from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_exec, get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerDllmMixin:
    def run_batch_dllm_overlap(self: Scheduler, batch: ScheduleBatch):
        """Launch first forward; resume denoising after the previous result commits.

        Called inside run_batch's existing forward stream/isolation context.
        Sync and FDFO reuse block metadata and read the preceding GPU step.
        """
        worker = self.tp_worker
        algorithm = worker.dllm_algorithm
        block_ids = [req.extend_range.end for req in batch.reqs]
        if not batch.dllm_step_id:
            batch.dllm_block_ids = torch.tensor(
                block_ids, dtype=torch.int64, device=batch.req_pool_indices.device
            )
            batch.input_ids = batch.input_ids.clone()
            if not algorithm.fdfo:
                batch.dllm_steps_left = algorithm.max_steps(algorithm.block_size)
        versions = batch.dllm_block_ids
        previous_done = None
        if batch.dllm_step_id:
            tokens, previous_done = self.future_map.resolve_dllm(
                batch.req_pool_indices, versions, batch.dllm_step_id
            )
            batch.input_ids = tokens.flatten()
        runner = worker.model_runner
        worker.set_hicache_consumer(batch.hicache_consumer_index)
        forward_batch = ForwardBatch.init_new(
            batch, runner, return_hidden_states_before_norm=False
        )
        # CPU has not committed the preceding result yet. Its block-owned
        # algorithm state, especially prompt masks/edit budgets, is authoritative.
        states = (
            batch.dllm_algo_state
            if batch.dllm_step_id
            else algorithm.init_fdfo_states(
                forward_batch, [req.dllm_algo_state for req in batch.reqs]
            )
        )
        out = runner.forward(forward_batch, pp_proxy_tensors=None)
        result = GenerationBatchResult(
            logits_output=out.logits_output,
            dllm_algo_state=states,
            can_run_cuda_graph=out.can_run_graph,
            dllm_block_ids=block_ids,
        )

        # Request phase may still describe the preceding block. Only a block
        # entirely inside the original prompt, without an actual mask token,
        # needs only the KV-persisting forward.
        pure_prefill = all(
            end <= len(req.origin_input_ids)
            and algorithm.mask_id
            not in req.origin_input_ids[end - algorithm.block_size : end]
            for req, end in zip(batch.reqs, block_ids)
        )

        def finish():
            blocks = forward_batch.input_ids.view(-1, algorithm.block_size)
            if pure_prefill or (
                batch.dllm_steps_left is not None and batch.dllm_steps_left <= 0
            ):
                # As in _run_sync, persist the final update after max_steps.
                done = torch.ones(
                    len(batch.reqs), dtype=torch.bool, device=blocks.device
                )
            else:
                done = algorithm.step(
                    forward_batch, out.logits_output.full_logits, states
                )
            if previous_done is not None:
                done = done | previous_done
            self.future_map.publish_dllm(
                batch.req_pool_indices, blocks, done, versions, batch.forward_iter
            )
            # Sync keeps all rows in the block until the entire batch completes.
            accepted = done if algorithm.fdfo else done.all().expand_as(done)
            result.accept_lens = accepted.to(torch.int32) * algorithm.block_size
            # Neither a later step nor graph input reuse may overwrite D2H data.
            result.next_token_ids = blocks.clone()
            return result

        # The shared event loop calls this after processing the previous result,
        # then uses its existing copy stream and copy_done lifetime boundary.
        result.delay_sample_func = finish
        return result

    def drain_dllm_before_control(self: Scheduler, recv_reqs):
        """Ordinary arrivals enqueue; controls wait before changing ownership."""
        if all(
            isinstance(req, (TokenizedGenerateReqInput, BatchTokenizedGenerateReqInput))
            for req in recv_reqs
        ):
            return
        while self.result_queue:
            batch, result = self.result_queue.popleft()
            self.process_batch_result(batch, result)
            if (
                not self.dllm_config.first_done_first_out_mode
                and not self.result_queue
                and result.dllm_next_batch is not None
                and not any(result.accept_length_per_req_cpu)
            ):
                # Preserve sync's current block budget/state while draining.
                batch = result.dllm_next_batch
                result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), result))
                self.launch_batch_sample_if_needed(result, batch)
        self.last_batch = None

    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if get_exec().dllm.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_overlap and self.result_queue:
            continuation = self.result_queue[-1][1].dllm_next_batch
            if continuation is not None:
                if all(
                    req.extend_range.end <= len(req.origin_input_ids)
                    and self.dllm_config.mask_id
                    not in req.origin_input_ids[
                        req.extend_range.end
                        - self.dllm_config.block_size : req.extend_range.end
                    ]
                    for req in continuation.reqs
                ):
                    # No denoising to relay. Let the common loop commit this
                    # prompt forward; keep next_batch intact (None means sealed).
                    return None
            return continuation
        if self.enable_priority_preemption:
            running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(running_batch=running_batch):
            return None

        running_bs = len(running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs, running_batch=running_batch)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder, running_batch=running_batch)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(
            can_run_list, forward_mode, adder=adder, running_batch=running_batch
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        block_results = (
            self.dllm_config.first_done_first_out_mode or self.enable_overlap
        )
        if self.enable_overlap:
            result.next_token_ids = result.next_token_ids.tolist()
            accepted = result.accept_lens.tolist()
            result.accept_length_per_req_cpu = accepted

        assert not block_results or result.accept_length_per_req_cpu is not None, (
            "Block dLLM result is missing accept lengths."
        )

        live_indices = list(range(batch.batch_size()))
        block_ids = getattr(result, "dllm_block_ids", None)
        if block_ids is not None:
            live_indices = [
                i
                for i in live_indices
                if block_ids[i] > batch.reqs[i].dllm_committed_block_id
            ]
            if not live_indices:
                return  # A redundant step of an already committed block.
            if any(result.accept_length_per_req_cpu[i] for i in live_indices):
                # Next step may already be writing the SAME KV slots. Finish
                # its delayed step and D2H before this result can free anything.
                # No new block is prepared until both results have been consumed.
                if self.result_queue:
                    next_batch, next_result = self.result_queue[0]
                    self.launch_batch_sample_if_needed(next_result, next_batch)
                    next_result.copy_done.synchronize()
                    # Stop same-block continuation. The common loop consumes
                    # this extra result after finishing the current result.
                    next_result.dllm_next_batch = None

        # Whole-block results also carry unresolved tokens for the next step.
        if block_results or result.next_token_ids:
            block_size = self.dllm_config.block_size
            algo_states = result.dllm_algo_state

            self.token_to_kv_pool_allocator.free_group_begin()
            for idx in live_indices:
                req = batch.reqs[idx]

                if not block_results:
                    next_token_ids = result.next_token_ids[idx]
                    new_tokens = len(next_token_ids)
                    if new_tokens == 0:
                        continue

                    req.full_untruncated_fill_ids[
                        req.extend_range.end - new_tokens : req.extend_range.end
                    ] = array("q", next_token_ids)
                    self.metrics_reporter.num_generated_tokens += new_tokens

                    req.output_ids.extend(next_token_ids)
                    req.update_finish_state(new_accepted_len=new_tokens)

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    continue

                next_token_ids = result.next_token_ids[idx]
                assert len(next_token_ids) == block_size

                if result.accept_length_per_req_cpu[idx] == 0:
                    # Unresolved: keep partial state and KV for the next FDFO round.
                    req.dllm_incomplete_ids = array("q", next_token_ids)
                    req.dllm_algo_state = (
                        algo_states[idx] if algo_states is not None else None
                    )
                    continue

                req.dllm_incomplete_ids = array("q")
                req.dllm_algo_state = None
                if block_ids is not None:
                    req.dllm_committed_block_id = block_ids[idx]

                # Mirror the resolved block into the committed fill ids so the
                # prefix cache keys on the real tokens, not the mask block, next
                # round. Index relative to extend_range.end (the truncated/
                # committed length), which can be shorter than
                # full_untruncated_fill_ids when the staging adder truncates the
                # block to the KV budget.
                req.full_untruncated_fill_ids[
                    req.extend_range.end - block_size : req.extend_range.end
                ] = array("q", next_token_ids)

                len_input = len(req.origin_input_ids)
                len_fill = req.extend_range.end
                if len_fill <= len_input:
                    continue

                if len_fill - len(next_token_ids) < len_input:
                    next_token_ids = next_token_ids[len_input - len_fill :]

                self.metrics_reporter.num_generated_tokens += len(next_token_ids)
                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=len(next_token_ids))

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()

            self.output_streamer.stream_output(
                [batch.reqs[i] for i in live_indices], batch.return_logprob
            )
            self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        running_batch: ScheduleBatch,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True

            # Try preemption if batch is full
            if running_batch.batch_is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_inflight_middle_chunks(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def pop_aborted_reqs(self, abort_all: bool, rid: str) -> List[Req]:
        aborted_reqs: List[Req] = []
        seen: Set[int] = set()

        for queue_name in ("waiting_queue", "staging_queue"):
            queue = getattr(self, queue_name)
            kept_queue = []
            for req in queue:
                if abort_all or req.rid.startswith(rid):
                    req_id = id(req)
                    if req_id not in seen:
                        aborted_reqs.append(req)
                        seen.add(req_id)
                else:
                    kept_queue.append(req)
            setattr(self, queue_name, kept_queue)

        return aborted_reqs

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
