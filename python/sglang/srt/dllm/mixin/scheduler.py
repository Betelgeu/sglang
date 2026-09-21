from __future__ import annotations

import logging
from array import array
from copy import copy
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_exec, get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
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

    def drain_dllm_results(self):
        """Cross a block/control boundary using the ordinary ordered queue."""
        while self.result_queue:
            batch, result = self.result_queue.popleft()
            self.process_batch_result(batch, result)
        self.last_batch = None

    def get_dllm_overlap_batch(self, last_batch):
        """Continue one speculative task; ordinary admission handles boundaries.

        FDFO always advances the same block. Once any row has committed, drain
        its extra task before ordinary scheduling mixes next blocks/new rows.
        This preserves first-done departure without a GPU-dependent CPU branch
        in the steady-state denoising path.
        """
        block_size = self.dllm_config.block_size
        fdfo = self.dllm_config.first_done_first_out_mode
        can_admit = (
            bool(self.waiting_queue)
            and len(self.dllm_manager.waiting_queue)
            < self.dllm_config.max_running_requests
        )
        boundary = can_admit or any(
            req.finished()
            or req.to_finish is not None
            or (fdfo and req.dllm_committed_end >= prefix + length)
            for req, prefix, length in zip(
                last_batch.reqs, last_batch.prefix_lens, last_batch.extend_lens
            )
        )
        # Full prompt chunks may span multiple blocks. Their regular prefill
        # transition also goes through admission, after their result is known.
        boundary |= any(length != block_size for length in last_batch.extend_lens)
        if not fdfo:
            needed = len(last_batch.reqs) * block_size
            boundary |= self.token_to_kv_pool_allocator.available_size() < needed
            boundary |= any(
                prefix + length + block_size
                > self.req_to_token_pool.req_to_token.shape[1]
                for prefix, length in zip(
                    last_batch.prefix_lens, last_batch.extend_lens
                )
            )
        if boundary:
            self.drain_dllm_results()
            return None

        if fdfo:
            batch = copy(last_batch)
            batch.dllm_relay = True
            batch.dllm_algo_state = None
            return batch

        # Sync: final prefix KV is ordered before the next forward on the same
        # stream. Keep it private to this request until actual tokens commit;
        # never stash mask-valued speculative prefixes into the radix cache.
        for req, prefix, length in zip(
            last_batch.reqs, last_batch.prefix_lens, last_batch.extend_lens
        ):
            end = prefix + length
            req.prefix_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, :end
            ].to(dtype=torch.int64, copy=True)
            req.dllm_block_offset = end
            req.set_extend_range(end, end + block_size)
            missing = end + block_size - len(req.full_untruncated_fill_ids)
            if missing > 0:
                req.full_untruncated_fill_ids.extend(
                    array("q", [self.dllm_config.mask_id] * missing)
                )
        batch = ScheduleBatch.init_new(
            last_batch.reqs[:],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        batch.prepare_for_extend()
        batch.decoding_reqs = None
        batch.prefill_stats = last_batch.prefill_stats
        return batch

    @torch.profiler.record_function("dllm/commit_result")
    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        tokens = result.next_token_ids.cpu().tolist()
        done = result.dllm_done.cpu().tolist()
        block_size = self.dllm_config.block_size
        emitted = []
        self.token_to_kv_pool_allocator.free_group_begin()
        for idx, req in enumerate(batch.reqs):
            req.dllm_inflight -= 1
            assert req.dllm_inflight >= 0
            end = batch.prefix_lens[idx] + batch.extend_lens[idx]
            was_finished = req.finished()
            if req.to_finish is not None:
                req.update_finish_state(new_accepted_len=0)
                if not was_finished:
                    emitted.append(req)

            # n+1 already ran: discard only at commit, retaining resources until
            # its completion. The snapshot boundary cannot move with req state.
            if not req.finished() and end > req.dllm_committed_end:
                if not done[idx]:
                    req.dllm_incomplete_ids = array("q", tokens[idx])
                    req.dllm_algo_state = {
                        name: value[idx]
                        for name, value in (result.dllm_algo_state or {}).items()
                    }
                else:
                    req.dllm_committed_end = end
                    req.dllm_incomplete_ids = array("q")
                    req.dllm_algo_state = None
                    req.full_untruncated_fill_ids[end - block_size : end] = array(
                        "q", tokens[idx]
                    )
                    start = max(0, len(req.origin_input_ids) - (end - block_size))
                    new_tokens = tokens[idx][start:]
                    if new_tokens:
                        req.output_ids.extend(new_tokens)
                        self.metrics_reporter.num_generated_tokens += len(new_tokens)
                        req.update_finish_state(new_accepted_len=len(new_tokens))
                        emitted.append(req)
                        if req.finished():
                            req.time_stats.set_completion_time()

            if req.finished() and req.dllm_inflight == 0 and req.kv.holds_kv:
                # Sync may own an uncommitted next block; abort may own a
                # partially denoised block. Only committed tokens enter cache.
                req.kv.kv_committed_len = max(
                    req.kv.cache_protected_len, req.dllm_committed_end
                )
                release_kv_cache(req, self.tree_cache)

        if emitted:
            self.output_streamer.stream_output(emitted, batch.return_logprob)
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
