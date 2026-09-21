from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

_is_npu = is_npu()


class DllmAlgorithm:
    """dLLM algorithm: subclasses implement ``step``; the base owns the
    synchronous and FDFO (``--dllm-fdfo``) execution loops in ``run``.
    """

    def __init__(self, config: DllmConfig):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        self.fdfo = config.first_done_first_out_mode

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)

    def init_step_state(self, forward_batch: ForwardBatch) -> dict[str, torch.Tensor]:
        return {}

    def max_steps(self, block_size: int) -> int:
        return block_size + 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Advance tokens/state in place and return device bool ``[B]`` done.

        Done means this forward persisted the final tokens' KV, not merely that
        all masks were filled. Terminal rows must preserve their tokens/state.
        """
        raise NotImplementedError

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> GenerationBatchResult:
        if self.fdfo:
            return self._run_fdfo(model_runner, forward_batch, algo_states)
        return self._run_sync(model_runner, forward_batch)

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        # Each synchronous call starts a new block. No-mask prompt rows are
        # already terminal; step leaves them unchanged while other rows denoise.
        states = self.init_step_state(forward_batch)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # NPU attention metadata is stable across the block's denoise steps.
        if _is_npu:
            forward_batch.mark_forward_metadata_ready()
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            # Sync mode intentionally retains its CPU completion decision.
            if all(done.cpu().tolist()):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        return self._make_result(forward_batch, out, done, states)

    def _run_fdfo(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> GenerationBatchResult:
        # The caller gathers mixed fresh/continuing rows into one batched dict.
        states = (
            self.init_step_state(forward_batch) if algo_states is None else algo_states
        )
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        done = self.step(forward_batch, out.logits_output.full_logits, states)
        return self._make_result(forward_batch, out, done, states)

    def _make_result(
        self,
        forward_batch: ForwardBatch,
        out,
        done: torch.Tensor,
        states: dict[str, torch.Tensor],
    ) -> GenerationBatchResult:
        # Imported after algorithm discovery to avoid the managers/algorithm
        # import cycle. Snapshots survive reuse of inputs and continuation state.
        from sglang.srt.managers.utils import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=out.logits_output,
            next_token_ids=forward_batch.input_ids.view(
                forward_batch.batch_size, self.block_size
            ).clone(),
            dllm_done=done.clone(),
            dllm_algo_state={key: value.clone() for key, value in states.items()},
            can_run_cuda_graph=out.can_run_graph,
        )
