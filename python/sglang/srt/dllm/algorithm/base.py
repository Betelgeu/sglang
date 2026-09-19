from __future__ import annotations

from typing import Any, List, Optional

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu

_is_npu = is_npu()


class DllmAlgorithm:
    """Denoise algorithms and per-request state, independent of scheduling."""

    def __init__(self, config: DllmConfig):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        self.fdfo = config.first_done_first_out_mode

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [None] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        return block_size + 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> torch.Tensor:
        """Advance tokens/state and return a bool tensor, one value per block.

        A completed block must already have its final KV persisted by this
        forward; filling the last mask alone does not permit departure.
        """
        raise NotImplementedError

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> GenerationBatchResult:
        if self.fdfo:
            return self._run_fdfo(model_runner, forward_batch, algo_states)
        return self._run_sync(model_runner, forward_batch)

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        return (input_ids != self.mask_id).sum(dim=1).tolist()

    def denoise_sync(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        out: ModelRunnerOutput,
    ) -> ModelRunnerOutput:
        """Finish a block from the logits of its already executed first forward."""
        states = self.init_step_state(forward_batch)
        if _is_npu:
            forward_batch.mark_forward_metadata_ready()
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(
                forward_batch, out.logits_output.full_logits, states
            ).tolist()
            if all(done):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        return out

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        start_list = self._block_start_list(forward_batch)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        tokens = []
        if any(start < self.block_size for start in start_list):
            out = self.denoise_sync(model_runner, forward_batch, out)
            blocks = forward_batch.input_ids.view(-1, self.block_size).tolist()
            tokens = [ids[start:] for ids, start in zip(blocks, start_list)]
        return GenerationBatchResult(
            logits_output=out.logits_output,
            next_token_ids=tokens,
            can_run_cuda_graph=out.can_run_graph,
        )

    def init_fdfo_states(self, forward_batch, algo_states):
        if algo_states is None:
            return self.init_step_state(forward_batch)
        fresh = None
        states = []
        for carried in algo_states:
            if carried is None:
                if fresh is None:
                    fresh = self.init_step_state(forward_batch)
                states.append(fresh[len(states)])
            else:
                states.append(carried)
        return states

    def _run_fdfo(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]],
    ) -> GenerationBatchResult:
        states = self.init_fdfo_states(forward_batch, algo_states)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        done = self.step(forward_batch, out.logits_output.full_logits, states).tolist()
        return GenerationBatchResult(
            logits_output=out.logits_output,
            next_token_ids=forward_batch.input_ids.view(-1, self.block_size).tolist(),
            accept_length_per_req_cpu=[self.block_size if d else 0 for d in done],
            dllm_algo_state=[None if d else state for d, state in zip(done, states)],
            can_run_cuda_graph=out.can_run_graph,
        )
