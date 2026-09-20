"""Block departure must wait for in-flight work and never emit twice."""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDllmOverlap(unittest.TestCase):
    def test_block_departure_waits_for_redundant_step_and_emits_once(self):
        order = []
        reqs = [
            SimpleNamespace(
                dllm_committed_block_id=-1,
                dllm_incomplete_ids=array("q"),
                dllm_algo_state=None,
                full_untruncated_fill_ids=array("q", [9, 9]),
                origin_input_ids=array("q"),
                output_ids=array("q"),
                extend_range=SimpleNamespace(end=2),
                update_finish_state=lambda **kw: None,
                finished=lambda: True,
                time_stats=Mock(),
            )
            for _ in range(2)
        ]

        def batch():
            return SimpleNamespace(
                reqs=reqs[:],
                batch_size=lambda: 2,
                return_logprob=False,
                prefill_stats=None,
                dp_cooperation_info=None,
            )

        def result(tokens, accepted, step):
            return GenerationBatchResult(
                next_token_ids=torch.tensor(tokens),
                accept_lens=torch.tensor(accepted),
                dllm_block_ids=[2, 2],
                copy_done=SimpleNamespace(
                    synchronize=lambda: order.append(("wait", step))
                ),
            )

        a = result([[1, 2], [3, 9]], [2, 0], 0)
        b = result([[1, 2], [3, 4]], [2, 2], 1)
        scheduler = SchedulerDllmMixin()
        scheduler.dllm_config = SimpleNamespace(
            first_done_first_out_mode=True, block_size=2
        )
        scheduler.enable_overlap = True
        scheduler.result_queue = deque([(batch(), b)])
        scheduler.token_to_kv_pool_allocator = Mock()
        scheduler.tree_cache = Mock()
        scheduler.metrics_reporter = Mock(num_generated_tokens=0)
        emitted = []
        scheduler.output_streamer = SimpleNamespace(
            stream_output=lambda rows, _: emitted.extend(rows)
        )
        scheduler.launch_batch_sample_if_needed = lambda r, b: order.append(
            ("finish", 1)
        )
        scheduler.process_batch_result = scheduler.process_batch_result_dllm
        with patch(
            "sglang.srt.dllm.mixin.scheduler.release_kv_cache",
            side_effect=lambda req, _: order.append(("release", reqs.index(req))),
        ):
            scheduler.process_batch_result_dllm(batch(), a)
            self.assertIsNone(b.dllm_next_batch)
            scheduler.process_batch_result_dllm(*scheduler.result_queue.popleft())
        self.assertEqual([list(r.output_ids) for r in reqs], [[1, 2], [3, 4]])
        self.assertLess(order.index(("wait", 1)), order.index(("release", 0)))
        self.assertEqual(sum(r is reqs[0] for r in emitted), 1)
        self.assertFalse(scheduler.result_queue)
        self.assertEqual(scheduler.metrics_reporter.num_generated_tokens, 4)


if __name__ == "__main__":
    unittest.main()
