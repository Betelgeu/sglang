"""Regression tests for dLLM on the shared overlap scheduler."""

import os
import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import _dllm_overlap_disable
from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def run_scheduler_overlap(algorithm, runner, forward_batch):
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(dllm_algo_state=None)
            for _ in range(forward_batch.batch_size)
        ]
    )
    worker = SimpleNamespace(
        dllm_algorithm=algorithm,
        model_runner=runner,
        prepare_dllm_batch=Mock(return_value=forward_batch),
    )
    scheduler = SimpleNamespace(tp_worker=worker)
    return SchedulerDllmMixin.run_batch_dllm_overlap(scheduler, batch)


class TestDllmOverlap(unittest.TestCase):
    def test_overlap_is_opt_in_and_explicit_disable_wins(self):
        view = SimpleNamespace(
            dllm_algorithm="LowConfidence",
            disable_overlap_schedule=False,
            device="cuda",
            tp_size=1,
            pp_size=1,
            dp_size=1,
            speculative_algorithm=None,
            disaggregation_mode="null",
        )
        with patch.dict(os.environ, SGLANG_ENABLE_DLLM_OVERLAP="0"):
            self.assertEqual(
                _dllm_overlap_disable(view), {"disable_overlap_schedule": True}
            )
        with patch.dict(os.environ, SGLANG_ENABLE_DLLM_OVERLAP="1"):
            self.assertEqual(_dllm_overlap_disable(view), {})
            view.tp_size = 2
            with self.assertRaisesRegex(ValueError, "TP1/PP1/DP1"):
                _dllm_overlap_disable(view)
            view.disable_overlap_schedule = True
            self.assertEqual(_dllm_overlap_disable(view), {})

    def test_fdfo_commits_partial_and_departing_blocks_before_output(self):
        def req(rid):
            return SimpleNamespace(
                rid=rid,
                dllm_incomplete_ids=array("q"),
                dllm_algo_state=None,
                full_untruncated_fill_ids=array("q", [99] * 4),
                origin_input_ids=array("q"),
                output_ids=array("q"),
                extend_range=SimpleNamespace(end=4),
                update_finish_state=Mock(),
                finished=lambda: False,
            )

        partial, departed = req("partial"), req("departed")
        state = {"steps": 3}
        batch = SimpleNamespace(
            reqs=[partial, departed],
            batch_size=lambda: 2,
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            accept_lens=None,
            next_token_ids=[[1, 99, 99, 99], [2, 3, 4, 5]],
            accept_length_per_req_cpu=[0, 4],
            dllm_algo_state=[state, None],
            can_run_cuda_graph=False,
        )
        scheduler = SimpleNamespace(
            dllm_config=SimpleNamespace(first_done_first_out_mode=True, block_size=4),
            token_to_kv_pool_allocator=Mock(),
            metrics_reporter=Mock(num_generated_tokens=0),
            output_streamer=Mock(),
            enable_overlap=False,
        )
        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release:
            SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, result)
            release.assert_not_called()
        self.assertEqual(list(partial.dllm_incomplete_ids), [1, 99, 99, 99])
        self.assertIs(partial.dllm_algo_state, state)
        self.assertEqual(list(partial.output_ids), [])
        self.assertEqual(list(departed.full_untruncated_fill_ids), [2, 3, 4, 5])
        self.assertEqual(list(departed.output_ids), [2, 3, 4, 5])
        self.assertFalse(departed.dllm_incomplete_ids)
        self.assertIsNone(departed.dllm_algo_state)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [partial, departed], False
        )

    def test_finished_request_is_released_only_after_copy_done(self):
        order = []
        req = SimpleNamespace(
            dllm_incomplete_ids=array("q"),
            dllm_algo_state=None,
            full_untruncated_fill_ids=array("q", [9, 9]),
            origin_input_ids=array("q"),
            output_ids=array("q"),
            extend_range=SimpleNamespace(end=2),
            update_finish_state=lambda **kw: order.append("finish"),
            finished=lambda: True,
            time_stats=Mock(),
        )
        batch = SimpleNamespace(
            reqs=[req],
            batch_size=lambda: 1,
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=SimpleNamespace(synchronize=lambda: order.append("wait_copy")),
            next_token_ids=torch.tensor([[2, 3]]),
            accept_lens=torch.tensor([2]),
            accept_length_per_req_cpu=None,
            dllm_algo_state=None,
            can_run_cuda_graph=False,
        )
        scheduler = SimpleNamespace(
            dllm_config=SimpleNamespace(first_done_first_out_mode=True, block_size=2),
            token_to_kv_pool_allocator=Mock(),
            tree_cache=Mock(),
            enable_overlap=True,
            metrics_reporter=Mock(num_generated_tokens=0),
            output_streamer=SimpleNamespace(
                stream_output=lambda *args: order.append("emit")
            ),
        )
        with patch(
            "sglang.srt.dllm.mixin.scheduler.release_kv_cache",
            side_effect=lambda *args: order.append("release"),
        ):
            SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, result)
        self.assertEqual(order, ["wait_copy", "finish", "release", "emit"])
        self.assertEqual(list(req.output_ids), [2, 3])

    def test_pending_requests_are_not_reinitialized_or_selected(self):
        class Request:
            def __init__(self, prefill):
                self.prefill = prefill
                self.init_next_round_input = Mock()

            def is_dllm_prefill(self):
                return self.prefill

        a, b, c = Request(False), Request(False), Request(True)
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        manager.waiting_queue = [a, b, c]
        manager.staging_queue = [a, b]
        manager.init_next_round({a})
        a.init_next_round_input.assert_not_called()
        b.init_next_round_input.assert_called_once()
        self.assertEqual(manager.staging_queue, [a])
        self.assertEqual(manager.get_decode_requests({a}), [b])
        self.assertEqual(manager.get_prefill_requests({a}), [c])

    def test_fdfo_async_result_keeps_final_kv_departure_boundary(self):
        config = DllmConfig("LowConfidence", {}, 2, 9, 2, True)
        algorithm = LowConfidence(config)
        batch = SimpleNamespace(batch_size=2, input_ids=torch.tensor([9, 9, 1, 2]))
        logits = torch.zeros(4, 10)
        logits[:, 3] = 20
        runner = SimpleNamespace(
            forward=Mock(
                return_value=SimpleNamespace(
                    logits_output=SimpleNamespace(full_logits=logits),
                    can_run_graph=False,
                )
            )
        )
        result = run_scheduler_overlap(algorithm, runner, batch)
        result.delay_sample_func()
        ids, accepted = result.next_token_ids, result.accept_lens
        self.assertIsInstance(ids, torch.Tensor)
        self.assertIsInstance(accepted, torch.Tensor)
        self.assertEqual(accepted.tolist(), [0, 2])
        self.assertEqual(ids.tolist(), [[3, 3], [1, 2]])
        result = run_scheduler_overlap(algorithm, runner, batch)
        result.delay_sample_func()
        accepted = result.accept_lens
        self.assertEqual(accepted.tolist(), [2, 2])

    def test_sync_continuation_does_not_repeat_first_forward(self):
        algorithm = LowConfidence(DllmConfig("LowConfidence", {}, 2, 9, 2, False))
        batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1, 2]))
        out = SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=None), can_run_graph=False
        )
        runner = SimpleNamespace(forward=Mock(return_value=out))
        result = run_scheduler_overlap(algorithm, runner, batch)
        result.delay_sample_func()
        runner.forward.assert_called_once()
        self.assertEqual(result.accept_lens.tolist(), [0])

    def test_sync_overlap_tracks_variable_lengths_and_pure_prefill(self):
        algorithm = LowConfidence(DllmConfig("LowConfidence", {}, 2, 9, 3, False))
        batch = SimpleNamespace(
            batch_size=3, input_ids=torch.tensor([1, 2, 1, 9, 9, 9])
        )
        logits = torch.zeros(6, 10)
        logits[:, 3] = 20
        runner = SimpleNamespace(
            forward=Mock(
                return_value=SimpleNamespace(
                    logits_output=SimpleNamespace(full_logits=logits),
                    can_run_graph=False,
                )
            )
        )
        result = run_scheduler_overlap(algorithm, runner, batch)
        self.assertEqual(batch.input_ids.tolist(), [1, 2, 1, 9, 9, 9])
        result.delay_sample_func()
        self.assertEqual(result.accept_lens.tolist(), [0, 1, 2])
        self.assertEqual(result.next_token_ids.tolist(), [[1, 2], [1, 3], [3, 3]])
        self.assertEqual(runner.forward.call_count, 2)

    def test_normal_results_are_cpu_lists(self):
        for fdfo in [False, True]:
            algorithm = LowConfidence(DllmConfig("LowConfidence", {}, 2, 9, 1, fdfo))
            batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1, 9]))
            logits = torch.zeros(2, 10)
            logits[:, 3] = 20
            runner = SimpleNamespace(
                forward=Mock(
                    return_value=SimpleNamespace(
                        logits_output=SimpleNamespace(full_logits=logits),
                        can_run_graph=False,
                    )
                )
            )
            result = algorithm.run(runner, batch)
            self.assertEqual(result.next_token_ids, [[1, 3]] if fdfo else [[3]])

    def test_joint_threshold_step_returns_tensor_in_all_modes(self):
        for fdfo in [False, True]:
            for vectorized in [False, True]:
                algorithm = JointThreshold(
                    DllmConfig(
                        "JointThreshold",
                        {"vectorized_decoding": vectorized},
                        2,
                        9,
                        1,
                        fdfo,
                    )
                )
                batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1, 2]))
                logits = torch.zeros(2, 10)
                done = algorithm.step(batch, logits, algorithm.init_step_state(batch))
                self.assertIsInstance(done, torch.Tensor)
                self.assertEqual(done.dtype, torch.bool)
                self.assertEqual(done.tolist(), [True])

    def test_common_loop_launches_independent_batch_before_committing_previous(self):
        order = []
        a, b = object(), object()
        remaining = [a, b]
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            dllm_config=object(),
            enable_overlap=True,
            _engine_paused=False,
            running_batch=None,
            last_batch=None,
            is_generation=False,
            ingest_requests=lambda: None,
            _apply_war_barrier=lambda: None,
            is_disable_overlap_for_batch=lambda *args, **kwargs: False,
        )

        def prepare(**kwargs):
            pending = SchedulerDllmMixin._dllm_pending_reqs(scheduler)
            if remaining:
                req = remaining.pop(0)
                self.assertNotIn(req, pending)
                if req is b:
                    self.assertEqual(pending, {a})
                batch = SimpleNamespace(reqs=[req])
                batch.copy = lambda: batch
            else:
                batch = None
            return SimpleNamespace(batch_to_run=batch, running_batch=None)

        scheduler.get_next_batch_to_run = prepare
        scheduler.run_batch = lambda batch: order.append(("launch", batch.reqs[0]))
        scheduler.process_batch_result = lambda batch, result: order.append(
            ("commit", batch.reqs[0])
        )
        scheduler.on_idle = lambda: setattr(scheduler, "gracefully_exit", True)
        Scheduler.event_loop_overlap(scheduler)
        self.assertEqual(
            order, [("launch", a), ("launch", b), ("commit", a), ("commit", b)]
        )
        self.assertFalse(scheduler.result_queue)

    def test_sync_uses_existing_delayed_sample_handoff(self):
        algorithm = LowConfidence(DllmConfig("LowConfidence", {}, 2, 9, 2, False))
        batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1, 2]))
        out = SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=None), can_run_graph=False
        )
        runner = SimpleNamespace(forward=Mock(return_value=out))
        result = run_scheduler_overlap(algorithm, runner, batch)
        self.assertIsNone(result.next_token_ids)
        self.assertIsNotNone(result.delay_sample_func)
        self.assertIs(result.delay_sample_func(), result)
        runner.forward.assert_called_once()
        self.assertEqual(result.next_token_ids.tolist(), [[1, 2]])
        self.assertEqual(result.accept_lens.tolist(), [0])

    def test_fdfo_delays_sampling_until_previous_result_has_been_processed(self):
        algorithm = LowConfidence(DllmConfig("LowConfidence", {}, 2, 9, 2, True))
        batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([9, 9]))
        logits = torch.zeros(2, 10)
        logits[:, 3] = 20
        out = SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits), can_run_graph=False
        )
        runner = SimpleNamespace(forward=Mock(return_value=out))
        result = run_scheduler_overlap(algorithm, runner, batch)
        self.assertIsNone(result.next_token_ids)
        self.assertEqual(batch.input_ids.tolist(), [9, 9])
        self.assertIs(result.delay_sample_func(), result)
        runner.forward.assert_called_once()
        self.assertEqual(result.next_token_ids.tolist(), [[3, 3]])
        self.assertEqual(result.accept_lens.tolist(), [0])

    def test_worker_normal_execution_does_not_schedule_a_callback(self):
        algorithm = LowConfidence(DllmConfig("LowConfidence", {}, 2, 9, 2, False))
        batch = SimpleNamespace(batch_size=1, input_ids=torch.tensor([1, 2]))
        runner = SimpleNamespace(
            forward=Mock(
                return_value=SimpleNamespace(logits_output=None, can_run_graph=False)
            )
        )
        worker = SimpleNamespace(
            dllm_algorithm=algorithm, model_runner=runner, enable_overlap=True
        )
        result = TpModelWorker._forward_batch_generation_dllm(worker, batch)
        self.assertIsNone(result.delay_sample_func)
        self.assertEqual(result.next_token_ids, [])

    def test_worker_prepares_dllm_batch_with_cache_consumer(self):
        batch = SimpleNamespace(hicache_consumer_index=7)
        worker = SimpleNamespace(model_runner=object(), set_hicache_consumer=Mock())
        with patch(
            "sglang.srt.managers.tp_worker.ForwardBatch.init_new", autospec=True
        ) as init:
            result = TpModelWorker.prepare_dllm_batch(worker, batch)
        worker.set_hicache_consumer.assert_called_once_with(7)
        init.assert_called_once_with(
            batch, worker.model_runner, return_hidden_states_before_norm=False
        )
        self.assertIs(result, init.return_value)

    def test_overlap_snapshot_preserves_dllm_routing(self):
        config = DllmConfig("LowConfidence", {}, 2, 9, 2, True)
        batch = ScheduleBatch(reqs=[], dllm_config=config)
        self.assertIs(batch.copy().dllm_config, config)

    def test_input_dispatch_drains_pending_before_abort_can_free_slots(self):
        order = []
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
            _poll_timeout_aborts=lambda: [],
            request_receiver=SimpleNamespace(recv_requests=lambda **kw: ["abort"]),
            metrics_reporter=Mock(),
            dllm_config=object(),
            enable_overlap=True,
            result_queue=deque([("batch", "result")]),
            last_batch=object(),
            process_batch_result=lambda *args: order.append("complete_and_commit"),
            process_input_requests=lambda reqs: order.append("dispatch_abort"),
        )
        self.assertEqual(Scheduler.ingest_requests(scheduler), ["abort"])
        self.assertEqual(order, ["complete_and_commit", "dispatch_abort"])
        self.assertIsNone(scheduler.last_batch)


if __name__ == "__main__":
    unittest.main()
