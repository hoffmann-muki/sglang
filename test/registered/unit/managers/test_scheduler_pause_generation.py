import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_runtime_checker_mixin import PoolStats

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


class TestSchedulerPauseGeneration(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.chunked_req = None
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.running_batch.batch_is_full = False
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.protected_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock()
        scheduler.result_queue = deque()
        scheduler.draft_worker = None
        # Support _kv_snap diagnostic logging in patched schedulers
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 1000
        scheduler.max_total_num_tokens = 1000
        scheduler._get_token_info = MagicMock(
            return_value=PoolStats(
                full_num_used=0,
                full_token_usage=0,
                full_available_size=1000,
                full_evictable_size=0,
            )
        )
        return scheduler

    def test_inplace_only_sets_flag(self):
        """in_place pause should only set _engine_paused and return."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.cur_batch = MagicMock()
        scheduler.chunked_req = MagicMock()

        original_last_batch = scheduler.last_batch
        original_cur_batch = scheduler.cur_batch
        original_chunked_req = scheduler.chunked_req

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        # All state must be preserved — no mutation
        self.assertIs(scheduler.last_batch, original_last_batch)
        self.assertIs(scheduler.cur_batch, original_cur_batch)
        self.assertIs(scheduler.chunked_req, original_chunked_req)

    def test_inplace_does_not_drain_overlap_queue(self):
        """in_place should not process the overlap result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        scheduler.last_batch = MagicMock()
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(len(scheduler.result_queue), 1)

    def test_inplace_does_not_merge_batch(self):
        """in_place should not filter or merge last_batch into running_batch."""
        scheduler = self._new_scheduler()
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True
        scheduler.last_batch = last_batch

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        last_batch.filter_batch.assert_not_called()
        scheduler.running_batch.merge_batch.assert_not_called()

    def test_abort_clears_state(self):
        """abort mode should clear last_batch and cur_batch."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.forward_mode.is_extend.return_value = False
        scheduler.cur_batch = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))

        self.assertTrue(scheduler._engine_paused)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.cur_batch)

    def test_retract_clears_running_batch(self):
        """retract mode should retract all requests from running_batch."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = None
        running_reqs = [MagicMock(), MagicMock()]
        for idx, req in enumerate(running_reqs):
            req.rid = f"running-req-{idx}"
        scheduler.running_batch.reqs = running_reqs
        scheduler.running_batch.__len__ = lambda self: len(self.reqs)
        scheduler.running_batch.is_empty.return_value = False
        scheduler.waiting_queue = []
        scheduler._add_request_to_queue = MagicMock()

        retracted = [MagicMock(), MagicMock()]
        scheduler.running_batch.retract_all.return_value = retracted
        scheduler.running_batch.filter_batch = MagicMock()
        scheduler.server_args = MagicMock()
        for idx, req in enumerate(retracted):
            req.rid = f"req-{idx}"
        scheduler.draft_worker = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler._engine_paused)
        scheduler.running_batch.retract_all.assert_called_once()
        scheduler.draft_worker.drain_pending_request_ids.assert_called_once_with(
            ["running-req-0", "running-req-1"]
        )
        scheduler.draft_worker.release_request_ids.assert_called_once_with(
            ["req-0", "req-1"],
            cache_prefix=False,
        )
        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        self.assertIsNone(scheduler.chunked_req)

    def test_abort_drains_overlap_queue(self):
        """abort with overlap enabled should drain the result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        mock_batch = MagicMock()
        mock_batch.forward_mode.is_extend.return_value = False
        scheduler.last_batch = mock_batch
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])
        scheduler.process_batch_result = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))

        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 0)

    def test_active_pool_idxs_ignores_remote_draft_executor_state(self):
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.is_empty.return_value = False
        req1 = MagicMock()
        req1.req_pool_idx = 5
        req2 = MagicMock()
        req2.req_pool_idx = None
        scheduler.last_batch.reqs = [req1, req2]
        scheduler.running_batch.is_empty.return_value = True
        scheduler.remote_draft_executor = MagicMock()
        scheduler.remote_draft_executor.active_req_pool_idxs.return_value = {3, 7}

        self.assertEqual(Scheduler._active_pool_idxs(scheduler), {5})

    def test_session_held_counts_include_remote_draft_executor_state(self):
        scheduler = self._new_scheduler()
        scheduler.last_batch = None
        scheduler.running_batch.is_empty.return_value = True
        scheduler.tree_cache.session_held_tokens.return_value = 11
        scheduler.tree_cache.session_held_full_tokens.return_value = 11
        scheduler.tree_cache.session_held_swa_tokens.return_value = 5
        scheduler.tree_cache.session_held_req_count.return_value = 2
        scheduler.remote_draft_executor = MagicMock()
        scheduler.remote_draft_executor.active_req_pool_idxs.return_value = {3}
        scheduler.remote_draft_executor.held_full_tokens.return_value = 7
        scheduler.remote_draft_executor.held_swa_tokens.return_value = 4
        scheduler.remote_draft_executor.held_req_count.return_value = 1

        self.assertEqual(Scheduler._session_held_tokens(scheduler), 18)
        self.assertEqual(Scheduler._session_held_full_tokens(scheduler), 18)
        self.assertEqual(Scheduler._session_held_swa_tokens(scheduler), 9)
        self.assertEqual(Scheduler._session_held_req_count(scheduler), 3)

    def test_remote_draft_release_helper_deduplicates_request_ids(self):
        scheduler = self._new_scheduler()
        scheduler.draft_worker = MagicMock()

        Scheduler._release_remote_draft_request_ids(
            scheduler,
            ["req-a", "req-b", "req-a"],
            cache_prefix=False,
        )

        scheduler.draft_worker.release_request_ids.assert_called_once_with(
            ["req-a", "req-b"],
            cache_prefix=False,
        )

    def test_remote_draft_drain_helper_deduplicates_request_ids(self):
        scheduler = self._new_scheduler()
        scheduler.draft_worker = MagicMock()

        Scheduler._drain_remote_draft_request_ids(
            scheduler,
            ["req-a", "req-b", "req-a"],
        )

        scheduler.draft_worker.drain_pending_request_ids.assert_called_once_with(
            ["req-a", "req-b"]
        )

    def test_remote_draft_drain_helper_ignores_local_draft_workers(self):
        scheduler = self._new_scheduler()
        scheduler.draft_worker = object()

        Scheduler._drain_remote_draft_request_ids(
            scheduler,
            ["req-a"],
        )

    def test_remote_draft_release_helper_ignores_local_draft_workers(self):
        scheduler = self._new_scheduler()
        scheduler.draft_worker = object()

        Scheduler._release_remote_draft_request_ids(
            scheduler,
            ["req-a"],
            cache_prefix=True,
        )

    def test_running_remote_draft_state_gate_waits_until_ready(self):
        scheduler = self._new_scheduler()
        scheduler.running_batch.is_empty.return_value = False
        scheduler.running_batch.reqs = [
            SimpleNamespace(rid="req-a"),
            SimpleNamespace(rid="req-b"),
            SimpleNamespace(rid="req-a"),
        ]
        scheduler.draft_worker = MagicMock()
        scheduler.draft_worker.has_pending_request_ids.return_value = True
        scheduler.draft_worker.pending_request_ids_ready.return_value = False

        ready = Scheduler._drain_ready_remote_draft_state_for_running_batch(scheduler)

        self.assertFalse(ready)
        scheduler.draft_worker.has_pending_request_ids.assert_called_once_with(
            ["req-a", "req-b"]
        )
        scheduler.draft_worker.pending_request_ids_ready.assert_called_once_with(
            ["req-a", "req-b"]
        )
        scheduler.draft_worker.drain_pending_request_ids.assert_not_called()

    def test_running_remote_draft_state_gate_drains_when_ready(self):
        scheduler = self._new_scheduler()
        scheduler.running_batch.is_empty.return_value = False
        scheduler.running_batch.reqs = [
            SimpleNamespace(rid="req-a"),
            SimpleNamespace(rid="req-b"),
            SimpleNamespace(rid="req-a"),
        ]
        scheduler.draft_worker = MagicMock()
        scheduler.draft_worker.has_pending_request_ids.return_value = True
        scheduler.draft_worker.pending_request_ids_ready.return_value = True

        ready = Scheduler._drain_ready_remote_draft_state_for_running_batch(scheduler)

        self.assertTrue(ready)
        scheduler.draft_worker.drain_pending_request_ids.assert_called_once_with(
            ["req-a", "req-b"]
        )

    def test_running_remote_draft_state_gate_ignores_non_remote_worker(self):
        scheduler = self._new_scheduler()
        scheduler.running_batch.is_empty.return_value = False
        scheduler.running_batch.reqs = [SimpleNamespace(rid="req-a")]
        scheduler.draft_worker = object()

        self.assertTrue(
            Scheduler._drain_ready_remote_draft_state_for_running_batch(scheduler)
        )

    def test_get_next_batch_to_run_can_prefill_while_remote_state_is_pending(self):
        scheduler = self._new_scheduler()
        scheduler._abort_on_waiting_timeout = MagicMock()
        scheduler._abort_on_running_timeout = MagicMock()
        scheduler.dllm_config = None
        scheduler.enable_hisparse = False
        scheduler.require_mlp_sync = False
        scheduler.running_batch.is_prefill_only = False
        scheduler._drain_ready_remote_draft_state_for_running_batch = MagicMock(
            return_value=False
        )
        new_batch = MagicMock()
        scheduler.get_new_batch_prefill = MagicMock(return_value=new_batch)
        scheduler.maybe_prepare_mlp_sync_batch = MagicMock(
            side_effect=lambda ret, **_: ret
        )
        scheduler._maybe_prepare_ngram_embedding = MagicMock(
            side_effect=lambda ret: ret
        )

        ret = Scheduler.get_next_batch_to_run(scheduler)

        self.assertIs(ret, new_batch)
        scheduler.get_new_batch_prefill.assert_called_once()

    def test_get_next_batch_to_run_defers_unmerged_prefill_until_remote_state_ready(
        self,
    ):
        scheduler = self._new_scheduler()
        scheduler._abort_on_waiting_timeout = MagicMock()
        scheduler._abort_on_running_timeout = MagicMock()
        scheduler.dllm_config = None
        scheduler.enable_hisparse = False
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.forward_mode.is_extend.return_value = True
        scheduler._drain_ready_remote_draft_state_for_running_batch = MagicMock(
            return_value=False
        )
        scheduler.get_new_batch_prefill = MagicMock()

        ret = Scheduler.get_next_batch_to_run(scheduler)

        self.assertIsNone(ret)
        scheduler.get_new_batch_prefill.assert_not_called()

    @patch("sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache")
    def test_finished_request_releases_remote_draft_state(self, mock_release_kv_cache):
        scheduler = self._new_scheduler()
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        scheduler.enable_hisparse = False
        scheduler.maybe_collect_routed_experts = MagicMock()
        scheduler.maybe_collect_customized_info = MagicMock()
        scheduler._release_remote_draft_request_ids = MagicMock()
        req = MagicMock()
        req.rid = "req-a"
        req.finished.return_value = True
        req.multimodal_inputs = None
        req.session = None
        req.time_stats = MagicMock()

        Scheduler._handle_finished_req(scheduler, req, 0, MagicMock())

        mock_release_kv_cache.assert_called_once_with(req, scheduler.tree_cache)
        scheduler._release_remote_draft_request_ids.assert_called_once_with(
            ["req-a"],
            cache_prefix=True,
        )


if __name__ == "__main__":
    unittest.main()
