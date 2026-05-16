import unittest
from collections import deque
from unittest.mock import MagicMock

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
        scheduler.running_batch.reqs = [MagicMock(), MagicMock()]
        scheduler.running_batch.__len__ = lambda self: len(self.reqs)
        scheduler.running_batch.is_empty.return_value = False
        scheduler.waiting_queue = []
        scheduler._add_request_to_queue = MagicMock()

        retracted = [MagicMock(), MagicMock()]
        scheduler.running_batch.retract_all.return_value = retracted
        scheduler.running_batch.filter_batch = MagicMock()
        scheduler.server_args = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler._engine_paused)
        scheduler.running_batch.retract_all.assert_called_once()
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

    def test_active_pool_idxs_ignores_tli_draft_executor_state(self):
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.is_empty.return_value = False
        req1 = MagicMock()
        req1.req_pool_idx = 5
        req2 = MagicMock()
        req2.req_pool_idx = None
        scheduler.last_batch.reqs = [req1, req2]
        scheduler.running_batch.is_empty.return_value = True
        scheduler.tli_draft_executor = MagicMock()
        scheduler.tli_draft_executor.active_req_pool_idxs.return_value = {3, 7}

        self.assertEqual(Scheduler._active_pool_idxs(scheduler), {5})

    def test_session_held_counts_include_tli_draft_executor_state(self):
        scheduler = self._new_scheduler()
        scheduler.last_batch = None
        scheduler.running_batch.is_empty.return_value = True
        scheduler.tree_cache.session_held_tokens.return_value = 11
        scheduler.tree_cache.session_held_full_tokens.return_value = 11
        scheduler.tree_cache.session_held_swa_tokens.return_value = 5
        scheduler.tree_cache.session_held_req_count.return_value = 2
        scheduler.tli_draft_executor = MagicMock()
        scheduler.tli_draft_executor.active_req_pool_idxs.return_value = {3}
        scheduler.tli_draft_executor.held_full_tokens.return_value = 7
        scheduler.tli_draft_executor.held_swa_tokens.return_value = 4
        scheduler.tli_draft_executor.held_req_count.return_value = 1

        self.assertEqual(Scheduler._session_held_tokens(scheduler), 18)
        self.assertEqual(Scheduler._session_held_full_tokens(scheduler), 18)
        self.assertEqual(Scheduler._session_held_swa_tokens(scheduler), 9)
        self.assertEqual(Scheduler._session_held_req_count(scheduler), 3)


if __name__ == "__main__":
    unittest.main()
