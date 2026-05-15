import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.speculative.remote_tli_worker import (
    RemoteTLIWorker,
    _BroadcastedDraftResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestRemoteTLIWorker(CustomTestCase):
    def _make_worker(self, *, rank: int, tp_size: int, draft_tp_size: int):
        worker = RemoteTLIWorker.__new__(RemoteTLIWorker)
        worker.tp_size = tp_size
        worker.tp_rank = rank
        worker.draft_tp_size = draft_tp_size
        worker.server_args = SimpleNamespace(tli_rpc_timeout=3.0)
        worker.client = SimpleNamespace(draft_forward=Mock())
        worker.target_worker = SimpleNamespace(
            world_group=SimpleNamespace(
                rank=rank,
                first_rank=0,
                cpu_group=object(),
                is_first_rank=(rank == 0),
            )
        )
        return worker

    def test_rank0_broadcasted_draft_forward_uses_client_on_root(self):
        worker = self._make_worker(rank=0, tp_size=4, draft_tp_size=1)
        worker.client.draft_forward.return_value = "draft-response"

        with patch(
            "sglang.srt.speculative.remote_tli_worker.broadcast_pyobj",
            side_effect=lambda data, *args, **kwargs: data,
        ) as mock_broadcast:
            response = worker._run_rank0_broadcasted_draft_forward(
                SimpleNamespace(),
                timeout=3.0,
                translate_request_to_draft_vocab=True,
                translate_response_to_target_vocab=True,
            )

        self.assertEqual(response, "draft-response")
        worker.client.draft_forward.assert_called_once()
        mock_broadcast.assert_called_once()

    def test_rank0_broadcasted_draft_forward_receives_broadcast_on_non_root(self):
        worker = self._make_worker(rank=2, tp_size=4, draft_tp_size=1)

        def fake_broadcast(data, rank, dist_group=None, src=0, force_cpu_device=True):
            del rank, dist_group, src, force_cpu_device
            if data:
                return data
            return [_BroadcastedDraftResult(ok=True, response="draft-response")]

        with patch(
            "sglang.srt.speculative.remote_tli_worker.broadcast_pyobj",
            side_effect=fake_broadcast,
        ) as mock_broadcast:
            response = worker._run_rank0_broadcasted_draft_forward(
                SimpleNamespace(),
                timeout=3.0,
                translate_request_to_draft_vocab=True,
                translate_response_to_target_vocab=True,
            )

        self.assertEqual(response, "draft-response")
        worker.client.draft_forward.assert_not_called()
        mock_broadcast.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
