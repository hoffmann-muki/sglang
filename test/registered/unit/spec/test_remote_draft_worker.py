import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.remote_draft_worker import (
    RemoteDraftWorker,
    _BroadcastedDraftResult,
    _PendingExtendAfterDecode,
)
from sglang.srt.speculative.draft_forward_protocol import DraftForwardResponse
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _StaticFuture:
    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        del timeout
        return self._result


class TestRemoteDraftWorker(CustomTestCase):
    def _make_worker(self, *, rank: int, tp_size: int, draft_tp_size: int):
        worker = RemoteDraftWorker.__new__(RemoteDraftWorker)
        worker.tp_size = tp_size
        worker.tp_rank = rank
        worker.remote_draft_tp_size = draft_tp_size
        worker.server_args = SimpleNamespace(draft_forward_rpc_timeout=3.0)
        worker.client = SimpleNamespace(draft_forward=Mock())
        worker.active_request_ids = set()
        worker._request_ordering = {}
        worker._pending_extend_after_decode = []
        worker._draft_forward_executor = Mock()
        worker.model_config = SimpleNamespace(dtype=torch.float32)
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
            "sglang.srt.speculative.remote_draft_worker.broadcast_pyobj",
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
            "sglang.srt.speculative.remote_draft_worker.broadcast_pyobj",
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

    def test_attach_ordering_metadata_advances_rounds_per_request(self):
        worker = self._make_worker(rank=0, tp_size=1, draft_tp_size=1)
        request = SimpleNamespace()

        worker._attach_ordering_metadata(
            request,
            request_ids=["req-a", "req-b"],
            token_positions=[7, 11],
            advance_round=True,
        )

        self.assertEqual(request.round_ids, [1, 1])
        self.assertEqual(request.token_positions, [7, 11])
        self.assertEqual(request.prefix_versions, [0, 0])

        next_prefix_versions = worker._next_prefix_versions(["req-a", "req-b"], [2, 0])
        worker._attach_ordering_metadata(
            request,
            request_ids=["req-a", "req-b"],
            token_positions=[10, 12],
            advance_round=True,
            prefix_versions=next_prefix_versions,
        )
        worker._commit_prefix_versions(["req-a", "req-b"], next_prefix_versions)

        self.assertEqual(request.round_ids, [2, 2])
        self.assertEqual(request.token_positions, [10, 12])
        self.assertEqual(request.prefix_versions, [3, 1])

    def test_validate_ordering_response_rejects_wrong_prefix_version(self):
        request = SimpleNamespace(
            request_id="decode:req-a",
            round_ids=[2],
            token_positions=[10],
            prefix_versions=[3],
        )
        response = SimpleNamespace(
            round_ids=[2],
            token_positions=[10],
            prefix_versions=[2],
        )

        with self.assertRaisesRegex(RuntimeError, "ordering metadata mismatch"):
            RemoteDraftWorker._validate_ordering_response(request, response)

    def test_release_request_ids_filters_inactive_requests(self):
        worker = self._make_worker(rank=0, tp_size=1, draft_tp_size=1)
        worker.active_request_ids = {"req-a", "req-b"}
        worker._request_ordering = {
            "req-a": object(),
            "req-b": object(),
            "inactive": object(),
        }
        worker._run_rank0_broadcasted_draft_forward = Mock(
            return_value=SimpleNamespace(
                round_ids=None,
                token_positions=None,
                prefix_versions=None,
            )
        )

        worker.release_request_ids(
            ["req-a", "inactive", "req-a"],
            cache_prefix=False,
        )

        worker._run_rank0_broadcasted_draft_forward.assert_called_once()
        release_request = worker._run_rank0_broadcasted_draft_forward.call_args.args[0]
        self.assertEqual(release_request.mode, "release")
        self.assertEqual(release_request.request_ids, ["req-a"])
        self.assertFalse(release_request.cache_prefix_on_release)
        self.assertEqual(worker.active_request_ids, {"req-b"})
        self.assertNotIn("req-a", worker._request_ordering)
        self.assertIn("req-b", worker._request_ordering)
        self.assertIn("inactive", worker._request_ordering)

    def test_release_request_ids_noops_without_active_matches(self):
        worker = self._make_worker(rank=0, tp_size=1, draft_tp_size=1)
        worker.active_request_ids = {"req-a"}
        worker._run_rank0_broadcasted_draft_forward = Mock()

        worker.release_request_ids(["req-b"], cache_prefix=True)

        worker._run_rank0_broadcasted_draft_forward.assert_not_called()

    def test_async_draft_forward_submits_rpc_on_root_only(self):
        response = DraftForwardResponse(
            request_id="extend_after_decode:req-a",
            parent_list=torch.empty((0,), dtype=torch.int64),
            top_scores_index=torch.empty((0,), dtype=torch.int64),
            draft_token_ids=torch.empty((0,), dtype=torch.int64),
            round_ids=[1],
            token_positions=[8],
            prefix_versions=[2],
        )

        root_worker = self._make_worker(rank=0, tp_size=4, draft_tp_size=1)
        root_future = _StaticFuture(response)
        root_worker._draft_forward_executor.submit.return_value = root_future
        request = SimpleNamespace(
            request_id="extend_after_decode:req-a",
            round_ids=[1],
            token_positions=[8],
            prefix_versions=[2],
        )

        future = root_worker._submit_draft_forward_async(
            request,
            timeout=3.0,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )

        self.assertIs(future, root_future)
        root_worker._draft_forward_executor.submit.assert_called_once_with(
            root_worker.client.draft_forward,
            request,
            timeout=3.0,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )

        non_root_worker = self._make_worker(rank=2, tp_size=4, draft_tp_size=1)

        future = non_root_worker._submit_draft_forward_async(
            request,
            timeout=3.0,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )

        self.assertIsNone(future)
        non_root_worker._draft_forward_executor.submit.assert_not_called()

    def test_resolve_async_draft_forward_broadcasts_root_result(self):
        worker = self._make_worker(rank=0, tp_size=4, draft_tp_size=1)
        response = DraftForwardResponse(
            request_id="extend_after_decode:req-a",
            parent_list=torch.empty((0,), dtype=torch.int64),
            top_scores_index=torch.empty((0,), dtype=torch.int64),
            draft_token_ids=torch.empty((0,), dtype=torch.int64),
            round_ids=[1],
            token_positions=[8],
            prefix_versions=[2],
        )
        request = SimpleNamespace(
            request_id="extend_after_decode:req-a",
            round_ids=[1],
            token_positions=[8],
            prefix_versions=[2],
        )
        pending = _PendingExtendAfterDecode(
            request=request,
            request_ids=["req-a"],
            spec_info=SimpleNamespace(),
            prefix_versions=[2],
            future=_StaticFuture(response),
            timeout=3.0,
        )

        with patch(
            "sglang.srt.speculative.remote_draft_worker.broadcast_pyobj",
            side_effect=lambda data, *args, **kwargs: data,
        ) as mock_broadcast:
            resolved = worker._resolve_draft_forward_async(pending)

        self.assertIs(resolved, response)
        mock_broadcast.assert_called_once()

    def test_resolve_async_draft_forward_receives_non_root_broadcast(self):
        worker = self._make_worker(rank=2, tp_size=4, draft_tp_size=1)
        response = DraftForwardResponse(
            request_id="extend_after_decode:req-a",
            parent_list=torch.empty((0,), dtype=torch.int64),
            top_scores_index=torch.empty((0,), dtype=torch.int64),
            draft_token_ids=torch.empty((0,), dtype=torch.int64),
            round_ids=[1],
            token_positions=[8],
            prefix_versions=[2],
        )
        pending = _PendingExtendAfterDecode(
            request=SimpleNamespace(
                request_id="extend_after_decode:req-a",
                round_ids=[1],
                token_positions=[8],
                prefix_versions=[2],
            ),
            request_ids=["req-a"],
            spec_info=SimpleNamespace(),
            prefix_versions=[2],
            future=None,
            timeout=3.0,
        )

        def fake_broadcast(data, rank, dist_group=None, src=0, force_cpu_device=True):
            del rank, dist_group, src, force_cpu_device
            if data:
                return data
            return [_BroadcastedDraftResult(ok=True, response=response)]

        with patch(
            "sglang.srt.speculative.remote_draft_worker.broadcast_pyobj",
            side_effect=fake_broadcast,
        ) as mock_broadcast:
            resolved = worker._resolve_draft_forward_async(pending)

        self.assertIs(resolved, response)
        mock_broadcast.assert_called_once()

    def test_drain_pending_extend_after_decode_only_matching_requests(self):
        worker = self._make_worker(rank=0, tp_size=1, draft_tp_size=1)
        pending_a = _PendingExtendAfterDecode(
            request=SimpleNamespace(request_id="extend_after_decode:req-a"),
            request_ids=["req-a"],
            spec_info=SimpleNamespace(),
            prefix_versions=[2],
            future=None,
            timeout=3.0,
        )
        pending_b = _PendingExtendAfterDecode(
            request=SimpleNamespace(request_id="extend_after_decode:req-b"),
            request_ids=["req-b"],
            spec_info=SimpleNamespace(),
            prefix_versions=[3],
            future=None,
            timeout=3.0,
        )
        worker._pending_extend_after_decode = [pending_a, pending_b]
        worker._finish_pending_extend_after_decode = Mock()

        worker._drain_pending_extend_after_decode_for_request_ids(["req-a"])

        worker._finish_pending_extend_after_decode.assert_called_once_with(pending_a)
        self.assertEqual(worker._pending_extend_after_decode, [pending_b])

    def test_drain_pending_extend_after_decode_keeps_pending_on_failure(self):
        worker = self._make_worker(rank=0, tp_size=1, draft_tp_size=1)
        pending_a = _PendingExtendAfterDecode(
            request=SimpleNamespace(request_id="extend_after_decode:req-a"),
            request_ids=["req-a"],
            spec_info=SimpleNamespace(),
            prefix_versions=[2],
            future=None,
            timeout=3.0,
        )
        pending_b = _PendingExtendAfterDecode(
            request=SimpleNamespace(request_id="extend_after_decode:req-b"),
            request_ids=["req-b"],
            spec_info=SimpleNamespace(),
            prefix_versions=[3],
            future=None,
            timeout=3.0,
        )
        worker._pending_extend_after_decode = [pending_a, pending_b]
        worker._finish_pending_extend_after_decode = Mock(
            side_effect=RuntimeError("boom")
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            worker._drain_pending_extend_after_decode_for_request_ids(["req-a"])

        self.assertEqual(worker._pending_extend_after_decode, [pending_a, pending_b])

    def test_finish_pending_extend_after_decode_commits_and_applies_state(self):
        worker = self._make_worker(rank=0, tp_size=1, draft_tp_size=1)
        response = DraftForwardResponse(
            request_id="extend_after_decode:req-a",
            parent_list=torch.empty((0,), dtype=torch.int64),
            top_scores_index=torch.empty((0,), dtype=torch.int64),
            draft_token_ids=torch.empty((0,), dtype=torch.int64),
        )
        spec_info = SimpleNamespace()
        pending = _PendingExtendAfterDecode(
            request=SimpleNamespace(request_id="extend_after_decode:req-a"),
            request_ids=["req-a"],
            spec_info=spec_info,
            prefix_versions=[4],
            future=None,
            timeout=3.0,
        )
        worker._resolve_draft_forward_async = Mock(return_value=response)
        worker._commit_prefix_versions = Mock()
        worker._apply_next_draft_state = Mock()

        worker._finish_pending_extend_after_decode(pending)

        worker._resolve_draft_forward_async.assert_called_once_with(pending)
        worker._commit_prefix_versions.assert_called_once_with(["req-a"], [4])
        worker._apply_next_draft_state.assert_called_once_with(
            spec_info,
            response,
            "extend_after_decode",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
