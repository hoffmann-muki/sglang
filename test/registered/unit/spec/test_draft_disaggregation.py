import asyncio
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import draft_disaggregation
from sglang.srt.speculative.remote_draft_executor import (
    RemoteDraftExecutorNotReadyError,
    RemoteDraftRequestState,
    RemoteDraftSchedulerExecutor,
    _slice_extend_hidden_states_for_prefixes,
)
from sglang.srt.speculative.eagle_utils import (
    _normalize_verified_id_for_tree_build,
)
from sglang.srt.speculative.draft_disaggregation import (
    DraftForwardExecutionNotWiredError,
    start_draft_forward_service,
    request_manager_backed_draft_forward_handler,
    draft_forward_service_bind_addr,
    draft_forward_service_enabled,
    unimplemented_draft_forward_handler,
)
from sglang.srt.speculative.draft_forward_protocol import (
    DraftForwardRequest,
    DraftForwardResponse,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_TEST_DRAFT_FORWARD_SERVICE_PORT = 32001


@dataclass
class _FakeSchedulerResult:
    success: bool
    message: str = ""
    response: DraftForwardResponse | None = None
    tp_rank: int = 0


class _FakeServer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.stop_calls = []

    async def stop(self, grace=0):
        self.stop_calls.append(grace)


class TestDraftDisaggregation(CustomTestCase):
    def _server_args(self, **kwargs):
        args = dict(
            host="127.0.0.1",
            draft_disaggregation_role="draft",
            draft_forward_service_host=None,
            draft_forward_service_port=_TEST_DRAFT_FORWARD_SERVICE_PORT,
            draft_forward_rpc_timeout=3.0,
            draft_forward_grpc_use_tls=False,
            draft_forward_grpc_keyfile=None,
            draft_forward_grpc_certfile=None,
            draft_forward_grpc_ca_certs=None,
        )
        args.update(kwargs)
        return SimpleNamespace(**args)

    def test_draft_service_bind_addr_defaults_to_server_host(self):
        self.assertEqual(
            draft_forward_service_bind_addr(self._server_args()),
            ("127.0.0.1", _TEST_DRAFT_FORWARD_SERVICE_PORT),
        )

    def test_draft_service_enabled_only_for_draft_role(self):
        self.assertTrue(draft_forward_service_enabled(self._server_args()))
        self.assertFalse(
            draft_forward_service_enabled(
                self._server_args(draft_disaggregation_role="target")
            )
        )

    def test_unimplemented_handler_fails_closed(self):
        request = DraftForwardRequest(
            request_id="req-closed",
            verified_id=torch.tensor([1]),
            hidden_states=torch.zeros(1, 1),
        )
        with self.assertRaisesRegex(
            DraftForwardExecutionNotWiredError,
            "draft-side model execution handler is not wired",
        ):
            asyncio.run(unimplemented_draft_forward_handler(request))

    def test_draft_executor_empty_response_echoes_ordering_metadata(self):
        request = DraftForwardRequest(
            request_id="release:req-1",
            request_ids=["req-1"],
            verified_id=torch.empty((0,), dtype=torch.int64),
            hidden_states=torch.empty((0,), dtype=torch.float32),
            mode="release",
            round_ids=[3],
            token_positions=[11],
            prefix_versions=[7],
        )

        response = RemoteDraftSchedulerExecutor._empty_response(request)

        self.assertEqual(response.round_ids, [3])
        self.assertEqual(response.token_positions, [11])
        self.assertEqual(response.prefix_versions, [7])

    def test_tokenizer_manager_backed_handler_round_trip(self):
        request = DraftForwardRequest(
            request_id="req-1",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = DraftForwardResponse(
            request_id="req-1",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def communicator(req):
            self.assertEqual(req.request.request_id, request.request_id)
            self.assertTrue(
                torch.equal(req.request.hidden_states, request.hidden_states)
            )
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        tokenizer_manager = SimpleNamespace(draft_forward_communicator=communicator)
        response = asyncio.run(
            request_manager_backed_draft_forward_handler(tokenizer_manager, request)
        )

        self.assertEqual(response.request_id, "req-1")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(
            torch.equal(response.top_scores_index, expected.top_scores_index)
        )
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_request_manager_backed_handler_round_trip(self):
        request = DraftForwardRequest(
            request_id="req-2",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = DraftForwardResponse(
            request_id="req-2",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def handle_draft_forward(req):
            self.assertEqual(req.request.request_id, request.request_id)
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        request_manager = SimpleNamespace(handle_draft_forward=handle_draft_forward)
        response = asyncio.run(
            request_manager_backed_draft_forward_handler(
                request_manager, request, timeout=3.0
            )
        )

        self.assertEqual(response.request_id, "req-2")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(
            torch.equal(response.top_scores_index, expected.top_scores_index)
        )
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_request_source_wrapper_backed_handler_round_trip(self):
        request = DraftForwardRequest(
            request_id="req-3",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = DraftForwardResponse(
            request_id="req-3",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def handle_draft_forward(req):
            self.assertEqual(req.request.request_id, request.request_id)
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        nested_source = SimpleNamespace(
            request_manager=SimpleNamespace(handle_draft_forward=handle_draft_forward)
        )
        response = asyncio.run(
            request_manager_backed_draft_forward_handler(nested_source, request)
        )

        self.assertEqual(response.request_id, "req-3")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(
            torch.equal(response.top_scores_index, expected.top_scores_index)
        )
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_remote_draft_executor_reports_held_state_counts(self):
        executor = RemoteDraftSchedulerExecutor.__new__(RemoteDraftSchedulerExecutor)
        executor.server_args = SimpleNamespace(page_size=8)
        executor.tree_cache = SimpleNamespace(supports_swa=lambda: True)
        executor.states = {
            (
                "req-1",
                0,
            ): RemoteDraftRequestState(
                request_id="req-1",
                tp_rank=0,
                req=SimpleNamespace(
                    req_pool_idx=3,
                    kv_allocated_len=17,
                    cache_protected_len=5,
                    swa_evicted_seqlen=7,
                ),
                seq_len=17,
            ),
            (
                "req-2",
                0,
            ): RemoteDraftRequestState(
                request_id="req-2",
                tp_rank=0,
                req=SimpleNamespace(
                    req_pool_idx=9,
                    kv_allocated_len=8,
                    cache_protected_len=0,
                    swa_evicted_seqlen=2,
                ),
                seq_len=8,
            ),
        }

        self.assertEqual(executor.held_full_tokens(), 27)
        self.assertEqual(executor.held_swa_tokens(), 23)
        self.assertEqual(executor.held_req_count(), 2)

    def test_extend_hidden_state_slicing_uses_target_and_draft_prefixes(self):
        hidden_states = torch.arange(4, dtype=torch.float32).reshape(4, 1)

        aligned = _slice_extend_hidden_states_for_prefixes(
            hidden_states,
            seq_lens=[5],
            target_prefix_lens=[1],
            draft_prefix_lens=[3],
        )

        self.assertTrue(torch.equal(aligned, torch.tensor([[2.0], [3.0]])))

    def test_extend_hidden_state_slicing_supports_batched_requests(self):
        hidden_states = torch.arange(7, dtype=torch.float32).reshape(7, 1)

        aligned = _slice_extend_hidden_states_for_prefixes(
            hidden_states,
            seq_lens=[5, 4],
            target_prefix_lens=[1, 2],
            draft_prefix_lens=[3, 2],
        )

        self.assertTrue(
            torch.equal(aligned, torch.tensor([[2.0], [3.0], [4.0], [5.0]]))
        )

    def test_extend_hidden_state_slicing_rejects_missing_draft_prefix(self):
        hidden_states = torch.arange(3, dtype=torch.float32).reshape(3, 1)

        with self.assertRaisesRegex(
            RemoteDraftExecutorNotReadyError,
            "draft node does not own the same prefix",
        ):
            _slice_extend_hidden_states_for_prefixes(
                hidden_states,
                seq_lens=[5],
                target_prefix_lens=[2],
                draft_prefix_lens=[1],
            )

    def test_extend_hidden_state_slicing_rejects_bad_target_metadata(self):
        hidden_states = torch.arange(3, dtype=torch.float32).reshape(3, 1)

        with self.assertRaisesRegex(
            RemoteDraftExecutorNotReadyError,
            "do not match target prefix metadata",
        ):
            _slice_extend_hidden_states_for_prefixes(
                hidden_states,
                seq_lens=[5],
                target_prefix_lens=[1],
                draft_prefix_lens=[1],
            )

    def test_start_draft_forward_service_starts_sidecar(self):
        fake_server = _FakeServer(
            kwargs={
                "host": "127.0.0.1",
                "port": _TEST_DRAFT_FORWARD_SERVICE_PORT,
            }
        )

        async def fake_serve_draft_forward_service(**kwargs):
            fake_server.kwargs = kwargs
            return fake_server

        request_manager = SimpleNamespace(handle_draft_forward=lambda req: [])

        with patch.object(
            draft_disaggregation,
            "serve_draft_forward_service",
            fake_serve_draft_forward_service,
        ):
            server = asyncio.run(
                start_draft_forward_service(request_manager, self._server_args())
            )

        self.assertIs(server, fake_server)
        self.assertEqual(server.kwargs["host"], "127.0.0.1")
        self.assertEqual(server.kwargs["port"], _TEST_DRAFT_FORWARD_SERVICE_PORT)
        self.assertTrue(callable(server.kwargs["request_handler"]))
        self.assertIsNone(server.kwargs["server_credentials"])

    def test_start_draft_forward_service_ignores_non_draft_role(self):
        server = asyncio.run(
            start_draft_forward_service(
                SimpleNamespace(handle_draft_forward=lambda req: []),
                self._server_args(draft_disaggregation_role="target"),
            )
        )
        self.assertIsNone(server)

    def test_verified_id_normalization_uses_last_token_per_row(self):
        draft_tokens = torch.zeros((1, 4), dtype=torch.int64)
        verified_id = torch.tensor([11, 22], dtype=torch.int64)

        normalized = _normalize_verified_id_for_tree_build(verified_id, draft_tokens)

        self.assertTrue(torch.equal(normalized, torch.tensor([22], dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
