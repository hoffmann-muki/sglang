import asyncio
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import tli_disaggregation
from sglang.srt.speculative.tli_disaggregation import (
    TLIDraftExecutionNotWiredError,
    start_tli_draft_service,
    tokenizer_manager_backed_tli_draft_handler,
    tli_draft_service_bind_addr,
    tli_draft_service_enabled,
    unimplemented_tli_draft_handler,
)
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_TEST_TLI_SERVICE_PORT = 32001


@dataclass
class _FakeSchedulerResult:
    success: bool
    message: str = ""
    response: TLIDraftResponse | None = None
    tp_rank: int = 0


class _FakeServer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.stop_calls = []

    async def stop(self, grace=0):
        self.stop_calls.append(grace)


class TestTLIDisaggregation(CustomTestCase):
    def _server_args(self, **kwargs):
        args = dict(
            host="127.0.0.1",
            tli_disaggregation_role="draft",
            tli_service_host=None,
            tli_service_port=_TEST_TLI_SERVICE_PORT,
            tli_rpc_timeout=3.0,
            tli_grpc_use_tls=False,
            tli_grpc_keyfile=None,
            tli_grpc_certfile=None,
            tli_grpc_ca_certs=None,
        )
        args.update(kwargs)
        return SimpleNamespace(**args)

    def test_draft_service_bind_addr_defaults_to_server_host(self):
        self.assertEqual(
            tli_draft_service_bind_addr(self._server_args()),
            ("127.0.0.1", _TEST_TLI_SERVICE_PORT),
        )

    def test_draft_service_enabled_only_for_draft_role(self):
        self.assertTrue(tli_draft_service_enabled(self._server_args()))
        self.assertFalse(
            tli_draft_service_enabled(
                self._server_args(tli_disaggregation_role="target")
            )
        )

    def test_unimplemented_handler_fails_closed(self):
        request = TLIDraftRequest(
            request_id="req-closed",
            verified_id=torch.tensor([1]),
            hidden_states=torch.zeros(1, 1),
        )
        with self.assertRaisesRegex(
            TLIDraftExecutionNotWiredError,
            "draft-side model execution handler is not wired",
        ):
            asyncio.run(unimplemented_tli_draft_handler(request))

    def test_tokenizer_manager_backed_handler_round_trip(self):
        request = TLIDraftRequest(
            request_id="req-1",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = TLIDraftResponse(
            request_id="req-1",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def communicator(req):
            self.assertEqual(req.request.request_id, request.request_id)
            self.assertTrue(torch.equal(req.request.hidden_states, request.hidden_states))
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        tokenizer_manager = SimpleNamespace(
            tli_draft_forward_communicator=communicator
        )
        response = asyncio.run(
            tokenizer_manager_backed_tli_draft_handler(tokenizer_manager, request)
        )

        self.assertEqual(response.request_id, "req-1")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(torch.equal(response.top_scores_index, expected.top_scores_index))
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_request_manager_backed_handler_round_trip(self):
        request = TLIDraftRequest(
            request_id="req-2",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = TLIDraftResponse(
            request_id="req-2",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def send_communicator_req(req, communicator_name, timeout=None):
            self.assertEqual(communicator_name, "tli_draft_forward_communicator")
            self.assertEqual(timeout, 3.0)
            self.assertEqual(req.request.request_id, request.request_id)
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        request_manager = SimpleNamespace(
            send_communicator_req=send_communicator_req
        )
        response = asyncio.run(
            tokenizer_manager_backed_tli_draft_handler(request_manager, request, timeout=3.0)
        )

        self.assertEqual(response.request_id, "req-2")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(torch.equal(response.top_scores_index, expected.top_scores_index))
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_nested_request_source_backed_handler_round_trip(self):
        request = TLIDraftRequest(
            request_id="req-3",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = TLIDraftResponse(
            request_id="req-3",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def tli_draft_forward_communicator(req):
            self.assertEqual(req.request.request_id, request.request_id)
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        nested_source = SimpleNamespace(
            state=SimpleNamespace(
                manager=SimpleNamespace(
                    tli_draft_forward_communicator=tli_draft_forward_communicator
                )
            )
        )
        response = asyncio.run(
            tokenizer_manager_backed_tli_draft_handler(nested_source, request)
        )

        self.assertEqual(response.request_id, "req-3")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(torch.equal(response.top_scores_index, expected.top_scores_index))
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_tuple_request_source_backed_handler_round_trip(self):
        request = TLIDraftRequest(
            request_id="req-4",
            verified_id=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 3),
            tp_rank=0,
            tp_size=1,
        )
        expected = TLIDraftResponse(
            request_id="req-4",
            parent_list=torch.tensor([[0, 1]]),
            top_scores_index=torch.tensor([[1, 0]]),
            draft_token_ids=torch.tensor([0, 1]),
        )

        async def tli_draft_forward_communicator(req):
            self.assertEqual(req.request.request_id, request.request_id)
            return [
                _FakeSchedulerResult(
                    success=True,
                    response=expected,
                    tp_rank=0,
                )
            ]

        nested_source = (
            SimpleNamespace(),
            SimpleNamespace(
                state=SimpleNamespace(
                    manager=SimpleNamespace(
                        tli_draft_forward_communicator=tli_draft_forward_communicator
                    )
                )
            ),
        )
        response = asyncio.run(
            tokenizer_manager_backed_tli_draft_handler(nested_source, request)
        )

        self.assertEqual(response.request_id, "req-4")
        self.assertTrue(torch.equal(response.parent_list, expected.parent_list))
        self.assertTrue(torch.equal(response.top_scores_index, expected.top_scores_index))
        self.assertTrue(torch.equal(response.draft_token_ids, expected.draft_token_ids))

    def test_start_tli_draft_service_starts_sidecar(self):
        fake_server = _FakeServer(
            kwargs={
                "host": "127.0.0.1",
                "port": _TEST_TLI_SERVICE_PORT,
            }
        )

        async def fake_serve_tli_speculative_service(**kwargs):
            fake_server.kwargs = kwargs
            return fake_server

        tokenizer_manager = SimpleNamespace(
            tli_draft_forward_communicator=lambda req: []
        )

        with patch.object(
            tli_disaggregation,
            "serve_tli_speculative_service",
            fake_serve_tli_speculative_service,
        ):
            server = asyncio.run(
                start_tli_draft_service(tokenizer_manager, self._server_args())
            )

        self.assertIs(server, fake_server)
        self.assertEqual(server.kwargs["host"], "127.0.0.1")
        self.assertEqual(server.kwargs["port"], _TEST_TLI_SERVICE_PORT)
        self.assertTrue(callable(server.kwargs["request_handler"]))
        self.assertIsNone(server.kwargs["server_credentials"])

    def test_start_tli_draft_service_ignores_non_draft_role(self):
        server = asyncio.run(
            start_tli_draft_service(
                SimpleNamespace(tli_draft_forward_communicator=lambda req: []),
                self._server_args(tli_disaggregation_role="target"),
            )
        )
        self.assertIsNone(server)


if __name__ == "__main__":
    unittest.main(verbosity=2)
