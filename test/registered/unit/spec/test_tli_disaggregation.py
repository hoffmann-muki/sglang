import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative import tli_disaggregation
from sglang.srt.speculative.tli_disaggregation import (
    TLIDraftExecutionNotWiredError,
    make_tli_service_ready_callback,
    tli_draft_service_bind_addr,
)
from sglang.srt.speculative.tli_protocol import TLIDraftRequest
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_TEST_TLI_SERVICE_PORT = 32001


class TestTLIDisaggregation(CustomTestCase):
    def _server_args(self, **kwargs):
        args = dict(
            host="127.0.0.1",
            tli_disaggregation_role="draft",
            tli_service_host=None,
            tli_service_port=_TEST_TLI_SERVICE_PORT,
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

    def test_unimplemented_handler_fails_closed(self):
        request = TLIDraftRequest(
            request_id="req-closed",
            verified_id=None,
            hidden_states=None,
        )
        with self.assertRaisesRegex(
            TLIDraftExecutionNotWiredError,
            "draft-side model execution handler is not wired",
        ):
            asyncio.run(tli_disaggregation.unimplemented_tli_draft_handler(request))

    def test_make_callback_starts_service(self):
        async def fake_serve_tli_speculative_service(**kwargs):
            return SimpleNamespace(kwargs=kwargs, stopped=False)

        with patch.object(
            tli_disaggregation,
            "serve_tli_speculative_service",
            fake_serve_tli_speculative_service,
        ):
            callback = make_tli_service_ready_callback(self._server_args())
            server = asyncio.run(callback(None, None, None))

        self.assertEqual(server.kwargs["host"], "127.0.0.1")
        self.assertEqual(server.kwargs["port"], _TEST_TLI_SERVICE_PORT)
        self.assertIs(
            server.kwargs["request_handler"],
            tli_disaggregation.unimplemented_tli_draft_handler,
        )
        self.assertIsNone(server.kwargs["server_credentials"])

    def test_make_callback_ignores_non_draft_role(self):
        callback = make_tli_service_ready_callback(
            self._server_args(tli_disaggregation_role="target")
        )
        self.assertIsNone(callback)


if __name__ == "__main__":
    unittest.main(verbosity=2)
