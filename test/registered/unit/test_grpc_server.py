import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_server import _call_serve_grpc_compat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestGrpcServerCompat(CustomTestCase):
    def test_compat_uses_supported_callback_name(self):
        seen = {}

        def fake_serve_grpc(server_args, model_info=None, request_manager_ready=None):
            seen["server_args"] = server_args
            seen["model_info"] = model_info
            seen["request_manager_ready"] = request_manager_ready
            return "ok"

        callback = lambda *args: args  # noqa: E731
        result = _call_serve_grpc_compat(
            fake_serve_grpc,
            SimpleNamespace(),
            model_info={"x": 1},
            on_request_manager_ready=callback,
        )

        self.assertEqual(result, "ok")
        self.assertIs(seen["request_manager_ready"], callback)

    def test_compat_passes_through_when_no_callback_requested(self):
        seen = {}

        def fake_serve_grpc(server_args, model_info=None):
            seen["server_args"] = server_args
            seen["model_info"] = model_info
            return "ok"

        result = _call_serve_grpc_compat(fake_serve_grpc, SimpleNamespace())

        self.assertEqual(result, "ok")
        self.assertIsNotNone(seen["server_args"])
        self.assertIsNone(seen["model_info"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
