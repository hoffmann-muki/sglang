import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.grpc_server import (
    _install_request_manager_capture,
    _start_smg_sidecars_when_ready,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _RequestManager:
    async def send_communicator_req(self, *args, **kwargs):
        return []


class _Scheduler:
    async def handle_tli_draft_forward(self, *args, **kwargs):
        return []


class TestGrpcServerCompat(CustomTestCase):
    def test_install_request_manager_capture_resolves_on_constructor(self):
        class _GrpcRequestManager(_RequestManager):
            pass

        async def run_test():
            request_manager = None
            with patch(
                "sglang.srt.entrypoints.grpc_server._iter_request_manager_classes",
                return_value=[_GrpcRequestManager],
            ):
                future, restore = _install_request_manager_capture(SimpleNamespace())
                try:
                    request_manager = _GrpcRequestManager()
                    self.assertIs(await future, request_manager)
                finally:
                    restore()

        asyncio.run(run_test())

    def test_start_smg_sidecars_when_ready_starts_both_sidecars(self):
        request_manager = SimpleNamespace(scheduler=_Scheduler())
        scheduler = _Scheduler()

        started = {}

        async def fake_start_sidecar_server(host, port, app):
            started["sidecar"] = (host, port, app)
            return SimpleNamespace(cleanup=lambda: None)

        async def fake_start_tli_draft_service(source, server_args, request_handler=None):
            started["tli"] = (source, server_args, request_handler)
            return SimpleNamespace(stop=lambda grace=0: None)

        async def run_test():
            request_manager_future = asyncio.get_running_loop().create_future()
            request_manager_future.set_result(request_manager)
            scheduler_future = asyncio.get_running_loop().create_future()
            scheduler_future.set_result(scheduler)

            with patch(
                "sglang.srt.entrypoints.grpc_server._start_sidecar_server",
                fake_start_sidecar_server,
            ), patch(
                "sglang.srt.entrypoints.grpc_server.start_tli_draft_service",
                fake_start_tli_draft_service,
            ):
                return await _start_smg_sidecars_when_ready(
                    SimpleNamespace(
                        host="127.0.0.1",
                        tli_disaggregation_role="draft",
                        tli_service_port=32001,
                    ),
                    scheduler_future,
                    request_manager_future,
                    SimpleNamespace(router=SimpleNamespace(add_get=lambda *args, **kwargs: None, add_post=lambda *args, **kwargs: None)),
                    "127.0.0.1",
                    30001,
                )

        sidecar_runner, tli_runner = asyncio.run(run_test())

        self.assertIn("sidecar", started)
        self.assertIn("tli", started)
        self.assertEqual(started["sidecar"][0], "127.0.0.1")
        self.assertEqual(started["sidecar"][1], 30001)
        self.assertIs(started["tli"][0], scheduler)
        self.assertIsNotNone(sidecar_runner)
        self.assertIsNotNone(tli_runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
