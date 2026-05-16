import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.entrypoints.grpc_server import (
    _install_request_manager_capture,
    _install_tli_draft_forward_bridge,
    _start_smg_sidecars_when_ready,
)
from sglang.srt.managers.io_struct import (
    TLIDraftForwardReqInput,
    TLIDraftForwardReqOutput,
)
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _RequestManager:
    async def send_communicator_req(self, *args, **kwargs):
        return []


class _BridgeRequestManager:
    def __init__(self):
        self.server_args = SimpleNamespace(tli_rpc_timeout=1.0)
        self.send_to_scheduler = _FakeSchedulerSocket()
        self.recv_from_scheduler = self.send_to_scheduler

    async def send_communicator_req(self, *args, **kwargs):
        return []


class _FakeSchedulerSocket:
    def __init__(self):
        self.sent = []
        self.recv_queue = asyncio.Queue()

    def send_pyobj(self, obj):
        self.sent.append(obj)
        if isinstance(obj, TLIDraftForwardReqInput):
            self.recv_queue.put_nowait(
                TLIDraftForwardReqOutput(
                    rid=obj.rid,
                    success=True,
                    message="",
                    response=TLIDraftResponse(
                        request_id=obj.request.request_id,
                        parent_list=[],
                        top_scores_index=[],
                        draft_token_ids=[],
                    ),
                    tp_rank=obj.request.tp_rank,
                )
            )
            self.recv_queue.put_nowait({"kind": "normal"})

    async def recv_pyobj(self):
        return await self.recv_queue.get()


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
        request_manager = _BridgeRequestManager()

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
                    request_manager_future,
                    SimpleNamespace(router=SimpleNamespace(add_get=lambda *args, **kwargs: None, add_post=lambda *args, **kwargs: None)),
                    "127.0.0.1",
                    30001,
                )

        sidecar_runner, tli_runner, restore = asyncio.run(run_test())

        self.assertIn("sidecar", started)
        self.assertIn("tli", started)
        self.assertEqual(started["sidecar"][0], "127.0.0.1")
        self.assertEqual(started["sidecar"][1], 30001)
        self.assertIs(started["tli"][0], request_manager)
        self.assertTrue(callable(started["tli"][0].handle_tli_draft_forward))
        self.assertTrue(
            callable(started["tli"][0].tli_draft_forward_communicator)
        )
        self.assertIsNotNone(sidecar_runner)
        self.assertIsNotNone(tli_runner)
        self.assertTrue(callable(restore))

    def test_install_tli_draft_forward_bridge_round_trip(self):
        request_manager = _BridgeRequestManager()
        restore = _install_tli_draft_forward_bridge(request_manager)

        async def run_test():
            request = TLIDraftForwardReqInput(
                rid="draft-1",
                request=TLIDraftRequest(
                    request_id="req-1",
                    verified_id=torch.tensor([0, 1]),
                    hidden_states=torch.zeros(2, 3),
                    tp_rank=0,
                    tp_size=1,
                ),
            )

            async def consume_scheduler_output():
                first = await request_manager.recv_from_scheduler.recv_pyobj()
                second = await request_manager.recv_from_scheduler.recv_pyobj()
                return first, second

            consumer_task = asyncio.create_task(consume_scheduler_output())
            try:
                response = await request_manager.handle_tli_draft_forward(request)
                first, second = await consumer_task
            finally:
                restore()

            return response, first, second

        response, first, second = asyncio.run(run_test())

        self.assertEqual(response.rid, "draft-1")
        self.assertTrue(response.success)
        self.assertEqual(response.tp_rank, 0)
        self.assertEqual(first.rid, "draft-1")
        self.assertEqual(first.request.request_id, "req-1")
        self.assertEqual(second, {"kind": "normal"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
