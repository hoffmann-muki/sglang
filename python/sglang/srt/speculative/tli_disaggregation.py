# SPDX-License-Identifier: Apache-2.0
"""Launcher helpers for disaggregated TLI speculative decoding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Awaitable, Callable

from sglang.srt.managers.io_struct import TLIDraftForwardReqInput
from sglang.srt.speculative.tli_grpc_transport import serve_tli_speculative_service
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse

logger = logging.getLogger(__name__)

TLIDraftHandler = Callable[
    [TLIDraftRequest], TLIDraftResponse | Awaitable[TLIDraftResponse]
]


class TLIDraftExecutionNotWiredError(RuntimeError):
    """Raised when the TLI RPC is reachable but no draft executor is installed."""


def tli_disaggregation_enabled(server_args) -> bool:
    return getattr(server_args, "tli_disaggregation_role", "none") != "none"


def tli_draft_service_enabled(server_args) -> bool:
    return getattr(server_args, "tli_disaggregation_role", "none") == "draft"


def tli_draft_service_bind_addr(server_args) -> tuple[str, int]:
    host = server_args.tli_service_host or server_args.host
    port = server_args.tli_service_port
    if port is None:
        raise ValueError("TLI draft service requires --tli-service-port.")
    return host, int(port)


def _read_file(path: str | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).read_bytes()


def build_tli_server_credentials(server_args):
    """Build grpc.ServerCredentials for draft-side TLS/mTLS, or None."""
    if not server_args.tli_grpc_use_tls:
        return None

    import grpc

    private_key = _read_file(server_args.tli_grpc_keyfile)
    certificate_chain = _read_file(server_args.tli_grpc_certfile)
    root_certificates = _read_file(server_args.tli_grpc_ca_certs)
    require_client_auth = root_certificates is not None
    return grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=root_certificates,
        require_client_auth=require_client_auth,
    )


def build_tli_channel_credentials(server_args):
    """Build grpc.ChannelCredentials for target-side TLS/mTLS, or None."""
    if not server_args.tli_grpc_use_tls:
        return None

    import grpc

    return grpc.ssl_channel_credentials(
        root_certificates=_read_file(server_args.tli_grpc_ca_certs),
        private_key=_read_file(server_args.tli_grpc_keyfile),
        certificate_chain=_read_file(server_args.tli_grpc_certfile),
    )


async def unimplemented_tli_draft_handler(
    request: TLIDraftRequest,
) -> TLIDraftResponse:
    raise TLIDraftExecutionNotWiredError(
        "TLI DraftForward RPC reached the draft node, but the draft-side model "
        "execution handler is not wired yet. This is intentional: returning an "
        "empty or synthetic TLIDraftResponse would corrupt speculative decoding "
        f"state. request_id={request.request_id!r}"
    )


async def scheduler_backed_tli_draft_handler(
    request_manager,
    request: TLIDraftRequest,
    timeout: float | None = None,
) -> TLIDraftResponse:
    results = await request_manager.send_communicator_req(
        TLIDraftForwardReqInput(request=request),
        "tli_draft_forward_communicator",
        timeout=timeout,
    )
    if not results:
        raise RuntimeError("TLI DraftForward received no scheduler response.")

    failures = [result for result in results if not result.success]
    if failures:
        messages = " | ".join(result.message for result in failures)
        raise RuntimeError(f"TLI DraftForward failed on draft scheduler: {messages}")

    for result in results:
        if result.tp_rank == request.tp_rank and result.response is not None:
            return result.response

    for result in results:
        if result.response is not None:
            return result.response

    raise RuntimeError("TLI DraftForward scheduler response did not include a payload.")


def make_tli_service_ready_callback(
    server_args,
    *,
    request_handler: TLIDraftHandler | None = None,
):
    """Create the gRPC launch hook used by the disaggregated draft node.

    The normal SGLang gRPC launcher calls this hook once the request manager is
    ready. Today this starts the typed TLI DraftForward service and installs a
    fail-closed handler until the scheduler-backed draft executor is available.
    """
    if not tli_draft_service_enabled(server_args):
        return None

    async def _on_ready(request_manager, srv_args, sched_info):
        del srv_args, sched_info

        host, port = tli_draft_service_bind_addr(server_args)
        handler = request_handler
        if handler is None:
            async def handler(request: TLIDraftRequest) -> TLIDraftResponse:
                return await scheduler_backed_tli_draft_handler(
                    request_manager,
                    request,
                    timeout=server_args.tli_rpc_timeout,
                )

        credentials = build_tli_server_credentials(server_args)
        server = await serve_tli_speculative_service(
            host=host,
            port=port,
            request_handler=handler,
            server_credentials=credentials,
        )
        scheme = "mTLS/TLS" if credentials is not None else "insecure"
        logger.info(
            "TLI DraftForward gRPC service started on %s:%d (%s).",
            host,
            port,
            scheme,
        )
        if request_handler is None:
            logger.warning(
                "TLI DraftForward service is using the scheduler-backed handler. "
                "Draft model execution remains fail-closed until the scheduler "
                "executor supports SGLang batch/KV reconstruction."
            )
        return server

    return _on_ready
