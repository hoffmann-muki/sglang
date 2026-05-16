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


def _request_source_candidate(obj, seen: set[int]):
    if obj is None:
        return None
    obj_id = id(obj)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    if hasattr(obj, "tli_draft_forward_communicator"):
        return getattr(obj, "tli_draft_forward_communicator")
    if hasattr(obj, "send_communicator_req"):
        return obj

    for attr_name in (
        "request_manager",
        "_request_manager",
        "server_state",
        "state",
        "scheduler",
        "tokenizer_manager",
        "manager",
        "app",
        "servicer",
        "communicator",
    ):
        if hasattr(obj, attr_name):
            candidate = _request_source_candidate(getattr(obj, attr_name), seen)
            if candidate is not None:
                return candidate
    return None


def _discover_request_source_communicator(request_source):
    seen: set[int] = set()
    candidate = _request_source_candidate(request_source, seen)
    if candidate is None:
        raise RuntimeError(
            "TLI draft request source does not expose a compatible communicator. "
            "Expected a tli_draft_forward_communicator callable or an object "
            "with send_communicator_req(...)."
        )
    return candidate


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


async def tokenizer_manager_backed_tli_draft_handler(
    tokenizer_manager,
    request: TLIDraftRequest,
    timeout: float | None = None,
) -> TLIDraftResponse:
    """Bridge DraftForward RPCs into the local scheduler communicator."""
    return await _draft_forward_via_request_source(
        tokenizer_manager,
        request,
        timeout=timeout,
    )


async def _draft_forward_via_request_source(
    request_source,
    request: TLIDraftRequest,
    timeout: float | None = None,
) -> TLIDraftResponse:
    communicator = _discover_request_source_communicator(request_source)
    if hasattr(communicator, "send_communicator_req"):
        results = await communicator.send_communicator_req(
            TLIDraftForwardReqInput(request=request),
            "tli_draft_forward_communicator",
            timeout=timeout,
        )
    else:
        results = await communicator(TLIDraftForwardReqInput(request=request))

    if not results:
        raise RuntimeError("TLI DraftForward received no scheduler response.")

    failures = [result for result in results if not result.success]
    if failures:
        messages = " | ".join(result.message for result in failures)
        raise RuntimeError(f"TLI DraftForward failed on draft scheduler: {messages}")

    matching_results = [
        result for result in results if result.tp_rank == request.tp_rank
    ]
    if len(matching_results) != 1:
        raise RuntimeError(
            "TLI DraftForward scheduler response did not include a unique payload "
            f"for tp_rank={request.tp_rank}. got_tp_ranks="
            f"{[result.tp_rank for result in results]}"
        )

    response = matching_results[0].response
    if response is None:
        raise RuntimeError(
            "TLI DraftForward scheduler response for the requested TP rank did not "
            f"include a payload (tp_rank={request.tp_rank})."
        )
    return response


async def start_tli_draft_service(
    request_source,
    server_args,
    *,
    request_handler: TLIDraftHandler | None = None,
):
    """Start the draft-side TLI gRPC sidecar from the serving runtime."""
    if not tli_draft_service_enabled(server_args):
        return None

    host, port = tli_draft_service_bind_addr(server_args)
    handler = request_handler
    if handler is None:

        async def _default_handler(request: TLIDraftRequest) -> TLIDraftResponse:
            return await _draft_forward_via_request_source(
                request_source,
                request,
                timeout=server_args.tli_rpc_timeout,
            )

        handler = _default_handler

    credentials = build_tli_server_credentials(server_args)
    server = await serve_tli_speculative_service(
        host=host,
        port=port,
        request_handler=handler,
        server_credentials=credentials,
    )
    scheme = "mTLS/TLS" if credentials is not None else "insecure"
    logger.info(
        "TLI DraftForward service started on %s:%d (%s).",
        host,
        port,
        scheme,
    )
    return server
