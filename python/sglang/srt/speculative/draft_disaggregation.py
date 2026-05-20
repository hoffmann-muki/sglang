# SPDX-License-Identifier: Apache-2.0
"""Launcher helpers for disaggregated draft-forward speculative decoding."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Awaitable, Callable

from sglang.srt.managers.io_struct import DraftForwardReqInput
from sglang.srt.speculative.draft_forward_grpc_transport import (
    serve_draft_forward_service,
)
from sglang.srt.speculative.draft_forward_protocol import (
    DraftForwardRequest,
    DraftForwardResponse,
)

logger = logging.getLogger(__name__)

DraftForwardHandler = Callable[
    [DraftForwardRequest], DraftForwardResponse | Awaitable[DraftForwardResponse]
]


class DraftForwardExecutionNotWiredError(RuntimeError):
    """Raised when the DraftForward RPC is reachable but no draft executor is installed."""


def draft_disaggregation_enabled(server_args) -> bool:
    return server_args.draft_disaggregation_role != "none"


def draft_forward_service_enabled(server_args) -> bool:
    return server_args.draft_disaggregation_role == "draft"


def draft_forward_service_bind_addr(server_args) -> tuple[str, int]:
    host = server_args.draft_forward_service_host
    host = host or server_args.host
    port = server_args.draft_forward_service_port
    if port is None:
        raise ValueError("DraftForward service requires --draft-forward-service-port.")
    return host, int(port)


def _read_file(path: str | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).read_bytes()


_REQUEST_SOURCE_ATTRS = (
    "bootstrap_server",
    "request_manager",
    "scheduler",
    "tokenizer_manager",
    "servicer",
    "manager",
)


def _has_direct_draft_forward_communicator(obj) -> bool:
    return any(
        hasattr(obj, attr_name)
        for attr_name in (
            "handle_draft_forward",
            "draft_forward_communicator",
            "send_communicator_req",
        )
    )


def _request_source_candidate(obj):
    if obj is None:
        return None

    if _has_direct_draft_forward_communicator(obj):
        return obj

    for attr_name in _REQUEST_SOURCE_ATTRS:
        candidate = getattr(obj, attr_name, None)
        if candidate is not None and _has_direct_draft_forward_communicator(candidate):
            return candidate

    if inspect.ismodule(obj):
        for attr_name in _REQUEST_SOURCE_ATTRS:
            candidate = getattr(obj, attr_name, None)
            if candidate is not None and _has_direct_draft_forward_communicator(
                candidate
            ):
                return candidate
    return None


def _discover_request_source_communicator(request_source):
    candidate = _request_source_candidate(request_source)
    if candidate is None:
        raise RuntimeError(
            "DraftForward request source does not expose a compatible communicator. "
            "Expected a request_manager/module/object with a direct "
            "handle_draft_forward(...), draft_forward_communicator, or "
            "send_communicator_req(...)."
        )
    return candidate


def build_draft_forward_server_credentials(server_args):
    """Build grpc.ServerCredentials for draft-side TLS/mTLS, or None."""
    if not server_args.draft_forward_grpc_use_tls:
        return None

    import grpc

    private_key = _read_file(server_args.draft_forward_grpc_keyfile)
    certificate_chain = _read_file(server_args.draft_forward_grpc_certfile)
    root_certificates = _read_file(server_args.draft_forward_grpc_ca_certs)
    require_client_auth = root_certificates is not None
    return grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=root_certificates,
        require_client_auth=require_client_auth,
    )


def build_draft_forward_channel_credentials(server_args):
    """Build grpc.ChannelCredentials for target-side TLS/mTLS, or None."""
    if not server_args.draft_forward_grpc_use_tls:
        return None

    import grpc

    return grpc.ssl_channel_credentials(
        root_certificates=_read_file(server_args.draft_forward_grpc_ca_certs),
        private_key=_read_file(server_args.draft_forward_grpc_keyfile),
        certificate_chain=_read_file(server_args.draft_forward_grpc_certfile),
    )


async def unimplemented_draft_forward_handler(
    request: DraftForwardRequest,
) -> DraftForwardResponse:
    raise DraftForwardExecutionNotWiredError(
        "DraftForward RPC reached the draft node, but the draft-side model "
        "execution handler is not wired yet. This is intentional: returning an "
        "empty or synthetic DraftForwardResponse would corrupt speculative decoding "
        f"state. request_id={request.request_id!r}"
    )


async def request_manager_backed_draft_forward_handler(
    request_manager,
    request: DraftForwardRequest,
    timeout: float | None = None,
) -> DraftForwardResponse:
    """Bridge DraftForward RPCs into the local scheduler communicator."""
    return await _draft_forward_via_request_source(
        request_manager,
        request,
        timeout=timeout,
    )


async def tokenizer_manager_backed_draft_forward_handler(
    tokenizer_manager,
    request: DraftForwardRequest,
    timeout: float | None = None,
) -> DraftForwardResponse:
    """Bridge DraftForward RPCs through a tokenizer manager wrapper."""
    return await request_manager_backed_draft_forward_handler(
        tokenizer_manager,
        request,
        timeout=timeout,
    )


async def _draft_forward_via_request_source(
    request_source,
    request: DraftForwardRequest,
    timeout: float | None = None,
) -> DraftForwardResponse:
    communicator = _discover_request_source_communicator(request_source)
    if hasattr(communicator, "handle_draft_forward"):
        results = await communicator.handle_draft_forward(
            DraftForwardReqInput(request=request)
        )
        if not isinstance(results, list):
            results = [results]
    elif hasattr(communicator, "send_communicator_req"):
        results = await communicator.send_communicator_req(
            DraftForwardReqInput(request=request),
            "draft_forward_communicator",
            timeout=timeout,
        )
    else:
        results = await communicator(DraftForwardReqInput(request=request))
        if not isinstance(results, list):
            results = [results]

    if not results:
        raise RuntimeError("DraftForward received no scheduler response.")

    failures = [result for result in results if not result.success]
    if failures:
        messages = " | ".join(result.message for result in failures)
        logger.error(
            "DraftForward RPC failed request_id=%r messages=%s",
            request.request_id,
            messages,
        )
        raise RuntimeError(f"DraftForward failed on draft scheduler: {messages}")

    matching_results = [
        result for result in results if result.tp_rank == request.tp_rank
    ]
    if len(matching_results) != 1:
        raise RuntimeError(
            "DraftForward scheduler response did not include a unique payload "
            f"for tp_rank={request.tp_rank}. got_tp_ranks="
            f"{[result.tp_rank for result in results]}"
        )

    response = matching_results[0].response
    if response is None:
        raise RuntimeError(
            "DraftForward scheduler response for the requested TP rank did not "
            f"include a payload (tp_rank={request.tp_rank})."
        )
    return response


async def start_draft_forward_service(
    request_source,
    server_args,
    *,
    request_handler: DraftForwardHandler | None = None,
):
    """Start the draft-side DraftForward gRPC sidecar from the serving runtime."""
    if not draft_forward_service_enabled(server_args):
        return None

    host, port = draft_forward_service_bind_addr(server_args)
    handler = request_handler
    if handler is None:

        async def _default_handler(
            request: DraftForwardRequest,
        ) -> DraftForwardResponse:
            return await _draft_forward_via_request_source(
                request_source,
                request,
                timeout=server_args.draft_forward_rpc_timeout,
            )

        handler = _default_handler

    credentials = build_draft_forward_server_credentials(server_args)
    server = await serve_draft_forward_service(
        host=host,
        port=port,
        request_handler=handler,
        server_credentials=credentials,
        stream_batch_window_s=getattr(
            server_args,
            "draft_forward_stream_batch_window_ms",
            0.0,
        )
        / 1000.0,
        stream_batch_max_requests=getattr(
            server_args,
            "draft_forward_stream_batch_max_requests",
            1,
        ),
        stream_batch_max_proposed_tokens=getattr(
            server_args,
            "draft_forward_stream_batch_max_proposed_tokens",
            2,
        ),
    )
    scheme = "mTLS/TLS" if credentials is not None else "insecure"
    logger.info(
        "DraftForward service started on %s:%d (%s).",
        host,
        port,
        scheme,
    )
    return server
