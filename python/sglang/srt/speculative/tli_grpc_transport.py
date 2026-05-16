# SPDX-License-Identifier: Apache-2.0
"""gRPC transport helpers for disaggregated TLI speculative decoding."""

from __future__ import annotations

import ctypes
import inspect
import logging
from importlib import import_module
from typing import Awaitable, Callable

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse
from sglang.srt.speculative.tli_token_translator import TLITokenTranslator

logger = logging.getLogger(__name__)

TLIDraftHandler = Callable[
    [TLIDraftRequest], TLIDraftResponse | Awaitable[TLIDraftResponse]
]

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _import_proto_modules():
    candidates = (
        (
            "sglang.runtime.v1.sglang_pb2",
            "sglang.runtime.v1.sglang_pb2_grpc",
        ),
        (
            "smg_grpc_proto.sglang_pb2",
            "smg_grpc_proto.sglang_pb2_grpc",
        ),
        (
            "sglang.srt.grpc.sglang_pb2",
            "sglang.srt.grpc.sglang_pb2_grpc",
        ),
    )
    errors = []
    for pb2_name, pb2_grpc_name in candidates:
        try:
            return import_module(pb2_name), import_module(pb2_grpc_name)
        except ImportError as exc:
            errors.append(f"{pb2_name} / {pb2_grpc_name}: {exc}")
    raise ImportError(
        "Unable to import generated TLI gRPC modules. Tried:\n"
        + "\n".join(errors)
    )


def _resolve_proto_module(proto_module=None):
    if proto_module is not None:
        return proto_module
    sglang_pb2, _ = _import_proto_modules()
    return sglang_pb2


def _resolve_grpc_module():
    _, sglang_pb2_grpc = _import_proto_modules()
    return sglang_pb2_grpc


def _dtype_to_str(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_STR[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype for TLI RPC: {dtype}") from exc


def _str_to_dtype(name: str) -> torch.dtype:
    try:
        return _STR_TO_DTYPE[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype for TLI RPC: {name}") from exc


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.detach().contiguous()
    total_bytes = tensor.numel() * tensor.element_size()
    if total_bytes == 0:
        return b""
    c_buf = (ctypes.c_char * total_bytes).from_address(tensor.data_ptr())
    return bytes(memoryview(c_buf))


def _tensor_from_bytes(
    *,
    shape: list[int] | tuple[int, ...],
    dtype: str,
    data: bytes,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    # `torch.frombuffer` warns on immutable bytes objects; copy into a writable
    # bytearray first since we immediately clone anyway.
    tensor = torch.frombuffer(
        bytearray(data), dtype=_str_to_dtype(dtype)
    ).reshape(list(shape))
    tensor = tensor.clone()
    if device != "cpu" and device != torch.device("cpu"):
        tensor = tensor.to(device)
    return tensor


def tensor_to_proto_tensor(tensor: torch.Tensor, proto_module=None):
    """Encode a torch tensor into the protobuf tensor envelope."""
    proto_module = _resolve_proto_module(proto_module)
    return proto_module.TensorData(
        shape=list(tensor.shape),
        dtype=_dtype_to_str(tensor.dtype),
        data=_tensor_to_bytes(tensor),
    )


def tensor_from_proto_tensor(proto_tensor, device: str | torch.device = "cpu"):
    """Decode a protobuf tensor envelope into a torch tensor."""
    if proto_tensor is None:
        return None
    return _tensor_from_bytes(
        shape=list(proto_tensor.shape),
        dtype=proto_tensor.dtype,
        data=bytes(proto_tensor.data),
        device=device,
    )


def _capture_hidden_mode_to_proto(mode: CaptureHiddenMode) -> int:
    return int(mode if mode is not None else CaptureHiddenMode.NULL)


def _capture_hidden_mode_from_proto(value) -> CaptureHiddenMode:
    return CaptureHiddenMode(int(value))


def _message_has_field(message, field_name: str) -> bool:
    has_field = getattr(message, "HasField", None)
    if callable(has_field):
        try:
            return has_field(field_name)
        except (ValueError, AttributeError):
            pass
    value = getattr(message, field_name, None)
    return value is not None


def _tensor_debug_summary(tensor) -> str:
    if tensor is None:
        return "None"
    shape = tuple(getattr(tensor, "shape", ()))
    dtype = getattr(tensor, "dtype", None)
    device = getattr(tensor, "device", None)
    numel = getattr(tensor, "numel", None)
    numel = numel() if callable(numel) else None
    return f"shape={shape} dtype={dtype} device={device} numel={numel}"


def _draft_request_debug_summary(request: TLIDraftRequest) -> str:
    return (
        f"request_id={request.request_id!r} mode={request.mode} "
        f"tp_rank={request.tp_rank} tp_size={request.tp_size} "
        f"request_ids={request.request_ids} "
        f"verified_id={_tensor_debug_summary(request.verified_id)} "
        f"hidden_states={_tensor_debug_summary(request.hidden_states)} "
        f"input_ids={_tensor_debug_summary(request.input_ids)} "
        f"accept_length={_tensor_debug_summary(request.accept_length)} "
        f"seq_lens_for_draft_extend={_tensor_debug_summary(request.seq_lens_for_draft_extend)} "
        f"mm_input_embeds={_tensor_debug_summary(request.mm_input_embeds)}"
    )


def _draft_response_debug_summary(response: TLIDraftResponse) -> str:
    return (
        f"request_id={response.request_id!r} mode={response.mode} "
        f"parent_list={_tensor_debug_summary(response.parent_list)} "
        f"top_scores_index={_tensor_debug_summary(response.top_scores_index)} "
        f"draft_token_ids={_tensor_debug_summary(response.draft_token_ids)} "
        f"next_hidden_states={_tensor_debug_summary(response.next_hidden_states)} "
        f"next_topk_p={_tensor_debug_summary(response.next_topk_p)} "
        f"next_topk_index={_tensor_debug_summary(response.next_topk_index)}"
    )


def draft_request_to_proto(
    request: TLIDraftRequest,
    *,
    translator: TLITokenTranslator | None = None,
    translate_to_draft_vocab: bool = True,
    proto_module=None,
):
    """Convert a request dataclass into the protobuf request envelope."""
    proto_module = _resolve_proto_module(proto_module)
    if translator is not None and translate_to_draft_vocab:
        request = request.to_draft_vocab(translator)
    message = proto_module.TliDraftRequest(
        request_id=request.request_id,
        mode=request.mode,
        capture_hidden_mode=_capture_hidden_mode_to_proto(
            request.capture_hidden_mode
        ),
        topk=request.topk,
        speculative_num_steps=request.speculative_num_steps,
        speculative_num_draft_tokens=request.speculative_num_draft_tokens,
        num_tokens_per_req=request.num_tokens_per_req,
        num_tokens_for_logprob_per_req=request.num_tokens_for_logprob_per_req,
        tp_rank=request.tp_rank,
        tp_size=request.tp_size,
    )
    verified_id = tensor_to_proto_tensor(request.verified_id, proto_module)
    hidden_states = tensor_to_proto_tensor(request.hidden_states, proto_module)
    if hasattr(message.verified_id, "CopyFrom"):
        message.verified_id.CopyFrom(verified_id)
        message.hidden_states.CopyFrom(hidden_states)
    else:
        message.verified_id = verified_id
        message.hidden_states = hidden_states
    if request.request_ids is not None:
        message.request_ids.extend(list(request.request_ids))
    if request.input_ids is not None:
        input_ids = tensor_to_proto_tensor(request.input_ids, proto_module)
        if hasattr(message.input_ids, "CopyFrom"):
            message.input_ids.CopyFrom(input_ids)
        else:
            message.input_ids = input_ids
    if request.accept_length is not None:
        accept_length = tensor_to_proto_tensor(request.accept_length, proto_module)
        if hasattr(message.accept_length, "CopyFrom"):
            message.accept_length.CopyFrom(accept_length)
        else:
            message.accept_length = accept_length
    if request.accept_length_cpu is not None:
        message.accept_length_cpu.extend(list(request.accept_length_cpu))
    if request.seq_lens_for_draft_extend is not None:
        seq_lens_for_draft_extend = tensor_to_proto_tensor(
            request.seq_lens_for_draft_extend, proto_module
        )
        if hasattr(message.seq_lens_for_draft_extend, "CopyFrom"):
            message.seq_lens_for_draft_extend.CopyFrom(seq_lens_for_draft_extend)
        else:
            message.seq_lens_for_draft_extend = seq_lens_for_draft_extend
    if request.seq_lens_for_draft_extend_cpu is not None:
        seq_lens_for_draft_extend_cpu = tensor_to_proto_tensor(
            request.seq_lens_for_draft_extend_cpu, proto_module
        )
        if hasattr(message.seq_lens_for_draft_extend_cpu, "CopyFrom"):
            message.seq_lens_for_draft_extend_cpu.CopyFrom(
                seq_lens_for_draft_extend_cpu
            )
        else:
            message.seq_lens_for_draft_extend_cpu = seq_lens_for_draft_extend_cpu
    if request.mm_input_embeds is not None:
        mm_input_embeds = tensor_to_proto_tensor(request.mm_input_embeds, proto_module)
        if hasattr(message.mm_input_embeds, "CopyFrom"):
            message.mm_input_embeds.CopyFrom(mm_input_embeds)
        else:
            message.mm_input_embeds = mm_input_embeds
    return message


def draft_request_from_proto(
    proto_request,
    *,
    proto_module=None,
) -> TLIDraftRequest:
    """Convert the protobuf request envelope into a dataclass."""
    if not _message_has_field(proto_request, "verified_id"):
        raise ValueError("TliDraftRequest.verified_id is required")
    if not _message_has_field(proto_request, "hidden_states"):
        raise ValueError("TliDraftRequest.hidden_states is required")
    return TLIDraftRequest(
        request_id=proto_request.request_id,
        verified_id=tensor_from_proto_tensor(proto_request.verified_id),
        hidden_states=tensor_from_proto_tensor(proto_request.hidden_states),
        request_ids=(
            list(proto_request.request_ids)
            if len(getattr(proto_request, "request_ids", [])) > 0
            else None
        ),
        input_ids=(
            tensor_from_proto_tensor(proto_request.input_ids)
            if _message_has_field(proto_request, "input_ids")
            else None
        ),
        tp_rank=int(getattr(proto_request, "tp_rank", 0)),
        tp_size=int(getattr(proto_request, "tp_size", 1) or 1),
        mode=proto_request.mode or "decode",
        capture_hidden_mode=_capture_hidden_mode_from_proto(
            proto_request.capture_hidden_mode
        ),
        topk=int(proto_request.topk),
        speculative_num_steps=int(proto_request.speculative_num_steps),
        speculative_num_draft_tokens=int(proto_request.speculative_num_draft_tokens),
        num_tokens_per_req=int(proto_request.num_tokens_per_req),
        num_tokens_for_logprob_per_req=int(
            proto_request.num_tokens_for_logprob_per_req
        ),
        accept_length=(
            tensor_from_proto_tensor(proto_request.accept_length)
            if _message_has_field(proto_request, "accept_length")
            else None
        ),
        accept_length_cpu=(
            list(proto_request.accept_length_cpu)
            if len(getattr(proto_request, "accept_length_cpu", [])) > 0
            else None
        ),
        seq_lens_for_draft_extend=(
            tensor_from_proto_tensor(proto_request.seq_lens_for_draft_extend)
            if _message_has_field(proto_request, "seq_lens_for_draft_extend")
            else None
        ),
        seq_lens_for_draft_extend_cpu=(
            tensor_from_proto_tensor(proto_request.seq_lens_for_draft_extend_cpu)
            if _message_has_field(proto_request, "seq_lens_for_draft_extend_cpu")
            else None
        ),
        mm_input_embeds=(
            tensor_from_proto_tensor(proto_request.mm_input_embeds)
            if _message_has_field(proto_request, "mm_input_embeds")
            else None
        ),
    )


def draft_response_to_proto(
    response: TLIDraftResponse,
    *,
    proto_module=None,
):
    """Convert a response dataclass into the protobuf response envelope."""
    proto_module = _resolve_proto_module(proto_module)
    message = proto_module.TliDraftResponse(
        request_id=response.request_id,
        mode=response.mode,
    )
    parent_list = tensor_to_proto_tensor(response.parent_list, proto_module)
    top_scores_index = tensor_to_proto_tensor(response.top_scores_index, proto_module)
    draft_token_ids = tensor_to_proto_tensor(response.draft_token_ids, proto_module)
    if hasattr(message.parent_list, "CopyFrom"):
        message.parent_list.CopyFrom(parent_list)
        message.top_scores_index.CopyFrom(top_scores_index)
        message.draft_token_ids.CopyFrom(draft_token_ids)
    else:
        message.parent_list = parent_list
        message.top_scores_index = top_scores_index
        message.draft_token_ids = draft_token_ids
    if response.next_hidden_states is not None:
        next_hidden_states = tensor_to_proto_tensor(
            response.next_hidden_states, proto_module
        )
        if hasattr(message.next_hidden_states, "CopyFrom"):
            message.next_hidden_states.CopyFrom(next_hidden_states)
        else:
            message.next_hidden_states = next_hidden_states
    if response.next_topk_p is not None:
        next_topk_p = tensor_to_proto_tensor(response.next_topk_p, proto_module)
        if hasattr(message.next_topk_p, "CopyFrom"):
            message.next_topk_p.CopyFrom(next_topk_p)
        else:
            message.next_topk_p = next_topk_p
    if response.next_topk_index is not None:
        next_topk_index = tensor_to_proto_tensor(response.next_topk_index, proto_module)
        if hasattr(message.next_topk_index, "CopyFrom"):
            message.next_topk_index.CopyFrom(next_topk_index)
        else:
            message.next_topk_index = next_topk_index
    return message


def draft_response_from_proto(
    proto_response,
    *,
    translator: TLITokenTranslator | None = None,
    translate_to_target_vocab: bool = False,
    proto_module=None,
) -> TLIDraftResponse:
    """Convert the protobuf response envelope into a dataclass."""
    if not _message_has_field(proto_response, "parent_list"):
        raise ValueError("TliDraftResponse.parent_list is required")
    if not _message_has_field(proto_response, "top_scores_index"):
        raise ValueError("TliDraftResponse.top_scores_index is required")
    if not _message_has_field(proto_response, "draft_token_ids"):
        raise ValueError("TliDraftResponse.draft_token_ids is required")
    response = TLIDraftResponse(
        request_id=proto_response.request_id,
        parent_list=tensor_from_proto_tensor(proto_response.parent_list),
        top_scores_index=tensor_from_proto_tensor(proto_response.top_scores_index),
        draft_token_ids=tensor_from_proto_tensor(proto_response.draft_token_ids),
        mode=proto_response.mode or "decode",
        next_hidden_states=(
            tensor_from_proto_tensor(proto_response.next_hidden_states)
            if _message_has_field(proto_response, "next_hidden_states")
            else None
        ),
        next_topk_p=(
            tensor_from_proto_tensor(proto_response.next_topk_p)
            if _message_has_field(proto_response, "next_topk_p")
            else None
        ),
        next_topk_index=(
            tensor_from_proto_tensor(proto_response.next_topk_index)
            if _message_has_field(proto_response, "next_topk_index")
            else None
        ),
    )
    if translator is not None and translate_to_target_vocab:
        return response.to_target_vocab(translator)
    return response


class TliSpeculativeServiceAdapter:
    """gRPC servicer adapter for DraftForward requests.

    The adapter intentionally keeps translation policy explicit:
    - request translation is optional and happens before the handler runs
    - response translation is optional and happens before the protobuf reply
    """

    def __init__(
        self,
        request_handler: TLIDraftHandler,
        *,
        translator: TLITokenTranslator | None = None,
        proto_module=None,
        translate_requests_to_draft_vocab: bool = False,
        translate_responses_to_target_vocab: bool = False,
    ):
        self.request_handler = request_handler
        self.translator = translator
        self.proto_module = proto_module
        self.translate_requests_to_draft_vocab = translate_requests_to_draft_vocab
        self.translate_responses_to_target_vocab = (
            translate_responses_to_target_vocab
        )

    async def DraftForward(self, request, context):
        try:
            tli_request = draft_request_from_proto(request)
            if self.translator is not None and self.translate_requests_to_draft_vocab:
                tli_request = tli_request.to_draft_vocab(self.translator)

            logger.info(
                "[TLI-DEBUG] gRPC adapter DraftForward enter %s",
                _draft_request_debug_summary(tli_request),
            )

            response = self.request_handler(tli_request)
            if inspect.isawaitable(response):
                response = await response
            if not isinstance(response, TLIDraftResponse):
                raise TypeError(
                    "TLI request handler must return TLIDraftResponse, got "
                    f"{type(response).__name__}"
                )
            logger.info(
                "[TLI-DEBUG] gRPC adapter DraftForward handler returned %s",
                _draft_response_debug_summary(response),
            )
            if self.translator is not None and self.translate_responses_to_target_vocab:
                response = response.to_target_vocab(self.translator)
                logger.info(
                    "[TLI-DEBUG] gRPC adapter DraftForward translated response %s",
                    _draft_response_debug_summary(response),
                )
            logger.info(
                "[TLI-DEBUG] gRPC adapter DraftForward proto encode enter request_id=%r "
                "mode=%s",
                response.request_id,
                response.mode,
            )
            proto_response = draft_response_to_proto(
                response, proto_module=self.proto_module
            )
            logger.info(
                "[TLI-DEBUG] gRPC adapter DraftForward proto encode exit request_id=%r "
                "mode=%s proto_type=%s",
                response.request_id,
                response.mode,
                type(proto_response).__name__,
            )
            return proto_response
        except Exception as exc:
            logger.exception("TLI DraftForward RPC failed: %s", exc)
            if context is not None:
                import grpc

                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
            raise


def add_tli_speculative_service_to_server(servicer, server):
    """Register the TLI gRPC service on a grpc.aio.Server or grpc.Server."""
    _, sglang_pb2_grpc = _import_proto_modules()
    sglang_pb2_grpc.add_TliSpeculativeServiceServicer_to_server(servicer, server)


class TliSpeculativeClient:
    """Client helper for the disaggregated TLI DraftForward RPC."""

    def __init__(
        self,
        target: str,
        *,
        translator: TLITokenTranslator | None = None,
        channel_credentials=None,
        options=None,
    ):
        import grpc

        _, sglang_pb2_grpc = _import_proto_modules()

        self.translator = translator
        self._grpc = grpc
        self._channel = (
            grpc.aio.secure_channel(target, channel_credentials, options=options)
            if channel_credentials is not None
            else grpc.aio.insecure_channel(target, options=options)
        )
        self._stub = sglang_pb2_grpc.TliSpeculativeServiceStub(self._channel)

    async def draft_forward(
        self,
        request: TLIDraftRequest,
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> TLIDraftResponse:
        if self.translator is not None and translate_request_to_draft_vocab:
            request = request.to_draft_vocab(self.translator)

        proto_request = draft_request_to_proto(
            request,
            proto_module=None,
            translator=None,
            translate_to_draft_vocab=False,
        )
        proto_response = await self._stub.DraftForward(
            proto_request,
            timeout=timeout,
        )
        response = draft_response_from_proto(proto_response)
        if self.translator is not None and translate_response_to_target_vocab:
            response = response.to_target_vocab(self.translator)
        return response

    async def close(self):
        await self._channel.close()


class TliSpeculativeBlockingClient:
    """Blocking client helper for scheduler/model-worker code paths."""

    def __init__(
        self,
        target: str,
        *,
        translator: TLITokenTranslator | None = None,
        channel_credentials=None,
        options=None,
    ):
        import grpc

        _, sglang_pb2_grpc = _import_proto_modules()

        self.translator = translator
        self._channel = (
            grpc.secure_channel(target, channel_credentials, options=options)
            if channel_credentials is not None
            else grpc.insecure_channel(target, options=options)
        )
        self._stub = sglang_pb2_grpc.TliSpeculativeServiceStub(self._channel)

    def draft_forward(
        self,
        request: TLIDraftRequest,
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> TLIDraftResponse:
        if self.translator is not None and translate_request_to_draft_vocab:
            request = request.to_draft_vocab(self.translator)

        logger.info(
            "[TLI-DEBUG] blocking client draft_forward enter %s timeout=%s "
            "translate_request_to_draft_vocab=%s translate_response_to_target_vocab=%s",
            _draft_request_debug_summary(request),
            timeout,
            translate_request_to_draft_vocab,
            translate_response_to_target_vocab,
        )
        proto_request = draft_request_to_proto(
            request,
            proto_module=None,
            translator=None,
            translate_to_draft_vocab=False,
        )
        proto_response = self._stub.DraftForward(proto_request, timeout=timeout)
        logger.info(
            "[TLI-DEBUG] blocking client draft_forward stub returned request_id=%r "
            "mode=%s proto_type=%s",
            request.request_id,
            request.mode,
            type(proto_response).__name__,
        )
        response = draft_response_from_proto(proto_response)
        logger.info(
            "[TLI-DEBUG] blocking client draft_forward decoded response %s",
            _draft_response_debug_summary(response),
        )
        if self.translator is not None and translate_response_to_target_vocab:
            response = response.to_target_vocab(self.translator)
            logger.info(
                "[TLI-DEBUG] blocking client draft_forward translated response %s",
                _draft_response_debug_summary(response),
            )
        return response

    def close(self):
        self._channel.close()


async def serve_tli_speculative_service(
    *,
    host: str,
    port: int,
    request_handler: TLIDraftHandler,
    translator: TLITokenTranslator | None = None,
    translate_requests_to_draft_vocab: bool = False,
    translate_responses_to_target_vocab: bool = False,
    server_credentials=None,
    options=None,
):
    """Start a standalone gRPC server for the TLI DraftForward RPC."""
    import grpc

    servicer = TliSpeculativeServiceAdapter(
        request_handler=request_handler,
        translator=translator,
        proto_module=None,
        translate_requests_to_draft_vocab=translate_requests_to_draft_vocab,
        translate_responses_to_target_vocab=translate_responses_to_target_vocab,
    )
    server = grpc.aio.server(options=options)
    add_tli_speculative_service_to_server(servicer, server)
    if server_credentials is not None:
        bound_port = server.add_secure_port(f"{host}:{port}", server_credentials)
    else:
        bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port != port:
        raise RuntimeError(
            f"Failed to bind TLI gRPC service on {host}:{port} (bound {bound_port})"
        )
    await server.start()
    return server
