# SPDX-License-Identifier: Apache-2.0
"""gRPC transport helpers for remote draft forwarding."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import logging
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from importlib import import_module
from typing import Awaitable, Callable

import torch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.draft_forward_protocol import (
    DraftForwardRequest,
    DraftForwardResponse,
)
from sglang.srt.speculative.tli_token_translator import TLITokenTranslator

logger = logging.getLogger(__name__)

DraftForwardHandler = Callable[
    [DraftForwardRequest], DraftForwardResponse | Awaitable[DraftForwardResponse]
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
        "Unable to import generated draft-forward gRPC modules. Tried:\n"
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
        raise ValueError(
            f"Unsupported tensor dtype for draft-forward RPC: {dtype}"
        ) from exc


def _str_to_dtype(name: str) -> torch.dtype:
    try:
        return _STR_TO_DTYPE[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tensor dtype for draft-forward RPC: {name}"
        ) from exc


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
    shape = list(shape)
    tensor_dtype = _str_to_dtype(dtype)
    numel = 1
    for dim in shape:
        numel *= dim
    expected_nbytes = numel * torch.empty((), dtype=tensor_dtype).element_size()
    if len(data) != expected_nbytes:
        raise ValueError(
            "Invalid draft-forward tensor payload size: "
            f"shape={shape}, dtype={dtype}, expected={expected_nbytes}, "
            f"got={len(data)}"
        )
    if numel == 0:
        return torch.empty(shape, dtype=tensor_dtype, device=device)

    # `torch.frombuffer` warns on immutable bytes objects; copy into a writable
    # bytearray first since we immediately clone anyway.
    tensor = torch.frombuffer(bytearray(data), dtype=tensor_dtype).reshape(shape)
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


def _validate_ordering_metadata(
    *,
    owner: str,
    request_ids: list[str] | None,
    round_ids: list[int] | None,
    token_positions: list[int] | None,
    prefix_versions: list[int] | None,
) -> None:
    metadata = {
        "round_ids": round_ids,
        "token_positions": token_positions,
        "prefix_versions": prefix_versions,
    }
    present = {name: values for name, values in metadata.items() if values is not None}
    if not present:
        return
    if len(present) != len(metadata):
        missing = sorted(set(metadata) - set(present))
        raise ValueError(
            f"{owner} ordering metadata must provide all fields together; "
            f"missing={missing}"
        )

    lengths = {name: len(values) for name, values in present.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"{owner} ordering metadata fields must have matching lengths; "
            f"got={lengths}"
        )
    if request_ids is not None and unique_lengths != {len(request_ids)}:
        raise ValueError(
            f"{owner} ordering metadata must align with request_ids; "
            f"metadata_len={next(iter(unique_lengths))}, "
            f"request_ids_len={len(request_ids)}"
        )


def draft_request_to_proto(
    request: DraftForwardRequest,
    *,
    translator: TLITokenTranslator | None = None,
    translate_to_draft_vocab: bool = True,
    proto_module=None,
):
    """Convert a request dataclass into the protobuf request envelope."""
    proto_module = _resolve_proto_module(proto_module)
    if translator is not None and translate_to_draft_vocab:
        request = request.to_draft_vocab(translator)
    _validate_ordering_metadata(
        owner="DraftForwardRequest",
        request_ids=request.request_ids,
        round_ids=request.round_ids,
        token_positions=request.token_positions,
        prefix_versions=request.prefix_versions,
    )
    message = proto_module.DraftForwardRequest(
        request_id=request.request_id,
        mode=request.mode,
        capture_hidden_mode=_capture_hidden_mode_to_proto(request.capture_hidden_mode),
        topk=request.topk,
        speculative_num_steps=request.speculative_num_steps,
        speculative_num_draft_tokens=request.speculative_num_draft_tokens,
        num_tokens_per_req=request.num_tokens_per_req,
        num_tokens_for_logprob_per_req=request.num_tokens_for_logprob_per_req,
        tp_rank=request.tp_rank,
        tp_size=request.tp_size,
        cache_prefix_on_release=request.cache_prefix_on_release,
    )
    if request.round_ids is not None:
        message.round_ids.extend(list(request.round_ids))
    if request.token_positions is not None:
        message.token_positions.extend(list(request.token_positions))
    if request.prefix_versions is not None:
        message.prefix_versions.extend(list(request.prefix_versions))
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
    if request.target_prefix_lens_for_draft_extend_cpu is not None:
        target_prefix_lens_for_draft_extend_cpu = tensor_to_proto_tensor(
            request.target_prefix_lens_for_draft_extend_cpu, proto_module
        )
        if hasattr(message.target_prefix_lens_for_draft_extend_cpu, "CopyFrom"):
            message.target_prefix_lens_for_draft_extend_cpu.CopyFrom(
                target_prefix_lens_for_draft_extend_cpu
            )
        else:
            message.target_prefix_lens_for_draft_extend_cpu = (
                target_prefix_lens_for_draft_extend_cpu
            )
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
) -> DraftForwardRequest:
    """Convert the protobuf request envelope into a dataclass."""
    if not _message_has_field(proto_request, "verified_id"):
        raise ValueError("DraftForwardRequest.verified_id is required")
    if not _message_has_field(proto_request, "hidden_states"):
        raise ValueError("DraftForwardRequest.hidden_states is required")
    return DraftForwardRequest(
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
        target_prefix_lens_for_draft_extend_cpu=(
            tensor_from_proto_tensor(
                proto_request.target_prefix_lens_for_draft_extend_cpu
            )
            if _message_has_field(
                proto_request,
                "target_prefix_lens_for_draft_extend_cpu",
            )
            else None
        ),
        mm_input_embeds=(
            tensor_from_proto_tensor(proto_request.mm_input_embeds)
            if _message_has_field(proto_request, "mm_input_embeds")
            else None
        ),
        round_ids=(
            list(proto_request.round_ids)
            if len(getattr(proto_request, "round_ids", [])) > 0
            else None
        ),
        token_positions=(
            list(proto_request.token_positions)
            if len(getattr(proto_request, "token_positions", [])) > 0
            else None
        ),
        prefix_versions=(
            list(proto_request.prefix_versions)
            if len(getattr(proto_request, "prefix_versions", [])) > 0
            else None
        ),
        cache_prefix_on_release=bool(
            getattr(proto_request, "cache_prefix_on_release", False)
        ),
    )


def draft_response_to_proto(
    response: DraftForwardResponse,
    *,
    proto_module=None,
):
    """Convert a response dataclass into the protobuf response envelope."""
    proto_module = _resolve_proto_module(proto_module)
    _validate_ordering_metadata(
        owner="DraftForwardResponse",
        request_ids=None,
        round_ids=response.round_ids,
        token_positions=response.token_positions,
        prefix_versions=response.prefix_versions,
    )
    message = proto_module.DraftForwardResponse(
        request_id=response.request_id,
        mode=response.mode,
    )
    if response.round_ids is not None:
        message.round_ids.extend(list(response.round_ids))
    if response.token_positions is not None:
        message.token_positions.extend(list(response.token_positions))
    if response.prefix_versions is not None:
        message.prefix_versions.extend(list(response.prefix_versions))
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
) -> DraftForwardResponse:
    """Convert the protobuf response envelope into a dataclass."""
    if not _message_has_field(proto_response, "parent_list"):
        raise ValueError("DraftForwardResponse.parent_list is required")
    if not _message_has_field(proto_response, "top_scores_index"):
        raise ValueError("DraftForwardResponse.top_scores_index is required")
    if not _message_has_field(proto_response, "draft_token_ids"):
        raise ValueError("DraftForwardResponse.draft_token_ids is required")
    response = DraftForwardResponse(
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
        round_ids=(
            list(proto_response.round_ids)
            if len(getattr(proto_response, "round_ids", [])) > 0
            else None
        ),
        token_positions=(
            list(proto_response.token_positions)
            if len(getattr(proto_response, "token_positions", [])) > 0
            else None
        ),
        prefix_versions=(
            list(proto_response.prefix_versions)
            if len(getattr(proto_response, "prefix_versions", [])) > 0
            else None
        ),
    )
    if translator is not None and translate_to_target_vocab:
        return response.to_target_vocab(translator)
    return response


class DraftForwardServiceAdapter:
    """gRPC servicer adapter for DraftForward requests.

    The adapter intentionally keeps translation policy explicit:
    - request translation is optional and happens before the handler runs
    - response translation is optional and happens before the protobuf reply
    """

    def __init__(
        self,
        request_handler: DraftForwardHandler,
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
        self.translate_responses_to_target_vocab = translate_responses_to_target_vocab

    async def _draft_forward_response(self, request):
        draft_request = draft_request_from_proto(request)
        if self.translator is not None and self.translate_requests_to_draft_vocab:
            draft_request = draft_request.to_draft_vocab(self.translator)

        response = self.request_handler(draft_request)
        if inspect.isawaitable(response):
            response = await response
        if not isinstance(response, DraftForwardResponse):
            raise TypeError(
                "DraftForward request handler must return DraftForwardResponse, got "
                f"{type(response).__name__}"
            )
        if self.translator is not None and self.translate_responses_to_target_vocab:
            response = response.to_target_vocab(self.translator)
        return draft_response_to_proto(response, proto_module=self.proto_module)

    def _set_context_error(self, context, exc: Exception):
        if context is None:
            return
        import grpc

        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(str(exc))

    async def DraftForward(self, request, context):
        try:
            return await self._draft_forward_response(request)
        except Exception as exc:
            logger.exception("DraftForward RPC failed: %s", exc)
            self._set_context_error(context, exc)
            raise

    async def DraftForwardStream(self, request_iterator, context):
        """Process streamed draft requests concurrently.

        Responses are yielded as individual requests complete. Per-request
        ordering is carried by round/prefix metadata; cross-request ordering is
        intentionally not imposed here.
        """

        queue = asyncio.Queue()
        tasks = set()
        sentinel = object()

        async def run_one(request):
            try:
                response = await self._draft_forward_response(request)
                await queue.put((response, None))
            except Exception as exc:  # pragma: no cover - exercised via caller
                await queue.put((None, exc))

        async def produce():
            try:
                async for request in request_iterator:
                    task = asyncio.create_task(run_one(request))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await queue.put((sentinel, None))

        producer = asyncio.create_task(produce())
        try:
            while True:
                response, exc = await queue.get()
                if response is sentinel:
                    break
                if exc is not None:
                    logger.error("DraftForwardStream RPC failed: %s", exc)
                    self._set_context_error(context, exc)
                    producer.cancel()
                    for task in list(tasks):
                        task.cancel()
                    raise exc
                yield response
        finally:
            if not producer.done():
                producer.cancel()
            for task in list(tasks):
                task.cancel()


def add_draft_forward_service_to_server(servicer, server):
    """Register the DraftForward gRPC service on a grpc.aio.Server or grpc.Server."""
    _, sglang_pb2_grpc = _import_proto_modules()
    sglang_pb2_grpc.add_DraftForwardServiceServicer_to_server(servicer, server)


class DraftForwardClient:
    """Client helper for the disaggregated DraftForward RPC."""

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
        self._stub = sglang_pb2_grpc.DraftForwardServiceStub(self._channel)

    async def draft_forward(
        self,
        request: DraftForwardRequest,
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> DraftForwardResponse:
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

    async def _iter_proto_requests(
        self,
        requests: Iterable[DraftForwardRequest] | AsyncIterable[DraftForwardRequest],
        *,
        translate_request_to_draft_vocab: bool,
    ) -> AsyncIterator:
        if isinstance(requests, AsyncIterable):
            async for request in requests:
                if self.translator is not None and translate_request_to_draft_vocab:
                    request = request.to_draft_vocab(self.translator)
                yield draft_request_to_proto(
                    request,
                    proto_module=None,
                    translator=None,
                    translate_to_draft_vocab=False,
                )
        else:
            for request in requests:
                if self.translator is not None and translate_request_to_draft_vocab:
                    request = request.to_draft_vocab(self.translator)
                yield draft_request_to_proto(
                    request,
                    proto_module=None,
                    translator=None,
                    translate_to_draft_vocab=False,
                )

    async def draft_forward_stream(
        self,
        requests: Iterable[DraftForwardRequest] | AsyncIterable[DraftForwardRequest],
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> AsyncIterator[DraftForwardResponse]:
        proto_responses = self._stub.DraftForwardStream(
            self._iter_proto_requests(
                requests,
                translate_request_to_draft_vocab=translate_request_to_draft_vocab,
            ),
            timeout=timeout,
        )
        async for proto_response in proto_responses:
            response = draft_response_from_proto(proto_response)
            if self.translator is not None and translate_response_to_target_vocab:
                response = response.to_target_vocab(self.translator)
            yield response

    async def close(self):
        await self._channel.close()


class BlockingDraftForwardClient:
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
        self._stub = sglang_pb2_grpc.DraftForwardServiceStub(self._channel)

    def draft_forward(
        self,
        request: DraftForwardRequest,
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> DraftForwardResponse:
        if self.translator is not None and translate_request_to_draft_vocab:
            request = request.to_draft_vocab(self.translator)

        proto_request = draft_request_to_proto(
            request,
            proto_module=None,
            translator=None,
            translate_to_draft_vocab=False,
        )
        proto_response = self._stub.DraftForward(proto_request, timeout=timeout)
        response = draft_response_from_proto(proto_response)
        if self.translator is not None and translate_response_to_target_vocab:
            response = response.to_target_vocab(self.translator)
        return response

    def _iter_proto_requests(
        self,
        requests: Iterable[DraftForwardRequest],
        *,
        translate_request_to_draft_vocab: bool,
    ) -> Iterator:
        for request in requests:
            if self.translator is not None and translate_request_to_draft_vocab:
                request = request.to_draft_vocab(self.translator)
            yield draft_request_to_proto(
                request,
                proto_module=None,
                translator=None,
                translate_to_draft_vocab=False,
            )

    def draft_forward_stream(
        self,
        requests: Iterable[DraftForwardRequest],
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> Iterator[DraftForwardResponse]:
        proto_responses = self._stub.DraftForwardStream(
            self._iter_proto_requests(
                requests,
                translate_request_to_draft_vocab=translate_request_to_draft_vocab,
            ),
            timeout=timeout,
        )
        for proto_response in proto_responses:
            response = draft_response_from_proto(proto_response)
            if self.translator is not None and translate_response_to_target_vocab:
                response = response.to_target_vocab(self.translator)
            yield response

    def close(self):
        self._channel.close()


class StreamingDraftForwardClient:
    """Blocking facade over a long-lived bidirectional DraftForward stream.

    The scheduler side is still synchronous today, but the transport underneath
    is not: multiple callers may submit requests concurrently, and responses are
    matched by request_id as they arrive instead of by send order.
    """

    def __init__(
        self,
        target: str,
        *,
        translator: TLITokenTranslator | None = None,
        channel_credentials=None,
        options=None,
    ):
        self.translator = translator
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[Future, bool]] = {}
        self._stream_error: Exception | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="draft-forward-stream",
            daemon=True,
        )
        self._thread.start()
        self._async_client_future = asyncio.run_coroutine_threadsafe(
            self._create_async_client(
                target,
                translator=translator,
                channel_credentials=channel_credentials,
                options=options,
            ),
            self._loop,
        )
        self._async_client_future.result()
        self._request_queue_future = asyncio.run_coroutine_threadsafe(
            self._create_request_queue(),
            self._loop,
        )
        self._request_queue = self._request_queue_future.result()
        self._stream_task = asyncio.run_coroutine_threadsafe(
            self._run_stream(),
            self._loop,
        )
        self._ready.wait()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _create_async_client(
        self,
        target: str,
        *,
        translator: TLITokenTranslator | None,
        channel_credentials,
        options,
    ) -> None:
        self._async_client = DraftForwardClient(
            target,
            translator=translator,
            channel_credentials=channel_credentials,
            options=options,
        )

    async def _create_request_queue(self) -> asyncio.Queue:
        return asyncio.Queue()

    async def _request_iterator(self) -> AsyncIterator:
        while True:
            item = await self._request_queue.get()
            if item is None:
                break
            request, translate_request_to_draft_vocab = item
            if self.translator is not None and translate_request_to_draft_vocab:
                request = request.to_draft_vocab(self.translator)
            yield draft_request_to_proto(
                request,
                proto_module=None,
                translator=None,
                translate_to_draft_vocab=False,
            )

    async def _run_stream(self):
        self._ready.set()
        try:
            proto_responses = self._async_client._stub.DraftForwardStream(
                self._request_iterator()
            )
            async for proto_response in proto_responses:
                response = draft_response_from_proto(proto_response)
                with self._lock:
                    pending = self._pending.pop(response.request_id, None)
                if pending is None:
                    logger.warning(
                        "Dropping orphan DraftForward streaming response request_id=%r.",
                        response.request_id,
                    )
                    continue
                future, translate_response_to_target_vocab = pending
                if self.translator is not None and translate_response_to_target_vocab:
                    response = response.to_target_vocab(self.translator)
                future.set_result(response)
        except Exception as exc:
            self._stream_error = exc
            if not self._closed.is_set():
                logger.exception("DraftForwardStream failed: %s", exc)
            self._fail_all_pending(exc)
        finally:
            if not self._closed.is_set() and self._stream_error is None:
                self._stream_error = RuntimeError("DraftForwardStream closed.")
                self._fail_all_pending(self._stream_error)

    def _fail_all_pending(self, exc: Exception):
        with self._lock:
            pending = [future for future, _ in self._pending.values()]
            self._pending.clear()
        for future in pending:
            future.set_exception(exc)

    def draft_forward(
        self,
        request: DraftForwardRequest,
        *,
        timeout: float | None = None,
        translate_request_to_draft_vocab: bool = True,
        translate_response_to_target_vocab: bool = True,
    ) -> DraftForwardResponse:
        if self._closed.is_set():
            raise RuntimeError("DraftForward streaming client is closed.")
        if self._stream_error is not None:
            raise RuntimeError("DraftForward streaming client is unavailable.") from (
                self._stream_error
            )
        future: Future = Future()
        with self._lock:
            if self._stream_error is not None:
                raise RuntimeError(
                    "DraftForward streaming client is unavailable."
                ) from (self._stream_error)
            if request.request_id in self._pending:
                raise RuntimeError(
                    f"Duplicate in-flight DraftForward request_id={request.request_id!r}"
                )
            self._pending[request.request_id] = (
                future,
                translate_response_to_target_vocab,
            )

        enqueue = asyncio.run_coroutine_threadsafe(
            self._request_queue.put((request, translate_request_to_draft_vocab)),
            self._loop,
        )
        try:
            enqueue.result(timeout=timeout)
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            with self._lock:
                self._pending.pop(request.request_id, None)
            raise TimeoutError(
                f"DraftForward streaming request timed out for {request.request_id!r}."
            ) from None
        except Exception:
            with self._lock:
                self._pending.pop(request.request_id, None)
            raise

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        asyncio.run_coroutine_threadsafe(self._request_queue.put(None), self._loop)
        try:
            self._stream_task.result(timeout=5)
        except Exception:
            self._stream_task.cancel()
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_client.close(),
                self._loop,
            ).result(timeout=5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()


async def serve_draft_forward_service(
    *,
    host: str,
    port: int,
    request_handler: DraftForwardHandler,
    translator: TLITokenTranslator | None = None,
    translate_requests_to_draft_vocab: bool = False,
    translate_responses_to_target_vocab: bool = False,
    server_credentials=None,
    options=None,
):
    """Start a standalone gRPC server for the DraftForward RPC."""
    import grpc

    servicer = DraftForwardServiceAdapter(
        request_handler=request_handler,
        translator=translator,
        proto_module=None,
        translate_requests_to_draft_vocab=translate_requests_to_draft_vocab,
        translate_responses_to_target_vocab=translate_responses_to_target_vocab,
    )
    server = grpc.aio.server(options=options)
    add_draft_forward_service_to_server(servicer, server)
    if server_credentials is not None:
        bound_port = server.add_secure_port(f"{host}:{port}", server_credentials)
    else:
        bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port != port:
        raise RuntimeError(
            f"Failed to bind DraftForward gRPC service on {host}:{port} "
            f"(bound {bound_port})"
        )
    await server.start()
    return server
