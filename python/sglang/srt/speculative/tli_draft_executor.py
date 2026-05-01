# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side draft executor for disaggregated TLI.

This module is the draft-node boundary between the TLI gRPC service and the
local SGLang scheduler/model worker. It owns per-request draft state and keeps
the execution path fail-closed until all SGLang batch/KV invariants needed for
remote draft forwards are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse


class TLIDraftExecutorNotReadyError(RuntimeError):
    """Raised when the scheduler path is wired but draft model execution is not."""


@dataclass
class TLIDraftRequestState:
    request_id: str
    tp_rank: int
    seq_len: int = 0


class TLIDraftSchedulerExecutor:
    """Owns draft-node request state for TLI DraftForward RPCs."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.model_worker = scheduler.tp_worker
        self.model_runner = scheduler.tp_worker.model_runner
        self.states: dict[tuple[str, int], TLIDraftRequestState] = {}

        if self.server_args.tli_disaggregation_role != "draft":
            raise ValueError(
                "TLIDraftSchedulerExecutor can only run on "
                "--tli-disaggregation-role draft."
            )

    def handle(self, request: TLIDraftRequest) -> TLIDraftResponse:
        self._validate_request(request)
        if request.mode == "release":
            self.release(request.request_ids or [request.request_id], request.tp_rank)
            return self._empty_response(request)

        self._ensure_states(request)
        raise TLIDraftExecutorNotReadyError(
            "TLI draft scheduler/model execution is not implemented yet. "
            "The gRPC/control-plane path is wired and draft-owned state is "
            "tracked, but running the draft model requires reconstructing "
            "SGLang ScheduleBatch/ForwardBatch state with correct KV allocation "
            f"for mode={request.mode!r}, request_id={request.request_id!r}, "
            f"tp_rank={request.tp_rank}."
        )

    def release(self, request_ids: list[str], tp_rank: int) -> None:
        for request_id in request_ids:
            self.states.pop((request_id, tp_rank), None)

    def clear(self) -> None:
        self.states.clear()

    def _validate_request(self, request: TLIDraftRequest) -> None:
        if request.tp_size != self.server_args.tp_size:
            raise ValueError(
                "TLI draft request tp_size does not match draft server tp_size: "
                f"request={request.tp_size}, server={self.server_args.tp_size}"
            )
        if request.tp_rank < 0 or request.tp_rank >= request.tp_size:
            raise ValueError(
                f"Invalid TLI draft request tp_rank={request.tp_rank}, "
                f"tp_size={request.tp_size}"
            )
        if request.mode not in ("extend", "decode", "extend_after_decode", "release"):
            raise ValueError(f"Unsupported TLI draft mode: {request.mode!r}")

    def _ensure_states(self, request: TLIDraftRequest) -> None:
        request_ids = request.request_ids or [request.request_id]
        for request_id in request_ids:
            self.states.setdefault(
                (request_id, request.tp_rank),
                TLIDraftRequestState(
                    request_id=request_id,
                    tp_rank=request.tp_rank,
                ),
            )

    @staticmethod
    def _empty_response(request: TLIDraftRequest) -> TLIDraftResponse:
        empty_i64 = torch.empty((0,), dtype=torch.int64)
        return TLIDraftResponse(
            request_id=request.request_id,
            parent_list=empty_i64,
            top_scores_index=empty_i64,
            draft_token_ids=empty_i64,
            mode=request.mode,
        )
