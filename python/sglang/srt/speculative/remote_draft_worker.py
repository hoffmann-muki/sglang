# SPDX-License-Identifier: Apache-2.0
"""Remote-target worker for disaggregated draft-forward speculative decoding.

This worker runs the target model locally and delegates draft-model operations to
the DraftForward gRPC service. It intentionally does not allocate or load a
local draft model.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from itertools import chain
from typing import Optional

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.trace import get_global_tracing_enabled
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.speculative.draft_disaggregation import (
    build_draft_forward_channel_credentials,
)
from sglang.srt.speculative.draft_forward_grpc_transport import (
    StreamingDraftForwardClient,
)
from sglang.srt.speculative.draft_forward_protocol import (
    DraftForwardRequest,
    DraftForwardResponse,
)
from sglang.srt.speculative.tli_token_translator import TLITokenTranslator
from sglang.srt.utils.common import broadcast_pyobj
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BroadcastedDraftResult:
    """Target-side wrapper used to fan a rank-0 draft RPC result out to TP ranks."""

    ok: bool
    response: DraftForwardResponse | None = None
    error: str | None = None


@dataclass(slots=True)
class _TargetRequestOrdering:
    """Target-side monotonically increasing per-request ordering state."""

    round_id: int = 0
    prefix_version: int = 0


class RemoteDraftWorker(EAGLEWorker):
    """Target-side worker that talks to a remote draft/speculator service."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        del gpu_id, dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port

        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.hot_token_id = None
        self.adaptive_controller = None
        self.model_config = target_worker.model_runner.model_config
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.active_request_ids: set[str] = set()
        self._request_ordering: dict[str, _TargetRequestOrdering] = {}
        self.remote_draft_tp_size = server_args.remote_draft_tp_size or self.tp_size
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        remote_draft_server_addr = server_args.remote_draft_server_addr
        if remote_draft_server_addr is None:
            raise ValueError("RemoteDraftWorker requires --remote-draft-server-addr.")

        target_tokenizer_path = server_args.tokenizer_path
        draft_tokenizer_path = server_args.remote_draft_tokenizer_path
        if draft_tokenizer_path is None:
            raise ValueError(
                "RemoteDraftWorker requires --remote-draft-tokenizer-path."
            )
        target_tokenizer = get_tokenizer(
            target_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_revision=server_args.revision,
            tokenizer_backend=server_args.tokenizer_backend,
        )
        draft_tokenizer = get_tokenizer(
            draft_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_backend=server_args.tokenizer_backend,
        )
        draft_vocab_size = len(draft_tokenizer.get_vocab())

        self.vocab_mapping = TLITokenTranslator(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            draft_vocab_size=draft_vocab_size,
            device=torch.device(self.device),
        )
        self.client = StreamingDraftForwardClient(
            remote_draft_server_addr,
            translator=self.vocab_mapping,
            channel_credentials=build_draft_forward_channel_credentials(server_args),
        )
        logger.info(
            "RemoteDraftWorker initialized; DraftForward target=%s draft_tp_size=%d "
            "mode=%s",
            remote_draft_server_addr,
            self.remote_draft_tp_size,
            "asymmetric" if self.uses_rank0_broadcast else "symmetric",
        )

    @property
    def draft_model_runner(self):
        raise RuntimeError("RemoteDraftWorker does not own a local draft model runner.")

    def clear_cache_pool(self):
        self._release_request_ids(sorted(self.active_request_ids), cache_prefix=False)
        self._request_ordering.clear()

    def capture_for_decode(self, logits_output, draft_input: EagleDraftInput):
        constrained_logits = self.vocab_mapping.constrain_draft_logits(
            logits_output.next_token_logits
        )
        probs = torch.softmax(constrained_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def _request_ids(self, batch) -> list[str]:
        request_ids = [req.rid for req in batch.reqs]
        self.active_request_ids.update(request_ids)
        for request_id in request_ids:
            self._request_ordering.setdefault(request_id, _TargetRequestOrdering())
        return request_ids

    def _request_ids_for_spec_info(
        self, batch, spec_info: EagleDraftInput
    ) -> list[str]:
        req_pool_indices = spec_info.req_pool_indices_for_draft_extend
        if req_pool_indices is None or len(req_pool_indices) == len(batch.reqs):
            return self._request_ids(batch)

        req_by_pool_idx = {req.req_pool_idx: req.rid for req in batch.reqs}
        request_ids = [
            req_by_pool_idx[int(req_pool_idx)]
            for req_pool_idx in req_pool_indices.to("cpu").tolist()
        ]
        self.active_request_ids.update(request_ids)
        for request_id in request_ids:
            self._request_ordering.setdefault(request_id, _TargetRequestOrdering())
        return request_ids

    def _batch_request_id(self, request_ids: list[str], mode: str) -> str:
        return f"{mode}:{','.join(request_ids)}"

    def _request_by_id(self, batch) -> dict[str, object]:
        return {req.rid: req for req in batch.reqs}

    def _token_positions_for_request_ids(
        self,
        request_ids: list[str],
        req_by_id: dict[str, object],
    ) -> list[int]:
        positions = []
        for request_id in request_ids:
            req = req_by_id.get(request_id)
            if req is None:
                positions.append(0)
            else:
                positions.append(len(req.origin_input_ids) + len(req.output_ids))
        return positions

    def _attach_ordering_metadata(
        self,
        request: DraftForwardRequest,
        *,
        request_ids: list[str],
        token_positions: list[int],
        advance_round: bool,
        prefix_versions: list[int] | None = None,
    ) -> DraftForwardRequest:
        round_ids = []
        request_prefix_versions = []
        for request_id in request_ids:
            state = self._request_ordering.setdefault(
                request_id,
                _TargetRequestOrdering(),
            )
            if advance_round:
                state.round_id += 1
            round_ids.append(state.round_id)
            request_prefix_versions.append(state.prefix_version)
        request.round_ids = round_ids
        request.token_positions = token_positions
        request.prefix_versions = prefix_versions or request_prefix_versions
        return request

    def _next_prefix_versions(
        self,
        request_ids: list[str],
        accepted_lengths: list[int],
    ) -> list[int]:
        prefix_versions = []
        for request_id, accepted_len in zip(request_ids, accepted_lengths):
            state = self._request_ordering.setdefault(
                request_id,
                _TargetRequestOrdering(),
            )
            prefix_versions.append(state.prefix_version + int(accepted_len) + 1)
        return prefix_versions

    def _commit_prefix_versions(
        self,
        request_ids: list[str],
        prefix_versions: list[int],
    ) -> None:
        for request_id, prefix_version in zip(request_ids, prefix_versions):
            self._request_ordering.setdefault(
                request_id,
                _TargetRequestOrdering(),
            ).prefix_version = prefix_version

    @staticmethod
    def _validate_ordering_response(
        request: DraftForwardRequest,
        response: DraftForwardResponse,
    ) -> None:
        expected = (
            getattr(request, "round_ids", None),
            getattr(request, "token_positions", None),
            getattr(request, "prefix_versions", None),
        )
        actual = (
            getattr(response, "round_ids", None),
            getattr(response, "token_positions", None),
            getattr(response, "prefix_versions", None),
        )
        if actual != expected:
            raise RuntimeError(
                "DraftForward response ordering metadata mismatch: "
                f"request_id={request.request_id!r}, expected={expected}, "
                f"actual={actual}"
            )

    @property
    def uses_rank0_broadcast(self) -> bool:
        return self.tp_size > 1 and self.remote_draft_tp_size == 1

    def _draft_tp_rank(self) -> int:
        return 0 if self.uses_rank0_broadcast else self.tp_rank

    def _draft_tp_size(self) -> int:
        return 1 if self.uses_rank0_broadcast else self.tp_size

    def _run_rank0_broadcasted_draft_forward(
        self,
        request: DraftForwardRequest,
        *,
        timeout: float | None,
        translate_request_to_draft_vocab: bool,
        translate_response_to_target_vocab: bool,
    ) -> DraftForwardResponse:
        if not self.uses_rank0_broadcast:
            response = self.client.draft_forward(
                request,
                timeout=timeout,
                translate_request_to_draft_vocab=translate_request_to_draft_vocab,
                translate_response_to_target_vocab=translate_response_to_target_vocab,
            )
            self._validate_ordering_response(request, response)
            return response

        world_group = self.target_worker.world_group
        is_root = world_group.is_first_rank
        payload: list[_BroadcastedDraftResult]
        if is_root:
            try:
                response = self.client.draft_forward(
                    request,
                    timeout=timeout,
                    translate_request_to_draft_vocab=translate_request_to_draft_vocab,
                    translate_response_to_target_vocab=translate_response_to_target_vocab,
                )
                payload = [_BroadcastedDraftResult(ok=True, response=response)]
            except Exception:
                logger.exception("DraftForward RPC failed on target root rank.")
                payload = [
                    _BroadcastedDraftResult(
                        ok=False,
                        error=traceback.format_exc(),
                    )
                ]
        else:
            payload = []

        broadcasted = broadcast_pyobj(
            payload,
            world_group.rank,
            world_group.cpu_group,
            src=world_group.first_rank,
        )[0]
        if not broadcasted.ok:
            raise RuntimeError(
                "DraftForward RPC failed on the target root rank:\n"
                f"{broadcasted.error}"
            )
        if broadcasted.response is None:
            raise RuntimeError(
                "DraftForward RPC succeeded but returned no response payload."
            )
        self._validate_ordering_response(request, broadcasted.response)
        return broadcasted.response

    def _draft_extend_input_ids(self, batch) -> torch.Tensor:
        """Return the full current token prefix for draft-side KV reconstruction."""
        full_input_ids = list(chain.from_iterable(req.fill_ids for req in batch.reqs))
        return torch.tensor(full_input_ids, dtype=torch.int64, device=self.device)

    def _target_prefix_lens_for_draft_extend(self, batch) -> torch.Tensor:
        prefix_lens = getattr(batch, "prefix_lens", None)
        if prefix_lens is None:
            prefix_lens = [0] * len(batch.reqs)
        return torch.tensor(prefix_lens, dtype=torch.int64)

    def _apply_next_draft_state(
        self,
        draft_input: EagleDraftInput,
        response: DraftForwardResponse,
        mode: str,
    ):
        if (
            response.next_topk_p is None
            or response.next_topk_index is None
            or response.next_hidden_states is None
        ):
            raise RuntimeError(
                f"DraftForward service returned a {mode} response without next "
                "draft state (next_topk_p, next_topk_index, next_hidden_states)."
            )
        draft_input.topk_p = response.next_topk_p.to(self.device)
        draft_input.topk_index = response.next_topk_index.to(self.device)
        draft_input.hidden_states = response.next_hidden_states.to(self.device)

    def _release_request_ids(
        self, request_ids: list[str], *, cache_prefix: bool = True
    ) -> None:
        if not request_ids:
            return
        release_request = DraftForwardRequest(
            request_id=f"release:{','.join(request_ids)}",
            request_ids=request_ids,
            verified_id=torch.empty((0,), dtype=torch.int64),
            hidden_states=torch.empty((0,), dtype=self.model_config.dtype),
            mode="release",
            tp_rank=self._draft_tp_rank(),
            tp_size=self._draft_tp_size(),
            cache_prefix_on_release=cache_prefix,
        )
        self._run_rank0_broadcasted_draft_forward(
            release_request,
            timeout=self.server_args.draft_forward_rpc_timeout,
            translate_request_to_draft_vocab=False,
            translate_response_to_target_vocab=False,
        )
        self.active_request_ids.difference_update(request_ids)
        for request_id in request_ids:
            self._request_ordering.pop(request_id, None)

    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            self.forward_draft_extend(
                batch,
                logits_output.hidden_states,
                next_token_ids,
                seq_lens_cpu,
                logits_output.mm_input_embeds,
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        spec_info = self.draft(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
        set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
            self.verify(batch, spec_info)
        )
        del model_worker_batch

        if get_global_tracing_enabled():
            for idx, req in enumerate(batch.reqs):
                accepted = verify_output.accept_length_per_req_cpu[idx]
                req.time_stats.set_spec_verify_end_time(accepted_tokens=accepted)

        finished_request_ids = [req.rid for req in batch.reqs if req.finished()]

        set_time_batch(batch.reqs, "set_spec_draft_extend_start_time", trace_only=True)
        if (
            self.server_args.enable_dp_attention
            or batch.spec_info.verified_id.numel() > 0
        ):
            self.forward_draft_extend_after_decode(batch)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[CACHE-DEBUG] target releasing finished draft requests request_ids=%s "
                "finished_request_ids=%s batch_size=%s accept_length=%s",
                [req.rid for req in batch.reqs],
                finished_request_ids,
                len(batch.reqs),
                getattr(batch.spec_info, "accept_length", None),
            )
        self._release_request_ids(finished_request_ids, cache_prefix=True)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[CACHE-DEBUG] target released finished draft requests request_ids=%s",
                finished_request_ids,
            )
        set_time_batch(batch.reqs, "set_spec_draft_extend_end_time", trace_only=True)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_drafts=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_draft_extend(
        self,
        batch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu,
        mm_input_embeds=None,
    ):
        request_ids = self._request_ids(batch)
        req_by_id = self._request_by_id(batch)
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        request = DraftForwardRequest(
            request_id=self._batch_request_id(request_ids, "extend"),
            request_ids=request_ids,
            input_ids=self._draft_extend_input_ids(batch),
            verified_id=next_token_ids,
            hidden_states=hidden_states,
            mode="extend",
            tp_rank=self._draft_tp_rank(),
            tp_size=self._draft_tp_size(),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            seq_lens_for_draft_extend=batch.seq_lens,
            seq_lens_for_draft_extend_cpu=seq_lens_cpu,
            target_prefix_lens_for_draft_extend_cpu=(
                self._target_prefix_lens_for_draft_extend(batch)
            ),
            mm_input_embeds=mm_input_embeds,
        )
        self._attach_ordering_metadata(
            request,
            request_ids=request_ids,
            token_positions=self._token_positions_for_request_ids(
                request_ids,
                req_by_id,
            ),
            advance_round=True,
        )
        response = self._run_rank0_broadcasted_draft_forward(
            request,
            timeout=self.server_args.draft_forward_rpc_timeout,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )
        self._apply_next_draft_state(batch.spec_info, response, "extend")

    def draft(self, batch):
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        request_ids = self._request_ids(batch)
        req_by_id = self._request_by_id(batch)
        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        request = DraftForwardRequest(
            request_id=self._batch_request_id(request_ids, "decode"),
            request_ids=request_ids,
            verified_id=spec_info.verified_id,
            hidden_states=spec_info.hidden_states,
            mode="decode",
            tp_rank=self._draft_tp_rank(),
            tp_size=self._draft_tp_size(),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            num_tokens_per_req=self.topk,
            num_tokens_for_logprob_per_req=self.topk,
        )
        self._attach_ordering_metadata(
            request,
            request_ids=request_ids,
            token_positions=self._token_positions_for_request_ids(
                request_ids,
                req_by_id,
            ),
            advance_round=True,
        )
        response = self._run_rank0_broadcasted_draft_forward(
            request,
            timeout=self.server_args.draft_forward_rpc_timeout,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.get_tree_verified_id(),
            response.parent_list.to(self.device),
            response.top_scores_index.to(self.device),
            response.draft_token_ids.to(self.device),
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def forward_draft_extend_after_decode(self, batch):
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        request_ids = self._request_ids_for_spec_info(batch, spec_info)
        req_by_id = self._request_by_id(batch)
        accept_length_cpu = (
            spec_info.accept_length.cpu().tolist()
            if spec_info.accept_length is not None
            else []
        )
        next_prefix_versions = self._next_prefix_versions(
            request_ids,
            accept_length_cpu,
        )
        request = DraftForwardRequest(
            request_id=self._batch_request_id(request_ids, "extend_after_decode"),
            request_ids=request_ids,
            verified_id=spec_info.verified_id,
            hidden_states=spec_info.hidden_states,
            mode="extend_after_decode",
            tp_rank=self._draft_tp_rank(),
            tp_size=self._draft_tp_size(),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=1,
            accept_length=spec_info.accept_length,
            accept_length_cpu=accept_length_cpu or None,
        )
        self._attach_ordering_metadata(
            request,
            request_ids=request_ids,
            token_positions=self._token_positions_for_request_ids(
                request_ids,
                req_by_id,
            ),
            advance_round=True,
            prefix_versions=next_prefix_versions,
        )
        response = self._run_rank0_broadcasted_draft_forward(
            request,
            timeout=self.server_args.draft_forward_rpc_timeout,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[CACHE-DEBUG] target draft extend-after-decode complete request_ids=%s "
                "accept_length=%s next_prefix_versions=%s",
                request_ids,
                accept_length_cpu,
                next_prefix_versions,
            )
        self._commit_prefix_versions(request_ids, next_prefix_versions)
        self._apply_next_draft_state(spec_info, response, "extend_after_decode")
