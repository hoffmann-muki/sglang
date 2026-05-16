# SPDX-License-Identifier: Apache-2.0
"""Remote-target worker for disaggregated TLI speculative decoding.

This worker runs the target model locally and delegates draft-model operations to
the TLI DraftForward gRPC service. It intentionally does not allocate or load a
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
from sglang.srt.speculative.tli_disaggregation import (
    build_tli_channel_credentials,
)
from sglang.srt.speculative.tli_grpc_transport import TliSpeculativeBlockingClient
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse
from sglang.srt.speculative.tli_token_translator import TLITokenTranslator
from sglang.srt.utils.common import broadcast_pyobj
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BroadcastedDraftResult:
    """Target-side wrapper used to fan a rank-0 draft RPC result out to TP ranks."""

    ok: bool
    response: TLIDraftResponse | None = None
    error: str | None = None


class RemoteTLIWorker(EAGLEWorker):
    """Target-side TLI worker that talks to a remote draft/speculator service."""

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
        self.remote_draft_tp_size = server_args.tli_draft_tp_size or self.tp_size
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        if server_args.tli_draft_server_addr is None:
            raise ValueError("RemoteTLIWorker requires --tli-draft-server-addr.")

        target_tokenizer_path = server_args.tokenizer_path
        draft_tokenizer_path = server_args.tli_draft_tokenizer_path
        if draft_tokenizer_path is None:
            raise ValueError("RemoteTLIWorker requires --tli-draft-tokenizer-path.")
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
        self.client = TliSpeculativeBlockingClient(
            server_args.tli_draft_server_addr,
            translator=self.vocab_mapping,
            channel_credentials=build_tli_channel_credentials(server_args),
        )
        logger.info(
            "RemoteTLIWorker initialized; draft RPC target=%s draft_tp_size=%d "
            "mode=%s",
            server_args.tli_draft_server_addr,
            self.remote_draft_tp_size,
            "asymmetric" if self.uses_rank0_broadcast else "symmetric",
        )

    @property
    def draft_model_runner(self):
        raise RuntimeError("RemoteTLIWorker does not own a local draft model runner.")

    def clear_cache_pool(self):
        self._release_request_ids(sorted(self.active_request_ids))

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
        return request_ids

    def _batch_request_id(self, request_ids: list[str], mode: str) -> str:
        return f"{mode}:{','.join(request_ids)}"

    @property
    def uses_rank0_broadcast(self) -> bool:
        return self.tp_size > 1 and self.remote_draft_tp_size == 1

    def _draft_tp_rank(self) -> int:
        return 0 if self.uses_rank0_broadcast else self.tp_rank

    def _draft_tp_size(self) -> int:
        return 1 if self.uses_rank0_broadcast else self.tp_size

    def _run_rank0_broadcasted_draft_forward(
        self,
        request: TLIDraftRequest,
        *,
        timeout: float | None,
        translate_request_to_draft_vocab: bool,
        translate_response_to_target_vocab: bool,
    ) -> TLIDraftResponse:
        if not self.uses_rank0_broadcast:
            return self.client.draft_forward(
                request,
                timeout=timeout,
                translate_request_to_draft_vocab=translate_request_to_draft_vocab,
                translate_response_to_target_vocab=translate_response_to_target_vocab,
            )

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
                logger.exception("TLI DraftForward RPC failed on target root rank.")
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
                "TLI DraftForward RPC failed on the target root rank:\n"
                f"{broadcasted.error}"
            )
        if broadcasted.response is None:
            raise RuntimeError(
                "TLI DraftForward RPC succeeded but returned no response payload."
            )
        return broadcasted.response

    def _draft_extend_input_ids(self, batch) -> torch.Tensor:
        """Return the full current token prefix for draft-side KV reconstruction."""
        full_input_ids = list(chain.from_iterable(req.fill_ids for req in batch.reqs))
        return torch.tensor(full_input_ids, dtype=torch.int64, device=self.device)

    def _apply_next_draft_state(
        self,
        draft_input: EagleDraftInput,
        response: TLIDraftResponse,
        mode: str,
    ):
        if (
            response.next_topk_p is None
            or response.next_topk_index is None
            or response.next_hidden_states is None
        ):
            raise RuntimeError(
                f"TLI draft service returned a {mode} response without next "
                "draft state (next_topk_p, next_topk_index, next_hidden_states)."
            )
        draft_input.topk_p = response.next_topk_p.to(self.device)
        draft_input.topk_index = response.next_topk_index.to(self.device)
        draft_input.hidden_states = response.next_hidden_states.to(self.device)

    def _release_request_ids(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        release_request = TLIDraftRequest(
            request_id=f"release:{','.join(request_ids)}",
            request_ids=request_ids,
            verified_id=torch.empty((0,), dtype=torch.int64),
            hidden_states=torch.empty((0,), dtype=self.model_config.dtype),
            mode="release",
            tp_rank=self._draft_tp_rank(),
            tp_size=self._draft_tp_size(),
        )
        self._run_rank0_broadcasted_draft_forward(
            release_request,
            timeout=self.server_args.tli_rpc_timeout,
            translate_request_to_draft_vocab=False,
            translate_response_to_target_vocab=False,
        )
        self.active_request_ids.difference_update(request_ids)

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
        self._release_request_ids(finished_request_ids)
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
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        request = TLIDraftRequest(
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
            mm_input_embeds=mm_input_embeds,
        )
        response = self._run_rank0_broadcasted_draft_forward(
            request,
            timeout=self.server_args.tli_rpc_timeout,
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
        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        request = TLIDraftRequest(
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
        response = self._run_rank0_broadcasted_draft_forward(
            request,
            timeout=self.server_args.tli_rpc_timeout,
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
            spec_info.verified_id,
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
        request = TLIDraftRequest(
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
            accept_length_cpu=(
                spec_info.accept_length.cpu().tolist()
                if spec_info.accept_length is not None
                else None
            ),
        )
        response = self._run_rank0_broadcasted_draft_forward(
            request,
            timeout=self.server_args.tli_rpc_timeout,
            translate_request_to_draft_vocab=True,
            translate_response_to_target_vocab=True,
        )
        self._apply_next_draft_state(spec_info, response, "extend_after_decode")
