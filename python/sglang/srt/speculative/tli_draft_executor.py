# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side draft executor for disaggregated TLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    fast_topk,
    get_last_loc_large_page_size_large_top_k,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse
from sglang.srt.speculative.tli_token_translator import TLITokenTranslator
from sglang.srt.speculative.eagle_worker import get_last_loc_large_page_size_top_k_1
from sglang.srt.utils.common import ceil_align
from sglang.srt.utils import next_power_of_2
from sglang.srt.utils.hf_transformers_utils import get_tokenizer


class TLIDraftExecutorNotReadyError(RuntimeError):
    """Raised when a safe draft-side execution invariant is not available yet."""


@dataclass
class TLIDraftRequestState:
    """Per-request draft KV and speculative state."""

    request_id: str
    tp_rank: int
    req: Req
    seq_len: int = 0
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


class TLIDraftSchedulerExecutor:
    """Owns draft-node request state for TLI DraftForward RPCs.

    The decode path intentionally reuses colocated EAGLE/TLI paged allocation
    helpers so draft-owned cache layout stays aligned with the local worker.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.model_worker = scheduler.tp_worker
        self.model_runner = scheduler.tp_worker.model_runner
        self.model_config = self.model_runner.model_config
        self.device = self.model_runner.device
        self.req_to_token_pool = scheduler.req_to_token_pool
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        self.tree_cache = scheduler.tree_cache
        self.tp_rank = scheduler.tp_worker.tp_rank
        self.states: dict[tuple[str, int], TLIDraftRequestState] = {}
        self._sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        self._translator: Optional[TLITokenTranslator] = None
        self._topk: Optional[int] = None
        self._speculative_num_steps: Optional[int] = None
        self._speculative_num_draft_tokens: Optional[int] = None
        self._draft_attn_backend = None
        self._draft_extend_attn_backend = None
        self._num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self._extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

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

        self._ensure_spec_config(request)

        if request.mode == "extend":
            return self._handle_extend(request)
        if request.mode == "decode":
            return self._handle_decode(request)
        if request.mode == "extend_after_decode":
            return self._handle_extend_after_decode(request)

        raise ValueError(f"Unsupported TLI draft mode: {request.mode!r}")

    def release(
        self, request_ids: list[str], tp_rank: int, *, cache_prefix: bool = False
    ) -> None:
        if cache_prefix:
            raise TLIDraftExecutorNotReadyError(
                "Draft-side release does not insert prefixes into the radix cache. "
                "The draft executor only frees its owned KV and request slots."
            )
        for request_id in request_ids:
            state = self.states.pop((request_id, tp_rank), None)
            if state is not None:
                self._release_request_state(state)

    def clear(self) -> None:
        for state in list(self.states.values()):
            self._release_request_state(state)
        self.states.clear()

    def active_req_pool_idxs(self) -> set[int]:
        """Return req pool indices intentionally held across RPC boundaries."""
        return {
            state.req.req_pool_idx
            for state in self.states.values()
            if state.req.req_pool_idx is not None
        }

    def held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Tokens intentionally retained by draft-owned request states.

        These are not part of the tree cache's session slot accounting, so the
        scheduler runtime checker needs a direct view of them.
        """
        total = 0
        for state in self.states.values():
            req = state.req
            in_batch = (
                active_pool_idxs is not None and req.req_pool_idx in active_pool_idxs
            )
            if req.req_pool_idx is None or in_batch:
                continue
            allocated = ceil_align(req.kv_allocated_len, self.server_args.page_size)
            total += allocated - req.cache_protected_len
        return total

    def held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """SWA tokens intentionally retained by draft-owned request states."""
        if not self.tree_cache.supports_swa():
            return 0

        total = 0
        for state in self.states.values():
            req = state.req
            in_batch = (
                active_pool_idxs is not None and req.req_pool_idx in active_pool_idxs
            )
            if req.req_pool_idx is None or in_batch:
                continue
            allocated = ceil_align(req.kv_allocated_len, self.server_args.page_size)
            total += allocated - max(req.cache_protected_len, req.swa_evicted_seqlen)
        return total

    def held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        """Req slots intentionally retained by draft-owned request states."""
        total = 0
        for state in self.states.values():
            req = state.req
            in_batch = (
                active_pool_idxs is not None and req.req_pool_idx in active_pool_idxs
            )
            if req.req_pool_idx is not None and not in_batch:
                total += 1
        return total

    def _validate_request(self, request: TLIDraftRequest) -> None:
        if request.tp_rank != self.tp_rank:
            raise ValueError(
                "TLI draft request was routed to the wrong TP rank: "
                f"request={request.tp_rank}, local={self.tp_rank}"
            )
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

    def _ensure_spec_config(self, request: TLIDraftRequest) -> None:
        spec_values = (
            request.topk,
            request.speculative_num_steps,
            request.speculative_num_draft_tokens,
        )
        if self._topk is None:
            self._topk = request.topk
            self._speculative_num_steps = request.speculative_num_steps
            self._speculative_num_draft_tokens = request.speculative_num_draft_tokens
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.model_runner,
                self._topk,
                self._speculative_num_steps,
            )
            self._draft_attn_backend = draft_backend_factory.create_decode_backend()
            self._draft_extend_attn_backend = (
                draft_backend_factory.create_draft_extend_backend()
            )
            self.model_runner.draft_attn_backend = self._draft_attn_backend
            return

        if spec_values != (
            self._topk,
            self._speculative_num_steps,
            self._speculative_num_draft_tokens,
        ):
            raise ValueError(
                "TLI draft request speculative parameters changed within the "
                "same draft executor. Restart the draft service or route to a "
                "separate executor for a different speculative shape: "
                f"request={spec_values}, executor="
                f"{(self._topk, self._speculative_num_steps, self._speculative_num_draft_tokens)}"
            )

    def _translator_or_create(self) -> TLITokenTranslator:
        if self._translator is not None:
            return self._translator
        target_tokenizer_path = self.server_args.tli_target_tokenizer_path
        if target_tokenizer_path is None:
            raise ValueError(
                "Draft role requires --tli-target-tokenizer-path to constrain TLI "
                "draft logits to the target/draft vocabulary intersection."
            )

        target_tokenizer = get_tokenizer(
            target_tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
            tokenizer_backend=self.server_args.tokenizer_backend,
        )
        draft_tokenizer = get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
            tokenizer_revision=self.server_args.revision,
            tokenizer_backend=self.server_args.tokenizer_backend,
        )
        target_vocab_size = len(target_tokenizer.get_vocab())

        self._translator = TLITokenTranslator(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=self.model_config.vocab_size,
            device=torch.device(self.device),
        )
        return self._translator

    def _handle_extend(self, request: TLIDraftRequest) -> TLIDraftResponse:
        request_ids = request.request_ids or [request.request_id]
        self.release(request_ids, request.tp_rank)
        batch = self._build_initial_extend_batch(request, request_ids)
        hidden_states = self._align_extend_hidden_states(request, batch)
        verified_id = request.verified_id.to(self.device, non_blocking=True)

        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=verified_id,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=request.seq_lens_for_draft_extend_cpu
        )
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.return_logprob = False
        if request.mm_input_embeds is not None:
            forward_batch.mm_input_embeds = request.mm_input_embeds.to(
                self.device, non_blocking=True
            )

        try:
            logits_output = self.model_runner.forward(forward_batch).logits_output
        except Exception:
            for state in self._states_from_batch(batch, request_ids):
                self._release_request_state(state)
            raise
        maybe_detect_nan(logits_output.next_token_logits, "tli_draft_extend")
        self._capture_for_decode(logits_output, forward_batch.spec_info)
        self._store_next_draft_state(request_ids, batch, forward_batch.spec_info)

        return self._state_response(request, forward_batch.spec_info)

    def _handle_decode(self, request: TLIDraftRequest) -> TLIDraftResponse:
        request_ids = request.request_ids or [request.request_id]
        batch = self._build_decode_batch(request, request_ids)
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.can_run_dp_cuda_graph = False
        if not forward_batch.forward_mode.is_idle() and self._speculative_num_steps > 1:
            self._draft_attn_backend.init_forward_metadata(forward_batch)

        parent_list, top_scores_index, draft_tokens = self._draft_forward(
            forward_batch
        )
        return TLIDraftResponse(
            request_id=request.request_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_token_ids=draft_tokens,
            mode=request.mode,
        )

    def _handle_extend_after_decode(
        self, request: TLIDraftRequest
    ) -> TLIDraftResponse:
        request_ids = request.request_ids or [request.request_id]
        batch, states, new_seq_lens = self._build_extend_after_decode_batch(
            request, request_ids
        )
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        forward_batch.can_run_dp_cuda_graph = False
        if not forward_batch.forward_mode.is_idle():
            attn_backend = self._draft_extend_attn_backend or self.model_runner.attn_backend
            attn_backend.init_forward_metadata(forward_batch)
            forward_batch.attn_backend = attn_backend

        try:
            logits_output = self.model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
        except Exception:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            raise
        maybe_detect_nan(logits_output.next_token_logits, "tli_draft_extend_after_decode")
        self._capture_for_decode(logits_output, forward_batch.spec_info)
        self._commit_extend_after_decode_state(states, new_seq_lens)
        self._store_next_draft_state(request_ids, batch, forward_batch.spec_info)

        return self._state_response(request, forward_batch.spec_info)

    def _build_initial_extend_batch(
        self, request: TLIDraftRequest, request_ids: list[str]
    ) -> ScheduleBatch:
        if request.input_ids is None:
            raise ValueError("TLI draft extend requires flattened input_ids.")
        if request.seq_lens_for_draft_extend_cpu is None:
            raise ValueError("TLI draft extend requires seq_lens_for_draft_extend_cpu.")

        seq_lens_cpu = request.seq_lens_for_draft_extend_cpu.cpu().tolist()
        if len(seq_lens_cpu) != len(request_ids):
            raise ValueError(
                "TLI draft extend request_ids and seq_lens length mismatch: "
                f"{len(request_ids)} request ids vs {len(seq_lens_cpu)} sequence lengths."
            )

        input_ids = request.input_ids.to("cpu").tolist()
        if sum(seq_lens_cpu) != len(input_ids):
            raise TLIDraftExecutorNotReadyError(
                "TLI draft extend requires the target to send the full current "
                "token prefix for each request so the draft node can reconstruct "
                "draft-owned KV independently. Got "
                f"sum(seq_lens)={sum(seq_lens_cpu)} but len(input_ids)={len(input_ids)}."
            )

        reqs: list[Req] = []
        offset = 0
        for request_id, seq_len in zip(request_ids, seq_lens_cpu):
            token_ids = input_ids[offset : offset + seq_len]
            offset += seq_len
            if len(token_ids) == 0:
                raise ValueError(f"TLI draft extend got empty input for {request_id!r}.")
            req = Req(
                rid=request_id,
                origin_input_text="",
                origin_input_ids=token_ids,
                sampling_params=self._sampling_params,
                eos_token_ids=self.model_config.hf_eos_token_id,
                vocab_size=self.model_config.vocab_size,
            )
            req.logprob_start_len = -1
            req.init_next_round_input(self.tree_cache)
            reqs.append(req)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.TLI,
        )
        batch.prepare_for_extend()
        return batch

    def _align_extend_hidden_states(
        self, request: TLIDraftRequest, batch: ScheduleBatch
    ) -> torch.Tensor:
        hidden_states = request.hidden_states.to(self.device, non_blocking=True)
        if hidden_states.shape[0] == batch.extend_num_tokens:
            return hidden_states

        seq_lens_cpu = request.seq_lens_for_draft_extend_cpu.cpu().tolist()
        if hidden_states.shape[0] != sum(seq_lens_cpu):
            raise TLIDraftExecutorNotReadyError(
                "TLI draft extend cannot align target hidden states with "
                "draft-owned prefix cache state. The target sent "
                f"{hidden_states.shape[0]} hidden rows, while the draft needs "
                f"{batch.extend_num_tokens} uncached rows and the full prefix "
                f"length is {sum(seq_lens_cpu)}."
            )

        parts = []
        offset = 0
        for seq_len, prefix_len in zip(seq_lens_cpu, batch.prefix_lens):
            parts.append(hidden_states[offset + prefix_len : offset + seq_len])
            offset += seq_len
        return torch.cat(parts, dim=0) if parts else hidden_states[:0]

    def _build_decode_batch(
        self, request: TLIDraftRequest, request_ids: list[str]
    ) -> ScheduleBatch:
        states = self._states_for_request(request, request_ids)
        for state in states:
            if (
                state.topk_p is None
                or state.topk_index is None
                or state.hidden_states is None
            ):
                raise RuntimeError(
                    f"TLI draft state for {state.request_id!r} is missing next "
                    "draft probabilities; an extend or extend_after_decode RPC "
                    "must succeed before decode."
                )

        batch = self._base_batch_from_states(states)
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = EagleDraftInput(
            topk_p=torch.stack([state.topk_p for state in states], dim=0),
            topk_index=torch.stack([state.topk_index for state in states], dim=0),
            hidden_states=torch.stack([state.hidden_states for state in states], dim=0),
            verified_id=request.verified_id.to(self.device, non_blocking=True),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            num_tokens_per_req=self._topk,
            num_tokens_for_logprob_per_req=self._topk,
        )
        self._draft_preprocess_decode(batch)
        return batch

    def _build_extend_after_decode_batch(
        self, request: TLIDraftRequest, request_ids: list[str]
    ) -> tuple[ScheduleBatch, list[TLIDraftRequestState], list[int]]:
        states = self._states_for_request(request, request_ids)
        if request.accept_length_cpu is None:
            if request.accept_length is None:
                raise ValueError("TLI extend_after_decode requires accept_length.")
            accept_length_cpu = request.accept_length.cpu().tolist()
        else:
            accept_length_cpu = request.accept_length_cpu
        if len(accept_length_cpu) != len(states):
            raise ValueError(
                "TLI extend_after_decode accept_length length mismatch: "
                f"{len(accept_length_cpu)} values for {len(states)} requests."
            )

        old_seq_lens = [state.seq_len for state in states]
        extend_lens = [length + 1 for length in accept_length_cpu]
        new_seq_lens = [old + ext for old, ext in zip(old_seq_lens, extend_lens)]

        batch = self._base_batch_from_states(states)
        batch.spec_info = EagleDraftInput(
            hidden_states=request.hidden_states.to(self.device, non_blocking=True),
            verified_id=request.verified_id.to(self.device, non_blocking=True),
            accept_length=torch.tensor(
                accept_length_cpu, dtype=torch.int32, device=self.device
            ),
            accept_length_cpu=accept_length_cpu,
            seq_lens_for_draft_extend=torch.tensor(
                new_seq_lens, dtype=torch.int64, device=self.device
            ),
            seq_lens_for_draft_extend_cpu=torch.tensor(
                new_seq_lens, dtype=torch.int64
            ),
            req_pool_indices_for_draft_extend=batch.req_pool_indices,
        )
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.prefix_lens = old_seq_lens
        batch.extend_lens = extend_lens
        batch.extend_num_tokens = sum(extend_lens)
        batch.extend_logprob_start_lens = [0] * len(states)
        batch.input_ids = request.verified_id.to(self.device, non_blocking=True)
        batch.seq_lens = torch.tensor(new_seq_lens, dtype=torch.int64, device=self.device)
        batch.seq_lens_cpu = torch.tensor(new_seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            new_seq_lens, dtype=torch.int32, device=self.device
        )
        batch.seq_lens_sum = sum(new_seq_lens)
        batch.return_logprob = False
        batch.return_hidden_states = False
        batch.out_cache_loc = self._alloc_extend_after_decode_slots(
            states, old_seq_lens, extend_lens
        )
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self._speculative_num_steps,
        )
        return batch, states, new_seq_lens

    def _commit_extend_after_decode_state(
        self,
        states: list[TLIDraftRequestState],
        new_seq_lens: list[int],
    ) -> None:
        for state, new_seq_len in zip(states, new_seq_lens):
            state.req.kv_committed_len = new_seq_len
            state.req.kv_allocated_len = new_seq_len
            state.seq_len = new_seq_len

    def _base_batch_from_states(self, states: list[TLIDraftRequestState]) -> ScheduleBatch:
        reqs = [state.req for state in states]
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.TLI,
        )
        seq_lens = [state.seq_len for state in states]
        batch.req_pool_indices = torch.tensor(
            [state.req.req_pool_idx for state in states],
            dtype=torch.int64,
            device=self.device,
        )
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        batch.seq_lens_sum = sum(seq_lens)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            self.model_config.vocab_size,
        )
        return batch

    def _alloc_extend_after_decode_slots(
        self,
        states: list[TLIDraftRequestState],
        old_seq_lens: list[int],
        extend_lens: list[int],
    ) -> torch.Tensor:
        total_tokens = sum(extend_lens)
        old_seq_lens_cpu = torch.tensor(old_seq_lens, dtype=torch.int64)
        extend_lens_cpu = torch.tensor(extend_lens, dtype=torch.int64)
        old_seq_lens_device = old_seq_lens_cpu.to(self.device, non_blocking=True)
        extend_lens_device = extend_lens_cpu.to(self.device, non_blocking=True)
        new_seq_lens_device = old_seq_lens_device + extend_lens_device
        new_seq_lens_cpu = old_seq_lens_cpu + extend_lens_cpu
        req_pool_indices = torch.tensor(
            [state.req.req_pool_idx for state in states],
            dtype=torch.int64,
            device=self.device,
        )
        if self.server_args.page_size == 1:
            out_cache_loc = alloc_token_slots(self.tree_cache, total_tokens)
        else:
            last_loc = get_last_loc(
                self.req_to_token_pool.req_to_token,
                req_pool_indices,
                old_seq_lens_device,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                tree_cache=self.tree_cache,
                prefix_lens=old_seq_lens_device,
                prefix_lens_cpu=old_seq_lens_cpu,
                seq_lens=new_seq_lens_device,
                seq_lens_cpu=new_seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=total_tokens,
            )

        offset = 0
        for state, old_seq_len, extend_len in zip(states, old_seq_lens, extend_lens):
            locs = torch.arange(
                old_seq_len,
                old_seq_len + extend_len,
                dtype=torch.int64,
                device=self.device,
            )
            self.req_to_token_pool.write(
                (state.req.req_pool_idx, locs),
                out_cache_loc[offset : offset + extend_len].to(torch.int32),
            )
            offset += extend_len
        return out_cache_loc

    def _draft_preprocess_decode(self, batch: ScheduleBatch) -> None:
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        num_seqs = batch.batch_size()
        if self.server_args.page_size == 1:
            alloc_len_per_decode = self._speculative_num_steps * self._topk
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self._topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self._speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self._speculative_num_steps
                extend_num_tokens = num_seqs * self._speculative_num_steps
            else:
                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self._num_new_pages_per_topk,
                    self._extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self._speculative_num_steps,
                    self._topk,
                    self.server_args.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.server_args.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu
                    + self._speculative_num_steps
                    + self.server_args.page_size
                    - 1
                ) // self.server_args.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu
                    // self.server_args.page_size
                    * self.server_args.page_size
                    + num_new_pages_per_topk
                    * (self.server_args.page_size * self._topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.server_args.page_size > 1 and self._topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (
                self._topk - 1
            )
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self._extend_lens,
            self._num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self._topk,
            self._speculative_num_steps,
            self.server_args.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self._speculative_num_steps + self.server_args.page_size),
        )

        if self.server_args.page_size > 1 and self._topk > 1:
            if duplicate_cache_len > 0:
                self.model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            out_cache_loc = out_cache_loc[
                : num_seqs * self._topk * self._speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self._topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "tli_draft_forward: NaN in initial topk_p")

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self._topk, self._speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self._speculative_num_steps, -1
        )

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []
        scores = None
        translator = self._translator_or_create()

        for i in range(self._speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self._topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == self._speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            if self.model_config.hf_config.architectures[0] == "GptOssForCausalLM":
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self._draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            logits_output = self.model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"tli_draft_forward step {i}")
            constrained_logits = translator.constrain_draft_logits(
                logits_output.next_token_logits
            )
            probs = torch.softmax(constrained_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self._topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                constrained_logits.shape[-1],
                f"tli_draft_forward step {i}: topk_index OOB",
            )
            hidden_states = logits_output.hidden_states

        return organize_draft_results(
            score_list, token_list, parents_list, self._speculative_num_draft_tokens
        )

    def _capture_for_decode(self, logits_output, draft_input: EagleDraftInput) -> None:
        constrained_logits = self._translator_or_create().constrain_draft_logits(
            logits_output.next_token_logits
        )
        probs = torch.softmax(constrained_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(
            probs, self._topk, dim=-1
        )
        draft_input.hidden_states = logits_output.hidden_states

    def _store_next_draft_state(
        self,
        request_ids: list[str],
        batch: ScheduleBatch,
        spec_info: EagleDraftInput,
    ) -> None:
        if len(request_ids) != len(batch.reqs):
            raise ValueError("TLI draft state/request batch size mismatch.")
        for i, request_id in enumerate(request_ids):
            state = self.states.get((request_id, self.tp_rank))
            if state is None:
                state = TLIDraftRequestState(
                    request_id=request_id,
                    tp_rank=self.tp_rank,
                    req=batch.reqs[i],
                )
                self.states[(request_id, self.tp_rank)] = state
            state.req = batch.reqs[i]
            state.seq_len = int(batch.seq_lens_cpu[i].item())
            state.topk_p = spec_info.topk_p[i].detach()
            state.topk_index = spec_info.topk_index[i].detach()
            state.hidden_states = spec_info.hidden_states[i].detach()
            self._assert_state_kv_invariant(state)

    @staticmethod
    def _assert_state_kv_invariant(state: TLIDraftRequestState) -> None:
        if state.req.req_pool_idx is None:
            raise RuntimeError(f"TLI draft state {state.request_id!r} has no req slot.")
        if state.seq_len < len(state.req.origin_input_ids):
            raise RuntimeError(
                "TLI draft request sequence length regressed below its prompt "
                f"length: request_id={state.request_id!r}, seq_len={state.seq_len}, "
                f"prompt_len={len(state.req.origin_input_ids)}"
            )
        if (
            state.req.kv_committed_len != state.seq_len
            or state.req.kv_allocated_len != state.seq_len
        ):
            raise RuntimeError(
                "TLI draft KV accounting mismatch: "
                f"request_id={state.request_id!r}, seq_len={state.seq_len}, "
                f"committed={state.req.kv_committed_len}, "
                f"allocated={state.req.kv_allocated_len}"
            )

    def _release_request_state(self, state: TLIDraftRequestState) -> None:
        req = state.req
        req_pool_idx = req.req_pool_idx
        if req_pool_idx is None:
            return
        if req.kv_committed_len != state.seq_len or req.kv_allocated_len != state.seq_len:
            raise RuntimeError(
                "TLI draft release saw mismatched KV accounting: "
                f"request_id={state.request_id!r}, seq_len={state.seq_len}, "
                f"committed={req.kv_committed_len}, allocated={req.kv_allocated_len}"
            )

        if req.mamba_pool_idx is not None and hasattr(
            self.req_to_token_pool, "free_mamba_cache"
        ):
            self.req_to_token_pool.free_mamba_cache(req)

        kv_indices = self.req_to_token_pool.req_to_token[
            req_pool_idx, : req.kv_allocated_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)
        self.req_to_token_pool.free(req)
        req.kv_committed_len = 0
        req.kv_allocated_len = 0

    def _states_from_batch(
        self, batch: ScheduleBatch, request_ids: list[str]
    ) -> list[TLIDraftRequestState]:
        if len(request_ids) != len(batch.reqs):
            raise ValueError("TLI draft state/request batch size mismatch.")
        states: list[TLIDraftRequestState] = []
        for i, request_id in enumerate(request_ids):
            state = self.states.get((request_id, self.tp_rank))
            if state is None:
                state = TLIDraftRequestState(
                    request_id=request_id,
                    tp_rank=self.tp_rank,
                    req=batch.reqs[i],
                    seq_len=int(batch.seq_lens_cpu[i].item()),
                )
            else:
                state.seq_len = int(batch.seq_lens_cpu[i].item())
            states.append(state)
        return states

    def _states_for_request(
        self, request: TLIDraftRequest, request_ids: list[str]
    ) -> list[TLIDraftRequestState]:
        states = []
        for request_id in request_ids:
            state = self.states.get((request_id, request.tp_rank))
            if state is None:
                raise RuntimeError(
                    f"TLI draft state for request {request_id!r} does not exist."
                )
            states.append(state)
        return states

    def _state_response(
        self, request: TLIDraftRequest, spec_info: EagleDraftInput
    ) -> TLIDraftResponse:
        return TLIDraftResponse(
            request_id=request.request_id,
            parent_list=torch.empty((0,), dtype=torch.int64, device=self.device),
            top_scores_index=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            mode=request.mode,
            next_hidden_states=spec_info.hidden_states,
            next_topk_p=spec_info.topk_p,
            next_topk_index=spec_info.topk_index,
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
