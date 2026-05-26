"""Standalone Fast_dLLM_v2 speculative worker for AR targets."""

from __future__ import annotations

import json
import logging
import traceback
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.co_draft.dllm_linear_adapter import (
    IndependentDllmAcceptedTokens,
    IndependentDllmDraftRequest,
)
from sglang.srt.speculative.co_draft.fast_dllm_v2_runner import (
    FastDllmV2ProposalRunner,
    FastDllmV2RunnerConfig,
    SGLangNativeFastDllmV2Runtime,
)
from sglang.srt.speculative.co_draft.tp import LocalDraftTpPlan
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.linear_verify import (
    LinearDraftBlock,
    build_linear_draft_block,
    build_linear_verify_input,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    SingleRankDraftGroup,
    single_rank_draft_context,
)
from sglang.srt.utils import broadcast_pyobj

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BroadcastedFastDllmV2Proposal:
    ok: bool
    payload: Optional[dict] = None
    error: Optional[str] = None


def _cpu_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().to("cpu")


class FastDllmV2Worker:
    """Spec-v1 worker that verifies independent Fast_dLLM_v2 token blocks.

    Fast_dLLM_v2 owns proposal generation through either the Transformers
    reference path or the SGLang-native draft runner. Once it emits token ids,
    the AR target verifies the linear block using the same target-side contract
    as DFlash, but without DFlash's target-hidden-conditioned draft cache.
    """

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
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.tp_rank = tp_rank
        self.device = target_worker.device
        self.tp_size = int(server_args.tp_size)
        self.draft_tp_size = int(server_args.speculative_draft_tp_size)
        self.draft_tp_plan = LocalDraftTpPlan(
            name="FAST_DLLM_V2",
            target_tp_size=self.tp_size,
            draft_tp_size=self.draft_tp_size,
        )
        self._is_draft_rank = self.draft_tp_plan.owns_rank(tp_rank)
        self.speculative_num_draft_tokens = int(
            server_args.speculative_num_draft_tokens
        )
        self.proposed_token_num = self.speculative_num_draft_tokens - 1
        self.runner_config = FastDllmV2RunnerConfig.from_server_args(server_args)
        self.native_draft_worker: Optional[TpModelWorker] = None
        self.native_draft_model_runner: Optional[ModelRunner] = None
        if self.runner_config.runtime not in (
            "transformers",
            "sglang_native",
            "native",
        ):
            raise ValueError(
                "FAST_DLLM_V2 runtime must be 'transformers' or "
                f"'sglang_native', got {self.runner_config.runtime!r}."
            )

        runtime = None
        if self._is_draft_rank and self.runner_config.runtime in (
            "sglang_native",
            "native",
        ):
            self.native_draft_model_runner = self._init_native_draft_model_runner(
                server_args=server_args,
                gpu_id=gpu_id,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
            )
            runtime = SGLangNativeFastDllmV2Runtime(self.native_draft_model_runner)

        self.runner = (
            FastDllmV2ProposalRunner(self.runner_config, runtime=runtime)
            if self._is_draft_rank
            else None
        )
        self._logged_first_verify = False
        self._proposal_profile_log_count = 0
        if self.tp_rank == 0:
            logger.info(
                "Initialized FAST_DLLM_V2 standalone speculator. "
                "model=%s, runtime=%s, target_tp=%s, draft_tp=%s, verify_block=%s, "
                "proposed_tokens=%s.",
                server_args.speculative_draft_model_path,
                self.runner_config.runtime,
                self.tp_size,
                self.draft_tp_size,
                self.speculative_num_draft_tokens,
                self.proposed_token_num,
            )
        elif not self._is_draft_rank:
            logger.info(
                "FAST_DLLM_V2: target TP rank %s is a passive draft participant.",
                tp_rank,
            )

    @contextmanager
    def _native_draft_context(self):
        if self.draft_tp_size == 1 and self.tp_size > 1:
            draft_group = SingleRankDraftGroup(self.target_worker.world_group)
            with (
                single_rank_draft_context(draft_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                yield
        else:
            with (
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                yield

    def _build_native_draft_args(self, server_args: ServerArgs) -> ServerArgs:
        draft_args = deepcopy(server_args)
        draft_args.model_path = self.runner_config.model_path
        draft_args.tokenizer_path = self.runner_config.tokenizer_path
        draft_args.served_model_name = self.runner_config.model_path
        draft_args.revision = draft_args.speculative_draft_model_revision
        draft_args.skip_tokenizer_init = True
        draft_args.trust_remote_code = self.runner_config.trust_remote_code
        # Fast_dLLM_v2 is an independent draft model, not a target-context
        # extension model. Let ModelConfig derive the draft's own safe context
        # length unless the Fast_dLLM_v2 algorithm config explicitly overrides it.
        draft_args.context_length = self.runner_config.context_length
        draft_args.speculative_algorithm = None
        draft_args.dllm_algorithm = self.runner_config.native_dllm_algorithm
        # Keep the native draft ModelRunner's DllmConfig in lockstep with the
        # Fast_dLLM_v2 proposal config. CUDA graph capture sizes DLLM input
        # buffers from DllmConfig.block_size, while the proposal loop sizes
        # ForwardBatch input_ids from runner_config.block_size.
        draft_args.dllm_algorithm_config = (
            server_args.speculative_fast_dllm_v2_algorithm_config
        )
        draft_args.disable_overlap_schedule = True
        draft_args.disable_cuda_graph = self.runner_config.native_disable_cuda_graph
        draft_args.disable_piecewise_cuda_graph = (
            self.runner_config.native_disable_piecewise_cuda_graph
        )
        if (
            not draft_args.disable_cuda_graph
            and self.runner_config.proposal_kwargs.get("selective_logits", True)
        ):
            draft_args.dllm_cuda_graph_logit_positions_size = (
                self.runner_config.small_block_size
            )
        if not draft_args.disable_cuda_graph:
            if self.runner_config.native_cuda_graph_bs is None:
                draft_args.cuda_graph_bs = [1]
            else:
                draft_args.cuda_graph_bs = list(
                    self.runner_config.native_cuda_graph_bs
                )
            if not draft_args.cuda_graph_bs:
                raise ValueError(
                    "FAST_DLLM_V2 native_cuda_graph_bs cannot be empty."
                )
            draft_args.cuda_graph_max_bs = (
                self.runner_config.native_cuda_graph_max_bs
                if self.runner_config.native_cuda_graph_max_bs is not None
                else max(draft_args.cuda_graph_bs)
            )
            if draft_args.cuda_graph_max_bs < max(draft_args.cuda_graph_bs):
                raise ValueError(
                    "FAST_DLLM_V2 native_cuda_graph_max_bs must be greater than "
                    "or equal to max(native_cuda_graph_bs)."
                )
        else:
            if self.runner_config.native_cuda_graph_max_bs is not None:
                draft_args.cuda_graph_max_bs = (
                    self.runner_config.native_cuda_graph_max_bs
                )
            if self.runner_config.native_cuda_graph_bs is not None:
                draft_args.cuda_graph_bs = list(
                    self.runner_config.native_cuda_graph_bs
                )
        draft_args.max_total_tokens = self._resolved_native_max_total_tokens()
        draft_args.max_running_requests = (
            self.runner_config.native_max_running_requests
        )

        if self.draft_tp_size == 1 and self.tp_size > 1:
            draft_args.tp_size = 1
            draft_args.ep_size = 1
            draft_args.enable_dp_attention = False

        draft_backend = draft_args.speculative_draft_attention_backend
        if draft_backend is None:
            draft_backend, _ = draft_args.get_attention_backends()
        if draft_backend is not None:
            draft_args.attention_backend = draft_backend
            draft_args.prefill_attention_backend = None
            draft_args.decode_attention_backend = None
            draft_args.speculative_draft_attention_backend = None
        if (
            not draft_args.disable_cuda_graph
            and draft_args.attention_backend != "flashinfer"
        ):
            logger.info(
                "FAST_DLLM_V2 native CUDA graph requires the flashinfer "
                "attention backend; overriding draft attention_backend=%s.",
                draft_args.attention_backend,
            )
            draft_args.attention_backend = "flashinfer"

        return draft_args

    def _minimum_native_total_tokens(self) -> int:
        active_requests = int(self.runner_config.proposal_batch_size)
        if active_requests <= 0:
            active_requests = 1
        active_requests = min(
            active_requests,
            int(self.runner_config.native_max_running_requests),
        )
        active_requests = max(1, active_requests)

        slack_blocks = max(0, int(self.runner_config.native_memory_pool_slack_blocks))
        # Each active proposal can hold a prefix/cache handle plus one refinement
        # block. Slack absorbs temporary block-cache refreshes and allocator
        # rounding without inheriting the target model's large serving pool.
        return max(
            int(self.runner_config.block_size) * 4,
            int(self.runner_config.block_size)
            * (2 * active_requests + slack_blocks),
        )

    def _resolved_native_max_total_tokens(self) -> int:
        minimum_tokens = self._minimum_native_total_tokens()
        configured_tokens = self.runner_config.native_max_total_tokens
        if configured_tokens is None:
            return max(4096, minimum_tokens)
        configured_tokens = int(configured_tokens)
        if configured_tokens < minimum_tokens:
            logger.warning(
                "FAST_DLLM_V2 native_max_total_tokens=%s is below the minimum "
                "safe pool size %s for proposal_batch_size=%s; using %s.",
                configured_tokens,
                minimum_tokens,
                self.runner_config.proposal_batch_size,
                minimum_tokens,
            )
            return minimum_tokens
        return configured_tokens

    def _native_draft_memory_pool_config(self) -> MemoryPoolConfig:
        max_total_tokens = self._resolved_native_max_total_tokens()
        max_running_requests = int(self.runner_config.native_max_running_requests)
        if max_running_requests <= 0:
            raise ValueError(
                "FAST_DLLM_V2 native_max_running_requests must be positive, "
                f"got {max_running_requests}."
            )

        return MemoryPoolConfig(
            max_total_num_tokens=max_total_tokens,
            max_running_requests=max_running_requests,
        )

    def _init_native_draft_model_runner(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
    ) -> ModelRunner:
        draft_args = self._build_native_draft_args(server_args)
        memory_pool_config = self._native_draft_memory_pool_config()
        saved_server_args = get_global_server_args()
        try:
            with self._native_draft_context():
                if self.draft_tp_size == 1 and self.tp_size > 1:
                    draft_model_config = ModelConfig.from_server_args(
                        draft_args,
                        model_path=draft_args.speculative_draft_model_path,
                        model_revision=draft_args.speculative_draft_model_revision,
                        is_draft_model=True,
                    )
                    draft_model_runner = ModelRunner(
                        model_config=draft_model_config,
                        mem_fraction_static=draft_args.mem_fraction_static,
                        gpu_id=gpu_id,
                        tp_rank=0,
                        tp_size=1,
                        moe_ep_rank=0,
                        moe_ep_size=1,
                        pp_rank=0,
                        pp_size=draft_args.pp_size,
                        nccl_port=nccl_port,
                        dp_rank=dp_rank,
                        attn_cp_rank=attn_cp_rank,
                        moe_dp_rank=moe_dp_rank,
                        server_args=draft_args,
                        is_draft_worker=True,
                        memory_pool_config=memory_pool_config,
                    )
                else:
                    self.native_draft_worker = TpModelWorker(
                        server_args=draft_args,
                        gpu_id=gpu_id,
                        tp_rank=self.draft_tp_plan.local_rank(self.tp_rank),
                        moe_ep_rank=moe_ep_rank,
                        pp_rank=0,
                        attn_cp_rank=attn_cp_rank,
                        moe_dp_rank=moe_dp_rank,
                        dp_rank=dp_rank,
                        nccl_port=nccl_port,
                        is_draft_worker=True,
                        memory_pool_config=memory_pool_config,
                    )
                    draft_model_runner = self.native_draft_worker.model_runner
        finally:
            set_global_server_args_for_scheduler(saved_server_args)

        if self.tp_rank == 0:
            logger.info(
                "FAST_DLLM_V2 native draft model loaded. model=%s, "
                "target_tp=%s, draft_tp=%s, local_rank=%s, "
                "draft_max_total_tokens=%s, draft_max_running_requests=%s, "
                "draft_cuda_graph=%s, draft_piecewise_cuda_graph=%s.",
                server_args.speculative_draft_model_path,
                self.tp_size,
                self.draft_tp_size,
                self.draft_tp_plan.local_rank(self.tp_rank),
                memory_pool_config.max_total_num_tokens,
                memory_pool_config.max_running_requests,
                not draft_args.disable_cuda_graph,
                not draft_args.disable_piecewise_cuda_graph,
            )
        return draft_model_runner

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        if self.runner is not None:
            self.runner.states.clear()

    def release_request_ids(
        self, request_ids: list[str], *, cache_prefix: bool = True
    ) -> None:
        del cache_prefix
        if self.runner is not None:
            self.runner.release(request_ids)

    def _request_input_ids(self, batch: ScheduleBatch) -> list[list[int]]:
        return [
            list(req.origin_input_ids) + list(req.output_ids) for req in batch.reqs
        ]

    def _build_proposal_on_draft_rank(self, batch: ScheduleBatch) -> LinearDraftBlock:
        if self.runner is None:
            raise RuntimeError(
                "FAST_DLLM_V2 proposal requested on a non-participating draft rank."
            )
        input_ids = self._request_input_ids(batch)
        current_token_ids = torch.tensor(
            [ids[-1] for ids in input_ids],
            dtype=torch.long,
            device=self.device,
        )
        prefix_lens = batch.seq_lens.to(device=self.device, dtype=torch.int64)
        request = IndependentDllmDraftRequest(
            request_ids=[req.rid for req in batch.reqs],
            input_ids=input_ids,
            current_token_ids=current_token_ids,
            prefix_lens=prefix_lens,
            proposed_token_num=self.proposed_token_num,
        )
        tokens = self.runner.propose(request)
        self._maybe_log_proposal_profile(tokens.metadata)
        return build_linear_draft_block(
            current_token_ids=tokens.current_token_ids,
            proposed_token_ids=tokens.proposed_token_ids,
            prefix_lens=tokens.prefix_lens,
        )

    def _maybe_log_proposal_profile(self, metadata: dict) -> None:
        profile = metadata.get("profile_total")
        if not profile:
            return

        interval = max(1, int(metadata.get("profile_log_interval", 1)))
        self._proposal_profile_log_count += 1
        if self._proposal_profile_log_count % interval:
            return

        logger.info(
            "FAST_DLLM_V2 proposal profile: %s",
            json.dumps(profile, sort_keys=True),
        )

    def _serialize_linear_block(self, block: LinearDraftBlock) -> dict:
        return {
            "draft_token": _cpu_tensor(block.draft_token),
            "positions": _cpu_tensor(block.positions),
            "draft_token_num": block.draft_token_num,
        }

    def _deserialize_linear_block(self, payload: dict) -> LinearDraftBlock:
        return LinearDraftBlock(
            draft_token=payload["draft_token"].to(self.device, non_blocking=True),
            positions=payload["positions"].to(self.device, non_blocking=True),
            draft_token_num=int(payload["draft_token_num"]),
        )

    def _broadcast_proposal_result(
        self,
        payload: list[_BroadcastedFastDllmV2Proposal],
    ) -> _BroadcastedFastDllmV2Proposal:
        world_group = self.target_worker.world_group
        return broadcast_pyobj(
            payload,
            world_group.rank,
            world_group.cpu_group,
            src=world_group.first_rank,
        )[0]

    def _propose_linear_block(self, batch: ScheduleBatch) -> LinearDraftBlock:
        payload: list[_BroadcastedFastDllmV2Proposal]
        if self._is_draft_rank:
            try:
                block = self._build_proposal_on_draft_rank(batch)
                payload = [
                    _BroadcastedFastDllmV2Proposal(
                        ok=True,
                        payload=self._serialize_linear_block(block),
                    )
                ]
            except Exception:
                logger.exception("FAST_DLLM_V2 proposal failed on draft rank.")
                payload = [
                    _BroadcastedFastDllmV2Proposal(
                        ok=False,
                        error=traceback.format_exc(),
                    )
                ]
        else:
            payload = []

        broadcasted = self._broadcast_proposal_result(payload)
        if not broadcasted.ok:
            raise RuntimeError(
                "FAST_DLLM_V2 proposal failed on draft rank:\n"
                f"{broadcasted.error}"
            )
        if broadcasted.payload is None:
            raise RuntimeError("FAST_DLLM_V2 proposal returned no payload.")
        return self._deserialize_linear_block(broadcasted.payload)

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

        block = self._propose_linear_block(batch)
        verify_input = build_linear_verify_input(
            block,
            requires_target_hidden=False,
        )
        verify_input.capture_hidden_mode = CaptureHiddenMode.NULL
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        batch.spec_algorithm = SpeculativeAlgorithm.FAST_DLLM_V2
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "FAST_DLLM_V2 speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch(),
                **kwargs,
            )

        old_output_lens = [len(req.output_ids) for req in batch.reqs]

        set_time_batch(batch.reqs, "set_spec_draft_start_time")
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time")

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)

        set_time_batch(batch.reqs, "set_spec_verify_start_time")
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch,
            is_verify=True,
            **kwargs,
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        (
            new_verified_id,
            _commit_lens,
            _next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )

        accepted_token_ids = [
            req.output_ids[old_len:]
            for req, old_len in zip(batch.reqs, old_output_lens, strict=True)
        ]
        if self.runner is not None:
            self.runner.extend_after_accept(
                IndependentDllmAcceptedTokens(
                    request_ids=[req.rid for req in batch.reqs],
                    accepted_token_ids=accepted_token_ids,
                )
            )
        for req, accepted in zip(
            batch.reqs, accept_length_per_req_cpu, strict=True
        ):
            req.time_stats.set_spec_verify_end_time(accepted_tokens=accepted)
            req.update_spec_acceptance_histogram(accepted)

        batch.forward_mode = ForwardMode.DECODE
        num_accepted_drafts = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "FAST_DLLM_V2 verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_drafts=num_accepted_drafts,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
