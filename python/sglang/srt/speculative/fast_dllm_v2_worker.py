"""Standalone Fast_dLLM_v2 speculative worker for AR targets."""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.co_draft.dllm_linear_adapter import (
    IndependentDllmAcceptedTokens,
    IndependentDllmDraftRequest,
)
from sglang.srt.speculative.co_draft.fast_dllm_v2_runner import (
    FastDllmV2ProposalRunner,
    FastDllmV2RunnerConfig,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.linear_verify import (
    build_linear_draft_block,
    build_linear_verify_input,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class FastDllmV2Worker:
    """Spec-v1 worker that verifies independent Fast_dLLM_v2 token blocks.

    Fast_dLLM_v2 owns proposal generation through its Hugging Face
    ``trust_remote_code`` path. Once it emits token ids, the AR target verifies
    the linear block using the same target-side contract as DFlash, but without
    DFlash's target-hidden-conditioned draft cache.
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
        del gpu_id, dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.tp_rank = tp_rank
        self.device = target_worker.device
        self.speculative_num_draft_tokens = int(
            server_args.speculative_num_draft_tokens
        )
        self.proposed_token_num = self.speculative_num_draft_tokens - 1
        self.runner = FastDllmV2ProposalRunner(
            FastDllmV2RunnerConfig.from_server_args(server_args)
        )
        self._logged_first_verify = False
        if self.tp_rank == 0:
            logger.info(
                "Initialized FAST_DLLM_V2 standalone speculator. "
                "model=%s, verify_block=%s, proposed_tokens=%s.",
                server_args.speculative_draft_model_path,
                self.speculative_num_draft_tokens,
                self.proposed_token_num,
            )

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        self.runner.states.clear()

    def release_request_ids(
        self, request_ids: list[str], *, cache_prefix: bool = True
    ) -> None:
        del cache_prefix
        self.runner.release(request_ids)

    def _request_input_ids(self, batch: ScheduleBatch) -> list[list[int]]:
        return [
            list(req.origin_input_ids) + list(req.output_ids) for req in batch.reqs
        ]

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            return

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
        block = build_linear_draft_block(
            current_token_ids=tokens.current_token_ids,
            proposed_token_ids=tokens.proposed_token_ids,
            prefix_lens=tokens.prefix_lens,
        )
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

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)

        set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)
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
        self.runner.extend_after_accept(
            IndependentDllmAcceptedTokens(
                request_ids=[req.rid for req in batch.reqs],
                accepted_token_ids=accepted_token_ids,
            )
        )
        for req, accepted in zip(
            batch.reqs, accept_length_per_req_cpu, strict=True
        ):
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
