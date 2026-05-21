# SPDX-License-Identifier: Apache-2.0
"""vLLM-style asymmetric colocated TLI worker."""

from __future__ import annotations

import copy
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_utils import (
    SingleRankDraftGroup,
    single_rank_draft_context,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.tli_worker import TLIWorker
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BroadcastedDraftResult:
    """Draft proposal result shared from rank 0 to the target TP group."""

    ok: bool
    payload: Optional[dict] = None
    draft_proposal_time: float = 0.0
    error: Optional[str] = None


def _cpu_tensor(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _device_tensor(tensor: Optional[torch.Tensor], device):
    if tensor is None:
        return None
    return tensor.to(device)


class AsymmetricTLIWorker(TLIWorker):
    """Run colocated TLI with a TP=1 draft on the target TP root rank.

    This follows vLLM's asymmetric TP shape: only the first target TP rank
    participates in draft generation, while the remaining target ranks remain
    passive draft participants and receive the root rank's verification payload.
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
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self._is_draft_rank = tp_rank == 0
        self._target_model_runner = target_worker.model_runner
        self._local_draft_tp_group = SingleRankDraftGroup(target_worker.world_group)
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.model_config = self._target_model_runner.model_config
        self.hot_token_id = None
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self._is_draft_rank:
            self._init_root_draft_model(
                server_args=server_args,
                gpu_id=gpu_id,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
            )
        else:
            logger.info(
                "Asymmetric colocated TLI: target TP rank %s is a passive draft participant.",
                tp_rank,
            )

    @property
    def target_model_runner(self):
        return self._target_model_runner

    @property
    def draft_model_runner(self):
        if not self._is_draft_rank:
            raise RuntimeError(
                "Non-root target TP ranks do not own a local draft model runner."
            )
        return self._draft_model_runner

    @property
    def model_runner(self):
        if self._is_draft_rank:
            return self.draft_model_runner
        return self.target_model_runner

    @contextmanager
    def _root_draft_context(self):
        with (
            single_rank_draft_context(self._local_draft_tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            yield

    def _init_root_draft_model(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
    ) -> None:
        draft_args = copy.copy(server_args)
        draft_args.tp_size = 1
        draft_args.ep_size = 1
        draft_args.enable_dp_attention = False
        draft_args.disable_cuda_graph = True
        draft_args.context_length = self.target_model_runner.model_config.context_len

        draft_model_config = ModelConfig.from_server_args(
            draft_args,
            model_path=draft_args.speculative_draft_model_path,
            model_revision=draft_args.speculative_draft_model_revision,
            is_draft_model=True,
        )
        with self._root_draft_context():
            self._draft_model_runner = ModelRunner(
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
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=self.target_model_runner.memory_pool_config,
            )
            self.model_config = self._draft_model_runner.model_config

            target_tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                tokenizer_revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )
            draft_tokenizer = get_tokenizer(
                server_args.remote_draft_tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                tokenizer_backend=server_args.tokenizer_backend,
            )
            self.vocab_mapping = self._create_vocab_mapping(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
            )
            self._try_prune_draft_lm_head(draft_args)
            self.init_attention_backend()

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        logger.info(
            "Asymmetric colocated TLI initialized: target_tp=%s, draft_tp=1, root_draft_rank=0.",
            server_args.tp_size,
        )

    def _create_vocab_mapping(self, *, target_tokenizer, draft_tokenizer):
        from sglang.srt.speculative.tli_token_translator import TLITokenTranslator

        return TLITokenTranslator(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=self.target_model_runner.model_config.vocab_size,
            draft_vocab_size=self.draft_model_runner.model_config.vocab_size,
            device=self.device,
        )

    def _broadcast_draft_result(
        self,
        payload: list[_BroadcastedDraftResult],
    ) -> _BroadcastedDraftResult:
        world_group = self.target_worker.world_group
        return broadcast_pyobj(
            payload,
            world_group.rank,
            world_group.cpu_group,
            src=world_group.first_rank,
        )[0]

    def _serialize_verify_input(self, spec_info: EagleVerifyInput) -> dict:
        return {
            "draft_token": _cpu_tensor(spec_info.draft_token),
            "custom_mask": _cpu_tensor(spec_info.custom_mask),
            "positions": _cpu_tensor(spec_info.positions),
            "retrieve_index": _cpu_tensor(spec_info.retrieve_index),
            "retrieve_next_token": _cpu_tensor(spec_info.retrieve_next_token),
            "retrieve_next_sibling": _cpu_tensor(spec_info.retrieve_next_sibling),
            "retrieve_cum_len": _cpu_tensor(spec_info.retrieve_cum_len),
            "spec_steps": spec_info.spec_steps,
            "topk": spec_info.topk,
            "draft_token_num": spec_info.draft_token_num,
            "capture_hidden_mode": spec_info.capture_hidden_mode,
            "seq_lens_sum": spec_info.seq_lens_sum,
            "seq_lens_cpu": _cpu_tensor(spec_info.seq_lens_cpu),
        }

    def _deserialize_verify_input(self, payload: dict) -> EagleVerifyInput:
        return EagleVerifyInput(
            draft_token=_device_tensor(payload["draft_token"], self.device),
            custom_mask=_device_tensor(payload["custom_mask"], self.device),
            positions=_device_tensor(payload["positions"], self.device),
            retrieve_index=_device_tensor(payload["retrieve_index"], self.device),
            retrieve_next_token=_device_tensor(
                payload["retrieve_next_token"], self.device
            ),
            retrieve_next_sibling=_device_tensor(
                payload["retrieve_next_sibling"], self.device
            ),
            retrieve_cum_len=_device_tensor(payload["retrieve_cum_len"], self.device),
            spec_steps=payload["spec_steps"],
            topk=payload["topk"],
            draft_token_num=payload["draft_token_num"],
            capture_hidden_mode=payload["capture_hidden_mode"],
            seq_lens_sum=payload["seq_lens_sum"],
            seq_lens_cpu=payload["seq_lens_cpu"],
        )

    def _draft_on_root(self, batch) -> tuple[EagleVerifyInput, float]:
        if not self._is_draft_rank:
            raise RuntimeError("Only the root draft rank can generate TLI proposals.")
        with self._root_draft_context():
            start_time = time.perf_counter()
            spec_info = self.draft(batch)
            draft_proposal_time = time.perf_counter() - start_time
        return spec_info, draft_proposal_time

    def _broadcast_draft(self, batch) -> tuple[EagleVerifyInput, float]:
        payload: list[_BroadcastedDraftResult]
        if self._is_draft_rank:
            try:
                spec_info, draft_proposal_time = self._draft_on_root(batch)
                payload = [
                    _BroadcastedDraftResult(
                        ok=True,
                        payload=self._serialize_verify_input(spec_info),
                        draft_proposal_time=draft_proposal_time,
                    )
                ]
            except Exception:
                logger.exception("Asymmetric colocated TLI draft failed on root rank.")
                payload = [
                    _BroadcastedDraftResult(
                        ok=False,
                        error=traceback.format_exc(),
                    )
                ]
        else:
            payload = []

        broadcasted = self._broadcast_draft_result(payload)
        if not broadcasted.ok:
            raise RuntimeError(
                "Asymmetric colocated TLI draft failed on root rank:\n"
                f"{broadcasted.error}"
            )
        if broadcasted.payload is None:
            raise RuntimeError("Asymmetric colocated TLI draft returned no payload.")
        return (
            self._deserialize_verify_input(broadcasted.payload),
            broadcasted.draft_proposal_time,
        )

    def _forward_draft_extend_on_root(self, batch, *args, **kwargs) -> float:
        if not self._is_draft_rank:
            return 0.0
        with self._root_draft_context():
            start_time = time.perf_counter()
            self.forward_draft_extend(batch, *args, **kwargs)
            return time.perf_counter() - start_time

    def _forward_draft_extend_after_decode_on_root(self, batch) -> float:
        if not self._is_draft_rank:
            return 0.0
        with self._root_draft_context():
            start_time = time.perf_counter()
            self.forward_draft_extend_after_decode(batch)
            return time.perf_counter() - start_time

    @staticmethod
    def _record_draft_timing(
        reqs,
        *,
        draft_proposal_time: float,
        grpc_communication_time: float = 0.0,
        draft_queue_scheduling_time: float = 0.0,
    ) -> None:
        """Record colocated draft timings using the shared speculative schema.

        In asymmetric colocated TLI, the draft runs in-process on the root
        target rank, so transport cost is zero and draft-side queueing is only
        present if the local draft path explicitly introduces it.
        """
        for req in reqs:
            req.time_stats.observe_draft_rpc_timing(
                grpc_communication_time=grpc_communication_time,
                draft_queue_scheduling_time=draft_queue_scheduling_time,
                draft_proposal_time=draft_proposal_time,
            )

    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            draft_proposal_time = self._forward_draft_extend_on_root(
                batch,
                logits_output.hidden_states,
                next_token_ids,
                seq_lens_cpu,
                logits_output.mm_input_embeds,
            )
            self._record_draft_timing(
                batch.reqs,
                draft_proposal_time=draft_proposal_time,
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        set_time_batch(batch.reqs, "set_spec_draft_start_time")
        spec_info, draft_proposal_time = self._broadcast_draft(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time")
        self._record_draft_timing(
            batch.reqs,
            draft_proposal_time=draft_proposal_time,
        )

        set_time_batch(batch.reqs, "set_spec_verify_start_time")
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
            self.verify(batch, spec_info)
        )
        del model_worker_batch

        for idx, req in enumerate(batch.reqs):
            accepted = verify_output.accept_length_per_req_cpu[idx]
            req.time_stats.set_spec_verify_end_time(accepted_tokens=accepted)

        set_time_batch(batch.reqs, "set_spec_draft_extend_start_time", trace_only=True)
        if batch.spec_info.verified_id.shape[0] > 0:
            draft_extend_time = self._forward_draft_extend_after_decode_on_root(batch)
            self._record_draft_timing(
                batch.reqs,
                draft_proposal_time=draft_extend_time,
            )
        set_time_batch(batch.reqs, "set_spec_draft_extend_end_time", trace_only=True)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_drafts=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
