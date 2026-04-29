# SPDX-License-Identifier: Apache-2.0
"""TLI (Token-Level Intersection) speculative decoding worker."""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.spec_utils import (
    fast_topk,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.standalone_worker import StandaloneWorker
from sglang.srt.speculative.vocab_mapping import VocabMapping
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)


class _PrunedReindexLMHead(nn.Module):
    def __init__(
        self,
        pruned_weight: torch.Tensor,
        pruned_bias: Optional[torch.Tensor],
        intersection_ids: torch.Tensor,
        full_vocab_size: int,
        use_fp32: bool = False,
    ):
        super().__init__()
        self.register_buffer("weight", pruned_weight)
        if pruned_bias is not None:
            self.register_buffer("bias", pruned_bias)
        else:
            self.bias = None
        self.register_buffer("intersection_ids", intersection_ids)
        self.full_vocab_size = full_vocab_size
        self.use_fp32 = use_fp32

    def set_lora(self, *args, **kwargs):
        pass

    def apply_lora(self, *args, **kwargs):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp32:
            compact = torch.matmul(x.to(torch.float32), self.weight.to(torch.float32).T)
        else:
            compact = torch.matmul(x.to(self.weight.dtype), self.weight.T)
        if self.bias is not None:
            compact = compact + self.bias
        out = torch.full(
            (x.shape[0], self.full_vocab_size),
            float("-inf"),
            dtype=compact.dtype,
            device=x.device,
        )
        out[:, self.intersection_ids] = compact
        return out


class TLIWorker(StandaloneWorker):
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
        self._defer_cuda_graphs = True
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        self.hot_token_id = None

        target_tokenizer_path = server_args.tokenizer_path or server_args.model_path
        draft_tokenizer_path = server_args.speculative_draft_model_path

        target_tokenizer = get_tokenizer(
            target_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_revision=server_args.revision,
        )
        draft_tokenizer = get_tokenizer(
            draft_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_revision=server_args.speculative_draft_model_revision,
        )

        self.vocab_mapping = VocabMapping(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            draft_vocab_size=self.draft_model_runner.model_config.vocab_size,
            device=self.device,
        )

        self._try_prune_draft_lm_head(server_args)

        self._defer_cuda_graphs = False
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().init_cuda_graphs()

    def init_cuda_graphs(self):
        if getattr(self, "_defer_cuda_graphs", False):
            return
        super().init_cuda_graphs()

    def _try_prune_draft_lm_head(self, server_args) -> None:
        try:
            lm_head = self.draft_model_runner.model.lm_head
        except AttributeError:
            logger.warning("Draft model has no lm_head; skipping LM head pruning.")
            return

        if not hasattr(lm_head, "weight"):
            logger.warning(
                "Draft LM head has no weight attribute; skipping LM head pruning."
            )
            return

        intersection_ids = self.vocab_mapping.intersection_draft_ids
        full_vocab_size = self.vocab_mapping.draft_vocab_size
        
        tp_size = getattr(self.draft_model_runner.tp_group, "world_size", 1)
        if tp_size > 1:
            if not hasattr(lm_head, "shard_indices"):
                logger.warning("Draft LM head is tensor parallel but missing shard_indices; skipping LM head pruning.")
                return
                
            start_idx = lm_head.shard_indices.org_vocab_start_index
            end_idx = lm_head.shard_indices.org_vocab_end_index
            
            # Keep only the intersection IDs that fall into this rank's partition
            mask = (intersection_ids >= start_idx) & (intersection_ids < end_idx)
            local_ids = intersection_ids[mask]
            
            # Map global intersection IDs to indices within the local weight tensor
            intersection_ids_local = local_ids - start_idx
            
            # The pruned reindex head must output the local pad length for all_gather to work
            full_vocab_size = lm_head.weight.data.shape[0]
            reindex_ids = intersection_ids_local
        else:
            reindex_ids = intersection_ids
            intersection_ids_local = intersection_ids
            
        pruned_weight = lm_head.weight.data[intersection_ids_local].clone()
        pruned_bias = (
            lm_head.bias.data[intersection_ids_local].clone()
            if getattr(lm_head, "bias", None) is not None
            else None
        )

        pruned_head = _PrunedReindexLMHead(
            pruned_weight=pruned_weight,
            pruned_bias=pruned_bias,
            intersection_ids=reindex_ids,
            full_vocab_size=full_vocab_size,
            use_fp32=getattr(server_args, "enable_fp32_lm_head", False),
        )
        self.draft_model_runner.model.lm_head = pruned_head
        logger.info(
            "Draft LM head pruned to %d/%d intersection tokens.",
            intersection_ids.numel(),
            full_vocab_size,
        )

    def forward_draft_extend(
        self,
        batch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu,
        mm_input_embeds=None,
    ):
        batch.input_ids = self.vocab_mapping.map_target_to_draft_ids(batch.input_ids)
        draft_next_token_ids = self.vocab_mapping.map_target_to_draft_ids(next_token_ids)
        super().forward_draft_extend(
            batch, hidden_states, draft_next_token_ids, seq_lens_cpu, mm_input_embeds
        )

    def forward_draft_extend_after_decode(self, batch):
        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        original_verified_id = spec_info.verified_id
        spec_info.verified_id = self.vocab_mapping.map_target_to_draft_ids(
            original_verified_id
        )
        super().forward_draft_extend_after_decode(batch)
        batch.spec_info.verified_id = self.vocab_mapping.map_draft_to_target_ids(
            batch.spec_info.verified_id
        )

    def capture_for_decode(self, logits_output, draft_input: EagleDraftInput):
        constrained_logits = self.vocab_mapping.constrain_draft_logits(
            logits_output.next_token_logits
        )
        probs = torch.softmax(constrained_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(self.vocab_mapping.map_draft_to_target_ids(tree_info[1]))
            parents_list.append(tree_info[2])

            if i == self.speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            if self.model_config.hf_config.architectures[0] == "GptOssForCausalLM":
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")

            constrained_logits = self.vocab_mapping.constrain_draft_logits(
                logits_output.next_token_logits
            )
            probs = torch.softmax(constrained_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                constrained_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={constrained_logits.shape[-1]}",
            )
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        return parent_list, top_scores_index, draft_tokens