# SPDX-License-Identifier: Apache-2.0
"""RPC-friendly request/response objects for disaggregated TLI speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Literal, Optional, TYPE_CHECKING

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

if TYPE_CHECKING:
    from sglang.srt.speculative.tli_token_translator import TLITokenTranslator

TLIDraftMode = Literal["extend", "decode", "extend_after_decode", "release"]


@dataclass(slots=True)
class TLIDraftRequest:
    """Request sent from the target server to the draft server."""

    request_id: str
    verified_id: torch.Tensor
    hidden_states: torch.Tensor
    request_ids: Optional[List[str]] = None
    input_ids: Optional[torch.Tensor] = None
    tp_rank: int = 0
    tp_size: int = 1
    mode: TLIDraftMode = "decode"
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    topk: int = 1
    speculative_num_steps: int = 1
    speculative_num_draft_tokens: int = 1
    num_tokens_per_req: int = 1
    num_tokens_for_logprob_per_req: int = 1
    accept_length: Optional[torch.Tensor] = None
    accept_length_cpu: Optional[List[int]] = None
    seq_lens_for_draft_extend: Optional[torch.Tensor] = None
    seq_lens_for_draft_extend_cpu: Optional[torch.Tensor] = None
    mm_input_embeds: Optional[torch.Tensor] = None

    def to_draft_vocab(self, translator: TLITokenTranslator) -> TLIDraftRequest:
        """Translate target-vocab token ids into draft-vocab token ids."""
        return replace(
            self,
            verified_id=translator.translate_target_to_draft_ids(self.verified_id),
            input_ids=(
                translator.translate_target_to_draft_ids(self.input_ids)
                if self.input_ids is not None
                else None
            ),
        )


@dataclass(slots=True)
class TLIDraftResponse:
    """Response sent from the draft server back to the target server."""

    request_id: str
    parent_list: torch.Tensor
    top_scores_index: torch.Tensor
    draft_token_ids: torch.Tensor
    mode: TLIDraftMode = "decode"
    next_hidden_states: Optional[torch.Tensor] = None
    next_topk_p: Optional[torch.Tensor] = None
    next_topk_index: Optional[torch.Tensor] = None

    def to_target_vocab(self, translator: TLITokenTranslator) -> TLIDraftResponse:
        """Translate draft-vocab token ids into target-vocab token ids."""
        return replace(
            self,
            draft_token_ids=translator.translate_draft_to_target_ids(
                self.draft_token_ids
            ),
        )
