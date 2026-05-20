# SPDX-License-Identifier: Apache-2.0
"""RPC-friendly request/response objects for remote draft forwarding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Literal, Optional, Sequence, TYPE_CHECKING

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

if TYPE_CHECKING:
    from sglang.srt.speculative.tli_token_translator import TLITokenTranslator

DraftForwardMode = Literal["extend", "decode", "extend_after_decode", "release"]


@dataclass(slots=True)
class DraftForwardRequest:
    """Request sent from the target server to the draft server."""

    request_id: str
    verified_id: torch.Tensor
    hidden_states: torch.Tensor
    request_ids: Optional[List[str]] = None
    input_ids: Optional[torch.Tensor] = None
    tp_rank: int = 0
    tp_size: int = 1
    mode: DraftForwardMode = "decode"
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
    target_prefix_lens_for_draft_extend_cpu: Optional[torch.Tensor] = None
    mm_input_embeds: Optional[torch.Tensor] = None
    round_ids: Optional[List[int]] = None
    token_positions: Optional[List[int]] = None
    prefix_versions: Optional[List[int]] = None
    cache_prefix_on_release: bool = False
    server_received_time: float = 0.0

    def to_draft_vocab(self, translator: TLITokenTranslator) -> DraftForwardRequest:
        """Translate token-id payloads into draft vocabulary.

        Only fields that represent vocabulary IDs are translated here. Structural
        metadata such as TP rank, request ordering, lengths, and hidden states
        intentionally remain unchanged.
        """
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
class DraftForwardResponse:
    """Response sent from the draft server back to the target server."""

    request_id: str
    parent_list: torch.Tensor
    top_scores_index: torch.Tensor
    draft_token_ids: torch.Tensor
    mode: DraftForwardMode = "decode"
    next_hidden_states: Optional[torch.Tensor] = None
    next_topk_p: Optional[torch.Tensor] = None
    next_topk_index: Optional[torch.Tensor] = None
    round_ids: Optional[List[int]] = None
    token_positions: Optional[List[int]] = None
    prefix_versions: Optional[List[int]] = None
    server_total_time: float = 0.0
    server_queue_scheduling_time: float = 0.0
    server_model_forward_time: float = 0.0

    def to_target_vocab(self, translator: TLITokenTranslator) -> DraftForwardResponse:
        """Translate token-id payloads back into target vocabulary.

        ``parent_list`` and ``top_scores_index`` are structural indices and are
        not vocab-dependent. ``draft_token_ids`` is the token-bearing payload that
        must be translated.
        """
        return replace(
            self,
            draft_token_ids=translator.translate_draft_to_target_ids(
                self.draft_token_ids
            ),
        )


def _request_ids(request: DraftForwardRequest) -> list[str]:
    return list(request.request_ids or [request.request_id])


def _extend_optional_list(values: Sequence[list[int] | None]) -> list[int] | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Cannot merge partially-present ordering metadata.")
    merged: list[int] = []
    for value in values:
        assert value is not None
        merged.extend(value)
    return merged


def _cat_optional_tensors(values: Sequence[torch.Tensor | None]) -> torch.Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Cannot merge partially-present tensor metadata.")
    return torch.cat([value for value in values if value is not None], dim=0)


def can_merge_draft_forward_requests(requests: Sequence[DraftForwardRequest]) -> bool:
    if len(requests) <= 1:
        return False

    first = requests[0]
    merge_keys = (
        "mode",
        "tp_rank",
        "tp_size",
        "capture_hidden_mode",
        "topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "num_tokens_per_req",
        "num_tokens_for_logprob_per_req",
        "cache_prefix_on_release",
    )
    for request in requests[1:]:
        if any(getattr(request, key) != getattr(first, key) for key in merge_keys):
            return False

    if first.mode == "release":
        return True

    if any(request.hidden_states is None for request in requests):
        return False
    if any(request.verified_id is None for request in requests):
        return False

    optional_tensor_fields = (
        "input_ids",
        "accept_length",
        "seq_lens_for_draft_extend",
        "seq_lens_for_draft_extend_cpu",
        "target_prefix_lens_for_draft_extend_cpu",
        "mm_input_embeds",
    )
    for field_name in optional_tensor_fields:
        present = [getattr(request, field_name) is not None for request in requests]
        if any(present) and not all(present):
            return False

    optional_list_fields = (
        "accept_length_cpu",
        "round_ids",
        "token_positions",
        "prefix_versions",
    )
    for field_name in optional_list_fields:
        present = [getattr(request, field_name) is not None for request in requests]
        if any(present) and not all(present):
            return False

    return True


def merge_draft_forward_requests(
    requests: Sequence[DraftForwardRequest],
) -> DraftForwardRequest:
    if not can_merge_draft_forward_requests(requests):
        raise ValueError("DraftForward requests are not merge-compatible.")

    request_ids: list[str] = []
    for request in requests:
        request_ids.extend(_request_ids(request))

    first = requests[0]
    return replace(
        first,
        request_id=f"merged:{','.join(request.request_id for request in requests)}",
        request_ids=request_ids,
        verified_id=torch.cat([request.verified_id for request in requests], dim=0),
        hidden_states=torch.cat([request.hidden_states for request in requests], dim=0),
        input_ids=_cat_optional_tensors([request.input_ids for request in requests]),
        accept_length=_cat_optional_tensors(
            [request.accept_length for request in requests]
        ),
        accept_length_cpu=_extend_optional_list(
            [request.accept_length_cpu for request in requests]
        ),
        seq_lens_for_draft_extend=_cat_optional_tensors(
            [request.seq_lens_for_draft_extend for request in requests]
        ),
        seq_lens_for_draft_extend_cpu=_cat_optional_tensors(
            [request.seq_lens_for_draft_extend_cpu for request in requests]
        ),
        target_prefix_lens_for_draft_extend_cpu=_cat_optional_tensors(
            [request.target_prefix_lens_for_draft_extend_cpu for request in requests]
        ),
        mm_input_embeds=_cat_optional_tensors(
            [request.mm_input_embeds for request in requests]
        ),
        round_ids=_extend_optional_list([request.round_ids for request in requests]),
        token_positions=_extend_optional_list(
            [request.token_positions for request in requests]
        ),
        prefix_versions=_extend_optional_list(
            [request.prefix_versions for request in requests]
        ),
    )


def _split_rows(
    tensor: torch.Tensor,
    row_counts: Sequence[int],
) -> list[torch.Tensor]:
    if tensor.dim() == 0:
        raise ValueError("Cannot split a scalar DraftForward response tensor.")
    if tensor.shape[0] == sum(row_counts):
        return list(torch.split(tensor, list(row_counts), dim=0))
    if tensor.numel() == 0:
        return [tensor.new_empty(tensor.shape) for _ in row_counts]
    if tensor.shape[0] != sum(row_counts):
        if len(row_counts) == 1:
            return [tensor]
        raise ValueError(
            "DraftForward merged response row count does not match requests: "
            f"rows={tensor.shape[0]}, expected={sum(row_counts)}."
        )
    raise AssertionError("unreachable DraftForward response split state")


def _split_optional_rows(
    tensor: torch.Tensor | None,
    row_counts: Sequence[int],
) -> list[torch.Tensor | None]:
    if tensor is None:
        return [None for _ in row_counts]
    return _split_rows(tensor, row_counts)


def _split_optional_list(
    values: list[int] | None,
    row_counts: Sequence[int],
) -> list[list[int] | None]:
    if values is None:
        return [None for _ in row_counts]
    splits: list[list[int] | None] = []
    offset = 0
    for row_count in row_counts:
        splits.append(values[offset : offset + row_count])
        offset += row_count
    if offset != len(values):
        raise ValueError(
            "DraftForward merged response metadata length does not match requests: "
            f"values={len(values)}, expected={offset}."
        )
    return splits


def split_merged_draft_forward_response(
    response: DraftForwardResponse,
    requests: Sequence[DraftForwardRequest],
) -> list[DraftForwardResponse]:
    row_counts = [len(_request_ids(request)) for request in requests]
    parent_lists = _split_rows(response.parent_list, row_counts)
    top_scores_indices = _split_rows(response.top_scores_index, row_counts)
    draft_token_ids = _split_rows(response.draft_token_ids, row_counts)
    next_hidden_states = _split_optional_rows(response.next_hidden_states, row_counts)
    next_topk_p = _split_optional_rows(response.next_topk_p, row_counts)
    next_topk_index = _split_optional_rows(response.next_topk_index, row_counts)
    round_ids = _split_optional_list(response.round_ids, row_counts)
    token_positions = _split_optional_list(response.token_positions, row_counts)
    prefix_versions = _split_optional_list(response.prefix_versions, row_counts)

    outputs: list[DraftForwardResponse] = []
    for i, request in enumerate(requests):
        outputs.append(
            DraftForwardResponse(
                request_id=request.request_id,
                parent_list=parent_lists[i],
                top_scores_index=top_scores_indices[i],
                draft_token_ids=draft_token_ids[i],
                mode=response.mode,
                next_hidden_states=next_hidden_states[i],
                next_topk_p=next_topk_p[i],
                next_topk_index=next_topk_index[i],
                round_ids=round_ids[i],
                token_positions=token_positions[i],
                prefix_versions=prefix_versions[i],
                server_total_time=response.server_total_time,
                server_queue_scheduling_time=response.server_queue_scheduling_time,
                server_model_forward_time=response.server_model_forward_time,
            )
        )
    return outputs
