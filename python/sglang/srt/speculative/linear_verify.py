"""Shared linear target-verification helpers for speculative draft blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash_info import DFlashVerifyInput


@dataclass(frozen=True, slots=True)
class LinearDraftBlock:
    """A flat draft-token block for target verification.

    This is the common output shape for DFlash and independent block-diffusion
    dLLM drafts such as Fast_dLLM_v2. The draft model may be very different, but
    once it proposes token ids, the AR target verifies a linear candidate block.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int


def build_linear_draft_block(
    *,
    current_token_ids: torch.Tensor,
    proposed_token_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> LinearDraftBlock:
    """Convert next-token proposals into the linear target-verify convention.

    Linear verification includes the already-verified current token at column 0.
    If a draft proposes ``K`` next tokens, the target verifies a block of
    ``K + 1`` tokens:

    ``[current_token, proposed_0, proposed_1, ... proposed_K-1]``.
    """

    if current_token_ids.ndim != 1:
        raise ValueError(
            "current_token_ids must be 1D, "
            f"got shape={tuple(current_token_ids.shape)}."
        )
    if proposed_token_ids.ndim != 2:
        raise ValueError(
            "proposed_token_ids must be 2D, "
            f"got shape={tuple(proposed_token_ids.shape)}."
        )
    if prefix_lens.ndim != 1:
        raise ValueError(
            f"prefix_lens must be 1D, got shape={tuple(prefix_lens.shape)}."
        )

    bs = int(current_token_ids.shape[0])
    if bs <= 0:
        raise ValueError("batch size must be positive.")
    if int(proposed_token_ids.shape[0]) != bs:
        raise ValueError(
            "proposed_token_ids batch size mismatch: "
            f"expected {bs}, got {int(proposed_token_ids.shape[0])}."
        )
    if int(prefix_lens.shape[0]) != bs:
        raise ValueError(
            f"prefix_lens batch size mismatch: expected {bs}, got {int(prefix_lens.shape[0])}."
        )

    device = proposed_token_ids.device
    current_token_ids = current_token_ids.to(device=device, dtype=torch.long)
    prefix_lens = prefix_lens.to(device=device, dtype=torch.int64)
    proposed_token_ids = proposed_token_ids.to(dtype=torch.long)

    candidates = torch.cat(
        [current_token_ids.unsqueeze(1), proposed_token_ids],
        dim=1,
    )
    draft_token_num = int(candidates.shape[1])
    offsets = torch.arange(draft_token_num, device=device, dtype=torch.int64)
    positions = prefix_lens.unsqueeze(1) + offsets.unsqueeze(0)

    return LinearDraftBlock(
        draft_token=candidates.reshape(-1),
        positions=positions.reshape(-1),
        draft_token_num=draft_token_num,
    )


def build_linear_verify_input(
    block: LinearDraftBlock,
    *,
    requires_target_hidden: bool = True,
) -> DFlashVerifyInput:
    """Create the existing target-verify payload for a linear draft block.

    DFlashVerifyInput is intentionally reused here because attention backends and
    CUDA graph capture already understand its linear verify semantics. This helper
    prevents new dLLM draft code from depending on DFlash's draft-model internals.
    """

    return DFlashVerifyInput(
        draft_token=block.draft_token,
        positions=block.positions,
        draft_token_num=block.draft_token_num,
        requires_target_hidden=requires_target_hidden,
    )
