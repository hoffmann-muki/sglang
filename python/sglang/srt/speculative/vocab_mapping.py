# SPDX-License-Identifier: Apache-2.0
"""Vocabulary mapping for Token-Level Intersection (TLI) speculative decoding."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_KNOWN_SPACE_PREFIXES = ("\u0120", "\u2581")


def _detect_space_sign(tokenizer) -> Optional[str]:
    """Detect a tokenizer's space-prefix character by encoding a literal space."""
    try:
        space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
        if space_ids:
            token = tokenizer.convert_ids_to_tokens(space_ids)[0]
            if token and token[0] in _KNOWN_SPACE_PREFIXES:
                return token[0]
    except Exception:
        pass
    return None


def _normalize_token(token: str, space_sign: Optional[str] = None) -> str:
    """Normalize a BPE token by replacing a tokenizer-specific space prefix."""
    if space_sign is not None:
        if token.startswith(space_sign):
            return " " + token[len(space_sign) :]
        return token
    for prefix in _KNOWN_SPACE_PREFIXES:
        if token.startswith(prefix):
            return " " + token[len(prefix) :]
    return token


class VocabMapping:
    """Map token IDs between target and draft vocabularies via intersection."""

    def __init__(
        self,
        target_tokenizer,
        draft_tokenizer,
        target_vocab_size: int,
        draft_vocab_size: int,
        device: torch.device,
    ):
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device

        self.target_unk_token_id: int = (
            target_tokenizer.unk_token_id
            if target_tokenizer.unk_token_id is not None
            else target_tokenizer.eos_token_id
        )
        self.draft_unk_token_id: int = (
            draft_tokenizer.unk_token_id
            if draft_tokenizer.unk_token_id is not None
            else draft_tokenizer.eos_token_id
        )
        if self.target_unk_token_id is None or self.draft_unk_token_id is None:
            raise ValueError(
                "Target or draft tokenizer has neither unk_token_id nor eos_token_id."
            )
        if target_tokenizer.unk_token_id is None:
            logger.warning(
                "Target tokenizer has no unk_token_id; using eos_token_id=%d as fallback.",
                self.target_unk_token_id,
            )
        if draft_tokenizer.unk_token_id is None:
            logger.warning(
                "Draft tokenizer has no unk_token_id; using eos_token_id=%d as fallback.",
                self.draft_unk_token_id,
            )

        target_space_sign = _detect_space_sign(target_tokenizer)
        draft_space_sign = _detect_space_sign(draft_tokenizer)

        target_vocab = target_tokenizer.get_vocab()
        draft_vocab = draft_tokenizer.get_vocab()

        target_normalized: dict[str, int] = {}
        for token, tid in target_vocab.items():
            norm = _normalize_token(token, target_space_sign)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized: dict[str, int] = {}
        for token, tid in draft_vocab.items():
            norm = _normalize_token(token, draft_space_sign)
            if norm not in draft_normalized:
                draft_normalized[norm] = tid

        common_tokens = set(target_normalized.keys()) & set(draft_normalized.keys())

        draft_to_target = torch.full((draft_vocab_size,), -1, dtype=torch.long)
        target_to_draft = torch.full((target_vocab_size,), -1, dtype=torch.long)
        intersection_mask_draft = torch.zeros(draft_vocab_size, dtype=torch.bool)

        for token in common_tokens:
            target_id = target_normalized[token]
            draft_id = draft_normalized[token]
            if target_id < target_vocab_size and draft_id < draft_vocab_size:
                draft_to_target[draft_id] = target_id
                target_to_draft[target_id] = draft_id
                intersection_mask_draft[draft_id] = True

        self.draft_to_target_ids = draft_to_target.to(device)
        self.target_to_draft_ids = target_to_draft.to(device)
        self.intersection_mask_draft = intersection_mask_draft.to(device)
        self.intersection_size = int(intersection_mask_draft.sum().item())
        self._target_unk_tensor = torch.tensor(
            self.target_unk_token_id, dtype=torch.long, device=device
        )
        self._draft_unk_tensor = torch.tensor(
            self.draft_unk_token_id, dtype=torch.long, device=device
        )
        self.intersection_draft_ids = intersection_mask_draft.nonzero(as_tuple=True)[0].to(
            device
        )

        logger.info(
            "VocabMapping initialized: target_vocab=%d, draft_vocab=%d, intersection=%d",
            target_vocab_size,
            draft_vocab_size,
            self.intersection_size,
        )

    def map_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        draft_ids = self.target_to_draft_ids[target_ids.to(torch.long)]
        draft_ids = torch.where(draft_ids >= 0, draft_ids, self._draft_unk_tensor)
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        target_ids = self.draft_to_target_ids[draft_ids.to(torch.long)]
        target_ids = torch.where(target_ids >= 0, target_ids, self._target_unk_tensor)
        return target_ids.to(draft_ids.dtype)

    def constrain_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~self.intersection_mask_draft, float("-inf"))