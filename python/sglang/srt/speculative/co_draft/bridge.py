"""Contracts for communication between colocated draft executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from sglang.srt.speculative.co_draft.executor import DraftProposal


@dataclass(slots=True)
class DraftAnchorBatch:
    """AR-produced anchors that can condition a dLLM completion pass.

    Token IDs are expressed in the target vocabulary unless a strategy adapter
    explicitly provides translated IDs in ``draft_token_ids``. This keeps the
    bridge vocabulary-neutral and makes tokenizer translation an adapter concern.
    """

    request_ids: list[str]
    target_token_ids: torch.Tensor
    draft_token_ids: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DraftCompletionRequest:
    """Request emitted by one draft executor for another executor to complete."""

    request_ids: list[str]
    anchors: DraftAnchorBatch
    max_completion_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class CoDraftBridge(ABC):
    """Bidirectional semantic bridge between colocated draft executors."""

    @abstractmethod
    def ar_to_dllm_completion_request(
        self,
        ar_proposal: DraftProposal,
        *,
        max_completion_tokens: int,
    ) -> DraftCompletionRequest:
        """Convert AR draft anchors into a dLLM completion request."""

    @abstractmethod
    def dllm_to_ar_anchor_batch(
        self,
        dllm_proposal: DraftProposal,
    ) -> DraftAnchorBatch:
        """Convert dLLM output into AR-readable anchors."""


class UnimplementedCoDraftBridge(CoDraftBridge):
    """Fail-fast bridge used until a concrete co-draft strategy is wired."""

    def ar_to_dllm_completion_request(
        self,
        ar_proposal: DraftProposal,
        *,
        max_completion_tokens: int,
    ) -> DraftCompletionRequest:
        raise NotImplementedError(
            "AR-to-dLLM co-draft bridge is not implemented yet."
        )

    def dllm_to_ar_anchor_batch(
        self,
        dllm_proposal: DraftProposal,
    ) -> DraftAnchorBatch:
        raise NotImplementedError(
            "dLLM-to-AR co-draft bridge is not implemented yet."
        )
