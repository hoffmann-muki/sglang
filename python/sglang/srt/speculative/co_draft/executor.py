"""Shared executor contracts for colocated AR+dLLM drafting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch

from sglang.srt.speculative.co_draft.tp import LocalDraftTpPlan

DraftExecutorKind = Literal["ar", "dllm"]
CoDraftStrategy = Literal[
    "ar_only",
    "dllm_only",
    "winner_take_all",
    "agreement_gate",
    "mixed_tree",
    "ar_qualifier",
]


@dataclass(slots=True)
class DraftTiming:
    queue_scheduling_time: float = 0.0
    proposal_time: float = 0.0
    other_time: float = 0.0


@dataclass(slots=True)
class DraftProposal:
    """Neutral proposal format before conversion to target verification input."""

    kind: DraftExecutorKind
    request_ids: list[str]
    draft_token_ids: torch.Tensor
    parent_list: Optional[torch.Tensor] = None
    top_scores_index: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DraftExecutorResult:
    proposal: Optional[DraftProposal]
    timing: DraftTiming = field(default_factory=DraftTiming)


class LocalDraftExecutor(ABC):
    """A colocated draft model that can maintain request state independently."""

    kind: DraftExecutorKind

    def __init__(self, *, tp_plan: LocalDraftTpPlan):
        self.tp_plan = tp_plan

    @abstractmethod
    def extend(self, *args, **kwargs) -> DraftExecutorResult:
        pass

    @abstractmethod
    def propose(self, *args, **kwargs) -> DraftExecutorResult:
        pass

    @abstractmethod
    def extend_after_accept(self, *args, **kwargs) -> DraftExecutorResult:
        pass

    @abstractmethod
    def release(self, request_ids: list[str]) -> None:
        pass


class ArTliDraftExecutor(LocalDraftExecutor):
    """Descriptor for the existing colocated TLI AR draft path.

    The executable AR path is still inherited from ``AsymmetricTLIWorker``.
    Strategy adapters should wrap those worker methods explicitly instead of
    assuming this descriptor can run draft kernels by itself.
    """

    kind: DraftExecutorKind = "ar"

    def __init__(self, *, tp_plan: LocalDraftTpPlan, owner):
        super().__init__(tp_plan=tp_plan)
        self.owner = owner

    def extend(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError(
            "AR co-draft execution is provided by the owning TLI worker."
        )

    def propose(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError(
            "AR co-draft execution is provided by the owning TLI worker."
        )

    def extend_after_accept(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError(
            "AR co-draft execution is provided by the owning TLI worker."
        )

    def release(self, request_ids: list[str]) -> None:
        return None


class DllmDraftExecutor(LocalDraftExecutor):
    """Descriptor for a colocated dLLM draft model with an independent TP shape."""

    kind: DraftExecutorKind = "dllm"

    def __init__(
        self,
        *,
        tp_plan: LocalDraftTpPlan,
        model_path: str,
        tokenizer_path: str,
        algorithm: str,
        algorithm_config: Optional[str],
    ):
        super().__init__(tp_plan=tp_plan)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config

    def extend(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError("dLLM draft extend adapter is not implemented yet.")

    def propose(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError("dLLM draft proposal adapter is not implemented yet.")

    def extend_after_accept(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError(
            "dLLM draft extend-after-accept adapter is not implemented yet."
        )

    def release(self, request_ids: list[str]) -> None:
        return None
