"""Shared executor contracts for colocated AR+dLLM drafting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch

from sglang.srt.speculative.co_draft.tp import LocalDraftTpPlan
from sglang.srt.speculative.linear_verify import LinearDraftBlock

if TYPE_CHECKING:
    from sglang.srt.speculative.co_draft.dllm_linear_adapter import (
        IndependentDllmProposalRunner,
    )

DraftExecutorKind = Literal["ar", "dllm"]
DllmDraftBackend = Literal["sglang_dllm", "fast_dllm_v2", "dflash"]
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


@dataclass(frozen=True, slots=True)
class LinearVerificationPlan:
    """Describe how a draft proposal should be verified by the AR target.

    DFlash already implements the target-side linear verification primitive. An
    independent dLLM draft such as Fast_dLLM_v2 can reuse that verify shape after
    it has produced a flat block of token ids, without reusing DFlash's
    target-hidden-conditioned draft model.
    """

    proposed_token_num: int
    uses_dflash_verify_contract: bool = True

    @property
    def draft_token_num(self) -> int:
        return self.proposed_token_num + 1


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
    """Descriptor for a colocated dLLM draft model with an independent TP shape.

    The backend names intentionally distinguish model families from verification:
    ``dflash`` is SGLang's target-hidden-conditioned DFlash draft model, while
    ``fast_dllm_v2`` is an independent block-diffusion LM whose token proposals
    should be verified through the shared linear target-verify path.
    """

    kind: DraftExecutorKind = "dllm"

    def __init__(
        self,
        *,
        tp_plan: LocalDraftTpPlan,
        model_path: str,
        tokenizer_path: str,
        algorithm: str,
        algorithm_config: Optional[str],
        backend: DllmDraftBackend,
        verification_plan: LinearVerificationPlan,
    ):
        super().__init__(tp_plan=tp_plan)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.backend = backend
        self.verification_plan = verification_plan
        from sglang.srt.speculative.co_draft.dllm_linear_adapter import (
            create_dllm_linear_adapter,
        )

        self.linear_adapter = create_dllm_linear_adapter(self)

    @property
    def is_independent_model(self) -> bool:
        return self.backend in ("sglang_dllm", "fast_dllm_v2")

    def build_linear_block(self, tokens) -> LinearDraftBlock:
        return self.linear_adapter.build_linear_block(tokens)

    def attach_runner(self, runner: "IndependentDllmProposalRunner") -> None:
        """Attach the runtime object that executes the independent dLLM draft."""

        self.linear_adapter.runner = runner

    def extend(self, *args, **kwargs) -> DraftExecutorResult:
        raise NotImplementedError(
            f"{self.backend} dLLM draft extend adapter is not implemented yet."
        )

    def propose(self, *args, **kwargs) -> DraftExecutorResult:
        tokens = self.linear_adapter.propose(*args, **kwargs)
        return DraftExecutorResult(
            proposal=self.linear_adapter.build_draft_proposal(tokens)
        )

    def extend_after_accept(self, *args, **kwargs) -> DraftExecutorResult:
        self.linear_adapter.extend_after_accept(*args, **kwargs)
        return DraftExecutorResult(proposal=None)

    def release(self, request_ids: list[str]) -> None:
        self.linear_adapter.release(request_ids)
