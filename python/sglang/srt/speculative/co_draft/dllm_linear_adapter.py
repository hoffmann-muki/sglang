"""Adapters for independent dLLM drafts that verify through a linear AR target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Optional, Protocol

import torch

from sglang.srt.speculative.co_draft.executor import DraftProposal
from sglang.srt.speculative.linear_verify import (
    LinearDraftBlock,
    build_linear_draft_block,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.co_draft.executor import DllmDraftExecutor


@dataclass(slots=True)
class IndependentDllmDraftTokens:
    """Token proposals from an independent dLLM draft model.

    ``proposed_token_ids`` contains only newly proposed tokens. The adapter
    prepends ``current_token_ids`` before target verification because SGLang's
    linear verify path follows the DFlash convention.
    """

    request_ids: list[str]
    current_token_ids: torch.Tensor
    proposed_token_ids: torch.Tensor
    prefix_lens: torch.Tensor
    scores: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndependentDllmDraftRequest:
    """Runtime request handed to an independent dLLM proposal runner.

    ``input_ids`` may be ragged because the draft model owns its own cache and
    can decide whether a request needs prefill, decode, or block-cache refresh.
    ``current_token_ids`` and ``prefix_lens`` define the target-verification
    anchor for the next draft block.
    """

    request_ids: list[str]
    input_ids: list[list[int]]
    current_token_ids: torch.Tensor
    prefix_lens: torch.Tensor
    proposed_token_num: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndependentDllmAcceptedTokens:
    """Accepted dLLM draft tokens that should advance draft-side state."""

    request_ids: list[str]
    accepted_token_ids: list[list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)


class IndependentDllmProposalRunner(Protocol):
    """Executes an independent dLLM and returns token blocks for verification."""

    def propose(
        self, request: IndependentDllmDraftRequest
    ) -> IndependentDllmDraftTokens:
        ...

    def release(self, request_ids: list[str]) -> None:
        ...

    def extend_after_accept(self, accepted: IndependentDllmAcceptedTokens) -> None:
        ...


class IndependentDllmLinearAdapter:
    """Converts independent dLLM token proposals into target-verifiable blocks."""

    def __init__(
        self,
        executor: "DllmDraftExecutor",
        runner: Optional[IndependentDllmProposalRunner] = None,
    ):
        self.executor = executor
        self.runner = runner

    def propose(
        self, request: IndependentDllmDraftRequest
    ) -> IndependentDllmDraftTokens:
        if self.runner is None:
            raise NotImplementedError(self._missing_runner_message())

        self._validate_request(request)
        tokens = self.runner.propose(request)
        self._validate(tokens, request=request)
        return tokens

    def release(self, request_ids: list[str]) -> None:
        if self.runner is not None:
            self.runner.release(request_ids)

    def extend_after_accept(self, accepted: IndependentDllmAcceptedTokens) -> None:
        self._validate_accepted(accepted)
        if self.runner is not None:
            self.runner.extend_after_accept(accepted)

    def build_linear_block(
        self, tokens: IndependentDllmDraftTokens
    ) -> LinearDraftBlock:
        self._validate(tokens)
        return build_linear_draft_block(
            current_token_ids=tokens.current_token_ids,
            proposed_token_ids=tokens.proposed_token_ids,
            prefix_lens=tokens.prefix_lens,
        )

    def build_draft_proposal(self, tokens: IndependentDllmDraftTokens) -> DraftProposal:
        block = self.build_linear_block(tokens)
        metadata = {
            "backend": self.executor.backend,
            "verification": "linear",
            **tokens.metadata,
        }
        return DraftProposal(
            kind="dllm",
            request_ids=tokens.request_ids,
            draft_token_ids=block.draft_token,
            scores=tokens.scores,
            metadata=metadata,
        )

    def _missing_runner_message(self) -> str:
        return (
            f"{self.executor.backend} proposal runner is not attached. Attach an "
            "IndependentDllmProposalRunner that owns draft-model execution and cache state."
        )

    def _validate_request(self, request: IndependentDllmDraftRequest) -> None:
        bs = len(request.request_ids)
        if bs <= 0:
            raise ValueError("Independent dLLM draft request batch must be non-empty.")
        if len(request.input_ids) != bs:
            raise ValueError(
                "input_ids batch size mismatch: "
                f"expected {bs}, got {len(request.input_ids)}."
            )
        if request.current_token_ids.dim() != 1:
            raise ValueError("current_token_ids must be a 1D tensor.")
        if int(request.current_token_ids.shape[0]) != bs:
            raise ValueError(
                "current_token_ids batch size mismatch: "
                f"expected {bs}, got {int(request.current_token_ids.shape[0])}."
            )
        if request.prefix_lens.dim() != 1:
            raise ValueError("prefix_lens must be a 1D tensor.")
        if int(request.prefix_lens.shape[0]) != bs:
            raise ValueError(
                "prefix_lens batch size mismatch: "
                f"expected {bs}, got {int(request.prefix_lens.shape[0])}."
            )
        if request.proposed_token_num <= 0:
            raise ValueError(
                "Independent dLLM draft request must ask for at least one proposed token."
            )

    def _validate_accepted(self, accepted: IndependentDllmAcceptedTokens) -> None:
        bs = len(accepted.request_ids)
        if bs <= 0:
            raise ValueError("Independent dLLM accepted-token batch must be non-empty.")
        if len(accepted.accepted_token_ids) != bs:
            raise ValueError(
                "accepted_token_ids batch size mismatch: "
                f"expected {bs}, got {len(accepted.accepted_token_ids)}."
            )
        for request_id, token_ids in zip(
            accepted.request_ids, accepted.accepted_token_ids
        ):
            if not request_id:
                raise ValueError("Accepted-token request id must be non-empty.")
            if any(not isinstance(token_id, int) for token_id in token_ids):
                raise ValueError("Accepted token ids must be Python integers.")

    def _validate(
        self,
        tokens: IndependentDllmDraftTokens,
        *,
        request: Optional[IndependentDllmDraftRequest] = None,
    ) -> None:
        bs = len(tokens.request_ids)
        if bs <= 0:
            raise ValueError("Independent dLLM draft batch must be non-empty.")
        if tokens.current_token_ids.dim() != 1:
            raise ValueError("current_token_ids must be a 1D tensor.")
        if int(tokens.current_token_ids.shape[0]) != bs:
            raise ValueError(
                "current_token_ids batch size mismatch: "
                f"expected {bs}, got {int(tokens.current_token_ids.shape[0])}."
            )
        if tokens.proposed_token_ids.dim() != 2:
            raise ValueError("proposed_token_ids must be a 2D tensor.")
        if int(tokens.proposed_token_ids.shape[0]) != bs:
            raise ValueError(
                "proposed_token_ids batch size mismatch: "
                f"expected {bs}, got {int(tokens.proposed_token_ids.shape[0])}."
            )
        if tokens.prefix_lens.dim() != 1:
            raise ValueError("prefix_lens must be a 1D tensor.")
        if int(tokens.prefix_lens.shape[0]) != bs:
            raise ValueError(
                "prefix_lens batch size mismatch: "
                f"expected {bs}, got {int(tokens.prefix_lens.shape[0])}."
            )
        if request is None:
            return
        if tokens.request_ids != request.request_ids:
            raise ValueError("Runner returned request ids in a different order.")
        proposed_token_num = int(tokens.proposed_token_ids.shape[1])
        if proposed_token_num != request.proposed_token_num:
            raise ValueError(
                "Runner returned the wrong number of proposed tokens: "
                f"expected {request.proposed_token_num}, got {proposed_token_num}."
            )


class FastDllmV2LinearAdapter(IndependentDllmLinearAdapter):
    """Adapter boundary for Fast_dLLM_v2 block proposals.

    The adapter validates runner inputs/outputs and converts Fast_dLLM_v2 token
    blocks into the shared linear target-verification shape.
    """

    def _missing_runner_message(self) -> str:
        return (
            "Fast_dLLM_v2 proposal runner is not attached. The adapter expects "
            "a runner that maintains Fast_dLLM_v2's block cache and returns "
            "next-token blocks."
        )


def create_dllm_linear_adapter(
    executor: "DllmDraftExecutor",
) -> IndependentDllmLinearAdapter:
    if executor.backend == "fast_dllm_v2":
        return FastDllmV2LinearAdapter(executor)
    return IndependentDllmLinearAdapter(executor)
