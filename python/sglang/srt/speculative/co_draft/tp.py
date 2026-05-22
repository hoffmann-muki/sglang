"""Tensor-parallel planning for colocated draft executors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalDraftTpPlan:
    """Describe which target TP ranks participate in a local draft executor."""

    name: str
    target_tp_size: int
    draft_tp_size: int
    root_rank: int = 0

    def __post_init__(self) -> None:
        if self.target_tp_size <= 0:
            raise ValueError("target_tp_size must be positive.")
        if self.draft_tp_size <= 0:
            raise ValueError(f"{self.name} draft_tp_size must be positive.")
        if self.draft_tp_size > self.target_tp_size:
            raise ValueError(
                f"{self.name} draft_tp_size={self.draft_tp_size} cannot exceed "
                f"target_tp_size={self.target_tp_size}."
            )
        if self.target_tp_size % self.draft_tp_size != 0:
            raise ValueError(
                f"{self.name} draft_tp_size={self.draft_tp_size} must divide "
                f"target_tp_size={self.target_tp_size}."
            )
        if self.root_rank < 0 or self.root_rank >= self.target_tp_size:
            raise ValueError(
                f"{self.name} root_rank={self.root_rank} is outside target TP size "
                f"{self.target_tp_size}."
            )
        if self.root_rank + self.draft_tp_size > self.target_tp_size:
            raise ValueError(
                f"{self.name} rank range [{self.root_rank}, "
                f"{self.root_rank + self.draft_tp_size}) exceeds target TP size "
                f"{self.target_tp_size}."
            )

    def owns_rank(self, tp_rank: int) -> bool:
        return self.root_rank <= tp_rank < self.root_rank + self.draft_tp_size

    def local_rank(self, tp_rank: int) -> int:
        if not self.owns_rank(tp_rank):
            raise ValueError(f"TP rank {tp_rank} does not participate in {self.name}.")
        return tp_rank - self.root_rank

    @property
    def is_asymmetric(self) -> bool:
        return self.draft_tp_size != self.target_tp_size
