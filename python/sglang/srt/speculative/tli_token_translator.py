# SPDX-License-Identifier: Apache-2.0
"""Cross-node token translation for TLI speculative decoding."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from sglang.srt.speculative.vocab_mapping import VocabMapping


class TLITokenTranslator(VocabMapping):
    """Bidirectional target/draft token translator for disaggregated TLI."""

    def translate_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        return super().map_target_to_draft_ids(target_ids)

    def translate_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return super().map_draft_to_target_ids(draft_ids)

    def translate_target_to_draft_list(self, target_ids: Sequence[int]) -> list[int]:
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long, device=self.device)
        return self.translate_target_to_draft_ids(target_ids_t).tolist()

    def translate_draft_to_target_list(self, draft_ids: Sequence[int]) -> list[int]:
        draft_ids_t = torch.as_tensor(draft_ids, dtype=torch.long, device=self.device)
        return self.translate_draft_to_target_ids(draft_ids_t).tolist()

    # Keep the VocabMapping-style names because the colocated TLI worker calls them.
    def map_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        return self.translate_target_to_draft_ids(target_ids)

    # Keep the VocabMapping-style names because the colocated TLI worker calls them.
    def map_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return self.translate_draft_to_target_ids(draft_ids)

    def to_state(self, to_cpu: bool = True) -> dict[str, Any]:
        """Serialize translator state for cross-process/cross-node initialization."""

        def _maybe_cpu(t: torch.Tensor) -> torch.Tensor:
            detached = t.detach().clone()
            return detached.cpu() if to_cpu else detached

        return {
            "target_vocab_size": self.target_vocab_size,
            "draft_vocab_size": self.draft_vocab_size,
            "target_unk_token_id": self.target_unk_token_id,
            "draft_unk_token_id": self.draft_unk_token_id,
            "draft_to_target_ids": _maybe_cpu(self.draft_to_target_ids),
            "target_to_draft_ids": _maybe_cpu(self.target_to_draft_ids),
            "intersection_mask_draft": _maybe_cpu(self.intersection_mask_draft),
        }

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], device: torch.device
    ) -> TLITokenTranslator:
        """Recreate a translator from serialized state."""
        required_keys = (
            "target_vocab_size",
            "draft_vocab_size",
            "target_unk_token_id",
            "draft_unk_token_id",
            "draft_to_target_ids",
            "target_to_draft_ids",
            "intersection_mask_draft",
        )
        for key in required_keys:
            if key not in state:
                raise ValueError(f"Missing key in translator state: {key}")

        obj = cls.__new__(cls)
        obj.target_vocab_size = int(state["target_vocab_size"])
        obj.draft_vocab_size = int(state["draft_vocab_size"])
        obj.device = device
        obj.target_unk_token_id = int(state["target_unk_token_id"])
        obj.draft_unk_token_id = int(state["draft_unk_token_id"])
        obj.draft_to_target_ids = torch.as_tensor(
            state["draft_to_target_ids"], dtype=torch.long, device=device
        )
        obj.target_to_draft_ids = torch.as_tensor(
            state["target_to_draft_ids"], dtype=torch.long, device=device
        )
        obj.intersection_mask_draft = torch.as_tensor(
            state["intersection_mask_draft"], dtype=torch.bool, device=device
        )
        if obj.draft_to_target_ids.shape != (obj.draft_vocab_size,):
            raise ValueError(
                "draft_to_target_ids has unexpected shape: "
                f"{tuple(obj.draft_to_target_ids.shape)}"
            )
        if obj.target_to_draft_ids.shape != (obj.target_vocab_size,):
            raise ValueError(
                "target_to_draft_ids has unexpected shape: "
                f"{tuple(obj.target_to_draft_ids.shape)}"
            )
        if obj.intersection_mask_draft.shape != (obj.draft_vocab_size,):
            raise ValueError(
                "intersection_mask_draft has unexpected shape: "
                f"{tuple(obj.intersection_mask_draft.shape)}"
            )
        obj.intersection_size = int(obj.intersection_mask_draft.sum().item())
        obj.intersection_draft_ids = obj.intersection_mask_draft.nonzero(as_tuple=True)[
            0
        ].to(device)
        obj._target_unk_tensor = torch.tensor(
            obj.target_unk_token_id, dtype=torch.long, device=device
        )
        obj._draft_unk_tensor = torch.tensor(
            obj.draft_unk_token_id, dtype=torch.long, device=device
        )
        return obj
