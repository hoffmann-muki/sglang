"""Fast_dLLM_v2 proposal runner for colocated co-draft experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

import torch
import yaml

from sglang.srt.speculative.co_draft.dllm_linear_adapter import (
    IndependentDllmAcceptedTokens,
    IndependentDllmDraftRequest,
    IndependentDllmDraftTokens,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.co_draft.executor import DllmDraftExecutor


@dataclass(frozen=True, slots=True)
class FastDllmV2RunnerConfig:
    """Runtime configuration for an independent Fast_dLLM_v2 draft runner."""

    model_path: str
    tokenizer_path: str
    proposed_token_num: int
    small_block_size: int = 8
    threshold: float = 0.9
    torch_dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = True
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_executor(
        cls, executor: "DllmDraftExecutor"
    ) -> "FastDllmV2RunnerConfig":
        raw_config = _load_algorithm_config(executor.algorithm_config)
        generation_kwargs = dict(raw_config.get("generation_kwargs", {}))
        return cls(
            model_path=executor.model_path,
            tokenizer_path=executor.tokenizer_path,
            proposed_token_num=executor.verification_plan.proposed_token_num,
            small_block_size=int(raw_config.get("small_block_size", 8)),
            threshold=float(raw_config.get("threshold", 0.9)),
            torch_dtype=str(raw_config.get("torch_dtype", "auto")),
            device_map=str(raw_config.get("device_map", "auto")),
            trust_remote_code=bool(raw_config.get("trust_remote_code", True)),
            generation_kwargs=generation_kwargs,
        )


@dataclass(slots=True)
class FastDllmV2RequestState:
    """Per-request state owned by the independent dLLM draft runner."""

    input_ids: list[int]
    accepted_token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class FastDllmV2Runtime(Protocol):
    """Backend that executes Fast_dLLM_v2 and may own model-specific caches."""

    def propose(
        self,
        config: FastDllmV2RunnerConfig,
        request: IndependentDllmDraftRequest,
        states: dict[str, FastDllmV2RequestState],
    ) -> IndependentDllmDraftTokens:
        ...

    def extend_after_accept(
        self,
        config: FastDllmV2RunnerConfig,
        accepted: IndependentDllmAcceptedTokens,
        states: dict[str, FastDllmV2RequestState],
    ) -> None:
        ...

    def release(
        self,
        config: FastDllmV2RunnerConfig,
        request_ids: list[str],
        states: dict[str, FastDllmV2RequestState],
    ) -> None:
        ...


class TransformersFastDllmV2Runtime:
    """Lazy Transformers runtime for Fast_dLLM_v2 proposal generation.

    This path intentionally uses the model's own ``generate`` implementation
    with ``trust_remote_code=True``. The surrounding runner owns request state;
    model-specific hierarchical/block-cache behavior remains inside the
    Fast_dLLM_v2 model implementation.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def propose(
        self,
        config: FastDllmV2RunnerConfig,
        request: IndependentDllmDraftRequest,
        states: dict[str, FastDllmV2RequestState],
    ) -> IndependentDllmDraftTokens:
        self._ensure_loaded(config)
        proposed = []
        for request_id in request.request_ids:
            input_ids = states[request_id].input_ids
            generated = self._generate_one(config, input_ids)
            proposed.append(generated)

        proposed_token_ids = torch.stack(proposed).to(request.current_token_ids.device)
        return IndependentDllmDraftTokens(
            request_ids=list(request.request_ids),
            current_token_ids=request.current_token_ids,
            proposed_token_ids=proposed_token_ids,
            prefix_lens=request.prefix_lens,
            metadata={
                "runner": "fast_dllm_v2",
                "runtime": "transformers",
                "small_block_size": config.small_block_size,
                "threshold": config.threshold,
            },
        )

    def extend_after_accept(
        self,
        config: FastDllmV2RunnerConfig,
        accepted: IndependentDllmAcceptedTokens,
        states: dict[str, FastDllmV2RequestState],
    ) -> None:
        return None

    def release(
        self,
        config: FastDllmV2RunnerConfig,
        request_ids: list[str],
        states: dict[str, FastDllmV2RequestState],
    ) -> None:
        return None

    def _ensure_loaded(self, config: FastDllmV2RunnerConfig) -> None:
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=config.trust_remote_code,
        )

    def _generate_one(
        self,
        config: FastDllmV2RunnerConfig,
        input_ids: list[int],
    ) -> torch.Tensor:
        assert self.model is not None
        assert self.tokenizer is not None
        device = self.model.device
        prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = self.model.generate(
            prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=config.proposed_token_num,
            small_block_size=config.small_block_size,
            threshold=config.threshold,
            **config.generation_kwargs,
        )
        new_tokens = output[
            0, prompt.shape[1] : prompt.shape[1] + config.proposed_token_num
        ]
        if new_tokens.numel() == config.proposed_token_num:
            return new_tokens

        pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                "Fast_dLLM_v2 generated fewer tokens than requested and no EOS/PAD "
                "token is available for padding."
            )
        padding = torch.full(
            (config.proposed_token_num - new_tokens.numel(),),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        return torch.cat([new_tokens, padding], dim=0)


class FastDllmV2ProposalRunner:
    """Stateful proposal runner for a colocated independent Fast_dLLM_v2 draft."""

    def __init__(
        self,
        config: FastDllmV2RunnerConfig,
        runtime: Optional[FastDllmV2Runtime] = None,
    ):
        self.config = config
        self.runtime = runtime or TransformersFastDllmV2Runtime()
        self.states: dict[str, FastDllmV2RequestState] = {}

    @classmethod
    def from_executor(
        cls, executor: "DllmDraftExecutor"
    ) -> "FastDllmV2ProposalRunner":
        return cls(FastDllmV2RunnerConfig.from_executor(executor))

    def propose(
        self, request: IndependentDllmDraftRequest
    ) -> IndependentDllmDraftTokens:
        self._refresh_states(request)
        return self.runtime.propose(self.config, request, self.states)

    def extend_after_accept(self, accepted: IndependentDllmAcceptedTokens) -> None:
        for request_id, accepted_token_ids in zip(
            accepted.request_ids, accepted.accepted_token_ids
        ):
            state = self.states.get(request_id)
            if state is None:
                continue
            state.input_ids.extend(accepted_token_ids)
            state.accepted_token_count += len(accepted_token_ids)
        self.runtime.extend_after_accept(self.config, accepted, self.states)

    def release(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.states.pop(request_id, None)
        self.runtime.release(self.config, request_ids, self.states)

    def _refresh_states(self, request: IndependentDllmDraftRequest) -> None:
        for request_id, input_ids in zip(request.request_ids, request.input_ids):
            self.states[request_id] = FastDllmV2RequestState(
                input_ids=list(input_ids)
            )


def _load_algorithm_config(config_path: Optional[str]) -> dict[str, Any]:
    if config_path is None:
        return {}

    path = Path(config_path)
    with path.open("r") as fin:
        if path.suffix.lower() == ".json":
            raw = json.load(fin)
        else:
            raw = yaml.safe_load(fin)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Fast_dLLM_v2 algorithm config must be a mapping.")
    return raw
