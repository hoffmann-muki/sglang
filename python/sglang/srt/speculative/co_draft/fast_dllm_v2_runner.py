"""Fast_dLLM_v2 proposal runner for colocated co-draft experiments."""

from __future__ import annotations

import importlib.util
import json
import logging
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

logger = logging.getLogger(__name__)


def _compute_default_rope_parameters(
    config: Any = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
) -> tuple[torch.Tensor, float]:
    """Compute standard RoPE parameters for the Fast_dLLM_v2 compatibility shim."""

    del seq_len
    if config is None:
        raise ValueError("Fast_dLLM_v2 default RoPE shim requires a config.")

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_theta = rope_parameters.get(
        "rope_theta", getattr(config, "rope_theta", 10000.0)
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_attention_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is None or num_attention_heads in (None, 0):
            raise ValueError(
                "Fast_dLLM_v2 default RoPE shim requires hidden_size and "
                "num_attention_heads when head_dim is not available."
            )
        head_dim = hidden_size // num_attention_heads

    partial_rotary_factor = rope_parameters.get(
        "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
    )
    rotary_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device)
            / rotary_dim
        )
    )
    return inv_freq, 1.0


def _ensure_transformers_default_rope_support() -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    # Fast_dLLM_v2 checkpoints use standard Qwen2.5-style RoPE when no
    # scaling override is present. Some Transformers releases expose that path
    # as a registered "default" rope_type and others compute it inline, so we
    # register a local compatibility alias when needed.
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    logger.warning(
        "Registered a local Transformers RoPE compatibility shim for the "
        "'default' rope_type."
    )


def _ensure_transformers_tied_weights_support() -> None:
    """Normalize tied-weight metadata for the Fast_dLLM_v2 remote model code.

    The Fast_dLLM_v2 checkpoint still declares tied weights using the legacy
    list-style contract, while newer Transformers releases expect an explicit
    mapping. We patch the base helper so the remote model can declare its tied
    embedding/LM-head pair in the format the loader expects.
    """

    from transformers.modeling_utils import PreTrainedModel

    original = getattr(
        PreTrainedModel.get_expanded_tied_weights_keys,
        "_sglang_fast_dllm_v2_compat",
        None,
    )
    if original is not None:
        return

    original_method = PreTrainedModel.get_expanded_tied_weights_keys

    def _patched_get_expanded_tied_weights_keys(self, all_submodels=False):
        tied_weights_keys = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied_weights_keys, list):
            model_type = getattr(getattr(self, "config", None), "model_type", None)
            if model_type == "Fast_dLLM_Qwen":
                self._tied_weights_keys = {
                    "lm_head.weight": "model.embed_tokens.weight"
                }
        return original_method(self, all_submodels=all_submodels)

    _patched_get_expanded_tied_weights_keys._sglang_fast_dllm_v2_compat = True
    PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded_tied_weights_keys
    logger.warning(
        "Registered a local Transformers tied-weights compatibility shim for "
        "Fast_dLLM_v2."
    )


def _ensure_transformers_dynamic_cache_indexing_support() -> None:
    """Provide legacy layer indexing for Fast_dLLM_v2 cache access."""

    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache, "__getitem__"):
        return

    def _legacy_getitem(self, layer_idx: int):
        key_cache = getattr(self, "key_cache", None)
        value_cache = getattr(self, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            return key_cache[layer_idx], value_cache[layer_idx]

        layers = getattr(self, "layers", None)
        if layers is not None:
            layer = layers[layer_idx]
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                return layer.keys, layer.values
            if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
                return layer.key_cache, layer.value_cache

        raise TypeError(
            "DynamicCache does not expose legacy layer key/value tensors."
        )

    DynamicCache.__getitem__ = _legacy_getitem
    logger.warning(
        "Registered a local Transformers DynamicCache indexing compatibility "
        "shim for Fast_dLLM_v2."
    )


@dataclass(frozen=True, slots=True)
class FastDllmV2RunnerConfig:
    """Runtime configuration for an independent Fast_dLLM_v2 draft runner."""

    model_path: str
    tokenizer_path: str
    proposed_token_num: int
    block_size: int = 32
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
            block_size=int(raw_config.get("block_size", 32)),
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
                "block_size": config.block_size,
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

        _ensure_transformers_default_rope_support()
        _ensure_transformers_tied_weights_support()
        _ensure_transformers_dynamic_cache_indexing_support()

        has_accelerate = self._has_accelerate()
        device_map = config.device_map if has_accelerate else None
        if not has_accelerate and config.device_map not in (None, "none"):
            logger.warning(
                "Fast_dLLM_v2 is falling back to a single-device load because "
                "accelerate is unavailable in this environment."
            )

        self.model = self._load_model(config, device_map=device_map)
        self.model = self._move_model_to_primary_device(self.model)
        self.tokenizer = self._load_tokenizer(config)

    def _has_accelerate(self) -> bool:
        return importlib.util.find_spec("accelerate") is not None

    def _load_model(
        self,
        config: FastDllmV2RunnerConfig,
        *,
        device_map: Optional[str],
    ) -> Any:
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "torch_dtype": config.torch_dtype,
            "trust_remote_code": config.trust_remote_code,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        return AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)

    def _load_tokenizer(self, config: FastDllmV2RunnerConfig) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=config.trust_remote_code,
        )

    def _move_model_to_primary_device(self, model):
        current_device = getattr(model, "device", torch.device("cpu"))
        if torch.cuda.is_available() and current_device.type != "cuda":
            try:
                return model.to(torch.device("cuda"))
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Fast_dLLM_v2 could not move the model to CUDA after CPU load: %s",
                    exc,
                )
        return model

    def _generate_one(
        self,
        config: FastDllmV2RunnerConfig,
        input_ids: list[int],
    ) -> torch.Tensor:
        assert self.model is not None
        assert self.tokenizer is not None
        device = self.model.device
        prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
        internal_max_new_tokens = _fast_dllm_v2_internal_generation_budget(
            prompt.shape[1],
            config.proposed_token_num,
            config.block_size,
        )
        output = self.model.generate(
            prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=internal_max_new_tokens,
            block_size=config.block_size,
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


def _fast_dllm_v2_internal_generation_budget(
    prompt_len: int,
    proposed_token_num: int,
    block_size: int,
) -> int:
    """Return the block-aligned budget Fast_dLLM_v2 needs for a short proposal."""

    if block_size <= 0:
        raise ValueError("Fast_dLLM_v2 block_size must be positive.")
    if proposed_token_num <= 0:
        raise ValueError("Fast_dLLM_v2 proposed_token_num must be positive.")

    remainder = prompt_len % block_size
    first_block_tokens = block_size + 1 if remainder == 0 else block_size - remainder + 1
    blocks = 1
    if proposed_token_num > first_block_tokens:
        remaining = proposed_token_num - first_block_tokens
        blocks += (remaining + block_size - 1) // block_size
    return blocks * block_size
