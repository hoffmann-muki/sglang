"""Fast_dLLM_v2 proposal runner for colocated co-draft experiments."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import types
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
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

FAST_DLLM_V2_REFERENCE_TRANSFORMERS_VERSION = "4.53.1"
FAST_DLLM_V2_MASK_ID = 151665
FAST_DLLM_V2_STOP_TOKEN = 151645
FAST_DLLM_V2_PROPOSAL_KWARGS = {
    "do_sample",
    "mask_id",
    "stop_token",
    "temperature",
    "top_p",
    "use_block_cache",
}


class _FastDllmV2ProposalUnsupported(RuntimeError):
    """Raised when SGLang's Fast_dLLM_v2 path cannot honor a request."""


def _compute_default_rope_parameters(
    config: Any = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
) -> tuple[torch.Tensor, float]:
    """Compute the Qwen-style RoPE parameters used by Fast_dLLM_v2."""

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

    current = ROPE_INIT_FUNCTIONS.get("default")
    if getattr(current, "_sglang_fast_dllm_v2_v453_compat", False):
        return

    # Fast_dLLM_v2 was validated against the Transformers 4.53.1 Qwen-style
    # "default" RoPE contract. Newer Transformers releases route "default"
    # through a different rope_parameters contract, which can produce identity
    # rotations for this remote model, so install the reference initializer.
    _compute_default_rope_parameters._sglang_fast_dllm_v2_v453_compat = True
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    logger.info("Registered Fast_dLLM_v2 Transformers 4.53.1 RoPE compatibility.")


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
    logger.info("Registered Fast_dLLM_v2 tied-weights compatibility.")


def _disable_transformers_hub_kernels_for_fast_dllm_v2() -> bool:
    """Disable Hub kernels before importing Fast_dLLM_v2 remote model code."""

    previous = os.environ.get("USE_HUB_KERNELS")
    if previous is None or previous.strip().upper() not in {"0", "OFF", "NO"}:
        os.environ["USE_HUB_KERNELS"] = "0"
        return True
    return False


def _repeat_kv_for_fast_dllm_v2(
    hidden_states: torch.Tensor, n_rep: int
) -> torch.Tensor:
    """Transformers 4.53.1 repeat_kv helper used by sdpa_attention_forward."""

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _fast_dllm_v2_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """SDPA wrapper matching the Transformers 4.53.1 call contract."""

    del kwargs
    if hasattr(module, "num_key_value_groups"):
        key = _repeat_kv_for_fast_dllm_v2(key, module.num_key_value_groups)
        value = _repeat_kv_for_fast_dllm_v2(value, module.num_key_value_groups)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _ensure_transformers_v453_sdpa_support() -> None:
    """Install the SDPA call contract Fast_dLLM_v2 was validated against."""

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(current, "_sglang_fast_dllm_v2_v453_compat", False):
        return

    _fast_dllm_v2_sdpa_attention_forward._sglang_fast_dllm_v2_v453_compat = True
    try:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _fast_dllm_v2_sdpa_attention_forward
    except TypeError:
        ALL_ATTENTION_FUNCTIONS.register(
            "sdpa", _fast_dllm_v2_sdpa_attention_forward
        )
    logger.info("Registered Fast_dLLM_v2 Transformers 4.53.1 SDPA compatibility.")


def _normalize_fast_dllm_v2_rope_config(config: Any) -> bool:
    """Restore the RoPE config shape used by Fast_dLLM_v2's reference stack.

    Transformers 5.x normalizes the checkpoint's plain Qwen RoPE fields into
    ``rope_scaling``/``rope_parameters`` dictionaries with ``rope_type="default"``.
    Fast_dLLM_v2's remote model was validated on Transformers 4.53.1, where
    those attributes are absent/None for this checkpoint. Leaving the 5.x
    normalized dictionaries in place makes the remote rotary module emit
    identity rotations (cos=1, sin=0), which corrupts attention while preserving
    tensor shapes.
    """

    if getattr(config, "model_type", None) != "Fast_dLLM_Qwen":
        return False

    changed = False
    for attr in ("rope_scaling", "rope_parameters"):
        value = getattr(config, attr, None)
        if isinstance(value, dict) and value.get("rope_type") == "default":
            setattr(config, attr, None)
            changed = True
    return changed


def _fast_dllm_v2_reference_rotary_forward(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = getattr(rotary_emb, "config", None)
    if config is None:
        raise ValueError("Fast_dLLM_v2 rotary compatibility requires a config.")

    inv_freq, attention_scaling = _compute_default_rope_parameters(
        config=config,
        device=x.device,
    )
    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    )
    position_ids_expanded = position_ids[:, None, :].float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _patch_fast_dllm_v2_rotary_embedding_forward(model: Any) -> bool:
    """Patch Fast_dLLM_v2 rotary modules to the reference Qwen RoPE behavior."""

    found = False
    changed = False
    for module in model.modules():
        if module.__class__.__name__ != "Fast_dLLM_QwenRotaryEmbedding":
            continue
        found = True
        if getattr(module.forward, "_sglang_fast_dllm_v2_v453_compat", False):
            continue

        def _forward(self, x, position_ids):
            return _fast_dllm_v2_reference_rotary_forward(self, x, position_ids)

        _forward._sglang_fast_dllm_v2_v453_compat = True
        module.forward = types.MethodType(_forward, module)
        changed = True
    if changed:
        logger.info(
            "Patched Fast_dLLM_v2 rotary embedding forward for Transformers "
            "4.53.1 compatibility."
        )
    return found


def _tensors_share_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device.type == "meta" or rhs.device.type == "meta":
        return False
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def _ensure_fast_dllm_v2_tied_embeddings(model: Any) -> bool:
    """Faithfully enforce Fast_dLLM_v2's tied embedding/LM-head contract."""

    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "Fast_dLLM_Qwen":
        return False
    if not getattr(config, "tie_word_embeddings", False):
        return False

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("Fast_dLLM_v2 requires input and output embeddings.")

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if input_weight is None or output_weight is None:
        raise ValueError("Fast_dLLM_v2 embeddings must expose weight tensors.")

    if _tensors_share_storage(input_weight, output_weight):
        return False

    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        try:
            tie_weights()
        except TypeError:
            tie_weights(recompute_mapping=True)

    output_embeddings = model.get_output_embeddings()
    output_weight = getattr(output_embeddings, "weight", None)
    if output_weight is not None and _tensors_share_storage(input_weight, output_weight):
        logger.warning("Fast_dLLM_v2 input embeddings and LM head were tied.")
        return True

    # Last-resort compatibility path for newer Transformers releases whose
    # metadata handling changed enough that the remote model's old list-style
    # tied-weight declaration was not applied during loading.
    output_embeddings.weight = input_weight
    logger.warning(
        "Fast_dLLM_v2 LM head was explicitly tied to input embeddings after "
        "load."
    )
    return True


def _ensure_transformers_legacy_dynamic_cache_support() -> None:
    """Use the list-backed DynamicCache API expected by Fast_dLLM_v2."""

    import transformers.cache_utils as cache_utils

    current_cache = cache_utils.DynamicCache
    try:
        cache = current_cache()
    except TypeError:
        cache = None
    if (
        cache is not None
        and hasattr(cache, "key_cache")
        and hasattr(cache, "value_cache")
        and hasattr(current_cache, "__getitem__")
    ):
        return

    class FastDllmV2LegacyDynamicCache:
        def __init__(self) -> None:
            self.key_cache: list[torch.Tensor] = []
            self.value_cache: list[torch.Tensor] = []
            self._seen_tokens = 0

        @property
        def seen_tokens(self):
            return self._seen_tokens

        def __getitem__(self, layer_idx: int):
            if layer_idx < len(self):
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access "
                f"layer with index {layer_idx}"
            )

        def __iter__(self):
            for layer_idx in range(len(self)):
                yield self[layer_idx]

        def __len__(self):
            return len(self.key_cache)

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cache_kwargs
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([], device=key_states.device))
                    self.value_cache.append(
                        torch.tensor([], device=value_states.device)
                    )
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif not self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            if layer_idx is None:
                layer_idx = 0
            if len(self.key_cache) <= layer_idx:
                return 0
            return self.key_cache[layer_idx].shape[-2]

        def get_max_length(self) -> Optional[int]:
            return None

        def get_max_cache_shape(self) -> Optional[int]:
            return None

        def crop(self, max_length: int) -> None:
            if max_length < 0:
                max_length = self.get_seq_length() - abs(max_length)
            if self.get_seq_length() <= max_length:
                return
            self._seen_tokens = max_length
            for layer_idx in range(len(self.key_cache)):
                if self.key_cache[layer_idx].numel():
                    self.key_cache[layer_idx] = self.key_cache[layer_idx][
                        ..., :max_length, :
                    ]
                    self.value_cache[layer_idx] = self.value_cache[layer_idx][
                        ..., :max_length, :
                    ]

        def reorder_cache(self, beam_idx: torch.LongTensor):
            for layer_idx in range(len(self.key_cache)):
                key_device = self.key_cache[layer_idx].device
                value_device = self.value_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, beam_idx.to(key_device)
                )
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, beam_idx.to(value_device)
                )

        def to_legacy_cache(self):
            return tuple(self[layer_idx] for layer_idx in range(len(self)))

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key_states, value_states) in enumerate(
                    past_key_values
                ):
                    cache.update(key_states, value_states, layer_idx)
            return cache

    cache_utils.DynamicCache = FastDllmV2LegacyDynamicCache
    logger.info("Registered Fast_dLLM_v2 legacy DynamicCache compatibility.")


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
    attn_implementation: Optional[str] = "sdpa"
    disable_hub_kernels: bool = True
    proposal_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_executor(
        cls, executor: "DllmDraftExecutor"
    ) -> "FastDllmV2RunnerConfig":
        raw_config = _load_algorithm_config(executor.algorithm_config)
        proposal_kwargs = dict(raw_config.get("proposal_kwargs", {}))
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
            attn_implementation=_optional_str(
                raw_config.get("attn_implementation", "sdpa")
            ),
            disable_hub_kernels=bool(raw_config.get("disable_hub_kernels", True)),
            proposal_kwargs=proposal_kwargs,
        )

    @classmethod
    def from_server_args(
        cls, server_args: "ServerArgs"
    ) -> "FastDllmV2RunnerConfig":
        raw_config = _load_algorithm_config(
            server_args.speculative_fast_dllm_v2_algorithm_config
        )
        proposal_kwargs = dict(raw_config.get("proposal_kwargs", {}))
        proposed_token_num = int(server_args.speculative_num_draft_tokens) - 1
        if proposed_token_num <= 0:
            raise ValueError(
                "FAST_DLLM_V2 requires --speculative-num-draft-tokens >= 2 "
                "because the linear verify block includes the current token."
            )
        tokenizer_path = raw_config.get(
            "tokenizer_path", server_args.speculative_draft_model_path
        )
        return cls(
            model_path=server_args.speculative_draft_model_path,
            tokenizer_path=tokenizer_path,
            proposed_token_num=proposed_token_num,
            block_size=int(raw_config.get("block_size", 32)),
            small_block_size=int(raw_config.get("small_block_size", 8)),
            threshold=float(raw_config.get("threshold", 0.9)),
            torch_dtype=str(raw_config.get("torch_dtype", "auto")),
            device_map=str(raw_config.get("device_map", "auto")),
            trust_remote_code=bool(raw_config.get("trust_remote_code", True)),
            attn_implementation=_optional_str(
                raw_config.get("attn_implementation", "sdpa")
            ),
            disable_hub_kernels=bool(raw_config.get("disable_hub_kernels", True)),
            proposal_kwargs=proposal_kwargs,
        )


@dataclass(slots=True)
class FastDllmV2RequestState:
    """Per-request state owned by the independent dLLM draft runner."""

    input_ids: list[int]
    accepted_token_count: int = 0
    draft_lookahead: list[int] = field(default_factory=list)
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

    SGLang mirrors Fast_dLLM_v2's block-diffusion sampler directly so
    speculative serving can stop once a contiguous proposal window is concrete.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._rope_config_normalized = False

    def propose(
        self,
        config: FastDllmV2RunnerConfig,
        request: IndependentDllmDraftRequest,
        states: dict[str, FastDllmV2RequestState],
    ) -> IndependentDllmDraftTokens:
        self._ensure_loaded(config)
        proposed = []
        for request_id in request.request_ids:
            state = states[request_id]
            generated = self._propose_from_state(config, state)
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
                "use_block_cache": bool(
                    config.proposal_kwargs.get("use_block_cache", False)
                ),
                **getattr(self, "model_metadata", {}),
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

        self._install_transformers_compat()
        hub_kernels_disabled = False
        if config.disable_hub_kernels:
            hub_kernels_disabled = _disable_transformers_hub_kernels_for_fast_dllm_v2()

        has_accelerate = self._has_accelerate()
        device_map = config.device_map if has_accelerate else None
        if not has_accelerate and config.device_map not in (None, "none"):
            logger.warning(
                "Fast_dLLM_v2 is falling back to a single-device load because "
                "accelerate is unavailable in this environment."
            )

        self.model = self._load_model(config, device_map=device_map)
        rotary_forward_patched = _patch_fast_dllm_v2_rotary_embedding_forward(
            self.model
        )
        self.model.eval()
        if config.disable_hub_kernels and hasattr(self.model, "config"):
            self.model.config.disable_custom_kernels = True
        tied_embeddings_after_load = _ensure_fast_dllm_v2_tied_embeddings(self.model)
        self.model = self._move_model_to_primary_device(self.model)
        transformers_version = self._transformers_version()
        self.model_metadata = {
            "tied_embeddings_after_load": tied_embeddings_after_load,
            "transformers_version": transformers_version,
            "reference_transformers_version": FAST_DLLM_V2_REFERENCE_TRANSFORMERS_VERSION,
            "hub_kernels_disabled": config.disable_hub_kernels,
            "hub_kernels_env_changed": hub_kernels_disabled,
            "sdpa_compat": "transformers_4.53.1",
            "rope_config_normalized": self._rope_config_normalized,
            "rotary_forward_patched": rotary_forward_patched,
        }
        self.tokenizer = self._load_tokenizer(config)

    def _install_transformers_compat(self) -> None:
        _ensure_transformers_default_rope_support()
        _ensure_transformers_tied_weights_support()
        _ensure_transformers_legacy_dynamic_cache_support()
        _ensure_transformers_v453_sdpa_support()

    def _has_accelerate(self) -> bool:
        return importlib.util.find_spec("accelerate") is not None

    def _load_model(
        self,
        config: FastDllmV2RunnerConfig,
        *,
        device_map: Optional[str],
    ) -> Any:
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_config = AutoConfig.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )
        self._rope_config_normalized = _normalize_fast_dllm_v2_rope_config(hf_config)

        kwargs: dict[str, Any] = {
            "config": hf_config,
            "torch_dtype": config.torch_dtype,
            "trust_remote_code": config.trust_remote_code,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation
        return AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)

    def _transformers_version(self) -> str:
        import transformers

        return transformers.__version__

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

    def _propose_from_state(
        self,
        config: FastDllmV2RunnerConfig,
        state: FastDllmV2RequestState,
    ) -> torch.Tensor:
        proposed_token_num = config.proposed_token_num
        lookahead = state.draft_lookahead
        if len(lookahead) < proposed_token_num:
            seed_input_ids = state.input_ids + lookahead
            generated = self._propose_one(config, seed_input_ids).tolist()
            lookahead.extend(int(token_id) for token_id in generated)

        device = self.model.device
        return torch.tensor(
            lookahead[:proposed_token_num],
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _initialize_proposal_block(
        input_tensor: torch.Tensor,
        seq_block_idx: torch.Tensor,
        block_idx: int,
        block_size: int,
        mask_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (seq_block_idx == block_idx).all():
            mask_count = block_size - input_tensor.shape[1] % block_size
            mask_tokens = torch.full(
                (input_tensor.shape[0], mask_count),
                mask_id,
                device=input_tensor.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_tensor, mask_tokens], dim=1)
            return x_init, x_init

        return input_tensor[:, : (block_idx + 1) * block_size], input_tensor

    @torch.no_grad()
    def _propose_one(
        self,
        config: FastDllmV2RunnerConfig,
        input_ids: list[int],
    ) -> torch.Tensor:
        """Run Fast_dLLM_v2 sampling only until the next proposal block is ready.

        In the speculative path we only need the next contiguous proposal
        tokens, so this mirrors Fast_dLLM_v2's reference block-diffusion loop
        and returns as soon as those positions are no longer masked.
        """

        assert self.model is not None
        assert self.tokenizer is not None
        self._validate_proposal_config(config)

        device = self.model.device
        prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
        original_prompt_len = prompt.shape[1]
        proposed_token_num = config.proposed_token_num
        block_size = config.block_size
        small_block_size = config.small_block_size
        if block_size <= 0 or small_block_size <= 0 or block_size % small_block_size:
            raise ValueError(
                "Fast_dLLM_v2 requires a positive small_block_size that divides "
                "block_size."
            )

        kwargs = config.proposal_kwargs
        mask_id = int(kwargs.get("mask_id", FAST_DLLM_V2_MASK_ID))
        stop_token = int(kwargs.get("stop_token", FAST_DLLM_V2_STOP_TOKEN))
        top_p = float(kwargs.get("top_p", 0.95))
        temperature = float(kwargs.get("temperature", 0.0))
        threshold = config.threshold
        use_block_cache = bool(kwargs.get("use_block_cache", False))

        max_new_tokens = _fast_dllm_v2_internal_generation_budget(
            original_prompt_len,
            proposed_token_num,
            block_size,
        )

        input_tensor = prompt
        seq_len = torch.tensor([original_prompt_len], device=device, dtype=torch.long)
        min_len = int(seq_len.min().item())
        num_blocks = max_new_tokens // block_size + int(seq_len.max().item()) // block_size

        if min_len > block_size:
            prefix_len = min_len // block_size * block_size
            output = self.model.forward(
                input_ids=input_tensor[:, :prefix_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                if input_tensor.shape[1] <= min_len:
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                else:
                    input_tensor[:, min_len] = next_token.squeeze(dim=-1)
                ready = self._first_contiguous_proposal(
                    input_tensor,
                    original_prompt_len,
                    proposed_token_num,
                    mask_id,
                )
                if ready is not None:
                    return ready
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        for block_idx in range(start_block_idx, num_blocks):
            x_init, input_tensor = self._initialize_proposal_block(
                input_tensor,
                seq_block_idx,
                block_idx,
                block_size,
                mask_id,
            )

            x_t = x_init.clone()
            block_past_key_values = None

            while True:
                ready = self._first_contiguous_proposal(
                    x_t,
                    original_prompt_len,
                    proposed_token_num,
                    mask_id,
                )
                if ready is not None:
                    return ready

                mask_idx = x_t[:, -block_size:] == mask_id
                if mask_idx.sum() == 0:
                    output = self.model.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size
                    start = -block_size + small_block_start_idx
                    end = (
                        None
                        if block_size == small_block_end_idx
                        else -block_size + small_block_end_idx
                    )

                    while True:
                        ready = self._first_contiguous_proposal(
                            x_t,
                            original_prompt_len,
                            proposed_token_num,
                            mask_id,
                        )
                        if ready is not None:
                            return ready

                        mask_idx = x_t[:, -block_size:] == mask_id
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        if use_block_cache:
                            block_start = -block_size + small_block_start_idx
                            if (
                                block_past_key_values is None
                                or (x_t[:, block_start] == mask_id).any()
                            ):
                                output = self.model.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                )
                                logits = output.logits
                                block_past_key_values = output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.model.forward(
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                ).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.model.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                            ).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                        x_1, p_1t = self.model.sample_with_top_p(
                            logits,
                            top_p=top_p,
                            temperature=temperature,
                        )
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)),
                            -1,
                        )
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = x1_p > threshold
                        max_prob_idx = x1_p.argmax(dim=-1)
                        row_idx = torch.arange(x_1.shape[0], device=x_1.device)
                        unmask_idx[row_idx, max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]
                        target_slice = x_t[:, start:end]
                        target_slice[unmask_idx] = x_1[unmask_idx]

                        if ((x_1 == stop_token) & unmask_idx).any():
                            ready = self._first_contiguous_proposal(
                                x_t,
                                original_prompt_len,
                                proposed_token_num,
                                mask_id,
                            )
                            if ready is not None:
                                return ready

            if input_tensor.shape[1] == x_t.shape[1]:
                input_tensor = x_t
            else:
                input_tensor[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_tensor = torch.cat([input_tensor, x_t[:, -1:]], dim=1)
                elif input_tensor.shape[1] <= (block_idx + 1) * block_size:
                    input_tensor = x_t
                else:
                    input_tensor[
                        seq_block_idx == block_idx, (block_idx + 1) * block_size
                    ] = x_t[
                        seq_block_idx == block_idx, (block_idx + 1) * block_size
                    ]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

        ready = self._first_contiguous_proposal(
            input_tensor,
            original_prompt_len,
            proposed_token_num,
            mask_id,
        )
        if ready is not None:
            return ready
        return self._pad_proposal(input_tensor, original_prompt_len, config, mask_id)

    def _validate_proposal_config(self, config: FastDllmV2RunnerConfig) -> None:
        assert self.model is not None
        unsupported_kwargs = sorted(
            set(config.proposal_kwargs) - FAST_DLLM_V2_PROPOSAL_KWARGS
        )
        if unsupported_kwargs:
            raise _FastDllmV2ProposalUnsupported(
                "SGLang's Fast_dLLM_v2 proposal path does not support "
                f"proposal_kwargs: {unsupported_kwargs}"
            )
        if not hasattr(self.model, "sample_with_top_p"):
            raise _FastDllmV2ProposalUnsupported(
                "Fast_dLLM_v2 model does not expose sample_with_top_p"
            )

    def _first_contiguous_proposal(
        self,
        token_ids: torch.Tensor,
        prompt_len: int,
        proposed_token_num: int,
        mask_id: int,
    ) -> Optional[torch.Tensor]:
        end = prompt_len + proposed_token_num
        if token_ids.shape[1] < end:
            return None
        candidate = token_ids[0, prompt_len:end]
        if (candidate == mask_id).any():
            return None
        return candidate.clone()

    def _pad_proposal(
        self,
        token_ids: torch.Tensor,
        prompt_len: int,
        config: FastDllmV2RunnerConfig,
        mask_id: int,
    ) -> torch.Tensor:
        assert self.tokenizer is not None
        device = token_ids.device
        candidate = token_ids[0, prompt_len : prompt_len + config.proposed_token_num]
        candidate = candidate[candidate != mask_id]
        if candidate.numel() >= config.proposed_token_num:
            return candidate[: config.proposed_token_num].clone()

        pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                "Fast_dLLM_v2 proposal produced too few concrete tokens and "
                "no EOS/PAD token is available for padding."
            )
        padding = torch.full(
            (config.proposed_token_num - candidate.numel(),),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        return torch.cat([candidate.clone(), padding], dim=0)


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
            self._consume_lookahead(state, accepted_token_ids)
        self.runtime.extend_after_accept(self.config, accepted, self.states)

    def release(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.states.pop(request_id, None)
        self.runtime.release(self.config, request_ids, self.states)

    def _refresh_states(self, request: IndependentDllmDraftRequest) -> None:
        for request_id, input_ids in zip(request.request_ids, request.input_ids):
            new_input_ids = list(input_ids)
            state = self.states.get(request_id)
            if state is None:
                self.states[request_id] = FastDllmV2RequestState(
                    input_ids=new_input_ids
                )
                continue

            old_input_ids = state.input_ids
            if len(new_input_ids) < len(old_input_ids) or (
                new_input_ids[: len(old_input_ids)] != old_input_ids
            ):
                state.input_ids = new_input_ids
                state.accepted_token_count = 0
                state.draft_lookahead.clear()
                state.metadata.clear()
                continue

            committed = new_input_ids[len(old_input_ids) :]
            if committed:
                state.input_ids = new_input_ids
                state.accepted_token_count += len(committed)
                self._consume_lookahead(state, committed)

    @staticmethod
    def _consume_lookahead(
        state: FastDllmV2RequestState,
        committed_token_ids: list[int],
    ) -> None:
        for token_id in committed_token_ids:
            if state.draft_lookahead and state.draft_lookahead[0] == int(token_id):
                del state.draft_lookahead[0]
                continue
            state.draft_lookahead.clear()
            break


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


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    parsed = str(value)
    if parsed.lower() in ("", "none", "null"):
        return None
    return parsed


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
