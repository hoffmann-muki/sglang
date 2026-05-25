"""Fast_dLLM_v2 proposal runner for colocated co-draft experiments."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
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
    "profile",
    "profile_log_interval",
    "stop_token",
    "temperature",
    "top_p",
    "trace_full_tensors",
    "trace_max_events",
    "trace_path",
    "trace_topk",
    "use_block_cache",
}


class _FastDllmV2ProposalUnsupported(RuntimeError):
    """Raised when SGLang's Fast_dLLM_v2 path cannot honor a request."""


class _FastDllmV2NativeTraceRecorder:
    """Small file-backed trace recorder for SGLang-native Fast_dLLM_v2 runs."""

    def __init__(
        self,
        path: str,
        *,
        max_events: int = 128,
        full_tensors: bool = False,
        topk: int = 8,
    ) -> None:
        self.path = Path(path)
        self.max_events = max(0, int(max_events))
        self.full_tensors = bool(full_tensors)
        self.topk = max(1, int(topk))
        self.phase = "uninitialized"
        self.events: list[dict[str, Any]] = []
        self.hook_handles: list[Any] = []
        self.hooked_model_id: Optional[int] = None
        self.trace: dict[str, Any] = {
            "metadata": {
                "runtime": "sglang_native",
                "trace_path": str(self.path),
                "trace_full_tensors": self.full_tensors,
                "trace_max_events": self.max_events,
                "trace_topk": self.topk,
            },
            "native": {"events": self.events},
        }

    def set_metadata(self, **metadata: Any) -> None:
        self.trace["metadata"].update(_json_safe(metadata))

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def install_model_hooks(self, model: Any) -> None:
        if model is None or self.hooked_model_id == id(model):
            return
        self.remove_hooks()
        self.hooked_model_id = id(model)
        for name in (
            "model.embed_tokens",
            "model.layers.0.input_layernorm",
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.rotary_emb",
            "model.layers.0.self_attn.attn",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.post_attention_layernorm",
            "model.layers.0.mlp",
            "model.norm",
            "logits_processor",
        ):
            module = _get_dotted_module(model, name)
            if module is not None:
                self._hook_module(name, module)

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.hooked_model_id = None

    def record(self, name: str, kind: str, payload: dict[str, Any]) -> None:
        if len(self.events) >= self.max_events:
            return
        self.events.append(
            {
                "phase": self.phase,
                "name": name,
                "kind": kind,
                "payload": _summarize_trace_object(
                    payload,
                    full_tensors=self.full_tensors,
                    topk=self.topk,
                ),
            }
        )
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.trace, self.path)
        self.path.with_suffix(".json").write_text(
            json.dumps(_json_safe(self.trace), indent=2) + "\n"
        )

    def _hook_module(self, name: str, module: torch.nn.Module) -> None:
        def pre_hook(_module, args, kwargs=None):
            self.record(
                name,
                "input",
                {
                    "args": args,
                    "kwargs": kwargs or {},
                },
            )

        def post_hook(_module, args, kwargs_or_output, output=None):
            if output is None:
                kwargs = {}
                actual_output = kwargs_or_output
            else:
                kwargs = kwargs_or_output or {}
                actual_output = output
            self.record(
                name,
                "output",
                {
                    "args": args,
                    "kwargs": kwargs,
                    "output": actual_output,
                },
            )

        try:
            self.hook_handles.append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            self.hook_handles.append(
                module.register_forward_hook(post_hook, with_kwargs=True)
            )
        except TypeError:
            self.hook_handles.append(module.register_forward_pre_hook(pre_hook))
            self.hook_handles.append(module.register_forward_hook(post_hook))


def _get_dotted_module(root: Any, dotted_name: str) -> Optional[torch.nn.Module]:
    current = root
    for part in dotted_name.split("."):
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, torch.nn.Module) else None


def _summarize_trace_object(
    value: Any,
    *,
    full_tensors: bool,
    topk: int,
) -> Any:
    if torch.is_tensor(value):
        return _summarize_trace_tensor(value, full_tensors=full_tensors, topk=topk)
    if isinstance(value, dict):
        return {
            str(k): _summarize_trace_object(v, full_tensors=full_tensors, topk=topk)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _summarize_trace_object(v, full_tensors=full_tensors, topk=topk)
            for v in value
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return repr(value)
    return repr(value)


def _summarize_trace_tensor(
    tensor: torch.Tensor,
    *,
    full_tensors: bool,
    topk: int,
) -> dict[str, Any]:
    detached = tensor.detach()
    cpu = detached.float().cpu()
    flat = cpu.reshape(-1)
    summary: dict[str, Any] = {
        "type": "tensor_summary",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "mean": float(cpu.mean()) if flat.numel() else 0.0,
        "std": float(cpu.std(unbiased=False)) if flat.numel() else 0.0,
        "min": float(cpu.min()) if flat.numel() else 0.0,
        "max": float(cpu.max()) if flat.numel() else 0.0,
        "head": flat[: min(8, flat.numel())].tolist(),
    }
    if detached.ndim >= 1 and flat.numel():
        values, indices = torch.topk(flat, k=min(topk, flat.numel()))
        summary["flat_topk"] = {
            "indices": indices.tolist(),
            "values": values.tolist(),
        }
    if detached.ndim >= 2 and detached.shape[-1] > 0:
        rows = cpu.reshape(-1, cpu.shape[-1])
        values, indices = torch.topk(rows[-1], k=min(topk, rows.shape[-1]))
        summary["last_row_topk"] = {
            "indices": indices.tolist(),
            "values": values.tolist(),
        }
    if full_tensors:
        summary["tensor"] = cpu
    return summary


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "tensor"}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


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
    runtime: str = "transformers"
    context_length: Optional[int] = None
    native_max_total_tokens: int = 4096
    native_max_running_requests: int = 8
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
            runtime=str(raw_config.get("runtime", "transformers")),
            context_length=_optional_int(raw_config.get("context_length")),
            native_max_total_tokens=int(
                raw_config.get("native_max_total_tokens", 4096)
            ),
            native_max_running_requests=int(
                raw_config.get("native_max_running_requests", 8)
            ),
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
            runtime=str(raw_config.get("runtime", "transformers")),
            context_length=_optional_int(raw_config.get("context_length")),
            native_max_total_tokens=int(
                raw_config.get("native_max_total_tokens", 4096)
            ),
            native_max_running_requests=int(
                raw_config.get("native_max_running_requests", 8)
            ),
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
class _SGLangNativeCacheState:
    """Ephemeral SGLang KV state used as native `past_key_values`."""

    req: Any
    req_pool_idx: int
    allocator_state_before_alloc: Any
    kv_len: int
    released: bool = False


@dataclass(slots=True)
class _SGLangNativeBlockCacheState:
    """Ephemeral native KV state for one Fast_dLLM_v2 refinement block."""

    req: Any
    req_pool_idx: int
    allocator_state_before_alloc: Any
    prefix_len: int
    block_size: int
    token_ids: torch.Tensor
    released: bool = False


class SGLangNativeFastDllmV2Runtime:
    """Native SGLang runtime shell for Fast_dLLM_v2 proposal generation.

    The proposal-control-flow pieces are intentionally shared with the
    Transformers-backed reference path. This runtime owns the SGLang
    ModelRunner/ForwardBatch bridge and ephemeral native KV handles used by the
    block-diffusion proposal loop.
    """

    def __init__(self, model_runner: Any):
        self.model_runner = model_runner
        self._active_native_caches: list[_SGLangNativeCacheState] = []
        self._active_native_block_caches: list[_SGLangNativeBlockCacheState] = []
        self._trace: Optional[_FastDllmV2NativeTraceRecorder] = None
        self._trace_path: Optional[str] = None

    def propose(
        self,
        config: FastDllmV2RunnerConfig,
        request: IndependentDllmDraftRequest,
        states: dict[str, FastDllmV2RequestState],
    ) -> IndependentDllmDraftTokens:
        self._configure_trace(config)
        self._trace_record(
            "proposal",
            "input",
            {
                "request_ids": list(request.request_ids),
                "prefix_lens": request.prefix_lens,
                "current_token_ids": request.current_token_ids,
                "proposed_token_num": request.proposed_token_num,
                "input_lens": [len(input_ids) for input_ids in request.input_ids],
                "input_ids": [
                    torch.tensor(input_ids, dtype=torch.long)
                    for input_ids in request.input_ids
                ],
                "config": {
                    "block_size": config.block_size,
                    "small_block_size": config.small_block_size,
                    "threshold": config.threshold,
                    "use_block_cache": bool(
                        config.proposal_kwargs.get("use_block_cache", False)
                    ),
                },
            },
        )
        proposed = []
        for request_id in request.request_ids:
            state = states[request_id]
            generated = _fast_dllm_v2_propose_from_state(self, config, state)
            proposed.append(generated)

        profiles = [
            states[request_id].metadata.get("last_profile")
            for request_id in request.request_ids
        ]
        profile_total = FastDllmV2BlockProposalEngine.aggregate_profiles(profiles)
        metadata = {
            "runner": "fast_dllm_v2",
            "runtime": "sglang_native",
            "block_size": config.block_size,
            "small_block_size": config.small_block_size,
            "threshold": config.threshold,
            "use_block_cache": bool(
                config.proposal_kwargs.get("use_block_cache", False)
            ),
            "profile_enabled": bool(config.proposal_kwargs.get("profile", False)),
        }
        if profile_total:
            metadata["profile_total"] = profile_total
            metadata["profile_log_interval"] = int(
                config.proposal_kwargs.get("profile_log_interval", 1)
            )

        proposed_token_ids = torch.stack(proposed).to(request.current_token_ids.device)
        self._trace_record(
            "proposal",
            "output",
            {
                "request_ids": list(request.request_ids),
                "proposed_token_ids": proposed_token_ids,
                "metadata": metadata,
            },
        )
        return IndependentDllmDraftTokens(
            request_ids=list(request.request_ids),
            current_token_ids=request.current_token_ids,
            proposed_token_ids=proposed_token_ids,
            prefix_lens=request.prefix_lens,
            metadata=metadata,
        )

    def extend_after_accept(
        self,
        config: FastDllmV2RunnerConfig,
        accepted: IndependentDllmAcceptedTokens,
        states: dict[str, FastDllmV2RequestState],
    ) -> None:
        return None

    @property
    def device(self) -> torch.device:
        device = getattr(self.model_runner, "device", "cuda")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def validate_proposal_config(self, config: FastDllmV2RunnerConfig) -> None:
        unsupported_kwargs = sorted(
            set(config.proposal_kwargs) - FAST_DLLM_V2_PROPOSAL_KWARGS
        )
        if unsupported_kwargs:
            raise _FastDllmV2ProposalUnsupported(
                "SGLang's Fast_dLLM_v2 native proposal path does not support "
                f"proposal_kwargs: {unsupported_kwargs}"
            )

    def _propose_one(
        self,
        config: FastDllmV2RunnerConfig,
        input_ids: list[int],
        profile: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        try:
            return FastDllmV2BlockProposalEngine(self).propose_one(
                config,
                input_ids,
                profile,
            )
        finally:
            self._release_active_native_caches()

    def forward(
        self,
        profile: Optional[dict[str, Any]],
        phase: str,
        **kwargs,
    ) -> Any:
        self._validate_native_forward_kwargs(phase, kwargs)
        input_ids = kwargs["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError(
                "Fast_dLLM_v2 native forward currently supports a single "
                f"request per draft invocation, got input_ids.shape={tuple(input_ids.shape)}."
            )

        past_key_values = kwargs.get("past_key_values")
        update_past_key_values = bool(kwargs.get("update_past_key_values", False))

        if profile is not None:
            start = time.perf_counter()
        block_cache_state = None
        if kwargs.get("use_block_cache"):
            block_cache_state = self._forward_with_block_cache(
                phase,
                input_ids,
                prefix_cache=past_key_values,
                block_cache=kwargs.get("block_past_key_values"),
                replace_position=kwargs.get("replace_position"),
            )
            logits = block_cache_state.logits
            cache_state = past_key_values
        else:
            logits, cache_state = self._forward_tokens(
                input_ids.reshape(-1),
                prefix_cache=past_key_values,
                keep_cache=update_past_key_values,
                block_size=kwargs.get("block_size"),
                phase=phase,
            )
        if profile is not None:
            elapsed = time.perf_counter() - start
            phase_calls_key = f"{phase}_forward_calls"
            phase_time_key = f"{phase}_forward_time_s"
            profile["forward_calls"] += 1
            profile["forward_time_s"] += elapsed
            profile[phase_calls_key] = profile.get(phase_calls_key, 0) + 1
            profile[phase_time_key] = profile.get(phase_time_key, 0.0) + elapsed
        logit_len = int(logits.shape[0]) if logits.ndim == 2 else input_ids.shape[1]
        return types.SimpleNamespace(
            logits=logits.view(1, logit_len, logits.shape[-1]),
            past_key_values=cache_state,
            block_past_key_values=(
                None if block_cache_state is None else block_cache_state.block_cache
            ),
        )

    def _validate_native_forward_kwargs(
        self,
        phase: str,
        kwargs: dict[str, Any],
    ) -> None:
        unsupported = []
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None and not isinstance(
            past_key_values, _SGLangNativeCacheState
        ):
            unsupported.append("foreign_past_key_values")
        block_past_key_values = kwargs.get("block_past_key_values")
        if block_past_key_values is not None and not isinstance(
            block_past_key_values, _SGLangNativeBlockCacheState
        ):
            unsupported.append("foreign_block_past_key_values")
        if kwargs.get("replace_position") is not None and not kwargs.get(
            "use_block_cache"
        ):
            unsupported.append("replace_position_without_block_cache")
        if unsupported:
            raise NotImplementedError(
                "Fast_dLLM_v2 native forward received unsupported cached "
                f"state (phase={phase!r}, unsupported={unsupported})."
            )

    def _forward_with_block_cache(
        self,
        phase: str,
        input_ids: torch.Tensor,
        *,
        prefix_cache: Optional[_SGLangNativeCacheState],
        block_cache: Optional[_SGLangNativeBlockCacheState],
        replace_position: Optional[int],
    ) -> Any:
        if block_cache is None:
            return self._refresh_block_cache(
                input_ids.reshape(-1),
                prefix_cache,
                phase=phase,
            )
        if replace_position is None:
            raise ValueError(
                "Fast_dLLM_v2 native block-cache reuse requires replace_position."
            )
        return self._reuse_block_cache(
            block_cache,
            input_ids.reshape(-1),
            int(replace_position),
            phase=phase,
        )

    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        prefix_cache: Optional[_SGLangNativeCacheState],
        keep_cache: bool,
        block_size: Optional[int],
        phase: str,
    ) -> tuple[torch.Tensor, Optional[_SGLangNativeCacheState]]:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        device = self.device
        input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
        num_tokens = int(input_ids.numel())
        if num_tokens <= 0:
            raise ValueError("Fast_dLLM_v2 native forward requires at least one token.")

        self._ensure_native_model_runner_ready()
        self._ensure_native_page_size_one()
        block_size = num_tokens if block_size is None else int(block_size)
        if block_size <= 0:
            raise ValueError(
                f"Fast_dLLM_v2 native forward requires block_size > 0, got {block_size}."
            )
        if num_tokens > block_size:
            if not keep_cache or num_tokens % block_size != 0:
                raise ValueError(
                    "Fast_dLLM_v2 native multi-block forward is only supported "
                    "for cached prefix construction with a whole number of blocks."
                )
            logits = None
            cache_state = prefix_cache
            for start in range(0, num_tokens, block_size):
                logits, cache_state = self._forward_tokens(
                    input_ids[start : start + block_size],
                    prefix_cache=cache_state,
                    keep_cache=True,
                    block_size=block_size,
                    phase=f"{phase}.block_{start // block_size}",
                )
            assert logits is not None
            return logits, cache_state
        if num_tokens != block_size:
            raise ValueError(
                "Fast_dLLM_v2 native DLLM_EXTEND forwards must be exactly one "
                f"diffusion block, got num_tokens={num_tokens}, block_size={block_size}."
            )

        allocator = self.model_runner.token_to_kv_pool_allocator
        cache_state = prefix_cache
        if cache_state is None:
            cache_state = self._allocate_native_cache_state()
            if keep_cache:
                self._active_native_caches.append(cache_state)

        prefix_len = int(cache_state.kv_len)
        req_pool_indices_tensor = torch.tensor(
            [cache_state.req_pool_idx],
            dtype=torch.int64,
            device=device,
        )
        token_to_kv_pool_state = None if keep_cache else allocator.backup_state()
        try:
            out_cache_loc = allocator.alloc(num_tokens)
            if out_cache_loc is None:
                raise RuntimeError(
                    "Fast_dLLM_v2 native forward ran out of KV slots while "
                    f"allocating {num_tokens} tokens."
                )

            self.model_runner.req_to_token_pool.write(
                (cache_state.req_pool_idx, slice(prefix_len, prefix_len + num_tokens)),
                out_cache_loc,
            )

            seq_len = prefix_len + num_tokens
            seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
            seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
            positions = torch.arange(
                prefix_len,
                seq_len,
                dtype=torch.int64,
                device=device,
            )
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DLLM_EXTEND,
                batch_size=1,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices_tensor,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=seq_len,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                extend_num_tokens=num_tokens,
                extend_seq_lens=torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=device,
                ),
                extend_prefix_lens=torch.tensor(
                    [prefix_len],
                    dtype=torch.int32,
                    device=device,
                ),
                extend_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
                extend_prefix_lens_cpu=[prefix_len],
                extend_seq_lens_cpu=[num_tokens],
                extend_logprob_start_lens_cpu=[0],
                return_logprob=False,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            with torch.inference_mode():
                full_logits = self._run_model_runner_forward(
                    forward_batch,
                    phase=phase,
                    input_ids=input_ids,
                    prefix_len=prefix_len,
                    block_size=num_tokens,
                    out_cache_loc=out_cache_loc,
                    req_pool_idx=cache_state.req_pool_idx,
                )
            if full_logits is None:
                raise RuntimeError(
                    "Fast_dLLM_v2 native model did not return full_logits for "
                    "DLLM_EXTEND forward."
                )
            if keep_cache:
                cache_state.kv_len = seq_len
            return full_logits, cache_state if keep_cache else None
        finally:
            if not keep_cache:
                if token_to_kv_pool_state is not None:
                    allocator.restore_state(token_to_kv_pool_state)
                if prefix_cache is None:
                    self._release_native_cache_state(cache_state)

    def _refresh_block_cache(
        self,
        block_ids: torch.Tensor,
        prefix_cache: Optional[_SGLangNativeCacheState],
        *,
        phase: str,
    ) -> Any:
        block_size = int(block_ids.numel())
        if block_size <= 0:
            raise ValueError("Fast_dLLM_v2 native block cache requires tokens.")

        prefix_len = 0 if prefix_cache is None else int(prefix_cache.kv_len)
        block_cache = self._allocate_native_block_cache_state(
            prefix_cache=prefix_cache,
            block_ids=block_ids,
        )
        self._active_native_block_caches.append(block_cache)
        logits = self._run_dllm_block_forward(
            input_ids=block_cache.token_ids,
            req_pool_idx=block_cache.req_pool_idx,
            prefix_len=prefix_len,
            block_size=block_size,
            out_cache_loc=self._block_cache_locs(block_cache),
            phase=phase,
        )
        return types.SimpleNamespace(logits=logits, block_cache=block_cache)

    def _reuse_block_cache(
        self,
        block_cache: _SGLangNativeBlockCacheState,
        input_ids: torch.Tensor,
        replace_position: int,
        *,
        phase: str,
    ) -> Any:
        if block_cache.released:
            raise RuntimeError("Fast_dLLM_v2 native block cache was already released.")
        num_tokens = int(input_ids.numel())
        if num_tokens <= 0:
            raise ValueError(
                "Fast_dLLM_v2 native block-cache reuse requires at least one token."
            )
        if replace_position < 0 or replace_position + num_tokens > block_cache.block_size:
            raise ValueError(
                "Fast_dLLM_v2 native block-cache replace range is out of bounds: "
                f"replace_position={replace_position}, num_tokens={num_tokens}, "
                f"block_size={block_cache.block_size}."
            )

        block_cache.token_ids[
            replace_position : replace_position + num_tokens
        ] = input_ids.to(block_cache.token_ids.device, dtype=torch.long)
        full_logits = self._run_dllm_block_forward(
            input_ids=block_cache.token_ids,
            req_pool_idx=block_cache.req_pool_idx,
            prefix_len=block_cache.prefix_len,
            block_size=block_cache.block_size,
            out_cache_loc=self._block_cache_locs(block_cache),
            phase=phase,
        )
        logits = full_logits[replace_position : replace_position + num_tokens]
        return types.SimpleNamespace(logits=logits, block_cache=block_cache)

    def _run_dllm_block_forward(
        self,
        *,
        input_ids: torch.Tensor,
        req_pool_idx: int,
        prefix_len: int,
        block_size: int,
        out_cache_loc: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        device = self.device
        input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
        if int(input_ids.numel()) != block_size:
            raise ValueError(
                "Fast_dLLM_v2 native block forward requires a complete block, "
                f"got {input_ids.numel()} tokens for block_size={block_size}."
            )

        seq_len = int(prefix_len) + int(block_size)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=input_ids,
            req_pool_indices=torch.tensor([req_pool_idx], dtype=torch.int64, device=device),
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_len,
            seq_lens_cpu=seq_lens_cpu,
            positions=torch.arange(
                prefix_len,
                seq_len,
                dtype=torch.int64,
                device=device,
            ),
            extend_num_tokens=block_size,
            extend_seq_lens=torch.tensor([block_size], dtype=torch.int32, device=device),
            extend_prefix_lens=torch.tensor(
                [prefix_len],
                dtype=torch.int32,
                device=device,
            ),
            extend_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
            extend_prefix_lens_cpu=[prefix_len],
            extend_seq_lens_cpu=[block_size],
            extend_logprob_start_lens_cpu=[0],
            return_logprob=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        full_logits = self._run_model_runner_forward(
            forward_batch,
            phase=phase,
            input_ids=input_ids,
            prefix_len=prefix_len,
            block_size=block_size,
            out_cache_loc=out_cache_loc,
            req_pool_idx=req_pool_idx,
        )
        if full_logits is None:
            raise RuntimeError(
                "Fast_dLLM_v2 native model did not return full_logits for "
                "DLLM_EXTEND block-cache forward."
            )
        return full_logits

    def _run_model_runner_forward(
        self,
        forward_batch: Any,
        *,
        phase: str,
        input_ids: torch.Tensor,
        prefix_len: int,
        block_size: int,
        out_cache_loc: torch.Tensor,
        req_pool_idx: int,
    ) -> Optional[torch.Tensor]:
        self._trace_set_phase(phase)
        self._trace_record(
            "model_runner.forward",
            "input",
            {
                "phase": phase,
                "input_ids": input_ids,
                "prefix_len": prefix_len,
                "block_size": block_size,
                "req_pool_idx": req_pool_idx,
                "seq_lens": forward_batch.seq_lens,
                "positions": forward_batch.positions,
                "out_cache_loc": out_cache_loc,
                "extend_prefix_lens": forward_batch.extend_prefix_lens,
                "extend_seq_lens": forward_batch.extend_seq_lens,
            },
        )
        with torch.inference_mode():
            logits_output = self.model_runner.forward(forward_batch).logits_output
        full_logits = getattr(logits_output, "full_logits", None)
        self._trace_record(
            "model_runner.forward",
            "output",
            {
                "phase": phase,
                "full_logits": full_logits,
            },
        )
        return full_logits

    def _allocate_native_cache_state(self) -> _SGLangNativeCacheState:
        self._ensure_native_model_runner_ready()
        allocator = self.model_runner.token_to_kv_pool_allocator
        fake_req = types.SimpleNamespace(
            rid="fast_dllm_v2_native",
            req_pool_idx=None,
            is_chunked=0,
            kv_committed_len=0,
        )
        req_pool_indices = self.model_runner.req_to_token_pool.alloc([fake_req])
        if req_pool_indices is None:
            raise RuntimeError("Fast_dLLM_v2 native forward ran out of request slots.")
        return _SGLangNativeCacheState(
            req=fake_req,
            req_pool_idx=int(req_pool_indices[0]),
            allocator_state_before_alloc=allocator.backup_state(),
            kv_len=0,
        )

    def _ensure_native_model_runner_ready(self) -> None:
        required_attrs = (
            "attn_backend",
            "forward",
            "req_to_token_pool",
            "token_to_kv_pool",
            "token_to_kv_pool_allocator",
        )
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(self.model_runner, attr)
        ]
        if missing_attrs:
            raise NotImplementedError(
                "Fast_dLLM_v2 native forward requires an initialized SGLang "
                f"ModelRunner, missing attributes={missing_attrs}."
            )

    def _ensure_native_page_size_one(self) -> None:
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator_page_size = int(getattr(allocator, "page_size", 1))
        if allocator_page_size != 1:
            raise NotImplementedError(
                "Fast_dLLM_v2 native forward currently supports "
                f"page_size=1 only, got page_size={allocator_page_size}."
            )

    def _allocate_native_block_cache_state(
        self,
        *,
        prefix_cache: Optional[_SGLangNativeCacheState],
        block_ids: torch.Tensor,
    ) -> _SGLangNativeBlockCacheState:
        self._ensure_native_model_runner_ready()
        self._ensure_native_page_size_one()
        allocator = self.model_runner.token_to_kv_pool_allocator
        block_ids = block_ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        block_size = int(block_ids.numel())
        prefix_len = 0 if prefix_cache is None else int(prefix_cache.kv_len)

        fake_req = types.SimpleNamespace(
            rid="fast_dllm_v2_native_block",
            req_pool_idx=None,
            is_chunked=0,
            kv_committed_len=0,
        )
        req_pool_indices = self.model_runner.req_to_token_pool.alloc([fake_req])
        if req_pool_indices is None:
            raise RuntimeError(
                "Fast_dLLM_v2 native block cache ran out of request slots."
            )

        allocator_state = allocator.backup_state()
        block_cache = _SGLangNativeBlockCacheState(
            req=fake_req,
            req_pool_idx=int(req_pool_indices[0]),
            allocator_state_before_alloc=allocator_state,
            prefix_len=prefix_len,
            block_size=block_size,
            token_ids=block_ids.clone(),
        )
        if prefix_len > 0:
            assert prefix_cache is not None
            prefix_locs = self.model_runner.req_to_token_pool.req_to_token[
                prefix_cache.req_pool_idx,
                :prefix_len,
            ]
            self.model_runner.req_to_token_pool.write(
                (block_cache.req_pool_idx, slice(0, prefix_len)),
                prefix_locs,
            )

        block_locs = allocator.alloc(block_size)
        if block_locs is None:
            allocator.restore_state(allocator_state)
            self.model_runner.req_to_token_pool.free(fake_req)
            block_cache.released = True
            raise RuntimeError(
                "Fast_dLLM_v2 native block cache ran out of KV slots while "
                f"allocating {block_size} tokens."
            )
        self.model_runner.req_to_token_pool.write(
            (block_cache.req_pool_idx, slice(prefix_len, prefix_len + block_size)),
            block_locs,
        )
        return block_cache

    def _block_cache_locs(
        self,
        block_cache: _SGLangNativeBlockCacheState,
    ) -> torch.Tensor:
        start = block_cache.prefix_len
        end = start + block_cache.block_size
        return self.model_runner.req_to_token_pool.req_to_token[
            block_cache.req_pool_idx,
            start:end,
        ].to(torch.int64)

    def _release_native_cache_state(
        self,
        cache_state: Optional[_SGLangNativeCacheState],
    ) -> None:
        if cache_state is None or cache_state.released:
            return
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator.restore_state(cache_state.allocator_state_before_alloc)
        self.model_runner.req_to_token_pool.free(cache_state.req)
        cache_state.released = True

    def _release_native_block_cache_state(
        self,
        block_cache: Optional[_SGLangNativeBlockCacheState],
    ) -> None:
        if block_cache is None or block_cache.released:
            return
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator.restore_state(block_cache.allocator_state_before_alloc)
        self.model_runner.req_to_token_pool.free(block_cache.req)
        block_cache.released = True

    def _release_active_native_caches(self) -> None:
        while self._active_native_block_caches:
            self._release_native_block_cache_state(
                self._active_native_block_caches.pop()
            )
        while self._active_native_caches:
            self._release_native_cache_state(self._active_native_caches.pop())

    def _configure_trace(self, config: FastDllmV2RunnerConfig) -> None:
        trace_path = config.proposal_kwargs.get("trace_path")
        if not trace_path:
            return
        trace_path = str(trace_path)
        if self._trace is None or self._trace_path != trace_path:
            self._trace = _FastDllmV2NativeTraceRecorder(
                trace_path,
                max_events=int(config.proposal_kwargs.get("trace_max_events", 128)),
                full_tensors=bool(
                    config.proposal_kwargs.get("trace_full_tensors", False)
                ),
                topk=int(config.proposal_kwargs.get("trace_topk", 8)),
            )
            self._trace_path = trace_path

        model = getattr(self.model_runner, "model", None)
        self._trace.install_model_hooks(model)
        model_config = getattr(self.model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        self._trace.set_metadata(
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            block_size=config.block_size,
            small_block_size=config.small_block_size,
            threshold=config.threshold,
            model_type=getattr(hf_config, "model_type", None),
            architectures=getattr(hf_config, "architectures", None),
            num_hidden_layers=getattr(hf_config, "num_hidden_layers", None),
            num_attention_heads=getattr(hf_config, "num_attention_heads", None),
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", None),
            hidden_size=getattr(hf_config, "hidden_size", None),
            rope_theta=getattr(hf_config, "rope_theta", None),
            rope_scaling=getattr(hf_config, "rope_scaling", None),
        )

    def _trace_set_phase(self, phase: str) -> None:
        if self._trace is not None:
            self._trace.set_phase(phase)

    def _trace_record(self, name: str, kind: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.record(name, kind, payload)

    def sample_with_top_p(
        self,
        profile: Optional[dict[str, Any]],
        logits: torch.Tensor,
        *,
        top_p: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if profile is not None:
            start = time.perf_counter()
        probs = _fast_dllm_v2_sample_probs(
            logits,
            top_p=top_p,
            temperature=temperature,
        )
        token_ids = probs.argmax(dim=-1)
        if profile is not None:
            profile["sampling_calls"] += 1
            profile["sampling_time_s"] += time.perf_counter() - start
        return token_ids, probs

    def proposal_pad_token_id(self) -> int:
        model_config = getattr(self.model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        token_id = getattr(hf_config, "eos_token_id", None)
        if token_id is None:
            token_id = getattr(hf_config, "pad_token_id", None)
        if isinstance(token_id, (list, tuple)):
            token_id = token_id[0] if token_id else None
        if token_id is None:
            token_id = FAST_DLLM_V2_STOP_TOKEN
        return int(token_id)

    def release(
        self,
        config: FastDllmV2RunnerConfig,
        request_ids: list[str],
        states: dict[str, FastDllmV2RequestState],
    ) -> None:
        return None


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


class FastDllmV2ProposalBackend(Protocol):
    """Model backend used by the backend-neutral Fast_dLLM_v2 proposal loop."""

    @property
    def device(self) -> torch.device:
        ...

    def validate_proposal_config(self, config: FastDllmV2RunnerConfig) -> None:
        ...

    def forward(
        self,
        profile: Optional[dict[str, Any]],
        phase: str,
        **kwargs,
    ) -> Any:
        ...

    def sample_with_top_p(
        self,
        profile: Optional[dict[str, Any]],
        logits: torch.Tensor,
        *,
        top_p: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def proposal_pad_token_id(self) -> int:
        ...


class FastDllmV2BlockProposalEngine:
    """Backend-neutral implementation of Fast_dLLM_v2 block proposals.

    This class owns the sampling/refinement algorithm. Backends provide only
    model execution, sampling, device placement, and padding-token resolution.
    Keeping the control flow here prevents the native SGLang path from drifting
    away from the validated Transformers reference path.
    """

    def __init__(self, backend: FastDllmV2ProposalBackend):
        self.backend = backend

    @staticmethod
    def new_profile() -> dict[str, Any]:
        return {
            "proposal_wall_time_s": 0.0,
            "forward_time_s": 0.0,
            "sampling_time_s": 0.0,
            "forward_calls": 0,
            "prefix_forward_calls": 0,
            "block_commit_forward_calls": 0,
            "refine_full_block_forward_calls": 0,
            "block_cache_refresh_forward_calls": 0,
            "block_cache_reuse_forward_calls": 0,
            "sampling_calls": 0,
            "blocks_started": 0,
            "small_blocks_visited": 0,
            "draft_model_invocations": 0,
            "lookahead_hit": 0,
            "lookahead_tokens_before": 0,
            "lookahead_tokens_after": 0,
            "lookahead_generated_tokens": 0,
        }

    @staticmethod
    def aggregate_profiles(
        profiles: list[Optional[dict[str, Any]]],
    ) -> dict[str, Any]:
        totals: dict[str, Any] = {}
        for profile in profiles:
            if not profile:
                continue
            for key, value in profile.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
        if not totals:
            return {}

        totals["requests_profiled"] = sum(1 for profile in profiles if profile)
        if totals.get("draft_model_invocations"):
            totals["forward_calls_per_invocation"] = (
                totals.get("forward_calls", 0) / totals["draft_model_invocations"]
            )
            totals["proposal_wall_time_per_invocation_s"] = (
                totals.get("proposal_wall_time_s", 0.0)
                / totals["draft_model_invocations"]
            )
        return totals

    @staticmethod
    def initialize_proposal_block(
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
    def propose_one(
        self,
        config: FastDllmV2RunnerConfig,
        input_ids: list[int],
        profile: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Run Fast_dLLM_v2 sampling until the next proposal window is ready."""

        self.backend.validate_proposal_config(config)

        device = self.backend.device
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
        num_blocks = (
            max_new_tokens // block_size + int(seq_len.max().item()) // block_size
        )

        if min_len > block_size:
            prefix_len = min_len // block_size * block_size
            output = self.backend.forward(
                profile,
                "prefix",
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
                ready = self.first_contiguous_proposal(
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
            if profile is not None:
                profile["blocks_started"] += 1
            x_init, input_tensor = self.initialize_proposal_block(
                input_tensor,
                seq_block_idx,
                block_idx,
                block_size,
                mask_id,
            )

            x_t = x_init.clone()
            block_past_key_values = None

            while True:
                ready = self.first_contiguous_proposal(
                    x_t,
                    original_prompt_len,
                    proposed_token_num,
                    mask_id,
                )
                if ready is not None:
                    return ready

                mask_idx = x_t[:, -block_size:] == mask_id
                if mask_idx.sum() == 0:
                    output = self.backend.forward(
                        profile,
                        "block_commit",
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
                    if profile is not None:
                        profile["small_blocks_visited"] += 1
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size
                    start = -block_size + small_block_start_idx
                    end = (
                        None
                        if block_size == small_block_end_idx
                        else -block_size + small_block_end_idx
                    )

                    while True:
                        ready = self.first_contiguous_proposal(
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
                                output = self.backend.forward(
                                    profile,
                                    "block_cache_refresh",
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                )
                                logits = output.logits
                                block_past_key_values = output.block_past_key_values
                                logits = torch.cat(
                                    [logits[:, :1, :], logits[:, :-1, :]], dim=1
                                )
                                logits = logits[:, start:end]
                            else:
                                logits = self.backend.forward(
                                    profile,
                                    "block_cache_reuse",
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                ).logits
                                logits = torch.cat(
                                    [logits[:, :1, :], logits[:, :-1, :]], dim=1
                                )
                        else:
                            logits = self.backend.forward(
                                profile,
                                "refine_full_block",
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                            ).logits
                            logits = torch.cat(
                                [logits[:, :1, :], logits[:, :-1, :]], dim=1
                            )
                            logits = logits[:, start:end]

                        x_1, p_1t = self.backend.sample_with_top_p(
                            profile,
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
                            ready = self.first_contiguous_proposal(
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

        ready = self.first_contiguous_proposal(
            input_tensor,
            original_prompt_len,
            proposed_token_num,
            mask_id,
        )
        if ready is not None:
            return ready
        return self.pad_proposal(input_tensor, original_prompt_len, config, mask_id)

    @staticmethod
    def first_contiguous_proposal(
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

    def pad_proposal(
        self,
        token_ids: torch.Tensor,
        prompt_len: int,
        config: FastDllmV2RunnerConfig,
        mask_id: int,
    ) -> torch.Tensor:
        device = token_ids.device
        candidate = token_ids[0, prompt_len : prompt_len + config.proposed_token_num]
        candidate = candidate[candidate != mask_id]
        if candidate.numel() >= config.proposed_token_num:
            return candidate[: config.proposed_token_num].clone()

        padding = torch.full(
            (config.proposed_token_num - candidate.numel(),),
            self.backend.proposal_pad_token_id(),
            dtype=torch.long,
            device=device,
        )
        return torch.cat([candidate.clone(), padding], dim=0)


def _fast_dllm_v2_propose_from_state(
    runtime: Any,
    config: FastDllmV2RunnerConfig,
    state: FastDllmV2RequestState,
) -> torch.Tensor:
    profile = (
        FastDllmV2BlockProposalEngine.new_profile()
        if config.proposal_kwargs.get("profile")
        else None
    )
    if profile is None:
        state.metadata.pop("last_profile", None)
    profile_start = time.perf_counter() if profile is not None else 0.0
    proposed_token_num = config.proposed_token_num
    lookahead = state.draft_lookahead
    if profile is not None:
        profile["lookahead_tokens_before"] = len(lookahead)

    if len(lookahead) < proposed_token_num:
        seed_input_ids = state.input_ids + lookahead
        generated = runtime._propose_one(config, seed_input_ids, profile).tolist()
        lookahead.extend(int(token_id) for token_id in generated)
        if profile is not None:
            profile["draft_model_invocations"] += 1
            profile["lookahead_generated_tokens"] += len(generated)
    elif profile is not None:
        profile["lookahead_hit"] += 1

    if profile is not None:
        profile["lookahead_tokens_after"] = len(lookahead)
        profile["proposal_wall_time_s"] = time.perf_counter() - profile_start
        state.metadata["last_profile"] = profile

    return torch.tensor(
        lookahead[:proposed_token_num],
        dtype=torch.long,
        device=runtime.device,
    )


def _fast_dllm_v2_sample_probs(
    logits: torch.Tensor,
    *,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    filtered_sorted_probs = sorted_probs.masked_fill(sorted_remove, 0.0)
    filtered_probs = torch.zeros_like(probs).scatter(
        -1,
        sorted_indices,
        filtered_sorted_probs,
    )
    normalizer = filtered_probs.sum(dim=-1, keepdim=True)
    return torch.where(normalizer > 0, filtered_probs / normalizer, probs)


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

        profiles = [
            states[request_id].metadata.get("last_profile")
            for request_id in request.request_ids
        ]
        profile_total = self._aggregate_profiles(profiles)
        proposed_token_ids = torch.stack(proposed).to(request.current_token_ids.device)
        metadata = {
            "runner": "fast_dllm_v2",
            "runtime": "transformers",
            "block_size": config.block_size,
            "small_block_size": config.small_block_size,
            "threshold": config.threshold,
            "use_block_cache": bool(
                config.proposal_kwargs.get("use_block_cache", False)
            ),
            "profile_enabled": bool(config.proposal_kwargs.get("profile", False)),
            **getattr(self, "model_metadata", {}),
        }
        if profile_total:
            metadata["profile_total"] = profile_total
            metadata["profile_log_interval"] = int(
                config.proposal_kwargs.get("profile_log_interval", 1)
            )

        return IndependentDllmDraftTokens(
            request_ids=list(request.request_ids),
            current_token_ids=request.current_token_ids,
            proposed_token_ids=proposed_token_ids,
            prefix_lens=request.prefix_lens,
            metadata=metadata,
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
        return _fast_dllm_v2_propose_from_state(self, config, state)

    @property
    def device(self) -> torch.device:
        assert self.model is not None
        return self.model.device

    def validate_proposal_config(self, config: FastDllmV2RunnerConfig) -> None:
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

    def forward(
        self,
        profile: Optional[dict[str, Any]],
        phase: str,
        **kwargs,
    ) -> Any:
        return self._profiled_forward(profile, phase, **kwargs)

    def sample_with_top_p(
        self,
        profile: Optional[dict[str, Any]],
        logits: torch.Tensor,
        *,
        top_p: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._profiled_sample_with_top_p(
            profile,
            logits,
            top_p=top_p,
            temperature=temperature,
        )

    def proposal_pad_token_id(self) -> int:
        assert self.tokenizer is not None
        pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                "Fast_dLLM_v2 proposal produced too few concrete tokens and "
                "no EOS/PAD token is available for padding."
            )
        return int(pad_id)

    @staticmethod
    def _new_profile() -> dict[str, Any]:
        return FastDllmV2BlockProposalEngine.new_profile()

    def _profiled_forward(
        self,
        profile: Optional[dict[str, Any]],
        phase: str,
        **kwargs,
    ) -> Any:
        if profile is None:
            return self.model.forward(**kwargs)

        start = time.perf_counter()
        output = self.model.forward(**kwargs)
        elapsed = time.perf_counter() - start
        phase_calls_key = f"{phase}_forward_calls"
        phase_time_key = f"{phase}_forward_time_s"
        profile["forward_calls"] += 1
        profile["forward_time_s"] += elapsed
        profile[phase_calls_key] = profile.get(phase_calls_key, 0) + 1
        profile[phase_time_key] = profile.get(phase_time_key, 0.0) + elapsed
        return output

    def _profiled_sample_with_top_p(
        self,
        profile: Optional[dict[str, Any]],
        logits: torch.Tensor,
        *,
        top_p: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if profile is None:
            return self.model.sample_with_top_p(
                logits,
                top_p=top_p,
                temperature=temperature,
            )

        start = time.perf_counter()
        output = self.model.sample_with_top_p(
            logits,
            top_p=top_p,
            temperature=temperature,
        )
        profile["sampling_calls"] += 1
        profile["sampling_time_s"] += time.perf_counter() - start
        return output

    @staticmethod
    def _aggregate_profiles(profiles: list[Optional[dict[str, Any]]]) -> dict[str, Any]:
        return FastDllmV2BlockProposalEngine.aggregate_profiles(profiles)

    @staticmethod
    def _initialize_proposal_block(
        input_tensor: torch.Tensor,
        seq_block_idx: torch.Tensor,
        block_idx: int,
        block_size: int,
        mask_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return FastDllmV2BlockProposalEngine.initialize_proposal_block(
            input_tensor,
            seq_block_idx,
            block_idx,
            block_size,
            mask_id,
        )

    @torch.no_grad()
    def _propose_one(
        self,
        config: FastDllmV2RunnerConfig,
        input_ids: list[int],
        profile: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        assert self.model is not None
        assert self.tokenizer is not None
        return FastDllmV2BlockProposalEngine(self).propose_one(
            config,
            input_ids,
            profile,
        )

    def _validate_proposal_config(self, config: FastDllmV2RunnerConfig) -> None:
        self.validate_proposal_config(config)

    def _first_contiguous_proposal(
        self,
        token_ids: torch.Tensor,
        prompt_len: int,
        proposed_token_num: int,
        mask_id: int,
    ) -> Optional[torch.Tensor]:
        return FastDllmV2BlockProposalEngine.first_contiguous_proposal(
            token_ids,
            prompt_len,
            proposed_token_num,
            mask_id,
        )

    def _pad_proposal(
        self,
        token_ids: torch.Tensor,
        prompt_len: int,
        config: FastDllmV2RunnerConfig,
        mask_id: int,
    ) -> torch.Tensor:
        return FastDllmV2BlockProposalEngine(self).pad_proposal(
            token_ids,
            prompt_len,
            config,
            mask_id,
        )


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


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    parsed = str(value)
    if parsed.lower() in ("", "none", "null"):
        return None
    return int(value)


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
