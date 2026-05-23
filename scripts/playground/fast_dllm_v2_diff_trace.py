"""Differential trace helper for Fast_dLLM_v2 Transformers compatibility.

Run this script once in the known-good Fast-dLLM environment and once in the
SGLang environment, then compare the two trace files. The trace intentionally
uses only Hugging Face/torch APIs plus optional SGLang compatibility shims so it
can be executed from both environments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = "Write a short Python function that returns the factorial of n."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace Fast_dLLM_v2 internals in two Transformers environments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace_parser = subparsers.add_parser("trace", help="write a trace artifact")
    trace_parser.add_argument("--model-path", required=True)
    trace_parser.add_argument("--out", required=True)
    trace_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    trace_parser.add_argument("--device-map", default="cuda:0")
    trace_parser.add_argument("--torch-dtype", default="auto")
    trace_parser.add_argument("--max-new-tokens", type=int, default=16)
    trace_parser.add_argument("--block-size", type=int, default=32)
    trace_parser.add_argument("--small-block-size", type=int, default=8)
    trace_parser.add_argument("--threshold", type=float, default=0.9)
    trace_parser.add_argument("--apply-sglang-compat", action="store_true")
    trace_parser.add_argument("--disable-hub-kernels", action="store_true")
    trace_parser.add_argument("--capture-calls", type=int, default=3)
    trace_parser.add_argument("--trust-remote-code", action="store_true", default=True)

    compare_parser = subparsers.add_parser("compare", help="compare two trace files")
    compare_parser.add_argument("--good", required=True)
    compare_parser.add_argument("--test", required=True)
    compare_parser.add_argument("--top", type=int, default=40)

    args = parser.parse_args()
    if args.command == "trace":
        write_trace(args)
    else:
        compare_traces(args)


def write_trace(args: argparse.Namespace) -> None:
    if args.disable_hub_kernels:
        os.environ["USE_HUB_KERNELS"] = "0"
    if args.apply_sglang_compat:
        apply_sglang_fast_dllm_v2_compat()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)

    tracer = FastDllmTrace(model, capture_calls=args.capture_calls)
    tracer.install_hooks()
    tracer.install_sdpa_wrapper()

    trace: dict[str, Any] = {
        "metadata": collect_metadata(args, model, tokenizer, text, inputs["input_ids"]),
        "forward": {},
        "generate": {},
    }

    with torch.no_grad():
        tracer.set_phase("forward")
        forward_out = model(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        trace["forward"] = {
            "events": tracer.pop_events("forward"),
            "logits_last": tensor_to_cpu(forward_out.logits[:, -1, :]),
            "logits_last_topk": topk_summary(forward_out.logits[:, -1, :], k=20),
            "hidden_states": summarize_hidden_states(
                getattr(forward_out, "hidden_states", None)
            ),
            "past_key_values": summarize_cache(
                getattr(forward_out, "past_key_values", None)
            ),
            "block_cache": summarize_cache(getattr(forward_out, "block_cache", None)),
        }

        tracer.set_phase("generate")
        output_ids = model.generate(
            inputs["input_ids"],
            tokenizer=tokenizer,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        trace["generate"] = {
            "events": tracer.pop_events("generate"),
            "output_ids": tensor_to_cpu(output_ids),
            "new_token_ids": tensor_to_cpu(new_ids),
            "new_text": tokenizer.decode(new_ids.tolist(), skip_special_tokens=False),
        }

    tracer.remove()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, out_path)
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(json_safe_summary(trace), indent=2) + "\n")
    print(f"wrote trace: {out_path}")
    print(f"wrote summary: {summary_path}")
    print("generated text:", trace["generate"]["new_text"])


def apply_sglang_fast_dllm_v2_compat() -> None:
    from sglang.srt.speculative.co_draft.fast_dllm_v2_runner import (
        _disable_transformers_hub_kernels_for_fast_dllm_v2,
        _ensure_transformers_default_rope_support,
        _ensure_transformers_legacy_dynamic_cache_support,
        _ensure_transformers_tied_weights_support,
        _ensure_transformers_v453_sdpa_support,
    )

    _ensure_transformers_default_rope_support()
    _ensure_transformers_tied_weights_support()
    _ensure_transformers_legacy_dynamic_cache_support()
    _ensure_transformers_v453_sdpa_support()
    _disable_transformers_hub_kernels_for_fast_dllm_v2()


class FastDllmTrace:
    def __init__(self, model: torch.nn.Module, capture_calls: int) -> None:
        self.model = model
        self.capture_calls = capture_calls
        self.phase = "uninitialized"
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.handles: list[Any] = []
        self._counts: dict[str, int] = {}
        self._original_sdpa = None

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.events.setdefault(phase, [])
        self._counts.clear()

    def pop_events(self, phase: str) -> list[dict[str, Any]]:
        return self.events.get(phase, [])

    def install_hooks(self) -> None:
        self._hook_module("embed_tokens", self.model.get_input_embeddings())
        core_model = getattr(self.model, "model", None)
        if core_model is not None and hasattr(core_model, "rotary_emb"):
            self._hook_module("rotary_emb", core_model.rotary_emb)
        layers = getattr(core_model, "layers", None)
        if layers is not None and len(layers) > 0:
            layer0 = layers[0]
            self._hook_module("layer0", layer0)
            if hasattr(layer0, "input_layernorm"):
                self._hook_module("layer0.input_layernorm", layer0.input_layernorm)
            if hasattr(layer0, "self_attn"):
                self._hook_module("layer0.self_attn", layer0.self_attn)
            if hasattr(layer0, "post_attention_layernorm"):
                self._hook_module(
                    "layer0.post_attention_layernorm",
                    layer0.post_attention_layernorm,
                )
            if hasattr(layer0, "mlp"):
                self._hook_module("layer0.mlp", layer0.mlp)
        if hasattr(self.model, "lm_head"):
            self._hook_module("lm_head", self.model.lm_head)

    def install_sdpa_wrapper(self) -> None:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except Exception:
            return

        self._original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def traced_sdpa(*args, **kwargs):
            name = "sdpa"
            should_capture = self._should_capture(name, "call")
            if should_capture:
                self._record(
                    name,
                    "input",
                    {
                        "args": summarize_object(args),
                        "kwargs": summarize_object(kwargs),
                        "query": find_tensor_by_index(args, 1),
                        "key": find_tensor_by_index(args, 2),
                        "value": find_tensor_by_index(args, 3),
                        "attention_mask": find_tensor_by_index(args, 4),
                    },
                )
            out = self._original_sdpa(*args, **kwargs)
            if should_capture:
                self._record(name, "output", {"output": summarize_object(out)})
            return out

        try:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = traced_sdpa
        except TypeError:
            ALL_ATTENTION_FUNCTIONS.register("sdpa", traced_sdpa)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._original_sdpa is None:
            return
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            try:
                ALL_ATTENTION_FUNCTIONS["sdpa"] = self._original_sdpa
            except TypeError:
                ALL_ATTENTION_FUNCTIONS.register("sdpa", self._original_sdpa)
        except Exception:
            pass

    def _hook_module(self, name: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return

        def pre_hook(_module, args, kwargs):
            if self._should_capture(name, "input"):
                self._record(
                    name,
                    "input",
                    {
                        "args": summarize_object(args),
                        "kwargs": summarize_object(kwargs),
                    },
                )

        def post_hook(_module, args, kwargs, output):
            if self._should_capture(name, "output"):
                self._record(
                    name,
                    "output",
                    {
                        "args": summarize_object(args),
                        "kwargs": summarize_object(kwargs),
                        "output": summarize_object(output),
                    },
                )

        self.handles.append(
            module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )
        self.handles.append(
            module.register_forward_hook(post_hook, with_kwargs=True)
        )

    def _should_capture(self, name: str, kind: str) -> bool:
        key = f"{self.phase}:{name}:{kind}"
        count = self._counts.get(key, 0)
        self._counts[key] = count + 1
        return count < self.capture_calls

    def _record(self, name: str, kind: str, payload: dict[str, Any]) -> None:
        self.events.setdefault(self.phase, []).append(
            {
                "name": name,
                "kind": kind,
                "payload": payload,
            }
        )


def collect_metadata(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    import transformers

    config = getattr(model, "config", None)
    return {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_type": getattr(config, "model_type", None),
        "model_path": args.model_path,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "apply_sglang_compat": args.apply_sglang_compat,
        "disable_hub_kernels": args.disable_hub_kernels,
        "use_hub_kernels_env": os.environ.get("USE_HUB_KERNELS"),
        "prompt": args.prompt,
        "chat_text": text,
        "input_ids": input_ids[0].detach().cpu().tolist(),
        "input_text": tokenizer.decode(input_ids[0].detach().cpu().tolist()),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "rope_parameters": getattr(config, "rope_parameters", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
    }


def summarize_object(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_summary(value)
    if isinstance(value, dict):
        return {str(k): summarize_object(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_object(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return summarize_cache(value)
    return repr(value)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    cpu = detached.float().cpu()
    flat = cpu.reshape(-1)
    head = flat[: min(8, flat.numel())].tolist()
    return {
        "type": "tensor_summary",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "mean": float(cpu.mean()) if flat.numel() else 0.0,
        "std": float(cpu.std(unbiased=False)) if flat.numel() else 0.0,
        "min": float(cpu.min()) if flat.numel() else 0.0,
        "max": float(cpu.max()) if flat.numel() else 0.0,
        "head": head,
        "tensor": cpu,
    }


def tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def find_tensor_by_index(args: tuple[Any, ...], index: int) -> Any:
    if len(args) <= index:
        return None
    value = args[index]
    if torch.is_tensor(value):
        return tensor_summary(value)
    return summarize_object(value)


def topk_summary(logits: torch.Tensor, k: int) -> dict[str, Any]:
    values, indices = torch.topk(logits.detach().float().cpu(), k=k, dim=-1)
    return {
        "indices": indices[0].tolist(),
        "values": values[0].tolist(),
    }


def summarize_hidden_states(hidden_states: Any) -> Any:
    if hidden_states is None:
        return None
    return [tensor_summary(tensor) for tensor in hidden_states[:3]]


def summarize_cache(cache: Any) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return {
            "type": type(cache).__name__,
            "key_cache": [tensor_summary(tensor) for tensor in cache.key_cache[:2]],
            "value_cache": [tensor_summary(tensor) for tensor in cache.value_cache[:2]],
            "seen_tokens": getattr(cache, "seen_tokens", None),
        }
    if isinstance(cache, (tuple, list)):
        summary = []
        for item in cache[:2]:
            summary.append(summarize_object(item))
        return {
            "type": type(cache).__name__,
            "len": len(cache),
            "head": summary,
        }
    return repr(cache)


def json_safe_summary(trace: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if torch.is_tensor(value):
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items() if k != "tensor"}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        return value

    return convert(trace)


def compare_traces(args: argparse.Namespace) -> None:
    good = torch.load(args.good, map_location="cpu")
    test = torch.load(args.test, map_location="cpu")

    print("good transformers:", good["metadata"].get("transformers_version"))
    print("test transformers:", test["metadata"].get("transformers_version"))
    print("good generated:", good["generate"].get("new_text"))
    print("test generated:", test["generate"].get("new_text"))
    print()

    compare_logits(good, test)
    compare_event_tensors(
        good["forward"].get("events", []),
        test["forward"].get("events", []),
        "forward",
        top=args.top,
    )
    compare_event_tensors(
        good["generate"].get("events", []),
        test["generate"].get("events", []),
        "generate",
        top=args.top,
    )


def compare_logits(good: dict[str, Any], test: dict[str, Any]) -> None:
    good_logits = good["forward"].get("logits_last")
    test_logits = test["forward"].get("logits_last")
    if not torch.is_tensor(good_logits) or not torch.is_tensor(test_logits):
        print("missing logits_last tensor")
        return
    print("first-step logits:")
    print_metric("forward.logits_last", good_logits, test_logits)
    good_top = good["forward"].get("logits_last_topk", {})
    test_top = test["forward"].get("logits_last_topk", {})
    print("good top ids:", good_top.get("indices", [])[:10])
    print("test top ids:", test_top.get("indices", [])[:10])
    print()


def compare_event_tensors(
    good_events: list[dict[str, Any]],
    test_events: list[dict[str, Any]],
    phase: str,
    top: int,
) -> None:
    good_map = collect_named_tensors(good_events)
    test_map = collect_named_tensors(test_events)
    shared = sorted(set(good_map) & set(test_map))
    rows = []
    for key in shared:
        lhs = good_map[key]
        rhs = test_map[key]
        if lhs.shape != rhs.shape:
            rows.append((float("inf"), key, f"shape {tuple(lhs.shape)} != {tuple(rhs.shape)}"))
            continue
        rows.append((max_abs_diff(lhs, rhs), key, metric_string(lhs, rhs)))

    rows.sort(key=lambda item: item[0], reverse=True)
    print(f"{phase} tensor diffs:")
    for _, key, metric in rows[:top]:
        print(f"{key}: {metric}")
    missing_good = sorted(set(test_map) - set(good_map))
    missing_test = sorted(set(good_map) - set(test_map))
    if missing_good:
        print(f"{phase} only in test:", missing_good[:top])
    if missing_test:
        print(f"{phase} only in good:", missing_test[:top])
    print()


def collect_named_tensors(events: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for event_idx, event in enumerate(events):
        prefix = f"{event_idx:03d}.{event['name']}.{event['kind']}"
        collect_tensors_from_object(prefix, event.get("payload"), out)
    return out


def collect_tensors_from_object(prefix: str, value: Any, out: dict[str, torch.Tensor]) -> None:
    if isinstance(value, dict):
        if value.get("type") == "tensor_summary" and torch.is_tensor(value.get("tensor")):
            out[prefix] = value["tensor"]
            return
        for key, child in value.items():
            if key == "tensor":
                continue
            collect_tensors_from_object(f"{prefix}.{key}", child, out)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_tensors_from_object(f"{prefix}.{index}", child, out)


def print_metric(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    print(f"{name}: {metric_string(lhs, rhs)}")


def metric_string(lhs: torch.Tensor, rhs: torch.Tensor) -> str:
    lhs_f = lhs.float()
    rhs_f = rhs.float()
    diff = (lhs_f - rhs_f).abs()
    denom = torch.maximum(lhs_f.abs(), rhs_f.abs()).clamp_min(1e-12)
    rel = (diff / denom).max().item()
    return (
        f"shape={tuple(lhs.shape)} "
        f"max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g} "
        f"max_rel={rel:.6g}"
    )


def max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


if __name__ == "__main__":
    main()
