# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config loading utilities."""

from pathlib import Path
from typing import Optional

from transformers import LlamaConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, logger, lru_cache_frozenset
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

from ..hf_transformers_patches import _ensure_gguf_version
from .common import (
    _CONFIG_REGISTRY,
    AutoConfig,
    DeepseekVLV2Config,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _load_deepseek_v32_model,
    _override_v_head_dim_if_zero,
    check_gguf_file,
    get_hf_text_config,
)
from .mistral_utils import is_mistral_model, load_mistral_config


def _set_architectures(config, arch_name):
    config.update({"architectures": [arch_name]})


def _is_eagle3_speculator_model(config_dict: dict) -> bool:
    architectures = config_dict.get("architectures") or []
    return bool(architectures) and architectures[0] == "Eagle3Speculator"


def _load_eagle3_speculator_model(model: str, config_dict: dict):
    spec_config = config_dict.get("speculators_config") or {}
    if hasattr(spec_config, "to_dict"):
        spec_config = spec_config.to_dict()
    if not isinstance(spec_config, dict):
        raise RuntimeError(
            f"Expected dict-valued speculators_config for {model}, got {type(spec_config)!r}."
        )

    layer_config = config_dict.get("transformer_layer_config") or {}
    if hasattr(layer_config, "to_dict"):
        layer_config = layer_config.to_dict()
    if not isinstance(layer_config, dict):
        raise RuntimeError(
            f"Expected dict-valued transformer_layer_config for {model}, got {type(layer_config)!r}."
        )

    llama_config_dict = dict(layer_config)
    target_vocab_size = llama_config_dict.get("vocab_size")
    draft_vocab_size = config_dict.get("draft_vocab_size")
    if target_vocab_size is None:
        target_vocab_size = draft_vocab_size
    if draft_vocab_size is None:
        draft_vocab_size = target_vocab_size
    for key in (
        "has_no_defaults_at_init",
        "norm_before_residual",
        "speculators_model_type",
        "speculators_version",
        "target_hidden_size",
        "torch_dtype",
        "transformer_layer_config",
    ):
        value = config_dict.get(key)
        if value is not None:
            llama_config_dict[key] = value

    if config_dict.get("target_hidden_size") is None:
        llama_config_dict.pop("target_hidden_size", None)

    if target_vocab_size is not None:
        llama_config_dict["vocab_size"] = target_vocab_size
    if draft_vocab_size is not None:
        llama_config_dict["draft_vocab_size"] = draft_vocab_size

    llama_config_dict.setdefault("model_type", "llama")
    llama_config_dict["num_hidden_layers"] = 1
    llama_config_dict["architectures"] = ["LlamaForCausalLMEagle3"]

    config = LlamaConfig.from_dict(llama_config_dict)
    config._name_or_path = model
    config.architectures = ["LlamaForCausalLMEagle3"]
    config.model_type = "llama"
    config.speculators_config = spec_config
    config.transformer_layer_config = layer_config
    return config


def _apply_deepseek_ocr_overrides(config, model):
    _override_v_head_dim_if_zero(config)
    _set_architectures(config, "DeepseekOCRForCausalLM")
    config._name_or_path = model


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        _ensure_gguf_version()
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if is_runai_obj_uri(model):
        model = ObjectStorageModel.get_path(model)

    if is_remote_url(model):
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if is_mistral_model(model):
        config = load_mistral_config(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
    else:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except (ValueError, KeyError) as e:
            if "deepseek_v32" in str(e):
                config = _load_deepseek_v32_model(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:
                config_dict, _ = PretrainedConfig.get_config_dict(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
                if _is_eagle3_speculator_model(config_dict):
                    config = _load_eagle3_speculator_model(model, config_dict)
                elif isinstance(e, ValueError):
                    raise
                else:
                    logger.warning(
                        "AutoConfig.from_pretrained raised KeyError for %s: %s. "
                        "Falling back to config registry lookup.",
                        model,
                        e,
                    )
                    model_type = config_dict.get("model_type")
                    if model_type in _CONFIG_REGISTRY:
                        config = _CONFIG_REGISTRY[model_type].from_dict(config_dict)
                        config._name_or_path = model
                    else:
                        raise

    if (
        config.architectures is not None
        and config.architectures[0] == "Phi4MMForCausalLM"
    ):
        from transformers import SiglipVisionConfig

        config.vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=448,
            intermediate_size=4304,
            model_type="siglip_vision_model",
            num_attention_heads=16,
            num_hidden_layers=26,
            patch_size=14,
        )

    if config.architectures in [
        ["LongcatCausalLM"],
        ["LongcatFlashForCausalLM"],
        ["LongcatFlashNgramForCausalLM"],
    ]:
        config.model_type = "longcat_flash"

    text_config = get_hf_text_config(config=config)

    if isinstance(model, str) and text_config is not None:
        items = (
            text_config.items()
            if hasattr(text_config, "items")
            else vars(text_config).items()
        )
        for key, val in items:
            if not hasattr(config, key) and val is not None:
                setattr(config, key, val)

    is_ocr = _is_deepseek_ocr_model(config)
    is_ocr2 = _is_deepseek_ocr2_model(config)

    if is_ocr2:
        _override_v_head_dim_if_zero(config)
        config.model_type = "deepseek-ocr"
        _set_architectures(config, "DeepseekOCRForCausalLM")
        config = DeepseekVLV2Config.from_pretrained(model, revision=revision)
        _apply_deepseek_ocr_overrides(config, model)
    elif config.model_type in _CONFIG_REGISTRY:
        model_type = config.model_type
        if model_type == "deepseek_vl_v2" and is_ocr:
            model_type = "deepseek-ocr"
        config = _CONFIG_REGISTRY[model_type].from_pretrained(model, revision=revision)

        # Re-check after reloading config from registry
        if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
            _apply_deepseek_ocr_overrides(config, model)
        else:
            config._name_or_path = model

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        _set_architectures(config, "MultiModalityCausalLM")

    if config.model_type == "gemma4":
        # Gemma4 configs use base attributes for SWA layers and `global_*`
        # variants for full-attention layers.  SGLang expects the opposite:
        # base = full-attention, `swa_*` = sliding-window overrides.
        text_config = config.text_config
        global_head_dim = getattr(text_config, "global_head_dim", None)
        global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)

        swa_head_dim = text_config.head_dim
        swa_kv_heads = text_config.num_key_value_heads

        text_config.swa_head_dim = swa_head_dim
        text_config.swa_v_head_dim = swa_head_dim
        text_config.swa_num_key_value_heads = swa_kv_heads

        if global_head_dim is not None:
            text_config.head_dim = global_head_dim
        if global_kv_heads is not None:
            text_config.num_key_value_heads = global_kv_heads

        if not hasattr(text_config, "v_head_dim"):
            text_config.v_head_dim = text_config.head_dim
        if not hasattr(text_config, "swa_v_head_dim"):
            text_config.swa_v_head_dim = text_config.swa_head_dim

    if config.model_type == "longcat_flash":
        _set_architectures(config, "LongcatFlashForCausalLM")

    if model_override_args:
        config.update(model_override_args)

    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config
