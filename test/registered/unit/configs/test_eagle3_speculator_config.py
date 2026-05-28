"""Unit tests for Eagle3 speculator config loading."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import LlamaConfig

from sglang.srt.utils.hf_transformers.config import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestEagle3SpeculatorConfigLoading(CustomTestCase):
    def _write_config(self, config_dir: Path, architectures: str) -> str:
        config_json = {
            "architectures": [architectures],
            "auto_map": {
                "AutoConfig": "remote.Eagle3SpeculatorConfig",
                "AutoModel": "remote.Eagle3SpeculatorForCausalLM",
            },
            "draft_vocab_size": 1234,
            "has_no_defaults_at_init": True,
            "norm_before_residual": True,
            "speculators_config": {
                "hidden_size": 16,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": False,
                "target_hidden_size": None,
                "vocab_size": 32000,
            },
            "speculators_model_type": "qwen3",
            "speculators_version": "1",
            "target_hidden_size": 32,
            "torch_dtype": "bfloat16",
            "transformer_layer_config": {"dummy": 1},
            "transformers_version": "4.0.0",
        }
        (config_dir / "config.json").write_text(json.dumps(config_json))
        return str(config_dir)

    @patch("sglang.srt.utils.hf_transformers.config.AutoConfig.from_pretrained")
    def test_eagle3_speculator_uses_native_llama_config(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = ValueError("Unrecognized model")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = self._write_config(Path(tmp_dir), "Eagle3Speculator")
            config = get_config(model_path, trust_remote_code=False)

        self.assertIsInstance(config, LlamaConfig)
        self.assertEqual(config.architectures, ["LlamaForCausalLMEagle3"])
        self.assertEqual(config.model_type, "llama")
        self.assertEqual(config.num_hidden_layers, 1)
        self.assertEqual(config.vocab_size, 1234)
        self.assertEqual(config.draft_vocab_size, 32000)
        self.assertEqual(config.target_hidden_size, 32)
        self.assertTrue(config.norm_before_residual)
        self.assertEqual(config.speculators_config["hidden_size"], 16)
        self.assertEqual(config.speculators_config["num_hidden_layers"], 1)

    @patch("sglang.srt.utils.hf_transformers.config.AutoConfig.from_pretrained")
    def test_non_eagle3_value_error_still_raises(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = ValueError("Unrecognized model")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = self._write_config(Path(tmp_dir), "SomeOtherCustom")
            with self.assertRaises(ValueError):
                get_config(model_path, trust_remote_code=False)


if __name__ == "__main__":
    unittest.main()
