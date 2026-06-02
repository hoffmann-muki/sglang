"""Unit tests for CUDA graph replay gating."""

from types import SimpleNamespace
import unittest

import torch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_runner(spec_algorithm: SpeculativeAlgorithm) -> CudaGraphRunner:
    runner = CudaGraphRunner.__new__(CudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=spec_algorithm,
        is_draft_worker=False,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_encoder_decoder=False, hf_config=SimpleNamespace()),
    )
    runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = False
    runner.enable_two_batch_overlap = False
    runner.disable_padding = False
    runner.max_bs = 8
    runner.graphs = {}
    runner.is_encoder_decoder = False
    runner.record_nolora_graph = False
    runner.enable_pdmux = False
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner.num_tokens_per_bs = 1
    runner.is_dllm = False
    return runner


def _make_batch(mode: ForwardMode):
    return SimpleNamespace(
        replace_embeds=None,
        batch_size=1,
        global_num_tokens_cpu=torch.tensor([1]),
        forward_mode=mode,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        can_run_tbo=True,
        can_run_dp_cuda_graph=True,
        input_ids=torch.zeros(1, dtype=torch.int64),
        encoder_lens=torch.ones(1, dtype=torch.int64),
    )


class TestCudaGraphRunnerStandalone(unittest.TestCase):
    def test_standalone_can_run_cuda_graph_replay(self):
        runner = _make_runner(SpeculativeAlgorithm.STANDALONE)

        self.assertTrue(runner.can_run(_make_batch(ForwardMode.TARGET_VERIFY)))
        self.assertTrue(runner.can_run(_make_batch(ForwardMode.DECODE)))


class TestCudaGraphRunnerTLI(unittest.TestCase):
    def test_tli_target_verify_can_run_cuda_graph_replay(self):
        runner = _make_runner(SpeculativeAlgorithm.TLI)
        runner.capture_hidden_mode = CaptureHiddenMode.FULL
        batch = _make_batch(ForwardMode.TARGET_VERIFY)
        batch.spec_info = SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL)

        self.assertTrue(runner.can_run(batch))

    def test_tli_target_verify_requires_matching_hidden_capture_mode(self):
        runner = _make_runner(SpeculativeAlgorithm.TLI)
        batch = _make_batch(ForwardMode.TARGET_VERIFY)
        batch.spec_info = SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL)

        self.assertFalse(runner.can_run(batch))


if __name__ == "__main__":
    unittest.main()
