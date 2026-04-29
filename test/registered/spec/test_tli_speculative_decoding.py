import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI,
    DEFAULT_DRAFT_MODEL_TLI,
    DEFAULT_TARGET_MODEL_TLI,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


register_cuda_ci(est_time=600, suite="stage-b-test-1-gpu-large")

GSM_DATASET_PATH = None


COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "TLI",
    "--speculative-num-steps",
    "4",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--mem-fraction-static",
    "0.7",
]


class _TliSpecBase(CustomTestCase):
    model = DEFAULT_TARGET_MODEL_TLI
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.69
    spec_decode_threshold = 2.5

    draft_model = None

    @classmethod
    def get_server_args(cls):
        return COMMON_SERVER_ARGS + ["--speculative-draft-model-path", cls.draft_model]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)

    def test_gsm8k(self):
        self._run_gsm8k()


class TestTLISameTokenizerTriton(_TliSpecBase):
    draft_model = DEFAULT_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "triton"]


class TestTLISameTokenizerFlashinfer(_TliSpecBase):
    draft_model = DEFAULT_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "flashinfer"]


class TestTLICrossFamilyTriton(_TliSpecBase):
    draft_model = DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "triton"]


class TestTLICrossFamilyFlashinfer(_TliSpecBase):
    draft_model = DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "flashinfer"]


if __name__ == "__main__":
    unittest.main(verbosity=2)