import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Perlmutter smoke matrix:
# keep the launch topology fixed and vary only request shape and backend.
COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
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
    "--disable-cuda-graph",
    "--disable-piecewise-cuda-graph",
    "--log-requests",
    "--log-requests-level",
    "3",
    "--log-requests-format",
    "json",
]


class _DraftForwardSpecBase(CustomTestCase):
    model = DEFAULT_TARGET_MODEL_TLI
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.69
    spec_decode_threshold = 2.5

    draft_model = None
    streaming_matrix_cases = (
        {
            "name": "pairwise-burst",
            "prompts": (
                "The capital city of France is",
                "The tallest mountain on Earth is",
            ),
            "max_new_tokens": 8,
        },
        {
            "name": "quartet-burst",
            "prompts": (
                "Explain photosynthesis in one sentence.",
                "Name a large planet in our solar system.",
                "The Pacific Ocean is the",
                "In mathematics, the derivative measures",
            ),
            "max_new_tokens": 8,
        },
    )

    @classmethod
    def get_server_args(cls):
        return COMMON_SERVER_ARGS + ["--remote-draft-tokenizer-path", cls.draft_model]

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
        self._flush_cache()
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

        server_info = self._server_info()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)

    def test_gsm8k(self):
        self._run_gsm8k()

    def _flush_cache(self):
        response = requests.get(self.base_url + "/flush_cache", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)

    def _server_info(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _post_generate(self, prompt: str, max_new_tokens: int):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        meta_info = payload["meta_info"]

        self.assertTrue(payload["text"])
        self.assertEqual(len(payload["output_ids"]), max_new_tokens)
        self.assertEqual(meta_info["completion_tokens"], max_new_tokens)
        self.assertGreater(meta_info["prompt_tokens"], 0)
        self.assertEqual(meta_info["finish_reason"]["type"], "length")
        self.assertIn("spec_verify_ct", meta_info)
        self.assertIn("spec_accepted_drafts", meta_info)
        self.assertIn("spec_proposed_drafts", meta_info)
        self.assertIn("spec_accept_rate", meta_info)
        self.assertGreater(meta_info["spec_verify_ct"], 0)
        self.assertGreaterEqual(meta_info["spec_accepted_drafts"], 0)
        self.assertGreaterEqual(
            meta_info["spec_proposed_drafts"], meta_info["spec_accepted_drafts"]
        )
        self.assertGreaterEqual(meta_info["spec_accept_rate"], 0.0)
        self.assertLessEqual(meta_info["spec_accept_rate"], 1.0)
        return payload

    def test_streaming_transport_matrix(self):
        self._flush_cache()

        for case in self.streaming_matrix_cases:
            with self.subTest(case=case["name"]):
                self._flush_cache()

                futures_by_index = {}
                completion_order = []
                with ThreadPoolExecutor(max_workers=len(case["prompts"])) as executor:
                    for index, prompt in enumerate(case["prompts"]):
                        future = executor.submit(
                            self._post_generate,
                            prompt,
                            case["max_new_tokens"],
                        )
                        futures_by_index[future] = index

                    for future in as_completed(futures_by_index):
                        completion_order.append(futures_by_index[future])
                        future.result()

                self.assertEqual(
                    sorted(completion_order),
                    list(range(len(case["prompts"]))),
                )

                server_info = self._server_info()
                avg_spec_accept_length = server_info["internal_states"][0][
                    "avg_spec_accept_length"
                ]
                self.assertGreater(avg_spec_accept_length, 0.0)


class TestDraftForwardSameTokenizerTriton(_DraftForwardSpecBase):
    draft_model = DEFAULT_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "triton"]


class TestDraftForwardSameTokenizerFlashinfer(_DraftForwardSpecBase):
    draft_model = DEFAULT_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "flashinfer"]


class TestDraftForwardCrossFamilyTriton(_DraftForwardSpecBase):
    draft_model = DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "triton"]


class TestDraftForwardCrossFamilyFlashinfer(_DraftForwardSpecBase):
    draft_model = DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI

    @classmethod
    def get_server_args(cls):
        return super().get_server_args() + ["--attention-backend", "flashinfer"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
