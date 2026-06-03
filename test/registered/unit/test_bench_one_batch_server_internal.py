import json
import tempfile
import unittest

from sglang.test.bench_one_batch_server_internal import (
    BenchOneCaseResult,
    aggregate_sglang_meta_info,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestBenchOneBatchServerMetrics(unittest.TestCase):
    def test_aggregate_sglang_meta_info_uses_exact_spec_counters(self):
        meta_infos = [
            {
                "completion_tokens": 10,
                "spec_verify_ct": 2,
                "spec_accepted_drafts": 8,
                "spec_proposed_drafts": 10,
                "cached_tokens": 3,
                "cached_tokens_details": {"device": 2, "host": 1},
                "latency_breakdown": {
                    "draft_proposal_time": 0.20,
                    "target_verification_time": 0.40,
                    "grpc_communication_time": 0.0,
                    "other_time": 0.10,
                },
            },
            {
                "completion_tokens": 9,
                "spec_verify_ct": 3,
                "spec_accepted_drafts": 6,
                "spec_proposed_drafts": 15,
                "cached_tokens": 5,
                "cached_tokens_details": {
                    "device": 1,
                    "storage": 4,
                    "storage_backend": "local",
                },
                "latency_breakdown": {
                    "draft_proposal_time": 0.30,
                    "target_verification_time": 0.50,
                    "grpc_communication_time": 0.02,
                    "other_time": 0.20,
                },
            },
        ]

        aggregated = aggregate_sglang_meta_info(meta_infos)

        self.assertAlmostEqual(aggregated["spec_accept_length"], 19 / 5)
        self.assertAlmostEqual(aggregated["spec_accept_rate"], 14 / 25)
        self.assertEqual(aggregated["spec_accepted_drafts"], 14)
        self.assertEqual(aggregated["spec_proposed_drafts"], 25)
        self.assertEqual(aggregated["spec_verify_ct"], 5)
        self.assertEqual(aggregated["cached_tokens"], 8)
        self.assertEqual(aggregated["cached_tokens_details"]["device"], 3)
        self.assertEqual(aggregated["cached_tokens_details"]["host"], 1)
        self.assertEqual(aggregated["cached_tokens_details"]["storage"], 4)
        self.assertEqual(aggregated["cached_tokens_details"]["storage_backend"], "local")
        self.assertAlmostEqual(
            aggregated["latency_breakdown"]["draft_proposal_time"], 0.25
        )
        self.assertAlmostEqual(
            aggregated["latency_breakdown"]["target_verification_time"], 0.45
        )
        self.assertAlmostEqual(
            aggregated["latency_breakdown"]["grpc_communication_time"], 0.01
        )

    def test_aggregate_sglang_meta_info_falls_back_to_weighted_accept_length(self):
        aggregated = aggregate_sglang_meta_info(
            [
                {"spec_accept_length": 5.0, "spec_verify_ct": 2},
                {"spec_accept_length": 3.0, "spec_verify_ct": 6},
            ]
        )

        self.assertAlmostEqual(aggregated["spec_accept_length"], 3.5)

    def test_aggregate_sglang_meta_info_empty(self):
        aggregated = aggregate_sglang_meta_info([])

        self.assertIsNone(aggregated["spec_accept_length"])
        self.assertIsNone(aggregated["spec_accept_rate"])
        self.assertIsNone(aggregated["latency_breakdown"])
        self.assertIsNone(aggregated["cached_tokens_details"])

    def test_result_jsonl_includes_extended_metrics(self):
        result = BenchOneCaseResult(
            run_name="case",
            batch_size=1,
            input_len=1024,
            output_len=16,
            latency=1.0,
            input_throughput=100.0,
            output_throughput=10.0,
            overall_throughput=110.0,
            last_ttft=0.1,
            last_gen_throughput=9.0,
            acc_length=4.5,
            cache_hit_rate=0.25,
            spec_accept_length=4.5,
            spec_accept_rate=0.875,
            spec_accepted_drafts=7,
            spec_proposed_drafts=8,
            spec_verify_ct=2,
            latency_breakdown={
                "draft_proposal_time": 0.1,
                "target_verification_time": 0.2,
            },
            cached_tokens=256,
            cached_tokens_details={"device": 256},
        )

        with tempfile.NamedTemporaryFile() as fout:
            result.dump_to_jsonl(fout.name)
            fout.seek(0)
            payload = json.loads(fout.read().decode())

        self.assertEqual(payload["spec_accept_length"], 4.5)
        self.assertEqual(payload["spec_accept_rate"], 0.875)
        self.assertEqual(payload["spec_accepted_drafts"], 7)
        self.assertEqual(payload["spec_proposed_drafts"], 8)
        self.assertEqual(payload["spec_verify_ct"], 2)
        self.assertEqual(payload["cached_tokens"], 256)
        self.assertEqual(payload["cached_tokens_details"], {"device": 256})
        self.assertEqual(payload["latency_breakdown"]["draft_proposal_time"], 0.1)


if __name__ == "__main__":
    unittest.main()
