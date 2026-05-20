import unittest

from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestReqTimeStatsLatencyBreakdown(CustomTestCase):
    def test_latency_breakdown_contains_expected_fields_and_other_time(self):
        stats = SchedulerReqTimeStats()
        stats.wait_queue_entry_time = 10.0
        stats.forward_entry_time = 10.2
        stats.observe_draft_rpc_timing(
            grpc_communication_time=0.1,
            draft_queue_scheduling_time=0.05,
            draft_proposal_time=0.3,
        )
        stats.target_verification_time = 0.25

        breakdown = stats.get_latency_breakdown(e2e_latency=1.0)

        self.assertEqual(
            set(breakdown),
            {
                "grpc_communication_time",
                "draft_proposal_time",
                "target_verification_time",
                "draft_queue_scheduling_time",
                "target_queue_scheduling_time",
                "other_time",
            },
        )
        self.assertAlmostEqual(breakdown["grpc_communication_time"], 0.1)
        self.assertAlmostEqual(breakdown["draft_queue_scheduling_time"], 0.05)
        self.assertAlmostEqual(breakdown["draft_proposal_time"], 0.3)
        self.assertAlmostEqual(breakdown["target_verification_time"], 0.25)
        self.assertAlmostEqual(breakdown["target_queue_scheduling_time"], 0.2)
        self.assertAlmostEqual(breakdown["other_time"], 0.1)

    def test_spec_verify_timing_accumulates_without_tracing(self):
        stats = SchedulerReqTimeStats()

        stats.set_spec_verify_start_time(100.0)
        stats.set_spec_verify_end_time(100.4, accepted_tokens=2)
        stats.set_spec_verify_start_time(101.0)
        stats.set_spec_verify_end_time(101.3, accepted_tokens=1)

        self.assertAlmostEqual(stats.target_verification_time, 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
