import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _Req:
    def __init__(self):
        self.spec_verify_ct = 0
        self.spec_accepted_drafts = 0
        self.spec_acceptance_histogram = []

    def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):
        if len(self.spec_acceptance_histogram) <= accepted_draft_tokens:
            self.spec_acceptance_histogram.extend(
                [0]
                * (accepted_draft_tokens - len(self.spec_acceptance_histogram) + 1)
            )
        self.spec_acceptance_histogram[accepted_draft_tokens] += 1


class TestSchedulerSpecMetrics(unittest.TestCase):
    def test_records_acceptance_from_v2_accept_lens(self):
        result = SimpleNamespace(
            accept_lens=None,
            accept_length_per_req_cpu=None,
            num_accepted_drafts=0,
        )
        reqs = [_Req(), _Req(), _Req()]
        batch = SimpleNamespace(reqs=reqs)

        SchedulerOutputProcessorMixin._record_spec_acceptance_from_accept_lens(
            None, result, batch, accept_lens=[5, 1, 3]
        )

        self.assertEqual(result.num_accepted_drafts, 6)
        self.assertEqual(result.accept_length_per_req_cpu, [4, 0, 2])
        self.assertEqual([req.spec_verify_ct for req in reqs], [1, 1, 1])
        self.assertEqual([req.spec_accepted_drafts for req in reqs], [4, 0, 2])
        self.assertEqual(reqs[0].spec_acceptance_histogram, [0, 0, 0, 0, 1])
        self.assertEqual(reqs[1].spec_acceptance_histogram, [1])
        self.assertEqual(reqs[2].spec_acceptance_histogram, [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
