import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _FailingDraftWorker:
    @property
    def model_runner(self):
        raise AssertionError(
            "colocated TLI init_disaggregation should not touch draft_worker.model_runner"
        )


class TestSchedulerDisaggregation(unittest.TestCase):
    def test_colocated_tli_init_disaggregation_is_noop_for_draft_worker(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            disaggregation_mode="null",
            disaggregation_transfer_backend="mooncake",
            draft_disaggregation_role="none",
        )
        scheduler.spec_algorithm = SpeculativeAlgorithm.TLI
        scheduler.draft_worker = _FailingDraftWorker()

        Scheduler.init_disaggregation(scheduler)

        self.assertEqual(scheduler.disaggregation_mode, DisaggregationMode.NULL)
        self.assertEqual(str(scheduler.transfer_backend), "TransferBackend.MOONCAKE")


if __name__ == "__main__":
    unittest.main()
