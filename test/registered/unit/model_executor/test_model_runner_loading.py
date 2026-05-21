import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.model_executor.model_runner import synchronize_model_loading


class _FakeTPGroup:
    def __init__(self, world_size, cpu_group=None):
        self.world_size = world_size
        self.cpu_group = cpu_group


class TestModelLoadingSync(unittest.TestCase):
    @patch("sglang.srt.model_executor.model_runner.get_tp_group")
    @patch("sglang.srt.model_executor.model_runner.dist.barrier")
    @patch("sglang.srt.model_executor.model_runner.dist.monitored_barrier")
    def test_single_rank_draft_skips_collective_barrier(
        self, mock_monitored_barrier, mock_barrier, mock_get_tp_group
    ):
        mock_get_tp_group.return_value = _FakeTPGroup(world_size=1)

        synchronize_model_loading("auto", tp_rank=0)

        mock_barrier.assert_not_called()
        mock_monitored_barrier.assert_not_called()

    @patch("sglang.srt.model_executor.model_runner.get_tp_group")
    @patch("sglang.srt.model_executor.model_runner.dist.barrier")
    @patch("sglang.srt.model_executor.model_runner.dist.monitored_barrier")
    def test_mooncake_uses_plain_barrier(
        self, mock_monitored_barrier, mock_barrier, mock_get_tp_group
    ):
        mock_get_tp_group.return_value = _FakeTPGroup(world_size=4, cpu_group=object())

        synchronize_model_loading("mooncake", tp_rank=2)

        mock_barrier.assert_called_once()
        mock_monitored_barrier.assert_not_called()

    @patch("sglang.srt.model_executor.model_runner.get_tp_group")
    @patch("sglang.srt.model_executor.model_runner.dist.barrier")
    @patch("sglang.srt.model_executor.model_runner.dist.monitored_barrier")
    def test_multi_rank_non_mooncake_uses_monitored_barrier(
        self, mock_monitored_barrier, mock_barrier, mock_get_tp_group
    ):
        mock_get_tp_group.return_value = _FakeTPGroup(world_size=4, cpu_group=object())

        synchronize_model_loading("auto", tp_rank=1)

        mock_barrier.assert_not_called()
        mock_monitored_barrier.assert_called_once()


if __name__ == "__main__":
    unittest.main()
