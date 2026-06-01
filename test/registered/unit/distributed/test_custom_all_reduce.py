"""Unit tests for custom all-reduce communicator sizing."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestCustomAllreduceSizing(unittest.TestCase):
    def test_graph_metadata_buffer_is_larger_than_payload_buffer(self):
        graph_meta_size = CustomAllreduce._resolve_graph_meta_buffer_size(
            CustomAllreduce._MAX_CAR_SIZE
        )
        self.assertGreaterEqual(graph_meta_size, 64 * 1024 * 1024)
        self.assertGreater(graph_meta_size, CustomAllreduce._MAX_CAR_SIZE)

    @patch(
        "sglang.srt.distributed.device_communicators.custom_all_reduce.get_int_env_var",
        return_value=512,
    )
    def test_graph_metadata_buffer_env_override(self, _mock_get_int_env_var):
        graph_meta_size = CustomAllreduce._resolve_graph_meta_buffer_size(
            CustomAllreduce._MAX_CAR_SIZE
        )
        self.assertEqual(graph_meta_size, 512 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
