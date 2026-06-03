"""Unit tests for spec-v2 schedule batch decode synchronization."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.test_utils import CustomTestCase


class TestSpecDecodeInputIds(CustomTestCase):

    def test_syncs_decode_input_ids_from_output_ids(self):
        """Spec-v2 decode should refresh input_ids from output_ids."""
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        batch.output_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64)

        ScheduleBatch._sync_spec_decode_input_ids(batch)

        self.assertTrue(torch.equal(batch.input_ids, batch.output_ids))

    def test_sync_keeps_existing_input_ids_when_no_output_ids(self):
        """Spec-v2 decode should leave input_ids alone if output_ids is absent."""
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        batch.output_ids = None

        ScheduleBatch._sync_spec_decode_input_ids(batch)

        self.assertTrue(torch.equal(batch.input_ids, torch.tensor([1, 2, 3, 4])))
