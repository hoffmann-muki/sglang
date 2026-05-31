import unittest

import torch

from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
from sglang.srt.utils import get_device, next_power_of_2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestDraftDecodeKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    def test_generate_draft_decode_kv_indices_overwrites_indptr_zero(self):
        batch_size = 64
        topk = 1
        speculative_num_steps = 4
        page_size = 1
        pool_len = 2048
        max_context_len = 1100

        req_pool_indices = torch.arange(
            batch_size, dtype=torch.int32, device=get_device()
        )
        req_to_token = torch.arange(
            batch_size * pool_len, dtype=torch.int32, device=get_device()
        ).reshape(batch_size, pool_len)
        seq_lens = torch.full(
            (batch_size,), 1024, dtype=torch.int32, device=get_device()
        )
        positions = seq_lens.repeat_interleave(topk)

        bs = batch_size * topk
        kv_indices = torch.empty(
            (speculative_num_steps, bs * max_context_len),
            dtype=torch.int32,
            device=get_device(),
        )
        kv_indptr = torch.full(
            (speculative_num_steps, bs + 1),
            66625,
            dtype=torch.int32,
            device=get_device(),
        )

        generate_draft_decode_kv_indices[
            (speculative_num_steps, batch_size, topk)
        ](
            req_pool_indices,
            req_to_token,
            seq_lens,
            kv_indices,
            kv_indptr,
            positions,
            pool_len,
            kv_indices.shape[1],
            kv_indptr.shape[1],
            next_power_of_2(batch_size),
            next_power_of_2(speculative_num_steps),
            next_power_of_2(bs),
            page_size,
        )

        kv_indptr_cpu = kv_indptr.cpu()
        self.assertTrue(torch.all(kv_indptr_cpu[:, 0] == 0))

        for step in range(speculative_num_steps):
            step_len = step + 1
            expected = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=kv_indptr_cpu.device
            )
            expected[1:] = torch.cumsum(positions.cpu(), dim=0) + (
                torch.arange(
                    1, bs + 1, dtype=torch.int32, device=kv_indptr_cpu.device
                )
                * step_len
            )
            self.assertTrue(torch.equal(kv_indptr_cpu[step], expected))


if __name__ == "__main__":
    unittest.main()
