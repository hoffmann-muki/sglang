import unittest

from sglang.srt.speculative.co_draft.bridge import UnimplementedCoDraftBridge
from sglang.srt.speculative.co_draft.executor import DllmDraftExecutor
from sglang.srt.speculative.co_draft.tp import LocalDraftTpPlan


class TestLocalDraftTpPlan(unittest.TestCase):
    def test_maps_participating_ranks(self):
        plan = LocalDraftTpPlan(
            name="test draft",
            target_tp_size=4,
            draft_tp_size=2,
        )

        self.assertTrue(plan.is_asymmetric)
        self.assertTrue(plan.owns_rank(0))
        self.assertTrue(plan.owns_rank(1))
        self.assertFalse(plan.owns_rank(2))
        self.assertEqual(plan.local_rank(0), 0)
        self.assertEqual(plan.local_rank(1), 1)

    def test_detects_asymmetric_draft_tp(self):
        plan = LocalDraftTpPlan(
            name="test draft",
            target_tp_size=4,
            draft_tp_size=1,
        )

        self.assertTrue(plan.is_asymmetric)

    def test_detects_symmetric_draft_tp(self):
        plan = LocalDraftTpPlan(
            name="test draft",
            target_tp_size=4,
            draft_tp_size=4,
        )

        self.assertFalse(plan.is_asymmetric)

    def test_rejects_draft_tp_larger_than_target_tp(self):
        with self.assertRaises(ValueError) as context:
            LocalDraftTpPlan(
                name="test draft",
                target_tp_size=2,
                draft_tp_size=4,
            )

        self.assertIn("cannot exceed", str(context.exception))

    def test_rejects_draft_tp_that_does_not_divide_target_tp(self):
        with self.assertRaises(ValueError) as context:
            LocalDraftTpPlan(
                name="test draft",
                target_tp_size=4,
                draft_tp_size=3,
            )

        self.assertIn("must divide", str(context.exception))

    def test_rejects_non_participating_local_rank_lookup(self):
        plan = LocalDraftTpPlan(
            name="test draft",
            target_tp_size=4,
            draft_tp_size=1,
        )

        with self.assertRaises(ValueError) as context:
            plan.local_rank(2)

        self.assertIn("does not participate", str(context.exception))

    def test_rejects_rank_range_that_exceeds_target_tp(self):
        with self.assertRaises(ValueError) as context:
            LocalDraftTpPlan(
                name="test draft",
                target_tp_size=4,
                draft_tp_size=2,
                root_rank=3,
            )

        self.assertIn("exceeds target TP size", str(context.exception))


class TestDllmDraftExecutor(unittest.TestCase):
    def test_descriptor_preserves_dllm_configuration(self):
        plan = LocalDraftTpPlan(
            name="dLLM draft",
            target_tp_size=4,
            draft_tp_size=1,
        )
        executor = DllmDraftExecutor(
            tp_plan=plan,
            model_path="dllm-model",
            tokenizer_path="dllm-tokenizer",
            algorithm="LowConfidence",
            algorithm_config="config.yaml",
        )

        self.assertEqual(executor.kind, "dllm")
        self.assertIs(executor.tp_plan, plan)
        self.assertEqual(executor.model_path, "dllm-model")
        self.assertEqual(executor.tokenizer_path, "dllm-tokenizer")
        self.assertEqual(executor.algorithm, "LowConfidence")
        self.assertEqual(executor.algorithm_config, "config.yaml")

    def test_dllm_runtime_adapter_is_explicitly_unimplemented(self):
        plan = LocalDraftTpPlan(
            name="dLLM draft",
            target_tp_size=4,
            draft_tp_size=1,
        )
        executor = DllmDraftExecutor(
            tp_plan=plan,
            model_path="dllm-model",
            tokenizer_path="dllm-tokenizer",
            algorithm="LowConfidence",
            algorithm_config=None,
        )

        with self.assertRaises(NotImplementedError):
            executor.propose()


class TestCoDraftBridge(unittest.TestCase):
    def test_unimplemented_bridge_fails_fast(self):
        bridge = UnimplementedCoDraftBridge()

        with self.assertRaises(NotImplementedError):
            bridge.ar_to_dllm_completion_request(
                ar_proposal=None,
                max_completion_tokens=4,
            )

        with self.assertRaises(NotImplementedError):
            bridge.dllm_to_ar_anchor_batch(dllm_proposal=None)


if __name__ == "__main__":
    unittest.main()
