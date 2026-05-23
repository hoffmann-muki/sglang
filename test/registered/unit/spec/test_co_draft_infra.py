import unittest
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

from sglang.srt.speculative.co_draft.bridge import UnimplementedCoDraftBridge
from sglang.srt.speculative.co_draft.dllm_linear_adapter import (
    FastDllmV2LinearAdapter,
    IndependentDllmAcceptedTokens,
    IndependentDllmDraftRequest,
    IndependentDllmDraftTokens,
)
from sglang.srt.speculative.co_draft.executor import (
    DllmDraftExecutor,
    LinearVerificationPlan,
)
from sglang.srt.speculative.co_draft.fast_dllm_v2_runner import (
    FastDllmV2ProposalRunner,
    FastDllmV2RunnerConfig,
    TransformersFastDllmV2Runtime,
    _fast_dllm_v2_internal_generation_budget,
    _ensure_transformers_dynamic_cache_indexing_support,
    _ensure_transformers_default_rope_support,
    _ensure_transformers_tied_weights_support,
)
from sglang.srt.speculative.linear_verify import (
    LinearDraftBlock,
    build_linear_draft_block,
    build_linear_verify_input,
)
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
            backend="fast_dllm_v2",
            verification_plan=LinearVerificationPlan(proposed_token_num=4),
        )

        self.assertEqual(executor.kind, "dllm")
        self.assertIs(executor.tp_plan, plan)
        self.assertEqual(executor.model_path, "dllm-model")
        self.assertEqual(executor.tokenizer_path, "dllm-tokenizer")
        self.assertEqual(executor.algorithm, "LowConfidence")
        self.assertEqual(executor.algorithm_config, "config.yaml")
        self.assertEqual(executor.backend, "fast_dllm_v2")
        self.assertTrue(executor.is_independent_model)
        self.assertEqual(executor.verification_plan.proposed_token_num, 4)
        self.assertEqual(executor.verification_plan.draft_token_num, 5)
        self.assertIsInstance(executor.linear_adapter, FastDllmV2LinearAdapter)

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
            backend="sglang_dllm",
            verification_plan=LinearVerificationPlan(proposed_token_num=4),
        )

        with self.assertRaises(NotImplementedError):
            executor.propose(None)

    def test_dllm_executor_uses_attached_runner(self):
        import torch

        class FakeFastDllmRunner:
            def __init__(self):
                self.released_request_ids = []

            def propose(self, request):
                return IndependentDllmDraftTokens(
                    request_ids=request.request_ids,
                    current_token_ids=request.current_token_ids,
                    proposed_token_ids=torch.tensor([[11, 12]]),
                    prefix_lens=request.prefix_lens,
                    metadata={"runner": "fake"},
                )

            def release(self, request_ids):
                self.released_request_ids.extend(request_ids)

        plan = LocalDraftTpPlan(
            name="dLLM draft",
            target_tp_size=4,
            draft_tp_size=1,
        )
        executor = DllmDraftExecutor(
            tp_plan=plan,
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            algorithm="FastDLLMv2",
            algorithm_config=None,
            backend="fast_dllm_v2",
            verification_plan=LinearVerificationPlan(proposed_token_num=2),
        )
        runner = FakeFastDllmRunner()
        executor.attach_runner(runner)

        result = executor.propose(
            IndependentDllmDraftRequest(
                request_ids=["r0"],
                input_ids=[[1, 2, 10]],
                current_token_ids=torch.tensor([10]),
                prefix_lens=torch.tensor([3]),
                proposed_token_num=2,
            )
        )

        self.assertIsNotNone(result.proposal)
        self.assertEqual(result.proposal.kind, "dllm")
        self.assertTrue(
            torch.equal(result.proposal.draft_token_ids, torch.tensor([10, 11, 12]))
        )
        self.assertEqual(result.proposal.metadata["backend"], "fast_dllm_v2")
        self.assertEqual(result.proposal.metadata["verification"], "linear")
        self.assertEqual(result.proposal.metadata["runner"], "fake")

        executor.release(["r0"])

        self.assertEqual(runner.released_request_ids, ["r0"])

    def test_dllm_runner_output_shape_is_validated(self):
        import torch

        class BadRunner:
            def propose(self, request):
                return IndependentDllmDraftTokens(
                    request_ids=request.request_ids,
                    current_token_ids=request.current_token_ids,
                    proposed_token_ids=torch.tensor([[11]]),
                    prefix_lens=request.prefix_lens,
                )

            def release(self, request_ids):
                return None

        plan = LocalDraftTpPlan(
            name="dLLM draft",
            target_tp_size=4,
            draft_tp_size=1,
        )
        executor = DllmDraftExecutor(
            tp_plan=plan,
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            algorithm="FastDLLMv2",
            algorithm_config=None,
            backend="fast_dllm_v2",
            verification_plan=LinearVerificationPlan(proposed_token_num=2),
        )
        executor.attach_runner(BadRunner())

        with self.assertRaisesRegex(ValueError, "wrong number of proposed tokens"):
            executor.propose(
                IndependentDllmDraftRequest(
                    request_ids=["r0"],
                    input_ids=[[1, 2, 10]],
                    current_token_ids=torch.tensor([10]),
                    prefix_lens=torch.tensor([3]),
                    proposed_token_num=2,
                )
            )

    def test_dllm_executor_extends_runner_after_accept(self):
        import torch

        class TrackingRunner:
            def __init__(self):
                self.accepted = None

            def propose(self, request):
                return IndependentDllmDraftTokens(
                    request_ids=request.request_ids,
                    current_token_ids=request.current_token_ids,
                    proposed_token_ids=torch.tensor([[11, 12]]),
                    prefix_lens=request.prefix_lens,
                )

            def extend_after_accept(self, accepted):
                self.accepted = accepted

            def release(self, request_ids):
                return None

        plan = LocalDraftTpPlan(
            name="dLLM draft",
            target_tp_size=4,
            draft_tp_size=1,
        )
        executor = DllmDraftExecutor(
            tp_plan=plan,
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            algorithm="FastDLLMv2",
            algorithm_config=None,
            backend="fast_dllm_v2",
            verification_plan=LinearVerificationPlan(proposed_token_num=2),
        )
        runner = TrackingRunner()
        executor.attach_runner(runner)

        result = executor.extend_after_accept(
            IndependentDllmAcceptedTokens(
                request_ids=["r0"],
                accepted_token_ids=[[11]],
            )
        )

        self.assertIsNone(result.proposal)
        self.assertEqual(runner.accepted.request_ids[0], "r0")
        self.assertEqual(runner.accepted.accepted_token_ids, [[11]])


class TestFastDllmV2ProposalRunner(unittest.TestCase):
    def test_config_loads_generation_options_from_yaml(self):
        with NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(
                "\n".join(
                    [
                        "block_size: 16",
                        "small_block_size: 4",
                        "threshold: 0.75",
                        "generation_max_new_tokens: 128",
                        "device_map: cuda:0",
                        "generation_kwargs:",
                        "  do_sample: false",
                    ]
                )
            )
            config_file.flush()

            plan = LocalDraftTpPlan(
                name="dLLM draft",
                target_tp_size=4,
                draft_tp_size=1,
            )
            executor = DllmDraftExecutor(
                tp_plan=plan,
                model_path="fast-dllm",
                tokenizer_path="fast-dllm-tokenizer",
                algorithm="FastDLLMv2",
                algorithm_config=config_file.name,
                backend="fast_dllm_v2",
                verification_plan=LinearVerificationPlan(proposed_token_num=3),
            )

            config = FastDllmV2RunnerConfig.from_executor(executor)

        self.assertEqual(config.model_path, "fast-dllm")
        self.assertEqual(config.tokenizer_path, "fast-dllm-tokenizer")
        self.assertEqual(config.proposed_token_num, 3)
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.small_block_size, 4)
        self.assertEqual(config.threshold, 0.75)
        self.assertEqual(config.generation_max_new_tokens, 128)
        self.assertEqual(config.device_map, "cuda:0")
        self.assertEqual(config.generation_kwargs, {"do_sample": False})

    def test_internal_generation_budget_is_block_aligned(self):
        self.assertEqual(
            _fast_dllm_v2_internal_generation_budget(
                prompt_len=29,
                proposed_token_num=2,
                block_size=32,
            ),
            32,
        )
        self.assertEqual(
            _fast_dllm_v2_internal_generation_budget(
                prompt_len=27,
                proposed_token_num=8,
                block_size=32,
            ),
            64,
        )
        self.assertEqual(
            _fast_dllm_v2_internal_generation_budget(
                prompt_len=32,
                proposed_token_num=8,
                block_size=32,
            ),
            32,
        )

    def test_runner_tracks_state_and_delegates_to_runtime(self):
        import torch

        class FakeRuntime:
            def __init__(self):
                self.propose_seen_state = None
                self.accepted_seen_state = None
                self.released = []

            def propose(self, config, request, states):
                self.propose_seen_state = dict(states)
                return IndependentDllmDraftTokens(
                    request_ids=request.request_ids,
                    current_token_ids=request.current_token_ids,
                    proposed_token_ids=torch.tensor([[101, 102]]),
                    prefix_lens=request.prefix_lens,
                    metadata={"runtime": "fake"},
                )

            def extend_after_accept(self, config, accepted, states):
                self.accepted_seen_state = dict(states)

            def release(self, config, request_ids, states):
                self.released.extend(request_ids)

        runtime = FakeRuntime()
        runner = FastDllmV2ProposalRunner(
            FastDllmV2RunnerConfig(
                model_path="fast-dllm",
                tokenizer_path="fast-dllm",
                proposed_token_num=2,
            ),
            runtime=runtime,
        )

        tokens = runner.propose(
            IndependentDllmDraftRequest(
                request_ids=["r0"],
                input_ids=[[10, 11, 12]],
                current_token_ids=torch.tensor([12]),
                prefix_lens=torch.tensor([3]),
                proposed_token_num=2,
            )
        )

        self.assertTrue(
            torch.equal(tokens.proposed_token_ids, torch.tensor([[101, 102]]))
        )
        self.assertEqual(
            runtime.propose_seen_state["r0"].input_ids,
            [10, 11, 12],
        )

        runner.extend_after_accept(
            IndependentDllmAcceptedTokens(
                request_ids=["r0"],
                accepted_token_ids=[[101]],
            )
        )

        self.assertEqual(runner.states["r0"].input_ids, [10, 11, 12, 101])
        self.assertEqual(runner.states["r0"].accepted_token_count, 1)
        self.assertEqual(
            runtime.accepted_seen_state["r0"].input_ids,
            [10, 11, 12, 101],
        )

        runner.release(["r0"])

        self.assertNotIn("r0", runner.states)
        self.assertEqual(runtime.released, ["r0"])

    def test_runner_falls_back_without_accelerate(self):
        import torch
        from unittest.mock import MagicMock

        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=2,
        )
        transformer_runtime = TransformersFastDllmV2Runtime()
        transformer_runtime._has_accelerate = MagicMock(return_value=False)
        fake_model = MagicMock()
        fake_model.device = torch.device("cpu")
        fake_model.generate = MagicMock(
            return_value=torch.tensor([[10, 11, 12, 101, 102]])
        )
        transformer_runtime._load_model = MagicMock(return_value=fake_model)
        transformer_runtime._load_tokenizer = MagicMock(return_value=MagicMock())

        request = IndependentDllmDraftRequest(
            request_ids=["r0"],
            input_ids=[[10, 11, 12]],
            current_token_ids=torch.tensor([12]),
            prefix_lens=torch.tensor([3]),
            proposed_token_num=2,
        )

        transformer_runtime.propose(
            config,
            request,
            {"r0": MagicMock(input_ids=[10, 11, 12])},
        )

        transformer_runtime._load_model.assert_called_once()
        self.assertEqual(
            transformer_runtime._load_model.call_args.kwargs["device_map"],
            None,
        )

    def test_default_rope_shim_registers_missing_default_key(self):
        import torch
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from unittest.mock import MagicMock

        sentinel = ROPE_INIT_FUNCTIONS.pop("default", None)
        try:
            _ensure_transformers_default_rope_support()

            self.assertIn("default", ROPE_INIT_FUNCTIONS)
            inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["default"](
                MagicMock(
                    hidden_size=8,
                    num_attention_heads=2,
                    rope_theta=10000.0,
                ),
                device=torch.device("cpu"),
            )
            self.assertEqual(attention_factor, 1.0)
            self.assertEqual(inv_freq.numel(), 2)
        finally:
            if sentinel is not None:
                ROPE_INIT_FUNCTIONS["default"] = sentinel
            else:
                ROPE_INIT_FUNCTIONS.pop("default", None)

    def test_tied_weights_shim_normalizes_legacy_list_metadata(self):
        from transformers.modeling_utils import PreTrainedModel

        original_method = PreTrainedModel.get_expanded_tied_weights_keys

        def fake_original(self, all_submodels=False):
            return self._tied_weights_keys

        try:
            PreTrainedModel.get_expanded_tied_weights_keys = fake_original
            _ensure_transformers_tied_weights_support()

            dummy = SimpleNamespace(
                _tied_weights_keys=["lm_head.weight"],
                config=SimpleNamespace(model_type="Fast_dLLM_Qwen"),
            )
            result = PreTrainedModel.get_expanded_tied_weights_keys(dummy, False)

            self.assertEqual(
                dummy._tied_weights_keys,
                {"lm_head.weight": "model.embed_tokens.weight"},
            )
            self.assertEqual(
                result,
                {"lm_head.weight": "model.embed_tokens.weight"},
            )
        finally:
            PreTrainedModel.get_expanded_tied_weights_keys = original_method

    def test_dynamic_cache_shim_adds_legacy_layer_indexing(self):
        from transformers.cache_utils import DynamicCache

        sentinel = getattr(DynamicCache, "__getitem__", None)
        if sentinel is not None:
            delattr(DynamicCache, "__getitem__")

        try:
            _ensure_transformers_dynamic_cache_indexing_support()

            cache = DynamicCache()
            cache.layers = [
                SimpleNamespace(keys="layer0-keys", values="layer0-values")
            ]

            self.assertEqual(cache[0], ("layer0-keys", "layer0-values"))
        finally:
            if sentinel is not None:
                DynamicCache.__getitem__ = sentinel
            else:
                delattr(DynamicCache, "__getitem__")


class TestLinearVerifyHelpers(unittest.TestCase):
    def test_builds_linear_block_from_independent_dllm_tokens(self):
        import torch

        block = build_linear_draft_block(
            current_token_ids=torch.tensor([10, 20]),
            proposed_token_ids=torch.tensor([[11, 12], [21, 22]]),
            prefix_lens=torch.tensor([5, 9]),
        )

        self.assertEqual(block.draft_token_num, 3)
        self.assertTrue(
            torch.equal(block.draft_token, torch.tensor([10, 11, 12, 20, 21, 22]))
        )
        self.assertTrue(
            torch.equal(block.positions, torch.tensor([5, 6, 7, 9, 10, 11]))
        )

    def test_builds_dflash_compatible_verify_input(self):
        import torch

        block = LinearDraftBlock(
            draft_token=torch.tensor([1, 2, 3, 4]),
            positions=torch.tensor([7, 8, 9, 10]),
            draft_token_num=2,
        )

        verify_input = build_linear_verify_input(block)

        self.assertEqual(verify_input.draft_token_num, 2)
        self.assertEqual(verify_input.num_tokens_per_batch, 2)
        self.assertTrue(torch.equal(verify_input.draft_token, block.draft_token))
        self.assertTrue(torch.equal(verify_input.positions, block.positions))

    def test_adapter_converts_independent_tokens_to_linear_proposal(self):
        import torch

        plan = LocalDraftTpPlan(
            name="dLLM draft",
            target_tp_size=4,
            draft_tp_size=1,
        )
        executor = DllmDraftExecutor(
            tp_plan=plan,
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            algorithm="FastDLLMv2",
            algorithm_config=None,
            backend="fast_dllm_v2",
            verification_plan=LinearVerificationPlan(proposed_token_num=2),
        )
        tokens = IndependentDllmDraftTokens(
            request_ids=["r0"],
            current_token_ids=torch.tensor([10]),
            proposed_token_ids=torch.tensor([[11, 12]]),
            prefix_lens=torch.tensor([5]),
        )

        block = executor.build_linear_block(tokens)
        proposal = executor.linear_adapter.build_draft_proposal(tokens)

        self.assertEqual(block.draft_token_num, 3)
        self.assertTrue(torch.equal(block.draft_token, torch.tensor([10, 11, 12])))
        self.assertEqual(proposal.kind, "dllm")
        self.assertEqual(proposal.metadata["backend"], "fast_dllm_v2")
        self.assertEqual(proposal.metadata["verification"], "linear")


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
