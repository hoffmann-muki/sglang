import unittest
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

from sglang.srt.dllm.config import DllmConfig
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
    FastDllmV2BlockProposalEngine,
    FastDllmV2ProposalRunner,
    FastDllmV2RequestState,
    FastDllmV2RunnerConfig,
    SGLangNativeFastDllmV2Runtime,
    TransformersFastDllmV2Runtime,
    _FastDllmV2NativeTraceRecorder,
    _SGLangNativeBlockCacheState,
    _SGLangNativeCacheState,
    _disable_transformers_hub_kernels_for_fast_dllm_v2,
    _fast_dllm_v2_internal_generation_budget,
    _ensure_fast_dllm_v2_tied_embeddings,
    _ensure_transformers_default_rope_support,
    _ensure_transformers_legacy_dynamic_cache_support,
    _ensure_transformers_tied_weights_support,
    _ensure_transformers_v453_sdpa_support,
    _normalize_fast_dllm_v2_rope_config,
    _patch_fast_dllm_v2_rotary_embedding_forward,
)
from sglang.srt.speculative.linear_verify import (
    LinearDraftBlock,
    build_linear_draft_block,
    build_linear_verify_input,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
    def test_native_model_entrypoint_uses_hf_architecture_name(self):
        from sglang.srt.models.fast_dllm_v2 import EntryClass

        self.assertEqual(EntryClass.__name__, "Fast_dLLM_QwenForCausalLM")

    def test_dllm_config_recognizes_fast_dllm_v2_native_architecture(self):
        from unittest.mock import patch

        server_args = SimpleNamespace(
            dllm_algorithm="HierarchyBlock",
            dllm_algorithm_config=None,
            max_running_requests=None,
            model_path="fast-dllm",
            revision=None,
        )

        with patch(
            "sglang.srt.dllm.config.ModelConfig.from_server_args"
        ) as from_server_args:
            from_server_args.return_value = SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Fast_dLLM_QwenForCausalLM"]
                )
            )

            config = DllmConfig.from_server_args(server_args)

        self.assertEqual(config.algorithm, "HierarchyBlock")
        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.mask_id, 151665)
        self.assertEqual(config.max_running_requests, 1)

    def test_standalone_config_uses_linear_verify_width(self):
        server_args = SimpleNamespace(
            speculative_draft_model_path="fast-dllm",
            speculative_fast_dllm_v2_algorithm_config=None,
            speculative_num_draft_tokens=9,
        )

        config = FastDllmV2RunnerConfig.from_server_args(server_args)

        self.assertEqual(config.model_path, "fast-dllm")
        self.assertEqual(config.tokenizer_path, "fast-dllm")
        self.assertEqual(config.proposed_token_num, 8)

    def test_standalone_config_loads_generation_options(self):
        with NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(
                "\n".join(
                    [
                        "tokenizer_path: fast-dllm-tokenizer",
                        "runtime: sglang_native",
                        "context_length: 2048",
                        "native_max_total_tokens: 8192",
                        "native_max_running_requests: 4",
                        "block_size: 16",
                        "small_block_size: 4",
                        "threshold: 0.8",
                        "device_map: cuda:0",
                        "proposal_kwargs:",
                        "  do_sample: false",
                    ]
                )
            )
            config_file.flush()
            server_args = SimpleNamespace(
                speculative_draft_model_path="fast-dllm",
                speculative_fast_dllm_v2_algorithm_config=config_file.name,
                speculative_num_draft_tokens=5,
            )

            config = FastDllmV2RunnerConfig.from_server_args(server_args)

        self.assertEqual(config.tokenizer_path, "fast-dllm-tokenizer")
        self.assertEqual(config.proposed_token_num, 4)
        self.assertEqual(config.runtime, "sglang_native")
        self.assertEqual(config.context_length, 2048)
        self.assertEqual(config.native_max_total_tokens, 8192)
        self.assertEqual(config.native_max_running_requests, 4)
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.small_block_size, 4)
        self.assertEqual(config.threshold, 0.8)
        self.assertEqual(config.device_map, "cuda:0")
        self.assertEqual(config.proposal_kwargs, {"do_sample": False})

    def test_config_loads_generation_options_from_yaml(self):
        with NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(
                "\n".join(
                    [
                        "block_size: 16",
                        "small_block_size: 4",
                        "native_max_total_tokens: 6144",
                        "native_max_running_requests: 6",
                        "threshold: 0.75",
                        "device_map: cuda:0",
                        "attn_implementation: sdpa",
                        "disable_hub_kernels: true",
                        "proposal_kwargs:",
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
        self.assertEqual(config.native_max_total_tokens, 6144)
        self.assertEqual(config.native_max_running_requests, 6)
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.small_block_size, 4)
        self.assertEqual(config.threshold, 0.75)
        self.assertEqual(config.device_map, "cuda:0")
        self.assertEqual(config.attn_implementation, "sdpa")
        self.assertTrue(config.disable_hub_kernels)
        self.assertEqual(config.proposal_kwargs, {"do_sample": False})

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

    def test_runtime_initializes_partial_block_with_padded_state(self):
        import torch

        runtime = TransformersFastDllmV2Runtime()
        input_tensor = torch.arange(26, dtype=torch.long).reshape(1, 26)
        seq_block_idx = torch.tensor([0], dtype=torch.long)

        x_init, updated_input = runtime._initialize_proposal_block(
            input_tensor=input_tensor,
            seq_block_idx=seq_block_idx,
            block_idx=0,
            block_size=32,
            mask_id=151666,
        )

        self.assertEqual(tuple(x_init.shape), (1, 32))
        self.assertEqual(tuple(updated_input.shape), (1, 32))
        self.assertTrue(torch.equal(x_init, updated_input))
        self.assertTrue(torch.equal(x_init[:, :26], input_tensor))
        self.assertTrue(torch.equal(x_init[:, 26:], torch.full((1, 6), 151666)))

    def test_block_proposal_engine_uses_backend_padding_contract(self):
        import torch

        class FakeBackend:
            def proposal_pad_token_id(self):
                return 99

        engine = FastDllmV2BlockProposalEngine(FakeBackend())
        token_ids = torch.tensor([[10, 11, 151665]], dtype=torch.long)
        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=3,
        )

        proposal = engine.pad_proposal(
            token_ids=token_ids,
            prompt_len=1,
            config=config,
            mask_id=151665,
        )

        self.assertTrue(torch.equal(proposal, torch.tensor([11, 99, 99])))

    def test_runner_consumes_matching_lookahead(self):
        state = FastDllmV2RequestState(
            input_ids=[10],
            draft_lookahead=[11, 12, 13],
        )

        FastDllmV2ProposalRunner._consume_lookahead(state, [11, 12])

        self.assertEqual(state.draft_lookahead, [13])

    def test_runner_clears_lookahead_on_divergent_bonus_token(self):
        state = FastDllmV2RequestState(
            input_ids=[10],
            draft_lookahead=[11, 12, 13],
        )

        FastDllmV2ProposalRunner._consume_lookahead(state, [11, 99])

        self.assertEqual(state.draft_lookahead, [])

    def test_runner_refresh_preserves_lookahead_for_external_progress(self):
        import torch

        runner = FastDllmV2ProposalRunner(
            FastDllmV2RunnerConfig(
                model_path="fast-dllm",
                tokenizer_path="fast-dllm",
                proposed_token_num=2,
            )
        )
        runner.states["r0"] = FastDllmV2RequestState(
            input_ids=[10],
            draft_lookahead=[11, 12, 13],
        )

        runner._refresh_states(
            IndependentDllmDraftRequest(
                request_ids=["r0"],
                input_ids=[[10, 11]],
                current_token_ids=torch.tensor([11]),
                prefix_lens=torch.tensor([2]),
                proposed_token_num=2,
            )
        )

        self.assertEqual(runner.states["r0"].input_ids, [10, 11])
        self.assertEqual(runner.states["r0"].draft_lookahead, [12, 13])

    def test_transformers_runtime_extends_short_lookahead_once(self):
        import torch
        from unittest.mock import MagicMock

        runtime = TransformersFastDllmV2Runtime()
        runtime.model = SimpleNamespace(device=torch.device("cpu"))
        runtime._propose_one = MagicMock(return_value=torch.tensor([14, 15, 16, 17]))
        state = FastDllmV2RequestState(
            input_ids=[10, 11],
            draft_lookahead=[12, 13],
        )
        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=4,
        )

        first = runtime._propose_from_state(config, state)
        second = runtime._propose_from_state(config, state)

        self.assertTrue(torch.equal(first, torch.tensor([12, 13, 14, 15])))
        self.assertTrue(torch.equal(second, torch.tensor([12, 13, 14, 15])))
        self.assertEqual(state.draft_lookahead, [12, 13, 14, 15, 16, 17])
        runtime._propose_one.assert_called_once_with(config, [10, 11, 12, 13], None)

    def test_native_runtime_fails_at_forward_adapter_until_wired(self):
        import torch

        runtime = SGLangNativeFastDllmV2Runtime(
            model_runner=SimpleNamespace(device=torch.device("cpu"))
        )
        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=2,
            runtime="sglang_native",
        )

        with self.assertRaisesRegex(NotImplementedError, "initialized SGLang ModelRunner"):
            runtime.propose(
                config,
                IndependentDllmDraftRequest(
                    request_ids=["r0"],
                    input_ids=[[10]],
                    current_token_ids=torch.tensor([10]),
                    prefix_lens=torch.tensor([1]),
                    proposed_token_num=2,
                ),
                {"r0": FastDllmV2RequestState(input_ids=[10])},
            )

    def test_native_runtime_can_serve_existing_lookahead_without_forward(self):
        import torch

        runtime = SGLangNativeFastDllmV2Runtime(
            model_runner=SimpleNamespace(device=torch.device("cpu"))
        )
        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=2,
            runtime="sglang_native",
        )

        tokens = runtime.propose(
            config,
            IndependentDllmDraftRequest(
                request_ids=["r0"],
                input_ids=[[10]],
                current_token_ids=torch.tensor([10]),
                prefix_lens=torch.tensor([1]),
                proposed_token_num=2,
            ),
            {"r0": FastDllmV2RequestState(input_ids=[10], draft_lookahead=[11, 12])},
        )

        self.assertTrue(torch.equal(tokens.proposed_token_ids, torch.tensor([[11, 12]])))
        self.assertEqual(tokens.metadata["runtime"], "sglang_native")

    def test_native_runtime_rejects_foreign_block_cache(self):
        import torch

        runtime = SGLangNativeFastDllmV2Runtime(
            model_runner=SimpleNamespace(device=torch.device("cpu"))
        )

        with self.assertRaisesRegex(NotImplementedError, "foreign_block_past_key_values"):
            runtime.forward(
                profile=None,
                phase="block_cache_reuse",
                input_ids=torch.tensor([[10, 11]], dtype=torch.long),
                use_block_cache=True,
                block_past_key_values=object(),
                replace_position=0,
            )

    def test_native_runtime_releases_block_caches_before_prefix_caches(self):
        import torch

        class FakeAllocator:
            page_size = 1

            def __init__(self):
                self.restored = []

            def restore_state(self, state):
                self.restored.append(state)

        class FakeReqPool:
            def __init__(self):
                self.freed = []

            def free(self, req):
                self.freed.append(req.rid)

        allocator = FakeAllocator()
        req_pool = FakeReqPool()
        runtime = SGLangNativeFastDllmV2Runtime(
            model_runner=SimpleNamespace(
                device=torch.device("cpu"),
                token_to_kv_pool_allocator=allocator,
                req_to_token_pool=req_pool,
            )
        )
        prefix_req = SimpleNamespace(rid="prefix")
        block_req = SimpleNamespace(rid="block")
        runtime._active_native_caches.append(
            _SGLangNativeCacheState(
                req=prefix_req,
                req_pool_idx=0,
                allocator_state_before_alloc="prefix-state",
                kv_len=4,
            )
        )
        runtime._active_native_block_caches.append(
            _SGLangNativeBlockCacheState(
                req=block_req,
                req_pool_idx=1,
                allocator_state_before_alloc="block-state",
                prefix_len=4,
                block_size=2,
                token_ids=torch.tensor([10, 11]),
            )
        )

        runtime._release_active_native_caches()

        self.assertEqual(allocator.restored, ["block-state", "prefix-state"])
        self.assertEqual(req_pool.freed, ["block", "prefix"])

    def test_native_runtime_builds_cached_prefix_in_block_sized_forwards(self):
        import types
        import torch

        class FakeAllocator:
            page_size = 1

            def __init__(self):
                self.next_loc = 0

            def backup_state(self):
                return self.next_loc

            def restore_state(self, state):
                self.next_loc = state

            def alloc(self, size):
                locs = torch.arange(self.next_loc, self.next_loc + size)
                self.next_loc += size
                return locs

        class FakeReqPool:
            def __init__(self):
                self.writes = []

            def alloc(self, reqs):
                reqs[0].req_pool_idx = 0
                return [0]

            def write(self, index, values):
                self.writes.append((index, values.clone()))

            def free(self, req):
                return None

        runtime = SGLangNativeFastDllmV2Runtime(
            model_runner=SimpleNamespace(
                device=torch.device("cpu"),
                attn_backend=object(),
                forward=lambda _batch: None,
                req_to_token_pool=FakeReqPool(),
                token_to_kv_pool=object(),
                token_to_kv_pool_allocator=FakeAllocator(),
            )
        )
        calls = []

        def fake_block_forward(
            self,
            *,
            input_ids,
            req_pool_idx,
            prefix_len,
            block_size,
            out_cache_loc,
            phase,
            logit_positions=None,
        ):
            calls.append(
                {
                    "input_ids": input_ids.tolist(),
                    "prefix_len": prefix_len,
                    "block_size": block_size,
                    "out_cache_loc": out_cache_loc.tolist(),
                    "logit_positions": (
                        None if logit_positions is None else logit_positions.tolist()
                    ),
                }
            )
            return torch.full((block_size, 4), len(calls), dtype=torch.float32)

        runtime._run_dllm_block_forward = types.MethodType(fake_block_forward, runtime)

        output = runtime.forward(
            profile=None,
            phase="prefix",
            input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
            use_cache=True,
            update_past_key_values=True,
            block_size=2,
        )

        self.assertEqual(
            calls,
            [
                {
                    "input_ids": [10, 11],
                    "prefix_len": 0,
                    "block_size": 2,
                    "out_cache_loc": [0, 1],
                    "logit_positions": None,
                },
                {
                    "input_ids": [12, 13],
                    "prefix_len": 2,
                    "block_size": 2,
                    "out_cache_loc": [2, 3],
                    "logit_positions": None,
                },
            ],
        )
        self.assertEqual(output.past_key_values.kv_len, 4)
        self.assertTrue(torch.equal(output.logits, torch.full((1, 2, 4), 2.0)))

    def test_shifted_small_block_logit_positions_match_sampler_shift(self):
        import torch

        self.assertTrue(
            torch.equal(
                FastDllmV2BlockProposalEngine.shifted_small_block_logit_positions(
                    0,
                    4,
                    device=torch.device("cpu"),
                ),
                torch.tensor([0, 0, 1, 2]),
            )
        )
        self.assertTrue(
            torch.equal(
                FastDllmV2BlockProposalEngine.shifted_small_block_logit_positions(
                    8,
                    12,
                    device=torch.device("cpu"),
                ),
                torch.tensor([7, 8, 9, 10]),
            )
        )

    def test_native_trace_recorder_writes_pt_and_json_summaries(self):
        import json
        import tempfile
        import torch
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "native_trace.pt"
            recorder = _FastDllmV2NativeTraceRecorder(
                str(trace_path),
                max_events=2,
                full_tensors=True,
                topk=2,
            )
            recorder.set_phase("prefix")
            recorder.record(
                "model_runner.forward",
                "output",
                {"full_logits": torch.tensor([[0.1, 0.3, 0.2]])},
            )

            loaded = torch.load(trace_path, map_location="cpu", weights_only=False)
            summary = json.loads(trace_path.with_suffix(".json").read_text())

        self.assertEqual(loaded["native"]["events"][0]["phase"], "prefix")
        self.assertIn(
            "tensor",
            loaded["native"]["events"][0]["payload"]["full_logits"],
        )
        self.assertNotIn(
            "tensor",
            summary["native"]["events"][0]["payload"]["full_logits"],
        )
        self.assertEqual(
            summary["native"]["events"][0]["payload"]["full_logits"][
                "last_row_topk"
            ]["indices"],
            [1, 2],
        )

    def test_native_trace_recorder_can_write_shape_only_summaries(self):
        import json
        import tempfile
        import torch
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "native_trace.pt"
            recorder = _FastDllmV2NativeTraceRecorder(
                str(trace_path),
                max_events=2,
                tensor_stats=False,
                flush_interval=2,
            )
            recorder.set_phase("prefix")
            recorder.record(
                "model_runner.forward",
                "output",
                {"full_logits": torch.tensor([[0.1, 0.3, 0.2]])},
            )
            recorder.flush()

            summary = json.loads(trace_path.with_suffix(".json").read_text())

        logits_summary = summary["native"]["events"][0]["payload"]["full_logits"]
        self.assertEqual(logits_summary["shape"], [1, 3])
        self.assertNotIn("mean", logits_summary)
        self.assertNotIn("last_row_topk", logits_summary)

    def test_transformers_runtime_metadata_reports_block_cache_setting(self):
        import torch

        runtime = TransformersFastDllmV2Runtime()
        runtime._ensure_loaded = lambda _config: None
        runtime._propose_from_state = lambda _config, _state: torch.tensor([11, 12])
        runtime.model_metadata = {}
        state = FastDllmV2RequestState(input_ids=[10])
        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=2,
        )

        tokens = runtime.propose(
            config,
            IndependentDllmDraftRequest(
                request_ids=["r0"],
                input_ids=[[10]],
                current_token_ids=torch.tensor([10]),
                prefix_lens=torch.tensor([1]),
                proposed_token_num=2,
            ),
            {"r0": state},
        )

        self.assertFalse(tokens.metadata["use_block_cache"])

    def test_transformers_runtime_reports_profile_metadata(self):
        import torch

        runtime = TransformersFastDllmV2Runtime()
        runtime._ensure_loaded = lambda _config: None
        runtime.model = SimpleNamespace(device=torch.device("cpu"))

        def _fake_propose_one(_config, _input_ids, profile):
            profile["forward_calls"] += 3
            profile["refine_full_block_forward_calls"] += 3
            profile["sampling_calls"] += 2
            profile["forward_time_s"] += 0.03
            profile["sampling_time_s"] += 0.01
            return torch.tensor([11, 12])

        runtime._propose_one = _fake_propose_one
        state = FastDllmV2RequestState(input_ids=[10])
        config = FastDllmV2RunnerConfig(
            model_path="fast-dllm",
            tokenizer_path="fast-dllm",
            proposed_token_num=2,
            proposal_kwargs={"profile": True, "profile_log_interval": 5},
        )

        tokens = runtime.propose(
            config,
            IndependentDllmDraftRequest(
                request_ids=["r0"],
                input_ids=[[10]],
                current_token_ids=torch.tensor([10]),
                prefix_lens=torch.tensor([1]),
                proposed_token_num=2,
            ),
            {"r0": state},
        )

        profile = tokens.metadata["profile_total"]
        self.assertTrue(tokens.metadata["profile_enabled"])
        self.assertEqual(tokens.metadata["profile_log_interval"], 5)
        self.assertEqual(profile["requests_profiled"], 1)
        self.assertEqual(profile["forward_calls"], 3)
        self.assertEqual(profile["refine_full_block_forward_calls"], 3)
        self.assertEqual(profile["sampling_calls"], 2)
        self.assertEqual(profile["draft_model_invocations"], 1)
        self.assertEqual(profile["forward_calls_per_invocation"], 3)

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

    def test_runtime_loads_without_accelerate(self):
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
        transformer_runtime._load_model = MagicMock(return_value=fake_model)
        transformer_runtime._load_tokenizer = MagicMock(return_value=MagicMock())

        transformer_runtime._ensure_loaded(config)

        transformer_runtime._load_model.assert_called_once()
        self.assertEqual(
            transformer_runtime._load_model.call_args.kwargs["device_map"],
            None,
        )

    def test_transformers_runtime_rejects_unsupported_proposal_kwargs(self):
        transformer_runtime = TransformersFastDllmV2Runtime()
        transformer_runtime.model = SimpleNamespace()

        with self.assertRaisesRegex(
            RuntimeError,
            "does not support proposal_kwargs",
        ):
            transformer_runtime._validate_proposal_config(
                FastDllmV2RunnerConfig(
                    model_path="fast-dllm",
                    tokenizer_path="fast-dllm",
                    proposed_token_num=2,
                    proposal_kwargs={"unsupported": True},
                )
            )

    def test_default_rope_shim_installs_reference_initializer(self):
        import torch
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from unittest.mock import MagicMock

        sentinel = ROPE_INIT_FUNCTIONS.get("default")
        ROPE_INIT_FUNCTIONS["default"] = lambda *args, **kwargs: (
            torch.zeros(2),
            1.0,
        )
        try:
            _ensure_transformers_default_rope_support()

            self.assertIn("default", ROPE_INIT_FUNCTIONS)
            self.assertTrue(
                getattr(
                    ROPE_INIT_FUNCTIONS["default"],
                    "_sglang_fast_dllm_v2_v453_compat",
                    False,
                )
            )
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
            self.assertFalse(torch.equal(inv_freq, torch.zeros_like(inv_freq)))
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

    def test_fast_dllm_tied_embeddings_are_enforced_after_load(self):
        import torch

        class DummyFastDllmModel:
            def __init__(self):
                self.config = SimpleNamespace(
                    model_type="Fast_dLLM_Qwen",
                    tie_word_embeddings=True,
                )
                self.input_embeddings = SimpleNamespace(
                    weight=torch.nn.Parameter(torch.ones(4, 3))
                )
                self.output_embeddings = SimpleNamespace(
                    weight=torch.nn.Parameter(torch.zeros(4, 3))
                )

            def get_input_embeddings(self):
                return self.input_embeddings

            def get_output_embeddings(self):
                return self.output_embeddings

        model = DummyFastDllmModel()

        changed = _ensure_fast_dllm_v2_tied_embeddings(model)

        self.assertTrue(changed)
        self.assertIs(
            model.get_output_embeddings().weight,
            model.get_input_embeddings().weight,
        )

    def test_fast_dllm_disables_hub_kernels_before_model_import(self):
        import os

        original_value = os.environ.pop("USE_HUB_KERNELS", None)
        try:
            changed = _disable_transformers_hub_kernels_for_fast_dllm_v2()

            self.assertTrue(changed)
            self.assertEqual(os.environ["USE_HUB_KERNELS"], "0")
        finally:
            if original_value is None:
                os.environ.pop("USE_HUB_KERNELS", None)
            else:
                os.environ["USE_HUB_KERNELS"] = original_value

    def test_fast_dllm_registers_transformers_v453_sdpa_contract(self):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        try:
            _ensure_transformers_v453_sdpa_support()

            sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
            module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
            query = torch.randn(1, 4, 3, 8)
            key = torch.randn(1, 2, 3, 8)
            value = torch.randn(1, 2, 3, 8)
            mask = torch.ones(3, 3, dtype=torch.bool).tril()

            out, weights = sdpa(
                module,
                query,
                key,
                value,
                mask,
                is_causal=False,
                scaling=8**-0.5,
                sliding_window=None,
            )

            self.assertIsNone(weights)
            self.assertEqual(out.shape, (1, 3, 4, 8))
        finally:
            try:
                ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
            except TypeError:
                ALL_ATTENTION_FUNCTIONS.register("sdpa", original_sdpa)

    def test_fast_dllm_rope_config_restores_reference_shape(self):
        config = SimpleNamespace(
            model_type="Fast_dLLM_Qwen",
            rope_scaling={"rope_type": "default", "rope_theta": 1000000.0},
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        )

        changed = _normalize_fast_dllm_v2_rope_config(config)

        self.assertTrue(changed)
        self.assertIsNone(config.rope_scaling)
        self.assertIsNone(config.rope_parameters)

    def test_fast_dllm_rope_config_preserves_non_default_scaling(self):
        config = SimpleNamespace(
            model_type="Fast_dLLM_Qwen",
            rope_scaling={"rope_type": "yarn", "factor": 2.0},
            rope_parameters={"rope_type": "yarn", "factor": 2.0},
        )

        changed = _normalize_fast_dllm_v2_rope_config(config)

        self.assertFalse(changed)
        self.assertEqual(config.rope_scaling["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")

    def test_fast_dllm_rotary_forward_patch_produces_non_identity_rope(self):
        import torch

        class Fast_dLLM_QwenRotaryEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    hidden_size=8,
                    num_attention_heads=2,
                    rope_theta=10000.0,
                )

            def forward(self, x, position_ids):
                return torch.ones(1, 4, 2), torch.zeros(1, 4, 2)

        model = torch.nn.Module()
        model.rotary_emb = Fast_dLLM_QwenRotaryEmbedding()

        patched = _patch_fast_dllm_v2_rotary_embedding_forward(model)
        cos, sin = model.rotary_emb(
            torch.zeros(1, 4, 8),
            torch.arange(4).reshape(1, 4),
        )
        expected_inv_freq = torch.tensor([1.0, 0.01])
        expected_freqs = torch.arange(4).reshape(1, 4, 1) * expected_inv_freq
        expected = torch.cat([expected_freqs, expected_freqs], dim=-1)

        self.assertTrue(patched)
        self.assertTrue(torch.allclose(cos, expected.cos(), atol=1e-6))
        self.assertTrue(torch.allclose(sin, expected.sin(), atol=1e-6))

    def test_dynamic_cache_shim_registers_list_backed_legacy_cache(self):
        import torch
        import transformers.cache_utils as cache_utils

        original_cache = cache_utils.DynamicCache

        try:
            _ensure_transformers_legacy_dynamic_cache_support()

            cache = cache_utils.DynamicCache()
            key_0 = torch.ones(1, 2, 3, 4)
            value_0 = torch.full((1, 2, 3, 4), 2.0)
            key_1 = torch.full((1, 2, 1, 4), 3.0)
            value_1 = torch.full((1, 2, 1, 4), 4.0)

            cache.update(key_0, value_0, 0)
            cache.update(key_1, value_1, 0)

            self.assertEqual(len(cache), 1)
            self.assertEqual(cache.seen_tokens, 4)
            self.assertEqual(cache.get_seq_length(0), 4)
            self.assertIsNone(cache.get_max_cache_shape())
            self.assertTrue(
                torch.equal(cache[0][0], torch.cat([key_0, key_1], dim=-2))
            )
            self.assertTrue(
                torch.equal(cache[0][1], torch.cat([value_0, value_1], dim=-2))
            )
            cache.crop(3)
            self.assertEqual(cache.get_seq_length(0), 3)
        finally:
            cache_utils.DynamicCache = original_cache


class TestLinearVerifyHelpers(unittest.TestCase):
    def test_fast_dllm_algorithm_reuses_linear_verify_contract(self):
        self.assertTrue(SpeculativeAlgorithm.FAST_DLLM_V2.uses_linear_verify())
        self.assertFalse(SpeculativeAlgorithm.FAST_DLLM_V2.is_dflash())

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
        self.assertTrue(verify_input.requires_target_hidden)
        self.assertEqual(verify_input.num_tokens_per_batch, 2)
        self.assertTrue(torch.equal(verify_input.draft_token, block.draft_token))
        self.assertTrue(torch.equal(verify_input.positions, block.positions))

    def test_builds_linear_verify_input_without_target_hidden(self):
        import torch

        block = LinearDraftBlock(
            draft_token=torch.tensor([1, 2]),
            positions=torch.tensor([7, 8]),
            draft_token_num=2,
        )

        verify_input = build_linear_verify_input(
            block,
            requires_target_hidden=False,
        )

        self.assertFalse(verify_input.requires_target_hidden)

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
