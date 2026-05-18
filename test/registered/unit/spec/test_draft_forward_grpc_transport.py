import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from sglang.srt.speculative.draft_forward_grpc_transport import (
    DraftForwardServiceAdapter,
    draft_request_from_proto,
    draft_request_to_proto,
    draft_response_from_proto,
    draft_response_to_proto,
    tensor_from_proto_tensor,
)
from sglang.srt.speculative.draft_forward_protocol import (
    DraftForwardRequest,
    DraftForwardResponse,
)
from sglang.srt.speculative.tli_token_translator import TLITokenTranslator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _TokenizerMock:
    def __init__(self, vocab, unk_token_id=0, eos_token_id=2, space_sign=None):
        self._vocab = dict(vocab)
        self.unk_token_id = unk_token_id
        self.eos_token_id = eos_token_id
        self._space_sign = space_sign

    def get_vocab(self):
        return self._vocab

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == " " and self._space_sign is not None:
            return {"input_ids": [0]}
        return {"input_ids": []}

    def convert_ids_to_tokens(self, ids):
        if self._space_sign is not None:
            return [f"{self._space_sign}token" for _ in ids]
        return ["token" for _ in ids]


def _make_translator(target_vocab, draft_vocab):
    return TLITokenTranslator(
        target_tokenizer=_TokenizerMock(target_vocab),
        draft_tokenizer=_TokenizerMock(draft_vocab),
        target_vocab_size=max(target_vocab.values()) + 1,
        draft_vocab_size=max(draft_vocab.values()) + 1,
        device=torch.device("cpu"),
    )


@dataclass
class _FakeTensorData:
    shape: list[int] = field(default_factory=list)
    dtype: str = ""
    data: bytes = b""


@dataclass
class _FakeDraftRequest:
    request_id: str = ""
    verified_id: _FakeTensorData | None = None
    hidden_states: _FakeTensorData | None = None
    request_ids: list[str] = field(default_factory=list)
    input_ids: _FakeTensorData | None = None
    tp_rank: int = 0
    tp_size: int = 1
    mode: str = "decode"
    capture_hidden_mode: int = 0
    topk: int = 1
    speculative_num_steps: int = 1
    speculative_num_draft_tokens: int = 1
    num_tokens_per_req: int = 1
    num_tokens_for_logprob_per_req: int = 1
    accept_length: _FakeTensorData | None = None
    accept_length_cpu: list[int] = field(default_factory=list)
    seq_lens_for_draft_extend: _FakeTensorData | None = None
    seq_lens_for_draft_extend_cpu: _FakeTensorData | None = None
    target_prefix_lens_for_draft_extend_cpu: _FakeTensorData | None = None
    mm_input_embeds: _FakeTensorData | None = None
    round_ids: list[int] = field(default_factory=list)
    token_positions: list[int] = field(default_factory=list)
    prefix_versions: list[int] = field(default_factory=list)
    cache_prefix_on_release: bool = False


@dataclass
class _FakeDraftResponse:
    request_id: str = ""
    parent_list: _FakeTensorData | None = None
    top_scores_index: _FakeTensorData | None = None
    draft_token_ids: _FakeTensorData | None = None
    mode: str = "decode"
    next_hidden_states: _FakeTensorData | None = None
    next_topk_p: _FakeTensorData | None = None
    next_topk_index: _FakeTensorData | None = None
    round_ids: list[int] = field(default_factory=list)
    token_positions: list[int] = field(default_factory=list)
    prefix_versions: list[int] = field(default_factory=list)


_FAKE_PROTO = SimpleNamespace(
    TensorData=_FakeTensorData,
    DraftForwardRequest=_FakeDraftRequest,
    DraftForwardResponse=_FakeDraftResponse,
)


class _FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class TestDraftForwardGrpcTransport(CustomTestCase):
    def setUp(self):
        self.translator = _make_translator(
            {"a": 0, "b": 1, "c": 2},
            {"a": 0, "b": 1, "d": 2},
        )

    def test_tensor_round_trip(self):
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        encoded = draft_request_to_proto(
            DraftForwardRequest(
                request_id="req-1",
                verified_id=tensor,
                hidden_states=torch.zeros(2, 3),
            ),
            proto_module=_FAKE_PROTO,
        )
        decoded = tensor_from_proto_tensor(encoded.verified_id)
        self.assertTrue(torch.equal(decoded, tensor))

    def test_response_proto_supports_empty_extend_tensors(self):
        response = DraftForwardResponse(
            request_id="req-empty-extend",
            mode="extend",
            parent_list=torch.empty((0,), dtype=torch.int64),
            top_scores_index=torch.empty((0,), dtype=torch.int64),
            draft_token_ids=torch.empty((0,), dtype=torch.int64),
            next_hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
            next_topk_p=torch.ones((1, 1), dtype=torch.float32),
            next_topk_index=torch.zeros((1, 1), dtype=torch.int64),
        )

        proto_response = draft_response_to_proto(
            response,
            proto_module=_FAKE_PROTO,
        )
        decoded = draft_response_from_proto(proto_response)

        self.assertEqual(decoded.request_id, response.request_id)
        self.assertEqual(decoded.mode, "extend")
        self.assertEqual(tuple(decoded.parent_list.shape), (0,))
        self.assertEqual(decoded.parent_list.dtype, torch.int64)
        self.assertEqual(tuple(decoded.top_scores_index.shape), (0,))
        self.assertEqual(decoded.top_scores_index.dtype, torch.int64)
        self.assertEqual(tuple(decoded.draft_token_ids.shape), (0,))
        self.assertEqual(decoded.draft_token_ids.dtype, torch.int64)
        self.assertTrue(
            torch.equal(decoded.next_hidden_states, response.next_hidden_states)
        )
        self.assertTrue(torch.equal(decoded.next_topk_p, response.next_topk_p))
        self.assertTrue(torch.equal(decoded.next_topk_index, response.next_topk_index))

    def test_request_proto_translation(self):
        request = DraftForwardRequest(
            request_id="req-1",
            mode="decode",
            verified_id=torch.tensor([0, 2, 1]),
            hidden_states=torch.zeros(3, 4),
            request_ids=["rid-0", "rid-1"],
            input_ids=torch.tensor([0, 1, 2]),
            tp_rank=1,
            tp_size=3,
            topk=2,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            accept_length=torch.tensor([1, 2, 3]),
            accept_length_cpu=[1, 2, 3],
            seq_lens_for_draft_extend_cpu=torch.tensor([4, 5]),
            target_prefix_lens_for_draft_extend_cpu=torch.tensor([1, 2]),
            round_ids=[7, 8],
            token_positions=[11, 12],
            prefix_versions=[13, 14],
            cache_prefix_on_release=True,
        )

        proto_request = draft_request_to_proto(
            request,
            translator=self.translator,
            proto_module=_FAKE_PROTO,
        )
        decoded = draft_request_from_proto(proto_request)

        self.assertEqual(decoded.request_id, "req-1")
        self.assertEqual(decoded.verified_id.tolist(), [0, 0, 1])
        self.assertEqual(decoded.request_ids, ["rid-0", "rid-1"])
        self.assertEqual(decoded.input_ids.tolist(), [0, 1, 0])
        self.assertEqual(decoded.tp_rank, 1)
        self.assertEqual(decoded.tp_size, 3)
        self.assertEqual(decoded.accept_length_cpu, [1, 2, 3])
        self.assertEqual(decoded.seq_lens_for_draft_extend_cpu.tolist(), [4, 5])
        self.assertEqual(
            decoded.target_prefix_lens_for_draft_extend_cpu.tolist(),
            [1, 2],
        )
        self.assertEqual(decoded.round_ids, [7, 8])
        self.assertEqual(decoded.token_positions, [11, 12])
        self.assertEqual(decoded.prefix_versions, [13, 14])
        self.assertTrue(decoded.cache_prefix_on_release)

    def test_response_proto_translation(self):
        response = DraftForwardResponse(
            request_id="req-1",
            parent_list=torch.tensor([[0, 1], [1, 0]]),
            top_scores_index=torch.tensor([[1, 0], [0, 1]]),
            draft_token_ids=torch.tensor([0, 1, 2]),
            next_hidden_states=torch.zeros(3, 4),
            round_ids=[5, 6],
            token_positions=[8, 9],
            prefix_versions=[13, 14],
        )

        proto_response = draft_response_to_proto(
            response,
            proto_module=_FAKE_PROTO,
        )
        decoded = draft_response_from_proto(
            proto_response,
            translator=self.translator,
            translate_to_target_vocab=True,
        )

        self.assertEqual(decoded.request_id, "req-1")
        self.assertEqual(decoded.draft_token_ids.tolist(), [0, 1, 0])
        self.assertTrue(torch.equal(decoded.parent_list, response.parent_list))
        self.assertTrue(
            torch.equal(decoded.top_scores_index, response.top_scores_index)
        )
        self.assertEqual(decoded.round_ids, [5, 6])
        self.assertEqual(decoded.token_positions, [8, 9])
        self.assertEqual(decoded.prefix_versions, [13, 14])

    def test_service_adapter_round_trip(self):
        async def handler(request):
            self.assertEqual(request.verified_id.tolist(), [0, 0, 1])
            return DraftForwardResponse(
                request_id=request.request_id,
                parent_list=torch.tensor([[0, 1]]),
                top_scores_index=torch.tensor([[1, 0]]),
                draft_token_ids=torch.tensor([0, 1, 2]),
            )

        adapter = DraftForwardServiceAdapter(
            handler,
            translator=self.translator,
            proto_module=_FAKE_PROTO,
            translate_requests_to_draft_vocab=False,
            translate_responses_to_target_vocab=False,
        )

        proto_request = draft_request_to_proto(
            DraftForwardRequest(
                request_id="req-1",
                mode="decode",
                verified_id=torch.tensor([0, 2, 1]),
                hidden_states=torch.zeros(3, 4),
            ),
            translator=self.translator,
            proto_module=_FAKE_PROTO,
        )
        response_proto = asyncio.run(
            adapter.DraftForward(proto_request, _FakeContext())
        )
        response = draft_response_from_proto(
            response_proto,
            translator=self.translator,
            translate_to_target_vocab=True,
        )
        self.assertEqual(response.draft_token_ids.tolist(), [0, 1, 0])

    def test_service_adapter_failure_raises(self):
        async def handler(_request):
            raise RuntimeError("boom")

        adapter = DraftForwardServiceAdapter(
            handler,
            proto_module=_FAKE_PROTO,
        )

        proto_request = draft_request_to_proto(
            DraftForwardRequest(
                request_id="req-err",
                mode="decode",
                verified_id=torch.tensor([0]),
                hidden_states=torch.zeros(1, 2),
            ),
            proto_module=_FAKE_PROTO,
        )
        with self.assertRaisesRegex(RuntimeError, "boom"):
            asyncio.run(adapter.DraftForward(proto_request, _FakeContext()))

    def test_service_adapter_stream_completes_requests_out_of_order(self):
        async def handler(request):
            if request.request_id == "slow":
                await asyncio.sleep(0.02)
            return DraftForwardResponse(
                request_id=request.request_id,
                parent_list=torch.tensor([[0, 1]]),
                top_scores_index=torch.tensor([[1, 0]]),
                draft_token_ids=torch.tensor([0, 1]),
                round_ids=request.round_ids,
                token_positions=request.token_positions,
                prefix_versions=request.prefix_versions,
            )

        async def request_stream():
            for request_id, round_id in (("slow", 1), ("fast", 2)):
                yield draft_request_to_proto(
                    DraftForwardRequest(
                        request_id=request_id,
                        mode="decode",
                        verified_id=torch.tensor([round_id]),
                        hidden_states=torch.zeros(1, 2),
                        round_ids=[round_id],
                        token_positions=[round_id * 10],
                        prefix_versions=[round_id * 100],
                    ),
                    proto_module=_FAKE_PROTO,
                )

        async def collect():
            adapter = DraftForwardServiceAdapter(
                handler,
                proto_module=_FAKE_PROTO,
            )
            return [
                draft_response_from_proto(response)
                async for response in adapter.DraftForwardStream(
                    request_stream(),
                    _FakeContext(),
                )
            ]

        responses = asyncio.run(collect())

        self.assertEqual(
            [response.request_id for response in responses], ["fast", "slow"]
        )
        self.assertEqual([response.round_ids for response in responses], [[2], [1]])
        self.assertEqual(
            [response.token_positions for response in responses],
            [[20], [10]],
        )
        self.assertEqual(
            [response.prefix_versions for response in responses],
            [[200], [100]],
        )

    def test_service_adapter_stream_micro_batches_compatible_requests(self):
        seen_requests = []

        async def handler(request):
            seen_requests.append(request)
            return DraftForwardResponse(
                request_id=request.request_id,
                parent_list=torch.tensor([[0, 1], [2, 3]]),
                top_scores_index=torch.tensor([[1, 0], [0, 1]]),
                draft_token_ids=torch.tensor([[11, 12], [21, 22]]),
                round_ids=request.round_ids,
                token_positions=request.token_positions,
                prefix_versions=request.prefix_versions,
            )

        async def request_stream():
            for request_id, round_id in (("req-a", 1), ("req-b", 2)):
                yield draft_request_to_proto(
                    DraftForwardRequest(
                        request_id=request_id,
                        request_ids=[request_id],
                        mode="decode",
                        verified_id=torch.tensor([round_id]),
                        hidden_states=torch.zeros(1, 2),
                        round_ids=[round_id],
                        token_positions=[round_id * 10],
                        prefix_versions=[round_id * 100],
                    ),
                    proto_module=_FAKE_PROTO,
                )

        async def collect():
            adapter = DraftForwardServiceAdapter(
                handler,
                proto_module=_FAKE_PROTO,
                stream_batch_window_s=0.001,
                stream_batch_max_requests=2,
                stream_batch_max_proposed_tokens=4,
            )
            return [
                draft_response_from_proto(response)
                async for response in adapter.DraftForwardStream(
                    request_stream(),
                    _FakeContext(),
                )
            ]

        responses = asyncio.run(collect())

        self.assertEqual(len(seen_requests), 1)
        self.assertEqual(seen_requests[0].request_ids, ["req-a", "req-b"])
        self.assertEqual(
            [response.request_id for response in responses],
            ["req-a", "req-b"],
        )
        self.assertEqual(responses[0].draft_token_ids.tolist(), [[11, 12]])
        self.assertEqual(responses[1].draft_token_ids.tolist(), [[21, 22]])
        self.assertEqual([response.round_ids for response in responses], [[1], [2]])

    def test_service_adapter_stream_flushes_at_proposed_token_cap(self):
        seen_requests = []

        async def handler(request):
            seen_requests.append(request)
            token_id = 100 + len(seen_requests)
            return DraftForwardResponse(
                request_id=request.request_id,
                parent_list=torch.tensor([[0]]),
                top_scores_index=torch.tensor([[0]]),
                draft_token_ids=torch.tensor([[token_id]]),
                round_ids=request.round_ids,
                token_positions=request.token_positions,
                prefix_versions=request.prefix_versions,
            )

        async def request_stream():
            for request_id, round_id in (("req-a", 1), ("req-b", 2)):
                yield draft_request_to_proto(
                    DraftForwardRequest(
                        request_id=request_id,
                        request_ids=[request_id],
                        mode="decode",
                        verified_id=torch.tensor([round_id]),
                        hidden_states=torch.zeros(1, 2),
                        speculative_num_draft_tokens=4,
                        round_ids=[round_id],
                        token_positions=[round_id],
                        prefix_versions=[round_id],
                    ),
                    proto_module=_FAKE_PROTO,
                )

        async def collect():
            adapter = DraftForwardServiceAdapter(
                handler,
                proto_module=_FAKE_PROTO,
                stream_batch_window_s=0.1,
                stream_batch_max_requests=2,
                stream_batch_max_proposed_tokens=4,
            )
            return [
                draft_response_from_proto(response)
                async for response in adapter.DraftForwardStream(
                    request_stream(),
                    _FakeContext(),
                )
            ]

        responses = asyncio.run(collect())

        self.assertEqual(len(seen_requests), 2)
        self.assertEqual(
            [request.request_ids for request in seen_requests],
            [["req-a"], ["req-b"]],
        )
        self.assertEqual(
            [response.request_id for response in responses],
            ["req-a", "req-b"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
