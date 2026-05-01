import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from sglang.srt.speculative.tli_grpc_transport import (
    TliSpeculativeServiceAdapter,
    draft_request_from_proto,
    draft_request_to_proto,
    draft_response_from_proto,
    draft_response_to_proto,
    tensor_from_proto_tensor,
)
from sglang.srt.speculative.tli_protocol import TLIDraftRequest, TLIDraftResponse
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
    mm_input_embeds: _FakeTensorData | None = None


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


_FAKE_PROTO = SimpleNamespace(
    TensorData=_FakeTensorData,
    TliDraftRequest=_FakeDraftRequest,
    TliDraftResponse=_FakeDraftResponse,
)


class _FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class TestTLIGRPCTransport(CustomTestCase):
    def setUp(self):
        self.translator = _make_translator(
            {"a": 0, "b": 1, "c": 2},
            {"a": 0, "b": 1, "d": 2},
        )

    def test_tensor_round_trip(self):
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        encoded = draft_request_to_proto(
            TLIDraftRequest(
                request_id="req-1",
                verified_id=tensor,
                hidden_states=torch.zeros(2, 3),
            ),
            proto_module=_FAKE_PROTO,
        )
        decoded = tensor_from_proto_tensor(encoded.verified_id)
        self.assertTrue(torch.equal(decoded, tensor))

    def test_request_proto_translation(self):
        request = TLIDraftRequest(
            request_id="req-1",
            mode="decode",
            verified_id=torch.tensor([0, 2, 1]),
            hidden_states=torch.zeros(3, 4),
            topk=2,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            accept_length=torch.tensor([1, 2, 3]),
            accept_length_cpu=[1, 2, 3],
        )

        proto_request = draft_request_to_proto(
            request,
            translator=self.translator,
            proto_module=_FAKE_PROTO,
        )
        decoded = draft_request_from_proto(proto_request)

        self.assertEqual(decoded.request_id, "req-1")
        self.assertEqual(decoded.verified_id.tolist(), [0, 0, 1])
        self.assertEqual(decoded.accept_length_cpu, [1, 2, 3])

    def test_response_proto_translation(self):
        response = TLIDraftResponse(
            request_id="req-1",
            parent_list=torch.tensor([[0, 1], [1, 0]]),
            top_scores_index=torch.tensor([[1, 0], [0, 1]]),
            draft_token_ids=torch.tensor([0, 1, 2]),
            next_hidden_states=torch.zeros(3, 4),
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

    def test_service_adapter_round_trip(self):
        async def handler(request):
            self.assertEqual(request.verified_id.tolist(), [0, 0, 1])
            return TLIDraftResponse(
                request_id=request.request_id,
                parent_list=torch.tensor([[0, 1]]),
                top_scores_index=torch.tensor([[1, 0]]),
                draft_token_ids=torch.tensor([0, 1, 2]),
            )

        adapter = TliSpeculativeServiceAdapter(
            handler,
            translator=self.translator,
            translate_requests_to_draft_vocab=False,
            translate_responses_to_target_vocab=False,
        )

        proto_request = draft_request_to_proto(
            TLIDraftRequest(
                request_id="req-1",
                mode="decode",
                verified_id=torch.tensor([0, 2, 1]),
                hidden_states=torch.zeros(3, 4),
            ),
            translator=self.translator,
            proto_module=_FAKE_PROTO,
        )
        response_proto = asyncio.run(adapter.DraftForward(proto_request, _FakeContext()))
        response = draft_response_from_proto(
            response_proto,
            translator=self.translator,
            translate_to_target_vocab=True,
        )
        self.assertEqual(response.draft_token_ids.tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
