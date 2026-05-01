import unittest

import torch

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


class TestTLIProtocol(CustomTestCase):
    def setUp(self):
        self.translator = _make_translator(
            {"a": 0, "b": 1, "c": 2},
            {"a": 0, "b": 1, "d": 2},
        )

    def test_request_translation(self):
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

        translated = request.to_draft_vocab(self.translator)

        self.assertEqual(translated.request_id, "req-1")
        self.assertEqual(translated.verified_id.tolist(), [0, 0, 1])
        self.assertEqual(translated.topk, 2)
        self.assertEqual(translated.speculative_num_steps, 3)
        self.assertEqual(translated.speculative_num_draft_tokens, 4)
        self.assertEqual(translated.accept_length_cpu, [1, 2, 3])

    def test_response_translation(self):
        response = TLIDraftResponse(
            request_id="req-1",
            parent_list=torch.tensor([[0, 1], [1, 0]]),
            top_scores_index=torch.tensor([[1, 0], [0, 1]]),
            draft_token_ids=torch.tensor([0, 1, 2]),
            next_hidden_states=torch.zeros(3, 4),
        )

        translated = response.to_target_vocab(self.translator)

        self.assertEqual(translated.request_id, "req-1")
        self.assertEqual(translated.draft_token_ids.tolist(), [0, 1, 0])
        self.assertTrue(torch.equal(translated.parent_list, response.parent_list))
        self.assertTrue(
            torch.equal(translated.top_scores_index, response.top_scores_index)
        )
        self.assertTrue(
            torch.equal(translated.next_hidden_states, response.next_hidden_states)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
