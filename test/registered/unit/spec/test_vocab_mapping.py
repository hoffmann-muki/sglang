import unittest

import torch

from sglang.srt.speculative.vocab_mapping import (
    VocabMapping,
    _detect_space_sign,
    _normalize_token,
)
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


def _make_mapping(target_vocab, draft_vocab, target_space_sign=None, draft_space_sign=None):
    target_tokenizer = _TokenizerMock(target_vocab, space_sign=target_space_sign)
    draft_tokenizer = _TokenizerMock(draft_vocab, space_sign=draft_space_sign)
    return VocabMapping(
        target_tokenizer=target_tokenizer,
        draft_tokenizer=draft_tokenizer,
        target_vocab_size=max(target_vocab.values()) + 1,
        draft_vocab_size=max(draft_vocab.values()) + 1,
        device=torch.device("cpu"),
    )


class TestTokenNormalization(CustomTestCase):
    def test_detect_space_sign(self):
        tokenizer = _TokenizerMock({"a": 0}, space_sign="\u0120")
        self.assertEqual(_detect_space_sign(tokenizer), "\u0120")

    def test_normalize_token(self):
        self.assertEqual(_normalize_token("\u2581hello"), " hello")
        self.assertEqual(_normalize_token("word", space_sign="\u2581"), "word")


class TestVocabMapping(CustomTestCase):
    def setUp(self):
        self.mapping = _make_mapping(
            {"a": 0, "b": 1, "c": 2},
            {"a": 0, "b": 1, "d": 2},
        )

    def test_intersection(self):
        self.assertEqual(self.mapping.intersection_size, 2)
        self.assertTrue(self.mapping.intersection_mask_draft[0].item())
        self.assertTrue(self.mapping.intersection_mask_draft[1].item())

    def test_map_target_to_draft(self):
        result = self.mapping.map_target_to_draft_ids(torch.tensor([0, 2, 1]))
        self.assertEqual(result.tolist(), [0, 0, 1])

    def test_map_draft_to_target(self):
        result = self.mapping.map_draft_to_target_ids(torch.tensor([0, 2, 1]))
        self.assertEqual(result.tolist(), [0, 0, 1])

    def test_constrain_logits(self):
        logits = torch.zeros(1, 3)
        constrained = self.mapping.constrain_draft_logits(logits)
        self.assertTrue(torch.isinf(constrained[0, 2]))
        self.assertFalse(torch.isinf(constrained[0, 0]))

    def test_eos_fallback(self):
        mapping = VocabMapping(
            target_tokenizer=_TokenizerMock({"a": 0}, unk_token_id=None, eos_token_id=9),
            draft_tokenizer=_TokenizerMock({"a": 0}, unk_token_id=None, eos_token_id=8),
            target_vocab_size=1,
            draft_vocab_size=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(mapping.target_unk_token_id, 9)
        self.assertEqual(mapping.draft_unk_token_id, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)