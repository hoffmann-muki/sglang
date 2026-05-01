import unittest

import torch

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


class TestTLITokenTranslator(CustomTestCase):
    def setUp(self):
        self.translator = _make_translator(
            {"a": 0, "b": 1, "c": 2},
            {"a": 0, "b": 1, "d": 2},
        )

    def test_translate_tensor_ids(self):
        draft = self.translator.translate_target_to_draft_ids(torch.tensor([0, 2, 1]))
        target = self.translator.translate_draft_to_target_ids(torch.tensor([0, 2, 1]))
        self.assertEqual(draft.tolist(), [0, 0, 1])
        self.assertEqual(target.tolist(), [0, 0, 1])

    def test_translate_list_ids(self):
        draft = self.translator.translate_target_to_draft_list([0, 2, 1])
        target = self.translator.translate_draft_to_target_list([0, 2, 1])
        self.assertEqual(draft, [0, 0, 1])
        self.assertEqual(target, [0, 0, 1])

    def test_backward_compatible_map_api(self):
        result = self.translator.map_target_to_draft_ids(torch.tensor([0, 2, 1]))
        self.assertEqual(result.tolist(), [0, 0, 1])

    def test_constrain_logits(self):
        logits = torch.zeros(1, 3)
        constrained = self.translator.constrain_draft_logits(logits)
        self.assertTrue(torch.isinf(constrained[0, 2]))
        self.assertFalse(torch.isinf(constrained[0, 0]))

    def test_state_round_trip(self):
        state = self.translator.to_state()
        loaded = TLITokenTranslator.from_state(state, device=torch.device("cpu"))
        out = loaded.translate_target_to_draft_ids(torch.tensor([0, 2, 1]))
        self.assertEqual(out.tolist(), [0, 0, 1])
        self.assertEqual(loaded.intersection_size, self.translator.intersection_size)

    def test_state_is_a_snapshot(self):
        state = self.translator.to_state()
        state["target_to_draft_ids"][0] = 99
        out = self.translator.translate_target_to_draft_ids(torch.tensor([0]))
        self.assertEqual(out.tolist(), [0])

    def test_missing_state_key_raises(self):
        state = self.translator.to_state()
        state.pop("target_to_draft_ids")
        with self.assertRaises(ValueError):
            TLITokenTranslator.from_state(state, device=torch.device("cpu"))

    def test_bad_state_shape_raises(self):
        state = self.translator.to_state()
        state["target_to_draft_ids"] = torch.zeros(10, dtype=torch.long)
        with self.assertRaises(ValueError):
            TLITokenTranslator.from_state(state, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
