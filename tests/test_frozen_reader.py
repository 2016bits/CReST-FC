import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from crestfc.frozen_reader import (capture_all, classify, decoder_prefix, encoder_text,
                                   patch_block, score_labels, score_layer_batch, validate_pairs)
from crestfc.vitaminc_claimflip import build_claimflip_pairs, shard_for_family


class _ToyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [7, 8, 9]


class FrozenReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128)
        self.model = Qwen2ForCausalLM(config).eval().requires_grad_(False)

    def test_capture_identity_patch_and_cleanup(self):
        prefix = [1, 2, 3, 4]
        captured = capture_all(self.model, torch.tensor([prefix]))
        self.assertEqual(set(captured), {0, 1})
        self.assertEqual(captured[0].shape, (1, 4, 32))
        candidates = [[5, 6], [7]]
        original, _ = score_labels(self.model, prefix, candidates)
        patched, _ = score_labels(self.model, prefix, candidates, captured[0], 0)
        self.assertTrue(all(abs(a-b) < 1e-5 for a, b in zip(original, patched)))
        self.assertEqual(len(self.model.model.layers[0]._forward_hooks), 0)
        with self.assertRaises(ValueError):
            with patch_block(self.model, 0, torch.zeros(1, 200, 32)):
                self.model.model(input_ids=torch.tensor([prefix]), use_cache=False)
        self.assertEqual(len(self.model.model.layers[0]._forward_hooks), 0)

    def test_label_sequence_shift_and_slots(self):
        prefix, answers = [1, 2, 3], [[4, 5], [6]]
        scores, means = score_labels(self.model, prefix, answers)
        with torch.inference_mode():
            ids = torch.tensor([prefix + answers[0]])
            logits = self.model(input_ids=ids, use_cache=False).logits.float()
            reference = (logits[0, 2].log_softmax(-1)[4] +
                         logits[0, 3].log_softmax(-1)[5]).item()
        self.assertAlmostEqual(scores[0], reference, places=5)
        self.assertAlmostEqual(means[0], reference / 2, places=5)
        self.assertEqual(decoder_prefix(_ToyTokenizer(), 3, 12), [12, 12, 12, 7, 8, 9])

    def test_batched_layers_match_separate_scoring(self):
        prefix = [1, 2, 3, 4]
        states = capture_all(self.model, torch.tensor([prefix]))
        candidates = [[5, 6], [7]]
        expected = [score_labels(self.model, prefix, candidates, states[layer])[0]
                    for layer in (0, 1)]
        actual = score_layer_batch(self.model, prefix, candidates,
                                   torch.cat([states[0], states[1]], dim=0))
        for pair_a, pair_b in zip(expected, actual):
            for a, b in zip(pair_a, pair_b):
                self.assertAlmostEqual(a, b, places=5)

    def test_pair_gate_and_ties(self):
        a = {"sample_id": "s", "claim": "A", "evidence": "yes", "label": "SUPPORTS", "split": "dev"}
        b = {"sample_id": "r", "claim": "A", "evidence": "no", "label": "REFUTES", "split": "dev"}
        pair = {"pair_id": "p", "claim": "A", "supports": a, "refutes": b}
        with self.assertRaises(ValueError):
            validate_pairs([pair], "dev", {}, False)
        selected, audit = validate_pairs([pair], "dev", {}, True)
        self.assertEqual(audit["selected_pairs"], 1)
        self.assertNotIn("yes", encoder_text(a, "claim_only_patch"))
        self.assertEqual(classify([1.0, 1.0])[0], "TIE")

    def test_claimflip_requires_same_family_and_exact_evidence(self):
        rows = [
            {"id": "family_a_1", "claim": "A is true", "gold_evidence": "Evidence X", "label": "SUPPORTS"},
            {"id": "family_a_2", "claim": "A is false", "gold_evidence": "Evidence X", "label": "REFUTES"},
            {"id": "family_a_3", "claim": "A is false", "gold_evidence": "Evidence Y", "label": "REFUTES"},
            {"id": "family_b_1", "claim": "B is false", "gold_evidence": "Evidence X", "label": "REFUTES"},
        ]
        pairs, audit = build_claimflip_pairs(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["supports_id"], pairs[0]["refutes_id"]),
                         ("family_a_1", "family_a_2"))
        self.assertEqual(shard_for_family("family_a", 5), shard_for_family("family_a", 5))


if __name__ == "__main__":
    unittest.main()
