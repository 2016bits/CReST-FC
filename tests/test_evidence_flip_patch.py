import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from crestfc.evidence_flip_patch import (
    LABELS,
    candidate_batch,
    clean_forward,
    margin,
    patched_scores,
    prediction,
    review_approved,
    select_random_donors,
)


def _example(pair_id, family, claim, support_evidence, refute_evidence):
    def row(label, text):
        return {
            "claim": claim,
            "label": label,
            "evidence": [{"evidence_id": "E1", "text": text}],
            "source_provenance": {"revision_family_id": family},
        }

    return {
        "pair_id": pair_id,
        "claim": claim,
        "supports": row("SUPPORTS", support_evidence),
        "refutes": row("REFUTES", refute_evidence),
    }


class EvidenceFlipPatchTests(unittest.TestCase):
    def test_review_gate_and_random_donor_independence(self):
        fields = {
            "approved": True,
            "reviewer": "reviewer-1",
            "claim_unchanged": True,
            "evidence_changed": True,
            "supports_label_correct": True,
            "refutes_label_correct": True,
            "evidence_sufficient": True,
        }
        self.assertTrue(review_approved(fields))
        self.assertFalse(review_approved({**fields, "reviewer": ""}))
        pairs = [
            _example("a", "fa", "Claim A", "short true", "short false"),
            _example("b", "fb", "Claim B", "medium true text", "medium false text"),
            _example("c", "fc", "Claim C", "long true text here", "long false text here"),
        ]
        mapping = select_random_donors(pairs, 42)
        by_id = {pair["pair_id"]: pair for pair in pairs}
        for pair in pairs:
            donor = by_id[mapping[pair["pair_id"]]]
            self.assertNotEqual(pair["pair_id"], donor["pair_id"])
            self.assertNotEqual(pair["claim"], donor["claim"])

    def test_identity_patch_reproduces_clean_score(self):
        torch.manual_seed(4)
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        model = Qwen2ForCausalLM(config).eval()
        prefix = list(range(1, 25))
        labels = [[50, 51], [52, 53]]
        positions = {"evidence_end": 8, "relation_anchor": 12, "answer_pre": 23}
        with torch.inference_mode():
            clean, states = clean_forward(model, prefix, positions, labels, [0, 1])
            patched = patched_scores(
                model, prefix, labels, 1, positions["relation_anchor"],
                states[1]["relation_anchor"],
            )
        self.assertTrue(torch.allclose(clean, patched, atol=2e-5))
        self.assertIn(prediction(clean), LABELS)
        self.assertAlmostEqual(margin(clean), float(clean[0] - clean[1]), places=6)
        ids, attention, length = candidate_batch(model, prefix, labels)
        self.assertEqual(tuple(ids.shape), tuple(attention.shape))
        self.assertEqual(length, len(prefix))


if __name__ == "__main__":
    unittest.main()
