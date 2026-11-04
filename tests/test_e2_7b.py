import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crestfc.e2_7b import candidate_ids, probe_axis_patch, reviewed, score_labels, source_margin
from crestfc.modeling import get_transformer_blocks, make_patch_hook, score_candidate_sequences


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [1 + ord(char) % 63 for char in text]
        data = torch.tensor([ids]) if return_tensors == "pt" else ids
        return {"input_ids": data}


class E2Tests(unittest.TestCase):
    def test_batched_scores_match_reference_with_and_without_patch(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128)
        model = Qwen2ForCausalLM(config).eval()
        tokenizer = CharacterTokenizer()
        prefix = "Claim: a\nEvidence: b\nRelation anchor:\nAnswer:"
        labels = candidate_ids(tokenizer)
        with torch.inference_mode():
            fast = score_labels(model, tokenizer, prefix, labels)
            reference = score_candidate_sequences(model, tokenizer, prefix,
                                                  candidates=("SUPPORTS", "REFUTES"))[0]
            self.assertTrue(torch.allclose(fast, reference, atol=2e-5))
            ids = tokenizer(prefix)["input_ids"]
            token_index = len(ids) - 10
            state = model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
            vector = state[0, token_index].clone()
            same = score_labels(model, tokenizer, prefix, labels, (1, token_index, vector))
            self.assertTrue(torch.allclose(fast, same, atol=2e-5))
            altered = vector + 0.2
            fast_patch = score_labels(model, tokenizer, prefix, labels, (1, token_index, altered))
            hook = make_patch_hook(token_index, altered.unsqueeze(0))
            reference_patch = score_candidate_sequences(model, tokenizer, prefix,
                candidates=("SUPPORTS", "REFUTES"),
                hook_block=get_transformer_blocks(model)[1], hook=hook)[0]
            self.assertTrue(torch.allclose(fast_patch, reference_patch, atol=2e-5))

    def test_probe_axis_decomposition_and_review_gate(self):
        source = torch.tensor([2., 3., -1.])
        target = torch.tensor([1., -2., 1.])
        probe = {"coef": np.array([2., 0., 0.]), "scale": np.ones(3)}
        axis, orthogonal = probe_axis_patch(source, target, probe)
        self.assertTrue(torch.allclose(axis, torch.tensor([2., -2., 1.])))
        self.assertTrue(torch.allclose(orthogonal, torch.tensor([1., 3., -1.])))
        self.assertAlmostEqual(source_margin(torch.tensor([-2., -1.]), "REFUTES"), 1.)
        fields = {"approved": True, "reviewer": "human", "claim_unchanged": True,
                  "evidence_changed": True, "supports_label_correct": True,
                  "refutes_label_correct": True, "evidence_sufficient": True}
        self.assertTrue(reviewed({}, fields))
        self.assertFalse(reviewed({}, {**fields, "evidence_sufficient": False}))


if __name__ == "__main__":
    unittest.main()
