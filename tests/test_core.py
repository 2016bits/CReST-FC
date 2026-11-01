import json
import unittest
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from crestfc.data import Example, normalize_label, validate_dataset, build_fever_wiki_index, adapt
from crestfc.evaluation import pair_metrics, clustered_bootstrap, parse_structured_output, evidence_prf
from crestfc.modeling import make_patch_hook, temporary_hook, label_margin, last_nonpadding, score_candidate_sequences, render_verdict
from crestfc.training import js_loss, flip_loss


class CoreTests(unittest.TestCase):
    def sample(self, sid, label, evidence, kind="none", role=None, metadata=None):
        return vars(Example(sample_id=sid, group_id="g", source_dataset="toy", source_id=sid,
                            split="train", claim="Acme was founded in 2010.",
                            evidence=[{"evidence_id": "E1", "text": evidence}], label=label,
                            pair_id="p" if kind != "none" else None, pair_kind=kind,
                            pair_role=role, claim_metadata=metadata or {},
                            source_provenance={"source_file": "toy", "source_sha256": "x"},
                            counterfactual_verified=kind != "none"))

    def test_ambiguous_label_rejected(self):
        with self.assertRaises(ValueError):
            normalize_label("NOT_SUPPORTED")
        self.assertEqual(normalize_label("supported"), "SUPPORTS")

    def test_e0_verdict_prompt(self):
        row = self.sample("a", "SUPPORTS", "Acme began in 2010.")
        expected = ("You are a fact-checking assistant.\n\n"
                    "Determine whether the evidence supports or refutes\n"
                    "the claim, using only the provided evidence.\n\n"
                    "Labels:\n"
                    "SUPPORTS: The evidence entails the claim.\n"
                    "REFUTES: The evidence contradicts the claim.\n"
                    "NEI: The evidence is insufficient to determine\n"
                    "whether the claim is supported or refuted.\n\n"
                    "Return exactly one label.\n\n"
                    "Claim: Acme was founded in 2010.\n"
                    "Evidence: [E1] Acme began in 2010.\n\nAnswer:")
        self.assertEqual(render_verdict(row), expected)
        self.assertIn("Evidence: [not provided]", render_verdict(row, "claim_only"))
        self.assertIn("Claim: [not provided]", render_verdict(row, "evidence_only"))

    def test_pair_invariants_and_split_leakage(self):
        a = self.sample("a", "SUPPORTS", "Acme was founded in 2010.", "relation_change", "A")
        b = self.sample("b", "REFUTES", "Acme was founded in 2012.", "relation_change", "B")
        pair = {"pair_id": "p", "sample_a": "a", "sample_b": "b", "kind": "relation_change", "audit_status": "approved"}
        self.assertEqual(validate_dataset([a, b], [pair], True)["n_pairs"], 1)
        b["split"] = "dev"
        with self.assertRaises(ValueError):
            validate_dataset([a, b], [pair])

    def test_pair_metrics_correctness(self):
        a = self.sample("a", "SUPPORTS", "Acme was founded in 2010.", "bias_change", "A", {"claim_publisher": "A"})
        b = self.sample("b", "SUPPORTS", "Acme was founded in 2010.", "bias_change", "B", {"claim_publisher": "B"})
        a["split"] = b["split"] = "dev"
        p = {"pair_id": "p", "sample_a": "a", "sample_b": "b", "kind": "bias_change", "audit_status": "approved"}
        m = pair_metrics([a, b], [p], {"a": "REFUTES", "b": "REFUTES"}, "dev")
        self.assertEqual(m["bias_consistency"], 1)
        self.assertEqual(m["bias_pair_correct"], 0)

    def test_patch_cleanup_tuple_and_identity(self):
        class Block(nn.Module):
            def forward(self, x):
                return x + 1, "cache"
        block = Block()
        x = torch.zeros(1, 3, 4)
        source = torch.ones(1, 4) * 7
        with temporary_hook(block, make_patch_hook(1, source)):
            y, cache = block(x)
            self.assertEqual(cache, "cache")
            self.assertTrue(torch.equal(y[0, 1], source[0]))
        y2, _ = block(x)
        self.assertEqual(float(y2[0, 1, 0]), 1.0)

    def test_gradients_and_losses(self):
        encoder = nn.Linear(3, 3)
        decoder = nn.Linear(3, 2)
        decoder.requires_grad_(False)
        z = encoder(torch.randn(1, 3))
        scores = decoder(z)
        loss = nn.functional.cross_entropy(scores, torch.tensor([1]))
        loss.backward()
        self.assertGreater(encoder.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(decoder.weight.grad)
        a = torch.tensor([[2., -2.]])
        b = torch.tensor([[-2., 2.]])
        self.assertAlmostEqual(js_loss(a, a).item(), 0, places=6)
        self.assertLess(flip_loss(a, b, "SUPPORTS", "REFUTES", ["SUPPORTS", "REFUTES"]).item(),
                        flip_loss(b, a, "SUPPORTS", "REFUTES", ["SUPPORTS", "REFUTES"]).item())

    def test_multi_token_teacher_forcing(self):
        class CharTokenizer:
            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                ids = [ord(c) for c in text]
                return {"input_ids": torch.tensor([ids]) if return_tensors else ids}
        class CausalToy(nn.Module):
            device = torch.device("cpu")
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(0.1))
            def forward(self, input_ids, use_cache=False):
                positions = torch.arange(input_ids.shape[1]).view(1, -1, 1)
                vocab = torch.arange(128).view(1, 1, -1)
                logits = self.bias * positions * vocab
                return type("Output", (), {"logits": logits})()
        model, tok = CausalToy(), CharTokenizer()
        scores = score_candidate_sequences(model, tok, "Q:", ["A", "BB"])
        self.assertEqual(tuple(scores.shape), (1, 2))
        prefix = tok("Q:", return_tensors="pt")["input_ids"]
        answer = tok(" BB", return_tensors="pt")["input_ids"]
        logits = model(torch.cat([prefix, answer], 1)).logits
        manual = logits[:, 1:4].log_softmax(-1).gather(-1, answer.unsqueeze(-1)).sum()
        self.assertTrue(torch.allclose(scores[0, 1], manual))
        scores.sum().backward()
        self.assertIsNotNone(model.bias.grad)

    def test_structured_output_rejects_invented_evidence(self):
        good = '{"verdict":"SUPPORTS","evidence_ids":["E1"],"justification":"E1 states the year."}'
        parsed = parse_structured_output(good, {"E1"}, ["SUPPORTS", "REFUTES"])
        self.assertTrue(parsed["valid"])
        bad = good.replace('"E1"', '"E9"', 1)
        self.assertEqual(parse_structured_output(bad, {"E1"}, ["SUPPORTS", "REFUTES"])["verdict"], "INVALID")
        self.assertEqual(evidence_prf({"E1"}, {"E1"})["f1"], 1.0)

    def test_padding_and_cluster_bootstrap(self):
        mask = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])
        self.assertEqual(last_nonpadding(mask).tolist(), [2, 1])
        rows = [self.sample("a", "SUPPORTS", "x"), self.sample("b", "SUPPORTS", "x")]
        ci = clustered_bootstrap(rows, {"a": 1., "b": 0.}, n_boot=20)
        self.assertEqual(ci["n_groups"], 1)
        self.assertEqual(ci["ci95"], [0.5, 0.5])

    def test_fever_indexed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wiki = root / "wiki"
            wiki.mkdir()
            (wiki / "pages.jsonl").write_text(json.dumps({"id": "Acme", "lines": "0\tAcme was founded in 2010.\n1\tIt is a company."}) + "\n")
            db = root / "wiki.sqlite"
            build_fever_wiki_index(wiki, db)
            raw = root / "train.jsonl"
            raw.write_text(json.dumps({"id": 1, "claim": "Acme was founded in 2010.",
                                       "label": "SUPPORTS", "evidence": [[[0, 0, "Acme", 0]]]}) + "\n")
            rows, audit = adapt(raw, "FEVER", "train", wiki_db=str(db))
            self.assertEqual(audit["accepted"], 1)
            self.assertEqual(rows[0].evidence[0]["text"], "Acme was founded in 2010.")


if __name__ == "__main__":
    unittest.main()
