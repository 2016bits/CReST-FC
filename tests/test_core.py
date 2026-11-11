import json
import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from crestfc.data import Example, normalize_label, validate_dataset, build_fever_wiki_index, adapt
from crestfc.evaluation import pair_metrics, clustered_bootstrap, parse_structured_output, parse_generated_verdict, evidence_prf
from crestfc.modeling import make_patch_hook, temporary_hook, label_margin, last_nonpadding, score_candidate_sequences, render_verdict, verdict_verbalizers
from crestfc.training import js_loss, flip_loss
from crestfc.fever_probe import resolve_gpu_tokens, shard_is_complete, partition_count, fit_probe
from crestfc.counterfactual_probe import make_fixed_claim_pairs, probe_score, evaluate
from crestfc.score_diagnostics import best_balanced_threshold, paired_metrics


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

    def test_generated_verdict_requires_one_exact_label(self):
        labels = ["SUPPORTS", "REFUTES"]
        verbalizers = ["SUPPORTED", "NOT_SUPPORTED"]
        self.assertEqual(parse_generated_verdict(" NOT_SUPPORTED\n", labels, verbalizers),
                         ("REFUTES", "NOT_SUPPORTED"))
        self.assertEqual(parse_generated_verdict("SUPPORTED", labels, verbalizers),
                         ("SUPPORTS", "SUPPORTED"))
        for text in ("NOT_SUPPORTED because...", "unsupported", "SUPPORTED or NOT_SUPPORTED", ""):
            self.assertEqual(parse_generated_verdict(text, labels, verbalizers),
                             ("INVALID", "INVALID"))

    def test_binary_fever_prompt_keeps_refutation_distinct_from_insufficiency(self):
        row = self.sample("x", "REFUTES", "Acme began in 2012.")
        prompt = render_verdict(row, scheme="fever_binary")
        self.assertEqual(verdict_verbalizers(["SUPPORTS", "REFUTES"], "fever_binary"),
                         ["SUPPORTS", "REFUTES"])
        self.assertIn("REFUTES: The evidence contradicts the claim.", prompt)
        self.assertNotIn("NEI", prompt)
        self.assertNotIn("NOT_SUPPORTED", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_visible_gpu_mapping(self):
        self.assertEqual(resolve_gpu_tokens([0, 1, 5], "0,1,2,3,4,7", 6), ["0", "1", "7"])
        with self.assertRaises(ValueError):
            resolve_gpu_tokens([6], "0,1,2,3,4,7", 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "activations"
            root.mkdir()
            (root / "shard0_complete.json").write_text(json.dumps({"files": ["train_full.npy"]}))
            self.assertFalse(shard_is_complete(Path(directory), 0))
            (root / "train_full.npy").write_bytes(b"done")
            self.assertTrue(shard_is_complete(Path(directory), 0))

    def test_legacy_probe_partition_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "activations").mkdir()
            manifest = {"gpus": "0,1,2,3,4,5"}
            self.assertEqual(partition_count(manifest, run), 6)
            (run / "activations" / "shard0_complete.json").write_text(
                json.dumps({"n_shards": 11, "files": ["train_full.npy"]}))
            self.assertEqual(partition_count(manifest, run), 11)
            self.assertEqual(partition_count({**manifest, "n_shards": 6}, run), 6)

    def test_fixed_claim_counterfactual_pairs_and_probe_score(self):
        def item(sid, label, claim, evidence, family="family"):
            row = self.sample(sid, label, evidence)
            row["claim"] = claim
            row["source_provenance"]["revision_family_id"] = family
            return row
        rows = [item("a", "SUPPORTS", "A is true.", "A is true."),
                item("b", "REFUTES", "A is true.", "A is false."),
                item("c", "REFUTES", "B is true.", "B is false.")]
        pairs, audit = make_fixed_claim_pairs(rows, "vitaminc", set())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["supports"]["sample_id"], "a")
        self.assertEqual(pairs[0]["refutes"]["sample_id"], "b")
        self.assertEqual(pairs[0]["claim"], "A is true.")
        self.assertEqual(audit["selected_pairs"], 1)
        excluded, _ = make_fixed_claim_pairs(rows, "vitaminc", {"a is true."})
        self.assertEqual(excluded, [])
        probe = {"mean": np.array([1., 0.]), "scale": np.array([2., 1.]),
                 "coef": np.array([1., -1.]), "intercept": np.array(0.)}
        self.assertEqual(probe_score(np.array([[1., 1.], [3., 0.]]), probe).tolist(), [-1., 1.])

    def test_frozen_probe_pair_metrics(self):
        from crestfc.fever_probe import sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e1 = root / "e1"
            e1.mkdir()
            (e1 / "manifest.json").write_text(json.dumps({"model_id": "Qwen/Qwen2.5-7B-Instruct"}))
            digest = sha256(e1 / "manifest.json")
            probe_dir = root / "probes"
            probe_dir.mkdir()
            for layer in range(28):
                metadata = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layer": layer,
                            "condition": "full", "read_position": "final_anchor",
                            "label_order": ["SUPPORTS", "REFUTES"], "e1_run_id": "e1",
                            "e1_manifest_sha256": digest}
                coef = np.zeros(3584)
                coef[0] = 1
                np.savez(probe_dir / f"layer{layer:02d}_full.npz",
                         mean=np.zeros(3584), scale=np.ones(3584), coef=coef,
                         intercept=np.array(0.), metadata=json.dumps(metadata))
            run = root / "run"
            (run / "activations").mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"e1_run": str(e1),
                "e1_manifest_sha256": digest, "n_shards": 1, "dataset": "toy"}))
            (run / "pairs.jsonl").write_text('{"pair_id":"p1"}\n{"pair_id":"p2"}\n')
            (run / "activations" / "shard0_index.jsonl").write_text(
                '{"pair_id":"p1","claim":"A","supports_id":"a","refutes_id":"b"}\n'
                '{"pair_id":"p2","claim":"B","supports_id":"c","refutes_id":"d"}\n')
            values = np.zeros((2, 2, 28, 3584), dtype=np.float32)
            values[0, 0, :, 0], values[0, 1, :, 0] = -1, 1
            values[1, 0, :, 0], values[1, 1, :, 0] = 1, -1
            np.save(run / "activations" / "shard0.npy", values)
            (run / "activations" / "shard0_complete.json").write_text(
                json.dumps({"n_shards": 1}))
            result = evaluate(SimpleNamespace(e1_run=str(e1), probe_dir=str(probe_dir)), run)
            self.assertEqual(result["n_pairs"], 2)
            row = result["selected_layer_metrics"][0]
            self.assertEqual(row["pair_correct_rate"], 0.5)
            self.assertEqual(row["predicted_flip_rate"], 1.0)
            self.assertEqual(row["correct_direction_rate"], 0.5)

    def test_fitted_probe_parameters_reproduce_predictions(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(120, 8)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        pred, _, parameters = fit_probe(x[:90], y[:90], x[90:], y[90:], 13,
                                         return_parameters=True)
        replay = (probe_score(x[90:], parameters) >= 0).astype(int)
        self.assertTrue(np.array_equal(replay, pred))

    def test_dev_threshold_calibration(self):
        supports = np.array([0., 1., 2.])
        refutes = np.array([3., 4., 5.])
        threshold = best_balanced_threshold(supports, refutes)
        self.assertGreater(threshold, 2.)
        self.assertLess(threshold, 3.)
        self.assertEqual(paired_metrics(supports, refutes, threshold)["pair_correct_rate"], 1.)

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

    def test_binary_support_prompt_and_label_mapping(self):
        row = self.sample("a", "REFUTES", "Acme began in 2012.")
        expected = ("You are an evidence-based fact verification assistant.\n\n"
                    "Determine whether the provided evidence sufficiently supports the claim.\n\n"
                    "Definitions:\n\n"
                    "- SUPPORTED: The evidence provides sufficient information to establish that the claim is true.\n"
                    "- NOT_SUPPORTED: The evidence contradicts the claim or does not provide sufficient information to establish that the claim is true.\n\n"
                    "Instructions:\n\n"
                    "- Base your decision strictly on the provided evidence.\n"
                    "- Do not rely on prior knowledge or assumptions.\n"
                    "- Do not confuse the absence of supporting evidence with evidence that the claim is false.\n"
                    "- Output exactly one label: SUPPORTED or NOT_SUPPORTED.\n"
                    "- Do not provide explanations.\n\n"
                    "Claim: Acme was founded in 2010.\n\n"
                    "Evidence: [E1] Acme began in 2012.\n\nLabel:")
        self.assertEqual(render_verdict(row, scheme="binary_support"), expected)
        self.assertEqual(verdict_verbalizers(["SUPPORTS", "REFUTES"], "binary_support"),
                         ["SUPPORTED", "NOT_SUPPORTED"])
        with self.assertRaises(ValueError):
            verdict_verbalizers(["SUPPORTS", "REFUTES", "NEI"], "binary_support")
        with tempfile.TemporaryDirectory() as directory:
            converted = Path(directory) / "converted_data"
            converted.mkdir()
            source = converted / "dev.json"
            source.write_text(json.dumps([{"id": 1, "claim": "Acme was founded in 2010.",
                                           "label": "refutes", "evidence": "Acme began in 2012."}]))
            rows, audit = adapt(source, "PolitiHop", "dev")
            self.assertEqual(audit["accepted"], 1)
            self.assertEqual(rows[0].label, "REFUTES")
            self.assertEqual(verdict_verbalizers(["SUPPORTS", rows[0].label], "binary_support")[1],
                             "NOT_SUPPORTED")

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
