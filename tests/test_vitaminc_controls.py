import copy
import unittest

from crestfc.vitaminc_controls import _metric_rows, select_random_evidence


def _sample(family, suffix, label, claim, evidence):
    return {"sample_id": f"{family}_{suffix}", "source_family": family,
            "label": label, "claim": claim,
            "evidence": [{"evidence_id": "E1", "text": evidence}]}


class VitaminCControlsTests(unittest.TestCase):
    def test_random_evidence_keeps_claim_and_is_label_blind(self):
        rows = [
            _sample("a", 1, "SUPPORTS", "A true", "Evidence A"),
            _sample("a", 2, "REFUTES", "A false", "Evidence A"),
            _sample("b", 1, "SUPPORTS", "B true", "Evidence B"),
            _sample("b", 2, "REFUTES", "B false", "Evidence B"),
            _sample("c", 1, "REFUTES", "C false", "Evidence C"),
        ]
        original = select_random_evidence(rows, 42)
        flipped = copy.deepcopy(rows)
        for row in flipped:
            row["label"] = "REFUTES" if row["label"] == "SUPPORTS" else "SUPPORTS"
        self.assertEqual(original, select_random_evidence(flipped, 42))
        by_id = {r["sample_id"]: r for r in rows}
        for donor in original:
            target = [r for r in rows if r["source_family"] == donor["source_family"]]
            donor_row = by_id[donor["donor_id"]]
            self.assertNotEqual(donor_row["source_family"], donor["source_family"])
            self.assertTrue(all(donor_row["evidence"][0]["text"] != r["evidence"][0]["text"]
                                for r in target))

    def test_pair_metrics_and_random_label_basis(self):
        samples = [_sample("a", "s", "SUPPORTS", "A true", "E"),
                   _sample("a", "r", "REFUTES", "A false", "E")]
        pair = {"pair_id": "p", "source_family": "a",
                "supports_id": "a_s", "refutes_id": "a_r"}
        by_id = {}
        for sample in samples:
            good = [0.0, -2.0] if sample["label"] == "SUPPORTS" else [-2.0, 0.0]
            bad = good[::-1]
            by_id[sample["sample_id"]] = {
                "gold_label": sample["label"],
                "full_scores": [good] * 28,
                "claim_only_scores": [bad] * 28,
                "evidence_only_scores": [good] * 28,
                "random_evidence_scores": [bad] * 28,
                "no_patch_scores": [0.0, 0.0],
                "direct_text_scores": good,
                "direct_generated_label": sample["label"],
            }
        metrics, pair_rows = _metric_rows(samples, [pair], by_id)
        lookup = {(m["condition"], m["layer"]): m for m in metrics}
        self.assertEqual(len(pair_rows), 4 * 28 + 3)
        self.assertEqual(lookup[("full_patch", 19)]["claimflip_direction"], 1.0)
        self.assertEqual(lookup[("full_patch", 19)]["claimflip_paircorrect"], 1.0)
        self.assertEqual(lookup[("claim_only_patch", 19)]["claimflip_paircorrect"], 0.0)
        self.assertEqual(lookup[("random_evidence_patch", 19)]["label_basis"], "original_gold_proxy")
        self.assertEqual(lookup[("no_patch_slots", -1)]["predict_tie"], 2)
        self.assertEqual(lookup[("direct_text_7b", -1)]["macro_f1"], 1.0)
        self.assertEqual(lookup[("direct_text_7b_generate", -1)]["claimflip_paircorrect"], 1.0)


if __name__ == "__main__":
    unittest.main()
