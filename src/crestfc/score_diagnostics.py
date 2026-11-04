"""Describe paired Probe scores and calibrate intercepts on VitaminC dev only."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .counterfactual_probe import load_probe, normalized_claim, probe_score
from .data import read_jsonl
from .fever_probe import ROOT, load_layer, partition_count


def score_arrays(run: Path) -> dict[int, dict]:
    output = defaultdict(lambda: {"pair_id": [], "supports": [], "refutes": []})
    for row in read_jsonl(run / "analysis" / "pair_predictions.jsonl"):
        item = output[int(row["layer"])]
        item["pair_id"].append(row["pair_id"])
        item["supports"].append(row["supports_refutes_logit"])
        item["refutes"].append(row["refutes_refutes_logit"])
    if set(output) != set(range(28)):
        raise ValueError(f"Expected all 28 layers in {run}")
    for item in output.values():
        item["supports"] = np.asarray(item["supports"], dtype=np.float64)
        item["refutes"] = np.asarray(item["refutes"], dtype=np.float64)
    reference = output[0]["pair_id"]
    if any(item["pair_id"] != reference for item in output.values()):
        raise ValueError("Pair ordering differs between layers")
    return dict(output)


def describe(values: np.ndarray) -> dict:
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(values.mean()), "sd": float(values.std()),
            "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]),
            "fraction_refutes_at_zero": float((values >= 0).mean())}


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    locations = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, locations, side="right") / len(a)
    cdf_b = np.searchsorted(b, locations, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def group_bootstrap(values: np.ndarray, inverse: np.ndarray, sample_groups: np.ndarray,
                    group_sizes: np.ndarray) -> list[float]:
    numerator = np.bincount(inverse, weights=values.astype(np.float64), minlength=len(group_sizes))
    estimates = numerator[sample_groups].sum(axis=1) / group_sizes[sample_groups].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def paired_metrics(a: np.ndarray, b: np.ndarray, threshold: float) -> dict:
    supports_ok = a < threshold
    refutes_ok = b >= threshold
    pair_ok = supports_ok & refutes_ok
    return {"supports_accuracy": float(supports_ok.mean()),
            "refutes_accuracy": float(refutes_ok.mean()),
            "balanced_accuracy": float((supports_ok.mean() + refutes_ok.mean()) / 2),
            "pair_correct_rate": float(pair_ok.mean()),
            "correct_direction_rate": float((b > a).mean())}


def best_balanced_threshold(a: np.ndarray, b: np.ndarray) -> float:
    """Choose a scalar threshold using calibration-side labels only."""
    candidates = np.unique(np.concatenate([a, b]))
    midpoints = (candidates[:-1] + candidates[1:]) / 2
    thresholds = np.concatenate([[np.nextafter(candidates[0], -np.inf), 0.0],
                                 midpoints, [np.nextafter(candidates[-1], np.inf)]])
    sorted_a, sorted_b = np.sort(a), np.sort(b)
    support_correct = np.searchsorted(sorted_a, thresholds, side="left")
    refute_correct = len(b) - np.searchsorted(sorted_b, thresholds, side="left")
    scores = (support_correct + refute_correct) / (len(a) + len(b))
    winners = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(thresholds[winners[np.argmin(np.abs(thresholds[winners]))]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-run", default=str(ROOT / "outputs/experiment_runs/counterfactual_e1_transfer/vitaminc"))
    parser.add_argument("--dev-run", default=str(ROOT / "outputs/experiment_runs/counterfactual_e1_calibration/vitaminc_dev"))
    parser.add_argument("--e1-run", default=str(ROOT / "outputs/experiment_runs/fever7b_e1_main"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/experiment_runs/counterfactual_e1_transfer/score_diagnostics"))
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    test_run, dev_run, e1_run = map(Path, (args.test_run, args.dev_run, args.e1_run))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test = score_arrays(test_run)
    dev = score_arrays(dev_run)
    test_pairs = {p["pair_id"]: p for p in read_jsonl(test_run / "pairs.jsonl")}
    dev_pairs = {p["pair_id"]: p for p in read_jsonl(dev_run / "pairs.jsonl")}
    if len(test_pairs) != len(test[0]["pair_id"]) or len(dev_pairs) != len(dev[0]["pair_id"]):
        raise ValueError("Pair predictions do not match prepared pair records")
    test_claims = {normalized_claim(p["claim"]) for p in test_pairs.values()}
    dev_claims = {normalized_claim(p["claim"]) for p in dev_pairs.values()}
    test_families = [test_pairs[pid]["supports"]["source_provenance"]["revision_family_id"]
                     for pid in test[0]["pair_id"]]
    dev_families = {p["supports"]["source_provenance"]["revision_family_id"]
                    for p in dev_pairs.values()}
    if test_claims & dev_claims or dev_families & set(test_families):
        raise ValueError("VitaminC dev/test Claim or source-family leakage")
    unique_families, inverse = np.unique(test_families, return_inverse=True)
    group_sizes = np.bincount(inverse)
    rng = np.random.default_rng(args.seed)
    sampled_groups = rng.integers(0, len(unique_families), size=(args.bootstrap, len(unique_families)))

    e1_manifest = json.loads((e1_run / "manifest.json").read_text())
    n_shards = partition_count(e1_manifest, e1_run)
    distributions = []
    hard_cases = []
    for layer in range(28):
        a, b = test[layer]["supports"], test[layer]["refutes"]
        if len(a) != len(test_families):
            raise ValueError("Layer row count differs from test pair index")
        direction = b > a
        pair_ok = (a < 0) & (b >= 0)
        no_cross_support = direction & (b < 0)
        no_cross_refute = direction & (a >= 0)
        no_cross = no_cross_support | no_cross_refute
        if int(pair_ok.sum() + no_cross.sum() + (~direction).sum()) != len(a):
            raise AssertionError("Score categories do not partition all pairs")
        x_dev, e1_index = load_layer(e1_run, "dev", "full", layer, n_shards)
        probe = load_probe(e1_run / "analysis" / "probes", layer, e1_manifest, e1_run)
        fever_scores = probe_score(x_dev, probe).astype(np.float64)
        fever_supports = fever_scores[[row["label"] == "SUPPORTS" for row in e1_index]]
        fever_refutes = fever_scores[[row["label"] == "REFUTES" for row in e1_index]]
        distributions.append({"layer": layer,
                              "test_supports": describe(a), "test_refutes": describe(b),
                              "fever_dev_supports": describe(fever_supports),
                              "fever_dev_refutes": describe(fever_refutes),
                              "ks_supports_fever_vs_vitaminc": ks_distance(fever_supports, a),
                              "ks_refutes_fever_vs_vitaminc": ks_distance(fever_refutes, b),
                              "pair_correct_rate": float(pair_ok.mean()),
                              "pair_correct_group_ci95": group_bootstrap(pair_ok, inverse, sampled_groups, group_sizes),
                              "correct_direction_rate": float(direction.mean()),
                              "correct_direction_group_ci95": group_bootstrap(direction, inverse, sampled_groups, group_sizes),
                              "direction_correct_no_threshold_cross": float(no_cross.mean()),
                              "direction_correct_both_supports": float(no_cross_support.mean()),
                              "direction_correct_both_refutes": float(no_cross_refute.mean()),
                              "direction_wrong_or_tied": float((~direction).mean()),
                              "small_positive_shift_le_0p5": float(((b - a > 0) & (b - a <= 0.5)).mean())})
        if layer in (18, 19, 20):
            order = np.argsort(b - a)[:30]
            for index in order:
                pair = test_pairs[test[layer]["pair_id"][index]]
                hard_cases.append({"layer": layer, "pair_id": pair["pair_id"],
                                   "claim": pair["claim"],
                                   "supports_evidence": pair["supports"]["evidence"],
                                   "refutes_evidence": pair["refutes"]["evidence"],
                                   "supports_score": float(a[index]), "refutes_score": float(b[index]),
                                   "score_shift": float(b[index] - a[index]),
                                   "status": "requires_manual_review"})
        print(f"layer={layer} direction={direction.mean():.4f} no_cross={no_cross.mean():.4f}", flush=True)

    with (output / "hard_cases_for_review.jsonl").open("w", encoding="utf8") as stream:
        for row in hard_cases:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "layer_score_distributions.csv").open("w", newline="", encoding="utf8") as stream:
        fields = ["layer", "population", "mean", "sd", "q05", "q25", "median", "q75", "q95",
                  "fraction_refutes_at_zero"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in distributions:
            for population in ("test_supports", "test_refutes", "fever_dev_supports", "fever_dev_refutes"):
                writer.writerow({"layer": row["layer"], "population": population, **row[population]})

    calibration = []
    dev_order = np.random.default_rng(args.seed).permutation(len(dev[0]["pair_id"]))
    calibrate_index, holdout_index = np.array_split(dev_order, 2)
    for layer in (18, 19, 20):
        da, db = dev[layer]["supports"], dev[layer]["refutes"]
        ta, tb = test[layer]["supports"], test[layer]["refutes"]
        threshold = best_balanced_threshold(da[calibrate_index], db[calibrate_index])
        original_test = paired_metrics(ta, tb, 0.0)
        adjusted_test = paired_metrics(ta, tb, threshold)
        original_pair = (ta < 0) & (tb >= 0)
        adjusted_pair = (ta < threshold) & (tb >= threshold)
        delta = adjusted_pair.astype(np.float64) - original_pair.astype(np.float64)
        calibration.append({"layer": layer, "threshold_from_dev_calibration_only": threshold,
                            "dev_calibration_n": len(calibrate_index),
                            "dev_holdout_n": len(holdout_index),
                            "dev_calibration_at_zero": paired_metrics(da[calibrate_index], db[calibrate_index], 0.0),
                            "dev_calibration_adjusted": paired_metrics(da[calibrate_index], db[calibrate_index], threshold),
                            "dev_holdout_at_zero": paired_metrics(da[holdout_index], db[holdout_index], 0.0),
                            "dev_holdout_adjusted": paired_metrics(da[holdout_index], db[holdout_index], threshold),
                            "test_at_zero": original_test, "test_adjusted_fixed_threshold": adjusted_test,
                            "test_pair_correct_delta_group_ci95": group_bootstrap(delta, inverse, sampled_groups, group_sizes)})

    def evidence_lengths(pairs: dict) -> dict:
        lengths = [len(" ".join(e["text"] for e in side["evidence"]))
                   for pair in pairs.values() for side in (pair["supports"], pair["refutes"])]
        return {"median_characters": float(np.median(lengths)),
                "q95_characters": float(np.quantile(lengths, 0.95))}

    e1_dev_rows = read_jsonl(e1_run / "data" / "dev.jsonl")
    e1_lengths = [len(" ".join(e["text"] for e in row["evidence"])) for row in e1_dev_rows]
    report = {"protocol": "Frozen E1 Probe; dev-only scalar threshold fit; fixed VitaminC test evaluation",
              "test_pairs": len(test_pairs), "test_claim_groups": len(test_claims),
              "test_source_families": len(unique_families),
              "max_pairs_per_source_family": int(group_sizes.max()),
              "dev_pairs": len(dev_pairs), "dev_source_families": len(dev_families),
              "dev_test_claim_overlap": 0, "dev_test_source_family_overlap": 0,
              "group_bootstrap_resamples": args.bootstrap, "seed": args.seed,
              "score_convention": "positive signed score = REFUTES; original threshold = 0",
              "fever_dev_evidence_length": {"median_characters": float(np.median(e1_lengths)),
                                             "q95_characters": float(np.quantile(e1_lengths, 0.95))},
              "vitaminc_test_evidence_length": evidence_lengths(test_pairs),
              "layers": distributions, "calibration": calibration,
              "interpretation_limit": "Wrong score direction flags candidate hard cases, not proven semantic incapacity. KS values are descriptive distribution differences, not causal explanations."}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    (output / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, layer in zip(axes, (18, 19, 20)):
        values = test[layer]
        lo = min(np.quantile(values["supports"], 0.01), np.quantile(values["refutes"], 0.01))
        hi = max(np.quantile(values["supports"], 0.99), np.quantile(values["refutes"], 0.99))
        bins = np.linspace(lo, hi, 65)
        ax.hist(values["supports"], bins=bins, density=True, alpha=0.48, label="SUPPORTS evidence")
        ax.hist(values["refutes"], bins=bins, density=True, alpha=0.48, label="REFUTES evidence")
        ax.axvline(0, color="black", linestyle="--", label="FEVER threshold 0")
        ax.axvline(calibration[layer - 18]["threshold_from_dev_calibration_only"],
                   color="purple", linestyle=":", label="VitaminC dev threshold")
        ax.set_title(f"Layer {layer}: VitaminC test paired scores")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Signed REFUTES Probe score")
    fig.tight_layout()
    fig.savefig(output / "vitaminc_score_distributions_L18_L19_L20.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"report": str(output / "report.json"), "calibration": calibration},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
