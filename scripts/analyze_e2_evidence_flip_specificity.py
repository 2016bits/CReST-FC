#!/usr/bin/env python3
"""Compare fixed-Claim Evidence Flip patches with same-label random donors.

Run after ``crestfc.evidence_flip_patch analyze`` has completed.  Resampling
keeps both patch directions from each pair together, and the simultaneous
interval covers all layer/position comparisons in this scan.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from crestfc.data import file_sha256, stable_id
from crestfc.evidence_flip_patch import MODES, POSITIONS, ROOT

DEFAULT_RUN = ROOT / "outputs/experiment_runs/e2_evidence_flip_vitaminc_dev_300_all28_4gpu_20261110"
DEFAULT_E1 = ROOT / "outputs/experiment_runs/fever7b_e1_main/analysis/candidate_layers.json"
DIRECTIONS = ("supports", "refutes")  # Target side; cf donor has the opposite label.


def collect(run: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf8"))
    layers = manifest["layers"]
    if layers != list(range(28)) or manifest["positions"] != list(POSITIONS) or manifest["modes"] != list(MODES):
        raise ValueError("Expected the complete 28-layer Evidence Flip protocol")
    pairs = [json.loads(line) for line in (run / "pairs.jsonl").open(encoding="utf8") if line.strip()]
    if len(pairs) != manifest["n_pairs"] or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Incomplete or duplicate pair source")
    families = [p["supports"]["source_provenance"]["revision_family_id"] for p in pairs]
    if len(set(families)) != len(pairs):
        raise ValueError("Pair bootstrap requires distinct source families")
    shape = (len(pairs), len(DIRECTIONS), len(layers), len(POSITIONS), len(MODES))
    effects = np.empty(shape, dtype=np.float64)
    direction = np.empty(shape, dtype=np.float64)
    flip = np.empty(shape, dtype=np.float64)
    flip_to_donor = np.empty(shape, dtype=np.float64)
    expected_keys = {(side, layer, pos, mode)
                     for side in DIRECTIONS for layer in layers for pos in POSITIONS for mode in MODES}
    for i, pair in enumerate(pairs):
        path = run / "pair_results" / f"{stable_id(pair['pair_id'])}.json"
        result = json.loads(path.read_text(encoding="utf8"))
        if result["pair_id"] != pair["pair_id"] or result["random_donor_pair_id"] != pair["random_donor_pair_id"]:
            raise ValueError(f"Result/pair mismatch: {path}")
        seen = set()
        for row in result["effects"]:
            key = (row["target_side"], row["layer"], row["position"], row["mode"])
            if key not in expected_keys or key in seen:
                raise ValueError(f"Unexpected or duplicate intervention in {path}: {key}")
            seen.add(key)
            j = DIRECTIONS.index(row["target_side"])
            k = row["layer"]
            m = POSITIONS.index(row["position"])
            n = MODES.index(row["mode"])
            effects[i, j, k, m, n] = row["margin_change_toward_donor"]
            direction[i, j, k, m, n] = row["correct_direction"]
            flip[i, j, k, m, n] = row["prediction_flipped"]
            flip_to_donor[i, j, k, m, n] = row["flipped_to_donor"]
        if seen != expected_keys:
            raise ValueError(f"Missing interventions in {path}")
    if not np.isfinite(effects).all():
        raise ValueError("Nonfinite margin effect")
    return manifest, effects, direction, flip, flip_to_donor


def analyze(run: Path, e1_path: Path, repetitions: int, seed: int) -> dict:
    manifest, effects, direction, flip, flip_to_donor = collect(run)
    n_pairs = effects.shape[0]
    cf = effects[..., MODES.index("cf_donor")]
    random = effects[..., MODES.index("random_donor")]
    adjusted = cf - random
    # Each row represents one independent pair; both intervention directions stay together.
    cf_pair = cf.mean(axis=1)
    random_pair = random.mean(axis=1)
    adjusted_pair = adjusted.mean(axis=1)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n_pairs, size=(repetitions, n_pairs))
    boot_cf = np.empty((repetitions, 28, len(POSITIONS)))
    boot_adjusted = np.empty_like(boot_cf)
    for b, indices in enumerate(sample_indices):
        boot_cf[b] = cf_pair[indices].mean(axis=0)
        boot_adjusted[b] = adjusted_pair[indices].mean(axis=0)
    cf_ci = np.quantile(boot_cf, (0.025, 0.975), axis=0)
    adjusted_ci = np.quantile(boot_adjusted, (0.025, 0.975), axis=0)
    adjusted_mean = adjusted_pair.mean(axis=0)
    standard_error = adjusted_pair.std(axis=0, ddof=1) / np.sqrt(n_pairs)
    safe_se = np.where(standard_error > 0, standard_error, np.inf)
    max_t = np.max(np.abs((boot_adjusted - adjusted_mean) / safe_se), axis=(1, 2))
    simultaneous_threshold = float(np.quantile(max_t, 0.95))
    adjusted_simultaneous_low = adjusted_mean - simultaneous_threshold * standard_error
    adjusted_simultaneous_high = adjusted_mean + simultaneous_threshold * standard_error
    e1 = json.loads(e1_path.read_text(encoding="utf8"))
    e1_f1 = {item["layer"]: item["full_macro_f1"] for item in e1["ranked_layers"]}
    rows = []
    for layer in range(28):
        for pos_index, pos in enumerate(POSITIONS):
            rows.append({
                "layer": layer,
                "position": pos,
                "n_pairs": n_pairs,
                "e1_probe_full_macro_f1": float(e1_f1[layer]),
                "cf_mean_margin_change": float(cf_pair[:, layer, pos_index].mean()),
                "cf_ci95_low": float(cf_ci[0, layer, pos_index]),
                "cf_ci95_high": float(cf_ci[1, layer, pos_index]),
                "random_mean_margin_change": float(random_pair[:, layer, pos_index].mean()),
                "cf_minus_random_mean": float(adjusted_mean[layer, pos_index]),
                "cf_minus_random_ci95_low": float(adjusted_ci[0, layer, pos_index]),
                "cf_minus_random_ci95_high": float(adjusted_ci[1, layer, pos_index]),
                "cf_minus_random_simultaneous95_low": float(adjusted_simultaneous_low[layer, pos_index]),
                "cf_minus_random_simultaneous95_high": float(adjusted_simultaneous_high[layer, pos_index]),
                "cf_minus_random_positive_pair_fraction": float(np.mean(adjusted_pair[:, layer, pos_index] > 0)),
                "cf_correct_direction_rate": float(direction[:, :, layer, pos_index, MODES.index("cf_donor")].mean()),
                "random_correct_direction_rate": float(direction[:, :, layer, pos_index, MODES.index("random_donor")].mean()),
                "cf_both_directions_correct_rate": float(np.mean(
                    direction[:, :, layer, pos_index, MODES.index("cf_donor")].all(axis=1))),
                "cf_prediction_flip_rate": float(flip[:, :, layer, pos_index, MODES.index("cf_donor")].mean()),
                "random_prediction_flip_rate": float(flip[:, :, layer, pos_index, MODES.index("random_donor")].mean()),
                "cf_flip_to_donor_rate": float(flip_to_donor[:, :, layer, pos_index, MODES.index("cf_donor")].mean()),
                "random_flip_to_donor_rate": float(flip_to_donor[:, :, layer, pos_index, MODES.index("random_donor")].mean()),
                "cf_minus_random_supports_to_refutes": float(adjusted[:, 1, layer, pos_index].mean()),
                "cf_minus_random_refutes_to_supports": float(adjusted[:, 0, layer, pos_index].mean()),
            })
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "specificity_by_layer_position.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for axis, pos in zip(axes, POSITIONS):
        selected = [row for row in rows if row["position"] == pos]
        xs = [row["layer"] for row in selected]
        axis.plot(xs, [row["cf_mean_margin_change"] for row in selected], label="Same Claim donor", color="tab:blue")
        axis.plot(xs, [row["random_mean_margin_change"] for row in selected], label="Random donor, same label", color="tab:orange")
        axis.plot(xs, [row["cf_minus_random_mean"] for row in selected], label="Paired difference", color="tab:green")
        axis.fill_between(xs, [row["cf_minus_random_simultaneous95_low"] for row in selected],
                          [row["cf_minus_random_simultaneous95_high"] for row in selected],
                          alpha=0.15, color="tab:green", label="Simultaneous 95% interval")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(ylabel="Mean donor-direction margin change", title=pos)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    axes[-1].set(xlabel="Transformer block output (zero-based)", xticks=list(range(28)))
    fig.tight_layout()
    fig.savefig(out / "specificity_layer_profile.png", dpi=180)
    plt.close(fig)

    ranked = {pos: sorted((row for row in rows if row["position"] == pos),
                          key=lambda row: row["cf_minus_random_mean"], reverse=True)[:8]
              for pos in POSITIONS}
    report = {
        "run_id": manifest["run_id"],
        "manifest_sha256": file_sha256(run / "manifest.json"),
        "e1_reference": str(e1_path),
        "e1_reference_sha256": file_sha256(e1_path),
        "n_pairs": n_pairs,
        "n_reviewed_pairs": manifest["n_reviewed_pairs"],
        "layers": manifest["layers"],
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "simultaneous_interval_comparisons": 28 * len(POSITIONS),
        "simultaneous_max_t_threshold": simultaneous_threshold,
        "max_identity_abs_effect": float(np.abs(effects[..., MODES.index("identity")]).max()),
        "ranked_cf_minus_random_by_position": ranked,
        "interpretation": (
            "Patching establishes an effect on this label margin. The same-label random donor "
            "isolates how much of the effect may come from an arbitrary donor carrying the label. "
            "A positive paired difference does not by itself identify an abstract relation variable; "
            "the E1 Probe uses a different prompt and read position."
        ),
    }
    (out / "specificity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    result = analyze(args.run.resolve(), args.e1.resolve(), args.bootstrap, args.seed)
    print(json.dumps({"run_id": result["run_id"], "n_pairs": result["n_pairs"],
                      "max_identity_abs_effect": result["max_identity_abs_effect"]}, indent=2))


if __name__ == "__main__":
    main()
