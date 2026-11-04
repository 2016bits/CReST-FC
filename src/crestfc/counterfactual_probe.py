"""Evaluate frozen E1 linear probes on fixed-claim, changed-evidence pairs.

Probe fitting is deliberately absent. This program requires serialized E1 probe
weights and StandardScaler statistics; E1's existing prediction checkpoints do
not contain them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch

from .data import adapt, file_sha256, read_jsonl, stable_id, write_jsonl
from .fever_probe import LABELS, ROOT, sha256
from .modeling import get_transformer_blocks, last_nonpadding, load_model, render_encoder

DATASETS = {
    "symmetric_fever": ("FEVER", "data/FEVER/converted_data/symmetric.json", "test"),
    "vitaminc": ("VitaminC", "data/VitaminC/converted_data/test_2.json", "test"),
    "vitaminc_dev": ("VitaminC", "data/VitaminC/converted_data/dev_2.json", "dev"),
}
N_LAYERS = 28
HIDDEN = 3584


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def normalized_claim(value: str) -> str:
    return " ".join(value.casefold().split())


def evidence_text(row: dict) -> str:
    return "\n".join(item["text"] for item in row["evidence"])


def make_fixed_claim_pairs(rows: list[dict], dataset: str, excluded_claims: set[str],
                           max_pairs: int = 0) -> tuple[list[dict], dict]:
    """Select one closest opposite-label evidence pair per exact normalized claim."""
    groups = defaultdict(list)
    excluded_rows = 0
    for row in rows:
        key = normalized_claim(row["claim"])
        if key in excluded_claims:
            excluded_rows += 1
            continue
        if row["label"] in LABELS and evidence_text(row).strip():
            groups[key].append(row)
    pairs = []
    for key in sorted(groups):
        group = groups[key]
        supports = [row for row in group if row["label"] == "SUPPORTS"]
        refutes = [row for row in group if row["label"] == "REFUTES"]
        candidates = []
        for a in supports:
            for b in refutes:
                if a["claim"] != b["claim"] or evidence_text(a) == evidence_text(b):
                    continue
                if dataset.startswith("vitaminc"):
                    family_a = a["source_provenance"]["revision_family_id"]
                    family_b = b["source_provenance"]["revision_family_id"]
                    if family_a != family_b:
                        continue
                similarity = SequenceMatcher(None, evidence_text(a), evidence_text(b)).ratio()
                candidates.append((similarity, a["sample_id"], b["sample_id"], a, b))
        if not candidates:
            continue
        similarity, _, _, a, b = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        pairs.append({"pair_id": f"{dataset}:{stable_id(a['sample_id'], b['sample_id'])}",
                      "claim": a["claim"], "supports": a, "refutes": b,
                      "evidence_similarity": round(similarity, 6),
                      "annotation_status": "source_labels_unreviewed"})
    pairs.sort(key=lambda pair: pair["pair_id"])
    if dataset == "vitaminc_dev":
        # The calibration subset uses independent revision families.
        seen_families = set()
        independent = []
        for pair in pairs:
            family = pair["supports"]["source_provenance"]["revision_family_id"]
            if family not in seen_families:
                independent.append(pair)
                seen_families.add(family)
        pairs = independent
    available = len(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs, {"input_rows": len(rows), "excluded_e1_overlap_rows": excluded_rows,
                   "candidate_pairs": available, "selected_pairs": len(pairs),
                   "unique_claims": len({normalized_claim(pair["claim"]) for pair in pairs})}


def e1_claims(e1_run: Path) -> set[str]:
    return {normalized_claim(row["claim"])
            for split in ("train", "dev")
            for row in read_jsonl(e1_run / "data" / f"{split}.jsonl")}


def prepare(args, run: Path) -> dict:
    e1_run = Path(args.e1_run).resolve()
    e1_manifest = json.loads((e1_run / "manifest.json").read_text())
    if e1_manifest["model_id"] != "Qwen/Qwen2.5-7B-Instruct" or e1_manifest["read_position"] != "final_anchor":
        raise ValueError("E1 model or read position differs from this evaluation protocol")
    dataset_name, relative_source, split = DATASETS[args.dataset]
    source = ROOT / relative_source
    rows, source_audit = adapt(source, dataset_name, split)
    excluded_claims = e1_claims(e1_run)
    if args.dataset == "vitaminc_dev":
        test_source = ROOT / DATASETS["vitaminc"][1]
        test_rows, _ = adapt(test_source, "VitaminC", "test")
        excluded_claims.update(normalized_claim(row.claim) for row in test_rows)
    pairs, pair_audit = make_fixed_claim_pairs([vars(row) for row in rows], args.dataset,
                                               excluded_claims, args.max_pairs)
    if not pairs:
        raise ValueError(f"No eligible fixed-claim counterfactual pairs in {source}")
    manifest = {"dataset": args.dataset, "source": str(source.resolve()),
                "source_sha256": file_sha256(source), "source_split": split,
                "source_audit": source_audit, "pair_audit": pair_audit,
                "e1_run": str(e1_run), "e1_manifest_sha256": sha256(e1_run / "manifest.json"),
                "model_id": e1_manifest["model_id"], "cache_dir": e1_manifest["cache_dir"],
                "read_position": "final_anchor", "condition": "full", "layers": list(range(N_LAYERS)),
                "n_shards": args.shards, "max_pairs": args.max_pairs,
                "label_order": list(LABELS), "pair_definition": "same exact claim; source SUPPORTS/REFUTES; distinct evidence"}
    if (run / "manifest.json").exists():
        old = json.loads((run / "manifest.json").read_text())
        if old != manifest:
            raise ValueError(f"Existing run manifest differs: {run}; use a new run ID")
        return manifest
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "pairs.jsonl", pairs)
    save_json(run / "manifest.json", manifest)
    return manifest


def probe_path(probe_dir: Path, layer: int) -> Path:
    return probe_dir / f"layer{layer:02d}_full.npz"


def load_probe(probe_dir: Path, layer: int, e1_manifest: dict, e1_run: Path) -> dict:
    path = probe_path(probe_dir, layer)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing original E1 probe parameters: {path}. E1 saved only predictions, "
            "not weights/scaler statistics. This experiment cannot score new samples "
            "without the original serialized Probe; it will not refit one.")
    with np.load(path, allow_pickle=False) as bundle:
        needed = {"mean", "scale", "coef", "intercept", "metadata"}
        if not needed <= set(bundle.files):
            raise ValueError(f"Incomplete E1 probe bundle: {path}; expected {sorted(needed)}")
        meta = json.loads(str(bundle["metadata"].item()))
        expected = {"model_id": e1_manifest["model_id"], "layer": layer,
                    "condition": "full", "read_position": "final_anchor",
                    "label_order": list(LABELS), "e1_run_id": e1_run.name,
                    "e1_manifest_sha256": sha256(e1_run / "manifest.json")}
        for field, value in expected.items():
            if meta.get(field) != value:
                raise ValueError(f"Probe metadata {field} mismatch in {path}: {meta.get(field)!r} != {value!r}")
        probe = {key: np.asarray(bundle[key], dtype=np.float32).copy()
                 for key in ("mean", "scale", "coef", "intercept")}
    if (probe["mean"].shape != (HIDDEN,) or probe["scale"].shape != (HIDDEN,)
            or probe["coef"].shape != (HIDDEN,) or probe["intercept"].size != 1
            or not all(np.isfinite(probe[key]).all() for key in probe)
            or (probe["scale"] <= 0).any()):
        raise ValueError(f"Invalid E1 probe dimensions or values: {path}")
    return probe


def check_probes(args) -> dict:
    e1_run = Path(args.e1_run).resolve()
    e1_manifest = json.loads((e1_run / "manifest.json").read_text())
    probe_dir = Path(args.probe_dir).resolve()
    files = {}
    for layer in range(N_LAYERS):
        load_probe(probe_dir, layer, e1_manifest, e1_run)
        files[str(layer)] = sha256(probe_path(probe_dir, layer))
    return {"probe_dir": str(probe_dir), "probe_sha256": files,
            "e1_run": str(e1_run), "e1_manifest_sha256": sha256(e1_run / "manifest.json")}


def shard_pairs(run: Path, shard: int, n_shards: int) -> list[dict]:
    return [pair for pair in read_jsonl(run / "pairs.jsonl")
            if int(stable_id(pair["pair_id"]), 16) % n_shards == shard]


def extract_shard(args, run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    shard, n_shards = args.shard, manifest["n_shards"]
    if shard < 0 or shard >= n_shards:
        raise ValueError(f"shard must be in [0,{n_shards})")
    out = run / "activations"
    marker = out / f"shard{shard}_complete.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta.get("n_shards") == n_shards and all((out / name).is_file() for name in meta["files"]):
            print(f"shard {shard} already complete", flush=True)
            return
    pairs = shard_pairs(run, shard, n_shards)
    if not pairs:
        raise ValueError(f"Empty shard {shard}; use fewer shards or more pairs")
    e1_manifest = json.loads((Path(manifest["e1_run"]) / "manifest.json").read_text())
    model, tokenizer = load_model(e1_manifest["snapshot"], manifest["cache_dir"], "bfloat16", "cuda")
    model.eval()
    blocks = get_transformer_blocks(model)
    if len(blocks) != N_LAYERS or model.config.hidden_size != HIDDEN:
        raise ValueError("Qwen model dimensions differ from E1")
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"shard{shard}.npy"
    partial = out / f"shard{shard}.partial.npy"
    array = np.lib.format.open_memmap(partial, mode="w+", dtype="float32",
                                      shape=(len(pairs), 2, N_LAYERS, HIDDEN))
    max_tokens = 0
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        rows = [row for pair in batch for row in (pair["supports"], pair["refutes"])]
        prompts = [render_encoder(row, "full")[0] for row in rows]
        encoded = tokenizer(prompts, add_special_tokens=False, padding=True,
                            truncation=False, return_tensors="pt")
        max_tokens = max(max_tokens, int(encoded["attention_mask"].sum(1).max()))
        if max_tokens > model.config.max_position_embeddings:
            raise ValueError(f"Input exceeds model context: {max_tokens}")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        positions = last_nonpadding(encoded["attention_mask"])
        storage = [None] * N_LAYERS
        handles = []
        for layer, block in enumerate(blocks):
            def capture(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                storage[layer] = hidden[torch.arange(hidden.shape[0], device=hidden.device),
                                        positions].float()
            handles.append(block.register_forward_hook(capture))
        try:
            with torch.inference_mode():
                model.model(**encoded, use_cache=False, return_dict=True)
                values = torch.stack(storage, dim=1).cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        array[start:start + len(batch)] = values.reshape(len(batch), 2, N_LAYERS, HIDDEN)
        if start % (args.batch_size * 100) == 0:
            print(f"shard={shard} {start + len(batch)}/{len(pairs)}", flush=True)
    array.flush()
    del array
    partial.replace(final)
    write_jsonl(out / f"shard{shard}_index.jsonl",
                ({"pair_id": pair["pair_id"], "claim": pair["claim"],
                  "supports_id": pair["supports"]["sample_id"],
                  "refutes_id": pair["refutes"]["sample_id"]} for pair in pairs))
    save_json(marker, {"shard": shard, "n_shards": n_shards, "n_pairs": len(pairs),
                       "max_prompt_tokens": max_tokens,
                       "files": [final.name, f"shard{shard}_index.jsonl"]})


def probe_score(activations: np.ndarray, probe: dict) -> np.ndarray:
    """Positive signed score means REFUTES; the E1 label order is SUPPORTS, REFUTES."""
    z = (np.asarray(activations, dtype=np.float32) - probe["mean"]) / probe["scale"]
    return z @ probe["coef"] + probe["intercept"].item()


def wilson_ci95(successes: int, total: int) -> list[float]:
    if total < 1:
        raise ValueError("Wilson interval requires at least one observation")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def evaluate(args, run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    e1_run = Path(manifest["e1_run"])
    e1_manifest = json.loads((e1_run / "manifest.json").read_text())
    audit = check_probes(args)
    if audit["e1_manifest_sha256"] != manifest["e1_manifest_sha256"]:
        raise ValueError("E1 manifest changed after pair preparation")
    n_shards = manifest["n_shards"]
    indexes, arrays = [], []
    for shard in range(n_shards):
        marker = run / "activations" / f"shard{shard}_complete.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Missing activation shard {shard}: {marker}")
        meta = json.loads(marker.read_text())
        if meta["n_shards"] != n_shards:
            raise ValueError(f"Partition mismatch in {marker}")
        index = read_jsonl(run / "activations" / f"shard{shard}_index.jsonl")
        array = np.load(run / "activations" / f"shard{shard}.npy", mmap_mode="r")
        if array.shape != (len(index), 2, N_LAYERS, HIDDEN):
            raise ValueError(f"Activation dimensions mismatch for shard {shard}")
        indexes.append(index)
        arrays.append(array)
    expected = {pair["pair_id"] for pair in read_jsonl(run / "pairs.jsonl")}
    observed = [row["pair_id"] for index in indexes for row in index]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicated counterfactual activation pairs")
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    metrics = []
    with (output / "pair_predictions.jsonl").open("w", encoding="utf8") as stream:
        for layer in range(N_LAYERS):
            probe = load_probe(Path(args.probe_dir), layer, e1_manifest, e1_run)
            score_a, score_b = [], []
            for index, array in zip(indexes, arrays):
                for start in range(0, len(index), 2048):
                    chunk = array[start:start + 2048, :, layer, :]
                    scores = probe_score(chunk, probe)
                    score_a.append(scores[:, 0])
                    score_b.append(scores[:, 1])
            a = np.concatenate(score_a)
            b = np.concatenate(score_b)
            correct_a = a < 0
            correct_b = b >= 0
            pair_correct = correct_a & correct_b
            shift = b - a
            ci_low, ci_high = wilson_ci95(int(pair_correct.sum()), len(a))
            metrics.append({"layer": layer, "n_pairs": len(a),
                            "supports_accuracy": float(correct_a.mean()),
                            "refutes_accuracy": float(correct_b.mean()),
                            "pair_correct_rate": float(pair_correct.mean()),
                            "pair_correct_ci95_low": ci_low,
                            "pair_correct_ci95_high": ci_high,
                            "predicted_flip_rate": float(((a >= 0) != (b >= 0)).mean()),
                            "correct_direction_rate": float((shift > 0).mean()),
                            "mean_refutes_score_shift": float(shift.mean()),
                            "median_refutes_score_shift": float(np.median(shift)),
                            "selected_from_e1": layer in {18, 19, 20, 21, 22, 23}})
            offset = 0
            for index in indexes:
                for j, item in enumerate(index):
                    k = offset + j
                    stream.write(json.dumps({"pair_id": item["pair_id"], "claim": item["claim"],
                                             "supports_id": item["supports_id"],
                                             "refutes_id": item["refutes_id"], "layer": layer,
                                             "supports_refutes_logit": float(a[k]),
                                             "refutes_refutes_logit": float(b[k]),
                                             "supports_pred": "REFUTES" if a[k] >= 0 else "SUPPORTS",
                                             "refutes_pred": "REFUTES" if b[k] >= 0 else "SUPPORTS",
                                             "refutes_score_shift": float(shift[k]),
                                             "pair_correct": bool(pair_correct[k])},
                                            ensure_ascii=False) + "\n")
                offset += len(index)
            print(f"layer={layer} pair_correct={metrics[-1]['pair_correct_rate']:.4f}", flush=True)
    import csv
    with (output / "layer_metrics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    save_json(output / "evaluation_manifest.json", {**audit, "run_manifest_sha256": sha256(run / "manifest.json"),
                                                     "n_pairs": len(expected), "n_layers": N_LAYERS,
                                                     "score_convention": "positive = REFUTES",
                                                     "probe_fitting_performed": False})
    return {"dataset": manifest["dataset"], "n_pairs": len(expected),
            "selected_layer_metrics": [row for row in metrics if row["selected_from_e1"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check-probes", "extract-shard", "evaluate"])
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--run-id", default="counterfactual_e1_transfer")
    parser.add_argument("--e1-run", default=str(ROOT / "outputs/experiment_runs/fever7b_e1_main"))
    parser.add_argument("--probe-dir", default=str(ROOT / "outputs/experiment_runs/fever7b_e1_main/analysis/probes"))
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    if args.shards < 1 or args.batch_size < 1 or args.max_pairs < 0:
        parser.error("shards and batch size must be positive; max pairs must be nonnegative")
    if args.command != "check-probes" and args.dataset is None:
        parser.error("--dataset is required")
    run = ROOT / "outputs/experiment_runs" / args.run_id / (args.dataset or "")
    if args.command == "prepare":
        print(json.dumps(prepare(args, run), ensure_ascii=False, indent=2))
    elif args.command == "check-probes":
        try:
            print(json.dumps(check_probes(args), ensure_ascii=False, indent=2))
        except FileNotFoundError as exc:
            parser.exit(2, f"{exc}\n")
    elif args.command == "extract-shard":
        extract_shard(args, run)
    else:
        print(json.dumps(evaluate(args, run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
