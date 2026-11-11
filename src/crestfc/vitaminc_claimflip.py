"""Full VitaminC dev E1-B scan with same-evidence, changed-claim pair metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .data import read_jsonl, write_jsonl
from .frozen_reader import (LABELS, QUESTION, capture_all, classify, decoder_prefix, digest,
                            dump, encoder_text, score_layer_batch, slot_id, candidate_ids)
from .modeling import load_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/VitaminC/converted_data/dev_2.json"
DEFAULT_SNAPSHOT = "/data/yangjun/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
CONDITIONS = ("full_patch", "claim_only_patch")
TIE_TOL = 1e-6


def family_id(row: dict) -> str:
    source_id = str(row["id"])
    if "_" not in source_id:
        raise ValueError(f"VitaminC ID lacks revision suffix: {source_id}")
    return source_id.rsplit("_", 1)[0]


def build_claimflip_pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    """One deterministic opposite-label claim pair per original family, same exact evidence."""
    groups = defaultdict(list)
    for row in rows:
        groups[(family_id(row), row["gold_evidence"])].append(row)
    by_family = defaultdict(list)
    for (family, evidence), group in groups.items():
        supports = [x for x in group if x["label"] == "SUPPORTS"]
        refutes = [x for x in group if x["label"] == "REFUTES"]
        for a in supports:
            for b in refutes:
                if a["claim"] != b["claim"]:
                    similarity = SequenceMatcher(None, a["claim"], b["claim"]).ratio()
                    by_family[family].append((similarity, a["id"], b["id"], a, b))
    pairs = []
    for family in sorted(by_family):
        similarity, _, _, a, b = max(by_family[family], key=lambda x: (x[0], x[1], x[2]))
        pairs.append({"pair_id": f"vitaminc_dev_claimflip:{family}", "source_family": family,
                      "supports_id": a["id"], "refutes_id": b["id"],
                      "claim_similarity": round(similarity, 6),
                      "evidence_sha256": hashlib.sha256(a["gold_evidence"].encode()).hexdigest(),
                      "review_status": "source_labels_unreviewed"})
    return pairs, {"candidate_family_count": len(by_family), "selected_pairs": len(pairs),
                   "same_evidence_group_count": len(groups),
                   "paired_sample_count": len({x for p in pairs for x in (p["supports_id"], p["refutes_id"])})}


def load_source(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("VitaminC source must be a nonempty JSON array")
    seen = set()
    for index, row in enumerate(rows):
        if set(row) != {"id", "claim", "gold_evidence", "label"}:
            raise ValueError(f"Unexpected VitaminC source schema at row {index}")
        if (row["label"] not in LABELS or not isinstance(row["claim"], str)
                or not isinstance(row["gold_evidence"], str)
                or not row["claim"].strip() or not row["gold_evidence"].strip()):
            raise ValueError(f"Invalid label or empty text at source row {index}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate VitaminC source ID: {row['id']}")
        seen.add(row["id"])
        family_id(row)
    return rows


def shard_for_family(family: str, n_shards: int) -> int:
    return int(hashlib.sha256(family.encode()).hexdigest(), 16) % n_shards


def prepare(args) -> dict:
    run = ROOT / "outputs/experiment_runs" / args.run_id
    source = Path(args.source).resolve()
    raw = load_source(source)
    pairs, pair_audit = build_claimflip_pairs(raw)
    if args.max_families:
        chosen = {p["source_family"] for p in pairs[:args.max_families]}
        raw = [row for row in raw if family_id(row) in chosen]
        pairs = [pair for pair in pairs if pair["source_family"] in chosen]
    if not raw or not pairs:
        raise ValueError("No eligible same-evidence claim-flip pairs")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(snapshot)
    manifest = {"protocol": "e1b_vitaminc_dev_claimflip_v1", "run_id": args.run_id,
                "source": str(source), "source_sha256": digest(source),
                "model_id": "Qwen/Qwen2.5-7B-Instruct", "snapshot": str(snapshot),
                "snapshot_config_sha256": digest(snapshot / "config.json"),
                "conditions": list(CONDITIONS), "layers": list(range(28)),
                "labels": list(LABELS), "n_samples": len(raw), "n_pairs": len(pairs),
                "n_families": len({family_id(x) for x in raw}), "n_shards": args.shards,
                "max_families": args.max_families, "layer_batch": args.layer_batch,
                "slot_text": args.slot_text, "no_training": True,
                "pair_definition": "same exact gold_evidence, same revision family, opposite source labels, distinct claims; one pair per family",
                "code_sha256": digest(Path(__file__)),
                "reader_code_sha256": digest(Path(__file__).with_name("frozen_reader.py"))}
    if (run / "manifest.json").exists():
        previous = json.loads((run / "manifest.json").read_text())
        if previous != manifest:
            repair_path = run / "code_repair.json"
            repair = json.loads(repair_path.read_text()) if repair_path.is_file() else {}
            old_protocol = {k: v for k, v in previous.items() if k != "code_sha256"}
            new_protocol = {k: v for k, v in manifest.items() if k != "code_sha256"}
            if (old_protocol != new_protocol or
                    repair.get("old_code_sha256") != previous.get("code_sha256") or
                    repair.get("new_code_sha256") != manifest["code_sha256"]):
                raise ValueError("Existing run protocol differs; use a new run ID or an audited code repair")
        return manifest
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "samples.jsonl", (
        {"sample_id": row["id"], "source_family": family_id(row), "split": "dev",
         "claim": row["claim"], "evidence": [{"evidence_id": "E1", "text": row["gold_evidence"]}],
         "label": row["label"], "evidence_mode": "oracle"} for row in raw))
    write_jsonl(run / "claimflip_pairs.jsonl", pairs)
    dump(run / "manifest.json", manifest)
    dump(run / "data_audit.json", {"source_rows": len(load_source(source)),
                                  "selected_rows": len(raw), "source_label_counts": dict(Counter(x["label"] for x in raw)),
                                  "reviewed_pairs": 0, "pair_audit": pair_audit,
                                  "actual_pairs": len(pairs),
                                  "shard_sizes": {str(i): sum(shard_for_family(family_id(x), args.shards) == i
                                                              for x in raw) for i in range(args.shards)}})
    return manifest


def score_all_layers(model, tokenizer, row: dict, condition: str,
                     slot_token: int, candidates: list[list[int]], layer_batch: int) -> list[list[float]]:
    ids = tokenizer(encoder_text(row, condition), add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(model.device)
    if ids.shape[1] > model.config.max_position_embeddings - max(map(len, candidates)) - 128:
        raise ValueError(f"Input too long for full-sequence patch: {row['sample_id']}")
    states = capture_all(model, ids)
    prefix = decoder_prefix(tokenizer, ids.shape[1], slot_token)
    expected_question = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}], tokenize=True, add_generation_prompt=True)
    if (prefix[:ids.shape[1]] != [slot_token] * ids.shape[1] or
            prefix[ids.shape[1]:] != list(expected_question)):
        raise ValueError(f"Decoder prefix differs from fixed slot + question template: {row['sample_id']}")
    scores = []
    for start in range(0, 28, layer_batch):
        source_batch = torch.cat([states[layer] for layer in range(start, min(start + layer_batch, 28))], dim=0)
        scores.extend(score_layer_batch(model, prefix, candidates, source_batch))
    if len(scores) != 28 or not all(np.isfinite(x).all() for x in scores):
        raise RuntimeError("Incomplete or nonfinite layer scores")
    return scores


def scan_shard(args) -> None:
    run = ROOT / "outputs/experiment_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text())
    if args.shard < 0 or args.shard >= manifest["n_shards"]:
        raise ValueError("Invalid shard")
    samples = [row for row in read_jsonl(run / "samples.jsonl")
               if shard_for_family(row["source_family"], manifest["n_shards"]) == args.shard]
    out = run / "shards"
    out.mkdir(exist_ok=True)
    output = out / f"shard{args.shard}.jsonl"
    marker = out / f"shard{args.shard}_complete.json"
    if marker.is_file():
        complete = json.loads(marker.read_text())
        if complete["n_samples"] == len(samples) and complete["sha256"] == digest(output):
            print(f"shard {args.shard} complete; reusing", flush=True)
            return
        raise ValueError(f"Shard completion marker invalid: {marker}")
    existing = read_jsonl(output) if output.is_file() else []
    done = {row["sample_id"] for row in existing}
    if len(done) != len(existing) or not done <= {row["sample_id"] for row in samples}:
        raise ValueError("Existing shard contains duplicate or foreign sample IDs")
    model, tokenizer = load_model(manifest["snapshot"], None, "bfloat16", "cuda")
    model.eval().requires_grad_(False)
    if model.config.num_hidden_layers != 28 or model.config.hidden_size != 3584:
        raise ValueError("Expected frozen Qwen2.5-7B with 28 blocks")
    sid, candidates = slot_id(tokenizer, manifest["slot_text"]), candidate_ids(tokenizer)
    dump(out / f"shard{args.shard}_runtime.json", {"gpu": torch.cuda.get_device_name(0),
                                                 "slot_id": sid, "candidate_ids": candidates,
                                                 "frozen_parameters": sum(p.requires_grad for p in model.parameters())})
    claim_cache = {}
    start_time = time.monotonic()
    with output.open("a", encoding="utf8") as stream:
        for index, row in enumerate(samples, 1):
            if row["sample_id"] in done:
                continue
            full = score_all_layers(model, tokenizer, row, "full_patch", sid, candidates,
                                    manifest["layer_batch"])
            if row["claim"] not in claim_cache:
                claim_cache[row["claim"]] = score_all_layers(model, tokenizer, row, "claim_only_patch",
                                                              sid, candidates, manifest["layer_batch"])
            claim = claim_cache[row["claim"]]
            record = {"sample_id": row["sample_id"], "source_family": row["source_family"],
                      "gold_label": row["label"], "full_scores": full, "claim_only_scores": claim}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 100 == 0 or index == len(samples):
                elapsed = time.monotonic() - start_time
                print(f"shard {args.shard}: {index}/{len(samples)} rows, {elapsed:.0f}s elapsed", flush=True)
    dump(marker, {"shard": args.shard, "n_samples": len(samples), "sha256": digest(output)})


def analyze(args) -> None:
    run = ROOT / "outputs/experiment_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text())
    samples = {row["sample_id"]: row for row in read_jsonl(run / "samples.jsonl")}
    pairs = read_jsonl(run / "claimflip_pairs.jsonl")
    by_id = {}
    for shard in range(manifest["n_shards"]):
        output = run / "shards" / f"shard{shard}.jsonl"
        marker = json.loads((run / "shards" / f"shard{shard}_complete.json").read_text())
        if marker["sha256"] != digest(output):
            raise ValueError(f"Shard {shard} changed after completion")
        for row in read_jsonl(output):
            sid = row["sample_id"]
            if sid in by_id or sid not in samples or row["gold_label"] != samples[sid]["label"]:
                raise ValueError(f"Duplicate, foreign or mismatched prediction: {sid}")
            if len(row["full_scores"]) != 28 or len(row["claim_only_scores"]) != 28:
                raise ValueError(f"Incomplete layer predictions: {sid}")
            by_id[sid] = row
    if len(by_id) != len(samples):
        raise ValueError(f"Predictions incomplete: {len(by_id)} / {len(samples)}")
    ordered = [by_id[sid] for sid in samples]
    truth = [row["gold_label"] for row in ordered]
    metric_rows = []
    for condition, field in (("full_patch", "full_scores"), ("claim_only_patch", "claim_only_scores")):
        for layer in range(28):
            pred = [classify(row[field][layer])[0] for row in ordered]
            correct_s = sum(y == p == "SUPPORTS" for y, p in zip(truth, pred))
            correct_r = sum(y == p == "REFUTES" for y, p in zip(truth, pred))
            n_s, n_r = truth.count("SUPPORTS"), truth.count("REFUTES")
            gaps, pair_correct, pair_tie = [], [], []
            for pair in pairs:
                s = by_id[pair["supports_id"]][field][layer]
                r = by_id[pair["refutes_id"]][field][layer]
                m_s, m_r = s[0] - s[1], r[0] - r[1]
                gap = m_s - m_r
                gaps.append(gap)
                pair_tie.append(abs(gap) <= TIE_TOL)
                pair_correct.append(m_s > TIE_TOL and m_r < -TIE_TOL)
            direction = sum(0.5 if tie else float(gap > 0)
                            for gap, tie in zip(gaps, pair_tie)) / len(pairs)
            metric_rows.append({"condition": condition, "layer": layer,
                                "n_samples": len(ordered), "n_pairs": len(pairs),
                                "macro_f1": float(f1_score(truth, pred, labels=LABELS,
                                                           average="macro", zero_division=0)),
                                "accuracy": sum(y == p for y, p in zip(truth, pred)) / len(truth),
                                "balanced_accuracy": 0.5 * (correct_s / n_s + correct_r / n_r),
                                "supports_recall": correct_s / n_s, "refutes_recall": correct_r / n_r,
                                "predict_supports": pred.count("SUPPORTS"),
                                "predict_refutes": pred.count("REFUTES"),
                                "predict_tie": pred.count("TIE"),
                                "claimflip_direction": direction,
                                "claimflip_paircorrect": sum(pair_correct) / len(pairs),
                                "claimflip_tie_rate": sum(pair_tie) / len(pairs),
                                "mean_pair_margin_gap": float(np.mean(gaps))})
    path = run / "layer_metrics.csv"
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for condition in CONDITIONS:
        values = [row for row in metric_rows if row["condition"] == condition]
        for ax, field in zip(axes, ("macro_f1", "claimflip_direction", "claimflip_paircorrect")):
            ax.plot([v["layer"] for v in values], [v[field] for v in values], label=condition)
            ax.set(xlabel="Encoder read layer", ylabel=field, ylim=(-.02, 1.02))
            ax.grid(alpha=.25)
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / "layer_curves.png", dpi=180); plt.close(fig)
    dump(run / "analysis_summary.json", {"protocol": manifest["protocol"],
                                         "n_samples": len(ordered), "n_pairs": len(pairs),
                                         "selection_allowed": False,
                                         "reason": "source labels are unreviewed; all 28 layers are descriptive dev results"})
    print(json.dumps({"metrics": str(path), "n_samples": len(ordered), "n_pairs": len(pairs)},
                     ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "scan-shard", "analyze"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--shard", type=int, default=-1)
    parser.add_argument("--max-families", type=int, default=0)
    parser.add_argument("--layer-batch", type=int, default=7)
    parser.add_argument("--slot-text", default="?")
    args = parser.parse_args()
    if args.shards < 1 or args.max_families < 0 or not 1 <= args.layer_batch <= 28:
        parser.error("Invalid shards, max-families or layer-batch")
    if args.stage == "prepare":
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2), flush=True)
    elif args.stage == "scan-shard":
        scan_shard(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
