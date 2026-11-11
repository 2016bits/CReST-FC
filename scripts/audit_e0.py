#!/usr/bin/env python3
"""Audit E0 source fidelity, prompt isolation, labels and saved metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import copy
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoTokenizer

from crestfc.data import normalize_label
from crestfc.modeling import render_verdict, verdict_verbalizers
from crestfc.evaluation import parse_generated_verdict

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("hover", "politihop", "symmetric_politihop", "fever", "symmetric_fever")
CONDITIONS = ("full", "claim_only", "evidence_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def check(name: str, run: Path, tokenizer, max_positions: int) -> dict:
    base = run / name
    cfg = yaml.safe_load((base / "baseline.yaml").read_text(encoding="utf8"))
    evaluated_conditions = cfg.get("conditions", list(CONDITIONS))
    prep = yaml.safe_load((base / "prepare.yaml").read_text(encoding="utf8"))
    source = Path(prep["source"]["path"])
    raw = json.loads(source.read_text(encoding="utf8"))
    samples = read_jsonl(base / "data/samples.jsonl")
    predictions = read_jsonl(base / "e0/predictions.jsonl")
    metrics = {condition: json.loads((base / "e0" / f"metrics_{condition}.json").read_text())
               for condition in evaluated_conditions}
    scheme, labels = cfg["verdict_scheme"], cfg["labels"]
    verbalizers = verdict_verbalizers(labels, scheme)
    by_id = {row["sample_id"]: row for row in samples}
    errors = Counter()
    source_counts = Counter(str(row.get("label", "")).upper() for row in raw)
    processed_counts = Counter(row["label"] for row in samples)
    changed_claim_whitespace = 0
    evidence_piece_count = 0
    lengths = []
    longest = (0, None, None)
    source_digest = sha256(source)
    for row in samples:
        provenance = row["source_provenance"]
        index = provenance["source_row"]
        if index < 0 or index >= len(raw):
            errors["invalid_source_row"] += 1
            continue
        original = raw[index]
        if provenance["source_sha256"] != source_digest:
            errors["source_hash_mismatch"] += 1
        if row["claim"] != original["claim"].strip():
            errors["claim_mismatch"] += 1
        if row["claim"] != original["claim"]:
            changed_claim_whitespace += 1
        if row["label"] != normalize_label(original["label"]):
            errors["gold_label_mismatch"] += 1
        if name == "hover":
            expected_evidence = [x["sentences"] for x in original["gold_evidence_list"] if x.get("sentences")]
        else:
            expected_evidence = [original["evidence"]]
        actual_evidence = [item["text"] for item in row["evidence"]]
        evidence_piece_count += len(actual_evidence)
        if actual_evidence != expected_evidence:
            errors["evidence_mismatch"] += 1
        evidence = "\n".join(f'[{item["evidence_id"]}] {item["text"]}' for item in row["evidence"])
        prompts = {condition: render_verdict(row, condition, scheme) for condition in CONDITIONS}
        if row["claim"] not in prompts["full"] or any(item["text"] not in prompts["full"] for item in row["evidence"]):
            errors["full_missing_field"] += 1
        altered = copy(row)
        altered["label"] = "NEI" if row["label"] != "NEI" else "SUPPORTS"
        if any(render_verdict(altered, condition, scheme) != prompts[condition] for condition in CONDITIONS):
            errors["gold_label_leaked"] += 1
        needle = f"Evidence: {evidence}\n\n"
        pos = prompts["full"].rfind(needle)
        if pos < 0 or prompts["claim_only"] != (prompts["full"][:pos] +
                                                 "Evidence: [not provided]\n\n" +
                                                 prompts["full"][pos + len(needle):]):
            errors["claim_only_other_difference"] += 1
        needle = f"Claim: {row['claim']}"
        pos = prompts["full"].find(needle)
        if pos < 0 or prompts["evidence_only"] != (prompts["full"][:pos] +
                                                    "Claim: [not provided]" +
                                                    prompts["full"][pos + len(needle):]):
            errors["evidence_only_other_difference"] += 1
        for condition, prompt in prompts.items():
            ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            total = len(ids) + max(len(tokenizer(" " + answer, add_special_tokens=False)["input_ids"])
                                   for answer in verbalizers)
            lengths.append(total)
            if total > longest[0]:
                longest = (total, row["sample_id"], condition)
            if total > max_positions:
                errors["over_model_context"] += 1
    by_condition = defaultdict(dict)
    for item in predictions:
        condition, sample_id = item["condition"], item["sample_id"]
        if condition not in evaluated_conditions or sample_id not in by_id or sample_id in by_condition[condition]:
            errors["prediction_key_invalid_or_duplicate"] += 1
            continue
        by_condition[condition][sample_id] = item
        if item["gold"] != by_id[sample_id]["label"]:
            errors["prediction_gold_mismatch"] += 1
        if item.get("inference_mode") == "generate":
            label, surface = parse_generated_verdict(item.get("raw_output", ""), labels, verbalizers)
            if item["prediction"] != label or item["prediction_surface"] != surface:
                errors["generation_parse_or_mapping_mismatch"] += 1
        else:
            if len(item["scores"]) != len(labels) or not all(math.isfinite(score) for score in item["scores"]):
                errors["score_invalid"] += 1
                continue
            winner = max(range(len(labels)), key=lambda i: item["scores"][i])
            if item["prediction"] != labels[winner] or item["prediction_surface"] != verbalizers[winner]:
                errors["prediction_argmax_or_mapping_mismatch"] += 1
    for condition in evaluated_conditions:
        pred = by_condition[condition]
        if len(pred) != len(samples):
            errors[f"{condition}_prediction_count_mismatch"] += 1
            continue
        truth = [row["label"] for row in samples]
        values = [pred[row["sample_id"]]["prediction"] for row in samples]
        saved = metrics[condition]
        independently = {
            "accuracy": accuracy_score(truth, values),
            "macro_f1": f1_score(truth, values, labels=labels, average="macro", zero_division=0),
            "balanced_accuracy": sum(
                sum(prediction == label for gold, prediction in zip(truth, values) if gold == label)
                / sum(gold == label for gold in truth)
                for label in labels if label in truth
            ) / sum(label in truth for label in labels),
            "invalid_rate": 1 - sum(value in labels for value in values) / len(values),
            "per_class_f1": dict(zip(labels, f1_score(truth, values, labels=labels,
                                                       average=None, zero_division=0))),
            "confusion_matrix": confusion_matrix(truth, values, labels=labels + ["INVALID"]).tolist(),
        }
        for key, value in independently.items():
            if isinstance(value, dict):
                matches = all(math.isclose(float(item), float(saved[key][label]), abs_tol=1e-12)
                              for label, item in value.items())
            elif isinstance(value, float):
                matches = math.isclose(value, saved[key], abs_tol=1e-12)
            else:
                matches = value == saved[key]
            if not matches:
                errors[f"{condition}_{key}_mismatch"] += 1
    return {"dataset": name, "source_path": str(source), "source_rows": len(raw),
            "samples": len(samples), "source_label_counts": dict(source_counts),
            "processed_label_counts": dict(processed_counts), "evidence_pieces": evidence_piece_count,
            "claim_whitespace_trimmed": changed_claim_whitespace,
            "verdict_scheme": scheme, "verbalizers": dict(zip(labels, verbalizers)),
            "max_model_positions": max_positions, "max_input_plus_answer_tokens": longest[0],
            "longest_sample_id": longest[1], "longest_condition": longest[2],
            "errors": dict(errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="e0_dataset_prompts")
    args = parser.parse_args()
    run = ROOT / "outputs/experiment_runs" / args.run_id
    cfg = yaml.safe_load((run / "fever/baseline.yaml").read_text(encoding="utf8"))
    spec = cfg["model"]
    snapshots = Path(spec["cache_dir"]) / ("models--" + spec["id"].replace("/", "--")) / "snapshots"
    snapshot = sorted(path for path in snapshots.glob("*") if (path / "config.json").is_file())[-1]
    model_config = json.loads((snapshot / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    launcher_path = run / "launcher_manifest.json"
    names = json.loads(launcher_path.read_text())["datasets"] if launcher_path.exists() else NAMES
    report = {"run_id": args.run_id, "model_id": spec["id"],
              "datasets": [check(name, run, tokenizer, model_config["max_position_embeddings"]) for name in names]}
    output = run / "e0_integrity_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(item["errors"] for item in report["datasets"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
