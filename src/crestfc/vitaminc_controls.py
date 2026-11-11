"""Supplement the completed VitaminC E1-B scan with frozen-reader controls."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import StoppingCriteria, StoppingCriteriaList

from .data import read_jsonl, write_jsonl
from .evaluation import parse_generated_verdict
from .frozen_reader import (LABELS, QUESTION, candidate_ids, classify, decoder_prefix,
                            digest, dump, score_labels, slot_id)
from .modeling import load_model, render_verdict
from .vitaminc_claimflip import ROOT, TIE_TOL, score_all_layers, shard_for_family

CONDITIONS = ("evidence_only_patch", "random_evidence_patch", "no_patch_slots",
              "direct_text_7b", "direct_text_7b_generate")
LAYER_CONDITIONS = ("full_patch", "claim_only_patch", "evidence_only_patch", "random_evidence_patch")
CONTROL_FIELDS = {"evidence_only_patch": "evidence_only_scores",
                  "random_evidence_patch": "random_evidence_scores",
                  "no_patch_slots": "no_patch_scores", "direct_text_7b": "direct_text_scores"}
ALL_FIELDS = {"full_patch": "full_scores", "claim_only_patch": "claim_only_scores", **CONTROL_FIELDS}


def select_random_evidence(samples: list[dict], seed: int = 42) -> list[dict]:
    """One label-blind, near-length donor from another family for each target family."""
    by_family: dict[str, list[dict]] = {}
    for row in samples:
        by_family.setdefault(row["source_family"], []).append(row)
    if len(by_family) < 2:
        raise ValueError("Random Evidence requires at least two source families")
    representatives = [min(rows, key=lambda x: x["sample_id"]) for rows in by_family.values()]
    pool = sorted((len(row["evidence"][0]["text"]), row["source_family"], row)
                  for row in representatives)
    sizes = [item[0] for item in pool]
    mapping = []
    for family, rows in sorted(by_family.items()):
        own_evidence = {r["evidence"][0]["text"] for r in rows}
        length = int(np.median([len(t) for t in own_evidence]))
        pos = bisect.bisect_left(sizes, length)
        candidates = []
        left, right = pos - 1, pos
        # Consider a local pool and sample among its 32 closest valid donors.
        while len(candidates) < 32 and (left >= 0 or right < len(pool)):
            if left < 0:
                take_right = True
            elif right == len(pool):
                take_right = False
            else:
                take_right = abs(sizes[right] - length) < abs(sizes[left] - length)
            item = pool[right] if take_right else pool[left]
            right += int(take_right)
            left -= int(not take_right)
            donor = item[2]
            if donor["source_family"] != family and donor["evidence"][0]["text"] not in own_evidence:
                candidates.append(donor)
        if not candidates:
            raise ValueError(f"No unrelated evidence for family {family}")
        index = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), 16) % len(candidates)
        donor = candidates[index]
        mapping.append({"source_family": family, "donor_id": donor["sample_id"],
                        "donor_family": donor["source_family"],
                        "donor_evidence_sha256": hashlib.sha256(donor["evidence"][0]["text"].encode()).hexdigest(),
                        "target_evidence_median_chars": length,
                        "donor_evidence_chars": len(donor["evidence"][0]["text"]),
                        "selection": "seeded choice among 32 nearest-length valid family representatives"})
    return mapping


def _source_run(args) -> tuple[Path, dict, list[dict], list[dict]]:
    source = (ROOT / "outputs/experiment_runs" / args.base_run).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["protocol"] != "e1b_vitaminc_dev_claimflip_v1" or
            manifest["layers"] != list(range(28)) or manifest["layer_batch"] != 7 or
            manifest["conditions"] != ["full_patch", "claim_only_patch"]):
        raise ValueError("Base run must be the complete 28-layer VitaminC E1-B protocol")
    samples = read_jsonl(source / "samples.jsonl")
    pairs = read_jsonl(source / "claimflip_pairs.jsonl")
    if len(samples) != manifest["n_samples"] or len(pairs) != manifest["n_pairs"]:
        raise ValueError("Base run input counts changed")
    return source, manifest, samples, pairs


def prepare(args) -> dict:
    source, base, all_samples, all_pairs = _source_run(args)
    if args.max_families:
        families = {p["source_family"] for p in all_pairs[:args.max_families]}
        samples = [r for r in all_samples if r["source_family"] in families]
        pairs = [p for p in all_pairs if p["source_family"] in families]
    else:
        samples, pairs = all_samples, all_pairs
    if not samples or not pairs:
        raise ValueError("No target samples or claimflip pairs")
    donors = select_random_evidence(all_samples, args.seed)
    target_families = {r["source_family"] for r in samples}
    donors = [d for d in donors if d["source_family"] in target_families]
    selected_ids = [r["sample_id"] for r in samples]
    donor_ids = [(d["source_family"], d["donor_id"]) for d in donors]
    run = ROOT / "outputs/experiment_runs" / args.run_id
    manifest = {"protocol": "e1b_vitaminc_controls_v1", "run_id": args.run_id,
                "base_run": str(source), "base_manifest_sha256": digest(source / "manifest.json"),
                "base_samples_sha256": digest(source / "samples.jsonl"),
                "base_pairs_sha256": digest(source / "claimflip_pairs.jsonl"),
                "model_snapshot": base["snapshot"], "model_config_sha256": base["snapshot_config_sha256"],
                "conditions": list(CONDITIONS), "layer_conditions": list(LAYER_CONDITIONS),
                "layers": list(range(28)), "layer_batch": 7, "slot_text": base["slot_text"],
                "n_samples": len(samples), "n_pairs": len(pairs), "n_families": len({r["source_family"] for r in samples}),
                "n_shards": args.shards, "max_families": args.max_families, "seed": args.seed,
                "selected_ids_sha256": hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest(),
                "donor_ids_sha256": hashlib.sha256(json.dumps(donor_ids).encode()).hexdigest(),
                "no_training": True, "random_evidence_gold_valid": False,
                "direct_text_prompt": "v4_fever_binary_supports_refutes",
                "decoder_question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
                "control_code_sha256": digest(Path(__file__)),
                "reader_code_sha256": digest(Path(__file__).with_name("frozen_reader.py")),
                "base_scan_code_sha256": digest(Path(__file__).with_name("vitaminc_claimflip.py"))}
    if (run / "manifest.json").is_file():
        if json.loads((run / "manifest.json").read_text()) != manifest:
            raise ValueError("Existing control run protocol differs; use a new run ID")
        if ([r["sample_id"] for r in read_jsonl(run / "samples.jsonl")] != selected_ids or
                read_jsonl(run / "claimflip_pairs.jsonl") != pairs or
                read_jsonl(run / "random_evidence_donors.jsonl") != donors):
            raise ValueError("Existing control inputs differ")
        return manifest
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "samples.jsonl", samples)
    write_jsonl(run / "claimflip_pairs.jsonl", pairs)
    write_jsonl(run / "random_evidence_donors.jsonl", donors)
    dump(run / "manifest.json", manifest)
    dump(run / "data_audit.json", {"target_labels": dict(Counter(x["label"] for x in samples)),
                                   "n_donors": len(donors), "reviewed_pairs": 0,
                                   "random_evidence_note": "Original gold labels are not ground truth after evidence replacement",
                                   "shard_sizes": {str(i): sum(shard_for_family(r["source_family"], args.shards) == i
                                                               for r in samples) for i in range(args.shards)}})
    return manifest


def _score_no_patch(model, tokenizer, row, sid, candidates):
    from .frozen_reader import encoder_text
    t = len(tokenizer.encode(encoder_text(row, "full_patch"), add_special_tokens=False))
    prefix = decoder_prefix(tokenizer, t, sid)
    expected = tokenizer.apply_chat_template([{"role": "user", "content": QUESTION}],
                                             tokenize=True, add_generation_prompt=True)
    if prefix[:t] != [sid] * t or prefix[t:] != list(expected):
        raise ValueError("No Patch decoder prefix is not fixed slots and question")
    return score_labels(model, prefix, candidates)[0], t


class _StopAtLabelOrNewline(StoppingCriteria):
    def __init__(self, tokenizer, prefix_tokens: int):
        self.tokenizer = tokenizer
        self.prefix_tokens = prefix_tokens

    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(input_ids[0, self.prefix_tokens:], skip_special_tokens=True)
        return generated.strip() in LABELS or "\n" in generated


@torch.inference_mode()
def _generate_direct(model, tokenizer, prompt_ids: list[int]) -> tuple[str, str, int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    output = model.generate(input_ids=ids, do_sample=False, max_new_tokens=8,
                            use_cache=True,
                            stopping_criteria=StoppingCriteriaList([_StopAtLabelOrNewline(tokenizer, len(prompt_ids))]),
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    generated = output[0, len(prompt_ids):]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    label, _ = parse_generated_verdict(raw, list(LABELS), list(LABELS))
    return label, raw, int(generated.numel())


def scan_shard(args) -> None:
    run = ROOT / "outputs/experiment_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text())
    if not 0 <= args.shard < manifest["n_shards"]:
        raise ValueError("Invalid shard")
    samples = [r for r in read_jsonl(run / "samples.jsonl")
               if shard_for_family(r["source_family"], manifest["n_shards"]) == args.shard]
    source_samples = {r["sample_id"]: r for r in read_jsonl(Path(manifest["base_run"]) / "samples.jsonl")}
    donors = {d["source_family"]: d for d in read_jsonl(run / "random_evidence_donors.jsonl")}
    out = run / "shards"
    out.mkdir(exist_ok=True)
    output = out / f"shard{args.shard}.jsonl"
    marker = out / f"shard{args.shard}_complete.json"
    if marker.is_file():
        done_marker = json.loads(marker.read_text())
        if done_marker["n_samples"] == len(samples) and done_marker["sha256"] == digest(output):
            print(f"control shard {args.shard} complete; reusing", flush=True)
            return
        raise ValueError("Control shard completion marker invalid")
    existing = read_jsonl(output) if output.is_file() else []
    done = {r["sample_id"] for r in existing}
    if len(done) != len(existing) or not done <= {r["sample_id"] for r in samples}:
        raise ValueError("Existing control shard has duplicate or foreign IDs")
    model, tokenizer = load_model(manifest["model_snapshot"], None, "bfloat16", "cuda")
    model.eval().requires_grad_(False)
    if model.config.num_hidden_layers != 28 or model.config.hidden_size != 3584:
        raise ValueError("Expected frozen Qwen2.5-7B with 28 blocks")
    sid, candidates = slot_id(tokenizer, manifest["slot_text"]), candidate_ids(tokenizer)
    dump(out / f"shard{args.shard}_runtime.json", {"gpu": torch.cuda.get_device_name(0),
                                                 "frozen_parameters": sum(p.requires_grad for p in model.parameters()),
                                                 "candidate_ids": candidates, "slot_id": sid})
    evidence_cache, no_patch_cache = {}, {}
    start = time.monotonic()
    with output.open("a", encoding="utf8") as stream:
        for index, row in enumerate(samples, 1):
            if row["sample_id"] in done:
                continue
            own_evidence = row["evidence"][0]["text"]
            donor_info = donors[row["source_family"]]
            donor = source_samples[donor_info["donor_id"]]
            donor_evidence = donor["evidence"][0]["text"]
            if donor["source_family"] == row["source_family"] or donor_evidence == own_evidence:
                raise ValueError(f"Invalid Random Evidence donor: {row['sample_id']}")
            if own_evidence not in evidence_cache:
                evidence_cache[own_evidence] = score_all_layers(
                    model, tokenizer, row, "evidence_only_patch", sid, candidates, 7)
            random_row = {**row, "evidence": [{"evidence_id": "E1", "text": donor_evidence}]}
            random_scores = score_all_layers(model, tokenizer, random_row, "full_patch", sid, candidates, 7)
            from .frozen_reader import encoder_text
            full_tokens = len(tokenizer.encode(encoder_text(row, "full_patch"), add_special_tokens=False))
            if full_tokens not in no_patch_cache:
                scores, checked_tokens = _score_no_patch(model, tokenizer, row, sid, candidates)
                if checked_tokens != full_tokens:
                    raise RuntimeError("No Patch slot count changed")
                no_patch_cache[full_tokens] = scores
            no_patch_scores = no_patch_cache[full_tokens]
            direct_ids = tokenizer.encode(render_verdict(row, scheme="fever_binary"), add_special_tokens=False)
            direct_scores = score_labels(model, direct_ids, candidates)[0]
            direct_label, direct_raw, direct_tokens = _generate_direct(model, tokenizer, direct_ids)
            record = {"sample_id": row["sample_id"], "source_family": row["source_family"],
                      "gold_label": row["label"], "donor_id": donor["sample_id"],
                      "donor_family": donor["source_family"], "donor_evidence_sha256": donor_info["donor_evidence_sha256"],
                      "full_source_tokens": full_tokens, "direct_prompt_tokens": len(direct_ids),
                      "evidence_only_scores": evidence_cache[own_evidence],
                      "random_evidence_scores": random_scores,
                      "no_patch_scores": no_patch_scores, "direct_text_scores": direct_scores,
                      "direct_generated_label": direct_label, "direct_generated_raw": direct_raw,
                      "direct_generated_tokens": direct_tokens}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 100 == 0 or index == len(samples):
                print(f"control shard {args.shard}: {index}/{len(samples)} rows, "
                      f"{time.monotonic() - start:.0f}s elapsed", flush=True)
    dump(marker, {"shard": args.shard, "n_samples": len(samples), "sha256": digest(output)})


def _read_complete_shards(run: Path, n_shards: int, expected: set[str], fields: dict[str, str]) -> dict[str, dict]:
    by_id = {}
    for shard in range(n_shards):
        output = run / "shards" / f"shard{shard}.jsonl"
        marker = json.loads((run / "shards" / f"shard{shard}_complete.json").read_text())
        if marker["sha256"] != digest(output):
            raise ValueError(f"Shard {shard} changed after completion")
        rows = read_jsonl(output)
        if marker["n_samples"] != len(rows):
            raise ValueError(f"Shard {shard} count differs")
        for row in rows:
            sid = row["sample_id"]
            if sid in by_id or sid not in expected:
                raise ValueError(f"Duplicate or foreign prediction: {sid}")
            for condition, field in fields.items():
                scores = row[field]
                if condition in LAYER_CONDITIONS:
                    if len(scores) != 28 or any(len(x) != 2 or not np.isfinite(x).all() for x in scores):
                        raise ValueError(f"Invalid {condition} scores: {sid}")
                elif len(scores) != 2 or not np.isfinite(scores).all():
                    raise ValueError(f"Invalid {condition} scores: {sid}")
            if fields == CONTROL_FIELDS:
                if row.get("direct_generated_label") not in (*LABELS, "INVALID"):
                    raise ValueError(f"Invalid generated label: {sid}")
                if (not isinstance(row.get("direct_generated_raw"), str) or
                        not 1 <= row.get("direct_generated_tokens", 0) <= 8):
                    raise ValueError(f"Invalid generated continuation: {sid}")
            by_id[sid] = row
    return by_id


PAIR_FIELDS = ("pair_id", "source_family", "condition", "layer", "supports_margin",
               "refutes_margin", "direction", "paircorrect")


def _metric_rows(samples: list[dict], pairs: list[dict], by_id: dict[str, dict],
                 pair_writer=None) -> tuple[list[dict], list[dict]]:
    truth = [row["label"] for row in samples]
    if any(y not in LABELS or by_id[row["sample_id"]]["gold_label"] != y for row, y in zip(samples, truth)):
        raise ValueError("Gold labels disagree with base samples")
    n_s, n_r = truth.count("SUPPORTS"), truth.count("REFUTES")
    metrics, pair_rows = [], []
    for condition, field in ALL_FIELDS.items():
        layers = range(28) if condition in LAYER_CONDITIONS else (-1,)
        for layer in layers:
            scores = [by_id[r["sample_id"]][field][layer] if layer >= 0 else by_id[r["sample_id"]][field]
                      for r in samples]
            pred = [classify(x)[0] for x in scores]
            correct_s = sum(y == p == "SUPPORTS" for y, p in zip(truth, pred))
            correct_r = sum(y == p == "REFUTES" for y, p in zip(truth, pred))
            gaps, correct, ties = [], [], []
            for pair in pairs:
                s = by_id[pair["supports_id"]][field]
                r = by_id[pair["refutes_id"]][field]
                if layer >= 0:
                    s, r = s[layer], r[layer]
                ms, mr = s[0] - s[1], r[0] - r[1]
                gap = ms - mr
                tie = abs(gap) <= TIE_TOL
                direction = 0.5 if tie else float(gap > 0)
                good = float(ms > TIE_TOL and mr < -TIE_TOL)
                gaps.append(gap); ties.append(tie); correct.append(good)
                pair_record = {"pair_id": pair["pair_id"], "source_family": pair["source_family"],
                               "condition": condition, "layer": layer,
                               "supports_margin": ms, "refutes_margin": mr,
                               "direction": direction, "paircorrect": good}
                if pair_writer is None:
                    pair_rows.append(pair_record)
                else:
                    pair_writer.writerow(pair_record)
            metrics.append({"condition": condition, "layer": layer, "n_samples": len(samples),
                            "n_pairs": len(pairs), "n_families": len({r["source_family"] for r in samples}),
                            "label_basis": "original_gold_proxy" if condition == "random_evidence_patch" else "gold",
                            "macro_f1": float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)),
                            "accuracy": sum(y == p for y, p in zip(truth, pred)) / len(truth),
                            "balanced_accuracy": 0.5 * (correct_s / n_s + correct_r / n_r),
                            "supports_recall": correct_s / n_s, "refutes_recall": correct_r / n_r,
                            "predict_supports": pred.count("SUPPORTS"), "predict_refutes": pred.count("REFUTES"),
                            "predict_tie": pred.count("TIE"), "predict_invalid": 0,
                            "claimflip_direction": sum(0.5 if t else float(g > 0) for g, t in zip(gaps, ties)) / len(pairs),
                            "claimflip_paircorrect": sum(correct) / len(pairs),
                            "claimflip_tie_rate": sum(ties) / len(pairs),
                            "mean_pair_margin_gap": float(np.mean(gaps))})
    generated = [by_id[r["sample_id"]]["direct_generated_label"] for r in samples]
    correct_s = sum(y == p == "SUPPORTS" for y, p in zip(truth, generated))
    correct_r = sum(y == p == "REFUTES" for y, p in zip(truth, generated))
    generated_paircorrect = []
    for pair in pairs:
        s = by_id[pair["supports_id"]]["direct_generated_label"]
        r = by_id[pair["refutes_id"]]["direct_generated_label"]
        good = float(s == "SUPPORTS" and r == "REFUTES")
        generated_paircorrect.append(good)
        pair_record = {"pair_id": pair["pair_id"], "source_family": pair["source_family"],
                       "condition": "direct_text_7b_generate", "layer": -1,
                       "supports_margin": None, "refutes_margin": None,
                       "direction": None, "paircorrect": good}
        if pair_writer is None:
            pair_rows.append(pair_record)
        else:
            pair_writer.writerow(pair_record)
    metrics.append({"condition": "direct_text_7b_generate", "layer": -1,
                    "n_samples": len(samples), "n_pairs": len(pairs),
                    "n_families": len({r["source_family"] for r in samples}),
                    "label_basis": "gold", "macro_f1": float(f1_score(truth, generated, labels=LABELS,
                                                                         average="macro", zero_division=0)),
                    "accuracy": sum(y == p for y, p in zip(truth, generated)) / len(truth),
                    "balanced_accuracy": 0.5 * (correct_s / n_s + correct_r / n_r),
                    "supports_recall": correct_s / n_s, "refutes_recall": correct_r / n_r,
                    "predict_supports": generated.count("SUPPORTS"),
                    "predict_refutes": generated.count("REFUTES"),
                    "predict_tie": 0, "predict_invalid": generated.count("INVALID"),
                    "claimflip_direction": None,
                    "claimflip_paircorrect": sum(generated_paircorrect) / len(pairs),
                    "claimflip_tie_rate": None, "mean_pair_margin_gap": None})
    return metrics, pair_rows


def analyze(args) -> None:
    run = ROOT / "outputs/experiment_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text())
    base_run = Path(manifest["base_run"])
    for filename, key in (("manifest.json", "base_manifest_sha256"),
                          ("samples.jsonl", "base_samples_sha256"),
                          ("claimflip_pairs.jsonl", "base_pairs_sha256")):
        if digest(base_run / filename) != manifest[key]:
            raise ValueError(f"Base {filename} changed")
    samples = read_jsonl(run / "samples.jsonl")
    pairs = read_jsonl(run / "claimflip_pairs.jsonl")
    expected = {r["sample_id"] for r in samples}
    controls = _read_complete_shards(run, manifest["n_shards"], expected, CONTROL_FIELDS)
    base_manifest = json.loads((base_run / "manifest.json").read_text())
    baseline = _read_complete_shards(base_run, base_manifest["n_shards"],
                                     {r["sample_id"] for r in read_jsonl(base_run / "samples.jsonl")},
                                     {"full_patch": "full_scores", "claim_only_patch": "claim_only_scores"})
    if set(controls) != expected:
        raise ValueError("Control predictions incomplete")
    by_id = {}
    for sid in expected:
        if controls[sid]["gold_label"] != baseline[sid]["gold_label"]:
            raise ValueError(f"Control/base labels disagree for {sid}")
        by_id[sid] = {**baseline[sid], **controls[sid]}
    with (run / "pair_metrics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        metrics, _ = _metric_rows(samples, pairs, by_id, writer)
    with (run / "layer_metrics.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for condition in (*ALL_FIELDS, "direct_text_7b_generate"):
        rows = [r for r in metrics if r["condition"] == condition]
        for ax, field in zip(axes, ("macro_f1", "claimflip_direction", "claimflip_paircorrect")):
            if condition in LAYER_CONDITIONS:
                ax.plot([r["layer"] for r in rows], [r[field] for r in rows], label=condition)
            elif rows[0][field] is not None:
                ax.axhline(rows[0][field], linestyle="--", label=condition)
            ax.set(xlabel="Encoder read layer (zero-based)", ylabel=field, ylim=(-.02, 1.02))
            ax.grid(alpha=.25)
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(run / "layer_curves.png", dpi=180); plt.close(fig)
    dump(run / "analysis_summary.json", {"n_samples": len(samples), "n_pairs": len(pairs),
                                         "n_families": manifest["n_families"],
                                         "source_labels_reviewed": False,
                                         "random_evidence_gold_valid": False,
                                         "random_evidence_interpretation": "Original-label agreement is a diagnostic only; replacement evidence has no verified target relation",
                                         "no_patch_slot_length": "matches each row's Full encoder token length",
                                         "direct_text_scoring": "both full-label teacher-forced log-likelihood and greedy generation with strict exact-label parsing",
                                         "selection_allowed": False})
    print(json.dumps({"metrics": str(run / "layer_metrics.csv"), "n_samples": len(samples),
                      "n_pairs": len(pairs)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "scan-shard", "analyze"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-run", default="e1b_vitaminc_claimflip_full")
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--shard", type=int, default=-1)
    parser.add_argument("--max-families", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.shards < 1 or args.max_families < 0:
        parser.error("Invalid shard count or max-families")
    if args.stage == "prepare":
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2), flush=True)
    elif args.stage == "scan-shard":
        scan_shard(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
