"""Command line entry points for the staged research pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from . import experiments, training, analysis, baselines
from .data import read_jsonl, validate_dataset, write_jsonl, build_fever_wiki_index


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audit_export(cfg):
    rows = read_jsonl(cfg["data"]["samples"])
    pairs = read_jsonl(cfg["data"]["pairs"])
    validate_dataset(rows, pairs)
    by_id = {x["sample_id"]: x for x in rows}
    pending = []
    for p in pairs:
        if p.get("audit_status") == "approved":
            continue
        a, b = by_id[p["sample_a"]], by_id[p["sample_b"]]
        pending.append({"pair_id": p["pair_id"], "kind": p["kind"], "claim": a["claim"],
                        "label_a": a["label"], "label_b": b["label"],
                        "evidence_a": a["evidence"], "evidence_b": b["evidence"],
                        "metadata_a": a["claim_metadata"], "metadata_b": b["claim_metadata"],
                        "label_correct": None, "bias_preserves_relation": None,
                        "edit_changes_relation": None, "evidence_sufficient": None,
                        "reviewer": None, "reason": None, "approved": False})
    write_jsonl(cfg["output_path"], pending)
    return {"pending": len(pending), "output_path": cfg["output_path"]}


def _audit_import(cfg):
    rows = read_jsonl(cfg["data"]["samples"])
    pairs = read_jsonl(cfg["data"]["pairs"])
    decisions = read_jsonl(cfg["review_path"])
    by_pair = {p["pair_id"]: p for p in pairs}
    by_sample = {r["sample_id"]: r for r in rows}
    for d in decisions:
        p = by_pair[d["pair_id"]]
        if p["audit_status"] != "pending":
            raise ValueError(f"Pair already resolved: {p['pair_id']}")
        required = ["label_correct", "evidence_sufficient"]
        required += ["bias_preserves_relation"] if p["kind"] == "bias_change" else ["edit_changes_relation"]
        if not d.get("reviewer") or not d.get("reason"):
            raise ValueError(f"Reviewer and reason required: {p['pair_id']}")
        if d.get("approved") and not all(d.get(k) is True for k in required):
            raise ValueError(f"Approval requires positive review fields: {p['pair_id']}")
        p["audit_status"] = "approved" if d.get("approved") else "rejected"
        p["reviewer"], p["audit_reason"] = d["reviewer"], d["reason"]
        if d.get("approved"):
            for sample_id in [p["sample_a"], p["sample_b"]]:
                by_sample[sample_id]["counterfactual_verified"] = True
                by_sample[sample_id]["verification_method"] = "human_review"
    validate_dataset(rows, pairs)
    out = Path(cfg["output_dir"])
    write_jsonl(out / "samples.jsonl", rows)
    write_jsonl(out / "pairs.jsonl", pairs)
    return {"decisions": len(decisions), "output_dir": str(out)}


def _merge(cfg):
    # Preserve high-priority evaluation groups; emit a decontaminated subset,
    # never overwrite the original official files.
    from collections import defaultdict
    all_rows, all_pairs = [], []
    for directory in cfg["input_dirs"]:
        directory = Path(directory)
        all_rows.extend(read_jsonl(directory / "samples.jsonl"))
        all_pairs.extend(read_jsonl(directory / "pairs.jsonl"))
    priority = {"train": 0, "dev": 1, "test": 2}
    keep_group, keep_family = {}, {}
    for row in all_rows:
        for mapping, key in [(keep_group, row["group_id"]),
                             (keep_family, row["source_provenance"].get("revision_family_id", row["source_id"]))]:
            if key not in mapping or priority[row["split"]] > priority[mapping[key]]:
                mapping[key] = row["split"]
    kept = [r for r in all_rows if r["split"] == keep_group[r["group_id"]]
            and r["split"] == keep_family[r["source_provenance"].get("revision_family_id", r["source_id"])]]
    kept_ids = {r["sample_id"] for r in kept}
    pairs = [p for p in all_pairs if p["sample_a"] in kept_ids and p["sample_b"] in kept_ids]
    report = validate_dataset(kept, pairs)
    out = Path(cfg["output_dir"])
    write_jsonl(out / "samples.jsonl", kept)
    write_jsonl(out / "pairs.jsonl", pairs)
    _report = {**report, "dropped_lower_priority_examples": len(all_rows) - len(kept),
               "split_priority": ["test", "dev", "train"], "input_dirs": cfg["input_dirs"]}
    (out / "merge_audit.json").write_text(json.dumps(_report, indent=2), encoding="utf8")
    return _report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crestfc")
    parser.add_argument("command", choices=["prepare", "index-wiki", "merge", "validate-data", "export-audit", "import-audit",
                                            "baseline", "extract-activations", "train-probes",
                                            "causal-patch", "select-layers", "calibrate-decoder", "train-baseline",
                                            "train-steering", "evaluate", "evaluate-counterfactual", "plot-probes", "report"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
    _seed(cfg.get("seed", 42))
    commands = {
        "prepare": experiments.prepare,
        "index-wiki": lambda c: build_fever_wiki_index(c["wiki_dir"], c["output_db"]),
        "merge": _merge,
        "validate-data": experiments.validate_data,
        "export-audit": _audit_export,
        "import-audit": _audit_import,
        "baseline": experiments.baseline,
        "extract-activations": experiments.extract,
        "train-probes": experiments.train_probes,
        "causal-patch": experiments.causal_patch,
        "select-layers": experiments.select_layers,
        "calibrate-decoder": training.calibrate,
        "train-baseline": baselines.train_baseline,
        "train-steering": training.steer,
        "evaluate": training.evaluate,
        "evaluate-counterfactual": training.evaluate,
        "plot-probes": analysis.plot_probes,
        "report": analysis.report,
    }
    result = commands[args.command](cfg)
    output = cfg.get("output_dir") or cfg.get("run_dir")
    if output and args.command not in {"validate-data", "select-layers"}:
        import platform, subprocess, datetime, hashlib
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf8")
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except Exception:
            git_commit = None
            git_dirty = None
        input_hashes = {}
        for candidate in [cfg.get("data", {}).get("samples"), cfg.get("data", {}).get("pairs"),
                          cfg.get("source", {}).get("path")]:
            if candidate and Path(candidate).is_file():
                h = hashlib.sha256()
                with open(candidate, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                input_hashes[candidate] = h.hexdigest()
        manifest = {"command": args.command, "utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "git_commit": git_commit, "git_dirty": git_dirty, "python": platform.python_version(),
                    "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "seed": cfg.get("seed", 42), "input_sha256": input_hashes}
        model_snapshots = {}
        for key in ["model", "encoder", "decoder"]:
            spec = cfg.get(key, {})
            if spec.get("id") and spec.get("cache_dir"):
                snapshots = Path(spec["cache_dir"]) / ("models--" + spec["id"].replace("/", "--")) / "snapshots"
                found = sorted(p.name for p in snapshots.glob("*") if (p / "config.json").exists())
                model_snapshots[key] = found[-1] if found else None
        manifest["local_model_snapshots"] = model_snapshots
        from .modeling import PROMPT_VERSION, VERDICT_PROMPT_VERSIONS
        manifest["prompt_version"] = (VERDICT_PROMPT_VERSIONS[cfg.get("verdict_scheme", "fever3")]
                                      if args.command in {"baseline", "train-baseline"}
                                      else PROMPT_VERSION)
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
