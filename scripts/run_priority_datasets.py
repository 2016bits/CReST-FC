#!/usr/bin/env python3
"""Run unreviewed E0 diagnostics on five requested datasets, one GPU each."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("hover", "HOVER", "data/HOVER/data/converted_data/dev.json", "dev"),
    ("politihop", "PolitiHop", "data/PolitiHopEGR/converted_data/dev.json", "dev"),
    ("symmetric_politihop", "PolitiHop", "data/PolitiHopEGR/converted_data/symmetric.json", "test"),
    ("fever", "FEVER", "data/FEVER/converted_data/dev.json", "dev"),
    ("symmetric_fever", "FEVER", "data/FEVER/converted_data/symmetric.json", "test"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("priority_%Y%m%d_%H%M%S"))
    parser.add_argument("--limit", type=int, default=20, help="Examples per dataset; 0 means all")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", default="/data/yangjun/.cache/huggingface")
    parser.add_argument("--inference-mode", choices=["candidate_score", "generate"],
                        default="candidate_score")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--gpu-tokens", default="0,1,2,3,4",
                        help="Physical GPU IDs assigned in dataset order")
    parser.add_argument("--conditions", default="full,claim_only,evidence_only")
    parser.add_argument("--datasets", default=",".join(item[0] for item in SOURCES))
    parser.add_argument("--fever-binary", action="store_true",
                        help="Use a SUPPORTS/REFUTES-only FEVER prompt as an auxiliary E1-matched task")
    parser.add_argument("--pin-e1-run", help="Require the exact Qwen snapshot and FEVER dev source from E1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    requested = args.datasets.split(",")
    if len(set(requested)) != len(requested) or any(name not in {item[0] for item in SOURCES} for name in requested):
        parser.error("--datasets must contain unique known dataset names")
    selected = [item for item in SOURCES if item[0] in requested]
    gpus = [int(token) for token in args.gpu_tokens.split(",")]
    if (len(gpus) != len(selected) or len(set(gpus)) != len(gpus) or
            not torch.cuda.is_available() or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in gpus)):
        raise RuntimeError("Provide one distinct available physical GPU ID per dataset")
    conditions = args.conditions.split(",")
    if not conditions or len(set(conditions)) != len(conditions) or any(
            condition not in {"full", "claim_only", "evidence_only"} for condition in conditions):
        parser.error("--conditions must use full,claim_only,evidence_only without duplicates")
    pinned = None
    if args.pin_e1_run:
        e1_run = Path(args.pin_e1_run).resolve()
        e1_manifest_path = e1_run / "manifest.json"
        pinned = json.loads(e1_manifest_path.read_text())
        if args.model_id != pinned["model_id"]:
            raise ValueError("E0 model ID differs from the E1 model ID")
        snapshots = Path(args.cache_dir) / ("models--" + args.model_id.replace("/", "--")) / "snapshots"
        matches = sorted(path for path in snapshots.glob("*") if (path / "config.json").exists())
        if not matches:
            raise FileNotFoundError(f"No local Qwen snapshot in {snapshots}")
        current = matches[-1]
        if current.resolve() != Path(pinned["snapshot"]).resolve():
            raise ValueError("Local Qwen snapshot differs from the E1 snapshot")
        fever_source = ROOT / "data/FEVER/converted_data/dev.json"
        digest = hashlib.sha256(fever_source.read_bytes()).hexdigest()
        if digest != pinned["dev_source_sha256"]:
            raise ValueError("FEVER dev data differs from the E1 dev source")
    os.chdir(ROOT)
    run = ROOT / "outputs" / "experiment_runs" / args.run_id
    run_config = {"model_id": args.model_id, "cache_dir": args.cache_dir,
                  "inference_mode": args.inference_mode, "max_new_tokens": args.max_new_tokens,
                  "generation_strategy": "greedy_stop_on_label_or_newline_v1"
                  if args.inference_mode == "generate" else None,
                  "gpu_tokens": gpus, "conditions": conditions, "datasets": [item[0] for item in selected],
                  "limit": args.limit, "fever_binary": args.fever_binary,
                  "e1_manifest_sha256": hashlib.sha256(e1_manifest_path.read_bytes()).hexdigest()
                  if pinned else None, "e1_snapshot": pinned["snapshot"] if pinned else None}
    config_path = run / "launcher_manifest.json"
    if run.exists() and any(run.iterdir()):
        old_config = json.loads(config_path.read_text()) if config_path.exists() else {}
        immutable = lambda value: {key: item for key, item in value.items() if key != "gpu_tokens"}
        if not args.resume or immutable(old_config) != immutable(run_config):
            raise FileExistsError(f"Existing run differs or --resume was omitted: {run}")
    for _, _, source, _ in selected:
        source_path = ROOT / source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    run.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf8")
    with (run / "gpu_launch_history.jsonl").open("a", encoding="utf8") as history:
        history.write(json.dumps({"utc_time": datetime.utcnow().isoformat() + "Z",
                                  "gpu_tokens": gpus, "resume": args.resume}) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
    processes = []
    for gpu, (name, dataset, source, split) in zip(gpus, selected):
        source_path = ROOT / source
        data_dir = run / name / "data"
        prep = {"seed": 42, "source": {"dataset": dataset, "path": str(source_path),
                                      "split": split, **({"limit": args.limit} if args.limit else {})},
                "output_dir": str(data_dir), "max_relation_pairs": 0, "add_bias_pairs": False}
        prep_path = run / name / "prepare.yaml"
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and prep_path.exists():
            if yaml.safe_load(prep_path.read_text()) != prep or not (data_dir / "samples.jsonl").is_file():
                raise ValueError(f"Prepared data differ from the resumed run: {name}")
        else:
            prep_path.write_text(yaml.safe_dump(prep, sort_keys=False), encoding="utf8")
            subprocess.run([sys.executable, "-m", "crestfc.cli", "prepare", "--config", str(prep_path)],
                           cwd=ROOT, env=env, check=True)
        audit = json.loads((data_dir / "audit.json").read_text(encoding="utf8"))
        if audit["accepted"] == 0:
            raise RuntimeError(f"No accepted examples for {name}: {audit['rejected']}")
        fever3 = name == "fever" and not args.fever_binary
        baseline = {"seed": 42, "model": {"id": args.model_id, "cache_dir": args.cache_dir,
                                           "dtype": "bfloat16", "device": "cuda"},
                    "data": {"samples": str(data_dir / "samples.jsonl"),
                             "pairs": str(data_dir / "pairs.jsonl")},
                    "labels": ["SUPPORTS", "REFUTES", "NEI"] if fever3 else ["SUPPORTS", "REFUTES"],
                    "verdict_scheme": "fever3" if fever3 else
                                      "fever_binary" if name == "fever" else "binary_support",
                    "split": split,
                    "conditions": conditions,
                    "inference_mode": args.inference_mode,
                    "generation_strategy": "greedy_stop_on_label_or_newline_v1"
                    if args.inference_mode == "generate" else None,
                    "max_new_tokens": args.max_new_tokens,
                    "resume": args.inference_mode == "generate",
                    "output_dir": str(run / name / "e0")}
        cfg_path = run / name / "baseline.yaml"
        if args.resume and cfg_path.exists():
            if yaml.safe_load(cfg_path.read_text()) != baseline:
                raise ValueError(f"Baseline config differs from the resumed run: {name}")
        else:
            cfg_path.write_text(yaml.safe_dump(baseline, sort_keys=False), encoding="utf8")
        child_env = env.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = (run / name / "e0.log").open("a" if args.resume else "w", encoding="utf8")
        proc = subprocess.Popen([sys.executable, "-m", "crestfc.cli", "baseline", "--config", str(cfg_path)],
                                cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((name, gpu, proc, log))
        print(f"GPU {gpu}: {name}, PID {proc.pid}, log {run/name/'e0.log'}", flush=True)
    failed = []
    for name, gpu, proc, log in processes:
        code = proc.wait()
        log.close()
        if code:
            failed.append(name)
        print(f"GPU {gpu}: {name} {'failed' if code else 'done'}", flush=True)
    if failed:
        raise SystemExit("Failed datasets: " + ", ".join(failed))


if __name__ == "__main__":
    main()
