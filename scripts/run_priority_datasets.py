#!/usr/bin/env python3
"""Run unreviewed E0 diagnostics on five requested datasets, one GPU each."""
from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
        raise RuntimeError("Five visible CUDA GPUs are required; do not set CUDA_VISIBLE_DEVICES")
    os.chdir(ROOT)
    run = ROOT / "outputs" / "experiment_runs" / args.run_id
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run}; use a new --run-id")
    for _, _, source, _ in SOURCES:
        source_path = ROOT / source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    run.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
    processes = []
    for gpu, (name, dataset, source, split) in enumerate(SOURCES):
        source_path = ROOT / source
        data_dir = run / name / "data"
        prep = {"seed": 42, "source": {"dataset": dataset, "path": str(source_path),
                                      "split": split, **({"limit": args.limit} if args.limit else {})},
                "output_dir": str(data_dir), "max_relation_pairs": 0, "add_bias_pairs": False}
        prep_path = run / name / "prepare.yaml"
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        prep_path.write_text(yaml.safe_dump(prep, sort_keys=False), encoding="utf8")
        subprocess.run([sys.executable, "-m", "crestfc.cli", "prepare", "--config", str(prep_path)],
                       cwd=ROOT, env=env, check=True)
        audit = json.loads((data_dir / "audit.json").read_text(encoding="utf8"))
        if audit["accepted"] == 0:
            raise RuntimeError(f"No accepted examples for {name}: {audit['rejected']}")
        baseline = {"seed": 42, "model": {"id": args.model_id, "cache_dir": args.cache_dir,
                                           "dtype": "bfloat16", "device": "cuda"},
                    "data": {"samples": str(data_dir / "samples.jsonl"),
                             "pairs": str(data_dir / "pairs.jsonl")},
                    "labels": ["SUPPORTS", "REFUTES", "NEI"], "split": split,
                    "conditions": ["full", "claim_only", "evidence_only"],
                    "output_dir": str(run / name / "e0")}
        cfg_path = run / name / "baseline.yaml"
        cfg_path.write_text(yaml.safe_dump(baseline, sort_keys=False), encoding="utf8")
        child_env = env.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = (run / name / "e0.log").open("w", encoding="utf8")
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
