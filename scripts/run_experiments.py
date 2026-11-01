#!/usr/bin/env python3
"""Run CReST-FC stages on a GPU with generated, reviewable YAML configs.

Examples:
    python scripts/run_experiments.py prepare --run-id trial1
    python scripts/run_experiments.py all --run-id trial1 --reviewed-data data/processed/pilot_reviewed
    python scripts/run_experiments.py e4 --run-id trial1 --reviewed-data ... --split test
"""
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
STAGES = ("prepare", "e0", "e1", "e2", "e3a", "e3b", "baseline-lora", "e4", "report", "all")


def config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf8"))


def data_paths(directory: Path) -> dict:
    return {"samples": str(directory / "samples.jsonl"),
            "pairs": str(directory / "pairs.jsonl")}


def require_data(directory: Path) -> None:
    for path in data_paths(directory).values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing data file: {path}")


def check_gpu(model_id: str, cache_dir: Path) -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Use Python 3.10; current version is {sys.version.split()[0]}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a GPU. Check the Python environment and CUDA wheel.")
    if not (torch.version.cuda or "").startswith("12."):
        raise RuntimeError(f"Expected a CUDA 12.x PyTorch wheel; got {torch.version.cuda}")
    snapshots = cache_dir / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not any((p / "config.json").is_file() for p in snapshots.glob("*")):
        raise FileNotFoundError(f"No complete local model snapshot for {model_id} in {cache_dir}")
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | "
          f"CUDA runtime {torch.version.cuda} | {torch.cuda.get_device_name(0)}", flush=True)


class Runner:
    def __init__(self, args):
        self.args = args
        self.run = ROOT / "outputs" / "experiment_runs" / args.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.pilot = ROOT / "data" / "processed" / "pilot"
        self.reviewed = (Path(args.reviewed_data).expanduser().resolve()
                         if args.reviewed_data else ROOT / "data" / "processed" / "pilot_reviewed")
        self.cache = Path(args.cache_dir).expanduser().resolve()

    def model(self) -> dict:
        return {"id": self.args.model_id, "cache_dir": str(self.cache),
                "dtype": "bfloat16", "device": "cuda"}

    def run_cli(self, command: str, base: str, stage: str, *, reviewed: bool = False,
                changes: dict | None = None, output: Path | None = None) -> dict:
        cfg = config(base)
        cfg["seed"] = self.args.seed
        for key in ("model", "encoder", "decoder"):
            if key in cfg:
                cfg[key] = self.model()
        if "data" in cfg:
            directory = self.reviewed if reviewed else self.pilot
            require_data(directory)
            cfg["data"] = data_paths(directory)
        if output is not None:
            cfg["output_dir"] = str(output)
        if self.args.unlimited:
            cfg.pop("limit", None)
            if "source" in cfg:
                cfg["source"].pop("limit", None)
        elif self.args.limit is not None:
            cfg["limit"] = self.args.limit
            if "source" in cfg:
                cfg["source"]["limit"] = self.args.limit
        if changes:
            cfg.update(changes)
        cfg_path = self.run / "launch_configs" / f"{stage}_{command}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        print(f"[{stage}] {command}: {cfg_path}", flush=True)
        subprocess.run([sys.executable, "-m", "crestfc.cli", command, "--config", str(cfg_path)],
                       cwd=ROOT, env=env, check=True)
        return cfg

    def selection(self) -> Path:
        path = self.run / "e2" / "candidate_layers_final.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run E2 first: {path}")
        value = json.loads(path.read_text())
        if not value.get("selected_layers"):
            raise RuntimeError("E2 selected no causal layer; review E2 controls and data before E3")
        return path

    def stage(self, name: str) -> None:
        if name == "prepare":
            self.run_cli("prepare", "prepare_vitaminc.yaml", "prepare_train",
                         output=ROOT / "data" / "processed" / "vitaminc_train_pilot")
            self.run_cli("prepare", "prepare_vitaminc_dev.yaml", "prepare_dev",
                         output=ROOT / "data" / "processed" / "vitaminc_dev_pilot")
            self.run_cli("merge", "merge_pilot.yaml", "prepare_merge", output=self.pilot)
            self.run_cli("export-audit", "audit_export.yaml", "prepare_audit")
            print(f"Review pairs in {ROOT / 'data/audits/pending_vitaminc_pilot.jsonl'} before E2/E3.")
        elif name == "e0":
            self.run_cli("baseline", "pilot.yaml", "e0", reviewed=self.args.reviewed_data is not None,
                         output=self.run / "e0")
        elif name == "e1":
            activation_dir = self.run / "e1" / "activations"
            use_reviewed = self.args.reviewed_data is not None
            for split in ("train", "dev"):
                self.run_cli("extract-activations", "probe.yaml", f"e1_{split}", reviewed=use_reviewed,
                             changes={"split": split}, output=activation_dir)
            self.run_cli("train-probes", "probe_train.yaml", "e1_probes", reviewed=use_reviewed,
                         changes={"activation_dir": str(activation_dir)},
                         output=self.run / "e1" / "probes")
            self.run_cli("plot-probes", "plot_probes.yaml", "e1_plot",
                         changes={"run_dir": str(self.run / "e1" / "probes")})
        elif name == "e2":
            require_data(self.reviewed)
            probe_path = self.run / "e1" / "probes" / "best_probes_on_dev.json"
            if not probe_path.is_file():
                raise FileNotFoundError(f"Run E1 first: {probe_path}")
            probes = json.loads(probe_path.read_text())
            ranked = sorted(((int(key.rsplit("_layer", 1)[1].split(".")[0]), val["macro_f1"])
                             for key, val in probes.items() if key.startswith("final_anchor_layer")),
                            key=lambda item: item[1], reverse=True)
            layers = [layer for layer, _ in ranked[:6]]
            if not layers:
                raise RuntimeError("E1 produced no final-anchor probe layers")
            self.run_cli("causal-patch", "patch.yaml", "e2", reviewed=True,
                         changes={"layers": layers}, output=self.run / "e2")
            self.run_cli("select-layers", "select_layers.yaml", "e2_select",
                         changes={"run_dir": str(self.run / "e2")})
        elif name == "e3a":
            selected = json.loads(self.selection().read_text())["selected_layers"]
            self.run_cli("calibrate-decoder", "decoder_calibration.yaml", "e3a", reviewed=True,
                         changes={"read_layer": max(selected), "guided_weight": 0.5},
                         output=self.run / "e3a")
        elif name == "e3b":
            selected_path = self.selection()
            checkpoint = self.run / "e3a" / "best"
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"Run E3a first: {checkpoint}")
            self.run_cli("train-steering", "steering.yaml", "e3b", reviewed=True,
                         changes={"selected_layers_path": str(selected_path),
                                  "allow_manual_layers": False,
                                  "calibration_checkpoint": str(checkpoint)},
                         output=self.run / "e3b")
        elif name == "baseline-lora":
            selected = json.loads(self.selection().read_text())["selected_layers"]
            self.run_cli("train-baseline", "baseline_lora.yaml", "baseline_lora", reviewed=True,
                         changes={"layers": selected, "include_approved_pairs": True},
                         output=self.run / "baseline_lora")
        elif name == "e4":
            checkpoint = self.run / "e3a" / "best"
            steering = self.run / "e3b"
            if not checkpoint.is_dir() or not (steering / "encoder_lora.pt").is_file():
                raise FileNotFoundError("Run E3a and E3b before E4")
            self.run_cli("evaluate", "evaluate.yaml", "e4", reviewed=True,
                         changes={"calibration_checkpoint": str(checkpoint),
                                  "steering_checkpoint": str(steering), "split": self.args.split,
                                  "guided": True},
                         output=self.run / "e4")
        elif name == "report":
            self.run_cli("report", "report.yaml", "report",
                         changes={"runs": [str(self.run / p) for p in
                                           ("e0", "e1/probes", "e2", "e4")]},
                         output=self.run / "report")
        else:
            raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", default="/data/yangjun/.cache/huggingface")
    parser.add_argument("--reviewed-data", help="Directory with reviewed samples.jsonl and pairs.jsonl")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlimited", action="store_true", help="Remove pilot sample limits")
    parser.add_argument("--limit", type=int, help="Override per-stage pilot sample limit")
    args = parser.parse_args()
    os.chdir(ROOT)
    check_gpu(args.model_id, Path(args.cache_dir).expanduser())
    runner = Runner(args)
    if args.stage == "all":
        if not args.reviewed_data:
            for stage in ("prepare", "e0", "e1"):
                runner.stage(stage)
            raise SystemExit("Stopped at human review gate. Import reviewed pairs, then rerun with --reviewed-data.")
        for stage in ("e0", "e1", "e2", "e3a", "e3b", "baseline-lora", "e4", "report"):
            runner.stage(stage)
    else:
        runner.stage(args.stage)


if __name__ == "__main__":
    main()
