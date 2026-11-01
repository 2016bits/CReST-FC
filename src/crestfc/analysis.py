"""Lightweight figures and evidence-bound research report assembly."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path


def plot_probes(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(cfg["run_dir"])
    with (run / "layer_probe_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    best = {}
    for row in rows:
        key = row["position_layer"]
        if key not in best or float(row["macro_f1"]) > float(best[key]["macro_f1"]):
            best[key] = row
    lines = defaultdict(list)
    for name, row in best.items():
        position, layer = name.rsplit("_layer", 1)
        lines[position].append((int(layer.split(".")[0]), float(row["macro_f1"])))
    if not lines:
        raise ValueError("No probe metrics to plot")
    for position, points in sorted(lines.items()):
        points.sort()
        plt.plot([x for x, _ in points], [y for _, y in points], marker=".", label=position)
    plt.xlabel("0-based transformer block")
    plt.ylabel("Dev macro-F1")
    plt.legend()
    plt.tight_layout()
    path = run / "layer_probe_macro_f1.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return {"figure": str(path), "positions": sorted(lines)}


def report(cfg):
    paths = [Path(p) for p in cfg["runs"]]
    entries = []
    for root in paths:
        for pattern in ["metrics.json", "metrics_full.json", "causal_summary.json", "best_probes_on_dev.json"]:
            path = root / pattern
            if path.is_file():
                entries.append({"path": str(path), "data": json.loads(path.read_text())})
    if not entries:
        raise ValueError("No result files in requested runs")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "report_data.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    lines = ["# CReST-FC experiment summary", "", "Generated from recorded run artifacts. Interpret only within each run's data protocol.", ""]
    for entry in entries:
        lines += [f'## {entry["path"]}', "", "```json", json.dumps(entry["data"], indent=2, ensure_ascii=False), "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf8")
    return {"n_artifacts": len(entries), "report": str(output / "report.md")}
