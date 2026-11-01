"""Single-model LoRA supervision baselines with matched dataset partitions."""
from __future__ import annotations
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import approved_pairs, read_jsonl, validate_dataset, write_jsonl
from .evaluation import classification_metrics, pair_metrics
from .modeling import add_lora, load_model, lora_state, render_verdict, score_candidate_sequences
from .training import _partition


def train_baseline(cfg):
    rows = read_jsonl(cfg["data"]["samples"])
    pairs = read_jsonl(cfg["data"]["pairs"]) if cfg["data"].get("pairs") else []
    validate_dataset(rows, pairs)
    fraction = cfg.get("decoder_fraction", 0.2)
    train = [r for r in rows if r["split"] == "train" and r["pair_kind"] == "none"
             and not _partition(r["group_id"], fraction)]
    if cfg.get("include_approved_pairs", False):
        reviewed = [r for a, b, _ in approved_pairs(rows, pairs, "train") for r in (a, b)
                    if not _partition(r["group_id"], fraction)]
        if not reviewed:
            raise ValueError("Counterfactual baseline requires approved train pairs")
        train += reviewed
    dev = [r for r in rows if r["split"] == "dev" and r["pair_kind"] == "none"]
    dev_pairs = approved_pairs(rows, pairs, "dev")
    if cfg.get("limit"):
        train, dev = train[:cfg["limit"]], dev[:min(cfg["limit"], 50)]
    if not train or not dev:
        raise ValueError("Need nonempty baseline train and dev")
    spec = cfg["model"]
    model, tok = load_model(spec["id"], spec.get("cache_dir"), spec.get("dtype", "bfloat16"),
                            spec.get("device", "cuda"))
    lc = cfg.get("lora", {})
    layers = cfg["layers"]
    params = add_lora(model, layers, rank=lc.get("rank", 8), alpha=lc.get("alpha", 16),
                      dropout=lc.get("dropout", 0.0))
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4))
    labels = cfg.get("labels", ["SUPPORTS", "REFUTES", "NEI"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    logs = []
    for epoch in range(cfg.get("epochs", 1)):
        random.Random(cfg.get("seed", 42) + epoch).shuffle(train)
        model.train()
        for i, row in enumerate(train):
            opt.zero_grad(set_to_none=True)
            scores = score_candidate_sequences(model, tok, render_verdict(row), labels)
            loss = F.cross_entropy(scores.float(), torch.tensor([labels.index(row["label"])], device=scores.device))
            loss.backward()
            opt.step()
            logs.append({"epoch": epoch, "step": i, "loss": float(loss.item())})
        model.eval()
        pred = {}
        with torch.no_grad():
            for row in dev + [r for a, b, _ in dev_pairs for r in (a, b)]:
                scores = score_candidate_sequences(model, tok, render_verdict(row), labels)
                pred[row["sample_id"]] = labels[scores.argmax(-1).item()]
        metric = classification_metrics(dev, pred, labels)
        metric.update(pair_metrics(rows, pairs, pred, "dev"))
        paired = [metric[k] for k in ["bias_pair_correct", "relation_pair_correct"] if metric[k] is not None]
        metric["selection_score"] = metric["macro_f1"] + (sum(paired) / len(paired) if paired else 0.0)
        if metric["selection_score"] > best:
            best = metric["selection_score"]
            torch.save(lora_state(model), output / "model_lora.pt")
            (output / "metadata.json").write_text(json.dumps({"model_id": spec["id"], "layers": layers,
                "lora": lc, "include_approved_pairs": cfg.get("include_approved_pairs", False),
                "train_n": len(train), "dev_n": len(dev), "dev_metrics": metric,
                "best_epoch": epoch, "trainable_parameters": sum(p.numel() for p in params)}, indent=2))
    write_jsonl(output / "train_log.jsonl", logs)
    return {"checkpoint": str(output), "train_n": len(train), "best_dev_selection_score": best}
