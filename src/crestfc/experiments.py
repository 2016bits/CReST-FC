"""E0/E1/E2 implementations and paper-oriented output files."""
from __future__ import annotations
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from .data import adapt, approved_pairs, make_pairs, read_jsonl, validate_dataset, write_jsonl
from .evaluation import classification_metrics, pair_metrics
from .modeling import (LABEL_ORDER, PROMPT_VERSION, VERDICT_PROMPT_VERSIONS, verdict_verbalizers,
                       add_lora, load_lora_state, extract_all, get_transformer_blocks,
                       label_margin, load_model, make_patch_hook, read_activation,
                       render_encoder, render_verdict, score_candidate_sequences,
                       temporary_hook, token_positions)


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def _load_rows(cfg):
    rows = read_jsonl(cfg["data"]["samples"])
    pairs = read_jsonl(cfg["data"]["pairs"]) if cfg["data"].get("pairs") else []
    validate_dataset(rows, pairs)
    return rows, pairs


def _model(cfg):
    return load_model(cfg["model"]["id"], cfg["model"].get("cache_dir"),
                      cfg["model"].get("dtype", "bfloat16"), cfg["model"].get("device", "cuda"))


def _labels(cfg):
    return cfg.get("labels", list(LABEL_ORDER))


def prepare(cfg):
    source = cfg["source"]
    out = Path(cfg["output_dir"])
    examples, audit = adapt(source["path"], source["dataset"], source["split"],
                            source.get("limit"), source.get("wiki_db"))
    samples, pairs = make_pairs(examples, cfg.get("max_relation_pairs", 200),
                                cfg.get("min_evidence_similarity", 0.7), cfg.get("add_bias_pairs", True),
                                cfg.get("seed", 42))
    rows = [vars(x) for x in samples]
    validation = validate_dataset(rows, pairs)
    write_jsonl(out / "samples.jsonl", rows)
    write_jsonl(out / "pairs.jsonl", pairs)
    _dump(out / "audit.json", {**audit, **validation, "pair_status": "pending_human_review",
                               "source": source, "prompt_version": PROMPT_VERSION})
    return {**audit, **validation}


def validate_data(cfg):
    rows, pairs = _load_rows(cfg)
    return validate_dataset(rows, pairs, cfg.get("require_review", False))


def baseline(cfg):
    rows, pairs = _load_rows(cfg)
    split = cfg.get("split", "dev")
    rows = [r for r in rows if r["split"] == split and (r["pair_kind"] == "none" or
            (cfg.get("include_approved_pairs", False) and r.get("counterfactual_verified", False)))]
    if cfg.get("limit"):
        rows = rows[:cfg["limit"]]
    model, tok = _model(cfg)
    if cfg.get("lora_checkpoint"):
        ckpt = Path(cfg["lora_checkpoint"])
        meta = json.loads((ckpt / "metadata.json").read_text())
        lc = meta["lora"]
        add_lora(model, meta["layers"], rank=lc.get("rank", 8), alpha=lc.get("alpha", 16), dropout=0.0)
        load_lora_state(model, ckpt / "model_lora.pt")
    model.eval()
    labels = _labels(cfg)
    scheme = cfg.get("verdict_scheme", "fever3")
    verbalizers = verdict_verbalizers(labels, scheme)
    out = Path(cfg["output_dir"])
    results = []
    for condition in cfg.get("conditions", ["full", "claim_only", "evidence_only", "full_bias"]):
        pred = {}
        for row in rows:
            with torch.inference_mode():
                scores = score_candidate_sequences(model, tok, render_verdict(row, condition, scheme), verbalizers)
            pred[row["sample_id"]] = labels[scores.argmax(-1).item()]
            results.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                            "condition": condition, "gold": row["label"],
                            "prediction": pred[row["sample_id"]],
                            "prediction_surface": verbalizers[scores.argmax(-1).item()],
                            "scores": scores.squeeze(0).tolist()})
        modes = sorted({r["evidence_mode"] for r in rows})
        _dump(out / f"metrics_{condition}.json", {"stage": "E0", "condition": condition,
                                                    "verdict_prompt_version": VERDICT_PROMPT_VERSIONS[scheme],
                                                    "verdict_scheme": scheme,
                                                    "verbalizers": dict(zip(labels, verbalizers)),
                                                    "evidence_mode": modes[0] if len(modes) == 1 else modes,
                                                    "labels": labels,
                                                    **classification_metrics(rows, pred, labels),
                                                    **pair_metrics(rows, pairs, pred, split)})
    write_jsonl(out / "predictions.jsonl", results)
    return {"n_predictions": len(results), "output_dir": str(out)}


def extract(cfg):
    rows, _ = _load_rows(cfg)
    split = cfg.get("split", "train")
    rows = [r for r in rows if r["split"] == split and (r["pair_kind"] == "none" or
            (cfg.get("include_approved_pairs", True) and r.get("counterfactual_verified", False)))]
    if cfg.get("pair_kind_filter"):
        rows = [r for r in rows if r["pair_kind"] == cfg["pair_kind_filter"]]
    if cfg.get("limit"):
        rows = rows[:cfg["limit"]]
    model, tok = _model(cfg)
    model.eval()
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"{split}_*_layer*.npy"):
        stale.unlink()
    labels = []
    positions = cfg.get("positions", ["final_anchor", "claim_end", "evidence_end"])
    for i, row in enumerate(rows):
        prompt, char_spans = render_encoder(row)
        if "edited_evidence_span_mean" in positions:
            edit = row.get("edited_span")
            if not edit or "evidence" not in char_spans:
                raise ValueError("Edited-span extraction requires relation pair samples with aligned evidence")
            side = edit["side"].lower()
            relevant = [(x[f"{side}_start"], x[f"{side}_end"]) for x in edit["edits"]]
            if not relevant:
                raise ValueError("No edited characters in relation pair")
            offset = char_spans["evidence"][0]
            char_spans["edited_evidence_span_mean"] = (offset + min(a for a, _ in relevant),
                                                        offset + max(b for _, b in relevant))
        tp = token_positions(tok, prompt, char_spans)
        selected = {name: tp[name.replace("_end", "")][-1:] if name.endswith("_end") else tp[name]
                    for name in positions}
        data = extract_all(model, tok, prompt, selected, cfg.get("layers"))
        for name, layer_values in data.items():
            for layer, value in layer_values.items():
                p = out / f"{split}_{name}_layer{layer}.npy"
                # Chunked per-layer files avoid retaining all layers in GPU memory.
                with p.open("ab") as f:
                    np.save(f, value.numpy().astype(np.float32), allow_pickle=False)
        labels.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                       "label": row["label"], "split": split})
    write_jsonl(out / f"{split}_index.jsonl", labels)
    _dump(out / f"{split}_metadata.json", {"model_id": cfg["model"]["id"],
                                           "prompt_version": PROMPT_VERSION,
                                           "positions": positions, "n": len(rows)})
    return {"n": len(rows), "output_dir": str(out)}


def _read_npy_stream(path):
    rows = []
    with open(path, "rb") as f:
        while True:
            try:
                rows.append(np.load(f, allow_pickle=False)[0])
            except (EOFError, ValueError):
                break
    return np.stack(rows)


def train_probes(cfg):
    root = Path(cfg["activation_dir"])
    train = read_jsonl(root / "train_index.jsonl")
    dev = read_jsonl(root / "dev_index.jsonl")
    if not train or not dev:
        raise ValueError("Need nonempty train and dev activations")
    labels = _labels(cfg)
    y_train = np.array([labels.index(x["label"]) for x in train])
    y_dev = np.array([labels.index(x["label"]) for x in dev])
    if len(np.unique(y_train)) < 2:
        raise ValueError("Probe training needs at least two classes")
    pair_data = _load_rows(cfg) if cfg.get("data", {}).get("pairs") else None
    result = []
    for p in sorted(root.glob("train_*_layer*.npy")):
        suffix = p.name[len("train_"):]
        dev_path = root / ("dev_" + suffix)
        if not dev_path.exists():
            continue
        X, V = _read_npy_stream(p), _read_npy_stream(dev_path)
        if len(X) != len(train) or len(V) != len(dev):
            raise ValueError(f"Activation/index count mismatch for {suffix}")
        for c in cfg.get("C", [0.01, 0.1, 1.0]):
            probe = LogisticRegression(C=c, max_iter=1000, random_state=cfg.get("seed", 42))
            probe.fit(X, y_train)
            pred = probe.predict(V)
            pair_stats = {}
            if pair_data:
                all_rows, all_pairs = pair_data
                indexed_pred = {row["sample_id"]: labels[int(y)] for row, y in zip(dev, pred)}
                pair_stats = pair_metrics(all_rows, all_pairs, indexed_pred, "dev")
            result.append({"position_layer": suffix, "C": c,
                           "accuracy": accuracy_score(y_dev, pred),
                           "macro_f1": f1_score(y_dev, pred, labels=list(range(len(labels))),
                                                average="macro", zero_division=0),
                           "balanced_accuracy": balanced_accuracy_score(y_dev, pred),
                           **pair_stats})
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "layer_probe_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]) if result else ["position_layer", "C", "accuracy", "macro_f1", "balanced_accuracy"])
        writer.writeheader()
        writer.writerows(result)
    best = {}
    for row in result:
        key = row["position_layer"]
        if key not in best or row["macro_f1"] > best[key]["macro_f1"]:
            best[key] = row
    _dump(out / "best_probes_on_dev.json", best)
    _dump(out / "probe_metadata.json", {"train_n": len(train), "dev_n": len(dev),
                                         "selection_split": "dev", "labels": labels})
    return {"n_probe_fits": len(result), "output_dir": str(out)}


def causal_patch(cfg):
    rows, pairs = _load_rows(cfg)
    selected = approved_pairs(rows, pairs, "dev")
    if not selected:
        raise ValueError("No approved dev pairs; review pairs before causal claims")
    if cfg.get("limit"):
        selected = selected[:cfg["limit"]]
    model, tok = _model(cfg)
    model.eval()
    labels = _labels(cfg)
    layers = cfg.get("layers", list(range(len(get_transformer_blocks(model)))))
    controls = [r for r in rows if r["split"] == "dev" and r["pair_kind"] == "none"]
    results = []
    with torch.inference_mode():
        for a, b, pair in selected:
            for source, target, direction in [(a, b, "A_to_B"), (b, a, "B_to_A")]:
                source_text, _ = render_encoder(source, "full_bias" if source.get("claim_metadata") else "full")
                target_text, _ = render_encoder(target, "full_bias" if target.get("claim_metadata") else "full")
                source_prompt = source_text + "\nVerdict:"
                target_prompt = target_text + "\nVerdict:"
                source_idx = token_positions(tok, source_prompt, {"anchor": (len(source_text)-len("Relation anchor:"), len(source_text))})["anchor"][-1]
                target_idx = token_positions(tok, target_prompt, {"anchor": (len(target_text)-len("Relation anchor:"), len(target_text))})["anchor"][-1]
                base = score_candidate_sequences(model, tok, target_prompt, labels)
                random_row = next((r for r in controls if r["group_id"] != target["group_id"]), None)
                random_text = None
                random_idx = None
                if random_row:
                    random_text, _ = render_encoder(random_row)
                    random_prompt = random_text + "\nVerdict:"
                    random_idx = token_positions(tok, random_prompt, {"anchor": (len(random_text)-len("Relation anchor:"), len(random_text))})["anchor"][-1]
                for layer in layers:
                    source_z = read_activation(model, tok, source_prompt, layer, [source_idx])
                    target_z = read_activation(model, tok, target_prompt, layer, [target_idx])
                    patches = [("source", source_z), ("same_input", target_z)]
                    if random_row:
                        patches.append(("random_sample", read_activation(model, tok, random_prompt, layer, [random_idx])))
                    for control, z in patches:
                        with temporary_hook(get_transformer_blocks(model)[layer], make_patch_hook(target_idx, z.to(model.device))):
                            patched = score_candidate_sequences(model, tok, target_prompt, labels)
                        index = labels.index(source["label"])
                        effect = (label_margin(patched, index) - label_margin(base, index)).item()
                        results.append({"pair_id": pair["pair_id"], "kind": pair["kind"],
                                        "direction": direction, "layer": layer, "control": control,
                                        "source_label": source["label"], "target_label": target["label"],
                                        "source_token": source_idx, "target_token": target_idx,
                                        "same_token_index": source_idx == target_idx,
                                        "effect": effect,
                                        "patched_prediction": labels[patched.argmax(-1).item()],
                                        "base_prediction": labels[base.argmax(-1).item()]})
    out = Path(cfg["output_dir"])
    write_jsonl(out / "causal_effects.jsonl", results)
    same_effects = [abs(x["effect"]) for x in results if x["control"] == "same_input"]
    max_same = max(same_effects, default=0.0)
    if max_same > cfg.get("same_input_tolerance", 0.05):
        raise RuntimeError(f"Same-input patch failed alignment check: max |IE|={max_same:.4f}")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in results:
        grouped[(row["kind"], row["layer"], row["control"])][row["pair_id"]].append(row["effect"])
    rng = np.random.default_rng(cfg.get("seed", 42))
    summary = []
    for (kind, layer, control), by_pair in grouped.items():
        values = [np.asarray(v) for v in by_pair.values()]
        boots = []
        for _ in range(cfg.get("bootstrap_reps", 1000)):
            picked = rng.integers(0, len(values), len(values))
            boots.append(float(np.mean(np.concatenate([values[i] for i in picked]))))
        flat = np.concatenate(values)
        summary.append({"kind": kind, "layer": layer, "control": control,
                        "n_pairs": len(values), "mean_ie": float(flat.mean()),
                        "median_ie": float(np.median(flat)),
                        "positive_fraction": float(np.mean(flat > 0)),
                        "group_bootstrap_ci95": [float(x) for x in np.quantile(boots, [0.025, 0.975])]})
    _dump(out / "causal_summary.json", {"max_abs_same_input_ie": max_same, "results": summary})
    _dump(out / "causal_metadata.json", {"n_approved_pairs": len(selected), "selection_split": "dev",
                                          "position": "final_anchor", "labels": labels,
                                          "controls": ["source", "same_input", "random_sample"]})
    return {"n_effects": len(results), "output_dir": str(out)}


def select_layers(cfg):
    rows = read_jsonl(Path(cfg["run_dir"]) / "causal_effects.jsonl")
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["kind"] == "relation_change":
            key = (row["pair_id"], row["direction"])
            grouped[row["layer"]][key][row["control"]] = row["effect"]
    if not grouped:
        raise ValueError("No reviewed relation-change causal effects")
    ranked = []
    rng = np.random.default_rng(cfg.get("seed", 42))
    for layer, pairs in grouped.items():
        by_pair = defaultdict(list)
        source, random_abs, same_abs = [], [], []
        for (pair_id, _), controls in pairs.items():
            if "source" not in controls or "random_sample" not in controls:
                continue
            effect = controls["source"]
            control = abs(controls["random_sample"])
            by_pair[pair_id].append(effect - control)
            source.append(effect)
            random_abs.append(control)
            same_abs.append(abs(controls.get("same_input", 0.0)))
        if not by_pair:
            continue
        cluster_values = [np.asarray(v) for v in by_pair.values()]
        boots = []
        for _ in range(cfg.get("bootstrap_reps", 1000)):
            picked = rng.integers(0, len(cluster_values), len(cluster_values))
            boots.append(float(np.mean(np.concatenate([cluster_values[i] for i in picked]))))
        adjusted = np.concatenate(cluster_values)
        ci = [float(x) for x in np.quantile(boots, [0.025, 0.975])]
        ranked.append({"layer": layer, "mean_ie_rel": float(np.mean(source)),
                       "mean_abs_random_ie": float(np.mean(random_abs)),
                       "mean_abs_same_input_ie": float(np.mean(same_abs)),
                       "control_adjusted_score": float(adjusted.mean()),
                       "control_adjusted_ci95": ci,
                       "n_pairs": len(by_pair), "positive_fraction": float(np.mean(np.asarray(source) > 0))})
    ranked.sort(key=lambda x: x["control_adjusted_score"], reverse=True)
    min_pairs = cfg.get("min_pairs", 20)
    qualified = [x for x in ranked if x["n_pairs"] >= min_pairs and x["control_adjusted_ci95"][0] > 0]
    result = {"selected_layers": [x["layer"] for x in qualified[:cfg.get("top_k", 2)]],
              "selection_split": "dev", "rule": "top_k_paired_control_adjusted_IE_with_group_CI_lower_gt_zero",
              "min_pairs": min_pairs, "ranked": ranked}
    _dump(Path(cfg["run_dir"]) / "candidate_layers_final.json", result)
    return result
