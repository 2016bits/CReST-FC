"""E3 decoder calibration and frozen-decoder encoder steering."""
from __future__ import annotations
import hashlib
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import approved_pairs, read_jsonl, validate_dataset, write_jsonl
from .evaluation import (classification_metrics, pair_metrics, clustered_bootstrap,
                         parse_structured_output, evidence_prf)
from .modeling import (LABEL_ORDER, LatentBridge, add_lora, get_transformer_blocks,
                       label_margin, load_lora_state, load_model, lora_state,
                       read_activation, render_encoder, token_positions)


def _json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


def _rows(cfg):
    rows = read_jsonl(cfg["data"]["samples"])
    pairs = read_jsonl(cfg["data"]["pairs"]) if cfg["data"].get("pairs") else []
    validate_dataset(rows, pairs)
    return rows, pairs


def _model(spec):
    return load_model(spec["id"], spec.get("cache_dir"), spec.get("dtype", "bfloat16"),
                      spec.get("device", "cuda"))


def _latent(encoder, tok, row, layer, detach=False):
    text, _ = render_encoder(row, "full_bias" if row.get("claim_metadata") else "full")
    idx = token_positions(tok, text, {"anchor": (len(text)-len("Relation anchor:"), len(text))})["final_anchor"]
    return read_activation(encoder, tok, text, layer, idx, detach=detach)


def _partition(group_id: str, fraction: float) -> bool:
    value = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) / 0xffffffff
    return value < fraction


def _score(bridge, latent, row, guided, labels):
    return bridge.score(latent, row if guided else None, guided=guided, candidates=labels)


def _ce(scores, label, labels):
    return F.cross_entropy(scores.float(), torch.tensor([labels.index(label)], device=scores.device))


def js_loss(scores_a, scores_b):
    a, b = scores_a.float().softmax(-1), scores_b.float().softmax(-1)
    m = 0.5 * (a + b)
    return 0.5 * (F.kl_div(m.clamp_min(1e-8).log(), a, reduction="batchmean") +
                  F.kl_div(m.clamp_min(1e-8).log(), b, reduction="batchmean"))


def flip_loss(scores_a, scores_b, ya, yb, labels, gamma=0.5):
    ia, ib = labels.index(ya), labels.index(yb)
    return (F.relu(gamma - label_margin(scores_a, ia) + label_margin(scores_b, ia)) +
            F.relu(gamma - label_margin(scores_b, ib) + label_margin(scores_a, ib))).mean()


def _save_calibration(out, decoder, bridge, cfg, epoch, metrics):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state(decoder), out / "decoder_lora.pt")
    torch.save({k: v.detach().cpu() for k, v in bridge.projector.state_dict().items()}, out / "projector.pt")
    _json(out / "metadata.json", {"encoder_id": cfg["encoder"]["id"],
                                  "decoder_id": cfg["decoder"]["id"],
                                  "encoder_hidden": bridge.projector.in_features,
                                  "decoder_hidden": bridge.projector.out_features,
                                  "read_layer": cfg["read_layer"], "write_layer": cfg.get("write_layer", 0),
                                  "decoder_lora_layers": cfg.get("decoder_lora_layers", [cfg.get("write_layer", 0)]),
                                  "lora": cfg.get("lora", {}), "labels": cfg.get("labels", list(LABEL_ORDER)),
                                  "guided_weight": cfg.get("guided_weight", 0.0),
                                  "structured_weight": cfg.get("structured_weight", 0.0),
                                  "decoder_trainable_parameters": sum(p.numel() for p in decoder.parameters() if p.requires_grad)
                                  + sum(p.numel() for p in bridge.projector.parameters() if p.requires_grad),
                                  "epoch": epoch, "dev_metrics": metrics})


def _load_calibration(path, decoder, tok):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    lora_cfg = meta["lora"]
    add_lora(decoder, meta["decoder_lora_layers"], rank=lora_cfg.get("rank", 8),
             alpha=lora_cfg.get("alpha", 16), dropout=lora_cfg.get("dropout", 0.0))
    load_lora_state(decoder, path / "decoder_lora.pt")
    bridge = LatentBridge(meta["encoder_hidden"], meta["decoder_hidden"],
                          decoder, tok, meta["write_layer"])
    bridge.projector.load_state_dict(torch.load(path / "projector.pt", map_location="cpu", weights_only=True))
    bridge.projector.requires_grad_(False)
    decoder.requires_grad_(False)
    decoder.eval()
    return bridge, meta


def calibrate(cfg):
    rows, _ = _rows(cfg)
    fraction = cfg.get("decoder_fraction", 0.2)
    train = [r for r in rows if r["split"] == "train" and r["pair_kind"] == "none"
             and _partition(r["group_id"], fraction)]
    dev = [r for r in rows if r["split"] == "dev" and r["pair_kind"] == "none"]
    if cfg.get("limit"):
        train, dev = train[:cfg["limit"]], dev[:min(cfg["limit"], 50)]
    if not train or not dev:
        raise ValueError("Need disjoint train_decoder and dev samples")
    if cfg.get("structured_weight", 0.0) and any(not r.get("gold_justification") for r in train + dev):
        raise ValueError("Structured calibration requires human-written gold_justification for every sample")
    encoder, etok = _model(cfg["encoder"])
    decoder, dtok = _model(cfg["decoder"])
    encoder.eval().requires_grad_(False)
    decoder.eval()
    lora_cfg = cfg.get("lora", {})
    decoder_params = add_lora(decoder, cfg.get("decoder_lora_layers", [cfg.get("write_layer", 0)]),
                              rank=lora_cfg.get("rank", 8), alpha=lora_cfg.get("alpha", 16),
                              dropout=lora_cfg.get("dropout", 0.0))
    bridge = LatentBridge(encoder.config.hidden_size, decoder.config.hidden_size,
                          decoder, dtok, cfg.get("write_layer", 0))
    params = decoder_params + list(bridge.projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4))
    labels = cfg.get("labels", list(LABEL_ORDER))
    out = Path(cfg["output_dir"])
    best = -1.0
    logs = []
    for epoch in range(cfg.get("epochs", 1)):
        random.Random(cfg.get("seed", 42) + epoch).shuffle(train)
        decoder.eval()  # frozen base has no dropout; adapter dropout remains disabled for stable bridge
        for i, row in enumerate(train):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                latent = _latent(encoder, etok, row, cfg["read_layer"], detach=False)
            scores = _score(bridge, latent, row, False, labels)
            loss = _ce(scores, row["label"], labels)
            if cfg.get("guided_weight", 0.0):
                guided = _score(bridge, latent, row, True, labels)
                loss = loss + cfg["guided_weight"] * _ce(guided, row["label"], labels)
            if cfg.get("structured_weight", 0.0):
                target = json.dumps({"verdict": row["label"],
                                     "evidence_ids": [e["evidence_id"] for e in row["evidence"] if e.get("is_gold")],
                                     "justification": row["gold_justification"]}, ensure_ascii=False)
                loss = loss + cfg["structured_weight"] * bridge.score_structured_target(latent, row, target)
            loss.backward()
            opt.step()
            logs.append({"epoch": epoch, "step": i, "loss": float(loss.item())})
        pred, null_pred = {}, {}
        with torch.no_grad():
            for row in dev:
                latent = _latent(encoder, etok, row, cfg["read_layer"], detach=False)
                scores = _score(bridge, latent, row, False, labels)
                null_scores = _score(bridge, torch.zeros_like(latent), row, False, labels)
                pred[row["sample_id"]] = labels[scores.argmax(-1).item()]
                null_pred[row["sample_id"]] = labels[null_scores.argmax(-1).item()]
        metrics = classification_metrics(dev, pred, labels)
        metrics["null_latent_macro_f1"] = classification_metrics(dev, null_pred, labels)["macro_f1"]
        metrics["null_latent_prediction_change_rate"] = sum(pred[k] != null_pred[k] for k in pred) / len(pred)
        metrics["selection_score"] = metrics["macro_f1"] + 0.1 * metrics["null_latent_prediction_change_rate"]
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            _save_calibration(out / "best", decoder, bridge, cfg, epoch, metrics)
    write_jsonl(out / "train_log.jsonl", logs)
    return {"train_n": len(train), "dev_n": len(dev), "best_dev_selection_score": best,
            "checkpoint": str(out / "best")}


def steer(cfg):
    rows, pairs = _rows(cfg)
    decoder_fraction = cfg.get("decoder_fraction", 0.2)
    train = [r for r in rows if r["split"] == "train" and r["pair_kind"] == "none"
             and not _partition(r["group_id"], decoder_fraction)]
    dev = [r for r in rows if r["split"] == "dev" and r["pair_kind"] == "none"]
    approved = [p for p in approved_pairs(rows, pairs, "train")
                if not _partition(p[0]["group_id"], decoder_fraction)]
    dev_approved = approved_pairs(rows, pairs, "dev")
    if cfg.get("limit"):
        train, dev, approved = train[:cfg["limit"]], dev[:min(cfg["limit"], 50)], approved[:cfg["limit"]]
    kinds = {p["kind"] for _, _, p in approved}
    if cfg.get("lambda_inv", 0) > 0 and "bias_change" not in kinds:
        raise ValueError("L_inv requires approved train bias_change pairs")
    if cfg.get("lambda_flip", 0) > 0 and "relation_change" not in kinds:
        raise ValueError("L_flip requires approved train relation_change pairs")
    dev_kinds = {p["kind"] for _, _, p in dev_approved}
    if cfg.get("require_dev_pairs", True):
        if cfg.get("lambda_inv", 0) > 0 and "bias_change" not in dev_kinds:
            raise ValueError("L_inv checkpoint selection requires approved dev bias_change pairs")
        if cfg.get("lambda_flip", 0) > 0 and "relation_change" not in dev_kinds:
            raise ValueError("L_flip checkpoint selection requires approved dev relation_change pairs")
    encoder, etok = _model(cfg["encoder"])
    decoder, dtok = _model(cfg["decoder"])
    bridge, meta = _load_calibration(cfg["calibration_checkpoint"], decoder, dtok)
    if meta["encoder_id"] != cfg["encoder"]["id"] or meta["decoder_id"] != cfg["decoder"]["id"]:
        raise ValueError("Calibration checkpoint model IDs differ from steering models")
    read_layer = meta["read_layer"]
    if cfg.get("require_latent_sensitivity", True) and meta["dev_metrics"]["null_latent_prediction_change_rate"] <= 0:
        raise ValueError("Calibrated decoder shows no dev prediction change with null latent")
    if cfg.get("selected_layers_path"):
        selection = json.loads(Path(cfg["selected_layers_path"]).read_text())
        if selection.get("selection_split") != "dev":
            raise ValueError("Layer selection must come from dev")
        selected = selection["selected_layers"]
    elif cfg.get("allow_manual_layers", False):
        selected = cfg["selected_layers"]
    else:
        raise ValueError("Use dev selected_layers_path; manual layers are only for smoke tests")
    if not selected:
        raise ValueError("No layer passed the dev causal control criterion")
    if any(layer > read_layer for layer in selected):
        raise ValueError("Encoder LoRA after read layer cannot receive bridge gradients")
    lora_cfg = cfg.get("lora", {})
    params = add_lora(encoder, selected, rank=lora_cfg.get("rank", 8),
                      alpha=lora_cfg.get("alpha", 16), dropout=lora_cfg.get("dropout", 0.0))
    encoder.train()
    decoder.eval()
    bridge.projector.eval()
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4))
    labels = meta["labels"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    logs = []
    tasks = [(r, None, None) for r in train] + [(a, b, p) for a, b, p in approved]
    if not tasks or not dev:
        raise ValueError("Need nonempty steer train and dev samples")
    for epoch in range(cfg.get("epochs", 1)):
        random.Random(cfg.get("seed", 42) + epoch).shuffle(tasks)
        for i, (a, b, pair) in enumerate(tasks):
            opt.zero_grad(set_to_none=True)
            za = _latent(encoder, etok, a, read_layer)
            if not za.requires_grad:
                raise RuntimeError("Encoder latent lost autograd graph")
            sa = _score(bridge, za, a, False, labels)
            loss = cfg.get("lambda_rel", 1.0) * _ce(sa, a["label"], labels)
            if b is not None:
                zb = _latent(encoder, etok, b, read_layer)
                sb = _score(bridge, zb, b, False, labels)
                loss = loss + cfg.get("lambda_rel", 1.0) * _ce(sb, b["label"], labels)
                if pair["kind"] == "bias_change":
                    loss = loss + cfg.get("lambda_inv", 0.2) * js_loss(sa, sb)
                else:
                    loss = loss + cfg.get("lambda_flip", 0.5) * flip_loss(
                        sa, sb, a["label"], b["label"], labels, cfg.get("margin_gamma", 0.5))
            loss.backward()
            grad_norm = float(sum(p.grad.abs().sum().item() for p in params if p.grad is not None))
            if grad_norm == 0 or not all(p.grad is None for p in decoder.parameters()):
                raise RuntimeError("Gradient bridge audit failed")
            opt.step()
            logs.append({"epoch": epoch, "step": i, "loss": float(loss.item()),
                         "encoder_lora_grad_l1": grad_norm, "pair_kind": pair["kind"] if pair else "none"})
        pred = {}
        encoder.eval()
        with torch.no_grad():
            dev_pair_rows = [r for a, b, _ in dev_approved for r in (a, b)]
            for row in dev + dev_pair_rows:
                latent = _latent(encoder, etok, row, read_layer)
                scores = _score(bridge, latent, row, False, labels)
                pred[row["sample_id"]] = labels[scores.argmax(-1).item()]
        encoder.train()
        metrics = classification_metrics(dev, pred, labels)
        metrics.update(pair_metrics(rows, pairs, pred, "dev"))
        paired = [metrics[k] for k in ["bias_pair_correct", "relation_pair_correct"] if metrics[k] is not None]
        metrics["selection_score"] = metrics["macro_f1"] + (sum(paired) / len(paired) if paired else 0.0)
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            torch.save(lora_state(encoder), out / "encoder_lora.pt")
            _json(out / "metadata.json", {"encoder_id": cfg["encoder"]["id"],
                                          "selected_layers": selected, "read_layer": read_layer,
                                          "lora": lora_cfg, "calibration_checkpoint": cfg["calibration_checkpoint"],
                                          "best_epoch": epoch, "dev_metrics": metrics,
                                          "n_approved_train_pairs": len(approved),
                                          "encoder_trainable_parameters": sum(p.numel() for p in params),
                                          "model_copies": 2,
                                          "optimizer_steps_per_epoch": len(tasks)})
    write_jsonl(out / "train_log.jsonl", logs)
    _json(out / "gradient_audit.json", {"encoder_lora_nonzero": True,
                                        "decoder_frozen": True, "projector_frozen": True})
    return {"train_n": len(train), "approved_pairs": len(approved),
            "best_dev_selection_score": best, "checkpoint": str(out)}


def evaluate(cfg):
    rows, pairs = _rows(cfg)
    split = cfg.get("split", "test")
    data = [r for r in rows if r["split"] == split and (r["pair_kind"] == "none" or
            r.get("counterfactual_verified", False))]
    if cfg.get("limit"):
        data = data[:cfg["limit"]]
    if not data:
        raise ValueError("No samples to evaluate")
    encoder, etok = _model(cfg["encoder"])
    decoder, dtok = _model(cfg["decoder"])
    bridge, meta = _load_calibration(cfg["calibration_checkpoint"], decoder, dtok)
    if meta["encoder_id"] != cfg["encoder"]["id"] or meta["decoder_id"] != cfg["decoder"]["id"]:
        raise ValueError("Calibration checkpoint model IDs differ from evaluation models")
    if cfg.get("steering_checkpoint"):
        sc = Path(cfg["steering_checkpoint"])
        smeta = json.loads((sc / "metadata.json").read_text())
        lc = smeta["lora"]
        add_lora(encoder, smeta["selected_layers"], rank=lc.get("rank", 8),
                 alpha=lc.get("alpha", 16), dropout=0)
        load_lora_state(encoder, sc / "encoder_lora.pt")
    encoder.eval().requires_grad_(False)
    labels = meta["labels"]
    guided = cfg.get("guided", False)
    if guided and not meta.get("guided_weight"):
        raise ValueError("Guided evaluation requires guided decoder calibration")
    structured = cfg.get("structured", False)
    if structured and not meta.get("structured_weight"):
        raise ValueError("Structured evaluation requires structured decoder calibration")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    latents = []
    with torch.inference_mode():
        for row in data:
            latents.append(_latent(encoder, etok, row, meta["read_layer"], detach=False))
    rng = random.Random(cfg.get("seed", 42))
    by_group = {}
    for j, row in enumerate(data):
        by_group.setdefault(row["group_id"], []).append(j)
    group_names = list(by_group)
    if len(group_names) < 2:
        raise ValueError("Shuffled-latent control requires at least two distinct groups")
    shuffled = []
    for row in data:
        group = rng.choice(group_names)
        while group == row["group_id"]:
            group = rng.choice(group_names)
        shuffled.append(rng.choice(by_group[group]))
    predictions = {mode: {} for mode in ["correct", "null", "shuffled"]}
    structured_predictions = {mode: {} for mode in predictions}
    evidence_scores = {mode: [] for mode in predictions}
    records = []
    with torch.inference_mode():
        for i, row in enumerate(data):
            for mode, latent in [("correct", latents[i]), ("null", torch.zeros_like(latents[i])),
                                 ("shuffled", latents[shuffled[i]])]:
                scores = _score(bridge, latent, row, guided, labels)
                pred = labels[scores.argmax(-1).item()]
                predictions[mode][row["sample_id"]] = pred
                record = {"sample_id": row["sample_id"], "mode": mode,
                                "gold": row["label"], "prediction": pred,
                                "latent_source_sample_id": data[shuffled[i]]["sample_id"] if mode == "shuffled" else
                                (row["sample_id"] if mode == "correct" else None),
                                "scores": scores.squeeze(0).tolist()}
                if structured:
                    raw = bridge.generate_structured(latent, row, cfg.get("max_new_tokens", 128))
                    parsed = parse_structured_output(raw, {e["evidence_id"] for e in row["evidence"]}, labels)
                    structured_predictions[mode][row["sample_id"]] = parsed["verdict"]
                    gold_ids = {e["evidence_id"] for e in row["evidence"] if e.get("is_gold")}
                    evidence_scores[mode].append(evidence_prf(gold_ids, set(parsed["evidence_ids"])))
                    record.update({"structured_raw": raw, "structured": parsed})
                records.append(record)
    out = Path(cfg["output_dir"])
    write_jsonl(out / "predictions.jsonl", records)
    report = {"stage": "E4", "split": split, "evidence_mode": "oracle",
              "inference_mode": "latent_guided_verdict" if guided else "latent_only_relation_readout",
              "n_examples": len(data), "model_id": cfg["encoder"]["id"],
              "model_copies": 2,
              "latency_ms_per_example_all_ablations": 1000 * (time.perf_counter() - started) / len(data),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "results": {mode: {**classification_metrics(data, pred, labels),
                                 **pair_metrics(rows, pairs, pred, split),
                                 "accuracy_group_bootstrap": clustered_bootstrap(
                                     data, {r["sample_id"]: float(pred[r["sample_id"]] == r["label"])
                                            for r in data}, cfg.get("bootstrap_reps", 1000), cfg.get("seed", 42))}
                          for mode, pred in predictions.items()}}
    if structured:
        report["structured_results"] = {
            mode: {**classification_metrics(data, pred, labels),
                   **pair_metrics(rows, pairs, pred, split),
                   "evidence_id_precision": sum(x["precision"] for x in evidence_scores[mode]) / len(data),
                   "evidence_id_recall": sum(x["recall"] for x in evidence_scores[mode]) / len(data),
                   "evidence_id_f1": sum(x["f1"] for x in evidence_scores[mode]) / len(data),
                   "justification_status": "requires_human_faithfulness_review"}
            for mode, pred in structured_predictions.items()}
    _json(out / "metrics.json", report)
    return {"output_dir": str(out), "n_examples": len(data),
            "correct_macro_f1": report["results"]["correct"]["macro_f1"]}
