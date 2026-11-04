"""Frozen 7B FEVER binary layer probes with input ablations and shard-safe output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .data import adapt, validate_dataset, write_jsonl, read_jsonl, stable_id
from .modeling import get_transformer_blocks, last_nonpadding, load_model, render_encoder

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("full", "claim_only", "evidence_only")
LABELS = ("SUPPORTS", "REFUTES")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_gpu_tokens(logical_gpus: list[int], visible_spec: str | None,
                       device_count: int) -> list[str]:
    """Map CUDA logical indexes to the device tokens visible to this process."""
    visible = ([token.strip() for token in visible_spec.split(",") if token.strip()]
               if visible_spec else [str(index) for index in range(device_count)])
    if (not logical_gpus or len(logical_gpus) != len(set(logical_gpus))
            or min(logical_gpus) < 0 or max(logical_gpus) >= min(len(visible), device_count)):
        raise ValueError(f"Invalid logical GPU list {logical_gpus}; visible devices are {visible}")
    return [visible[index] for index in logical_gpus]


def shard_is_complete(run: Path, shard: int) -> bool:
    out = run / "activations"
    marker = out / f"shard{shard}_complete.json"
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    files = value.get("files", [])
    return bool(files) and all((out / name).is_file() for name in files)


def partition_count(manifest: dict, run: Path) -> int:
    """Preserve the partition count recorded by an already-started run."""
    if "n_shards" in manifest:
        return int(manifest["n_shards"])
    markers = sorted((run / "activations").glob("shard*_complete.json"))
    if markers:
        counts = {int(json.loads(path.read_text())["n_shards"]) for path in markers}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent shard partition counts: {counts}")
        return counts.pop()
    return len(manifest["gpus"].split(","))


def snapshot(cache_dir: str, model_id: str) -> Path:
    root = Path(cache_dir) / ("models--" + model_id.replace("/", "--")) / "snapshots"
    matches = sorted(p for p in root.glob("*") if (p / "config.json").is_file()
                     and (p / "model.safetensors.index.json").is_file()
                     and (p / "tokenizer.json").is_file())
    if not matches:
        raise FileNotFoundError(f"Complete local {model_id} snapshot not found in {cache_dir}")
    selected = matches[-1]
    index = json.loads((selected / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values()) if not (selected / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing weight files: {missing}")
    return selected


def selected_rows(source: Path, split: str, per_label: int, seed: int) -> tuple[list[dict], dict]:
    examples, source_audit = adapt(source, "FEVER", split)
    rows = [vars(item) for item in examples if item.label in LABELS and item.pair_kind == "none"]
    by_label = {label: [row for row in rows if row["label"] == label] for label in LABELS}
    rng = random.Random(seed)
    chosen = []
    for label in LABELS:
        if not by_label[label]:
            raise ValueError(f"No {label} examples in {source}")
        rng.shuffle(by_label[label])
        chosen.extend(by_label[label][:per_label] if per_label else by_label[label])
    rng.shuffle(chosen)
    return chosen, source_audit


def prepare(args, run: Path) -> dict:
    import sklearn
    import transformers

    train_source = ROOT / "data/FEVER/converted_data/train.json"
    dev_source = ROOT / "data/FEVER/converted_data/dev.json"
    train, train_audit = selected_rows(train_source, "train", args.train_per_label, args.seed)
    dev, dev_audit = selected_rows(dev_source, "dev", args.dev_per_label, args.seed)
    # Claims repeated across source splits are excluded from probe fitting.
    dev_claims = {stable_id(row["claim"].strip().lower()) for row in dev}
    before = len(train)
    train = [row for row in train if stable_id(row["claim"].strip().lower()) not in dev_claims]
    if len({row["label"] for row in train}) < 2:
        raise ValueError("Train split has fewer than two labels after leakage removal")
    validation = validate_dataset(train + dev, [])
    if validation["claim_hash_cross_split"] or validation["revision_family_cross_split"]:
        raise ValueError(f"Cross-split leakage: {validation}")
    data_dir = run / "data"
    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "dev.jsonl", dev)
    manifest = {"model_id": args.model_id, "snapshot": str(snapshot(args.cache_dir, args.model_id)),
                "cache_dir": args.cache_dir, "seed": args.seed, "gpus": args.gpus,
                "n_shards": len(args.gpus.split(",")),
                "cuda_visible_devices_at_launch": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "shard_device_tokens": getattr(args, "shard_device_tokens", None),
                "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                             "transformers": transformers.__version__, "sklearn": sklearn.__version__},
                "code_sha256": {name: sha256(Path(__file__).with_name(name))
                                for name in ("fever_probe.py", "modeling.py", "data.py")},
                "train_per_label": args.train_per_label, "dev_per_label": args.dev_per_label,
                "train_source": str(train_source.resolve()), "dev_source": str(dev_source.resolve()),
                "train_source_sha256": train[0]["source_provenance"]["source_sha256"],
                "dev_source_sha256": dev[0]["source_provenance"]["source_sha256"],
                "train_source_audit": train_audit, "dev_source_audit": dev_audit,
                "train_n": len(train), "dev_n": len(dev),
                "train_labels": dict(Counter(row["label"] for row in train)),
                "dev_labels": dict(Counter(row["label"] for row in dev)),
                "dropped_train_claim_overlap": before - len(train),
                "validation": validation, "conditions": CONDITIONS,
                "read_position": "final_anchor", "prompt_version": "v1_anchor_last_plain",
                "evidence_mode": "retrieved", "label_space": LABELS}
    dump(run / "manifest.json", manifest)
    return manifest


def shard_rows(run: Path, split: str, shard: int, n_shards: int) -> list[dict]:
    return [row for row in read_jsonl(run / "data" / f"{split}.jsonl")
            if int(stable_id(row["sample_id"]), 16) % n_shards == shard]


def extract_shard(args, run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    n_shards = partition_count(manifest, run)
    shard = args.shard
    out = run / "activations"
    complete = out / f"shard{shard}_complete.json"
    if complete.is_file():
        value = json.loads(complete.read_text())
        if all((out / name).is_file() for name in value["files"]):
            print(f"Shard {shard} already complete", flush=True)
            return
    model, tokenizer = load_model(manifest["model_id"], manifest["cache_dir"], "bfloat16", "cuda")
    model.eval()
    blocks = get_transformer_blocks(model)
    n_layers, hidden = len(blocks), model.config.hidden_size
    max_positions = model.config.max_position_embeddings
    out.mkdir(parents=True, exist_ok=True)
    files = []
    max_seen = 0
    for split in ("train", "dev"):
        rows = shard_rows(run, split, shard, n_shards)
        if not rows:
            raise ValueError(f"Empty {split} shard {shard}; use more examples or fewer GPUs")
        index = [{"sample_id": row["sample_id"], "group_id": row["group_id"],
                  "label": row["label"], "split": split} for row in rows]
        write_jsonl(out / f"{split}_shard{shard}_index.jsonl", index)
        files.append(f"{split}_shard{shard}_index.jsonl")
        for condition in CONDITIONS:
            final_name = f"{split}_{condition}_shard{shard}.npy"
            partial = out / (final_name + ".partial")
            if partial.exists():
                partial.unlink()
            array = np.lib.format.open_memmap(partial, mode="w+", dtype="float32",
                                              shape=(len(rows), n_layers, hidden))
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                prompts = [render_encoder(row, condition)[0] for row in batch]
                encoded = tokenizer(prompts, add_special_tokens=False, padding=True,
                                    truncation=False, return_tensors="pt")
                counts = encoded["attention_mask"].sum(dim=1)
                max_seen = max(max_seen, int(counts.max().item()))
                if max_seen > max_positions:
                    raise ValueError(f"Prompt exceeds model context ({max_seen}>{max_positions})")
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                positions = last_nonpadding(encoded["attention_mask"])
                storage = [None] * n_layers
                handles = []
                for layer, block in enumerate(blocks):
                    def capture(_module, _inputs, output, layer=layer):
                        h = output[0] if isinstance(output, tuple) else output
                        storage[layer] = h[torch.arange(h.shape[0], device=h.device), positions].float()
                    handles.append(block.register_forward_hook(capture))
                try:
                    with torch.inference_mode():
                        # The base transformer exposes the same block outputs and
                        # avoids allocating full-vocabulary LM logits per token.
                        model.model(**encoded, use_cache=False, return_dict=True)
                        batch_acts = torch.stack(storage, dim=1).cpu().numpy()
                finally:
                    for handle in handles:
                        handle.remove()
                array[start:start + len(batch)] = batch_acts
                if (start // args.batch_size) % 100 == 0:
                    print(f"shard={shard} {split}/{condition} {start + len(batch)}/{len(rows)}", flush=True)
            array.flush()
            del array
            partial.replace(out / final_name)
            files.append(final_name)
    dump(complete, {"shard": shard, "n_shards": n_shards, "n_layers": n_layers,
                    "hidden_size": hidden, "max_prompt_tokens": max_seen, "files": files})


def load_layer(run: Path, split: str, condition: str, layer: int, n_shards: int):
    chunks, index = [], []
    for shard in range(n_shards):
        root = run / "activations"
        rows = read_jsonl(root / f"{split}_shard{shard}_index.jsonl")
        array = np.load(root / f"{split}_{condition}_shard{shard}.npy", mmap_mode="r")
        if len(rows) != array.shape[0]:
            raise ValueError(f"Activation/index mismatch: {split}/{condition}/shard{shard}")
        chunks.append(np.asarray(array[:, layer, :], dtype=np.float32))
        index.extend(rows)
    return np.concatenate(chunks), index


def fit_probe(x_train, y_train, x_dev, y_dev, seed: int, device: str = "cpu",
              return_parameters: bool = False):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)
    if device == "cpu":
        probe = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                                   max_iter=1000, random_state=seed)
        probe.fit(x_train, y_train)
        pred = probe.predict(x_dev)
        converged = bool(max(probe.n_iter_) < probe.max_iter)
        if return_parameters:
            parameters = {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy(),
                          "coef": probe.coef_[0].copy(),
                          "intercept": np.asarray(probe.intercept_[0]).copy()}
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for probe fitting on {device}")
        x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        counts = np.bincount(y_train, minlength=2)
        weights = torch.as_tensor(len(y_train) / (2 * counts[y_train]),
                                  dtype=torch.float32, device=device)
        weight_sum = weights.sum()
        coef = torch.nn.Parameter(torch.zeros(x.shape[1], device=device))
        bias = torch.nn.Parameter(torch.zeros((), device=device))
        optimizer = torch.optim.LBFGS([coef, bias], lr=1.0, max_iter=100,
                                       tolerance_grad=1e-6, tolerance_change=1e-9,
                                       history_size=10, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = x.mv(coef) + bias
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y,
                                                                           reduction="none")
            objective = (losses * weights).sum() / weight_sum
            objective = objective + coef.square().sum() * (0.5 / (0.1 * weight_sum))
            objective.backward()
            return objective

        optimizer.step(closure)
        converged = optimizer.state[coef]["n_iter"] < 100
        if return_parameters:
            parameters = {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy(),
                          "coef": coef.detach().cpu().numpy().copy(),
                          "intercept": bias.detach().cpu().numpy().copy()}
        with torch.inference_mode():
            dev_x = torch.as_tensor(x_dev, dtype=torch.float32, device=device)
            pred = ((dev_x.mv(coef) + bias) >= 0).to(torch.int64).cpu().numpy()
        del x, y, weights, dev_x, coef, bias, optimizer
    metrics = {"accuracy": float(accuracy_score(y_dev, pred)),
               "macro_f1": float(f1_score(y_dev, pred, labels=[0, 1], average="macro")),
               "balanced_accuracy": float(balanced_accuracy_score(y_dev, pred)),
               "converged": converged}
    return (pred, metrics, parameters) if return_parameters else (pred, metrics)


def save_probe_parameters(run: Path, layer: int, condition: str,
                          parameters: dict, extra_metadata: dict | None = None) -> Path:
    """Save a frozen probe in the format used by external pair evaluation."""
    manifest = json.loads((run / "manifest.json").read_text())
    out = run / "analysis" / "probes"
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"model_id": manifest["model_id"], "layer": layer, "condition": condition,
                "read_position": manifest["read_position"], "label_order": list(LABELS),
                "e1_run_id": run.name, "e1_manifest_sha256": sha256(run / "manifest.json"),
                "train_source_sha256": manifest["train_source_sha256"],
                "dev_source_sha256": manifest["dev_source_sha256"]}
    metadata.update(extra_metadata or {})
    final = out / f"layer{layer:02d}_{condition}.npz"
    temporary = out / f"layer{layer:02d}_{condition}.partial.npz"
    np.savez(temporary, mean=parameters["mean"], scale=parameters["scale"],
             coef=parameters["coef"], intercept=parameters["intercept"],
             metadata=json.dumps(metadata, ensure_ascii=False))
    temporary.replace(final)
    return final


def export_e1_probes(args, run: Path) -> dict:
    """Refit E1 probes on cached activations and verify historical dev predictions.

    Historical checkpoints contain predictions but no fitted parameters. Exact
    agreement with those checkpoints is required before a reconstructed probe
    can be transferred to an external dataset.
    """
    manifest = json.loads((run / "manifest.json").read_text())
    n_shards = partition_count(manifest, run)
    metadata = [json.loads((run / "activations" / f"shard{i}_complete.json").read_text())
                for i in range(n_shards)]
    n_layers = metadata[0]["n_layers"]
    if any(item["n_layers"] != n_layers or item["n_shards"] != n_shards for item in metadata):
        raise ValueError("E1 activation shards have inconsistent dimensions or partitioning")
    method = "torch_lbfgs_c0.1_v1" if args.probe_device != "cpu" else "sklearn_liblinear_c0.1_v1"
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if len(layers) != len(set(layers)) or any(x < 0 or x >= n_layers for x in layers):
        raise ValueError(f"Invalid layer list: {args.layers}")
    conditions = [item.strip() for item in args.conditions.split(",")]
    if not conditions or len(conditions) != len(set(conditions)) or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"Invalid condition list: {args.conditions}")
    probe_dir = run / "analysis" / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for layer in layers:
        checkpoint = run / "analysis" / "checkpoints" / f"layer{layer:02d}.npz"
        with np.load(checkpoint, allow_pickle=False) as saved:
            info = json.loads(str(saved["metadata"].item()))
            if info["config"]["method"] != method:
                raise ValueError(f"Original layer {layer} used {info['config']['method']}; requested {method}")
            original_predictions = {condition: saved[condition].copy() for condition in conditions}
        checkpoint_digest = sha256(checkpoint)
        for condition in conditions:
            original_pred = original_predictions[condition]
            target = probe_dir / f"layer{layer:02d}_{condition}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    old = json.loads(str(saved["metadata"].item()))
                if (old.get("e1_manifest_sha256") != sha256(run / "manifest.json")
                        or old.get("original_checkpoint_sha256") != checkpoint_digest
                        or old.get("probe_method") != method
                        or old.get("reproduced_dev_predictions") is not True):
                    raise ValueError(f"Existing Probe bundle has incompatible provenance: {target}")
                print(f"layer={layer} {condition} verified Probe reused", flush=True)
            else:
                x_train, train_index = load_layer(run, "train", condition, layer, n_shards)
                x_dev, dev_index = load_layer(run, "dev", condition, layer, n_shards)
                y_train = np.array([LABELS.index(row["label"]) for row in train_index])
                y_dev = np.array([LABELS.index(row["label"]) for row in dev_index])
                if len(x_train) != manifest["train_n"] or len(x_dev) != manifest["dev_n"]:
                    raise ValueError(f"Layer {layer}/{condition} row count differs from E1 manifest")
                predicted, metrics, parameters = fit_probe(x_train, y_train, x_dev, y_dev,
                                                           manifest["seed"], args.probe_device,
                                                           return_parameters=True)
                mismatches = int(np.count_nonzero(predicted != original_pred))
                if mismatches:
                    raise RuntimeError(f"Layer {layer}/{condition} refit disagrees with original "
                                       f"E1 dev predictions on {mismatches}/{len(original_pred)} examples")
                save_probe_parameters(run, layer, condition, parameters,
                                      {"probe_method": method, "converged": metrics["converged"],
                                       "fitted_from_cached_e1_activations": True,
                                       "reproduced_dev_predictions": True,
                                       "original_checkpoint_sha256": checkpoint_digest})
                print(f"layer={layer} {condition} Probe exported; "
                      f"dev predictions {len(original_pred)}/{len(original_pred)} match", flush=True)
                del x_train, x_dev
            records.append({"layer": layer, "condition": condition, "path": str(target),
                            "sha256": sha256(target), "original_checkpoint_sha256": checkpoint_digest})
            if len(layers) == n_layers and set(conditions) == set(CONDITIONS):
                dump(probe_dir / "export_progress.json", {"method": method, "completed": records})
    if len(layers) == n_layers and set(conditions) == set(CONDITIONS):
        dump(probe_dir / "export_manifest.json", {
            "e1_run": str(run), "e1_manifest_sha256": sha256(run / "manifest.json"),
            "method": method, "layers": list(range(n_layers)), "conditions": list(CONDITIONS),
            "bundles": records,
            "note": "Reconstructed by refitting on cached E1 activations; all dev predictions match historical checkpoints."})
    return {"exported_or_verified_bundles": len(records), "probe_dir": str(probe_dir)}


def analyze(args, run: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((run / "manifest.json").read_text())
    n_shards = partition_count(manifest, run)
    metadata = [json.loads((run / "activations" / f"shard{i}_complete.json").read_text())
                for i in range(n_shards)]
    n_layers = metadata[0]["n_layers"]
    if any(item["n_layers"] != n_layers or item["hidden_size"] != metadata[0]["hidden_size"]
           for item in metadata):
        raise ValueError("Shard model dimensions differ")
    if any(item["n_shards"] != n_shards for item in metadata):
        raise ValueError("Shard partition counts differ")
    for split in ("train", "dev"):
        expected = {row["sample_id"] for row in read_jsonl(run / "data" / f"{split}.jsonl")}
        observed = [row["sample_id"] for shard in range(n_shards)
                    for row in read_jsonl(run / "activations" / f"{split}_shard{shard}_index.jsonl")]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError(f"Incomplete or duplicated {split} activation partition")
    results = []
    shuffled = []
    shuffled_predictions = {}
    predictions = {condition: {} for condition in CONDITIONS}
    dev_index_all = [row for shard in range(n_shards)
                     for row in read_jsonl(run / "activations" / f"dev_shard{shard}_index.jsonl")]
    dev_groups = [row["group_id"] for row in dev_index_all]
    dev_truth = np.array([LABELS.index(row["label"]) for row in dev_index_all])
    checkpoint_dir = run / "analysis" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_config = {"method": "torch_lbfgs_c0.1_v1" if args.probe_device != "cpu"
                         else "sklearn_liblinear_c0.1_v1", "seed": manifest["seed"],
                         "n_shards": n_shards, "train_n": manifest["train_n"],
                         "dev_n": manifest["dev_n"]}
    for layer in range(n_layers):
        checkpoint = checkpoint_dir / f"layer{layer:02d}.npz"
        if checkpoint.is_file():
            with np.load(checkpoint, allow_pickle=False) as saved:
                info = json.loads(str(saved["metadata"].item()))
                if info["config"] != checkpoint_config:
                    raise ValueError(f"Probe checkpoint configuration mismatch: {checkpoint}")
                results.extend(info["results"])
                shuffled.append(info["shuffled"])
                for condition in CONDITIONS:
                    predictions[condition][layer] = saved[condition].copy()
                shuffled_predictions[layer] = saved["shuffled_predictions"].copy()
            print(f"layer={layer} reused completed probe checkpoint", flush=True)
            continue
        layer_results = []
        layer_parameters = {}
        for condition in CONDITIONS:
            x_train, train_index = load_layer(run, "train", condition, layer, n_shards)
            x_dev, dev_index = load_layer(run, "dev", condition, layer, n_shards)
            y_train = np.array([LABELS.index(row["label"]) for row in train_index])
            y_dev = np.array([LABELS.index(row["label"]) for row in dev_index])
            groups = [row["group_id"] for row in dev_index]
            if groups != dev_groups or not np.array_equal(y_dev, dev_truth):
                raise ValueError("Dev activation indexes differ across layers or conditions")
            pred, metrics, parameters = fit_probe(x_train, y_train, x_dev, y_dev,
                                                  manifest["seed"], args.probe_device,
                                                  return_parameters=True)
            layer_parameters[condition] = parameters
            row = {"layer": layer, "condition": condition, **metrics}
            results.append(row)
            layer_results.append(row)
            predictions[condition][layer] = pred
            if condition == "full":
                y_shuffled = np.random.default_rng(manifest["seed"]).permutation(y_train)
                shuffled_pred, control = fit_probe(x_train, y_shuffled, x_dev, y_dev,
                                                   manifest["seed"], args.probe_device)
                shuffled_predictions[layer] = shuffled_pred
                shuffled_row = {"layer": layer, "condition": "shuffled_train_labels", **control}
                shuffled.append(shuffled_row)
            print(f"layer={layer} {condition} dev_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        temporary = checkpoint_dir / f"layer{layer:02d}.partial.npz"
        np.savez_compressed(temporary, metadata=json.dumps({"config": checkpoint_config,
                                                              "results": layer_results,
                                                              "shuffled": shuffled_row}),
                            shuffled_predictions=shuffled_predictions[layer],
                            **{condition: predictions[condition][layer] for condition in CONDITIONS})
        temporary.replace(checkpoint)
        checkpoint_digest = sha256(checkpoint)
        for condition in CONDITIONS:
            save_probe_parameters(run, layer, condition, layer_parameters[condition],
                                  {"probe_method": checkpoint_config["method"],
                                   "converged": next(row["converged"] for row in layer_results
                                                     if row["condition"] == condition),
                                   "fitted_from_cached_e1_activations": True,
                                   "created_during_e1_analysis": True,
                                   "reproduced_dev_predictions": True,
                                   "original_checkpoint_sha256": checkpoint_digest})
    output = run / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    import csv
    with (output / "layer_probe_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results + shuffled)
    curves = {condition: {row["layer"]: row for row in results + shuffled if row["condition"] == condition}
              for condition in (*CONDITIONS, "shuffled_train_labels")}
    candidates = []
    for layer in range(n_layers):
        full = curves["full"][layer]
        control = max(curves["claim_only"][layer]["macro_f1"],
                      curves["evidence_only"][layer]["macro_f1"])
        candidates.append({"layer": layer, "full_macro_f1": full["macro_f1"],
                           "claim_only_macro_f1": curves["claim_only"][layer]["macro_f1"],
                           "evidence_only_macro_f1": curves["evidence_only"][layer]["macro_f1"],
                           "shuffled_label_macro_f1": curves["shuffled_train_labels"][layer]["macro_f1"],
                           "descriptive_ablation_gap": full["macro_f1"] - control})
    ranked = sorted(candidates, key=lambda row: (row["full_macro_f1"], row["descriptive_ablation_gap"]),
                    reverse=True)
    enough_examples = manifest["train_n"] >= 200 and manifest["dev_n"] >= 200
    provisional = [row["layer"] for row in ranked[:6]] if enough_examples else []
    group_members = {}
    for index, group in enumerate(dev_groups):
        group_members.setdefault(group, []).append(index)
    group_keys = list(group_members)
    rng = np.random.default_rng(manifest["seed"])
    intervals = {}
    for layer in provisional:
        full_scores, gap_scores, shuffled_gap_scores = [], [], []
        for _ in range(200):
            sampled = rng.choice(len(group_keys), size=len(group_keys), replace=True)
            indices = np.fromiter((index for group_number in sampled
                                   for index in group_members[group_keys[group_number]]), dtype=np.int64)
            gold = dev_truth[indices]
            full_f1 = f1_score(gold, predictions["full"][layer][indices], labels=[0, 1],
                               average="macro", zero_division=0)
            claim_f1 = f1_score(gold, predictions["claim_only"][layer][indices], labels=[0, 1],
                                average="macro", zero_division=0)
            evidence_f1 = f1_score(gold, predictions["evidence_only"][layer][indices], labels=[0, 1],
                                   average="macro", zero_division=0)
            shuffled_f1 = f1_score(gold, shuffled_predictions[layer][indices], labels=[0, 1],
                                   average="macro", zero_division=0)
            full_scores.append(full_f1)
            gap_scores.append(full_f1 - max(claim_f1, evidence_f1))
            shuffled_gap_scores.append(full_f1 - shuffled_f1)
        intervals[layer] = {"full_macro_f1_ci95": np.quantile(full_scores, [0.025, 0.975]).tolist(),
                            "descriptive_gap_ci95": np.quantile(gap_scores, [0.025, 0.975]).tolist(),
                            "shuffled_label_gap_ci95": np.quantile(shuffled_gap_scores, [0.025, 0.975]).tolist()}
    selected = [layer for layer in provisional
                if intervals[layer]["descriptive_gap_ci95"][0] > 0
                and intervals[layer]["shuffled_label_gap_ci95"][0] > 0]
    selection = {"selection_split": "dev", "block_indexing": "0-based", "read_position": "final_anchor",
                 "selected_candidate_layers": selected, "provisional_top_layers": provisional,
                 "bootstrap": "200 dev group resamples", "intervals": intervals,
                 "minimum_selection_samples": {"train": 200, "dev": 200},
                 "selection_status": "exploratory" if enough_examples else "smoke_only",
                 "ranked_layers": ranked,
                 "interpretation": "Linear decodability only; ablation gap is descriptive, not causal evidence."}
    dump(output / "candidate_layers.json", selection)
    plt.figure(figsize=(10, 5))
    for condition in (*CONDITIONS, "shuffled_train_labels"):
        plt.plot(range(n_layers), [curves[condition][i]["macro_f1"] for i in range(n_layers)],
                 marker=".", label=condition)
    plt.xlabel("0-based Qwen2.5-7B transformer block")
    plt.ylabel("Dev macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "layer_probe_macro_f1.png", dpi=180)
    plt.close()
    return selection


def run_all(args, run: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch cannot initialize CUDA; visible devices: "
                           f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}. "
                           "Check a fresh CUDA tensor allocation before resuming.")
    gpus = [int(item) for item in args.gpus.split(",")]
    worker_gpus = ([int(item) for item in args.resume_gpus.split(",")]
                   if args.resume_gpus else gpus)
    if args.resume_gpus and not args.resume:
        raise ValueError("--resume-gpus requires --resume")
    device_tokens = resolve_gpu_tokens(worker_gpus, os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       torch.cuda.device_count())
    args.shard_device_tokens = device_tokens
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError(f"Run exists: {run}; use --resume or a new --run-id")
    if args.resume:
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["gpus"] != args.gpus or manifest["model_id"] != args.model_id:
            raise ValueError("Resume parameters differ from existing run")
        current_hashes = {name: sha256(Path(__file__).with_name(name))
                          for name in ("fever_probe.py", "modeling.py", "data.py")}
        if manifest.get("code_sha256") != current_hashes:
            changed = {name for name, digest in current_hashes.items()
                       if manifest.get("code_sha256", {}).get(name) != digest}
            if changed != {"fever_probe.py"} or not (run / "device_override.json").is_file():
                raise ValueError("Code changed since this run began; resume would mix activation protocols")
            dump(run / "orchestration_resume_audit.json", {
                "reason": "GPU mapping and legacy shard-count repair; original extraction partition retained",
                "original_code_sha256": manifest["code_sha256"],
                "resume_code_sha256": current_hashes,
                "device_tokens": device_tokens,
            })
    else:
        prepare(args, run)
        if args.resume_gpus:
            raise ValueError("--resume-gpus requires --resume")
    manifest = json.loads((run / "manifest.json").read_text())
    n_shards = partition_count(manifest, run)
    if n_shards != len(gpus):
        dump(run / "partition_repair.json", {
            "issue": "Legacy code used the character count of the GPU string as the partition count",
            "original_gpu_string": manifest["gpus"], "n_shards": n_shards,
            "completed_shards_preserved": [i for i in range(n_shards) if shard_is_complete(run, i)],
            "worker_gpus": worker_gpus,
        })
    log_dir = run / "logs"
    log_dir.mkdir(exist_ok=True)
    pending_shards = [shard for shard in range(n_shards) if not shard_is_complete(run, shard)]
    if args.resume_shard_gpu is not None:
        if not args.resume or len(pending_shards) > 1:
            raise ValueError("--resume-shard-gpu requires --resume and at most one pending shard")
        override_token = (resolve_gpu_tokens([args.resume_shard_gpu],
                                             os.environ.get("CUDA_VISIBLE_DEVICES"),
                                             torch.cuda.device_count())[0] if pending_shards else None)
        if pending_shards:
            dump(run / "resume_device_assignment.json", {
                "pending_shard": pending_shards[0], "logical_gpu": args.resume_shard_gpu,
                "physical_device_token": override_token,
                "completed_shards_reused": [shard for shard in range(n_shards)
                                             if shard not in pending_shards],
            })
    else:
        override_token = None
    for shard in range(n_shards):
        if shard not in pending_shards:
            print(f"shard {shard}: already complete; reusing activations", flush=True)
    def launch(shard: int, worker: int):
        gpu, token = worker_gpus[worker], device_tokens[worker]
        if override_token is not None:
            token, gpu = override_token, args.resume_shard_gpu
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = token
        env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
        stream = (log_dir / f"gpu{gpu}_shard{shard}.log").open("a" if args.resume else "w")
        cmd = [sys.executable, "-m", "crestfc.fever_probe", "extract-shard", "--run-id", args.run_id,
               "--shard", str(shard), "--batch-size", str(args.batch_size)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(f"logical GPU {gpu} (visible token {token}) shard {shard}: PID {proc.pid}; log {stream.name}", flush=True)
        return gpu, shard, proc, stream
    processes = [launch(shard, worker) for worker, shard in
                 enumerate(pending_shards[:len(worker_gpus)])]
    failed = []
    next_shard = len(processes)
    for worker in range(len(processes)):
        gpu, shard, proc, stream = processes[worker]
        code = proc.wait()
        stream.close()
        if code:
            failed.append((gpu, shard))
        while next_shard < len(pending_shards) and worker == next_shard % len(worker_gpus):
            processes.append(launch(pending_shards[next_shard], worker))
            next_shard += 1
    for gpu, shard, proc, stream in processes[len(worker_gpus):]:
        code = proc.wait()
        stream.close()
        if code:
            failed.append((gpu, shard))
    if failed:
        raise RuntimeError(f"Extraction failed for {failed}; inspect logs, then rerun with --resume")
    selection = analyze(args, run)
    print(json.dumps({"run": str(run), "candidate_layers": selection["selected_candidate_layers"]},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "extract-shard", "analyze", "export-probes"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--cache-dir", default="/data/yangjun/.cache/huggingface")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--resume-gpus", help="Comma-separated logical GPUs for resumed shard workers")
    parser.add_argument("--train-per-label", type=int, default=0,
                        help="0 uses every source train example of each label")
    parser.add_argument("--dev-per-label", type=int, default=0,
                        help="0 uses the full source dev split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-device", default="cuda:0",
                        help="Device for linear probe fitting; use cpu for sklearn liblinear")
    parser.add_argument("--layers", default="all", help="Comma-separated zero-based layers for export-probes")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="Comma-separated E1 conditions for export-probes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-shard-gpu", type=int,
                        help="Run the sole unfinished shard on this logical visible GPU")
    args = parser.parse_args()
    if args.train_per_label < 0 or args.dev_per_label < 0 or args.batch_size < 1:
        parser.error("limits must be nonnegative and batch size positive")
    run = ROOT / "outputs/experiment_runs" / args.run_id
    if args.command == "all":
        run_all(args, run)
    elif args.command == "extract-shard":
        if args.shard is None:
            parser.error("extract-shard needs --shard")
        extract_shard(args, run)
    elif args.command == "analyze":
        analyze(args, run)
    else:
        print(json.dumps(export_e1_probes(args, run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
