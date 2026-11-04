"""Frozen Qwen2.5-7B causal tracing at the E1 relation anchor.

E2-0 pairs -> E2-1 clean decisions -> E2-2 bidirectional patches ->
E2-3 controls/local patches -> E2-4 grouped statistics -> E2-5 dev-only selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .counterfactual_probe import evidence_text, load_probe, normalized_claim, probe_path
from .data import read_jsonl, stable_id, write_jsonl
from .fever_probe import ROOT, sha256
from .modeling import get_transformer_blocks, load_model, make_patch_hook, render_encoder, temporary_hook

LABELS = ("SUPPORTS", "REFUTES")
DIRECTIONS = ("SUPPORTS_to_REFUTES", "REFUTES_to_SUPPORTS")
E1_CANDIDATES = (18, 19, 20, 21, 22, 23)
CONTROL_LAYERS = (4, 8, 12, 27)
ANSWER_SUFFIX = (" \nDetermine whether the evidence supports or refutes the claim, "
                 "using only the evidence. Return exactly one label.\nAnswer:")
MAIN_MODES = ("source", "same_input", "random_sample", "wrong_relation")
LOCAL_MODES = ("probe_axis", "orthogonal", "nonrelation_token", "answer_token", "answer_same_input")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def pair_family(pair: dict) -> str:
    return pair["supports"]["source_provenance"].get("revision_family_id", pair["pair_id"])


def check_pair(pair: dict) -> None:
    a, b = pair["supports"], pair["refutes"]
    if (pair["claim"] != a["claim"] or a["claim"] != b["claim"]
            or a["label"] != "SUPPORTS" or b["label"] != "REFUTES"
            or not evidence_text(a).strip() or not evidence_text(b).strip()
            or evidence_text(a) == evidence_text(b)
            or a["split"] != b["split"]):
        raise ValueError(f"Invalid fixed-claim relation-change pair: {pair['pair_id']}")


def read_reviews(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    reviews = {}
    for row in read_jsonl(path):
        pid = row["pair_id"]
        if pid in reviews:
            raise ValueError(f"Duplicate review for {pid}")
        reviews[pid] = row
    return reviews


def reviewed(pair: dict, review: dict | None) -> bool:
    return bool(review and review.get("approved") is True and review.get("reviewer")
                and all(review.get(key) is True for key in
                        ("claim_unchanged", "evidence_changed", "supports_label_correct",
                         "refutes_label_correct", "evidence_sufficient")))


def prepare(args, run: Path) -> dict:
    source = Path(args.pair_source).resolve()
    source_manifest = json.loads((source.parent / "manifest.json").read_text())
    split = source_manifest["source_split"]
    if split != "dev" and args.phase == "dev":
        raise ValueError("Dev selection may only use a dev pair source")
    if split != "test" and args.phase == "test":
        raise ValueError("Test evaluation may only use a test pair source")
    locked_layers = []
    selection_path = Path(args.locked_dev_selection).resolve() if args.locked_dev_selection else None
    if args.phase == "test":
        if selection_path is None:
            raise ValueError("Test evaluation requires --locked-dev-selection from a reviewed dev E2 run")
        selection = json.loads(selection_path.read_text())
        locked_layers = selection.get("selected_e3_layers", [])
        if not selection.get("e3_selection_eligible") or not locked_layers:
            raise ValueError("Dev E3 layer selection is ineligible or empty")
    e1_run = Path(args.e1_run).resolve()
    e1_manifest = json.loads((e1_run / "manifest.json").read_text())
    if args.phase == "test" and (selection.get("phase") != "dev"
                                  or selection.get("e1_manifest_sha256") != sha256(e1_run / "manifest.json")
                                  or selection.get("n_reviewed_pairs") != selection.get("n_pairs")
                                  or any(not isinstance(layer, int) or layer < 0 or layer >= 28
                                         for layer in locked_layers)):
        raise ValueError("Locked layer selection does not match reviewed dev and the frozen E1 run")
    if source_manifest.get("e1_manifest_sha256") != sha256(e1_run / "manifest.json"):
        raise ValueError("Counterfactual pair source was prepared against a different E1 run")
    if e1_manifest["read_position"] != "final_anchor" or e1_manifest["model_id"] != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("E1 model/prompt protocol differs from this E2 implementation")
    if not Path(e1_manifest["snapshot"]).is_dir():
        raise FileNotFoundError(e1_manifest["snapshot"])
    probe_dir = e1_run / "analysis" / "probes"
    probe_hashes = {}
    for layer in E1_CANDIDATES:
        load_probe(probe_dir, layer, e1_manifest, e1_run)
        probe_hashes[str(layer)] = sha256(probe_path(probe_dir, layer))
    reviews = read_reviews(Path(args.review_file).resolve() if args.review_file else None)
    pairs = []
    rejected = Counter()
    for pair in read_jsonl(source):
        check_pair(pair)
        review = reviews.get(pair["pair_id"])
        is_reviewed = reviewed(pair, review)
        if not is_reviewed and not args.allow_unreviewed:
            rejected["unreviewed"] += 1
            continue
        candidate = dict(pair)
        candidate["e2_review_status"] = "approved" if is_reviewed else "source_labels_unreviewed"
        if is_reviewed:
            candidate["e2_reviewer"] = review["reviewer"]
        pairs.append(candidate)
    if not pairs:
        raise ValueError("No eligible pairs. Supply reviewed labels or use --allow-unreviewed for an exploratory pilot")
    # A seeded subset avoids taking the first source-order claims in a small pilot.
    pairs.sort(key=lambda pair: stable_id(str(args.seed), pair["pair_id"]))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    pairs.sort(key=lambda pair: pair["pair_id"])
    if len(pairs) < 2:
        raise ValueError("At least two distinct Claims are needed for random-donor controls")
    if len(pairs) < args.shards:
        raise ValueError("Selected pairs are fewer than GPU shards; reduce GPU_TOKENS or increase MAX_PAIRS")
    claims = [normalized_claim(pair["claim"]) for pair in pairs]
    if len(claims) != len(set(claims)):
        raise ValueError("Each E2 pair must have a unique Claim in the selected sample")
    if len({pair_family(pair) for pair in pairs}) < 2:
        raise ValueError("Random-donor controls require at least two independent source families")
    layers = sorted(set(args.layers))
    if any(layer < 0 or layer >= 28 for layer in layers) or not set(E1_CANDIDATES) <= set(layers):
        raise ValueError("E2 must include all E1 candidate layers 18–23 and valid 0-based controls")
    if any(not any(int(stable_id(pair["pair_id"]), 16) % args.shards == shard for pair in pairs)
           for shard in range(args.shards)):
        raise ValueError("At least one GPU shard is empty; use fewer GPU_TOKENS or more pairs")
    manifest = {"phase": args.phase, "pair_source": str(source),
                "pair_source_sha256": sha256(source), "source_dataset": source_manifest["dataset"],
                "pair_source_manifest_sha256": sha256(source.parent / "manifest.json"),
                "source_split": split, "review_file": str(Path(args.review_file).resolve()) if args.review_file else None,
                "review_file_sha256": sha256(Path(args.review_file)) if args.review_file else None,
                "locked_dev_selection": str(selection_path) if selection_path else None,
                "locked_dev_selection_sha256": sha256(selection_path) if selection_path else None,
                "locked_e3_layers": locked_layers,
                "allow_unreviewed": args.allow_unreviewed, "reviewed_pairs": sum(
                    pair["e2_review_status"] == "approved" for pair in pairs),
                "rejected": dict(rejected), "n_pairs": len(pairs), "max_pairs": args.max_pairs,
                "model_id": e1_manifest["model_id"], "snapshot": e1_manifest["snapshot"],
                "cache_dir": e1_manifest["cache_dir"], "e1_run": str(e1_run),
                "e1_manifest_sha256": sha256(e1_run / "manifest.json"),
                "e1_probe_sha256": probe_hashes,
                "read_position": "E1 final Relation anchor token",
                "decision_suffix": ANSWER_SUFFIX, "candidate_labels": list(LABELS),
                "candidate_layers": list(E1_CANDIDATES), "control_layers": list(CONTROL_LAYERS),
                "layers": layers, "shards": args.shards, "seed": args.seed,
                "main_modes": list(MAIN_MODES), "local_modes": list(LOCAL_MODES),
                "code_sha256": sha256(Path(__file__))}
    if (run / "manifest.json").exists():
        old = json.loads((run / "manifest.json").read_text())
        if old != manifest:
            raise ValueError("Existing E2 run manifest differs; use a new run ID")
        return manifest
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "pairs.jsonl", pairs)
    write_jsonl(run / "review_queue.jsonl", (
        {"pair_id": pair["pair_id"], "claim": pair["claim"],
         "supports_evidence": pair["supports"]["evidence"],
         "refutes_evidence": pair["refutes"]["evidence"],
         "approved": False, "reviewer": None,
         "claim_unchanged": None, "evidence_changed": None,
         "supports_label_correct": None, "refutes_label_correct": None,
         "evidence_sufficient": None} for pair in pairs))
    dump(run / "manifest.json", manifest)
    return manifest


def candidate_ids(tokenizer) -> list[list[int]]:
    result = [tokenizer(" " + label, add_special_tokens=False)["input_ids"] for label in LABELS]
    if any(not ids for ids in result):
        raise ValueError("Empty label continuation")
    return result


def candidate_batch(model, tokenizer, prefix: str,
                    label_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise ValueError("Empty scoring prefix")
    max_cont = max(map(len, label_ids))
    pad = tokenizer.pad_token_id
    ids = torch.full((len(label_ids), len(prefix_ids) + max_cont), pad,
                     dtype=torch.long, device=model.device)
    attention = torch.zeros_like(ids)
    for i, continuation in enumerate(label_ids):
        content = prefix_ids + continuation
        ids[i, :len(content)] = torch.as_tensor(content, device=model.device)
        attention[i, :len(content)] = 1
    return ids, attention, len(prefix_ids)


def score_labels(model, tokenizer, prefix: str, label_ids: list[list[int]],
                 patch: tuple[int, int, torch.Tensor] | None = None) -> torch.Tensor:
    """Teacher-forced sequence log-probabilities with two labels in one forward."""
    ids, attention, prefix_len = candidate_batch(model, tokenizer, prefix, label_ids)
    max_cont = max(map(len, label_ids))
    if patch is None:
        hidden = model.model(input_ids=ids, attention_mask=attention, use_cache=False,
                             return_dict=True).last_hidden_state
    else:
        layer, token_index, vector = patch
        if token_index < 0 or token_index >= prefix_len:
            raise ValueError("Patch token must lie within the fixed prefix")
        source = vector.to(model.device, model.dtype).reshape(1, -1).expand(len(label_ids), -1)
        with temporary_hook(get_transformer_blocks(model)[layer], make_patch_hook(token_index, source)):
            hidden = model.model(input_ids=ids, attention_mask=attention, use_cache=False,
                                 return_dict=True).last_hidden_state
    start = prefix_len - 1
    logits = model.lm_head(hidden[:, start:start + max_cont, :]).float()
    log_probs = logits.log_softmax(dim=-1)
    values = []
    for i, continuation in enumerate(label_ids):
        token_ids = torch.as_tensor(continuation, device=model.device)
        selected = log_probs[i, :len(continuation), :].gather(-1, token_ids[:, None])
        values.append(selected.sum())
    return torch.stack(values).detach().cpu()


def capture_pair(model, tokenizer, label_ids: list[list[int]], pair: dict,
                 layers: list[int]) -> dict:
    """Read the E1 anchor in the same forward geometry used for decision scoring.

    Candidate tokens follow the answer and cannot influence the earlier anchor.
    """
    result = {}
    blocks = get_transformer_blocks(model)
    for side in ("supports", "refutes"):
        prefix, _ = render_encoder(pair[side], "full")
        e1_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        decision_prefix = prefix + ANSWER_SUFFIX
        decision_ids = tokenizer(decision_prefix, add_special_tokens=False)["input_ids"]
        if decision_ids[:len(e1_ids)] != e1_ids:
            raise ValueError("Decision suffix changed E1 anchor tokenization")
        input_ids, attention, _ = candidate_batch(model, tokenizer, decision_prefix, label_ids)
        anchors, first, answer = {}, {}, {}
        handles = []
        for layer in layers:
            def hook(_module, _inputs, output, layer=layer):
                state = output[0] if isinstance(output, tuple) else output
                anchors[layer] = state[0, len(e1_ids) - 1].detach().clone()
                first[layer] = state[0, 0].detach().clone()
                answer[layer] = state[0, len(decision_ids) - 1].detach().clone()
            handles.append(blocks[layer].register_forward_hook(hook))
        try:
            model.model(input_ids=input_ids, attention_mask=attention,
                        use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        result[side] = {"prefix": prefix, "anchor_index": len(e1_ids) - 1,
                        "token_count": len(e1_ids), "decision_index": len(decision_ids) - 1,
                        "anchor": anchors, "first": first, "answer": answer}
    return result


def probe_axis_patch(source: torch.Tensor, target: torch.Tensor, probe: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch only the E1 linear decision direction, plus its orthogonal complement."""
    weight = torch.as_tensor(probe["coef"] / probe["scale"],
                             dtype=torch.float32, device=target.device)
    delta = source.float() - target.float()
    projection = torch.dot(delta, weight) / torch.dot(weight, weight).clamp_min(1e-12) * weight
    return target.float() + projection, target.float() + delta - projection


def source_margin(scores: torch.Tensor, source_label: str) -> float:
    difference = float(scores[1] - scores[0])
    return difference if source_label == "REFUTES" else -difference


def prediction(scores: torch.Tensor) -> str:
    return "REFUTES" if float(scores[1]) >= float(scores[0]) else "SUPPORTS"


def run_pair(model, tokenizer, label_ids: list[list[int]], pair: dict,
             random_pair: dict, layers: list[int], e1_run: Path, e1_manifest: dict) -> dict:
    states = capture_pair(model, tokenizer, label_ids, pair, layers)
    random_states = capture_pair(model, tokenizer, label_ids, random_pair, layers)
    baseline = {}
    for side in ("supports", "refutes"):
        scores = score_labels(model, tokenizer, states[side]["prefix"] + ANSWER_SUFFIX, label_ids)
        baseline[side] = {"supports_logprob": float(scores[0]), "refutes_logprob": float(scores[1]),
                          "refutes_margin": float(scores[1] - scores[0]),
                          "prediction": prediction(scores), "gold_label": pair[side]["label"],
                          "correct": prediction(scores) == pair[side]["label"],
                          "anchor_index": states[side]["anchor_index"],
                          "prompt_tokens_before_suffix": states[side]["token_count"]}
    effects = []
    probes = {layer: load_probe(e1_run / "analysis" / "probes", layer, e1_manifest, e1_run)
              for layer in layers if layer in E1_CANDIDATES}
    for source_side, target_side in (("supports", "refutes"), ("refutes", "supports")):
        source, target = states[source_side], states[target_side]
        target_label = pair[target_side]["label"]
        source_label = pair[source_side]["label"]
        base_scores = torch.tensor([baseline[target_side]["supports_logprob"],
                                    baseline[target_side]["refutes_logprob"]])
        base_source_margin = source_margin(base_scores, source_label)
        for layer in layers:
            source_anchor = source["anchor"][layer]
            target_anchor = target["anchor"][layer]
            modes = {"source": (target["anchor_index"], source_anchor),
                     "same_input": (target["anchor_index"], target_anchor),
                     "random_sample": (target["anchor_index"], random_states[source_side]["anchor"][layer]),
                     "wrong_relation": (target["anchor_index"], random_states[target_side]["anchor"][layer])}
            if layer in E1_CANDIDATES:
                axis, orthogonal = probe_axis_patch(source_anchor, target_anchor, probes[layer])
                modes.update({"probe_axis": (target["anchor_index"], axis),
                              "orthogonal": (target["anchor_index"], orthogonal),
                              "nonrelation_token": (0, source["first"][layer]),
                              "answer_token": (target["decision_index"], source["answer"][layer]),
                              "answer_same_input": (target["decision_index"], target["answer"][layer])})
            for mode, (index, vector) in modes.items():
                patched = score_labels(model, tokenizer, target["prefix"] + ANSWER_SUFFIX,
                                       label_ids, (layer, index, vector))
                patched_source_margin = source_margin(patched, source_label)
                effects.append({"pair_id": pair["pair_id"], "family_id": pair_family(pair),
                                "direction": f"{source_label}_to_{target_label}",
                                "source_label": source_label, "target_label": target_label,
                                "layer": layer, "mode": mode, "patch_token_index": index,
                                "source_anchor_index": source["anchor_index"],
                                "target_anchor_index": target["anchor_index"],
                                "same_anchor_index": source["anchor_index"] == target["anchor_index"],
                                "token_length_difference": abs(source["token_count"] - target["token_count"]),
                                "source_margin_before": base_source_margin,
                                "source_margin_after": patched_source_margin,
                                "effect": patched_source_margin - base_source_margin,
                                "base_prediction": baseline[target_side]["prediction"],
                                "patched_prediction": prediction(patched),
                                "target_clean_correct": baseline[target_side]["correct"],
                                "source_clean_correct": baseline[source_side]["correct"],
                                "patch_delta_norm": float((vector.float() -
                                                           (target["answer"][layer].float() if mode in
                                                            {"answer_token", "answer_same_input"} else
                                                            target["first"][layer].float() if index == 0 else
                                                            target_anchor.float())).norm())})
    return {"pair_id": pair["pair_id"], "claim": pair["claim"],
            "family_id": pair_family(pair), "review_status": pair["e2_review_status"],
            "random_donor_pair_id": random_pair["pair_id"],
            "baseline": baseline, "effects": effects}


def shard_pairs(run: Path, shard: int, n_shards: int) -> list[dict]:
    return [pair for pair in read_jsonl(run / "pairs.jsonl")
            if int(stable_id(pair["pair_id"]), 16) % n_shards == shard]


def run_shard(args, run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    if not 0 <= args.shard < manifest["shards"]:
        raise ValueError("Shard index out of range")
    pairs = read_jsonl(run / "pairs.jsonl")
    subset = shard_pairs(run, args.shard, manifest["shards"])
    if not subset:
        raise ValueError("Empty E2 shard")
    by_id = {pair["pair_id"]: index for index, pair in enumerate(pairs)}
    e1_run = Path(manifest["e1_run"])
    e1_manifest = json.loads((e1_run / "manifest.json").read_text())
    for key, digest in manifest["e1_probe_sha256"].items():
        if sha256(probe_path(e1_run / "analysis" / "probes", int(key))) != digest:
            raise ValueError(f"E1 probe layer {key} changed after E2 preparation")
    model, tokenizer = load_model(manifest["snapshot"], manifest["cache_dir"], "bfloat16", "cuda")
    model.eval()
    if len(get_transformer_blocks(model)) != 28:
        raise ValueError("Expected Qwen2.5-7B with 28 transformer blocks")
    label_ids = candidate_ids(tokenizer)
    out = run / "pair_results"
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, pair in enumerate(subset):
            final = out / f"{pair['pair_id'].split(':')[-1]}.json"
            if final.exists():
                saved = json.loads(final.read_text())
                if saved.get("pair_id") == pair["pair_id"] and len(saved.get("effects", [])) == (
                        2 * sum(len(MAIN_MODES) + (len(LOCAL_MODES) if layer in E1_CANDIDATES else 0)
                                for layer in manifest["layers"])):
                    print(f"shard={args.shard} pair={index + 1}/{len(subset)} reused", flush=True)
                    continue
                raise ValueError(f"Incomplete or incompatible pair result: {final}")
            position = by_id[pair["pair_id"]]
            random_pair = next(pairs[(position + offset) % len(pairs)] for offset in range(1, len(pairs))
                               if normalized_claim(pairs[(position + offset) % len(pairs)]["claim"])
                               != normalized_claim(pair["claim"])
                               and pair_family(pairs[(position + offset) % len(pairs)]) != pair_family(pair))
            result = run_pair(model, tokenizer, label_ids, pair, random_pair, manifest["layers"],
                              e1_run, e1_manifest)
            temporary = final.with_suffix(".partial")
            dump(temporary, result)
            temporary.replace(final)
            print(f"shard={args.shard} pair={index + 1}/{len(subset)} complete", flush=True)
    dump(run / "pair_results" / f"shard{args.shard}_complete.json", {
        "shard": args.shard, "n_pairs": len(subset), "pair_ids": [pair["pair_id"] for pair in subset]})


def _group_ci(rows: list[dict], value_key: str, reps: int, seed: int) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(float(row[value_key]))
    keys = list(groups)
    if not keys:
        raise ValueError("No rows for bootstrap")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        sampled = rng.integers(0, len(keys), len(keys))
        values = [value for i in sampled for value in groups[keys[i]]]
        boots.append(float(np.mean(values)))
    return np.quantile(boots, [0.025, 0.975]).astype(float).tolist()


def analyze(args, run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    pairs = read_jsonl(run / "pairs.jsonl")
    expected_effects = 2 * sum(len(MAIN_MODES) + (len(LOCAL_MODES) if layer in E1_CANDIDATES else 0)
                               for layer in manifest["layers"])
    results = {}
    for pair in pairs:
        path = run / "pair_results" / f"{pair['pair_id'].split(':')[-1]}.json"
        if not path.exists():
            raise FileNotFoundError(f"E2 pair unfinished: {path}")
        result = json.loads(path.read_text())
        if result["pair_id"] != pair["pair_id"]:
            raise ValueError(f"Pair result ID mismatch: {path}")
        if len(result.get("effects", [])) != expected_effects:
            raise ValueError(f"Incomplete pair effects: {path}")
        results[pair["pair_id"]] = result
    baselines = []
    effects = []
    for pair in pairs:
        result = results[pair["pair_id"]]
        a, b = result["baseline"]["supports"], result["baseline"]["refutes"]
        baselines.append({"pair_id": pair["pair_id"], "family_id": result["family_id"],
                          "review_status": result["review_status"],
                          "supports_prediction": a["prediction"], "refutes_prediction": b["prediction"],
                          "supports_margin": a["refutes_margin"], "refutes_margin": b["refutes_margin"],
                          "pair_correct": bool(a["correct"] and b["correct"]),
                          "direction_correct": bool(b["refutes_margin"] > a["refutes_margin"]),
                          "token_length_difference": abs(a["prompt_tokens_before_suffix"] -
                                                         b["prompt_tokens_before_suffix"])})
        effects.extend(result["effects"])
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "baseline.jsonl", baselines)
    write_jsonl(out / "causal_effects.jsonl", effects)
    max_same = max(abs(row["effect"]) for row in effects if row["mode"] in
                   {"same_input", "answer_same_input"})
    baseline_by_pair = {row["pair_id"]: row for row in baselines}
    summary = []
    for layer in manifest["layers"]:
        for mode in list(MAIN_MODES) + (list(LOCAL_MODES) if layer in E1_CANDIDATES else []):
            selected = [row for row in effects if row["layer"] == layer and row["mode"] == mode]
            if not selected:
                continue
            subsets = [("all", selected),
                       ("clean_pair_correct", [row for row in selected
                                               if baseline_by_pair[row["pair_id"]]["pair_correct"]]),
                       ("length_matched", [row for row in selected
                                           if baseline_by_pair[row["pair_id"]]["token_length_difference"] == 0])]
            subsets.extend((f"direction_{direction}",
                            [row for row in selected if row["direction"] == direction])
                           for direction in DIRECTIONS)
            for subset_name, subset in subsets:
                if not subset:
                    continue
                values = np.asarray([row["effect"] for row in subset])
                summary.append({"layer": layer, "mode": mode, "subset": subset_name,
                                "n_pairs": len({row["pair_id"] for row in subset}),
                                "n_directions": len(subset), "mean_ie": float(values.mean()),
                                "median_ie": float(np.median(values)),
                                "positive_fraction": float((values > 0).mean()),
                                "source_label_prediction_rate": float(np.mean([
                                    row["patched_prediction"] == row["source_label"] for row in subset])),
                                "source_label_flip_fraction": float(np.mean([
                                    row["base_prediction"] != row["source_label"] and
                                    row["patched_prediction"] == row["source_label"] for row in subset])),
                                "away_from_source_flip_fraction": float(np.mean([
                                    row["base_prediction"] == row["source_label"] and
                                    row["patched_prediction"] != row["source_label"] for row in subset])),
                                "family_bootstrap_ci95": _group_ci(subset, "effect", args.bootstrap, manifest["seed"])})
    with (out / "causal_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    probe_agreement = []
    source_parent = Path(manifest["pair_source"]).parent
    probe_predictions = source_parent / "analysis" / "pair_predictions.jsonl"
    if probe_predictions.exists():
        probe_by_layer = defaultdict(dict)
        ids = set(results)
        for row in read_jsonl(probe_predictions):
            if row["pair_id"] in ids and row["layer"] in E1_CANDIDATES:
                probe_by_layer[row["layer"]][row["pair_id"]] = row
        for layer in E1_CANDIDATES:
            if len(probe_by_layer[layer]) != len(pairs):
                continue
            both = Counter((bool(probe_by_layer[layer][row["pair_id"]]["pair_correct"]),
                            row["pair_correct"])
                           for row in baselines)
            direction = Counter((probe_by_layer[layer][row["pair_id"]]["refutes_score_shift"] > 0,
                                 row["direction_correct"]) for row in baselines)
            probe_agreement.append({"layer": layer, "probe_correct_model_wrong": both[(True, False)],
                                    "probe_wrong_model_correct": both[(False, True)],
                                    "both_correct": both[(True, True)], "both_wrong": both[(False, False)],
                                    "probe_direction_correct_model_wrong": direction[(True, False)],
                                    "probe_direction_wrong_model_correct": direction[(False, True)],
                                    "both_direction_correct": direction[(True, True)],
                                    "both_direction_wrong": direction[(False, False)]})
    e1_selection = json.loads((Path(manifest["e1_run"]) / "analysis" / "candidate_layers.json").read_text())
    e1_f1 = {row["layer"]: row["full_macro_f1"] for row in e1_selection["ranked_layers"]}
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(11, 5))
    profile_modes = (("source", "#1f77b4"), ("random_sample", "#ff7f0e"),
                     ("wrong_relation", "#2ca02c"))
    for mode, color in profile_modes:
        profile = {row["layer"]: row for row in summary
                   if row["mode"] == mode and row["subset"] == "all"}
        if profile:
            xs = sorted(profile)
            axis.plot(xs, [profile[layer]["mean_ie"] for layer in xs],
                      marker="o", markersize=3, color=color, label=f"{mode} mean IE")
            if mode == "source":
                axis.fill_between(xs,
                                  [profile[layer]["family_bootstrap_ci95"][0] for layer in xs],
                                  [profile[layer]["family_bootstrap_ci95"][1] for layer in xs],
                                  color=color, alpha=0.15)
    for layer in (18, 19, 20):
        if layer in manifest["layers"]:
            axis.axvline(layer, color="gray", linewidth=0.7, alpha=0.5)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(xlabel="Transformer block (0-based)", ylabel="Source-label margin change (IE)",
             title="E2 causal effects and E1 linear decodability")
    secondary = axis.twinx()
    xs = sorted(set(manifest["layers"]) & set(e1_f1))
    secondary.plot(xs, [e1_f1[layer] for layer in xs], color="black", linestyle="--",
                   linewidth=1.2, label="E1 full Probe Macro-F1")
    secondary.set_ylabel("E1 Macro-F1")
    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = secondary.get_legend_handles_labels()
    axis.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "causal_layer_profile.png", dpi=180)
    plt.close(figure)
    candidate_scores = []
    for layer in manifest["layers"]:
        data = {(row["mode"], row["subset"]): row for row in summary if row["layer"] == layer}
        if ("source", "all") not in data:
            continue
        source = data[("source", "all")]
        by_direction = defaultdict(dict)
        for row in effects:
            if row["layer"] == layer and row["mode"] in {"source", "random_sample"}:
                by_direction[(row["pair_id"], row["direction"])][row["mode"]] = row
        adjusted_rows = []
        for modes in by_direction.values():
            if "source" in modes and "random_sample" in modes:
                adjusted_rows.append({"family_id": modes["source"]["family_id"],
                                      "adjusted": modes["source"]["effect"] -
                                                  abs(modes["random_sample"]["effect"])})
        if not adjusted_rows:
            continue
        adjusted = float(np.mean([row["adjusted"] for row in adjusted_rows]))
        directional_scores = []
        for direction in DIRECTIONS:
            source_rows = [modes["source"] for (pair_id, name), modes in by_direction.items()
                           if name == direction and "source" in modes]
            adjusted_direction = [{"family_id": modes["source"]["family_id"],
                                   "adjusted": modes["source"]["effect"] -
                                               abs(modes["random_sample"]["effect"])}
                                  for (pair_id, name), modes in by_direction.items()
                                  if name == direction and "source" in modes and "random_sample" in modes]
            if source_rows and adjusted_direction:
                directional_scores.append({"direction": direction,
                                           "source_mean_ie": float(np.mean([row["effect"] for row in source_rows])),
                                           "source_ci95": _group_ci(source_rows, "effect", args.bootstrap,
                                                                    manifest["seed"]),
                                           "control_adjusted_ci95": _group_ci(
                                               adjusted_direction, "adjusted", args.bootstrap,
                                               manifest["seed"])})
        candidate_scores.append({"layer": layer, "e1_full_macro_f1": e1_f1[layer],
                                 "source_mean_ie": source["mean_ie"],
                                 "source_ci95": source["family_bootstrap_ci95"],
                                 "random_abs_mean_ie": float(np.mean([
                                     abs(modes["random_sample"]["effect"])
                                     for modes in by_direction.values() if "random_sample" in modes])),
                                 "control_adjusted_score": adjusted,
                                 "control_adjusted_ci95": _group_ci(adjusted_rows, "adjusted",
                                                                    args.bootstrap, manifest["seed"]),
                                 "n_pairs": source["n_pairs"],
                                 "n_clean_pairs": data.get(("source", "clean_pair_correct"), {}).get("n_pairs", 0),
                                 "clean_pair_source_mean_ie": data.get(("source", "clean_pair_correct"), {}).get("mean_ie"),
                                 "directional_scores": directional_scores})
    if manifest["phase"] == "dev":
        candidate_scores.sort(key=lambda row: row["control_adjusted_score"], reverse=True)
    else:
        candidate_scores.sort(key=lambda row: row["layer"])
    eligible = (manifest["phase"] == "dev" and manifest["reviewed_pairs"] == manifest["n_pairs"]
                and max_same <= args.same_input_tolerance)
    selected = [row["layer"] for row in candidate_scores
                if eligible and row["n_pairs"] >= args.min_pairs
                and row["source_ci95"][0] > 0
                and row["control_adjusted_ci95"][0] > 0
                and len(row["directional_scores"]) == 2
                and all(direction["source_ci95"][0] > 0 and
                        direction["control_adjusted_ci95"][0] > 0
                        for direction in row["directional_scores"])][:args.top_k]
    from scipy.stats import spearmanr
    rho = (float(spearmanr([row["e1_full_macro_f1"] for row in candidate_scores],
                           [row["control_adjusted_score"] for row in candidate_scores]).statistic)
           if len(candidate_scores) >= 3 else None)
    if rho is not None and not np.isfinite(rho):
        rho = None
    report = {"phase": manifest["phase"], "n_pairs": len(pairs),
              "n_reviewed_pairs": manifest["reviewed_pairs"],
              "e1_manifest_sha256": manifest["e1_manifest_sha256"],
              "clean_pair_correct_rate": float(np.mean([row["pair_correct"] for row in baselines])),
              "clean_direction_correct_rate": float(np.mean([row["direction_correct"] for row in baselines])),
              "max_abs_same_input_ie": max_same, "same_input_tolerance": args.same_input_tolerance,
              "same_input_passed": max_same <= args.same_input_tolerance,
              "candidate_layer_scores": candidate_scores, "probe_decision_agreement": probe_agreement,
              "e1_f1_vs_causal_score_spearman": rho,
              "most_causal_layer": next((row["layer"] for row in candidate_scores
                                          if row["source_ci95"][0] > 0 and
                                          row["control_adjusted_ci95"][0] > 0 and
                                          len(row["directional_scores"]) == 2 and
                                          all(direction["source_ci95"][0] > 0 and
                                                  direction["control_adjusted_ci95"][0] > 0
                                                  for direction in row["directional_scores"])), None)
                                          if manifest["phase"] == "dev" else None,
              "e1_candidate_layers": list(E1_CANDIDATES),
              "locked_dev_selected_layers": manifest.get("locked_e3_layers", []),
              "locked_test_layer_scores": [row for row in candidate_scores
                                           if row["layer"] in manifest.get("locked_e3_layers", [])],
              "selected_e3_layers": selected, "e3_selection_eligible": eligible,
              "selection_reason": ("reviewed dev pairs and same-input check passed" if eligible else
                                   "test evaluation uses frozen dev layers; no test layer selection"
                                   if manifest["phase"] == "test" else
                                   "exploratory only: requires reviewed dev pairs and same-input control pass"),
              "selection_rule": "dev-only: source IE and paired source-minus-abs(random) IE family-CI lower > 0 overall and separately in both directions; rank by adjusted score",
              "bootstrap": f"{args.bootstrap} source-family resamples",
              "interpretation": "Patching at one block output is causal for this prompt/output metric; it does not prove exclusive storage or semantic purity."}
    dump(out / "e2_report.json", report)
    dump(out / "e3_layer_selection.json", {key: report[key] for key in
                                           ("selected_e3_layers", "e3_selection_eligible", "selection_reason",
                                            "selection_rule", "candidate_layer_scores", "phase",
                                            "e1_manifest_sha256", "n_reviewed_pairs", "n_pairs")})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run-shard", "analyze"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-source", default=str(ROOT / "outputs/experiment_runs/counterfactual_e1_calibration/vitaminc_dev/pairs.jsonl"))
    parser.add_argument("--phase", choices=["dev", "test"], default="dev")
    parser.add_argument("--e1-run", default=str(ROOT / "outputs/experiment_runs/fever7b_e1_main"))
    parser.add_argument("--review-file")
    parser.add_argument("--locked-dev-selection")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(28)))
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--same-input-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_pairs < 0 or args.shards < 1 or args.bootstrap < 100 or args.top_k < 1:
        parser.error("Invalid limits")
    run = ROOT / "outputs" / "experiment_runs" / args.run_id
    if args.command == "prepare":
        print(json.dumps(prepare(args, run), ensure_ascii=False, indent=2))
    elif args.command == "run-shard":
        run_shard(args, run)
    else:
        print(json.dumps(analyze(args, run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
