"""Same-layer causal patching for fixed-Claim VitaminC Evidence Flips.

The frozen Qwen2.5-7B model receives a normal binary fact-checking prompt.  For
each fixed-Claim SUPPORTS/REFUTES pair, block outputs are patched in both
directions at the last Evidence token, the Relation Anchor, and the token just
before the answer.  The output metric is the teacher-forced
log P(SUPPORTS) - log P(REFUTES) margin.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .data import file_sha256, read_jsonl, stable_id, write_jsonl
from .modeling import get_transformer_blocks, load_model, make_patch_hook, temporary_hook

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_SOURCE = (
    ROOT / "outputs/experiment_runs/counterfactual_e1_calibration/vitaminc_dev/pairs.jsonl"
)
DEFAULT_SNAPSHOT = (
    "/data/yangjun/.cache/huggingface/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
)
LABELS = ("SUPPORTS", "REFUTES")
POSITIONS = ("evidence_end", "relation_anchor", "answer_pre")
MODES = ("cf_donor", "identity", "random_donor", "opposite_direction_donor")
DEFAULT_LAYERS = (0, 3, 6, 9, 12, 15, 18, 19, 20, 23, 27)
PROMPT_VERSION = "evidence_flip_chat_v1"
TIE_TOL = 1e-6


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


def pair_family(pair: dict) -> str:
    return str(pair["supports"].get("source_provenance", {}).get(
        "revision_family_id", pair["pair_id"]
    ))


def evidence_text(row: dict) -> str:
    return "\n".join(str(item["text"]).strip() for item in row["evidence"])


def validate_pair(pair: dict) -> None:
    supports, refutes = pair["supports"], pair["refutes"]
    if (
        pair.get("claim") != supports.get("claim")
        or supports.get("claim") != refutes.get("claim")
        or supports.get("label") != "SUPPORTS"
        or refutes.get("label") != "REFUTES"
        or not evidence_text(supports)
        or not evidence_text(refutes)
        or evidence_text(supports) == evidence_text(refutes)
    ):
        raise ValueError(f"Invalid fixed-Claim Evidence Flip: {pair.get('pair_id')}")


def read_reviews(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    result = {}
    for row in read_jsonl(path):
        pair_id = row.get("pair_id")
        if not pair_id or pair_id in result:
            raise ValueError(f"Missing or duplicate review pair_id: {pair_id}")
        result[pair_id] = row
    return result


def review_approved(review: dict | None) -> bool:
    required = (
        "claim_unchanged",
        "evidence_changed",
        "supports_label_correct",
        "refutes_label_correct",
        "evidence_sufficient",
    )
    return bool(
        review
        and review.get("approved") is True
        and str(review.get("reviewer", "")).strip()
        and all(review.get(field) is True for field in required)
    )


def review_row(pair: dict) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "claim": pair["claim"],
        "supports_evidence": pair["supports"]["evidence"],
        "refutes_evidence": pair["refutes"]["evidence"],
        "approved": False,
        "reviewer": None,
        "claim_unchanged": None,
        "evidence_changed": None,
        "supports_label_correct": None,
        "refutes_label_correct": None,
        "evidence_sufficient": None,
    }


def _ranked_pairs(source: Path, seed: int) -> list[dict]:
    pairs = read_jsonl(source)
    if not pairs:
        raise ValueError("Pair source is empty")
    seen = set()
    for pair in pairs:
        validate_pair(pair)
        if pair["pair_id"] in seen:
            raise ValueError(f"Duplicate pair_id: {pair['pair_id']}")
        seen.add(pair["pair_id"])
    return sorted(pairs, key=lambda pair: stable_id(str(seed), pair["pair_id"]))


def make_review_queue(args, run: Path) -> dict:
    source = Path(args.pair_source).resolve()
    pairs = _ranked_pairs(source, args.seed)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    run.mkdir(parents=True, exist_ok=True)
    queue = run / "review_queue.jsonl"
    write_jsonl(queue, (review_row(pair) for pair in pairs))
    audit = {
        "pair_source": str(source),
        "pair_source_sha256": file_sha256(source),
        "seed": args.seed,
        "n_pairs": len(pairs),
        "review_queue": str(queue),
    }
    dump(run / "review_queue_manifest.json", audit)
    return audit


def _mean_evidence_chars(pair: dict) -> float:
    return 0.5 * (len(evidence_text(pair["supports"])) + len(evidence_text(pair["refutes"])))


def select_random_donors(pairs: list[dict], seed: int) -> dict[str, str]:
    """Choose an independent, approximately length-matched donor pair."""
    mapping = {}
    for target in pairs:
        target_length = _mean_evidence_chars(target)
        candidates = [
            donor
            for donor in pairs
            if donor["pair_id"] != target["pair_id"]
            and donor["claim"].strip().casefold() != target["claim"].strip().casefold()
            and pair_family(donor) != pair_family(target)
        ]
        if not candidates:
            raise ValueError("Random controls require at least two independent Claims/families")
        donor = min(
            candidates,
            key=lambda item: (
                abs(_mean_evidence_chars(item) - target_length),
                stable_id(str(seed), target["pair_id"], item["pair_id"]),
            ),
        )
        mapping[target["pair_id"]] = donor["pair_id"]
    return mapping


def shard_for_pair(pair_id: str, shards: int) -> int:
    return int(stable_id(pair_id), 16) % shards


def prepare(args, run: Path) -> dict:
    source = Path(args.pair_source).resolve()
    snapshot = Path(args.snapshot).resolve()
    review_path = Path(args.review_file).resolve() if args.review_file else None
    if not source.is_file():
        raise FileNotFoundError(source)
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(snapshot)
    reviews = read_reviews(review_path)
    eligible = []
    rejected = Counter()
    for pair in _ranked_pairs(source, args.seed):
        review = reviews.get(pair["pair_id"])
        approved = review_approved(review)
        if review_path is not None and not approved:
            rejected["not_approved"] += 1
            continue
        if not approved and not args.allow_unreviewed:
            rejected["unreviewed"] += 1
            continue
        selected = dict(pair)
        selected["review_status"] = "approved" if approved else "source_labels_unreviewed"
        selected["reviewer"] = review.get("reviewer") if approved else None
        eligible.append(selected)
    if args.max_pairs:
        eligible = eligible[: args.max_pairs]
    if len(eligible) < 2:
        raise ValueError(
            "Need at least two eligible pairs. Supply an approved --review-file, or use "
            "--allow-unreviewed only for an exploratory smoke test."
        )
    if len(eligible) < args.shards:
        raise ValueError("Selected pairs are fewer than shards")
    donors = select_random_donors(eligible, args.seed)
    selected_pairs = []
    for pair in eligible:
        item = dict(pair)
        item["random_donor_pair_id"] = donors[pair["pair_id"]]
        selected_pairs.append(item)
    layers = sorted(set(args.layers))
    if not layers or any(layer < 0 or layer >= 28 for layer in layers):
        raise ValueError("Layers must be unique Qwen block indices in [0, 27]")
    if any(not any(shard_for_pair(pair["pair_id"], args.shards) == shard for pair in selected_pairs)
           for shard in range(args.shards)):
        raise ValueError("A shard is empty; use fewer shards or more pairs")
    prompt_spec = {
        "version": PROMPT_VERSION,
        "labels": list(LABELS),
        "system": "You are a fact-checking assistant. Use only the supplied evidence.",
    }
    manifest = {
        "protocol": "e2_evidence_flip_same_layer_v1",
        "run_id": args.run_id,
        "pair_source": str(source),
        "pair_source_sha256": file_sha256(source),
        "review_file": str(review_path) if review_path else None,
        "review_file_sha256": file_sha256(review_path) if review_path else None,
        "allow_unreviewed": bool(args.allow_unreviewed),
        "n_pairs": len(selected_pairs),
        "n_reviewed_pairs": sum(pair["review_status"] == "approved" for pair in selected_pairs),
        "rejected": dict(rejected),
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "snapshot": str(snapshot),
        "snapshot_config_sha256": file_sha256(snapshot / "config.json"),
        "dtype": "bfloat16",
        "layers": layers,
        "positions": list(POSITIONS),
        "modes": list(MODES),
        "shards": args.shards,
        "seed": args.seed,
        "prompt": prompt_spec,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt_spec, sort_keys=True).encode()
        ).hexdigest(),
        "margin": "logP(SUPPORTS)-logP(REFUTES); intervention effect is signed toward donor label",
        "no_training": True,
        "code_sha256": file_sha256(Path(__file__)),
    }
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf8"))
        if old != manifest:
            raise ValueError("Existing run manifest differs; use a new run ID")
        return manifest
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "pairs.jsonl", selected_pairs)
    write_jsonl(run / "review_queue.jsonl", (review_row(pair) for pair in selected_pairs))
    dump(run / "manifest.json", manifest)
    dump(
        run / "data_audit.json",
        {
            "selected_pairs": len(selected_pairs),
            "reviewed_pairs": manifest["n_reviewed_pairs"],
            "review_status": dict(Counter(pair["review_status"] for pair in selected_pairs)),
            "shard_sizes": {
                str(shard): sum(
                    shard_for_pair(pair["pair_id"], args.shards) == shard
                    for pair in selected_pairs
                )
                for shard in range(args.shards)
            },
        },
    )
    return manifest


def _content(row: dict) -> tuple[str, dict[str, tuple[int, int]]]:
    parts = ["Claim:\n", row["claim"], "\n\nEvidence:\n"]
    evidence_start = sum(map(len, parts))
    for index, item in enumerate(row["evidence"]):
        parts.append(f"[{item['evidence_id']}] ")
        parts.append(item["text"].strip())
        if index + 1 < len(row["evidence"]):
            parts.append("\n")
    evidence_end = sum(map(len, parts))
    parts.append("\n\n")
    anchor_start = sum(map(len, parts))
    parts.append("Relation anchor:")
    anchor_end = sum(map(len, parts))
    parts.append(
        "\nDetermine whether the evidence supports or refutes the claim. "
        "Return exactly one label: SUPPORTS or REFUTES."
    )
    return "".join(parts), {
        "evidence": (evidence_start, evidence_end),
        "relation_anchor": (anchor_start, anchor_end),
    }


def _last_overlapping(offsets: Iterable[tuple[int, int]], span: tuple[int, int]) -> int:
    start, end = span
    indices = [index for index, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not indices:
        raise ValueError(f"No token overlaps span {span}")
    return indices[-1]


def prompt_encoding(tokenizer, row: dict) -> tuple[list[int], dict[str, int], str]:
    content, content_spans = _content(row)
    messages = [
        {
            "role": "system",
            "content": "You are a fact-checking assistant. Use only the supplied evidence.",
        },
        {"role": "user", "content": content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    template_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    ids = list(encoded["input_ids"])
    if ids != list(template_ids):
        raise ValueError("Chat template tokenization is not reproducible")
    content_start = rendered.find(content)
    if content_start < 0:
        raise ValueError("Rendered chat template does not contain the user content")
    evidence_span = tuple(content_start + value for value in content_spans["evidence"])
    anchor_span = tuple(content_start + value for value in content_spans["relation_anchor"])
    positions = {
        "evidence_end": _last_overlapping(encoded["offset_mapping"], evidence_span),
        "relation_anchor": _last_overlapping(encoded["offset_mapping"], anchor_span),
        "answer_pre": len(ids) - 1,
    }
    if not (positions["evidence_end"] < positions["relation_anchor"] < positions["answer_pre"]):
        raise ValueError(f"Unexpected causal position order: {positions}")
    return ids, positions, rendered


def candidate_ids(tokenizer) -> list[list[int]]:
    values = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in LABELS]
    if any(not ids for ids in values):
        raise ValueError("A label continuation is empty")
    return values


def candidate_batch(
    model, prefix_ids: list[int], labels: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prefix_length = len(prefix_ids)
    maximum = max(map(len, labels))
    ids = torch.full(
        (len(labels), prefix_length + maximum),
        model.config.pad_token_id,
        dtype=torch.long,
        device=model.device,
    )
    attention = torch.zeros_like(ids)
    for index, continuation in enumerate(labels):
        content = prefix_ids + continuation
        ids[index, : len(content)] = torch.as_tensor(content, device=model.device)
        attention[index, : len(content)] = 1
    return ids, attention, prefix_length


def _scores_from_hidden(
    model,
    hidden: torch.Tensor,
    prefix_length: int,
    labels: list[list[int]],
) -> torch.Tensor:
    maximum = max(map(len, labels))
    logits = model.lm_head(hidden[:, prefix_length - 1 : prefix_length - 1 + maximum]).float()
    log_probs = logits.log_softmax(dim=-1)
    values = []
    for index, continuation in enumerate(labels):
        token_ids = torch.as_tensor(continuation, device=hidden.device)
        selected = log_probs[index, : len(continuation)].gather(-1, token_ids[:, None])
        values.append(selected.sum())
    return torch.stack(values).detach().cpu()


def clean_forward(
    model,
    prefix_ids: list[int],
    positions: dict[str, int],
    labels: list[list[int]],
    layers: list[int],
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    ids, attention, prefix_length = candidate_batch(model, prefix_ids, labels)
    captured: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    handles = []
    blocks = get_transformer_blocks(model)
    for layer in layers:
        def hook(_module, _inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            for name, position in positions.items():
                captured[layer][name] = hidden[0, position].detach().clone()

        handles.append(blocks[layer].register_forward_hook(hook))
    try:
        hidden = model.model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    if any(set(captured[layer]) != set(POSITIONS) for layer in layers):
        raise RuntimeError("Failed to capture every requested layer/position")
    return _scores_from_hidden(model, hidden, prefix_length, labels), captured


def patched_scores(
    model,
    prefix_ids: list[int],
    labels: list[list[int]],
    layer: int,
    position: int,
    vector: torch.Tensor,
) -> torch.Tensor:
    ids, attention, prefix_length = candidate_batch(model, prefix_ids, labels)
    if not 0 <= position < prefix_length:
        raise ValueError("Patch position must be in the prompt prefix")
    source = vector.to(model.device, model.dtype).reshape(1, -1).expand(len(labels), -1)
    with temporary_hook(get_transformer_blocks(model)[layer], make_patch_hook(position, source)):
        hidden = model.model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    return _scores_from_hidden(model, hidden, prefix_length, labels)


def prediction(scores: torch.Tensor | list[float]) -> str:
    return "SUPPORTS" if float(scores[0]) >= float(scores[1]) else "REFUTES"


def margin(scores: torch.Tensor | list[float]) -> float:
    return float(scores[0]) - float(scores[1])


def capture_pair(model, tokenizer, labels: list[list[int]], pair: dict, layers: list[int]) -> dict:
    result = {}
    for side in ("supports", "refutes"):
        prefix_ids, positions, rendered = prompt_encoding(tokenizer, pair[side])
        scores, states = clean_forward(model, prefix_ids, positions, labels, layers)
        result[side] = {
            "prefix_ids": prefix_ids,
            "positions": positions,
            "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "scores": scores,
            "states": states,
        }
    return result


def run_pair(
    model,
    tokenizer,
    labels: list[list[int]],
    pair: dict,
    random_pair: dict,
    layers: list[int],
) -> dict:
    current = capture_pair(model, tokenizer, labels, pair, layers)
    random_states = capture_pair(model, tokenizer, labels, random_pair, layers)
    baseline = {}
    for side in ("supports", "refutes"):
        scores = current[side]["scores"]
        baseline[side] = {
            "gold_label": pair[side]["label"],
            "supports_logprob": float(scores[0]),
            "refutes_logprob": float(scores[1]),
            "margin": margin(scores),
            "prediction": prediction(scores),
            "correct": prediction(scores) == pair[side]["label"],
            "positions": current[side]["positions"],
            "prompt_tokens": len(current[side]["prefix_ids"]),
            "prompt_sha256": current[side]["prompt_sha256"],
        }
    effects = []
    for source_side, target_side in (("supports", "refutes"), ("refutes", "supports")):
        source_label = pair[source_side]["label"]
        target_label = pair[target_side]["label"]
        clean_scores = current[target_side]["scores"]
        clean_margin = margin(clean_scores)
        for layer in layers:
            for position_name in POSITIONS:
                target_vector = current[target_side]["states"][layer][position_name]
                interventions = {
                    "cf_donor": (source_label, pair["pair_id"],
                                 current[source_side]["states"][layer][position_name]),
                    "identity": (target_label, pair["pair_id"], target_vector),
                    "random_donor": (source_label, random_pair["pair_id"],
                                     random_states[source_side]["states"][layer][position_name]),
                    "opposite_direction_donor": (
                        target_label,
                        random_pair["pair_id"],
                        random_states[target_side]["states"][layer][position_name],
                    ),
                }
                for mode in MODES:
                    donor_label, donor_pair_id, vector = interventions[mode]
                    patched = patched_scores(
                        model,
                        current[target_side]["prefix_ids"],
                        labels,
                        layer,
                        current[target_side]["positions"][position_name],
                        vector,
                    )
                    patched_margin = margin(patched)
                    sign = 1.0 if donor_label == "SUPPORTS" else -1.0
                    change = sign * (patched_margin - clean_margin)
                    patched_prediction = prediction(patched)
                    effects.append(
                        {
                            "pair_id": pair["pair_id"],
                            "family_id": pair_family(pair),
                            "intervention_direction": f"{source_label}_to_{target_label}",
                            "source_label": source_label,
                            "target_label": target_label,
                            "target_side": target_side,
                            "layer": layer,
                            "position": position_name,
                            "target_token_index": current[target_side]["positions"][position_name],
                            "source_token_index": (
                                current[source_side]["positions"][position_name]
                                if mode == "cf_donor"
                                else current[target_side]["positions"][position_name]
                                if mode == "identity"
                                else random_states[source_side]["positions"][position_name]
                                if mode == "random_donor"
                                else random_states[target_side]["positions"][position_name]
                            ),
                            "mode": mode,
                            "donor_pair_id": donor_pair_id,
                            "donor_label": donor_label,
                            "clean_supports_logprob": float(clean_scores[0]),
                            "clean_refutes_logprob": float(clean_scores[1]),
                            "patched_supports_logprob": float(patched[0]),
                            "patched_refutes_logprob": float(patched[1]),
                            "clean_margin": clean_margin,
                            "patched_margin": patched_margin,
                            "margin_change_toward_donor": change,
                            "correct_direction": change > TIE_TOL,
                            "direction_tie": abs(change) <= TIE_TOL,
                            "clean_prediction": prediction(clean_scores),
                            "patched_prediction": patched_prediction,
                            "prediction_flipped": patched_prediction != prediction(clean_scores),
                            "flipped_to_donor": (
                                prediction(clean_scores) != donor_label
                                and patched_prediction == donor_label
                            ),
                            "target_clean_correct": prediction(clean_scores) == target_label,
                            "patch_delta_norm": float((vector.float() - target_vector.float()).norm()),
                        }
                    )
    return {
        "pair_id": pair["pair_id"],
        "family_id": pair_family(pair),
        "review_status": pair["review_status"],
        "random_donor_pair_id": random_pair["pair_id"],
        "baseline": baseline,
        "effects": effects,
    }


def result_path(run: Path, pair_id: str) -> Path:
    return run / "pair_results" / f"{stable_id(pair_id)}.json"


def run_shard(args, run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf8"))
    if not 0 <= args.shard < manifest["shards"]:
        raise ValueError("Shard index is out of range")
    pairs = read_jsonl(run / "pairs.jsonl")
    by_id = {pair["pair_id"]: pair for pair in pairs}
    subset = [
        pair for pair in pairs
        if shard_for_pair(pair["pair_id"], manifest["shards"]) == args.shard
    ]
    if not subset:
        raise ValueError("Empty shard")
    model, tokenizer = load_model(manifest["snapshot"], None, manifest["dtype"], "cuda")
    model.eval().requires_grad_(False)
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(get_transformer_blocks(model)) != 28 or model.config.hidden_size != 3584:
        raise ValueError("Expected Qwen2.5-7B with 28 blocks and hidden size 3584")
    labels = candidate_ids(tokenizer)
    output = run / "pair_results"
    output.mkdir(parents=True, exist_ok=True)
    expected_effects = 2 * len(manifest["layers"]) * len(POSITIONS) * len(MODES)
    with torch.inference_mode():
        for index, pair in enumerate(subset, 1):
            final = result_path(run, pair["pair_id"])
            if final.exists():
                saved = json.loads(final.read_text(encoding="utf8"))
                if saved.get("pair_id") == pair["pair_id"] and len(saved.get("effects", [])) == expected_effects:
                    print(f"shard={args.shard} pair={index}/{len(subset)} reused", flush=True)
                    continue
                raise ValueError(f"Incomplete existing result: {final}")
            random_pair = by_id[pair["random_donor_pair_id"]]
            result = run_pair(model, tokenizer, labels, pair, random_pair, manifest["layers"])
            dump(final, result)
            print(f"shard={args.shard} pair={index}/{len(subset)} complete", flush=True)
    dump(
        output / f"shard{args.shard}_complete.json",
        {
            "shard": args.shard,
            "n_pairs": len(subset),
            "pair_ids": [pair["pair_id"] for pair in subset],
        },
    )


def _cluster_ci(rows: list[dict], field: str, repetitions: int, seed: int) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(float(row[field]))
    pair_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(pair_ids), len(pair_ids))
        values.append(float(np.mean([
            value for index in sampled for value in grouped[pair_ids[index]]
        ])))
    return np.quantile(values, [0.025, 0.975]).astype(float).tolist()


def _summary_row(rows: list[dict], subset: str, bootstrap: int, seed: int) -> dict:
    changes = np.asarray([row["margin_change_toward_donor"] for row in rows], dtype=float)
    interval = _cluster_ci(rows, "margin_change_toward_donor", bootstrap, seed)
    return {
        "layer": rows[0]["layer"],
        "position": rows[0]["position"],
        "mode": rows[0]["mode"],
        "subset": subset,
        "n_pairs": len({row["pair_id"] for row in rows}),
        "n_directions": len(rows),
        "mean_margin_change": float(changes.mean()),
        "median_margin_change": float(np.median(changes)),
        "margin_change_ci95_low": interval[0],
        "margin_change_ci95_high": interval[1],
        "correct_direction_rate": float(np.mean([row["correct_direction"] for row in rows])),
        "direction_tie_rate": float(np.mean([row["direction_tie"] for row in rows])),
        "prediction_flip_rate": float(np.mean([row["prediction_flipped"] for row in rows])),
        "flip_to_donor_rate": float(np.mean([row["flipped_to_donor"] for row in rows])),
        "donor_prediction_rate_after": float(np.mean([
            row["patched_prediction"] == row["donor_label"] for row in rows
        ])),
        "mean_patch_delta_norm": float(np.mean([row["patch_delta_norm"] for row in rows])),
    }


def analyze(args, run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf8"))
    pairs = read_jsonl(run / "pairs.jsonl")
    expected_effects = 2 * len(manifest["layers"]) * len(POSITIONS) * len(MODES)
    results = []
    for pair in pairs:
        path = result_path(run, pair["pair_id"])
        if not path.exists():
            raise FileNotFoundError(f"Unfinished pair: {path}")
        result = json.loads(path.read_text(encoding="utf8"))
        if result.get("pair_id") != pair["pair_id"] or len(result.get("effects", [])) != expected_effects:
            raise ValueError(f"Invalid pair result: {path}")
        results.append(result)
    analysis = run / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    baselines = []
    effects = []
    for result in results:
        supports = result["baseline"]["supports"]
        refutes = result["baseline"]["refutes"]
        baselines.append({
            "pair_id": result["pair_id"],
            "family_id": result["family_id"],
            "review_status": result["review_status"],
            "supports_prediction": supports["prediction"],
            "refutes_prediction": refutes["prediction"],
            "supports_margin": supports["margin"],
            "refutes_margin": refutes["margin"],
            "pair_correct": supports["correct"] and refutes["correct"],
            "clean_direction_correct": supports["margin"] > refutes["margin"],
        })
        effects.extend(result["effects"])
    write_jsonl(analysis / "clean_baseline.jsonl", baselines)
    write_jsonl(analysis / "causal_effects.jsonl", effects)
    summary = []
    pair_summary = []
    for layer in manifest["layers"]:
        for position in POSITIONS:
            for mode in MODES:
                selected = [
                    row for row in effects
                    if row["layer"] == layer and row["position"] == position and row["mode"] == mode
                ]
                subsets = [("all", selected)]
                subsets.extend(
                    (f"direction_{direction}", [
                        row for row in selected if row["intervention_direction"] == direction
                    ])
                    for direction in ("SUPPORTS_to_REFUTES", "REFUTES_to_SUPPORTS")
                )
                subsets.append(("clean_target_correct", [
                    row for row in selected if row["target_clean_correct"]
                ]))
                for subset_name, rows in subsets:
                    if rows:
                        summary.append(_summary_row(rows, subset_name, args.bootstrap, manifest["seed"]))
                by_pair = defaultdict(list)
                for row in selected:
                    by_pair[row["pair_id"]].append(row)
                for pair_id, rows in by_pair.items():
                    if len(rows) != 2:
                        raise ValueError("Expected exactly two patch directions per pair")
                    pair_summary.append({
                        "pair_id": pair_id,
                        "layer": layer,
                        "position": position,
                        "mode": mode,
                        "mean_margin_change": float(np.mean([
                            row["margin_change_toward_donor"] for row in rows
                        ])),
                        "both_directions_correct": all(row["correct_direction"] for row in rows),
                        "any_prediction_flip": any(row["prediction_flipped"] for row in rows),
                        "both_predictions_flip": all(row["prediction_flipped"] for row in rows),
                        "both_flip_to_donor": all(row["flipped_to_donor"] for row in rows),
                    })
    with (analysis / "layer_position_summary.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (analysis / "pair_summary.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_summary[0]))
        writer.writeheader()
        writer.writerows(pair_summary)

    overall = {
        (row["layer"], row["position"], row["mode"]): row
        for row in summary if row["subset"] == "all"
    }
    candidate_rows = []
    for layer in manifest["layers"]:
        for position in POSITIONS:
            cf = overall[(layer, position, "cf_donor")]
            identity = overall[(layer, position, "identity")]
            random = overall[(layer, position, "random_donor")]
            opposite = overall[(layer, position, "opposite_direction_donor")]
            paired = [
                row for row in pair_summary
                if row["layer"] == layer and row["position"] == position and row["mode"] == "cf_donor"
            ]
            candidate_rows.append({
                "layer": layer,
                "position": position,
                "cf_mean_margin_change": cf["mean_margin_change"],
                "cf_ci95": [cf["margin_change_ci95_low"], cf["margin_change_ci95_high"]],
                "cf_correct_direction_rate": cf["correct_direction_rate"],
                "cf_prediction_flip_rate": cf["prediction_flip_rate"],
                "cf_flip_to_donor_rate": cf["flip_to_donor_rate"],
                "cf_both_directions_correct_rate": float(np.mean([
                    row["both_directions_correct"] for row in paired
                ])),
                "identity_mean_abs_effect": float(np.mean([
                    abs(row["margin_change_toward_donor"])
                    for row in effects
                    if row["layer"] == layer and row["position"] == position
                    and row["mode"] == "identity"
                ])),
                "random_donor_mean_margin_change": random["mean_margin_change"],
                "opposite_direction_mean_margin_change": opposite["mean_margin_change"],
                "identity_correct_direction_rate": identity["correct_direction_rate"],
            })
    candidate_rows.sort(
        key=lambda row: (
            row["cf_both_directions_correct_rate"],
            row["cf_correct_direction_rate"],
            row["cf_mean_margin_change"],
        ),
        reverse=True,
    )
    clean_pair_rate = float(np.mean([row["pair_correct"] for row in baselines]))
    reviewed = manifest["n_reviewed_pairs"] == manifest["n_pairs"]
    report = {
        "protocol": manifest["protocol"],
        "n_pairs": len(pairs),
        "n_reviewed_pairs": manifest["n_reviewed_pairs"],
        "clean_pair_correct_rate": clean_pair_rate,
        "clean_direction_correct_rate": float(np.mean([
            row["clean_direction_correct"] for row in baselines
        ])),
        "max_identity_abs_effect": max(row["identity_mean_abs_effect"] for row in candidate_rows),
        "reviewed_inference_eligible": reviewed,
        "ranked_layer_positions": candidate_rows,
        "interpretation": (
            "Same-layer patching establishes causal influence on this prompt and label-margin metric. "
            "Cross-sample patches can transmit Evidence content and do not alone prove an independent "
            "abstract relation variable."
        ),
        "status": (
            "reviewed dev analysis" if reviewed
            else "exploratory only: one or more source-label pairs are not human approved"
        ),
    }
    dump(analysis / "report.json", report)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fields = (
        ("cf_mean_margin_change", "Mean margin change toward donor"),
        ("cf_correct_direction_rate", "Correct-direction rate"),
        ("cf_prediction_flip_rate", "Prediction flip rate"),
    )
    for position in POSITIONS:
        rows = sorted(
            (row for row in candidate_rows if row["position"] == position),
            key=lambda row: row["layer"],
        )
        for axis, (field, ylabel) in zip(axes, fields):
            axis.plot(
                [row["layer"] for row in rows],
                [row[field] for row in rows],
                marker="o",
                label=position,
            )
            axis.set(xlabel="Block output (0-based)", ylabel=ylabel)
            axis.grid(alpha=0.25)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(analysis / "causal_layer_position_profile.png", dpi=180)
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review-queue", "prepare", "run-shard", "analyze"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-source", default=str(DEFAULT_PAIR_SOURCE))
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--review-file")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.max_pairs < 0 or args.shards < 1 or args.bootstrap < 100:
        parser.error("max-pairs/shards/bootstrap is invalid")
    run = ROOT / "outputs/experiment_runs" / args.run_id
    if args.command == "review-queue":
        print(json.dumps(make_review_queue(args, run), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(args, run), ensure_ascii=False, indent=2))
    elif args.command == "run-shard":
        run_shard(args, run)
    else:
        print(json.dumps(analyze(args, run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
