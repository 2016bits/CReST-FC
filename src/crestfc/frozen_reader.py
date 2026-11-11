"""E1-B: training-free full-sequence activation reading with a frozen Qwen decoder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .data import read_jsonl, write_jsonl
from .modeling import get_transformer_blocks, load_model, render_verdict

LABELS = ("SUPPORTS", "REFUTES")
CONDITIONS = ("full_patch", "claim_only_patch", "evidence_only_patch",
              "no_patch_slots", "random_donor_patch", "cf_donor_patch", "direct_text_7b")
ENCODER_VERSION = "e1b_ctx_v1"
DECODER_VERSION = "e1b_diag_v1"
QUESTION = ("You are given a claim and evidence through internal representations.\n"
            "Determine whether the evidence SUPPORTS or REFUTES the claim.\n"
            "Answer using exactly one label: SUPPORTS or REFUTES.")
TIE_TOL = 1e-6


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def evidence_text(row: dict) -> str:
    evidence = row["evidence"]
    return "\n".join(x["text"] for x in evidence) if isinstance(evidence, list) else str(evidence)


def family(pair: dict) -> str:
    row = pair["supports"]
    provenance = row.get("source_provenance", {})
    return str(pair.get("source_family") or provenance.get("revision_family_id") or
               row.get("group_id") or pair["claim"].casefold())


def validate_pairs(pairs: list[dict], split: str, reviews: dict[str, dict],
                   allow_unreviewed: bool) -> tuple[list[dict], dict]:
    selected, rejected = [], Counter()
    ids, sample_ids, family_splits = set(), set(), defaultdict(set)
    for pair in pairs:
        pid = pair["pair_id"]
        if pid in ids:
            raise ValueError(f"Duplicate pair ID: {pid}")
        ids.add(pid)
        a, b = pair["supports"], pair["refutes"]
        pair_split = a.get("split", pair.get("split"))
        if pair_split != b.get("split", pair.get("split")):
            raise ValueError(f"Pair crosses splits: {pid}")
        family_splits[family(pair)].add(pair_split)
        if pair_split != split:
            rejected["other_split"] += 1
            continue
        if (a["label"] != "SUPPORTS" or b["label"] != "REFUTES" or
                not a["claim"].strip() or a["claim"] != b["claim"] or
                pair["claim"] != a["claim"] or not evidence_text(a).strip() or
                not evidence_text(b).strip() or evidence_text(a) == evidence_text(b)):
            raise ValueError(f"Invalid same-claim evidence-flip pair: {pid}")
        for row in (a, b):
            if row["sample_id"] in sample_ids:
                raise ValueError(f"Repeated sample ID: {row['sample_id']}")
            sample_ids.add(row["sample_id"])
        review = reviews.get(pid, {})
        approved = bool(review.get("approved") is True and review.get("reviewer") and
                        all(review.get(key) is True for key in
                            ("claim_unchanged", "evidence_changed", "supports_label_correct",
                             "refutes_label_correct", "evidence_sufficient")))
        if not approved and not allow_unreviewed:
            rejected["unreviewed"] += 1
            continue
        selected.append({**pair, "e1b_review_status": "approved" if approved else "source_labels_unreviewed"})
    if any(len(values) > 1 for values in family_splits.values()):
        raise ValueError("Source family crosses input splits")
    if not selected:
        raise ValueError("No eligible pairs; supply reviewed pairs or explicitly allow an exploratory pilot")
    return selected, {"input_pairs": len(pairs), "selected_pairs": len(selected),
                      "reviewed_pairs": sum(p["e1b_review_status"] == "approved" for p in selected),
                      "source_families": len({family(p) for p in selected}), "rejected": dict(rejected),
                      "evidence_modes": dict(Counter(row.get("evidence_mode", "unspecified")
                                                     for p in selected for row in (p["supports"], p["refutes"])))}


def encoder_text(row: dict, condition: str) -> str:
    if condition == "full_patch":
        return f"Claim:\n{row['claim']}\n\nEvidence:\n{evidence_text(row)}\n"
    if condition == "claim_only_patch":
        return f"Claim:\n{row['claim']}\n\nEvidence:\n[not provided]\n"
    if condition == "evidence_only_patch":
        return f"Claim:\n[not provided]\n\nEvidence:\n{evidence_text(row)}\n"
    raise ValueError(condition)


def slot_id(tokenizer, slot_text: str) -> int:
    ids = tokenizer.encode(slot_text, add_special_tokens=False)
    if len(ids) != 1 or ids[0] in tokenizer.all_special_ids:
        raise ValueError(f"Slot text must be one ordinary token: {slot_text!r} -> {ids}")
    return ids[0]


def decoder_prefix(tokenizer, slots: int, token_id: int) -> list[int]:
    if slots < 1:
        raise ValueError("At least one source token is required")
    question = tokenizer.apply_chat_template([{"role": "user", "content": QUESTION}],
                                              tokenize=True, add_generation_prompt=True)
    if not question:
        raise ValueError("Empty decoder question")
    return [token_id] * slots + list(question)


@torch.inference_mode()
def capture_all(model, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    blocks = get_transformer_blocks(model)
    captured, handles = {}, []
    for index, block in enumerate(blocks):
        def capture(_module, _args, output, index=index):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[:2] != input_ids.shape or not torch.isfinite(hidden).all():
                raise ValueError(f"Invalid block {index} activation")
            captured[index] = hidden.detach().clone()
        handles.append(block.register_forward_hook(capture))
    try:
        model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(blocks):
        raise RuntimeError("Not every Transformer block produced an activation")
    return captured


@contextmanager
def patch_block(model, layer: int, source: torch.Tensor):
    """Replace exactly the first T decoder Block-output states, including on both candidate rows."""
    if source.ndim != 3 or not torch.isfinite(source).all():
        raise ValueError("Source must be finite [B,T,D] activations")
    count = 0
    def hook(_module, _args, output):
        nonlocal count
        count += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if (source.shape[0] not in {1, hidden.shape[0]} or source.shape[1] > hidden.shape[1]
                or source.shape[2] != hidden.shape[2]):
            raise ValueError("Source/decoder patch dimensions disagree")
        patched = hidden.clone()
        patched[:, :source.shape[1], :] = source.to(hidden.device, hidden.dtype).expand(
            hidden.shape[0], -1, -1)
        return (patched, *output[1:]) if isinstance(output, tuple) else patched
    handle = get_transformer_blocks(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        if count == 0:
            raise RuntimeError("Decoder patch hook was never called")


def candidate_ids(tokenizer) -> list[list[int]]:
    candidates = [tokenizer.encode(" " + label, add_special_tokens=False) for label in LABELS]
    if any(not row for row in candidates) or candidates[0] == candidates[1]:
        raise ValueError("Invalid candidate continuation tokens")
    return candidates


@torch.inference_mode()
def score_labels(model, prefix_ids: list[int], candidates: list[list[int]],
                 source: torch.Tensor | None = None, write_layer: int = 0) -> tuple[list[float], list[float]]:
    if not prefix_ids or any(not item for item in candidates):
        raise ValueError("Empty prefix or label continuation")
    scores = []
    for answer in candidates:
        ids = torch.tensor([prefix_ids + answer], dtype=torch.long, device=model.device)
        if ids.shape[1] > model.config.max_position_embeddings:
            raise ValueError("Decoder input exceeds model context")
        if source is None:
            hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 use_cache=False, return_dict=True).last_hidden_state
        else:
            if source.shape[1] > len(prefix_ids):
                raise ValueError("Source activation exceeds decoder slot prefix")
            with patch_block(model, write_layer, source):
                hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     use_cache=False, return_dict=True).last_hidden_state
        logits = model.lm_head(hidden[:, len(prefix_ids)-1:len(prefix_ids)+len(answer)-1]).float()
        logp = logits.log_softmax(dim=-1)
        targets = torch.tensor(answer, dtype=torch.long, device=logp.device)
        scores.append(float(logp[0, torch.arange(len(answer), device=logp.device), targets].sum()))
    return scores, [value / len(answer) for value, answer in zip(scores, candidates)]


@torch.inference_mode()
def score_layer_batch(model, prefix_ids: list[int], candidates: list[list[int]],
                      layer_sources: torch.Tensor, write_layer: int = 0) -> list[list[float]]:
    """Score several read layers in one decoder forward; results match score_labels."""
    if (layer_sources.ndim != 3 or not len(prefix_ids) or len(candidates) != 2
            or any(not c for c in candidates) or layer_sources.shape[0] < 1
            or layer_sources.shape[1] > len(prefix_ids)):
        raise ValueError("Invalid batched read layers or label continuations")
    n, prefix_len = layer_sources.shape[0], len(prefix_ids)
    suffix_len = max(map(len, candidates))
    ids = torch.full((n * 2, prefix_len + suffix_len), model.config.pad_token_id or 0,
                     dtype=torch.long, device=model.device)
    mask = torch.zeros_like(ids)
    for layer_index in range(n):
        for answer_index, answer in enumerate(candidates):
            row_index = 2 * layer_index + answer_index
            content = prefix_ids + answer
            ids[row_index, :len(content)] = torch.tensor(content, device=model.device)
            mask[row_index, :len(content)] = 1
    if ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Decoder input exceeds model context")
    patched_sources = layer_sources.repeat_interleave(2, dim=0)
    with patch_block(model, write_layer, patched_sources):
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                             return_dict=True).last_hidden_state
    logits = model.lm_head(hidden[:, prefix_len - 1:prefix_len + suffix_len - 1]).float()
    logp = logits.log_softmax(dim=-1)
    scores = []
    for layer_index in range(n):
        pair = []
        for answer_index, answer in enumerate(candidates):
            row_index = 2 * layer_index + answer_index
            token_indices = torch.tensor(answer, dtype=torch.long, device=logp.device)
            positions = torch.arange(len(answer), device=logp.device)
            pair.append(float(logp[row_index, positions, token_indices].sum()))
        scores.append(pair)
    return scores


def classify(scores: list[float]) -> tuple[str, float]:
    margin = scores[0] - scores[1]
    return ("TIE" if abs(margin) <= TIE_TOL else LABELS[0] if margin > 0 else LABELS[1]), margin


def review_map(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    result = {row["pair_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate review IDs")
    return result


def prepare(args) -> tuple[Path, list[dict], dict]:
    root = Path(__file__).resolve().parents[2]
    run = (root / "outputs/experiment_runs" / args.run_id).resolve()
    source = Path(args.pair_source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.phase == "test" and (args.allow_unreviewed or not args.locked_dev_selection):
        raise ValueError("Test requires reviewed pairs and a locked dev selection")
    pairs, audit = validate_pairs(read_jsonl(source), args.phase,
                                  review_map(Path(args.review_file).resolve() if args.review_file else None),
                                  args.allow_unreviewed)
    pairs.sort(key=lambda p: hashlib.sha256(f"{args.seed}:{p['pair_id']}".encode()).hexdigest())
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if len(pairs) < 2:
        raise ValueError("At least two pairs are needed for independent random-donor control")
    if len({family(p) for p in pairs}) < 2:
        raise ValueError("Random donor needs two independent source families")
    audit["selected_after_limit"] = len(pairs)
    audit["reviewed_after_limit"] = sum(p["e1b_review_status"] == "approved" for p in pairs)
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(snapshot / "config.json")
    if args.phase == "test":
        locked = json.loads(Path(args.locked_dev_selection).read_text())
        if (not locked.get("eligible") or not locked.get("selected_layers") or
                locked.get("model_snapshot") != str(snapshot)):
            raise ValueError("Locked dev selection is ineligible or uses a different model")
        if sorted(args.layers) != sorted(locked["scanned_layers"]):
            raise ValueError("Test layers differ from locked dev scan")
        dev_run = Path(args.locked_dev_selection).resolve().parent
        dev_manifest = json.loads((dev_run / "manifest.json").read_text())
        dev_pairs = read_jsonl(dev_run / "selected_pairs.jsonl")
        if (dev_manifest["phase"] != "dev" or dev_manifest["model_snapshot"] != str(snapshot)
                or dev_manifest["encoder_prompt_version"] != ENCODER_VERSION
                or dev_manifest["decoder_prompt_version"] != DECODER_VERSION):
            raise ValueError("Locked dev protocol differs from test")
        dev_claims = {p["claim"].casefold().strip() for p in dev_pairs}
        dev_families = {family(p) for p in dev_pairs}
        if (dev_claims & {p["claim"].casefold().strip() for p in pairs} or
                dev_families & {family(p) for p in pairs}):
            raise ValueError("Dev/test Claim or source family overlaps")
    manifest = {"protocol": "e1b_frozen_reader_v1", "phase": args.phase,
                "pair_source": str(source), "pair_source_sha256": digest(source),
                "review_file": str(Path(args.review_file).resolve()) if args.review_file else None,
                "review_sha256": digest(Path(args.review_file)) if args.review_file else None,
                "locked_dev_selection": str(Path(args.locked_dev_selection).resolve()) if args.locked_dev_selection else None,
                "model_id": "Qwen/Qwen2.5-7B-Instruct", "model_snapshot": str(snapshot),
                "model_config_sha256": digest(snapshot / "config.json"),
                "encoder_prompt_version": ENCODER_VERSION, "decoder_prompt_version": DECODER_VERSION,
                "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
                "read_boundary": "block_output_full_sequence", "write_boundary": "block0_output",
                "labels": list(LABELS), "no_training": True, "tie_tolerance": TIE_TOL,
                "layers": args.layers, "seed": args.seed, "max_pairs": args.max_pairs,
                "allow_unreviewed": args.allow_unreviewed,
                "code_sha256": digest(Path(__file__))}
    if (run / "manifest.json").exists():
        if json.loads((run / "manifest.json").read_text()) != manifest:
            raise ValueError("Run manifest changed; use another run ID")
    else:
        run.mkdir(parents=True, exist_ok=True)
        dump(run / "manifest.json", manifest)
        dump(run / "data_audit.json", audit)
        write_jsonl(run / "selected_pairs.jsonl", pairs)
    return run, pairs, audit


def choose_random_donor(pairs: list[dict], index: int, target_len: int,
                        lengths: dict[str, int]) -> dict:
    candidates = [p for j, p in enumerate(pairs) if j != index and family(p) != family(pairs[index])]
    if not candidates:
        raise ValueError("No independent random donor")
    return min(candidates, key=lambda p: (abs(lengths[p["pair_id"]] - target_len), p["pair_id"]))


def make_record(pair: dict, row: dict, condition: str, layer: int | None,
                donor: dict | None, scores: list[float], means: list[float],
                source_tokens: int, prefix_tokens: int) -> dict:
    prediction, margin = classify(scores)
    return {"pair_id": pair["pair_id"], "sample_id": row["sample_id"], "source_family": family(pair),
            "split": row["split"], "review_status": pair["e1b_review_status"],
            "evidence_mode": row.get("evidence_mode", "unspecified"),
            "condition": condition, "read_layer": layer, "write_layer": 0 if layer is not None else None,
            "donor_id": donor["sample_id"] if donor else None,
            "donor_gold": donor["label"] if donor else None,
            "num_source_tokens": source_tokens, "num_slots": source_tokens if layer is not None or condition == "no_patch_slots" else 0,
            "prefix_tokens": prefix_tokens, "score_supports": scores[0], "score_refutes": scores[1],
            "mean_score_supports": means[0], "mean_score_refutes": means[1],
            "margin": margin, "predicted_label": prediction, "gold_label": row["label"],
            "prompt_version": DECODER_VERSION if condition != "direct_text_7b" else "v4_fever_binary_supports_refutes"}


def scan(args, run: Path, pairs: list[dict]) -> None:
    model, tokenizer = load_model(str(Path(args.snapshot).resolve()), None, "bfloat16", "cuda")
    model.eval().requires_grad_(False)
    blocks = get_transformer_blocks(model)
    if len(blocks) != 28 or model.config.hidden_size != 3584:
        raise ValueError("Expected Qwen2.5-7B with 28 blocks and hidden size 3584")
    sid, candidates = slot_id(tokenizer, args.slot_text), candidate_ids(tokenizer)
    lengths = {}
    for p in pairs:
        lengths[p["pair_id"]] = len(tokenizer.encode(encoder_text(p["supports"], "full_patch"),
                                                 add_special_tokens=False))
    dump(run / "token_protocol.json", {"slot_text": args.slot_text, "slot_id": sid,
                                      "candidate_token_ids": dict(zip(LABELS, candidates)),
                                      "question": QUESTION, "question_token_count":
                                      len(decoder_prefix(tokenizer, 1, sid))-1})
    sanity_path = run / "sanity_checks.json"
    if not sanity_path.exists():
        first = pairs[0]["supports"]
        source_ids = tokenizer(encoder_text(first, "full_patch"), add_special_tokens=False,
                               return_tensors="pt")["input_ids"].to(model.device)
        source_states = capture_all(model, source_ids)
        prefix = decoder_prefix(tokenizer, source_ids.shape[1], sid)
        decoder_states = capture_all(model, torch.tensor([prefix], device=model.device))
        clean, _ = score_labels(model, prefix, candidates)
        identity, _ = score_labels(model, prefix, candidates, decoder_states[0])
        repeated, _ = score_labels(model, prefix, candidates)
        difference = max(abs(a-b) for a, b in zip(clean, identity))
        repeat_difference = max(abs(a-b) for a, b in zip(clean, repeated))
        checks = {"frozen_parameters": sum(p.requires_grad for p in model.parameters()),
                  "captured_blocks": len(source_states), "source_tokens": source_ids.shape[1],
                  "source_shape": list(source_states[0].shape),
                  "identity_max_score_difference": difference,
                  "repeat_max_score_difference": repeat_difference,
                  "decoder_hooks_after": sum(len(b._forward_hooks) for b in blocks)}
        dump(sanity_path, checks)
        if (checks["frozen_parameters"] or checks["captured_blocks"] != 28 or
                difference > 1e-3 or repeat_difference > 1e-3 or checks["decoder_hooks_after"]):
            raise RuntimeError(f"E1-B sanity check failed: {checks}")
    out = run / "pair_results"
    out.mkdir(exist_ok=True)
    for index, pair in enumerate(pairs):
        target = out / (hashlib.sha256(pair["pair_id"].encode()).hexdigest()[:24] + ".json")
        if target.is_file():
            saved = json.loads(target.read_text())
            if saved.get("pair_id") == pair["pair_id"] and saved.get("layers") == args.layers:
                print(f"pair {index+1}/{len(pairs)} reused", flush=True)
                continue
            raise ValueError("Existing pair result conflicts with run protocol")
        rows = (pair["supports"], pair["refutes"])
        captured = {}
        for condition in ("full_patch", "claim_only_patch", "evidence_only_patch"):
            for row in rows:
                source = encoder_text(row, condition)
                ids = tokenizer(source, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
                if ids.shape[1] > model.config.max_position_embeddings:
                    raise ValueError(f"Encoder exceeds context: {row['sample_id']}")
                captured[(condition, row["sample_id"])] = capture_all(model, ids)
        # Use one unrelated donor for both members of the target pair. Its label is
        # chosen by seeded pair order, independently of either target gold label.
        target_length = round(sum(captured[("full_patch", row["sample_id"])][0].shape[1]
                                  for row in rows) / len(rows))
        random_pair = choose_random_donor(pairs, index, target_length, lengths)
        random_row = random_pair["supports"] if index % 2 == 0 else random_pair["refutes"]
        random_ids = tokenizer(encoder_text(random_row, "full_patch"), add_special_tokens=False,
                               return_tensors="pt")["input_ids"].to(model.device)
        random_states = capture_all(model, random_ids)
        random_prefix = decoder_prefix(tokenizer, random_ids.shape[1], sid)
        records = []
        for row in rows:
            own = captured[("full_patch", row["sample_id"])]
            other = rows[1] if row is rows[0] else rows[0]
            num_slots = own[0].shape[1]
            prefix = decoder_prefix(tokenizer, num_slots, sid)
            decoded = tokenizer.decode(prefix, skip_special_tokens=False)
            if row["claim"] in decoded or evidence_text(row) in decoded or row["sample_id"] in decoded:
                raise ValueError("Claim, evidence or sample ID leaked into decoder text")
            no_scores, no_means = score_labels(model, prefix, candidates)
            records.append(make_record(pair, row, "no_patch_slots", None, None,
                                       no_scores, no_means, num_slots, len(prefix)))
            direct_prompt = render_verdict(row, scheme="fever_binary")
            direct_ids = tokenizer.encode(direct_prompt, add_special_tokens=False)
            direct_scores, direct_means = score_labels(model, direct_ids, candidates)
            records.append(make_record(pair, row, "direct_text_7b", None, None,
                                       direct_scores, direct_means, 0, len(direct_ids)))
            for condition in ("full_patch", "claim_only_patch", "evidence_only_patch"):
                states = captured[(condition, row["sample_id"])]
                t = states[0].shape[1]
                condition_prefix = decoder_prefix(tokenizer, t, sid)
                for layer in args.layers:
                    scores, means = score_labels(model, condition_prefix, candidates, states[layer])
                    records.append(make_record(pair, row, condition, layer, row,
                                               scores, means, t, len(condition_prefix)))
            for layer in args.layers:
                for condition, donor, states, pref in (
                    ("random_donor_patch", random_row, random_states, random_prefix),
                    ("cf_donor_patch", other, captured[("full_patch", other["sample_id"])],
                     decoder_prefix(tokenizer, captured[("full_patch", other["sample_id"])][0].shape[1], sid))):
                    scores, means = score_labels(model, pref, candidates, states[layer])
                    records.append(make_record(pair, row, condition, layer, donor,
                                               scores, means, states[layer].shape[1], len(pref)))
        pending = target.with_suffix(".tmp")
        dump(pending, {"pair_id": pair["pair_id"], "layers": args.layers, "records": records})
        pending.replace(target)
        print(f"pair {index+1}/{len(pairs)} complete ({len(records)} scores)", flush=True)


def _ci(values: np.ndarray, groups: list[str], reps: int, seed: int) -> list[float]:
    unique = sorted(set(groups))
    by = {g: values[np.asarray(groups) == g] for g in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        chosen = rng.integers(0, len(unique), len(unique))
        parts = [by[unique[i]] for i in chosen]
        draws.append(float(np.concatenate(parts).mean()))
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def analyze(args, run: Path, pairs: list[dict], audit: dict) -> dict:
    records = []
    for pair in pairs:
        path = run / "pair_results" / (hashlib.sha256(pair["pair_id"].encode()).hexdigest()[:24] + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete scan: {path}")
        result = json.loads(path.read_text())
        if result["pair_id"] != pair["pair_id"] or result["layers"] != args.layers:
            raise ValueError("Saved pair results disagree with manifest")
        records.extend(result["records"])
    write_jsonl(run / "predictions.jsonl", records)
    by_key = defaultdict(list)
    for record in records:
        by_key[(record["condition"], record["read_layer"])].append(record)
    metrics, pair_rows, ci_rows = [], [], []
    for (condition, layer), rows in sorted(by_key.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])):
        truth = [r["gold_label"] for r in rows]
        pred = [r["predicted_label"] for r in rows]
        each_pair = defaultdict(dict)
        for row in rows:
            each_pair[row["pair_id"]][row["gold_label"]] = row
        directions, correct, gaps, groups = [], [], [], []
        for pair in pairs:
            s, r = (each_pair[pair["pair_id"]][label] for label in LABELS)
            gap = s["margin"] - r["margin"]
            direction = 0.5 if abs(gap) <= TIE_TOL else float(gap > 0)
            good = float(s["margin"] > TIE_TOL and r["margin"] < -TIE_TOL)
            directions.append(direction); correct.append(good); gaps.append(gap); groups.append(family(pair))
            pair_rows.append({"pair_id": pair["pair_id"], "source_family": family(pair),
                              "condition": condition, "read_layer": layer,
                              "m_plus": s["margin"], "m_minus": r["margin"],
                              "direction": direction, "pair_correct": good})
        item = {"condition": condition, "read_layer": layer, "n_samples": len(rows), "n_pairs": len(pairs),
                "n_families": len(set(groups)), "accuracy": float(accuracy_score(truth, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)),
                "supports_recall": sum(y == p == "SUPPORTS" for y, p in zip(truth, pred)) / truth.count("SUPPORTS"),
                "refutes_recall": sum(y == p == "REFUTES" for y, p in zip(truth, pred)) / truth.count("REFUTES"),
                "tie_rate": float(np.mean([p == "TIE" for p in pred])),
                "direction": float(np.mean(directions)), "pair_correct": float(np.mean(correct)),
                "direction_tie_rate": float(np.mean([abs(g) <= TIE_TOL for g in gaps])),
                "mean_pair_gap": float(np.mean(gaps)), "median_pair_gap": float(np.median(gaps)),
                "mean_margin": float(np.mean([r["margin"] for r in rows]))}
        metrics.append(item)
        for measure, values in (("direction", directions), ("pair_correct", correct)):
            ci_rows.append({"condition": condition, "read_layer": layer, "measure": measure,
                            "mean": float(np.mean(values)), "ci95": _ci(np.asarray(values), groups,
                                                                            args.bootstrap_reps, args.seed)})
    for name, data in (("metrics_by_layer.csv", metrics), ("pairs.csv", pair_rows),
                       ("bootstrap_ci.csv", ci_rows)):
        with (run / name).open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    for field in ("macro_f1", "direction", "pair_correct"):
        fig, ax = plt.subplots(figsize=(9, 4))
        for condition in ("full_patch", "claim_only_patch", "evidence_only_patch",
                          "random_donor_patch", "cf_donor_patch"):
            curve = sorted((r for r in metrics if r["condition"] == condition and r["read_layer"] is not None),
                           key=lambda r: r["read_layer"])
            if curve:
                ax.plot([r["read_layer"] for r in curve], [r[field] for r in curve],
                        marker=".", label=condition)
        ax.set(xlabel="Encoder read layer (zero-based)", ylabel=field, ylim=(-0.03, 1.03))
        ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(figures / f"{field}_by_layer.png", dpi=180)
        plt.close(fig)
    eligible = (args.phase == "dev" and not args.allow_unreviewed and
                audit["reviewed_after_limit"] == len(pairs) and len(pairs) >= 200 and
                args.layers == list(range(28)))
    selected = []
    if eligible:
        by = {(r["condition"], r["read_layer"]): r for r in metrics}
        pair_values = {(r["pair_id"], r["condition"], r["read_layer"]): r["pair_correct"]
                       for r in pair_rows}
        for layer in args.layers:
            full = by[("full_patch", layer)]
            if full["direction"] <= .5:
                continue
            controls_pass = True
            for ctrl in ("claim_only_patch", "random_donor_patch", "no_patch_slots"):
                ctrl_layer = None if ctrl == "no_patch_slots" else layer
                differences = np.asarray([
                    pair_values[(p["pair_id"], "full_patch", layer)] -
                    pair_values[(p["pair_id"], ctrl, ctrl_layer)] for p in pairs])
                if _ci(differences, [family(p) for p in pairs], args.bootstrap_reps,
                       args.seed + layer)[0] <= 0:
                    controls_pass = False
                    break
            if controls_pass:
                selected.append(layer)
    selection = {"phase": args.phase, "eligible": eligible, "selected_layers": selected,
                 "scanned_layers": args.layers, "model_snapshot": str(Path(args.snapshot).resolve()),
                 "rule": "dev-only: full direction > 0.5 and cluster-bootstrap 95% lower bound of paired PairCorrect gap > 0 vs claim/random/no-patch",
                 "n_pairs": len(pairs), "reviewed_pairs": audit["reviewed_after_limit"]}
    dump(run / "layer_selection.json", selection)
    return {"n_predictions": len(records), "n_pairs": len(pairs), "metrics": str(run / "metrics_by_layer.csv"),
            "selection_eligible": eligible}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "scan", "analyze", "all"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-source", required=True)
    parser.add_argument("--snapshot", default="/data/yangjun/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75")
    parser.add_argument("--phase", choices=["dev", "test"], default="dev")
    parser.add_argument("--review-file")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--locked-dev-selection")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--slot-text", default="?")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    args.layers = list(range(28)) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    if (not args.layers or len(set(args.layers)) != len(args.layers) or
            any(v < 0 or v >= 28 for v in args.layers) or args.max_pairs < 0 or args.bootstrap_reps < 100):
        parser.error("Use unique layers in 0..27, nonnegative max pairs, and at least 100 bootstrap replicates")
    args.layers.sort()
    run, pairs, audit = prepare(args)
    if args.stage in {"scan", "all"}:
        scan(args, run, pairs)
    if args.stage in {"analyze", "all"}:
        print(json.dumps(analyze(args, run, pairs, audit), ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps({"run": str(run), "audit": audit}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
