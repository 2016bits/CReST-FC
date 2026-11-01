"""Classification, pair metrics and group-clustered uncertainty estimates."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from .data import approved_pairs


def parse_structured_output(text: str, allowed_ids: set[str], labels: list[str]) -> dict:
    """Strictly parse generated JSON; preserve invalid outputs as INVALID."""
    try:
        if not text.lstrip().startswith("{"):
            raise ValueError("Output must begin with a JSON object")
        start = text.index("{")
        obj, end = json.JSONDecoder().raw_decode(text[start:])
        if text[start + end:].strip() or not isinstance(obj, dict):
            raise ValueError("Trailing text or non-object JSON")
        verdict = obj["verdict"]
        ids = obj["evidence_ids"]
        justification = obj["justification"]
        if verdict not in labels or not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError("Invalid verdict or evidence list")
        if len(ids) != len(set(ids)) or any(x not in allowed_ids for x in ids):
            raise ValueError("Invented or duplicate evidence ID")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError("Missing justification")
        return {"verdict": verdict, "evidence_ids": ids,
                "justification": justification, "valid": True, "error": None}
    except (ValueError, KeyError, TypeError) as exc:
        return {"verdict": "INVALID", "evidence_ids": [], "justification": "",
                "valid": False, "error": str(exc)}


def evidence_prf(gold_ids: set[str], predicted_ids: set[str]) -> dict:
    tp = len(gold_ids & predicted_ids)
    precision = tp / len(predicted_ids) if predicted_ids else (1.0 if not gold_ids else 0.0)
    recall = tp / len(gold_ids) if gold_ids else (1.0 if not predicted_ids else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(rows: list[dict], predictions: dict[str, str], labels: list[str]) -> dict:
    y = [row["label"] for row in rows]
    pred = [predictions.get(row["sample_id"], "INVALID") for row in rows]
    valid = [p in labels for p in pred]
    return {
        "n_examples": len(rows),
        "accuracy": float(accuracy_score(y, pred)) if y else None,
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)) if y else None,
        "balanced_accuracy": float(np.mean([sum(p == label for truth, p in zip(y, pred) if truth == label)
                                             / sum(truth == label for truth in y)
                                             for label in labels if label in y])) if y else None,
        "invalid_rate": 1 - float(np.mean(valid)) if y else None,
        "per_class_f1": {label: float(v) for label, v in zip(labels, f1_score(y, pred, labels=labels,
                                                                               average=None, zero_division=0))},
        "confusion_labels": labels + ["INVALID"],
        "confusion_matrix": confusion_matrix(y, pred, labels=labels + ["INVALID"]).tolist(),
    }


def pair_metrics(rows: list[dict], pairs: list[dict], predictions: dict[str, str], split: str) -> dict:
    selected = approved_pairs(rows, pairs, split)
    out = {"n_bias_pairs": 0, "n_relation_pairs": 0, "bias_consistency": None,
           "bias_pair_correct": None, "relation_pair_correct": None, "raw_relation_flip_rate": None}
    bias, relation = [], []
    for a, b, pair in selected:
        if a["sample_id"] not in predictions or b["sample_id"] not in predictions:
            continue
        pa, pb = predictions[a["sample_id"]], predictions[b["sample_id"]]
        if pair["kind"] == "bias_change":
            bias.append((pa == pb, pa == a["label"] and pb == b["label"]))
        else:
            relation.append((pa == a["label"] and pb == b["label"], pa != pb))
    out["n_bias_pairs"], out["n_relation_pairs"] = len(bias), len(relation)
    if bias:
        out["bias_consistency"] = float(np.mean([x[0] for x in bias]))
        out["bias_pair_correct"] = float(np.mean([x[1] for x in bias]))
    if relation:
        out["relation_pair_correct"] = float(np.mean([x[0] for x in relation]))
        out["raw_relation_flip_rate"] = float(np.mean([x[1] for x in relation]))
    return out


def clustered_bootstrap(rows: list[dict], values: dict[str, float], n_boot: int = 1000,
                        seed: int = 42) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row["sample_id"] in values:
            groups[row["group_id"]].append(values[row["sample_id"]])
    if not groups:
        return {"mean": None, "ci95": None, "n_groups": 0}
    group_vals = [np.asarray(v, dtype=float) for v in groups.values()]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(group_vals), len(group_vals))
        sample = [group_vals[i] for i in idx]
        means.append(float(sum(x.sum() for x in sample) / sum(len(x) for x in sample)))
    return {"mean": float(np.mean(list(values.values()))),
            "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "n_groups": len(group_vals)}
