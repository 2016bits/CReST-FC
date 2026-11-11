#!/usr/bin/env python3
"""Build a self-contained, offline HTML reviewer for same-claim evidence-flip pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/experiment_runs/e2_vitaminc_review_queue_300"
TEMPLATE = Path(__file__).resolve().parent / "templates/evidence_review.html"
FIELDS = ("claim_unchanged", "evidence_changed", "supports_label_correct",
          "refutes_label_correct", "evidence_sufficient")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def text_of(row: dict) -> str:
    evidence = row["evidence"]
    return "\n".join(item["text"] for item in evidence)


def diff_segments(left: str, right: str) -> tuple[list[dict], list[dict]]:
    """Preserve whitespace while highlighting word-level replacements and insertions."""
    tokenize = lambda value: re.findall(r"\S+|\s+", value)
    a, b = tokenize(left), tokenize(right)
    left_out, right_out = [], []
    for tag, a0, a1, b0, b1 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        changed = tag != "equal"
        if a0 != a1:
            left_out.append({"text": "".join(a[a0:a1]), "changed": changed})
        if b0 != b1:
            right_out.append({"text": "".join(b[b0:b1]), "changed": changed})
    if "".join(part["text"] for part in left_out) != left:
        raise AssertionError("Left evidence changed during diff rendering")
    if "".join(part["text"] for part in right_out) != right:
        raise AssertionError("Right evidence changed during diff rendering")
    return left_out, right_out


def build_data(pairs: list[dict], queue: list[dict], source_sha256: str) -> dict:
    if not pairs or len(pairs) != len(queue):
        raise ValueError("Pair and review queue counts differ or are empty")
    queued = {item["pair_id"]: item for item in queue}
    if len(queued) != len(queue):
        raise ValueError("Duplicate review queue pair ID")
    items = []
    seen, families = set(), set()
    for pair in pairs:
        pid = pair["pair_id"]
        if pid in seen or pid not in queued:
            raise ValueError(f"Duplicate or missing queue pair: {pid}")
        seen.add(pid)
        q = queued[pid]
        support, refute = pair["supports"], pair["refutes"]
        if (support["label"] != "SUPPORTS" or refute["label"] != "REFUTES" or
                support["claim"] != refute["claim"] or pair["claim"] != support["claim"] or
                q["claim"] != pair["claim"] or q["supports_evidence"] != support["evidence"] or
                q["refutes_evidence"] != refute["evidence"]):
            raise ValueError(f"Pair/queue mismatch: {pid}")
        left, right = text_of(support), text_of(refute)
        if not left.strip() or not right.strip() or left == right:
            raise ValueError(f"Evidence invalid or identical: {pid}")
        family = support["source_provenance"]["revision_family_id"]
        if family != refute["source_provenance"]["revision_family_id"]:
            raise ValueError(f"Pair crosses revision families: {pid}")
        if family in families:
            raise ValueError(f"Repeated revision family: {family}")
        families.add(family)
        left_diff, right_diff = diff_segments(left, right)
        items.append({"pair_id": pid, "claim": pair["claim"], "source_family": family,
                      "supports_id": support["sample_id"], "refutes_id": refute["sample_id"],
                      "supports_evidence": support["evidence"],
                      "refutes_evidence": refute["evidence"],
                      "supports_segments": left_diff, "refutes_segments": right_diff,
                      "evidence_similarity": pair.get("evidence_similarity")})
    if seen != set(queued):
        raise ValueError("Review queue contains pair IDs absent from pair source")
    return {"protocol": "crestfc_evidence_review_v1", "source_sha256": source_sha256,
            "total": len(items), "fields": list(FIELDS), "pairs": items}


def write_csv(path: Path, data: dict) -> None:
    def safe(value):
        text = "" if value is None else str(value)
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("pair_id", "source_family", "claim",
                                                    "supports_evidence", "refutes_evidence",
                                                    "evidence_similarity", "reviewer", *FIELDS,
                                                    "decision", "notes"))
        writer.writeheader()
        for item in data["pairs"]:
            writer.writerow({"pair_id": safe(item["pair_id"]), "source_family": safe(item["source_family"]),
                             "claim": safe(item["claim"]),
                             "supports_evidence": safe("\n".join(x["text"] for x in item["supports_evidence"])),
                             "refutes_evidence": safe("\n".join(x["text"] for x in item["refutes_evidence"])),
                             "evidence_similarity": item["evidence_similarity"],
                             "reviewer": "", **{key: "" for key in FIELDS},
                             "decision": "", "notes": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    pair_path, queue_path = run / "pairs.jsonl", run / "review_queue.jsonl"
    source_hash = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    data = build_data(read_jsonl(pair_path), read_jsonl(queue_path), source_hash)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Script data is parsed as text, but the HTML parser still recognizes </script>.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = TEMPLATE.read_text(encoding="utf8")
    if template.count("__REVIEW_DATA_JSON__") != 1:
        raise ValueError("Review template must contain one data placeholder")
    output = (args.output or run / "review.html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REVIEW_DATA_JSON__", payload), encoding="utf8")
    csv_output = (args.csv_output or run / "review_sheet.csv").resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_output, data)
    print(json.dumps({"html": str(output), "csv": str(csv_output), "pairs": data["total"],
                      "source_sha256": source_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
