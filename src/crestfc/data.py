"""Source adapters, pair construction and leakage checks.

Original dataset files remain untouched. Automatic counterfactuals are candidates
until reviewed; the audit gate is enforced again by training and evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

LABELS = {"SUPPORTS", "REFUTES", "NEI", "CONFLICTING_EVIDENCE"}
LABEL_MAP = {
    "SUPPORTS": "SUPPORTS", "SUPPORTED": "SUPPORTS", "REFUTES": "REFUTES",
    "REFUTED": "REFUTES", "NEI": "NEI", "NOT ENOUGH INFO": "NEI",
    "NOT_ENOUGH_INFO": "NEI", "CONFLICTING EVIDENCE": "CONFLICTING_EVIDENCE",
    "CONFLICTING_EVIDENCE": "CONFLICTING_EVIDENCE",
}


def normalize_label(value: str) -> str:
    key = str(value).strip().upper().replace("-", "_")
    if key not in LABEL_MAP:
        raise ValueError(f"Unknown or ambiguous label {value!r}; do not coerce to REFUTES/NEI")
    return LABEL_MAP[key]


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]


@dataclass
class Example:
    sample_id: str
    group_id: str
    source_dataset: str
    source_id: str
    split: str
    claim: str
    evidence: list[dict[str, Any]]
    label: str
    label_space: str = "fever3"
    evidence_mode: str = "oracle"
    pair_id: str | None = None
    pair_kind: str = "none"
    pair_role: str | None = None
    claim_metadata: dict[str, Any] = field(default_factory=dict)
    edited_span: dict[str, Any] | None = None
    counterfactual_verified: bool = False
    verification_method: str | None = None
    gold_justification: str | None = None
    source_provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        assert self.sample_id and self.group_id and self.claim.strip()
        assert self.split in {"train", "dev", "test"}
        assert self.label in LABELS
        assert self.label_space in {"fever3", "averitec4"}
        assert self.label != "CONFLICTING_EVIDENCE" or self.label_space == "averitec4"
        assert self.evidence_mode in {"oracle", "retrieved", "insufficient"}
        assert self.evidence or self.evidence_mode == "insufficient"
        assert self.pair_kind in {"none", "bias_change", "relation_change"}
        assert (self.pair_id is None) == (self.pair_kind == "none")
        if self.label == "NEI":
            assert self.evidence_mode == "insufficient", "NEI requires audited insufficient evidence"
        for e in self.evidence:
            assert e.get("evidence_id") and isinstance(e.get("text"), str) and e["text"].strip()
        assert self.source_provenance, "Source revision/path required"


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _records(path: Path) -> list[dict]:
    if path.suffix == ".tsv":
        with path.open(encoding="utf8") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array in {path}")
    return value


def _evidence(row: dict, dataset: str) -> list[dict]:
    if dataset == "HOVER":
        items = row.get("gold_evidence_list") or []
        return [{"evidence_id": f"E{i+1}", "text": x["sentences"],
                 "doc_id": x.get("title"), "sent_id": str(x.get("index")), "is_gold": True}
                for i, x in enumerate(items) if x.get("sentences")]
    if dataset == "PolitiHop":
        if isinstance(row.get("evidence"), str) and row["evidence"].strip():
            return [{"evidence_id": "E1", "text": row["evidence"], "is_gold": False}]
        raw = row.get("annotated_evidence") or row.get("relevant_text_url_sentences")
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = []
        if isinstance(raw, dict):
            raw = list(raw.values())
        if not isinstance(raw, list):
            raw = []
        return [{"evidence_id": f"E{i+1}", "text": str(x), "is_gold": True}
                for i, x in enumerate(raw) if isinstance(x, str) and x.strip()]
    raw = row.get("gold_evidence") or row.get("evidence")
    if isinstance(raw, str) and raw.strip():
        return [{"evidence_id": "E1", "text": raw, "is_gold": True}]
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return [{"evidence_id": f"E{i+1}", "text": x, "is_gold": True}
                for i, x in enumerate(raw) if x.strip()]
    if raw is not None:
        raise ValueError("Indexed evidence needs a resolved corpus; do not treat indices as text")
    return []


def adapt(path: str | Path, dataset: str, split: str, limit: int | None = None,
          wiki_db: str | None = None) -> tuple[list[Example], dict]:
    path = Path(path)
    dataset = dataset.upper() if dataset not in {"VitaminC", "PolitiHop"} else dataset
    if split not in {"train", "dev", "test"}:
        raise ValueError(split)
    rows = _records(path)
    conn = None
    if dataset == "FEVER" and wiki_db:
        import sqlite3
        conn = sqlite3.connect(wiki_db)
    examples = []
    source_hash = file_sha256(path)
    rejected = Counter()
    for i, row in enumerate(rows):
        try:
            # Only EGR-FV's explicitly binarized converted PolitiHop records are
            # compatible with the FEVER label space. Raw ratings remain excluded.
            if dataset == "PolitiHop" and (path.parent.name != "converted_data" or
                                             str(row.get("label", "")).lower() not in {"supports", "refutes"} or
                                             not isinstance(row.get("evidence"), str)):
                rejected["politihop_label_requires_mapping"] += 1
                continue
            label = normalize_label(row["label"])
            if label == "NEI":
                rejected["nei_requires_insufficient_evidence_audit"] += 1
                continue
            evidence = (_resolve_fever_evidence(row.get("evidence"), conn)
                        if dataset == "FEVER" and isinstance(row.get("evidence"), list)
                        and row["evidence"] and isinstance(row["evidence"][0], list)
                        else _evidence(row, dataset))
            if not evidence:
                rejected["missing_resolved_evidence"] += 1
                continue
            claim = row["claim"].strip()
            if not claim:
                rejected["empty_claim"] += 1
                continue
            source_id = str(row.get("id", row.get("uid", row.get("index", i))))
            group_id = f"{dataset}:{stable_id(claim.lower())}"
            family_id = source_id.rsplit("_", 1)[0] if dataset == "VitaminC" else source_id
            converted_retrieval = (dataset in {"FEVER", "PolitiHop"} and
                                   path.parent.name == "converted_data" and
                                   isinstance(row.get("evidence"), str))
            if converted_retrieval:
                for item in evidence:
                    item["is_gold"] = False
            ex = Example(sample_id=f"{dataset}:{split}:{source_id}", group_id=group_id,
                         source_dataset=dataset, source_id=source_id, split=split,
                         claim=claim, evidence=evidence, label=label,
                         evidence_mode="retrieved" if converted_retrieval else "oracle",
                         gold_justification=row.get("gold_justification"),
                         source_provenance={"source_file": str(path.resolve()), "source_row": i,
                                            "source_sha256": source_hash,
                                            "revision_family_id": f"{dataset}:{family_id}"})
            ex.validate()
            examples.append(ex)
            if limit and len(examples) >= limit:
                break
        except (ValueError, KeyError, AssertionError, TypeError) as exc:
            rejected[f"invalid:{type(exc).__name__}"] += 1
    if conn is not None:
        conn.close()
    return examples, {"source_rows": len(rows), "accepted": len(examples), "rejected": dict(rejected)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_pairs(examples: list[Example], max_relation_pairs: int = 0,
               min_similarity: float = 0.7, add_bias_pairs: bool = True, seed: int = 42) -> tuple[list[Example], list[dict]]:
    """Build reviewable candidates; no automatic pair is human-verified."""
    by_claim: dict[tuple[str, str], list[Example]] = defaultdict(list)
    out = list(examples)
    pairs: list[dict] = []
    for x in examples:
        by_claim[(x.split, x.group_id)].append(x)
        if add_bias_pairs:
            a, b = _copy(x), _copy(x)
            pair_id = f"bias:{x.sample_id}"
            for role, item, publisher in [("A", a, "Publisher A"), ("B", b, "Publisher B")]:
                item.sample_id = f"{x.sample_id}:bias:{role}"
                item.pair_id, item.pair_kind, item.pair_role = pair_id, "bias_change", role
                item.claim_metadata = {"claim_publisher": publisher}
                item.counterfactual_verified = False
                item.verification_method = "pending_human_review"
            out.extend([a, b])
            pairs.append({"pair_id": pair_id, "sample_a": a.sample_id, "sample_b": b.sample_id,
                          "kind": "bias_change", "changed_fields": ["claim_metadata.claim_publisher"],
                          "audit_status": "pending"})
    count = 0
    if max_relation_pairs < 0:
        raise ValueError("max_relation_pairs must be nonnegative")
    keys = sorted(by_claim)
    random.Random(seed).shuffle(keys)
    for key in keys:
        group = by_claim[key]
        if count >= max_relation_pairs:
            break
        candidates = [(a, b) for a in group for b in group if a.label != b.label]
        if not candidates:
            continue
        best = max(candidates, key=lambda ab: SequenceMatcher(None, ab[0].evidence[0]["text"],
                                                                 ab[1].evidence[0]["text"]).ratio())
        a, b = best
        sim = SequenceMatcher(None, a.evidence[0]["text"], b.evidence[0]["text"]).ratio()
        if sim < min_similarity:
            continue
        old_text, new_text = a.evidence[0]["text"], b.evidence[0]["text"]
        edits = [{"a_start": i, "a_end": j, "b_start": k, "b_end": l,
                  "a_text": old_text[i:j], "b_text": new_text[k:l]}
                 for op, i, j, k, l in SequenceMatcher(None, old_text, new_text).get_opcodes()
                 if op != "equal"]
        a, b = _copy(a), _copy(b)
        a.edited_span = {"evidence_id": "E1", "edits": edits, "side": "A"}
        b.edited_span = {"evidence_id": "E1", "edits": edits, "side": "B"}
        pair_id = f"relation:{stable_id(a.sample_id, b.sample_id)}"
        for role, item in [("A", a), ("B", b)]:
            item.sample_id = f"{item.sample_id}:relation:{role}"
            item.pair_id, item.pair_kind, item.pair_role = pair_id, "relation_change", role
            item.counterfactual_verified = False
            item.verification_method = "pending_human_review"
        out.extend([a, b])
        pairs.append({"pair_id": pair_id, "sample_a": a.sample_id, "sample_b": b.sample_id,
                      "kind": "relation_change", "changed_fields": ["evidence"],
                      "evidence_similarity": round(sim, 4), "audit_status": "pending"})
        count += 1
    return out, pairs


def _copy(x: Example) -> Example:
    return Example(**json.loads(json.dumps(asdict(x))))


def validate_dataset(rows: list[dict], pairs: list[dict] | None = None, require_review: bool = False) -> dict:
    seen: dict[str, dict] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    claims: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        x = Example(**row)
        x.validate()
        if x.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {x.sample_id}")
        seen[x.sample_id] = row
        groups[x.group_id].add(x.split)
        claims[stable_id(x.claim.lower())].add(x.split)
        families[x.source_provenance.get("revision_family_id", x.source_id)].add(x.split)
    leaks = [g for g, splits in groups.items() if len(splits) > 1]
    if leaks:
        raise ValueError(f"Group split leakage: {leaks[:5]} ({len(leaks)} groups)")
    if pairs:
        pair_ids = set()
        for pair in pairs:
            pid = pair["pair_id"]
            if pid in pair_ids:
                raise ValueError(f"Duplicate pair: {pid}")
            pair_ids.add(pid)
            a, b = seen[pair["sample_a"]], seen[pair["sample_b"]]
            if a["pair_id"] != pid or b["pair_id"] != pid or a["pair_role"] != "A" or b["pair_role"] != "B":
                raise ValueError(f"Pair linkage invalid: {pid}")
            if a["group_id"] != b["group_id"] or a["split"] != b["split"] or a["claim"] != b["claim"]:
                raise ValueError(f"Pair group/claim/split mismatch: {pid}")
            if pair["kind"] == "bias_change":
                if a["label"] != b["label"] or a["evidence"] != b["evidence"]:
                    raise ValueError(f"Invalid bias pair: {pid}")
                if a["claim_metadata"] == b["claim_metadata"]:
                    raise ValueError(f"Bias unchanged: {pid}")
            elif pair["kind"] == "relation_change":
                if a["label"] == b["label"] or a["evidence"] == b["evidence"]:
                    raise ValueError(f"Invalid relation pair: {pid}")
            else:
                raise ValueError(f"Unknown pair kind: {pid}")
            if require_review and (pair.get("audit_status") != "approved" or
                                   not a["counterfactual_verified"] or not b["counterfactual_verified"]):
                raise ValueError(f"Unreviewed pair: {pid}")
    return {"n_examples": len(rows), "n_groups": len(groups), "n_pairs": len(pairs or []),
            "claim_hash_cross_split": sum(len(v) > 1 for v in claims.values()),
            "revision_family_cross_split": sum(len(v) > 1 for v in families.values())}


def approved_pairs(rows: list[dict], pairs: list[dict], split: str) -> list[tuple[dict, dict, dict]]:
    seen = {x["sample_id"]: x for x in rows}
    return [(seen[p["sample_a"]], seen[p["sample_b"]], p) for p in pairs
            if p.get("audit_status") == "approved" and seen[p["sample_a"]]["split"] == split
            and seen[p["sample_a"]].get("counterfactual_verified")
            and seen[p["sample_b"]].get("counterfactual_verified")]


def build_fever_wiki_index(wiki_dir: str | Path, output_db: str | Path) -> dict:
    """Index official FEVER wiki-pages JSONL files without loading the corpus in RAM."""
    import sqlite3
    wiki_dir, output_db = Path(wiki_dir), Path(output_db)
    files = sorted(wiki_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No FEVER wiki-pages JSONL files in {wiki_dir}")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_db)
    conn.execute("CREATE TABLE IF NOT EXISTS sentences (doc_id TEXT, sent_id INTEGER, text TEXT, PRIMARY KEY (doc_id,sent_id))")
    n = 0
    for path in files:
        with path.open(encoding="utf8") as f:
            batch = []
            for line in f:
                page = json.loads(line)
                for raw in page.get("lines", "").split("\n"):
                    parts = raw.split("\t", 2)
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
                        batch.append((page["id"], int(parts[0]), parts[1]))
                if len(batch) >= 10000:
                    conn.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?)", batch)
                    n += len(batch)
                    batch.clear()
            if batch:
                conn.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?)", batch)
                n += len(batch)
            conn.commit()
    conn.close()
    return {"indexed_sentences": n, "files": len(files), "output_db": str(output_db)}


def _resolve_fever_evidence(raw, conn) -> list[dict]:
    if conn is None:
        raise ValueError("Original FEVER evidence indices require wiki_db")
    if not isinstance(raw, list):
        raise ValueError("Expected FEVER evidence set list")
    # Multiple annotated evidence sets are alternatives. Use the first complete set,
    # preserving all sentences in that set for oracle classification.
    for evidence_set in raw:
        if not isinstance(evidence_set, list):
            continue
        result = []
        for entry in evidence_set:
            if not isinstance(entry, list) or len(entry) < 4 or entry[2] is None:
                result = []
                break
            doc, sent = entry[2], entry[3]
            found = conn.execute("SELECT text FROM sentences WHERE doc_id=? AND sent_id=?",
                                 (doc, sent)).fetchone()
            if not found:
                result = []
                break
            result.append({"evidence_id": f"E{len(result)+1}", "text": found[0],
                           "doc_id": doc, "sent_id": str(sent), "is_gold": True})
        if result:
            return result
    raise ValueError("No complete FEVER evidence set could be resolved")
