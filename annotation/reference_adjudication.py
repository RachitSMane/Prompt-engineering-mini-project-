"""Review / approval (adjudication) of the existing D2 reference annotations.

The 300 reference annotations in ``data/annotations/annotations.json`` are NOT modified. Their original provenance
(1 human-created, 299 AI-assisted) stays as it is. This module stores, in a separate file
(``data/annotations/reference_adjudication.json``), what a named person actually decided about each record:

* ``approved``  — accepted unchanged as a D2 reference label (``method`` = ``bulk`` or ``individual``);
* ``modified``  — accepted with a correction; the correction is stored here, the original label stays untouched;
* ``excluded``  — rejected (e.g. unreadable review), with a reason.

An approval is a review/acceptance decision. It does NOT turn an AI-assisted label into a human-created one:
``original_provenance`` is copied from the annotation file and never changed, and the dashboard reports
independent human labels separately (``annotation/human_annotation.py``).

Rules
* Bulk approval needs a reviewer name and an explicit confirmation; it only touches records that are valid, have
  status ``complete`` and have no decision yet (idempotent: running it twice changes nothing).
* Flagged records are never bulk-approved; they need an individual decision.
* Each decision stores a SHA-256 of the original label. If the annotation file changes later, that decision no longer
  counts (``stale``) until it is reviewed again.
* Every write takes a short lock and re-reads the file (two browser sessions cannot erase each other's decisions);
  read-only demo mode blocks every write.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from schemas.evaluation_schema import Annotation
from utils.readonly import ensure_writable
from utils.storage import write_json_atomic

REQUIRED_REFERENCE_LABELS = 300
ACCEPTED = ("approved", "modified")
DECISIONS = ACCEPTED + ("excluded",)
LOCK_STALE_S = 60


class AdjudicationError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def label_sha256(record: dict) -> str:
    keep = {k: record.get(k) for k in ("review_id", "product_id", "overall_sentiment", "aspects", "complaint_categories")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_reference(annotations_json: Path, sample_csv: Path) -> dict:
    """Load and validate the existing reference annotations against the sample. Never writes."""
    data = json.loads(Path(annotations_json).read_text(encoding="utf-8"))
    method = data.get("meta", {}).get("annotation_method", {})
    human_ids = set(method.get("human_labels", []))
    raw = data["annotations"]
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    sample = pd.read_csv(sample_csv, dtype=str)
    sample_ids = sample["review_id"].tolist()
    sample_products = dict(zip(sample["review_id"], sample["product_id"]))
    problems: list[str] = []
    ids = [r.get("review_id") for r in rows]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        problems.append(f"duplicate review ids: {dups[:10]}")
    if len(set(sample_ids)) != len(sample_ids):
        problems.append("duplicate review ids in the sample")
    records: dict[str, dict] = {}
    invalid: dict[str, str] = {}
    for r in rows:
        rid = r.get("review_id")
        try:
            Annotation.model_validate(r)
        except ValidationError as exc:
            invalid[rid] = "; ".join(e["msg"] for e in exc.errors()[:3])
            continue
        if rid not in sample_products:
            invalid[rid] = "review id not in the annotation sample"
            continue
        if sample_products[rid] != r.get("product_id"):
            invalid[rid] = f"product id {r.get('product_id')} differs from the sample ({sample_products[rid]})"
            continue
        records[rid] = r
    missing = [rid for rid in sample_ids if rid not in records and rid not in invalid]
    if missing:
        problems.append(f"{len(missing)} sample review(s) have no reference annotation: {missing[:10]}")
    if invalid:
        problems.append(f"{len(invalid)} invalid annotation(s): {dict(list(invalid.items())[:5])}")
    provenance = {rid: ("human_created" if rid in human_ids else "ai_assisted") for rid in records}
    return {"records": records, "sample_ids": sample_ids, "invalid": invalid, "missing": missing, "problems": problems,
            "provenance": provenance, "approved_by_meta": method.get("approved_by"), "valid": not problems}


class ReferenceAdjudicationStore:
    def __init__(self, path: Path, annotations_json: Path, sample_csv: Path, required: int = REQUIRED_REFERENCE_LABELS):
        self.path = Path(path)
        self.annotations_json = Path(annotations_json)
        self.sample_csv = Path(sample_csv)
        self.required = required
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @classmethod
    def default(cls, settings) -> "ReferenceAdjudicationStore":
        p = settings.paths
        return cls(p.annotations / "reference_adjudication.json", p.annotations_json, p.annotation_sample_csv)

    # ------------------------------------------------------------------ storage
    def _read(self) -> dict:
        if not self.path.exists():
            return {"meta": {"description": "Review/approval decisions on the existing D2 reference annotations. "
                             "The annotations themselves (annotations.json) are never modified; original provenance is "
                             "kept per record. Approval is not independent human annotation.",
                             "created_at": _now()}, "decisions": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    @contextmanager
    def _transaction(self):
        ensure_writable(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 10
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > LOCK_STALE_S:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() > deadline:
                    raise AdjudicationError("the reference store is being changed by another session; try again")
                time.sleep(0.1)
        try:
            data = self._read()  # always re-read under the lock
            yield data
            data["meta"]["updated_at"] = _now()
            write_json_atomic(self.path, data)
        finally:
            self.lock_path.unlink(missing_ok=True)

    def decisions(self) -> dict[str, dict]:
        return self._read()["decisions"]

    # ------------------------------------------------------------------ actions
    def _decision(self, ref: dict, rid: str, decision: str, reviewer: str, method: str, **extra) -> dict:
        rec = ref["records"][rid]
        return {"review_id": rid, "product_id": rec["product_id"], "decision": decision, "reviewer": reviewer.strip(),
                "method": method, "decided_at": _now(), "original_provenance": ref["provenance"][rid],
                "original_status": rec.get("status", "complete"), "original_label_sha256": label_sha256(rec), **extra}

    @staticmethod
    def _reviewer(reviewer: str) -> str:
        if not reviewer or len(reviewer.strip()) < 2:
            raise AdjudicationError("enter the reviewer's name")
        return reviewer.strip()

    def bulk_approve(self, reviewer: str, *, confirmed: bool) -> dict:
        """Approve every valid, complete, undecided reference record. Idempotent. Flagged records are skipped."""
        reviewer = self._reviewer(reviewer)
        if not confirmed:
            raise AdjudicationError("confirm that you have reviewed and accept these reference labels")
        ref = load_reference(self.annotations_json, self.sample_csv)
        if ref["invalid"] or ref["missing"]:
            raise AdjudicationError("reference annotations failed validation: " + "; ".join(ref["problems"]))
        stats = {"approved_now": 0, "already_decided": 0, "skipped_flagged": 0}
        with self._transaction() as data:
            for rid in ref["sample_ids"]:
                rec = ref["records"][rid]
                cur = data["decisions"].get(rid)
                if cur and cur.get("original_label_sha256") == label_sha256(rec):
                    stats["already_decided"] += 1
                    continue
                if rec.get("status", "complete") != "complete":
                    stats["skipped_flagged"] += 1
                    continue
                d = self._decision(ref, rid, "approved", reviewer, "bulk")
                d["history"] = (cur or {}).get("history", []) + [{"action": "bulk_approved", "by": reviewer, "at": d["decided_at"]}]
                data["decisions"][rid] = d
                stats["approved_now"] += 1
            data["meta"].setdefault("bulk_approvals", []).append({"by": reviewer, "at": _now(), **stats})
        return stats

    def decide(self, rid: str, decision: str, reviewer: str, *, reason: str = "", correction: dict | None = None) -> dict:
        """Individual decision (approve / modify / exclude). May replace an earlier decision; history is kept."""
        reviewer = self._reviewer(reviewer)
        if decision not in DECISIONS:
            raise AdjudicationError(f"unknown decision {decision!r}")
        ref = load_reference(self.annotations_json, self.sample_csv)
        if rid not in ref["records"]:
            raise AdjudicationError(f"{rid} is not a valid reference record")
        extra: dict = {}
        if decision == "excluded":
            if not reason.strip():
                raise AdjudicationError("a reason is required to exclude a record")
            extra["reason"] = reason.strip()
        if decision == "modified":
            if not correction:
                raise AdjudicationError("a correction is required")
            merged = {**ref["records"][rid], **correction}
            Annotation.model_validate(merged)  # the corrected label must be valid
            extra["corrected_label"] = {k: merged[k] for k in ("overall_sentiment", "aspects", "complaint_categories")}
            if reason:
                extra["reason"] = reason.strip()
        with self._transaction() as data:
            cur = data["decisions"].get(rid)
            d = self._decision(ref, rid, decision, reviewer, "individual", **extra)
            d["history"] = (cur or {}).get("history", []) + [{"action": decision, "by": reviewer, "at": d["decided_at"],
                                                              "previous": (cur or {}).get("decision")}]
            data["decisions"][rid] = d
        return d

    # ------------------------------------------------------------------ views
    def progress(self) -> dict:
        ref = load_reference(self.annotations_json, self.sample_csv)
        dec = self.decisions()
        counts = {k: 0 for k in DECISIONS}
        stale = []
        for rid in ref["sample_ids"]:
            d = dec.get(rid)
            if not d or rid not in ref["records"]:
                continue
            if d.get("original_label_sha256") != label_sha256(ref["records"][rid]):
                stale.append(rid)
                continue
            counts[d["decision"]] += 1
        accepted = counts["approved"] + counts["modified"]
        decided = accepted + counts["excluded"]
        available = len(ref["records"])
        prov = {"human_created": 0, "ai_assisted": 0}
        for rid, d in dec.items():
            if rid in ref["records"] and rid not in stale and d["decision"] in ACCEPTED:
                prov[ref["provenance"][rid]] += 1
        reviewers = sorted({d["reviewer"] for d in dec.values()})
        return {
            "available": available, "sample_size": len(ref["sample_ids"]), "required": self.required,
            "accepted": accepted, "approved": counts["approved"], "modified": counts["modified"], "excluded": counts["excluded"],
            "awaiting_review": len(ref["sample_ids"]) - decided, "stale_decisions": stale,
            "accepted_by_original_provenance": prov,
            "flagged_in_source": sorted(r for r, x in ref["records"].items() if x.get("status", "complete") != "complete"),
            "reviewers": reviewers, "validation_problems": ref["problems"],
            "ready": ref["valid"] and accepted >= self.required and decided == len(ref["sample_ids"]) and not stale,
        }

    def accepted_gold(self) -> list[Annotation]:
        """Accepted reference labels as ``Annotation`` objects (corrections applied), in sample order."""
        ref = load_reference(self.annotations_json, self.sample_csv)
        dec = self.decisions()
        out = []
        for rid in ref["sample_ids"]:
            d, rec = dec.get(rid), ref["records"].get(rid)
            if not d or not rec or d["decision"] not in ACCEPTED or d.get("original_label_sha256") != label_sha256(rec):
                continue
            label = {**rec, **(d.get("corrected_label") or {}), "status": "complete"}
            out.append(Annotation.model_validate(label))
        return out


def reference_dataframe(store: "ReferenceAdjudicationStore") -> pd.DataFrame:
    """Accepted reference labels with their provenance and approval record (one row per review)."""
    ref = load_reference(store.annotations_json, store.sample_csv)
    dec = store.decisions()
    rows = []
    for g in store.accepted_gold():
        d = dec[g.review_id]
        rows.append({"review_id": g.review_id, "product_id": g.product_id, "overall_sentiment": g.overall_sentiment.value,
                     "aspects": ";".join(f"{a.aspect_category.value}:{a.sentiment.value}" for a in g.aspects),
                     "complaint_categories": ";".join(c.value for c in g.complaint_categories),
                     "original_provenance": ref["provenance"][g.review_id], "decision": d["decision"], "method": d["method"],
                     "reviewer": d["reviewer"], "decided_at": d["decided_at"]})
    return pd.DataFrame(rows, columns=["review_id", "product_id", "overall_sentiment", "aspects", "complaint_categories",
                                       "original_provenance", "decision", "method", "reviewer", "decided_at"])


def export_reference(store: "ReferenceAdjudicationStore", out_dir: Path) -> dict:
    ensure_writable(out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = reference_dataframe(store)
    df.to_csv(out_dir / "approved_reference.csv", index=False)
    write_json_atomic(out_dir / "approved_reference.json", {"progress": store.progress(), "records": df.to_dict("records")})
    return {"csv": out_dir / "approved_reference.csv", "json": out_dir / "approved_reference.json", "rows": len(df)}
