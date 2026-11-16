"""Independent (blind) human annotation workflow for the 300-review D2 evaluation set.

Why a separate store
--------------------
``data/annotations/annotations.json`` holds the reference labels used so far: 1 human label and 299 AI-assisted
labels. It stays untouched so the reported D2 numbers remain reproducible. Genuine human labels are collected in
``data/annotations/human_annotations.json`` (this module), which by construction can never contain an AI label
presented as a human one.

Provenance (who decided the label)
  * ``human_created``  — a human labelled the review WITHOUT seeing any AI prediction or AI-assisted label (blind).
  * ``human_verified`` — a human checked an AI-assisted label (it was shown) and accepted or corrected it.
  * ``ai_assisted``    — produced by a model and not decided by a human. Such labels are never stored here as labels;
                         they remain in the legacy reference file and are only ever shown as "AI-assisted".

Status (workflow state)
  * ``pending``   — no human label yet (implicit: the review is in the queue but has no record);
  * ``completed`` — a first human label has been saved;
  * ``flagged``   — a human marked the review as unusable/ambiguous (excluded; a reserve review can replace it);
  * ``verified``  — legacy/optional: a SECOND, different person checked the completed label. Second-person
                    verification is NOT part of the project's required workflow (decided 2026-11-15) and the app no
                    longer offers it; the status stays valid so stored records always load unchanged.

Annotation completion (``progress()['annotation_complete']``): every queued review has a human label (completed) or
a flag — i.e. ``pending == 0``. Completed labels are human-created but NOT independently verified by a second person.

The strict "verified" gate below applies only to the optional CLI ``evaluate``/``finalise`` path; it is kept so a
completed label is never silently reported as verified ground truth (use ``evaluate --include-unverified`` for an
explicitly labelled analysis against the completed human labels).
Evaluation eligibility (that CLI path): status ``verified`` AND provenance ``human_created`` or ``human_verified``.
Every change is appended to the record's ``history`` and written atomically, so progress is saved after each action
and annotation can be resumed at any time.
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from annotation.annotation_manager import AI_ASSISTED_MARKER, format_aspects_cell, parse_aspects_cell
from schemas.evaluation_schema import AnnotatedAspect, Annotation
from schemas.taxonomy import ComplaintCategory, OverallSentiment
from utils.readonly import ensure_writable
from utils.storage import read_json, write_json_atomic

Provenance = Literal["human_created", "ai_assisted", "human_verified"]
Status = Literal["pending", "completed", "flagged", "verified"]
HUMAN_PROVENANCE = ("human_created", "human_verified")
REQUIRED_HUMAN_LABELS = 300

EXPORT_COLUMNS = [
    "review_id", "product_id", "overall_sentiment", "aspects", "complaint_categories", "provenance", "status",
    "blind", "annotator", "verified_by", "independent_verification", "notes", "created_at", "updated_at", "verified_at",
]


class AnnotationConflict(ValueError):
    """Raised instead of silently overwriting another person's label, a verified label, or a newer saved state."""


def _same(a: str | None, b: str | None) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    status: Status
    provenance: Provenance
    overall_sentiment: OverallSentiment | None = None
    aspects: list[AnnotatedAspect] = Field(default_factory=list)
    complaint_categories: list[ComplaintCategory] = Field(default_factory=list)
    annotator: str = Field(..., min_length=1)
    blind: bool = Field(..., description="True if no AI prediction / AI-assisted label was shown before the label was submitted.")
    verified_by: str | None = None
    verified_at: str | None = None
    notes: str = ""
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    history: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "HumanLabel":
        if self.provenance == "ai_assisted":
            raise ValueError("AI-assisted labels are not stored as labels in the human annotation store")
        if self.provenance == "human_created" and not self.blind:
            raise ValueError("a human_created label must be blind (no AI label shown); use provenance human_verified instead")
        if self.provenance == "human_verified" and self.blind:
            raise ValueError("human_verified means an AI-assisted label was shown and checked, so blind must be False")
        if self.notes.strip().startswith(AI_ASSISTED_MARKER):
            raise ValueError(f"notes start with the AI-assisted marker '{AI_ASSISTED_MARKER}': this is not a human label")
        if self.status in ("completed", "verified") and self.overall_sentiment is None:
            raise ValueError("a completed/verified label needs an overall sentiment")
        if self.status == "pending":
            raise ValueError("pending reviews have no stored record")
        if self.status == "verified":
            if not self.verified_by:
                raise ValueError("a verified label needs verified_by")
            if self.verified_by.strip().lower() == self.annotator.strip().lower():
                raise ValueError("verification must be done by a different person than the annotator")
        cats = [a.aspect_category for a in self.aspects]
        if len(cats) != len(set(cats)):
            raise ValueError("each aspect category may be labelled only once per review")
        self.complaint_categories = list(dict.fromkeys(self.complaint_categories))
        return self

    @property
    def eligible(self) -> bool:
        return self.status == "verified" and self.provenance in HUMAN_PROVENANCE

    @property
    def independent_verification(self) -> bool:
        return bool(self.verified_by) and self.verified_by.strip().lower() != self.annotator.strip().lower()

    def label_fields(self) -> dict:
        return {
            "overall_sentiment": self.overall_sentiment.value if self.overall_sentiment else None,
            "aspects": sorted(format_aspects_cell(self.aspects).split("; ")) if self.aspects else [],
            "complaint_categories": sorted(c.value for c in self.complaint_categories),
        }

    def to_annotation(self) -> Annotation:
        """Convert to the schema the metrics code uses. Only verified human labels are converted."""
        if not self.eligible:
            raise ValueError(f"{self.review_id}: only verified human labels can be used as ground truth (status={self.status}, provenance={self.provenance})")
        return Annotation(
            review_id=self.review_id, product_id=self.product_id, overall_sentiment=self.overall_sentiment,
            aspects=self.aspects, complaint_categories=self.complaint_categories, annotator=self.annotator,
            notes=f"{self.provenance}; verified by {self.verified_by}", status="complete",
        )


class HumanAnnotationStore:
    """JSON store of genuine human labels. Every mutation is validated, logged in ``history`` and written atomically."""

    def __init__(self, path: Path):
        self.path = path
        self._in_txn = False
        self._load()

    def _load(self) -> None:
        raw = read_json(self.path, default={}) or {}
        self.meta: dict = raw.get("meta", {})
        self.reserve_ids: list[str] = list(raw.get("reserve_review_ids", []))
        self._items: dict[str, HumanLabel] = {rid: HumanLabel.model_validate(r) for rid, r in raw.get("labels", {}).items()}

    @contextmanager
    def _transaction(self, timeout_s: float = 15.0, stale_s: float = 60.0):
        """Exclusive section for one mutation: take a lock file, RE-READ the saved state (so a label saved meanwhile by
        another browser session or the CLI is seen, checked for conflicts and never overwritten), write, release."""
        ensure_writable(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {_now()}".encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > stale_s:  # left behind by a crashed process
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise AnnotationConflict(f"the annotation store is locked by another session ({lock.name}); try again")
                time.sleep(0.05)
        try:
            self._load()
            self._in_txn = True
            yield
        finally:
            self._in_txn = False
            lock.unlink(missing_ok=True)

    @classmethod
    def default(cls, settings) -> "HumanAnnotationStore":
        return cls(settings.paths.annotations / "human_annotations.json")

    # ------------------------------------------------------------------ persistence
    def _flush(self) -> None:
        ensure_writable(self.path)
        if not self._in_txn:  # every read-modify-write must re-read the file under the lock first
            raise RuntimeError("HumanAnnotationStore writes are only allowed inside _transaction()")
        self.meta.setdefault("description", "Genuine human labels collected with the blind annotation workflow (annotation/human_annotation.py). "
                                            "Never populated from model output or from the AI-assisted reference labels.")
        self.meta.update({"updated_at": _now(), "count": len(self._items), "required_human_labels": REQUIRED_HUMAN_LABELS})
        write_json_atomic(self.path, {"meta": self.meta, "reserve_review_ids": self.reserve_ids,
                                      "labels": {k: v.model_dump(mode="json") for k, v in self._items.items()}})

    def _put(self, label: HumanLabel, action: str, by: str, **extra) -> HumanLabel:
        ensure_writable(self.path)
        label.history.append({"at": _now(), "action": action, "by": by, **extra})
        label.updated_at = _now()
        label = HumanLabel.model_validate(label.model_dump())  # re-validate the final state
        self._items[label.review_id] = label
        self._flush()
        return label

    # ------------------------------------------------------------------ queries
    def get(self, review_id: str) -> HumanLabel | None:
        return self._items.get(review_id)

    def all(self) -> list[HumanLabel]:
        return list(self._items.values())

    def status_of(self, review_id: str) -> Status:
        lab = self._items.get(review_id)
        return lab.status if lab else "pending"

    def queue(self, sample_ids: list[str]) -> list[str]:
        """The annotation queue: the 300-review sample plus any reserve reviews added to replace flagged ones."""
        return list(dict.fromkeys(list(sample_ids) + self.reserve_ids))

    def next_pending(self, queue: list[str], after: str | None = None) -> str | None:
        start = queue.index(after) + 1 if after in queue else 0
        for rid in queue[start:] + queue[:start]:
            if rid not in self._items:
                return rid
        return None

    def progress(self, queue: list[str]) -> dict:
        labs = [self._items[rid] for rid in queue if rid in self._items]
        by_status = {s: sum(lab.status == s for lab in labs) for s in ("completed", "flagged", "verified")}
        eligible = [lab for lab in labs if lab.eligible]
        pending = len(queue) - len(labs)
        labelled = by_status["completed"] + by_status["verified"]  # a human label exists (verified is legacy-only)
        return {
            "queue": len(queue),
            "pending": pending,
            **by_status,
            # Annotation completion (project workflow): every queued review has a human decision (label or flag).
            # Second-person verification is NOT part of the required workflow and is not needed for completion.
            "labelled": labelled,
            "annotation_complete": pending == 0 and len(queue) > 0,
            "human_created": sum(lab.provenance == "human_created" for lab in labs),
            "human_verified_provenance": sum(lab.provenance == "human_verified" for lab in labs),
            "eligible": len(eligible),
            "eligible_blind": sum(lab.provenance == "human_created" for lab in eligible),
            "required": REQUIRED_HUMAN_LABELS,
            "remaining_to_required": max(0, REQUIRED_HUMAN_LABELS - len(eligible)),
            "annotators": sorted({lab.annotator for lab in labs}),
            "verifiers": sorted({lab.verified_by for lab in labs if lab.verified_by}),
        }

    def gold(self, queue: list[str] | None = None, include_unverified: bool = False) -> list[Annotation]:
        """Ground truth for D2: verified human labels only (``include_unverified`` adds completed ones, for provisional
        analysis only — never for the reported result)."""
        ids = set(queue) if queue is not None else None
        out = []
        for lab in self._items.values():
            if ids is not None and lab.review_id not in ids:
                continue
            if lab.provenance not in HUMAN_PROVENANCE:
                continue
            if lab.eligible:
                out.append(lab.to_annotation())
            elif include_unverified and lab.status == "completed":
                out.append(Annotation(review_id=lab.review_id, product_id=lab.product_id, overall_sentiment=lab.overall_sentiment,
                                      aspects=lab.aspects, complaint_categories=lab.complaint_categories, annotator=lab.annotator,
                                      notes=f"{lab.provenance}; NOT verified", status="complete"))
        return out

    # ------------------------------------------------------------------ actions
    def save_blind_label(self, review_id: str, product_id: str, *, annotator: str, overall_sentiment: str,
                         aspects: list[AnnotatedAspect] | None = None, complaint_categories: list[str] | None = None,
                         notes: str = "", allow_relabel: bool = True) -> HumanLabel:
        """Save a label made WITHOUT any AI suggestion visible -> provenance human_created, status completed.

        Only the annotator who owns a label may re-label it (logged in ``history``; a verified label goes back to
        completed and must be verified again). A label by another person, or a checked-AI label, is never overwritten:
        ``AnnotationConflict`` is raised instead (disagreements are resolved on the verification step)."""
        with self._transaction():
            prev = self._items.get(review_id)
            if prev is not None and not allow_relabel:
                raise AnnotationConflict(f"{review_id} already has a label ({prev.status}); import never overwrites existing labels")
            if prev is not None and not _same(prev.annotator, annotator):
                raise AnnotationConflict(f"{review_id} already has a {prev.status} label by {prev.annotator}; it is not overwritten. "
                                         "Correct it on the verification step instead.")
            if prev is not None and prev.provenance != "human_created":
                raise AnnotationConflict(f"{review_id} has a {prev.provenance} label; a blind label cannot replace it without a reset")
            return self._save_blind_unlocked(review_id, product_id, prev, annotator=annotator, overall_sentiment=overall_sentiment,
                                             aspects=aspects, complaint_categories=complaint_categories, notes=notes)

    def _save_blind_unlocked(self, review_id, product_id, prev, *, annotator, overall_sentiment, aspects, complaint_categories, notes) -> HumanLabel:
        lab = HumanLabel(
            review_id=review_id, product_id=product_id, status="completed", provenance="human_created", blind=True,
            overall_sentiment=overall_sentiment, aspects=aspects or [], complaint_categories=complaint_categories or [],
            annotator=annotator.strip(), notes=notes, created_at=prev.created_at if prev else _now(),
            history=list(prev.history) if prev else [],
        )
        return self._put(lab, "relabelled" if prev else "labelled", annotator.strip(), previous=prev.label_fields() if prev else None)

    def flag(self, review_id: str, product_id: str, *, annotator: str, reason: str) -> HumanLabel:
        if not reason.strip():
            raise ValueError("give a reason for flagging")
        with self._transaction():
            prev = self._items.get(review_id)
            if prev is not None and not _same(prev.annotator, annotator):
                raise AnnotationConflict(f"{review_id} already has a {prev.status} label by {prev.annotator}; only that annotator can flag it")
            if prev is not None and prev.status == "verified":
                raise AnnotationConflict(f"{review_id} is verified (by {prev.verified_by}); reset it explicitly before flagging")
            lab = HumanLabel(review_id=review_id, product_id=product_id, status="flagged", provenance="human_created", blind=True,
                             annotator=annotator.strip(), notes=reason, created_at=prev.created_at if prev else _now(),
                             history=list(prev.history) if prev else [])
            return self._put(lab, "flagged", annotator.strip(), reason=reason)

    def verify(self, review_id: str, *, reviewer: str, corrections: dict | None = None) -> HumanLabel:
        """A second person confirms (or corrects) a completed label -> status verified.
        ``corrections`` may contain overall_sentiment / aspects / complaint_categories."""
        with self._transaction():
            return self._verify_unlocked(review_id, reviewer=reviewer, corrections=corrections)

    def _verify_unlocked(self, review_id: str, *, reviewer: str, corrections: dict | None) -> HumanLabel:
        lab = self._items.get(review_id)
        if lab is not None and lab.status == "verified":
            raise AnnotationConflict(f"{review_id} was already verified by {lab.verified_by}")
        if lab is None or lab.status != "completed":
            raise ValueError(f"{review_id}: only completed labels can be verified (current status {self.status_of(review_id)})")
        if reviewer.strip().lower() == lab.annotator.strip().lower():
            raise ValueError("you cannot verify your own label; ask a different annotator")
        before = lab.label_fields()
        data = lab.model_dump()
        data.update({k: v for k, v in (corrections or {}).items() if k in ("overall_sentiment", "aspects", "complaint_categories")})
        data.update({"status": "verified", "verified_by": reviewer.strip(), "verified_at": _now()})
        new = HumanLabel.model_validate(data)
        changed = new.label_fields() != before
        return self._put(new, "verified_with_corrections" if changed else "verified", reviewer.strip(),
                         previous=before if changed else None)

    def save_checked_ai_label(self, review_id: str, product_id: str, *, annotator: str, overall_sentiment: str,
                              aspects: list[AnnotatedAspect] | None = None, complaint_categories: list[str] | None = None,
                              ai_label: dict | None = None, notes: str = "") -> HumanLabel:
        """A human checked an AI-assisted label that WAS shown to them -> provenance human_verified (not blind),
        status completed. It still needs verification by a second person to become eligible."""
        with self._transaction():
            prev = self._items.get(review_id)
            if prev is not None and prev.provenance == "human_created":
                raise AnnotationConflict(f"{review_id} already has a blind human label; it is not replaced by a checked AI label")
            if prev is not None and not _same(prev.annotator, annotator):
                raise AnnotationConflict(f"{review_id} already has a label by {prev.annotator}; it is not overwritten")
            lab = HumanLabel(review_id=review_id, product_id=product_id, status="completed", provenance="human_verified", blind=False,
                             overall_sentiment=overall_sentiment, aspects=aspects or [], complaint_categories=complaint_categories or [],
                             annotator=annotator.strip(), notes=notes, history=list(prev.history) if prev else [])
            return self._put(lab, "checked_ai_label", annotator.strip(), ai_label=ai_label)

    def reset(self, review_id: str, *, by: str) -> None:
        """Remove a label (back to pending). Only its annotator may do this. The removed record is kept in ``meta.removed``."""
        with self._transaction():
            lab = self._items.get(review_id)
            if lab is None:
                return
            if not _same(lab.annotator, by):
                raise AnnotationConflict(f"{review_id} was labelled by {lab.annotator}; only that annotator can remove it")
            self._items.pop(review_id)
            self.meta.setdefault("removed", []).append({"at": _now(), "by": by, "record": lab.model_dump(mode="json")})
            self._flush()

    def add_reserve(self, reviews: pd.DataFrame, exclude: set[str], n: int, seed: int) -> list[str]:
        """Add ``n`` replacement reviews (for flagged ones) from outside the sample, reproducibly (seeded, sorted pool)."""
        with self._transaction():
            pool = sorted(set(reviews["review_id"]) - set(exclude) - set(self.reserve_ids))
            rng = random.Random(f"{seed}-reserve-{len(self.reserve_ids)}")
            picked = rng.sample(pool, min(n, len(pool)))
            self.reserve_ids += picked
            self._flush()
            return picked

    # ------------------------------------------------------------------ import / export
    def to_dataframe(self, eligible_only: bool = False) -> pd.DataFrame:
        rows = []
        for lab in self._items.values():
            if eligible_only and not lab.eligible:
                continue
            rows.append({
                "review_id": lab.review_id, "product_id": lab.product_id,
                "overall_sentiment": lab.overall_sentiment.value if lab.overall_sentiment else "",
                "aspects": format_aspects_cell(lab.aspects), "complaint_categories": "; ".join(c.value for c in lab.complaint_categories),
                "provenance": lab.provenance, "status": lab.status, "blind": lab.blind, "annotator": lab.annotator,
                "verified_by": lab.verified_by or "", "independent_verification": lab.independent_verification, "notes": lab.notes,
                "created_at": lab.created_at, "updated_at": lab.updated_at, "verified_at": lab.verified_at or "",
            })
        return pd.DataFrame(rows, columns=EXPORT_COLUMNS)

    def export_reference(self, out_dir: Path) -> dict[str, Path]:
        """Final human-verified reference dataset (eligible labels only) as JSON + CSV, with a provenance summary."""
        ensure_writable(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        eligible = [lab for lab in self._items.values() if lab.eligible]
        doc = {"generated_at": _now(), "count": len(eligible),
               "provenance": {p: sum(lab.provenance == p for lab in eligible) for p in HUMAN_PROVENANCE},
               "required": REQUIRED_HUMAN_LABELS, "meets_requirement": len(eligible) >= REQUIRED_HUMAN_LABELS,
               "labels": [lab.model_dump(mode="json") for lab in eligible]}
        paths = {"json": out_dir / "human_reference.json", "csv": out_dir / "human_reference.csv"}
        paths["json"].write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        self.to_dataframe(eligible_only=True).to_csv(paths["csv"], index=False, encoding="utf-8")
        return paths

    def import_blind_csv(self, df: pd.DataFrame, product_ids: dict[str, str], default_annotator: str = "") -> dict:
        """Import labels filled offline in the BLANK template (it contains no AI labels) as blind human labels.
        Rows whose notes carry the AI-assisted marker, or that name another provenance, are rejected."""
        imported, skipped, errors = 0, 0, []
        self._load()
        seen: set[str] = set()
        for i, row in df.fillna("").iterrows():
            if not str(row.get("overall_sentiment", "")).strip():
                skipped += 1
                continue
            rid = str(row.get("review_id", "")).strip()
            try:
                prov = str(row.get("provenance", "") or "human_created").strip()
                if prov != "human_created":
                    raise ValueError(f"provenance '{prov}' cannot be imported as a blind human label")
                if rid not in product_ids:
                    raise ValueError(f"review id {rid!r} is not in the annotation queue")
                annotator = str(row.get("annotator", "")).strip() or default_annotator
                if not annotator:
                    raise ValueError("annotator is required")
                if self.get(rid) is not None or rid in seen:
                    raise AnnotationConflict(f"{rid} already has a label ({self.status_of(rid)}) or appears twice; import never overwrites labels")
                seen.add(rid)
                self.save_blind_label(rid, product_ids[rid], annotator=annotator, overall_sentiment=str(row["overall_sentiment"]).strip().lower(),
                                      aspects=parse_aspects_cell(row.get("aspects", "")),
                                      complaint_categories=[c.strip().lower() for c in str(row.get("complaint_categories", "")).replace(",", ";").split(";") if c.strip()],
                                      notes=str(row.get("notes", "")), allow_relabel=False)
                imported += 1
            except (ValidationError, ValueError) as exc:
                errors.append(f"row {i + 2} ({rid}): {str(exc).splitlines()[0]}")
        return {"imported": imported, "skipped_unlabelled": skipped, "errors": errors}


def migrate_legacy_human_labels(store: HumanAnnotationStore, legacy_path: Path) -> list[str]:
    """Copy the labels that ``annotations.json`` documents as HUMAN (``meta.annotation_method.human_labels``) into the
    human store as blind human_created/completed labels. AI-assisted labels are never copied. Idempotent."""
    raw = read_json(legacy_path, default={}) or {}
    human_ids = (raw.get("meta", {}).get("annotation_method", {}) or {}).get("human_labels", [])
    moved = []
    for rid in human_ids:
        a = raw.get("annotations", {}).get(rid)
        if not a or store.get(rid) or (a.get("notes") or "").startswith(AI_ASSISTED_MARKER) or a.get("status") != "complete":
            continue
        lab = store.save_blind_label(rid, a["product_id"], annotator=a["annotator"], overall_sentiment=a["overall_sentiment"],
                                     aspects=[AnnotatedAspect(aspect_category=x["aspect_category"], sentiment=x["sentiment"]) for x in a.get("aspects", [])],
                                     complaint_categories=a.get("complaint_categories", []), notes=a.get("notes", ""))
        with store._transaction():  # re-read under the lock, then record the migration in the label's history
            lab = store._items[rid]
            lab.history[-1]["action"] = "migrated_from_annotations_json"
            lab.history[-1]["original_annotated_at"] = a.get("annotated_at")
            store._flush()
        moved.append(rid)
    return moved
