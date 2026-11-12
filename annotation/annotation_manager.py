"""Human annotation workflow for the 300-review evaluation set.

* ``build_annotation_sample`` draws a reproducible, rating-stratified sample.
* ``AnnotationStore`` persists labels in ``data/annotations/annotations.json``
  (atomic writes) and imports/exports CSV.

Labels saved through the UI or the CSV template are human labels. Nothing in the
pipeline writes model predictions into this store. A label file may also contain
documented AI-assisted reference labels (per-record ``notes`` beginning with
``AI_ASSISTED_MARKER`` and ``meta.annotation_method``); ``label_provenance``
reports the split so the app and reports never call them "human" labels.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from schemas.evaluation_schema import AnnotatedAspect, Annotation
from utils.storage import read_json, write_json_atomic

SAMPLE_COLUMNS = ["sample_order", "review_id", "product_id", "product_title", "rating", "review_text"]
ANNOTATION_CSV_COLUMNS = [
    "review_id", "product_id", "review_text", "overall_sentiment", "aspects",
    "complaint_categories", "annotator", "notes", "status", "annotated_at",
]
AI_ASSISTED_MARKER = "AI-assisted label"


def label_provenance(annotations) -> dict:
    """Count human vs AI-assisted labels (marker at the start of ``notes``)."""
    ai = sum(1 for a in annotations if (a.notes or "").startswith(AI_ASSISTED_MARKER))
    return {"total": len(annotations), "human": len(annotations) - ai, "ai_assisted": ai}


def provenance_text(prov: dict) -> str:
    if not prov.get("total"):
        return "no reference labels"
    if not prov["ai_assisted"]:
        return f"{prov['total']} human labels"
    return f"{prov['total']} reference labels: {prov['human']} human, {prov['ai_assisted']} AI-assisted (documented in annotations.json)"


TEMPLATE_HELP = (
    "aspects format: 'category:sentiment; category:sentiment' e.g. 'noise:negative; performance:positive'. "
    "complaint_categories format: 'cat1; cat2'. Leave both empty if none."
)


def build_annotation_sample(reviews: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Rating-stratified sample so minority sentiment classes are represented.

    Target n/5 reviews per star rating; shortfalls are filled randomly from the
    remaining reviews. The natural rating distribution is heavily skewed to 5★,
    so a purely random sample would leave too few negative/mixed examples for a
    meaningful macro-F1. (Documented in docs/experiment_methodology.md.)
    """
    if len(reviews) <= n:
        sample = reviews.copy()
    else:
        rng = random.Random(seed)
        pool = reviews.sort_values("review_id").reset_index(drop=True)
        ratings = sorted(pool["rating"].dropna().unique())
        per = n // max(1, len(ratings))
        chosen: list[int] = []
        for r in ratings:
            idx = pool.index[pool["rating"] == r].tolist()
            chosen += rng.sample(idx, min(per, len(idx)))
        remaining = sorted(set(pool.index) - set(chosen))
        chosen += rng.sample(remaining, n - len(chosen))
        sample = pool.loc[chosen]
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle order shown to annotator
    sample.insert(0, "sample_order", range(1, len(sample) + 1))
    return sample[[c for c in SAMPLE_COLUMNS if c in sample.columns]]


def save_sample(sample: pd.DataFrame, path: Path, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8")
    return True


def load_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return pd.read_csv(path, dtype={"review_id": str, "product_id": str}, keep_default_na=False)


def write_template(sample: pd.DataFrame, path: Path) -> None:
    """Blank CSV for annotating offline (e.g. in Excel) and importing later."""
    tpl = pd.DataFrame({c: "" for c in ANNOTATION_CSV_COLUMNS}, index=range(len(sample)))
    tpl["review_id"] = sample["review_id"].values
    tpl["product_id"] = sample["product_id"].values
    tpl["review_text"] = sample["review_text"].values
    tpl["status"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    tpl.to_csv(path, index=False, encoding="utf-8")


def _split(value: str) -> list[str]:
    return [p.strip() for p in str(value or "").replace(",", ";").split(";") if p.strip()]


def parse_aspects_cell(value: str) -> list[AnnotatedAspect]:
    aspects = []
    for part in _split(value):
        if ":" not in part:
            raise ValueError(f"aspect '{part}' must look like 'category:sentiment'")
        cat, sent = (x.strip().lower() for x in part.split(":", 1))
        aspects.append(AnnotatedAspect(aspect_category=cat, sentiment=sent))
    return aspects


def format_aspects_cell(aspects: list[AnnotatedAspect]) -> str:
    return "; ".join(f"{a.aspect_category.value}:{a.sentiment.value}" for a in aspects)


class AnnotationStore:
    def __init__(self, path: Path):
        self.path = path
        raw = read_json(path, default={}) or {}
        self.meta: dict = raw.get("meta", {})
        self._items: dict[str, Annotation] = {rid: Annotation.model_validate(a) for rid, a in raw.get("annotations", {}).items()}

    # --------------------------------------------------------------- CRUD
    def get(self, review_id: str) -> Annotation | None:
        return self._items.get(review_id)

    def save(self, annotation: Annotation) -> None:
        self._items[annotation.review_id] = annotation
        self._flush()

    def delete(self, review_id: str) -> None:
        if self._items.pop(review_id, None) is not None:
            self._flush()

    def _flush(self) -> None:
        # Keep an existing description (e.g. one documenting AI-assisted labels); only new files get the default.
        self.meta.setdefault("description", "Human-created ground-truth labels. Never populated from model output.")
        self.meta.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "count": len(self._items),
            }
        )
        write_json_atomic(self.path, {"meta": self.meta, "annotations": {k: v.model_dump(mode="json") for k, v in self._items.items()}})

    # --------------------------------------------------------------- queries
    def all(self) -> list[Annotation]:
        return list(self._items.values())

    def completed(self, review_ids: list[str] | None = None) -> list[Annotation]:
        items = [a for a in self._items.values() if a.status == "complete"]
        if review_ids is not None:
            allowed = set(review_ids)
            items = [a for a in items if a.review_id in allowed]
        return items

    def progress(self, sample_ids: list[str]) -> dict:
        ids = set(sample_ids)
        done = [a for a in self._items.values() if a.review_id in ids]
        complete = sum(a.status == "complete" for a in done)
        flagged = len(done) - complete
        # Flagged reviews are excluded from evaluation, so readiness is measured against the evaluable reviews.
        return {"total": len(ids), "complete": complete, "flagged": flagged, "remaining": len(ids) - len(done),
                "evaluable": len(ids) - flagged}

    # --------------------------------------------------------------- export / import
    def to_dataframe(self, sample: pd.DataFrame | None = None) -> pd.DataFrame:
        texts = dict(zip(sample["review_id"], sample["review_text"])) if sample is not None and len(sample) else {}
        rows = [
            {
                "review_id": a.review_id,
                "product_id": a.product_id,
                "review_text": texts.get(a.review_id, ""),
                "overall_sentiment": a.overall_sentiment.value,
                "aspects": format_aspects_cell(a.aspects),
                "complaint_categories": "; ".join(c.value for c in a.complaint_categories),
                "annotator": a.annotator,
                "notes": a.notes,
                "status": a.status,
                "annotated_at": a.annotated_at,
            }
            for a in self._items.values()
        ]
        return pd.DataFrame(rows, columns=ANNOTATION_CSV_COLUMNS)

    def to_json(self) -> str:
        return json.dumps([a.model_dump(mode="json") for a in self._items.values()], ensure_ascii=False, indent=2)

    def import_csv(self, df: pd.DataFrame, default_annotator: str = "") -> dict:
        """Import human labels from the CSV template. Returns counts and row errors."""
        imported, skipped, errors = 0, 0, []
        df = df.fillna("")
        for i, row in df.iterrows():
            if not str(row.get("overall_sentiment", "")).strip():
                skipped += 1  # unlabelled row
                continue
            try:
                ann = Annotation(
                    review_id=str(row["review_id"]).strip(),
                    product_id=str(row["product_id"]).strip(),
                    overall_sentiment=str(row["overall_sentiment"]).strip().lower(),
                    aspects=parse_aspects_cell(row.get("aspects", "")),
                    complaint_categories=[c.lower() for c in _split(row.get("complaint_categories", ""))],
                    annotator=str(row.get("annotator", "")).strip() or default_annotator,
                    notes=str(row.get("notes", "")),
                    status=str(row.get("status", "")).strip().lower() or "complete",
                )
                self._items[ann.review_id] = ann
                imported += 1
            except (ValidationError, ValueError, KeyError) as exc:
                errors.append(f"row {i + 2}: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
        if imported:
            self._flush()
        return {"imported": imported, "skipped_unlabelled": skipped, "errors": errors}
