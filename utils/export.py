"""JSON and CSV export with stable, documented columns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from schemas.review_schema import ExtractionFailure, ReviewRecord
from schemas.taxonomy import AspectCategory

# Fixed column order => every CSV export has identical columns, even when empty.
RECORD_CSV_COLUMNS = [
    "review_id", "product_id", "product_title", "rating", "review_text",
    "overall_sentiment", "confidence", "num_aspects", "aspect_categories",
    "positive_aspects", "negative_aspects", "neutral_aspects", "complaint_categories",
    "aspects_json", "rationale", "strategy", "provider", "model_name", "prompt_version",
    "is_mock", "validation_success", "repair_attempts", "local_repairs",
    "latency_s", "input_tokens", "output_tokens", "extracted_at",
]
ASPECT_CSV_COLUMNS = ["review_id", "product_id", "aspect_name", "aspect_category", "sentiment", "evidence", "confidence", "strategy", "model_name", "is_mock"]
FAILURE_CSV_COLUMNS = ["review_id", "product_id", "review_text", "error", "strategy", "model_name", "is_mock", "repair_attempts"]


def _aspects_by(record: ReviewRecord, sentiment: str) -> str:
    return "; ".join(f"{a.aspect_category.value}:{a.aspect_name}" for a in record.aspects if a.sentiment.value == sentiment)


def records_to_json(records: Iterable[ReviewRecord], include_attempts: bool = True) -> str:
    exclude = None if include_attempts else {"extraction_metadata": {"attempts"}}
    data = [r.model_dump(mode="json", exclude=exclude) for r in records]
    text = json.dumps(data, ensure_ascii=False, indent=2)
    json.loads(text)  # guarantee validity
    return text


def records_to_dataframe(records: Iterable[ReviewRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        m = r.extraction_metadata
        rows.append(
            {
                "review_id": r.review_id,
                "product_id": r.product_id,
                "product_title": r.product_title or "",
                "rating": r.rating,
                "review_text": r.review_text,
                "overall_sentiment": r.overall_sentiment.value,
                "confidence": r.confidence,
                "num_aspects": len(r.aspects),
                "aspect_categories": "; ".join(dict.fromkeys(a.aspect_category.value for a in r.aspects)),
                "positive_aspects": _aspects_by(r, "positive"),
                "negative_aspects": _aspects_by(r, "negative"),
                "neutral_aspects": _aspects_by(r, "neutral"),
                "complaint_categories": "; ".join(c.value for c in r.complaint_categories),
                "aspects_json": json.dumps([a.model_dump(mode="json") for a in r.aspects], ensure_ascii=False),
                "rationale": r.rationale or "",
                "strategy": m.strategy,
                "provider": m.provider,
                "model_name": m.model_name,
                "prompt_version": m.prompt_version,
                "is_mock": m.is_mock,
                "validation_success": m.validation_success,
                "repair_attempts": m.repair_attempts,
                "local_repairs": m.local_repairs,
                "latency_s": m.total_latency_s,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "extracted_at": m.extracted_at,
            }
        )
    return pd.DataFrame(rows, columns=RECORD_CSV_COLUMNS)


def aspects_to_dataframe(records: Iterable[ReviewRecord]) -> pd.DataFrame:
    rows = [
        {
            "review_id": r.review_id,
            "product_id": r.product_id,
            "aspect_name": a.aspect_name,
            "aspect_category": a.aspect_category.value,
            "sentiment": a.sentiment.value,
            "evidence": a.evidence,
            "confidence": a.confidence,
            "strategy": r.extraction_metadata.strategy,
            "model_name": r.extraction_metadata.model_name,
            "is_mock": r.extraction_metadata.is_mock,
        }
        for r in records
        for a in r.aspects
    ]
    return pd.DataFrame(rows, columns=ASPECT_CSV_COLUMNS)


def failures_to_dataframe(failures: Iterable[ExtractionFailure]) -> pd.DataFrame:
    rows = [
        {
            "review_id": f.review_id,
            "product_id": f.product_id,
            "review_text": f.review_text,
            "error": f.error,
            "strategy": f.extraction_metadata.strategy,
            "model_name": f.extraction_metadata.model_name,
            "is_mock": f.extraction_metadata.is_mock,
            "repair_attempts": f.extraction_metadata.repair_attempts,
        }
        for f in failures
    ]
    return pd.DataFrame(rows, columns=FAILURE_CSV_COLUMNS)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_records(records: list[ReviewRecord], out_dir: Path, stem: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{stem}.json",
        "csv": out_dir / f"{stem}.csv",
        "aspects_csv": out_dir / f"{stem}_aspects.csv",
    }
    paths["json"].write_text(records_to_json(records), encoding="utf-8")
    records_to_dataframe(records).to_csv(paths["csv"], index=False, encoding="utf-8")
    aspects_to_dataframe(records).to_csv(paths["aspects_csv"], index=False, encoding="utf-8")
    return paths


ALL_ASPECT_CATEGORIES = [c.value for c in AspectCategory]
