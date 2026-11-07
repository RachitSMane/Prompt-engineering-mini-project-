"""Pydantic v2 models for structured review extraction.

Two layers:
  * ``LLMExtraction`` – exactly what the LLM must return (validated, grounded).
  * ``ReviewRecord``  – the final stored record = identifiers + extraction + metadata.

Evidence grounding: when validated with ``context={"review_text": ...}``, every
aspect's ``evidence`` must be a (near-)verbatim quote from the review. This is
how we enforce "do not invent aspects / no sentiment without textual evidence".
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from schemas.taxonomy import AspectCategory, AspectSentiment, ComplaintCategory, OverallSentiment

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def evidence_is_grounded(evidence: str, review_text: str, threshold: float = 0.85) -> bool:
    """True if ``evidence`` appears (almost) verbatim in ``review_text``.

    Exact match after normalisation (case, punctuation, whitespace), otherwise a
    fuzzy sliding-window match so trivial LLM typos/ellipses do not fail.
    """
    ev, src = normalize_for_match(evidence), normalize_for_match(review_text)
    if not ev:
        return False
    if ev in src:
        return True
    # LLMs often join two fragments with "..." — accept if every fragment is grounded
    fragments = [f for f in re.split(r"\.\.\.|…", evidence) if normalize_for_match(f)]
    if len(fragments) > 1 and all(normalize_for_match(f) in src for f in fragments):
        return True
    ev_words, src_words = ev.split(), src.split()
    n = len(ev_words)
    if n == 0 or len(src_words) < n:
        return SequenceMatcher(None, ev, src).ratio() >= threshold
    best = 0.0
    for start in range(0, len(src_words) - n + 1):
        window = " ".join(src_words[start : start + n])
        best = max(best, SequenceMatcher(None, ev, window).ratio())
        if best >= threshold:
            return True
    return False


class Aspect(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    aspect_name: str = Field(..., min_length=1, max_length=60, description="Short name of the aspect as discussed, e.g. 'ice maker', 'noise level'.")
    aspect_category: AspectCategory = Field(..., description="Category from the aspect taxonomy.")
    sentiment: AspectSentiment = Field(..., description="Sentiment expressed toward this aspect.")
    evidence: str = Field(..., min_length=3, max_length=300, description="Verbatim quote from the review supporting the aspect and sentiment.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this aspect judgement (0-1).")

    @field_validator("aspect_name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = _WS_RE.sub(" ", v).strip().lower()
        if not v:
            raise ValueError("aspect_name must not be blank")
        return v

    @field_validator("evidence")
    @classmethod
    def _grounded(cls, v: str, info: ValidationInfo) -> str:
        v = v.strip().strip('"').strip()
        ctx = info.context or {}
        review_text = ctx.get("review_text")
        if review_text and ctx.get("strict_grounding", True) and not evidence_is_grounded(v, review_text):
            raise ValueError(
                f"evidence {v[:80]!r} is not a verbatim quote from the review text; "
                "copy the exact words from the review"
            )
        return v


class LLMExtraction(BaseModel):
    """The structured object every prompting strategy must return."""

    model_config = ConfigDict(extra="forbid")

    overall_sentiment: OverallSentiment = Field(..., description="Overall sentiment of the whole review.")
    aspects: list[Aspect] = Field(default_factory=list, description="Product aspects explicitly discussed. Empty list if none.")
    complaint_categories: list[ComplaintCategory] = Field(default_factory=list, description="Complaint categories present. Empty list if no complaint.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in this extraction (0-1).")
    rationale: str | None = Field(default=None, max_length=400, description="Optional one- or two-sentence justification. No step-by-step reasoning.")

    @field_validator("complaint_categories")
    @classmethod
    def _dedupe_complaints(cls, v: list[ComplaintCategory]) -> list[ComplaintCategory]:
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _dedupe_aspects(self) -> "LLMExtraction":
        seen: set[tuple[str, str, str]] = set()
        unique: list[Aspect] = []
        for a in self.aspects:
            key = (a.aspect_name, a.aspect_category.value, a.sentiment.value)
            if key not in seen:
                seen.add(key)
                unique.append(a)
        self.aspects = unique
        return self


class AttemptLog(BaseModel):
    attempt: int
    kind: str  # "initial" | "llm_repair" | "provider_error"
    raw_output: str
    success: bool
    errors: list[str] = Field(default_factory=list)
    local_fixes: list[str] = Field(default_factory=list, description="Deterministic fixes applied before validation (no LLM call).")
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


class ExtractionMetadata(BaseModel):
    strategy: str
    provider: str
    model_name: str
    prompt_version: str
    is_mock: bool
    validation_success: bool
    repair_attempts: int = Field(0, description="Number of LLM repair calls made.")
    local_repairs: int = Field(0, description="Number of deterministic (no-LLM) fixes applied.")
    total_latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts: list[AttemptLog] = Field(default_factory=list)
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ReviewInput(BaseModel):
    """A raw review to be analysed."""

    review_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    review_text: str = Field(..., min_length=1)
    product_title: str | None = None
    rating: float | None = None
    timestamp: Any | None = None


class ReviewRecord(BaseModel):
    review_id: str
    product_id: str
    review_text: str
    overall_sentiment: OverallSentiment
    aspects: list[Aspect]
    complaint_categories: list[ComplaintCategory]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str | None = None
    product_title: str | None = None
    rating: float | None = None
    timestamp: Any | None = None
    extraction_metadata: ExtractionMetadata

    @classmethod
    def from_extraction(cls, review: ReviewInput, extraction: LLMExtraction, metadata: ExtractionMetadata) -> "ReviewRecord":
        return cls(
            review_id=review.review_id,
            product_id=review.product_id,
            review_text=review.review_text,
            product_title=review.product_title,
            rating=review.rating,
            timestamp=review.timestamp,
            overall_sentiment=extraction.overall_sentiment,
            aspects=extraction.aspects,
            complaint_categories=extraction.complaint_categories,
            confidence=extraction.confidence,
            rationale=extraction.rationale,
            extraction_metadata=metadata,
        )


class ExtractionFailure(BaseModel):
    """Controlled error returned when no valid output could be obtained."""

    review_id: str
    product_id: str
    review_text: str
    error: str
    extraction_metadata: ExtractionMetadata
