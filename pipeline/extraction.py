"""Stage 1 — EXTRACT: raw review text -> validated ``ReviewRecord``.

Batchable and resumable: results are appended to a JSONL cache keyed by
(review id, text hash). Re-running skips reviews that already succeeded, so an
interrupted 300-review run continues where it stopped and no API call is wasted.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from config import Settings
from llm.base import LLMClient
from pipeline.repair import generate_structured
from pipeline.validation import normalize_extraction
from prompts.common import PROMPT_VERSION, extraction_format_instructions
from prompts.strategies import PromptStrategy
from schemas.review_schema import ExtractionFailure, ExtractionMetadata, LLMExtraction, ReviewInput, ReviewRecord
from utils.storage import append_jsonl, read_jsonl, slug

logger = logging.getLogger(__name__)

EXTRACTION_TASK_DESCRIPTION = (
    "Convert a customer product review into a JSON object with overall_sentiment, aspects "
    "(aspect_name, aspect_category, sentiment, verbatim evidence, confidence), complaint_categories, "
    "confidence and an optional short rationale, using only the allowed label values."
)

ExtractionOutcome = ReviewRecord | ExtractionFailure


def cache_key(review: ReviewInput) -> str:
    return f"{review.review_id}:{hashlib.sha1(review.review_text.encode()).hexdigest()[:10]}"


class ExtractionStore:
    """JSONL-backed cache of extraction outcomes for one (source, strategy, model)."""

    def __init__(self, path: Path):
        self.path = path
        self._ok: dict[str, ReviewRecord] = {}
        self._failed: dict[str, ExtractionFailure] = {}
        for row in read_jsonl(path):
            key = row.get("cache_key")
            if row.get("status") == "ok":
                self._ok[key] = ReviewRecord.model_validate(row["record"])
                self._failed.pop(key, None)
            elif row.get("status") == "failed" and key not in self._ok:
                self._failed[key] = ExtractionFailure.model_validate(row["record"])

    @classmethod
    def for_run(cls, settings: Settings, is_mock: bool, source: str, strategy: str, model_name: str) -> "ExtractionStore":
        name = f"{slug(source)}__{strategy}__{slug(model_name)}__{PROMPT_VERSION}.jsonl"
        return cls(settings.paths.mode_dir(is_mock) / "extractions" / name)

    def get(self, review: ReviewInput) -> ExtractionOutcome | None:
        key = cache_key(review)
        return self._ok.get(key) or self._failed.get(key)

    def put(self, review: ReviewInput, outcome: ExtractionOutcome) -> None:
        key = cache_key(review)
        status = "ok" if isinstance(outcome, ReviewRecord) else "failed"
        append_jsonl(self.path, [{"cache_key": key, "status": status, "record": outcome.model_dump(mode="json")}])
        if status == "ok":
            self._ok[key] = outcome  # type: ignore[assignment]
            self._failed.pop(key, None)
        else:
            self._failed[key] = outcome  # type: ignore[assignment]

    @property
    def records(self) -> list[ReviewRecord]:
        return list(self._ok.values())

    @property
    def failures(self) -> list[ExtractionFailure]:
        return list(self._failed.values())


@dataclass
class ExtractionRunResult:
    records: list[ReviewRecord] = field(default_factory=list)
    failures: list[ExtractionFailure] = field(default_factory=list)
    from_cache: int = 0
    new_calls: int = 0

    @property
    def total(self) -> int:
        return len(self.records) + len(self.failures)


class ReviewExtractor:
    def __init__(self, client: LLMClient, strategy: PromptStrategy, settings: Settings):
        self.client = client
        self.strategy = strategy
        self.settings = settings

    def extract(self, review: ReviewInput) -> ExtractionOutcome:
        messages = self.strategy.build_messages(review.review_text)
        result = generate_structured(
            self.client,
            messages,
            LLMExtraction,
            task="extraction",
            task_description=EXTRACTION_TASK_DESCRIPTION,
            format_instructions=extraction_format_instructions(),
            payload={"review_id": review.review_id, "review_text": review.review_text, "strategy": self.strategy.name},
            context={"review_text": review.review_text, "strict_grounding": self.settings.strict_evidence_grounding},
            source_text=review.review_text,
            max_repairs=self.settings.max_repair_attempts,
            normalizer=normalize_extraction,
        )
        metadata = ExtractionMetadata(
            strategy=self.strategy.name,
            provider=self.client.provider,
            model_name=self.client.model_name,
            prompt_version=self.strategy.version,
            is_mock=self.client.is_mock,
            validation_success=result.success,
            repair_attempts=result.repair_attempts,
            local_repairs=result.local_repairs,
            total_latency_s=result.total_latency_s,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            attempts=result.attempts,
        )
        if result.success:
            return ReviewRecord.from_extraction(review, result.value, metadata)  # type: ignore[arg-type]
        return ExtractionFailure(
            review_id=review.review_id,
            product_id=review.product_id,
            review_text=review.review_text,
            error=result.error or "unknown error",
            extraction_metadata=metadata,
        )

    def extract_many(
        self,
        reviews: Iterable[ReviewInput],
        store: ExtractionStore | None = None,
        progress: Callable[[int, int, ExtractionOutcome], None] | None = None,
        retry_failed: bool = True,
    ) -> ExtractionRunResult:
        reviews = list(reviews)
        run = ExtractionRunResult()
        for i, review in enumerate(reviews, 1):
            cached = store.get(review) if store else None
            if cached is not None and (isinstance(cached, ReviewRecord) or not retry_failed):
                outcome = cached
                run.from_cache += 1
            else:
                outcome = self.extract(review)
                run.new_calls += 1
                if store:
                    store.put(review, outcome)
            (run.records if isinstance(outcome, ReviewRecord) else run.failures).append(outcome)  # type: ignore[arg-type]
            if progress:
                progress(i, len(reviews), outcome)
        logger.info(
            "Extraction (%s, %s): %d ok, %d failed, %d cached, %d new",
            self.strategy.name, self.client.model_name, len(run.records), len(run.failures), run.from_cache, run.new_calls,
        )
        return run
