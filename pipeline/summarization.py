"""Stage 3 — SUMMARISE, and the full extract -> aggregate -> summarise chain."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import Settings
from llm.base import LLMClient
from pipeline.aggregation import aggregate_all
from pipeline.extraction import ExtractionRunResult, ExtractionStore, ReviewExtractor
from pipeline.repair import StructuredResult, generate_structured
from prompts.common import render
from prompts.strategies import get_strategy
from prompts.summarization import (
    SUMMARY_PROMPT_VERSION,
    TEMPLATE,
    allowed_review_ids,
    evidence_block,
    statistics_block,
    summary_format_instructions,
)
from schemas.evaluation_schema import ProductAggregate, ProductSummary, ProductSummaryContent
from schemas.review_schema import ReviewInput
from utils.storage import append_jsonl, log_activity, read_jsonl

logger = logging.getLogger(__name__)

SUMMARY_TASK_DESCRIPTION = (
    "Write a product-level summary of customer reviews as JSON (overall_opinion, appreciated, complaints, "
    "frequent_aspects, improvement_areas, recommendations) using only the provided statistics and evidence; "
    "every appreciated/complaint point must cite supporting review ids from the evidence."
)


def summarize_product(client: LLMClient, aggregate: ProductAggregate, settings: Settings, source_extraction: str) -> tuple[ProductSummary | None, StructuredResult]:
    stats, evidence = statistics_block(aggregate), evidence_block(aggregate)
    messages = render(
        TEMPLATE,
        format_instructions=summary_format_instructions(),
        product_title=aggregate.product_title or "unknown title",
        product_id=aggregate.product_id,
        total_reviews=aggregate.total_reviews,
        statistics=stats,
        evidence=evidence,
    )
    allowed = allowed_review_ids(aggregate)
    result = generate_structured(
        client,
        messages,
        ProductSummaryContent,
        task="summary",
        task_description=SUMMARY_TASK_DESCRIPTION,
        format_instructions=summary_format_instructions(),
        payload={"aggregate": aggregate.model_dump(mode="json")},
        context={"allowed_review_ids": allowed},
        source_text=f"STATISTICS:\n{stats}\n\nEVIDENCE:\n{evidence}",
        max_repairs=settings.max_repair_attempts,
    )
    if not result.success:
        return None, result
    sid = hashlib.sha1(f"{aggregate.product_id}|{source_extraction}|{client.model_name}|{result.attempts[-1].raw_output}".encode()).hexdigest()[:12]
    summary = ProductSummary(
        summary_id=sid,
        product_id=aggregate.product_id,
        product_title=aggregate.product_title,
        content=result.value,  # type: ignore[arg-type]
        aggregate=aggregate,
        source_extraction=source_extraction,
        provider=client.provider,
        model_name=client.model_name,
        is_mock=client.is_mock,
        prompt_version=SUMMARY_PROMPT_VERSION,
        repair_attempts=result.repair_attempts,
        latency_s=result.total_latency_s,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    return summary, result


class SummaryStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def for_mode(cls, settings: Settings, is_mock: bool) -> "SummaryStore":
        return cls(settings.paths.mode_dir(is_mock) / "summaries" / "summaries.jsonl")

    def add(self, summary: ProductSummary) -> None:
        append_jsonl(self.path, [summary.model_dump(mode="json")])

    def all(self) -> list[ProductSummary]:
        return [ProductSummary.model_validate(r) for r in read_jsonl(self.path)]

    def latest(self) -> dict[tuple[str, str], ProductSummary]:
        """Most recent summary per (product, extraction source)."""
        out: dict[tuple[str, str], ProductSummary] = {}
        for s in self.all():
            out[(s.product_id, s.source_extraction)] = s
        return out

    def get(self, summary_id: str) -> ProductSummary | None:
        return next((s for s in self.all() if s.summary_id == summary_id), None)


@dataclass
class ChainResult:
    extraction: ExtractionRunResult
    aggregates: dict[str, ProductAggregate]
    summaries: dict[str, ProductSummary] = field(default_factory=dict)
    summary_errors: dict[str, str] = field(default_factory=dict)
    reused_summaries: int = 0


def _reusable(prev: ProductSummary, client: LLMClient, aggregate: ProductAggregate) -> bool:
    return (
        prev.provider == client.provider
        and prev.model_name == client.model_name
        and prev.is_mock == client.is_mock
        and prev.prompt_version == SUMMARY_PROMPT_VERSION
        and prev.aggregate.model_dump(mode="json") == aggregate.model_dump(mode="json")
    )


def run_chain(
    client: LLMClient,
    reviews: list[ReviewInput],
    settings: Settings,
    strategy: str = "few_shot",
    source: str = "dataset",
    summarize: bool = True,
    min_reviews_for_summary: int = 3,
    progress: Callable[[str, int, int], None] | None = None,
    retry_failed: bool = True,
    reuse_summaries: bool = False,
) -> ChainResult:
    """extract -> aggregate -> summarise for every product in ``reviews``.

    ``retry_failed=False`` keeps cached output failures as failures instead of calling the model again.
    ``reuse_summaries=True`` (resume) reuses a stored summary instead of generating a new one, but only when it was
    produced from the identical aggregate with the same extraction source, provider, model and prompt version.
    """
    store = ExtractionStore.for_run(settings, client.is_mock, source, strategy, client.model_name)
    extractor = ReviewExtractor(client, get_strategy(strategy), settings)
    extraction = extractor.extract_many(
        reviews, store=store, progress=(lambda i, n, _o: progress("extract", i, n)) if progress else None, retry_failed=retry_failed,
    )
    aggregates = aggregate_all(extraction.records)
    result = ChainResult(extraction=extraction, aggregates=aggregates)
    if summarize:
        sstore = SummaryStore.for_mode(settings, client.is_mock)
        source_name = store.path.stem
        eligible = {pid: a for pid, a in aggregates.items() if a.total_reviews >= min_reviews_for_summary}
        stored = sstore.latest() if reuse_summaries else {}
        for i, (pid, agg) in enumerate(eligible.items(), 1):
            prev = stored.get((pid, source_name))
            if prev is not None and _reusable(prev, client, agg):
                result.summaries[pid] = prev
                result.reused_summaries += 1
                if progress:
                    progress("summarise", i, len(eligible))
                continue
            summary, sres = summarize_product(client, agg, settings, source_name)
            if summary:
                sstore.add(summary)
                result.summaries[pid] = summary
            else:
                result.summary_errors[pid] = sres.error or "unknown error"
            if progress:
                progress("summarise", i, len(eligible))
    log_activity(
        settings.paths.activity_log, "prompt_chain", client.is_mock, strategy=strategy, source=source,
        reviews=len(reviews), products=len(aggregates), summaries=len(result.summaries),
    )
    return result
