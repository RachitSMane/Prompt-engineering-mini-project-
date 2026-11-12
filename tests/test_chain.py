"""Aggregation, summarisation (grounding of citations) and LLM-as-a-Judge."""

import json

import pytest
from pydantic import ValidationError

from evaluation.summary_evaluation import JudgementStore, deterministic_checks, export_inspection_sample, judge_summary
from llm.mock import MockLLMClient
from pipeline.aggregation import aggregate_all, aggregate_product
from pipeline.summarization import SummaryStore, run_chain, summarize_product
from schemas.evaluation_schema import JudgeVerdict, ProductSummaryContent
from schemas.review_schema import Aspect, ExtractionMetadata, ReviewInput, ReviewRecord
from tests.conftest import ScriptedLLM


def _meta():
    return ExtractionMetadata(strategy="few_shot", provider="t", model_name="t", prompt_version="v1", is_mock=True, validation_success=True)


def _rec(rid, pid, overall, aspects, complaints=(), rating=None, ts="2022-05-01"):
    return ReviewRecord(
        review_id=rid, product_id=pid, review_text=" ".join(e for _, _, e in aspects) or "nothing", overall_sentiment=overall,
        aspects=[Aspect(aspect_name=c, aspect_category=c, sentiment=s, evidence=e, confidence=0.9) for c, s, e in aspects],
        complaint_categories=list(complaints), confidence=0.9, rating=rating, timestamp=ts, extraction_metadata=_meta(),
    )


@pytest.fixture
def records():
    return [
        _rec("r1", "P1", "positive", [("performance", "positive", "works great"), ("noise", "positive", "very quiet")], rating=5),
        _rec("r2", "P1", "negative", [("noise", "negative", "too loud"), ("quality_durability", "negative", "broke fast")], ["noise_issue", "poor_durability"], rating=1),
        _rec("r3", "P1", "mixed", [("performance", "positive", "cools well"), ("noise", "negative", "rattles a lot")], ["noise_issue"], rating=3, ts="2023-01-01"),
        _rec("r4", "P1", "neutral", [], rating=None),
        _rec("x1", "P2", "positive", [("price_value", "positive", "cheap price")], rating=5),
    ]


def test_aggregation_statistics(records):
    agg = aggregate_product([r for r in records if r.product_id == "P1"])
    assert agg.total_reviews == 4
    assert agg.sentiment_counts == {"positive": 1, "negative": 1, "neutral": 1, "mixed": 1}
    assert agg.sentiment_percentages["positive"] == 25.0
    noise = next(s for s in agg.aspect_stats if s.aspect_category == "noise")
    assert (noise.mentions, noise.positive, noise.negative) == (3, 1, 2)
    assert noise.net_sentiment == pytest.approx(-1 / 3, abs=1e-3)
    assert agg.aspect_stats[0].aspect_category == "noise"  # most mentioned first
    assert agg.complaint_counts == {"noise_issue": 2, "poor_durability": 1}
    assert agg.complaint_percentages["noise_issue"] == 50.0
    assert agg.reviews_with_complaints == 2
    assert agg.average_rating == 3.0
    assert {t["period"] for t in agg.aspect_trend} == {2022, 2023}
    assert {q.review_id for q in agg.negative_evidence} <= {"r2", "r3"}


def test_aggregate_all_groups_products(records):
    aggs = aggregate_all(records)
    assert set(aggs) == {"P1", "P2"} and aggs["P2"].total_reviews == 1


def test_aggregate_rejects_mixed_or_empty(records):
    with pytest.raises(ValueError):
        aggregate_product(records)
    with pytest.raises(ValueError):
        aggregate_product([])


def _summary_json(ids):
    return json.dumps({
        "overall_opinion": "Opinions are split: 25.0% positive and 25.0% negative.",
        "appreciated": [{"point": "Cooling performance is praised.", "supporting_review_ids": [ids[0]]}],
        "complaints": [{"point": "Noise is the most common complaint.", "supporting_review_ids": [ids[1]]}],
        "frequent_aspects": ["noise", "performance"],
        "improvement_areas": ["Reduce operating noise."],
        "recommendations": ["Investigate rattling reports."],
    })


def test_summary_rejects_unknown_review_ids_then_repairs(settings, records):
    agg = aggregate_product([r for r in records if r.product_id == "P1"])
    good_ids = sorted({q.review_id for q in agg.positive_evidence + agg.negative_evidence})
    client = ScriptedLLM([_summary_json(["r999", "r2"]), _summary_json(good_ids[:2])])
    summary, res = summarize_product(client, agg, settings, "src")
    assert summary is not None and res.repair_attempts == 1
    assert "unknown review ids" in res.attempts[0].errors[0]
    assert summary.aggregate.total_reviews == 4  # statistics come from code, attached to the summary


def test_summary_schema_requires_citations():
    with pytest.raises(ValidationError):
        ProductSummaryContent.model_validate({"overall_opinion": "fine product overall", "appreciated": [{"point": "good", "supporting_review_ids": []}]})


def test_deterministic_checks_flag_invented_percentages(settings, records):
    agg = aggregate_product([r for r in records if r.product_id == "P1"])
    ids = sorted({q.review_id for q in agg.positive_evidence + agg.negative_evidence})
    bad = json.loads(_summary_json(ids))
    bad["overall_opinion"] = "An impressive 87% of buyers love it; 25.0% are positive."
    summary, _ = summarize_product(ScriptedLLM([json.dumps(bad)]), agg, settings, "src")
    checks = deterministic_checks(summary)
    assert checks["percentages_not_in_statistics"] == [87.0]
    assert "noise" in checks["top_aspects_mentioned"]


def test_judge_structured_verdict_and_repair(settings, records):
    agg = aggregate_product([r for r in records if r.product_id == "P1"])
    ids = sorted({q.review_id for q in agg.positive_evidence + agg.negative_evidence})
    summary, _ = summarize_product(ScriptedLLM([_summary_json(ids)]), agg, settings, "src")
    verdict = {
        "faithfulness": {"score": 4, "justification": "Supported by r2 and r3."},
        "coverage": {"score": 4, "justification": "Covers noise and performance."},
        "consistency": {"score": 5, "justification": "Matches the 25% split."},
        "hallucination": {"score": 5, "justification": "No unsupported claims."},
        "unsupported_claims": [],
        "overall_comment": "Accurate summary.",
    }
    bad = dict(verdict, coverage={"score": 7, "justification": "too high"})
    client = ScriptedLLM([json.dumps(bad), json.dumps(verdict)])
    j, res = judge_summary(client, summary, records, settings)
    assert j is not None and res.repair_attempts == 1 and j.verdict.mean_score == 4.5
    prompt = "\n".join(m.content for m in client.calls[0]["messages"])
    for must in ("ORIGINAL REVIEWS", "STRUCTURED EXTRACTIONS", "AGGREGATED STATISTICS", "SUMMARY TO EVALUATE", "[r2]"):
        assert must in prompt
    assert "[x1]" not in prompt  # only this product's reviews
    store = JudgementStore(settings.paths.outputs / "j.jsonl")
    store.add(j)
    assert store.latest_by_summary()[summary.summary_id].verdict.consistency.score == 5
    md = export_inspection_sample([j], {summary.summary_id: summary})
    assert "human agrees?" in md and "MOCK" in md  # ScriptedLLM is flagged mock


def test_judge_scores_bounded():
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate({"faithfulness": {"score": 0, "justification": "xxxxx"}})


def test_full_chain_with_mock(settings):
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good.", "Installation was easy but it leaks."]
    reviews = [ReviewInput(review_id=f"C{i}", product_id=f"P{i % 2}", review_text=t) for i, t in enumerate(texts * 2)]
    res = run_chain(MockLLMClient(malformed_rate=30), reviews, settings, strategy="zero_shot", source="unit", min_reviews_for_summary=1)
    assert len(res.extraction.records) == 8 and set(res.aggregates) == {"P0", "P1"}
    assert set(res.summaries) == {"P0", "P1"} and all(s.is_mock for s in res.summaries.values())
    assert len(SummaryStore.for_mode(settings, True).all()) == 2
    assert (settings.paths.outputs / "mock" / "extractions").exists() and not (settings.paths.outputs / "live").exists()
    j, _ = judge_summary(MockLLMClient(), next(iter(res.summaries.values())), res.extraction.records, settings)
    assert j.is_mock and "MOCK" in j.verdict.overall_comment


def test_corrupted_summary_is_a_distinct_unfaithful_copy(settings, records):
    from evaluation.summary_evaluation import CORRUPTION_FABRICATED_FEATURE, corrupt_summary

    agg = aggregate_product([r for r in records if r.product_id == "P1"])
    ids = sorted({q.review_id for q in agg.positive_evidence + agg.negative_evidence})
    summary, _ = summarize_product(ScriptedLLM([_summary_json(ids)]), agg, settings, "src")
    bad = corrupt_summary(summary)
    assert bad.summary_id == f"{summary.summary_id}-corrupted" and summary.content.complaints  # original untouched
    assert bad.content.complaints == []
    assert bad.content.appreciated[0].point == CORRUPTION_FABRICATED_FEATURE
    assert bad.content.appreciated[0].supporting_review_ids[0] in agg.review_ids
    assert "%" in bad.content.overall_opinion and bad.content.overall_opinion != summary.content.overall_opinion
    assert deterministic_checks(bad)["percentages_not_in_statistics"]  # the fabricated number is caught deterministically too


def test_chain_reuses_seeded_predictions_without_llm_calls(settings, tmp_path):
    from pipeline.extraction import ExtractionStore

    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good."]
    reviews = [ReviewInput(review_id=f"S{i}", product_id="P0", review_text=t) for i, t in enumerate(texts)]
    first = run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="zero_shot", source="seedsrc", summarize=False)
    assert first.extraction.new_calls == 3
    src = ExtractionStore.for_run(settings, True, "seedsrc", "zero_shot", MockLLMClient().model_name).path
    target = ExtractionStore.for_run(settings, True, "seeded", "zero_shot", MockLLMClient().model_name)
    added = target.seed_from(src, review_ids={"S0", "S1"})
    assert added == {"ok": 2, "failed": 0, "skipped_existing": 0}
    assert target.seed_from(src, review_ids={"S0"})["skipped_existing"] == 1  # idempotent

    class NoCalls(MockLLMClient):
        def invoke(self, *a, **k):
            raise AssertionError("seeded reviews must not be sent to the model")

    second = run_chain(NoCalls(), reviews[:2], settings, strategy="zero_shot", source="seeded", summarize=False)
    assert second.extraction.from_cache == 2 and second.extraction.new_calls == 0


def test_chain_can_keep_cached_failures(settings):
    from pipeline.extraction import ExtractionStore
    from schemas.review_schema import ExtractionFailure

    review = ReviewInput(review_id="F1", product_id="P0", review_text="Totally broke on day one, awful.")
    store = ExtractionStore.for_run(settings, True, "failsrc", "zero_shot", MockLLMClient().model_name)
    store.put(review, ExtractionFailure(review_id="F1", product_id="P0", review_text=review.review_text, error="invalid", extraction_metadata=_meta()))

    class NoCalls(MockLLMClient):
        def invoke(self, *a, **k):
            raise AssertionError("cached failure must not be retried")

    res = run_chain(NoCalls(), [review], settings, strategy="zero_shot", source="failsrc", summarize=False, retry_failed=False)
    assert len(res.extraction.failures) == 1 and res.extraction.new_calls == 0


def test_chain_resume_reuses_matching_summaries_without_llm_calls(settings):
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good.", "Installation was easy but it leaks."]
    reviews = [ReviewInput(review_id=f"U{i}", product_id=f"P{i % 2}", review_text=t) for i, t in enumerate(texts * 2)]
    first = run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="zero_shot", source="resume", min_reviews_for_summary=1)
    assert len(first.summaries) == 2 and first.reused_summaries == 0

    class NoCalls(MockLLMClient):
        def invoke(self, *a, **k):
            raise AssertionError("stored summaries must be reused on resume")

    again = run_chain(NoCalls(), reviews, settings, strategy="zero_shot", source="resume", min_reviews_for_summary=1, reuse_summaries=True)
    assert again.reused_summaries == 2 and again.summary_errors == {}
    assert {p: s.summary_id for p, s in again.summaries.items()} == {p: s.summary_id for p, s in first.summaries.items()}
    assert len(SummaryStore.for_mode(settings, True).all()) == 2  # no duplicate rows written


def test_chain_resume_regenerates_when_aggregate_changed(settings):
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good."]
    reviews = [ReviewInput(review_id=f"V{i}", product_id="P0", review_text=t) for i, t in enumerate(texts)]
    run_chain(MockLLMClient(malformed_rate=0), reviews[:2], settings, strategy="zero_shot", source="changed", min_reviews_for_summary=1)
    res = run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="zero_shot", source="changed", min_reviews_for_summary=1, reuse_summaries=True)
    assert res.reused_summaries == 0 and len(res.summaries) == 1  # different input -> a new summary, not a stale one
