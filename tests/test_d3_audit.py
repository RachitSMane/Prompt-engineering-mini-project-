"""Deterministic D3 audits (evidence traceability, judge-quality heuristics) and the GPU/memory guard. Offline."""

from __future__ import annotations

import json

import pytest

from evaluation.d3_audit import audit_judgement, audit_summary, claim_polarity
from evaluation.summary_evaluation import judge_summary
from llm.mock import MockLLMClient
from pipeline.summarization import run_chain
from schemas.evaluation_schema import SummaryPoint
from schemas.review_schema import ReviewInput
from tests.conftest import ScriptedLLM
from utils import gpu_guard


@pytest.fixture
def chain(settings):
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good.",
             "Terrible value, overpriced and it stopped working.", "Easy to install and works well."]
    reviews = [ReviewInput(review_id=f"E{i}", product_id="P0", review_text=t) for i, t in enumerate(texts * 2)]
    return run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="few_shot", source="dataset", min_reviews_for_summary=1)


def test_clean_summary_passes_id_and_statistics_checks(chain):
    summary = next(iter(chain.summaries.values()))
    res = audit_summary(summary, chain.extraction.records)
    assert res["unknown_ids"] == [] and res["wrong_product_ids"] == []
    assert res["statistics_match_source"] is True and res["stat_checks"]["sentiment_percentages"]
    assert res["points_traceable"] + res["points_partial"] + res["points_weak"] == len(summary.content.appreciated) + len(summary.content.complaints)


def test_unknown_and_foreign_ids_are_reported(chain):
    summary = next(iter(chain.summaries.values()))
    records = list(chain.extraction.records)
    foreign = records[0].model_copy(update={"review_id": "X9", "product_id": "OTHER"})
    bad = summary.model_copy(update={"content": summary.content.model_copy(update={
        "appreciated": [SummaryPoint(point="Great product overall", supporting_review_ids=["NOPE", "X9"])]})})
    res = audit_summary(bad, records + [foreign])
    assert res["unknown_ids"] == ["NOPE"] and res["wrong_product_ids"] == ["X9"] and not res["ids_and_statistics_verified"]


def test_tampered_statistics_and_fabricated_percentage_are_reported(chain):
    summary = next(iter(chain.summaries.values()))
    agg = summary.aggregate.model_copy(update={"sentiment_percentages": {**summary.aggregate.sentiment_percentages, "positive": 99.9}})
    content = summary.content.model_copy(update={"overall_opinion": summary.content.overall_opinion + " Fully 77.7% of buyers love it."})
    res = audit_summary(summary.model_copy(update={"aggregate": agg, "content": content}), chain.extraction.records)
    assert res["stat_checks"]["sentiment_percentages"] is False and 77.7 in res["percentages_not_in_statistics"]
    assert any("aggregate mismatch" in i for i in res["issues"])


def test_point_with_wrong_direction_and_no_overlap_is_weak(chain):
    summary = next(iter(chain.summaries.values()))
    neg_ids = [r.review_id for r in chain.extraction.records if r.overall_sentiment.value == "negative"][:1]
    if not neg_ids:
        pytest.skip("mock produced no negative record")
    bad = summary.model_copy(update={"content": summary.content.model_copy(update={
        "appreciated": [SummaryPoint(point="Stunning chrome finish admired by chefs", supporting_review_ids=neg_ids)]})})
    res = audit_summary(bad, chain.extraction.records)
    assert res["points"][0]["status"] == "weak" and res["points_weak"] >= 1


def test_claim_polarity():
    assert claim_polarity("the product did not provide good value for money") == "negative"
    assert claim_polarity("The basket is less sturdy than the original") == "negative"
    assert claim_polarity("The product offers good value for money") == "positive"


def test_judge_flagging_a_source_supported_complaint_is_a_likely_false_positive(settings, chain):
    summary = next(iter(chain.summaries.values()))
    assert summary.aggregate.complaint_counts.get("poor_value", 0) > 0 or summary.aggregate.complaint_counts  # mock has complaints
    cat = next(iter(summary.aggregate.complaint_counts))
    claim_text = {"poor_value": "the product did not provide good value", "noise_issue": "the product is too loud",
                  "poor_durability": "the product broke and did not last", "performance_issue": "the product did not work well"}.get(cat, "poor value issue")
    content = summary.content.model_copy(update={"overall_opinion": summary.content.overall_opinion + " " + claim_text + "."})
    s2 = summary.model_copy(update={"content": content})
    out = {"faithfulness_score": 3, "faithfulness_justification": "x x x x x", "coverage_score": 4, "coverage_justification": "x x x x x",
           "consistency_score": 3, "consistency_justification": "x x x x x", "hallucination_score": 4, "hallucination_justification": "x x x x x",
           "unsupported_claims": [{"claim": claim_text, "reason": "r", "evidence_review_ids": ["E1", "ZZZ"]}], "overall_comment": "x x x x x"}
    j, _ = judge_summary(ScriptedLLM([json.dumps(out)]), s2, chain.extraction.records, settings, version="v2")
    res = audit_judgement(j, s2)
    assert res["scores_valid"] and res["justifications_present"] and res["hallucination_consistent_with_claims"]
    assert res["judge_evidence_ids_valid"] is False  # ZZZ does not exist
    if cat in ("poor_value", "noise_issue", "poor_durability", "performance_issue"):
        assert res["claims"][0]["likely_false_positive"] is True


def test_guard_stops_on_low_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_guard, "free_ram_mb", lambda: 500)
    monkeypatch.setattr(gpu_guard, "read_gpu_temp", lambda: 50)
    g = gpu_guard.GpuGuard(tmp_path / "g.csv", min_free_ram_mb=2048)
    with pytest.raises(gpu_guard.LowMemory):
        g.wait_until_safe()
    assert "low_memory" in (tmp_path / "g.csv").read_text()
    assert issubclass(gpu_guard.LowMemory, gpu_guard.GpuTooHot)  # callers stop on it exactly like on heat


def test_guard_aborts_hot_and_passes_cool(monkeypatch):
    monkeypatch.setattr(gpu_guard, "free_ram_mb", lambda: 10_000)
    monkeypatch.setattr(gpu_guard, "read_gpu_temp", lambda: 95)
    with pytest.raises(gpu_guard.GpuTooHot):
        gpu_guard.GpuGuard(None).wait_until_safe()
    monkeypatch.setattr(gpu_guard, "read_gpu_temp", lambda: 60)
    assert gpu_guard.GpuGuard(None).wait_until_safe() == 60
    monkeypatch.setattr(gpu_guard, "read_gpu_temp", lambda: None)
    assert gpu_guard.GpuGuard(None).wait_until_safe() is None


def _with_opinion(summary, agg_update: dict, sentence: str):
    agg = summary.aggregate.model_copy(update=agg_update)
    return summary.model_copy(update={"aggregate": agg, "content": summary.content.model_copy(update={"overall_opinion": sentence})})


def test_count_statements_checked_against_total_and_misleading_denominator(chain):
    """Regression (B001DHKBXU): 'poor durability (8 out of 15 reviews)' where 15 = complaining reviews, total = 19."""
    summary = next(iter(chain.summaries.values()))
    agg = {"total_reviews": 19, "reviews_with_complaints": 15, "complaint_counts": {"poor_durability": 8}}
    ok = audit_summary(_with_opinion(summary, agg, "15 out of 19 reviews contain complaints; durability (8 of 19 reviews)."), chain.extraction.records)
    assert [c["status"] for c in ok["count_statements"]] == ["ok", "ok"] and ok["count_statements_bad"] == 0
    bad = audit_summary(_with_opinion(summary, agg, "Poor durability (8 out of 15 reviews) and 13 of 19 reviews praise it."), chain.extraction.records)
    assert [c["status"] for c in bad["count_statements"]] == ["misleading_denominator", "not_in_statistics"]
    assert not bad["ids_and_statistics_verified"]
    assert any("misleading" in i for i in bad["issues"]) and any("counts not in statistics" in i for i in bad["issues"])


def test_corrected_copy_is_separate_and_requires_exact_single_match(chain):
    from scripts.build_corrected_summaries import correct

    summary = next(iter(chain.summaries.values()))
    s = summary.model_copy(update={"content": summary.content.model_copy(update={"overall_opinion": "Durability (8 out of 15 reviews)."})})
    fixed, applied = correct(s, [("(8 out of 15 reviews)", "(8 out of 19 reviews)", "test")])
    assert fixed.summary_id == f"{s.summary_id}-corr1" and fixed.content.overall_opinion == "Durability (8 out of 19 reviews)."
    assert s.content.overall_opinion == "Durability (8 out of 15 reviews)." and applied[0]["source"] == "test"  # original untouched
    with pytest.raises(ValueError, match="exactly one"):
        correct(s, [("(9 out of 15 reviews)", "x", "test")])
