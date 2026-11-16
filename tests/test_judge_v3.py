"""Judge v3: one call, raw response kept, no repair, failures separate, claim-consistent scores, no score transfer."""

from __future__ import annotations

import json

import pytest

from evaluation.judge_v3 import (
    JudgeOutputV3, V3Store, judge_items, judge_once, parse_v3, prompt_sha256, reconcile_hallucination, rubric_note,
    run_config, summarise_v3,
)
from llm.base import LLMError
from llm.mock import MockLLMClient
from pipeline.summarization import run_chain
from prompts.judge import RUBRIC
from schemas.review_schema import ReviewInput
from scripts.judge_v3 import claim_detects, nc_report, original_vs_corrected, reason_only
from tests.conftest import ScriptedLLM


def v3(**over) -> dict:
    d = {
        "faithfulness_score": 4, "faithfulness_justification": "Mostly supported by E1 and E2.",
        "coverage_score": 4, "coverage_justification": "Main themes covered.",
        "consistency_score": 3, "consistency_justification": "Tone slightly too positive.",
        "unsupported_claims": [{"claim": "it is very quiet", "reason": "no review mentions noise", "evidence_review_ids": ["E0"]}],
        "hallucination_score": 4, "hallucination_justification": "One unsupported detail.",
        "overall_comment": "Reasonable summary with one unsupported detail.",
    }
    d.update(over)
    return d


@pytest.fixture
def chain(settings):
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good."]
    reviews = [ReviewInput(review_id=f"E{i}", product_id="P0", review_text=t) for i, t in enumerate(texts * 2)]
    return run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="few_shot", source="dataset", min_reviews_for_summary=1)


def _cfg(settings, client):
    return run_config(client, settings)


def test_v3_uses_the_unchanged_rubric_and_claims_before_score():
    from evaluation.judge_v3 import SYSTEM_V3

    assert RUBRIC.split("List every")[0].strip() in SYSTEM_V3
    keys = list(JudgeOutputV3.model_json_schema()["properties"])
    assert keys.index("unsupported_claims") < keys.index("hallucination_score")
    assert len(prompt_sha256()) == 64


def test_inconsistent_verdict_is_accepted_not_repaired():
    out, err = parse_v3(json.dumps(v3(hallucination_score=2, unsupported_claims=[])))
    assert err is None and out.hallucination_score == 2 and out.unsupported_claims == []


@pytest.mark.parametrize("raw,n,expect", [(5, 0, (5, "consistent")), (2, 3, (2, "consistent")),
                                          (4, 0, (None, "inconsistent_score_below_5_without_claims")),
                                          (5, 2, (None, "inconsistent_score_5_with_claims"))])
def test_inconsistent_verdicts_are_excluded_never_changed(raw, n, expect):
    assert reconcile_hallucination(raw, n) == expect


def test_rubric_note_flags_score_4_with_several_claims():
    assert rubric_note(4, 2) and not rubric_note(4, 1) and not rubric_note(3, 3)


def test_single_call_keeps_raw_and_records_reconciliation(settings, chain):
    summary = next(iter(chain.summaries.values()))
    raw = json.dumps(v3(hallucination_score=2, unsupported_claims=[]))
    client = ScriptedLLM([raw])
    ok, rec = judge_once(client, summary, chain.extraction.records, kind="original", config=_cfg(settings, client), run_id="t")
    assert ok and len(client.calls) == 1 and client.calls[0]["task"] == "judge_v3"
    assert rec["raw_response"] == raw and rec["hallucination_score_raw"] == 2
    assert rec["scores"]["hallucination"] is None and rec["scores"]["mean"] is None
    assert rec["hallucination_reconciliation"] == "inconsistent_score_below_5_without_claims"
    assert rec["unsupported_claims"] == []  # nothing invented
    assert rec["config"]["repair_attempts"] == 0 and rec["config"]["calls_per_summary"] == 1


def test_invalid_output_is_a_separate_failure_with_raw_and_no_scores(settings, chain, tmp_path):
    summary = next(iter(chain.summaries.values()))
    client = ScriptedLLM(["not json at all"])
    store = V3Store(tmp_path / "v3")
    stats = judge_items(client, [("original", summary, None)], chain.extraction.records, store, config=_cfg(settings, client), run_id="t")
    assert stats == {"judged": 0, "failed": 1, "skipped_stored": 0} and len(client.calls) == 1
    assert store.judgements() == []
    f = store.failures()[0]
    assert f["raw_response"] == "not json at all" and "scores" not in f and f["valid"] is False


def test_resume_never_repeats_valid_or_failed_items_and_kinds_are_separate(settings, chain, tmp_path):
    summary = next(iter(chain.summaries.values()))
    store = V3Store(tmp_path / "v3")
    c1 = ScriptedLLM([json.dumps(v3()), "broken"])
    judge_items(c1, [("original", summary, None), ("corrected", summary, None)], chain.extraction.records, store,
                config=_cfg(settings, c1), run_id="a")
    c2 = ScriptedLLM([json.dumps(v3())])
    stats = judge_items(c2, [("original", summary, None), ("corrected", summary, None), ("nc_corrupted", summary, None)],
                        chain.extraction.records, store, config=_cfg(settings, c2), run_id="b")
    assert stats == {"judged": 1, "failed": 0, "skipped_stored": 2} and len(c2.calls) == 1
    assert sorted(j["evaluated_object"] for j in store.judgements()) == ["nc_corrupted", "original"]


def test_provider_error_records_nothing(settings, chain, tmp_path):
    class Down(ScriptedLLM):
        def invoke(self, *a, **k):
            raise LLMError("ollama down")

    summary = next(iter(chain.summaries.values()))
    store = V3Store(tmp_path / "v3")
    with pytest.raises(LLMError):
        judge_items(Down([]), [("original", summary, None)], chain.extraction.records, store, config=_cfg(settings, ScriptedLLM([])), run_id="t")
    assert store.done() == set()


def test_unknown_kind_rejected(settings, chain):
    summary = next(iter(chain.summaries.values()))
    with pytest.raises(ValueError):
        judge_once(ScriptedLLM([json.dumps(v3())]), summary, chain.extraction.records, kind="mixed", config={}, run_id="t")


def _rec(kind, sid, pid, claims, scores, **extra):
    base = {"evaluated_object": kind, "summary_id": sid, "product_id": pid, "scores": scores,
            "unsupported_claims": [{"claim": c, "reason": ""} for c in claims], "justifications": {},
            "hallucination_score_raw": scores["hallucination"] or 4, "hallucination_reconciliation": "consistent" if scores["hallucination"] else "inconsistent_score_below_5_without_claims",
            "rubric_note": None, "claims_found_in_summary": len(claims), "judge_evidence_ids_unknown": []}
    base.update(extra)
    return base


S4 = {"faithfulness": 4, "coverage": 4, "consistency": 4, "hallucination": 4, "mean": 4.0}
S3 = {"faithfulness": 3, "coverage": 4, "consistency": 4, "hallucination": 3, "mean": 3.5}


def test_corrected_scores_are_never_taken_from_the_original():
    js = [_rec("original", "s1", "P1", ["80% positive"], S3), _rec("corrected", "s1-corr1", "P1", [], S4)]
    (row,) = original_vs_corrected(js)
    assert row["corrected_scores_v3"] == S4 and row["original_scores_v3"] == S3
    assert row["corrected_summary_id"] == "s1-corr1" and row["original_summary_id"] == "s1"
    assert summarise_v3(js, [], "corrected")["valid"] == 1 and summarise_v3(js, [], "original")["mean_scores"]["mean"] == 3.5


def test_v3_negative_control_uses_claim_text_only():
    o = _rec("original", "s1", "P1", [], S4)
    hit = _rec("nc_corrupted", "s1-nc-feature", "P1", ["Wi-Fi app control"], S3, nc_kind="invented_feature",
               planted_text="x", keywords=["wi-fi"], original_summary_id="s1")
    miss = _rec("nc_corrupted", "s1-nc-statistic", "P1", [], S4, nc_kind="fabricated_statistic", planted_text="y",
                keywords=["91%"], original_summary_id="s1", justifications={"h": "91% would buy again, as stated"})
    assert claim_detects(hit, hit["keywords"]) and not claim_detects(miss, miss["keywords"]) and reason_only(miss, miss["keywords"])
    rep = nc_report([o, hit, miss], [])
    assert rep["overall"]["detected"] == 1 and rep["overall"]["missed"] == 1
    assert rep["overall"]["keyword_only_in_reason_or_justification"] == 1
    assert rep["overall"]["faithfulness_lower"] == 1 and rep["keyword_control_on_v3_originals"]["named"] == 0


def test_excluded_hallucination_scores_are_left_out_of_means_and_pair_comparisons():
    none = {"faithfulness": 2, "coverage": 3, "consistency": 2, "hallucination": None, "mean": None}
    js = [_rec("original", "s1", "P1", [], none), _rec("original", "s2", "P2", ["x"], S3)]
    sm = summarise_v3(js, [], "original")
    assert sm["mean_scores"]["hallucination"] == 3 and sm["hallucination_consistent"] == 1
    assert sm["mean_scores"]["faithfulness"] == 2.5
    nc = _rec("nc_corrupted", "s1-nc-feature", "P1", [], S4, nc_kind="invented_feature", planted_text="x",
              keywords=["wi-fi"], original_summary_id="s1")
    rep = nc_report(js + [nc], [])
    assert rep["overall"]["pairs_comparable_on_hallucination"] == 0 and rep["overall"]["hallucination_lower"] == 0
