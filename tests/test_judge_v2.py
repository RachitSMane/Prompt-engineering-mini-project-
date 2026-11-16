"""Judge v2: flat schema, structural normaliser, validation, repair, bounded retries, resume, traceability,
negative-control planting. Offline: scripted / mock clients only."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evaluation.negative_control import FEATURE_POOL, build_planted, planted_claim_detected, plant_invented_feature
from evaluation.summary_evaluation import JudgementStore, JudgeProviderStopped, claim_in_text, judge_summaries, judge_summary
from llm.base import LLMError, LLMResponse
from llm.mock import MockLLMClient
from pipeline.summarization import run_chain
from pipeline.validation import normalize_judge_v2, parse_json
from schemas.evaluation_schema import JudgeOutputV2, JudgeVerdict
from schemas.review_schema import ReviewInput
from tests.conftest import ScriptedLLM


def v2(**over) -> dict:
    d = {
        "faithfulness_score": 4, "faithfulness_justification": "Mostly supported by R1 and R2.",
        "coverage_score": 4, "coverage_justification": "Main themes covered.",
        "consistency_score": 3, "consistency_justification": "Tone slightly too positive.",
        "hallucination_score": 4, "hallucination_justification": "One unsupported detail.",
        "unsupported_claims": [{"claim": "it is very quiet", "reason": "no review mentions noise", "evidence_review_ids": ["E0"]}],
        "overall_comment": "Reasonable summary with one unsupported detail.",
    }
    d.update(over)
    return d


@pytest.fixture
def chain(settings):
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good."]
    reviews = [ReviewInput(review_id=f"E{i}", product_id="P0", review_text=t) for i, t in enumerate(texts * 2)]
    res = run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="few_shot", source="dataset", min_reviews_for_summary=1)
    return res


# ------------------------------------------------------------------ schema
def test_flat_schema_accepts_valid_and_converts():
    out = JudgeOutputV2.model_validate(v2())
    verdict = JudgeVerdict.from_v2(out)
    assert verdict.faithfulness.score == 4 and verdict.unsupported_claims == ["it is very quiet"]
    assert verdict.unsupported_claim_details[0].evidence_review_ids == ["E0"]


@pytest.mark.parametrize("bad, msg", [
    ({"faithfulness_score": 7}, "less than or equal to 5"),
    ({"coverage_score": 0}, "greater than or equal to 1"),
    ({"consistency_score": 3.5}, "whole number"),
    ({"faithfulness_score": "4/5"}, "valid integer"),
    ({"hallucination_score": True}, "boolean"),
    ({"coverage_justification": ""}, "at least 5"),
])
def test_invalid_scores_and_justifications_rejected(bad, msg):
    with pytest.raises(ValidationError, match=msg):
        JudgeOutputV2.model_validate(v2(**bad))


def test_missing_field_rejected():
    d = v2()
    del d["consistency_justification"]
    with pytest.raises(ValidationError, match="consistency_justification"):
        JudgeOutputV2.model_validate(d)


def test_whole_number_strings_and_floats_are_accepted():
    assert JudgeOutputV2.model_validate(v2(faithfulness_score="4", coverage_score=3.0)).coverage_score == 3


def test_hallucination_score_must_agree_with_claims():
    with pytest.raises(ValidationError, match="unsupported_claims is empty"):
        JudgeOutputV2.model_validate(v2(hallucination_score=2, unsupported_claims=[]))
    with pytest.raises(ValidationError, match="5 means no unsupported claims"):
        JudgeOutputV2.model_validate(v2(hallucination_score=5))
    assert JudgeOutputV2.model_validate(v2(hallucination_score=5, unsupported_claims=[])).hallucination_score == 5


def test_stored_v1_judgements_still_load():
    v1 = {"faithfulness": {"score": 4, "justification": "fine fine"}, "coverage": {"score": 4, "justification": "fine fine"},
          "consistency": {"score": 3, "justification": "fine fine"}, "hallucination": {"score": 2, "justification": "fine fine"},
          "unsupported_claims": [], "overall_comment": "fine fine"}
    assert JudgeVerdict.model_validate(v1).unsupported_claim_details == []


# ------------------------------------------------------------------ parsing / normaliser
def test_normaliser_fixes_the_v1_nesting_failure():
    """The exact failure seen with llama3.2: unsupported_claims copied into every dimension object."""
    nested = {d: {"score": 4, "justification": f"{d} ok ok", "unsupported_claims": ["it is very quiet"]} for d in ("faithfulness", "coverage", "consistency", "hallucination")}
    nested["overall_comment"] = "Overall fine."
    obj, notes = normalize_judge_v2(nested)
    out = JudgeOutputV2.model_validate(obj)
    assert out.faithfulness_score == 4 and [c.claim for c in out.unsupported_claims] == ["it is very quiet"]  # de-duplicated
    assert "moved nested unsupported_claims to the top level" in notes and "flattened nested dimension objects" in notes


def test_normaliser_never_invents_values():
    obj, _ = normalize_judge_v2({"faithfulness": 4, "overall_comment": "x"})  # bare integer, no justification
    with pytest.raises(ValidationError):
        JudgeOutputV2.model_validate(obj)
    obj, notes = normalize_judge_v2(v2(unsupported_claims=None, hallucination_score=5))
    assert obj["unsupported_claims"] == [] and "replaced null unsupported_claims with []" in notes


def test_markdown_fence_extra_text_and_wrapper_are_handled():
    raw = "Here is my evaluation:\n```json\n" + json.dumps({"evaluation": v2()}) + "\n```\nThanks!"
    obj, fixes, err = parse_json(raw)
    assert err is None and "removed markdown code fence" in fixes
    obj, notes = normalize_judge_v2(obj)
    assert JudgeOutputV2.model_validate(obj).coverage_score == 4 and "unwrapped nested wrapper object" in notes


def test_plain_string_claims_converted():
    obj, notes = normalize_judge_v2(v2(unsupported_claims=["it is very quiet"]))
    assert obj["unsupported_claims"] == [{"claim": "it is very quiet"}] and "converted plain-string claims to claim objects" in notes


# ------------------------------------------------------------------ judge_summary: repair & failure
def test_v2_repair_then_success(settings, chain):
    summary = next(iter(chain.summaries.values()))
    client = ScriptedLLM(["not json at all", json.dumps(v2(hallucination_score=5, unsupported_claims=[]))])
    j, res = judge_summary(client, summary, chain.extraction.records, settings, version="v2")
    assert j is not None and j.prompt_version == "v2" and j.repair_attempts == 1
    assert client.calls[0]["task"] == "judge_v2" and client.calls[1]["task"] == "repair"
    repair_prompt = "\n".join(m.content for m in client.calls[1]["messages"])
    assert "unsupported_claims" in repair_prompt and "not json at all" in repair_prompt


def test_v2_persistent_invalid_output_is_a_failure_without_scores(settings, chain):
    summary = next(iter(chain.summaries.values()))
    bad = json.dumps(v2(hallucination_score=2, unsupported_claims=[]))  # internally inconsistent every time
    client = ScriptedLLM([bad, bad, bad])
    j, res = judge_summary(client, summary, chain.extraction.records, settings, version="v2")
    assert j is None and not res.success and len(res.attempts) == 1 + settings.max_repair_attempts
    assert "unsupported_claims is empty" in res.error


def test_v2_judgement_records_claim_traceability(settings, chain):
    summary = next(iter(chain.summaries.values()))
    phrase = summary.content.overall_opinion[:60]
    out = v2(unsupported_claims=[{"claim": phrase, "reason": "x"}, {"claim": "built-in espresso grinder", "reason": "y"}], hallucination_score=3)
    j, _ = judge_summary(ScriptedLLM([json.dumps(out)]), summary, chain.extraction.records, settings, version="v2")
    assert j.deterministic_checks["unsupported_claims_flagged"] == 2
    assert j.deterministic_checks["unsupported_claims_traceable_to_summary"] == 1  # the invented one is not in the summary


def test_claim_in_text():
    assert claim_in_text("The ice maker is very quiet", "Reviewers say the ice maker is very quiet at night.")
    assert not claim_in_text("solar powered", "The ice maker is quiet.")


# ------------------------------------------------------------------ batch: resume, retries, stored failures
class _FlakyThenOk(ScriptedLLM):
    def __init__(self, responses, fail_times):
        super().__init__(responses)
        self.fail_times = fail_times

    def invoke(self, messages, *, task, payload=None):
        if self.fail_times:
            self.fail_times -= 1
            raise LLMError("ollama call failed: ReadTimeout")
        return super().invoke(messages, task=task, payload=payload)


def _ok():
    return json.dumps(v2(hallucination_score=5, unsupported_claims=[]))


def test_batch_resume_never_repeats_successful_judgements(settings, chain):
    summaries = list(chain.summaries.values())
    store = JudgementStore.for_mode(settings, True, "v2")
    client = ScriptedLLM([_ok()] * len(summaries))
    r1 = judge_summaries(client, summaries, chain.extraction.records, settings, store, version="v2", retry_wait_s=0)
    assert len(r1.judged) == len(summaries) and r1.new_calls == len(summaries)
    client2 = ScriptedLLM([])  # would raise if called
    r2 = judge_summaries(client2, summaries, chain.extraction.records, settings, store, version="v2", retry_wait_s=0)
    assert r2.new_calls == 0 and len(r2.reused) == len(summaries) and client2.calls == []
    assert len(store.all()) == len(summaries)  # no duplicates
    assert store.path.parent.name == "v2"


def test_stored_failure_is_final_unless_retry_requested(settings, chain):
    summary = next(iter(chain.summaries.values()))
    store = JudgementStore.for_mode(settings, True, "v2")
    bad = "nope"
    r1 = judge_summaries(ScriptedLLM([bad] * 3), [summary], chain.extraction.records, settings, store, version="v2", retry_wait_s=0)
    assert r1.failed == [summary.summary_id]
    fail = store.failures()[0]
    assert len(fail["attempts"]) == 3 and fail["attempts"][0]["raw_output"] == "nope"  # raw outputs kept
    r2 = judge_summaries(ScriptedLLM([]), [summary], chain.extraction.records, settings, store, version="v2", retry_wait_s=0)
    assert r2.skipped_stored_failure == [summary.summary_id] and r2.new_calls == 0
    r3 = judge_summaries(ScriptedLLM([_ok()]), [summary], chain.extraction.records, settings, store, version="v2", retry_failed=True, retry_wait_s=0)
    assert len(r3.judged) == 1 and store.open_failures() == {}


def test_provider_errors_are_retried_with_a_bound(settings, chain):
    summary = next(iter(chain.summaries.values()))
    store = JudgementStore.for_mode(settings, True, "v2")
    r = judge_summaries(_FlakyThenOk([_ok()], fail_times=2), [summary], chain.extraction.records, settings, store, version="v2", provider_retries=2, retry_wait_s=0)
    assert len(r.judged) == 1
    with pytest.raises(JudgeProviderStopped):
        judge_summaries(_FlakyThenOk([], fail_times=5), [next(iter(chain.summaries.values()))], chain.extraction.records, settings,
                        JudgementStore.for_mode(settings, True, "v2b"), version="v2", provider_retries=2, retry_wait_s=0)
    assert not JudgementStore.for_mode(settings, True, "v2b").failures_path.exists()  # outage is not recorded as a model failure


def test_v1_and_v2_stores_are_separate(settings):
    v1 = JudgementStore.for_mode(settings, False, "v1")
    v2s = JudgementStore.for_mode(settings, False, "v2")
    assert v1.path.parent.name == "judgements" and v2s.path.parent.name == "v2" and v1.path != v2s.path


def test_mock_v2_end_to_end(settings, chain):
    summary = next(iter(chain.summaries.values()))
    j, _ = judge_summary(MockLLMClient(), summary, chain.extraction.records, settings, version="v2")
    assert j and j.is_mock and j.prompt_version == "v2"


# ------------------------------------------------------------------ negative control
def test_planted_feature_is_absent_from_reviews_and_detected_by_keyword(settings, chain):
    summary = next(iter(chain.summaries.values()))
    texts = {r.review_id: r.review_text for r in chain.extraction.records}
    texts[summary.aggregate.review_ids[0]] += " It has wi-fi."  # make the first pooled feature unusable for this product
    planted = plant_invented_feature(summary, texts, index=0)
    assert planted.planted_text != FEATURE_POOL[0][0]  # skipped because "wi-fi" occurs in the reviews
    assert planted.summary.content.appreciated[0].point == planted.planted_text
    assert planted.summary.summary_id.endswith("-nc-feature") and planted.summary.content.complaints == summary.content.complaints
    out = v2(unsupported_claims=[{"claim": planted.planted_text, "reason": "no review mentions it"}], hallucination_score=2)
    j, _ = judge_summary(ScriptedLLM([json.dumps(out)]), planted.summary, chain.extraction.records, settings, version="v2")
    assert planted_claim_detected(j, planted.keywords)
    j_miss, _ = judge_summary(ScriptedLLM([_ok()]), planted.summary, chain.extraction.records, settings, version="v2")
    assert not planted_claim_detected(j_miss, planted.keywords)


def test_fabricated_statistic_not_close_to_any_real_statistic(chain):
    summary = next(iter(chain.summaries.values()))
    items = build_planted(summary, {}, index=0)
    stat = next(p for p in items if p.kind == "fabricated_statistic")
    pct = int(stat.keywords[0].rstrip("%"))
    known = list(summary.aggregate.sentiment_percentages.values()) + list(summary.aggregate.complaint_percentages.values())
    assert all(abs(pct - k) > 3 for k in known)
    assert stat.summary.content.overall_opinion.startswith(summary.content.overall_opinion.rstrip()[:50])


def test_planted_keyword_only_in_reason_is_not_a_detection(settings, chain):
    """Regression (B0081N3YW4, 2026-11-14): the judge cited the planted 91% as 'the actual statistic' in a reason while
    attacking a true claim. That is the opposite of detecting it and must not count."""
    from evaluation.negative_control import planted_keyword_in_reason_only

    summary = next(iter(chain.summaries.values()))
    stat = next(p for p in build_planted(summary, {}, index=0) if p.kind == "fabricated_statistic")
    pct = stat.keywords[0]
    out = v2(unsupported_claims=[{"claim": "The product received a mixed reception", "reason": f"The actual statistic is {pct} of reviewers would buy it again"}])
    j, _ = judge_summary(ScriptedLLM([json.dumps(out)]), stat.summary, chain.extraction.records, settings, version="v2")
    assert not planted_claim_detected(j, stat.keywords)
    assert planted_keyword_in_reason_only(j, stat.keywords)
    out_ok = v2(unsupported_claims=[{"claim": f"{pct} of reviewers say they would buy this exact product again", "reason": "no such statistic"}])
    j_ok, _ = judge_summary(ScriptedLLM([json.dumps(out_ok)]), stat.summary, chain.extraction.records, settings, version="v2")
    assert planted_claim_detected(j_ok, stat.keywords) and not planted_keyword_in_reason_only(j_ok, stat.keywords)


def test_negative_control_summary_recomputes_detection_without_editing_rows(settings, chain):
    from scripts.judge_negative_control import summarise

    summary = next(iter(chain.summaries.values()))
    stat = next(p for p in build_planted(summary, {}, index=0) if p.kind == "fabricated_statistic")
    pct = stat.keywords[0]
    orig, _ = judge_summary(ScriptedLLM([_ok()]), summary, chain.extraction.records, settings, version="v2")
    reason_only = v2(unsupported_claims=[{"claim": "mixed reception", "reason": f"actually {pct} would buy again"}])
    named = v2(unsupported_claims=[{"claim": f"{pct} of reviewers would buy this exact product again", "reason": "fabricated"}], hallucination_score=2)
    rows = []
    for out in (reason_only, named):
        j, _ = judge_summary(ScriptedLLM([json.dumps(out)]), stat.summary, chain.extraction.records, settings, version="v2")
        rows.append({"product_id": "P0", "kind": stat.kind, "planted_text": stat.planted_text, "keywords": list(stat.keywords),
                     "original_judgement_id": orig.judgement_id, "original_scores": {"faithfulness": 4, "coverage": 4, "consistency": 3, "hallucination": 4, "mean": 3.75},
                     "original_flagged_planted_keywords": False, "corrupted_judged": True, "detected": True,  # stored with the old rule
                     "corrupted_scores": {"faithfulness": 4, "coverage": 4, "consistency": 3, "hallucination": out["hallucination_score"], "mean": 3.5},
                     "corrupted_judgement": j.model_dump(mode="json")})
    frozen = json.dumps(rows, sort_keys=True)
    s = summarise(rows, [], "scripted", {orig.judgement_id: orig})
    assert json.dumps(rows, sort_keys=True) == frozen  # input rows untouched
    k = s["by_kind"][stat.kind]
    assert (k["planted_claim_detected"], k["missed"], k["detected_with_previous_rule"], k["keyword_in_reason_only"]) == (1, 1, 2, 1)
    assert k["hallucination_lower"] == 1 and s["overall"]["corrupted_judged"] == 2
    assert s["detection_rule"] == "claim_text_only" and s["missed"][0]["kind"] == stat.kind


def test_semantic_review_requires_manual_reading_of_every_keyword_miss(settings, chain):
    from scripts.nc_semantic_review import review

    summary = next(iter(chain.summaries.values()))
    stat = next(p for p in build_planted(summary, {}, index=0) if p.kind == "fabricated_statistic")
    named = v2(unsupported_claims=[{"claim": f"{stat.keywords[0]} of reviewers would buy this exact product again", "reason": "fabricated"}])
    j, _ = judge_summary(ScriptedLLM([json.dumps(named)]), stat.summary, chain.extraction.records, settings, version="v2")
    row = {"product_id": "P0", "kind": stat.kind, "planted_text": stat.planted_text, "keywords": list(stat.keywords), "corrupted_judgement": j.model_dump(mode="json")}
    out = review([row], manual={})
    assert out["keyword_detected_in_claims"] == 1 and out["items"][0]["category"] == "named_in_claims" and "not a human review" in out["reviewer"]
    assert "NOT independent human ground truth" in out["assessment_type"] and out["pairs_in_file"] == out["n"] == 1
    j_miss, _ = judge_summary(ScriptedLLM([_ok()]), stat.summary, chain.extraction.records, settings, version="v2")
    miss = row | {"corrupted_judgement": j_miss.model_dump(mode="json")}
    with pytest.raises(ValueError, match="no manual classification"):  # an unread miss is never silently counted
        review([miss], manual={})
    with pytest.raises(ValueError, match="duplicate"):
        review([row, row], manual={})
    with pytest.raises(ValueError, match="not used"):  # stale manual entry
        review([row], manual={("ZZ", "invented_feature"): ("not_mentioned", "x")})
    with pytest.raises(ValueError, match="keyword match but also a manual entry"):
        review([row], manual={("P0", stat.kind): ("not_mentioned", "x")})
    cats = review([miss], manual={("P0", stat.kind): ("not_mentioned", "no mention")})["items"][0]
    assert cats["category"] == "not_mentioned" and cats["keyword_detected"] is False and cats["planted_keyword_in_reason_only"] is False
