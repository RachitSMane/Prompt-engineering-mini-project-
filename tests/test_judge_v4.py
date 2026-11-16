"""Judge v4: derived scores, consistency rules, LLM-stage parsing/failure handling, resumability, object separation.

All tests use a fake in-process client (no Ollama, no network)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.claim_checks import Stats
from evaluation.judge_v4 import (
    V4Store, build_summary_evaluation, combine, deterministic_stage, effective_llm_status, evaluate_claim, evaluate_items,
    item_from_summary, judge_claim, run_llm,
)
from llm.base import LLMError, LLMResponse
from schemas.claim_schema import (
    CONTRADICTED_CAP, UNSUPPORTED_CAP, Claim, ClaimEvaluation, ClaimStatus, DeterministicFinding, NumericCheck, SummaryEvaluationV4,
    derive_scores,
)
from schemas.evaluation_schema import ProductSummary, ProductSummaryContent, SummaryPoint
from tests.test_claims import make_agg

S, P, U, C, V = (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED,
                 ClaimStatus.UNVERIFIABLE)
REVIEWS = {"Ra": {"product_id": "P1", "rating": 5.0, "title": "Great", "text": "Easy to install. It works great."},
           "Rb": {"product_id": "P1", "rating": 1.0, "title": "Bad", "text": "It does not fit my fridge at all."},
           "Rc": {"product_id": "P1", "rating": 2.0, "title": "Meh", "text": "The lid broke after a week."}}


class FakeClient:
    provider, model_name, is_mock = "fake", "fake-judge", True

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def invoke(self, messages, *, task, payload=None):
        self.calls += 1
        return LLMResponse(text=self.fn(messages, payload or {}), model_name=self.model_name, provider=self.provider, latency_s=0.01, is_mock=True)


def verdict(status, quote="", expl="checked against the cited review"):
    return json.dumps({"evidence_quote": quote or "NONE", "explanation": expl, "status": status})


def summary(content=None) -> ProductSummary:
    content = content or ProductSummaryContent(
        overall_opinion="The product has a majority of positive reviews. Users appreciate the installation.",
        appreciated=[SummaryPoint(point="Installation is easy.", supporting_review_ids=["Ra"])],
        complaints=[SummaryPoint(point="It does not fit some fridges.", supporting_review_ids=["Rb"]),
                    SummaryPoint(point="The product is not durable.", supporting_review_ids=["Rc"])],
        frequent_aspects=["performance"], improvement_areas=["Improve durability"], recommendations=["Check the model"])
    return ProductSummary(summary_id="S1", product_id="P1", content=content, aggregate=make_agg(), source_extraction="x", provider="fake",
                          model_name="fake", is_mock=True, prompt_version="v1")


# --------------------------------------------------------------------------- derived scores
@pytest.mark.parametrize("seed", range(200))
def test_derived_score_rules_hold_for_random_claim_sets(seed):
    rng = random.Random(seed)
    sts = [rng.choice([S, P, U, C, V]) for _ in range(rng.randint(1, 12))]
    d = derive_scores(sts)
    ver = [x for x in sts if x != V]
    if d["faithfulness_score"] == 100.0:
        assert ver and all(x == S for x in ver)
    if C in sts:
        assert d["faithfulness_score"] is not None and d["faithfulness_score"] <= CONTRADICTED_CAP
        assert d["validity_status"] in ("CONTRADICTED_CLAIMS", "INSUFFICIENT_EVIDENCE")
    if U in sts:
        assert d["faithfulness_score"] <= UNSUPPORTED_CAP and d["validity_status"] != "FAITHFUL"
    assert d["evidence_coverage"] == round(len(ver) / len(sts), 4)
    assert sum(d["counts"].values()) == len(sts)


def test_score_monotone_in_bad_claims():
    base = [S] * 8
    scores = [derive_scores(base[:8 - k] + [U] * k)["faithfulness_score"] for k in range(9)]
    assert scores == sorted(scores, reverse=True)
    assert derive_scores([S, S, S, C])["faithfulness_score"] < derive_scores([S, S, S, U])["faithfulness_score"]


def test_unverifiable_excluded_and_insufficient_evidence():
    assert derive_scores([S, V])["faithfulness_score"] == 100.0
    assert derive_scores([S, V, V])["validity_status"] == "INSUFFICIENT_EVIDENCE"
    assert derive_scores([V])["faithfulness_score"] is None


# --------------------------------------------------------------------------- consistency enforced by the schema
def ce(status, cid="S1:appreciated:0", **kw):
    claim = Claim(claim_id=cid, summary_id="S1", product_id="P1", source_field="appreciated", index=0, kind="point", text="x")
    return ClaimEvaluation(claim=claim, status=status, explanation="e", method="llm", **kw)


def test_summary_evaluation_rejects_tampered_score_or_lists():
    evs = [ce(S, "a"), ce(U, "b"), ce(C, "c")]
    good = build_summary_evaluation(kind="original", summary_id="S1", product_id="P1", claim_evals=evs, run_id="r", config={})
    data = good.model_dump()
    assert SummaryEvaluationV4.model_validate(data)
    for field, value in (("faithfulness_score", 100.0), ("unsupported_claims", []), ("contradicted_claims", []),
                         ("validity_status", "FAITHFUL"), ("counts", {**data["counts"], "UNSUPPORTED": 0})):
        with pytest.raises(ValidationError):
            SummaryEvaluationV4.model_validate({**data, field: value})


def test_failed_numeric_claim_is_reported_and_cannot_be_supported():
    num = NumericCheck(mention="80%", kind="percentage", extracted_value=80, expected_value=86.7, denominator=15, calculation="c",
                       passed=False, severity="error")
    with pytest.raises(ValidationError):
        ce(S, numeric_checks=[num])
    ev = build_summary_evaluation(kind="original", summary_id="S1", product_id="P1", claim_evals=[ce(C, "n", numeric_checks=[num]), ce(S, "s")],
                                  run_id="r", config={})
    assert ev.failed_numeric_claims == ["n"] and ev.numeric_claims_failed == 1
    with pytest.raises(ValidationError):
        SummaryEvaluationV4.model_validate({**ev.model_dump(), "failed_numeric_claims": []})


def test_deterministic_cap_cannot_be_beaten():
    f = DeterministicFinding(check="x", outcome="fail", status_cap=U, detail="d")
    with pytest.raises(ValidationError):
        ce(S, findings=[f])
    with pytest.raises(ValidationError):
        ce(V, findings=[f])  # a definite finding never disappears into UNVERIFIABLE
    assert ce(C, findings=[f]).status == C


# --------------------------------------------------------------------------- combination
def test_combine_rules():
    assert combine(None, S, True) == (S, "llm")
    assert combine(None, None, True) == (V, "none")
    assert combine(P, S, True)[0] == P          # deterministic cap wins over a better LLM verdict
    assert combine(P, V, True)[0] == P          # UNVERIFIABLE never overrides a deterministic finding
    assert combine(U, C, True)[0] == C          # the worse verdict wins
    assert combine(S, None, False) == (S, "deterministic")


def test_unverified_quote_turns_support_into_unverifiable():
    assert effective_llm_status({"valid": True, "status": "SUPPORTED", "quote_verified": False}) == V
    assert effective_llm_status({"valid": True, "status": "CONTRADICTED", "quote_verified": False}) == V
    assert effective_llm_status({"valid": True, "status": "UNSUPPORTED", "quote_verified": False}) == U
    assert effective_llm_status({"valid": True, "status": "SUPPORTED", "quote_verified": True}) == S
    assert effective_llm_status({"valid": False}) is None


# --------------------------------------------------------------------------- LLM stage
def _point_claim():
    it = item_from_summary("original", summary())
    return it, [c for c in it.claims if c.kind == "point"][0]


def test_judge_claim_records_quote_verification():
    it, c = _point_claim()
    det = deterministic_stage(c, it.stats, REVIEWS)
    ok, rec = judge_claim(FakeClient(lambda m, p: verdict("SUPPORTED", "Easy to install")), c, det, it.stats, REVIEWS, kind="original",
                          config={}, run_id="t")
    assert ok and rec["quote_verified"] and rec["status"] == "SUPPORTED" and rec["raw_response"]
    ok, rec = judge_claim(FakeClient(lambda m, p: verdict("SUPPORTED", "Installation took two hours")), c, det, it.stats, REVIEWS,
                          kind="original", config={}, run_id="t")
    assert ok and not rec["quote_verified"]
    assert evaluate_claim(c, det, rec).status == V


def test_evidence_prompt_contains_only_cited_reviews():
    it, c = _point_claim()
    seen = {}

    def fn(m, p):
        seen["user"] = m[1].content
        return verdict("SUPPORTED", "Easy to install")

    judge_claim(FakeClient(fn), c, deterministic_stage(c, it.stats, REVIEWS), it.stats, REVIEWS, kind="original", config={}, run_id="t")
    assert "[Ra]" in seen["user"] and "[Rb]" not in seen["user"] and "[Rc]" not in seen["user"]


@pytest.mark.parametrize("raw", ["not json", json.dumps({"status": "MAYBE", "explanation": "x", "evidence_quote": "y"}),
                                 json.dumps({"status": "SUPPORTED"})])
def test_malformed_output_is_a_stored_failure_without_invented_label(raw, tmp_path):
    it = item_from_summary("original", summary())
    store = V4Store(tmp_path)
    stats = run_llm(FakeClient(lambda m, p: raw), [it], REVIEWS, store, config={}, run_id="t")
    assert stats["failed"] == 3 and stats["valid"] == 0
    assert all(f["raw_response"] == raw and "validation_error" in f for f in store.failures())
    ev = evaluate_items([it], REVIEWS, store, run_id="t", config={})[0]
    pts = [e for e in ev.claim_evaluations if e.claim.kind == "point"]
    assert all(e.llm_status is None for e in pts)


def test_provider_error_stores_nothing_and_run_resumes(tmp_path):
    it = item_from_summary("original", summary())
    store = V4Store(tmp_path)

    def boom(m, p):
        raise LLMError("ollama down")

    class Boom(FakeClient):
        def invoke(self, messages, *, task, payload=None):
            raise LLMError("down")

    with pytest.raises(LLMError):
        run_llm(Boom(boom), [it], REVIEWS, store, config={}, run_id="t")
    assert store.records() == [] and store.failures() == []
    client = FakeClient(lambda m, p: verdict("SUPPORTED", "Easy to install"))
    first = run_llm(client, [it], REVIEWS, store, config={}, run_id="t")
    again = run_llm(client, [it], REVIEWS, store, config={}, run_id="t")
    assert first["called"] == 3 and again["called"] == 0 and again["skipped_stored"] == 3


def test_object_kinds_never_mix(tmp_path):
    store = V4Store(tmp_path)
    orig = item_from_summary("original", summary())
    corr = item_from_summary("corrected", summary())  # same ids, different kind
    run_llm(FakeClient(lambda m, p: verdict("CONTRADICTED", "It does not fit my fridge at all")), [orig], REVIEWS, store, config={}, run_id="t")
    ev_c = evaluate_items([corr], REVIEWS, store, run_id="t", config={})[0]
    assert all(e.llm_status is None for e in ev_c.claim_evaluations if e.claim.kind == "point")


def test_end_to_end_unsupported_and_contradicted_claims_are_listed(tmp_path):
    store = V4Store(tmp_path)
    it = item_from_summary("original", summary())

    def fn(m, p):
        text = m[1].content
        if "Installation is easy" in text:
            return verdict("SUPPORTED", "Easy to install")
        if "does not fit" in text:
            return verdict("SUPPORTED", "It does not fit my fridge at all")
        return verdict("UNSUPPORTED")

    run_llm(FakeClient(fn), [it], REVIEWS, store, config={}, run_id="t")
    ev = evaluate_items([it], REVIEWS, store, run_id="t", config={})[0]
    by = {e.claim.claim_id: e for e in ev.claim_evaluations}
    assert by["S1:complaints:1"].status == U and "S1:complaints:1" in ev.unsupported_claims
    assert ev.faithfulness_score is not None and ev.faithfulness_score <= UNSUPPORTED_CAP
    # "Improve durability": no negative durability statistic and the durability complaint was NOT verified -> unsupported
    assert by["S1:improvement_areas:0"].status == U
    SummaryEvaluationV4.model_validate(ev.model_dump())


def test_derived_claim_supported_by_verified_complaint(tmp_path):
    store = V4Store(tmp_path)
    it = item_from_summary("original", summary())
    run_llm(FakeClient(lambda m, p: verdict("SUPPORTED", "The lid broke after a week") if "not durable" in m[1].content
                       else verdict("SUPPORTED", "Easy to install")), [it], REVIEWS, store, config={}, run_id="t")
    ev = evaluate_items([it], REVIEWS, store, run_id="t", config={})[0]
    by = {e.claim.claim_id: e for e in ev.claim_evaluations}
    assert by["S1:complaints:1"].status == S
    assert by["S1:improvement_areas:0"].status == S
    assert "verified against its cited reviews" in by["S1:improvement_areas:0"].explanation


# --------------------------------------------------------------------------- real data (skipped if the run artefacts are absent)
ROOT = Path(__file__).resolve().parents[1]
SRC = "dataset__few_shot__qwen2.5-3b__v1"


def _live():
    from config import load_settings
    from scripts.run_judge import load_inputs

    s = load_settings()
    if not (s.paths.mode_dir(False) / "summaries" / "summaries.jsonl").exists():
        pytest.skip("live summaries not present")
    summ, recs = load_inputs(s, False, SRC)
    return s, summ, recs


def test_all_15_summaries_decompose_and_evaluate_without_llm(tmp_path):
    from evaluation.judge_v4 import load_review_index

    s, summ, recs = _live()
    reviews = load_review_index(s.paths.reviews_csv)
    items = [item_from_summary("original", x, recs) for x in summ]
    evs = evaluate_items(items, reviews, V4Store(tmp_path), run_id="t", config={})
    assert len(evs) == 15 and all(ev.total_claims > 0 for ev in evs)
    by = {ev.product_id: ev for ev in evs}
    # known deterministic defects are found without any LLM
    assert any(n.mention == "80%" and not n.passed for e in by["B0BTY63SFY"].claim_evaluations for n in e.numeric_checks)
    assert by["B00DM8K1NS"].counts["CONTRADICTED"] >= 1 and by["B00LHRAGHW"].counts["CONTRADICTED"] >= 1
    assert by["B01FZJXITU"].counts["CONTRADICTED"] >= 1 and by["B07XG8QQL5"].counts["CONTRADICTED"] >= 1


def test_corrected_copies_are_separate_objects():
    from scripts.judge_v4 import corrected_items

    s, summ, recs = _live()
    items = corrected_items(s.paths.mode_dir(False), SRC, recs)
    if not items:
        pytest.skip("no corrected copies")
    orig_ids = {x.summary_id for x in summ}
    assert all(it.kind == "corrected" and it.summary_id not in orig_ids and it.original_summary_id in orig_ids for it in items)


def test_json_and_csv_exports_agree():
    import csv

    d = ROOT / "outputs" / "live" / "d3" / "judge_v4" / SRC
    if not (d / "evaluations.jsonl").exists():
        pytest.skip("v4 report not built")
    evs = [SummaryEvaluationV4.model_validate_json(line) for line in (d / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = list(csv.DictReader((d / "summaries.csv").open(encoding="utf-8")))
    assert len(rows) == len(evs)
    for ev, r in zip(evs, rows):
        assert r["summary_id"] == ev.summary_id and r["validity_status"] == ev.validity_status
        assert (r["faithfulness_score"] or None) == (None if ev.faithfulness_score is None else str(ev.faithfulness_score))
        assert int(r["CONTRADICTED"]) == ev.counts["CONTRADICTED"] and int(r["UNSUPPORTED"]) == ev.counts["UNSUPPORTED"]
    claims = list(csv.DictReader((d / "claims.csv").open(encoding="utf-8")))
    assert len(claims) == sum(ev.total_claims for ev in evs)
    status = {(e.claim.claim_id, ev.evaluated_object): e.status.value for ev in evs for e in ev.claim_evaluations}
    assert all(status[(r["claim_id"], r["evaluated_object"])] == r["status"] for r in claims)
