"""Anchored 1-5 rubric derived from judge v4 claim results (no LLM, no new inference).

The project rubric (``prompts/judge.py::RUBRIC``, problem statement §4.6) has four 1-5 dimensions: FAITHFULNESS,
COVERAGE, CONSISTENCY (agreement with the aggregated statistics) and HALLUCINATION (unsupported claims; higher = better).
Judge v1-v3 let an LLM write these scores, which produced scores that contradicted the judge's own claim lists. Here every
dimension is a pure function of stored data:

* FAITHFULNESS   = ``SummaryEvaluationV4.faithfulness_band_1_5`` (already derived from the claim statuses).
* COVERAGE       = which important themes of the product (from the aggregate) the summary mentions at all.
* CONSISTENCY    = the deterministic statistic findings on the claims that state statistics (overall-opinion sentences and
                   the frequent-aspect list): wrong numbers, wrong majority/lean direction, misleading denominators.
* HALLUCINATION  = the number of UNSUPPORTED / CONTRADICTED claims and whether any number failed its check.

The anchors below were written on 2026-11-16 by the AI coding assistant, following the wording of
``prompts/judge.py::RUBRIC``, before any rubric value was computed. After the first computation they were revised ONCE, for
two definitional errors (documented in reports/D3_COMPLETION_REPORT_2026-11-16.md): (1) the catch-all aspect ``other`` is
not a theme; (2) a "failed number" for HALLUCINATION means a numeric check with severity ``error`` (a wrong value, which v4
always turns into a CONTRADICTED claim), not a ``misleading`` one (e.g. a misleading denominator, which v4 rates
PARTIALLY_SUPPORTED). They are not calibrated against human ratings.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.claim_checks import ASPECT_TO_COMPLAINT, Stats, aspects_in, complaints_in
from schemas.claim_schema import ClaimStatus, SummaryEvaluationV4

RUBRIC_VERSION = "v4-derived-2026-11-16.r2"

#: A theme must be mentioned by at least this many reviews to count as "important".
MIN_THEME_MENTIONS = 2
TOP_ASPECTS = 3
TOP_COMPLAINTS = 3

#: Deterministic checks that compare a claim with the aggregated statistics (consistency dimension).
STATISTIC_CHECKS = frozenset({
    "numeric_count", "numeric_percentage", "average_rating", "aspect_majority", "aspect_statement", "frequent_aspects",
    "mixed_sentiment", "overall_negative", "overall_positive", "sentiment_lean", "sentiment_majority",
    "most_common_complaint", "universal_quantifier",
})

ANCHORS = {
    "faithfulness": "band of the derived 0-100 faithfulness score: 5 >= 95, 4 >= 80, 3 >= 60, 2 >= 40, else 1 "
                    "(schemas/claim_schema.py::derive_scores)",
    "coverage": "themes = top-3 aspects (excluding 'other') by mentions and top-3 complaint categories, each mentioned by >= 2 reviews; the "
                "rank-1 aspect and rank-1 complaint are MAJOR, the others MINOR. 5 = every theme mentioned; 4 = misses exactly "
                "one minor theme; 3 = misses a major theme or >= 2 minor themes, but mentions >= 50% of themes; 2 = mentions "
                ">= 25% of themes; 1 = mentions fewer than 25%",
    "consistency": "per statistic-stating claim, the worst deterministic statistic finding: 'fail' = MAJOR mismatch, "
                   "'misleading' = MINOR mismatch; a failed numeric check = WRONG NUMBER. 5 = no mismatch; 4 = exactly one "
                   "minor mismatch; 3 = one major mismatch without a wrong number, or >= 2 minor mismatches; 2 = a wrong "
                   "number or 2 major mismatches; 1 = >= 3 major mismatches",
    "hallucination": "n = UNSUPPORTED + CONTRADICTED claims (all claims). 5 = n == 0; 4 = n == 1 and it is UNSUPPORTED; "
                     "3 = n <= 3 with at most one CONTRADICTED and no failed number; 2 = n <= 6 and no failed number; "
                     "1 = n > 6 or any number with a wrong value (numeric check severity 'error': a fabricated or misattributed "
                     "statistic); misleading denominators are not wrong values",
}


@dataclass(frozen=True)
class Theme:
    kind: str  # "aspect" | "complaint"
    label: str
    mentions: int
    major: bool


def important_themes(st: Stats) -> list[Theme]:
    aspects = sorted(((a, v["mentions"]) for a, v in st.aspects.items() if a != "other" and v["mentions"] >= MIN_THEME_MENTIONS),
                     key=lambda kv: (-kv[1], kv[0]))[:TOP_ASPECTS]
    complaints = sorted(((c, n) for c, n in st.complaint_counts.items() if n >= MIN_THEME_MENTIONS),
                        key=lambda kv: (-kv[1], kv[0]))[:TOP_COMPLAINTS]
    return ([Theme("aspect", a, n, i == 0) for i, (a, n) in enumerate(aspects)]
            + [Theme("complaint", c, n, i == 0) for i, (c, n) in enumerate(complaints)])


_COMPLAINT_TO_ASPECT = {v: k for k, v in ASPECT_TO_COMPLAINT.items()}


def theme_mentioned(theme: Theme, ev: SummaryEvaluationV4) -> bool:
    for ce in ev.claim_evaluations:
        text = ce.claim.text
        if theme.kind == "aspect":
            if theme.label in aspects_in(text):
                return True
        else:
            negative_role = ce.claim.polarity == "negative" or ce.claim.source_field in ("overall_opinion", "frequent_aspects")
            if not negative_role:
                continue
            if theme.label in complaints_in(text):
                return True
            asp = _COMPLAINT_TO_ASPECT.get(theme.label)
            if asp and asp in aspects_in(text):
                return True
    return False


def coverage(ev: SummaryEvaluationV4, st: Stats) -> dict:
    themes = important_themes(st)
    if not themes:
        return {"score": None, "themes": [], "missed": [], "reason": "no theme is mentioned by >= 2 reviews"}
    rows = [{"kind": t.kind, "label": t.label, "mentions": t.mentions, "major": t.major, "mentioned": theme_mentioned(t, ev)}
            for t in themes]
    missed = [r for r in rows if not r["mentioned"]]
    share = (len(rows) - len(missed)) / len(rows)
    missed_major = sum(1 for r in missed if r["major"])
    missed_minor = len(missed) - missed_major
    if not missed:
        score = 5
    elif missed_major == 0 and missed_minor == 1:
        score = 4
    elif share >= 0.5:
        score = 3
    elif share >= 0.25:
        score = 2
    else:
        score = 1
    return {"score": score, "themes": rows, "missed": [f"{r['kind']}:{r['label']}" for r in missed],
            "mentioned_share": round(share, 3)}


def consistency(ev: SummaryEvaluationV4) -> dict:
    major, minor, wrong_numbers, details = [], [], [], []
    for ce in ev.claim_evaluations:
        if ce.claim.source_field not in ("overall_opinion", "frequent_aspects"):
            continue
        outcomes = [f.outcome for f in ce.findings if f.check in STATISTIC_CHECKS]
        cid = ce.claim.claim_id
        if any(n.severity == "error" for n in ce.numeric_checks):
            wrong_numbers.append(cid)
        if "fail" in outcomes:
            major.append(cid)
        elif "misleading" in outcomes or any(n.severity == "misleading" for n in ce.numeric_checks):
            minor.append(cid)
        for f in ce.findings:
            if f.check in STATISTIC_CHECKS and f.outcome in ("fail", "misleading"):
                details.append(f"{cid}: {f.check} {f.outcome}")
    if not major and not minor and not wrong_numbers:
        score = 5
    elif not major and not wrong_numbers and len(minor) == 1:
        score = 4
    elif len(major) >= 3:
        score = 1
    elif wrong_numbers or len(major) == 2:
        score = 2
    else:  # one major mismatch without a wrong number, or >= 2 minor mismatches
        score = 3
    return {"score": score, "major_mismatch_claims": major, "minor_mismatch_claims": minor,
            "wrong_number_claims": wrong_numbers, "findings": details}


def wrong_value_claims(ev: SummaryEvaluationV4) -> list[str]:
    return [ce.claim.claim_id for ce in ev.claim_evaluations if any(n.severity == "error" for n in ce.numeric_checks)]


def hallucination(ev: SummaryEvaluationV4) -> dict:
    u = [ce.claim.claim_id for ce in ev.claim_evaluations if ce.status == ClaimStatus.UNSUPPORTED]
    c = [ce.claim.claim_id for ce in ev.claim_evaluations if ce.status == ClaimStatus.CONTRADICTED]
    failed_numbers = wrong_value_claims(ev)
    n = len(u) + len(c)
    if n == 0:
        score = 5
    elif n == 1 and not c:
        score = 4
    elif failed_numbers or n > 6:
        score = 1
    elif n <= 3 and len(c) <= 1:
        score = 3
    else:
        score = 2
    return {"score": score, "unsupported_claims": u, "contradicted_claims": c, "wrong_value_claims": failed_numbers,
            "misleading_numeric_claims": [x for x in ev.failed_numeric_claims if x not in failed_numbers]}


def rubric_for(ev: SummaryEvaluationV4, st: Stats) -> dict:
    """All four dimensions for one stored v4 evaluation. Pure function of its claims and the product statistics."""
    cov = coverage(ev, st)
    con = consistency(ev)
    hal = hallucination(ev)
    return {
        "summary_id": ev.summary_id, "product_id": ev.product_id, "evaluated_object": ev.evaluated_object,
        "original_summary_id": ev.original_summary_id, "rubric_version": RUBRIC_VERSION,
        "faithfulness": ev.faithfulness_band_1_5, "faithfulness_score_0_100": ev.faithfulness_score,
        "coverage": cov["score"], "consistency": con["score"], "hallucination": hal["score"],
        "validity_status": ev.validity_status,
        "coverage_detail": cov, "consistency_detail": con, "hallucination_detail": hal,
    }


def check_rubric_consistent(row: dict, ev: SummaryEvaluationV4) -> list[str]:
    """Invariants that make a rubric row impossible to contradict its own claims; returns the violated rules."""
    problems = []
    bad = [ce for ce in ev.claim_evaluations if ce.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED)]
    if (row["hallucination"] == 5) != (not bad):
        problems.append("hallucination 5 <=> no UNSUPPORTED/CONTRADICTED claim")
    wrong = wrong_value_claims(ev)
    if wrong and row["hallucination"] != 1:
        problems.append("a number with a wrong value must give hallucination 1")
    stat_wrong = [cid for cid in wrong if cid in row["consistency_detail"]["wrong_number_claims"]]
    if stat_wrong and row["consistency"] > 2:
        problems.append("a wrong number in a statistic claim must give consistency <= 2")
    if any(ce.status == ClaimStatus.SUPPORTED and ce.claim.claim_id in wrong for ce in ev.claim_evaluations):
        problems.append("a claim with a wrong number is SUPPORTED")
    if row["faithfulness"] != ev.faithfulness_band_1_5:
        problems.append("faithfulness band differs from the stored evaluation")
    if any(ce.status == ClaimStatus.CONTRADICTED for ce in ev.claim_evaluations) and (row["faithfulness"] or 0) >= 4:
        problems.append("a CONTRADICTED claim cannot coexist with faithfulness >= 4")
    return problems
