"""D2 report builder: the independent metric code is correct on a hand-checked example, treats invalid outputs as
wrong (denominator = all reviews, no confusion-matrix column), and reproduces the stored metrics of the real run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_d2_report import ROOT, check_transcription, failure_category, metrics_for, recompute, verify

RUN = ROOT / "outputs" / "live" / "experiments" / "20261111-030912_qwen2.5-3b"


def test_metrics_by_hand_with_an_invalid_output():
    y_true = ["positive", "positive", "negative", "neutral", "mixed"]
    y_pred = ["positive", "__invalid__", "negative", "positive", "negative"]
    m = metrics_for(y_true, y_pred)
    assert m["n"] == 5 and m["correct"] == 2 and m["accuracy"] == pytest.approx(0.4) and m["invalid"] == 1
    pos = m["per_class"]["positive"]
    assert pos["precision"] == pytest.approx(0.5) and pos["recall"] == pytest.approx(0.5) and pos["invalid"] == 1
    assert sum(m["confusion_matrix"][0]) == 1  # the invalid output is in no predicted column
    assert m["per_class"]["neutral"]["f1"] == 0 and m["per_class"]["mixed"]["f1"] == 0
    assert m["macro_f1"] == pytest.approx((0.5 + 2 * 0.5 * 1 / 1.5 + 0 + 0) / 4)


def test_failure_category_uses_the_first_error_only():
    err = "Output still invalid after 2 repair attempt(s): overall_sentiment: Field required; aspects.0.aspect_category: Input should be 'a'"
    assert failure_category(err) == "schema (missing / extra field)"
    assert failure_category("invalid JSON: Expecting value at line 1") == "invalid JSON"


@pytest.mark.skipif(not (RUN / "metrics.json").exists(), reason="saved D2 run not present")
def test_recomputation_reproduces_the_stored_run_and_flags_the_transcription_slip():
    stored = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    data = recompute(RUN)
    rows = verify(data, stored)
    assert data["same_review_set"] and all(r["match"] for r in rows) and len(rows) == 96
    assert data["run_ids"]["n"] == 299 and data["sample"]["size"] == 300 and data["run_ids"]["sample_not_in_run"] == ["R5cfb12ef75b4"]
    assert data["reference"]["human"] == 1 and data["reference"]["ai_assisted"] == 298
    bad = [t for t in check_transcription(data) if not t["match"]]
    assert [(t["strategy"], t["value"]) for t in bad] == [("zero_shot", "accuracy")]


@pytest.mark.skipif(not (RUN / "approved_reference" / "metrics.json").exists(), reason="approved-reference evaluation not present")
def test_sensitivity_recomputation_reproduces_the_approved_reference_evaluation():
    from scripts.build_d2_report import recompute_sensitivity

    sens = recompute_sensitivity(RUN)
    assert sens["same_review_set"] and sens["n"] == 300 and sens["unique"] == 300
    assert all(v["match"] for v in sens["verification"]) and len(sens["verification"]) == 72
    assert sens["extra_review_ids"] == ["R5cfb12ef75b4"] and sens["corrections_applied"] == 0
    for s in ("zero_shot", "few_shot", "cot"):
        assert sens["strategies"][s]["extra_predictions"]["R5cfb12ef75b4"]["prediction"] == "neutral"
        assert sens["strategies"][s]["classification"]["per_class"]["neutral"]["support"] == 6


def test_chart_check_detects_a_missing_value_label():
    from scripts.build_d2_report import chart_check, grouped_bars

    svg = grouped_bars("t", ["a"], {"few_shot": [0.5]}, 1.0)
    assert chart_check({"x": svg}, {"x": ["0.500"]})[0]["match"]
    assert not chart_check({"x": svg}, {"x": ["0.501"]})[0]["match"]
