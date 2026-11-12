"""Metric calculations checked against hand-computed values."""

import pytest

from evaluation.classification_metrics import (
    INVALID,
    collapse_predicted_aspects,
    evaluate_predictions,
    multilabel_metrics,
    single_label_metrics,
)
from evaluation.prompt_comparison import comparison_table, evaluate_run, evaluation_status, run_experiment
from llm.mock import MockLLMClient
from schemas.evaluation_schema import AnnotatedAspect, Annotation
from schemas.review_schema import Aspect, ExtractionMetadata, ReviewInput, ReviewRecord


def test_single_label_hand_computed():
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "neutral"]
    m = single_label_metrics(y_true, y_pred, ["positive", "negative", "neutral", "mixed"])
    assert m["accuracy"] == 0.75
    # positive: P=1, R=.5, F1=.6667 ; negative: P=.5, R=1, F1=.6667 ; neutral: 1,1,1 ;
    # mixed occurs in neither gold nor predictions -> excluded from the macro average
    assert m["per_class"]["positive"]["f1-score"] == pytest.approx(0.6667, abs=1e-4)
    assert m["macro_f1"] == pytest.approx((0.6667 + 0.6667 + 1) / 3, abs=1e-3)
    assert m["weighted_f1"] == pytest.approx((2 * 0.6667 + 0.6667 + 1) / 4, abs=1e-3)
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]


def test_invalid_prediction_counts_as_wrong():
    m = single_label_metrics(["positive", "negative"], ["positive", INVALID], ["positive", "negative"])
    assert m["accuracy"] == 0.5 and m["invalid_predictions"] == 1
    assert m["per_class"]["negative"]["recall"] == 0.0


def test_multilabel_hand_computed():
    t = [{"a", "b"}, {"a"}, set()]
    p = [{"a"}, {"a", "c"}, set()]
    m = multilabel_metrics(t, p, ["a", "b", "c"])
    # micro: TP=2, FP=1, FN=1 -> P=R=F1=2/3
    assert m["micro_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["subset_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["hamming_loss"] == pytest.approx(2 / 9, abs=1e-4)
    assert m["per_label"]["b"]["recall"] == 0.0
    # label a: TP=2 -> F1=1 ; label b: FN=1 -> F1=0 ; label c: predicted only (FP) -> F1=0, still counted
    assert m["per_label"]["a"]["f1"] == 1.0
    assert m["macro_f1_present_labels"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["macro_labels"] == ["a", "b", "c"]


def test_multilabel_macro_excludes_labels_absent_everywhere():
    m = multilabel_metrics([{"a"}, {"b"}], [{"a"}, {"b"}], ["a", "b", "c", "d"])
    assert m["macro_f1_present_labels"] == 1.0 and m["macro_labels"] == ["a", "b"]


def test_macro_f1_ignores_classes_absent_from_gold_and_predictions():
    """Regression: perfect predictions without any 'neutral' used to give macro-F1 0.667."""
    m = single_label_metrics(["positive", "negative", "positive"], ["positive", "negative", "positive"], ["positive", "negative", "neutral"])
    assert m["macro_f1"] == 1.0 and m["macro_labels"] == ["positive", "negative"]
    # a class that is only predicted (false positive) is still penalised
    m = single_label_metrics(["positive", "positive"], ["positive", "neutral"], ["positive", "negative", "neutral"])
    assert m["macro_labels"] == ["positive", "neutral"]
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0) / 2, abs=1e-4)


def _meta(**kw):
    base = dict(strategy="zero_shot", provider="t", model_name="t", prompt_version="v1", is_mock=True, validation_success=True)
    base.update(kw)
    return ExtractionMetadata(**base)


def _rec(rid, overall, aspects, complaints):
    return ReviewRecord(
        review_id=rid, product_id="P", review_text="x", overall_sentiment=overall, confidence=0.9,
        aspects=[Aspect(aspect_name=c, aspect_category=c, sentiment=s, evidence="xxx", confidence=conf) for c, s, conf in aspects],
        complaint_categories=complaints, extraction_metadata=_meta(),
    )


def test_collapse_majority_and_tiebreak():
    r = _rec("1", "mixed", [("noise", "negative", 0.9), ("noise", "positive", 0.4), ("performance", "positive", 0.5), ("performance", "positive", 0.5), ("performance", "negative", 0.9)], [])
    assert collapse_predicted_aspects(r) == {"noise": "negative", "performance": "positive"}
    assert collapse_predicted_aspects(None) == {}


def test_evaluate_predictions_aspect_matching():
    gold = [
        Annotation(review_id="1", product_id="P", overall_sentiment="mixed", annotator="h",
                   aspects=[AnnotatedAspect(aspect_category="noise", sentiment="negative"), AnnotatedAspect(aspect_category="performance", sentiment="positive")],
                   complaint_categories=["noise_issue"]),
        Annotation(review_id="2", product_id="P", overall_sentiment="positive", annotator="h", aspects=[AnnotatedAspect(aspect_category="price_value", sentiment="positive")]),
        Annotation(review_id="3", product_id="P", overall_sentiment="negative", annotator="h"),
    ]
    preds = {
        "1": _rec("1", "mixed", [("noise", "negative", 0.9), ("performance", "negative", 0.8)], ["noise_issue"]),
        "2": _rec("2", "positive", [("price_value", "positive", 0.9), ("shipping_packaging", "positive", 0.9)], []),
        "3": None,  # failed extraction
    }
    m = evaluate_predictions(gold, preds)
    assert m["n_evaluated"] == 3 and m["n_failed_extractions"] == 1
    assert m["overall_sentiment"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    # detection: gold {noise,perf},{price},{} ; pred {noise,perf},{price,ship},{} -> TP=3 FP=1 FN=0
    assert m["aspect_detection"]["micro_precision"] == 0.75 and m["aspect_detection"]["micro_recall"] == 1.0
    # sentiment on matched categories: noise ok, performance wrong, price ok
    assert m["aspect_sentiment"]["n"] == 3 and m["aspect_sentiment"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    # joint pairs: TP=2 (noise:neg, price:pos), FP=2, FN=1
    assert m["aspect_joint"]["micro_precision"] == 0.5
    assert m["complaints"]["micro_f1"] == 1.0


def test_evaluation_status_labels():
    assert evaluation_status(True, 300, 300) == "mock_demo"
    assert evaluation_status(False, 0, 300) == "awaiting_annotations"
    assert evaluation_status(False, 120, 300) == "provisional"
    assert evaluation_status(False, 300, 300) == "complete"


def test_run_experiment_end_to_end_mock(settings):
    reviews = [ReviewInput(review_id=f"E{i}", product_id="P", review_text=t) for i, t in enumerate(
        ["Works great and very quiet.", "Broke after a week, total junk.", "Fits my fridge perfectly, good price.", "It arrived."])]
    gold = [Annotation(review_id="E0", product_id="P", overall_sentiment="positive", annotator="h"),
            Annotation(review_id="E1", product_id="P", overall_sentiment="negative", annotator="h")]
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot", "cot"], reviews, gold)
    metrics = evaluate_run(run_dir, gold, required=300)
    assert metrics["status"] == "mock_demo" and metrics["n_gold_available"] == 2
    table = comparison_table(metrics)
    assert set(table["strategy"]) == {"Zero-shot", "Chain-of-Thought"}
    assert (run_dir / "report.md").read_text(encoding="utf-8").count("NOT AN EXPERIMENTAL RESULT") == 1
    assert "mock" in str(run_dir) and run_dir.name.startswith("MOCK-")
    # resuming adds a strategy without recomputing existing ones
    run_experiment(settings, MockLLMClient(malformed_rate=0), ["few_shot"], reviews, gold, run_id=run_dir.name)
    assert set(evaluate_run(run_dir, gold, 300)["strategies"]) == {"zero_shot", "cot", "few_shot"}


def test_resumed_run_evaluates_added_reviews(settings):
    """Regression: reviews added on resume were predicted but silently left out of evaluation."""
    r = [ReviewInput(review_id=f"A{i}", product_id="P", review_text="Works great and very quiet.") for i in range(3)]
    g = [Annotation(review_id=f"A{i}", product_id="P", overall_sentiment="positive", annotator="h") for i in range(3)]
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot"], r[:1], g[:1])
    run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot"], r, g, run_id=run_dir.name)
    m = evaluate_run(run_dir, g, 300)
    assert m["n_gold_available"] == 3
    assert m["strategies"]["zero_shot"]["classification"]["n_evaluated"] == 3


def test_interrupted_run_with_unstarted_strategy_evaluates_as_incomplete(settings):
    """Regression: a strategy listed in config.json with no predictions crashed render_report (KeyError 'per_class')."""
    import json

    r = [ReviewInput(review_id=f"U{i}", product_id="P", review_text="Works great and very quiet.") for i in range(2)]
    g = [Annotation(review_id=f"U{i}", product_id="P", overall_sentiment="positive", annotator="h") for i in range(2)]
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot"], r, g, required_gold=2)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg["strategies"].append("cot")  # as if the run stopped before CoT started
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    m = evaluate_run(run_dir, g, 2)
    assert m["missing_predictions"]["cot"] == 2 and m["strategies"]["cot"]["coverage"] == 0.0
    # mock runs are always labelled mock_demo; the live status logic is checked directly
    assert evaluation_status(False, 2, 2, missing_predictions=2) == "incomplete"
    assert "cot: no predictions yet" in (run_dir / "report.md").read_text(encoding="utf-8")
    import pandas as pd
    assert pd.isna(comparison_table(m).set_index("strategy").loc["Chain-of-Thought", "sentiment_macro_f1"])


def test_resume_does_not_resample_stored_output_failures(settings):
    """A stored output failure is final on resume: one attempt per review per strategy, as for the others."""
    from tests.conftest import ScriptedLLM

    r = [ReviewInput(review_id="Z0", product_id="P", review_text="Works great and very quiet.")]
    scripted = ScriptedLLM(["not json"] * (settings.max_repair_attempts + 1))  # initial attempt + every repair invalid
    run_dir = run_experiment(settings, scripted, ["zero_shot"], r, [], run_id="unit-failure-run", required_gold=1)
    lines = (run_dir / "predictions_zero_shot.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and '"status": "failed"' in lines[0]

    run_experiment(settings, ScriptedLLM([]), ["zero_shot"], r, [], run_id=run_dir.name, required_gold=1)  # raises if called
    assert len((run_dir / "predictions_zero_shot.jsonl").read_text(encoding="utf-8").splitlines()) == 1
