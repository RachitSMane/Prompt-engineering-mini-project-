"""Label provenance reporting, D1 export, and judge-specific readiness. Offline: no LLM or network calls."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from annotation.annotation_manager import AI_ASSISTED_MARKER, label_provenance, provenance_text
from evaluation.prompt_comparison import evaluate_run, run_experiment
from llm.mock import MockLLMClient
from schemas.evaluation_schema import Annotation
from schemas.review_schema import ReviewInput

ROOT = Path(__file__).resolve().parents[1]


def _gold(rid: str, sentiment: str, ai: bool) -> Annotation:
    return Annotation(review_id=rid, product_id="P", overall_sentiment=sentiment, annotator="a",
                      notes=f"{AI_ASSISTED_MARKER} (test)." if ai else "")


def _reviews() -> list[ReviewInput]:
    texts = ["Works great and very quiet.", "Broke after a week, total junk.", "Fits my fridge perfectly, good price."]
    return [ReviewInput(review_id=f"E{i}", product_id="P", review_text=t) for i, t in enumerate(texts)]


# ------------------------------------------------------------------ provenance
def test_label_provenance_counts_marker():
    gold = [_gold("A", "positive", ai=False), _gold("B", "negative", ai=True), _gold("C", "mixed", ai=True)]
    prov = label_provenance(gold)
    assert prov == {"total": 3, "human": 1, "ai_assisted": 2}
    assert "1 human, 2 AI-assisted" in provenance_text(prov)
    assert provenance_text(label_provenance([_gold("A", "positive", ai=False)])) == "1 human labels"


def test_report_states_provenance_and_coverage(settings):
    """The report must not call AI-assisted reference labels human annotations, and must report coverage."""
    gold = [_gold("E0", "positive", ai=False), _gold("E1", "negative", ai=True), _gold("E2", "positive", ai=True)]
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot"], _reviews(), gold, required_gold=3)
    metrics = evaluate_run(run_dir, gold, 3)
    assert metrics["reference_provenance"] == {"total": 3, "human": 1, "ai_assisted": 2}
    assert metrics["strategies"]["zero_shot"]["coverage"] == 1.0
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "1 human, 2 AI-assisted" in report and "not manual annotations" in report
    assert "Ground truth = human annotations" not in report


# ------------------------------------------------------------------ D1 export
def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_d1", ROOT / "scripts" / "export_d1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_d1_export_writes_revalidated_json_and_csv(settings, tmp_path):
    reviews = _reviews()
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot"], reviews, [], required_gold=3)
    mod = _load_export_module()
    report = mod.export_strategy(run_dir, "zero_shot", [r.review_id for r in reviews], tmp_path)
    assert report["coverage"] == 1.0 and report["predictions_stored"] == 3 and report["missing_predictions"] == 0
    assert report["valid_records"] + report["failed_output"] + report["failed_provider"] == 3
    assert report["revalidation"]["revalidated_ok"] == report["valid_records"] and not report["revalidation"]["revalidation_errors"]
    out = tmp_path / "zero_shot"
    assert len(json.loads((out / "records.json").read_text(encoding="utf-8"))) == report["valid_records"]
    assert len(pd.read_csv(out / "records.csv")) == report["valid_records"]
    pd.read_csv(out / "records_aspects.csv")
    pd.read_csv(out / "failures.csv")
    assert json.loads((out / "validation_report.json").read_text(encoding="utf-8"))["strategy"] == "zero_shot"
    assert "zero_shot" in mod.summary_md([report], run_dir.name)


def test_d1_export_counts_failures_not_drops_them(settings, tmp_path):
    """With a malformed-output mock and no repairs, failures are exported and counted, never silently dropped."""
    reviews = [ReviewInput(review_id=f"F{i}", product_id="P", review_text="Loud and broke quickly, poor value.") for i in range(6)]
    s = replace(settings, max_repair_attempts=0)
    run_dir = run_experiment(s, MockLLMClient(malformed_rate=100), ["zero_shot"], reviews, [], required_gold=6)
    report = _load_export_module().export_strategy(run_dir, "zero_shot", [r.review_id for r in reviews], tmp_path)
    assert report["predictions_stored"] == 6
    assert report["valid_records"] + report["failed_output"] == 6
    assert len(pd.read_csv(tmp_path / "zero_shot" / "failures.csv")) == report["failed_output"] + report["failed_provider"]


# ------------------------------------------------------------------ judge readiness
def test_judge_status_checks_judge_model_not_extraction_model(settings, monkeypatch):
    pytest.importorskip("streamlit")
    import ui.common as common

    s = replace(settings, llm_provider="ollama", llm_model="qwen2.5:3b", judge_provider="ollama", judge_model="llama3.2",
                ollama_base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(common, "_ollama_models", lambda base_url: ["qwen2.5:3b"])  # judge model not pulled
    assert common.live_status(s)[0] is True
    ready, label, reason = common.judge_status(s)
    assert ready is False and label == "Model not pulled" and "llama3.2" in reason
    monkeypatch.setattr(common, "_ollama_models", lambda base_url: ["qwen2.5:3b", "llama3.2:latest"])
    assert common.judge_status(s)[0] is True
    monkeypatch.setattr(common, "_ollama_models", lambda base_url: None)  # server down
    assert common.judge_status(s)[1] == "Ollama not running"


def test_d3_export_marks_failed_judgements_without_scores(settings):
    from evaluation.summary_evaluation import JudgementStore, judge_summary
    from pipeline.summarization import run_chain

    spec = importlib.util.spec_from_file_location("export_d3", ROOT / "scripts" / "export_d3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    texts = ["Works great, very quiet.", "Too loud and broke after a month.", "Fits perfectly and the price was good."]
    reviews = [ReviewInput(review_id=f"E{i}", product_id=f"P{i % 2}", review_text=t) for i, t in enumerate(texts * 2)]
    res = run_chain(MockLLMClient(malformed_rate=0), reviews, settings, strategy="few_shot", source="dataset", min_reviews_for_summary=1)
    (ok_pid, ok_summary), (bad_pid, bad_summary) = sorted(res.summaries.items())
    j, _ = judge_summary(MockLLMClient(), ok_summary, res.extraction.records, settings)
    store = JudgementStore.for_mode(settings, True)
    store.add(j)
    (store.path.parent / "judge_failures.jsonl").write_text(json.dumps({"summary_id": bad_summary.summary_id, "product_id": bad_pid, "error": "invalid verdict"}) + "\n", encoding="utf-8")

    out = mod.build(ok_summary.source_extraction, settings, True)
    jdf = pd.read_csv(out / "judgements.csv").set_index("product_id")
    assert jdf.loc[ok_pid, "judge_status"] == "ok" and not pd.isna(jdf.loc[ok_pid, "faithfulness"])
    assert jdf.loc[bad_pid, "judge_status"].startswith("failed") and pd.isna(jdf.loc[bad_pid, "faithfulness"])  # no score invented
    assert jdf.loc[bad_pid, "judge_error"] == "invalid verdict"
    report = (out / "d3_report.md").read_text(encoding="utf-8")
    assert "FAILED" in report and "MOCK" in report  # failure stated; mock data labelled as mock
