"""The Summary Evaluation page loads the judge v4 results (read-only demo mode, no model calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "live" / "d3" / "judge_v4" / "dataset__few_shot__qwen2.5-3b__v1" / "report.json"


@pytest.mark.skipif(not REPORT.exists(), reason="judge v4 report not built")
def test_summary_evaluation_page_shows_v4_results(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "ui" / "summary_evaluation.py"), default_timeout=180)
    at.run()
    assert not at.exception
    # while the human claim validation is incomplete the page opens in the blind view; switch to the results
    if any(sc.key == "se_page_mode" for sc in at.segmented_control):
        at.segmented_control(key="se_page_mode").set_value("Evaluation results").run()
        assert not at.exception
    assert any("judge v4" in h.value for h in at.subheader)
    labels = [m.label for m in at.metric]
    assert "Claims evaluated" in labels and "Negative control specificity" in labels
    for view in ("Claims", "Negative control", "Run metadata", "Summaries"):
        at.segmented_control(key="v4_view").set_value(view).run()
        assert not at.exception, view
