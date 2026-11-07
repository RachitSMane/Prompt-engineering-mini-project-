"""Headless smoke tests of every Streamlit page (no browser, no API calls).

Uses streamlit.testing.AppTest. Mock mode is forced on so pages that call the
LLM use the offline fixture client. Outputs go to a temporary directory.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "review_analysis", "prompt_comparison", "product_insights", "summary_evaluation", "annotation_page"]


@pytest.fixture
def isolated(monkeypatch, settings):
    """Point the app at real dataset/sample files but temporary output/annotation dirs."""
    import shutil

    import ui.common as common
    from config import Paths, load_settings

    real = load_settings().paths
    if not real.reviews_csv.exists() or not real.annotation_sample_csv.exists():
        pytest.skip("dataset not prepared")
    ann_dir = settings.paths.annotations
    ann_dir.mkdir(parents=True)
    shutil.copy(real.annotation_sample_csv, ann_dir / "annotation_sample.csv")
    paths = Paths(root=real.root, data_raw=real.data_raw, data_processed=real.data_processed, annotations=ann_dir, outputs=settings.paths.outputs, reports=settings.paths.reports)
    test_settings = replace(settings, paths=paths, mock_malformed_rate=15)
    common.get_settings.clear()
    monkeypatch.setattr(common, "get_settings", lambda: test_settings)
    return test_settings


def _page(name: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "ui" / f"{name}.py"), default_timeout=60)
    at.session_state["use_mock"] = True
    return at


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_without_exception(isolated, name):
    at = _page(name).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page should render a title"


def test_review_analysis_runs_mock_extraction(isolated):
    at = _page("review_analysis").run()
    at.number_input[0].set_value(5)
    next(b for b in at.button if "Run extraction" in b.label).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Valid records" and m.value == "5" for m in at.metric)
    assert any("MOCK" in w.value for w in at.warning)
    assert (isolated.paths.outputs / "mock" / "extractions").exists()
    assert not (isolated.paths.outputs / "live").exists()


def test_annotation_save_persists(isolated):
    at = _page("annotation_page").run()
    at.text_input(key="annotator").set_value("tester")
    at.radio[0].set_value("positive")
    next(b for b in at.button if "Save" in b.label).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    from annotation.annotation_manager import AnnotationStore

    saved = AnnotationStore(isolated.paths.annotations_json).all()
    assert len(saved) == 1 and saved[0].annotator == "tester" and saved[0].overall_sentiment.value == "positive"


def test_live_mode_without_key_shows_message(isolated, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "ui" / "review_analysis.py"), default_timeout=60)
    at.session_state["use_mock"] = False
    at.run()
    assert not at.exception
    assert any("API key" in i.value for i in at.info)
    assert next(b for b in at.button if "Run extraction" in b.label).disabled


def _click(at: AppTest, text: str) -> AppTest:
    next(b for b in at.button if text in b.label).click()
    return at.run()


def test_full_offline_ui_workflow(isolated):
    """Product Insights extract -> summary -> Summary Evaluation judge -> Prompt Comparison run (all mock)."""
    at = _page("product_insights").run()
    at.number_input[0].set_value(3)
    at = _click(at, "Run extraction")
    assert not at.exception, [e.value for e in at.exception]
    assert any("valid records" in s.value for s in at.success)

    at = _page("product_insights").run()  # new session sees the stored extraction
    assert any("Product" == s.label for s in at.selectbox)
    at = _click(at, "Generate summary for this product")
    assert not at.exception, [e.value for e in at.exception]
    assert any("Overall customer opinion" in m.value for m in at.markdown)
    assert any("MOCK template summary" in w.value for w in at.warning)

    at = _page("summary_evaluation").run()
    at = _click(at, "Judge all unjudged summaries")
    assert not at.exception, [e.value for e in at.exception]
    at = _page("summary_evaluation").run()
    assert any(m.label == "Faithfulness" for m in at.metric)
    assert any("MOCK judge verdict" in w.value for w in at.warning)

    at = _page("prompt_comparison").run()
    at.radio[0].set_value("First N reviews of the annotation sample")
    at.run()
    at.number_input[0].set_value(4)
    at = _click(at, "Run experiment")
    assert not at.exception, [e.value for e in at.exception]
    assert any("Run saved" in s.value for s in at.success)
    assert any("NOT AN EXPERIMENTAL RESULT" in e.value for e in at.error)
    assert any("Mock demo" in w.value for w in at.warning)


def test_annotation_progress_survives_restart(isolated):
    at = _page("annotation_page").run()
    at.text_input(key="annotator").set_value("tester")
    at.radio[0].set_value("mixed")
    at = _click(at, "Save")
    assert any("Review 2 / 300" in m.value for m in at.markdown)  # "Save & next" advanced
    at = _page("annotation_page").run()  # brand-new session = app restart
    assert not at.exception
    assert any("1 complete" in (p.proto.text or "") for p in at.get("progress"))
    assert any("Review 2 / 300" in m.value for m in at.markdown)  # resumes at the first unlabelled review


def test_navigation_between_pages(isolated):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["use_mock"] = True
    at.run()
    for page in PAGES[1:]:
        at.switch_page(f"ui/{page}.py").run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert at.title


def test_csv_upload_and_extraction(isolated):
    """Upload a CSV with non-standard column names, map columns, run mock extraction."""
    csv = (
        "asin,content,stars\n"
        "B1,The motor is very loud and it broke after a week.,1\n"
        "B1,Fits perfectly and the price was great.,5\n"
        "B2,Installation was easy; works fine.,five\n"
        "B2,,3\n"
    ).encode()
    at = _page("review_analysis").run()
    at.radio[0].set_value("Upload CSV").run()
    at.file_uploader[0].set_value(("reviews.csv", csv, "text/csv")).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [s.value for s in at.selectbox[:2]] == ["content", "asin"]  # columns guessed
    assert any("3 valid reviews" in c.value for c in at.caption)  # empty row removed
    at.number_input[0].set_value(3)
    at = _click(at, "Run extraction")
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Valid records" and m.value == "3" for m in at.metric)
    assert list((isolated.paths.outputs / "mock" / "extractions").glob("upload-reviews__*.jsonl"))
