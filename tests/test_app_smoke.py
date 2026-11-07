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
