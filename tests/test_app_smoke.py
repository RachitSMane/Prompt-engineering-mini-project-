"""Headless smoke tests of every Streamlit page (no browser, no API calls).

Uses streamlit.testing.AppTest. Mock mode is forced on so pages that call the
LLM use the offline fixture client. Outputs go to a temporary directory.
"""

from __future__ import annotations

import re
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


def _texts(at: AppTest) -> list[str]:
    """All visible text: native alerts plus the ReviewIQ callouts/cards rendered as markdown."""
    out: list[str] = []
    for kind in ("markdown", "warning", "error", "info", "success", "caption"):
        out += [e.value for e in getattr(at, kind)]
    return out


def _shows(at: AppTest, text: str) -> bool:
    return any(text in t for t in _texts(at))


_KPI = re.compile(r'riq-kpi-label">([^<]+)(?:<span[^>]*>[^<]*</span>)?</div><div class="riq-kpi-value[^"]*">([^<]+)<')


def _kpis(at: AppTest) -> dict[str, list[str]]:
    """KPI cards rendered by ui.components.kpi(): label -> values (a label can appear in several rows)."""
    found: dict[str, list[str]] = {}
    for m in at.markdown:
        for label, value in _KPI.findall(m.value):
            found.setdefault(label, []).append(value)
    return found


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
    assert "5" in _kpis(at)["Valid records"]
    assert _shows(at, "Offline mock mode")
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
    assert _shows(at, "API key")
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
    assert _shows(at, "MOCK template summary")

    at = _page("summary_evaluation").run()
    at = _click(at, "Judge all unjudged summaries")
    assert not at.exception, [e.value for e in at.exception]
    at = _page("summary_evaluation").run()
    assert "Faithfulness" in _kpis(at)
    assert _shows(at, "MOCK judge verdict")

    at = _page("prompt_comparison").run()
    at.radio[0].set_value("First N reviews of the annotation sample")
    at.run()
    at.number_input[0].set_value(4)
    at = _click(at, "Run experiment")
    assert not at.exception, [e.value for e in at.exception]
    assert any("Run saved" in s.value for s in at.success)
    assert _shows(at, "NOT AN EXPERIMENTAL RESULT")
    assert _shows(at, "Mock demo")


def test_annotation_progress_survives_restart(isolated):
    at = _page("annotation_page").run()
    at.text_input(key="annotator").set_value("tester")
    at.radio[0].set_value("mixed")
    at = _click(at, "Save & next")
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
    assert _shows(at, "3 valid reviews")  # empty row removed
    assert _kpis(at)["Invalid records"] == ["1"]
    at.number_input[0].set_value(3)
    at = _click(at, "Run extraction")
    assert not at.exception, [e.value for e in at.exception]
    assert "3" in _kpis(at)["Valid records"]
    assert list((isolated.paths.outputs / "mock" / "extractions").glob("upload-reviews__*.jsonl"))


def test_review_analysis_filters(isolated):
    at = _page("review_analysis").run()
    at.number_input[0].set_value(8)
    at = _click(at, "Run extraction")
    total = int(_kpis(at)["Valid records"][0])
    sentiment = next(m for m in at.multiselect if m.label == "Sentiment")
    sentiment.set_value(["positive"]).run()
    assert not at.exception, [e.value for e in at.exception]
    caption = next(t for t in _texts(at) if "records · select a row" in t)
    shown = int(re.search(r"(\d+) of (\d+) records", caption).group(1))
    assert int(re.search(r"(\d+) of (\d+) records", caption).group(2)) == total
    assert 0 < shown <= total


def test_annotation_unsaved_changes_guard(isolated):
    """Navigating away with unsaved labels asks for confirmation instead of silently discarding them."""
    from annotation.annotation_manager import AnnotationStore

    at = _page("annotation_page").run()
    at.radio[0].set_value("negative").run()
    assert _shows(at, "Unsaved changes")
    at = _click(at, "Next unlabelled")
    assert not at.exception, [e.value for e in at.exception]
    assert any("Review 1 / 300" in m.value for m in at.markdown)  # did not move
    assert _shows(at, "You have unsaved changes on review 1")
    at = _click(at, "Discard and continue")
    assert any("Review 2 / 300" in m.value for m in at.markdown)
    assert AnnotationStore(isolated.paths.annotations_json).all() == []  # nothing was written


def test_annotation_validation_errors_are_shown(isolated):
    at = _page("annotation_page").run()
    at.radio[0].set_value("positive")
    at = _click(at, "Save & next")  # no annotator name
    assert any("annotator name" in e.value for e in at.error)
    assert any("Review 1 / 300" in m.value for m in at.markdown)


def test_prompt_experiments_with_human_labels(isolated):
    """With some human labels the metric cards, charts and tabs render real (here: mock-fixture) numbers."""
    from annotation.annotation_manager import AnnotationStore, load_sample
    from schemas.evaluation_schema import Annotation

    sample = load_sample(isolated.paths.annotation_sample_csv).head(6)
    store = AnnotationStore(isolated.paths.annotations_json)
    for i, row in enumerate(sample.itertuples()):
        store.save(Annotation(review_id=row.review_id, product_id=row.product_id, overall_sentiment=["positive", "negative", "mixed"][i % 3], aspects=[], complaint_categories=[], annotator="test"))
    at = _page("prompt_comparison").run()
    at = _click(at, "Run experiment")  # default scope: human-annotated reviews only
    assert not at.exception, [e.value for e in at.exception]
    at = _page("prompt_comparison").run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert html.count("riq-strategy-primary\"") == 3  # three strategy cards with a numeric macro-F1
    assert "Pending" not in re.findall(r'riq-strategy-primary[^>]*>([^<]+)<', html)
    assert _shows(at, "Provisional") or _shows(at, "Mock demo")


def test_theme_tokens_and_chart_template():
    import tomllib

    import plotly.io as pio

    from evaluation.visualizations import sentiment_pie
    from ui.theme import COLORS, SENTIMENT, css_variables

    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert cfg["base"] == "dark"
    assert cfg["backgroundColor"] == COLORS["bg"] and cfg["textColor"] == COLORS["text"] and cfg["primaryColor"] == COLORS["accent"]
    assert "--riq-positive: #34D399" in css_variables() and SENTIMENT["negative"] == "#F87171"
    assert pio.templates.default == "reviewiq"
    fig = sentiment_pie({"positive": 3, "negative": 1, "neutral": 0, "mixed": 0})
    assert fig.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"  # no white chart backgrounds
    assert list(fig.data[0].marker.colors) == [SENTIMENT["positive"], SENTIMENT["negative"]]
