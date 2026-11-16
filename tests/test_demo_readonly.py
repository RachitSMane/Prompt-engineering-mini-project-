"""Read-only demo mode (REVIEWIQ_DEMO_READONLY=true).

Checks that (1) every backend write path and every LLM client path refuses in demo mode, even when called
directly (bypassing the UI), (2) the Streamlit pages render with write actions disabled, (3) the saved-results
view of Review Analysis makes zero LLM calls and still offers working downloads, (4) normal mode is unchanged
and (5) rendering every page in demo mode against the REAL saved results leaves those files byte-identical.

No test makes a network or LLM call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import ScriptedLLM, valid_extraction
from utils.readonly import DEMO_BANNER, ENV_VAR, ReadOnlyModeError, ensure_writable, is_readonly

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "review_analysis", "prompt_comparison", "product_insights", "summary_evaluation", "annotation_page"]


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "true")


# ------------------------------------------------------------------ flag parsing
@pytest.mark.parametrize("value,expected", [("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
                                            ("false", False), ("0", False), ("", False), ("no", False)])
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(ENV_VAR, value)
    assert is_readonly() is expected


def test_flag_unset_is_normal_mode():
    assert not is_readonly()
    ensure_writable("anything")  # no exception


# ------------------------------------------------------------------ backend write paths
def test_storage_helpers_refuse(demo, tmp_path):
    from utils.storage import append_jsonl, log_activity, write_json_atomic

    with pytest.raises(ReadOnlyModeError):
        append_jsonl(tmp_path / "a.jsonl", [{"x": 1}])
    with pytest.raises(ReadOnlyModeError):
        write_json_atomic(tmp_path / "b.json", {"x": 1})
    with pytest.raises(ReadOnlyModeError):
        log_activity(tmp_path / "activity.jsonl", "test", False)
    assert list(tmp_path.iterdir()) == []  # nothing created, not even parent dirs or temp files


def test_existing_file_is_not_modified(demo, tmp_path):
    from utils.storage import append_jsonl, write_json_atomic

    f = tmp_path / "results.jsonl"
    f.write_text('{"keep": true}\n', encoding="utf-8")
    j = tmp_path / "metrics.json"
    j.write_text('{"macro_f1": 0.6}', encoding="utf-8")
    with pytest.raises(ReadOnlyModeError):
        append_jsonl(f, [{"new": 1}])
    with pytest.raises(ReadOnlyModeError):
        write_json_atomic(j, {"macro_f1": 0.0})
    assert f.read_text(encoding="utf-8") == '{"keep": true}\n'
    assert j.read_text(encoding="utf-8") == '{"macro_f1": 0.6}'


def test_extraction_store_refuses(demo, settings, review):
    from pipeline.extraction import ExtractionStore, ReviewExtractor
    from prompts.strategies import get_strategy

    store = ExtractionStore.for_run(settings, False, "dataset", "few_shot", "m")
    llm = ScriptedLLM([json.dumps(valid_extraction())])
    with pytest.raises(ReadOnlyModeError):
        ReviewExtractor(llm, get_strategy("few_shot"), settings).extract_many([review], store=store)
    assert not store.path.exists()


def test_summary_and_judgement_stores_refuse(demo, settings):
    from evaluation.summary_evaluation import JudgementStore
    from pipeline.summarization import SummaryStore

    for store in (SummaryStore.for_mode(settings, False), JudgementStore.for_mode(settings, False)):
        with pytest.raises(ReadOnlyModeError):
            store.add(type("Stub", (), {"model_dump": lambda self, **_k: {"summary_id": "x"}})())
        assert not store.path.exists()


def test_annotation_writes_refuse(demo, settings, monkeypatch):
    from annotation.annotation_manager import AnnotationStore, save_sample, write_template
    from schemas.evaluation_schema import Annotation

    path = settings.paths.annotations_json
    monkeypatch.delenv(ENV_VAR)  # create one label in normal mode first
    store = AnnotationStore(path)
    store.save(Annotation(review_id="R1", product_id="P1", overall_sentiment="positive", annotator="t"))
    before = path.read_bytes()
    monkeypatch.setenv(ENV_VAR, "true")

    store = AnnotationStore(path)  # reading still works
    assert len(store.all()) == 1
    with pytest.raises(ReadOnlyModeError):
        store.save(Annotation(review_id="R2", product_id="P1", overall_sentiment="negative", annotator="t"))
    with pytest.raises(ReadOnlyModeError):
        store.delete("R1")
    try:
        store.import_csv(pd.DataFrame([{"review_id": "R3", "product_id": "P1", "review_text": "x", "overall_sentiment": "neutral",
                                        "aspects": "", "complaint_categories": "", "annotator": "t", "notes": "", "status": "", "annotated_at": ""}]))
    except ReadOnlyModeError:
        pass
    assert path.read_bytes() == before

    sample = pd.DataFrame({"review_id": ["R1"], "product_id": ["P1"], "review_text": ["x"]})
    with pytest.raises(ReadOnlyModeError):
        save_sample(sample, settings.paths.annotation_sample_csv)
    with pytest.raises(ReadOnlyModeError):
        write_template(sample, settings.paths.annotations / "annotation_template.csv")
    assert not settings.paths.annotation_sample_csv.exists()


def test_metrics_reevaluation_and_exports_refuse(demo, tmp_path):
    from evaluation.prompt_comparison import evaluate_run
    from utils.export import export_records

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"review_ids": [], "strategies": []}), encoding="utf-8")
    with pytest.raises(ReadOnlyModeError):
        evaluate_run(run, [], 0)
    assert not (run / "metrics.json").exists() and not (run / "report.md").exists()
    with pytest.raises(ReadOnlyModeError):
        export_records([], tmp_path / "export", "x")
    assert not (tmp_path / "export").exists()


# ------------------------------------------------------------------ LLM paths
def test_factory_refuses_live_and_mock_clients(demo, settings):
    from llm.base import LLMUnavailableError
    from llm.factory import get_llm_client

    for use_mock in (False, True):
        for role in ("extraction", "judge"):
            with pytest.raises(LLMUnavailableError, match="read-only demo mode"):
                get_llm_client(settings, use_mock=use_mock, role=role)


def test_clients_built_outside_the_factory_refuse(demo):
    """Bypassing the factory still cannot reach the model."""
    from llm.langchain_client import LangChainLLMClient
    from llm.mock import MockLLMClient

    class Boom:
        def invoke(self, *_a, **_k):
            raise AssertionError("the underlying model must not be invoked in demo mode")

    live = object.__new__(LangChainLLMClient)
    live.provider, live.model_name, live._model = "ollama", "qwen2.5:3b", Boom()
    with pytest.raises(ReadOnlyModeError):
        live.invoke([], task="extraction")
    with pytest.raises(ReadOnlyModeError):
        MockLLMClient(malformed_rate=0).invoke([], task="extraction", payload={})


# ------------------------------------------------------------------ normal mode unchanged
def test_normal_mode_writes_and_clients_work(settings, tmp_path):
    from llm.factory import get_llm_client
    from utils.storage import append_jsonl, read_jsonl, write_json_atomic

    append_jsonl(tmp_path / "a.jsonl", [{"x": 1}])
    write_json_atomic(tmp_path / "b.json", {"y": 2})
    assert read_jsonl(tmp_path / "a.jsonl") == [{"x": 1}]
    client = get_llm_client(settings, use_mock=True)
    assert client.is_mock and client.invoke([], task="judge", payload={}).text


# ------------------------------------------------------------------ Streamlit pages (temporary data)
st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest


def _seed_live_outputs(settings):
    """Write a small 'live' extraction file, summary and judgement in normal mode (temp dir)."""
    from evaluation.summary_evaluation import JudgementStore
    from pipeline.aggregation import aggregate_all
    from pipeline.extraction import ExtractionStore, ReviewExtractor
    from prompts.strategies import get_strategy
    from schemas.review_schema import ReviewInput

    reviews = [ReviewInput(review_id=f"R{i}", product_id="P1", product_title="Test product", review_text=t)
               for i, t in enumerate(["The ice maker works great and installation was easy. But it is very loud at night."] * 3)]
    reviews = [r.model_copy(update={"review_text": r.review_text + f" ({i})"}) for i, r in enumerate(reviews)]
    data = [valid_extraction() for _ in reviews]
    store = ExtractionStore.for_run(settings, False, "dataset", "few_shot", "scripted-test-model")
    res = ReviewExtractor(ScriptedLLM([json.dumps(d) for d in data]), get_strategy("few_shot"), settings).extract_many(reviews, store=store)
    assert len(res.records) == 3
    aggs = aggregate_all(res.records)
    # one saved summary (made with the offline mock client, so it is flagged is_mock) for the judge page
    from llm.mock import MockLLMClient
    from pipeline.summarization import SummaryStore, summarize_product

    summary, _ = summarize_product(MockLLMClient(malformed_rate=0), aggs["P1"], settings, store.path.stem)
    assert summary is not None
    SummaryStore.for_mode(settings, False).add(summary)
    return store, aggs, JudgementStore


@pytest.fixture
def isolated_demo(monkeypatch, settings):
    import shutil

    import ui.common as common
    from config import Paths, load_settings

    real = load_settings().paths
    if not real.reviews_csv.exists() or not real.annotation_sample_csv.exists():
        pytest.skip("dataset not prepared")
    ann_dir = settings.paths.annotations
    ann_dir.mkdir(parents=True)
    shutil.copy(real.annotation_sample_csv, ann_dir / "annotation_sample.csv")
    paths = Paths(root=real.root, data_raw=real.data_raw, data_processed=real.data_processed, annotations=ann_dir,
                  outputs=settings.paths.outputs, reports=settings.paths.reports)
    test_settings = replace(settings, paths=paths)
    store, _aggs, _ = _seed_live_outputs(test_settings)
    common.get_settings.clear()
    monkeypatch.setattr(common, "get_settings", lambda: test_settings)

    calls = []

    def forbidden(*_a, **_k):
        calls.append(1)
        raise AssertionError("no LLM client may be created in demo mode")

    import llm.factory

    monkeypatch.setattr(common, "get_llm_client", forbidden)
    monkeypatch.setattr(llm.factory, "get_llm_client", forbidden)
    monkeypatch.setenv(ENV_VAR, "true")
    return test_settings, store, calls


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "ui" / f"{name}.py"), default_timeout=90)


def _texts(at) -> list[str]:
    out = []
    for kind in ("markdown", "warning", "error", "info", "success", "caption"):
        out += [e.value for e in getattr(at, kind)]
    return out


def _button(at, text):
    return next(b for b in at.button if text in b.label)


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("name", PAGES)
def test_pages_render_in_demo_mode(isolated_demo, name):
    test_settings, _store, calls = isolated_demo
    before = _hash_tree(test_settings.paths.outputs)
    at = _page(name).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title
    assert calls == []
    assert _hash_tree(test_settings.paths.outputs) == before


def test_demo_banner_on_every_page(isolated_demo):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(DEMO_BANNER in t for t in _texts(at))
    for page in PAGES[1:]:
        at.switch_page(f"ui/{page}.py").run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert any(DEMO_BANNER in t for t in _texts(at)), page
    assert at.session_state["use_mock"] is False  # demo mode shows the real (live) saved results


def test_write_buttons_disabled(isolated_demo):
    at = _page("review_analysis").run()
    at.radio[0].set_value("Prepared dataset (Amazon Appliances)").run()
    assert _button(at, "Run extraction").disabled

    at = _page("product_insights").run()
    assert _button(at, "Run extraction").disabled
    assert _button(at, "Generate summary for this product").disabled
    assert _button(at, "Generate summaries for all products").disabled
    assert any("Generating summaries is disabled" in t for t in _texts(at))

    at = _page("summary_evaluation").run()
    assert _button(at, "Judge all unjudged summaries").disabled
    assert _button(at, "Judge this summary").disabled
    assert any("Judging is disabled" in t for t in _texts(at))

    at = _page("annotation_page").run()
    assert next(b for b in at.button if b.label == "Save & next").disabled
    assert any("Labels are read-only" in t for t in _texts(at))


def test_ui_handlers_cannot_bypass_the_guard(isolated_demo):
    """A browser (and AppTest) cannot click a disabled button. If a handler were reached anyway, the backend
    functions those handlers call refuse, so nothing is written and no client is created."""
    from evaluation.summary_evaluation import JudgementStore
    from pipeline.summarization import SummaryStore
    from ui.common import get_client

    test_settings, _store, calls = isolated_demo
    before = _hash_tree(test_settings.paths.outputs)
    at = _page("product_insights").run()
    assert _button(at, "Generate summary for this product").disabled
    with pytest.raises(Exception, match="Cannot update a disabled"):
        _button(at, "Generate summary for this product").click()
    assert get_client() is None and get_client(role="judge") is None  # explains demo mode instead of returning a client
    summary = SummaryStore.for_mode(test_settings, False).all()[0]
    with pytest.raises(ReadOnlyModeError):
        SummaryStore.for_mode(test_settings, False).add(summary)
    with pytest.raises(ReadOnlyModeError):
        JudgementStore.for_mode(test_settings, False).add(summary)
    assert calls == []
    assert _hash_tree(test_settings.paths.outputs) == before


def test_saved_results_view_zero_llm_calls_and_downloads(isolated_demo):
    test_settings, store, calls = isolated_demo
    before = store.path.read_bytes()
    at = _page("review_analysis").run()
    assert at.radio[0].value == "Saved results (no LLM calls)"  # default source in demo mode
    assert not at.exception, [e.value for e in at.exception]
    texts = _texts(at)
    assert any("Saved results · cached, not newly generated" in t and "0 LLM calls" in t for t in texts)
    assert any("MOCK fixture outputs" in t for t in texts)  # the scripted test model is flagged is_mock, so it is labelled as such
    assert at.dataframe, "records table should be shown"
    assert len(at.dataframe[0].value) == 3

    # filters work without any LLM call
    sent = next(m for m in at.multiselect if m.label == "Sentiment")
    sent.set_value(["positive"]).run()
    assert not at.exception
    assert any("No records match the filters" in t for t in _texts(at))
    sent.set_value(["mixed"]).run()
    assert len(at.dataframe[0].value) == 3

    # downloads are generated in memory from the loaded records
    labels = [d.label for d in at.get("download_button")]
    assert {"JSON (records)", "CSV (records)", "CSV (aspects)", "CSV (failures)"} <= set(labels)
    assert calls == []
    assert store.path.read_bytes() == before


def test_saved_results_download_payloads(isolated_demo):
    """The export functions used by the download buttons produce the stored records without touching the file."""
    from pipeline.extraction import ExtractionStore
    from utils.export import aspects_to_dataframe, dataframe_to_csv_bytes, records_to_dataframe, records_to_json

    _s, store, _c = isolated_demo
    before = store.path.read_bytes()
    recs = ExtractionStore(store.path).records
    payload = json.loads(records_to_json(recs))
    assert len(payload) == 3 and payload[0]["review_id"] == "R0"
    assert dataframe_to_csv_bytes(records_to_dataframe(recs)).decode().count("\n") == 4  # header + 3 rows
    assert len(aspects_to_dataframe(recs)) == 9
    assert store.path.read_bytes() == before


# ------------------------------------------------------------------ real saved results stay byte-identical
REAL_FILES = ["outputs/live", "data/annotations"]


@pytest.fixture
def real_demo(monkeypatch):
    import ui.common as common

    if not (ROOT / "outputs" / "live" / "summaries" / "summaries.jsonl").exists():
        pytest.skip("no saved live results in this checkout")
    common.get_settings.clear()
    monkeypatch.setenv(ENV_VAR, "true")
    calls = []

    def forbidden(*_a, **_k):
        calls.append(1)
        raise AssertionError("no LLM client may be created in demo mode")

    import llm.factory

    monkeypatch.setattr(common, "get_llm_client", forbidden)
    monkeypatch.setattr(llm.factory, "get_llm_client", forbidden)
    yield calls
    common.get_settings.clear()


def test_real_results_unchanged_after_demo_session(real_demo):
    before = {d: _hash_tree(ROOT / d) for d in REAL_FILES}
    for name in PAGES:
        at = _page(name).run()
        assert not at.exception, (name, [e.value for e in at.exception])
    at = _page("prompt_comparison").run()
    assert _button(at, "Re-evaluate").disabled
    run_dir = ROOT / "outputs" / "live" / "experiments" / "20261111-030912_qwen2.5-3b"
    if run_dir.exists():  # calling the handler's backend directly is refused before anything is computed or written
        from evaluation.prompt_comparison import evaluate_run

        with pytest.raises(ReadOnlyModeError):
            evaluate_run(run_dir, [], 299)
    at = _page("summary_evaluation").run()
    assert all(b.disabled for b in at.button if "Judge" in b.label or "Re-judge" in b.label)
    at = _page("review_analysis").run()
    assert len(at.dataframe[0].value) == 285  # the stored valid qwen2.5:3b records
    assert any("real local-LLM inference" in t and "ollama/qwen2.5:3b" in t for t in _texts(at))
    assert real_demo == []
    assert {d: _hash_tree(ROOT / d) for d in REAL_FILES} == before


def test_reference_approval_is_disabled_and_writes_nothing_in_demo_mode(isolated_demo):
    import shutil

    from annotation.reference_adjudication import ReferenceAdjudicationStore
    from config import load_settings

    test_settings, _store, calls = isolated_demo
    shutil.copy(load_settings().paths.annotations_json, test_settings.paths.annotations_json)
    at = _page("annotation_page").run()
    assert not at.exception, [e.value for e in at.exception]
    assert _button(at, "Approve Existing Reference Annotations").disabled
    assert _button(at, "Save decision").disabled
    assert not ReferenceAdjudicationStore.default(test_settings).path.exists()
    at = _page("prompt_comparison").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not calls
