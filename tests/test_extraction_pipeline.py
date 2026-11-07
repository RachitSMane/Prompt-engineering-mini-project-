"""Strategies, caching/resume, mock client and exports."""

import json

import pandas as pd
import pytest

from config import load_settings
from llm.base import LLMUnavailableError
from llm.factory import get_llm_client
from llm.mock import CORRUPTIONS, MockLLMClient, corrupt, lexicon_extract
from pipeline.extraction import ExtractionStore, ReviewExtractor
from prompts.few_shot import FEW_SHOT_EXAMPLES
from prompts.strategies import STRATEGIES, get_strategy
from schemas.review_schema import LLMExtraction, ReviewInput
from utils.export import ASPECT_CSV_COLUMNS, RECORD_CSV_COLUMNS, aspects_to_dataframe, records_to_dataframe, records_to_json
from tests.conftest import REVIEW_TEXT, ScriptedLLM


def test_three_strategies_share_definitions_and_schema():
    msgs = {name: s.build_messages(REVIEW_TEXT) for name, s in STRATEGIES.items()}
    systems = {name: m[0].content for name, m in msgs.items()}
    for text in systems.values():
        assert "installation_difficulty" in text  # same taxonomy
        assert "overall_sentiment" in text  # same schema instructions
        assert REVIEW_TEXT not in text  # review only in user turn
    assert "EXAMPLE 1" in msgs["few_shot"][1].content and "EXAMPLE 1" not in msgs["zero_shot"][1].content
    assert "ANALYSIS PROCEDURE" in systems["cot"] and "ANALYSIS PROCEDURE" not in systems["zero_shot"]


def test_unknown_strategy():
    with pytest.raises(ValueError):
        get_strategy("tree_of_thought")


def test_few_shot_examples_are_schema_valid():
    for ex in FEW_SHOT_EXAMPLES:
        LLMExtraction.model_validate(ex["output"], context={"review_text": ex["review"]})


def test_store_resumes_and_skips_cached(settings, review, valid_json):
    store = ExtractionStore(settings.paths.outputs / "x.jsonl")
    client = ScriptedLLM([valid_json])
    ex = ReviewExtractor(client, get_strategy("zero_shot"), settings)
    run1 = ex.extract_many([review], store=store)
    assert run1.new_calls == 1
    store2 = ExtractionStore(settings.paths.outputs / "x.jsonl")  # reload from disk
    run2 = ReviewExtractor(ScriptedLLM([]), get_strategy("zero_shot"), settings).extract_many([review], store=store2)
    assert run2.from_cache == 1 and run2.new_calls == 0 and len(run2.records) == 1


def test_failed_reviews_are_retried(settings, review, valid_json):
    store = ExtractionStore(settings.paths.outputs / "y.jsonl")
    ReviewExtractor(ScriptedLLM(["a", "b", "c"]), get_strategy("zero_shot"), settings).extract_many([review], store=store)
    assert len(store.failures) == 1
    run = ReviewExtractor(ScriptedLLM([valid_json]), get_strategy("zero_shot"), settings).extract_many([review], store=store)
    assert len(run.records) == 1 and not ExtractionStore(store.path).failures


def test_factory_refuses_live_without_key(settings, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from dataclasses import replace

    with pytest.raises(LLMUnavailableError, match="GOOGLE_API_KEY"):
        get_llm_client(replace(settings, llm_provider="gemini"), use_mock=False)
    assert get_llm_client(settings, use_mock=True).is_mock


def test_mock_mode_flag_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    s = load_settings()
    assert s.is_mock and s.llm_model == "mock-offline-v1" and not s.live_llm_available()


def test_lexicon_mock_output_is_schema_valid():
    data = lexicon_extract(REVIEW_TEXT)
    ex = LLMExtraction.model_validate(data, context={"review_text": REVIEW_TEXT})
    cats = {a.aspect_category.value for a in ex.aspects}
    assert {"noise", "installation_setup"} <= cats


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_every_mock_corruption_is_recovered(settings, kind):
    """Every kind of malformed mock output must end in a valid record via local or LLM repair."""
    review = ReviewInput(review_id="C1", product_id="P", review_text=REVIEW_TEXT)
    bad = corrupt(lexicon_extract(REVIEW_TEXT), kind)
    client = ScriptedLLM([bad, json.dumps(lexicon_extract(REVIEW_TEXT))])
    out = ReviewExtractor(client, get_strategy("zero_shot"), settings).extract(review)
    assert out.extraction_metadata.validation_success, kind


def test_mock_client_end_to_end_marks_mock(settings):
    client = MockLLMClient(malformed_rate=100)
    reviews = [ReviewInput(review_id=f"M{i}", product_id="P", review_text=REVIEW_TEXT) for i in range(10)]
    run = ReviewExtractor(client, get_strategy("few_shot"), settings).extract_many(reviews)
    assert len(run.records) == 10
    assert all(r.extraction_metadata.is_mock for r in run.records)
    assert sum(r.extraction_metadata.repair_attempts + r.extraction_metadata.local_repairs > 0 for r in run.records) == 10


def test_exports_have_consistent_columns_and_valid_json(settings, review, valid_json):
    rec = ReviewExtractor(ScriptedLLM([valid_json]), get_strategy("zero_shot"), settings).extract(review)
    df = records_to_dataframe([rec])
    assert list(df.columns) == RECORD_CSV_COLUMNS
    assert list(records_to_dataframe([]).columns) == RECORD_CSV_COLUMNS
    assert list(aspects_to_dataframe([rec]).columns) == ASPECT_CSV_COLUMNS and len(aspects_to_dataframe([rec])) == 3
    data = json.loads(records_to_json([rec]))
    assert data[0]["aspects"][2]["sentiment"] == "negative"
    path = settings.paths.outputs / "r.csv"
    path.parent.mkdir(parents=True)
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert back.loc[0, "complaint_categories"] == "noise_issue"
    assert json.loads(back.loc[0, "aspects_json"])[0]["aspect_name"] == "ice maker"
