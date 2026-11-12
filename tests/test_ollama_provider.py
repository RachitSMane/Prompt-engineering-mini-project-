"""Local Ollama provider and provider-outage safety. Offline: no Ollama server or network needed."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from config import is_local_url, load_settings
from evaluation.prompt_comparison import ExperimentStopped, evaluate_run, evaluation_status, run_experiment
from llm.base import LLMError, LLMResponse, LLMUnavailableError
from llm.factory import get_llm_client
from pipeline.extraction import ExtractionStopped, ExtractionStore, ReviewExtractor
from prompts.strategies import get_strategy
from schemas.evaluation_schema import Annotation
from schemas.review_schema import ReviewInput
from tests.conftest import REVIEW_TEXT, valid_extraction


class FlakyLLM:
    """Live-looking client that answers validly, then fails like an unreachable provider after ``fail_after`` calls."""

    provider = "ollama"
    model_name = "qwen2.5:3b"
    is_mock = False

    def __init__(self, fail_after: int | None = None):
        self.fail_after, self.calls = fail_after, 0

    def invoke(self, messages, *, task, payload=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise LLMError("ollama call failed: ConnectError: connection refused")
        return LLMResponse(text=json.dumps(valid_extraction()), model_name=self.model_name, provider=self.provider, latency_s=0.01, input_tokens=10, output_tokens=5)


def _reviews(n: int) -> list[ReviewInput]:
    return [ReviewInput(review_id=f"R{i}", product_id="P1", review_text=REVIEW_TEXT) for i in range(n)]


def _gold(n: int) -> list[Annotation]:
    return [Annotation(review_id=f"R{i}", product_id="P1", overall_sentiment="mixed", aspects=[], complaint_categories=[], annotator="t") for i in range(n)]


# ------------------------------------------------------------------ configuration
def test_ollama_settings_from_env(monkeypatch):
    for k, v in {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen2.5:3b", "OLLAMA_BASE_URL": "http://127.0.0.1:11434/", "OLLAMA_NUM_CTX": "8192"}.items():
        monkeypatch.setenv(k, v)
    s = load_settings()
    assert (s.llm_provider, s.llm_model, s.ollama_base_url, s.ollama_num_ctx) == ("ollama", "qwen2.5:3b", "http://127.0.0.1:11434", 8192)
    assert s.live_llm_available() and s.api_key_env() is None  # no API key needed


@pytest.mark.parametrize("url,local", [("http://127.0.0.1:11434", True), ("http://localhost:11434", True), ("http://[::1]:11434", True),
                                       ("https://ollama.com", False), ("http://192.168.1.5:11434", False)])
def test_is_local_url(url, local):
    assert is_local_url(url) is local


def test_remote_ollama_endpoint_rejected_by_default(settings):
    remote = replace(settings, llm_provider="ollama", llm_model="qwen2.5:3b", ollama_base_url="https://ollama.example.com")
    assert not remote.live_llm_available()
    with pytest.raises(LLMUnavailableError, match="not a local address"):
        get_llm_client(remote, use_mock=False)
    assert replace(remote, ollama_allow_remote=True).live_llm_available()  # explicit opt-in only


def test_ollama_chat_model_is_local_and_bypasses_rate_limiter(settings):
    pytest.importorskip("langchain_ollama")
    from llm.langchain_client import build_chat_model

    s = replace(settings, llm_provider="ollama", ollama_base_url="http://127.0.0.1:11434", ollama_num_ctx=8192, llm_timeout_seconds=300, llm_requests_per_minute=10)
    m = build_chat_model("ollama", "qwen2.5:3b", s)
    assert type(m).__name__ == "ChatOllama"
    assert m.base_url == "http://127.0.0.1:11434" and m.model == "qwen2.5:3b"
    assert m.num_ctx == 8192 and m.num_predict == s.llm_max_output_tokens and m.temperature == s.llm_temperature
    assert m.client_kwargs == {"timeout": 300}
    assert m.rate_limiter is None  # the cloud requests-per-minute limiter is not applied locally


def test_cloud_providers_keep_rate_limiter(settings, monkeypatch):
    pytest.importorskip("langchain_groq")
    from llm.langchain_client import build_chat_model

    monkeypatch.setenv("GROQ_API_KEY", "test-not-used")
    m = build_chat_model("groq", "openai/gpt-oss-120b", replace(settings, llm_requests_per_minute=10))
    assert m.rate_limiter is not None


# ------------------------------------------------------------------ outage safety
def test_provider_error_stops_batch_and_is_not_cached(settings):
    store = ExtractionStore(settings.paths.outputs / "x.jsonl")
    with pytest.raises(ExtractionStopped) as info:
        ReviewExtractor(FlakyLLM(fail_after=2), get_strategy("zero_shot"), settings).extract_many(_reviews(5), store=store)
    assert info.value.review_id == "R2" and info.value.completed == 2
    reloaded = ExtractionStore(store.path)
    assert len(reloaded.records) == 2 and reloaded.failures == []  # the outage is not stored as a model failure
    client = FlakyLLM()
    run = ReviewExtractor(client, get_strategy("zero_shot"), settings).extract_many(_reviews(5), store=reloaded)
    assert client.calls == 3 and run.from_cache == 2 and len(run.records) == 5  # resume repeats no successful call


def test_experiment_stops_cleanly_and_is_never_complete_until_resumed(settings):
    reviews, gold = _reviews(4), _gold(4)
    with pytest.raises(ExperimentStopped) as info:
        run_experiment(settings, FlakyLLM(fail_after=6), ["zero_shot", "few_shot"], reviews, gold, run_id="R-test", required_gold=4)
    run_dir = info.value.run_dir
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "incomplete" and metrics["missing_predictions"] == {"zero_shot": 0, "few_shot": 2}
    assert metrics["strategies"]["few_shot"]["classification"]["n_evaluated"] == 2  # missing reviews are not scored as wrong
    assert metrics["strategies"]["few_shot"]["classification"]["n_failed_extractions"] == 0

    client = FlakyLLM()
    run_experiment(settings, client, ["zero_shot", "few_shot"], reviews, gold, run_id="R-test", required_gold=4)
    assert client.calls == 2  # only the two missing few-shot reviews
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "complete" and not any(metrics["missing_predictions"].values())


def test_legacy_provider_failures_count_as_missing_not_wrong(settings):
    reviews, gold = _reviews(2), _gold(2)
    run_dir = run_experiment(settings, FlakyLLM(), ["zero_shot"], reviews, gold, run_id="R-legacy", required_gold=2)
    pred = run_dir / "predictions_zero_shot.jsonl"
    rows = [json.loads(line) for line in pred.read_text(encoding="utf-8").splitlines()]
    rec = rows[1]["record"]
    legacy = {"review_id": rec["review_id"], "product_id": rec["product_id"], "review_text": rec["review_text"],
              "error": "LLM provider error: groq call failed: RateLimitError", "extraction_metadata": rec["extraction_metadata"]}
    pred.write_text(json.dumps(rows[0]) + "\n" + json.dumps({"cache_key": rows[1]["cache_key"], "status": "failed", "record": legacy}) + "\n", encoding="utf-8")
    metrics = evaluate_run(run_dir, gold, 2)
    assert metrics["status"] == "incomplete"
    cl = metrics["strategies"]["zero_shot"]["classification"]
    assert cl["n_evaluated"] == 1 and cl["n_failed_extractions"] == 0


def test_status_never_complete_with_missing_predictions():
    assert evaluation_status(False, 299, 299, missing_predictions=3) == "incomplete"
    assert evaluation_status(False, 299, 299) == "complete"
    assert evaluation_status(False, 5, 299) == "provisional"
    assert evaluation_status(True, 299, 299, missing_predictions=3) == "mock_demo"
