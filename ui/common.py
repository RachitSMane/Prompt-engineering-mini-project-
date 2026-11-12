"""Shared Streamlit helpers: settings, LLM mode, banners and cached loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import Settings, load_settings
from llm.base import LLMClient, LLMUnavailableError
from llm.factory import get_llm_client
from pipeline.extraction import ExtractionStore
from ui.components import callout, esc, kv_html, md
from utils.data_loader import load_reviews
from utils.storage import read_json, read_jsonl

MOCK_BANNER = (
    "<b>Offline mock mode.</b> Outputs come from a deterministic keyword fixture, not a language model. "
    "They exist to develop and demonstrate the pipeline and must never be reported as experimental results."
)


@st.cache_resource
def get_settings() -> Settings:
    return load_settings()


def use_mock() -> bool:
    return bool(st.session_state.get("use_mock", False))


def mode_dir() -> Path:
    return get_settings().paths.mode_dir(use_mock())


@st.cache_data(ttl=15, show_spinner=False)
def _ollama_models(base_url: str) -> list[str] | None:
    from llm.langchain_client import ollama_models

    return ollama_models(base_url)  # lists installed models only; no inference


def provider_status(s: Settings, provider: str, model: str) -> tuple[bool, str, str]:
    """(ready, short label, explanation) for one provider/model pair. Lists models only; no inference."""
    if not s.live_llm_available(provider):
        label = "Ollama endpoint blocked" if provider == "ollama" else "No LLM key"
        return False, label, s.live_unavailable_reason(provider)
    if provider == "ollama":
        from llm.langchain_client import model_installed

        names = _ollama_models(s.ollama_base_url)
        if names is None:
            return False, "Ollama not running", f"No Ollama server answered at {s.ollama_base_url}."
        if not model_installed(model, names):
            return False, "Model not pulled", f"Ollama is running but '{model}' is not installed (ollama pull {model})."
        return True, "Local LLM ready", ""
    return True, "LLM connected", ""


def live_status(s: Settings) -> tuple[bool, str, str]:
    """Status of the configured extraction provider."""
    return provider_status(s, s.llm_provider, s.llm_model)


def judge_status(s: Settings) -> tuple[bool, str, str]:
    """Status of the configured LLM-as-a-Judge (it can differ from the extraction model)."""
    return provider_status(s, s.effective_judge_provider, s.effective_judge_model)


def render_sidebar() -> None:
    """Compact connection + mode block shown under the navigation."""
    s = get_settings()
    live, label, _ = live_status(s)
    if "use_mock" not in st.session_state:
        st.session_state["use_mock"] = s.is_mock or not live
    with st.sidebar:
        llm_row = (
            f'<div class="riq-side-row"><span class="riq-dot on"></span>{esc(label)}<span class="riq-side-sub">{esc(s.llm_provider)} · {esc(s.llm_model)}</span></div>'
            if live
            else f'<div class="riq-side-row"><span class="riq-dot off"></span>{esc(label)}<span class="riq-side-sub">{esc(s.llm_provider)}</span></div>'
        )
        md(f'<div class="riq-side-status"><div class="riq-side-label">Workspace</div>{llm_row}</div>')
        st.toggle(
            "Offline mock mode",
            key="use_mock",
            help="Uses deterministic fixtures instead of an LLM. Mock outputs are stored separately in outputs/mock/ and are never experimental results.",
        )
        with st.expander("About", icon=":material/info:"):
            md(
                '<div class="riq-side-note">Aspect-based review analytics with validated structured outputs.</div>'
                + kv_html([
                    ("Extraction", f"{s.llm_provider} / {s.llm_model}"),
                    ("Judge", f"{s.effective_judge_provider} / {s.effective_judge_model}"),
                    ("Artefacts", f"outputs/{'mock' if use_mock() else 'live'}/"),
                ])
            )
            if not live:
                st.caption(live_status(s)[2])


def mock_banner() -> None:
    if use_mock():
        callout(MOCK_BANNER, "warning", tag="Mock")


MOCK_HINT = "or switch on <b>Offline mock mode</b> in the sidebar to try the pipeline with fixtures (not real results)."


def setup_message(s: Settings) -> tuple[str, str]:
    """(how to enable the configured provider, callout tag)."""
    if s.llm_provider == "ollama":
        return (
            f"Start Ollama (<code>ollama serve</code>), install the model (<code>ollama pull {esc(s.llm_model)}</code>) and reload the page, {MOCK_HINT}",
            "Local LLM",
        )
    env = s.api_key_env() or "the provider's API key"
    return f"Copy <code>.env.example</code> to <code>.env</code>, set <code>{esc(env)}</code> and restart the app, {MOCK_HINT}", "API key"


def get_client(role: str = "extraction") -> LLMClient | None:
    """Return the client for the current mode, or None after showing a clear message."""
    s = get_settings()
    try:
        return get_llm_client(s, use_mock=use_mock(), role=role)
    except LLMUnavailableError as exc:
        msg, tag = setup_message(s)
        callout(f"<b>This action needs a live LLM.</b> {esc(exc)} {msg}", "accent", tag=tag)
        return None


def llm_ready() -> bool:
    s = get_settings()
    return use_mock() or live_status(s)[0]


def require_llm_note() -> None:
    if not llm_ready():
        s = get_settings()
        msg, tag = setup_message(s)
        callout(f"<b>Live LLM features are disabled.</b> {esc(live_status(s)[2])} {msg}", "accent", tag=tag)


@st.cache_data(show_spinner=False)
def cached_reviews(path_str: str, mtime: float) -> pd.DataFrame:
    return load_reviews(Path(path_str))


def dataset() -> pd.DataFrame | None:
    p = get_settings().paths.reviews_csv
    if not p.exists():
        return None
    return cached_reviews(str(p), p.stat().st_mtime)


def dataset_card() -> dict:
    return read_json(get_settings().paths.data_processed / "dataset_card.json", default={}) or {}


def extraction_files() -> list[Path]:
    d = mode_dir() / "extractions"
    return sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []


def load_store(path: Path) -> ExtractionStore:
    return ExtractionStore(path)


def activity(limit: int = 10) -> pd.DataFrame:
    rows = [r for r in read_jsonl(get_settings().paths.activity_log) if r.get("mode") == ("mock" if use_mock() else "live")]
    df = pd.DataFrame(rows[-limit:][::-1])
    return df


def pretty(label: str) -> str:
    return label.replace("_", " ")
