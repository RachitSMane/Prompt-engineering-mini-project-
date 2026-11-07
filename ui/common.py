"""Shared Streamlit helpers: settings, LLM mode, banners and cached loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import Settings, load_settings
from llm.base import LLMClient, LLMUnavailableError
from llm.factory import get_llm_client
from pipeline.extraction import ExtractionStore
from utils.data_loader import load_reviews
from utils.storage import read_json, read_jsonl

MOCK_BANNER = (
    "🧪 **MOCK / OFFLINE MODE** — outputs below come from a deterministic keyword fixture, **not a language model**. "
    "They exist only to develop and demonstrate the pipeline and must never be reported as experimental results."
)


@st.cache_resource
def get_settings() -> Settings:
    return load_settings()


def use_mock() -> bool:
    return bool(st.session_state.get("use_mock", False))


def mode_dir() -> Path:
    return get_settings().paths.mode_dir(use_mock())


def render_sidebar() -> None:
    s = get_settings()
    live = s.live_llm_available()
    if "use_mock" not in st.session_state:
        st.session_state["use_mock"] = s.is_mock or not live
    with st.sidebar:
        st.markdown("### LLM status")
        if live:
            st.success(f"Live LLM ready: **{s.llm_provider} / {s.llm_model}**")
        else:
            st.error("Live LLM unavailable")
            st.caption(s.live_unavailable_reason())
        st.toggle(
            "Offline MOCK mode (development only)",
            key="use_mock",
            help="Uses deterministic fixtures instead of an LLM. Mock outputs are stored separately in outputs/mock/ and are never experimental results.",
        )
        if use_mock():
            st.warning("Mock mode ON — results are fixtures, not LLM outputs.")
        st.caption(f"Viewing **{'mock' if use_mock() else 'live'}** artefacts · judge: {s.effective_judge_provider}/{s.effective_judge_model}")


def mock_banner() -> None:
    if use_mock():
        st.warning(MOCK_BANNER)


def get_client(role: str = "extraction") -> LLMClient | None:
    """Return the client for the current mode, or None after showing a clear message."""
    s = get_settings()
    try:
        return get_llm_client(s, use_mock=use_mock(), role=role)
    except LLMUnavailableError as exc:
        st.info(
            f"🔑 **This action needs a live LLM.** {exc}\n\n"
            "Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` (free key from Google AI Studio), and restart the app. "
            "Alternatively, switch on *Offline MOCK mode* in the sidebar to try the pipeline with fixtures (not real results)."
        )
        return None


def llm_ready() -> bool:
    s = get_settings()
    return use_mock() or s.live_llm_available()


def require_llm_note() -> None:
    if not llm_ready():
        st.info("🔑 Live LLM features are disabled: no API key configured. See the sidebar, or enable Offline MOCK mode for a fixture-based demo.")


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
