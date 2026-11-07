"""Reusable ReviewIQ UI components.

Small functions that render consistent building blocks (page header, KPI cards, panels,
callouts, empty states, ranked bars, badges). HTML-returning helpers end in `_html` or
return `str`; helpers that render directly call Streamlit. All text is HTML-escaped.
"""

from __future__ import annotations

import html as _html
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ui.theme import COLORS, SENTIMENT, css_variables

_CSS_PATH = Path(__file__).with_name("styles.css")
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}

SENTIMENT_TONE = {"positive": "positive", "negative": "negative", "neutral": "neutral", "mixed": "mixed"}


def esc(value: object) -> str:
    return _html.escape(str(value), quote=True)


def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def inject_css() -> None:
    """Load the stylesheet once per run (style-only st.html takes no layout space)."""
    st.html(f"<style>{css_variables()}\n{_CSS_PATH.read_text(encoding='utf-8')}</style>")


# --------------------------------------------------------------------------- atoms
def icon(name: str, size: int = 16) -> str:
    return f'<span class="riq-icon" style="font-size:{size}px" aria-hidden="true">{esc(name)}</span>'


def badge(text: str, tone: str = "neutral", dot: bool = False) -> str:
    d = '<span class="riq-dot" style="background:currentColor"></span>' if dot else ""
    return f'<span class="riq-badge riq-tone-{tone}">{d}{esc(text)}</span>'


def mode_badge(is_mock: bool) -> str:
    return badge("Offline mock", "warning", dot=True) if is_mock else badge("Live", "positive", dot=True)


def sentiment_badge(value: str | None) -> str:
    if not value:
        return badge("—", "neutral")
    return badge(value, SENTIMENT_TONE.get(value, "neutral"))


def chips(values: Iterable[str]) -> str:
    vals = [v for v in values if v]
    return "".join(f'<span class="riq-chip">{esc(v.replace("_", " "))}</span>' for v in vals) or '<span class="riq-chip">none</span>'


# --------------------------------------------------------------------------- page structure
def page_header(title: str, subtitle: str, *, is_mock: bool | None = None, meta: Sequence[str] = ()) -> None:
    """Title (an accessible h1), one-line subtitle and a compact mode badge on the right."""
    left, right = st.columns([5, 2], vertical_alignment="top")
    with left:
        st.title(title, anchor=False)
        md(f'<p class="riq-subtitle">{esc(subtitle)}</p>')
    with right:
        badges = list(meta) + ([mode_badge(is_mock)] if is_mock is not None else [])
        if badges:
            md(f'<div class="riq-header-meta">{"".join(badges)}</div>')


def section(title: str, caption: str | None = None, right_html: str = "") -> None:
    cap = f'<span class="riq-section-caption">{esc(caption)}</span>' if caption else ""
    md(f'<div class="riq-section"><div><div class="riq-section-title">{esc(title)}</div>{cap}</div><div>{right_html}</div></div>')


@contextmanager
def panel(title: str | None = None, caption: str | None = None, right_html: str = "", height: int | str = "content") -> Iterator[None]:
    """A bordered surface with an optional title row. Use sparingly: one level, never nested."""
    with st.container(border=True, height=height):
        if title:
            md(f'<div class="riq-panel-title"><span>{esc(title)}</span><span style="margin-left:auto">{right_html}</span></div>')
        if caption:
            md(f'<div class="riq-panel-caption">{esc(caption)}</div>')
        yield


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the ReviewIQ template (Streamlit's own theme is disabled)."""
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CONFIG, key=key)


# --------------------------------------------------------------------------- KPI cards
def kpi(label: str, value: object, caption: str = "", tone: str | None = None, icon_name: str | None = None) -> str:
    cls = f" riq-{tone}" if tone else ""
    muted = " riq-muted" if str(value) in ("—", "Pending") else ""
    ic = icon(icon_name, 16) if icon_name else ""
    return (
        f'<div class="riq-kpi{cls}"><div class="riq-kpi-label">{esc(label)}{ic}</div>'
        f'<div class="riq-kpi-value{muted}">{esc(value)}</div><div class="riq-kpi-caption">{esc(caption)}</div></div>'
    )


def kpi_row(cards: Sequence[str], min_width: int = 180) -> None:
    md(f'<div class="riq-kpi-grid" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">{"".join(cards)}</div>')


# --------------------------------------------------------------------------- feedback
def callout(body_html: str, tone: str = "accent", tag: str | None = None) -> None:
    """Compact inline notice. `body_html` may contain trusted inline HTML (<b>, <code>)."""
    t = f'<span class="riq-callout-tag">{esc(tag)}</span>' if tag else ""
    md(f'<div class="riq-callout riq-tone-{tone}">{t}<div>{body_html}</div></div>')


def empty_state(title: str, body_html: str = "", icon_name: str = "inbox") -> None:
    md(
        f'<div class="riq-empty"><div class="riq-empty-icon">{icon(icon_name, 20)}</div>'
        f'<div class="riq-empty-title">{esc(title)}</div><div class="riq-empty-body">{body_html}</div></div>'
    )


# --------------------------------------------------------------------------- data display
def ranked_bars_html(items: Sequence[tuple[str, int]], color: str = COLORS["accent"], limit: int = 8, total: int | None = None) -> str:
    items = list(items)[:limit]
    if not items:
        return '<div class="riq-panel-caption">None recorded.</div>'
    top = max(c for _, c in items) or 1
    rows = []
    for label, count in items:
        pct = 100 * count / top
        val = f"{count}" if total is None else f"{100 * count / max(1, total):.0f}%"
        rows.append(
            f'<div class="riq-rank" title="{esc(label)}: {count}"><span class="riq-rank-label">{esc(str(label).replace("_", " "))}</span>'
            f'<span class="riq-rank-track"><span class="riq-rank-fill" style="display:block;width:{pct:.1f}%;background:{color}"></span></span>'
            f'<span class="riq-rank-value">{esc(val)}</span></div>'
        )
    return "".join(rows)


def kv_html(rows: Sequence[tuple[str, object]]) -> str:
    return '<div class="riq-kv">' + "".join(f'<span class="k">{esc(k)}</span><span class="v" title="{esc(v)}">{esc(v)}</span>' for k, v in rows) + "</div>"


def score_html(score: int, max_score: int = 5) -> str:
    level = "good" if score >= 4 else "mid" if score == 3 else "bad"
    segs = "".join(f'<span class="{"on" if i < score else ""}"></span>' for i in range(max_score))
    return f'<div class="riq-score {level}" role="img" aria-label="{score} of {max_score}">{segs}</div>'


def sentiment_color(value: str) -> str:
    return SENTIMENT.get(value, COLORS["accent"])
