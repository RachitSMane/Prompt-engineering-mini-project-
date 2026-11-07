"""Plotly figure builders used by the dashboard (pure functions, easy to test)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SENTIMENT_COLORS = {"positive": "#2E7D32", "negative": "#C62828", "neutral": "#78909C", "mixed": "#F9A825"}
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=340)


def sentiment_pie(counts: dict[str, int], title: str = "Overall sentiment") -> go.Figure:
    df = pd.DataFrame({"sentiment": list(counts), "count": list(counts.values())})
    df = df[df["count"] > 0]
    fig = px.pie(df, names="sentiment", values="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.45, title=title)
    fig.update_layout(**LAYOUT)
    return fig


def aspect_sentiment_bar(stats: list[dict], title: str = "Aspect sentiment") -> go.Figure:
    """Diverging bar: positive to the right, negative to the left."""
    df = pd.DataFrame(stats)
    if df.empty:
        return go.Figure().update_layout(title=title + " (no aspects)", **LAYOUT)
    df = df.sort_values("mentions")
    fig = go.Figure()
    fig.add_bar(y=df["aspect_category"], x=-df["negative"], name="negative", orientation="h", marker_color=SENTIMENT_COLORS["negative"])
    fig.add_bar(y=df["aspect_category"], x=df["neutral"], name="neutral", orientation="h", marker_color=SENTIMENT_COLORS["neutral"])
    fig.add_bar(y=df["aspect_category"], x=df["positive"], name="positive", orientation="h", marker_color=SENTIMENT_COLORS["positive"])
    fig.update_layout(barmode="relative", title=title, xaxis_title="reviews (negative ← → positive)", **{**LAYOUT, "height": max(300, 28 * len(df) + 80)})
    return fig


def counts_bar(counts: dict[str, int], title: str, color: str = "#C62828") -> go.Figure:
    df = pd.DataFrame({"label": list(counts), "count": list(counts.values())}).sort_values("count")
    if df.empty:
        return go.Figure().update_layout(title=title + " (none)", **LAYOUT)
    fig = px.bar(df, x="count", y="label", orientation="h", title=title)
    fig.update_traces(marker_color=color)
    fig.update_layout(yaxis_title="", **{**LAYOUT, "height": max(260, 28 * len(df) + 80)})
    return fig


def aspect_trend_line(trend: list[dict], top_n: int = 5) -> go.Figure:
    df = pd.DataFrame(trend)
    if df.empty:
        return go.Figure().update_layout(title="Aspect sentiment trend (no dated data)", **LAYOUT)
    top = df.groupby("aspect_category")["mentions"].sum().nlargest(top_n).index
    df = df[df["aspect_category"].isin(top)].sort_values("period")
    fig = px.line(df, x="period", y="net_sentiment", color="aspect_category", markers=True, title="Net aspect sentiment by year (top aspects)", hover_data=["mentions"])
    fig.update_layout(yaxis_range=[-1.05, 1.05], **LAYOUT)
    return fig


def confusion_heatmap(matrix: list[list[int]], labels: list[str], title: str) -> go.Figure:
    fig = px.imshow(matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Human label", color="count"), title=title)
    fig.update_layout(**{**LAYOUT, "height": 360})
    return fig


def metric_comparison_bar(table: pd.DataFrame, metrics: list[str], title: str = "Strategy comparison") -> go.Figure:
    cols = [m for m in metrics if m in table.columns]
    df = table.melt(id_vars="strategy", value_vars=cols, var_name="metric", value_name="value").dropna()
    fig = px.bar(df, x="metric", y="value", color="strategy", barmode="group", title=title, text_auto=".3f")
    fig.update_layout(yaxis_range=[0, 1.05], **{**LAYOUT, "height": 400})
    return fig


def judge_scores_bar(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure().update_layout(title="No judgements yet", **LAYOUT)
    dims = ["faithfulness", "coverage", "consistency", "hallucination"]
    m = df[dims].mean().reset_index()
    m.columns = ["dimension", "mean score"]
    fig = px.bar(m, x="dimension", y="mean score", text_auto=".2f", title=f"Mean judge scores over {len(df)} summaries (1-5)")
    fig.update_layout(yaxis_range=[0, 5.2], **LAYOUT)
    return fig
