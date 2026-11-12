"""Plotly figure builders used by the dashboard (pure functions, easy to test).

All figures use the shared dark "reviewiq" template, registered here as the Plotly default.
Colours come from ui/theme.py, so every page uses the same palette.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from ui.theme import COLORS, FONT, SENTIMENT, SERIES

SENTIMENT_COLORS = dict(SENTIMENT)
LAYOUT = dict(margin=dict(l=8, r=8, t=8, b=8), height=320)

_AXIS = dict(
    gridcolor=COLORS["border_soft"],
    linecolor=COLORS["border"],
    zerolinecolor=COLORS["border"],
    tickfont=dict(size=11, color=COLORS["text_2"]),
    title=dict(font=dict(size=11, color=COLORS["muted"])),
    automargin=True,
)

TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT["family"], size=12, color=COLORS["text_2"]),
        title=dict(font=dict(size=13, color=COLORS["text"]), x=0, xanchor="left"),
        colorway=SERIES,
        xaxis=_AXIS,
        yaxis=_AXIS,
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", xanchor="left", font=dict(size=11, color=COLORS["text_2"]), bgcolor="rgba(0,0,0,0)", title=dict(text="")),
        hoverlabel=dict(bgcolor=COLORS["elevated"], bordercolor=COLORS["border"], font=dict(family=FONT["family"], size=12, color=COLORS["text"])),
        bargap=0.35,
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=COLORS["text_2"]))),
    )
)
pio.templates["reviewiq"] = TEMPLATE
pio.templates.default = "reviewiq"


def _finish(fig: go.Figure, title: str | None, height: int, **layout) -> go.Figure:
    top = 36 if title else (32 if fig.layout.showlegend is not False and len(fig.data) > 1 else 8)
    fig.update_layout(template="reviewiq", height=height, margin=dict(l=8, r=8, t=top, b=8), title=dict(text=title or ""), **layout)
    return fig


def _empty(title: str | None, message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=COLORS["muted"], size=12), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _finish(fig, title, height)


def sentiment_pie(counts: dict[str, int], title: str | None = None) -> go.Figure:
    """Donut of overall sentiment with the total in the centre."""
    df = pd.DataFrame({"sentiment": list(counts), "count": list(counts.values())})
    df = df[df["count"] > 0]
    if df.empty:
        return _empty(title, "No reviews")
    fig = go.Figure(
        go.Pie(
            labels=df["sentiment"],
            values=df["count"],
            hole=0.66,
            sort=False,
            marker=dict(colors=[SENTIMENT_COLORS.get(s, COLORS["accent"]) for s in df["sentiment"]], line=dict(color=COLORS["surface"], width=2)),
            textinfo="none",
            hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(text=f"<b>{int(df['count'].sum()):,}</b><br><span style='font-size:11px;color:{COLORS['muted']}'>reviews</span>", showarrow=False, font=dict(size=20, color=COLORS["text"]))
    return _finish(fig, title, 280, showlegend=True, legend=dict(orientation="v", x=1.0, xanchor="left", y=0.5, yanchor="middle"))


def aspect_sentiment_bar(stats: list[dict], title: str | None = None) -> go.Figure:
    """Diverging bar: negative mentions to the left, neutral and positive to the right."""
    df = pd.DataFrame(stats)
    if df.empty:
        return _empty(title, "No aspects extracted")
    df = df.sort_values("mentions")
    df["label"] = df["aspect_category"].str.replace("_", " ")
    fig = go.Figure()
    for name, sign in (("negative", -1), ("neutral", 1), ("positive", 1)):
        fig.add_bar(
            y=df["label"], x=sign * df[name], name=name, orientation="h", marker_color=SENTIMENT_COLORS[name],
            customdata=df[name], hovertemplate="%{y} · " + name + ": %{customdata}<extra></extra>",
        )
    return _finish(fig, title, max(260, 26 * len(df) + 70), barmode="relative", bargap=0.4, xaxis_title="mentions  (← negative · positive →)")


def counts_bar(counts: dict[str, int], title: str | None = None, color: str = SENTIMENT["negative"]) -> go.Figure:
    df = pd.DataFrame({"label": [k.replace("_", " ") for k in counts], "count": list(counts.values())}).sort_values("count")
    if df.empty:
        return _empty(title, "None recorded")
    fig = go.Figure(go.Bar(x=df["count"], y=df["label"], orientation="h", marker_color=color, hovertemplate="%{y}: %{x}<extra></extra>"))
    return _finish(fig, title, max(220, 26 * len(df) + 60), xaxis_title="reviews")


def distribution_bar(counts: dict, title: str | None = None, x_title: str = "", y_title: str = "reviews", color: str = COLORS["accent"]) -> go.Figure:
    """Vertical bar of a categorical distribution (e.g. star ratings)."""
    fig = go.Figure(go.Bar(x=[str(k) for k in counts], y=list(counts.values()), marker_color=color, hovertemplate="%{x}: %{y}<extra></extra>"))
    return _finish(fig, title, 240, xaxis_title=x_title, yaxis_title=y_title, xaxis_type="category")


def histogram(values, title: str | None = None, x_title: str = "", nbins: int = 30, color: str = "#38BDF8") -> go.Figure:
    fig = go.Figure(go.Histogram(x=list(values), nbinsx=nbins, marker_color=color, hovertemplate="%{x}: %{y}<extra></extra>"))
    return _finish(fig, title, 240, xaxis_title=x_title, yaxis_title="reviews", bargap=0.08)


def aspect_trend_line(trend: list[dict], top_n: int = 5, title: str | None = None) -> go.Figure:
    df = pd.DataFrame(trend)
    if df.empty:
        return _empty(title, "No dated reviews")
    top = df.groupby("aspect_category")["mentions"].sum().nlargest(top_n).index
    df = df[df["aspect_category"].isin(top)].sort_values("period").copy()
    df["period"] = df["period"].astype(str)
    df["aspect"] = df["aspect_category"].str.replace("_", " ")
    fig = px.line(df, x="period", y="net_sentiment", color="aspect", markers=True, hover_data=["mentions"], color_discrete_sequence=SERIES)
    return _finish(fig, title, 300, yaxis_range=[-1.05, 1.05], yaxis_title="net sentiment", xaxis_title="", xaxis_type="category", legend_title_text="")


def confusion_heatmap(matrix: list[list[int]], labels: list[str], title: str | None = None) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=labels, y=labels, text=matrix, texttemplate="%{text}", textfont=dict(color=COLORS["text"], size=12),
            colorscale=[[0, COLORS["surface_2"]], [1, COLORS["accent_hover"]]], showscale=False, xgap=2, ygap=2,
            hovertemplate="reference: %{y}<br>predicted: %{x}<br>count: %{z}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return _finish(fig, title, 300, xaxis_title="predicted", yaxis_title="reference label")


def metric_comparison_bar(table: pd.DataFrame, metrics: list[str], title: str | None = None) -> go.Figure:
    cols = [m for m in metrics if m in table.columns]
    df = table.melt(id_vars="strategy", value_vars=cols, var_name="metric", value_name="value").dropna()
    if df.empty:
        return _empty(title, "No metrics yet")
    df["metric"] = df["metric"].str.replace("_", " ")
    fig = px.bar(df, x="metric", y="value", color="strategy", barmode="group", text_auto=".2f", color_discrete_sequence=SERIES)
    fig.update_traces(textfont_size=10, textposition="outside", cliponaxis=False)
    return _finish(fig, title, 340, yaxis_range=[0, 1.1], xaxis_title="", yaxis_title="score")


def strategy_bar(table: pd.DataFrame, value: str, title: str | None = None, y_title: str = "", fmt: str = ".2f") -> go.Figure:
    """One bar per strategy (e.g. latency or macro-F1), coloured with the strategy series."""
    df = table[["strategy", value]].dropna()
    if df.empty:
        return _empty(title, "No data")
    fig = go.Figure(go.Bar(x=df["strategy"], y=df[value], marker_color=SERIES[: len(df)], text=df[value], texttemplate="%{text:" + fmt + "}", textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>"))
    return _finish(fig, title, 280, yaxis_title=y_title, xaxis_title="")


def grouped_strategy_bar(table: pd.DataFrame, values: list[str], title: str | None = None, y_title: str = "") -> go.Figure:
    df = table.melt(id_vars="strategy", value_vars=values, var_name="series", value_name="value").dropna()
    if df.empty:
        return _empty(title, "No data")
    df["series"] = df["series"].str.replace("mean_", "").str.replace("_", " ")
    fig = px.bar(df, x="strategy", y="value", color="series", barmode="group", color_discrete_sequence=[COLORS["accent"], "#38BDF8"])
    return _finish(fig, title, 280, yaxis_title=y_title, xaxis_title="")


def per_class_bar(per_class_by_strategy: dict[str, dict], metric: str = "f1-score", title: str | None = None) -> go.Figure:
    """Grouped bar: per-class score for each strategy."""
    rows = [{"strategy": s, "class": c, "value": v.get(metric)} for s, pc in per_class_by_strategy.items() for c, v in pc.items()]
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return _empty(title, "No per-class scores")
    fig = px.bar(df, x="class", y="value", color="strategy", barmode="group", color_discrete_sequence=SERIES, text_auto=".2f")
    fig.update_traces(textfont_size=10, textposition="outside", cliponaxis=False)
    return _finish(fig, title, 300, yaxis_range=[0, 1.1], xaxis_title="", yaxis_title=metric.replace("-score", " score"))


def judge_scores_bar(rows: list[dict], title: str | None = None) -> go.Figure:
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(title, "No judgements yet")
    dims = ["faithfulness", "coverage", "consistency", "hallucination"]
    m = df[dims].mean()
    fig = go.Figure(go.Bar(x=[d.capitalize() for d in dims], y=m.values, marker_color=COLORS["accent"], text=m.values, texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.2f} / 5<extra></extra>"))
    return _finish(fig, title, 280, yaxis_range=[0, 5.4], yaxis_title="mean score (1–5)", xaxis_title="")


def judge_scores_per_summary(rows: list[dict], title: str | None = None) -> go.Figure:
    """Heatmap-style table of scores: one row per summary, one column per dimension."""
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty(title, "No judgements yet")
    dims = ["faithfulness", "coverage", "consistency", "hallucination"]
    labels = [str(t)[:38] for t in df.get("product_title", df.get("product_id"))]
    fig = go.Figure(
        go.Heatmap(
            z=df[dims].values, x=[d.capitalize() for d in dims], y=labels, text=df[dims].values, texttemplate="%{text}",
            zmin=1, zmax=5, colorscale=[[0, "#3B1F2B"], [0.5, COLORS["surface_2"]], [1, "#1F4D44"]], showscale=False, xgap=2, ygap=2,
            textfont=dict(color=COLORS["text"]), hovertemplate="%{y}<br>%{x}: %{z} / 5<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _finish(fig, title, max(220, 26 * len(df) + 60))
