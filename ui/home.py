"""Page 1 — Overview dashboard."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from annotation.annotation_manager import AnnotationStore, load_sample
from evaluation.prompt_comparison import list_runs
from evaluation.summary_evaluation import JudgementStore
from evaluation.visualizations import aspect_sentiment_bar, distribution_bar, histogram, sentiment_pie
from pipeline.aggregation import overview_stats
from pipeline.summarization import SummaryStore
from ui.common import activity, dataset, dataset_card, extraction_files, get_settings, load_store, mock_banner, use_mock
from ui.components import badge, chart, empty_state, esc, kpi, kpi_row, kv_html, md, page_header, panel, ranked_bars_html, section
from ui.theme import COLORS, SENTIMENT

s = get_settings()
mock = use_mock()
page_header("Overview", "Understand customer sentiment and product performance across your review dataset.", is_mock=mock)
mock_banner()

df = dataset()
card = dataset_card()
records = {}
for f in extraction_files():
    for r in load_store(f).records:
        records.setdefault(r.review_id, r)
ov = overview_stats(list(records.values())) if records else None

# ---------------------------------------------------------------- KPIs
n_total = len(df) if df is not None else 0
n_products = df["product_id"].nunique() if df is not None else 0
mode_word = "mock" if mock else "live"
if ov:
    sc, tot = ov["sentiment_counts"], ov["total"]
    pos_pct, neg_pct = 100 * sc.get("positive", 0) / tot, 100 * sc.get("negative", 0) / tot
    cards = [
        kpi("Total reviews", f"{n_total:,}" if df is not None else "—", f"{tot:,} processed ({mode_word})", icon_name="reviews"),
        kpi("Products analysed", f"{ov['products']}", f"of {n_products} products in the dataset", icon_name="inventory_2"),
        kpi("Positive sentiment", f"{pos_pct:.1f}%", f"{sc.get('positive', 0):,} of {tot:,} processed reviews", tone="positive", icon_name="trending_up"),
        kpi("Negative sentiment", f"{neg_pct:.1f}%", f"{sc.get('negative', 0):,} of {tot:,} processed reviews", tone="negative", icon_name="trending_down"),
    ]
else:
    cards = [
        kpi("Total reviews", f"{n_total:,}" if df is not None else "—", "in the prepared dataset" if df is not None else "dataset not prepared", icon_name="reviews"),
        kpi("Products analysed", "—", f"{n_products} products available · none processed", icon_name="inventory_2"),
        kpi("Positive sentiment", "—", "no processed reviews yet", icon_name="trending_up"),
        kpi("Negative sentiment", "—", "no processed reviews yet", icon_name="trending_down"),
    ]
kpi_row(cards)

# ---------------------------------------------------------------- sentiment & aspects
if not ov:
    empty_state(
        f"No reviews processed in {mode_word} mode yet",
        "Run extraction on <b>Review Analysis</b> or <b>Product Intelligence</b>. Sentiment and aspect analytics appear here once structured records exist.",
        "analytics",
    )
else:
    stats = [{"aspect_category": c, "mentions": sum(v.values()), "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0)} for c, v in ov["aspect_counts"].items()]
    left, right = st.columns([2, 3], gap="medium")
    with left, panel("Overall sentiment", f"Share of {ov['total']:,} processed reviews"):
        chart(sentiment_pie(ov["sentiment_counts"]), key="ov_pie")
    with right, panel("Aspect sentiment breakdown", "Mentions per aspect category, split by sentiment"):
        chart(aspect_sentiment_bar(stats), key="ov_aspects")

    a, b, c = st.columns(3, gap="medium")
    with a, panel("Most mentioned aspects", "Aspect categories by number of mentions"):
        md(ranked_bars_html(sorted(((x["aspect_category"], x["mentions"]) for x in stats), key=lambda t: -t[1]), COLORS["accent"]))
    with b, panel("Top complaint categories", f"{sum(bool(r.complaint_categories) for r in records.values())} reviews contain at least one complaint"):
        md(ranked_bars_html(list(ov["complaint_counts"].items()), SENTIMENT["negative"]))
    with c, panel("Recent activity", f"Latest {mode_word} pipeline runs"):
        act = activity(8)
        if act.empty:
            md('<div class="riq-panel-caption">No activity recorded yet.</div>')
        else:
            def _ago(ts: str) -> str:
                try:
                    secs = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
                except (TypeError, ValueError):
                    return ""
                for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
                    if secs >= n:
                        return f"{int(secs // n)}{unit} ago"
                return "just now"

            def _detail(row) -> str:
                keys = [("reviews", "reviews"), ("ok", "valid"), ("failed", "failed"), ("product_id", ""), ("n_reviews", "reviews"), ("strategy", "")]
                def fmt(v):
                    return int(v) if isinstance(v, float) and v.is_integer() else v

                parts = [f"{fmt(row[k])} {lbl}".strip() for k, lbl in keys if k in row and row[k] == row[k] and row[k] is not None]
                return " · ".join(parts[:3])

            rows = act.to_dict("records")
            md("".join(
                f'<div class="riq-activity"><span class="riq-dot"></span><span class="riq-activity-main">{esc(str(r.get("action", "")).replace("_", " ").capitalize())} '
                f'<span>{esc(_detail(r))}</span></span><span class="riq-activity-time">{esc(_ago(r.get("timestamp", "")))}</span></div>'
                for r in rows
            ))

# ---------------------------------------------------------------- dataset
section("Dataset & data quality", card.get("source", "Amazon Reviews 2023") if card else None)
if df is None:
    empty_state("Dataset not prepared", "Run <code>python scripts/prepare_dataset.py</code> (no API key needed).", "database")
else:
    a, b, c = st.columns([3, 3, 2], gap="medium")
    with a, panel("Star rating distribution", "Reviews per star rating"):
        rd = df["rating"].value_counts().sort_index()
        chart(distribution_bar({f"{int(k)}★": int(v) for k, v in rd.items()}, x_title=""), key="ov_ratings")
    with b, panel("Review length", f"Words per review · {n_products} products × up to {card.get('reviews_per_product_cap', '—')} reviews"):
        chart(histogram(df["review_text"].str.split().str.len().clip(upper=200), x_title="words (clipped at 200)"), key="ov_length")
    with c, panel("Quality"):
        dropped = card.get("dropped_for_selected_products", {})
        flt = card.get("filters", {})
        md(kv_html([
            ("Category", card.get("category", s.dataset_category)),
            ("Raw reviews", f"{card.get('total_raw_reviews_in_category', 0):,}"),
            ("Final reviews", f"{len(df):,}"),
            ("Mean rating", f"{df['rating'].mean():.2f}"),
            ("Median length", f"{int(df['review_text'].str.split().str.len().median())} words"),
            ("Length filter", f"{flt.get('min_words', '?')}–{flt.get('max_words', '?')} words"),
            ("Removed: length", f"{dropped.get('invalid_length_or_empty', 0):,}"),
            ("Removed: duplicates", f"{dropped.get('duplicate_user_product', 0) + dropped.get('duplicate_text', 0):,}"),
            ("Missing text", f"{int((df['review_text'].str.strip() == '').sum())}"),
            ("Seed", card.get("random_seed", s.random_seed)),
        ]))

# ---------------------------------------------------------------- deliverables
sample = load_sample(s.paths.annotation_sample_csv)
ann = AnnotationStore(s.paths.annotations_json)
prog = ann.progress(sample["review_id"].tolist()) if len(sample) else {"complete": 0, "total": s.annotation_sample_size}
runs = list_runs(s, mock)
n_sum = len(SummaryStore.for_mode(s, mock).all())
n_judge = len(JudgementStore.for_mode(s, mock).all())
d2_ready = prog["complete"] >= s.annotation_sample_size and runs and not mock
d3_ready = n_sum and n_judge and not mock
section("Project deliverables")
md(
    '<div class="riq-kpi-grid">'
    + kpi("D1 · Structured extraction", "Implemented", "Pydantic schema · parsers · JSON/CSV export · repair")
    + kpi("D2 · Prompt comparison", f"{prog['complete']} / {s.annotation_sample_size}", f"human labels · {len(runs)} {mode_word} run(s)")
    + kpi("D3 · Summaries + judge", f"{n_sum} / {n_judge}", f"{mode_word} summaries / judgements")
    + "</div>"
    + '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:-4px">'
    + badge("D1 implemented", "positive", dot=True)
    + badge("D2 ready for evaluation" if d2_ready else "D2 pending: 300 human labels + live run", "positive" if d2_ready else "warning", dot=True)
    + badge("D3 live results available" if d3_ready else "D3 pending live run", "positive" if d3_ready else "warning", dot=True)
    + "</div>"
)
