"""Page 4 — Product Intelligence: aggregate statistics and grounded product summaries."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.visualizations import aspect_sentiment_bar, aspect_trend_line, counts_bar, sentiment_pie
from pipeline.aggregation import aggregate_all
from pipeline.extraction import ExtractionStore, ReviewExtractor
from pipeline.summarization import SummaryStore, summarize_product
from prompts.strategies import STRATEGIES, get_strategy
from schemas.evaluation_schema import ProductAggregate, ProductSummary
from ui.common import dataset, extraction_files, get_client, get_settings, llm_ready, load_store, mock_banner, pretty, require_llm_note, use_mock
from ui.components import badge, callout, chart, empty_state, esc, kpi, kpi_row, md, page_header, panel, section
from utils.data_loader import dataframe_to_inputs
from utils.storage import log_activity

s = get_settings()
page_header("Product Intelligence", "Discover what customers think about individual products.", is_mock=use_mock())
mock_banner()

# ---------------------------------------------------------------- extraction source
with st.expander("Extract reviews for products (stage 1)", icon=":material/play_circle:", expanded=not extraction_files()):
    require_llm_note()
    df = dataset()
    if df is None:
        empty_state("Dataset not prepared", "Run <code>python scripts/prepare_dataset.py</code>.", "database")
    else:
        c1, c2 = st.columns(2)
        strategy = c1.selectbox("Strategy", list(STRATEGIES), index=1, format_func=lambda k: STRATEGIES[k].label)
        per = c2.number_input("Reviews per product", 3, 100, 20, help="Cost control: reviews extracted per product (cached, resumable).")
        titles = df.drop_duplicates("product_id").set_index("product_id")["product_title"].to_dict()
        prods = st.multiselect("Products", list(titles), format_func=lambda p: f"{titles[p][:60]} ({p})", placeholder="All products")
        sub = (df[df["product_id"].isin(prods)] if prods else df).groupby("product_id", group_keys=False).head(int(per))
        md(f'<div class="riq-panel-caption">{len(sub)} reviews → up to {len(sub)} LLM calls (cached results are skipped).</div>')
        if st.button("Run extraction", type="primary", icon=":material/play_arrow:", disabled=not llm_ready()):
            client = get_client()
            if client:
                store = ExtractionStore.for_run(s, client.is_mock, "dataset", strategy, client.model_name)
                bar = st.progress(0.0)
                res = ReviewExtractor(client, get_strategy(strategy), s).extract_many(dataframe_to_inputs(sub), store=store, progress=lambda i, n, _o: bar.progress(i / n, text=f"Extracted {i}/{n}"))
                log_activity(s.paths.activity_log, "product_extraction", client.is_mock, strategy=strategy, reviews=len(sub), ok=len(res.records), failed=len(res.failures))
                st.success(f"{len(res.records)} valid records, {len(res.failures)} controlled failures ({res.from_cache} from cache).")

files = extraction_files()
if not files:
    empty_state("No extraction results in this mode yet", "Run stage 1 above to extract structured records for one or more products.", "inventory_2")
    st.stop()

c1, c2 = st.columns([3, 2])
src = c2.selectbox("Extraction result set", files, format_func=lambda p: p.stem)
store = load_store(src)
records = store.records
if not records:
    callout("This result set has no valid records.", "warning")
    st.stop()
aggregates: dict[str, ProductAggregate] = aggregate_all(records)
pid = c1.selectbox("Product", list(aggregates), format_func=lambda p: f"{(aggregates[p].product_title or p)[:70]} · {aggregates[p].total_reviews} reviews")
agg = aggregates[pid]

# ---------------------------------------------------------------- product header & KPIs
dominant = max(agg.sentiment_counts, key=lambda k: agg.sentiment_counts[k])
md(
    f'<div style="margin:1.1rem 0 0.4rem"><div style="font-size:18px;font-weight:600;color:var(--riq-text)">{esc(agg.product_title or pid)}</div>'
    f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px">{badge(pid, "neutral")}{badge("Calculated statistics", "accent")}'
    f'<span class="riq-panel-caption" style="margin:0;align-self:center">Computed with pandas from {agg.total_reviews} structured records · no LLM involved</span></div></div>'
)
kpi_row([
    kpi("Total reviews", agg.total_reviews, f"average rating {agg.average_rating:.2f} ★" if agg.average_rating else "in this result set", icon_name="reviews"),
    kpi("Overall sentiment", dominant.capitalize(), f"{agg.sentiment_percentages[dominant]}% of reviews", tone=dominant if dominant in ("positive", "negative", "mixed") else None, icon_name="sentiment_satisfied"),
    kpi("Positive reviews", f"{agg.sentiment_percentages['positive']}%", f"{agg.sentiment_counts['positive']} reviews", tone="positive", icon_name="thumb_up"),
    kpi("Negative reviews", f"{agg.sentiment_percentages['negative']}%", f"{agg.sentiment_counts['negative']} reviews", tone="negative", icon_name="thumb_down"),
    kpi("With complaints", agg.reviews_with_complaints, f"{100 * agg.reviews_with_complaints / max(1, agg.total_reviews):.0f}% of reviews", icon_name="report"),
], min_width=170)

# ---------------------------------------------------------------- aspect performance
section("Aspect performance", "Only aspect categories that occur in this product's reviews")
aspect_df = pd.DataFrame([x.model_dump() for x in agg.aspect_stats])
a, b = st.columns([3, 2], gap="medium")
with a, panel("Sentiment by aspect", "Mentions split by sentiment"):
    chart(aspect_sentiment_bar([x.model_dump() for x in agg.aspect_stats]), key="pi_aspects")
with b, panel("Aspect statistics"):
    if aspect_df.empty:
        md('<div class="riq-panel-caption">No aspects extracted.</div>')
    else:
        view = aspect_df[["aspect_category", "mentions", "positive", "negative", "neutral", "net_sentiment"]].copy()
        view["aspect_category"] = view["aspect_category"].map(pretty)
        st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            column_config={
                "aspect_category": st.column_config.TextColumn("aspect"),
                "net_sentiment": st.column_config.ProgressColumn("net sentiment", min_value=-1.0, max_value=1.0, format="%.2f"),
            },
        )

# ---------------------------------------------------------------- complaints
section("Customer complaints", f"{agg.reviews_with_complaints} of {agg.total_reviews} reviews report at least one complaint")
a, b = st.columns([3, 2], gap="medium")
with a, panel("Most reported complaint categories", "Number of reviews per category"):
    chart(counts_bar(agg.complaint_counts), key="pi_complaints")
with b, panel("Sentiment distribution"):
    chart(sentiment_pie(agg.sentiment_counts), key="pi_pie")

with panel("Net aspect sentiment over time", "Top aspects by mentions · net = (positive − negative) / mentions"):
    chart(aspect_trend_line(agg.aspect_trend), key="pi_trend")

a, b = st.columns(2, gap="medium")
with a, panel("Positive evidence", "Verbatim quotes from reviews"):
    md("".join(f'<div class="riq-quote pos">“{esc(q.quote)}”<span class="riq-quote-meta">{esc(pretty(q.aspect_category))} · {esc(q.review_id)}</span></div>' for q in agg.positive_evidence) or '<div class="riq-panel-caption">None.</div>')
with b, panel("Negative evidence", "Verbatim quotes from reviews"):
    md("".join(f'<div class="riq-quote neg">“{esc(q.quote)}”<span class="riq-quote-meta">{esc(pretty(q.aspect_category))} · {esc(q.review_id)}</span></div>' for q in agg.negative_evidence) or '<div class="riq-panel-caption">None.</div>')

# ---------------------------------------------------------------- summary
section("Product summary", "Stage 3 · generated by an LLM from the statistics and evidence above", right_html=badge("AI-generated", "accent", dot=True))
sstore = SummaryStore.for_mode(s, use_mock())
existing = sstore.latest().get((pid, src.stem))


def _list(title: str, items: list[str]) -> str:
    body = "".join(f"<li>{esc(x)}</li>" for x in items) or "<li>none</li>"
    return f'<div class="riq-list-title">{esc(title)}</div><ul class="riq-list">{body}</ul>'


def render_summary(summary: ProductSummary) -> None:
    rec_by_id = {r.review_id: r for r in records}
    c = summary.content
    if summary.is_mock:
        callout("<b>MOCK template summary</b> — not generated by an LLM. Shown only to demonstrate the layout and chain.", "warning", tag="Mock")
    with panel():
        md(f'<div class="riq-list-title" style="margin-top:0">Overall customer opinion</div><div class="riq-summary-opinion">{esc(c.overall_opinion)}</div>')

    def points(title: str, items, tone: str) -> None:
        md(f'<div class="riq-list-title">{esc(title)}</div>')
        if not items:
            md('<div class="riq-panel-caption">none</div>')
        for p in items:
            with st.expander(f"{p.point}  ·  {len(p.supporting_review_ids)} cited review(s)"):
                for rid in p.supporting_review_ids:
                    r = rec_by_id.get(rid)
                    md(f'<div class="riq-quote {tone}">{esc(r.review_text if r else "(review not in this result set)")}<span class="riq-quote-meta">{esc(rid)}</span></div>')

    a, b = st.columns(2, gap="medium")
    with a:
        points("What customers appreciate", c.appreciated, "pos")
    with b:
        points("Common complaints", c.complaints, "neg")
    with panel():
        a, b, d = st.columns(3, gap="medium")
        a.markdown(_list("Frequently mentioned aspects", c.frequent_aspects), unsafe_allow_html=True)
        b.markdown(_list("Improvement opportunities", c.improvement_areas), unsafe_allow_html=True)
        d.markdown(_list("Recommendations", c.recommendations), unsafe_allow_html=True)
    md(f'<div class="riq-panel-caption">{esc(summary.provider)}/{esc(summary.model_name)} · prompt {esc(summary.prompt_version)} · repairs {summary.repair_attempts} · {esc(summary.created_at)} · id {esc(summary.summary_id)}</div>')


if existing:
    render_summary(existing)
else:
    empty_state("No summary for this product yet", "Generate one below. It is grounded in the statistics and the cited reviews above.", "summarize")

c1, c2, _ = st.columns([2, 2, 3])
if c1.button("Generate summary for this product", type="primary", icon=":material/auto_awesome:", disabled=not llm_ready(), width="stretch"):
    client = get_client()
    if client:
        with st.spinner("Summarising…"):
            summary, res = summarize_product(client, agg, s, src.stem)
        if summary:
            sstore.add(summary)
            log_activity(s.paths.activity_log, "product_summary", client.is_mock, product_id=pid)
            st.rerun()
        else:
            st.error(f"Summary failed after repair attempts: {res.error}")
if c2.button("Generate summaries for all products", icon=":material/library_add:", disabled=not llm_ready(), width="stretch"):
    client = get_client()
    if client:
        bar = st.progress(0.0)
        errors = []
        for i, (p, a_) in enumerate(aggregates.items(), 1):
            summary, res = summarize_product(client, a_, s, src.stem)
            if summary:
                sstore.add(summary)
            else:
                errors.append(f"{p}: {res.error}")
            bar.progress(i / len(aggregates), text=f"{i}/{len(aggregates)}")
        log_activity(s.paths.activity_log, "product_summaries_batch", client.is_mock, products=len(aggregates), errors=len(errors))
        for e in errors:
            st.error(e)
        st.rerun()
