"""Page 4 — Product Insights: aggregate statistics and grounded product summaries."""

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
from utils.data_loader import dataframe_to_inputs
from utils.storage import log_activity

s = get_settings()
st.title("📦 Product Insights")
st.caption("Stage 2 — AGGREGATE (computed with pandas) → Stage 3 — SUMMARISE (LLM, grounded in statistics and evidence).")
mock_banner()

# ---------------------------------------------------------------- extraction source
with st.expander("▶ Extract reviews for the products (Stage 1)", expanded=not extraction_files()):
    require_llm_note()
    df = dataset()
    if df is None:
        st.error("Dataset not prepared. Run `python scripts/prepare_dataset.py`.")
    else:
        c1, c2 = st.columns(2)
        strategy = c1.selectbox("Strategy", list(STRATEGIES), index=1, format_func=lambda k: STRATEGIES[k].label)
        per = c2.number_input("Reviews per product", 3, 100, 20, help="Cost control: reviews extracted per product (cached, resumable).")
        titles = df.drop_duplicates("product_id").set_index("product_id")["product_title"].to_dict()
        prods = st.multiselect("Products (empty = all)", list(titles), format_func=lambda p: f"{titles[p][:60]} ({p})")
        sub = (df[df["product_id"].isin(prods)] if prods else df).groupby("product_id", group_keys=False).head(int(per))
        st.caption(f"{len(sub)} reviews → ~{len(sub)} LLM calls (cached results are skipped).")
        if st.button("Run extraction", type="primary", disabled=not llm_ready()):
            client = get_client()
            if client:
                store = ExtractionStore.for_run(s, client.is_mock, "dataset", strategy, client.model_name)
                bar = st.progress(0.0)
                res = ReviewExtractor(client, get_strategy(strategy), s).extract_many(dataframe_to_inputs(sub), store=store, progress=lambda i, n, _o: bar.progress(i / n, text=f"Extracted {i}/{n}"))
                log_activity(s.paths.activity_log, "product_extraction", client.is_mock, strategy=strategy, reviews=len(sub), ok=len(res.records), failed=len(res.failures))
                st.success(f"{len(res.records)} valid records, {len(res.failures)} controlled failures ({res.from_cache} from cache).")

files = extraction_files()
if not files:
    st.info("No extraction results in this mode yet. Run Stage 1 above.")
    st.stop()

src = st.selectbox("Extraction result set", files, format_func=lambda p: p.stem)
store = load_store(src)
records = store.records
if not records:
    st.warning("This result set has no valid records.")
    st.stop()
aggregates: dict[str, ProductAggregate] = aggregate_all(records)
pid = st.selectbox("Product", list(aggregates), format_func=lambda p: f"{(aggregates[p].product_title or p)[:70]} ({p}) — {aggregates[p].total_reviews} reviews")
agg = aggregates[pid]

# ---------------------------------------------------------------- statistics
st.subheader(agg.product_title or pid)
k = st.columns(5)
k[0].metric("Reviews analysed", agg.total_reviews)
k[1].metric("Average rating", f"{agg.average_rating:.2f} ★" if agg.average_rating else "—")
k[2].metric("Positive", f"{agg.sentiment_percentages['positive']}%")
k[3].metric("Negative", f"{agg.sentiment_percentages['negative']}%")
k[4].metric("With complaints", agg.reviews_with_complaints)

a, b = st.columns(2)
a.plotly_chart(sentiment_pie(agg.sentiment_counts), width="stretch")
b.plotly_chart(aspect_sentiment_bar([x.model_dump() for x in agg.aspect_stats], "Positive vs negative aspects"), width="stretch")
a, b = st.columns(2)
a.plotly_chart(counts_bar(agg.complaint_counts, "Common complaints"), width="stretch")
b.plotly_chart(aspect_trend_line(agg.aspect_trend), width="stretch")

with st.expander("Aspect statistics table"):
    st.dataframe(pd.DataFrame([x.model_dump() for x in agg.aspect_stats]), width="stretch", hide_index=True)
pos, neg = st.columns(2)
with pos:
    st.markdown("**👍 Positive evidence**")
    for q in agg.positive_evidence:
        st.markdown(f"- *“{q.quote}”* — `{pretty(q.aspect_category)}` · {q.review_id}")
with neg:
    st.markdown("**👎 Negative evidence**")
    for q in agg.negative_evidence:
        st.markdown(f"- *“{q.quote}”* — `{pretty(q.aspect_category)}` · {q.review_id}")

# ---------------------------------------------------------------- summary
st.divider()
st.subheader("Product summary (Stage 3)")
sstore = SummaryStore.for_mode(s, use_mock())
existing = sstore.latest().get((pid, src.stem))


def render_summary(summary: ProductSummary) -> None:
    rec_by_id = {r.review_id: r for r in records}
    c = summary.content
    if summary.is_mock:
        st.warning("MOCK template summary — not generated by an LLM.")
    st.markdown(f"**Overall customer opinion.** {c.overall_opinion}")

    def points(title: str, items) -> None:
        st.markdown(f"**{title}**")
        if not items:
            st.caption("none")
        for p in items:
            with st.expander(f"{p.point}  ·  cites {', '.join(p.supporting_review_ids)}"):
                for rid in p.supporting_review_ids:
                    r = rec_by_id.get(rid)
                    st.markdown(f"`{rid}` — {r.review_text if r else '(review not in this result set)'}")

    a, b = st.columns(2)
    with a:
        points("What customers appreciate", c.appreciated)
    with b:
        points("Common complaints", c.complaints)
    a, b, d = st.columns(3)
    a.markdown("**Frequently mentioned aspects**\n" + "\n".join(f"- {x}" for x in c.frequent_aspects))
    b.markdown("**Areas requiring improvement**\n" + "\n".join(f"- {x}" for x in c.improvement_areas))
    d.markdown("**Actionable recommendations**\n" + "\n".join(f"- {x}" for x in c.recommendations))
    st.caption(f"{summary.provider}/{summary.model_name} · prompt {summary.prompt_version} · repairs {summary.repair_attempts} · {summary.created_at} · id {summary.summary_id}")


if existing:
    render_summary(existing)
else:
    st.caption("No summary yet for this product and result set.")

c1, c2 = st.columns(2)
if c1.button("✨ Generate summary for this product", type="primary", disabled=not llm_ready()):
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
if c2.button("Generate summaries for all products in this result set", disabled=not llm_ready()):
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
