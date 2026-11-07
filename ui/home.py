"""Page 1 — Home / Dashboard."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from annotation.annotation_manager import AnnotationStore, load_sample
from evaluation.prompt_comparison import list_runs
from evaluation.summary_evaluation import JudgementStore
from evaluation.visualizations import aspect_sentiment_bar, counts_bar, sentiment_pie
from pipeline.aggregation import overview_stats
from pipeline.summarization import SummaryStore
from ui.common import activity, dataset, dataset_card, extraction_files, get_settings, load_store, mock_banner, use_mock

s = get_settings()
st.title("📊 Aspect-Based Customer Review Analytics")
st.caption("Structured LLM outputs · prompt-technique comparison · grounded product summaries")
mock_banner()

st.markdown(
    """
An LLM pipeline that turns unstructured **Amazon product reviews** into **validated structured records**
(overall sentiment, aspect-level sentiment with evidence, complaint categories). It then compares
**zero-shot, few-shot and Chain-of-Thought** prompts against human labels, and produces **per-product summaries**
through an *extract → aggregate → summarise* chain that an **LLM-as-a-Judge** evaluates.
"""
)

# ---------------------------------------------------------------- deliverable status
sample = load_sample(s.paths.annotation_sample_csv)
ann = AnnotationStore(s.paths.annotations_json)
prog = ann.progress(sample["review_id"].tolist()) if len(sample) else {"complete": 0, "total": s.annotation_sample_size}
runs = list_runs(s, use_mock())
summaries = SummaryStore.for_mode(s, use_mock()).all()
judgements = JudgementStore.for_mode(s, use_mock()).all()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### D1 · Structured extraction")
    st.write("Pydantic schema · LangChain parsers · JSON/CSV export · automatic repair")
    st.success("Implemented")
with c2:
    st.markdown("#### D2 · Prompt comparison")
    st.write(f"Human labels: **{prog['complete']} / {s.annotation_sample_size}** · runs ({'mock' if use_mock() else 'live'}): **{len(runs)}**")
    if prog["complete"] >= s.annotation_sample_size and runs and not use_mock():
        st.success("Ready for full evaluation")
    else:
        st.warning("Pending: needs 300 human annotations + a live run")
with c3:
    st.markdown("#### D3 · Summaries + judge")
    st.write(f"Summaries: **{len(summaries)}** · judgements: **{len(judgements)}**")
    (st.success if summaries and judgements and not use_mock() else st.warning)("Live results available" if summaries and judgements and not use_mock() else "Pending live run")

st.divider()

# ---------------------------------------------------------------- dataset
st.subheader("Dataset")
df = dataset()
card = dataset_card()
if df is None:
    st.error("Dataset not prepared yet. Run `python scripts/prepare_dataset.py` (no API key needed).")
else:
    k = st.columns(4)
    k[0].metric("Reviews", f"{len(df):,}")
    k[1].metric("Products", df["product_id"].nunique())
    k[2].metric("Mean rating", f"{df['rating'].mean():.2f} ★")
    k[3].metric("Median length", f"{int(df['review_text'].str.split().str.len().median())} words")
    st.caption(f"Source: {card.get('source', 'Amazon Reviews 2023')} — category **{card.get('category', s.dataset_category)}**, "
               f"sampled from {card.get('total_raw_reviews_in_category', 0):,} raw reviews (seed {card.get('random_seed', s.random_seed)}).")
    a, b = st.columns(2)
    rd = df["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="reviews")
    a.plotly_chart(px.bar(rd, x="rating", y="reviews", title="Star rating distribution").update_layout(height=300, margin=dict(t=40, b=10)), width="stretch")
    per = df.groupby("product_title").size().sort_values().rename("reviews").reset_index()
    per["product_title"] = per["product_title"].str.slice(0, 45)
    b.plotly_chart(px.bar(per, x="reviews", y="product_title", orientation="h", title="Reviews per product").update_layout(height=300, margin=dict(t=40, b=10), yaxis_title=""), width="stretch")

st.divider()

# ---------------------------------------------------------------- processed
st.subheader(f"Processed reviews ({'mock' if use_mock() else 'live'} results)")
records = {}
for f in extraction_files():
    for r in load_store(f).records:
        records.setdefault(r.review_id, r)
if not records:
    st.info("No reviews processed yet in this mode. Use **Review Analysis** or **Product Insights** to run extraction.")
else:
    ov = overview_stats(list(records.values()))
    k = st.columns(3)
    k[0].metric("Unique reviews processed", ov["total"])
    k[1].metric("Products covered", ov["products"])
    k[2].metric("Reviews with complaints", sum(bool(r.complaint_categories) for r in records.values()))
    a, b = st.columns(2)
    a.plotly_chart(sentiment_pie(ov["sentiment_counts"]), width="stretch")
    stats = [{"aspect_category": cat, "mentions": sum(v.values()), "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0)} for cat, v in ov["aspect_counts"].items()]
    b.plotly_chart(aspect_sentiment_bar(stats, "Most discussed aspects"), width="stretch")
    st.plotly_chart(counts_bar(ov["complaint_counts"], "Complaint categories"), width="stretch")

st.subheader("Recent processing activity")
act = activity()
if act.empty:
    st.caption("No activity recorded yet.")
else:
    st.dataframe(act, width="stretch", hide_index=True)
