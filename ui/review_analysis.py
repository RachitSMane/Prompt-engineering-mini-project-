"""Page 2 — Review Analysis: upload/select reviews, run extraction, inspect & download."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation.visualizations import aspect_sentiment_bar, counts_bar, sentiment_pie
from pipeline.aggregation import overview_stats
from pipeline.extraction import ExtractionStore, ReviewExtractor
from prompts.strategies import STRATEGIES, get_strategy
from schemas.review_schema import ReviewRecord
from ui.common import dataset, get_client, get_settings, llm_ready, mock_banner, require_llm_note, use_mock
from utils.data_loader import dataframe_to_inputs, normalize_uploaded_reviews
from utils.export import aspects_to_dataframe, dataframe_to_csv_bytes, failures_to_dataframe, records_to_dataframe, records_to_json
from utils.storage import log_activity, slug

UI_MAX_REVIEWS = 200
s = get_settings()

st.title("🔎 Review Analysis")
st.caption("Stage 1 — EXTRACT: unstructured review → validated structured JSON (with automatic repair).")
mock_banner()
require_llm_note()

# ---------------------------------------------------------------- input
source_kind = st.radio("Review source", ["Prepared dataset (Amazon Appliances)", "Upload CSV"], horizontal=True)
reviews_df: pd.DataFrame | None = None
source_name = "dataset"

if source_kind.startswith("Upload"):
    up = st.file_uploader("CSV file with review text and product identifier columns", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up, keep_default_na=False)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            raw = None
        if raw is not None and not raw.empty:
            cols = list(raw.columns)

            def guess(*names: str) -> int:
                for i, c in enumerate(cols):
                    if c.lower() in names:
                        return i
                return 0

            c1, c2, c3, c4 = st.columns(4)
            text_col = c1.selectbox("Review text column", cols, index=guess("review_text", "text", "review", "content"))
            prod_col = c2.selectbox("Product ID column", cols, index=guess("product_id", "asin", "parent_asin", "product"))
            id_col = c3.selectbox("Review ID column (optional)", ["(generate)"] + cols, index=(guess("review_id", "id") + 1) if any(c.lower() in ("review_id", "id") for c in cols) else 0)
            title_col = c4.selectbox("Product title column (optional)", ["(none)"] + cols)
            try:
                reviews_df = normalize_uploaded_reviews(raw, text_col, prod_col, None if id_col == "(generate)" else id_col, None if title_col == "(none)" else title_col)
                st.caption(f"{len(reviews_df)} valid reviews · {reviews_df.attrs.get('dropped', 0)} empty/duplicate rows removed")
                source_name = f"upload-{slug(Path(up.name).stem)}"
            except ValueError as exc:
                st.error(str(exc))
else:
    df = dataset()
    if df is None:
        st.error("Dataset not prepared. Run `python scripts/prepare_dataset.py`.")
    else:
        titles = df.drop_duplicates("product_id").set_index("product_id")["product_title"].to_dict()
        chosen = st.multiselect("Filter products (optional)", list(titles), format_func=lambda p: f"{titles[p][:60]} ({p})")
        reviews_df = df[df["product_id"].isin(chosen)] if chosen else df

if reviews_df is None or reviews_df.empty:
    st.stop()

with st.expander(f"Raw reviews ({len(reviews_df)})", expanded=False):
    st.dataframe(reviews_df[[c for c in ["review_id", "product_id", "product_title", "rating", "review_text"] if c in reviews_df.columns]], width="stretch", hide_index=True)

# ---------------------------------------------------------------- run
c1, c2, c3 = st.columns([2, 2, 1])
strategy_name = c1.selectbox("Prompting strategy", list(STRATEGIES), format_func=lambda k: STRATEGIES[k].label, index=1)
n = c2.number_input("Number of reviews to analyse", 1, min(UI_MAX_REVIEWS, len(reviews_df)), min(10, len(reviews_df)),
                    help=f"Capped at {UI_MAX_REVIEWS} in the UI; use scripts/run_chain.py for larger batches. Results are cached, so re-running is free.")
c3.write("")
c3.write("")
run = c3.button("▶ Run extraction", type="primary", disabled=not llm_ready(), width="stretch")
st.caption(STRATEGIES[strategy_name].description)

subset = reviews_df.head(int(n))
inputs = dataframe_to_inputs(subset)
state_key = f"extraction::{source_name}::{strategy_name}::{use_mock()}"

if run:
    client = get_client()
    if client:
        store = ExtractionStore.for_run(s, client.is_mock, source_name, strategy_name, client.model_name)
        extractor = ReviewExtractor(client, get_strategy(strategy_name), s)
        bar = st.progress(0.0, text="Starting…")
        try:
            result = extractor.extract_many(inputs, store=store, progress=lambda i, total, _o: bar.progress(i / total, text=f"Extracted {i}/{total}"))
        except Exception as exc:  # unexpected errors are shown, never swallowed
            st.error(f"Extraction stopped: {exc}")
            st.stop()
        bar.empty()
        st.session_state[state_key] = {"records": result.records, "failures": result.failures, "cached": result.from_cache, "new": result.new_calls}
        log_activity(s.paths.activity_log, "review_extraction", client.is_mock, source=source_name, strategy=strategy_name, reviews=len(inputs), ok=len(result.records), failed=len(result.failures))

res = st.session_state.get(state_key)
if not res:
    st.info("Choose a strategy and click **Run extraction**.")
    st.stop()

records: list[ReviewRecord] = res["records"]
failures = res["failures"]
st.divider()
st.subheader("Results")
k = st.columns(5)
k[0].metric("Valid records", len(records))
k[1].metric("Controlled failures", len(failures))
k[2].metric("From cache", res["cached"])
k[3].metric("Needed LLM repair", sum(r.extraction_metadata.repair_attempts > 0 for r in records))
k[4].metric("Local fixes", sum(r.extraction_metadata.local_repairs > 0 for r in records))

if records:
    table = records_to_dataframe(records)
    st.dataframe(table[["review_id", "product_id", "overall_sentiment", "positive_aspects", "negative_aspects", "complaint_categories", "confidence", "repair_attempts", "latency_s"]], width="stretch", hide_index=True)
    ov = overview_stats(records)
    a, b = st.columns(2)
    a.plotly_chart(sentiment_pie(ov["sentiment_counts"]), width="stretch")
    stats = [{"aspect_category": c, "mentions": sum(v.values()), "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0)} for c, v in ov["aspect_counts"].items()]
    b.plotly_chart(aspect_sentiment_bar(stats), width="stretch")
    st.plotly_chart(counts_bar(ov["complaint_counts"], "Complaint categories"), width="stretch")

    st.subheader("Inspect a record")
    by_id = {r.review_id: r for r in records}
    rid = st.selectbox("Review", list(by_id), format_func=lambda i: f"{i} — {by_id[i].review_text[:80]}")
    rec = by_id[rid]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"> {rec.review_text}")
        st.markdown(f"**Overall:** `{rec.overall_sentiment.value}` · **confidence** {rec.confidence:.2f}")
        if rec.aspects:
            st.dataframe(pd.DataFrame([a.model_dump(mode="json") for a in rec.aspects]), width="stretch", hide_index=True)
        else:
            st.caption("No identifiable product aspects.")
        st.markdown("**Complaints:** " + (", ".join(f"`{c.value}`" for c in rec.complaint_categories) or "none"))
    with right:
        st.json(rec.model_dump(mode="json", exclude={"extraction_metadata": {"attempts"}}), expanded=False)
    with st.expander(f"Repair log ({len(rec.extraction_metadata.attempts)} attempt(s))"):
        for att in rec.extraction_metadata.attempts:
            st.markdown(f"**Attempt {att.attempt} — {att.kind}** · {'✅ valid' if att.success else '❌ invalid'}")
            if att.local_fixes:
                st.caption("Local fixes: " + "; ".join(att.local_fixes))
            if att.errors:
                st.code("\n".join(att.errors), language="text")
            st.code(att.raw_output[:3000], language="json")

if failures:
    st.subheader("Controlled failures")
    st.caption("No valid output after all repair attempts. These are reported, never silently accepted.")
    st.dataframe(failures_to_dataframe(failures), width="stretch", hide_index=True)

st.subheader("Download")
d = st.columns(4)
stem = f"{source_name}_{strategy_name}{'_MOCK' if use_mock() else ''}"
d[0].download_button("JSON (records)", records_to_json(records), f"{stem}.json", "application/json", width="stretch")
d[1].download_button("CSV (records)", dataframe_to_csv_bytes(records_to_dataframe(records)), f"{stem}.csv", "text/csv", width="stretch")
d[2].download_button("CSV (aspects)", dataframe_to_csv_bytes(aspects_to_dataframe(records)), f"{stem}_aspects.csv", "text/csv", width="stretch")
d[3].download_button("CSV (failures)", dataframe_to_csv_bytes(failures_to_dataframe(failures)), f"{stem}_failures.csv", "text/csv", width="stretch")
