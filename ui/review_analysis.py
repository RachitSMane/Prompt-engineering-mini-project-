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
from ui.components import badge, chart, chips, empty_state, esc, kpi, kpi_row, md, page_header, panel, section, sentiment_badge
from utils.data_loader import dataframe_to_inputs, normalize_uploaded_reviews
from utils.export import aspects_to_dataframe, dataframe_to_csv_bytes, failures_to_dataframe, records_to_dataframe, records_to_json
from utils.storage import log_activity, slug

UI_MAX_REVIEWS = 200
s = get_settings()

page_header("Review Analysis", "Inspect customer feedback and extract structured product intelligence.", is_mock=use_mock())
mock_banner()
require_llm_note()

# ---------------------------------------------------------------- input
section("1 · Source", "Use the prepared Amazon Appliances dataset or upload your own CSV")
source_kind = st.radio("Review source", ["Prepared dataset (Amazon Appliances)", "Upload CSV"], horizontal=True, label_visibility="collapsed")
reviews_df: pd.DataFrame | None = None
source_name = "dataset"

if source_kind.startswith("Upload"):
    up = st.file_uploader(
        "Upload reviews CSV",
        type=["csv"],
        help="Required: a review text column and a product identifier column. Optional: review ID and product title. "
        "Empty and duplicate reviews are removed automatically.",
    )
    md('<div class="riq-panel-caption">CSV with a header row · required columns: review text, product ID · optional: review ID, product title, rating.</div>')
    if up is None:
        empty_state("No file uploaded", "Drop a CSV above. You map its columns in the next step, so any column names work.", "upload_file")
    else:
        try:
            raw = pd.read_csv(up, keep_default_na=False)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            raw = None
        if raw is not None and raw.empty:
            st.error("The uploaded CSV has no rows.")
        elif raw is not None:
            cols = list(raw.columns)

            def guess(*names: str) -> int:
                for i, c in enumerate(cols):
                    if c.lower() in names:
                        return i
                return 0

            with panel("Column mapping", "Tell ReviewIQ which columns hold the review data"):
                c1, c2, c3, c4 = st.columns(4)
                text_col = c1.selectbox("Review text column", cols, index=guess("review_text", "text", "review", "content"))
                prod_col = c2.selectbox("Product ID column", cols, index=guess("product_id", "asin", "parent_asin", "product"))
                id_col = c3.selectbox("Review ID column (optional)", ["(generate)"] + cols, index=(guess("review_id", "id") + 1) if any(c.lower() in ("review_id", "id") for c in cols) else 0)
                title_col = c4.selectbox("Product title column (optional)", ["(none)"] + cols)
            invalid: int | str = "—"
            try:
                reviews_df = normalize_uploaded_reviews(raw, text_col, prod_col, None if id_col == "(generate)" else id_col, None if title_col == "(none)" else title_col)
                invalid = int(reviews_df.attrs.get("dropped", 0))
                source_name = f"upload-{slug(Path(up.name).stem)}"
            except ValueError as exc:
                st.error(str(exc))
            processed_key = f"extraction::{source_name}::"
            done = [k for k in st.session_state if str(k).startswith(processed_key)]
            kpi_row([
                kpi("Rows uploaded", f"{len(raw):,}", up.name, icon_name="upload_file"),
                kpi("Valid records", f"{len(reviews_df):,}" if reviews_df is not None else "—", f"{len(reviews_df) if reviews_df is not None else 0} valid reviews ready", tone="positive", icon_name="check_circle"),
                kpi("Invalid records", f"{invalid}", "empty or duplicate rows removed", tone="negative" if invalid not in (0, "—") else None, icon_name="block"),
                kpi("Processing status", "Processed" if done else "Not processed", "results available below" if done else "run extraction below", icon_name="pending_actions"),
            ])
else:
    df = dataset()
    if df is None:
        empty_state("Dataset not prepared", "Run <code>python scripts/prepare_dataset.py</code> (no API key needed).", "database")
    else:
        titles = df.drop_duplicates("product_id").set_index("product_id")["product_title"].to_dict()
        chosen = st.multiselect("Filter products (optional)", list(titles), format_func=lambda p: f"{titles[p][:60]} ({p})", placeholder="All 15 products" if len(titles) == 15 else "All products")
        reviews_df = df[df["product_id"].isin(chosen)] if chosen else df

if reviews_df is None or reviews_df.empty:
    st.stop()

with st.expander(f"Preview raw reviews ({len(reviews_df):,})", icon=":material/table_view:"):
    st.dataframe(reviews_df[[c for c in ["review_id", "product_id", "product_title", "rating", "review_text"] if c in reviews_df.columns]], width="stretch", hide_index=True)

# ---------------------------------------------------------------- run
section("2 · Extract", "Convert reviews into validated structured records (cached, with automatic repair)")
with panel():
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    strategy_name = c1.selectbox("Prompting strategy", list(STRATEGIES), format_func=lambda k: STRATEGIES[k].label, index=1)
    n = c2.number_input("Number of reviews to analyse", 1, min(UI_MAX_REVIEWS, len(reviews_df)), min(10, len(reviews_df)),
                        help=f"Capped at {UI_MAX_REVIEWS} in the UI; use scripts/run_chain.py for larger batches. Results are cached, so re-running is free.")
    run = c3.button("Run extraction", type="primary", icon=":material/play_arrow:", disabled=not llm_ready(), width="stretch")
    md(f'<div class="riq-panel-caption">{esc(STRATEGIES[strategy_name].description)}</div>')

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
    empty_state("No results yet", "Choose a prompting strategy and click <b>Run extraction</b>. Structured records, filters and exports appear here.", "manage_search")
    st.stop()

records: list[ReviewRecord] = res["records"]
failures = res["failures"]

# ---------------------------------------------------------------- results
section("3 · Results", f"{STRATEGIES[strategy_name].label} · {len(inputs)} reviews requested")
kpi_row([
    kpi("Valid records", len(records), "passed schema validation", tone="positive"),
    kpi("Controlled failures", len(failures), "no valid output after repair", tone="negative" if failures else None),
    kpi("From cache", res["cached"], "re-used, no new LLM call"),
    kpi("Needed LLM repair", sum(r.extraction_metadata.repair_attempts > 0 for r in records), "fixed by a repair prompt"),
    kpi("Local fixes", sum(r.extraction_metadata.local_repairs > 0 for r in records), "fixed without an LLM call"),
], min_width=160)

if records:
    by_id = {r.review_id: r for r in records}

    f1, f2, f3, f4 = st.columns(4)
    products = sorted({r.product_id for r in records})
    f_prod = f1.multiselect("Product", products, placeholder="All products")
    f_sent = f2.multiselect("Sentiment", ["positive", "negative", "neutral", "mixed"], placeholder="All sentiments")
    f_asp = f3.multiselect("Aspect", sorted({a.aspect_category.value for r in records for a in r.aspects}), format_func=lambda x: x.replace("_", " "), placeholder="Any aspect")
    f_comp = f4.multiselect("Complaint category", sorted({c.value for r in records for c in r.complaint_categories}), format_func=lambda x: x.replace("_", " "), placeholder="Any complaint")

    def keep(r: ReviewRecord) -> bool:
        return (
            (not f_prod or r.product_id in f_prod)
            and (not f_sent or r.overall_sentiment.value in f_sent)
            and (not f_asp or any(a.aspect_category.value in f_asp for a in r.aspects))
            and (not f_comp or any(c.value in f_comp for c in r.complaint_categories))
        )

    shown = [r for r in records if keep(r)]
    view = pd.DataFrame([
        {
            "Review ID": r.review_id,
            "Product": r.product_title or r.product_id,
            "Review": r.review_text,
            "Sentiment": r.overall_sentiment.value,
            "Aspects": len(r.aspects),
            "Complaints": ", ".join(c.value.replace("_", " ") for c in r.complaint_categories) or "—",
            "Confidence": r.confidence,
        }
        for r in shown
    ])
    md(f'<div class="riq-panel-caption">{len(shown)} of {len(records)} records · select a row to inspect it</div>')
    if view.empty:
        empty_state("No records match the filters", "Clear one or more filters to see records.", "filter_alt_off")
        selected = None
    else:
        event = st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            **({"height": 420} if len(view) > 11 else {}),
            on_select="rerun",
            selection_mode="single-row",
            key=f"ra_table::{state_key}",
            column_config={
                "Review ID": st.column_config.TextColumn(width=120),
                "Product": st.column_config.TextColumn(width=170),
                "Review": st.column_config.TextColumn(width="large"),
                "Sentiment": st.column_config.TextColumn(width=90),
                "Aspects": st.column_config.NumberColumn(width=70, format="%d"),
                "Complaints": st.column_config.TextColumn(width=190),
                "Confidence": st.column_config.ProgressColumn(width="small", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        sel = getattr(getattr(event, "selection", None), "rows", None) or []
        selected = shown[sel[0]] if sel and sel[0] < len(shown) else shown[0]

    # ------------------------------------------------------------ record detail
    if selected is not None:
        rec = selected
        section("Record detail", f"{rec.review_id} · {rec.product_title or rec.product_id}")
        left, right = st.columns([3, 2], gap="medium")
        with left:
            md(
                f'<div class="riq-review-meta"><span><b>Review</b> {esc(rec.review_id)}</span><span><b>Product</b> {esc(rec.product_id)}</span>'
                f'<span><b>Confidence</b> {rec.confidence:.2f}</span></div><div class="riq-review">{esc(rec.review_text)}</div>'
            )
            md(
                f'<div style="margin:12px 0 4px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
                f'<span class="riq-list-title" style="margin:0">Overall</span>{sentiment_badge(rec.overall_sentiment.value)}'
                f'<span class="riq-list-title" style="margin:0 0 0 10px">Complaints</span>{chips(c.value for c in rec.complaint_categories)}</div>'
            )
            if rec.aspects:
                st.dataframe(
                    pd.DataFrame([
                        {"Aspect": a.aspect_name, "Category": a.aspect_category.value.replace("_", " "), "Sentiment": a.sentiment.value, "Evidence": a.evidence, "Confidence": a.confidence}
                        for a in rec.aspects
                    ]),
                    width="stretch",
                    hide_index=True,
                    column_config={"Evidence": st.column_config.TextColumn(width="large"), "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")},
                )
            else:
                md('<div class="riq-panel-caption">No identifiable product aspects.</div>')
        with right:
            t_json, t_log = st.tabs(["Validated JSON", f"Repair log ({len(rec.extraction_metadata.attempts)})"])
            with t_json:
                st.json(rec.model_dump(mode="json", exclude={"extraction_metadata": {"attempts"}}), expanded=1)
            with t_log:
                for att in rec.extraction_metadata.attempts:
                    md(f'<div style="display:flex;gap:8px;align-items:center;margin:6px 0"><b style="font-size:13px">Attempt {att.attempt} · {esc(att.kind)}</b>'
                       f'{badge("valid", "positive") if att.success else badge("invalid", "negative")}</div>')
                    if att.local_fixes:
                        st.caption("Local fixes: " + "; ".join(att.local_fixes))
                    if att.errors:
                        st.code("\n".join(att.errors), language="text")
                    st.code(att.raw_output[:3000], language="json")

    # ------------------------------------------------------------ analytics
    section("Result analytics", "Computed from the valid records above")
    ov = overview_stats(records)
    stats = [{"aspect_category": c, "mentions": sum(v.values()), "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0)} for c, v in ov["aspect_counts"].items()]
    a, b, c = st.columns([2, 3, 3], gap="medium")
    with a, panel("Overall sentiment"):
        chart(sentiment_pie(ov["sentiment_counts"]), key="ra_pie")
    with b, panel("Aspect sentiment"):
        chart(aspect_sentiment_bar(stats), key="ra_aspects")
    with c, panel("Complaint categories"):
        chart(counts_bar(ov["complaint_counts"]), key="ra_complaints")

if failures:
    section("Controlled failures", "No valid output after all repair attempts. These are reported, never silently accepted.")
    st.dataframe(failures_to_dataframe(failures), width="stretch", hide_index=True)

section("Export", "Download validated records and failures")
d = st.columns(4)
stem = f"{source_name}_{strategy_name}{'_MOCK' if use_mock() else ''}"
d[0].download_button("JSON (records)", records_to_json(records), f"{stem}.json", "application/json", icon=":material/data_object:", width="stretch")
d[1].download_button("CSV (records)", dataframe_to_csv_bytes(records_to_dataframe(records)), f"{stem}.csv", "text/csv", icon=":material/download:", width="stretch")
d[2].download_button("CSV (aspects)", dataframe_to_csv_bytes(aspects_to_dataframe(records)), f"{stem}_aspects.csv", "text/csv", icon=":material/download:", width="stretch")
d[3].download_button("CSV (failures)", dataframe_to_csv_bytes(failures_to_dataframe(failures)), f"{stem}_failures.csv", "text/csv", icon=":material/download:", width="stretch")
