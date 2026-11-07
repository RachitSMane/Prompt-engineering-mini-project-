"""Page 6 — Data Annotation: manual labelling of the 300-review evaluation set."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from annotation.annotation_manager import AnnotationStore, build_annotation_sample, load_sample, save_sample, write_template
from config import PROJECT_ROOT
from schemas.evaluation_schema import AnnotatedAspect, Annotation
from schemas.taxonomy import (
    ASPECT_CATEGORY_DEFINITIONS,
    COMPLAINT_CATEGORY_DEFINITIONS,
    AspectCategory,
    ComplaintCategory,
    OverallSentiment,
    values,
)
from ui.common import dataset, get_settings, pretty
from utils.export import dataframe_to_csv_bytes

s = get_settings()
st.title("✍️ Data Annotation")
st.caption("Human ground truth for Deliverable 2. Labels are saved after every review to data/annotations/annotations.json.")
st.info("Label each review **yourself**, from the text. Never copy model predictions here: these labels are the ground truth the prompts are evaluated against.")

with st.expander("📖 Annotation guide"):
    st.markdown((PROJECT_ROOT / "annotation" / "annotation_guide.md").read_text(encoding="utf-8"))

sample = load_sample(s.paths.annotation_sample_csv)
if sample.empty:
    df = dataset()
    if df is None:
        st.error("Prepare the dataset first: `python scripts/prepare_dataset.py`.")
        st.stop()
    st.warning("No annotation sample yet.")
    if st.button(f"Build the {s.annotation_sample_size}-review annotation sample (seed {s.random_seed})", type="primary"):
        smp = build_annotation_sample(df, s.annotation_sample_size, s.random_seed)
        save_sample(smp, s.paths.annotation_sample_csv)
        write_template(load_sample(s.paths.annotation_sample_csv), s.paths.annotations / "annotation_template.csv")
        st.rerun()
    st.stop()

store = AnnotationStore(s.paths.annotations_json)
ids = sample["review_id"].tolist()
prog = store.progress(ids)
st.progress(prog["complete"] / max(1, prog["total"]), text=f"{prog['complete']} complete · {prog['flagged']} flagged · {prog['remaining']} remaining (of {prog['total']})")

annotator = st.text_input("Annotator name / initials", value=st.session_state.get("annotator", ""), key="annotator")

tab_label, tab_data = st.tabs(["Label reviews", "Export / import / review"])

with tab_label:
    if "ann_idx" not in st.session_state:
        nxt = next((i for i, rid in enumerate(ids) if store.get(rid) is None), 0)
        st.session_state["ann_idx"] = nxt

    def go(delta: int) -> None:
        st.session_state["ann_idx"] = max(0, min(len(ids) - 1, st.session_state["ann_idx"] + delta))

    def go_next_unlabelled() -> None:
        cur = st.session_state["ann_idx"]
        order = list(range(cur + 1, len(ids))) + list(range(0, cur + 1))
        st.session_state["ann_idx"] = next((i for i in order if store.get(ids[i]) is None), cur)

    nav = st.columns([1, 1, 2, 2])
    nav[0].button("◀ Previous", on_click=go, args=(-1,), width="stretch")
    nav[1].button("Next ▶", on_click=go, args=(1,), width="stretch")
    nav[2].button("Jump to next unlabelled ⏭", on_click=go_next_unlabelled, width="stretch")
    # No widget key: the widget is re-created whenever ann_idx changes, so it always shows the current position.
    jump = nav[3].number_input("Go to review #", 1, len(ids), st.session_state["ann_idx"] + 1)
    if int(jump) - 1 != st.session_state["ann_idx"]:
        st.session_state["ann_idx"] = int(jump) - 1

    idx = st.session_state["ann_idx"]
    row = sample.iloc[idx]
    rid = row["review_id"]
    existing = store.get(rid)
    status_txt = "✅ labelled" if existing and existing.status == "complete" else "🚩 flagged" if existing else "⬜ not labelled"
    st.markdown(f"#### Review {idx + 1} / {len(ids)} · `{rid}` · {status_txt}")
    st.caption(f"Product: {row.get('product_title', '')} ({row['product_id']})")
    st.markdown(f"<div style='font-size:1.1rem;padding:0.8rem 1rem;border-left:4px solid #888;background:rgba(128,128,128,0.08)'>{row['review_text']}</div>", unsafe_allow_html=True)
    if st.toggle("Show star rating (hidden by default to avoid bias)", key=f"showrating_{rid}"):
        st.caption(f"Rating: {row.get('rating', '?')} ★")

    k = f"_{rid}"
    ov_opts = values(OverallSentiment)
    overall = st.radio("Overall sentiment", ov_opts, index=ov_opts.index(existing.overall_sentiment.value) if existing else None, horizontal=True, key="ov" + k)

    cat_opts = values(AspectCategory)
    pre_aspects = {a.aspect_category.value: a.sentiment.value for a in existing.aspects} if existing else {}
    cats = st.multiselect("Aspect categories discussed", cat_opts, default=list(pre_aspects), format_func=pretty, key="cats" + k,
                          help="\n".join(f"{c.value}: {d}" for c, d in ASPECT_CATEGORY_DEFINITIONS.items()))
    aspect_labels: dict[str, str] = {}
    if cats:
        cols = st.columns(min(4, len(cats)))
        for i, cat in enumerate(cats):
            opts = ["positive", "negative", "neutral"]
            aspect_labels[cat] = cols[i % len(cols)].selectbox(f"{pretty(cat)} sentiment", opts, index=opts.index(pre_aspects.get(cat, "positive")), key=f"as_{cat}{k}")
    comp_opts = values(ComplaintCategory)
    complaints = st.multiselect("Complaint categories (empty if none)", comp_opts, default=[c.value for c in existing.complaint_categories] if existing else [], format_func=pretty, key="cc" + k,
                                help="\n".join(f"{c.value}: {d}" for c, d in COMPLAINT_CATEGORY_DEFINITIONS.items()))
    notes = st.text_input("Notes (optional)", value=existing.notes if existing else "", key="nt" + k)
    flagged = st.checkbox("Flag as unusable / ambiguous (excluded from evaluation)", value=bool(existing and existing.status == "flagged"), key="fl" + k)

    b = st.columns([2, 1, 3])
    if b[0].button("💾 Save & next", type="primary", width="stretch"):
        if not annotator.strip():
            st.error("Enter your annotator name first.")
        elif overall is None:
            st.error("Choose an overall sentiment.")
        else:
            try:
                ann = Annotation(
                    review_id=rid,
                    product_id=row["product_id"],
                    overall_sentiment=overall,
                    aspects=[AnnotatedAspect(aspect_category=c, sentiment=sv) for c, sv in aspect_labels.items()],
                    complaint_categories=complaints,
                    annotator=annotator.strip(),
                    notes=notes,
                    status="flagged" if flagged else "complete",
                )
                store.save(ann)
                go_next_unlabelled()
                st.rerun()
            except ValidationError as exc:
                st.error(f"Invalid annotation: {exc.errors()[0]['msg']}")
    if existing and b[1].button("🗑 Delete label", width="stretch"):
        store.delete(rid)
        st.rerun()

with tab_data:
    ann_df = store.to_dataframe(sample)
    st.markdown(f"**{len(ann_df)} saved annotations**")
    st.dataframe(ann_df, width="stretch", hide_index=True)
    d = st.columns(3)
    d[0].download_button("Export annotations (CSV)", dataframe_to_csv_bytes(ann_df), "annotations.csv", "text/csv", width="stretch")
    d[1].download_button("Export annotations (JSON)", store.to_json(), "annotations.json", "application/json", width="stretch")
    tpl_path = s.paths.annotations / "annotation_template.csv"
    if tpl_path.exists():
        d[2].download_button("Blank CSV template", tpl_path.read_bytes(), "annotation_template.csv", "text/csv", width="stretch")
    if len(ann_df):
        a, b = st.columns(2)
        a.markdown("Overall sentiment distribution (human labels)")
        a.bar_chart(ann_df["overall_sentiment"].value_counts())
        b.markdown("Annotations by annotator")
        b.bar_chart(ann_df["annotator"].value_counts())

    st.markdown("#### Import labels from a filled CSV template")
    up = st.file_uploader("Filled annotation CSV", type=["csv"], key="ann_import")
    if up is not None and st.button("Import"):
        try:
            res = store.import_csv(pd.read_csv(up, keep_default_na=False), default_annotator=annotator.strip())
            st.success(f"Imported {res['imported']} · skipped {res['skipped_unlabelled']} unlabelled rows")
            for e in res["errors"][:20]:
                st.error(e)
        except Exception as exc:
            st.error(f"Import failed: {exc}")
