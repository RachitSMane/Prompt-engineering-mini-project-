"""Page 6 — Human Annotation: manual labelling of the 300-review evaluation set."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from annotation.annotation_manager import AnnotationStore, build_annotation_sample, load_sample, save_sample, write_template
from config import PROJECT_ROOT
from evaluation.visualizations import counts_bar
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
from ui.components import badge, callout, chart, empty_state, esc, kpi, kpi_row, md, page_header, panel
from ui.theme import COLORS
from utils.export import dataframe_to_csv_bytes

s = get_settings()
page_header("Human Annotation", "Create the manually labelled ground-truth dataset for prompt evaluation.", meta=[badge("Ground truth", "accent")])

sample = load_sample(s.paths.annotation_sample_csv)
if sample.empty:
    df = dataset()
    if df is None:
        empty_state("Dataset not prepared", "Run <code>python scripts/prepare_dataset.py</code> first.", "database")
        st.stop()
    empty_state("No annotation sample yet", f"Build a {s.annotation_sample_size}-review sample, stratified by star rating (seed {s.random_seed}).", "edit_note")
    if st.button(f"Build the {s.annotation_sample_size}-review annotation sample (seed {s.random_seed})", type="primary"):
        smp = build_annotation_sample(df, s.annotation_sample_size, s.random_seed)
        save_sample(smp, s.paths.annotation_sample_csv)
        write_template(load_sample(s.paths.annotation_sample_csv), s.paths.annotations / "annotation_template.csv")
        st.rerun()
    st.stop()

store = AnnotationStore(s.paths.annotations_json)
ids = sample["review_id"].tolist()
prog = store.progress(ids)
pct = 100 * prog["complete"] / max(1, prog["total"])

# ---------------------------------------------------------------- progress
kpi_row([
    kpi("Total reviews", prog["total"], "stratified evaluation sample", icon_name="list_alt"),
    kpi("Completed", prog["complete"], f"{prog['flagged']} flagged (excluded)", tone="positive", icon_name="task_alt"),
    kpi("Remaining", prog["remaining"], "not yet labelled", icon_name="pending"),
    kpi("Completion", f"{pct:.0f}%", "required for Deliverable 2", tone="positive" if pct >= 100 else None, icon_name="donut_large"),
])
st.progress(prog["complete"] / max(1, prog["total"]), text=f"{prog['complete']} complete · {prog['flagged']} flagged · {prog['remaining']} remaining (of {prog['total']})")
callout(
    "Label each review <b>yourself</b>, from the text. Never copy model predictions here: these labels are the ground truth the prompts are evaluated against. "
    "Every save is written immediately to <code>data/annotations/annotations.json</code>.",
    "accent",
    tag="Guideline",
)
with st.expander("Annotation guide", icon=":material/menu_book:"):
    st.markdown((PROJECT_ROOT / "annotation" / "annotation_guide.md").read_text(encoding="utf-8"))

tab_label, tab_data = st.tabs(["Label reviews", "Export / import / review"])

with tab_label:
    if "ann_idx" not in st.session_state:
        st.session_state["ann_idx"] = next((i for i, rid in enumerate(ids) if store.get(rid) is None), 0)

    def next_unlabelled(cur: int) -> int:
        order = list(range(cur + 1, len(ids))) + list(range(0, cur + 1))
        return next((i for i in order if store.get(ids[i]) is None), cur)

    idx = st.session_state["ann_idx"]

    # ------------------------------------------------------------ queue strip (current review highlighted)
    lo = max(0, min(idx - 12, len(ids) - 25))
    cells = []
    for i in range(lo, min(len(ids), lo + 25)):
        a = store.get(ids[i])
        cls = "cur" if i == idx else ("flag" if a and a.status == "flagged" else "done" if a else "")
        cells.append(f'<span class="{cls}" title="{esc(ids[i])}">{i + 1}</span>')
    q1, q2 = st.columns([5, 1], vertical_alignment="center")
    q1.markdown(f'<div class="riq-queue" aria-label="Review queue">{"".join(cells)}</div>', unsafe_allow_html=True)
    jump = q2.number_input("Go to review #", 1, len(ids), idx + 1, key=f"jump_{idx}", label_visibility="collapsed", help="Go to review number")

    row = sample.iloc[idx]
    rid = row["review_id"]
    existing = store.get(rid)
    status_badge = badge("labelled", "positive", dot=True) if existing and existing.status == "complete" else badge("flagged", "warning", dot=True) if existing else badge("not labelled", "neutral", dot=True)

    left, right = st.columns([5, 4], gap="large")
    with left:
        md(
            f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0 6px"><span style="font-size:16px;font-weight:600">Review {idx + 1} / {len(ids)}</span>{status_badge}</div>'
            f'<div class="riq-review-meta"><span><b>ID</b> {esc(rid)}</span><span><b>Product</b> {esc(row.get("product_title", "") or "")} ({esc(row["product_id"])})</span></div>'
            f'<div class="riq-review">{esc(row["review_text"])}</div>'
        )
        if st.toggle("Show star rating (hidden by default to avoid bias)", key=f"showrating_{rid}"):
            md(f'<div class="riq-panel-caption" style="margin-top:6px">Rating: {esc(row.get("rating", "?"))} ★</div>')

    with right, panel("Labels"):
        k = f"_{rid}"
        annotator = st.text_input("Annotator name / initials", value=st.session_state.get("annotator", ""), key="annotator")
        ov_opts = values(OverallSentiment)
        overall = st.radio("Overall sentiment", ov_opts, index=ov_opts.index(existing.overall_sentiment.value) if existing else None, horizontal=True, key="ov" + k)

        cat_opts = values(AspectCategory)
        pre_aspects = {a.aspect_category.value: a.sentiment.value for a in existing.aspects} if existing else {}
        cats = st.multiselect("Aspect categories discussed", cat_opts, default=list(pre_aspects), format_func=pretty, key="cats" + k,
                              help="\n".join(f"{c.value}: {d}" for c, d in ASPECT_CATEGORY_DEFINITIONS.items()), placeholder="Select aspects mentioned in the review")
        aspect_labels: dict[str, str] = {}
        if cats:
            cols = st.columns(min(3, len(cats)))
            for i, cat in enumerate(cats):
                opts = ["positive", "negative", "neutral"]
                aspect_labels[cat] = cols[i % len(cols)].selectbox(f"{pretty(cat)}", opts, index=opts.index(pre_aspects.get(cat, "positive")), key=f"as_{cat}{k}")
        comp_opts = values(ComplaintCategory)
        complaints = st.multiselect("Complaint categories", comp_opts, default=[c.value for c in existing.complaint_categories] if existing else [], format_func=pretty, key="cc" + k,
                                    help="\n".join(f"{c.value}: {d}" for c, d in COMPLAINT_CATEGORY_DEFINITIONS.items()), placeholder="None (leave empty if no complaint)")
        notes = st.text_input("Notes (optional)", value=existing.notes if existing else "", key="nt" + k)
        flagged = st.checkbox("Flag as unusable / ambiguous (excluded from evaluation)", value=bool(existing and existing.status == "flagged"), key="fl" + k)

    # ------------------------------------------------------------ unsaved-change detection
    if existing:
        dirty = (
            overall != existing.overall_sentiment.value
            or aspect_labels != pre_aspects
            or set(complaints) != {c.value for c in existing.complaint_categories}
            or notes != (existing.notes or "")
            or flagged != (existing.status == "flagged")
        )
    else:
        dirty = bool(overall or cats or complaints or notes or flagged)

    def save() -> bool:
        if not annotator.strip():
            st.error("Enter your annotator name first.")
            return False
        if overall is None:
            st.error("Choose an overall sentiment.")
            return False
        try:
            store.save(Annotation(
                review_id=rid,
                product_id=row["product_id"],
                overall_sentiment=overall,
                aspects=[AnnotatedAspect(aspect_category=c, sentiment=sv) for c, sv in aspect_labels.items()],
                complaint_categories=complaints,
                annotator=annotator.strip(),
                notes=notes,
                status="flagged" if flagged else "complete",
            ))
            return True
        except ValidationError as exc:
            st.error(f"Invalid annotation: {exc.errors()[0]['msg']}")
            return False

    def set_idx(target: int) -> None:
        """Move to a review; clear stale 'go to' inputs and any pending navigation."""
        for key in [x for x in st.session_state if str(x).startswith("jump_")]:
            del st.session_state[key]
        st.session_state.pop("ann_pending", None)
        st.session_state["ann_idx"] = target

    def navigate(target: int) -> None:
        target = max(0, min(len(ids) - 1, target))
        if target == idx:
            return
        if not dirty:
            set_idx(target)
            st.rerun()
        elif st.session_state.get("ann_pending") != target:
            st.session_state["ann_pending"] = target  # ask for confirmation below instead of losing edits
            st.rerun()

    b = st.columns([1, 1, 1.4, 1.6, 1.2])
    if b[0].button("Previous", icon=":material/chevron_left:", width="stretch", disabled=idx == 0):
        navigate(idx - 1)
    if b[1].button("Save", icon=":material/save:", width="stretch"):
        if save():
            st.rerun()
    if b[2].button("Save & next", type="primary", icon=":material/check:", width="stretch"):
        if save():
            set_idx(next_unlabelled(idx))
            st.rerun()
    if b[3].button("Next unlabelled", icon=":material/skip_next:", width="stretch"):
        navigate(next_unlabelled(idx))
    if existing and b[4].button("Delete label", icon=":material/delete:", width="stretch"):
        store.delete(rid)
        st.rerun()
    if int(jump) - 1 != idx:
        navigate(int(jump) - 1)

    if dirty:
        md(f'<div style="margin-top:6px">{badge("Unsaved changes", "warning", dot=True)}</div>')
    pending = st.session_state.get("ann_pending")
    if pending is not None and pending != idx:
        callout(f"<b>You have unsaved changes on review {idx + 1}.</b> Save them, or discard them and go to review {pending + 1}.", "warning", tag="Unsaved")
        c1, c2, _ = st.columns([1.2, 1, 3])
        if c1.button("Discard and continue", icon=":material/close:", width="stretch"):
            for key in [f"ov{k}", f"cats{k}", f"cc{k}", f"nt{k}", f"fl{k}"] + [f"as_{c}{k}" for c in cat_opts]:
                st.session_state.pop(key, None)
            set_idx(pending)
            st.rerun()
        if c2.button("Keep editing", width="stretch"):
            st.session_state.pop("ann_pending", None)
            st.session_state.pop(f"jump_{idx}", None)
            st.rerun()

with tab_data:
    ann_df = store.to_dataframe(sample)
    md(f'<div class="riq-panel-caption">{len(ann_df)} saved annotations</div>')
    st.dataframe(ann_df, width="stretch", hide_index=True)
    d = st.columns(3)
    d[0].download_button("Export annotations (CSV)", dataframe_to_csv_bytes(ann_df), "annotations.csv", "text/csv", icon=":material/table:", width="stretch")
    d[1].download_button("Export annotations (JSON)", store.to_json(), "annotations.json", "application/json", icon=":material/data_object:", width="stretch")
    tpl_path = s.paths.annotations / "annotation_template.csv"
    if tpl_path.exists():
        d[2].download_button("Blank CSV template", tpl_path.read_bytes(), "annotation_template.csv", "text/csv", icon=":material/download:", width="stretch")
    if len(ann_df):
        a, b = st.columns(2, gap="medium")
        with a, panel("Overall sentiment (human labels)"):
            chart(counts_bar(ann_df["overall_sentiment"].value_counts().to_dict(), color=COLORS["accent"]), key="an_sent")
        with b, panel("Annotations by annotator"):
            chart(counts_bar(ann_df["annotator"].value_counts().to_dict(), color="#38BDF8"), key="an_who")

    with panel("Import labels from a filled CSV template"):
        up = st.file_uploader("Filled annotation CSV", type=["csv"], key="ann_import")
        if up is not None and st.button("Import", icon=":material/upload:"):
            try:
                res = store.import_csv(pd.read_csv(up, keep_default_na=False), default_annotator=st.session_state.get("annotator", "").strip())
                st.success(f"Imported {res['imported']} · skipped {res['skipped_unlabelled']} unlabelled rows")
                for e in res["errors"][:20]:
                    st.error(e)
            except Exception as exc:
                st.error(f"Import failed: {exc}")
