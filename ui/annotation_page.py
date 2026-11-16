"""Page 6 — Human Annotation: blind, independent human labelling of the 300-review D2 evaluation set.

Genuine human labels go to ``data/annotations/human_annotations.json`` (annotation/human_annotation.py).
The blind labelling tab never shows a model prediction or an AI-assisted label. Second-person verification is not
part of the project workflow: completion = every queued review has a human label or a flag (pending == 0). The legacy reference labels
(1 human + 299 AI-assisted, ``annotations.json``) are shown read-only in their own tab, clearly marked.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from annotation.annotation_manager import AnnotationStore, build_annotation_sample, label_provenance, load_sample, provenance_text, save_sample, write_template
from annotation.human_annotation import HumanAnnotationStore
from annotation.reference_adjudication import AdjudicationError, ReferenceAdjudicationStore, load_reference, reference_dataframe
from utils.readonly import ReadOnlyModeError
from config import PROJECT_ROOT
from schemas.evaluation_schema import AnnotatedAspect
from schemas.taxonomy import ASPECT_CATEGORY_DEFINITIONS, COMPLAINT_CATEGORY_DEFINITIONS, AspectCategory, ComplaintCategory, OverallSentiment, values
from ui.common import dataset, get_settings, pretty
from ui.components import badge, callout, empty_state, esc, kpi, kpi_row, md, page_header, panel, section
from utils.export import dataframe_to_csv_bytes
from utils.readonly import DISABLED_REASON, is_readonly

s = get_settings()
RO = is_readonly()
page_header("Human Annotation", "Review the D2 reference annotations and the independent, blind human annotations of the 300-review evaluation set.", meta=[badge("Reference labels", "accent")])

sample = load_sample(s.paths.annotation_sample_csv)
if sample.empty:
    df = dataset()
    if df is None:
        empty_state("Dataset not prepared", "Run <code>python scripts/prepare_dataset.py</code> first.", "database")
        st.stop()
    empty_state("No annotation sample yet", f"Build a {s.annotation_sample_size}-review sample, stratified by star rating (seed {s.random_seed}).", "edit_note")
    if st.button(f"Build the {s.annotation_sample_size}-review annotation sample (seed {s.random_seed})", type="primary", disabled=RO):
        smp = build_annotation_sample(df, s.annotation_sample_size, s.random_seed)
        save_sample(smp, s.paths.annotation_sample_csv)
        write_template(load_sample(s.paths.annotation_sample_csv), s.paths.annotations / "annotation_template.csv")
        st.rerun()
    st.stop()

hstore = HumanAnnotationStore.default(s)
queue = hstore.queue(sample["review_id"].tolist())
reviews_by_id = sample.set_index("review_id")
if hstore.reserve_ids:  # reserve reviews (replacements for flagged ones) come from the full dataset
    full = dataset()
    if full is not None:
        extra = full[full["review_id"].isin(hstore.reserve_ids)].set_index("review_id")
        reviews_by_id = pd.concat([reviews_by_id, extra[[c for c in reviews_by_id.columns if c in extra.columns]]])
prog = hstore.progress(queue)

# ---------------------------------------------------------------- D2 reference dataset (approval of existing annotations)
rstore = ReferenceAdjudicationStore.default(s)
try:
    rprog = rstore.progress()
except Exception as exc:  # malformed source file: show it instead of crashing
    rprog = None
    callout(f"<b>Reference annotations could not be validated:</b> {esc(str(exc))}", "warning", tag="Reference")
if rprog:
    section("D2 reference dataset", "Review and approval of the existing 300 reference annotations used to score the three prompt strategies")
    kpi_row([
        kpi("Reference annotations available", f"{rprog['available']} / {rprog['sample_size']}", "validated records in annotations.json", icon_name="dataset"),
        kpi("Reviewed / approved", f"{rprog['accepted']} / {rprog['required']}",
            f"{rprog['approved']} approved · {rprog['modified']} modified · {rprog['excluded']} excluded", tone="positive" if rprog["ready"] else None, icon_name="fact_check"),
        kpi("Awaiting review", rprog["awaiting_review"], "need a reviewer decision", tone="warning" if rprog["awaiting_review"] else None, icon_name="pending_actions"),
        kpi("D2 reference dataset", "Ready" if rprog["ready"] else "Not ready",
            "all reference records accepted" if rprog["ready"] else "approve or decide every record", tone="positive" if rprog["ready"] else "warning", icon_name="task_alt"),
    ])
    st.progress(min(1.0, rprog["accepted"] / max(1, rprog["required"])),
                text=f"{rprog['accepted']} of {rprog['required']} reference annotations accepted · {rprog['awaiting_review']} awaiting review"
                     + (f" · reviewer(s): {', '.join(rprog['reviewers'])}" if rprog["reviewers"] else ""))
    _ap = rprog["accepted_by_original_provenance"]
    md(f'<div class="riq-panel-caption"><b>Provenance is unchanged by approval.</b> Original source of the reference annotations: '
       f'1 human-created, {rprog["available"] - 1} AI-assisted (mentor-approved at generation, per annotations.json). Accepted so far: '
       f'{_ap["human_created"]} human-created, {_ap["ai_assisted"]} AI-assisted. Approval records that a named person reviewed and accepted '
       'a label; it does not make an AI-assisted label human-created. Decisions are stored separately in '
       '<code>data/annotations/reference_adjudication.json</code>; <code>annotations.json</code> is never modified.</div>')
    if rprog["stale_decisions"]:
        callout(f"{len(rprog['stale_decisions'])} decision(s) no longer match the source label and need re-review: "
                f"{esc(', '.join(rprog['stale_decisions'][:10]))}", "warning", tag="Changed")
    with st.expander("Approve existing reference annotations" + (" — done" if rprog["ready"] else ""), icon=":material/fact_check:",
                     expanded=not rprog["ready"] and not RO):
        md('<div class="riq-panel-caption">Approves every valid reference record that has status <i>complete</i> and no decision yet. '
           'Flagged records are never bulk-approved; decide them individually below. Running it again changes nothing.</div>')
        with st.form("ref_bulk"):
            _rev = st.text_input("Reviewer name (recorded with every approval)", key="ref_reviewer")
            _ok = st.checkbox("I have reviewed these reference labels and accept them as the D2 comparison dataset. "
                              "I understand they remain AI-assisted labels.", key="ref_confirm")
            _go = st.form_submit_button("Approve Existing Reference Annotations", type="primary", disabled=RO, icon=":material/done_all:")
        if _go:
            try:
                _st = rstore.bulk_approve(_rev, confirmed=_ok)
                st.success(f"Approved {_st['approved_now']} record(s); {_st['already_decided']} already decided; "
                           f"{_st['skipped_flagged']} flagged record(s) need an individual decision.")
                st.rerun()
            except (AdjudicationError, ReadOnlyModeError) as exc:
                st.error(str(exc))
        if rprog["accepted"]:
            _rdf = reference_dataframe(rstore)
            st.download_button(f"Approved reference dataset (CSV, {len(_rdf)})", dataframe_to_csv_bytes(_rdf), "approved_reference.csv",
                               "text/csv", icon=":material/download:")
        _ref = load_reference(s.paths.annotations_json, s.paths.annotation_sample_csv)
        _dec = rstore.decisions()
        _todo = [rid for rid in _ref["sample_ids"] if rid not in _dec or rid in rprog["stale_decisions"]]
        _pick_from = _todo + [rid for rid in _ref["sample_ids"] if rid not in _todo]
        md('<div class="riq-panel-caption"><b>Individual decision</b> (approve, correct or exclude one record; replaces an earlier '
           'decision and keeps its history).</div>')
        _rid = st.selectbox("Record", _pick_from, format_func=lambda r: f"{r} · " + (_dec[r]["decision"] if r in _dec else "awaiting review")
                            + (" · flagged in source" if r in rprog["flagged_in_source"] else ""), key="ref_rid")
        if _rid:
            _rec = _ref["records"][_rid]
            if _rid in reviews_by_id.index:
                md(f'<div class="riq-panel-caption">{esc(str(reviews_by_id.loc[_rid, "review_text"])[:1200])}</div>')
            md(f'<div class="riq-panel-caption">Reference label: <b>{esc(_rec["overall_sentiment"])}</b> · aspects '
               f'{esc(", ".join(a["aspect_category"] + ":" + a["sentiment"] for a in _rec["aspects"]) or "–")} · complaints '
               f'{esc(", ".join(_rec["complaint_categories"]) or "–")} · original provenance <b>{esc(_ref["provenance"][_rid])}</b>'
               + (f' · note: {esc(_rec.get("notes", "")[:200])}' if _rec.get("notes") else "") + '</div>')
            with st.form(f"ref_one_{_rid}"):
                _rev1 = st.text_input("Reviewer name", key=f"ref_rev1_{_rid}")
                _choice = st.selectbox("Decision", ["approve as is", "correct overall sentiment", "exclude"], key=f"ref_choice_{_rid}")
                _new = st.selectbox("Corrected overall sentiment", values(OverallSentiment), index=values(OverallSentiment).index(_rec["overall_sentiment"]), key=f"ref_new_{_rid}")
                _why = st.text_input("Reason (required to exclude or correct)", key=f"ref_why_{_rid}")
                _save = st.form_submit_button("Save decision", disabled=RO)
            if _save:
                try:
                    if _choice == "approve as is":
                        rstore.decide(_rid, "approved", _rev1)
                    elif _choice == "exclude":
                        rstore.decide(_rid, "excluded", _rev1, reason=_why)
                    else:
                        if not _why.strip():
                            raise AdjudicationError("a reason is required for a correction")
                        rstore.decide(_rid, "modified", _rev1, reason=_why, correction={"overall_sentiment": _new})
                    st.rerun()
                except (AdjudicationError, ReadOnlyModeError, ValueError) as exc:
                    st.error(str(exc))

# ---------------------------------------------------------------- independent human labels (separate, stronger evidence)
section("Independent human annotations", "Blind labels created by people from the review text alone (separate from the reference approval)")
_done = prog["annotation_complete"]
kpi_row([
    kpi("Human-created annotations", prog["human_created"], "labelled blind by a person (no AI label shown)", icon_name="person"),
    kpi("Completed annotations", f"{prog['labelled']} / {prog['queue']}", "reviews with a saved human label",
        tone="positive" if _done else None, icon_name="task_alt"),
    kpi("Pending annotations", prog["pending"], "reviews without a human label yet", tone="warning" if prog["pending"] else "positive", icon_name="pending_actions"),
    kpi("Flagged annotations", prog["flagged"], "marked unusable / ambiguous by the annotator", tone="warning" if prog["flagged"] else None, icon_name="flag"),
])
st.progress(min(1.0, (prog["labelled"] + prog["flagged"]) / max(1, prog["queue"])),
            text=f"{prog['labelled']} completed · {prog['flagged']} flagged · {prog['pending']} pending — independent human annotations")
if _done:
    callout(f"<b>All {prog['labelled']} independent human annotations are complete.</b> Each was created blind by a named annotator "
            f"({esc(', '.join(prog['annotators']))}). Second-person verification is not required for this project, so these labels have "
            "<b>not</b> been independently verified by a second annotator.", "positive", tag="Complete")
md('<div class="riq-panel-caption">These human-created annotations are stored separately from the AI-assisted reference annotations '
   '(<code>human_annotations.json</code> vs <code>annotations.json</code>) and are never merged with them. The reported D2 results measure '
   'agreement with the approved (mostly AI-assisted) reference labels above, not with these human annotations.</div>')
callout(
    "<b>Integrity rules.</b> Label each review yourself from its text. The <b>Label (blind)</b> tab never shows a model prediction or an "
    "AI-assisted label. AI-assisted reference labels are never counted as human labels. Every save is written immediately, so you can "
    "stop and resume at any time.",
    "accent", tag="Guideline",
)
if RO:
    callout(f"<b>Labels are read-only.</b> Labelling, flagging and import are disabled. {esc(DISABLED_REASON)} Progress, the guide and exports still work.", "warning", tag="Demo")
with st.expander("Annotation guide", icon=":material/menu_book:"):
    st.markdown((PROJECT_ROOT / "annotation" / "annotation_guide.md").read_text(encoding="utf-8"))

tab_label, tab_queue, tab_legacy = st.tabs(["Label (blind)", "Queue & export", "AI-assisted reference labels (legacy)"])

ov_opts, cat_opts, comp_opts = values(OverallSentiment), values(AspectCategory), values(ComplaintCategory)


def label_form(prefix: str, rid: str, pre=None) -> tuple:
    """Overall sentiment, aspects with sentiment, complaints. ``pre`` is a HUMAN label (never an AI one)."""
    k = f"{prefix}_{rid}"
    overall = st.radio("Overall sentiment", ov_opts, index=ov_opts.index(pre.overall_sentiment.value) if pre and pre.overall_sentiment else None, horizontal=True, key="ov" + k)
    pre_aspects = {a.aspect_category.value: a.sentiment.value for a in pre.aspects} if pre else {}
    cats = st.multiselect("Aspect categories discussed", cat_opts, default=list(pre_aspects), format_func=pretty, key="cats" + k,
                          help="\n".join(f"{c.value}: {d}" for c, d in ASPECT_CATEGORY_DEFINITIONS.items()), placeholder="Select aspects mentioned in the review")
    aspect_labels: dict[str, str] = {}
    if cats:
        cols = st.columns(min(3, len(cats)))
        for i, cat in enumerate(cats):
            opts = ["positive", "negative", "neutral"]
            aspect_labels[cat] = cols[i % len(cols)].selectbox(pretty(cat), opts, index=opts.index(pre_aspects.get(cat, "positive")), key=f"as_{cat}{k}")
    complaints = st.multiselect("Complaint categories", comp_opts, default=[c.value for c in pre.complaint_categories] if pre else [], format_func=pretty, key="cc" + k,
                                help="\n".join(f"{c.value}: {d}" for c, d in COMPLAINT_CATEGORY_DEFINITIONS.items()), placeholder="None (leave empty if no complaint)")
    return overall, [AnnotatedAspect(aspect_category=c, sentiment=v) for c, v in aspect_labels.items()], complaints


def review_box(rid: str, title: str) -> None:
    row = reviews_by_id.loc[rid]
    md(f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0 6px"><span style="font-size:16px;font-weight:600">{esc(title)}</span>'
       f'{badge(hstore.status_of(rid), "positive" if hstore.status_of(rid) in ("completed", "verified") else "neutral", dot=True)}</div>'
       f'<div class="riq-review-meta"><span><b>ID</b> {esc(rid)}</span><span><b>Product</b> {esc(str(row.get("product_title", "") or ""))} ({esc(str(row["product_id"]))})</span></div>'
       f'<div class="riq-review">{esc(str(row["review_text"]))}</div>')


# ---------------------------------------------------------------- blind labelling
with tab_label:
    annotator = st.text_input("Your name / initials (recorded with every label)", key="h_annotator")
    if "h_cur" not in st.session_state or st.session_state["h_cur"] not in queue:
        st.session_state["h_cur"] = hstore.next_pending(queue) or queue[0]
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Review", queue, index=queue.index(st.session_state["h_cur"]),
                        format_func=lambda r: f"#{queue.index(r) + 1} · {r} · {hstore.status_of(r)}", key=f"h_pick_{st.session_state['h_cur']}")
    if pick != st.session_state["h_cur"]:
        st.session_state["h_cur"] = pick
        st.rerun()
    if c2.button("Next pending", icon=":material/skip_next:", width="stretch"):
        nxt = hstore.next_pending(queue, after=st.session_state["h_cur"])
        if nxt:
            st.session_state["h_cur"] = nxt
            st.rerun()
        st.info("No pending reviews left.")
    rid = st.session_state["h_cur"]
    existing = hstore.get(rid)
    c3.markdown(f'<div class="riq-panel-caption">Resume point: first pending is #{queue.index(hstore.next_pending(queue)) + 1 if hstore.next_pending(queue) else "–"}</div>', unsafe_allow_html=True)

    left, right = st.columns([5, 4], gap="large")
    with left:
        review_box(rid, f"Review {queue.index(rid) + 1} / {len(queue)}")
        if st.toggle("Show star rating (hidden by default to avoid bias)", key=f"h_rating_{rid}"):
            md(f'<div class="riq-panel-caption" style="margin-top:6px">Rating: {esc(str(reviews_by_id.loc[rid].get("rating", "?")))} ★</div>')
        md('<div class="riq-panel-caption" style="margin-top:8px">No model prediction or AI-assisted label is shown on this tab.</div>')
    with right, panel("Your labels"):
        mine = existing if existing and existing.provenance == "human_created" and existing.annotator.strip().lower() == annotator.strip().lower() else None
        locked = existing is not None and mine is None
        if locked:
            callout(f"This review already has a human label ({esc(existing.status)}, by {esc(existing.annotator)}). "
                    "It is not shown here, so it cannot bias you; pick another review.", "neutral", tag="Labelled")
        else:
            overall, aspects, complaints = label_form("blind", rid, pre=mine)
            notes = st.text_input("Notes (optional)", value=mine.notes if mine else "", key=f"h_notes_{rid}")
            b = st.columns(3)
            if b[0].button("Save & next", type="primary", icon=":material/check:", width="stretch", disabled=RO, key=f"h_save_{rid}"):
                if not annotator.strip():
                    st.error("Enter your name first.")
                elif overall is None:
                    st.error("Choose an overall sentiment.")
                else:
                    try:
                        hstore.save_blind_label(rid, str(reviews_by_id.loc[rid]["product_id"]), annotator=annotator, overall_sentiment=overall,
                                                aspects=aspects, complaint_categories=complaints, notes=notes)
                        st.session_state["h_cur"] = hstore.next_pending(queue, after=rid) or rid
                        st.rerun()
                    except (ValidationError, ValueError) as exc:
                        st.error(f"Not saved: {str(exc).splitlines()[0]}")
            with b[1].popover("Flag review", icon=":material/flag:", width="stretch", disabled=RO):
                reason = st.text_input("Why is it unusable / ambiguous?", key=f"h_flag_reason_{rid}")
                if st.button("Flag", key=f"h_flag_{rid}", disabled=RO):
                    try:
                        hstore.flag(rid, str(reviews_by_id.loc[rid]["product_id"]), annotator=annotator or "?", reason=reason)
                        st.session_state["h_cur"] = hstore.next_pending(queue, after=rid) or rid
                        st.rerun()
                    except (ValidationError, ValueError) as exc:
                        st.error(str(exc).splitlines()[0])
            if mine and b[2].button("Remove my label", icon=":material/delete:", width="stretch", disabled=RO, key=f"h_del_{rid}"):
                try:
                    hstore.reset(rid, by=annotator)
                    st.rerun()
                except ValueError as exc:  # AnnotationConflict: changed by another session meanwhile
                    st.error(f"Not removed: {str(exc).splitlines()[0]}")

# ---------------------------------------------------------------- queue & export
with tab_queue:
    qdf = pd.DataFrame([{"#": i + 1, "review_id": r, "status": hstore.status_of(r),
                         "provenance": (hstore.get(r).provenance if hstore.get(r) else ""),
                         "annotator": (hstore.get(r).annotator if hstore.get(r) else ""),
                         "updated_at": (hstore.get(r).updated_at if hstore.get(r) else ""),
                         "reserve": r in hstore.reserve_ids} for i, r in enumerate(queue)])
    f1, f2 = st.columns(2)
    _status_opts = ["pending", "completed", "flagged"] + (["verified"] if prog["verified"] else [])  # "verified" only if legacy records exist
    sel = f1.multiselect("Status", _status_opts, default=[], placeholder="All statuses", key="h_qstatus")
    st.dataframe(qdf[qdf["status"].isin(sel)] if sel else qdf, width="stretch", hide_index=True)
    all_df = hstore.to_dataframe()
    d = st.columns(2)
    d[0].download_button(f"Human annotations (CSV, {len(all_df)})", dataframe_to_csv_bytes(all_df), "human_annotations.csv", "text/csv",
                         icon=":material/table:", width="stretch", help="All human-created labels and flags. Not second-person verified.")
    tpl_path = s.paths.annotations / "annotation_template.csv"
    if tpl_path.exists():
        d[1].download_button("Blank offline template (no AI labels)", tpl_path.read_bytes(), "annotation_template.csv", "text/csv", icon=":material/download:", width="stretch")
    with panel("Import blind labels from the filled blank template"):
        up = st.file_uploader("Filled template CSV", type=["csv"], key="h_import")
        if up is not None and st.button("Import as blind human labels", icon=":material/upload:", disabled=RO):
            res = hstore.import_blind_csv(pd.read_csv(up, keep_default_na=False, dtype=str), dict(zip(reviews_by_id.index, reviews_by_id["product_id"].astype(str))),
                                          default_annotator=st.session_state.get("h_annotator", "").strip())
            st.success(f"Imported {res['imported']} · skipped {res['skipped_unlabelled']} unlabelled rows")
            for e in res["errors"][:20]:
                st.error(e)
    md(f'<div class="riq-panel-caption">Flagged reviews: {prog["flagged"]}. Add replacement reviews with '
       '<code>python scripts/human_annotation.py reserve --n &lt;flagged&gt;</code>, then label them here.</div>')

# ---------------------------------------------------------------- legacy reference labels (read-only)
with tab_legacy:
    legacy = AnnotationStore(s.paths.annotations_json)
    ids = set(sample["review_id"])
    prov = label_provenance([a for a in legacy.all() if a.review_id in ids])
    callout(f"<b>Not human ground truth.</b> {esc(provenance_text(prov))}. These labels produced the reported D2 numbers "
            "(agreement with AI-assisted reference labels). They are shown read-only and are never copied into the human store "
            "(only the documented human label was migrated, as a blind human label).", "warning", tag="AI-assisted")
    ldf = legacy.to_dataframe(sample)
    st.dataframe(ldf, width="stretch", hide_index=True)
    st.download_button("Export legacy reference labels (CSV)", dataframe_to_csv_bytes(ldf), "annotations.csv", "text/csv", icon=":material/table:")
