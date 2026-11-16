"""Human claim validation (single annotator) — section of the Summary Evaluation page.

One named person labels each of the 239 claims of the 15 original summaries against the evidence of the frozen claim
template. Judge v4 verdicts, scores and explanations are never read or shown here; the page hides all judge results while
labelling is incomplete (see ``ui/summary_evaluation.py``). Labels are stored in ``data/annotations/d3_claim_validation.json``
by ``evaluation.claim_gold.ClaimValidationStore`` (one label per claim; changes are explicit revisions with history).
"""

from __future__ import annotations

import re

import streamlit as st

from evaluation.claim_gold import (
    LABEL_GUIDE, LABELS, GUIDE_NOTES, AnnotationInputError, ClaimValidationStore, DuplicateAnnotationError, export_csv_bytes,
    export_rows, product_statistics,
)
from evaluation.d3_status import DEFAULT_SOURCE, paths
import ui.common as common
from ui.components import empty_state, section
from utils.readonly import DISABLED_REASON, ReadOnlyModeError, is_readonly

FIELD_LABEL = {"overall_opinion": "Overall opinion (statistic sentence)", "appreciated": "Appreciated point (cited reviews)",
               "complaints": "Complaint point (cited reviews)", "frequent_aspects": "Frequent aspects (statistics)",
               "improvement_areas": "Improvement area (statistics)", "recommendations": "Recommendation (statistics)"}
_REVIEW_SPLIT = re.compile(r"\n\n(?=\[R[0-9a-f]{12}\])")
_REVIEW_HEAD = re.compile(r"^\[(R[0-9a-f]{12})\] \(([^)]*)\) ?")


def _paths(source: str = DEFAULT_SOURCE) -> dict:
    return paths(common.get_settings().paths.mode_dir(False), source)


def validation_state(source: str = DEFAULT_SOURCE) -> dict:
    """Cheap status used by the page to decide whether judge results must stay hidden."""
    p = _paths(source)
    if not p["template"].exists():
        return {"available": False, "complete": False, "store_exists": False}
    store = ClaimValidationStore(p["store"], p["template"])
    prog = store.progress()
    return {"available": True, "complete": prog["complete"], "store_exists": p["store"].exists(), **prog}


def log_results_opened(source: str = DEFAULT_SOURCE) -> None:
    """Record (once per session) that judge results were opened while labelling was incomplete."""
    if is_readonly() or st.session_state.get("hcv_results_logged"):
        return
    p = _paths(source)
    if not p["store"].exists():  # labelling has not started; exposure before starting is a documented general limitation
        return
    try:
        ClaimValidationStore(p["store"], p["template"]).log_event("judge_results_opened_before_complete",
                                                                  "Evaluation results view opened on the Summary Evaluation page")
        st.session_state["hcv_results_logged"] = True
    except ReadOnlyModeError:
        pass


def _go(cid: str) -> None:
    st.session_state["hcv_cid"] = cid


def _go_from_jump(key: str) -> None:
    st.session_state["hcv_cid"] = st.session_state[key]


def render(source: str = DEFAULT_SOURCE) -> None:
    p = _paths(source)
    section("Human claim validation", "One annotator labels every claim against its evidence · judge v4 verdicts stay hidden")
    if not p["template"].exists():
        empty_state("Claim template not exported",
                    f"Run <code>python scripts/claim_gold.py export --source {source}</code> first.", "inbox")
        return
    store = ClaimValidationStore(p["store"], p["template"])
    data = store.data()
    labels = {r["claim_id"]: r for r in data["labels"]}
    prog = store.progress()
    readonly = is_readonly()

    st.caption("Single-annotator validation: the comparison with judge v4 is computed only after **all** claims are labelled, "
               "and measures agreement with one person, not inter-annotator reliability or ground truth. Judge outputs are hidden "
               "on this page while you label; please do not consult judge outputs or reports elsewhere while labelling.")
    with st.container(horizontal=True):
        st.metric("Completed", prog["completed"], border=True)
        st.metric("Remaining", prog["remaining"], border=True)
        st.metric("Total claims", prog["total"], border=True)
    st.progress(prog["completed"] / prog["total"] if prog["total"] else 0.0,
                text=f"{prog['completed']} of {prog['total']} claims labelled")

    with st.expander("Labelling guide", icon=":material/menu_book:"):
        for lab in LABELS:
            st.markdown(f"**{lab}** — {LABEL_GUIDE[lab]}")
        st.markdown(GUIDE_NOTES)

    if prog["complete"]:
        st.success("Run `python scripts/d3_completion.py --source " + source + " --verify` to generate the comparison with judge v4 "
                   "and the final report. Switch the page view to *Evaluation results* to see judge outputs.",
                   title="All claims are labelled")

    # -- annotator (single, fixed after the first saved label)
    if data.get("annotator"):
        annotator = data["annotator"]
        st.text_input("Annotator", value=annotator, disabled=True, key="hcv_annotator_fixed",
                      help="This validation uses one annotator; the name is fixed by the first saved label.")
    else:
        annotator = st.text_input("Your name (annotator)", key="hcv_annotator", max_chars=80, disabled=readonly,
                                  help="Required before the first label. It is stored with every label and cannot change later.")

    # -- navigation
    only_open = st.toggle("Show only unreviewed claims", value=True, key="hcv_only_open")
    order = store.order
    ids = [c for c in order if c not in labels] if only_open else list(order)
    if not ids:
        st.info("No unreviewed claims left. Turn off the filter to review or revise saved labels.")
        _export(store, data)
        return
    cur = st.session_state.get("hcv_cid")
    if cur not in ids:  # first visit (resume), or the current claim was just labelled under the filter
        after = [c for c in ids if cur in order and order.index(c) > order.index(cur)] if cur in order else []
        cur = after[0] if after else (prog["next_unreviewed"] if prog["next_unreviewed"] in ids else ids[0])
        st.session_state["hcv_cid"] = cur
    pos = ids.index(cur)
    number = {c: i + 1 for i, c in enumerate(order)}
    row = store.by_id[cur]

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.button("Previous", icon=":material/chevron_left:", disabled=pos == 0, on_click=_go,
                  args=(ids[pos - 1] if pos > 0 else cur,), key="hcv_prev")
        jump_key = f"hcv_jump_{cur}"
        st.selectbox("Claim", ids, index=pos, key=jump_key, on_change=_go_from_jump, args=(jump_key,),
                     format_func=lambda c: f"#{number[c]} · {store.by_id[c]['product_id']} · {store.by_id[c]['claim_field']}"
                     + (" · labelled" if c in labels else ""))
        st.button("Next", icon=":material/chevron_right:", icon_position="right", disabled=pos == len(ids) - 1, on_click=_go,
                  args=(ids[pos + 1] if pos < len(ids) - 1 else cur,), key="hcv_next")

    # -- the claim and its evidence
    with st.container(border=True):
        st.markdown(f"**Claim #{number[cur]} of {len(order)}** · product `{row['product_id']}` · summary `{row['summary_id']}` · "
                    f"{FIELD_LABEL.get(row['claim_field'], row['claim_field'])}")
        st.markdown(f"> {row['claim_text']}")
        st.caption(f"Claim id `{cur}` · cited review ids: {row['cited_review_ids'] or 'none (checked against the statistics)'}")

    stats_text = product_statistics(store.template).get(row["product_id"], "")
    if row["evidence"].startswith("STATISTICS"):
        st.markdown("**Evidence: deterministic product statistics**")
        st.code(row["evidence"].removeprefix("STATISTICS\n"), language=None, wrap_lines=True)
    else:
        st.markdown("**Evidence: full text of the cited reviews**")
        for block in _REVIEW_SPLIT.split(row["evidence"]):
            m = _REVIEW_HEAD.match(block)
            with st.container(border=True):
                if m:
                    st.caption(f"Review `{m.group(1)}` · {m.group(2)}")
                    st.markdown(block[m.end():])
                else:
                    st.markdown(block)
        if stats_text:
            with st.expander("Product statistics (context)", icon=":material/bar_chart:"):
                st.code(stats_text.removeprefix("STATISTICS\n"), language=None, wrap_lines=True)

    # -- label form
    saved = labels.get(cur)
    revise = False
    if saved:
        st.info(f"Saved label: **{saved['label']}**" + (f" — {saved['explanation']}" if saved["explanation"] else "")
                + f" · {saved['labelled_at']}" + (f" · revision {saved['revision']}" if saved.get("revision") else ""),
                title="Already labelled")
        revise = st.checkbox("Revise this label (the previous label is kept in the revision history)", key=f"hcv_revise_{cur}",
                             disabled=readonly)
    if readonly:
        st.warning(DISABLED_REASON, title="Read-only mode: labels cannot be saved")
    locked = readonly or (saved is not None and not revise)
    with st.form(key=f"hcv_form_{cur}_{'rev' if revise else 'new'}", border=True):
        choice = st.radio("Label", LABELS, index=None, key=f"hcv_label_{cur}_{revise}", disabled=locked,
                          captions=[LABEL_GUIDE[x] for x in LABELS])
        expl = st.text_area("Explanation (optional)", key=f"hcv_expl_{cur}_{revise}", max_chars=2000, disabled=locked,
                            placeholder="e.g. the cited review says the opposite; the number is the complaint count, not the share")
        submitted = st.form_submit_button("Save label and go to next", icon=":material/save:", type="primary", disabled=locked)
    if submitted:
        if choice is None:
            st.error("Choose a label before saving.")
        else:
            try:
                store.add(cur, choice, expl, annotator, revise=revise, interface="streamlit")
            except DuplicateAnnotationError as e:
                st.warning(str(e), title="Not saved: duplicate")
            except (AnnotationInputError, ReadOnlyModeError) as e:
                st.error(str(e), title="Not saved")
            else:
                st.toast(f"Saved claim #{number[cur]} as {choice}", icon=":material/check:")
                if not only_open and pos < len(ids) - 1:
                    st.session_state["hcv_cid"] = ids[pos + 1]
                st.rerun()

    _export(store, data)


def _export(store: ClaimValidationStore, data: dict) -> None:
    rows = export_rows(data, store.template)
    st.download_button(f"Export completed labels ({len(rows)}) as CSV", export_csv_bytes(rows), file_name="d3_human_claim_labels.csv",
                       mime="text/csv", icon=":material/download:", disabled=not rows, key="hcv_export")
