"""Page 3 — Prompt Experiments: zero-shot vs few-shot vs CoT against the reference labels."""

from __future__ import annotations

import copy
import json
import re

import pandas as pd
import streamlit as st

from annotation.annotation_manager import AnnotationStore, label_provenance, load_sample, provenance_text
from evaluation.prompt_comparison import (
    MOCK_BANNER,
    STATUS_TEXT,
    comparison_table,
    evaluate_run,
    list_runs,
    load_run_predictions,
    run_experiment,
)
from evaluation.d2_human_eval import EXPECTED_LABELS as H_EXPECTED, OUT_DIR as H_DIR, stored_inputs_match, validate_labels
from evaluation.visualizations import confusion_heatmap, grouped_strategy_bar, metric_comparison_bar, per_class_bar, strategy_bar
from prompts.strategies import STRATEGIES
from ui.common import get_client, get_settings, llm_ready, mock_banner, require_llm_note, use_mock
from ui.components import badge, callout, chart, empty_state, esc, kpi, kpi_row, kv_html, md, page_header, panel, section
from ui.theme import SERIES
from utils.data_loader import dataframe_to_inputs
from utils.export import dataframe_to_csv_bytes, records_to_dataframe
from utils.readonly import DISABLED_REASON, is_readonly
from utils.storage import read_json

s = get_settings()
mock = use_mock()
page_header("Prompt Experiments", "Evaluate how prompting strategies affect structured review classification.", is_mock=mock)
mock_banner()

sample = load_sample(s.paths.annotation_sample_csv)
store = AnnotationStore(s.paths.annotations_json)
if sample.empty:
    empty_state("No annotation sample", "Build the 300-review sample on the <b>Human Annotation</b> page first.", "edit_note")
    st.stop()
gold = store.completed(sample["review_id"].tolist())
n_gold = len(gold)
n_required = store.progress(sample["review_id"].tolist())["evaluable"]  # flagged (unreadable) reviews are excluded from evaluation
prov = label_provenance(gold)
runs = list_runs(s, mock)
H_STORE = s.paths.annotations / "human_annotations.json"
try:
    hval = validate_labels(H_STORE, s.paths.annotation_sample_csv) if H_STORE.exists() and not mock else None
except Exception as _e:  # never hide a broken label store
    hval = {"valid": False, "problems": [str(_e)], "n_labels": 0, "annotators": {}}


def _md_to_html(text: str) -> str:
    """Render the backend's small Markdown banners (bold only) as safe HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text.lstrip("> ").strip()))


def _fmt(v, kind: str = "score") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "s":
        return f"{v:.2f}s"
    if kind == "int":
        return f"{v:,.0f}"
    return f"{v:.3f}"


# ---------------------------------------------------------------- readiness
kpi_row([
    (kpi("Human labels (primary reference)", f"{hval['n_labels']} / {H_EXPECTED}",
         f"blind, human-created · {len(hval['annotators'])} annotators · " + ("validated" if hval["valid"] else "NOT valid"),
         tone="positive" if hval["valid"] else "negative", icon_name="edit_note") if hval else
     kpi("Reference labels", f"{n_gold} / {n_required}", f"{prov['human']} human · {prov['ai_assisted']} AI-assisted · {s.annotation_sample_size - n_required} flagged excluded", tone="positive" if n_gold >= n_required else None, icon_name="edit_note")),
    kpi("Experiment runs", len(runs), f"{'mock' if mock else 'live'} runs saved", icon_name="history"),
    kpi("Strategies", len(STRATEGIES), " · ".join(x.label for x in STRATEGIES.values()), icon_name="science"),
    kpi("Primary metric", "Macro F1", "overall sentiment, 4 classes", icon_name="target"),
])
if not hval:
    st.progress(min(1.0, n_gold / max(1, n_required)), text=f"Reference labels: {n_gold} / {n_required} evaluable labelled reviews ({s.annotation_sample_size - n_required} flagged and excluded)")
if prov["ai_assisted"] and not hval:
    callout(
        f"<b>Reference labels are mostly AI-assisted.</b> {esc(provenance_text(prov))}. They are not manual annotations, "
        "so the scores below measure agreement with these reference labels. See <code>meta.annotation_method</code> in annotations.json.",
        "accent",
        tag="Provenance",
    )
if n_gold < n_required and not hval:
    callout(
        f"<b>Real evaluation is pending.</b> The evaluation set needs {n_required} labelled reviews; {n_gold} are done. "
        "Until then any metrics are marked <i>provisional</i>. Label reviews on the <b>Human Annotation</b> page. Model predictions are never written into the reference labels.",
        "warning",
        tag="Pending",
    )

# ---------------------------------------------------------------- run experiment
with st.expander("New experiment / resume a run", icon=":material/science:", expanded=not runs):
    require_llm_note()
    strategies = st.multiselect("Strategies", list(STRATEGIES), default=list(STRATEGIES), format_func=lambda k: STRATEGIES[k].label)
    md("".join(f'<div class="riq-panel-caption"><b style="color:var(--riq-text-2)">{esc(STRATEGIES[k].label)}</b> — {esc(STRATEGIES[k].description)}</div>' for k in strategies))
    c1, c2, c3 = st.columns([2, 1, 2])
    scope = c1.radio("Reviews", ["Annotated reviews only", "First N reviews of the annotation sample"], help="Classification metrics are only computed where reference labels exist.")
    pool = sample[sample["review_id"].isin({g.review_id for g in gold})] if scope.startswith("Annotated") else sample
    n = c2.number_input("Max reviews", 1, max(1, len(pool)), max(1, min(len(pool), 300)), disabled=pool.empty)
    existing = [r["run_id"] for r in runs]
    resume = c3.selectbox("Run", ["(new run)"] + existing, help="Resuming re-uses saved predictions (no repeated API calls).")
    calls = int(n) * len(strategies)
    callout(
        (f"Estimated <b>~{calls} local model calls</b> (+ repairs) on this machine; no rate limit applies, speed depends on your hardware. "
         if s.llm_provider == "ollama" else
         f"Estimated <b>~{calls} LLM calls</b> (+ repairs). At the configured rate limit of {s.llm_requests_per_minute}/min that is about "
         f"<b>{calls / max(1, s.llm_requests_per_minute):.0f} min</b>. ")
        + "For long runs, <code>scripts/run_experiment.py</code> is more robust.",
        "accent",
        tag="Cost",
    )
    confirmed = True
    if not mock:
        kind = "local model calls" if s.llm_provider == "ollama" else "live API calls"
        confirmed = st.checkbox(f"I understand this starts ~{calls} {kind} to {s.llm_provider} / {s.llm_model}.", key="confirm_live_experiment")
    if st.button("Run experiment", type="primary", icon=":material/play_arrow:", disabled=pool.empty or not strategies or not llm_ready() or not confirmed):
        client = get_client()
        if client:
            reviews = dataframe_to_inputs(pool.head(int(n)))
            bar = st.progress(0.0)
            try:
                run_dir = run_experiment(s, client, strategies, reviews, gold, run_id=None if resume == "(new run)" else resume, required_gold=n_required,
                                         progress=lambda name, i, total: bar.progress(i / total, text=f"{STRATEGIES[name].label}: {i}/{total}"))
                st.success(f"Run saved: {run_dir.name}")
                st.session_state["selected_run"] = run_dir.name
                runs = list_runs(s, mock)
            except Exception as exc:
                st.error(f"Experiment stopped: {exc}. Saved predictions are kept; resume the run to continue.")

# ---------------------------------------------------------------- results
if not runs:
    empty_state(f"No {'mock' if mock else 'live'} experiment runs yet", "Start one above. Results, charts and downloads appear here.", "science")
    st.stop()

section("Results", "Primary: stored predictions scored against 300 blind human-created labels. Historical: the same run scored against 299 mostly AI-assisted reference labels.")
ids = [r["run_id"] for r in runs]
default = ids.index(st.session_state["selected_run"]) if st.session_state.get("selected_run") in ids else 0
c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
run_id = c1.selectbox("Experiment run", ids, index=default, format_func=lambda i: next(f"{r['run_id']} · {r['model_name']} · {r['n_reviews']} reviews · {', '.join(r['strategies'])}" for r in runs if r["run_id"] == i))
run = next(r for r in runs if r["run_id"] == run_id)
run_dir = run["path"]
if c2.button("Re-evaluate", icon=":material/refresh:", width="stretch", disabled=is_readonly(),
             help=DISABLED_REASON if is_readonly() else "Recomputes metrics from saved predictions with the current annotations; no LLM calls."):
    evaluate_run(run_dir, gold, n_required)
metrics = read_json(run_dir / "metrics.json")
if not metrics:
    if is_readonly():  # computing metrics would write metrics.json
        callout(f"<b>This run has no saved metrics.json.</b> {esc(DISABLED_REASON)}", "warning", tag="Demo")
        st.stop()
    metrics = evaluate_run(run_dir, gold, n_required)
config = read_json(run_dir / "config.json")
status = metrics["status"]
if metrics["is_mock"]:
    callout(_md_to_html(MOCK_BANNER), "negative", tag="Not a result")
_flagged_excluded = len(sample) - n_required  # flagged (unreadable) sample reviews, excluded from the evaluation
hm = None if metrics["is_mock"] else read_json(run_dir / H_DIR / "metrics.json")
VIEW_H, VIEW_A = "Human-labelled · 300 reviews (primary)", "AI-assisted reference · historical"
view = VIEW_A
if hm or (hval and not metrics["is_mock"]):
    view = st.segmented_control("Reference labels", [VIEW_H, VIEW_A], default=VIEW_H, key="pc_reference_view") or VIEW_H
if view == VIEW_H and not hm:
    callout("<b>No human-labelled evaluation for this run yet.</b> " + (
        "Run <code>python scripts/d2_human_eval.py evaluate</code> (no LLM calls) or use the button below." if hval and hval["valid"] else
        "The human label set is not valid: " + esc("; ".join(hval["problems"][:5]) if hval else "no label store")), "warning", tag="Pending")
    if hval and hval["valid"] and st.button("Evaluate against the 300 human labels", icon=":material/rule:", disabled=is_readonly(),
                                            help=DISABLED_REASON if is_readonly() else "Re-scores stored predictions; no model is called."):
        from evaluation.d2_human_eval import D2ValidationError, evaluate as _h_evaluate
        try:
            _h_evaluate(run_dir, H_STORE, s.paths.annotation_sample_csv, legacy_path=s.paths.annotations_json)
            st.rerun()
        except D2ValidationError as _e:
            st.error("Not evaluated: " + "; ".join(_e.problems[:10]))
    st.stop()
if view == VIEW_H:
    _cur = stored_inputs_match(run_dir, H_STORE, s.paths.annotation_sample_csv)
    _hv = hm["reference_validation"]
    _ok = hm["status"] == "complete" and hm["independent_check_passed"] and _cur["current"]
    _sup = hm["reviews_from_supplementary_predictions"]
    callout(
        f"<b>Primary D2 evaluation — {hm['n_evaluated']} blind human-created labels.</b> "
        f"Annotators: {esc(', '.join(f'{a} ({n})' for a, n in _hv['annotators'].items()))}; one label per review, not verified by a second person. "
        f"Re-evaluation of <b>stored predictions</b> (no fresh inference): {hm['n_evaluated'] - len(_sup)} from the run "
        f"+ {len(_sup)} supplementary ({esc(', '.join(_sup))}); identical review set for every strategy; "
        f"failed extractions count as wrong · independent cross-check {'passed' if hm['independent_check_passed'] else 'FAILED'} · evaluated {esc(hm['evaluated_at'])}"
        + ("" if _cur["current"] else f" · <b>out of date: changed inputs {esc(', '.join(_cur['changed']))}; re-run the evaluation</b>"),
        "positive" if _ok else "warning", tag="Primary · human-labelled" if _ok else "Primary · check")
    _q = hm.get("label_quality_signals") or {}
    if _q.get("positive_label_on_1_2_star"):
        callout(f"<b>Label quality.</b> {_q['positive_label_on_1_2_star']} reviews rated 1–2 stars carry the human label <i>positive</i>; "
                f"{_q.get('human_positive_few_shot_negative', '–')} human-<i>positive</i> reviews were predicted <i>negative</i> by few-shot "
                f"({_q.get('human_positive_few_shot_negative_on_1_2_star', '–')} of them 1–2 stars). Some of these labels look wrong, so the scores partly "
                "measure label noise. No label was changed; see <code>reports/D2_HUMAN_LABELLED_REPORT.md</code>.", "warning", tag="Caution")
    _h = hm.get("historical_label_agreement") or {}
    if _h:
        md(f'<div class="riq-panel-caption">The human labels match the historical (mostly AI-assisted) reference labels on {_h["same_overall_sentiment"]} '
           f'of {_h["n"]} reviews, so these scores are not comparable with the historical evaluation (switch the view above to see it).</div>')
if view == VIEW_A:
    callout(
        f"<b>Historical evaluation — original experiment vs AI-assisted reference labels.</b> {metrics['n_gold_available']} reviews"
        + (f" (annotation sample {len(sample)}; {_flagged_excluded} flagged review(s) excluded before the run)" if _flagged_excluded > 0 else "")
        + f" · evaluated {esc(metrics['evaluated_at'])}. <b>Status:</b> {esc(STATUS_TEXT[status])}. "
        "The strategy cards and all tabs below show this evaluation. It is kept unchanged as a historical experiment"
        + ("; the primary D2 result is the human-labelled evaluation." if hm or hval else "."),
        "neutral" if (hm or hval) else ("positive" if status == "complete" else "warning"),
        tag="Historical" if (hm or hval) else {"complete": "Complete", "provisional": "Provisional", "incomplete": "Incomplete", "mock_demo": "Mock", "awaiting_annotations": "Pending"}.get(status, status),
    )

# ---------------------------------------------------------------- D2 against the approved reference dataset
if not metrics["is_mock"] and view == VIEW_A:
    from annotation.reference_adjudication import ReferenceAdjudicationStore
    from evaluation.reference_evaluation import OUT_DIR as _REF_DIR, STATUS_TEXT as _REF_STATUS, evaluate_reference

    _rstore = ReferenceAdjudicationStore.default(s)
    try:
        _rprog = _rstore.progress()
    except Exception:
        _rprog = None
    _refm = read_json(run_dir / _REF_DIR / "metrics.json")
    with panel("Historical sensitivity analysis — approved-reference evaluation",
               "A separate evaluation on the approved (mostly AI-assisted) reference annotations (no LLM calls). It is not the primary D2 result."):
        if _rprog:
            _ap = _rprog["accepted_by_original_provenance"]
            md(f'<div class="riq-panel-caption">Reference dataset: {_rprog["accepted"]} of {_rprog["required"]} accepted '
               f'({"ready" if _rprog["ready"] else "not ready — approve them on the Human Annotation page"}); original provenance of the accepted '
               f'labels: {_ap["human_created"]} human-created, {_ap["ai_assisted"]} AI-assisted. These are approved reference labels, '
               'not independent human ground truth.</div>')
        if st.button("Evaluate against approved references", icon=":material/rule:", disabled=is_readonly() or not _rprog or not _rprog["accepted"],
                     help=DISABLED_REASON if is_readonly() else "Writes <run>/approved_reference/; the run's metrics.json is not changed."):
            _refm = evaluate_reference(run_dir, _rstore.accepted_gold(), progress=_rprog, provenance=_rprog["accepted_by_original_provenance"])
        if _refm:
            _stale = bool(_rprog) and (_refm["reference_progress"]["accepted"] != _rprog["accepted"])
            _done = _refm["status"] == "complete" and _refm.get("independent_check_passed") and not _stale
            _extra = sorted(set(_refm["evaluated_review_ids"]) - set(config.get("review_ids", [])))
            callout(f"<b>{esc(_REF_STATUS[_refm['status']])}.</b> {_refm['n_evaluated_common']} reviews, identical for every strategy · "
                    f"independent cross-check {'passed' if _refm.get('independent_check_passed') else 'not passed'} · evaluated {esc(_refm['evaluated_at'])}"
                    + (" · <b>out of date: the approvals changed since this evaluation</b>" if _stale else ""),
                    "positive" if _done else "warning", tag="Sensitivity · complete" if _done else "Sensitivity")
            if _extra or _refm["n_evaluated_common"] != metrics["n_gold_available"]:
                _dm = {n: _refm["strategies"][n]["classification"]["overall_sentiment"]["macro_f1"]
                       - metrics["strategies"][n]["classification"]["overall_sentiment"]["macro_f1"]
                       for n in _refm["strategies"] if n in metrics["strategies"] and _refm["n_evaluated_common"]}
                md(f'<div class="riq-panel-caption"><b>Not the same evaluation as the primary result.</b> Denominator '
                   f'{_refm["n_evaluated_common"]} here vs {metrics["n_gold_available"]} in the primary evaluation'
                   + (f'; added review(s): {esc(", ".join(_extra))} (excluded from the original run, approved and predicted later)' if _extra else "")
                   + ('. Macro-F1 difference vs primary: ' + esc(", ".join(f"{STRATEGIES[n].label if n in STRATEGIES else n} {v:+.4f}" for n, v in _dm.items())) if _dm else "")
                   + '. See the D2 report, §14.</div>')
            if _refm["n_evaluated_common"]:
                st.dataframe(pd.DataFrame([{
                    "strategy": STRATEGIES[n].label if n in STRATEGIES else n, "n": e["classification"]["overall_sentiment"]["n"],
                    "valid": e["valid_predictions"], "invalid": e["invalid_predictions"], "failure rate": e["failure_rate"],
                    "accuracy": e["classification"]["overall_sentiment"]["accuracy"],
                    "macro P": e["classification"]["overall_sentiment"]["macro_precision"],
                    "macro R": e["classification"]["overall_sentiment"]["macro_recall"],
                    "macro F1": e["classification"]["overall_sentiment"]["macro_f1"],
                    "weighted F1": e["classification"]["overall_sentiment"]["weighted_f1"],
                    "cross-check": "match" if _refm["independent_check"].get(n, {}).get("matches") else "MISMATCH",
                } for n, e in _refm["strategies"].items()]), width="stretch", hide_index=True)
            if any(_refm["needs_inference"].values()):
                md('<div class="riq-panel-caption">Accepted references without a prediction: ' + esc("; ".join(
                    f"{n}: {', '.join(v)}" for n, v in _refm["needs_inference"].items() if v)) +
                   '. Predict them with <code>scripts/reference_annotations.py predict-missing</code> (local LLM, separate supplementary files).</div>')
        else:
            md('<div class="riq-panel-caption">Not evaluated yet. Approve the reference annotations on the Human Annotation page, then evaluate here '
               '(or run <code>scripts/reference_annotations.py evaluate --run &lt;run&gt;</code>).</div>')

# ---------------------------------------------------------------- metrics of the selected reference set
ml_ok = True
if view == VIEW_H:
    mv = copy.deepcopy(metrics)  # operational metrics (validity, latency, tokens) stay those of the run itself
    ml_ok = hm["multilabel_validity"]["valid"]
    for _n, _e in hm["strategies"].items():
        _cl = _e["classification"]
        mv["strategies"][_n]["classification"] = _cl if ml_ok else {k: _cl[k] for k in ("n_evaluated", "n_failed_extractions", "overall_sentiment")}
    mv["n_gold_available"] = hm["n_evaluated"]
else:
    mv = metrics
table = comparison_table(mv)
has_cls = mv["n_gold_available"] > 0
names = list(mv["strategies"])
label_of = {k: (STRATEGIES[k].label if k in STRATEGIES else k) for k in names}
# Strategies with at least one scored prediction (an interrupted run may not have reached every strategy yet).
scored = [k for k in names if mv["strategies"][k].get("classification", {}).get("overall_sentiment", {}).get("n")]
unscored = [label_of[k] for k in names if k not in scored]

# ---------------------------------------------------------------- strategy cards
best = table["sentiment_macro_f1"].astype(float).idxmax() if has_cls and table["sentiment_macro_f1"].notna().any() else None
cards = []
for i, row in table.iterrows():
    primary = _fmt(row["sentiment_macro_f1"]) if has_cls else "Pending"
    head_badge = badge("Best macro F1", "accent") if best == i else (badge("mock", "warning") if metrics["is_mock"] else "")
    cards.append(
        f'<div class="riq-strategy{" riq-best" if best == i else ""}" style="--riq-strategy-color:{SERIES[i % len(SERIES)]}">'
        f'<div class="riq-strategy-head"><span class="riq-strategy-name">{esc(row["strategy"])}</span>{head_badge}</div>'
        f'<div class="riq-strategy-primary{" riq-muted" if primary == "Pending" else ""}">{esc(primary)}</div>'
        f'<div class="riq-strategy-primary-label">Macro F1 · overall sentiment</div>'
        + kv_html([
            ("Accuracy", _fmt(row["sentiment_accuracy"])),
            ("Precision (macro)", _fmt(row["sentiment_macro_precision"])),
            ("Recall (macro)", _fmt(row["sentiment_macro_recall"])),
            ("Weighted F1", _fmt(row["sentiment_weighted_f1"])),
            ("Validation success", _fmt(row["validity_rate"], "pct")),
            ("Mean latency", _fmt(row["mean_latency_s"], "s")),
            ("Tokens in / out", f'{_fmt(row["mean_input_tokens"], "int")} / {_fmt(row["mean_output_tokens"], "int")}'),
        ])
        + "</div>"
    )
md(f'<div class="riq-strategy-grid">{"".join(cards)}</div>')
if has_cls and unscored:
    md(f'<div class="riq-panel-caption">No predictions yet for: {esc(", ".join(unscored))}. Resume the run to complete it.</div>')
if not has_cls:
    md('<div class="riq-panel-caption">Classification metrics appear once reference annotations overlap this run. Operational metrics (validity, latency, tokens) are shown now.</div>')

tab_perf, tab_cm, tab_pc, tab_eff, tab_run = st.tabs(["Performance", "Confusion matrices", "Per-class", "Efficiency", "Run details"])

with tab_perf:
    if not has_cls:
        empty_state("Classification metrics pending", "These need reference labels for the reviews in this run. Annotate on <b>Human Annotation</b>, then click <b>Re-evaluate</b>.", "hourglass_empty")
    else:
        a, b = st.columns([2, 3], gap="medium")
        _ref = "300 human labels" if view == VIEW_H else "historical AI-assisted reference"
        with a, panel("Macro F1 by strategy", f"Overall sentiment · primary metric · {_ref}"):
            chart(strategy_bar(table, "sentiment_macro_f1", y_title="macro F1", fmt=".3f"), key="pc_macro")
        with b, panel("All task metrics", "Sentiment, complaints (multi-label) and aspects" if ml_ok else "Overall sentiment (multi-label metrics not valid for this reference)"):
            chart(metric_comparison_bar(table, ["sentiment_accuracy", "sentiment_macro_precision", "sentiment_macro_recall", "sentiment_macro_f1", "sentiment_weighted_f1"]
                                        + (["complaint_micro_f1", "complaint_macro_f1", "aspect_detection_micro_f1", "aspect_sentiment_macro_f1", "aspect_joint_micro_f1"] if ml_ok else [])), key="pc_all")
        with panel("Overall sentiment (4 classes)", _ref):
            st.dataframe(table[["strategy", "n_gold", "sentiment_accuracy", "sentiment_macro_precision", "sentiment_macro_recall", "sentiment_macro_f1", "sentiment_weighted_f1"]], width="stretch", hide_index=True)
        with panel("Complaints (multi-label) and aspects", _ref):
            if ml_ok:
                st.dataframe(table[["strategy", "complaint_micro_f1", "complaint_macro_f1", "complaint_hamming_loss", "aspect_detection_micro_f1", "aspect_sentiment_macro_f1", "aspect_joint_micro_f1"]], width="stretch", hide_index=True)
            else:
                md(f'<div class="riq-panel-caption"><b>Not valid for the human labels — not reported.</b> {esc(hm["multilabel_validity"]["reason"])} '
                   'Switch to the historical view for multi-label results against the AI-assisted reference labels.</div>')

with tab_cm:
    if not has_cls:
        empty_state("Confusion matrices need reference labels", "Rows are reference labels, columns are predictions.", "grid_on")
    else:
        cols = st.columns(max(len(scored), 1), gap="medium")
        for col, name in zip(cols, scored):
            entry = mv["strategies"][name]
            with col, panel(label_of[name], "Overall sentiment · rows = " + ("human label" if view == VIEW_H else "reference label")):
                cm = entry["classification"]["overall_sentiment"]
                chart(confusion_heatmap(cm["confusion_matrix"], cm["labels"]), key=f"cm_{name}")
                asent = entry["classification"].get("aspect_sentiment", {})
                if asent.get("n"):
                    md('<div class="riq-panel-caption">Aspect sentiment (matched categories)</div>')
                    chart(confusion_heatmap(asent["confusion_matrix"], asent["labels"]), key=f"cma_{name}")

with tab_pc:
    if not has_cls or not scored:
        empty_state("Per-class scores need reference labels", "", "bar_chart")
    else:
        with panel("Per-class F1 · overall sentiment"):
            chart(per_class_bar({label_of[n_]: mv["strategies"][n_]["classification"]["overall_sentiment"]["per_class"] for n_ in scored}), key="pc_perclass")
        strat = st.selectbox("Per-class detail for", scored, format_func=lambda k: label_of[k])
        cl = mv["strategies"][strat].get("classification", {})
        a, b = st.columns(2, gap="medium")
        with a, panel("Overall sentiment per class"):
            st.dataframe(pd.DataFrame(cl["overall_sentiment"]["per_class"]).T, width="stretch")
        with b, panel("Complaint categories per label", "Labels with support > 0"):
            if "complaints" in cl:
                st.dataframe(pd.DataFrame(cl["complaints"]["per_label"]).T.query("support > 0"), width="stretch")
            else:
                md('<div class="riq-panel-caption">Not valid for the human labels (complaint categories were almost never annotated).</div>')

with tab_eff:
    a, b = st.columns(2, gap="medium")
    with a, panel("Mean latency per review", "seconds"):
        chart(strategy_bar(table, "mean_latency_s", y_title="seconds"), key="pc_latency")
    with b, panel("Mean tokens per review", "input and output"):
        chart(grouped_strategy_bar(table, ["mean_input_tokens", "mean_output_tokens"], y_title="tokens"), key="pc_tokens")
    with panel("Validation and repair"):
        st.dataframe(
            table[["strategy", "validity_rate", "first_attempt_valid_rate", "needed_llm_repair", "mean_latency_s", "mean_input_tokens", "mean_output_tokens"]],
            width="stretch",
            hide_index=True,
            column_config={
                "validity_rate": st.column_config.ProgressColumn("validity rate", min_value=0.0, max_value=1.0, format="%.3f"),
                "first_attempt_valid_rate": st.column_config.ProgressColumn("first-attempt valid", min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )
    if metrics["is_mock"]:
        md('<div class="riq-panel-caption">Mock token counts are rough character-based estimates and latency is 0. These are not measurements.</div>')

with tab_run:
    a, b, c = st.columns(3, gap="medium")
    with a, panel("Configuration"):
        md(kv_html([
            ("Run ID", config["run_id"]),
            ("Created", config.get("created_at", "")),
            ("Prompt version", config.get("prompt_version", "")),
            ("Temperature", config.get("temperature", "")),
            ("Repair attempts", config.get("max_repair_attempts", "")),
            ("Evidence grounding", "strict" if config.get("strict_evidence_grounding") else "lenient"),
            ("Git commit", str(config.get("git_commit") or "—")[:10]),
        ]))
    with b, panel("Model"):
        md(kv_html([
            ("Provider", config.get("provider", "")),
            ("Model", config.get("model_name", "")),
            ("Mode", "mock fixture" if metrics["is_mock"] else "live LLM"),
            ("Strategies", len(config["strategies"])),
        ]))
    with c, panel("Dataset"):
        md(kv_html([
            ("Reviews in run", f"{len(config['review_ids']):,}"),
            ("Reference labels in run (historical)", metrics["n_gold_available"]),
            ("Human labels evaluated (primary)", hm["n_evaluated"] if hm else "—"),
            ("Required labels", metrics["required_gold"]),
            ("Annotation sample", f"{len(sample):,} reviews"),
            ("Sampling", f"stratified by rating · seed {s.random_seed}"),
        ]))

    with panel("Experiment history", f"All {'mock' if mock else 'live'} runs"):
        hist = pd.DataFrame([{"run": r["run_id"], "created": r["created_at"], "model": r["model_name"], "reviews": r["n_reviews"], "strategies": ", ".join(r["strategies"])} for r in runs])
        st.dataframe(hist, width="stretch", hide_index=True)

    report = (run_dir / "report.md").read_text(encoding="utf-8") if (run_dir / "report.md").exists() else ""
    section("Downloads")
    d = st.columns(3)
    d[0].download_button("Historical report (Markdown)", report, f"{run_id}_report.md", "text/markdown", icon=":material/description:", width="stretch")
    d[1].download_button("Historical metrics (JSON)", json.dumps(metrics, indent=2), f"{run_id}_metrics.json", "application/json", icon=":material/data_object:", width="stretch")
    d[2].download_button("Comparison table, current view (CSV)", dataframe_to_csv_bytes(table), f"{run_id}_comparison.csv", "text/csv", icon=":material/table:", width="stretch")
    d = st.columns(max(1, len(config["strategies"])))
    for col, name in zip(d, config["strategies"]):
        recs, _ = load_run_predictions(run_dir, name)
        col.download_button(f"Predictions — {name} (CSV)", dataframe_to_csv_bytes(records_to_dataframe(recs)), f"{run_id}_{name}_predictions.csv", "text/csv", icon=":material/download:", width="stretch")
    if hm:
        _hd = run_dir / H_DIR
        d = st.columns(3)
        d[0].download_button("Human-labelled report (Markdown)", (_hd / "report.md").read_text(encoding="utf-8") if (_hd / "report.md").exists() else "",
                             f"{run_id}_human_labelled_report.md", "text/markdown", icon=":material/description:", width="stretch", key="dl_h_report")
        d[1].download_button("Human-labelled metrics (JSON)", json.dumps(hm, indent=2), f"{run_id}_human_labelled_metrics.json", "application/json",
                             icon=":material/data_object:", width="stretch", key="dl_h_metrics")
        d[2].download_button("300 human labels (CSV)", (_hd / "reference_labels.csv").read_bytes() if (_hd / "reference_labels.csv").exists() else b"",
                             f"{run_id}_human_reference_labels.csv", "text/csv", icon=":material/table:", width="stretch", key="dl_h_labels")
    with st.expander("Preview report", icon=":material/article:"):
        st.markdown(report)
    with st.expander("Raw run configuration", icon=":material/settings:"):
        st.json({k: v for k, v in config.items() if k != "review_ids"}, expanded=False)
