"""Page 5 — Summary Evaluation with LLM-as-a-Judge."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation.summary_evaluation import (
    JudgementStore,
    JudgeProviderStopped,
    available_judge_versions,
    export_inspection_sample,
    get_judge_client,
    judge_summaries,
    judgements_table,
)
from evaluation.d3_audit import audit_judgement, audit_summary
from evaluation.visualizations import judge_scores_bar, judge_scores_per_summary
from utils.storage import read_json
from pipeline.extraction import ExtractionStore
from pipeline.summarization import SummaryStore
from prompts.judge import CURRENT_JUDGE_VERSION, JUDGE_PROMPT_VERSION, RUBRIC, V2_RULES
from schemas.evaluation_schema import ProductSummary
from ui.common import get_settings, judge_status, mock_banner, mode_dir, pretty, use_mock
from utils.readonly import DISABLED_REASON, is_readonly
from ui.components import badge, callout, chart, empty_state, esc, kv_html, md, page_header, panel, score_html, section
from utils.export import dataframe_to_csv_bytes
from llm.base import LLMUnavailableError

DIMENSIONS = [
    ("faithfulness", "Faithfulness", "claims supported by the source"),
    ("coverage", "Coverage", "main themes represented"),
    ("consistency", "Consistency", "agrees with the statistics"),
    ("hallucination", "Hallucination assessment", "5 = no unsupported claims"),
]

s = get_settings()
page_header("Summary Evaluation", "Measure the reliability and faithfulness of AI-generated product summaries.", is_mock=use_mock())
mock_banner()
callout(
    "<b>LLM-as-a-Judge is an automated assessment, not objective human evaluation.</b> The judge can be wrong or biased "
    "(for example towards fluent text, or towards its own model family). Use the detailed view and the manual inspection sample to audit it.",
    "accent",
    tag="Automated",
)

# ---------------------------------------------------------------- human claim validation (single annotator, blind)
# While the human labelling of the judge v4 claims is incomplete, the page opens in the blind validation view and stops
# there, so no judge verdict (v1-v4) is visible next to the claims being labelled.
if not use_mock():
    from ui.human_claim_validation import log_results_opened, render as _render_hcv, validation_state  # noqa: E402

    _hcv = validation_state()
    if _hcv["available"]:
        _HCV_VIEWS = ["Human claim validation (blind)", "Evaluation results"]
        _hcv_default = _HCV_VIEWS[1] if _hcv["complete"] else _HCV_VIEWS[0]
        _hcv_view = st.segmented_control("Page view", _HCV_VIEWS, default=_hcv_default, key="se_page_mode") or _hcv_default
        if _hcv_view == _HCV_VIEWS[0]:
            _render_hcv()
            st.stop()
        if not _hcv["complete"]:
            st.warning("Human claim validation is not finished. This view shows judge verdicts; reading them before you finish "
                       "labelling biases the validation. Opening it is recorded in the label store.",
                       title=f"Judge results visible ({_hcv['completed']}/{_hcv['total']} claims labelled)")
            log_results_opened()

sstore = SummaryStore.for_mode(s, use_mock())
VERSION_LABEL = {"v2": "v2 — fixed judge (flat schema, constrained decoding)", "v1": "v1 — original run (preserved, read-only)"}
versions = list(dict.fromkeys(available_judge_versions(s, use_mock()) + [CURRENT_JUDGE_VERSION]))
version = st.radio("Judge run", versions, format_func=lambda v: VERSION_LABEL.get(v, v), horizontal=True, key="se_version",
                   help="v1 and v2 results are stored separately and never mixed. New judging always uses the current version.")
jstore = JudgementStore.for_mode(s, use_mock(), version)
CAN_JUDGE_VERSION = version == CURRENT_JUDGE_VERSION
latest = sstore.latest()
summaries: dict[str, ProductSummary] = {x.summary_id: x for x in latest.values()}
if not summaries:
    empty_state("No summaries in this mode yet", "Generate them on the <b>Product Intelligence</b> page, then judge them here.", "summarize")
    st.stop()
judged = {sid: j for sid, j in jstore.latest_by_summary().items() if sid in summaries}
# Judge calls whose output stayed invalid after the bounded repairs. They are final for this version (not silently
# re-sampled) and are shown as failures, never as "not judged yet".
failed = {k: v for k, v in jstore.open_failures().items() if k in summaries}


def _of(found, expected) -> str:
    """'2 of 3 (price value, performance)' from the deterministic-check lists."""
    if isinstance(found, list) and isinstance(expected, list):
        return f"{len(found)} of {len(expected)}" + (f" ({', '.join(pretty(x) for x in found)})" if found else "")
    return f"{found} of {expected}"


def records_for(summary: ProductSummary):
    path = mode_dir() / "extractions" / f"{summary.source_extraction}.jsonl"
    return ExtractionStore(path).records if path.exists() else []


def run_judge(targets: list[ProductSummary], retry_failed: bool = False) -> None:
    """Resumable v2 judging: stored judgements are reused, failures are stored with their raw outputs."""
    try:
        client = get_judge_client(s, use_mock=use_mock(), version=CURRENT_JUDGE_VERSION)
    except LLMUnavailableError as exc:
        st.error(str(exc))
        return
    by_source: dict[str, list] = {}
    for summ in targets:
        by_source.setdefault(summ.source_extraction, []).append(summ)
    bar = st.progress(0.0)
    done = 0
    try:
        for source_targets in by_source.values():
            recs = records_for(source_targets[0])
            if not recs:
                st.error(f"Source extractions for {source_targets[0].source_extraction} not found; cannot judge.")
                continue
            res = judge_summaries(client, source_targets, recs, s, jstore, version=CURRENT_JUDGE_VERSION, retry_failed=retry_failed, source="app")
            for sid in res.failed:
                st.error(f"Judge failed for {summaries[sid].product_id}; the failure and raw outputs are stored.")
            done += len(source_targets)
            bar.progress(done / len(targets), text=f"Judged {done}/{len(targets)}")
    except JudgeProviderStopped as exc:
        st.error(f"Stopped: {exc}. Everything judged so far is stored.")
        return
    st.rerun()


# ---------------------------------------------------------------- actions
judge_ready, _, judge_reason = (True, "", "") if use_mock() else judge_status(s)
if not CAN_JUDGE_VERSION:
    judge_ready = False
if is_readonly():
    judge_ready = False
    callout(f"<b>Judging is disabled.</b> {esc(DISABLED_REASON)} Saved judgements and recorded judge failures are shown unchanged.", "warning", tag="Demo")
elif not CAN_JUDGE_VERSION:
    callout(f"<b>{esc(VERSION_LABEL.get(version, version))}.</b> These stored results are shown unchanged; new judging uses {esc(CURRENT_JUDGE_VERSION)}.", "neutral", tag="Preserved")
elif not judge_ready:
    callout(f"<b>The judge model is not available.</b> {esc(judge_reason)} Judging is disabled; existing judgements are still shown.", "accent", tag="Judge")
unjudged = [x for sid, x in summaries.items() if sid not in judged and sid not in failed]
n_failed = sum(sid in failed for sid in summaries)
c1, c2 = st.columns([3, 2], vertical_alignment="center")
same_model = s.effective_judge_model == s.llm_model and not use_mock()
c1.markdown(
    f'<div class="riq-panel-caption" style="margin:0">Judge model <b style="color:var(--riq-text)">{esc(s.effective_judge_provider)} / {esc(s.effective_judge_model)}</b>'
    + (" · same as the summariser; set JUDGE_PROVIDER / JUDGE_MODEL to reduce self-preference bias" if same_model else "")
    + f" · <b>{esc(version)}</b>: {sum(sid in judged for sid in summaries)} of {len(summaries)} summaries judged"
    + (f' · <b style="color:var(--riq-negative, #f87171)">{n_failed} judge call(s) failed</b> (invalid judge output after repairs; no score)' if n_failed else "")
    + f" · {len(unjudged)} pending"
    + "</div>",
    unsafe_allow_html=True,
)
if c2.button(f"Judge all unjudged summaries ({len(unjudged)})", type="primary", icon=":material/gavel:", disabled=not unjudged or not judge_ready, width="stretch"):
    run_judge(unjudged)
with st.expander("Judge rubric", icon=":material/rule:"):
    st.code(RUBRIC + ("\n\n" + V2_RULES if version != JUDGE_PROMPT_VERSION else ""), language="text")

# v1 vs v2 status per summary (both stores read-only here)
_v1 = JudgementStore.for_mode(s, use_mock(), JUDGE_PROMPT_VERSION)
_v2 = JudgementStore.for_mode(s, use_mock(), CURRENT_JUDGE_VERSION)
if _v1.path.exists() and (_v2.path.exists() or _v2.failures_path.exists()):
    def _st(store, sid):
        if sid in store.latest_by_summary():
            v = store.latest_by_summary()[sid].verdict
            return f"ok F{v.faithfulness.score} C{v.coverage.score} K{v.consistency.score} H{v.hallucination.score}"
        return "failed" if sid in store.open_failures() else "pending"
    with st.expander("v1 vs v2 status per summary", icon=":material/compare:"):
        st.dataframe(pd.DataFrame([{"product": (x.product_title or x.product_id)[:60], "product_id": x.product_id, "v1": _st(_v1, sid), "v2": _st(_v2, sid)}
                                   for sid, x in summaries.items()]), width="stretch", hide_index=True)

# ---------------------------------------------------------------- overview
judged_list = [judged[sid] for sid in summaries if sid in judged]
rows = judgements_table(judged_list, summaries)
if not rows:
    empty_state("No judgements yet", "Click <b>Judge all unjudged summaries</b> to score them on faithfulness, coverage, consistency and hallucination.", "gavel")
else:
    table = pd.DataFrame(rows)
    cards = []
    for key, label, cap in DIMENSIONS:
        mean = table[key].mean()
        cards.append(
            f'<div class="riq-kpi"><div class="riq-kpi-label">{esc(label)}</div>'
            f'<div class="riq-kpi-value">{mean:.2f}<span style="font-size:14px;color:var(--riq-muted);font-weight:500"> / 5</span></div>'
            f'{score_html(round(mean))}<div class="riq-kpi-caption">{esc(cap)} · mean of {len(table)}</div></div>'
        )
    md(f'<div class="riq-kpi-grid">{"".join(cards)}</div>')
    if any(r["is_mock"] for r in rows):
        callout("<b>MOCK judge verdicts</b> — placeholder values, not an LLM assessment.", "warning", tag="Mock")

    a, b = st.columns([2, 3], gap="medium")
    with a, panel("Mean score per dimension", "1 = poor · 5 = excellent"):
        chart(judge_scores_bar(rows), key="se_mean")
    with b, panel("Scores per summary"):
        chart(judge_scores_per_summary(rows), key="se_heat")

    section("Evaluation history", "Latest judgement per summary")
    hist = table[["product_title", "product_id", "faithfulness", "coverage", "consistency", "hallucination", "mean_score", "unsupported_claims", "judge_model"]].copy()
    st.dataframe(
        hist,
        width="stretch",
        hide_index=True,
        column_config={
            "product_title": st.column_config.TextColumn("product", width="large"),
            "mean_score": st.column_config.ProgressColumn("mean", min_value=1.0, max_value=5.0, format="%.2f"),
        },
    )
    d = st.columns(3)
    d[0].download_button("Judgements (CSV)", dataframe_to_csv_bytes(table), "judgements.csv", "text/csv", icon=":material/table:", width="stretch")
    d[1].download_button("Judgements (JSON)", json.dumps([j.model_dump(mode="json") for j in judged_list], indent=2), "judgements.json", "application/json", icon=":material/data_object:", width="stretch")
    d[2].download_button("Manual inspection sample (MD)", export_inspection_sample(judged_list, summaries, n=3), "judge_inspection_sample.md", "text/markdown", icon=":material/fact_check:", width="stretch")

# ---------------------------------------------------------------- evidence verification & negative control
@st.cache_data(show_spinner=False)
def _evidence_audit(summary_json: tuple, source_path: str) -> list[dict]:
    from schemas.evaluation_schema import ProductSummary as _PS

    recs = ExtractionStore(Path(source_path)).records if Path(source_path).exists() else []
    return [audit_summary(_PS.model_validate_json(x), recs) for x in summary_json]


_src_paths = {x.source_extraction: str(mode_dir() / "extractions" / f"{x.source_extraction}.jsonl") for x in summaries.values()}
audits: dict[str, dict] = {}
for _src, _path in _src_paths.items():
    _group = tuple(x.model_dump_json() for x in summaries.values() if x.source_extraction == _src)
    for a in _evidence_audit(_group, _path):
        audits[a["summary_id"]] = a
section("Evidence verification", "Deterministic, no LLM: cited ids, re-computed statistics, percentages, claim traceability")
if audits:
    n_ok = sum(a["ids_and_statistics_verified"] for a in audits.values())
    pts = [p for a in audits.values() for p in a["points"]]
    md(kv_html([
        ("Summaries with valid ids + matching statistics + only sourced percentages", f"{n_ok} of {len(audits)}"),
        ("Summary points traceable / partial / weak", f"{sum(p['status'] == 'traceable' for p in pts)} / {sum(p['status'] == 'partial' for p in pts)} / {sum(p['status'] == 'weak' for p in pts)}"),
        ("Cited ids unknown or from another product", str(sum(len(a["unknown_ids"]) + len(a["wrong_product_ids"]) for a in audits.values()))),
    ]))
    issues = [{"product_id": a["product_id"], "issue": i} for a in audits.values() for i in a["issues"]]
    if issues:
        st.dataframe(pd.DataFrame(issues), width="stretch", hide_index=True)
    _corr = []
    for _src in _src_paths:
        _cp = mode_dir() / "d3" / "corrected_summaries" / _src / "corrected_summaries.jsonl"
        if _cp.exists():
            _corr += [json.loads(x) for x in _cp.read_text(encoding="utf-8").splitlines() if x.strip()]
    if _corr:
        with st.expander(f"Separately labelled corrections ({len(_corr)}) — originals unchanged", icon=":material/edit_document:"):
            md(f'<div class="riq-panel-caption">{esc(_corr[0]["provenance"])}. The evaluated originals, their judgements and the negative control '
               'all use the original summaries. <b>v1/v2 scores and the v2 negative control belong to the original summaries.</b> The corrected '
               'copies were judged only in the separate judge v3 run, under their own ids (see the Judge v3 section); no score is transferred.</div>')
            st.dataframe(pd.DataFrame([{"product_id": c["product_id"], "corrected_summary_id": c["corrected_summary_id"], "old": r["old"], "new": r["new"],
                                        "source of new wording": r["source"]} for c in _corr for r in c["replacements"]]), width="stretch", hide_index=True)

section("Negative control", "Does the judge notice deliberately planted unsupported claims?")
_nc_v2 = None
_nc_v1 = None
for _src in _src_paths:
    _nc_v2 = _nc_v2 or read_json(mode_dir() / "d3" / "judge_negative_control_v2" / _src / "summary.json")
    _nc_v1 = _nc_v1 or read_json(mode_dir() / "d3" / "judge_sanity" / _src / "summary.json")
if version != JUDGE_PROMPT_VERSION:
    if _nc_v2 and _nc_v2.get("by_kind"):
        def _nc_row(name: str, v: dict) -> dict:
            ci = v.get("detection_wilson95")
            return {"planted defect": name, "judged": v.get("corrupted_judged"), "detected": v.get("planted_claim_detected"),
                    "missed": v.get("missed", (v.get("corrupted_judged") or 0) - (v.get("planted_claim_detected") or 0)),
                    "detection rate": v.get("detection_rate"), "95% CI (Wilson)": f"{ci[0]:.0%}–{ci[1]:.0%}" if ci else "–",
                    "false-negative rate": v.get("false_negative_rate"), "judge failed": v.get("corrupted_judge_failed"),
                    "faithfulness lower": v.get("faithfulness_lower"), "hallucination lower": v.get("hallucination_lower"),
                    "hallucination higher": v.get("hallucination_higher", "–"), "mean score lower": v.get("mean_score_lower")}
        _nc_rows = [_nc_row(k.replace("_", " "), v) for k, v in _nc_v2["by_kind"].items()]
        if _nc_v2.get("overall"):
            _nc_rows.append(_nc_row("all planted claims", _nc_v2["overall"]))
        st.dataframe(pd.DataFrame(_nc_rows), width="stretch", hide_index=True)
        _ov = _nc_v2.get("overall", {})
        if _ov.get("missed"):
            callout(f"<b>The judge missed {_ov['missed']} of {_ov['corrupted_judged']} planted unsupported claims.</b> "
                    f"Faithfulness fell in only {_ov.get('faithfulness_lower')} and hallucination in only {_ov.get('hallucination_lower')} of "
                    f"{_ov['corrupted_judged']} corrupted summaries, so the scores barely respond to planted errors.", "warning", tag="Limitation")
        if _ov.get("keyword_in_reason_only"):
            md(f'<div class="riq-panel-caption">{_ov["keyword_in_reason_only"]} pair(s) mention the planted claim only in a <i>reason</i> '
               f'(the judge used the fabricated number as if it were true); these are counted as missed. The earlier rule, which also '
               f'searched reasons, counted {_ov.get("detected_with_previous_rule")} detections.</div>')
        if _nc_v2.get("missed"):
            with st.expander(f"Missed planted claims ({len(_nc_v2['missed'])}, keyword rule)", icon=":material/search_off:"):
                st.dataframe(pd.DataFrame(_nc_v2["missed"]), width="stretch", hide_index=True)
        _sem = None
        for _src in _src_paths:
            _sem = _sem or read_json(mode_dir() / "d3" / "judge_negative_control_v2" / _src / "semantic_review.json")
        if _sem and _sem.get("items"):
            md(f'<div class="riq-panel-caption"><b>Keyword match ≠ understanding.</b> Reading every stored judgement: planted claim listed as '
               f'unsupported {_sem["named_in_claims_incl_paraphrase"]} / {_sem["n"]} (incl. paraphrases the keywords miss); mentioned as unsupported '
               f'anywhere {_sem["noticed_anywhere"]} / {_sem["n"]} (some only in a justification while the claim list stays empty, a contradictory verdict); '
               f'used as fact {_sem["accepted_as_fact"]}; not mentioned {_sem["not_mentioned"]}. These counts are context only and are '
               f'not combined with the keyword detection rate above. Reviewer: {esc(_sem.get("reviewer", ""))}. '
               f'<b>{esc(_sem.get("assessment_type", "model-assisted assessment; not human ground truth"))}</b>.</div>')
            with st.expander("Semantic review per planted claim", icon=":material/manage_search:"):
                st.dataframe(pd.DataFrame(_sem["items"])[["product_id", "kind", "category", "keyword_detected", "hallucination_score", "evidence"]],
                             width="stretch", hide_index=True)
        if _nc_v2.get("original_false_positive"):
            fp = _nc_v2["original_false_positive"]
            any_fp = _nc_v2.get("originals_flagging_any_claim")
            md(f'<div class="riq-panel-caption">Controls: the judge named a planted keyword in {fp["flagged"]} of {fp["n"]} keyword checks on the '
               f'{fp.get("distinct_original_judgements", "?")} original summaries. This is a narrow measure: '
               + (f'the judge listed at least one "unsupported" claim for <b>{any_fp["flagged"]} of {any_fp["n"]}</b> original summaries, '
                  'many of them legitimate content, so its general false-positive rate is high.' if any_fp else "see the judge evaluation report.")
               + '</div>')
        md(f'<div class="riq-panel-caption">{esc(_nc_v2.get("note", ""))} Small sample (15 products): rates are indicative only, not statistically reliable.</div>')
    else:
        md('<div class="riq-panel-caption">The v2 negative control has not been run yet (<code>scripts/judge_negative_control.py</code>).</div>')
elif _nc_v1:
    md(kv_html([("Pairs scored", _nc_v1.get("pairs_judged")), ("Corrupted judgements failed", _nc_v1.get("failed_corrupted_judgements")),
                ("Corrupted scored lower (mean)", _nc_v1.get("corrupted_scored_lower_mean")), ("Corrupted faithfulness lower", _nc_v1.get("corrupted_faithfulness_lower")),
                ("Invented feature flagged", _nc_v1.get("flagged_fabricated_feature"))]))

# ---------------------------------------------------------------- judge v3 (separately labelled run, read-only view)
_v3 = None
for _src in _src_paths:
    _v3 = _v3 or read_json(mode_dir() / "d3" / "judge_v3_report" / _src / "report.json")
if _v3:
    section("Judge v3 (separate run)", "One call per summary, raw response kept, no repair; corrected copies judged separately")
    callout("<b>v3 is a separately labelled run.</b> v1 and v2 above are unchanged. v3 makes one judge call per summary with no "
            "repair, so no claim is produced under validator pressure. A hallucination score is reported only when it agrees with the "
            "judge's own claim list; contradictory verdicts are counted, not repaired. Scores are model opinions.", "accent", tag="v3")

    def _v3_row(name: str, b: dict) -> dict:
        m = b.get("mean_scores", {})
        return {"evaluated": name, "attempted": b.get("attempted"), "valid": b.get("valid"), "validation failures": b.get("validation_failures"),
                "faithfulness": m.get("faithfulness"), "coverage": m.get("coverage"), "consistency": m.get("consistency"),
                "hallucination (consistent only)": m.get("hallucination"),
                "consistent verdicts": f'{b.get("hallucination_consistent", "–")} / {b.get("valid")}',
                "with claims": b.get("summaries_with_claims"), "claims": b.get("claims_total"),
                "claims found in summary": b.get("claims_found_in_summary")}

    st.dataframe(pd.DataFrame([_v3_row("original summaries", _v3["originals"]), _v3_row("corrected copies", _v3["corrected"]),
                               _v3_row("negative-control copies", _v3["nc_corrupted"])]), width="stretch", hide_index=True)
    if _v3.get("original_vs_corrected"):
        md('<div class="riq-panel-caption"><b>Original vs corrected copy</b> — each row has its own v3 judgement; no score is transferred '
           'between the original and the corrected summary.</div>')
        st.dataframe(pd.DataFrame([{"product_id": r["product_id"], "original id": r["original_summary_id"], "corrected id": r["corrected_summary_id"],
                                    "original (v3)": json.dumps(r["original_scores_v3"]) if r["original_scores_v3"] else "–",
                                    "corrected (v3)": json.dumps(r["corrected_scores_v3"]),
                                    "claims on original": len(r["original_claims"] or []), "claims on corrected": len(r["corrected_claims"])}
                                   for r in _v3["original_vs_corrected"]]), width="stretch", hide_index=True)
    _n3 = _v3.get("negative_control_v3", {})
    if _n3.get("overall", {}).get("corrupted_judged"):
        _o3 = _n3["overall"]
        _ci3 = _o3.get("detection_wilson95")
        md(f'<div class="riq-panel-caption"><b>v3 negative control</b> (same 30 corrupted copies, same claim-text keyword rule): detected '
           f'{_o3["detected"]} / {_o3["corrupted_judged"]}' + (f' (95% CI {_ci3[0]:.0%}–{_ci3[1]:.0%})' if _ci3 else "") +
           f', missed {_o3["missed"]}; faithfulness lower in {_o3["faithfulness_lower"]} of {_o3["pairs_with_v3_original"]} pairs. '
           f'Validation failures: {_n3.get("validation_failures", 0)}. The v2 result above (18 / 30) is kept as the historical result; '
           'the two runs are reported side by side, never pooled.</div>')
        st.dataframe(pd.DataFrame([{"planted defect": k.replace("_", " "), "judged": v["corrupted_judged"], "detected": v["detected"], "missed": v["missed"],
                                    "faithfulness lower": v["faithfulness_lower"], "hallucination lower": v["hallucination_lower"]}
                                   for k, v in _n3.get("by_kind", {}).items()]), width="stretch", hide_index=True)

# ---------------------------------------------------------------- judge v4 (final D3 evaluation, read-only view)
from ui.d3_v4_section import render as _render_v4  # noqa: E402

_render_v4(mode_dir(), list(_src_paths))

# ---------------------------------------------------------------- detailed view
section("Detailed evaluation", "Generated summary, its sources and the judge's assessment")
sid = st.selectbox("Summary", list(summaries), format_func=lambda i: f"{(summaries[i].product_title or summaries[i].product_id)[:70]} · {'judged' if i in judged else ('judge failed' if i in failed else 'not judged')}")
summ = summaries[sid]
j = judged.get(sid)
c = summ.content
left, right = st.columns([3, 2], gap="medium")
with left:
    with panel("Generated summary", right_html=badge("AI-generated", "accent")):
        if summ.is_mock:
            callout("<b>MOCK template summary.</b>", "warning", tag="Mock")
        md(f'<div class="riq-summary-opinion">{esc(c.overall_opinion)}</div>')
        md(
            '<div class="riq-list-title">What customers appreciate</div><ul class="riq-list">'
            + "".join(f"<li>{esc(p.point)} <span class='riq-chip'>{esc(', '.join(p.supporting_review_ids))}</span></li>" for p in c.appreciated)
            + '</ul><div class="riq-list-title">Common complaints</div><ul class="riq-list">'
            + "".join(f"<li>{esc(p.point)} <span class='riq-chip'>{esc(', '.join(p.supporting_review_ids))}</span></li>" for p in c.complaints)
            + "</ul>"
        )
        with st.expander("Full summary JSON", icon=":material/data_object:"):
            st.json(c.model_dump(mode="json"), expanded=True)
    agg = summ.aggregate
    with panel("Source · aggregated statistics", right_html=badge("Calculated", "neutral")):
        sp = agg.sentiment_percentages
        md(kv_html([
            ("Reviews", agg.total_reviews),
            ("Positive / negative", f"{sp.get('positive', 0)}% / {sp.get('negative', 0)}%"),
            ("Neutral / mixed", f"{sp.get('neutral', 0)}% / {sp.get('mixed', 0)}%"),
            ("Top complaint", next((f"{pretty(k)} ({v})" for k, v in agg.complaint_counts.items()), "none")),
        ]))
        asp = pd.DataFrame([x.model_dump() for x in agg.aspect_stats])
        if not asp.empty:
            asp = asp[["aspect_category", "mentions", "positive", "negative", "neutral"]].assign(aspect_category=lambda d: d["aspect_category"].map(pretty)).rename(columns={"aspect_category": "aspect"})
            st.dataframe(asp, width="stretch", hide_index=True)
    with st.expander(f"Source · evidence quotes ({len(agg.positive_evidence) + len(agg.negative_evidence)})", icon=":material/format_quote:"):
        md("".join(
            f'<div class="riq-quote {"pos" if q.sentiment == "positive" else "neg"}">“{esc(q.quote)}”<span class="riq-quote-meta">{esc(q.review_id)} · {esc(q.sentiment)} · {esc(pretty(q.aspect_category))}</span></div>'
            for q in agg.positive_evidence + agg.negative_evidence
        ))
with right:
    with panel("Judge assessment"):
        if not j and sid in failed:
            callout("<b>Judge failed.</b> The judge model's output was still invalid after the repair attempts, so this summary has no score. "
                    "The failure is kept as the result (one judge attempt per summary) and is not re-run automatically.", "warning", tag="Failed")
            md(f'<div class="riq-panel-caption">{esc(failed[sid].get("judge_model", ""))} · {esc((failed[sid].get("error") or "")[:400])}</div>')
            if failed[sid].get("attempts"):
                with st.expander(f"Raw judge outputs ({len(failed[sid]['attempts'])} attempts)", icon=":material/data_object:"):
                    for a in failed[sid]["attempts"]:
                        st.caption(f"attempt {a['attempt']} · {a['kind']} · errors: {'; '.join(a.get('errors', []))[:300]}")
                        st.code(a.get("raw_output", "")[:3000], language="json")
            if CAN_JUDGE_VERSION and st.button("Retry this failed judgement", icon=":material/refresh:", disabled=not judge_ready,
                                              help="Explicit re-run; the new outcome is stored next to the failure"):
                run_judge([summ], retry_failed=True)
        elif not j:
            md('<div class="riq-panel-caption">Not judged yet.</div>')
            if st.button("Judge this summary", icon=":material/gavel:", disabled=not judge_ready):
                run_judge([summ])
        else:
            if j.is_mock:
                callout("<b>MOCK judge verdict</b> — placeholder values, not an LLM assessment.", "warning", tag="Mock")
            v = j.verdict
            md("".join(
                f'<div class="riq-dim"><div class="riq-dim-head"><span>{esc(label)}</span><span>{getattr(v, key).score} / 5</span></div>'
                f'{score_html(getattr(v, key).score)}<div class="riq-dim-body">{esc(getattr(v, key).justification)}</div></div>'
                for key, label, _ in DIMENSIONS
            ))
            details = {d.claim: d for d in v.unsupported_claim_details}
            claims = "".join(
                f'<div class="riq-quote neg">“{esc(cl)}”'
                + (f'<span class="riq-quote-meta">{esc(details[cl].reason)}{" · checked " + esc(", ".join(details[cl].evidence_review_ids)) if details[cl].evidence_review_ids else ""}</span>' if cl in details and (details[cl].reason or details[cl].evidence_review_ids) else "")
                + "</div>"
                for cl in v.unsupported_claims
            ) or '<div class="riq-panel-caption">None flagged.</div>'
            md(f'<div class="riq-list-title">Unsupported claims</div>{claims}<div class="riq-list-title">Overall comment</div><div class="riq-dim-body">{esc(v.overall_comment)}</div>')
    a = audits.get(sid)
    if a:
        with panel("Evidence verification", "Every cited id and statistic checked against the source", right_html=badge("verified" if a["ids_and_statistics_verified"] else "issues", "positive" if a["ids_and_statistics_verified"] else "warning")):
            md(kv_html([("Cited ids", f'{a["cited_ids"]} ({len(a["unknown_ids"])} unknown, {len(a["wrong_product_ids"])} other product)'),
                        ("Statistics match re-aggregation", "yes" if a["statistics_match_source"] else "no"),
                        ("Points traceable / partial / weak", f'{a["points_traceable"]} / {a["points_partial"]} / {a["points_weak"]}')]))
            for i in a["issues"]:
                md(f'<div class="riq-panel-caption">⚠ {esc(i)}</div>')
    if j:
        q = audit_judgement(j, summ)
        if q["claims"]:
            with panel("Judge claim check", "Are the claims the judge called unsupported contradicted by the source data?"):
                st.dataframe(pd.DataFrame([{"claim": c["claim"], "polarity": c["polarity"], "source categories supporting it": ", ".join(c["source_categories_supporting"]),
                                            "likely judge false positive": c["likely_false_positive"]} for c in q["claims"]]), width="stretch", hide_index=True)
        with panel("Deterministic checks", "Computed without an LLM", right_html=badge("Calculated", "neutral")):
            dc = j.deterministic_checks
            pct = dc.get("percentages_not_in_statistics") or []
            md(kv_html([
                ("Percentages not in statistics", ", ".join(map(str, pct)) if pct else "none"),
                ("Top-3 aspects mentioned", _of(dc.get("top_aspects_mentioned"), dc.get("top_aspects"))),
                ("Top complaints mentioned", _of(dc.get("top_complaints_mentioned"), dc.get("top_complaints"))),
                ("Cited IDs belong to product", "yes" if dc.get("cited_ids_all_in_product") else "no"),
                *([("Flagged claims found in summary text", f'{dc.get("unsupported_claims_traceable_to_summary")} of {dc.get("unsupported_claims_flagged")}')] if "unsupported_claims_flagged" in dc else []),
            ]))
        md(f'<div class="riq-panel-caption">{esc(j.judge_provider)}/{esc(j.judge_model)} · prompt {esc(j.prompt_version)} · repairs {j.repair_attempts} · {esc(j.created_at)}</div>')

