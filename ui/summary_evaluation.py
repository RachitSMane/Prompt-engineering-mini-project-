"""Page 5 — Summary Evaluation with LLM-as-a-Judge."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from evaluation.summary_evaluation import JudgementStore, export_inspection_sample, judge_summary, judgements_table
from evaluation.visualizations import judge_scores_bar, judge_scores_per_summary
from pipeline.extraction import ExtractionStore
from pipeline.summarization import SummaryStore
from prompts.judge import RUBRIC
from schemas.evaluation_schema import ProductSummary
from ui.common import get_client, get_settings, llm_ready, mock_banner, mode_dir, pretty, require_llm_note, use_mock
from ui.components import badge, callout, chart, empty_state, esc, kv_html, md, page_header, panel, score_html, section
from utils.export import dataframe_to_csv_bytes

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

sstore = SummaryStore.for_mode(s, use_mock())
jstore = JudgementStore.for_mode(s, use_mock())
latest = sstore.latest()
summaries: dict[str, ProductSummary] = {x.summary_id: x for x in latest.values()}
if not summaries:
    empty_state("No summaries in this mode yet", "Generate them on the <b>Product Intelligence</b> page, then judge them here.", "summarize")
    st.stop()
judged = jstore.latest_by_summary()


def _of(found, expected) -> str:
    """'2 of 3 (price value, performance)' from the deterministic-check lists."""
    if isinstance(found, list) and isinstance(expected, list):
        return f"{len(found)} of {len(expected)}" + (f" ({', '.join(pretty(x) for x in found)})" if found else "")
    return f"{found} of {expected}"


def records_for(summary: ProductSummary):
    path = mode_dir() / "extractions" / f"{summary.source_extraction}.jsonl"
    return ExtractionStore(path).records if path.exists() else []


def run_judge(targets: list[ProductSummary]) -> None:
    client = get_client(role="judge")
    if not client:
        return
    bar = st.progress(0.0)
    for i, summ in enumerate(targets, 1):
        recs = records_for(summ)
        if not recs:
            st.error(f"Source extractions for {summ.product_id} not found; cannot judge.")
            continue
        j, res = judge_summary(client, summ, recs, s)
        if j:
            jstore.add(j)
        else:
            st.error(f"Judge failed for {summ.product_id}: {res.error}")
        bar.progress(i / len(targets), text=f"Judged {i}/{len(targets)}")
    st.rerun()


# ---------------------------------------------------------------- actions
require_llm_note()
unjudged = [x for sid, x in summaries.items() if sid not in judged]
c1, c2 = st.columns([3, 2], vertical_alignment="center")
same_model = s.effective_judge_model == s.llm_model and not use_mock()
c1.markdown(
    f'<div class="riq-panel-caption" style="margin:0">Judge model <b style="color:var(--riq-text)">{esc(s.effective_judge_provider)} / {esc(s.effective_judge_model)}</b>'
    + (" · same as the summariser; set JUDGE_PROVIDER / JUDGE_MODEL to reduce self-preference bias" if same_model else "")
    + f" · {len(judged)} of {len(summaries)} summaries judged</div>",
    unsafe_allow_html=True,
)
if c2.button(f"Judge all unjudged summaries ({len(unjudged)})", type="primary", icon=":material/gavel:", disabled=not unjudged or not llm_ready(), width="stretch"):
    run_judge(unjudged)
with st.expander("Judge rubric", icon=":material/rule:"):
    st.code(RUBRIC, language="text")

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

# ---------------------------------------------------------------- detailed view
section("Detailed evaluation", "Generated summary, its sources and the judge's assessment")
sid = st.selectbox("Summary", list(summaries), format_func=lambda i: f"{(summaries[i].product_title or summaries[i].product_id)[:70]} · {'judged' if i in judged else 'not judged'}")
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
        if not j:
            md('<div class="riq-panel-caption">Not judged yet.</div>')
            if st.button("Judge this summary", icon=":material/gavel:", disabled=not llm_ready()):
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
            claims = "".join(f'<div class="riq-quote neg">“{esc(cl)}”</div>' for cl in v.unsupported_claims) or '<div class="riq-panel-caption">None flagged.</div>'
            md(f'<div class="riq-list-title">Unsupported claims</div>{claims}<div class="riq-list-title">Overall comment</div><div class="riq-dim-body">{esc(v.overall_comment)}</div>')
    if j:
        with panel("Deterministic checks", "Computed without an LLM", right_html=badge("Calculated", "neutral")):
            dc = j.deterministic_checks
            pct = dc.get("percentages_not_in_statistics") or []
            md(kv_html([
                ("Percentages not in statistics", ", ".join(map(str, pct)) if pct else "none"),
                ("Top-3 aspects mentioned", _of(dc.get("top_aspects_mentioned"), dc.get("top_aspects"))),
                ("Top complaints mentioned", _of(dc.get("top_complaints_mentioned"), dc.get("top_complaints"))),
                ("Cited IDs belong to product", "yes" if dc.get("cited_ids_all_in_product") else "no"),
            ]))
        md(f'<div class="riq-panel-caption">{esc(j.judge_provider)}/{esc(j.judge_model)} · prompt {esc(j.prompt_version)} · repairs {j.repair_attempts} · {esc(j.created_at)}</div>')
        if st.button("Re-judge", icon=":material/refresh:", disabled=not llm_ready()):
            run_judge([summ])
