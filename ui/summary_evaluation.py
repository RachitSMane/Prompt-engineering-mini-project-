"""Page 5 — Summary Evaluation with LLM-as-a-Judge."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from evaluation.summary_evaluation import JudgementStore, export_inspection_sample, judge_summary, judgements_table
from evaluation.visualizations import judge_scores_bar
from pipeline.extraction import ExtractionStore
from pipeline.summarization import SummaryStore
from prompts.judge import RUBRIC
from schemas.evaluation_schema import ProductSummary
from ui.common import get_client, get_settings, llm_ready, mock_banner, mode_dir, require_llm_note, use_mock
from utils.export import dataframe_to_csv_bytes

s = get_settings()
st.title("🧑‍⚖️ Summary Evaluation")
st.caption("LLM-as-a-Judge scoring of product summaries: faithfulness, coverage, consistency, hallucination (1–5).")
mock_banner()
st.info(
    "**Important:** judge scores are an automated assessment by a language model. They are **not** objective human "
    "evaluation, and the judge can be wrong or biased (e.g. towards fluent text, or towards its own model family). "
    "Use the inspection view below to audit them manually."
)
with st.expander("Judge rubric"):
    st.code(RUBRIC, language="text")

sstore = SummaryStore.for_mode(s, use_mock())
jstore = JudgementStore.for_mode(s, use_mock())
latest = sstore.latest()
summaries: dict[str, ProductSummary] = {x.summary_id: x for x in latest.values()}
if not summaries:
    st.info("No summaries in this mode yet. Generate them on the **Product Insights** page.")
    st.stop()
judged = jstore.latest_by_summary()


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


require_llm_note()
st.caption(f"Judge model: **{s.effective_judge_provider} / {s.effective_judge_model}**" + (" (same as the summariser; a different judge model reduces self-preference bias — set JUDGE_PROVIDER/JUDGE_MODEL)" if s.effective_judge_model == s.llm_model and not use_mock() else ""))
c1, c2 = st.columns(2)
unjudged = [x for sid, x in summaries.items() if sid not in judged]
if c1.button(f"Judge all unjudged summaries ({len(unjudged)})", type="primary", disabled=not unjudged or not llm_ready()):
    run_judge(unjudged)

# ---------------------------------------------------------------- overview
rows = judgements_table([judged[sid] for sid in summaries if sid in judged], summaries)
if rows:
    st.plotly_chart(judge_scores_bar(rows), width="stretch")
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True)
    d = st.columns(3)
    d[0].download_button("Judgements (CSV)", dataframe_to_csv_bytes(table), "judgements.csv", "text/csv", width="stretch")
    d[1].download_button("Judgements (JSON)", json.dumps([judged[sid].model_dump(mode="json") for sid in summaries if sid in judged], indent=2), "judgements.json", "application/json", width="stretch")
    d[2].download_button("Manual inspection sample (MD)", export_inspection_sample([judged[sid] for sid in summaries if sid in judged], summaries, n=3), "judge_inspection_sample.md", "text/markdown", width="stretch")
else:
    st.caption("No judgements yet.")

# ---------------------------------------------------------------- inspection
st.divider()
st.subheader("Inspect a summary and its judgement")
sid = st.selectbox("Summary", list(summaries), format_func=lambda i: f"{(summaries[i].product_title or summaries[i].product_id)[:70]} {'✅' if i in judged else '—'}")
summ = summaries[sid]
j = judged.get(sid)
left, right = st.columns(2)
with left:
    st.markdown("#### Generated summary")
    if summ.is_mock:
        st.warning("MOCK template summary.")
    st.json(summ.content.model_dump(mode="json"), expanded=True)
    st.markdown("#### Source: aggregated statistics")
    agg = summ.aggregate
    st.write({"total_reviews": agg.total_reviews, "sentiment_percentages": agg.sentiment_percentages, "complaint_counts": agg.complaint_counts})
    st.dataframe(pd.DataFrame([x.model_dump() for x in agg.aspect_stats])[["aspect_category", "mentions", "positive", "negative", "neutral"]], width="stretch", hide_index=True)
    with st.expander("Source: evidence quotes"):
        for q in agg.positive_evidence + agg.negative_evidence:
            st.markdown(f"- `{q.review_id}` ({q.sentiment}, {q.aspect_category}) *“{q.quote}”*")
with right:
    st.markdown("#### Judge verdict")
    if not j:
        st.caption("Not judged yet.")
        if st.button("Judge this summary", disabled=not llm_ready()):
            run_judge([summ])
    else:
        if j.is_mock:
            st.warning("MOCK judge verdict — placeholder values, not an LLM assessment.")
        v = j.verdict
        m = st.columns(4)
        for col, dim in zip(m, ["faithfulness", "coverage", "consistency", "hallucination"]):
            col.metric(dim.capitalize(), f"{getattr(v, dim).score}/5")
        for dim in ["faithfulness", "coverage", "consistency", "hallucination"]:
            st.markdown(f"**{dim.capitalize()}** — {getattr(v, dim).justification}")
        st.markdown("**Unsupported claims flagged:** " + (", ".join(f"“{c}”" for c in v.unsupported_claims) or "none"))
        st.markdown(f"**Overall comment:** {v.overall_comment}")
        st.markdown("#### Deterministic checks (no LLM)")
        dc = j.deterministic_checks
        st.write(
            {
                "percentages in summary not found in statistics": dc.get("percentages_not_in_statistics"),
                "top-3 aspects mentioned": f"{dc.get('top_aspects_mentioned')} of {dc.get('top_aspects')}",
                "top complaints mentioned": f"{dc.get('top_complaints_mentioned')} of {dc.get('top_complaints')}",
                "all cited review ids belong to product": dc.get("cited_ids_all_in_product"),
            }
        )
        st.caption(f"{j.judge_provider}/{j.judge_model} · prompt {j.prompt_version} · repairs {j.repair_attempts} · {j.created_at}")
        if st.button("Re-judge", disabled=not llm_ready()):
            run_judge([summ])
