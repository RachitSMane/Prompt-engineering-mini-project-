"""Deliverable 3 export: product summaries, aggregation, evidence and LLM-as-a-Judge results as files.

Reads the latest summary per product for one extraction source (``outputs/<mode>/summaries/summaries.jsonl``)
and the latest judgement per summary (``outputs/<mode>/judgements/judgements.jsonl``), and writes to
``outputs/<mode>/d3/<source>/``:

  aggregates.csv        one row per product: review count, sentiment %, complaint rate, top aspects/complaints
  aspect_stats.csv      one row per (product, aspect category)
  evidence.csv          the evidence quotes given to the summariser (review id, aspect, sentiment, quote)
  summaries.json        the generated summaries (content + provenance)
  judgements.csv        judge scores per product + deterministic checks + citation-polarity check
  d3_report.md          per-product summaries with evidence, judge verdicts and an overview

No LLM calls are made. The citation-polarity check is deterministic: every "appreciated" point should cite
reviews whose extraction contains a positive aspect, every "complaint" point reviews with a negative aspect.

Usage:  python scripts/export_d3.py --source dataset__few_shot__qwen2.5-3b__v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.summary_evaluation import JudgementStore  # noqa: E402
from pipeline.extraction import ExtractionStore  # noqa: E402
from pipeline.summarization import SummaryStore  # noqa: E402
from schemas.evaluation_schema import ProductSummary, SummaryJudgement  # noqa: E402
from schemas.review_schema import ReviewRecord  # noqa: E402
from utils.storage import read_jsonl  # noqa: E402

DIMS = ("faithfulness", "coverage", "consistency", "hallucination")


def citation_polarity(summary: ProductSummary, records: dict[str, ReviewRecord]) -> dict:
    """Share of cited reviews whose extraction has an aspect of the point's polarity."""
    checks = {"appreciated": "positive", "complaints": "negative"}
    out: dict = {}
    for field, polarity in checks.items():
        cited = [rid for p in getattr(summary.content, field) for rid in p.supporting_review_ids]
        ok = [rid for rid in cited if rid in records and any(a.sentiment.value == polarity for a in records[rid].aspects)]
        out[f"{field}_citations"] = len(cited)
        out[f"{field}_citations_matching_polarity"] = len(ok)
    total = out["appreciated_citations"] + out["complaints_citations"]
    matched = out["appreciated_citations_matching_polarity"] + out["complaints_citations_matching_polarity"]
    out["citation_polarity_rate"] = round(matched / total, 3) if total else None
    return out


def build(source: str, settings, is_mock: bool) -> Path:
    mode_dir = settings.paths.mode_dir(is_mock)
    summaries = {pid: s for (pid, src), s in SummaryStore.for_mode(settings, is_mock).latest().items() if src == source}
    if not summaries:
        sys.exit(f"No summaries for source '{source}' in {mode_dir / 'summaries'}")
    judgements: dict[str, SummaryJudgement] = JudgementStore.for_mode(settings, is_mock).latest_by_summary()
    fail_file = mode_dir / "judgements" / "judge_failures.jsonl"
    failures = {r["summary_id"]: r for r in read_jsonl(fail_file)} if fail_file.exists() else {}
    records = {r.review_id: r for r in ExtractionStore(mode_dir / "extractions" / f"{source}.jsonl").records}
    out = mode_dir / "d3" / source
    out.mkdir(parents=True, exist_ok=True)

    agg_rows, aspect_rows, ev_rows, judge_rows = [], [], [], []
    for pid, s in sorted(summaries.items(), key=lambda kv: kv[1].product_title or kv[0]):
        a = s.aggregate
        agg_rows.append({
            "product_id": pid, "product_title": a.product_title, "reviews_extracted": a.total_reviews,
            **{f"pct_{k}": v for k, v in a.sentiment_percentages.items()},
            "reviews_with_complaints": a.reviews_with_complaints,
            "top_aspects": "; ".join(f"{x.aspect_category} ({x.mentions}, net {x.net_sentiment:+.2f})" for x in a.aspect_stats[:5]),
            "top_complaints": "; ".join(f"{k} ({v})" for k, v in list(a.complaint_counts.items())[:5]),
        })
        aspect_rows += [{"product_id": pid, **x.model_dump()} | {"top_aspect_names": "; ".join(x.top_aspect_names)} for x in a.aspect_stats]
        ev_rows += [{"product_id": pid, **q.model_dump()} for q in a.positive_evidence + a.negative_evidence]
        j = judgements.get(s.summary_id)
        row = {"product_id": pid, "product_title": a.product_title, "summary_id": s.summary_id,
               "summary_model": s.model_name, "summary_repairs": s.repair_attempts, "summary_input_tokens": s.input_tokens,
               **citation_polarity(s, records)}
        row["judge_status"] = "ok" if j else ("failed (invalid judge output)" if s.summary_id in failures else "not run")
        if not j and s.summary_id in failures:
            row["judge_error"] = failures[s.summary_id].get("error", "")
        if j:
            row |= {d: getattr(j.verdict, d).score for d in DIMS} | {
                "mean_score": j.verdict.mean_score, "unsupported_claims": len(j.verdict.unsupported_claims),
                "judge_model": j.judge_model, "judge_repairs": j.repair_attempts,
                "pct_not_in_statistics": len(j.deterministic_checks.get("percentages_not_in_statistics", [])),
                "top_aspect_coverage": j.deterministic_checks.get("top_aspect_coverage"),
                "cited_ids_all_in_product": j.deterministic_checks.get("cited_ids_all_in_product"),
            }
        judge_rows.append(row)

    pd.DataFrame(agg_rows).to_csv(out / "aggregates.csv", index=False, encoding="utf-8")
    pd.DataFrame(aspect_rows).to_csv(out / "aspect_stats.csv", index=False, encoding="utf-8")
    pd.DataFrame(ev_rows).to_csv(out / "evidence.csv", index=False, encoding="utf-8")
    jdf = pd.DataFrame(judge_rows)
    jdf.to_csv(out / "judgements.csv", index=False, encoding="utf-8")
    (out / "summaries.json").write_text(json.dumps([s.model_dump(mode="json") for s in summaries.values()], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "d3_report.md").write_text(render_report(source, summaries, judgements, jdf, failures), encoding="utf-8")
    return out


def render_report(source: str, summaries: dict[str, ProductSummary], judgements: dict[str, SummaryJudgement], jdf: pd.DataFrame, failures: dict | None = None) -> str:
    failures = failures or {}
    any_s = next(iter(summaries.values()))
    judged = jdf.dropna(subset=["faithfulness"]) if "faithfulness" in jdf else jdf.iloc[0:0]
    lines = [f"# D3 — Extract → Aggregate → Summarise · `{source}`", ""]
    if any_s.is_mock:
        lines += ["> **MOCK / OFFLINE DATA — deterministic fixtures, not LLM summaries or judgements. Not a result.**", ""]
    lines += [
             f"- Products summarised: {len(summaries)} · summaries judged: {len(judged)}",
             f"- Summariser: `{any_s.provider}/{any_s.model_name}` (prompt {any_s.prompt_version})" + (
                 f" · judge: `{judged['judge_model'].iloc[0]}`" if len(judged) else ""),
             "- Summaries are **model-generated** from the aggregated statistics and evidence quotes; they are not verified ground truth.",
             "- Judge scores are an **automated model assessment**, not objective truth (see Limitations).", ""]
    if len(judged):
        lines += ["## Judge score overview (1–5)", "", "| dimension | mean | min | max | share of 5s |", "| --- | --- | --- | --- | --- |"]
        for d in DIMS:
            col = judged[d].astype(int)
            lines.append(f"| {d} | {col.mean():.2f} | {col.min()} | {col.max()} | {(col == 5).mean():.0%} |")
        perfect = int((judged[list(DIMS)].astype(int) == 5).all(axis=1).sum())
        lines += ["", f"- Summaries with a perfect 5 on every dimension: {perfect} of {len(judged)}",
                  f"- Unsupported claims flagged by the judge: {int(judged['unsupported_claims'].sum())} in total",
                  f"- Deterministic: summaries quoting a percentage absent from the statistics: {int((judged['pct_not_in_statistics'] > 0).sum())}",
                  f"- Deterministic: mean top-3 aspect coverage {judged['top_aspect_coverage'].astype(float).mean():.2f}; "
                  f"citation-polarity rate (cited reviews matching the point's sentiment) {jdf['citation_polarity_rate'].astype(float).mean():.2f}", ""]
        if len(judged) > 2 and judged["citation_polarity_rate"].notna().sum() > 2:
            r = judged[["faithfulness", "citation_polarity_rate"]].astype(float).corr().iloc[0, 1]
            lines += [f"- Correlation between judge faithfulness and the deterministic citation-polarity rate: {r:.2f} "
                      "(low agreement means the two signals measure different things or the judge is lenient).", ""]
    lines += ["## Per-product results", ""]
    for s in sorted(summaries.values(), key=lambda x: x.product_title or x.product_id):
        a, c = s.aggregate, s.content
        lines += [f"### {a.product_title or s.product_id} (`{s.product_id}`)", "",
                  f"Reviews extracted: {a.total_reviews} · sentiment %: `{a.sentiment_percentages}` · reviews with complaints: {a.reviews_with_complaints}", "",
                  f"**Overall opinion (generated):** {c.overall_opinion}", ""]
        for title, pts in (("Appreciated", c.appreciated), ("Complaints", c.complaints)):
            if pts:
                lines.append(f"**{title}:**")
                lines += [f"- {p.point} — cites {', '.join(f'`{r}`' for r in p.supporting_review_ids)}" for p in pts]
                lines.append("")
        quotes = a.positive_evidence[:3] + a.negative_evidence[:3]
        if quotes:
            lines.append("**Evidence given to the summariser (sample):**")
            lines += [f'- `{q.review_id}` ({q.sentiment}, {q.aspect_category}): "{q.quote}"' for q in quotes]
            lines.append("")
        j = judgements.get(s.summary_id)
        if j:
            v = j.verdict
            lines += [f"**Judge ({j.judge_model}):** " + " · ".join(f"{d} {getattr(v, d).score}" for d in DIMS) + f" · mean {v.mean_score}", ""]
            lines += [f"- *{d}*: {getattr(v, d).justification}" for d in DIMS]
            lines += [f"- Unsupported claims: {v.unsupported_claims or 'none flagged'}", f"- Comment: {v.overall_comment}", ""]
        elif s.summary_id in failures:
            lines += [f"**Judge:** FAILED — the judge output was still invalid after the repair attempts, so no score exists for this summary "
                      f"(error: `{failures[s.summary_id].get('error', '')[:300]}`).", ""]
        else:
            lines += ["**Judge:** not run for this summary.", ""]
    lines += ["## Limitations of LLM-as-a-Judge", "",
              "- The judge is itself a small local LLM (3B parameters); it can miss unsupported claims, be lenient towards fluent text and "
              "misread statistics. Its scores are model opinions, not measurements.",
              "- It sees truncated review texts (80 words each) and the same extractions the summariser used, so extraction errors are invisible to it.",
              "- Scores are not calibrated against human ratings in this project; the deterministic checks and the inspection sample are provided "
              "so that a human can audit them.", ""]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="Export D3 product summaries and judge results.")
    p.add_argument("--source", required=True, help="extraction source name, e.g. dataset__few_shot__qwen2.5-3b__v1")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    out = build(args.source, load_settings(), args.mock)
    print(f"Wrote {out}")
    print((out / "d3_report.md").read_text(encoding="utf-8")[:3000])


if __name__ == "__main__":
    main()
