"""Build the final project report and the presentation from the curated results in reports/.

  python scripts/build_final_report.py --d2-run <run_id> [--compare-run <run_id>] --d3-source <source> [--discussion reports/discussion.md]

Every number is read from files produced by the pipeline (reports/d2, reports/d1, reports/d3, reports/verification.md);
nothing is typed in by hand. Screenshots are embedded from reports/screenshots/ (captured from the running app).
Writes reports/final/ReviewIQ_Final_Report.{html,pdf} and reports/presentation/ReviewIQ_Presentation.{html,pdf}
(PDF only when a local Chrome/Edge is available).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
TEAM = [("Rachit S Mane", "PES1UG24AM214"), ("P. Chinni Krishna Reddy", "PES1UG24AM209"),
        ("Pranav V. Menon", "PES1UG24AM198"), ("Akash R.", "PES1UG24AM212")]
LABEL = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "Chain-of-Thought"}
SCREENS = [("01_overview", "Overview — dataset size, deliverable status and processed reviews"),
           ("02_review_analysis", "Review Analysis — the extraction page in its initial state (data source, prompting strategy and run controls); no extraction was run in this browser session, so no records are shown"),
           ("03_prompt_experiments", "Prompt Experiments — measured comparison of the three strategies"),
           ("04_product_intelligence", "Product Intelligence — calculated statistics, aspect performance, complaint categories and evidence quotes for one product; the generated summary lower on the page is outside the captured area"),
           ("05_summary_evaluation", "Summary Evaluation — LLM-as-a-Judge scores and deterministic checks"),
           ("06_human_annotation", "Human Annotation — reference labels with provenance")]

e = html.escape


def img_tag(path: Path, alt: str, cls: str = "shot") -> str:
    if not path.exists():
        return f'<p class="missing">[missing image: {e(path.name)}]</p>'
    return f'<img class="{cls}" alt="{e(alt)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def d3_outcome(d: dict) -> dict:
    """D3 judge outcome counts read from the curated files (never typed in by hand)."""
    j = d.get("d3_judge")
    has = j is not None and len(j) and "judge_status" in j
    sc = d.get("d3_sanity") or {}
    return {"summaries": len(d.get("d3_summaries") or []),
            "judged": int((j["judge_status"] == "ok").sum()) if has else 0,
            "failed": int(j["judge_status"].astype(str).str.startswith("failed").sum()) if has else 0,
            "pairs": sc.get("pairs_judged", 0), "faith_lower": sc.get("corrupted_faithfulness_lower", 0),
            "feature_flagged": sc.get("flagged_fabricated_feature", 0)}


def table_html(df: pd.DataFrame, cols: dict[str, str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{e(v)}</th>" for v in cols.values())
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                s = "—"
            elif c in fmt:
                s = format(v, fmt[c])
            else:
                s = str(v)
            cells.append(f"<td>{e(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def md_inline(text: str) -> str:
    """Tiny Markdown subset for the hand-written discussion: paragraphs, bullets, **bold**, `code`."""
    out, items = [], []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        fmt = lambda s: re.sub(r"`(.+?)`", r"<code>\1</code>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e(s)))
        if lines[0].startswith("### "):
            out.append(f"<h3>{fmt(lines[0][4:])}</h3>")
            lines = lines[1:]
        if lines and all(l.lstrip().startswith("- ") for l in lines):
            items = "".join(f"<li>{fmt(l.lstrip()[2:])}</li>" for l in lines)
            out.append(f"<ul>{items}</ul>")
        elif lines:
            out.append(f"<p>{fmt(' '.join(lines))}</p>")
    return "\n".join(out)


def load(args) -> dict:
    d2dir = REPORTS / "d2" / args.d2_run
    data = {
        "d2_table": pd.read_csv(d2dir / "comparison_table.csv"),
        "d2_per_class": pd.read_csv(d2dir / "per_class_metrics.csv"),
        "d2_metrics": json.loads((d2dir / "metrics.json").read_text(encoding="utf-8")),
        "d2_config": json.loads((d2dir / "config.json").read_text(encoding="utf-8")),
        "d2_charts": d2dir / "charts",
        "d1_reports": {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in (REPORTS / "d1" / args.d2_run).glob("*/validation_report.json")},
        "verification": (REPORTS / "verification.md").read_text(encoding="utf-8") if (REPORTS / "verification.md").exists() else "",
        "verification_independent": (REPORTS / "verification_independent.md").read_text(encoding="utf-8") if (REPORTS / "verification_independent.md").exists() else "",
        "discussion": Path(args.discussion).read_text(encoding="utf-8") if args.discussion and Path(args.discussion).exists() else "",
    }
    if args.compare_run:
        cdir = REPORTS / "d2" / args.compare_run
        data["cmp_table"] = pd.read_csv(cdir / "comparison_table.csv")
        data["cmp_metrics"] = json.loads((cdir / "metrics.json").read_text(encoding="utf-8"))
    d3dir = REPORTS / "d3" / args.d3_source
    data["d3_judge"] = pd.read_csv(d3dir / "judgements.csv") if (d3dir / "judgements.csv").exists() else pd.DataFrame()
    data["d3_agg"] = pd.read_csv(d3dir / "aggregates.csv") if (d3dir / "aggregates.csv").exists() else pd.DataFrame()
    data["d3_summaries"] = json.loads((d3dir / "summaries.json").read_text(encoding="utf-8")) if (d3dir / "summaries.json").exists() else []
    sanity = d3dir / "judge_sanity" / "summary.json"
    data["d3_sanity"] = json.loads(sanity.read_text(encoding="utf-8")) if sanity.exists() else None
    data["d3_source"] = args.d3_source
    return data


CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;color:#1d2433;max-width:1000px;margin:0 auto;padding:28px 40px;line-height:1.5;font-size:11pt}
h1{font-size:24pt;margin:0 0 4px;color:#111a33} h2{font-size:16pt;color:#111a33;border-bottom:2px solid #6366f1;padding-bottom:4px;margin-top:28px;page-break-after:avoid}
h3{font-size:12.5pt;margin:18px 0 6px;color:#28325a;page-break-after:avoid} table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:9.5pt;page-break-inside:avoid}
th,td{border:1px solid #d3d8e6;padding:4px 7px;text-align:left;vertical-align:top} th{background:#eef0fa} code{background:#f2f3f8;padding:1px 4px;border-radius:3px;font-size:9.5pt}
.cover{border:1px solid #d3d8e6;border-radius:8px;padding:22px 26px;margin-bottom:18px;background:#f7f8fd} .muted{color:#5b6478;font-size:10pt}
.shot{width:100%;border:1px solid #c9cfe0;border-radius:6px;margin:6px 0 2px} figure{margin:10px 0 16px;page-break-inside:avoid;break-inside:avoid} figcaption{font-size:9.5pt;color:#5b6478}
.chart{width:100%;border:1px solid #e1e4ef;border-radius:4px} .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px} .note{background:#fff7e6;border-left:4px solid #f59e0b;padding:8px 12px;margin:10px 0;font-size:10pt}
.ok{background:#ecfdf3;border-left:4px solid #10b981;padding:8px 12px;margin:10px 0;font-size:10pt} .missing{color:#b91c1c}
@media print{body{padding:0 14px} h2{page-break-before:auto} img.shot{width:auto;max-width:100%;max-height:228mm;display:block;margin:6px auto 2px}}
"""


def report_html(d: dict, args) -> str:
    t, cfg, m = d["d2_table"], d["d2_config"], d["d2_metrics"]
    best = t.sort_values("sentiment_macro_f1", ascending=False).iloc[0]
    prov = m.get("reference_provenance", {})
    s = []
    if m.get("is_mock"):
        s.append("<div class='note'><b>MOCK / OFFLINE DATA — deterministic fixtures, not experimental results.</b></div>")
    s.append(f"""<div class="cover"><div class="muted">PES University · Prompt Engineering Mini Project · 5th Semester AIML</div>
<h1>ReviewIQ — Aspect-Based Customer Review Analytics with Structured Outputs</h1><div class="muted">Final project report</div>
<p>{' · '.join(f'{e(n)} ({e(r)})' for n, r in TEAM)}</p></div>""")
    o = d3_outcome(d)
    s.append(f"""<h2>Abstract</h2><p>ReviewIQ converts unstructured Amazon appliance reviews into validated structured records (overall sentiment,
aspect-level sentiment with verbatim evidence, and complaint categories) using a local large language model served by Ollama, with no cloud
API. Outputs are validated with Pydantic and repaired automatically when invalid. On {m['n_gold_available']} evaluable reviews, three prompting
strategies were compared with <code>{e(cfg['model_name'])}</code>; the highest overall-sentiment macro-F1 was {best['sentiment_macro_f1']:.3f}
({e(str(best['strategy']))}). A prompt chain aggregates extractions per product and generated cited summaries for all {o['summaries']} products; a separate
local LLM judge returned a valid verdict for only <b>{o['judged']} of {o['summaries']}</b> summaries (<b>{o['failed']} judge outputs failed</b> schema validation and have no
score). In a negative control with deliberately corrupted summaries, the judge's faithfulness score decreased in <b>{o['faith_lower']} of {o['pairs']}</b> scored pairs, so its
faithfulness scores are not a reliable quality measure. The reference labels are mostly AI-assisted ({prov.get('human', '?')} human, {prov.get('ai_assisted', '?')} AI-assisted), so the
scores measure agreement with those labels; this and the other limitations are discussed in Section 10.</p>""")
    s.append("""<h2>1. Introduction</h2><p>Star ratings say how much customers liked a product but not why. Aspect-based sentiment analysis identifies
what customers talk about (fit, installation, noise, durability, price) and how they feel about each aspect. Large language models can perform this
analysis from free text, but their outputs are unreliable: they can produce invalid JSON, labels outside the taxonomy or evidence that does not appear
in the review. ReviewIQ therefore treats the model as an unreliable component inside a validated pipeline.</p>
<h2>2. Problem statement and objectives</h2><p><i>"Develop an LLM pipeline that converts unstructured product reviews into validated structured records
containing sentiment, product aspects and complaint categories, and generates per-product summaries. The pipeline shall use public e-commerce review data
and compare prompting techniques for sentiment analysis and multi-label classification."</i></p>
<ul><li><b>D1</b> — schema-based extraction with Pydantic and LangChain, JSON/CSV export and automatic repair of malformed outputs.</li>
<li><b>D2</b> — comparison of zero-shot, few-shot and Chain-of-Thought prompting on the 300-review labelled sample, with macro-F1.</li>
<li><b>D3</b> — prompt chain extract → aggregate → summarise with per-product summaries and LLM-as-a-Judge faithfulness evaluation.</li></ul>
<h2>3. Background</h2><p>The dataset is the Amazon Reviews 2023 corpus (Hou et al., 2024). Prompting strategies follow the standard zero-shot,
few-shot (in-context demonstrations) and Chain-of-Thought (explicit reasoning procedure) formulations. Evaluation uses macro-averaged F1, which weights
classes equally and is therefore suited to the class-imbalanced sentiment labels. LLM-as-a-Judge uses a language model with a rubric to rate generated
text; its known weaknesses (leniency, self-preference, sensitivity to fluency) motivate the separate judge model and deterministic checks used here.</p>""")
    s.append(f"""<h2>4. System architecture</h2><p>Reviews are processed by a strategy-specific prompt, sent to the local model through LangChain's
<code>ChatOllama</code>, parsed, validated with Pydantic (including evidence grounding against the review text) and repaired when invalid. Valid records are
aggregated per product with pandas; a summarisation prompt receives only the statistics and evidence quotes; a separate judge model scores each summary.
All inference uses <code>http://127.0.0.1:11434</code>; non-local endpoints are rejected by configuration.</p>
<pre style="font-size:8.5pt;background:#f7f8fd;padding:10px;border-radius:6px">reviews → [EXTRACT] prompt → local LLM → parse → validate (+grounding) → repair loop → ReviewRecord / controlled failure
        → [AGGREGATE] pandas statistics + evidence quotes → [SUMMARISE] cited summary (validated ids) → [JUDGE] rubric 1–5 + deterministic checks</pre>""")
    d1 = d["d1_reports"]
    d1df = pd.DataFrame([{"strategy": LABEL.get(k, k), **{x: v.get(x) for x in ("predictions_stored", "reviews_in_run", "valid_records", "failed_output",
                         "failed_provider", "first_attempt_valid", "needed_llm_repair", "repaired_successfully")},
                          "revalidated": f"{v['revalidation']['revalidated_ok']}/{v['revalidation']['reloaded_records']}",
                          "mean_s": v["runtime_s"]["mean"], "total_h": round(v["runtime_s"]["total"] / 3600, 2),
                          "in_tok": v["tokens"]["input_total"], "out_tok": v["tokens"]["output_total"]} for k, v in d1.items()])
    s.append(f"""<h2>5. Methodology</h2><h3>5.1 Dataset</h3><p>Amazon Reviews 2023, Appliances category (2,128,605 raw reviews), prepared deterministically
(seed 42): text cleaning, 5–400 words, products with 200–5,000 valid reviews, 15 random products, de-duplication, 100 reviews per product — 1,500 reviews.
The evaluation sample has 300 reviews stratified by star rating; the rating is never shown to the model or used as ground truth.</p>
<h3>5.2 Annotation and label provenance</h3><p>The reference labels contain {prov.get('total', '?')} evaluable records: <b>{prov.get('human', '?')} human
label</b> and <b>{prov.get('ai_assisted', '?')} AI-assisted labels</b> generated by an AI model (Claude) from the review text only, following the annotation
guide, with the project mentor's approval. One unreadable review is flagged and excluded. The provenance is recorded in the label file and shown in the
application next to every metric.</p>
<h3>5.3 Controlled comparison</h3><p>All strategies used the same {len(cfg['review_ids'])} review ids, model <code>{e(cfg['model_name'])}</code>, temperature
{cfg['temperature']}, output cap {cfg['settings']['llm_max_output_tokens']} tokens, context {cfg['settings']['ollama_num_ctx']} tokens, up to
{cfg['max_repair_attempts']} repair attempts and strict evidence grounding. Few-shot demonstrations are synthetic, so they cannot overlap the test set.
Chain-of-Thought performs its procedure silently and returns the same schema.</p>""")
    s.append(f"""<h2>6. D1 — Structured extraction: implementation and results</h2><p>Each review produces either a validated <code>ReviewRecord</code>
or a controlled failure. The table summarises the extraction runs of the D2 experiment (the same records are exported as JSON and CSV; every exported
record was reloaded and re-validated, including evidence grounding).</p>
{table_html(d1df, {"strategy": "strategy", "predictions_stored": "predicted", "reviews_in_run": "reviews", "valid_records": "valid", "failed_output": "failed (output)",
                   "failed_provider": "failed (provider)", "first_attempt_valid": "valid first try", "needed_llm_repair": "needed repair", "repaired_successfully": "repaired OK",
                   "revalidated": "re-validated", "mean_s": "mean s", "total_h": "hours", "in_tok": "input tokens", "out_tok": "output tokens"})}""")
    s.append(f"""<h2>7. D2 — Prompting strategy comparison</h2><p>Status: <b>{e(m['status'])}</b> · reference labels in run: {m['n_gold_available']} /
{m['required_gold']} · missing predictions: {e(json.dumps(m.get('missing_predictions')))}.</p>
{table_html(t, {"strategy": "strategy", "n_gold": "n", "coverage": "coverage", "failed": "failed", "sentiment_accuracy": "accuracy", "sentiment_macro_precision": "macro P",
                "sentiment_macro_recall": "macro R", "sentiment_macro_f1": "macro F1", "sentiment_weighted_f1": "weighted F1"},
            {"coverage": ".3f", "sentiment_accuracy": ".3f", "sentiment_macro_precision": ".3f", "sentiment_macro_recall": ".3f", "sentiment_macro_f1": ".3f", "sentiment_weighted_f1": ".3f"})}
<h3>Complaints and aspects</h3>
{table_html(t, {"strategy": "strategy", "complaint_micro_f1": "complaint micro F1", "complaint_macro_f1": "complaint macro F1", "complaint_hamming_loss": "Hamming loss",
                "aspect_detection_micro_f1": "aspect detection F1", "aspect_sentiment_macro_f1": "aspect sentiment macro F1", "aspect_joint_micro_f1": "joint F1"},
            {k: ".3f" for k in ("complaint_micro_f1", "complaint_macro_f1", "complaint_hamming_loss", "aspect_detection_micro_f1", "aspect_sentiment_macro_f1", "aspect_joint_micro_f1")})}
<h3>Reliability and cost</h3>
{table_html(t, {"strategy": "strategy", "validity_rate": "validity", "first_attempt_valid_rate": "valid first try", "needed_llm_repair": "needed repair",
                "mean_latency_s": "mean s / review", "mean_input_tokens": "mean input tokens", "mean_output_tokens": "mean output tokens"},
            {"validity_rate": ".3f", "first_attempt_valid_rate": ".3f", "mean_latency_s": ".1f", "mean_input_tokens": ".0f", "mean_output_tokens": ".0f"})}
<h3>Per-class results (overall sentiment)</h3>
{table_html(d["d2_per_class"].assign(strategy=lambda x: x["strategy"].map(lambda v: LABEL.get(v, v))), {"strategy": "strategy", "class": "class", "precision": "precision", "recall": "recall", "f1-score": "F1", "support": "support"},
            {"precision": ".3f", "recall": ".3f", "f1-score": ".3f"})}
<figure>{img_tag(d['d2_charts'] / 'macro_f1.png', 'macro F1', 'chart')}<figcaption>Overall-sentiment macro-F1 per strategy.</figcaption></figure>
<figure>{img_tag(d['d2_charts'] / 'per_class_f1.png', 'per-class F1', 'chart')}<figcaption>Per-class F1 per strategy.</figcaption></figure>
<div class="grid2">{''.join(f"<figure>{img_tag(d['d2_charts'] / f'confusion_{n}.png', n, 'chart')}<figcaption>Confusion matrix — {LABEL[n]} (rows = reference, columns = prediction).</figcaption></figure>" for n in LABEL if (d['d2_charts'] / f'confusion_{n}.png').exists())}</div>""")
    if "cmp_table" in d:
        c, cm = d["cmp_table"], d["cmp_metrics"]
        s.append(f"""<h3>Model comparison: {e(cm['model_name'])} on a subset</h3><p>Status: <b>{e(cm['status'])}</b> · reviews: {cm['n_gold_available']} (subset; not the full
evaluation). Compare with the {e(cfg['model_name'])} rows for the same reviews in the discussion below.</p>
{table_html(c, {"strategy": "strategy", "n_gold": "n", "coverage": "coverage", "sentiment_accuracy": "accuracy", "sentiment_macro_f1": "macro F1",
                "complaint_micro_f1": "complaint micro F1", "aspect_detection_micro_f1": "aspect detection F1", "validity_rate": "validity", "mean_latency_s": "mean s"},
            {k: ".3f" for k in ("coverage", "sentiment_accuracy", "sentiment_macro_f1", "complaint_micro_f1", "aspect_detection_micro_f1", "validity_rate")} | {"mean_latency_s": ".1f"})}""")
    j = d["d3_judge"]
    if len(j):
        dims = ["faithfulness", "coverage", "consistency", "hallucination"]
        judged = j.dropna(subset=["faithfulness"]) if "faithfulness" in j else j.iloc[0:0]
        agg = d["d3_agg"]
        n_rev = int(agg["reviews_extracted"].sum()) if len(agg) else 0
        rng = f"{int(agg['reviews_extracted'].min())}–{int(agg['reviews_extracted'].max())}" if len(agg) else "?"
        s.append(f"""<h2>8. D3 — Prompt chain: extract → aggregate → summarise</h2>
<div class='note'><b>Scope:</b> the chain was run on the <b>{n_rev} validly extracted reviews of the 299-review evaluation sample</b>
({len(agg)} products, {rng} reviews per product), <b>not on the full 1,500-review dataset</b>. Extractions were reused from the D2 run (source
<code>{e(d['d3_source'])}</code>) after a compatibility check of provider, model, prompt version and validation settings, so no review was sent to the
model twice. Reviews whose extraction failed are not part of the aggregates.</div>
<p>{len(d['d3_summaries'])} products were summarised from their extracted
reviews; {len(judged)} summaries were judged successfully{f" and <b>{int(j['judge_status'].astype(str).str.startswith('failed').sum())} judge call(s) failed</b> (the judge output was still invalid after the repair attempts; no score exists for those summaries and none was filled in)" if 'judge_status' in j else ""}. Summaries are model output and are kept with their statistics, evidence quotes and cited review ids.</p>
{table_html(j.assign(product=lambda x: x['product_title'].astype(str).str.slice(0, 48)), {"product": "product", **({"judge_status": "judge"} if "judge_status" in j else {}), **{k: k for k in dims}, "mean_score": "mean", "unsupported_claims": "unsupported claims",
                "citation_polarity_rate": "citation polarity", "top_aspect_coverage": "top-aspect coverage", "summary_repairs": "summary repairs"},
            {"citation_polarity_rate": ".2f", "top_aspect_coverage": ".2f"})}""")
        if len(judged):
            s.append("<p>" + " · ".join(f"mean {k} {judged[k].astype(float).mean():.2f}" for k in dims) +
                     f" · summaries with all scores 5: {int((judged[dims].astype(int) == 5).all(axis=1).sum())} of {len(judged)}.</p>")
        if d["d3_summaries"]:
            ex = d["d3_summaries"][0]
            c = ex["content"]
            s.append(f"""<h3>Example summary — {e(ex.get('product_title') or ex['product_id'])}</h3><p><b>Overall opinion (generated):</b> {e(c['overall_opinion'])}</p>
<ul>{''.join(f"<li>👍 {e(p['point'])} <span class='muted'>[{', '.join(p['supporting_review_ids'])}]</span></li>" for p in c['appreciated'])}
{''.join(f"<li>👎 {e(p['point'])} <span class='muted'>[{', '.join(p['supporting_review_ids'])}]</span></li>" for p in c['complaints'])}</ul>""")
    sc = d.get("d3_sanity")
    if sc and sc.get("pairs_judged"):
        mo, mc = sc["mean_original"], sc["mean_corrupted"]
        dims4 = ["faithfulness", "coverage", "consistency", "hallucination", "mean"]
        skipped = sc.get("skipped_original_not_judged") or []
        s.append(f"""<h3>Negative control: does the judge notice a corrupted summary?</h3><p>{f"Only the {sc['products'] - len(skipped)} summaries whose original judgement succeeded were eligible; {len(skipped)} were skipped because their original judgement had failed (failed judgements are not re-sampled). " if skipped else ""}Each eligible summary was copied and deliberately corrupted
({'; '.join(e(x) for x in sc['corruption'])}) and judged with the same judge model ({e(sc['judge_model'])}) and prompt. Corrupted copies are stored separately and
never shown as results.</p>
<table><thead><tr><th></th>{''.join(f'<th>{k}</th>' for k in dims4)}</tr></thead><tbody>
<tr><td>original (mean of {sc['pairs_judged']})</td>{''.join(f'<td>{mo[k]:.2f}</td>' for k in dims4)}</tr>
<tr><td>corrupted (mean of {sc['pairs_judged']})</td>{''.join(f'<td>{mc[k]:.2f}</td>' for k in dims4)}</tr></tbody></table>
<p>Corrupted copy scored lower on the mean in <b>{sc['corrupted_scored_lower_mean']} of {sc['pairs_judged']}</b> pairs, equal in {sc['corrupted_scored_equal_mean']},
higher in {sc['corrupted_scored_higher_mean']}; lower faithfulness in {sc['corrupted_faithfulness_lower']}. The judge listed the invented feature as an unsupported
claim in {sc['flagged_fabricated_feature']} and the fabricated percentage in {sc['flagged_fabricated_percentage']} of {sc['pairs_judged']} cases.
{sc['failed_corrupted_judgements']} corrupted judgement(s) failed validation.</p>""")
    s.append("<h2>9. Evaluation, testing and verification</h2>")
    if d.get("verification_independent"):
        s.append("<h3>Independent recomputation of the D2 metrics</h3><div class='ok'>" + md_inline("\n\n".join(
            l for l in d["verification_independent"].splitlines() if l.startswith(("Recomputed", "**Result")))) + "</div>")
    s.append(f"<div class='ok'>{md_inline(d['verification'].replace('# Verification of curated results', ''))}</div>" if d["verification"] else "")
    if d["discussion"]:
        s.append("<h2>Discussion</h2>" + md_inline(d["discussion"]))
    s.append("<h2>Screenshots (captured from the running application)</h2>")
    for name, cap in SCREENS:
        s.append(f"<figure>{img_tag(REPORTS / 'screenshots' / f'{name}.png', cap)}<figcaption>{e(cap)}</figcaption></figure>")
    s.append("""<h2>10. Limitations</h2><ul>
<li>298 of the 299 evaluable reference labels are AI-assisted; the scores measure agreement with these labels, not independent manual annotation, and there is no inter-annotator agreement.</li>
<li>The models are small enough for a 4 GB GPU. The 3B model misses aspects and complaints. """ + ("The 7B model only partly fits in VRAM and was compared on a subset only." if args.compare_run else "The 7B model only partly fits in VRAM (about 88 s per review in the smoke test), so no 7B evaluation on the 299 reviews was run; only a one-review smoke test exists.") + """</li>
<li>The 3B model occasionally loops; a 1,024-token cap truncates such outputs, which are then repaired or counted as failures.</li>
<li>Aspect matching is at category level; the evaluation sample is rating-stratified, so accuracy describes the balanced set.</li>
<li>LLM-as-a-Judge scores come from a 3B model and are not calibrated against human ratings. D3 product summaries are based on the evaluation sample (""" + (f"{int(d['d3_agg']['reviews_extracted'].min())}–{int(d['d3_agg']['reviews_extracted'].max())}" if len(d["d3_agg"]) else "?") + """ extracted reviews per product), not on all 100 reviews per product.</li></ul>
<h2>11. Conclusion</h2><p>ReviewIQ demonstrates a complete, locally hosted pipeline that turns free-text reviews into validated structured data, compares
prompting strategies under controlled conditions with transparent coverage and failure accounting, and produces cited product summaries for all """ + str(o["summaries"]) + """ products. The LLM-as-a-Judge stage is auditable but
did not work reliably with the small local judge: only """ + f"{o['judged']} of {o['summaries']} judge outputs were valid and {o['failed']} failed" + """, and in the
negative control the judge's faithfulness score decreased in """ + f"{o['faith_lower']} of {o['pairs']}" + """ scored pairs (the invented feature was flagged in
""" + f"{o['feature_flagged']} of {o['pairs']}" + """), with too few pairs for a statistical conclusion. The deterministic checks are the more trustworthy D3 evidence. The validation and repair layer makes even a small local model usable for structured extraction, while the measured results show
the limits of 3B-parameter models for fine-grained aspect and complaint extraction.</p>
<h2>References</h2><ol><li>Hou, Y., Li, J., He, Z., Yan, A., Chen, X., &amp; McAuley, J. (2024). <i>Bridging Language and Items for Retrieval and Recommendation.</i> arXiv:2403.03952. (Amazon Reviews 2023 dataset)</li>
<li>Project repository: https://github.com/RachitSMane/Prompt-engineering-mini-project-</li>
<li>Software: Ollama, LangChain (langchain-core, langchain-ollama), Pydantic, pandas, scikit-learn, Streamlit, Plotly — versions in reports/execution_log.md.</li></ol>""")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>ReviewIQ — Final Report</title><style>{CSS}</style></head><body>{''.join(s)}</body></html>"


SLIDE_CSS = """
@page{size:1280px 720px;margin:0} body{margin:0;font-family:'Segoe UI',Arial,sans-serif;background:#0b1020;color:#e6e9f5}
.slide{width:1280px;height:720px;box-sizing:border-box;padding:48px 64px;page-break-after:always;position:relative;overflow:hidden;background:linear-gradient(135deg,#0b1020,#151c38)}
h1{font-size:44px;margin:0 0 12px;color:#fff} h2{font-size:32px;margin:0 0 18px;color:#a5b4fc} p,li{font-size:20px;line-height:1.45} .muted{color:#9aa3c0}
table{border-collapse:collapse;font-size:17px;width:100%} th,td{border:1px solid #2c3561;padding:6px 10px;text-align:left} th{background:#1e2650;color:#c7d2fe}
.big{font-size:56px;font-weight:700;color:#fff} .kpis{display:flex;gap:22px;margin:18px 0} .kpi{background:#1a2147;border:1px solid #2c3561;border-radius:12px;padding:16px 20px;flex:1}
.kpi .l{color:#9aa3c0;font-size:16px} .shot{max-height:470px;max-width:100%;border-radius:10px;border:1px solid #2c3561} .chart{max-height:440px;max-width:100%;border-radius:8px;background:#fff}
.foot{position:absolute;bottom:22px;left:64px;right:64px;font-size:14px;color:#6b7393;display:flex;justify-content:space-between} code{background:#1e2650;padding:1px 6px;border-radius:4px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px}
"""


def slides_html(d: dict, args) -> str:
    t, cfg, m = d["d2_table"], d["d2_config"], d["d2_metrics"]
    prov = m.get("reference_provenance", {})
    best = t.sort_values("sentiment_macro_f1", ascending=False).iloc[0]
    o = d3_outcome(d)
    foot = "<div class='foot'><span>ReviewIQ · PES University · Prompt Engineering Mini Project</span><span>{n}</span></div>"
    slides = []
    add = lambda body: slides.append(f"<section class='slide'>{body}{foot.format(n=len(slides) + 1)}</section>")
    add(f"""<div class='muted'>PES University · Prompt Engineering Mini Project · 5th Semester AIML</div><h1 style='margin-top:120px'>ReviewIQ</h1>
<p style='font-size:28px'>Aspect-Based Customer Review Analytics with Structured Outputs</p><p class='muted' style='margin-top:60px'>{' · '.join(f'{e(n)} ({e(r)})' for n, r in TEAM)}</p>""")
    add("""<h2>Problem</h2><ul><li>Thousands of free-text reviews; star ratings say <b>how much</b>, not <b>why</b>.</li>
<li>Goal: validated structured records — overall sentiment, aspects with evidence, complaint categories — and per-product summaries.</li>
<li>Compare prompting techniques for sentiment and multi-label classification on public e-commerce data.</li>
<li>Constraint: run on a student laptop, free, with no review data leaving the machine.</li></ul>""")
    add("""<h2>Solution</h2><ul><li>Local LLMs through <b>Ollama</b> (qwen2.5:3b extraction, llama3.2 judge) — no API key, no cloud.</li>
<li><b>Pydantic</b> schema + evidence grounding; <b>automatic repair</b> of malformed outputs; controlled failures.</li>
<li>Three prompting strategies behind one interface; resumable, cached experiments with honest status labels.</li>
<li>Extract → Aggregate → Summarise chain with cited summaries and an LLM-as-a-Judge; six-page Streamlit dashboard.</li></ul>""")
    add("""<h2>Architecture</h2><p style='font-family:Consolas,monospace;font-size:19px;line-height:1.9'>reviews → <b>EXTRACT</b> (prompt → local LLM → parse → validate + grounding → repair)<br>
→ <b>AGGREGATE</b> (pandas statistics, evidence quotes)<br>→ <b>SUMMARISE</b> (cited summary, ids validated)<br>→ <b>JUDGE</b> (separate model, rubric 1–5 + deterministic checks)</p>
<p class='muted'>LangChain ChatOllama · 127.0.0.1:11434 only · live and mock outputs physically separated</p>""")
    add("""<h2>Three deliverables</h2><div class='cols'><div><p><b>D1 Structured extraction</b><br>Pydantic schema, LangChain parser instructions, JSON/CSV export,
repair loop, resumable cache, outage-safe batches.</p><p><b>D2 Prompt comparison</b><br>Zero-shot · Few-shot (synthetic examples) · Chain-of-Thought (silent procedure),
same model/data/settings; macro-F1 primary.</p></div><div><p><b>D3 Prompt chain</b><br>Extract → aggregate → summarise per product, every point cites review ids;
LLM-as-a-Judge (faithfulness, coverage, consistency, hallucination) + deterministic checks.</p></div></div>""")
    rows = "".join(f"<tr><td>{e(str(r['strategy']))}</td><td>{r['sentiment_macro_f1']:.3f}</td><td>{r['sentiment_accuracy']:.3f}</td><td>{r['coverage']:.3f}</td><td>{r['validity_rate']:.3f}</td><td>{r['needed_llm_repair']}</td><td>{r['mean_latency_s']:.1f}</td></tr>" for _, r in t.iterrows())
    add(f"""<h2>D2 results — {e(cfg['model_name'])}, {m['n_gold_available']} reviews</h2><table><tr><th>strategy</th><th>macro F1</th><th>accuracy</th><th>coverage</th><th>validity</th><th>repairs</th><th>s / review</th></tr>{rows}</table>
<p>Best macro-F1: <b>{e(str(best['strategy']))}</b> ({best['sentiment_macro_f1']:.3f}). Status: <b>{e(m['status'])}</b>.{" <b style='color:#fbbf24'>MOCK DATA — NOT A RESULT</b>" if m.get('is_mock') else ""}</p>
<p class='muted'>Reference labels: {prov.get('human', '?')} human + {prov.get('ai_assisted', '?')} AI-assisted — scores measure agreement with these labels.</p>""")
    add(f"""<h2>Per-class F1 and confusion</h2><div class='cols'><div>{img_tag(d['d2_charts'] / 'per_class_f1.png', 'per class', 'chart')}</div>
<div>{img_tag(d['d2_charts'] / f"confusion_{next((k for k, v in LABEL.items() if v == best['strategy']), 'zero_shot')}.png", 'confusion', 'chart')}</div></div>""")
    j = d["d3_judge"]
    if len(j) and "faithfulness" in j:
        judged = j.dropna(subset=["faithfulness"])
        dims = ["faithfulness", "coverage", "consistency", "hallucination"]
        add(f"""<h2>D3 results — product summaries and judge</h2><div class='kpis'>{''.join(f"<div class='kpi'><div class='l'>{k}</div><div class='big'>{judged[k].astype(float).mean():.2f}</div><div class='l'>mean of {len(judged)} (1–5)</div></div>" for k in dims)}</div>
<p>{len(d['d3_summaries'])} products summarised · <b>{len(judged)} judged, {int(j['judge_status'].astype(str).str.startswith('failed').sum()) if 'judge_status' in j else 0} judge call(s) failed</b> (invalid judge output; no score) · citation-polarity rate {j['citation_polarity_rate'].astype(float).mean():.2f} · unsupported claims flagged: {int(judged['unsupported_claims'].sum())}</p>
<p class='muted'>Scope: the 299-review evaluation sample ({int(d['d3_agg']['reviews_extracted'].sum()) if len(d['d3_agg']) else '?'} extracted reviews), not the full 1,500. Judge = llama3.2 (3B); scores are model opinions, audited with deterministic checks — not objective truth.</p>""")
        sc = d.get("d3_sanity")
        if sc and sc.get("pairs_judged"):
            add(f"""<h2>Does the judge catch a bad summary?</h2><p>Negative control: each summary with a valid original judgement ({sc['products'] - len(sc.get('skipped_original_not_judged') or [])} of {sc['products']}) copied and corrupted (inverted opinion + fabricated %, invented feature, complaints removed), judged by the same model; {sc['pairs_judged']} pair(s) scored, {sc['failed_corrupted_judgements']} corrupted judgement(s) failed.</p>
<div class='kpis'><div class='kpi'><div class='l'>mean score — original</div><div class='big'>{sc['mean_original']['mean']:.2f}</div></div>
<div class='kpi'><div class='l'>mean score — corrupted</div><div class='big'>{sc['mean_corrupted']['mean']:.2f}</div></div>
<div class='kpi'><div class='l'>corrupted scored lower (mean)</div><div class='big'>{sc['corrupted_scored_lower_mean']}/{sc['pairs_judged']}</div></div>
<div class='kpi'><div class='l'>corrupted faithfulness lower</div><div class='big'>{sc['corrupted_faithfulness_lower']}/{sc['pairs_judged']}</div></div></div>
<p>Invented feature flagged in {sc['flagged_fabricated_feature']}/{sc['pairs_judged']} · fabricated percentage flagged in {sc['flagged_fabricated_percentage']}/{sc['pairs_judged']} · equal scores in {sc['corrupted_scored_equal_mean']}/{sc['pairs_judged']}.{" Too few pairs for a conclusion." if sc['pairs_judged'] < 10 else ""}</p>""")
    for name, cap in SCREENS[:6]:
        p = REPORTS / "screenshots" / f"{name}.png"
        if p.exists():
            add(f"<h2>Demo — {e(cap.split(' — ')[0])}</h2>{img_tag(p, cap)}<p class='muted'>{e(cap)}</p>")
    add("""<h2>Demo flow</h2><ol><li>Overview: dataset (1,500 reviews, 15 products) and deliverable status.</li><li>Review Analysis: run one review live, show JSON, evidence and repair log.</li>
<li>Prompt Experiments: comparison table, macro-F1, confusion matrices, provenance note.</li><li>Product Intelligence: aggregated aspects/complaints and the cited summary.</li>
<li>Summary Evaluation: judge scores, unsupported claims, deterministic checks.</li><li>Human Annotation: reference labels with human / AI-assisted badges.</li></ol>""")
    add("""<h2>Limitations</h2><ul><li>Reference labels mostly AI-assisted (1 human) — agreement, not independent ground truth.</li>
<li>Small local models: 3B misses aspects/complaints; """ + ("7B partly in VRAM, compared on a subset." if args.compare_run else "7B too slow on a 4 GB GPU for the 299-review evaluation (smoke test only).") + """</li><li>Occasional runaway generation, handled by an output cap + repair.</li>
<li>Category-level aspect matching; rating-stratified sample; 3B judge not human-calibrated.</li></ul>""")
    add("""<h2>Conclusion</h2><ul><li>A complete, local, reproducible pipeline: validated extraction, fair prompt comparison, cited summaries, auditable judge.</li>
<li>Judge reliability: """ + f"{o['judged']}/{o['summaries']} valid judge outputs ({o['failed']} failed); faithfulness lower for corrupted summaries in {o['faith_lower']}/{o['pairs']} scored pairs" + """ — not a reliable quality measure.</li>
<li>Validation + repair make a small local model usable for structured output; the metrics show where 3B models fall short.</li>
<li>Next: human re-annotation, constrained JSON decoding, larger local model, human-calibrated judging.</li></ul><p class='big' style='margin-top:40px'>Thank you</p>""")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>ReviewIQ — Presentation</title><style>{SLIDE_CSS}</style></head><body>{''.join(slides)}</body></html>"


def to_pdf(html_path: Path, landscape_slides: bool = False) -> Path | None:
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        return None
    pdf = html_path.with_suffix(".pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=False)
    return pdf if pdf.exists() else None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="Build the final report and presentation from reports/.")
    p.add_argument("--d2-run", required=True)
    p.add_argument("--compare-run", default=None)
    p.add_argument("--d3-source", required=True)
    p.add_argument("--discussion", default=str(REPORTS / "discussion.md"))
    args = p.parse_args()
    d = load(args)
    for sub, name, builder in (("final", "ReviewIQ_Final_Report.html", report_html), ("presentation", "ReviewIQ_Presentation.html", slides_html)):
        out = REPORTS / sub / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(builder(d, args), encoding="utf-8")
        pdf = to_pdf(out)
        print(f"Wrote {out}" + (f" and {pdf}" if pdf else " (no local browser for PDF)"))


if __name__ == "__main__":
    main()
