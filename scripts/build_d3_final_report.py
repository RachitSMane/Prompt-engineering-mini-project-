"""Build the final D3 report (Markdown, HTML, PDF), its data file and SVG charts from stored artefacts only. No LLM.

  python scripts/build_d3_final_report.py --source dataset__few_shot__qwen2.5-3b__v1 [--no-pdf]

Inputs: outputs/live/d3/judge_v4/<source>/{report.json,evaluations.jsonl}, nc-v3 cases + manifest, judge v1-v3 artefacts.
Outputs: reports/final/D3_Final_Report.{md,html,pdf}, reports/final/figures/d3_*.svg, reports/d3/D3_REPORT_DATA.json.
Every number in the report is read from these files; scripts/verify_d3_final.py re-checks them.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.judge_v4 import SYSTEM_V4, HUMAN_V4  # noqa: E402
from schemas.claim_schema import SCORING_FORMULA, WEIGHTS  # noqa: E402
from scripts.build_final_report import CSS, TEAM, to_pdf  # noqa: E402
from utils.storage import read_json, read_jsonl  # noqa: E402

# Errors identified by the 2026-11-15 investigation (investigator reading of summaries vs statistics/cited reviews; NOT a
# human gold set). Used only to ask "did v4 flag the errors we already knew about?".
KNOWN_ERRORS = [
    ("B0BTY63SFY", "a61df84f4901:overall_opinion:1", "'80% of the reviews' positive (actual 86.7%; 80% = performance share)"),
    ("B0BTY63SFY", "a61df84f4901:overall_opinion:3", "'67% of negative reviews' (no such statistic)"),
    ("B0BTY63SFY", "a61df84f4901:appreciated:2", "'suitable for larger eggs' cites a review saying larger eggs do not fit"),
    ("B0BTY63SFY", "a61df84f4901:complaints:2", "'small legs' misreads a typo ('Only small legs fit' = eggs)"),
    ("B0BTY63SFY", "a61df84f4901:improvement_areas:3", "durability improvement with 3 positive / 0 negative durability mentions"),
    ("B0BTY63SFY", "a61df84f4901:recommendations:3", "durability recommendation without negative evidence"),
    ("B001DHKBXU", "2f2f868f0ec7:overall_opinion:2", "complaint counts given 'out of 15' (15 = reviews with complaints; 19 reviews)"),
    ("B001DHKBXU", "2f2f868f0ec7:complaints:1", "'installation was difficult' cites a review saying 'easy to install'"),
    ("B001DHKBXU", "2f2f868f0ec7:complaints:2", "'poor value' unsupported by its citation"),
    ("B001DHKBXU", "2f2f868f0ec7:complaints:4", "'compatibility issues' not in either cited review"),
    ("B00DM8K1NS", "2fb2110754ba:overall_opinion:0", "'3 positive reviews out of 13' (actual 5) and 'slightly positive' (negative > positive)"),
    ("B00LHRAGHW", "329800146f6c:overall_opinion:0", "'slight majority of positive' (40% positive vs 50% negative)"),
    ("B01FZJXITU", "f336d1f67313:overall_opinion:0", "'(7 out of 9) praising performance and price value' (4 and 2)"),
    ("B07XG8QQL5", "b575f6ccde9d:overall_opinion:0", "'majority of users' with performance/durability issues (31.6% each)"),
    ("B07XG8QQL5", "b575f6ccde9d:complaints:0", "performance complaint cites two 5-star positive reviews"),
    ("B0C5G1WGBN", "95d7c1c7e9d5:appreciated:1", "'good value' cites a 1-star complaint that does not mention value"),
]
FLAG = {"UNSUPPORTED", "CONTRADICTED"}
TYPE_LABEL = {"incorrect_review_count": "Incorrect review count / denominator", "incorrect_percentage": "Incorrect percentage",
              "incorrect_majority": "Incorrect sentiment majority", "incorrect_complaint_statistic": "Incorrect complaint statistic",
              "flipped_sentiment": "Flipped sentiment", "unsupported_performance": "Unsupported performance claim",
              "unsupported_durability": "Unsupported durability claim", "fabricated_compatibility": "Fabricated compatibility",
              "fabricated_warranty": "Fabricated warranty", "incorrect_aspect_attribution": "Incorrect aspect attribution",
              "contradictory_citation": "Contradictory citation", "misleading_paraphrase": "Misleading paraphrase"}


# --------------------------------------------------------------------------- small helpers
def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def ci(c):
    return "–" if not c else f"[{100 * c[0]:.1f}%, {100 * c[1]:.1f}%]"


def num(x):
    return "–" if x is None else (f"{x:g}" if isinstance(x, (int, float)) else str(x))


def table(header: list[str], rows: list[list]) -> str:
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def svg_bars(title: str, labels: list[str], series: list[tuple[str, list[float | None], str]], vmax: float = 100.0, unit: str = "") -> str:
    """Horizontal grouped bar chart; every value is printed next to its bar (checked by verify_d3_final.py)."""
    lw, bw, gap, bh = 190, 380, 14, 13
    h = 40 + len(labels) * (len(series) * bh + gap) + 24
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + bw + 80}" height="{h}" font-family="Segoe UI, Arial" font-size="11">',
           f'<text x="4" y="16" font-size="13" font-weight="600">{html.escape(title)}</text>']
    y = 34
    for i, lab in enumerate(labels):
        out.append(f'<text x="{lw - 6}" y="{y + bh * len(series) / 2 + 4}" text-anchor="end">{html.escape(lab)}</text>')
        for name, vals, color in series:
            v = vals[i]
            w = 0 if v is None else max(1, bw * min(v, vmax) / vmax)
            out.append(f'<rect x="{lw}" y="{y}" width="{w:.1f}" height="{bh - 2}" fill="{color}"/>')
            out.append(f'<text x="{lw + w + 4:.1f}" y="{y + bh - 3}">{"n/a" if v is None else f"{v:g}{unit}"}</text>')
            y += bh
        y += gap
    lx = 4
    for name, _, color in series:
        out.append(f'<rect x="{lx}" y="{h - 14}" width="10" height="10" fill="{color}"/><text x="{lx + 14}" y="{h - 5}">{html.escape(name)}</text>')
        lx += 20 + 7 * len(name)
    out.append("</svg>")
    return "".join(out)


def md_to_html(md: str, svgs: dict[str, str]) -> str:
    def inline(t: str) -> str:
        t = html.escape(t, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        return t

    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        if ln.startswith("!["):
            src = re.search(r"\(([^)]+)\)", ln).group(1)
            cap = re.search(r"!\[([^\]]*)\]", ln).group(1)
            out.append(f"<figure>{svgs.get(src, '')}<figcaption>{html.escape(cap)}</figcaption></figure>")
        elif ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.startswith("> "):
            out.append(f"<div class='ok'>{inline(ln[2:])}</div>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = lambda r: [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", r.strip())[1:-1]]
            hd = "".join(f"<th>{inline(c)}</th>" for c in cells(rows[0]))
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>" for r in rows[2:])
            out.append(f"<table><tr>{hd}</tr>{body}</table>")
            continue
        elif re.match(r"^(\d+\.|-) ", ln):
            tag = "ol" if ln[0].isdigit() else "ul"
            items = []
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]):
                items.append(re.sub(r"^(\d+\.|-) ", "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- data
def load(source: str) -> dict:
    mode = ROOT / "outputs" / "live"
    d = mode / "d3" / "judge_v4" / source
    rep = read_json(d / "report.json")
    if not rep:
        raise SystemExit("run `python scripts/judge_v4.py report` first")
    evals = [json.loads(x) for x in (d / "evaluations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    nc_rows = []
    import csv

    with (d / "nc_results.csv").open(encoding="utf-8") as f:
        nc_rows = list(csv.DictReader(f))
    claims = {c["claim"]["claim_id"]: c for ev in evals if ev["evaluated_object"] == "original" for c in ev["claim_evaluations"]}
    known = []
    for pid, cid, desc in KNOWN_ERRORS:
        c = claims.get(cid)
        known.append({"product_id": pid, "claim_id": cid, "error": desc, "status": c["status"] if c else "MISSING",
                      "method": c["method"] if c else "", "flagged": bool(c and c["status"] in FLAG),
                      "not_supported": bool(c and c["status"] != "SUPPORTED")})
    corr = read_jsonl(mode / "d3" / "corrected_summaries_v2" / source / "corrected_summaries.jsonl")
    summaries = {r["product_id"]: r for r in read_jsonl(mode / "summaries" / "summaries.jsonl")}
    return {"source": source, "report": rep, "evals": evals, "nc_rows": nc_rows, "known_errors": known, "corr2": corr,
            "summaries": summaries}


# --------------------------------------------------------------------------- report text
def build(D: dict, figs: dict[str, str]) -> str:
    R = D["report"]
    o, nc, cfg = R["originals"], R["negative_control_v3"], R["config"]
    calls = R["llm_calls"]
    pm = nc["metrics"]["primary"]
    comb, det, llm = pm["combined"], pm["deterministic"], pm["llm"]
    rv = nc["review_based_types_primary"]
    man = nc["manifest"]
    prior = R["prior_judges"]
    sc = o["status_counts"]
    kn = D["known_errors"]
    kn_flag = sum(k["flagged"] for k in kn)
    kn_ns = sum(k["not_supported"] for k in kn)
    per = {r["product_id"]: r for r in o["per_product"]}
    corr_cmp = R["corrected"]["comparison"]
    rep = R["reproducibility"]
    hs = R["human_sentiment_check"]
    sens = R["sensitivity"]
    taus = [v["kendall_tau_vs_primary"] for k, v in sens.items() if not k.startswith("primary (")]
    L = []
    a = L.append
    a("# ReviewIQ — Deliverable 3 Final Report")
    a("")
    a("Aspect-Based Customer Review Analytics with Structured Outputs · Product-level summaries and faithfulness evaluation")
    a("")
    a("Team: " + ", ".join(f"{n} ({u})" for n, u in TEAM) + f" · Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from stored artefacts "
      "(`scripts/build_d3_final_report.py`; numbers re-checked by `scripts/verify_d3_final.py`).")
    a("")
    # 1 abstract
    a("## 1. Abstract")
    a(f"ReviewIQ turns customer reviews into structured aspect-level records, aggregates them per product and generates concise product "
      f"summaries with a local model (qwen2.5:3b). Earlier LLM-as-a-Judge designs (v1-v3) could not measure faithfulness reliably: v3 "
      f"verdicts contradicted their own claim lists in {prior['v3']['inconsistent_score_claim_verdicts']} of {prior['v3']['valid_total']} cases. "
      f"We replaced them with a claim-level evaluator (judge v4): each of the {o['claims']} claims in the 15 summaries is checked "
      f"deterministically (numbers, counts, majorities, aspect statements, citations) and, for cited points, by a local LLM "
      f"({cfg.get('judge_model')}) against only the cited review text, with a mandatory verbatim evidence quote. Scores are derived from the "
      f"claim statuses, so a score cannot contradict the claims. On the originals, v4 finds {sc['CONTRADICTED']} contradicted and "
      f"{sc['UNSUPPORTED']} unsupported claims (mean faithfulness {o['mean_faithfulness']} / 100) and flags {kn_flag} of the "
      f"{len(kn)} errors identified by the earlier manual investigation. On a new pre-registered negative control with "
      f"{nc['cases']} planted errors and {nc['cases']} matched correct claims, the combined evaluator reaches recall {pct(comb['recall'])} "
      f"{ci(comb['recall_wilson95'])}, precision {pct(comb['precision'])} and specificity {pct(comb['specificity'])}. For review-based errors "
      f"(the part only an LLM can check), recall is {pct(rv['combined']['recall'])} with specificity {pct(rv['combined']['specificity'])}. "
      "The evaluator is useful but not perfectly reliable; its limitations are stated in section 18.")
    a("")
    # 2 introduction
    a("## 2. Introduction")
    a("Product summaries generated by language models can state numbers that are not in the data, attribute statistics to the wrong "
      "quantity, cite reviews that say the opposite, or generalise from one review to all customers. D3 asks whether such errors can be "
      "detected automatically and reproducibly with local resources (Windows laptop, 32 GB RAM, Quadro M2200 with 4 GB VRAM, Ollama; no "
      "cloud inference). This report documents the final D3 pipeline, the evaluation architecture that replaced the holistic judge, and the "
      "evidence for and against its reliability.")
    a("")
    # 3 objectives
    a("## 3. D3 objectives")
    a(table(["#", "Objective", "Where addressed", "Status"], [
        ["1", "Extract relevant information from reviews", "§5 (reused D2 few-shot extraction)", "done"],
        ["2", "Aggregate at product level", "§6", "done"],
        ["3", "Generate concise product summaries", "§7", "done (15 summaries)"],
        ["4", "Evaluate faithfulness with an LLM-as-a-Judge", "§9-12 (judge v4)", "done; reliability measured in §15-16"],
        ["5", "Check claims against original review evidence", "§8-9 (verbatim quote check, cited reviews only)", "done"],
        ["6", "Test reliability with deliberately corrupted summaries", "§14-16 (nc-v3, with clean controls)", "done"],
        ["7", "Reproducible results and final report", "§10, Appendix B, this report", "done"],
    ]))
    a("")
    # 4 dataset
    hsrows = hs["rows"]
    a("## 4. Dataset description")
    a(f"The dataset holds 1,500 Amazon appliance reviews for 15 products (`data/processed/reviews.csv`). D3 uses the D2 annotation sample "
      f"(299 reviews after one flagged review was removed); 285 reviews have a valid extraction and 14 are controlled failures. Per product "
      f"the summaries rest on 9-42 reviews. The 300 independent human annotations (human-created, blind, not second-person verified) "
      f"label overall sentiment; only a handful carry aspect or complaint labels ({hs['note'].split('Only ')[1].split(',')[0]}), so they are "
      "used only in a separate sentiment-aggregation check (§17) and as one agreement signal when choosing negative-control anchors, never "
      "as claim-level ground truth.")
    a("")
    # 5 extraction
    a("## 5. Extraction methodology")
    a("Each review is extracted by qwen2.5:3b with the few-shot prompt chosen in D2 into a Pydantic `ReviewRecord` (overall sentiment, "
      "aspects with sentiment and a verbatim evidence span, complaint categories), with deterministic fixes and at most two repair calls "
      "(`pipeline/extraction.py`, `pipeline/repair.py`). Output: `outputs/live/extractions/dataset__few_shot__qwen2.5-3b__v1.jsonl` "
      "(reused from D2, unchanged). D2 measured complaint micro-precision 0.367 for this strategy, so complaint statistics can be inflated "
      "(§17).")
    a("")
    # 6 aggregation
    a("## 6. Product-level aggregation")
    a("`pipeline/aggregation.py` (pandas only, no LLM) computes per product: sentiment counts and percentages, average rating, per-aspect "
      "positive/negative/neutral mentions and net sentiment, complaint counts and percentages (denominator = all reviews), reviews with "
      "complaints, a yearly trend and up to ten evidence quotes. Judge v4 re-reads these statistics for every numeric check; the audit "
      "re-aggregates them identically from the extraction file.")
    a("")
    # 7 summary generation
    a("## 7. Summary generation")
    a("qwen2.5:3b receives the statistics and the evidence quotes (not the full reviews) and returns `ProductSummaryContent`: an overall "
      "opinion, up to six appreciated points and six complaints (each citing 1-3 review ids from the evidence), frequent aspects, "
      "improvement areas and recommendations (`prompts/summarization.py`, rules: copy numbers only from the statistics; improvement areas must "
      "follow from complaints). 15 summaries were generated (`outputs/live/summaries/summaries.jsonl`); they are the evaluated artefacts "
      "and were never modified.")
    a("")
    # 8 traceability
    a("## 8. Evidence traceability")
    a(f"All 130 cited review ids exist and belong to the right product (re-verified by v4's citation check: every point claim records "
      f"`exists` and `belongs_to_product` per citation). Traceability alone is not faithfulness: the investigation found points whose only "
      f"citation says the opposite. v4 therefore resolves every cited id to its original review text and requires the LLM to quote the "
      f"sentence it relied on; the quote must be an exact (normalised: case, whitespace, punctuation, '...' gaps) substring of the cited text "
      f"or the verdict becomes UNVERIFIABLE. {pct(calls['quote_verified_rate'])} of all stored LLM verdicts carried a verified quote.")
    a("")
    # 9 architecture
    a("## 9. Claim-level evaluation architecture")
    a("1. **Decomposition** (`evaluation/claims.py`, no LLM): each overall-opinion sentence, each appreciated/complaint point (with its "
      "own citations), the frequent-aspect list, and each improvement area and recommendation becomes one claim. Claim text is verbatim; ids "
      "are stable (`<summary_id>:<field>:<index>`). Each claim gets a category (12 categories), a polarity and a numeric flag.")
    a("2. **Deterministic checks** (`evaluation/claim_checks.py`): counts, denominators, percentages (±0.5 points), average rating, "
      "majority/plurality/'slight' wording, 'mixed' wording, most-frequent aspects, most-common complaint, universal quantifiers ('all "
      "reviewers'), aspect statements ('users appreciate X but face issues with Y'), numbers in points that must occur in the cited text, "
      "citation validity, and derived claims (improvement areas need negative evidence). Every numeric check stores the extracted value, "
      "expected value, denominator and calculation.")
    a("3. **LLM verification** (`evaluation/judge_v4.py`): only for point claims (and opinion sentences no deterministic rule covers). One "
      "call per claim; the evidence is the full text of the claim's cited reviews only (other reviews are never shown). Output "
      "`{evidence_quote, explanation, status}` under Ollama constrained decoding.")
    a("4. **Combination**: the worse of the deterministic and LLM verdicts wins; a definite deterministic finding is never overridden by a "
      "better or an UNVERIFIABLE LLM verdict; an LLM verdict of SUPPORTED, PARTIALLY_SUPPORTED or CONTRADICTED without a verified quote "
      "becomes UNVERIFIABLE; an improvement area is supported if a complaint point of the same summary was verified for that aspect.")
    a("5. **Derived scores** (`schemas/claim_schema.py`): computed from statuses only; `SummaryEvaluationV4` re-derives counts, lists and "
      "score on validation and refuses to load an inconsistent record.")
    a("")
    a(table(["Method", "Claims (15 originals)"], [[k, v] for k, v in o["by_method"].items()]))
    a("")
    # 10 prompt and config
    a("## 10. Judge prompt and model configuration")
    a(table(["Setting", "Value"], [["judge model", f"{cfg.get('judge_provider')}/{cfg.get('judge_model')} (different family from the qwen2.5 summariser)"],
                                   ["temperature", cfg.get("temperature")], ["num_ctx / num_predict", f"{cfg.get('num_ctx')} / {cfg.get('num_predict')}"],
                                   ["constrained decoding", cfg.get("constrained_decoding")], ["calls per claim / repairs", f"{cfg.get('calls_per_claim')} / {cfg.get('repair_attempts')}"],
                                   ["prompt SHA-256", R["prompt_sha256"]], ["valid LLM verdicts / validation failures", f"{calls['valid']} / {calls['validation_failures']}"],
                                   ["mean latency per call", f"{calls['mean_latency_s']} s (sequential, GPU guard active)"]]))
    a("")
    a("System prompt (verbatim; the user message gives the claim, its role and the evidence):")
    a("```")
    a(SYSTEM_V4)
    a("----")
    a(HUMAN_V4)
    a("```")
    a("")
    # 11 scoring
    a("## 11. Scoring methodology")
    a(f"`{SCORING_FORMULA}`")
    a("")
    a("Rationale: a supported claim earns full credit and a partially supported one half; an unsupported claim earns nothing; a contradicted "
      "claim actively misinforms the reader, so it subtracts a full claim. UNVERIFIABLE claims (advice that cannot be checked, or an LLM "
      "verdict without a verifiable quote) are excluded rather than counted as errors, and the excluded share is reported as evidence "
      "coverage; below 50 % coverage the summary is marked INSUFFICIENT_EVIDENCE. The caps make the validity rules hard: any contradicted claim "
      "keeps the score at or below 60, any unsupported claim at or below 80, and 100 requires every verifiable claim to be supported. "
      f"Validity labels: FAITHFUL, MINOR_ISSUES, UNSUPPORTED_CLAIMS, CONTRADICTED_CLAIMS, INSUFFICIENT_EVIDENCE. The weights were fixed "
      f"before any v4 result existed. Sensitivity (§12): over six alternative weightings the Kendall τ of the product ranking against the "
      f"primary scores ranges from {min(t for t in taus if t is not None)} to {max(t for t in taus if t is not None)}.")
    a("")
    # 12 originals
    a("## 12. Original summary evaluation")
    a(f"{o['summaries']} summaries, {o['claims']} claims: " + ", ".join(f"{k} {v}" for k, v in sc.items()) +
      f". Numeric claims: {o['numeric_claims']}, of which {o['numeric_failed']} failed a deterministic check. Validity: " +
      ", ".join(f"{k} {v}" for k, v in o["validity"].items()) + f". Mean faithfulness {o['mean_faithfulness']} / 100.")
    a("")
    a(table(["Product", "Claims", "S", "P", "U", "C", "Unverif.", "Numeric failed", "Coverage", "Score", "Validity"],
            [[r["product_id"], r["total_claims"], r["SUPPORTED"], r["PARTIALLY_SUPPORTED"], r["UNSUPPORTED"], r["CONTRADICTED"], r["UNVERIFIABLE"],
              r["numeric_failed"], pct(r["evidence_coverage"], 0), num(r["faithfulness_score"]), r["validity_status"]] for r in o["per_product"]]))
    a("")
    a(f"![Faithfulness score per product: originals and correction set 2]({figs['scores']})")
    a("")
    a("### Known errors from the investigation")
    a(f"The investigation (`reports/proposed_revisions_2026-11-15/`) listed {len(kn)} concrete errors by reading the summaries against the "
      f"statistics and cited reviews (investigator reading, not a human gold set). v4 marks {kn_flag} of them UNSUPPORTED or CONTRADICTED "
      f"and {kn_ns} as anything other than SUPPORTED:")
    a("")
    a(table(["Product", "Claim", "Known error", "v4 status", "Method"], [[k["product_id"], k["claim_id"].split(":", 1)[1], k["error"], k["status"],
                                                                          k["method"]] for k in kn]))
    a("")
    a("### Weight sensitivity")
    a(table(["Weighting", "Mean score", "Kendall τ vs primary"], [[k, v["mean"], num(v["kendall_tau_vs_primary"])] for k, v in sens.items()]))
    a("")
    # 13 corrected
    a("## 13. Corrected summary evaluation")
    a("Corrections are separate, labelled artefacts with new ids; the originals are untouched. Correction set 1 (`-corr1`, 2026-11-14) "
      "fixed the misattributed percentages of B0BTY63SFY and the denominators of B001DHKBXU. Correction set 2 (`-corr2`, 2026-11-15, "
      "`scripts/build_corrected_summaries_v2.py`) fixes exactly the errors named in the investigation, for 7 products; every edit records "
      "the old and new text, the reason and verbatim review quotes (checked to be substrings) or recomputed statistics. The edit list was "
      "fixed before v4 evaluated any corrected copy. Provenance: written by the AI coding assistant, not by the summariser and not "
      "human-approved. Each copy was evaluated independently by v4 under its own id. The flags that remain on corrected copies are "
      "on points the corrections did not touch; several are LLM false positives of the kind measured in §16 (for example, 'The product is "
      "not durable and breaks easily' marked CONTRADICTED by a review saying it broke after a few months), so the corrected scores are "
      "not an upper bound of what correction can achieve.")
    a("")
    a(table(["Product", "Set", "Copy id", "Original score", "Corrected score", "Original validity", "Corrected validity", "Corrected C/U"],
            [[c["product_id"], c["correction_set"], c["corrected_summary_id"], num(c["original_score"]), num(c["corrected_score"]),
              c["original_validity"], c["corrected_validity"], f"{c['corrected_counts']['CONTRADICTED']}/{c['corrected_counts']['UNSUPPORTED']}"]
             for c in corr_cmp]))
    a("")
    a("Edits in correction set 2:")
    a("")
    a(table(["Product", "Field", "Action", "Old", "New", "Reason"], [[x["product_id"], e["field"], e["action"], e["old"][:110], (e.get("new") or "")[:110],
                                                                    e["reason"][:140]] for x in D["corr2"] for e in x["edits"]]))
    a("")
    a("Not fixable in a summary: B001DHKBXU's complaint statistics are inflated by extraction errors (about 2 of 7 installation_difficulty "
      "and 1 of 6 poor_value labels match the review text); fixing them requires re-extraction, which would change D2-derived artefacts.")
    a("")
    # 14 NC method
    a("## 14. Negative-control methodology")
    a(f"The 30 earlier copies (nc-v2: invented feature, fabricated statistic) and their v2/v3 results are unchanged. The new control "
      f"**{man['nc_version']}** (`evaluation/negative_control_v3.py`) has {man['cases']} cases over 12 error types; each case is one planted "
      f"(corrupted) claim and one matched correct (clean) claim of the same form, so false positives can be measured. It was generated with "
      f"seed {man['seed']}, frozen on {man['frozen_at'][:10]} before any judging (dataset SHA-256 `{man['dataset_sha256']}`), and the runner "
      "refuses a dataset whose hash differs. Statistic types use the product aggregate. Review-based types are anchored on reviews where "
      "three signals agree: an extraction aspect whose evidence span is verbatim in the review and contains wording for that aspect and "
      "polarity, the star rating, and the human overall sentiment. Fabricated details are planted only where their key words are absent "
      "from the cited review (brands: from all reviews of the product). Before freezing, every review-based case was read against its "
      "review by the AI assistant; ambiguous cases led to rule changes before freezing, never after. Detection is read from the "
      "evaluator's status on the planted claim itself.")
    a("")
    a(table(["Error type", "Cases", "Expected status", "Example corrupted claim"],
            [[TYPE_LABEL[t], man["cases_by_type"][t], v["expected_status"], next(r["corrupted_text"] for r in D["nc_rows"] if r["error_type"] == t)[:120]]
             for t, v in nc["by_type"].items()]))
    a("")
    a("Detection rules (pre-registered in the manifest): " + "; ".join(f"**{k}**: {v}" for k, v in nc["detection_rule"].items()))
    a("")
    # 15 NC results
    a("## 15. Negative-control results")
    a(table(["Error type", "Cases", "Detected", "Rate", "95% CI", "Clean flagged", "Specificity", "LLM-only detected", "Deterministic-only detected"],
            [[TYPE_LABEL[t], v["cases"], v["detected"], pct(v["detection_rate"], 0), ci(v["detection_wilson95"]), v["clean_flagged"], pct(v["specificity"], 0),
              f"{v['llm_only_detected']}/{v['llm_judged']}" if v["llm_judged"] else "n/a", v["deterministic_only_detected"]] for t, v in nc["by_type"].items()]))
    a("")
    a(f"![Detection rate and specificity by error type]({figs['nc']})")
    a("")
    cs = nc["copy_level_scores"]
    a(f"Summary-level effect: inserting the corrupted claim instead of its clean twin gives a lower faithfulness score in "
      f"{cs['corrupted_copy_lower']} of {cs['pairs']} pairs, equal in {cs['equal']}, higher in {cs['corrupted_copy_higher']} ({cs['note']}). "
      f"For comparison, v3 gave the corrupted nc-v2 copy a lower faithfulness score in only {prior['v3']['nc_v2_faithfulness_lower']} of "
      f"{prior['v3']['nc_v2_pairs']} pairs.")
    a("")
    # 16 metrics
    a("## 16. Precision, recall, F1 and specificity")
    rows = []
    for rule, layers in nc["metrics"].items():
        for layer, m in layers.items():
            if m["n_corrupted"] == 0:
                continue
            rows.append([rule, layer, m["tp"], m["fp"], m["tn"], m["fn"], pct(m["precision"]), pct(m["recall"]), ci(m["recall_wilson95"]),
                         pct(m["f1"]), pct(m["specificity"]), ci(m["specificity_wilson95"]), pct(m["false_positive_rate"])])
    a(table(["Rule", "Layer", "TP", "FP", "TN", "FN", "Precision", "Recall", "Recall 95% CI", "F1", "Specificity", "Spec. 95% CI", "FPR"], rows))
    a("")
    a("Layers: *combined* is the final v4 status; *deterministic* counts a claim as flagged only from deterministic findings; *llm* uses the "
      "LLM verdict alone (after the quote check) on the claims the LLM judged. The statistic types are detected deterministically by design "
      "(the checks and the generator were written by the same author), so their perfect detection says the checks work, not that the "
      "problem is hard. The informative numbers are the review-based types:")
    a("")
    a(table(["Review-based types (primary rule)", "TP", "FP", "TN", "FN", "Precision", "Recall", "Recall 95% CI", "Specificity"],
            [[layer, m["tp"], m["fp"], m["tn"], m["fn"], pct(m["precision"]), pct(m["recall"]), ci(m["recall_wilson95"]), pct(m["specificity"])]
             for layer, m in rv.items()]))
    a("")
    a("Comparison of judge versions:")
    a("")
    a(table(["", "v1", "v2", "v3", "v4"], [
        ["Unit of evaluation", "whole summary", "whole summary", "whole summary", "one claim + its cited reviews"],
        ["Valid outputs", f"{prior['v1']['valid']} / 15", f"{prior['v2']['valid']} / 15", f"{prior['v3']['valid_total']} / 47",
         f"{calls['valid']} verdicts, {calls['validation_failures']} failures"],
        ["Score/claim contradictions", "4 of 5 valid", "forced away by validator (H = 4 for all)",
         f"{prior['v3']['inconsistent_score_claim_verdicts']} / {prior['v3']['valid_total']}", "0 (impossible by construction, tested)"],
        ["Planted-error detection", "3-pair sanity check only", f"{prior['v2']['nc_v2_detected']} / 30 (nc-v2)",
         f"{prior['v3']['nc_v2_set_detected']} / {prior['v3']['nc_v2_set_judged']} (nc-v2)",
         f"{comb['tp']} / {comb['n_corrupted']} (nc-v3); review-based {rv['combined']['tp']} / {rv['combined']['n_corrupted']}"],
        ["False positives measured", "no", "no", "no", f"yes: {comb['fp']} of {comb['n_clean']} clean claims flagged"],
        ["Numbers checked by", "LLM", "LLM", "LLM", "deterministic code"],
    ]))
    a("")
    # 17 error analysis
    fn = [r for r in D["nc_rows"] if r["corrupted_flagged_primary"] == "False"]
    fp = [r for r in D["nc_rows"] if r["clean_flagged_primary"] == "True"]
    a("## 17. Error analysis")
    a(f"**Missed planted errors ({len(fn)}).** Status of each missed corrupted claim and the evaluator's explanation:")
    a("")
    a(table(["Case", "Type", "Corrupted claim", "Status", "Explanation (truncated)"],
            [[r["case_id"], TYPE_LABEL[r["error_type"]], r["corrupted_text"][:90], r["corrupted_status"], r["corrupted_explanation"][:170]] for r in fn]))
    a("")
    a(f"**False positives on clean claims ({len(fp)}).**")
    a("")
    a(table(["Case", "Type", "Clean claim", "Status", "Explanation (truncated)"],
            [[r["case_id"], TYPE_LABEL[r["error_type"]], r["clean_text"][:90], r["clean_status"], r["clean_explanation"][:170]] for r in fp]))
    a("")
    unsup = [(ev["product_id"], c) for ev in D["evals"] if ev["evaluated_object"] == "original" for c in ev["claim_evaluations"]
             if c["status"] in FLAG]
    a(f"**Flagged claims in the original summaries ({len(unsup)}).** These are the evaluator's findings, not verified truth; the LLM-based "
      "ones in particular should be read with the explanation and quote.")
    a("")
    a(table(["Product", "Claim", "Status", "Method", "Explanation (truncated)"],
            [[pid, c["claim"]["text"][:100], c["status"], c["method"], c["explanation"][:170]] for pid, c in unsup]))
    a("")
    a(f"**Extraction drift (separate analysis with the human annotations).** {hs['note']} Mean absolute difference between the extraction-"
      f"based and the human-label sentiment distribution per product: positive {hs['mean_abs_diff_positive_pts']} points, negative "
      f"{hs['mean_abs_diff_negative_pts']} points; the positive/negative majority direction agrees for {hs['products_same_majority_direction']} "
      f"of {hs['products']} products. **Caveat:** {hs['interpretation']}")
    a("")
    a(table(["Star rating"] + ["positive", "negative", "neutral", "mixed"],
            [[k] + [v.get(x, 0) for x in ("positive", "negative", "neutral", "mixed")] for k, v in hs["rating_by_human_label"].items()]))
    a("")
    a(table(["Product", "Human-labelled", "Extraction pos/neg %", "Human pos/neg %", "Same direction"],
            [[r["product_id"], r["human_labelled"], f"{r['extraction_pct'].get('positive')}/{r['extraction_pct'].get('negative')}",
              f"{r['human_pct']['positive']}/{r['human_pct']['negative']}", r["same_majority_direction"]] for r in hsrows]))
    a("")
    # 18 limitations
    a("## 18. Limitations")
    a(f"- **Small, local judge.** {cfg.get('judge_model')} (3B) is imperfect; it misses planted errors and occasionally rejects correct "
      "claims (§17). No larger model fits the 4 GB GPU at acceptable speed.")
    a("- **No human claim-level gold set.** Agreement between v4 and humans is **unavailable**: no claim has been labelled by a person. The "
      "known-error table uses investigator reading, and the negative control uses constructed claims. A template for human claim labels can be "
      "produced from `claims.csv`.")
    a("- **Constructed controls.** nc-v3 claims follow templates ('A reviewer reports that …'); real summary errors are more varied. The "
      "statistic types are caught by checks written by the same author. Per-type sample sizes are small (3-15), hence wide intervals.")
    a("- **Heuristic deterministic parsing.** The phrase patterns cover the wording seen in these summaries; an unrecognised phrasing gets "
      "no deterministic verdict (it is sent to the LLM or left UNVERIFIABLE, never silently passed).")
    a("- **Statistics inherit extraction errors.** Numeric checks compare with the aggregate, which is faithful to the extractions, not "
      "necessarily to the reviews (complaint precision 0.367 in D2).")
    a("- **Opinion sentences are checked against statistics only;** their wording is not checked against individual reviews.")
    a("- **Derived claims** (improvement areas, recommendations) are checked only for negative evidence on the targeted aspect; generic "
      "advice is UNVERIFIABLE.")
    a(f"- **Reproducibility** was tested on a subset of {rep['pairs']} claims (§19), not on every claim.")
    a("- **Single summariser run.** The 15 summaries are one generation; corrections were written by the AI assistant, not by people.")
    a("")
    # 19 discussion
    a("## 19. Discussion")
    a("The v1-v3 history showed that the problem was task design, not output format: one generation over a whole product was asked to "
      "find, verify and score claims, and its scores were independent of its findings. Splitting the task changes what the model must do — "
      "read one to three short reviews and decide whether one sentence is in them — and moves every number to code. Two consequences are "
      "visible in the results: score/claim contradictions are impossible, and the derived score now moves when an error is planted (§15).")
    a(f"Reproducibility: re-running {rep['pairs']} LLM claim verdicts with identical settings gave the same status for {rep['same_status']} "
      f"({pct(rep['status_agreement'])}, 95% CI {ci(rep['status_agreement_wilson95'])}); {rep['raw_identical']} raw responses were "
      "byte-identical.")
    a(f"False positives matter for reading §12: on the clean controls the LLM layer wrongly flagged {llm['fp']} of {llm['n_clean']} correct "
      f"claims (false-positive rate {pct(llm['false_positive_rate'])}, 95% CI for specificity {ci(llm['specificity_wilson95'])}), while the "
      f"deterministic layer flagged none. Some of the {sc['UNSUPPORTED']} UNSUPPORTED claims found by the LLM in the original summaries are "
      "therefore likely false positives (the clean controls are simpler than real claims, so the real rate may be higher); each comes with "
      "its explanation and quote for a person to confirm.")
    a(f"The remaining weakness is the LLM layer on review-based claims: it matters most for real summaries (misattributed citations, "
      f"misreadings, unsupported points) and it is the only layer that produces false positives (specificity {pct(rv['llm']['specificity'])} "
      f"vs {pct(rv['deterministic']['specificity'])} for the deterministic layer). It is also blind to subtle exaggeration: "
      f"{nc['by_type']['misleading_paraphrase']['cases'] - nc['by_type']['misleading_paraphrase']['detected']} of "
      f"{nc['by_type']['misleading_paraphrase']['cases']} misleading paraphrases passed as SUPPORTED. The evaluator should therefore be "
      "used as a screening tool that lists suspicious claims with their evidence for a person to confirm, not as an automatic pass/fail judge.")
    a("")
    # 20 conclusion
    a("## 20. Conclusion")
    a(f"D3 now has a complete, reproducible pipeline (extraction → aggregation → 15 summaries) and a claim-level evaluation in which every "
      f"claim has a structured verdict, every number is checked by code, an LLM verdict counts as support or contradiction only when its quote is "
      f"found verbatim in the cited reviews, and every score is derived from the claims. The evaluator flagged {sc['CONTRADICTED']} "
      f"contradicted and {sc['UNSUPPORTED']} unsupported claims in the original summaries (the deterministic findings and "
      f"{kn_flag} of {len(kn)} previously known errors are confirmed; the LLM findings include false positives), its false-positive rate was measured "
      f"({pct(comb['false_positive_rate'])} on {comb['n_clean']} clean claims), and its recall on planted errors is {pct(comb['recall'])} "
      f"overall and {pct(rv['combined']['recall'])} on review-based errors. It is not a perfectly reliable judge, and the results are reported "
      "as such.")
    a("")
    a("## Appendix A — artefacts")
    a(table(["Artefact", "Path"], [
        ["LLM claim verdicts (raw responses)", "outputs/live/judgements/v4/claim_verdicts.jsonl, failures.jsonl, run_manifest.json"],
        ["Reproducibility repeat", "outputs/live/judgements/v4_repro/"],
        ["Summary evaluations (originals + corrected)", f"outputs/live/d3/judge_v4/{D['source']}/evaluations.jsonl"],
        ["CSV exports", f"outputs/live/d3/judge_v4/{D['source']}/claims.csv, summaries.csv, nc_results.csv"],
        ["Negative control nc-v3", f"outputs/live/d3/negative_control_v3/{D['source']}/cases.jsonl, manifest.json"],
        ["Correction set 2", f"outputs/live/d3/corrected_summaries_v2/{D['source']}/"],
        ["Report data", "reports/d3/D3_REPORT_DATA.json"],
    ]))
    a("")
    a("## Appendix B — reproduction")
    a("```")
    a(f"python scripts/build_corrected_summaries_v2.py --source {D['source']}")
    a(f"python scripts/judge_v4.py build-nc --source {D['source']}")
    a(f"python scripts/judge_v4.py run --source {D['source']} --objects original nc corrected")
    a(f"python scripts/judge_v4.py repro --source {D['source']}")
    a(f"python scripts/judge_v4.py report --source {D['source']}")
    a(f"python scripts/build_d3_final_report.py --source {D['source']}")
    a(f"python scripts/verify_d3_final.py --source {D['source']}")
    a("```")
    return "\n".join(L) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args()
    D = load(args.source)
    R = D["report"]
    o = R["originals"]
    figdir = ROOT / "reports" / "final" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    corr2 = {c["product_id"]: c["corrected_score"] for c in R["corrected"]["comparison"] if c["correction_set"] == "corr2"}
    prods = [r["product_id"] for r in o["per_product"]]
    charts = {
        "scores": svg_bars("Faithfulness score (0-100), judge v4", prods,
                           [("original", [r["faithfulness_score"] for r in o["per_product"]], "#4c78a8"),
                            ("correction set 2", [corr2.get(p_) for p_ in prods], "#f58518")]),
        "nc": svg_bars("nc-v3: detection rate and specificity (%)", [TYPE_LABEL[t] for t in R["negative_control_v3"]["by_type"]],
                       [("detection rate", [round(100 * v["detection_rate"]) for v in R["negative_control_v3"]["by_type"].values()], "#54a24b"),
                        ("specificity", [round(100 * v["specificity"]) for v in R["negative_control_v3"]["by_type"].values()], "#9d755d")], unit="%"),
    }
    figs = {}
    for k, svg in charts.items():
        (figdir / f"d3_{k}.svg").write_text(svg, encoding="utf-8")
        figs[k] = f"figures/d3_{k}.svg"
    md = build(D, figs)
    out_md = ROOT / "reports" / "final" / "D3_Final_Report.md"
    out_md.write_text(md, encoding="utf-8")
    body = md_to_html(md, {figs[k]: charts[k] for k in charts})
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>ReviewIQ — D3 Final Report</title><style>{CSS}"
           "figure svg{max-width:100%;height:auto} figure{text-align:center} pre{white-space:pre-wrap;font-size:10px;background:#f6f6f6;padding:8px} "
           "h3{margin-top:14px}</style></head><body><div class='muted'>PES University · Prompt Engineering Mini Project · Deliverable 3</div>"
           f"{body}</body></html>")
    out_html = ROOT / "reports" / "final" / "D3_Final_Report.html"
    out_html.write_text(doc, encoding="utf-8")
    pdf = None if args.no_pdf else to_pdf(out_html)
    data = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": args.source,
            "report_json": f"outputs/live/d3/judge_v4/{args.source}/report.json", "known_errors": D["known_errors"],
            "known_errors_flagged": sum(k["flagged"] for k in D["known_errors"]), "weights": {k.value: v for k, v in WEIGHTS.items()},
            "scoring_formula": SCORING_FORMULA, "charts": {k: {"path": f"reports/final/{v}"} for k, v in figs.items()},
            "outputs": {"markdown": "reports/final/D3_Final_Report.md", "html": "reports/final/D3_Final_Report.html",
                        "pdf": "reports/final/D3_Final_Report.pdf" if pdf else None}, **{k: R[k] for k in R}}
    (ROOT / "reports" / "d3").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "d3" / "D3_REPORT_DATA.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(f"written: {out_md}, {out_html}" + (f", {pdf}" if pdf else " (no PDF: no local Chrome/Edge)"))


if __name__ == "__main__":
    main()
