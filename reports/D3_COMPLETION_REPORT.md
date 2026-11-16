# D3 completion report — Extract → Aggregate → Summarise + LLM-as-a-Judge (updated 2026-11-14)

> **Revision 2026-11-15 — superseded as the final D3 result.** The final D3 evaluation is the claim-level judge v4 with negative
> control nc-v3 and correction set 2: see [`final/D3_Final_Report.md`](final/D3_Final_Report.md) (also `.html` / `.pdf`), data in
> `d3/D3_REPORT_DATA.json`, and the investigation that motivated it in `proposed_revisions_2026-11-15/`. The text below is kept
> unchanged as the 2026-11-14 record of judges v1–v3. Two statements in it are now known to be too weak or outdated. First, "13/15 pass
> the evidence audit" checks ids, numbers and word overlap, not meaning; v4 finds factual errors in more summaries. Second, the `-corr1`
> copies were later judged, by v3 and v4.

Base commit `580adb2` (branch `main`). Nothing committed or pushed. All inference ran on local Ollama: summariser
`qwen2.5:3b`, judge `llama3.2`. Every figure below was re-checked on 2026-11-14 against the saved files by
independent scripts that do not use the project's own metric code.

Detailed reports (single source for each topic; this file only summarises them):
- `D3_JUDGE_EVALUATION.md` — judge v1/v2 results and judge quality
- `D3_NEGATIVE_CONTROL.md` — planted-claim experiment
- `D3_EVIDENCE_AUDIT.md` — deterministic audit of the 15 summaries

## 1. Architecture

```
reviews (299-review evaluation sample)
  └─ 1. EXTRACT    qwen2.5:3b few-shot, Pydantic-validated, repair loop  -> outputs/live/extractions/dataset__few_shot__qwen2.5-3b__v1.jsonl
                   (reused from D2 run 20261111-030912_qwen2.5-3b: 285 valid, 14 controlled failures)
  └─ 2. AGGREGATE  pandas only (pipeline/aggregation.py): sentiment %, aspect stats, complaint %, evidence quotes with review ids
  └─ 3. SUMMARISE  qwen2.5:3b from statistics + evidence only; every point must cite review ids of that product
                   -> outputs/live/summaries/summaries.jsonl (15 summaries, unchanged originals)
  └─ 4. CHECK      deterministic (no LLM): evidence audit (evaluation/d3_audit.py, scripts/audit_d3.py)
  └─ 5. JUDGE      llama3.2: v1 (original, preserved) and v2 (flat schema + constrained decoding), stored separately
                   -> outputs/live/judgements/{judgements.jsonl, judge_failures.jsonl, v2/judgements.jsonl}
  └─ 6. NEGATIVE CONTROL  one planted unsupported claim per corrupted copy -> outputs/live/d3/judge_negative_control_v2/
```

## 2. Status per product (verified 2026-11-14)

| Product | Summary | Judge v1 | Judge v2 (F/C/K/H, claims) | Evidence audit | Negative control (feature / statistic named?) |
|---|---|---|---|---|---|
| B001DHKBXU | yes | failed | 3/4/2/4, 2 | **misleading denominators (7 counts "out of 15" of 19 reviews)** (corrected copy available) | no / yes |
| B001ONHE8S | yes | ok | 5/5/5/4, 1 | verified | no / yes |
| B001TH7GZU | yes | failed | 3/4/2/4, 2 | verified | yes / no |
| B0081N3YW4 | yes | failed | 3/4/3/4, 1 | verified | no / **no** (91 % used as fact in a reason) |
| B008YDTFYW | yes | ok | 3/4/4/4, 1 | verified | no / yes |
| B00B0ZMNOU | yes | failed | 3/4/2/4, 2 | verified | yes / yes |
| B00DM8K1NS | yes | ok | 3/4/4/4, 2 | verified | yes / yes |
| B00LHRAGHW | yes | failed | 3/4/2/4, 2 | verified | no / yes |
| B01FZJXITU | yes | failed | 4/3/3/4, 2 | verified | yes / no |
| B01LZHJF55 | yes | failed | 3/4/3/4, 2 | verified | yes / yes |
| B0759KR8JX | yes | failed | 4/4/3/4, 1 | verified (40 % quoted from a cited review) | yes / no |
| B07XG8QQL5 | yes | ok | 2/3/2/4, 3 | verified ids/statistics; 1 weakly traceable point | yes / no |
| B09WWM1WLS | yes | ok | 3/3/2/4, 1 (**placeholder evidence ids**) | verified | yes / yes |
| B0BTY63SFY | yes | failed | 4/4/4/4, 1 | **misattributed 80 % and 67 %** (corrected copy available) | yes / no |
| B0C5G1WGBN | yes | failed | 3/4/2/4, 3 | verified | yes / no |

All judge scores in this table belong to the **original** summaries. The corrected copies of B001DHKBXU and
B0BTY63SFY carry no v1/v2 scores; they have their own, separate judge v3 judgements (§3), and no score is
transferred between an original and its copy.

## 3. Results

| Item | Result |
|---|---|
| Product summaries | **15 / 15** exist (unchanged originals) |
| Evidence: cited ids exist and belong to the product | **130 / 130** |
| Evidence: statistics identical to a fresh re-aggregation | **15 / 15** |
| Evidence: percentages sourced | 8 / 10; **B0BTY63SFY: 80 % and 67 % misattributed.** 80 % is the positive share of performance mentions in the aspect statistics (8/10), stated as "80% of the reviews" (actual 86.7 %); 67 % is 2 of the 3 reviews with complaints, stated as "67% of negative reviews" (actual 2 of 2) |
| Evidence: count statements ("N out of M reviews") — check added 2026-11-14 | 10 found; 3 correct; **7 in B001DHKBXU use the 15 complaining reviews as denominator although the product has 19 reviews** |
| Evidence: summary points traceable / partial / weak | 41 / 41 / **1** (B07XG8QQL5) |
| Summaries fully verified (ids + statistics + sourced percentages + correct counts) | **13 / 15** (was reported as 14/15 before the count check) |
| Judge v1 | 5 valid / 10 failed (preserved unchanged) |
| **Judge v2** | **15 / 15 valid, 0 failed, 0 pending**. Means: F 3.27, C 3.87, K 2.87, H 4.00 |
| v2 quality | Schema-valid 15/15. **H = 4 for all 15.** 17 / 26 flagged claims likely legitimate content; 6 / 26 not in the summary; 1 judgement cites placeholder ids `review_id_1/2`; **7 / 15 claim lists produced only after the validator rejected a claim-free answer**; 9 / 15 list 2–3 claims at H = 4 (rubric: 4 = one minor detail); missed the real misattributed 80 % / 67 % |
| **Judge v3 (separate run, 2026-11-14; one call, no repair, raw kept)** | 47 / 47 valid, 0 validation failures. Originals: F 2.47, C 3.27, K 2.60; hallucination score contradicts the judge's own (empty) claim list in **12 / 15** (reported as None, not repaired). Corrected copies judged separately (scores never transferred): B001DHKBXU identical to its original, B0BTY63SFY F +1 / K −1; no claims listed on either. v3 negative control **5 / 30** detected (v2: 18 / 30) |
| **Negative control (v2), all 15 products, 30 planted claims** | Planted claim named **18 / 30 (60 %, 95 % CI 42–75 %)**, **missed 12**. Invented feature 10/15, fabricated statistic 8/15. Faithfulness lower 2/30, hallucination lower 2/30 (higher 2/30). Keyword false positives on the 15 originals 0/30 checks — but all 15 originals list ≥ 1 "unsupported" claim |

## 4. Changes on 2026-11-14

1. **Negative control completed:** 16 remaining judge calls in a detached process, with the watchdog active and its
   thresholds unchanged; 0 judge failures. This was the only inference of the day.
2. **Detection-rule defect fixed** (`evaluation/negative_control.py`). Detection used to match the claim's `reason` as
   well. In B0081N3YW4 the judge wrote "The actual statistic is 91 % of reviewers…": it accepted the planted number as
   fact, and that counted as a detection. Detection now requires the keyword in a claim.
   - Recomputed from the stored judgements, with no re-run: 18 / 30 (previously 19 / 30).
   - `pairs.jsonl` is unchanged; the earlier summary is kept as `summary_previous_rule_claims_and_reasons.json`.
3. **Semantic review** (`scripts/nc_semantic_review.py`): all 30 judgements read in full, in separate categories. It
   is a Claude (model-assisted) assessment, **not human ground truth**, and is reported separately from the keyword
   rate (`D3_NEGATIVE_CONTROL.md` §3b).
4. **Evidence audit extended** with a count-statement check (`evaluation/d3_audit.py`). The 2026-11-13 audit files are
   kept with a date suffix. B0BTY63SFY's 80 % / 67 % were re-traced: they are **misattributed** ratios (wrong
   denominators), not invented numbers.
5. **Separately labelled corrected copies** of B001DHKBXU and B0BTY63SFY (`scripts/build_corrected_summaries.py`,
   deterministic, no LLM):
   - Both pass the audit, and the originals are byte-identical.
   - At creation they were unjudged. Later on 2026-11-14 they received their own **judge v3** judgements, stored as `evaluated_object = corrected` under the corrected ids; v1/v2 scores refer to the originals only.
6. **Summary Evaluation page:** the negative-control panel now shows
   - an "all planted claims" row with missed count and Wilson CI;
   - a limitation callout;
   - the reason-only case;
   - the list of missed claims;
   - the semantic review, labelled as model-assisted;
   - the narrow-control caveat;
   - the corrected copies (v1/v2 scores belong to the originals; their own v3 scores are in the Judge v3 section).
7. **Tests added:** `tests/test_judge_v2.py` +3, `tests/test_d3_audit.py` +2.

## 5. Acceptance criteria

| # | Criterion | Implementation | Execution | Evidence / quality |
|---|---|---|---|---|
| 1 | Extract → aggregate → summarise pipeline works | done | done (15 summaries) | statistics reproduce exactly (15 / 15) |
| 2 | All 15 summaries exist | done | done | – |
| 3 | Summary evidence verified | done (ids, statistics, percentages, counts, traceability) | done (all 15) | **Not met for 2 of 15 original summaries:** B0BTY63SFY (misattributed %) and B001DHKBXU (misleading denominators); 1 weak point in B07XG8QQL5. Corrected copies exist, but the evaluated originals are unchanged |
| 4 | All 15 have valid judge results, or failures are reported | done | v2 15 / 15; v3 15 / 15 (+ 2 corrected copies, 30 NC copies); v1 5 ok / 10 failed (reported) | v2 outputs are schema-valid but contain placeholder ids (1), untraceable claims (6 / 26) and validator-induced claims (7 / 15) |
| 5 | Negative control performed | done | done: 15 products, 30 planted claims (v2), same 30 re-judged by v3 | v2 keyword rule 18 / 30 detected, **12 missed**; v3 5 / 30; n = 15 (wide CI) |
| 6 | Judge reliability evaluated honestly | done | done | **The judge is not reliable** (v2 and v3): scores barely respond to planted errors or corrections; v2 H = 4 for every original; v3 score contradicts its claim list in 37 / 47 |
| 7 | UI displays the correct saved results | done | verified on real data, read-only | – |
| 8 | Tests pass | done | 269 passed, 1 skipped | – |

**Verdict: D3 is implemented and fully executed, but NOT fully compliant on evidence and reliability.**
- Two of the 15 evaluated summaries fail the evidence audit.
- The LLM judge is shown to be an unreliable faithfulness measure, in v2 and in the no-repair v3 run.

Both are reported as findings, not hidden. The corrected copies pass the deterministic audit and now have their own
v3 judgements; the judge does not distinguish them from the originals. Closing the reliability gap needs a better
(larger or human-calibrated) judge.

## 6. Reproduce (no LLM unless marked)

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
.venv\Scripts\python.exe scripts\audit_d3.py --source dataset__few_shot__qwen2.5-3b__v1
.venv\Scripts\python.exe scripts\judge_negative_control.py --source dataset__few_shot__qwen2.5-3b__v1 --report-only
.venv\Scripts\python.exe scripts\build_manual_inspection.py --source dataset__few_shot__qwen2.5-3b__v1
# LLM (resumable; with everything stored these make 0 new calls)
$env:LLM_MAX_OUTPUT_TOKENS="2048"
.venv\Scripts\python.exe scripts\run_judge.py --source dataset__few_shot__qwen2.5-3b__v1
.venv\Scripts\python.exe scripts\judge_negative_control.py --source dataset__few_shot__qwen2.5-3b__v1
Remove-Item Env:LLM_MAX_OUTPUT_TOKENS
```
