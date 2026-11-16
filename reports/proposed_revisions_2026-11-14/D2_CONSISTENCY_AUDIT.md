# D2 consistency audit (read-only), 2026-11-14

**Verdict: NEEDS ATTENTION.** Both metric sets are correct and reproducible, and the D2 report is internally
consistent. But two different evaluations are shown side by side without explanation, and several documents still
describe the reference approval as not yet done.

Nothing was modified: no prediction, metric, annotation or report file was changed. Proposed corrections are in this
folder only.

## 1. Source of the discrepancy

| | Report (`D2_FINAL_REPORT.md`) | Dashboard values (0.4435 / 0.6322 / 0.4276) |
|---|---|---|
| File | `outputs/live/experiments/20261111-030912_qwen2.5-3b/metrics.json` (written 2026-11-12 03:46 UTC) | `outputs/live/experiments/20261111-030912_qwen2.5-3b/approved_reference/metrics.json` (written 2026-11-14 16:55 UTC) |
| Produced by | `evaluation/prompt_comparison.py::evaluate_run` | `evaluation/reference_evaluation.py::evaluate_reference` (button *Evaluate against approved references*) |
| Reference set | `annotations.json`, status `complete` → 299 labels (flagged `R5cfb12ef75b4` excluded) | the approved references in `data/annotations/reference_adjudication.json` → 300 labels |
| Approvals | – | 299 bulk-approved by "Rachit" at 16:48:12 UTC; `R5cfb12ef75b4` individually approved by "Rachit" at 16:53:32 UTC (`original_status: flagged`, provenance `ai_assisted`, decision `approved`, label unchanged) |
| Predictions | the 299 original predictions per strategy (2026-11-11) | the same 299 + 1 supplementary prediction per strategy for `R5cfb12ef75b4` (`supplementary/`, 2026-11-14 16:18–16:20 UTC, `qwen2.5:3b`, prompt v1, 0 repairs) |
| Shown on the dashboard | Prompt Experiments: "Reference labels 299 / 299", strategy cards ("Best macro F1 0.603"), *Performance* / *Confusion* / *Per-class* / *Efficiency* / *Run details* tabs | Prompt Experiments: the **D2 reference evaluation** panel (n = 300, tagged "D2 complete"); Overview "D2 · Approved reference labels 300 / 300" |

**Mechanism.** The one extra review has the reference label **neutral** (AI-assisted, confidence 0.30, marked
"DEBATABLE"; the text is garbled). **All three strategies predicted neutral for it.** It becomes the 6th neutral
example and a correct one. Because neutral is tiny, its F1 jumps, and macro-F1 weights neutral equally with the
large classes:

| Strategy | Neutral F1, 299 → 300 | Macro-F1, 299 → 300 | Accuracy, 299 → 300 | Weighted F1, 299 → 300 |
|---|---|---|---|---|
| Zero-shot | 0.0476 → 0.0698 | 0.4380 → **0.4435** (+0.0055) | 0.6120 → 0.6133 | 0.6797 → 0.6780 |
| Few-shot | 0.1667 → 0.2857 | 0.6025 → **0.6322** (+0.0297) | 0.8127 → 0.8133 | 0.8170 → 0.8172 |
| CoT | 0.1010 → 0.1188 | 0.4232 → **0.4276** (+0.0044) | 0.5753 → 0.5767 | 0.6461 → 0.6446 |

Positive, negative and mixed rows are identical in both evaluations. The ranking is unchanged.

**Ruled out:**
- *A different run:* both use run `20261111-030912_qwen2.5-3b`.
- *A different metric calculation:* both use the same metric code and denominators (all reviews, failures scored as
  wrong).
- *A stale or wrongly loaded result:* the panel's stale check passes (300 accepted at evaluation and now).

The difference is the reference set: 299 versus 300, by the inclusion of the formerly flagged review.

**Independent verification.** Both sets were recomputed with the stdlib metric function in
`scripts/build_d2_report.py` (no project metric code), directly from the raw prediction files and
`annotations.json`. 299 reviews reproduce `metrics.json`; 300 reviews (with the supplementary rows) reproduce
`approved_reference/metrics.json`. Accuracy, macro P/R/F1, weighted F1, per-class F1 and the confusion matrices
are identical in both cases. The 300 evaluated ids are unique and include `R5cfb12ef75b4`.

**Comparability of the supplementary predictions:**
- Same model, temperature 0, prompt version v1. The extraction prompts are unchanged since the run's commit
  `045f16b`.
- Pipeline code changed since `045f16b` only in outage handling, cache seeding and the judge normaliser
  (`git diff 045f16b -- pipeline/`), none of which changes how a successful extraction is validated.
- They were, however, produced 3 days later in a separate session, for a review that had been excluded before the
  original run.

## 2. Which metrics to use

Both are correct. They answer different questions:

| Use | Metrics | Why |
|---|---|---|
| **D2 headline (recommended)**: the evaluation on the approved 300-review reference dataset | Approved reference, n = 300: macro-F1 0.4435 / **0.6322** / 0.4276; accuracy 0.6133 / 0.8133 / 0.5767 | Matches the D2 definition the team adopted on 2026-11-14 (all 300 reference annotations, reviewed and approved under a named reviewer); status `complete`; independent cross-check passed; reproduced independently |
| Original pre-specified run analysis (report as sensitivity) | `metrics.json`, n = 299: 0.4380 / **0.6025** / 0.4232; accuracy 0.6120 / 0.8127 / 0.5753 | The flagged review was excluded **before** the run as unreadable, and all predictions come from one session |

The submission must disclose that the two differ only by `R5cfb12ef75b4`: a garbled review with a low-confidence,
debatable AI label, predicted later. With only 6 neutral examples, that single review moves few-shot macro-F1 by
+0.03. **If the team does not stand behind approving that review, use the 299-review figures instead; the ranking
and conclusions are the same.**

Either way the labels are mostly AI-assisted (299 of 300, approved; 1 human). The metrics are agreement with
approved AI-assisted reference labels, **not** independent human ground truth (0 / 300 second-person-verified
human labels).

## 3. Is the existing D2 report internally consistent?

**Yes, for what it covers (the 299-review run).**
- Markdown, HTML and JSON were written by the same build (22:41:00). They carry identical numbers: 89 / 89 decimal
  values match between Markdown and HTML, and every headline value equals `D2_REPORT_DATA.json`, which records
  96 / 96 verified values.
- The PDF was printed from that HTML 4 s later. A fresh print of the HTML gives the same 11 pages and 25 / 25
  identical decompressed content streams.
- The 300-sample versus 299-evaluated distinction is stated correctly.

**But it is incomplete for the current state.** It does not mention the approved 300-review evaluation that now
exists, and its claim that the supplementary predictions "are not part of any number in this report" is true but
leaves the reader without the 300-review result. Proposed addendum: `D2_FINAL_REPORT_ADDENDUM_proposed.md`.

Minor finding (no effect on any number): `scripts/build_d2_report.py` uses `PROVIDER_ERROR_PREFIX = "Provider error"`,
while the project's prefix is `"LLM provider error"` (`pipeline/extraction.py`). No stored row has either prefix,
which is why all 96 values still match. Proposed fix: `proposed_code_changes.diff`.

## 4. Dashboard corrections required

1. **Prompt Experiments** shows both evaluations on one page with no explanation. The "Best macro F1 0.603" card
   and the tabs (n = 299) sit next to a panel tagged "D2 complete" with 0.6322 (n = 300). Proposed: label the top
   section "Original run evaluation: 299 reviews (flagged review excluded before the run)". In the D2 panel, state
   the difference ("+1 review, R5cfb12ef75b4, neutral; macro-F1 differs by …"). See `proposed_code_changes.diff`.
2. **Overview** "D2 complete" is correct only under the approved-reference definition. Its caption already says
   "approved (mostly AI-assisted) reference labels". No change is needed beyond item 1.

## 5. Stale or defective documentation (proposed fixes, not applied)

| File | Problem | Proposal |
|---|---|---|
| `reports/RESUME_CHECKPOINT.md` | says "0/300 approved"; 3 control characters (`scripts\r…` turned into a carriage return) | corrected copy (control characters) in this folder; status text to update |
| `reports/D3_JUDGE_EVALUATION.md` | 2 carriage returns in `scripts\run_judge.py` | corrected copy in this folder |
| `reports/HUMAN_ANNOTATION_GUIDE.md` | 2 bell characters in `reports\d2\annotation_progress.md` | corrected copy in this folder |
| `reports/DELIVERABLE_COMPLETION_REPORT.md` | D2 rows say approval "not executed: 0 / 300"; D2 verdict "not yet shown as complete" | update to 300 / 300 approved, evaluation complete (n = 300) and the sensitivity statement above |
| `reports/D2_COMPLETION_REPORT.md` | "0 / 300 approved (waiting for the reviewer)" | same |
| `reports/FINAL_EXPERIMENT_RESULTS.md` | "Status: 0/300 approved" | same |
| `reports/FINAL_TEST_REPORT.md` | "No reference approval was recorded" | "Approval recorded by Rachit (299 bulk + 1 individual), 2026-11-14 16:48–16:53 UTC; evaluated 16:55 UTC" |
| `reports/VIVA_EXPLANATION.md` | "Status today: 0 / 300 approved" | same |

The control characters came from escape sequences in my scripted edits earlier on 2026-11-14 (`\r` → carriage
return, `\a` → bell).

## 6. Files inspected

- `outputs/live/experiments/20261111-030912_qwen2.5-3b/`: `config.json`, `metrics.json`, `report.md`,
  `predictions_{zero_shot,few_shot,cot}.jsonl`, `supplementary/predictions_*.jsonl`,
  `approved_reference/metrics.json`, `approved_reference/report.md`, `human_reference/` (listing).
- `data/annotations/`: `annotations.json`, `annotation_sample.csv`, `reference_adjudication.json`,
  `human_annotations.json` (listing).
- Code: `ui/prompt_comparison.py`, `ui/home.py`, `ui/annotation_page.py`, `evaluation/prompt_comparison.py`,
  `evaluation/reference_evaluation.py`, `evaluation/classification_metrics.py`, `annotation/reference_adjudication.py`,
  `pipeline/extraction.py` (plus `git diff 045f16b -- pipeline/ prompts/`), `scripts/build_d2_report.py`.
- Report files: `reports/D2_FINAL_REPORT.md`, `reports/final/D2_Final_Report.html`,
  `reports/final/D2_Final_Report.pdf`, `reports/d2/D2_REPORT_DATA.json`, and every `reports/*.md` for stale status
  text and control characters.
- Live dashboard: Overview, Prompt Experiments and Human Annotation rendered read-only (AppTest), with values
  captured.
