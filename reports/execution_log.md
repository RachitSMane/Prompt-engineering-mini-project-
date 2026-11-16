# ReviewIQ — Execution Log (project completion, 2026-11-11)

All times are local (IST, UTC+05:30). All LLM inference ran on the local Ollama server at
`http://127.0.0.1:11434`; no cloud API was configured or contacted.

## Environment

| Item | Value |
|---|---|
| OS / hardware | Windows, 32 GB RAM, NVIDIA Quadro M2200 (4 GB VRAM) |
| Python | 3.11.9 virtual environment in `.venv` (the original project venv was also 3.11) |
| Key packages | langchain-core 1.7.7, langchain-ollama 1.1.0, ollama 0.6.3, pydantic 2.14.3, streamlit 1.67.0, pandas 3.1.1, scikit-learn 1.9.1, plotly 7.1.0, pytest 9.2.0 (`pip check`: no broken requirements) |
| Ollama | 0.40.15; models installed: `qwen2.5:3b` (3.1B, Q4_K_M), `qwen2.5:7b` (Q4_K_M), `llama3.2` (3B), `nomic-embed-text` |

## Timeline

| Time | Step | Result |
|---|---|---|
| 08:20 | Restored `data/annotations/annotations.json` from the recovery backup | source and destination SHA-256 `c8781acc…7c968eba`; 300 records, 300 unique ids, 299 complete, 1 flagged (`R5cfb12ef75b4`), all pass the `Annotation` schema; ids identical to the 300-review sample; 1 human label (`R3314d62a5df6`), 299 AI-assisted labels (documented in `meta.annotation_method`) |
| 08:22–08:25 | `python scripts/prepare_dataset.py` (downloaded `Appliances.jsonl.gz` 270 MB + `meta_Appliances.jsonl.gz` 66 MB from the McAuley Lab URL in `data/README.md`) | 1,500 reviews, 15 products × 100, 1,500 unique ids, ratings 150/55/64/145/1,086 — identical to the documented dataset card; all 300 sample ids present with identical text |
| 08:25 | Created `.venv` (Python 3.11.9), `pip install -r requirements.txt` | installed; `pip check` clean |
| 08:26 | Full test suite | 115 passed, 1 skipped (optional `langchain-groq` not installed) |
| 08:27 | One-review smoke test, review `R3314d62a5df6`, zero-shot, cloud keys blanked, non-local sockets blocked | qwen2.5:3b: valid first attempt, 25.7 s warm, 1,819/163 tokens, 1 of 3 reference aspects, complaint missed (identical to the historical run). qwen2.5:7b: valid first attempt, 88.2 s warm (2.2 of 4.9 GB in VRAM), 2 of 3 aspects with correct evidence, complaint missed. Only connection: 127.0.0.1:11434 |
| 08:31 | D2 run `20261111-030130_qwen2.5-3b` started (LLM_MAX_OUTPUT_TOKENS=2048) | **aborted at 4/897**: review `Rb1ed3e0396d5` made the model repeat complaint labels until the 2,048-token cap (298.6 s, 1.4 s under the 300 s timeout). Kept in `outputs/live/experiments_aborted/` with `ABORTED.md` |
| 08:39 | D2 run `20261111-030912_qwen2.5-3b` started with LLM_MAX_OUTPUT_TOKENS=1024 (2.5–10× a normal valid output; runaway outputs are truncated and go through the repair path) | see below |
| 08:39–12:05 | D2 zero-shot stage (299 reviews) | 282 valid, 17 failed (counted as wrong); 26 needed an LLM repair; mean 41 s/review, 3.4 h. Interim check of the metric code (no LLM): macro-F1 0.438, accuracy 0.612 |
| 10:30–12:00 | GPU slowdown observed | laptop GPU at 78–81 °C with SM clock 1,037 MHz (later 405 MHz between calls); generation fell from ~7 to ~3.5–4 tokens/s; on AC power, no driver throttle reason reported; Windows power plan "Balanced" left unchanged |

## Decisions

- **Evaluation set = 299 reviews.** The 300th sample review (`R5cfb12ef75b4`) is flagged as unreadable and has no
  valid label, so it cannot be scored; it is excluded explicitly and reported as such.
- **Output-token cap 1,024 for extraction** (see the aborted run above). Product summaries and judge verdicts are
  larger structured objects and are generated with 2,048.
- **Reference labels are reported with their provenance** (1 human, 298 AI-assisted in the evaluated 299). The app
  and reports previously called them "human labels"; this was corrected.

## Session 2 (2026-11-11, from 15:13 IST)

| Time | Step | Result |
|---|---|---|
| 15:13 | State check | D2 process had died at 12:48 (few-shot 49/299); Ollama up, all 4 models installed; GPU idle; 28 GB disk free; annotations hash still `c8781acc…`; test suite 120 passed / 1 skipped (optional `langchain-groq`) |
| 15:14 | Resumed D2 run | **Defect found:** on resume the runner re-ran stored zero-shot *output* failures (`retry_failed=True`), which would give those reviews a second sample. Stopped after 5 retries — all 5 failed again, so no prediction changed. The 5 rows were moved to `retried_failures_removed.jsonl` (original file kept as `.bak-with-5-retries`, see `RETRY_NOTE.md` in the run folder) |
| 15:25 | **Defect found:** `evaluate_run` / the Prompt Experiments page crashed (`KeyError: 'per_class'`) for a run whose CoT stage had not started | fixed in `evaluation/prompt_comparison.py` and `ui/prompt_comparison.py`; regression tests added |
| 15:32 | D2 resumed with `retry_failed=False` (one attempt per review per strategy) | running |
| 15:35 | All six Streamlit pages checked through the Streamlit agent API | status `ready`, 0 exceptions, no server warnings |
| 15:56 | GPU watchdog armed (`logs/gpu_monitor.csv`, 20 s samples; stop if ≥ 95 °C for 3 samples — card slowdown temperature 96 °C) | max 90 °C in the first hour |
| 16:57 | Slow few-shot throughput investigated (~138 s/review over 30 min) | one runaway review looped to the 1,024-token cap on all 3 attempts (993 s, stored as a failure as designed); GPU on AC but in power state P8 with platform power limit 21.5 W (default 80 W) and SM clock 405 MHz at 71 °C — a firmware/Windows power cap, not thermal. Hardware power limits deliberately not changed. Watchdog loops duplicated after monitor re-arming; fixed (script now self-exits after 29 min) — duplicate CSV rows before 16:58 are harmless repeats |
| 19:11 | D2 few-shot finished | 299/299 stored (285 valid, 14 output failures); 49 from the first session + 250 new |
| 23:08 | D2 CoT finished; process exited 0 | 299/299 stored (279 valid, 20 output failures). Max GPU temperature during D2 (watchdog): 91 °C; no threshold events |
| 23:09 | `--reevaluate` + `scripts/verify_metrics_independent.py` | status **complete**; every strategy 299/299 ids, 0 duplicates, 0 out-of-run ids; independent recomputation matches all stored metrics. Macro-F1: zero-shot 0.4380, few-shot **0.6025**, CoT 0.4232 |
| 23:10 | D1 export (`scripts/export_d1.py`) | all records re-validated after reload (282/282, 285/285, 279/279); JSON + 3 CSVs per strategy parse |
| 23:10 | D3 started: `run_chain.py --sample --strategy few_shot --seed-from-run 20261111-030912_qwen2.5-3b --no-retry-failed --judge`, LLM_MAX_OUTPUT_TOKENS=2048 | compatibility check passed; seeded 285 ok + 14 failed few-shot outcomes from D2 (best macro-F1) |

## Session 3 (2026-11-12, from 08:06 IST)

| Time | Step | Result |
|---|---|---|
| 08:06 | State check after the previous session closed | D2 complete (zero-shot/few-shot/CoT 299/299 each, unique ids; zero-shot SHA-256 `c3ad1430…e33d95a` unchanged); no inference or watchdog process alive; D3 had died ~23:32 with 5/15 summaries, 0 judgements; GPU 49 °C |
| 08:08 | **Defect found:** `run_chain.py` had no summary/judgement reuse, so "re-run the same command" would regenerate the 5 stored summaries and append duplicates | added opt-in `--resume` (`run_chain(reuse_summaries=True)`): a stored summary/judgement is reused only if extraction source, provider, model, prompt version and the full aggregate are identical; 2 regression tests; offline check confirmed all 5 stored summaries match |
| 08:12 | GPU watchdog re-armed (`..\tools\gpu_watchdog.py`, same CSV/threshold); D3 resumed with `--resume` | seeding idempotent (`skipped_existing: 299`), 0 new extraction calls |
| 08:13–08:26 | D3 summaries | 15/15 stored (5 reused + 10 new, 53–125 s each, 0 repairs), no duplicate rows; GPU plateaued at 86–90 °C with SM clock reduced to 721–987 MHz |
| 08:24–08:28 | **Watchdog gap (~4 min)** | first watchdog exited without a thermal stop (most likely its idle-exit fired because process detection returned nothing; not reproducible interactively). Last sample before the gap 88 °C, first after 87 °C — below threshold. Replaced by a hardened version (exit reasons/crashes logged to `logs/gpu_watchdog.log`, unreadable process list treated as "running"); the session monitor now also alerts if `gpu_monitor.csv` is stale > 90 s |
| 08:28 | Judge (llama3.2) | first judge output failure stored (verdict dimensions not objects, still invalid after 2 repairs); kept as a failure, not re-sampled |
| 08:28–08:59 | D3 judge (llama3.2, 15 summaries) | **5 judged, 10 failed** (judge output violated the verdict schema: flat per-dimension scores, or `unsupported_claims` nested inside each dimension; still invalid after 2 repairs). 0 duplicates, all 15 products accounted for. Failures were only logged by `run_chain.py`; they are now persisted in `outputs/live/judgements/judge_failures.jsonl` (this run's 10 reconstructed from `logs/d3_chain.out`) and a `--resume` keeps them as failures. Not re-sampled. Peak GPU 90 °C |
| 08:55 | **Defect found:** `judge_sanity_check.py` re-judged originals without a stored judgement — i.e. it would re-sample the 10 failed judgements | added `--only-judged-originals` (skipped products listed in `summary.json`) |
| 09:00–09:12 | Judge negative control (`--only-judged-originals`) | 5 eligible pairs → 3 scored, 2 corrupted judgements failed validation. Corrupted mean lower in 3/3 (3.17 → 2.92, via coverage/consistency) but **faithfulness lower in 0/3** (4.00 → 4.00); invented feature flagged 0/3, fabricated percentage flagged 3/3. n = 3: inconclusive; the judge's faithfulness/hallucination scores do not discriminate |
| 09:06–09:08 | System stall (~2.5 min) | nvidia-smi, Ollama and PowerShell timed out together; one `NA` sample (below the 3-sample fail-safe). Temperatures 86 °C before / 73 °C after; watchdog kept running as designed |
| 09:01 | D3 export (`export_d3.py`) | added explicit `judge_status` (`ok` / `failed (invalid judge output)`) + `judge_error` columns so failed judgements are not confused with "not run"; no scores filled in |
| 09:13 | `build_reports.py` | curated D1/D2/D3 + negative control into `reports/`; recomputation matches; curated files byte-identical to `outputs/live` |
| 09:17–09:21 | Screenshots (Playwright 1.65.1 installed into the venv as the documented optional tool; local Chrome) | first capture: pages 02 and 04 caught mid-render (skeleton placeholders) → capture now also waits for `stSkeleton`; all 6 re-captured and inspected. **Defect found:** Summary Evaluation showed failed judgements as "Not judged yet" and offered to re-judge them → page now reads `judge_failures.jsonl`, shows "judge failed" + error, excludes them from the batch button, and records new app-side failures |
| 09:22 | Final report + presentation | report/slides now state the 10 failed judgements, the 10 skipped negative-control products and faithfulness-lower 0/3 (values read from files); D3 section added to `reports/discussion.md` |
| 09:24 | Tests | 129 passed, 1 skipped (optional `langchain-groq`); new tests: summary resume/reuse (2), D3 export failure status, UI judge-failure display. `pip check` clean; no secrets tracked; zero-shot hash `c3ad1430…` and annotations hash `c8781acc…` unchanged |
| 09:25 | Shutdown | watchdog exited itself after 10 idle min; Streamlit stopped; no project processes left; no models loaded. Nothing committed or pushed |

## Documentation pass (2026-11-12, after the submission audit; no inference, no results changed)

| Step | Result |
|---|---|
| D2 metric check | metrics.json, `verification_independent.md`, `comparison_table.csv`, report and slides agree: overall-sentiment macro-F1 0.4380 / 0.6025 / 0.4232, accuracy 0.6120 / 0.8127 / 0.5753 (zero-shot / few-shot / CoT); PDFs show 3-decimal rounding with correct column labels |
| Captions | Review Analysis caption now describes the initial (empty) page state; Product Intelligence caption no longer claims the summary is visible |
| Discussion | few-shot mean input tokens corrected 2,885 → 2,884.47 (862,456 / 299) |
| Abstract / conclusion / closing slide | now state 5 of 15 valid judge outputs, 10 failed, faithfulness lower in 0 of 3 scored negative-control pairs (values read from the curated files) |
| Layout | screenshots capped at 228 mm in print so each stays on one page with its caption (report 21 → 15 pages) |
| Annotation `annotator` field | `Akash` on all 300 records; the guide defines it as the labeller's name; meta + notes establish 1 human (R3314d62a5df6) / 299 Claude-generated labels; Akash's role for the 299 is not documented anywhere → field left unchanged, question raised with the team |
| Verification | D1/D2/D3 and negative-control figures in both PDFs match the original outputs; 6 embedded screenshots byte-identical to `reports/screenshots/`; 129 passed / 1 skipped; zero-shot `c3ad1430…` and annotations `c8781acc…` unchanged |

## Session 5 (2026-11-14, from 08:59 IST): final verification and completion

| Time | Step | Result |
|---|---|---|
| 09:05 | State check | v1 5 ok / 10 failed; v2 15 ok / 0 failed; negative control 14/30 pairs (stopped 2026-11-13 19:44 by watchdog fail-safe); no inference running; GPU 47 C; 14 GB RAM free |
| 09:10 | Backup `Downloads\ReviewIQ-pre-final-backup-20261114` | 391 files, SHA-256 manifest; earlier backups untouched |
| 09:12 | GPU watchdog started (unchanged thresholds); negative control resumed as a detached process | resumed at 14 stored pairs |
| 09:15-09:45 | Independent audit of the D3 stores (scratch script, no project metric code) | all records schema-valid, genuine Ollama outputs. New: 7 misleading "out of 15" counts in B001DHKBXU; 7/15 v2 claim lists produced after validator rejections; 9/15 verdicts violate the H=4 rubric |
| 10:28 | Negative control finished | 30/30 pairs, 0 judge failures; peak GPU 91 C; watchdog exited itself 10:37 (idle) |
| ~10:40 | Independent re-computation from `pairs.jsonl` | all invariants hold. **Defect:** the detection rule also matched reasons; B0081N3YW4 counted as detected although the judge used the fabricated 91 % as fact. Fixed (claims only), recomputed without re-run: 18/30 (was 19/30); previous summary preserved |
| ~10:50 | Evidence audit: count-statement check added; `audit_d3.py` re-run (previous outputs kept with date suffix) | 13/15 summaries fully verified (was 14/15) |
| ~11:00 | D2 human store: conflict protection (owner-only changes, protected verified labels, locked re-reading transactions, import never overwrites) | 6 new tests; real store still 0/300 verified, 1 completed |
| 11:00 | D1 export verifier (scratch output) and D2 independent metrics | 27/27 PASS, identical to stored report; all metrics match |
| 11:02 | All six pages, real data, read-only | 0 exceptions, 0 generation calls (1 `GET /api/tags`), write buttons disabled, 0 files changed |
| 11:05 | Full test suite | 235 passed, 1 skipped (optional `langchain_groq`), 0 failed |
| — | Protected files | zero-shot `c3ad1430…`, annotations `c8781acc…`, judgements, summaries, metrics unchanged; nothing committed or pushed |

## Session 6 (2026-11-14, afternoon): reconciliation, no inference

| Step | Result |
|---|---|
| Semantic review of the 30 negative-control judgements (`scripts/nc_semantic_review.py`, model-assisted, labelled not-human) | 18 keyword / 1 paraphrase / 2 justification-only (self-contradictory) / 1 accepted as fact / 8 not mentioned; kept separate from the keyword rate 18/30; `pairs.jsonl` and previous-rule summary unchanged |
| B0BTY63SFY 80 % / 67 % re-traced at record level | misattributed real ratios (80 % = performance positive share 8/10; 67 % = 2 of 3 complaining reviews; actual: 86.7 % positive reviews, 2 of 2 negative reviews), not invented; wording corrected in all reports |
| Corrected copies (`scripts/build_corrected_summaries.py`) | B001DHKBXU (7 denominators) and B0BTY63SFY (2 sentences); replay-verified; pass the audit; stored separately; originals byte-identical; not judged |
| Human store hardening | writes outside a locked transaction refused; import never relabels; migration history under lock; +4 tests |
| Verification | 241 passed / 1 skipped; D1 27/27 PASS; D2 metrics all match; 6 pages read-only: 0 exceptions, 0 generation calls, 0 files changed; backup manifest OK; protected files identical to backup; `pairs.jsonl` append-only |

## Session 7 (2026-11-14, evening): judge v3, D2 reference approval workflow, final verification

| Time (local) | Step | Result |
|---|---|---|
| 19:42 | GPU watchdog started (thresholds unchanged); judge v3 pilot (1 call) | pilot showed the first reconciliation rule raised a contradictory score to 5; rule replaced; pilot kept in `judgements/v3_pilot_superseded_reconciliation/` |
| 19:45–21:46 | Judge v3 full run, detached (`run_judge.py --version v3`) | 47/47 valid, 0 failures; peak GPU 87 °C; no thermal stop |
| during the run | D2: `verify_d2_records.py` (11/11 PASS), `human_annotation.py finalise` / `progress-report`, Overview indicators fixed, D1/D2 reports, viva and demo guides | – |
| during the run | User request: bulk approval of the existing reference annotations | `annotation/reference_adjudication.py`, UI button, evaluation into `approved_reference/`; **no approval recorded by the assistant** (it needs the reviewer's name); dry run on a temporary copy reproduces `metrics.json` |
| 21:50 | `predict-missing` for R5cfb12ef75b4 (3 × `qwen2.5:3b`) | 3/3 ok → `supplementary/`; original prediction files byte-identical; Ollama swapped out `llama3.2`; all models unloaded afterwards |
| 21:55 | Full test suite | **269 passed, 1 skipped**, 0 failed |
| 22:00 | Verifiers | D1 27/27; D2 records 11/11; D2 metrics match; NC claim-only 18/30; evidence audit 13/15 originals, 2/2 corrected; v3 report |
| 22:05 | Six pages read-only on real data | 0 exceptions, 1 `GET /api/tags`, 0 generation calls, 0 files changed |
| 22:10 | Screenshots | 6/6 → `reports/screenshots/final_2026-11-14/`; Streamlit stopped |
| end | Protected files and backups | see `DELIVERABLE_COMPLETION_REPORT.md`; nothing committed or pushed |

## Session 8 (2026-11-14, late evening): D2 final report — primary 299 + sensitivity 300, no inference

| Step | Result |
|---|---|
| Reviewer actions found (not made by the assistant) | `reference_adjudication.json`: 299 bulk + `R5cfb12ef75b4` individually approved by Rachit, 16:48–16:53 UTC; `approved_reference/metrics.json` evaluated 16:55 UTC (300 reviews) |
| Read-only consistency audit | source of the 0.4435 / 0.6322 / 0.4276 figures traced to the 300-review approved-reference evaluation; `reports/proposed_revisions_2026-11-14/` |
| Decision (user) | primary = original 299-review run; 300-review evaluation = sensitivity analysis |
| Backup before changes | `Downloads\ReviewIQ-d2-final-revision-backup-20261114` (26 files, SHA-256 manifest) |
| D2 report rebuilt (`scripts/build_d2_report.py`) | 17 sections incl. §14 sensitivity; primary 96/96 and sensitivity 72/72 values recomputed identically; 8/8 charts label-checked |
| Dashboard | Prompt Experiments: primary-evaluation label; sensitivity panel relabelled with denominator, added review and macro-F1 difference; Overview badge reworded |
| Documentation | stale "0/300 approved" statements corrected; 7 control characters (from earlier escape slips) fixed |
