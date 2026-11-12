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
