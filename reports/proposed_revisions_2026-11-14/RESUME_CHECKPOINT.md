# ReviewIQ — resume checkpoint

Use this file to continue the project completion workflow after any interruption. Every stage is resumable; no stage
needs to be restarted. All commands run from the repository root with the project venv
(`.venv\Scripts\python.exe`, Python 3.11.9). All inference uses local Ollama only (`http://127.0.0.1:11434`).

## Status update (2026-11-14)

| Item | Status | Evidence |
|---|---|---|
| D3 judge v2 | done, 15/15 valid | `outputs/live/judgements/v2/judgements.jsonl` |
| D3 negative control (v2) | **done, 30/30 pairs** | `outputs/live/d3/judge_negative_control_v2/dataset__few_shot__qwen2.5-3b__v1/` — 18/30 named, 12 missed |
| D3 evidence audit | done, 13/15 verified | `outputs/live/d3/audit/dataset__few_shot__qwen2.5-3b__v1/evidence_audit.json` |
| D3 judge v3 (separate run) | **done, 47/47 valid** (15 originals, 2 corrected, 30 NC) | `outputs/live/judgements/v3/`; report `outputs/live/d3/judge_v3_report/<source>/report.json` |
| D2 reference annotations | 300/300 validated; **0/300 approved** (needs the reviewer) | `scripts\reference_annotations.py status`; approval → `data/annotations/reference_adjudication.json` |
| D2 prediction for R5cfb12ef75b4 | done (3/3, supplementary) | `outputs/live/experiments/20261111-030912_qwen2.5-3b/supplementary/` |
| D2 independent human labels | 0/300 verified (1 completed) | `scripts\human_annotation.py status` |
| Tests | **269 passed, 1 skipped** | `reports/FINAL_TEST_REPORT.md` |

No LLM work is outstanding. Next steps (no LLM): (1) the reviewer approves the references and decides
R5cfb12ef75b4; (2) evaluate against the approved references (`scripts\reference_annotations.py evaluate --run
20261111-030912_qwen2.5-3b`). See `DELIVERABLE_COMPLETION_REPORT.md` §3. Judge v3 resumes with
`$env:LLM_MAX_OUTPUT_TOKENS="2048"; $env:LLM_TIMEOUT_SECONDS="600"; .venv\Scripts\python.exe scripts\run_judge.py --version v3
--source dataset__few_shot__qwen2.5-3b__v1` (all stored → 0 calls). The stage table below is historical (2026-11-12).

## Stage status (updated 2026-11-12 09:30 IST)

| # | Stage | Status | Evidence |
|---|---|---|---|
| 0 | Annotations restored + verified | done | SHA-256 `c8781acc…7c968eba` (source = destination); 300 records, 299 complete, 1 flagged `R5cfb12ef75b4` |
| 0 | Dataset prepared | done | `data/processed/reviews.csv`: 1,500 reviews, 15 products, unique ids |
| 1 | D2 zero-shot (qwen2.5:3b, 299 reviews) | **done — do not touch** | `outputs/live/experiments/20261111-030912_qwen2.5-3b/predictions_zero_shot.jsonl` (299 rows, 282 valid, 17 failed) |
| 2 | D2 few-shot | **done** | `predictions_few_shot.jsonl` (299 rows, 285 valid, 14 failed) |
| 3 | D2 CoT | **done** | `predictions_cot.jsonl` (299 rows, 279 valid, 20 failed) |
| 4 | D2 final metrics + independent verification | **done** | status complete; `reports/verification_independent.md` (all match) |
| 5 | D1 export (from the D2 predictions) | **done** | `outputs/live/d1/20261111-030912_qwen2.5-3b/` |
| 6 | D3 chain on the 299-review sample + judge (`<best>` = `few_shot`) | **done** | 15/15 summaries; judge (llama3.2) 5 ok / 10 failed (stored in `outputs/live/judgements/judge_failures.jsonl`, final, not re-sampled) |
| 7 | Judge negative control | **done** (`--only-judged-originals`) | 5 eligible → 3 pairs scored, 2 corrupted judgements failed; faithfulness lower 0/3; `outputs/live/d3/judge_sanity/<source>/summary.json` |
| 8 | Curate reports, screenshots, final report, presentation | **done** | `reports/d1,d2,d3`, `reports/screenshots/` (6, verified visually), `reports/final/`, `reports/presentation/` |
| 9 | Full tests, security and git checks | **done** | 129 passed / 1 skipped (optional langchain-groq); pip check clean; no secrets tracked; nothing committed |

## Resume commands

```powershell
# 0. Pre-checks: Ollama running and models installed (do NOT pull again)
ollama list
# 1-3. Resume D2 (skips every stored prediction; stored output failures are final; one attempt per review per strategy)
.venv\Scripts\python.exe scripts\run_experiment.py --resume 20261111-030912_qwen2.5-3b --only-annotated >> logs\d2_qwen2.5-3b.out 2>&1
# 4. Final metrics from saved predictions (no LLM calls) and independent check (exit code 1 on any mismatch)
.venv\Scripts\python.exe scripts\run_experiment.py --reevaluate 20261111-030912_qwen2.5-3b
.venv\Scripts\python.exe scripts\verify_metrics_independent.py --run 20261111-030912_qwen2.5-3b --out reports\verification_independent.md
# 5. D1 export
.venv\Scripts\python.exe scripts\export_d1.py --run 20261111-030912_qwen2.5-3b
# 6. D3 on the evaluation sample, reusing the D2 predictions of the chosen strategy (<best> = highest macro-F1)
$env:LLM_MAX_OUTPUT_TOKENS="2048"
.venv\Scripts\python.exe scripts\run_chain.py --sample --strategy <best> --seed-from-run 20261111-030912_qwen2.5-3b --no-retry-failed --judge --resume   # --resume reuses stored summaries/judgements (identical inputs/model/prompt); without it they are regenerated and duplicated
# 7. Judge negative control (resumable; skips pairs already judged)
.venv\Scripts\python.exe scripts\judge_sanity_check.py --source dataset__<best>__qwen2.5-3b__v1 --only-judged-originals   # never re-judges a failed original
Remove-Item Env:LLM_MAX_OUTPUT_TOKENS
# 8. Exports, curation, screenshots, report
.venv\Scripts\python.exe scripts\export_d3.py --source dataset__<best>__qwen2.5-3b__v1
.venv\Scripts\python.exe scripts\build_reports.py --d2-run 20261111-030912_qwen2.5-3b --d3-source dataset__<best>__qwen2.5-3b__v1
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8601 --server.headless true   # separate terminal
.venv\Scripts\python.exe scripts\capture_screenshots.py --url http://127.0.0.1:8601     # needs: pip install playwright (optional tool)
.venv\Scripts\python.exe scripts\build_final_report.py --d2-run 20261111-030912_qwen2.5-3b --d3-source dataset__<best>__qwen2.5-3b__v1
# 9. Tests
.venv\Scripts\python.exe -m pytest -q -rs
```

## Safety rules in force

- GPU watchdog (`..	ools\gpu_watchdog.py "<repo>"`, outside the repo; start it BEFORE inference, self-exits 10 min after inference ends; launch via `Invoke-CimMethod Win32_Process Create` so it survives a closed terminal) samples every 20 s into `logs/gpu_monitor.csv`; inference is stopped if the GPU stays ≥ 95 °C for 3 samples
  (hardware slowdown is 96 °C). A stop loses at most the one in-flight review; everything stored is kept.
- qwen2.5:7b is never loaded while a qwen2.5:3b run is active. The 7B comparison is optional and lower priority than finishing D2.
- Never edit `predictions_zero_shot.jsonl`; never modify `C:\Users\Lenovo\Downloads\ReviewIQ-Recovery-Backup`.
- Interruption history of this run: see `reports/execution_log.md` and `RETRY_NOTE.md` in the run folder.
