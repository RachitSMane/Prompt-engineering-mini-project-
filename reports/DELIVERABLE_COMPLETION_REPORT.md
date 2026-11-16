# ReviewIQ — deliverable completion report (final, 2026-11-14)

Base commit `580adb2` (branch `main`). **Nothing has been committed or pushed.** All inference used local Ollama
only. Every number below was re-checked on 2026-11-14 against the saved result files, mostly with independent
scripts.

Backups (never modified; manifest re-verified at the end):
- `Downloads\ReviewIQ-Recovery-Backup`
- `Downloads\ReviewIQ-pre-completion-backup-20261113`
- `Downloads\ReviewIQ-pre-final-backup-20261114` (391 files, SHA-256 manifest)

| Topic | Report |
|---|---|
| D1 | `D1_COMPLETION_REPORT.md` |
| D2 full experiment report (verified, with charts; PDF in `final/D2_Final_Report.pdf`) | `D2_FINAL_REPORT.md` |
| D2 (experiments, reference approval, human labels) | `D2_COMPLETION_REPORT.md`, `HUMAN_ANNOTATION_GUIDE.md`, `d2/record_integrity.md`, `d2/annotation_progress.md` |
| D3 overview / judge v1–v3 / negative control / evidence | `D3_COMPLETION_REPORT.md`, `D3_JUDGE_EVALUATION.md`, `D3_NEGATIVE_CONTROL.md`, `D3_EVIDENCE_AUDIT.md` |
| All results in one place | `FINAL_EXPERIMENT_RESULTS.md` |
| Tests and verification | `FINAL_TEST_REPORT.md` |
| Demo and viva | `MENTOR_DEMO_GUIDE.md`, `VIVA_EXPLANATION.md` |
| Documentation consistency audit | `CONSISTENCY_AUDIT_2026-11-14.md` |

## 1. Compliance matrix

| Requirement | Implementation | Executed | Independently verified | Limitations / outstanding |
|---|---|---|---|---|
| **D1** Pydantic schema, LangChain extraction (local Ollama) | done | 3 strategies × 299 reviews (846 valid, 51 controlled failures) | `test_schemas.py`, export re-validation | small 3B model: shallow extractions |
| D1 Repair, validation, controlled failures, diagnostics | done | done | failures stored with every raw attempt; tested | – |
| D1 JSON + CSV export | done | done | **27 / 27 checks PASS** (`verify_d1_exports.py`) | – |
| **D2** Zero-shot / few-shot / CoT on identical reviews | done | 299 reviews each | **11 / 11 record checks PASS** (`verify_d2_records.py`) | – |
| D2 Metrics (accuracy, macro P/R/F1, weighted F1, per-class, confusion) | done | done | stdlib recomputation: all values match | – |
| D2 300 reference annotations | validated: 300 unique ids, 0 invalid, 0 missing | – | `reference_annotations.py validate` | provenance: 1 human-created, 299 AI-assisted (mentor-approved at generation per `annotations.json`) |
| D2 Bulk approval / adjudication of the existing references | done: button, CLI, individual approve/correct/exclude, idempotent, locked, read-only safe, originals never edited | **done**: 300 / 300 accepted (299 bulk + `R5cfb12ef75b4` individually, reviewer Rachit, 2026-11-14 16:48–16:53 UTC; no label corrected; provenance unchanged: 1 human-created, 299 AI-assisted) | 10 unit + 2 UI tests | – |
| D2 Evaluation on the approved references (**sensitivity analysis**) | done (`<run>/approved_reference/`, original `metrics.json` untouched) | 300 reviews, 2026-11-14 16:55 UTC | 72 / 72 values recomputed identically | macro-F1 0.4435 / 0.6322 / 0.4276; secondary to the primary 299-review result |
| D2 Predictions for the formerly flagged review | done (`predict-missing`, supplementary files) | **3 / 3 predicted** (`qwen2.5:3b`, 2026-11-14) | original prediction files byte-identical | – |
| D2 Independent human annotations (blind) | done | **300 / 300 completed** (human-created; 0 pending, 0 flagged; updated 2026-11-15) | annotation tests | second-person verification not required by the project; labels **not** independently verified; kept separate from AI-assisted references; D2 results unchanged |
| **D3** Extract → aggregate → summarise | done | 15 / 15 summaries | statistics reproduce 15 / 15 | – |
| D3 Evidence audit | done | all 15 + 2 corrected copies | **13 / 15 originals pass**; both corrected copies pass; 130 / 130 cited ids valid | B0BTY63SFY misattributed %, B001DHKBXU 7 misleading denominators (originals kept) |
| D3 Judge v2 | done | 15 / 15 valid | independent store re-check | H = 4 for all; 7 / 15 claim lists induced by repairs |
| D3 **Judge v3** (one call, no repair, raw kept, corrected copies separate) | done | **47 / 47 valid** (15 originals, 2 corrected, 30 NC), 0 failures | report recomputed from stored records | score contradicts its own claim list in 37 / 47; no response to the corrections |
| D3 Negative control | done | v2: 30 planted claims; same 30 re-judged by v3 | claim-only recomputation from `pairs.jsonl`: 18 / 30 | **v2 18 / 30 detected, 12 missed**; v3 5 / 30; model-assisted semantic review of v2 (18 / 1 / 2 / 1 / 8) kept separate; n = 15 |
| D3 Judge reliability | evaluated | – | – | **not reliable** (v2 and v3) |
| UI: six pages | done | read-only on real data | 0 exceptions, 0 generation calls, 0 files changed | screenshots in `screenshots/final_2026-11-14/` |
| Tests | – | **269 passed, 1 skipped**, 0 failed | – | – |

### Deliverable verdicts

| Deliverable | Verdict |
|---|---|
| **D1** | **Complete**, independently verified |
| **D2** | **Complete against approved, mostly AI-assisted reference labels.** **Primary:** original 299-review run, macro-F1 0.4380 / **0.6025** / 0.4232. **Sensitivity:** 300-review approved-reference evaluation, 0.4435 / 0.6322 / 0.4276 (adds `R5cfb12ef75b4`). Both were recomputed independently; full write-up in `D2_FINAL_REPORT.md`. **Not independent human ground truth** (0 / 300 second-person-verified human labels) |
| **D3** | **Implemented and fully executed; not fully compliant on evidence and judge reliability.** 2 / 15 original summaries fail the evidence audit (corrected copies pass). Judges v2 and v3 are both unreliable faithfulness measures |

## 2. What changed on 2026-11-14 (final pass)

| Area | Change |
|---|---|
| Inference | Judge v3: 47 single calls (14:15–16:16 UTC, peak 87 °C, watchdog active and its thresholds unchanged). `predict-missing`: 3 `qwen2.5:3b` calls for `R5cfb12ef75b4` → `<run>/supplementary/`. Earlier the same day: 16 v2 negative-control calls. Nothing was re-run or overwritten |
| `evaluation/judge_v3.py`, `scripts/judge_v3.py`, `scripts/run_judge.py` | Judge v3 (see `D3_JUDGE_EVALUATION.md` §6); `run_judge.py --version v3` delegates to it |
| `annotation/reference_adjudication.py`, `evaluation/reference_evaluation.py`, `scripts/reference_annotations.py` | Approval of the existing references (separate `reference_adjudication.json`), approved-reference export, evaluation into `approved_reference/`, `predict-missing` |
| `scripts/verify_d2_records.py` | Read-only record-integrity check (ids, duplicates, identical sets, invalid outputs, provenance) |
| `scripts/human_annotation.py` | `finalise` (export + evaluate only at ≥ 300 verified human labels) and `progress-report` |
| `ui/annotation_page.py` | D2 reference dataset panel (available / approved / awaiting / readiness, bulk approval button, individual decisions, export). Independent human labels are reported separately |
| `ui/prompt_comparison.py` | Sensitivity-analysis panel for the approved-reference evaluation (status, provenance, per-strategy metrics, cross-check, denominator and macro-F1 difference vs the primary run); primary-evaluation label on the original run |
| `ui/home.py` | Deliverable indicators come from stored records: D2 = approved references plus evaluation status, with human-verified labels shown separately; D3 = latest judge version. The misleading "299/299 · ready for evaluation" badge was removed |
| `ui/summary_evaluation.py` | Judge v3 section (originals vs corrected, v3 negative control, contradiction counts); corrected-copy wording |
| Tests | +28: 241 → 269 (`test_judge_v3.py` 15, `test_reference_adjudication.py` 9, `test_human_annotation.py` +2, `test_app_smoke.py` +1, `test_demo_readonly.py` +1) |
| Reports | New: D1, D2 completion reports, viva explanation, mentor demo guide, record-integrity and annotation-progress reports. Updated: this report, the D3, judge, negative-control and evidence reports, test report, results, resume checkpoint, execution log, annotation guide. Six new screenshots in a separate folder |

**Unchanged (verified by hash against the 2026-11-14 backup):** all three prediction files, `metrics.json`,
`config.json`, `annotations.json`, `human_annotations.json`, the summaries, extractions, v1/v2 judgements and v1
failures. `pairs.jsonl` holds 30 rows, the first 14 identical to the backup.

## 3. Remaining work

| # | Task | Who | How |
|---|---|---|---|
| 1–3 | ~~Approve the references, decide `R5cfb12ef75b4`, evaluate~~ | done | done 2026-11-14 16:48–16:55 UTC by the reviewer (see the D2 rows above) |
| 4 | ~~Independent human annotations~~ | done | 300 / 300 completed blind (2026-11-15); second-person verification not required |
| 5 | Optional: a reliable judge | – | a larger or human-calibrated judge (inference) |
| 6 | Optional: regenerate `reports/final/*.html/pdf`; decide whether to commit | you | – |

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501                       # normal mode (approval)
$env:REVIEWIQ_DEMO_READONLY = "true"; .venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501   # demo
```

## 4. Limitations

- **D2 references:** D2 metrics measure agreement with mostly AI-assisted reference labels, even after approval.
  Approval records acceptance; it does not create independent human labels.
- **D3 sample:** the summaries use the 299-review sample (9–42 reviews per product).
- **Negative control:** n = 15 products; keyword detection; a 3B judge partly on CPU.
- **Superseded files:** `reports/final/ReviewIQ_Final_Report.html` / `.pdf` and the `DEMO_*.md` files date from
  2026-11-12 and are superseded (marked). Current: this report, `MENTOR_DEMO_GUIDE.md` and
  `screenshots/final_2026-11-14/`.
- **Model loading:** during `predict-missing`, Ollama still held `llama3.2` from the judge run when `qwen2.5:3b` was
  requested. It swapped the models (only `qwen2.5:3b` was resident afterwards), and both were unloaded at the end.
