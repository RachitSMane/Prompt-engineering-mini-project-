# Final test report (final, 2026-11-14)

Everything below was executed on 2026-11-14 with `.venv\Scripts\python.exe` (Python 3.11.9). No test makes an LLM
call; tests use scripted or mock clients and temporary directories.

## Full suite

```
.venv\Scripts\python.exe -m pytest -q -rs
269 passed, 1 skipped in 41.90s
SKIPPED tests\test_ollama_provider.py:83: could not import 'langchain_groq' (optional cloud provider, deliberately not installed)
```

0 failures (final run, 2026-11-14 evening, after all code changes). Earlier counts: 226 (2026-11-13), 241 (2026-11-14 morning). 43 tests were added on 2026-11-14 (15 in the morning, 28 in the final pass; see below).

| File | Tests | Area |
|---|---|---|
| `test_annotation.py` | 9 | legacy annotation store |
| `test_app_smoke.py` | 21 | UI / integration (AppTest, isolated data) |
| `test_chain.py` | 14 | D3 extract → aggregate → summarise, resume |
| `test_d1_completion.py` | 8 | D1 repair, diagnostics, export verification |
| `test_d3_audit.py` | 10 | D3 evidence traceability, judge heuristics, GPU/RAM guard |
| `test_demo_readonly.py` | 32 | read-only demo mode |
| `test_extraction_pipeline.py` | 19 | D1 extraction |
| `test_human_annotation.py` | 36 | D2 annotation workflow, provenance, conflicts, human-label evaluation, `finalise` / progress report |
| `test_judge_v3.py` | 15 | D3 judge v3: single call, raw kept, failures separate, consistency rule, kinds separate, no score transfer, v3 negative control |
| `test_judge_v2.py` | 29 | D3 judge schema, normalisation, repair, retries, resume, negative control |
| `test_metrics.py` | 12 | D2 metrics |
| `test_ollama_provider.py` | 13 (1 skipped) | local provider |
| `test_provenance_and_exports.py` | 6 | provenance, exports |
| `test_reference_adjudication.py` | 9 | reference approval: validation, idempotent bulk approval, flagged skipped, decisions, stale detection, read-only, export, evaluation |
| `test_schemas.py` | 17 | Pydantic schemas |
| `test_validation_and_repair.py` | 20 | D1 validation and repair |

### Tests added on 2026-11-14

| Test | Covers |
|---|---|
| `test_judge_v2.py::test_planted_keyword_only_in_reason_is_not_a_detection` | Regression for B0081N3YW4: a judge that cites the planted number as fact in a *reason* has not detected it |
| `test_judge_v2.py::test_negative_control_summary_recomputes_detection_without_editing_rows` | The summary recomputes detection from the stored judgements, keeps the old count for transparency, and never edits input rows |
| `test_d3_audit.py::test_count_statements_checked_against_total_and_misleading_denominator` | Regression for B001DHKBXU: "N out of M reviews" counts must match the total; a complaining-review denominator is flagged |
| `test_human_annotation.py::test_another_annotator_cannot_overwrite_flag_or_reset_a_label` | No overwrite, flag, reset or checked-AI replacement by a different person |
| `test_human_annotation.py::test_annotator_names_compare_case_insensitively` | Same person with different casing or spacing may re-label |
| `test_human_annotation.py::test_verified_label_cannot_be_flagged_or_verified_twice` | Verified labels are protected |
| `test_human_annotation.py::test_two_sessions_never_erase_each_others_labels` | Two concurrent store instances: no lost update, stale session sees the other label, no double verification |
| `test_human_annotation.py::test_locked_store_raises_conflict_and_stale_lock_is_recovered` | Lock contention and stale-lock recovery |
| `test_human_annotation.py::test_import_never_overwrites_existing_or_duplicate_rows` | CSV import never overwrites or double-imports |
| `test_human_annotation.py::test_writes_outside_a_transaction_are_refused` | No read-modify-write can bypass the lock and the re-read |
| `test_human_annotation.py::test_import_never_relabels_even_the_same_annotators_label` | Import cannot overwrite the importing annotator's own label (check runs inside the lock) |
| `test_human_annotation.py::test_migration_rerun_preserves_history_and_never_overwrites` | Migration is idempotent, keeps the original timestamp and history, never touches a verified label |
| `test_human_annotation.py::test_ai_assisted_label_has_no_path_to_verified_human_label` | Verification, migration, import and the schema all refuse to turn an AI-assisted label into an eligible human label |
| `test_judge_v2.py::test_semantic_review_requires_manual_reading_of_every_keyword_miss` | Semantic review: each pair exactly once, no duplicates, no stale or contradictory manual entries, labelled as not human |
| `test_d3_audit.py::test_corrected_copy_is_separate_and_requires_exact_single_match` | Corrected copy has a new id, the original is untouched, every replacement must match exactly once |

### Tests added in the final pass (28)

- `test_judge_v3.py` (15):
  - unchanged rubric, claims before the score;
  - an inconsistent verdict is accepted, never repaired;
  - inconsistent hallucination scores are excluded, never changed;
  - one call with the raw response kept;
  - invalid output becomes a separate failure without scores;
  - resume never repeats valid or failed items, and the object kinds stay separate;
  - a provider error records nothing; an unknown kind is rejected;
  - corrected scores never come from the original;
  - the v3 negative control uses the claim text only;
  - excluded scores are left out of means and comparisons.
- `test_reference_adjudication.py` (9):
  - validation of ids, fields and provenance, and detection of invalid records;
  - idempotent bulk approval that keeps `annotations.json` and skips flagged records;
  - reviewer and confirmation required;
  - individual approve / correct / exclude, with history;
  - stale detection after a source change;
  - read-only mode blocks every write;
  - the evaluation uses identical ids, writes a separate folder and leaves `metrics.json` untouched;
  - the export holds accepted rows only, with provenance and reviewer.
- `test_human_annotation.py` +2: the progress report counts only verified human labels; `finalise` refuses and
  writes nothing below 300.
- `test_app_smoke.py` +1: bulk approval from the page (reviewer recorded, 299 approved once, provenance kept,
  `annotations.json` unchanged, idempotent).
- `test_demo_readonly.py` +1: the approval and decision buttons are disabled and nothing is written in demo mode.

## Final verification (2026-11-14 evening, no LLM)

| Check | Result |
|---|---|
| `verify_d1_exports.py` | 3 / 3 strategies PASS, 27 / 27 checks, 0 failures |
| `verify_d2_records.py` | 11 / 11 PASS: 299 identical ids for all strategies, 0 duplicates, invalid 17 / 14 / 20 |
| `verify_metrics_independent.py` | all values match for 3 / 3 strategies (exit 0) |
| `reference_annotations.py validate` | 300 records, 300 unique ids, 0 invalid, 0 missing; 1 human-created, 299 AI-assisted; 1 flagged in source |
| Reference evaluation dry run (temporary copy, never written to the project) | 299 approved → macro-F1 0.4380 / 0.6025 / 0.4232 and confusion matrices identical to `metrics.json`; cross-check passes |
| D3 evidence audit (in memory) | 13 / 15 originals verified (B001DHKBXU, B0BTY63SFY fail); both corrected copies verified |
| Negative control v2, recomputed from `pairs.jsonl` (claim text only) | 18 / 30 (feature 10 / 15, statistic 8 / 15); the old claims-plus-reasons rule gives 19 / 30 (documented); the semantic review file says 18 / 19 / 21 / 1 / 8 and is labelled model-assisted, not human |
| Judge v3 report | 47 / 47 valid, 0 failures; originals 12 / 15 contradictory; v3 NC 5 / 30 |
| Six pages, real data, read-only (AppTest, HTTP recorded, `outputs/` and `data/` hashed) | 0 exceptions on all six pages and the navigation shell; write buttons disabled; 1 HTTP request (`GET /api/tags`); 0 generation calls; **0 files changed**. Summary Evaluation shows the Judge v3 section, the contradiction counts, "no score is transferred" and the v2 18 / 30 |
| Screenshots | 6 / 6 captured from the read-only app into `reports/screenshots/final_2026-11-14/`, none showing an exception; older screenshots kept |
| Protected files vs backup | see `DELIVERABLE_COMPLETION_REPORT.md` §2 (all byte-identical) |

## Verification of saved results (morning pass, no LLM)

| Check | Result |
|---|---|
| `scripts/verify_d1_exports.py --run 20261111-030912_qwen2.5-3b` (output to a scratch file) | PASS: 9/9 checks for each of zero-shot, few-shot and CoT; identical to `reports/d1/export_verification.md` |
| `scripts/verify_metrics_independent.py --run 20261111-030912_qwen2.5-3b` | all values match, exit 0 |
| Independent re-check of the D3 stores (separate script, not project code) | 15 summaries; v1 5 ok / 10 failed; v2 15 ok / 0 failed; all records schema-valid, genuine Ollama outputs (non-mock, latency > 0), summary/product ids consistent |
| Independent re-computation of the negative control from `pairs.jsonl` | 30 rows, 30 unique pairs, 15 products; planted text present with exactly one change; planted feature keywords absent from every product's reviews; fabricated % > 3 points from real statistics; scores match the stored originals; 1 detection-rule defect found and fixed (see `D3_NEGATIVE_CONTROL.md` §3) |
| `scripts/audit_d3.py` (re-run, previous output kept) | 13/15 summaries fully verified |
| `scripts/nc_semantic_review.py` (no LLM) | 30 items, 30 unique pairs; 18 keyword / 1 paraphrase / 2 justification-only / 1 accepted as fact / 8 not mentioned; `pairs.jsonl` and the previous-rule summary unchanged |
| `scripts/build_corrected_summaries.py` (no LLM) + independent replay | each old string occurs exactly once; replaying the replacements on the original reproduces the corrected text exactly; numbers recompute from the extraction records; both copies pass the audit; no judge scores in the file; `summaries.jsonl` identical to the backup |
| `pairs.jsonl` vs backup | append-only: first 14 rows identical, 16 rows added by the completed run |
| SHA-256 of protected files | zero-shot predictions `c3ad1430…`, legacy annotations `c8781acc…`; v1/v2 judgements, v1 failures, summaries, extractions, all three prediction files, `metrics.json`, `annotations.json` and `human_annotations.json` byte-identical to the 2026-11-14 backup |
| Backup `Downloads\ReviewIQ-pre-final-backup-20261114` | `sha256sum -c MANIFEST.sha256`: all 391 files OK |

## All six pages on the real saved data, read-only demo mode

AppTest run with `REVIEWIQ_DEMO_READONLY=true`. Every HTTP request was recorded, and all of `outputs/` and `data/` was
hashed before and after.

| Page | Exceptions | Read-only banner | Write buttons enabled |
|---|---|---|---|
| Overview | 0 | yes (from the `app.py` navigation shell, shown above every page) | none |
| Review Analysis | 0 | yes | none |
| Prompt Experiments | 0 | yes | none |
| Product Intelligence | 0 | yes | none |
| Summary Evaluation (v2, and v1 selected) | 0 | yes | none |
| Human Annotation | 0 | yes | none |

- **HTTP:** 1 request in total (`GET /api/tags`, the availability check); **0 generation or embedding calls**.
- **Files changed under `outputs/` or `data/`:** none.
- **Summary Evaluation:** shows the negative-control table with the "all planted claims" row, the "missed 12 of 30"
  limitation, the keyword-vs-understanding caveat, the semantic review labelled "NOT independent human ground truth",
  the 13 of 15 evidence result, the corrected copies (v3-only judgements), and separate v1 and v2 views (v1
  lists its failures).
- **Human Annotation:** shows "Verified human labels 0 / 300", "1 labelled once; needs a second person" and
  "299 of 300 pending". The blind tab states that no model prediction is shown.

## Notes and checks not executed

- LLM inference on the final day: 16 v2 negative-control calls (morning), 47 judge v3 calls and 3 `predict-missing` calls (evening). Nothing earlier was re-run or overwritten.
- No human review of the negative-control or judge claims. The semantic review is model-assisted.
- The reference approval was recorded by the reviewer through the app (299 bulk + 1 individual, 2026-11-14 16:48–16:53 UTC), not by the assistant. Tests exercise the approval only on isolated test data.

- Screenshots were retaken in the final pass (`reports/screenshots/final_2026-11-14/`); the older folders are kept.
- Normal (non-read-only) mode was checked only by the automated smoke tests on isolated data. It was not checked on
  the real data, to avoid any chance of writing to the real stores.
