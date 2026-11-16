# Documentation consistency audit (2026-11-14, final)

This audit is non-destructive. No LLM or Ollama call was made. No prediction, summary, judgement, annotation,
negative-control or backup file was changed. Nothing was committed or pushed. The inconsistencies below were
recorded **before** any document was edited.

## 1. Reference values (re-checked from the saved files during this audit)

| Item | Value | Source checked |
|---|---|---|
| Full test suite | 241 passed, 1 skipped (optional `langchain_groq`), 0 failed | `pytest -q -rs`, re-run in this audit |
| D1 export verification | 3 / 3 strategies PASS (9 / 9 checks each, 27 / 27); output identical to `reports/d1/export_verification.md` | `scripts/verify_d1_exports.py`, output to a scratch file |
| Negative control, keyword rule on the claim text | 18 / 30 detected (60 %, Wilson 95 % CI 42–75 %), 12 missed; old rule 19 / 30 kept separately | `judge_negative_control_v2/.../summary.json` |
| Semantic review (model-assisted, Claude; not human ground truth) | 18 keyword / 1 paraphrase / 2 explanation-only / 1 accepted as fact / 8 not mentioned (n = 30) | `semantic_review.json` |
| Corrected copies | B001DHKBXU (`2f2f868f0ec7-corr1`) and B0BTY63SFY (`a61df84f4901-corr1`), stored separately; no score or judgement fields | `corrected_summaries/.../corrected_summaries.jsonl` |
| D2 human store | 0 / 300 verified; 1 completed (R3314d62a5df6, Akash); 299 pending; 300 still required | `scripts/human_annotation.py status` |
| Protected files | predictions (3), `metrics.json`, summaries, v1/v2 judgements, v1 failures, `annotations.json`, `human_annotations.json` byte-identical to the 2026-11-14 backup; `pairs.jsonl` 30 rows, first 14 identical | `cmp` against `Downloads\ReviewIQ-pre-final-backup-20261114` |
| Backup manifest | `sha256sum -c MANIFEST.sha256`: OK | backup folder (read only) |

## 2. Files inspected

`DELIVERABLE_COMPLETION_REPORT.md`, `FINAL_TEST_REPORT.md`, `RESUME_CHECKPOINT.md`, `D3_COMPLETION_REPORT.md`,
`D3_NEGATIVE_CONTROL.md`, `D3_JUDGE_EVALUATION.md`, `D3_EVIDENCE_AUDIT.md`, `FINAL_EXPERIMENT_RESULTS.md`,
`HUMAN_ANNOTATION_GUIDE.md`, `execution_log.md`, `reports/README.md`, `DEMO_GUIDE.md`, `DEMO_QA.md`,
`DEMO_TALK_TRACK.md`, `DEMO_CHECKLIST.md`, `reports/d1/export_verification.md`, the repository `README.md`, and
`reports/final/ReviewIQ_Final_Report.html` (text search only).

There are **no separate D1 or D2 completion reports** in the repository. The D1 and D2 status lives in
`DELIVERABLE_COMPLETION_REPORT.md` (matrix and verdicts), `FINAL_EXPERIMENT_RESULTS.md` and
`HUMAN_ANNOTATION_GUIDE.md`. None was created, because doing so would duplicate those reports.

## 3. Findings

### Consistent (no change needed)

- The test count is 241 / 1 / 0 in `FINAL_TEST_REPORT.md`, `RESUME_CHECKPOINT.md` (status table),
  `DELIVERABLE_COMPLETION_REPORT.md` and `execution_log.md` session 6. The 15 tests listed as added match
  226 → 241.
- `execution_log.md` also contains 115, 120, 129 and 235. These are timestamped historical entries, superseded
  by the later sessions. They were left unchanged, because the log is a record.
- `RESUME_CHECKPOINT.md` stage table (129 passed; old 3-pair negative control) is explicitly headed "historical
  (2026-11-12)", below the 2026-11-14 status table. It was left unchanged.
- The negative control is reported as 18 / 30 detected and 12 missed in every main report. The semantic review is
  always separate, labelled model-assisted (Claude), and called "not human ground truth". 19 / 30 appears only as
  the superseded rule or as the separately labelled paraphrase context.
- Corrected copies: every D3 report says they exist separately, the originals are unchanged, and they are not judged.
  No score is attached to them in the data file.
- D2: every main report gives 0 / 300 verified and 1 completed awaiting a second person, and keeps software and
  human work apart. None claims the human-ground-truth requirement is met.
- Judge reliability: every main report says the judge is **not** reliable. No report claims a 0 % false-positive
  rate; the 0 / 30 keyword control is explicitly "not a false-positive rate".
- Read-only demo (6 pages, 0 exceptions, 1 `GET /api/tags`, 0 generation calls, 0 files changed) and D1
  (27 / 27) are stated identically in `FINAL_TEST_REPORT.md` and `DELIVERABLE_COMPLETION_REPORT.md`, and match
  this audit's re-run.

### Inconsistencies found

| # | File | Problem | Action |
|---|---|---|---|
| 1 | `README.md` §8 | "Re-labelling a review on the *Human Annotation* page replaces it with a human label." This is no longer true and is misleading: human labels go to the separate blind store, and the legacy file is read-only | corrected |
| 2 | `README.md` §19 | Human Annotation page described as "browse/edit the 300 reference labels" | corrected |
| 3 | `README.md` §22 | "Product summaries use 20 reviews per product". The final summaries use the 299-review sample (9–42 per product) | corrected |
| 4 | `reports/README.md` | Says "Status: pending … no live LLM experiment has been run … this folder contains no results" | status replaced with a pointer to the current reports |
| 5 | `DEMO_GUIDE.md`, `DEMO_QA.md`, `DEMO_TALK_TRACK.md`, `DEMO_CHECKLIST.md` | Written for the 2026-11-12 state: judge v1 only (5 / 15), 3-pair negative control (0 / 3), "160 passed", the old annotation page. `DEMO_GUIDE.md`'s 2026-11-13 note says v2 is "5 judged, 10 pending" (now 15 / 15) | dated "superseded" notice added at the top of each; expected test count in `DEMO_CHECKLIST.md` and `DEMO_QA.md` updated to 241 / 1. The script bodies are otherwise unchanged (they need rewriting by the presenter) |
| 6 | `reports/final/ReviewIQ_Final_Report.html` / `.pdf` (2026-11-12) | Contains the old negative control ("faithfulness decreased in 0 of 3 scored pairs") and the v1-only judge | **not regenerated** (that would overwrite a deliverable); documented as superseded in `reports/README.md` and `DELIVERABLE_COMPLETION_REPORT.md` §5 |
| 7 | `HUMAN_ANNOTATION_GUIDE.md` line 3 | Header "Status on 2026-11-13", although the figures were re-checked on 2026-11-14 (they are unchanged) | date updated |
| 8 | `D3_COMPLETION_REPORT.md` §2 | The per-product table shows judge v2 scores in the same row as "corrected copy available". This could be read as scores for the corrected copies | footnote added: the scores belong to the original summaries |

## 4. What remains (unchanged by this audit)

- **D2:** 300 human-verified labels, by two different people. Then `scripts\human_annotation.py export` and
  `evaluate`.
- **D3:** not fully compliant. 2 of the 15 original summaries fail the evidence audit (the corrected copies are
  unjudged), and the judge is unreliable. Closing either needs new inference and your approval.
- **Optional:** rewrite the demo scripts and regenerate the final HTML/PDF and screenshots; have a human check the
  B07XG8QQL5 weak point and the judge's claims; decide whether to commit.
