# Human annotation guide (Deliverable 2 ground truth)

**Status on 2026-11-14:** 1 genuine human label exists (completed, **not yet verified**). **0 of 300** labels are
verified, so **300 verified human labels are still needed**. The D2 numbers reported so far measure agreement with
AI-assisted reference labels, not with human annotation.

## 00. D2 reference dataset: approving the existing annotations (2026-11-14)

The **D2 comparison dataset** is the 300 existing reference annotations in `annotations.json` (1 human-created, 299
AI-assisted, recorded as mentor-approved at generation). They don't have to be re-entered. The top of the
Human Annotation page shows:

| Indicator | Meaning (all from stored records) |
|---|---|
| Reference annotations available | validated records (300 / 300: unique ids, valid fields, product ids match the sample) |
| Reviewed / approved | records a named reviewer accepted (approved or corrected) |
| Awaiting review | records with no decision yet |
| D2 reference dataset | **Ready** when all 300 are accepted |
| Independent human-created labels / Second-person verified | the separate blind workflow below (1 / 0) |

**How to approve:**
1. In normal mode, type your name and tick the confirmation.
2. Click **Approve Existing Reference Annotations**. The 299 complete records are approved once; a second click
   changes nothing.
3. The flagged record `R5cfb12ef75b4` (garbled text) needs an *Individual decision*: approve, correct the sentiment,
   or exclude it with a reason. Any record can later be corrected or excluded, and its history is kept.

**What is stored:** decisions go to `data/annotations/reference_adjudication.json`, with reviewer, time, method and
a hash of the original label. `annotations.json` is never modified. Each accepted record keeps its
`original_provenance`, so an AI-assisted label stays AI-assisted. If a source label changes later, its approval
becomes stale and must be re-done.

**Next:** Prompt Experiments -> **Evaluate against approved references** (no LLM). CLI equivalent:
`scripts\reference_annotations.py validate | status | approve --reviewer "Name" --confirm | decide | export | evaluate --run <run>`.

## 0. Quick card for independent human annotators (optional, stronger evidence)

You need **two people**: an annotator (A) and a verifier (B). Each needs about 2.5 hours for 300 reviews.

1. Start the app in **normal mode** (§3), open **Human Annotation**, and type your own name.
2. **A — Label (blind):**
   - read the review only, and decide from the text;
   - pick the overall sentiment, then each aspect with its sentiment, then the complaints;
   - click **Save & next**.
   - Unreadable or ambiguous review? Use **Flag review** with a reason.
   - Labels save immediately, so you can stop at any time and continue later.
3. **B — Verify (second annotator):** type your own name, check each of A's labels, correct it if needed, and click
   **Confirm as verified**. Start with `R3314d62a5df6`, which is already labelled by Akash.
4. **Never** copy the AI-assisted labels from the legacy tab, and never label from the model's predictions. A label
   counts only if a person decided it.
5. Check progress at any time:
   `.venv\Scripts\python.exe scripts\human_annotation.py progress-report --out reports\d2\annotation_progress.md`
6. At **300 verified**, run:
   `.venv\Scripts\python.exe scripts\human_annotation.py finalise --run 20261111-030912_qwen2.5-3b`.
   It exports the reference and evaluates all three strategies. Below 300 it refuses and changes nothing.

## 1. Why there is a new workflow

The old annotation page wrote straight into `data/annotations/annotations.json`, which holds 1 human and 299
AI-assisted labels. When you opened a review, the form was **pre-filled with the AI-assisted label and its note**,
so pressing *Save*, or editing the note, silently turned an AI label into a "human" one. That is now impossible:

- Genuine human labels go to a **separate file**, `data/annotations/human_annotations.json`
  (`annotation/human_annotation.py`).
- The **Label (blind)** tab never shows a model prediction or an AI-assisted label. The form starts empty. If another
  person has already labelled the review, their label is **not** shown either.
- The legacy store now refuses to overwrite an AI-assisted label with an unmarked one, through both the UI path and
  CSV import (`AnnotationStore.save`).
- `annotations.json` is unchanged and only shown read-only, in the tab *AI-assisted reference labels (legacy)*.
- The single documented human label (`R3314d62a5df6`, by Akash, saved before any AI label existed) was copied into the
  human store as a blind label, with the action `migrated_from_annotations_json` in its history. No AI-assisted
  label was copied.

## 2. Provenance and status

| Provenance | Meaning | Counts towards the 300? |
|---|---|---|
| `human_created` | A human labelled the review **blind** (no AI label or prediction shown) | yes, once verified (strongest evidence) |
| `human_verified` | A human checked an AI-assisted label that **was shown** and accepted or corrected it | yes, once verified; reported separately |
| `ai_assisted` | Model-generated, not decided by a human | **never**. These labels cannot be stored in the human file (enforced by the schema) |

| Status | Meaning |
|---|---|
| `pending` | In the queue, no human label yet |
| `completed` | First human label saved |
| `flagged` | Unusable or ambiguous (reason required); excluded; replace it with a reserve review |
| `verified` | A **different person** confirmed or corrected the completed label |

**Eligible for D2:** status `verified` and provenance `human_created` or `human_verified`. Verifying your own label is
refused. Re-labelling a verified review sets it back to `completed`. Every action is appended to the record's
`history`.

**No silent overwrites (enforced in `annotation/human_annotation.py` since 2026-11-14, tested):**
- Only the annotator who made a label can re-label, flag or remove it (names compare case-insensitively). Anyone else
  gets an `AnnotationConflict` error, and the label is unchanged. Disagreements are resolved on the **Verify** step,
  where corrections are logged with the previous values.
- A verified label cannot be flagged (remove it explicitly first) or verified a second time.
- A checked-AI label never replaces a blind label, and vice versa.
- CSV import never overwrites an existing label, not even the importing annotator's own, and the check runs inside
  the lock. A review that appears twice in one file is imported once.
- The store refuses any write that does not go through the locked, re-reading transaction (`RuntimeError`).
  Migration of the legacy human label is idempotent: a rerun changes nothing and keeps the label's history.
- An AI-assisted label has no path to a verified human label. It cannot be stored in the human file (schema), cannot
  be migrated (only the documented human id is copied), and cannot be imported (AI marker or provenance rejected).
  Verification requires an existing human label. These rules are covered by tests.
- Every change takes a short lock (`human_annotations.json.lock`) and re-reads the saved file first. Two people with
  the app open at the same time therefore can't erase each other's labels. A lock left by a crashed session expires
  after 60 s.

**Current count (2026-11-14):** 0 verified human labels out of 300. One label is completed and awaits a second
person: R3314d62a5df6, by Akash, migrated from the legacy file. It was made at 12:20 UTC on 2026-11-10, before the
AI-assisted labels were generated at 12:34, so it was blind. **300 verified labels are still required.** Two different
people are needed: an annotator and a verifier.

## 3. Running the workflow

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
# normal mode (NOT demo mode: demo mode blocks every save)
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open **Human Annotation**.

1. **Label (blind):** enter your name, read the review (the star rating is hidden unless you toggle it), choose the
   overall sentiment, then the aspects with a sentiment for each, then the complaint categories. Click **Save & next**.
   Use **Flag review** (with a reason) for unusable reviews.
   - Saves are atomic and immediate. Closing the app loses nothing, and it reopens at the first pending review.
2. **Verify (second annotator):** a teammate enters *their* name and sees only labels made by others. They check each
   one, correct it if needed, and click **Confirm as verified**. Corrections are logged with the previous values.
3. **Queue & export:** filter by status. Download the **verified human reference** (CSV) or all human labels.
   - Offline route: fill the **blank template**, which contains no AI labels, and use *Import as blind human labels*.
     Rows carrying the AI marker or another provenance are rejected.

The order you label in doesn't matter. At about 30 s per review, labelling 300 takes about 2.5 h, and verification
roughly as long again (two people).

## 4. Command-line helpers (no LLM calls)

```powershell
.venv\Scripts\python.exe scripts\human_annotation.py status                 # progress towards 300 verified labels
.venv\Scripts\python.exe scripts\human_annotation.py reserve --n 1          # replacement review(s) for flagged ones
.venv\Scripts\python.exe scripts\human_annotation.py export                 # -> data\annotations\human_reference\human_reference.{json,csv}
.venv\Scripts\python.exe scripts\human_annotation.py evaluate --run 20261111-030912_qwen2.5-3b
.venv\Scripts\python.exe scripts\human_annotation.py finalise --run 20261111-030912_qwen2.5-3b   # export + evaluate, only at >= 300 verified
.venv\Scripts\python.exe scripts\human_annotation.py progress-report --out reports\d2\annotation_progress.md
```

The `evaluate` command:
- scores all three strategies on the **same** verified human-labelled reviews;
- counts controlled failures as wrong predictions;
- recomputes every overall-sentiment metric independently with the standard library and compares the two;
- writes the results to `outputs/live/experiments/<run>/human_reference/`, leaving the original `metrics.json` untouched.

Its status is `complete` only when at least 300 verified human labels are evaluated and no prediction is missing.

**Flagged or reserve reviews need predictions.** A reserve review (added to replace a flagged one) has no D2
predictions yet. `evaluate` lists it under `needs_inference`, and the status stays `incomplete` until you run the
three strategies on it. That's 3 calls per review, about 2.5 min with `qwen2.5:3b`. Every review that already has a
prediction is skipped:

```powershell
.venv\Scripts\python.exe scripts\run_experiment.py --resume 20261111-030912_qwen2.5-3b --human-labelled
```

This adds the reviews to the same run, so all three strategies keep one shared review set. Resuming also rebuilds
that run's original `metrics.json` and `report.md` from the legacy labels: the values stay the same, but the
timestamp changes. Back the run folder up before doing this.

## 5. What is (and is not) claimed

- No human label has been created, copied or inferred by software. The only human label is the one Akash made.
- The AI-assisted labels stay AI-assisted everywhere: in the legacy tab, in the reports, and in the provenance of
  the original `metrics.json`.
- D2 can be called fully compliant only after 300 verified human labels exist and
  `scripts\human_annotation.py evaluate` reports status `complete`.
