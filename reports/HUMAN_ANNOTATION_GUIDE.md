# Human annotation guide (Deliverable 2 ground truth)

**Status on 2026-11-15:** all **300 independent human annotations are complete** (300 human-created, blind labels;
0 pending; 0 flagged; annotators Akash, Rachit, p.chinni krishna reddy and pranav menon). **Second-person
verification is not part of this project's required workflow** (project decision, 2026-11-15), so these labels are
human-created but have **not** been independently verified by a second annotator. They are stored separately from
the AI-assisted reference annotations and are never merged with them. The reported D2 numbers measure agreement with
the approved (mostly AI-assisted) reference labels, not with these human annotations.

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
| Human-created / Completed / Pending / Flagged annotations | the separate blind workflow below (300 / 300 / 0 / 0) |

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

## 0. Quick card for independent human annotators

The 300 annotations are complete; this card describes how they were made (and how a replacement for a flagged review
would be labelled).

1. Start the app in **normal mode** (§3), open **Human Annotation**, and type your own name.
2. **Label (blind):**
   - read the review only, and decide from the text;
   - pick the overall sentiment, then each aspect with its sentiment, then the complaints;
   - click **Save & next**.
   - Unreadable or ambiguous review? Use **Flag review** with a reason.
   - Labels save immediately, so you can stop at any time and continue later.
3. **Never** copy the AI-assisted labels from the legacy tab, and never label from the model's predictions. A label
   counts only if a person decided it.
4. Check progress at any time:
   `.venv\Scripts\python.exe scripts\human_annotation.py progress-report --out reports\d2\annotation_progress.md`
5. Annotation is **complete** when no review is pending (every queued review has a human label or a flag).
   No second-person verification step is required.

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
| `human_created` | A human labelled the review **blind** (no AI label or prediction shown) | yes (all 300 current labels) |
| `human_verified` | A human checked an AI-assisted label that **was shown** and accepted or corrected it | reported separately (none exist) |
| `ai_assisted` | Model-generated, not decided by a human | **never**. These labels cannot be stored in the human file (enforced by the schema) |

| Status | Meaning |
|---|---|
| `pending` | In the queue, no human label yet |
| `completed` | Human label saved — the annotation is complete |
| `flagged` | Unusable or ambiguous (reason required); excluded; replace it with a reserve review |
| `verified` | Legacy/optional only: a different person confirmed the label. The app no longer offers this step and no current label has it; the status stays valid so stored records always load |

**Completion** (`progress()['annotation_complete']`, shown on the page and by `progress-report`): no review in the
queue is `pending`. Completed labels are human-created; they are **not** described as verified anywhere.
Every action is appended to the record's `history`.

**No silent overwrites (enforced in `annotation/human_annotation.py` since 2026-11-14, tested):**
- Only the annotator who made a label can re-label, flag or remove it (names compare case-insensitively). Anyone else
  gets an `AnnotationConflict` error, and the label is unchanged.
- A checked-AI label never replaces a blind label, and vice versa.
- CSV import never overwrites an existing label, not even the importing annotator's own, and the check runs inside
  the lock. A review that appears twice in one file is imported once.
- The store refuses any write that does not go through the locked, re-reading transaction (`RuntimeError`).
  Migration of the legacy human label is idempotent: a rerun changes nothing and keeps the label's history.
- An AI-assisted label has no path to a verified human label. It cannot be stored in the human file (schema), cannot
  be migrated (only the documented human id is copied), and cannot be imported (AI marker or provenance rejected).
  These rules are covered by tests.
- Every change takes a short lock (`human_annotations.json.lock`) and re-reads the saved file first. Two people with
  the app open at the same time therefore can't erase each other's labels. A lock left by a crashed session expires
  after 60 s.

**Current count (2026-11-15):** 300 / 300 completed human-created annotations, 0 pending, 0 flagged. One of them,
R3314d62a5df6 (by Akash), was migrated from the legacy file; it was made at 12:20 UTC on 2026-11-10, before the
AI-assisted labels were generated at 12:34, so it was blind. None has been verified by a second annotator, and none
needs to be under the project's workflow.

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
2. **Queue & export:** filter by status (with annotator and last-update time per review). Download all **human
   annotations** (CSV).
   - Offline route: fill the **blank template**, which contains no AI labels, and use *Import as blind human labels*.
     Rows carrying the AI marker or another provenance are rejected.

The order you label in doesn't matter. At about 30 s per review, labelling 300 takes about 2.5 h.

## 4. Command-line helpers (no LLM calls)

```powershell
.venv\Scripts\python.exe scripts\human_annotation.py status                 # completed / pending / flagged counts
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

**Note (2026-11-15):** `export`, `evaluate` and `finalise` still use only `verified` labels as ground truth, so they
never present the completed (unverified) labels as verified. With the current data they find 0 verified labels;
`finalise` refuses and changes nothing. An analysis against the 300 completed human annotations is possible with
`evaluate --run <run> --include-unverified`; its report is explicitly marked as including unverified labels. It has
not been run, and it would not change the reported D2 results.

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

- No human label has been created, copied or inferred by software. All 300 were entered by the named annotators.
- The 300 human annotations are human-created and complete; they have **not** been independently verified by a
  second annotator, and that step is not part of this project's required workflow.
- The AI-assisted labels stay AI-assisted everywhere: in the legacy tab, in the reports, and in the provenance of
  the original `metrics.json`. Human and AI-assisted annotations are never merged.
- The D2 results (primary: the 299-review run; sensitivity: the 300-review approved-reference run) are unchanged and
  remain agreement with the approved, mostly AI-assisted reference labels.
