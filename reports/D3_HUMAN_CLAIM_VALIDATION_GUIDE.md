# D3 human claim validation — single-annotator methodology and guide

## Decision record (2026-11-16)

The project owner decided that judge v4 is validated against **one** human annotator. This replaces criterion 4 of
`reports/proposed_revisions_2026-11-15/D3_INVESTIGATION_AND_COMPLETION_PLAN.md` §14 (two independent annotators,
inter-annotator Cohen's kappa, adjudication of disagreements). The plan, `reports/D3_COMPLETION_REPORT_2026-11-16.md` and the
earlier two-annotator files in `outputs/live/d3/claim_gold/dataset__few_shot__qwen2.5-3b__v1/` (`preregistration.json`,
`agreement.json`, `LABELLING_GUIDE.txt`) are kept unchanged as history. No human label existed when the methodology changed.

## What is measured, and what is not

* One named person labels every one of the **239 claims** of the 15 original summaries exactly once.
* After **all** claims are labelled, judge v4's final claim status is compared with the human labels: exact agreement
  (with 95 % Wilson interval), per-class precision / recall / F1, the 5×5 confusion matrix, flagged-claim
  (unsupported + contradicted) precision / recall / specificity, agreement per judge layer (deterministic vs LLM), Cohen's
  kappa between the judge and this annotator, and every disagreement with both explanations.
* These numbers describe **agreement between judge v4 and one person**. They are not inter-annotator reliability, not a
  human consensus, and the human labels are not universal ground truth.
* The thresholds proposed in the plan (flagged recall ≥ 0.7, precision ≥ 0.6) are reported for information only; meeting
  them is not required for completion, and the result is reported as is.

## Completion rule (criterion C5, `evaluation/d3_status.py`)

C5 passes only when **all** of the following hold. D3 turns green only if C1–C9 all pass.

1. The frozen template `claim_gold_template.csv` has 239 unique claim ids that equal the judge v4 original claims.
2. The label store `data/annotations/d3_claim_validation.json` has exactly one label for each of the 239 claims, no
   duplicates, valid labels, all required fields, one non-AI annotator name, and template / claim-text / evidence hashes
   matching the template.
3. `outputs/live/d3/claim_gold/<source>/single_annotator/comparison.json` was generated from exactly the current label file
   (its SHA-256 is stored). Changing any label afterwards makes it stale until it is regenerated.

## How to annotate

1. Start the app (normal mode; read-only demo mode cannot save labels):

   ```powershell
   cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
   .venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```

2. Open http://127.0.0.1:8501 and go to **Summary Evaluation**. While labelling is incomplete, the page opens in the view
   **Human claim validation (blind)**. All judge results (v1–v4) are hidden in this view.
3. Enter your name once; it is fixed by the first saved label (single annotator).
4. For each claim: read the claim, the product/summary/field, the cited review ids and the evidence (full text of the cited
   reviews, or the product statistics for statistic sentences; the statistics are also available as context for cited
   points). Choose one label, optionally write an explanation, and press **Save label and go to next**.
5. **Resume:** close the browser or stop Streamlit at any time. Every label is saved immediately; on reopening, the page
   continues at the first unreviewed claim.
6. **Navigation:** *Previous* / *Next* and the claim selector. *Show only unreviewed claims* (on by default) hides
   finished claims; turn it off to review saved labels.
7. **Changing a label:** a saved claim is locked. Tick *Revise this label* to change it; the previous label is kept in
   the store's revision history. Saving the same claim twice without revising is refused (no duplicates).
8. **Export:** *Export completed labels as CSV* (labels only, no judge output), or
   `python scripts/claim_gold.py export-labels --source dataset__few_shot__qwen2.5-3b__v1`.
9. When all 239 claims are labelled, run

   ```powershell
   .venv\Scripts\python.exe scripts\d3_completion.py --source dataset__few_shot__qwen2.5-3b__v1 --verify
   ```

   This validates the labels, writes the comparison and the label CSV, runs the full test suite and
   `verify_d3_final.py`, recomputes the completion status (Overview badge) and writes
   `reports/D3_COMPLETION_REPORT_SINGLE_ANNOTATOR.md`.

Progress and integrity at any time: `python scripts/claim_gold.py status --source dataset__few_shot__qwen2.5-3b__v1`.

## Labels

| Label | Use when |
|---|---|
| supported | The evidence directly says what the claim says (reasonable paraphrase is fine). |
| partially_supported | Part of the claim is supported, or the claim is broader, stronger or more specific than the evidence. |
| unsupported | The evidence does not mention what the claim says. |
| contradicted | The evidence says the opposite, or gives a different number. |
| unverifiable | The claim cannot be checked against this evidence (e.g. generic advice), or the evidence is unreadable. |

Statistic sentences (overall opinion, frequent aspects) are judged against the statistics. Improvement areas and
recommendations are supported only if the statistics show complaints about that aspect. Cited points are judged against
the full text of their cited reviews only.

## Blinding and its limits

* The validation view never reads judge v4 output; the comparison is computed only when all claims are labelled, so no
  partial agreement can be seen while labelling.
* Opening **Evaluation results** before labelling is complete shows a warning, and the event is recorded in the label store
  (`events`, with the number of labels at that time); the final report lists such events.
* The annotator is a project member who may already have read earlier judge outputs and the investigation reports, which
  name several erroneous claims. The labelling is therefore blind only within the interface. This is stated in the report.

## Limitations of single-annotator validation

* No estimate of how much people disagree on these labels; one person's misreadings or personal threshold for
  "supported" versus "partially supported" cannot be detected.
* Agreement can be inflated if the annotator's reading was shaped by earlier judge outputs, or deflated by label-boundary
  ambiguity (partially_supported versus unsupported) that a second annotator would expose.
* 239 claims from 15 summaries of one summariser run; per-class sample sizes for contradicted and unverifiable are small.
