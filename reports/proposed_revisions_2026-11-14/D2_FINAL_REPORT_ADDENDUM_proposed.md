# PROPOSED addendum to `reports/D2_FINAL_REPORT.md` (not applied)

Insert after §9 (Overall results), or as Appendix B. All values come from
`outputs/live/experiments/20261111-030912_qwen2.5-3b/approved_reference/metrics.json` and were reproduced
independently from the raw files (see `D2_CONSISTENCY_AUDIT.md`).

---

## 9b. Evaluation on the approved 300-review reference dataset

On 2026-11-14 the team reviewed the 300 existing reference annotations and recorded its approval: 299 bulk-approved
and the formerly flagged review `R5cfb12ef75b4` individually approved, by reviewer "Rachit"
(`data/annotations/reference_adjudication.json`). The original annotations and their provenance are unchanged
(1 human-created, 299 AI-assisted). The three strategies were then scored on all 300. For `R5cfb12ef75b4`, the
predictions come from a separate 2026-11-14 session with the same model, prompt version and temperature
(`supplementary/`).

| Strategy | n | Accuracy | Macro P | Macro R | **Macro F1** | Weighted F1 | Valid / invalid |
|---|---|---|---|---|---|---|---|
| Zero-shot | 300 | 0.6133 | 0.6642 | 0.4909 | 0.4435 | 0.6780 | 283 / 17 |
| **Few-shot** | 300 | **0.8133** | 0.7185 | 0.6156 | **0.6322** | **0.8172** | 286 / 14 |
| Chain-of-Thought | 300 | 0.5767 | 0.7226 | 0.5831 | 0.4276 | 0.6446 | 280 / 20 |

Per-class F1 (support positive 111, negative 141, neutral 6, mixed 42):

| Class | Zero-shot | Few-shot | CoT |
|---|---|---|---|
| positive | 0.6821 | 0.9231 | 0.6182 |
| negative | 0.8520 | 0.8639 | 0.8401 |
| neutral | 0.0698 | 0.2857 | 0.1188 |
| mixed | 0.1702 | 0.4561 | 0.1333 |

**Difference from §9 (299 reviews):** exactly one review. `R5cfb12ef75b4` has the reference label *neutral*
(AI-assisted, confidence 0.30, "DEBATABLE"; garbled text), and all three strategies predicted *neutral*. It is the
6th neutral example, so neutral F1 rises and macro-F1 increases by +0.0055 / +0.0297 / +0.0044. Accuracy changes by
about 0.001. The positive, negative and mixed results are identical, and so is the ranking.

This illustrates the class-imbalance point in §11: one rare-class example can move macro-F1 by up to 0.03.

**Which figures are the D2 result.** The 300-review evaluation is the evaluation on the approved D2 reference
dataset (status *complete*, independent cross-check passed). The 299-review figures in §9 are the original run,
with the unreadable review excluded before the run. They are reported as a sensitivity analysis. Both are agreement
with approved, mostly AI-assisted reference labels, **not** independent human ground truth.
