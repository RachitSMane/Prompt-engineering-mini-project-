# Final experiment results (verified on 2026-11-14)

Every inference ran locally on Ollama (`qwen2.5:3b` for extraction and summaries, `llama3.2` as judge). No mock data
appears in this report.

## D1 — Structured extraction (run `20261111-030912_qwen2.5-3b`, 299 reviews per strategy)

| Strategy | Valid | Controlled failures | Validity | First-attempt valid | Mean latency |
|---|---|---|---|---|---|
| Zero-shot | 282 | 17 | 94.3 % | — | 41.2 s |
| Few-shot | 285 | 14 | 95.3 % | 282 | 53.8 s |
| Chain-of-thought | 279 | 20 | 93.3 % | — | 47.5 s |

**Few-shot repairs:**
- 85 records needed a deterministic local fix;
- 17 needed an LLM repair, of which 3 succeeded and 14 stayed controlled failures.

The most common errors before repair were invalid JSON, an invalid aspect category, an invalid complaint enum, and
evidence that wasn't verbatim or was longer than 300 characters.

**Export check:** `scripts/verify_d1_exports.py` passes all 9 consistency checks for every strategy (see
`reports/d1/export_verification.md`). It confirms that:
- the JSON matches the stored predictions and re-validates with grounding;
- the CSV values equal the JSON;
- there is one aspect row per aspect;
- the failures CSV equals the stored failures;
- every review is accounted for.

## D2 — Prompt comparison

### Against the AI-assisted reference labels (the only completed evaluation)

The reference is 299 labels: **1 human and 298 AI-assisted**. **This is agreement with AI-assisted labels, not the
required evaluation against 300 manual labels.**

| Strategy | Macro-F1 | Accuracy | Macro P | Macro R | Weighted F1 | Failures (scored wrong) |
|---|---|---|---|---|---|---|
| Zero-shot | 0.4380 | 0.6120 | 0.6611 | 0.4659 | 0.6797 | 17 |
| **Few-shot** | **0.6025** | **0.8127** | 0.6917 | 0.5823 | 0.8170 | 14 |
| Chain-of-thought | 0.4232 | 0.5753 | 0.7201 | 0.5831 | 0.6461 | 20 |

Per-class F1 (positive / negative / neutral / mixed; supports 111 / 141 / 5 / 42):
- zero-shot 0.682 / 0.852 / 0.048 / 0.170;
- few-shot 0.923 / 0.864 / 0.167 / 0.456;
- CoT 0.618 / 0.840 / 0.101 / 0.133.

Confusion matrices (rows = reference, columns = prediction, order positive, negative, neutral, mixed):
- zero-shot `[[59,2,44,0],[0,118,16,1],[0,3,2,0],[3,13,17,4]]`;
- few-shot `[[102,3,2,1],[1,127,3,1],[0,4,1,0],[7,19,1,13]]`;
- CoT `[[51,1,51,0],[0,113,20,0],[0,0,5,0],[3,14,18,3]]`.

Zero-shot and CoT push positive reviews into neutral. All three strategies miss most mixed reviews.

`scripts/verify_metrics_independent.py`, which uses the standard library only, was re-run on 2026-11-13: all values
match, exit code 0.

### Sensitivity analysis: the approved 300-review reference dataset

- **Reference set:** the 300 existing reference annotations, validated (300 unique ids, 0 invalid, 0 missing; 1
  human-created, 299 AI-assisted). A named reviewer accepts them on the Human Annotation page; the approval is
  stored separately and the provenance is unchanged.
- **Status:** 300 / 300 accepted (299 bulk + `R5cfb12ef75b4` individually, reviewer Rachit, 2026-11-14 16:48–16:53 UTC; no label corrected; provenance unchanged: 1 human-created, 299 AI-assisted).
- **Evaluation** (`approved_reference/metrics.json`, 2026-11-14 16:55 UTC, 300 reviews): macro-F1 0.4435 / **0.6322**
  / 0.4276, accuracy 0.6133 / 0.8133 / 0.5767 (zero-shot / few-shot / CoT). Recomputed independently: 72 / 72
  values identical.
- **The only difference from the primary table above is `R5cfb12ef75b4`:** flagged as garbled and excluded before
  the run. Its AI label is neutral, at confidence 0.30, and marked debatable. All three strategies predicted
  neutral for it, so it adds one correct neutral and changes neutral F1, and therefore macro-F1 (+0.0055 /
  +0.0297 / +0.0044).
- **The primary D2 result remains the 299-review table above.** Details: `D2_FINAL_REPORT.md` §14.

### Independent human annotations

| | |
|---|---|
| Completed human-created annotations (updated 2026-11-15) | **300 / 300** (0 pending, 0 flagged) |
| Second-person verified | not required by the project; 0 |
| Status of `scripts\human_annotation.py evaluate` | `awaiting_human_labels` (written to `outputs/live/experiments/20261111-030912_qwen2.5-3b/human_reference/`) |

No metrics against the human annotations have been computed (the reported D2 results are unchanged). The pipeline is ready:
- one shared review set for all three strategies;
- failures scored as wrong;
- an independent stdlib cross-check, which the tests cover.

When 300 labels are verified, run:
`.venv\Scripts\python.exe scripts\human_annotation.py evaluate --run 20261111-030912_qwen2.5-3b`.

**No new inference is needed for the 299 sample reviews that already have predictions.**

## D3 — Summaries and judge

Figures re-verified on 2026-11-14 against the saved files.

- **Summaries:** 15/15, generated from statistics and evidence with cited review ids.
- **Evidence audit (deterministic):** 130/130 cited ids valid and from the right product; statistics reproduce
  15/15; **13/15 summaries fully verified**. B0BTY63SFY has misattributed 80 % / 67 % (real ratios, wrong denominator); B001DHKBXU gives 7 complaint
  counts "out of 15 reviews" although the product has 19. One point in B07XG8QQL5 is weakly supported.
- **Judge v1** (original, preserved): 5 ok / 10 failed. v1 negative control: faithfulness lower in 0/3 pairs,
  invented feature flagged 0/3.
- **Judge v2** (fixed structure): **15/15 valid, 0 failed, 0 pending**. Means F 3.27, C 3.87, K 2.87, H 4.00
  (H = 4 for all 15). 7/15 needed repairs, each after a first answer that listed no claims.
- **Judge v2 negative control (all 15 products, 30 planted claims):** planted claim named in **18/30 (60 %, 95 % CI
  42–75 %)**, **12 missed**. Invented feature 10/15, fabricated statistic 8/15. Faithfulness lower 2/30,
  hallucination lower 2/30. **The judge is not a reliable faithfulness measure.**

- **Semantic reading of the 30 negative-control judgements** (model-assisted, not human ground truth, not combined with
  the keyword rate): 1 paraphrased detection, 2 self-contradictory verdicts, 1 planted claim used as fact, 8 not mentioned.
- **Corrected copies** of B001DHKBXU and B0BTY63SFY (deterministic, separately stored) pass the audit. Every v1/v2
  score above refers to the originals. The copies have only their own judge v3 judgements.
- **Judge v3** (separate run, 2026-11-14; one call per summary, no repair, raw response kept): 47/47 valid, 0
  validation failures.
  - Originals: F 2.47, C 3.27, K 2.60. The hallucination score contradicts the judge's own empty claim list in
    **12/15** (reported as None, not repaired).
  - Corrected copies: B001DHKBXU scored identically to its original; B0BTY63SFY F +1, K -1; no claims on either.
  - v3 negative control on the same 30 copies: **5/30** detected by the claim list (v2: 18/30). In all 25 misses the
    planted keyword appears only in the justification text.
  - **v3 is also not a reliable faithfulness measure.** Details: `D3_JUDGE_EVALUATION.md` §6.

Details: `D3_COMPLETION_REPORT.md`, `D3_JUDGE_EVALUATION.md`, `D3_NEGATIVE_CONTROL.md`, `D3_EVIDENCE_AUDIT.md`.
