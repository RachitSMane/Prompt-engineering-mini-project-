# ReviewIQ — likely mentor questions and accurate answers

> **Superseded figures (audit 2026-11-14).** This file was written for the 2026-11-12 state. Current results: judge v2 valid for 15 / 15 summaries (v1 5 / 15 preserved); negative control on 15 products with 30 planted claims, 18 / 30 detected and 12 missed (the judge is not reliable); evidence audit 13 / 15; D2 0 / 300 verified human labels; tests 269 passed, 1 skipped (final pass; judge v3 and the reference-approval workflow added). Use `MENTOR_DEMO_GUIDE.md` instead of this file.

All figures are from the saved results: D2 run `20261111-030912_qwen2.5-3b/metrics.json`, D3 source
`dataset__few_shot__qwen2.5-3b__v1`, negative control
`outputs/live/d3/judge_sanity/dataset__few_shot__qwen2.5-3b__v1/summary.json`.

## Data and labels

**Q: Where does the data come from?**
Amazon Reviews 2023 (McAuley Lab, UCSD), Appliances category. 15 products × 100 reviews = 1,500 reviews, sampled with
seed 42 and filtered by length, with duplicates removed (`data/processed/dataset_card.json`).

**Q: How were the ground-truth labels made? Are they human?**
Mostly not. Of the 300-review stratified sample, 299 are labelled and 1 is flagged as unreadable and excluded. Of the 299
evaluable labels, **1 is human and 298 are AI-assisted**. That provenance is recorded per label (note prefix
"AI-assisted label", plus `meta.annotation_method` in `annotations.json`) and shown in the app. So D2 measures **agreement
with AI-assisted reference labels**, not accuracy against human judgement. The next step would be human re-annotation of at
least a subset, with an inter-annotator agreement check (e.g. Cohen's κ).

**Q: Couldn't the AI-assisted labels be biased towards the model you're evaluating?**
Yes, that's a real risk: if the labelling model shares biases with qwen2.5, agreement is inflated. That's exactly why I
present the D2 numbers as agreement with reference labels and list human validation as the main open issue.

## Deliverable 1: structured extraction

**Q: How do you guarantee valid structured output?**
Three steps. (1) A Pydantic schema with fixed enums for sentiment, aspect category and complaint category. (2) Deterministic
local fixes first (77 records needed one), then up to 2 LLM repair prompts that send back the validation errors (3 records
needed one). (3) Strict evidence grounding: each aspect's evidence must be a quote from the review. If nothing validates,
the record is a **controlled failure** (14 of 299 for few-shot), stored and reported, never silently accepted or re-sampled.

**Q: Is what you showed real model output or pre-computed?**
It is real local inference (Ollama `qwen2.5:3b`, run on 2026-11-11), **saved and shown from cache**. The demo runs in
read-only mode, so no new inference happens; the Review Analysis page says so ("Saved results · cached, not newly
generated · 0 LLM calls"). Mock fixtures exist only for offline development, live in a separate folder, and are labelled
"Mock" whenever they are shown.

**Q: Can you run it live now?**
In normal mode, yes, but a few-shot extraction takes about 54 s per review on this laptop (mean latency 53.8 s), and it
adds to the saved extraction cache. To protect the reported results I demo from saved outputs. If needed I can start a
separate normal-mode instance afterwards and run a single review.

## Deliverable 2: prompt comparison

**Q: What are the results?** (overall sentiment, 4 classes, 299 reviews)

| Strategy | Macro-F1 | Accuracy | Weighted F1 | Valid outputs | Mean latency | Mean input tokens |
|---|---|---|---|---|---|---|
| Zero-shot | 0.438 | 0.612 | 0.680 | 282 / 299 (94.3 %) | 41.2 s | 2,057 |
| **Few-shot** | **0.6025** | **0.813** | **0.817** | 285 / 299 (95.3 %) | 53.8 s | 2,885 |
| Chain-of-thought | 0.423 | 0.575 | 0.646 | 279 / 299 (93.3 %) | 47.5 s | 2,298 |

**Q: Why macro-F1?**
The classes are imbalanced (141 negative, 111 positive, 42 mixed, 5 neutral). Macro-F1 weights each class equally, so a model
can't score well by getting only the majority classes right.

**Q: Why did chain-of-thought do worse?**
From the confusion matrices: zero-shot and CoT move many positive reviews into **neutral**. CoT labels 51 positives neutral,
so its neutral precision is 0.05 even though its neutral recall is 1.0. Few-shot's examples anchor the label boundaries
(102 positives correct). A 3B model doesn't seem to benefit from free-form reasoning here. That's an observation from one
model and one dataset, not a general claim.

**Q: What are few-shot's weaknesses?**
"Mixed" reviews: recall is only 0.31 (13 of 42), and most of the rest are called negative. Neutral has only 5 examples, so
its per-class numbers are too noisy to interpret.

**Q: Were failed outputs counted?**
Failed extractions count as wrong predictions in the classification metrics, so a strategy can't gain by failing on hard
reviews. Each review got exactly one attempt per strategy (plus the in-call repairs); failures were never re-sampled.

**Q: Are the few-shot examples taken from the test set?**
No. They are 5 fixed **synthetic** examples, not from the evaluation sample (stated in the strategy description in the app).

## Deliverable 3: summaries and judge

**Q: How do you stop the summary from hallucinating?**
It is generated from pandas-computed statistics and verbatim evidence quotes, not from free text, and every point must cite
review IDs, which the UI links back to the full reviews. Deterministic checks (no LLM) then verify that every percentage in
the summary appears in the statistics, that the top aspects and complaints are mentioned, and that the cited IDs belong to
the product. All 5 judged summaries pass the percentage and cited-ID checks. These checks don't catch qualitative
misstatements, though.

**Q: How reliable is the LLM judge?**
Not reliable, and I report that openly.
- **Coverage:** `llama3.2` produced schema-valid output for only **5 of 15** summaries. The other 10 failed validation even
  after repairs, mostly by nesting `unsupported_claims` inside each score. Those are stored as final failures; I did not
  re-run the judge to fish for scores.
- **Internal inconsistency:** the 5 valid judgements score hallucination 1–2 out of 5 (5 = no unsupported claims), yet 4 of
  them list no unsupported claims. For Napco B09WWM1WLS the justification literally says "does not introduce any unsupported
  claims" next to a score of 2.
- Mean scores of the 5: faithfulness 3.8, coverage 3.8, consistency 3.0, hallucination 1.4.

**Q: Did you validate the judge?**
Yes, with a negative control. I corrupted summaries three ways: inverted opinion with a fabricated percentage, an invented
feature ("built-in Wi-Fi app control"), and all complaints removed. Only originals with a valid judgement were used, so 5
were eligible: 3 pairs were scored and 2 corrupted judgements failed. Results:
- the mean score was lower for the corrupted summary in **3/3** pairs (3.17 → 2.92 on average);
- **faithfulness was lower in 0/3** (4.0 for both);
- the fabricated percentage was flagged in 3/3, the invented feature in **0/3**.
Conclusion: on this sample the judge does **not** discriminate reliably on faithfulness. With n = 3, nothing statistical
can be claimed either way. I would treat the judge scores as exploratory and rely on the deterministic checks and the cited
evidence.

**Q: Why a different model as judge?**
To reduce self-preference bias (a model grading its own output). The cost is that llama3.2 (3B) is weak at strict JSON
output, which caused the failures. A larger judge, or a judge with constrained/structured decoding, is the obvious next step.

**Q: Could you fix the judge failures by parsing them differently?**
Possibly: most failures are a consistent nesting mistake that could be fixed deterministically. But the raw judge outputs
of those 10 calls weren't stored, so applying it would mean re-running the judge. To stay fair, that would have to be a
separately labelled re-run of all 15 summaries, not a retry of just the failed ones. It hasn't been done.

## Engineering

**Q: Why local models?**
Reproducibility, no API costs, and no data leaving the machine. Ollama on `127.0.0.1` only; remote endpoints are refused
unless explicitly allowed in config.

**Q: What is "demo mode"?**
`REVIEWIQ_DEMO_READONLY=true`. The backend refuses every result write and every LLM client creation or call, and the UI
disables the buttons and explains why. It's covered by 31 tests, including one that renders all six pages against the real
saved results and checks that the files are byte-identical afterwards. Full suite (2026-11-14, final): 269 passed, 1 skipped.

**Q: What would you do next?**
1. Human annotation (at least a subset) plus agreement statistics.
2. A stronger or constrained-decoding judge, re-run on all 15 summaries as a labelled new run, plus a larger negative
   control.
3. Extract all 1,500 reviews (currently only the 299-review evaluation sample, 9–42 reviews per product).
4. Better handling of "mixed" sentiment (few-shot examples, or a two-stage classifier).
