# Experiment Methodology

## Research question

For converting e-commerce reviews into structured records, how do **zero-shot**, **few-shot** and
**Chain-of-Thought (CoT)** prompts compare on overall-sentiment classification (single-label),
complaint classification (multi-label) and aspect-level sentiment, measured against human labels?

## Evaluation set — 300 manually labelled reviews

* **Population:** 1,500 cleaned reviews of 15 products. They come from the Amazon Reviews 2023
  *Appliances* category (see `data/README.md`).
* **Sampling:** 300 reviews, stratified by star rating (≈60 per rating, seed 42,
  `scripts/build_annotation_sample.py`). The natural distribution is 72% 5-star. A random sample would hold
  only ~10 reviews per minority rating, so per-class F1 for negative/mixed/neutral would be extremely noisy.
  *Consequence:* the reported accuracy describes this balanced set, not the natural distribution.
  Macro-F1, which weights classes equally, is the headline metric.
* **Labels:** created by a human annotator through the annotation page or the CSV template, following
  `annotation/annotation_guide.md`. The annotator does not see the rating by default. Flagged reviews are excluded.
* **Integrity:** no model output is ever written into `data/annotations/annotations.json`.

## Controlled variables (fair comparison)

| Variable | Control |
|---|---|
| Model | One `LLMClient` instance for all strategies within a run; `config.json` records provider/model |
| Decoding | `temperature = 0` (configurable, recorded) |
| Inputs | Same review IDs, same order, same cleaned text (`utils/data_loader.clean_text`) |
| Label definitions & rules | Identical blocks from `prompts/common.py` / `schemas/taxonomy.py` |
| Output schema | Identical `LLMExtraction` + identical repair procedure and limit |
| Information | The model sees only review text, never the star rating |

Only the prompting technique varies:

* **Zero-shot:** instructions + definitions + review.
* **Few-shot:** the same, plus 5 fixed **synthetic** demonstrations written by the author
  (positive multi-aspect, negative multi-complaint, mixed, compatibility issue, no-aspect neutral).
  They are not drawn from the dataset, so they cannot overlap the evaluation set.
* **CoT:** the same, plus an explicit 7-step analysis procedure. The model performs it silently and returns
  only the JSON and an optional ≤400-character rationale. No reasoning trace is requested or stored.

## Metrics

**Macro averaging rule (all tasks):** macro scores average over the labels that occur in the gold labels
**or** the predictions. This matches scikit-learn's default when no label list is given. A label absent from
both has undefined precision and recall; scoring it as F1 = 0 would lower macro-F1 for reasons unrelated to
the strategy. A label that is only *predicted* is still included, so false positives are penalised.
The labels used are stored as `macro_labels` in `metrics.json`.

### Overall sentiment (4 classes: positive, negative, neutral, mixed)
Accuracy, macro precision/recall/F1 (**primary**), weighted F1, per-class report, confusion matrix.
The confusion matrix covers the four valid classes. Predictions that failed validation are counted separately
as `invalid_predictions`, so a row can sum to less than its support.

### Complaint categories (multi-label, 14 labels)
Micro P/R/F1, macro-F1 (rule above, reported as `complaint_macro_f1`), weighted and sample-averaged F1,
Hamming loss and subset accuracy. A plain macro over all 14 labels is also stored (`macro_f1`).

### Aspects — category-level matching
Free-text aspect names ("ice maker" vs "icemaker") are hard to match reliably, so matching happens at the
**aspect category** level:

1. Human labels give one sentiment per category per review.
2. Predicted aspects are collapsed to one sentiment per category by majority vote, with ties broken by summed
   confidence (`collapse_predicted_aspects`).
3. **Detection:** category sets compared as multi-label (micro P/R/F1).
4. **Aspect sentiment:** for categories present in both gold and prediction, 3-class macro-F1 and a confusion matrix.
   Sentiment quality is measured separately from detection.
5. **Joint:** `category:sentiment` pairs compared as sets (micro F1), the strictest view.

### Failed extractions
A review with no valid output after repair stays in the denominator. Its sentiment counts as wrong and its
sets are empty. Strategies that produce more invalid output are penalised rather than evaluated on an
easier subset. Validity rate, first-attempt validity and repair counts are reported separately.

### Operational metrics
Mean latency per review, mean input/output tokens (LangChain `usage_metadata`), validity rate, and the
number of reviews that needed an LLM repair.

## Evaluation status labels

| Status | Meaning |
|---|---|
| `awaiting_annotations` | No human labels overlap the run; only operational metrics |
| `provisional` | Some, but fewer than 300, human labels |
| `complete` | Evaluated against the full 300-review human-labelled set |
| `mock_demo` | Offline fixture run; numbers are not results |

## Summary evaluation (LLM-as-a-Judge)

* The judge receives the original review texts (truncated to 80 words), per-review structured extractions,
  aggregated statistics and the generated summary.
* Rubric: faithfulness, coverage, consistency and hallucination, each scored 1–5 with anchored level descriptions
  (`prompts/judge.py`). The output is a validated `JudgeVerdict` with justifications and verbatim unsupported claims.
* Deterministic checks run alongside: percentages not found in the statistics, coverage of the top-3 aspects and
  top complaints, and validity of cited IDs.
* **Limitations:** LLM judges can favour fluent text and their own model family. Set `JUDGE_PROVIDER`/`JUDGE_MODEL`
  to use a different model. The exported inspection sample lets a human record agreement with each score.
  Judge scores are reported as model-based assessments, not human evaluation.

## Reproducibility

Each experiment directory stores `config.json` with the settings snapshot, prompt version, strategies,
the exact review IDs and the git commit, alongside per-strategy predictions with raw outputs and repair logs.
`python scripts/run_experiment.py --reevaluate <run_id>` recomputes metrics after annotations change,
without any LLM calls.

## Recommended procedure once an API key is available

1. Finish all 300 annotations on the Data Annotation page.
2. Pilot: `python scripts/run_experiment.py --limit 20`. Check the validity rate and inspect a few raw outputs.
3. Full run: `python scripts/run_experiment.py`, about 900 calls. Resume with `--resume <run_id>` if interrupted.
4. Chain: `python scripts/run_chain.py --per-product 30 --judge`.
5. Fill in the "human agrees?" column of the judge inspection sample for a few products.
6. Report the numbers from `report.md` exactly as produced.
