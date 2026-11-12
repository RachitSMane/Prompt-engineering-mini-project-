# Prompt Comparison Report — `20261111-030912_qwen2.5-3b`

**Evaluation status:** Complete — evaluated against the full reference-labelled set (299 / 299 labelled reviews in this run).

**Reference labels:** 299 reference labels: 1 human, 298 AI-assisted (documented in annotations.json). AI-assisted labels are not manual annotations; scores measure agreement with these reference labels.

## Configuration

- Provider / model: `ollama` / `qwen2.5:3b` (temperature 0.0)
- Prompt version: `v1` · strategies: zero_shot, few_shot, cot
- Reviews in run: 299 · max repair attempts: 2 · strict grounding: True
- Created: 2026-11-11T03:09:12+00:00 · git commit: `045f16b`

## Overall sentiment (4 classes)

| strategy | n_gold | sentiment_accuracy | sentiment_macro_precision | sentiment_macro_recall | sentiment_macro_f1 | sentiment_weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 299 | 0.612 | 0.6611 | 0.4659 | 0.438 | 0.6797 |
| Few-shot | 299 | 0.8127 | 0.6917 | 0.5823 | 0.6025 | 0.817 |
| Chain-of-Thought | 299 | 0.5753 | 0.7201 | 0.5831 | 0.4232 | 0.6461 |

## Complaints (multi-label) and aspects

| strategy | complaint_micro_f1 | complaint_macro_f1 | complaint_hamming_loss | aspect_detection_micro_f1 | aspect_sentiment_macro_f1 | aspect_joint_micro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 0.3593 | 0.3029 | 0.0707 | 0.5221 | 0.5514 | 0.3938 |
| Few-shot | 0.4153 | 0.3599 | 0.0767 | 0.6488 | 0.6479 | 0.5579 |
| Chain-of-Thought | 0.3854 | 0.2846 | 0.0686 | 0.5028 | 0.5524 | 0.3698 |

### zero_shot: per-class sentiment metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| positive | 0.9516 | 0.5315 | 0.6821 | 111 |
| negative | 0.8676 | 0.8369 | 0.852 | 141 |
| neutral | 0.0253 | 0.4 | 0.0476 | 5 |
| mixed | 0.8 | 0.0952 | 0.1702 | 42 |

Confusion matrix (rows = reference label, columns = prediction, order ['positive', 'negative', 'neutral', 'mixed']): `[[59, 2, 44, 0], [0, 118, 16, 1], [0, 3, 2, 0], [3, 13, 17, 4]]`

### few_shot: per-class sentiment metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| positive | 0.9273 | 0.9189 | 0.9231 | 111 |
| negative | 0.8301 | 0.9007 | 0.8639 | 141 |
| neutral | 0.1429 | 0.2 | 0.1667 | 5 |
| mixed | 0.8667 | 0.3095 | 0.4561 | 42 |

Confusion matrix (rows = reference label, columns = prediction, order ['positive', 'negative', 'neutral', 'mixed']): `[[102, 3, 2, 1], [1, 127, 3, 1], [0, 4, 1, 0], [7, 19, 1, 13]]`

### cot: per-class sentiment metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| positive | 0.9444 | 0.4595 | 0.6182 | 111 |
| negative | 0.8828 | 0.8014 | 0.8401 | 141 |
| neutral | 0.0532 | 1.0 | 0.101 | 5 |
| mixed | 1.0 | 0.0714 | 0.1333 | 42 |

Confusion matrix (rows = reference label, columns = prediction, order ['positive', 'negative', 'neutral', 'mixed']): `[[51, 1, 51, 0], [0, 113, 20, 0], [0, 0, 5, 0], [3, 14, 18, 3]]`

## Reliability, latency and token usage

| strategy | coverage | failed | validity_rate | first_attempt_valid_rate | needed_llm_repair | mean_latency_s | mean_input_tokens | mean_output_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 1.0 | 17 | 0.9431 | 0.913 | 26 | 41.1856 | 2057.4 | 214.5 |
| Few-shot | 1.0 | 14 | 0.9532 | 0.9431 | 17 | 53.7686 | 2884.5 | 227.9 |
| Chain-of-Thought | 1.0 | 20 | 0.9331 | 0.9164 | 25 | 47.5273 | 2297.6 | 223.1 |

## Methodology notes

- Reference labels come from data/annotations/annotations.json (never from this run's model output); their human / AI-assisted split is stated above. The star rating is never used as ground truth.
- The flagged (unreadable) review is excluded; coverage = share of run reviews with a stored prediction.
- Failed extractions are kept: wrong sentiment, empty aspect/complaint sets.
- Aspects are matched by aspect category; aspect sentiment is scored on categories found in both gold and prediction.
- Macro averages use the labels that occur in the reference labels or the predictions (labels absent from both are excluded).
