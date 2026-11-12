# Independent metric verification — `20261111-030912_qwen2.5-3b`

Recomputed with the Python standard library only (no project code, no scikit-learn) from the raw prediction files and `data/annotations/annotations.json`, then compared with the stored `metrics.json` (tolerance 0.0001).

- Reviews in run: 299 · complete reference labels in run: 299

- Reference class counts (all four classes checked): {'positive': 111, 'negative': 141, 'neutral': 5, 'mixed': 42}

| strategy | ids accounted | extra ids | duplicate rows | valid | failed (scored wrong) | missing | coverage | accuracy | macro P | macro R | macro F1 | matches metrics.json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_shot | 299/299 | 0 | 0 | 282 | 17 | 0 | 1.0000 | 0.6120 | 0.6611 | 0.4659 | 0.4380 | yes |
| few_shot | 299/299 | 0 | 0 | 285 | 14 | 0 | 1.0000 | 0.8127 | 0.6917 | 0.5823 | 0.6025 | yes |
| cot | 299/299 | 0 | 0 | 279 | 20 | 0 | 1.0000 | 0.5753 | 0.7201 | 0.5831 | 0.4232 | yes |

### zero_shot

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| positive | 0.9516 | 0.5315 | 0.6821 | 111 |
| negative | 0.8676 | 0.8369 | 0.8520 | 141 |
| neutral | 0.0253 | 0.4000 | 0.0476 | 5 |
| mixed | 0.8000 | 0.0952 | 0.1702 | 42 |

Confusion matrix (rows = reference, columns = prediction, order ['positive', 'negative', 'neutral', 'mixed']; failed predictions appear in no column): `[[59, 2, 44, 0], [0, 118, 16, 1], [0, 3, 2, 0], [3, 13, 17, 4]]`

### few_shot

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| positive | 0.9273 | 0.9189 | 0.9231 | 111 |
| negative | 0.8301 | 0.9007 | 0.8639 | 141 |
| neutral | 0.1429 | 0.2000 | 0.1667 | 5 |
| mixed | 0.8667 | 0.3095 | 0.4561 | 42 |

Confusion matrix (rows = reference, columns = prediction, order ['positive', 'negative', 'neutral', 'mixed']; failed predictions appear in no column): `[[102, 3, 2, 1], [1, 127, 3, 1], [0, 4, 1, 0], [7, 19, 1, 13]]`

### cot

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| positive | 0.9444 | 0.4595 | 0.6182 | 111 |
| negative | 0.8828 | 0.8014 | 0.8401 | 141 |
| neutral | 0.0532 | 1.0000 | 0.1010 | 5 |
| mixed | 1.0000 | 0.0714 | 0.1333 | 42 |

Confusion matrix (rows = reference, columns = prediction, order ['positive', 'negative', 'neutral', 'mixed']; failed predictions appear in no column): `[[51, 1, 51, 0], [0, 113, 20, 0], [0, 0, 5, 0], [3, 14, 18, 3]]`

**Result:** all recomputed values match the stored metrics (per-class P/R/F1 for all four classes, confusion matrices, macro P/R/F1, accuracy, invalid predictions, coverage, missing predictions; no duplicate or out-of-run ids).
