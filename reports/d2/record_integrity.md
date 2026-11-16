# D2 record integrity — run `20261111-030912_qwen2.5-3b`

Read-only, standard library only, no project code and no LLM (`scripts/verify_d2_records.py`).

Evaluable reference labels: **299** (1 human, 298 AI-assisted); excluded (flagged): R5cfb12ef75b4.

| Strategy | Rows | Unique review ids | Ids with several rows (retries) | Valid | Invalid (scored as wrong) |
|---|---|---|---|---|---|
| zero_shot | 299 | 299 | 0 | 282 | 17 |
| few_shot | 299 | 299 | 0 | 285 | 14 |
| cot | 299 | 299 | 0 | 279 | 20 |

| Check | Result | Detail |
|---|---|---|
| zero_shot: every row has a review id | PASS | 0 rows without id |
| zero_shot: no review has two successful predictions | PASS | 0 ids |
| zero_shot: evaluated ids == evaluable reference ids | PASS | missing 0, extra 0 |
| few_shot: every row has a review id | PASS | 0 rows without id |
| few_shot: no review has two successful predictions | PASS | 0 ids |
| few_shot: evaluated ids == evaluable reference ids | PASS | missing 0, extra 0 |
| cot: every row has a review id | PASS | 0 rows without id |
| cot: no review has two successful predictions | PASS | 0 ids |
| cot: evaluated ids == evaluable reference ids | PASS | missing 0, extra 0 |
| all three strategies use the identical review set | PASS | 299 ids |
| reference provenance recorded (human vs AI-assisted) | PASS | 1 human, 298 AI-assisted |

**ALL CHECKS PASS.** The reference labels are mostly AI-assisted, so these metrics measure agreement with AI-assisted labels, not with human ground truth.
