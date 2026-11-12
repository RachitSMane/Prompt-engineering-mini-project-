# Verification of curated results

Metrics recomputed from the saved prediction files (no LLM calls) and compared with the stored `metrics.json`.

## `20261111-030912_qwen2.5-3b`

- status: **complete** · reference labels in run: 299 / 299
- missing predictions: {'zero_shot': 0, 'few_shot': 0, 'cot': 0}
- recomputation matches stored metrics: **yes**

- D1 export copied from `outputs\live\d1\20261111-030912_qwen2.5-3b`
- D3 export copied from `outputs\live\d3\dataset__few_shot__qwen2.5-3b__v1`
- Judge negative control copied from `outputs\live\d3\judge_sanity\dataset__few_shot__qwen2.5-3b__v1`

- All 7 JSON and 7 CSV files under `reports/` parse successfully.
