# D1 — Structured outputs: completion report (2026-11-14)

**Verdict: complete.** Every D1 requirement is implemented, was executed on the real data with local Ollama
(`qwen2.5:3b`), and was independently verified on 2026-11-14. No D1 inference was re-run for this report.

## 1. Requirement status

| Requirement | Implementation | Executed | Independently verified |
|---|---|---|---|
| Pydantic schema: overall sentiment, aspects with category + sentiment + evidence, complaint categories | `schemas/review_schema.py`, `schemas/taxonomy.py` (13 aspect categories, closed complaint enum, written definitions) | yes | `tests/test_schemas.py` (17 tests); export re-validation below |
| LangChain structured extraction (local Ollama) | `pipeline/extraction.py`, `llm/langchain_client.py`, prompts in `prompts/` | 3 strategies × 299 reviews | yes |
| Automatic validation and repair | `pipeline/validation.py` (deterministic local fixes), `pipeline/repair.py` (bounded LLM repair, max 2) | yes | `tests/test_validation_and_repair.py` (20) |
| Controlled failure handling | an output that is still invalid after repair is stored as a failure with every raw attempt and error; never dropped, never given invented values | 51 controlled failures | failures CSV = stored failures (below) |
| Diagnostics | per-strategy validity, first-attempt validity, repairs, latency, tokens (`metrics.json` → `operational`); `validation_report.json` per export | yes | `tests/test_d1_completion.py` (8) |
| JSON and CSV export | `scripts/export_d1.py` → `outputs/live/d1/20261111-030912_qwen2.5-3b/<strategy>/{records.json, records.csv, records_aspects.csv, failures.json, failures.csv, validation_report.json}` | yes | `scripts/verify_d1_exports.py`: **27 / 27 PASS** |

## 2. Results (run `20261111-030912_qwen2.5-3b`, 299 reviews per strategy)

| Strategy | Valid | Controlled failures | Validity | First-attempt valid | Needed LLM repair | Local fixes | Mean latency |
|---|---|---|---|---|---|---|---|
| Zero-shot | 282 | 17 | 94.3 % | 91.3 % | 26 | 28 | 41.2 s |
| Few-shot | 285 | 14 | 95.3 % | 94.3 % | 17 | 110 | 53.8 s |
| Chain-of-thought | 279 | 20 | 93.3 % | 91.6 % | 25 | 25 | 47.5 s |

Typical errors before repair: invalid JSON, aspect categories outside the taxonomy, invalid complaint values, and
evidence that was not verbatim or was longer than 300 characters.

## 3. Independent export verification (re-run 2026-11-14)

`scripts/verify_d1_exports.py --run 20261111-030912_qwen2.5-3b` reads the exports and the stored predictions,
without using the export code. It runs 9 checks per strategy:

1. the JSON parses and re-validates against the schema, with evidence grounding;
2. JSON ids are unique;
3. the JSON matches the stored records;
4. the CSV has one row per record;
5. the CSV values equal the JSON;
6. the aspects CSV has one row per aspect;
7. the failures CSV matches the stored failures;
8. no review is both a record and a failure;
9. every review in the run is accounted for.

| Strategy | JSON records | Aspect rows | Failures | Reviews | Result |
|---|---|---|---|---|---|
| CoT | 279 | 352 | 20 | 299 | PASS (9 / 9) |
| Few-shot | 285 | 625 | 14 | 299 | PASS (9 / 9) |
| Zero-shot | 282 | 371 | 17 | 299 | PASS (9 / 9) |

The output of the re-run (written to a scratch file) is identical to the stored `reports/d1/export_verification.md`.

**Reproducibility.** The exports are rebuilt deterministically from the stored predictions
(`scripts/export_d1.py`), with no model call. The predictions are byte-identical to the 2026-11-14 backup
(zero-shot SHA-256 `c3ad1430…`).

## 4. Limitations

- `qwen2.5:3b` is a small model: extractions are valid but shallow. Aspect detection micro-F1 is 0.52 / 0.65 / 0.50
  (zero-shot / few-shot / CoT) against the reference labels, which are mostly AI-assisted (see the D2 report).
- Evidence grounding uses fuzzy string matching, so paraphrased evidence is rejected (counted as a failure, not
  accepted).
- A failure is a controlled outcome, not a crash. It counts as a wrong prediction in D2.

## 5. Reproduce (no LLM)

```powershell
.venv\Scripts\python.exe scripts\export_d1.py --run 20261111-030912_qwen2.5-3b
.venv\Scripts\python.exe scripts\verify_d1_exports.py --run 20261111-030912_qwen2.5-3b --out reports\d1\export_verification.md
```
