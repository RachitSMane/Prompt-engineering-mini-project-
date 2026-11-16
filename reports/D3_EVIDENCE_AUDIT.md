# D3 — Summary evidence audit (deterministic, no LLM)

Source: `dataset__few_shot__qwen2.5-3b__v1`. That's 15 stored product summaries (`outputs/live/summaries/summaries.jsonl`)
built from 285 valid extraction records. Code: `evaluation/d3_audit.py::audit_summary`; runner: `scripts/audit_d3.py`.
Output: `outputs/live/d3/audit/dataset__few_shot__qwen2.5-3b__v1/evidence_audit.json`. First run 2026-11-13; re-run on 2026-11-14 with an added count-statement check (the 2026-11-13 output is kept as `evidence_audit_2026-11-13.json`).

No summary was rewritten. The originals are the evaluated artefacts.

## Checks

| Check | Method |
|---|---|
| Cited review ids exist | every `supporting_review_ids` entry must be a record in the source extraction file |
| Cited reviews belong to the product | the record's `product_id` must equal the summary's product |
| Statistics match the aggregation source | re-aggregate the product's records with `pipeline.aggregation.aggregate_product` and compare total, sentiment counts and %, complaint counts, numeric aspect statistics, and review ids. `top_aspect_names` is compared tie-aware: among names with equal counts the order follows record order, which differs between the cache at summary time and now |
| Percentages are real | every `NN%` in the summary text must equal a computed sentiment or complaint percentage (±0.5); otherwise it is traced manually |
| Count statements are real (added 2026-11-14) | every "N out of M reviews" / "N of M reviews" in the text: M must be the product's review total and N a real count. A complaint count over the number of *complaining* reviews is reported as a misleading denominator |
| Claims are traceable | each appreciated or complaint point is checked against its cited reviews. **Direction:** appreciated → a positive aspect or positive/mixed review; complaint → a negative aspect or negative/mixed review. **Content-word overlap** with the cited review texts, evidence quotes and aspect categories ≥ 0.3. Both pass = `traceable`, one passes = `partial`, neither = `weak` |
| Frequent aspects | every `frequent_aspects` entry must match an aspect category or name in the statistics |

## Results

| | |
|---|---|
| Summaries | **15** |
| Cited review ids | 130; **0 unknown, 0 from another product** |
| Statistics identical to a fresh re-aggregation | **15 / 15** |
| Percentages in summary text | 10, of which 7 match a statistic, 1 is quoted from a cited review (recognised automatically), and **2 are unsupported** |
| Count statements in summary text | 10, of which 3 correct and **7 misleading** (all in B001DHKBXU) |
| Summaries with valid ids + matching statistics + only sourced percentages + correct counts | **13 / 15** (B0BTY63SFY and B001DHKBXU fail) |
| Summary points (appreciated + complaints) | 83: **41 traceable, 41 partial, 1 weak** |
| Frequent aspects not in statistics | 0 |

### Defects found (reported, not silently fixed)

1. **B0BTY63SFY (77L Deviled Egg Containers): misattributed statistics.** The overall opinion says "positive sentiments
   from 80% of the reviews" and "67% of negative reviews mentioning these aspects". Neither is a stated sentiment or
   complaint statistic. Traced by hand on 2026-11-14 (the 2026-11-13 report called them "fabricated", which was too
   strong): 80 % equals the positive share of *performance* mentions in the aspect statistics (8 of 10), but the share of
   positive *reviews* is 86.7 % (13 of 15). 67 % equals 2 of the 3 reviews with complaints, but **both** negative reviews
   (R80f511ff60c4, Rd1d0ae81cb05; 2 of 2) mention compatibility or performance. As written, both statements are wrong.
   - This is a genuine hallucination by the summariser (`qwen2.5:3b`).
   - Neither judge flagged it. The v2 judge's only "unsupported claim" for this summary was a sentence about stacking
     eggs, and its consistency justification quotes the correct 86.7% without noticing the 80%.
2. **B07XG8QQL5 (Samsung heating element): one weakly traceable complaint point.** "Performance issues are the most
   common complaint, with users experiencing heating coil failure and sensor malfunctions."
   - Only 1 of the 3 cited reviews is negative or mixed, and the content-word overlap is 0.18. The "sensor malfunctions"
     part is not supported by the cited reviews.
3. **B001DHKBXU (Whirlpool lid switch): misleading denominators (found 2026-11-14).** The overall opinion correctly says
   "15 out of 19 reviews containing complaints", then lists every complaint as "N out of 15 reviews" (poor durability
   8, installation difficulty 7, poor value 6, product defect 5, customer service 4, performance 4, compatibility 1).
   The counts are right, but 15 is the number of complaining reviews; the product has **19** reviews, so "8 out of 15
   reviews" overstates the share (42 %, not 53 %). The percentage-only check had passed this summary. Neither judge
   flagged it (v1 failed on this product; v2 instead flagged "did not provide good value for money", which the data
   support).
4. **B0759KR8JX: not a defect.** "40% cheaper than the brand name filter" is not a computed statistic, but it is quoted
   from the cited review R351847be0a2c ("40% less expensive than the brand name filter").

### Separately labelled corrections (2026-11-14; originals unchanged)

The two failing summaries have **corrected copies** written by `scripts/build_corrected_summaries.py` (no LLM) to
`outputs/live/d3/corrected_summaries/dataset__few_shot__qwen2.5-3b__v1/corrected_summaries.jsonl`.

- **Provenance:** a deterministic text correction by the AI coding assistant (Claude Code), with every new number
  taken from the product's own aggregate and extraction records. The corrections were not produced by the summariser
  and were not written or approved by a human.
- **Changes:** exact replacements only, each matching exactly once.
  - B001DHKBXU: 7 × "out of 15 reviews" → "out of 19 reviews".
  - B0BTY63SFY: "with positive sentiments from 80% of the reviews." → "and 86.7% of the reviews were positive
    overall."; and "with 67% of negative reviews mentioning these aspects." → "with both negative reviews (2 of 2)
    mentioning these aspects."
- **Checks run (2026-11-14):**
  - Replaying the replacements on the original reproduces the corrected text exactly. The cited ids, the other summary
    fields and the aggregate are unchanged.
  - The numbers recompute from the extraction records: complaint counts over 19 reviews; 13 of 15 positive; 2 of 2
    negative reviews mention compatibility or performance.
  - Both copies pass `audit_summary` with 0 issues.
  - `summaries.jsonl` is byte-identical to the backup.
- **Judging:** every stored v1/v2 judgement and v2 negative-control score refers to the **original** summary. The
  corrected copies were judged only by the separate judge v3 run (2026-11-14), under their own ids
  (`D3_JUDGE_EVALUATION.md` §6); no score is transferred between an original and its copy. The evaluated D3 results, including "13 / 15 verified", are those of the
  originals.
- **Not corrected:** the weak point in B07XG8QQL5 ("sensor malfunctions"). Deciding whether to drop it needs a human
  reading of the cited reviews (`scripts/build_manual_inspection.py`).

### Meaning of "partial"

41 points are `partial`. In most of them the cited reviews point the right way, but the point paraphrases (low word
overlap), or one cited review is mixed in tone. These are not proven defects, but they are not mechanically verified
either. `scripts/build_manual_inspection.py` writes all 83 points next to the full text of the reviews they cite, for
human checking. **No human inspection has been done yet.**

## Per product

| Product | Records | Cited ids | Traceable / partial / weak | Unsupported % | Ids + statistics verified |
|---|---|---|---|---|---|
| B001DHKBXU | 19 | 10 | 4 / 2 / 0 | – (**7 misleading counts**) | **no: misleading denominators** |
| B001ONHE8S | 16 | 10 | 4 / 2 / 0 | – | yes |
| B001TH7GZU | 28 | 9 | 2 / 3 / 0 | – | yes |
| B0081N3YW4 | 42 | 9 | 0 / 4 / 0 | – | yes |
| B008YDTFYW | 11 | 12 | 1 / 8 / 0 | – | yes |
| B00B0ZMNOU | 24 | 10 | 7 / 2 / 0 | – | yes |
| B00DM8K1NS | 13 | 7 | 1 / 4 / 0 | – | yes |
| B00LHRAGHW | 10 | 8 | 1 / 5 / 0 | – | yes |
| B01FZJXITU | 9 | 6 | 1 / 3 / 0 | – | yes |
| B01LZHJF55 | 24 | 7 | 4 / 0 / 0 | – | yes |
| B0759KR8JX | 14 | 8 | 6 / 0 / 0 | – (40 % is quoted from cited review R351847be0a2c) | yes |
| B07XG8QQL5 | 19 | 7 | 2 / 0 / **1** | – | yes (1 weak point) |
| B09WWM1WLS | 19 | 10 | 5 / 0 / 0 | – | yes |
| B0BTY63SFY | 15 | 10 | 3 / 3 / 0 | **80, 67** (misattributed) | **no: misattributed statistics** |
| B0C5G1WGBN | 22 | 7 | 0 / 5 / 0 | – | yes |

## Judge-side evidence problems (details in `D3_JUDGE_EVALUATION.md`)

- **Placeholder evidence ids:** the v2 judgement of B09WWM1WLS cites `review_id_1` and `review_id_2`, copied from the
  prompt skeleton; they exist in no product. The same placeholders appear in the negative-control judgement of
  B0081N3YW4 (fabricated statistic). All other judge evidence ids are valid.
- **Claims not in the summary:** 6 of the 26 claims flagged by judge v2 are not found in the summary text.
- **Repair pressure:** 7 of 15 v2 judgements (and 13 of 30 negative-control judgements) were accepted only after the
  validator rejected a first answer that listed no unsupported claims.
- **Missed defects:** neither judge flagged B0BTY63SFY's misattributed 80 % / 67 % (real ratios, wrong denominator) or B001DHKBXU's misleading
  denominators.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\audit_d3.py --source dataset__few_shot__qwen2.5-3b__v1
.venv\Scripts\python.exe scripts\build_manual_inspection.py --source dataset__few_shot__qwen2.5-3b__v1
```
