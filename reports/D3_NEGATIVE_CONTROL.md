# D3 — Negative control for the LLM judge (v2; v3 re-check in §7)

**Status: executed on all 15 products — 30 corrupted summaries, 0 judge failures.**
- 2026-11-13: pilot on 3 products, then continued to 7 products until the watchdog fail-safe stopped it (`nvidia-smi`
  unreadable 3×).
- 2026-11-14, 09:12–10:28 IST: resumed and completed with the GPU watchdog active and its thresholds unchanged.
- Results re-verified independently from `pairs.jsonl` on 2026-11-14.
- **n = 15 products: indicative, not statistically reliable.**

Judge: `llama3.2` via local Ollama, judge v2 (flat schema, constrained decoding). Controls: the stored v2 judgements
of the 15 original summaries (all successful).

## 1. Method

Code: `evaluation/negative_control.py`; runner: `scripts/judge_negative_control.py`.

- **Control:** the original summary with its stored v2 judgement (15 distinct judgements).
- **Corrupted copies:** two per product, each differing from the original by **exactly one planted defect**:
  1. `invented_feature` — an appreciated point praising a feature from a pool of six (Wi-Fi/voice control, solar
     power, UV sterilisation, touchscreen with recipes, lifetime guarantee, smartphone energy app). Before use, the
     code checks that **none of the feature's keywords occurs in any review of that product**; the independent
     re-check confirmed this for all 15. The point cites a real review id, so only a judge that reads the sources can
     catch it.
  2. `fabricated_statistic` — "In total, NN% of reviewers say they would buy this exact product again." appended to
     the unchanged overall opinion. NN is more than 3 points from every real sentiment or complaint percentage of the
     product (re-checked).
- **Detection (current rule, 2026-11-14):** one of the judge's `unsupported_claims` contains a planted keyword.
  The claim's `reason` is **not** counted (see §3).
- **False negative:** the corrupted summary's judgement does not name the planted claim.
- **Keyword control:** the original summary's judgement already names the planted keyword, which would make detection
  meaningless. This check is narrow; the broader false-positive picture is in §4.
- Corrupted summaries and their judgements exist only in
  `outputs/live/d3/judge_negative_control_v2/dataset__few_shot__qwen2.5-3b__v1/pairs.jsonl`. They never enter the
  summary or judgement stores shown in the app.

## 2. Results (n = 15 products, 30 corrupted summaries)

| Defect | Judged | Named (detected) | Missed | Detection rate | 95 % Wilson CI | False-negative rate | Faithfulness lower | Hallucination lower / higher | Mean lower |
|---|---|---|---|---|---|---|---|---|---|
| Invented feature | 15 | **10** | **5** | **66.7 %** | 42–85 % | 33.3 % | 1 / 15 | 2 / 1 | 6 / 15 |
| Fabricated statistic | 15 | **8** | **7** | **53.3 %** | 30–75 % | 46.7 % | 1 / 15 | 0 / 1 | 4 / 15 |
| **All planted claims** | **30** | **18** | **12** | **60.0 %** | 42–75 % | 40.0 % | 2 / 30 | 2 / 2 | 10 / 30 |

- **Keyword control:** 0 of 30 checks (15 originals × 2 keywords). No original judgement mentioned a planted keyword
  before it was planted. **This is not a false-positive rate for the judge.** It only shows the planted keywords weren't
  already present. On the same 15 original summaries, the judge listed at least one "unsupported" claim **15 / 15**
  times (26 claims; 17 match categories in the product's own data). Its general false-positive behaviour is therefore
  poor (§4).
- **Mean score:** lower for 10 corrupted summaries, unchanged for 16 and **higher for 4**.
- **Missed (12):**
  - Invented feature (5): B001DHKBXU (Wi-Fi), B001ONHE8S (solar), B0081N3YW4 (touchscreen), B008YDTFYW (lifetime
    guarantee), B00LHRAGHW (solar).
  - Fabricated statistic (7): B001TH7GZU (88 %), B0081N3YW4 (91 %), B01FZJXITU (91 %), B0759KR8JX (97 %),
    B07XG8QQL5 (93 %), B0BTY63SFY (91 %), B0C5G1WGBN (86 %).

| Product | Planted defect | Named? | Faithfulness | Hallucination | Mean | Repairs |
|---|---|---|---|---|---|---|
| B001DHKBXU | invented feature (wi-fi) | **no** | 3→3 | 4→4 | 3.25→3.25 | 0 |
| B001DHKBXU | fabricated statistic (97%) | yes | 3→3 | 4→4 | 3.25→3.25 | 2 |
| B001ONHE8S | invented feature (solar) | **no** | 5→3 | 4→5 | 4.75→3.00 | 1 |
| B001ONHE8S | fabricated statistic (93%) | yes | 5→4 | 4→4 | 4.75→3.50 | 2 |
| B001TH7GZU | invented feature (uv light) | yes | 3→3 | 4→4 | 3.25→3.25 | 0 |
| B001TH7GZU | fabricated statistic (88%) | **no** | 3→3 | 4→4 | 3.25→3.25 | 2 |
| B0081N3YW4 | invented feature (touchscreen) | **no** | 3→3 | 4→4 | 3.50→3.25 | 1 |
| B0081N3YW4 | fabricated statistic (91%) | **no** (see §3) | 3→3 | 4→4 | 3.50→3.50 | 1 |
| B008YDTFYW | invented feature (lifetime) | **no** | 3→3 | 4→4 | 3.75→3.75 | 0 |
| B008YDTFYW | fabricated statistic (86%) | yes | 3→3 | 4→4 | 3.75→3.75 | 0 |
| B00B0ZMNOU | invented feature (smartphone) | yes | 3→3 | 4→4 | 3.25→2.75 | 0 |
| B00B0ZMNOU | fabricated statistic (97%) | yes | 3→3 | 4→4 | 3.25→2.75 | 0 |
| B00DM8K1NS | invented feature (wi-fi) | yes | 3→3 | 4→4 | 3.75→3.25 | 0 |
| B00DM8K1NS | fabricated statistic (93%) | yes | 3→3 | 4→4 | 3.75→3.25 | 0 |
| B00LHRAGHW | invented feature (solar) | **no** | 3→3 | 4→4 | 3.25→3.25 | 1 |
| B00LHRAGHW | fabricated statistic (88%) | yes | 3→3 | 4→4 | 3.25→3.25 | 0 |
| B01FZJXITU | invented feature (uv light) | yes | 4→4 | 4→2 | 3.50→3.00 | 0 |
| B01FZJXITU | fabricated statistic (91%) | **no** | 4→4 | 4→5 | 3.50→3.50 | 1 |
| B01LZHJF55 | invented feature (touchscreen) | yes | 3→3 | 4→4 | 3.50→3.50 | 0 |
| B01LZHJF55 | fabricated statistic (86%) | yes | 3→3 | 4→4 | 3.50→3.25 | 0 |
| B0759KR8JX | invented feature (lifetime) | yes | 4→4 | 4→2 | 3.75→3.25 | 0 |
| B0759KR8JX | fabricated statistic (97%) | **no** | 4→4 | 4→4 | 3.75→3.75 | 1 |
| B07XG8QQL5 | invented feature (smartphone) | yes | 2→2 | 4→4 | 2.75→2.75 | 0 |
| B07XG8QQL5 | fabricated statistic (93%) | **no** | 2→3 | 4→4 | 2.75→3.25 | 0 |
| B09WWM1WLS | invented feature (wi-fi) | yes | 3→3 | 4→4 | 3.00→3.25 | 0 |
| B09WWM1WLS | fabricated statistic (88%) | yes | 3→3 | 4→4 | 3.00→3.25 | 0 |
| B0BTY63SFY | invented feature (solar) | yes | 4→4 | 4→4 | 4.00→4.00 | 2 |
| B0BTY63SFY | fabricated statistic (91%) | **no** | 4→4 | 4→4 | 4.00→4.00 | 2 |
| B0C5G1WGBN | invented feature (uv light) | yes | 3→3 | 4→4 | 3.25→3.25 | 1 |
| B0C5G1WGBN | fabricated statistic (86%) | **no** | 3→3 | 4→4 | 3.25→3.50 | 2 |

## 3. Correction made during verification (no re-run)

The earlier rule counted a detection when the keyword appeared in a claim **or in its reason**. I inspected all 19
pairs it called "detected" by hand and found one wrong case: **B0081N3YW4, fabricated statistic**.
- The judge's only claim was "The product received a 'mixed reception' with a majority of negative reviews."
- Its reason was "The actual statistic is 91 % of reviewers saying they would buy the product again, which is a
  positive sentiment."
- So the judge **accepted the fabricated 91 % as fact** and used it to reject a true statement — the opposite of
  detection.

What changed:
- Detection now requires the keyword in a claim (`planted_claim_detected`). A mention only in a reason is reported
  separately (`keyword_in_reason_only`: 1 case).
- The figures are recomputed from the stored judgements; `pairs.jsonl` is byte-identical. The earlier summary is kept
  as `summary_previous_rule_claims_and_reasons.json` (19 / 30 under the old rule).
- Regression tests: `test_planted_keyword_only_in_reason_is_not_a_detection` and
  `test_negative_control_summary_recomputes_detection_without_editing_rows`.
- The other 18 detections each name the planted sentence. One (B008YDTFYW) does so as the paraphrase "Users would buy
  this exact product again".

## 3b. Keyword detection vs semantic review (separate measures; never combined)

The **official detection figure is the keyword rule: 18 / 30 (60 %)**. A keyword match shows only that the judge
*listed* the planted sentence as unsupported. It does not show that the judge checked the evidence: the matching
reasons are generic ("Not supported by the reviews"), and some verdicts list many claims, including true ones
(B001DHKBXU's fabricated-statistic verdict lists 7). Keywords can also miss paraphrases.

So every stored judgement was also read in full: `scripts/nc_semantic_review.py` →
`semantic_review.json`, with all 30 pairs included exactly once.
- **Assessment type:** this is a **model-assisted assessment** by Claude Code (the AI assistant), quoting the stored
  text for every pair. **It is not independent human ground truth** and does not replace the keyword rate.
- **Automatic part:** the 18 keyword matches are re-checked by the script, and the matching claim and reason recorded.
- **Manual part:** the 12 keyword misses were classified by hand.

| Category (separate) | Count | Pairs |
|---|---|---|
| Named in claims, keyword match (automatic) | 18 | as in §2 |
| Named in claims, paraphrase without keyword | 1 | B07XG8QQL5 statistic ("Users would buy this product again.") |
| Mentioned as unsupported only in a justification; claim list empty, hallucination 5 (self-contradictory verdict) | 2 | B001ONHE8S feature ("solar power feature"), B01FZJXITU statistic ("users would buy the product again, which is not supported") |
| Planted claim used as fact (keyword only in a *reason*) | 1 | B0081N3YW4 statistic |
| Not mentioned anywhere | 8 | B001DHKBXU, B0081N3YW4, B008YDTFYW, B00LHRAGHW (features); B001TH7GZU, B0759KR8JX, B0BTY63SFY, B0C5G1WGBN (statistics) |

The rule-based rate stays 18 / 30. The semantic reading is reported only as context: the judge listed the planted
claim in 19 / 30 cases if a paraphrase is accepted, and mentioned it as unsupported somewhere in 21 / 30. In 2 of
those 21 the verdict contradicts itself (it says "unsupported" but lists no claim and scores hallucination 5). In
9 / 30 cases the planted claim went unnoticed (8) or was accepted as true (1).

## 4. Interpretation

- **The judge's claim list caught a planted error 60 % of the time and missed 40 %** (12 of 30). Invented features
  (10/15) were caught somewhat more often than fabricated percentages (8/15). That is the reverse of the 7-product
  interim result (3/7 vs 6/7), which shows how unstable rates from small samples are.
- **The scores barely respond.** Faithfulness fell in 2 of 30 corrupted summaries and hallucination in 2 of 30.
  - In 2 cases, hallucination *rose* to 5 ("no unsupported claims") while the planted claim went unnoticed
    (B001ONHE8S solar, B01FZJXITU 91 %).
  - Four corrupted summaries got a *higher* mean score than their originals.
- **Controls:** the narrow keyword control is clean (0/30). But every one of the 15 original judgements lists at
  least one "unsupported" claim, and 17 of those 26 claims match categories present in the product's own data
  (`D3_JUDGE_EVALUATION.md` §4). A summary-level "flags anything" detector would therefore have a 100 % false-positive
  rate on the originals; only claim-level matching makes detection measurable at all.
- **Validator pressure:** 13 of the 30 corrupted-summary judgements were repaired. Every first rejection was
  "hallucination below 5 but no claims" (`D3_JUDGE_EVALUATION.md` finding 5).
- **Real-data false negative:** neither judge flagged the genuinely misattributed 80 % / 67 % (real ratios, wrong denominator) in B0BTY63SFY's original
  summary, and the planted 91 % in the same summary was missed too.
- **Conclusion:** the negative control **does not support** using this judge as a reliable faithfulness measure. Its
  claim list is a useful *pointer* for a human reviewer; its numeric scores are not.

## 5. Limitations

- n = 15 products with 2 planted defects each, and a single judge model (3B, temperature 0, partial CPU offload). The
  confidence intervals are wide.
- Keyword matching can be loose (a claim containing the keyword for another reason would count) or strict (a
  paraphrase without the keyword would not). All 30 judgements are in `pairs.jsonl`, and the 18 detections were
  checked by hand.
- Only two defect types. Subtler corruptions (inverted sentiment, a dropped complaint, wrong counts) were not tested.
- The fabricated-statistic sentence uses one template, so the 15 cases are not fully independent.

## 6. Reproduce

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
.venv\Scripts\python.exe scripts\judge_negative_control.py --source dataset__few_shot__qwen2.5-3b__v1 --report-only   # no LLM; rewrites summary.json from pairs.jsonl
# full run (LLM, resumable; with all 30 pairs stored it makes 0 new calls)
$env:LLM_MAX_OUTPUT_TOKENS="2048"
.venv\Scripts\python.exe scripts\judge_negative_control.py --source dataset__few_shot__qwen2.5-3b__v1
Remove-Item Env:LLM_MAX_OUTPUT_TOKENS
```

## 7. The same 30 corrupted copies judged by judge v3 (2026-11-14, separate run)

Judge v3 makes one call per summary, with no repair and the raw response kept (see `D3_JUDGE_EVALUATION.md` §6).
The corrupted copies were rebuilt and checked to be **identical** to the stored v2 pairs; `pairs.jsonl` was only
read. Same claim-text keyword rule. Results: `outputs/live/d3/judge_v3_report/<source>/report.json`.

| | v2 (historical result, unchanged) | v3 (separate run) |
|---|---|---|
| Planted claim named in the claim list | **18 / 30** (60 %, CI 42–75 %) | **5 / 30** (16.7 %, CI 7–34 %) |
| Missed | 12 / 30 | 25 / 30 |
| Invented feature / fabricated statistic | 10 / 15 · 8 / 15 | 4 / 15 · 1 / 15 |
| Faithfulness lower than its original | 2 / 30 | 3 / 30 |
| Hallucination comparison | lower 2 / 30, higher 2 / 30 | not possible: every pair has a contradictory verdict |
| Validation failures / repairs | 0 / 13 of 30 repaired | 0 / none (no repair in v3) |

- **Planted keyword in the justification text:** in all 25 v3 misses, the keyword appears in the judge's
  justification text. The judge often calls the planted sentence unsupported in prose but leaves the claim list
  empty. This count does not tell rejection apart from acceptance, so it is **not** counted as a detection.
- **Not a semantic review:** no model-assisted semantic review of the v3 judgements was made. The v2 semantic review
  (§3b; 18 / 1 / 2 / 1 / 8) applies to v2 only.
- **Never pooled:** the v2 and v3 rates are reported side by side and never combined.

**Interpretation:** v2's higher rate is partly an artefact of validator pressure, which forced the judge to fill
the claim list. Without that pressure (v3), the structured verdict misses most planted errors. Neither run supports
using this 3B judge as a faithfulness measure. Limitations as in §5: n = 15 products, a 3B local judge, keyword
detection.
