# D3 investigation and completion plan

**Date:** 2026-11-15 · **Type:** read-only investigation (no source, prompt, summary, judgement, dataset or report was
changed; nothing was regenerated; no LLM was called) · **Scope:** D3 only — extract → aggregate → summarise →
LLM-as-a-Judge faithfulness evaluation, negative controls. D1/D2 untouched.

Every number below was recomputed on 2026-11-15 directly from the stored artifacts with throw-away scripts (not the
project's own report code). Where a finding comes from my own reading of reviews rather than from a computation, it is
marked **[investigator reading — needs human confirmation]**.

---

## 0. Executive summary

| Area | Verdict |
|---|---|
| Pipeline (extract → aggregate → summarise) | **Implemented and reproducible.** 15 summaries, 130/130 citations valid, statistics re-aggregate identically. |
| Summary faithfulness | **Weaker than reported.** The existing audit (13/15 pass) checks ids, numbers and word overlap, not meaning. Reading the summaries against the statistics and the cited reviews finds factual errors in at least **6 of 15** summaries, including claims whose cited review says the opposite. |
| Statistics feeding the summaries | Faithful to the *extractions*, not necessarily to the *reviews*: extraction complaint precision is 0.37 (D2, few-shot). In B001DHKBXU about 2 of 7 `installation_difficulty` and 1 of 6 `poor_value` labels match the review text. |
| LLM judge v1/v2/v3 | **Not a reliable faithfulness measure.** v3: 37/47 verdicts contradict their own claim list; corrupted copies got the **same or a better** raw hallucination score than their originals in **30/30** pairs; claim lists on originals are mostly false positives. |
| Negative control | Planted-error detection 5/30 (v3, claim list). But the judge *named* 24 of the 25 "missed" planted errors as unsupported in its prose — the failure is structural (empty claim list), not only perceptual. The control set itself is too easy/narrow (one of its two error types is caught 15/15 by a regex). |
| Can D3 be declared complete today? | **No.** The pipeline deliverable exists; the *evaluation* deliverable (a faithfulness assessment that demonstrably detects errors) does not. |

**Root cause of the contradictions:** the judge is a 3B model asked to do, in one generation over 2–5k tokens of
context, claim extraction, evidence retrieval across 9–42 reviews, verification, and four holistic scores — and the
scores are generated independently of the claim list. Under JSON-schema-constrained decoding it emits
`"unsupported_claims": []` and then describes unsupported claims in prose.

**Recommended fix (first):** replace the holistic judge with a **claim-level verifier** (judge v4): claims are taken
deterministically from the structured summary, numeric/statistical claims are verified deterministically, the LLM only
labels one claim at a time against the full text of that claim's cited reviews, and every score is **derived** from the
claim labels — so a score/claim contradiction is impossible by construction. Validate it against a small
**human-labelled claim gold set** and an **extended, pre-registered negative control** with clean controls for
specificity.

---

## 1. D3 requirement checklist

Sources: `docs/ReviewIQ_Problem_Statement_and_Solution.pdf` §4.4–4.6 (the only formal requirement document in the
repo; there is no separate PRD/SRS file), README §1/§15/§16, and the 7 requirements in the 2026-11-15 brief.

| # | Requirement | Status | Evidence / gap |
|---|---|---|---|
| R1 | Extract relevant information from reviews | **Done** (reused from D2) | 299-review sample → 285 valid records, 14 controlled failures (`outputs/live/extractions/dataset__few_shot__qwen2.5-3b__v1.jsonl`). Quality limited: complaint micro-F1 0.415, micro-precision 0.367 (D2 `metrics.json`, few_shot). |
| R2 | Aggregate at product level (PDF 4.4: distributions, frequent aspects, ± balance, recurring complaints, representative quotes) | **Done** | `pipeline/aggregation.py`; pandas only; 15/15 aggregates re-computed identically by the audit. |
| R3 | Concise product summaries; every praise/complaint cites valid review ids; no invented numbers (PDF 4.5) | **Partly** | Citation validity enforced by schema (130/130 valid). "No invented numbers" violated in 2 summaries (B0BTY63SFY 80 %/67 %) and count/denominator errors in 2 more (B001DHKBXU, B00DM8K1NS). Citation *polarity* not enforced (cited review contradicts point in ≥ 3 points). |
| R4 | Evaluate with an LLM-as-a-Judge on an anchored 1–5 rubric (faithfulness, coverage, consistency, unsupported claims) (PDF 4.6) | **Implemented, not reliable** | v1 5/15 valid; v2 15/15 valid but H = 4 for all; v3 47/47 valid but 37/47 self-contradictory and no discrimination (§5–7). |
| R5 | Assess factual faithfulness against original review evidence | **Not achieved** | No component checks individual claims against their cited review text semantically. Deterministic audit = ids + numbers + word overlap. Judge = holistic. |
| R6 | Demonstrate evaluation with deliberately corrupted summaries / negative controls | **Done, but weak design and failing result** | 30 planted copies (2 types × 15). v2 18/30, v3 5/30 detected. No clean controls → no specificity; one error type trivially regex-detectable. |
| R7 | Reproducible results and documentation | **Partly** | Stores are append-only, versioned, raw responses kept (v3), prompt hash stored; one repeat (v3 pilot vs final, B001DHKBXU) is byte-identical. But D3 exports (`d3_report.md`, `judgements.csv`) still reflect **v1 only**; `corrected_summaries/README.json` says the corrected copies were "never judged" although v3 judged them; no repeat-run study. |
| PDF 4.6 | Deterministic checks for invented figures and invalid citations | **Done (narrow)** | Percentage check ±0.5; count regex "N (out) of M reviews"; id checks. Misses "3 positive reviews out of 13", "slight majority of positive", "majority of users", misattributed shares. |

---

## 2. Architecture and data flow

```
data/processed/reviews.csv (1,500 reviews, 15 products)
  │  annotation sample: 299 reviews (300 minus 1 flagged)          scripts/run_chain.py --sample --seed-from-run <D2 run>
  ▼
[1] EXTRACT   qwen2.5:3b few-shot, Pydantic ReviewRecord, deterministic fixes + ≤2 LLM repairs
              pipeline/extraction.py, pipeline/repair.py, pipeline/validation.py, prompts/few_shot.py
              → outputs/live/extractions/dataset__few_shot__qwen2.5-3b__v1.jsonl   (285 ok / 14 failed; reused from D2)
  ▼
[2] AGGREGATE pandas only: sentiment counts/%, avg rating, per-aspect ± counts and net sentiment, complaint counts/%
              (denominator = all reviews), reviews_with_complaints, yearly trend, ≤5 positive + ≤5 negative evidence
              quotes (≤2 per category, highest extraction confidence)            pipeline/aggregation.py
              → embedded in each summary as `aggregate`; exported to outputs/live/d3/<source>/aggregates.csv, aspect_stats.csv, evidence.csv
  ▼
[3] SUMMARISE qwen2.5:3b, prompt v1 (prompts/summarization.py): input = STATISTICS json + EVIDENCE quotes only
              (not the full reviews); output ProductSummaryContent {overall_opinion, appreciated[≤6], complaints[≤6],
              frequent_aspects, improvement_areas, recommendations}; each point's supporting_review_ids validated ⊆
              evidence ids (schemas/evaluation_schema.py SummaryPoint)          pipeline/summarization.py
              → outputs/live/summaries/summaries.jsonl (15 rows, 15 products, 14×0 repairs, 1×1 repair)
  ▼
[4] DETERMINISTIC CHECKS (no LLM)
     a. deterministic_checks()   % in text vs stats, top-aspect/complaint keyword coverage, ids ⊆ product   evaluation/summary_evaluation.py
     b. evidence audit           ids, re-aggregation, %, "N of M reviews", point traceability (sentiment direction +
                                 content-word overlap ≥ 0.3), frequent aspects    evaluation/d3_audit.py, scripts/audit_d3.py
                                 → outputs/live/d3/audit/<source>/evidence_audit.json
     c. citation-polarity check  scripts/export_d3.py → judgements.csv
  ▼
[5] JUDGE (llama3.2, separate family; full product reviews truncated to 80 words + per-review extractions + statistics + summary JSON)
     v1  nested JudgeVerdict, ≤2 repairs                      prompts/judge.py TEMPLATE        → outputs/live/judgements/{judgements,judge_failures}.jsonl
     v2  flat JudgeOutputV2, constrained decoding, normaliser, cross-field validator (H<5 ⇔ claims), ≤2 repairs
                                                              prompts/judge.py TEMPLATE_V2     → outputs/live/judgements/v2/judgements.jsonl
     v3  flat JudgeOutputV3, claims before H, 1 call, no repair, raw kept, post-hoc reconciliation (H reported only if
         consistent with claims)                              evaluation/judge_v3.py, scripts/judge_v3.py → outputs/live/judgements/v3/
  ▼
[6] NEGATIVE CONTROL
     v1  corrupt_summary(): 3 defects at once (3 products)    evaluation/summary_evaluation.py → outputs/live/d3/judge_sanity/
     v2  1 planted defect per copy: invented_feature / fabricated_statistic, keyword detection on claim text
                                                              evaluation/negative_control.py, scripts/judge_negative_control.py
                                                              → outputs/live/d3/judge_negative_control_v2/ (v2 judge); v3 judge re-used the same 30 copies (identical planted texts, verified)
     semantic review of v2 NC                                 scripts/nc_semantic_review.py → semantic_review.json
  ▼
[7] CORRECTIONS (separate ids, originals untouched)           scripts/build_corrected_summaries.py → outputs/live/d3/corrected_summaries/
[8] REPORTING / UI                                            scripts/export_d3.py, scripts/build_reports.py, reports/D3_*.md,
                                                              ui/summary_evaluation.py, ui/product_insights.py
```

### Missing or inconsistent steps in the flow

1. **No stage verifies extractions against review text for D3.** Statistics are faithful to extractions by
   construction ("statistics match source" in the audit is circular), but extraction errors flow straight into
   summaries (B001DHKBXU, §9.2).
2. **No claim decomposition.** The summary is already structured (each point has cited ids), but neither the audit nor
   the judge treats each point / opinion sentence as a unit with its own evidence.
3. **The judge receives the whole product, not the claim's evidence.** For a 3B model this turns verification into
   retrieval over 9–42 reviews.
4. **Summarisation sees only ≤10 evidence quotes + statistics**, yet writes aggregate claims ("users appreciate X") and
   free-text improvement areas/recommendations that are not tied to any evidence id and are never checked.
5. **Exports are stale.** `outputs/live/d3/<source>/d3_report.md` and `judgements.csv` show v1 (5 judged, 10 failed);
   v2/v3 results exist only in separate files/reports. The UI judge selector offers v1/v2; v3 is a read-only section.
6. **Corrected-copy metadata is stale:** `corrected_summaries/README.json` says "The corrected copies have never been
   judged"; v3 judged both (`evaluated_object = corrected`).

---

## 3. Files and modules per stage

| Stage | Code | Prompts / schemas | Scripts | Artifacts | Tests |
|---|---|---|---|---|---|
| Extract | `pipeline/extraction.py`, `pipeline/repair.py`, `pipeline/validation.py` | `prompts/few_shot.py`, `schemas/review_schema.py`, `schemas/taxonomy.py` | `scripts/run_chain.py` | `outputs/live/extractions/*.jsonl` | `tests/test_chain.py` (+ D1/D2 tests) |
| Aggregate | `pipeline/aggregation.py` | `schemas/evaluation_schema.py` (`ProductAggregate`, `AspectStat`, `EvidenceQuote`) | `scripts/run_chain.py`, `scripts/export_d3.py` | `outputs/live/d3/<source>/aggregates.csv`, `aspect_stats.csv`, `evidence.csv` | `tests/test_chain.py` |
| Summarise | `pipeline/summarization.py` | `prompts/summarization.py` (v1), `ProductSummaryContent`, `SummaryPoint` | `scripts/run_chain.py` | `outputs/live/summaries/summaries.jsonl` | `tests/test_chain.py` |
| Deterministic checks | `evaluation/summary_evaluation.py::deterministic_checks`, `evaluation/d3_audit.py` | – | `scripts/audit_d3.py`, `scripts/export_d3.py` | `outputs/live/d3/audit/<source>/*.json` | `tests/test_d3_audit.py` (10) |
| Judge v1/v2 | `evaluation/summary_evaluation.py` | `prompts/judge.py`, `JudgeVerdict`, `JudgeOutputV2`, `UnsupportedClaim` | `scripts/run_judge.py`, `scripts/run_chain.py --judge` | `outputs/live/judgements/`, `.../v2/` | `tests/test_judge_v2.py` (24) |
| Judge v3 | `evaluation/judge_v3.py` | `JudgeOutputV3`, `SYSTEM_V3` (reuses `prompts.judge.RUBRIC`) | `scripts/judge_v3.py` | `outputs/live/judgements/v3/`, `outputs/live/d3/judge_v3_report/` | `tests/test_judge_v3.py` (12) |
| Negative control | `evaluation/negative_control.py`, `summary_evaluation.py::corrupt_summary` | – | `scripts/judge_negative_control.py`, `scripts/judge_sanity_check.py`, `scripts/nc_semantic_review.py` | `outputs/live/d3/judge_negative_control_v2/`, `judge_sanity/` | `tests/test_judge_v2.py`, `tests/test_judge_v3.py` |
| Corrections | – | – | `scripts/build_corrected_summaries.py` | `outputs/live/d3/corrected_summaries/` | `tests/test_d3_audit.py` |
| Manual inspection | – | – | `scripts/build_manual_inspection.py` | `outputs/live/d3/manual_inspection/` | – |
| Reports / UI | `ui/summary_evaluation.py`, `ui/product_insights.py` | – | `scripts/export_d3.py`, `scripts/build_reports.py`, `scripts/build_final_report.py` | `reports/D3_*.md` | `tests/test_app_smoke.py` |

D3-specific tests run on 2026-11-15: `test_d3_audit.py`, `test_judge_v2.py`, `test_judge_v3.py`, `test_chain.py` —
**68 passed**. They test plumbing (schema, storage, reconciliation, detection rule), not judge quality.

---

## 4. Historical results re-verified against the artifacts

| Reported claim | Recomputed 2026-11-15 | Status |
|---|---|---|
| 15 product summaries generated | 15 rows, 15 products, all `qwen2.5:3b`, prompt v1, not mock, one source | ✅ |
| Judge v1: 5 valid, 10 failures | 5 valid / 10 failures (9 × `faithfulness.unsupported_claims: Extra inputs`, 1 × `faithfulness: Input should be a valid dict`) | ✅ — additionally **4 of the 5 valid v1 verdicts have hallucination 1–2 with zero claims** (already contradictory) |
| Judge v2: 15/15 valid | 15/15; 7/15 needed repairs (4 × 2, 3 × 1) | ✅ |
| v2 hallucination = 4 for every product | H = 4 in 15/15; F {2:1, 3:10, 4:3, 5:1}; consistency {2:7, 3:4, 4:3, 5:1}; 26 claims; 18/26 traceable to summary (0.6 overlap) / 20/26 (audit rule); 17/26 flagged "likely false positive" by the audit's category heuristic; 1 verdict cites placeholder ids `review_id_1`, `review_id_2` | ✅ |
| Judge v3: 47/47 valid | 47 valid, 0 failures (15 original, 2 corrected, 30 NC); 2026-11-14 14:15→16:16 UTC; 153.9 s mean latency; max GPU 87 °C; T = 0, num_ctx 8192, num_predict 2048, constrained decoding | ✅ |
| Evidence audit: 13/15 originals pass | 13/15 under the audit's criterion (ids + re-aggregation + percentages + counts); B0BTY63SFY and B001DHKBXU fail. Points: 83 = 41 traceable / 41 partial / 1 weak (B07XG8QQL5, not counted as a fail) | ✅ but see §8: criterion too weak |
| 130/130 cited ids valid | 130 citations (101 unique), all exist and belong to the product | ✅ |
| v3 score/claim contradictions 37/47 | 37/47 `inconsistent_score_below_5_without_claims` (originals 12/15, corrected 2/2, NC 23/30); 0 cases of "5 with claims"; plus 4 verdicts with H = 4 and ≥ 2 claims (rubric: 4 = one minor detail) | ✅ |
| v3 detected 5/30 planted errors | 5/30 (invented feature 4/15, fabricated statistic 1/15), Wilson 95 % [0.07, 0.34] | ✅ |
| (related) v2 NC detected 18/30 | 18/30 (feature 10/15, statistic 8/15) | ✅ — see §6 for why v2 > v3 |

Other verified facts used below: the v3 prompt fits the context (largest product B0081N3YW4 ≈ 5.1k tokens of
8,192 incl. 2,048 output); 80-word truncation hides only **2 of 177** cited evidence quotes; v2 and v3 negative controls
used **byte-identical planted texts (30/30)**; the one repeated v3 call (pilot vs final, B001DHKBXU original, same
prompt hash) produced a **byte-identical raw response**.

---

## 5. Judge v1, v2 and v3 compared

| | v1 | v2 | v3 |
|---|---|---|---|
| Output shape | nested `JudgeVerdict` | flat `JudgeOutputV2` | flat `JudgeOutputV3` (claims before H) |
| Constrained decoding | no | yes (shape) | yes (shape) |
| Repairs | ≤ 2 | ≤ 2 + normaliser | 0 (1 call) |
| Claim/score coupling | none | **validator forces** H<5 ⇔ ≥1 claim; repair loop re-asks | none at generation; post-hoc: H reported only if consistent |
| Valid outputs | 5/15 originals | 15/15 originals | 47/47 (15 + 2 + 30) |
| Self-consistent verdicts | 1/5 | 15/15 (forced) | 10/47 |
| Hallucination scores | {1:3, 2:2} with 0–1 claims | 4 for all 15 | raw {2:11, 3:10, 4:26} over 47 (originals {2:5, 3:6, 4:4}) |
| Faithfulness scores | {3:1, 4:4} | {2:1, 3:10, 4:3, 5:1} | {2:31, 3:11, 4:5} — never 1 or 5 |
| Claims on originals | 1 | 26 (17 likely FP) | 7 (in 3 summaries; ≥ 3 clear FP, §7.4) |
| NC detection (claim list) | (v1 sanity: 3 pairs, corrupted scored = original on F) | 18/30 | 5/30 |
| Main failure | schema non-compliance | claims induced by validator pressure; score collapses to 4 | claim list left empty; scores unrelated to content |

**Interpretation.** v1 failed structurally. v2 "fixed" validity by forcing consistency; the model satisfied the
validator with the cheapest valid answer (H = 4 plus one or two claims), so the H dimension carries no information and
7/15 claim lists exist only because of the repair loop. v3 removed the pressure and exposed the underlying behaviour: the
model does not populate the list it is asked for, and its scores sit in a narrow 2–4 band regardless of input. The
progression shows that **the problem is not the output format but the task design**.

---

## 6. Score/claim-list contradictions — detailed analysis

### 6.1 What the contradiction looks like
In 37/47 v3 verdicts the raw hallucination score is 2–4 while `unsupported_claims` is `[]`. In **25** of those the
justifications/overall comment explicitly name unsupported claims (e.g. B001DHKBXU: *"The summary introduces several
unsupported claims, such as 'The Whirlpool 3949238 Washer Lid Switch received a…'"*, list empty, H = 4). Faithfulness is
< 5 in all 37 as well.

### 6.2 Causes (ranked by evidence)

1. **Scores and claims are independent generations.** The rubric asks for holistic 1–5 scores; nothing makes the score
   a function of the list. A 3B model produces a typical score (2–4) as a prior; v3's own data show F never reaches 1 or
   5 and raw H is 4 for 21/30 corrupted copies.
2. **Constrained decoding makes `[]` the easiest path.** With a JSON-schema grammar the array field is reached after
   three justifications; an empty array is a valid two-token continuation. The model then "reasons" in the later
   `hallucination_justification`, which is free text. v2 avoided this only by rejecting `[]` and re-asking.
3. **The task is retrieval + verification over the whole product.** Up to 42 reviews (≈ 5k tokens) per call. The model
   misreads the statistics block: it calls *"a majority of negative reviews"* unsupported when 64.3 % are negative
   (B0081N3YW4) and *"generally well-received"* unsupported at 77.8 % positive (B01FZJXITU).
4. **"Unsupported" is undefined for aggregate claims.** The judge rejects "users appreciate the ease of installation"
   because it is "not supported by all reviews" (B00DM8K1NS, installation 4 positive / 1 negative). There is no
   notion of partial support, a support threshold, or paraphrase tolerance.
5. **No missing-vs-contradicted distinction and no evidence anchoring.** Claims are free text; the judge invented claims
   that are not in the summary (B07XG8QQL5: *"The product has a 90-day warranty"* — taken from a review, not the summary;
   *"The product is durable and reliable"* — the summary says the opposite).
6. **Reconciliation hides rather than resolves.** v3 sets H = None for inconsistent verdicts; means then cover only
   3/15 originals and 7/30 NC copies — not a usable measurement.

### 6.3 Conclusion on scoring
Yes — **scores must be derived from claim-level results**, not generated by the LLM. With per-claim labels
(supported / partially supported / unsupported / contradicted) the rubric anchors become deterministic rules (e.g.
H = 5 iff no unsupported/contradicted claims; H ≤ 2 if any contradicted or fabricated statistic). A summary can then
never receive a high faithfulness score while its own claim analysis lists unsupported claims.

---

## 7. Negative-control error analysis

### 7.1 What was planted
30 copies = 15 products × 2 single-defect types (`evaluation/negative_control.py`):

* **invented_feature** — an appreciated point praising a feature from a 6-item pool (Wi-Fi/voice, solar power, UV
  sterilisation, colour touchscreen, lifetime guarantee, smartphone energy app), chosen only if none of its keywords
  occurs in the product's reviews; it cites the product's first review id.
* **fabricated_statistic** — "In total, NN % of reviewers say they would buy this exact product again." appended to
  the overall opinion, NN ∈ {97, 93, 88, 91, 86} and > 3 points from every computed percentage.

### 7.2 Did the judge have the evidence, and were the errors detectable?
* Yes: the judge received all reviews of the product (80-word truncation; keywords were checked on full texts),
  the extractions and the statistics.
* **Fabricated statistic:** detectable from SOURCE 3 alone. The existing deterministic percentage check catches it
  **15/15** — this half of the control tests something a regex already guarantees.
* **Invented feature:** detectable only as *absence* across all reviews (hard for a 3B model), although the cited
  review id makes it checkable against one review. Construction and detection rely on the same keyword list (circular
  for any keyword-based checker; fine for an LLM).

### 7.3 Detected vs missed, and why (v3)
Detection rule: a planted keyword appears in the text of a listed claim (reason field ignored).

| Category (my classification of the 30 v3 verdicts) | invented feature | fabricated statistic | total |
|---|---|---|---|
| A. In the claim list (= reported detection) | 4 | 1 | **5** |
| B. Not in the list, but rejected as unsupported/incorrect in the justification text | 11 | 13 | **24** |
| C. Accepted as fact | 0 | 1 (B0BTY63SFY: *"the actual statistic is 91 % of reviewers saying they would buy…"*) | **1** |

Per missed case (25): 24 are **structural misses** (cause §6.2-2: the error was recognised but the list was left empty
or filled with other claims, e.g. B00DM8K1NS feature: 3 other claims listed, Wi-Fi only in prose), 1 is a **genuine
perceptual miss** (planted number absorbed as fact). Product list for B: B001DHKBXU ×2, B001ONHE8S ×2, B0081N3YW4 ×2,
B008YDTFYW (feature), B00B0ZMNOU (statistic), B00DM8K1NS ×2, B00LHRAGHW (statistic), B01FZJXITU (statistic),
B01LZHJF55 ×2, B0759KR8JX ×2, B07XG8QQL5 ×2, B09WWM1WLS ×2, B0BTY63SFY (feature), B0C5G1WGBN ×2, B001TH7GZU
(statistic, rejected by contrast with the true 32.1 %).

### 7.4 Does the evaluation measure faithfulness detection or JSON generation?
Neither cleanly:

* **The judge's scores do not discriminate at all.** Corrupted vs its own original: raw hallucination **better in
  16/30, equal 14/30, worse 0/30**; faithfulness equal 26, lower 3, higher 1; sum of the four raw scores higher for the
  corrupted copy in 12, lower in 4. Mean raw H: originals 2.93, corrupted 3.53.
* **The judge's prose "detects" almost everything because it flags almost everything.** It claims unsupported content
  in all 15 originals. Of the 7 listed claims on originals: 1 clear true positive (B0C5G1WGBN "good value for money",
  cited review R4cc079d33dae is a complaint), ≥ 3 clear false positives (B07XG8QQL5 "90-day warranty" and "durable and
  reliable" are not in the summary; B0C5G1WGBN "works well" is supported by its own citation Rf06c65f9d7b5,
  "Overall is working great"), the rest debatable/unverified.
* **No specificity measurement.** There is no set of verified-correct claims, so a judge that rejects everything would
  score 30/30 on prose detection.
* **Too narrow and too easy.** Two blatant insertion types; none of the error types actually found in the real
  summaries (misattributed share, wrong denominator/count, wrong majority direction, citation that contradicts the
  point, polarity flip, over-generalisation, omitted top complaint).
* **Why v2 > v3 (18 vs 5):** the v2 validator required ≥ 1 claim whenever H < 5 and re-asked up to twice, so the list
  was always populated and the salient planted sentence was the obvious candidate. The v2 "detections" are partly
  manufactured by the repair pressure, which is why they are not evidence of a better judge.

**Do not edit the 30 copies.** Keep them and their v2/v3 results as historical evidence; add a new control set (§11, §12).

---

## 8. Evidence audit of all 15 summaries

Columns: audit = existing deterministic audit; **investigator** = my reading of the overall opinion against the
statistics and of key points against the cited review text **[needs human confirmation]**.

| Product | n | Sentiment (+/−/0/mixed %) | Audit | Investigator findings |
|---|---|---|---|---|
| B001DHKBXU | 19 | 21.1/78.9/0/0 | FAIL (7 misleading denominators) | §9.2: denominators; "installation was difficult" cites a review saying "easy to install"; "compatibility issues" cites a review saying "exact match"; poor-value/installation stats inflated by extraction errors |
| B001ONHE8S | 16 | 62.5/31.2/0/6.2 | pass | "slight leaning towards positive" understates 62.5 % (minor tone) |
| B001TH7GZU | 28 | 32.1/64.3/0/3.6 | pass (3 partial) | "mixed reception" for 64 % negative (minor tone) |
| B0081N3YW4 | 42 | 21.4/64.3/2.4/11.9 | pass (4 partial, 0 traceable) | "mixed reception with a majority of negative reviews" — acceptable |
| B008YDTFYW | 11 | 54.5/36.4/0/9.1 | pass (8 partial) | "slight majority of positive" correct |
| B00B0ZMNOU | 24 | 25.0/66.7/0/8.3 | pass | "mixed overall sentiment" while 66.7 % negative (minor tone; numbers right) |
| **B00DM8K1NS** | 13 | 38.5/46.2/7.7/7.7 | pass | **"3 positive reviews out of 13" — actual 5**; **"neutral to slightly positive tone" — negative (46.2 %) exceeds positive (38.5 %)**. Audit misses it: its count regex needs "N of M reviews" |
| **B00LHRAGHW** | 10 | 40.0/50.0/10.0/0 | pass | **"slight majority of positive reviews … generally well-received" — wrong direction (50 % negative)** |
| **B01FZJXITU** | 9 | 77.8/22.2/0/0 | pass | **"majority of reviews (7 out of 9) praising its performance and price value"** — 7 is the positive-review count; performance positive = 4, price positive = 2 (misattributed count, same pattern as B0BTY63SFY's 80 %) |
| B01LZHJF55 | 24 | 33.3/58.3/4.2/4.2 | pass | "slightly negative" understates 58.3 % (minor) |
| B0759KR8JX | 14 | 71.4/28.6/0/0 | pass | "40 % cheaper" quoted from review (correctly recognised) |
| **B07XG8QQL5** | 19 | 21.1/68.4/5.3/5.3 | pass (1 weak point) | **"a majority of users experiencing performance issues and poor durability"** — performance_issue 6/19, poor_durability 6/19 (31.6 % each); "sensor malfunctions" unsupported (audit weak point) |
| B09WWM1WLS | 19 | 36.8/47.4/10.5/5.3 | pass | consistent with statistics |
| **B0BTY63SFY** | 15 | 86.7/13.3/0/0 | FAIL (80 %, 67 %) | §9.1: 80 %/67 % misattributed; "suitable for larger eggs" cites a review saying larger eggs don't fit; "small legs" is a misreading of "Only small eggs fit"; durability improvement areas contradict 3+/0− |
| B0C5G1WGBN | 22 | 36.4/59.1/0/4.5 | pass (5 partial, 0 traceable) | "good value for money" cites R4cc079d33dae, which is a complaint and says nothing about value |

**Tally:** clear factual errors in **6/15** (B001DHKBXU, B0BTY63SFY, B00DM8K1NS, B00LHRAGHW, B01FZJXITU, B07XG8QQL5)
plus an unsupported point in B0C5G1WGBN; tone imprecision in 5 more. Point-level semantic support was **not**
systematically checked for all 83 points — only 41 are "traceable" even lexically. This is exactly the gap a claim-level
gold set must close (§11).

---

## 9. The two problematic products

### 9.1 B0BTY63SFY — 77L Deviled Egg Containers (summary `a61df84f4901`)
Statistics: 15 reviews; 13 positive (86.7 %), 2 negative (R80f511ff60c4, Rd1d0ae81cb05); avg 4.2; 3 reviews with
complaints; aspects performance 8+/2−, ease_of_use 7+/1−, size_capacity 5+/0−, compatibility 2+/2−,
quality_durability 3+/0−, installation_setup 0+/2−.

| # | Claim (summary) | Evidence | Verdict |
|---|---|---|---|
| 1 | "positive sentiments from 80 % of the reviews" | positive reviews 86.7 %; 80 % = performance mentions 8/10 | **Unsupported (misattributed statistic)** — known, corrected in `-corr1` |
| 2 | "67 % of negative reviews mentioning these aspects" | both negative reviews (2/2); 67 % = 2/3 complaining reviews | **Unsupported (misattributed)** — known, corrected |
| 3 | Appreciated: "suitable for holding larger eggs and fit well in the fridge" [R80f511ff60c4] | R80f511ff60c4 (3★, negative): *"Only small legs fit. A view put larger eggs the let will not close properly"* | **Contradicted by its only citation** — NEW, not in audit or correction (partial support exists in uncited R1c492cdbe101 and R24b8b37aae15) |
| 4 | Complaint: "do not fit properly for larger eggs and are too tall for the container" [R90688f008c98, R80f511ff60c4] | R90688f008c98: duck eggs "too tall for the container"; R80f511ff60c4 larger eggs | Partially supported (garbled: the eggs are too tall) |
| 5 | Complaint: "lids do not seal properly and the containers do not stack or snap together" [Rd1d0ae81cb05, Rbd2a0049f12f] | "The lids do not seal at all"; "hoped they would stack better, even snap together" | Supported |
| 6 | Complaint: "small legs that do not fit properly" [Rbd2a0049f12f] + improvement area + 2 recommendations about "included legs" | Rbd2a0049f12f: "little feet … have to line up just right"; "Only small legs fit" in R80f511ff60c4 is a typo for eggs | **Mostly unsupported** — misreading amplified into 3 items (one recommendation duplicated) — NEW |
| 7 | Improvement: "Enhancing the design and material quality for durability"; recommendation: "Invest in better material quality" | quality_durability 3+/0−; Rbd2a0049f12f "fairly thin but sturdy enough" | **Unsupported** (violates summariser rule 6) — NEW |
| 8 | "concerns about the installation setup" | installation_setup 0+/2− (extraction labels lid snapping as installation) | Consistent with stats; semantically an extraction artefact |

Judges: v2 flagged only a stacking sentence; v3 original F 3, raw H 2, no claims, prose flags "great for storing" (claim
is supported by R1c492cdbe101 "They're great for storing lots of eggs"). **Neither judge flagged #1–#3, #6, #7.**
Corrected copy `a61df84f4901-corr1` fixes only #1 and #2; v3 scored it F 4, raw H 2, no claims — so it still contains
#3, #6, #7.

### 9.2 B001DHKBXU — Whirlpool 3949238 Washer Lid Switch (summary `2f2f868f0ec7`)
Statistics: 19 reviews; 4 positive (21.1 %), 15 negative (78.9 %), 0 mixed; 15 with complaints; complaint counts
poor_durability 8, installation_difficulty 7, poor_value 6, product_defect 5, customer_service 4, performance 4,
compatibility_fit 1, usability 1.

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 1 | Seven "N out of 15 reviews" complaint shares | counts are over 19 reviews; 15 = complaining reviews | **Misleading denominator** (known, corrected in `-corr1`) |
| 2 | "mixed to negative overall sentiment" | 0 % mixed, 78.9 % negative | Imprecise tone (minor) |
| 3 | Appreciated: "Installation was easy and the part arrived on time" [R326a034e9581, R67a4f9906504] | "easy to install"; "got the part in a couple of days" | Supported |
| 4 | Complaint: "Installation was difficult and the part did not work as intended" [R326a034e9581, R037714d5ebd7] | R326a034e9581: **"easy to install"**; R037714d5ebd7: "Installed part, but did not work" | **"Installation was difficult" contradicted/unsupported by both citations**; the same review supports #3 and is cited against it — NEW |
| 5 | "failed in less than 3 months" [R55ca2e9dc390] | verbatim | Supported |
| 6 | "did not provide good value for money" [R28878e067e18] | review went to Amazon because Sears was expensive; no value complaint about this part | **Unsupported by citation** — NEW |
| 7 | "issues with functionality and customer service" [R55ca2e9dc390, R037714d5ebd7] | "would recommend not buying from SupplyEdge"; "Returned" | Partially supported (seller, not customer service) |
| 8 | "did not perform as expected and had compatibility issues" [R55ca2e9dc390, R28878e067e18] | R037714d5ebd7 (uncited) says "exact match"; neither cited review reports incompatibility | **"compatibility issues" unsupported** — NEW |
| 9 | Complaint statistics themselves | of 7 `installation_difficulty` labels ≈ 2 genuine (R88206a7f14bb, R2079a0152739); of 6 `poor_value` ≈ 1 genuine (Re98c674dfc8e) — R5caf7dd50929, R2ac41d087893, Rc3ae51a75bdb *praise* the low price | **Upstream extraction error**: the summary faithfully repeats statistics that are unfaithful to the reviews — NEW |

Judges: v1 failed; v2 flagged "did not provide good value for money" (the audit's heuristic called this a false positive
because `poor_value` is a top complaint — in fact the judge was right about the cited review and the statistic is
inflated); v3 F 2, raw H 4, no claims. **No judge flagged the denominators, #4 or #8.** Corrected copy `-corr1` fixes
only #1.

### 9.3 Recommendation for these products
Keep originals and `-corr1` copies. Create `-corr2` copies (new ids, deterministic text edits with a source note per
edit, provenance "AI coding assistant, not human-approved") addressing #3, #6, #7 (B0BTY63SFY) and #4, #6, #8
(B001DHKBXU); state that #9 cannot be fixed in the summary without re-extraction. Evaluate `-corr2` separately under
judge v4; never transfer scores.

---

## 10. Missing functionality and technical deficiencies

**Evaluation design (critical)**
1. No claim-level faithfulness verification; scores are holistic LLM opinions.
2. Scores not derived from claims → contradictions (37/47) and no discrimination (0/30 corrupted copies scored worse
   on raw H).
3. Judge context = whole product; claim evidence (cited reviews) not isolated; reviews truncated.
4. No label set {supported, partially supported, unsupported, contradicted}; no paraphrase/aggregate-claim policy; no
   requirement that the judge quote the evidence span (and no substring check of that quote).
5. Judge can list claims that are not in the summary (6/7 traceable in v3 originals; v2 18–20/26).
6. No human ground truth for judge calibration (README §19 acknowledges).
7. Negative control: 2 blatant types, no clean controls, keyword detection rule, no per-type realistic errors.
8. Reproducibility: one repeat only; no label-agreement study across runs/models.

**Deterministic checks (high)**
9. Count regex misses "3 positive reviews out of 13", "7 out of 9 praising …"; no check of qualitative statistic claims
   ("majority", "slight majority", "most", "well-received") against sentiment/complaint shares; no attribution check
   (which statistic a number belongs to).
10. Citation polarity is computed in `export_d3.py` but not part of the audit pass criterion or of generation-time
    validation; same review cited for opposite points is not detected.
11. Improvement areas/recommendations are unchecked (no evidence ids, no link to negative aspects).

**Pipeline (medium)**
12. Extraction errors propagate into statistics (complaint precision 0.37); D3 has no extraction-faithfulness check.
13. Summariser input is ≤10 quotes; summariser misreads typos ("legs") and violates rule 6.

**Reporting / consistency (medium)**
14. `d3_report.md` / `judgements.csv` exports show v1 only; UI selector lacks v3; corrected-copy README stale; audit
    "13/15 pass" is presented as faithfulness although it is id/number/lexical traceability.

---

## 11. Recommended improvements (ranked)

| Rank | Improvement | Why | Cost (local) |
|---|---|---|---|
| **1** | **Claim-level verifier "judge v4" with derived scores.** (a) Deterministic claim decomposition from `ProductSummaryContent`: each appreciated/complaint point = 1 claim with its cited ids; overall-opinion split into sentences; improvement areas/recommendations = "derived" claims. (b) Deterministic verifiers for numbers, counts, denominators, majority/direction words, attribution, citation polarity, frequent aspects. (c) LLM labels **one claim at a time** against the **full text of its cited reviews** (≤ ~800 tokens), output `{label ∈ supported/partially_supported/unsupported/contradicted, evidence_quote, rationale}` with an enum under constrained decoding; `evidence_quote` must be a substring of a cited review (else label downgraded and flagged). (d) Scores computed by fixed rules from labels; LLM never writes a score. | Removes the contradiction by construction; turns verification into a task a 3B model can do; makes every verdict auditable | ~125 claim checks per 15 summaries × ~10–20 s ≈ 25–45 min with llama3.2 3B, sequential; deterministic part instant |
| **2** | **Human claim gold set.** Export the ~125 original-summary claims + cited review texts to CSV; 2 team members label each claim with the same 4 labels (blind to judge output); report Cohen's κ between humans and between judge v4 and humans, per-label precision/recall with CIs | The only way to show the judge measures faithfulness; must be genuine human work (never AI-filled) | ~1–2 h per annotator; no inference |
| **3** | **Negative control v3 (new, pre-registered, seeded).** Keep the 30 old copies. Add per product: misattributed statistic, wrong denominator/count, wrong majority direction, polarity flip of a point, citation swapped to a contradicting review, over-generalisation, invented feature (re-used type), plus **clean controls** (claims labelled supported in the gold set). Detection = verifier label on the planted claim unit (no keywords). Report per-type sensitivity, specificity on clean claims, Wilson CIs, and deterministic-only vs LLM vs combined | Measures real detection ability and false-positive rate; includes the error types the real summaries actually contain | Only the planted claims need LLM calls (~7 × 15 ≈ 105 short calls ≈ 20–35 min) |
| **4** | **Extend deterministic audit** (count/majority/attribution/polarity/duplicate-citation/derived-claim checks) and make it the first stage of v4 | Catches 4 of the 6 erroneous summaries without any LLM | none |
| **5** | **Corrected copies `-corr2`** for B0BTY63SFY and B001DHKBXU (§9.3), evaluated separately | Completes the "fix and re-evaluate" story honestly | 2 summaries × ~8 claims |
| **6** | **Reproducibility study**: re-run v4 on a fixed subset (e.g. 5 summaries + 20 planted claims) once more; report label agreement; optional second verifier model (e.g. `qwen2.5:3b`, noting same-family bias) on disagreements only | Shows stability; cheap | ~10 min |
| **7** | **Reporting consistency**: version-aware `export_d3.py`; UI v4 section; fix corrected-copy README; rename the audit outcome to "ids/statistics/lexical traceability" | Removes contradictions between artifacts | none |
| 8 (optional) | Extraction-faithfulness spot check for D3 statistics (re-use the 300 human annotations for complaint categories on the 299 reviews: complaint-count error per product) | Quantifies how far statistics drift from human labels; no inference | none |
| 9 (optional, later) | Summariser v2 (polarity-validated citations, evidence-linked improvement areas) as a *separate* experiment | Better summaries; does **not** replace the evaluated originals | 15 summaries |

Not recommended: larger judge models (4 GB VRAM; 7B at Q4 is borderline and slow), concurrent inference, repeated full
re-judging of all objects with the holistic rubric, or editing the existing negative-control set.

---

## 12. Phased implementation plan (after approval)

**Phase 0 — freeze & baseline (no inference).** Hash manifest of all D1/D2/D3 artifacts; backup folder; record current
test count. Exit: manifest written, backup verified.

**Phase 1 — deterministic claim layer (no inference).**
- New `evaluation/claims.py`: `Claim` model (id, summary_id, field, text, cited_ids, kind ∈ {point, opinion_sentence,
  derived}), deterministic decomposition.
- New `evaluation/claim_checks.py`: numeric/percentage/count/denominator/majority/attribution/polarity/duplicate-citation
  checks → per-claim deterministic findings.
- Extend `evaluation/d3_audit.py` to use them (old output files kept; new file name with date).
- Exit: all 15 originals decomposed; deterministic findings reproduce the known defects (B0BTY63SFY 80/67, B001DHKBXU
  denominators) and the new ones (B00DM8K1NS "3 of 13", B00LHRAGHW direction, B01FZJXITU attribution, B07XG8QQL5
  "majority"); unit tests green.

**Phase 2 — human gold set (human work, no inference).**
- `scripts/export_claim_gold_template.py` → CSV of claims + cited review texts (no judge output).
- Team labels; `scripts/import_claim_gold.py` validates (annotator names, labels, timestamps) into
  `data/annotations/claim_gold.json` (separate store, provenance human).
- Exit: ≥ 2 annotators per claim, κ reported, disagreements adjudicated by a named person.

**Phase 3 — judge v4 claim verifier (local inference, sequential).**
- New `evaluation/judge_v4.py`, schema `ClaimVerdictV4`, prompt with label definitions + 3 short examples; constrained
  enum; evidence-quote substring check; derived-score rules in one pure function; store `outputs/live/judgements/v4/`
  (raw responses, prompt hash, config; failures separate; resumable; GPU guard as in v3).
- Pilot on 2 summaries; then 15 originals + 2 `-corr1` + 2 `-corr2`.
- Exit: 0 contradictions possible (test), all calls stored, agreement with gold reported.

**Phase 4 — negative control v3 (local inference for planted claims only).**
- New `evaluation/negative_control_v3.py` (old module untouched), seeded generator, manifest of planted claim ids and
  types written **before** judging; clean-control claims drawn from gold-"supported" claims.
- Exit: per-type sensitivity, specificity, CIs; deterministic-only vs v4 vs combined.

**Phase 5 — reproducibility + corrections.** Re-run subset; `-corr2` copies via `scripts/build_corrected_summaries.py`
(new output file, old kept).

**Phase 6 — reporting & UI.** `reports/D3_FINAL_REPORT.md` (new), version-aware `scripts/export_d3.py`, v4 section in
`ui/summary_evaluation.py`, fix `corrected_summaries/README.json` (new version note, not deletion), update
`D3_COMPLETION_REPORT.md` via a new dated revision. Exit: every number in the report regenerated by a verification
script from stored files.

Estimated local inference total: ≈ 1.5–2.5 h of sequential llama3.2 calls across Phases 3–5 (vs ≈ 2 h for the single v3
run), plus 2–4 person-hours of human labelling.

---

## 13. Tests required

| Test | Asserts |
|---|---|
| `test_claims_decomposition` | every point becomes exactly one claim with its cited ids; opinion sentences split deterministically; ids stable across runs |
| `test_claim_checks_numbers` | flags B0BTY63SFY "80 %"/"67 %", B001DHKBXU "N out of 15", B00DM8K1NS "3 positive reviews out of 13", B01FZJXITU "7 out of 9 praising …"; accepts B0759KR8JX quoted "40 %" |
| `test_claim_checks_direction` | "slight majority of positive" with 40/50 → error; "majority of negative" with 64.3 % → ok |
| `test_citation_polarity_and_duplicates` | appreciated point citing only negative reviews is flagged; same review cited for opposite points is flagged |
| `test_derived_scores_consistent` | property test over random label lists: H = 5 ⇔ no unsupported/contradicted; F monotone non-increasing in unsupported share; **no input can produce F ≥ 4 with any contradicted claim** |
| `test_v4_parse_and_failure` | malformed/off-enum output → stored failure with raw text, no label invented; provider error → nothing stored, resumable |
| `test_evidence_quote_must_be_substring` | non-substring quote → label capped at `unsupported`/flagged |
| `test_v4_never_mixes_objects` | original / corrected / nc kinds stored separately; no score transfer between ids |
| `test_negative_control_v3_generation` | seeded, one defect per copy, planted claim id recorded, manifest written before judging, old 30 copies untouched |
| `test_detection_metric` | detection read from the planted claim's label, specificity from clean claims, Wilson CI correct |
| `test_gold_import_provenance` | rejects rows without annotator, AI marker or invalid label; never merges with judge output |
| `test_reports_match_artifacts` | every number in the D3 final report recomputed from stored files |
| `test_d1_d2_untouched` | hash manifest of D1/D2 artifacts unchanged |
| existing | all current tests (274 passed / 1 skipped at last full run) still pass |

---

## 14. Criteria for declaring D3 complete

D3 is complete when **all** of the following hold, verified by scripts from stored files:

1. **Pipeline:** 15 summaries from the documented source; 100 % citation validity; statistics re-aggregate identically
   (already true).
2. **Claim-level audit:** every claim of the 15 originals has (a) deterministic findings and (b) a v4 label with an
   evidence quote that is a substring of a cited review; all summaries with factual errors are listed per claim.
3. **Consistency by construction:** zero verdicts in which a derived score contradicts the claim labels (enforced by
   code and tested).
4. **Judge validity:** v4 agreement with the human claim gold set reported (κ, per-label precision/recall with 95 % CI).
   The thresholds that count as "reliable" are a team decision to be fixed **before** the run (suggested: recall on
   unsupported+contradicted ≥ 0.7 and precision ≥ 0.6); whatever the outcome, it is reported as is.
5. **Negative control:** new pre-registered control with per-type sensitivity, specificity on clean claims and CIs;
   old v2/v3 results reported alongside, unaltered.
6. **Reproducibility:** repeat-run label agreement on a fixed subset reported; prompt hashes and configs stored.
7. **Corrections:** `-corr2` copies for B0BTY63SFY and B001DHKBXU evaluated separately; originals unchanged.
8. **Honest limitations stated:** small 3B judge, extraction errors propagate into statistics (with the measured size),
   human gold set size, single summariser run.
9. **Artifacts consistent:** exports, UI and reports show the same versions and numbers; stale notes corrected.
10. **No collateral change:** D1/D2 hash manifest unchanged; full test suite green.

---

## Appendix A — files that would be modified or added (on approval)

New: `evaluation/claims.py`, `evaluation/claim_checks.py`, `evaluation/judge_v4.py`, `evaluation/negative_control_v3.py`,
`scripts/judge_v4.py`, `scripts/export_claim_gold_template.py`, `scripts/import_claim_gold.py`,
`scripts/negative_control_v3.py`, `scripts/verify_d3_final.py`, `tests/test_claims.py`, `tests/test_judge_v4.py`,
`tests/test_negative_control_v3.py`, `reports/D3_FINAL_REPORT.md`.

Modified: `schemas/evaluation_schema.py` (add `Claim`, `ClaimVerdictV4`; existing models unchanged),
`evaluation/d3_audit.py` (extended checks, new output file), `scripts/build_corrected_summaries.py` (`-corr2`, new
output file), `scripts/export_d3.py` (version-aware), `ui/summary_evaluation.py` (v4 section),
`outputs/live/d3/corrected_summaries/.../README.json` (append a dated note), `reports/D3_COMPLETION_REPORT.md` (dated
revision).

Not modified: prompts/summaries/extractions of the original run, `prompts/judge.py`, `evaluation/judge_v3.py`,
`evaluation/negative_control.py`, all v1/v2/v3 judgement stores, the 30 negative-control copies, D1/D2 code and data.

## Appendix B — reproduction of this investigation
All figures were computed read-only from: `outputs/live/summaries/summaries.jsonl`,
`outputs/live/extractions/dataset__few_shot__qwen2.5-3b__v1.jsonl`, `outputs/live/judgements/{judgements,judge_failures}.jsonl`,
`outputs/live/judgements/v2/judgements.jsonl`, `outputs/live/judgements/v3/{judgements.jsonl,run_manifest.json}`,
`outputs/live/judgements/v3_pilot_superseded_reconciliation/`, `outputs/live/d3/audit/.../{evidence_audit,judge_quality}.json`,
`outputs/live/d3/judge_negative_control_v2/.../{pairs,summary}.jsonl`, `outputs/live/d3/judge_v3_report/.../report.json`,
`outputs/live/d3/corrected_summaries/...`, `outputs/live/experiments/20261111-030912_qwen2.5-3b/metrics.json`. Prompt
sizes were estimated by rendering `TEMPLATE_V3` offline (characters / 4); no model was called.
