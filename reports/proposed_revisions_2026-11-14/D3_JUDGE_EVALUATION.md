# D3 — LLM-as-a-Judge evaluation (v1 original, v2 fixed, v3 no-repair)

Everything ran on local Ollama. The judge is `llama3.2` (3B), a different family from the summariser `qwen2.5:3b`,
to reduce self-preference. Source: `dataset__few_shot__qwen2.5-3b__v1`, 15 product summaries. Judgements produced on 2026-11-13; audit updated on 2026-11-14.
v1, v2 and v3 are stored separately and **never averaged together**. v3 (2026-11-14) is described in §6.

## 1. Judge v1 (original run, preserved unchanged)

Files: `outputs/live/judgements/judgements.jsonl` and `judge_failures.jsonl`. The failures were reconstructed from
logs; raw outputs were not stored at the time.

- **5 valid judgements, 10 failures.** Means of the 5: faithfulness 3.8, coverage 3.8, consistency 3.0,
  hallucination 1.4.
- **Contradicts itself:** in 4 of the 5 judgements the hallucination score is 1–2 while no unsupported claims are
  listed.
- **Cause of the failures:** the prompt sent a generated schema for a nested model, which forbids extra keys.
  `llama3.2` either returned bare integers per dimension, or (after a repair prompt) copied `unsupported_claims` into
  every dimension object. 9 of the 10 failures were this structural problem; 1 was malformed JSON.

## 2. Judge v2 (fix)

| What changed | Where |
|---|---|
| The model is asked for a flat JSON object, with one score and one justification per dimension and a single top-level `unsupported_claims` list (`{claim, reason, evidence_review_ids}`) | `schemas/evaluation_schema.py::JudgeOutputV2` |
| The prompt shows a literal JSON skeleton and explicit rules | `prompts/judge.py::TEMPLATE_V2` |
| Ollama constrained decoding (`format=` the JSON schema); Pydantic still validates the result | `llm/langchain_client.py`, `get_judge_client` |
| A structural normaliser only moves or renames values the judge produced (it flattens the v1 nesting and lifts nested claims). It never invents a score, justification or claim | `pipeline/validation.py::normalize_judge_v2` |
| Scores must be whole numbers 1–5 (3.5, "4/5", booleans, 0 and 7 are rejected), justifications must be non-empty, and extra keys are not allowed | validators |
| Consistency rule: a hallucination score below 5 must list at least one claim, and a listed claim rules out a 5 | `JudgeOutputV2` |
| At most 2 repairs per summary; provider errors are retried at most twice, then the run stops without recording anything; failures are stored with every raw output; reruns are resumable and never repeat a success | `judge_summaries`, `JudgementStore(v2)` |
| Before every call the run pauses at ≥ 88 °C, stops at ≥ 94 °C or when free RAM drops below 2 GB, and runs in a detached process | `utils/gpu_guard.py`, `scripts/run_judge.py` |

## 3. Judge v2 results: 15 / 15 valid, 0 failures

The v2 run was resumed from the 5 judgements already stored, which were kept. Then came a one-product pilot
(B00B0ZMNOU), then the remaining 9.
- Total judge time: 69.5 min.
- Repairs: 7 of the 15 judgements needed one or two repair rounds (11 in total). Every first-attempt rejection was
  caused by the consistency rule.
- Peak GPU temperature: 89 °C, at which point the guard paused the run once.

| Product | F | C | K | H | Claims flagged | Likely false positives* | Repairs | Latency | v1 outcome (F/C/K/H) |
|---|---|---|---|---|---|---|---|---|---|
| B001DHKBXU Whirlpool lid switch | 3 | 4 | 2 | 4 | 2 | 2 | 0 | 335 s | failed |
| B001ONHE8S Silverware basket | 5 | 5 | 5 | 4 | 1 | 0 | 2 | 415 s | 4/4/3/1 |
| B001TH7GZU Stanco drip bowls | 3 | 4 | 2 | 4 | 2 | 2 | 1 | 498 s | failed |
| B0081N3YW4 SIMPLECUPS | 3 | 4 | 3 | 4 | 1 | 1 | 1 | 688 s | failed |
| B008YDTFYW Whirlpool basket | 3 | 4 | 4 | 4 | 1 | 1 | 1 | 539 s | 4/4/3/1 |
| B00B0ZMNOU Solofill cup | 3 | 4 | 2 | 4 | 2 | 2 | 0 | 102 s | failed |
| B00DM8K1NS Whirlpool icemaker | 3 | 4 | 4 | 4 | 2 | 1 | 0 | 71 s | 4/4/4/1 |
| B00LHRAGHW GE timer knob | 3 | 4 | 2 | 4 | 2 | 1 | 0 | 74 s | failed |
| B01FZJXITU Dryer gas valve coil | 4 | 3 | 3 | 4 | 2 | 0 | 0 | 80 s | failed |
| B01LZHJF55 Tier1 filter combo | 3 | 4 | 3 | 4 | 2 | 0 | 2 | 193 s | failed |
| B0759KR8JX UKF8001 filter | 4 | 4 | 3 | 4 | 1 | 1 | 0 | 129 s | failed |
| B07XG8QQL5 Samsung heating element | 2 | 3 | 2 | 4 | 3 | 3 | 0 | 171 s | 3/3/2/2 |
| B09WWM1WLS Napco igniter | 3 | 3 | 2 | 4 | 1 | 0 | 2 | 411 s | 4/4/3/2 |
| B0BTY63SFY Egg containers | 4 | 4 | 4 | 4 | 1 | 1 | 2 | 289 s | failed |
| B0C5G1WGBN BLACK+DECKER dryer | 3 | 4 | 2 | 4 | 3 | 2 | 0 | 175 s | failed |

**v2 means (n = 15):** faithfulness 3.27, coverage 3.87, consistency 2.87, hallucination **4.00**.

\* See §4. This is a keyword heuristic, not proven wrong.

## 4. Quality audit of the v2 verdicts (`evaluation/d3_audit.py::audit_judgement`, no LLM)

| Check | Result |
|---|---|
| Scores are whole numbers 1–5 | 15 / 15 |
| Every dimension has a non-empty justification | 15 / 15 |
| Claims are at the top level, and the hallucination score agrees with the claim list | 15 / 15 (enforced by the schema) |
| Evidence ids cited by the judge exist | 14 / 15. **B09WWM1WLS cites `review_id_1`, `review_id_2`**, placeholders copied from the prompt skeleton |
| Flagged claims found in the summary text | 20 / 26. The other 6 are paraphrases or claims that aren't in the summary (e.g. "The manufacturer is authentic") |
| Flagged claims contradicted by the source data (**likely false positives**) | **17 / 26**. The claim matches a complaint or aspect category of the same polarity in that product's own statistics. Examples: "did not provide good value for money" when `poor_value` is a recorded complaint (B001DHKBXU); "the basket is less sturdy than the original" when `poor_durability` is recorded (B008YDTFYW) |

**Findings:**

1. **Valid JSON, unreliable content.** Every v2 output is schema-valid, which shows the structural fix works. It does
   not show the judgements are correct.
2. **The hallucination score doesn't discriminate.** All 15 summaries got hallucination = 4, and each lists 1–3
   claims. The consistency rule removed v1's self-contradiction, but the model now meets it in the minimal way:
   always one or more claims, always a 4.
3. **It frequently treats legitimate complaints as hallucinations.** 17 of the 26 flagged claims match a category
   actually present in the source statistics. The heuristic can't prove an individual case wrong, so these are
   candidates for manual checking. Even so, the pattern clearly includes negative but source-supported statements.
4. **It misses a real defect.** The evidence audit (`D3_EVIDENCE_AUDIT.md`) found that B0BTY63SFY's summary states
   "80%" and "67%" with the wrong denominators (real ratios, misattributed). Neither judge flagged it; the v2 consistency justification even quotes the correct 86.7%.
5. **The claim lists are partly produced by validator pressure (found 2026-11-14 from `logs/judge_v2.out`).** In all 7
   repaired judgements, the judge's *first* answer gave hallucination 3–4 with **no** unsupported claims, which the
   consistency rule rejected. In 4 of them the second answer swung to hallucination 5 *with* claims (rejected again);
   the accepted third answer was hallucination 4 with 1–2 claims. So for 7 of 15 summaries the listed claims were
   written only after the validator demanded them, which plausibly explains part of the false-positive rate. The same
   pattern occurs in the negative control: 13 of 30 corrupted-summary judgements were repaired, every first rejection
   being "hallucination below 5 but no claims". Per-product history: B001ONHE8S, B01LZHJF55, B09WWM1WLS, B0BTY63SFY
   (2 repairs: no claims → H 5 with claims → accepted); B001TH7GZU, B0081N3YW4, B008YDTFYW (1 repair). Successful
   judgements store only `repair_attempts`; the rejected raw attempts exist only in the log.
6. **The judge ignores its own rubric.** The rubric defines hallucination 4 as "one minor unsupported detail", yet
   9 of the 15 verdicts list 2–3 unsupported claims and still give 4, and 5 hallucination justifications speak of
   "several" unsupported claims while scoring 4 (B001DHKBXU, B0081N3YW4, B008YDTFYW, B01LZHJF55, B0C5G1WGBN).
7. **Negative control (completed 2026-11-14, all 15 products, 30 planted claims):** the judge named the planted claim
   in **18 / 30 (60 %, 95 % CI 42–75 %)** and **missed 12**; faithfulness fell in only 2 / 30 and hallucination in
   2 / 30. A separate model-assisted reading of all 30 judgements (not human ground truth, not combined with the
   keyword rate) adds 1 paraphrased detection and 2 self-contradictory verdicts that call the planted content
   unsupported in a justification while listing no claim and scoring hallucination 5. Details in `D3_NEGATIVE_CONTROL.md`.

**Design lesson (applied in v3, §6):** the hard H-vs-claims rule plus
automatic repair pushes a small model to invent claims. A fairer design would accept the first answer and *record* the
inconsistency, or ask for the claim list first and derive the hallucination score deterministically from it.

**Overall:** judge v2 delivers a complete, validated, auditable verdict for all 15 summaries, which fixes D3's
structured-output failure. On its quality, the evidence above says it is **not a reliable faithfulness measure**.
Its scores are reported as model opinions, alongside the deterministic evidence audit, which is the more trustworthy
signal.

## 5. Reproduce

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
$env:LLM_MAX_OUTPUT_TOKENS="2048"
.venv\Scripts\python.exe scripts\run_judge.py --source dataset__few_shot__qwen2.5-3b__v1   # resumable; all 15 stored -> 0 new calls
Remove-Item Env:LLM_MAX_OUTPUT_TOKENS
.venv\Scripts\python.exe scripts\audit_d3.py --source dataset__few_shot__qwen2.5-3b__v1     # no LLM
```

A fresh v2 store would re-judge all 15 summaries. Because the judge runs partly on the CPU, run-to-run differences in
wording are possible even at temperature 0.

## 6. Judge v3 (separately labelled run, 2026-11-14)

**Why:** v2's consistency validator and repair loop produced 7 of its 15 claim lists. v3 tests the judge without
that pressure.

**Design** (`evaluation/judge_v3.py`, `scripts/judge_v3.py`, launched as `run_judge.py --version v3` under the
unchanged GPU watchdog):
- Same rubric text as v1/v2 and the same flat fields. The claim list is requested *before* the hallucination score.
- **Exactly one call per summary**, no repair, no consistency validator. The **raw response is stored for every
  call**. Validation failures go to a separate `failures.jsonl` (there were none).
- **Score/claim consistency without forcing claims:** the judge's raw hallucination score is always kept. It is
  *reported* only when it agrees with the judge's own claim list (5 with no claims, or 1–4 with at least one
  claim). Otherwise it is reported as `None` and counted as contradictory; no claim is added and no score is
  changed. A pilot used a rule that raised such scores to 5. It was replaced before the full run because it
  contradicted the judge's own justification. The pilot is kept in `judgements/v3_pilot_superseded_reconciliation/`
  and is not used anywhere.
- Original summaries, corrected copies and negative-control copies are separate `evaluated_object` kinds. No score
  is transferred between summary ids.
- **Recorded in `judgements/v3/run_manifest.json`:**
  - model `llama3.2` (Ollama, constrained decoding on);
  - temperature 0, `num_ctx` 8192, `num_predict` 2048, timeout 600 s;
  - prompt SHA-256 `b8570467e4d4…`;
  - run 2026-11-14 14:15–16:16 UTC, 47 calls, peak GPU 87 °C.

**Results** (`outputs/live/d3/judge_v3_report/dataset__few_shot__qwen2.5-3b__v1/report.json`):

| Evaluated | Valid / attempted | Mean F / C / K | Hallucination reported (consistent) | Contradictory (H < 5, no claims) | Summaries with claims (claims; found in summary) |
|---|---|---|---|---|---|
| 15 original summaries | 15 / 15 | 2.47 / 3.27 / 2.60 | 3 / 15 (all H = 3) | **12 / 15** | 3 (7; 6) |
| 2 corrected copies | 2 / 2 | 3.0 / 3.5 / 2.5 | 0 / 2 | 2 / 2 | 0 |
| 30 negative-control copies | 30 / 30 | 2.40 / 3.23 / 2.30 | 7 / 30 | 23 / 30 | 7 (17; 17) |

**Original vs corrected** (each with its own v3 judgement):

| Product | Original (v3) F / C / K | Corrected copy (v3) F / C / K | Claims listed (original → corrected) |
|---|---|---|---|
| B001DHKBXU | 2 / 3 / 2 | 2 / 3 / 2 | 0 → 0 |
| B0BTY63SFY | 3 / 4 / 4 | 4 / 4 / 3 | 0 → 0 |

The judge did not flag the misleading denominators or the misattributed percentages on the originals. It scored
the corrected B001DHKBXU identically to the original. For B0BTY63SFY, faithfulness rose by 1 and consistency fell
by 1: no systematic response to a verified correction.

**v3 negative control** (the same 30 corrupted copies as v2, checked to be identical to the stored v2 pairs; same
claim-text keyword rule):

| Planted defect | Judged | Detected (claim list) | Missed | 95 % CI | Faithfulness lower |
|---|---|---|---|---|---|
| Invented feature | 15 | 4 | 11 | 11–52 % | 1 |
| Fabricated statistic | 15 | 1 | 14 | 1–30 % | 2 |
| **All** | **30** | **5 (16.7 %)** | **25** | **7–34 %** | **3 / 30** |

- No hallucination comparison was possible: in every pair, the original or the corrupted verdict is contradictory.
- The planted keyword never appears in the claim lists of the 30 v3 original judgements (0 / 30 keyword control).
- **In all 25 misses the planted keyword appears in the justification text.** Spot checks show the judge often
  *names* the planted sentence as unsupported in prose ("states that the product 'runs on solar power…', which is
  not supported") while leaving `unsupported_claims` empty. This count does not tell rejection apart from
  acceptance; it is **not** a detection under the rule and is not added to the 5.

**Conclusion on v3:**
- Without validator pressure, `llama3.2` writes its criticism into the justification but rarely fills the
  structured claim list.
- Its hallucination score contradicts its own claim list in 37 of 47 verdicts.
- Its scores do not respond to planted errors (faithfulness lower in 3 / 30) or to verified corrections.

v3 removes the repair artefact but shows that the structured verdict itself is unreliable. **Neither v2 nor v3 is a
reliable faithfulness measure.** v2's 18 / 30 stays the historical result. v3's 5 / 30 is reported beside it and
never pooled. The deterministic evidence audit (13 / 15 originals pass; both corrected copies pass) remains the
trustworthy D3 signal.

```powershell
$env:LLM_MAX_OUTPUT_TOKENS="2048"; $env:LLM_TIMEOUT_SECONDS="600"
.venv\Scripts\python.exe scripts\run_judge.py --version v3 --source dataset__few_shot__qwen2.5-3b__v1   # resumable; all 47 stored -> 0 new calls
.venv\Scripts\python.exe scripts\run_judge.py --version v3 --source dataset__few_shot__qwen2.5-3b__v1 --report-only   # analysis, no LLM
```
