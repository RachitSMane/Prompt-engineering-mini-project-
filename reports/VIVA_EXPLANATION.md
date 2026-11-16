# ReviewIQ — viva / presentation explanation (D1, D2, D3)

A short spoken explanation (about 5 minutes), followed by likely questions with honest answers. Every number is
from the saved results and was re-verified on 2026-11-14. Inference ran locally with Ollama only: `qwen2.5:3b`
for extraction and summaries, `llama3.2` as the judge.

## The one-sentence pitch

ReviewIQ turns raw Amazon appliance reviews into validated structured data (D1). It compares three prompting
strategies for that extraction (D2). It then turns the data into evidence-cited product summaries, checked
deterministically and by an LLM judge whose reliability we tested (D3).

## D1 — Structured outputs (complete)

- **What:** each review becomes a Pydantic-validated object: overall sentiment, aspects (13 categories, each with a
  sentiment and a verbatim evidence quote) and complaint categories. LangChain builds the prompts, and local
  Ollama runs them.
- **Robustness:** invalid output goes through deterministic local fixes and then at most 2 LLM repair attempts.
  Anything still invalid is stored as a **controlled failure**, with the raw output and the error, never dropped or
  filled with invented values.
- **Result:** 282 / 285 / 279 valid out of 299 (zero-shot / few-shot / CoT), with 51 controlled failures in total.
- **Verification:** an independent script checks the JSON and CSV exports against the stored predictions (9 checks
  × 3 strategies = **27 / 27 PASS**).

## D2 — Prompt engineering (experiments complete; human ground truth outstanding)

- **What:** zero-shot, few-shot and chain-of-thought prompts were run on the same 299 reviews and scored with
  accuracy, precision, recall, F1 and macro-F1.
- **Result:**

  | Strategy | Macro-F1 | Accuracy |
  |---|---|---|
  | **Few-shot** | **0.603** | **0.813** |
  | Zero-shot | 0.438 | 0.612 |
  | CoT | 0.423 | 0.575 |

- **Why:** zero-shot and CoT label mildly positive reviews "neutral" (CoT's reasoning makes the model hedge).
  Few-shot examples teach the boundary: positive recall rises from about 0.5 to 0.92. All three miss "mixed".
  Few-shot costs about 40 % more input tokens.
- **Verification:** every metric was recomputed with the standard library only, and all values match. All three
  strategies used the identical 299 review ids, with no duplicates.
- **Reference labels:** the comparison dataset is the 300 existing reference annotations (1 human, 299
  AI-assisted, mentor-approved). A named reviewer approves them in the app, and the approval is stored separately
  without changing provenance.
  - **Status:** 300 / 300 accepted by the reviewer.
  - The **primary** result stays the original 299-review run. The 300-review evaluation adds the formerly excluded,
    garbled review `R5cfb12ef75b4` (neutral, low-confidence label) as a **sensitivity analysis**: macro-F1
    0.4435 / 0.6322 / 0.4276. Same ranking; one extra correct neutral moves few-shot macro-F1 by +0.03.
- **Honest caveat:** these numbers are agreement with mostly AI-assisted labels, not accuracy against independent
  human ground truth. A blind two-person human workflow exists, and its labels are counted separately (0 / 300
  verified). We did not fake them.

## D3 — Summaries and faithfulness (implemented and executed; not fully compliant)

- **Chain:**
  1. **Extract:** the D1 records.
  2. **Aggregate:** pandas computes statistics only.
  3. **Summarise:** `qwen2.5:3b` writes from the statistics and evidence, and every point must cite review ids.
  4. **Evaluate:** a deterministic evidence audit plus an LLM judge.
- **Evidence audit (deterministic, the most trustworthy signal):** **13 / 15** summaries pass. All 130 cited ids are
  valid.
  - **B0BTY63SFY** misattributes two real ratios (80 % and 67 %).
  - **B001DHKBXU** says "N out of 15 reviews" where the product has 19.
  - We keep the originals unchanged, and separately labelled corrected copies exist.
- **Judge v2:** valid JSON for 15 / 15. On its own the negative control shows it is **not reliable**:
  - 30 planted unsupported claims in corrupted copies;
  - the judge's claim list named **18 / 30** and missed **12**;
  - faithfulness fell in only 2 / 30.
  - A separate reading of all 30 judgements by Claude, an AI assistant (not human ground truth), finds 1 paraphrased
    detection and 2 self-contradictory verdicts.
- **Judge v3 (separate run):** one call per summary, no repair (v2's repair loop forced claims into 7 of 15 claim
  lists), raw responses kept, and the corrected copies judged separately. 47/47 valid.
  - In 12 of 15 originals the hallucination score contradicts the judge's own empty claim list.
  - The corrected B001DHKBXU was scored identically to the original.
  - v3 detected only 5 of the 30 planted claims, because the judge criticises them in prose but leaves the claim
    list empty.
- **Bottom line:** the pipeline works and is auditable, but the 3B local judge cannot be trusted as a faithfulness
  measure. We say so, and we rely on the deterministic audit.

## Likely questions

**Are the D2 labels human?** One is; the others are AI-assisted, mentor-approved reference labels, marked as such
everywhere. Approval in the app records who accepted them; it does not make them human-created.

**Why not just relabel the AI labels as human?** That would be fabrication. A label counts only if a person decided
it blind. Our 300 independent human annotations were all made that way; second-person verification was not part of
the project's workflow, so we describe them as human-created, not as verified.

**Why few-shot?** It is best on macro-F1, accuracy, validity and aspect detection. Its extra cost (tokens and
latency) is moderate.

**How do you know the summaries are faithful?** The deterministic audit re-traces every cited id, statistic,
percentage and count to the source records. It caught 2 of 15 summaries; the LLM judge flagged neither.

**Why keep an unreliable judge?** The assignment asks for LLM-as-a-judge. We implemented it, tested it with a
negative control, and report that it is unreliable instead of hiding it.

**What would you do next?** Collect the 300 human labels, use a larger judge or calibrate it against human ratings,
and extract all 1,500 reviews instead of the 299-review sample.
