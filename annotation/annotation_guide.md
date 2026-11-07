# Annotation Guide — 300-Review Evaluation Set

These human labels are the **ground truth** for comparing the zero-shot, few-shot
and Chain-of-Thought prompts. Quality matters more than speed. Plan on 25–40 seconds
per review; most reviews are short (median 22 words), so 300 reviews take roughly 2–3 hours.
You can split that over several sessions, because progress is saved after every review.

> **Integrity rule:** label every review yourself, from the review text. Do **not** copy
> model predictions into the annotation tool. Labels produced by an LLM must never be
> described as manual annotations.

## Where to annotate

* **Streamlit app → "Data Annotation" page** (recommended). Each label is saved immediately to
  `data/annotations/annotations.json`.
* **Offline (Excel/Sheets):** fill `data/annotations/annotation_template.csv` and import it on the
  same page. Use the formats shown in the *CSV format* section below.

The 300 reviews are a reproducible, rating-stratified sample (≈60 per star rating) drawn by
`scripts/build_annotation_sample.py` with seed 42. The star rating is **hidden by default**, so that
you judge the text rather than the stars. The models never see the rating either.

## Step-by-step

1. **Read the whole review.**
2. **Overall sentiment** — choose exactly one:

| Label | Use when |
|---|---|
| `positive` | Satisfied overall. Any criticism is absent or trivial. |
| `negative` | Dissatisfied overall. Any praise is absent or trivial. |
| `neutral` | No clear opinion: factual, descriptive, or too vague to judge. |
| `mixed` | Substantial praise **and** substantial criticism ("works great but very loud"). |

3. **Aspects** — tick every aspect category the review *explicitly* discusses, then give each one
   **one** sentiment: `positive`, `negative` or `neutral` (mentioned without any evaluation).
   If the review covers the same category twice with opposite opinions, choose the dominant one
   and add a note.

| Category | What it covers |
|---|---|
| `performance` | How well it does its core job (cooling, heating, filtering, cleaning, power, results). "Works great" = performance. |
| `quality_durability` | Build quality, reliability, lasting, breaking or failing over time. |
| `design_build` | Appearance, materials, finish, physical construction. |
| `ease_of_use` | Controls, operation, cleaning/maintenance effort, convenience. |
| `installation_setup` | Installing, assembling, mounting, initial setup, installation instructions. |
| `noise` | Sound level, vibration, rattling, humming. |
| `size_capacity` | Dimensions, capacity, footprint, weight. |
| `compatibility` | Fits / works with the reviewer's model or appliance ("fits my Whirlpool perfectly"). |
| `price_value` | Price, value for money, cheaper than OEM. |
| `customer_service` | Seller/manufacturer support, warranty, returns, refunds. |
| `shipping_packaging` | Delivery speed, packaging, condition on arrival. |
| `safety` | Smoke, burning smell, overheating, sparks, hazardous leaks. |
| `other` | A genuine product aspect that fits none of the above (add a note). |

4. **Complaint categories** — tick every complaint that is present (multi-label). Leave this
   empty for reviews with no complaint.

| Complaint | Use when |
|---|---|
| `product_defect` | Arrived broken, dead on arrival, manufacturing defect. |
| `poor_durability` | Worked at first, but broke or wore out too soon. |
| `performance_issue` | Works, but performs poorly at its core job. |
| `noise_issue` | Too loud, rattles, vibrates. |
| `installation_difficulty` | Hard to install, poor installation instructions. |
| `usability_issue` | Hard or inconvenient to operate, clean or maintain. |
| `compatibility_fit_issue` | Did not fit or work with the expected model. |
| `poor_value` | Overpriced, not worth the money. |
| `shipping_damage` | Damaged in transit, poor packaging, late or lost. |
| `missing_parts` | Parts or hardware missing. |
| `not_as_described` | Differs from listing, counterfeit, misleading. |
| `customer_service_issue` | Bad support, warranty or returns experience. |
| `safety_concern` | Hazard: smoke, burning smell, overheating, sparks. |
| `other` | A genuine complaint not covered above (add a note). |

5. **Flag** reviews that are unreadable, not in English or genuinely ambiguous (status `flagged`).
   Flagged reviews are excluded from evaluation.

## Decision rules and edge cases

* **Only label what is written.** "Exact replacement" → `compatibility:positive`. Don't infer
  durability from "just installed it".
* **Negation:** "not loud at all" → `noise:positive`.
* **Expectation statements:** "hope it lasts" → no `quality_durability` label (no evaluation yet).
* **Star rating vs text:** label the text. A 4★ review whose text only criticises is `negative`
  or `mixed`.
* **Complaint ⇔ negative aspect:** most complaints have a matching negative aspect, but not always
  (`missing_parts`, `not_as_described`).
* **Returned items:** returning a product is not itself a complaint category. Label the *reason*
  for the return.
* **Very short reviews** ("Great!", "As described"): `positive`, and usually no aspects.
  "As described" → no aspect.

## Worked examples

| Review | Overall | Aspects | Complaints |
|---|---|---|---|
| "Easy to install, functions as expected. Fantastic price!!" | positive | installation_setup:positive; performance:positive; price_value:positive | — |
| "Fits my Kenmore fridge perfectly" | positive | compatibility:positive | — |
| "The plates fit but slide around and there is no way to level them." | negative | compatibility:negative; design_build:negative | compatibility_fit_issue |
| "Works fine but arrived with a dented corner." | mixed | performance:positive; shipping_packaging:negative | shipping_damage |
| "Bought for my mom." | neutral | — | — |

## CSV format (offline annotation)

| column | format |
|---|---|
| `overall_sentiment` | one of `positive`, `negative`, `neutral`, `mixed` (leave empty = not yet labelled) |
| `aspects` | `category:sentiment; category:sentiment` e.g. `noise:negative; performance:positive` |
| `complaint_categories` | `cat1; cat2` e.g. `noise_issue; poor_value` |
| `annotator` | your name or initials |
| `status` | `complete` (default) or `flagged` |

## Optional: inter-annotator agreement

If a teammate is available, have them independently label the same 50 reviews and report Cohen's
kappa on overall sentiment. This makes the ground truth more credible in the report.
