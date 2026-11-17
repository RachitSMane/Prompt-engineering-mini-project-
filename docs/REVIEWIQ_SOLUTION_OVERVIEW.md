# ReviewIQ – Aspect-Based Customer Review Analytics with Structured Outputs

*From thousands of free-text reviews to validated, product-level insight — on a local LLM.*

PES University · Prompt Engineering Mini Project · 5th Semester AIML · Team: Rachit S Mane, P. Chinni Krishna Reddy, Pranav V. Menon, Akash R.

## WHY – The problem

- E-commerce platforms receive very large volumes of customer reviews. They hold valuable feedback, but almost all of it is **unstructured text**.
- A star rating or overall sentiment says *how much* a customer liked a product, not *why*: it cannot show **which features** (noise, installation, durability, fit, price…) customers like or dislike.
- Reading and tagging reviews by hand is slow, inconsistent and does not scale.
- Product and support teams need **structured, actionable insight**: what customers praise, what they complain about, and what to fix first.

## WHAT – Our solution

ReviewIQ is an LLM pipeline that runs **entirely on local hardware** (Ollama, `qwen2.5:3b`; no API key, no review text leaves the machine). It converts each review into a schema-validated record containing:

- **Overall sentiment** – positive, negative, neutral or mixed.
- **Product aspects** from a fixed taxonomy of 13 categories, each with **aspect-level sentiment** and a near-verbatim **evidence quote** from the review.
- **Complaint categories** (14-category taxonomy, multi-label).
- **Confidence values** (0–1) – the model's self-reported confidence, not a calibrated probability.
- **Validated JSON and CSV exports**, plus a six-page Streamlit dashboard for exploring results.

The LLM is treated as an unreliable component: every output is parsed, validated, checked against the review text, and then repaired or rejected.

## Major components (deliverables)

| | What is implemented |
|---|---|
| **D1 – Structured Review Extraction** | LLM extraction with LangChain `PydanticOutputParser` and Pydantic v2 schema validation; evidence grounding; malformed-output repair (local fixes, then up to two LLM repair calls) with controlled failures instead of crashes; JSON and CSV export. |
| **D2 – Prompt Engineering Experiments** | Zero-shot, Few-shot and Chain-of-Thought prompts compared on identical inputs (300 reviews, same model and settings). Evaluated against **300 blind human-created sentiment labels**: macro F1 (primary), accuracy, per-class precision/recall/F1, confusion matrices, plus latency and token efficiency. |
| **D3 – Product Summarisation & Faithfulness** | Extract → aggregate → summarise prompt chain: pandas aggregation per product, then product summaries whose points cite review ids. Faithfulness is checked **claim by claim** by an LLM-as-a-Judge with deterministic checks, tested with a negative control of deliberately planted errors. |

## Experimental results (measured, not claimed)

- **D2:** Few-shot performed best (macro F1 **0.352**, accuracy 0.473) vs zero-shot 0.285 and CoT 0.280. These scores are **modest** and show that sentiment classification with a 3B local model remains hard; few-shot used ~40% more input tokens and ~31% more latency than zero-shot. The human labels cover sentiment; only 8 of 300 include aspects, so aspect/complaint metrics are not reported against them.
- **D3:** On 123 planted errors the judge detected 96.7% (false-positive rate 9.8%). Its agreement with one human annotator on 239 summary claims was only 45.2% (Cohen's κ 0.06) – a single-annotator check, not ground truth.

## Value and outcome

ReviewIQ turns scattered customer feedback into **organised product intelligence**: sentiment at review and aspect level, recurring complaints, and evidence-backed product summaries. This makes it easier to see where customers are satisfied, which problems keep recurring, and where products can be improved – while reporting honestly where the model and its evaluation remain limited.
