# Aspect-Based Customer Review Analytics with Structured Outputs

An LLM pipeline that converts unstructured e-commerce product reviews into **validated structured records**
(overall sentiment, product aspects with aspect-level sentiment and evidence, complaint categories). It compares
**zero-shot, few-shot and Chain-of-Thought** prompting against human labels, and generates **per-product
summaries** through an *extract → aggregate → summarise* prompt chain that an **LLM-as-a-Judge** evaluates.

> Mini-project — *Jackfruit Problem, Problem Statement 2*.

> ⚠️ **Current status:** the full pipeline is implemented and tested offline. **No live LLM experiment
> has been run yet** because no API key is configured, and the **300 manual annotations are pending**.
> This repository therefore reports **no model performance numbers**. Everything you can run without a
> key uses a clearly labelled *offline mock* that is not a language model. See [§19](#19-sample-outputs) and [§20](#20-limitations-and-future-improvements).

---

## Contents
1. [Introduction](#1-introduction) · 2. [Problem statement](#2-problem-statement) · 3. [Objectives](#3-objectives) ·
4. [Technologies](#4-technologies-used) · 5. [Architecture](#5-system-architecture) · 6. [Folder structure](#6-project-folder-structure) ·
7. [Dataset](#7-dataset-and-data-preparation) · 8. [JSON schema](#8-structured-json-schema) · 9. [Prompting strategies](#9-prompt-engineering-strategies) ·
10. [Extraction pipeline](#10-extraction-pipeline) · 11. [Malformed output repair](#11-malformed-output-repair) · 12. [Aggregation & summarisation](#12-aggregation-and-summarisation) ·
13. [Evaluation methodology](#13-evaluation-methodology) · 14. [LLM-as-a-Judge](#14-llm-as-a-judge) · 15. [Installation](#15-installation) ·
16. [Configuration](#16-environment-configuration) · 17. [Running the app](#17-running-the-application) · 18. [Tests](#18-running-tests) ·
19. [Sample outputs](#19-sample-outputs) · 20. [Limitations & future work](#20-limitations-and-future-improvements)

---

## 1. Introduction
Online stores collect thousands of free-text reviews. Star ratings say *how much* customers liked a product,
but not *why*. Aspect-based sentiment analysis finds **what** customers talk about (noise, installation, fit,
durability…) and **how they feel** about each aspect. This project uses a large language model for that
analysis. It treats the LLM as an unreliable component: every output is validated against a schema, checked
against the review text, and repaired or rejected.

## 2. Problem statement
> *"Develop an LLM pipeline that converts unstructured product reviews into validated structured records
> containing sentiment, product aspects and complaint categories, and generates per-product summaries. The
> pipeline shall use public e-commerce review data and compare prompting techniques for sentiment analysis
> and multi-label classification."*

| Deliverable | Requirement | Where |
|---|---|---|
| **D1** | JSON schema + extraction pipeline (LangChain parsers / Pydantic), JSON & CSV export, automatic repair of malformed outputs | `schemas/`, `pipeline/extraction.py`, `pipeline/validation.py`, `pipeline/repair.py`, `utils/export.py` |
| **D2** | Compare zero-shot, few-shot and CoT prompts on 300 manually labelled reviews; report macro-F1 | `prompts/`, `annotation/`, `evaluation/classification_metrics.py`, `evaluation/prompt_comparison.py` |
| **D3** | Prompt chain extract → aggregate → summarise; per-product summaries; faithfulness via LLM-as-a-Judge | `pipeline/aggregation.py`, `pipeline/summarization.py`, `evaluation/summary_evaluation.py` |

## 3. Objectives
1. Define a strict, documented schema and taxonomy for review analysis.
2. Extract structured records reliably, with grounded evidence and automatic repair.
3. Compare three prompting techniques fairly (same model, data, schema, definitions).
4. Evaluate against human ground truth with standard single- and multi-label metrics.
5. Produce product summaries grounded in computed statistics and cited reviews.
6. Evaluate the summaries with a rubric-based LLM judge plus deterministic checks.
7. Provide an interactive dashboard and a reproducible, testable codebase.

## 4. Technologies used
| Purpose | Technology |
|---|---|
| Language | Python 3.11 |
| Structured output / validation | Pydantic v2 |
| Prompt templates, output parsers, model integration | LangChain (`langchain-core`, `langchain-google-genai`) |
| Default LLM provider | Google Gemini (`gemini-2.5-flash`, configurable). OpenAI, Anthropic, Groq and Ollama are supported |
| Data processing | pandas, NumPy |
| Metrics | scikit-learn |
| Frontend / charts | Streamlit, Plotly |
| Configuration | python-dotenv |
| Testing | pytest, `streamlit.testing.AppTest` |

## 5. System architecture
```
 reviews ─► [EXTRACT] prompt strategy ─► LLM ─► parse ─► validate (+grounding) ─► repair loop ─► ReviewRecord / controlled failure
                                                                                                  │
            [AGGREGATE] pandas: sentiment %, aspect stats, complaints, trends, evidence quotes ◄────┘
                                                                                                  │
            [SUMMARISE] LLM: grounded summary, every point cites review IDs (validated) ◄──────────┘
                                                                                                  │
            [JUDGE]     LLM-as-a-Judge (rubric 1–5) + deterministic checks ◄───────────────────────┘
```
* **Provider-independent LLM layer:** the pipeline talks only to an `LLMClient` interface (`llm/base.py`).
  `llm/factory.py` builds a LangChain chat model for the configured provider, or the offline `MockLLMClient`.
* **Live and mock are physically separated:** `outputs/live/` vs `outputs/mock/`, with an `is_mock` flag on every record.
  Live calls never silently fall back to mock.

Details: [`docs/architecture.md`](docs/architecture.md).

## 6. Project folder structure
```
├── app.py                      Streamlit entry point (st.navigation)
├── config.py                   all settings from .env (no secrets in code)
├── requirements.txt · .env.example · .gitignore
├── schemas/
│   ├── taxonomy.py             single source of truth for all labels + definitions
│   ├── review_schema.py        Aspect, LLMExtraction, ReviewRecord, metadata, grounding check
│   └── evaluation_schema.py    Annotation, ProductAggregate, ProductSummary, JudgeVerdict
├── llm/
│   ├── base.py                 LLMClient protocol, LLMResponse, errors
│   ├── factory.py              returns live or mock client
│   ├── langchain_client.py     Gemini / OpenAI / Anthropic / Groq / Ollama via LangChain
│   └── mock.py                 OFFLINE deterministic fixtures (development only)
├── prompts/
│   ├── common.py               shared rules, taxonomy block, PydanticOutputParser instructions
│   ├── zero_shot.py · few_shot.py · cot.py · strategies.py (common interface)
│   └── repair.py · summarization.py · judge.py
├── pipeline/
│   ├── validation.py           JSON extraction, local fixes, normalisation, error formatting
│   ├── repair.py               generic generate → validate → repair loop
│   ├── extraction.py           Stage 1 + resumable JSONL cache
│   ├── aggregation.py          Stage 2 (pandas)
│   └── summarization.py        Stage 3 + full chain runner
├── evaluation/
│   ├── classification_metrics.py   single-label, multi-label, aspect metrics
│   ├── prompt_comparison.py        experiment runner, reports
│   ├── summary_evaluation.py       LLM-as-a-Judge + deterministic checks
│   └── visualizations.py           Plotly figures
├── annotation/
│   ├── annotation_manager.py   sample, persistence, CSV import/export
│   └── annotation_guide.md
├── ui/                         the six dashboard pages + shared helpers
├── scripts/                    prepare_dataset · build_annotation_sample · run_experiment · run_chain
├── utils/                      data_loader · export · storage · logging_config
├── tests/                      87 tests (no network)
├── data/                       raw/ · processed/ · annotations/  (see data/README.md)
├── outputs/                    generated artefacts (git-ignored; live/ and mock/)
├── reports/                    curated final results (pending)
└── docs/                       architecture.md · experiment_methodology.md
```

## 7. Dataset and data preparation
* **Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), **Appliances** category,
  2.13 M raw reviews. It was chosen for concrete, comparable aspects (fit/compatibility, installation, noise, durability, price).
* **Pipeline** (`scripts/prepare_dataset.py`, deterministic with seed 42):
  1. clean the text (HTML removed, whitespace collapsed);
  2. keep reviews of 5–400 words;
  3. keep products with 200–5,000 valid reviews and a title;
  4. randomly pick 15 products;
  5. de-duplicate per user+product and per identical text;
  6. sample 100 reviews per product.

  Result: **1,500 reviews / 15 products**.
* `parent_asin` is kept as `product_id` for aggregation. The star rating is never shown to the LLM.
* **Annotation sample:** 300 reviews, stratified by rating (62/55/60/60/63 for 1★–5★), covering all 15 products.

Source, licence notes, fields and statistics: [`data/README.md`](data/README.md).

## 8. Structured JSON schema
Each record (`ReviewRecord`) contains:

| Field | Type / constraint |
|---|---|
| `review_id`, `product_id`, `review_text` | identifiers and source text |
| `overall_sentiment` | `positive` · `negative` · `neutral` · `mixed` |
| `aspects[]` | list (may be empty) of **Aspect** |
| `complaint_categories[]` | multi-label from the complaint taxonomy (empty if none) |
| `confidence` | float 0–1 |
| `rationale` | optional, ≤400 characters, a concise justification (no reasoning trace) |
| `extraction_metadata` | strategy, provider, model, prompt version, `is_mock`, validation success, repair attempts, local fixes, latency, tokens, **every attempt's raw output and errors** |

**Aspect:** `aspect_name` (normalised), `aspect_category` (13 categories), `sentiment` (`positive`/`negative`/`neutral`),
`evidence` (**must be a near-verbatim quote from the review**, otherwise validation fails), `confidence` 0–1.

**Aspect categories:** performance, quality_durability, design_build, ease_of_use, installation_setup, noise, size_capacity,
compatibility, price_value, customer_service, shipping_packaging, safety, other.

**Complaint categories:** product_defect, poor_durability, performance_issue, noise_issue, installation_difficulty, usability_issue,
compatibility_fit_issue, poor_value, shipping_damage, missing_parts, not_as_described, customer_service_issue, safety_concern, other.

Every label has a written definition in `schemas/taxonomy.py`. To extend the taxonomy, add an enum member and its definition;
a test checks that the two stay in sync. Extra fields are forbidden (`extra="forbid"`), and duplicate aspects and complaints are removed.

## 9. Prompt engineering strategies
All three strategies share the system role, the 9 extraction rules, the taxonomy definitions and the
LangChain `PydanticOutputParser` format instructions (`prompts/common.py`). They differ **only** in technique:

| Strategy | What is added |
|---|---|
| **Zero-shot** | nothing: instructions + definitions + review |
| **Few-shot** | 5 fixed **synthetic** labelled examples written by the author (positive multi-aspect, negative multi-complaint, mixed, compatibility issue, neutral/no aspects). Not taken from the dataset, so they cannot leak into the test set |
| **Chain-of-Thought** | a 7-step analysis procedure (split statements → identify aspects → map categories → decide sentiment with negation → check complaints → weigh overall → verify evidence). The model performs it *silently* and returns only the JSON + concise rationale |

`prompts/strategies.py` exposes a common `PromptStrategy.build_messages(review_text)` interface, so the pipeline can switch strategies without code changes.

## 10. Extraction pipeline
`ReviewExtractor` (`pipeline/extraction.py`) builds the messages, calls `generate_structured(...)`, and returns either a
`ReviewRecord` or a controlled `ExtractionFailure`. `extract_many` processes batches with a progress callback and an
`ExtractionStore` (JSONL) cache keyed by review ID + text hash:
* **resumable:** an interrupted run continues where it stopped;
* **no repeated API calls:** successful reviews are skipped;
* **failed reviews are retried** on the next run.

Exports (`utils/export.py`): JSON (validated by re-parsing), records CSV with a **fixed column list**, a long-format
aspects CSV and a failures CSV.

## 11. Malformed output repair
LLMs can return invalid JSON, prose around the JSON, wrong enum values, wrong types, missing fields, incomplete aspects
or invented evidence. `pipeline/repair.py` handles all of these:

1. **Capture** the raw response.
2. **Extract** the JSON object: strip fences and prose with a string-aware brace scanner, and detect truncation.
3. **Local fixes** that need no LLM call (trailing commas, smart quotes). Each fix is logged.
4. **Normalise** formatting only (`"Positive"` → `"positive"`, unwrap `{"result": {...}}`). Invalid meanings such as
   `"very good"` are *not* guessed.
5. **Validate** with Pydantic, including evidence grounding against the review text.
6. If invalid, send a **repair prompt** containing the original output, the schema, the source text and the exact errors.
7. **Retry** up to `MAX_REPAIR_ATTEMPTS` (default 2).
8. **Log** every attempt (raw output, errors, local fixes, tokens, latency) in `extraction_metadata.attempts`.
9. If all attempts fail, return a **controlled failure**. Invalid data is never stored as a record.

The same loop validates summaries (cited review IDs must exist) and judge verdicts (scores 1–5). Tests cover every
failure type listed above (`tests/test_validation_and_repair.py`).

## 12. Aggregation and summarisation
**Aggregate** (`pipeline/aggregation.py`, pure pandas, no LLM): totals; positive/negative/neutral/mixed counts and
percentages; per aspect category its mentions, positive/negative/neutral counts, net sentiment and top aspect names;
complaint counts and percentages; yearly aspect-sentiment trend; and diverse evidence quotes with review IDs.

**Summarise** (`pipeline/summarization.py`): the LLM receives only the statistics and the evidence quotes. It returns
`ProductSummaryContent`:
* overall opinion;
* what customers appreciate;
* common complaints;
* frequent aspects;
* improvement areas;
* recommendations.

Each appreciated or complaint point **must cite review IDs from the evidence**, and unknown IDs fail validation and
trigger repair. The prompt forbids calculating or inventing numbers, and a deterministic check later flags any
percentage that does not appear in the statistics.

## 13. Evaluation methodology
Ground truth = **300 human-labelled reviews** (`annotation/`, Data Annotation page). Metrics (`evaluation/classification_metrics.py`):

| Task | Metrics |
|---|---|
| Overall sentiment (4 classes) | accuracy, **macro P/R/F1**, weighted F1, per-class report, confusion matrix |
| Complaints (multi-label) | micro/macro/weighted/samples F1, Hamming loss, subset accuracy, per-label report |
| Aspects | category-level **detection** F1 · **aspect sentiment** macro-F1 on matched categories · strict **joint** (category:sentiment) F1 |
| Operational | validity rate, first-attempt validity, LLM repairs, latency, input/output tokens |

Failed extractions stay in the denominator and count as wrong. Each run is labelled **awaiting annotations**,
**provisional** (<300 labels), **complete** or **mock demo**. Full details, including why the sample is
rating-stratified: [`docs/experiment_methodology.md`](docs/experiment_methodology.md).

## 14. LLM-as-a-Judge
`evaluation/summary_evaluation.py` sends the judge four inputs:
* the original reviews;
* the per-review structured extractions;
* the aggregated statistics;
* the generated summary.

The judge applies an anchored 1–5 rubric for **faithfulness, coverage, consistency and hallucination**
(5 = no hallucination) and returns a Pydantic `JudgeVerdict` with justifications and a verbatim list of
unsupported claims. Each judgement stores its model, prompt version and repair count, plus
**deterministic checks**: invented percentages, coverage of the top aspects and complaints, and cited-ID validity.
An exportable **manual inspection sample** shows the evidence, statistics, summary and scores side by side with a
"human agrees?" column.

> Judge scores are model-based assessments, **not objective human evaluation**. To reduce self-preference bias,
> set `JUDGE_PROVIDER`/`JUDGE_MODEL` to a different model.

## 15. Installation
```bash
git clone https://github.com/RachitSMane/Prompt-engineering-mini-project-.git
cd Prompt-engineering-mini-project-
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py          # downloads ~340 MB once, builds data/processed/reviews.csv
python scripts/build_annotation_sample.py  # keeps the committed 300-review sample
```

## 16. Environment configuration
```bash
cp .env.example .env      # Windows: copy .env.example .env
```
| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` · `openai` · `anthropic` · `groq` · `ollama` · `mock` |
| `LLM_MODEL` | `gemini-2.5-flash` | model name for the provider |
| `GOOGLE_API_KEY` | – | free key from https://aistudio.google.com/apikey |
| `LLM_TEMPERATURE` | `0` | deterministic decoding for experiments |
| `LLM_REQUESTS_PER_MINUTE` | `10` | client-side rate limit (free tier) |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | same as extraction | optional separate judge model |
| `MAX_REPAIR_ATTEMPTS` | `2` | LLM repair retries |
| `STRICT_EVIDENCE_GROUNDING` | `true` | require verbatim evidence |
| `MOCK_MALFORMED_RATE` | `15` | % of mock outputs deliberately corrupted |

For other providers, `pip install langchain-openai` (or `-anthropic`, `-groq`, `-ollama`) and set the matching key.
**Never commit `.env`**; it is git-ignored.

**Without an API key**, the app starts in *Offline MOCK mode* and shows a clear message wherever a live LLM is needed.
Mock outputs come from a keyword fixture, are saved under `outputs/mock/`, and carry banners. They are **never results**.

## 17. Running the application
```bash
streamlit run app.py
```
| Page | What you can do |
|---|---|
| 🏠 Home / Dashboard | deliverable status, dataset statistics, processed reviews, sentiment & aspect charts, recent activity |
| 🔎 Review Analysis | use the dataset or upload a CSV (column mapping), run extraction with any strategy, inspect JSON + repair logs, download JSON/CSV |
| ⚖️ Prompt Comparison | run/resume experiments, macro-F1 and other metrics, confusion matrices, latency & tokens, download report |
| 📦 Product Insights | extract per product, aggregated statistics, positive/negative aspects, complaints, trends, generate grounded summaries |
| 🧑‍⚖️ Summary Evaluation | run the judge, scores per dimension, justifications, deterministic checks, inspection export |
| ✍️ Data Annotation | label the 300 reviews (guide, progress, navigation), import/export CSV & JSON |

Long runs are more robust from the command line:
```bash
python scripts/run_experiment.py --limit 20          # pilot
python scripts/run_experiment.py                     # all strategies × 300 reviews (~900 calls)
python scripts/run_experiment.py --resume <run_id>   # continue after interruption
python scripts/run_experiment.py --reevaluate <id>   # recompute metrics after more annotations (no LLM calls)
python scripts/run_chain.py --per-product 30 --judge # extract → aggregate → summarise → judge
# add --mock to any of these for an offline pipeline demo (not a result)
```

## 18. Running tests
```bash
python -m pytest -q
```
87 tests run offline (scripted fake LLM + mock client; no API calls). They cover:
* schema validation, enums, missing and extra fields, multiple aspects, mixed sentiment, no-aspect reviews and evidence grounding;
* JSON parsing and every malformed-output type, the repair loop, controlled failures and provider errors;
* caching and resume, CSV/JSON export columns;
* aggregation numbers and metric calculations checked against hand-computed values;
* summary citation validation, the judge, and annotation persistence, import and export;
* headless rendering and interaction tests for all six Streamlit pages.

## 19. Sample outputs
> The examples below were produced in **MOCK mode** to show the *format* and the *repair mechanism*.
> They are not LLM outputs and not experimental results.

**Repair in action:** a deliberately corrupted first response, rejected by validation and fixed by the repair step:
```text
review:   "Fit our dishwasher perfectly as indicated by the model numbers. Great product."
attempt 0 (initial)  → {"overall_sentiment": "very good", "aspects": [...], ...}
          errors     → overall_sentiment: Input should be 'positive', 'negative', 'neutral' or 'mixed' (got 'very_good')
attempt 1 (llm_repair) → valid ✅   (record stores both attempts in extraction_metadata.attempts)
```
On a 600-review mock run with 15% injected corruption: 46 records needed an LLM repair, 32 were fixed by local
deterministic fixes, and 0 failed. This shows the mechanism works; it says nothing about a real model's error rate.

**Structured record (abridged):**
```json
{
  "review_id": "Rd77288b05bc2", "product_id": "B001DHKBXU",
  "review_text": "Easy to install and works great!",
  "overall_sentiment": "positive",
  "aspects": [
    {"aspect_name": "install", "aspect_category": "installation_setup", "sentiment": "positive",
     "evidence": "Easy to install and works great!", "confidence": 0.6}
  ],
  "complaint_categories": [], "confidence": 0.5,
  "extraction_metadata": {"strategy": "few_shot", "provider": "mock", "model_name": "mock-offline-v1",
    "is_mock": true, "validation_success": true, "repair_attempts": 0, "attempts": ["..."]}
}
```
Real results will be added to `reports/` after the live runs (see the checklist below).

## 20. Limitations and future improvements
**Pending work (required for D2/D3 results):**
- [ ] Add a Gemini API key to `.env`.
- [ ] Complete the 300 manual annotations (Data Annotation page, roughly 2–3 hours).
- [ ] Run `scripts/run_experiment.py` (all strategies, 300 reviews) and copy `report.md` to `reports/`.
- [ ] Run `scripts/run_chain.py --judge` and fill in the human-agreement column of the inspection sample.

**Limitations:**
- One annotator → no inter-annotator agreement yet (the guide describes an optional 50-review kappa check).
- Aspect matching is at category level; aspect-name matching (e.g. with embeddings) is future work.
- The rating-stratified evaluation set does not reflect the natural (72% 5★) distribution; macro-F1 is the headline metric for that reason.
- Evidence grounding uses fuzzy string matching (85% similarity). Paraphrased evidence is rejected, which can raise repair counts.
- LLM-as-a-Judge shares the biases of LLMs; a different judge model and human spot-checks are recommended.
- Free-tier rate limits make the full experiment take roughly 1.5 hours; the runs are resumable.

**Future improvements:** confidence calibration analysis, dynamic few-shot example selection (outside the test set),
a second annotator, multilingual reviews, async batch calls for paid tiers.
