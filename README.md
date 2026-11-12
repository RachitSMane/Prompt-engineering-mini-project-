# ReviewIQ — Aspect-Based Customer Review Analytics with Structured Outputs

ReviewIQ turns unstructured e-commerce product reviews into **validated structured records** (overall sentiment,
product aspects with aspect-level sentiment and verbatim evidence, complaint categories). It compares
**zero-shot, few-shot and Chain-of-Thought** prompting on a 300-review evaluation sample, and produces
**per-product summaries** through an *extract → aggregate → summarise* prompt chain that a separate
**LLM-as-a-Judge** evaluates. All inference runs **locally with Ollama**: no API key, no cloud service, and no
review text leaves the machine.

> **PES University · Prompt Engineering Mini Project · 5th Semester AIML**
> Team: Rachit S Mane (PES1UG24AM214) · P. Chinni Krishna Reddy (PES1UG24AM209) · Pranav V. Menon (PES1UG24AM198) · Akash R. (PES1UG24AM212)
> Problem statement: *Jackfruit Problem, Problem Statement 2* — see [`docs/ReviewIQ_Problem_Statement_and_Solution.pdf`](docs/ReviewIQ_Problem_Statement_and_Solution.pdf).

<!-- RESULTS-STATUS -->

---

## Contents
1. [Overview](#1-overview) · 2. [Problem statement](#2-problem-statement) · 3. [Objectives](#3-objectives) · 4. [Features](#4-features) ·
5. [Technology stack](#5-technology-stack) · 6. [Architecture](#6-architecture) · 7. [Dataset](#7-dataset) · 8. [Annotation methodology](#8-annotation-methodology) ·
9. [Structured extraction pipeline](#9-structured-extraction-pipeline) · 10. [Pydantic schema](#10-pydantic-schema) · 11. [Prompting strategies](#11-prompting-strategies) ·
12. [Evaluation methodology](#12-evaluation-methodology) · 13. [Experimental results](#13-experimental-results) · 14. [Prompt chain](#14-prompt-chain-extract--aggregate--summarise) ·
15. [Aggregation](#15-aggregation-methodology) · 16. [LLM-as-a-Judge](#16-llm-as-a-judge-methodology) · 17. [Installation](#17-installation) · 18. [Ollama setup](#18-ollama-setup) ·
19. [Running the application](#19-running-the-application) · 20. [Running experiments](#20-running-experiments) · 21. [Reproducing the results](#21-reproducing-the-results) ·
22. [Limitations](#22-limitations) · 23. [Future improvements](#23-future-improvements) · 24. [Team and academic information](#24-team-and-academic-information)

---

## 1. Overview
Online stores collect thousands of free-text reviews. Star ratings say *how much* customers liked a product, but not
*why*. Aspect-based sentiment analysis finds **what** customers talk about (fit, installation, noise, durability, price…)
and **how they feel** about each aspect. ReviewIQ uses a local large language model for this analysis and treats it as an
unreliable component: every output is parsed, validated against a Pydantic schema, checked against the review text, and
repaired or rejected. A Streamlit dashboard with six pages exposes the whole pipeline.

## 2. Problem statement
> *"Develop an LLM pipeline that converts unstructured product reviews into validated structured records containing
> sentiment, product aspects and complaint categories, and generates per-product summaries. The pipeline shall use
> public e-commerce review data and compare prompting techniques for sentiment analysis and multi-label classification."*

| Deliverable | Requirement | Implementation |
|---|---|---|
| **D1** | JSON schema + extraction pipeline (LangChain / Pydantic), JSON & CSV export, automatic repair of malformed outputs | `schemas/`, `pipeline/extraction.py`, `pipeline/validation.py`, `pipeline/repair.py`, `utils/export.py`, `scripts/export_d1.py` |
| **D2** | Compare zero-shot, few-shot and CoT prompts on the 300-review labelled sample; report macro-F1 | `prompts/`, `annotation/`, `evaluation/classification_metrics.py`, `evaluation/prompt_comparison.py`, `scripts/run_experiment.py` |
| **D3** | Prompt chain extract → aggregate → summarise; per-product summaries; faithfulness via LLM-as-a-Judge | `pipeline/aggregation.py`, `pipeline/summarization.py`, `evaluation/summary_evaluation.py`, `scripts/run_chain.py`, `scripts/export_d3.py` |

## 3. Objectives
1. Define a strict, documented schema and taxonomy for review analysis.
2. Extract structured records reliably, with grounded evidence and automatic repair.
3. Compare three prompting techniques fairly (same model, data, schema, definitions and decoding).
4. Evaluate against reference labels with standard single- and multi-label metrics, reporting coverage and failures.
5. Produce product summaries grounded in computed statistics and cited reviews.
6. Evaluate the summaries with a rubric-based LLM judge plus deterministic checks, and state the judge's limits.
7. Run entirely on local hardware, with a reproducible, tested codebase and an interactive dashboard.

## 4. Features
* **Validated extraction** with Pydantic v2: enums, field limits, `extra="forbid"`, and evidence that must be a near-verbatim quote of the review.
* **Automatic repair**: JSON extraction from prose/fences, deterministic local fixes, then up to two LLM repair calls with the exact validation errors; every attempt is logged.
* **Failure safety**: a malformed output becomes a controlled failure (never a crash); a provider outage stops the batch cleanly, keeps every completed result cached, and marks the experiment **Incomplete** until resumed.
* **Three prompting strategies** behind one interface, compared on identical inputs.
* **Metrics**: macro-F1 (primary), accuracy, per-class precision/recall/F1, confusion matrices, multi-label complaint metrics, aspect detection/sentiment metrics, validity, repair, coverage, latency and token usage.
* **Prompt chain** with pandas aggregation, citation-validated summaries and an LLM-as-a-Judge with deterministic cross-checks.
* **Local-only inference** through Ollama (`127.0.0.1`), with non-local endpoints rejected by default; optional cloud providers remain available but are never selected by default.
* **Six-page Streamlit dashboard** with live/mock separation, exports and an annotation tool.

## 5. Technology stack
| Purpose | Technology |
|---|---|
| Language | Python 3.11 |
| Local LLM runtime | Ollama 0.40 — `qwen2.5:3b` (extraction, summaries), `llama3.2` (judge), `qwen2.5:7b` (model comparison) |
| LLM integration, prompts, parsers | LangChain (`langchain-core`, `langchain-ollama==1.1.0`) |
| Structured output / validation | Pydantic v2 |
| Data processing / metrics | pandas, NumPy, scikit-learn |
| Dashboard / charts | Streamlit, Plotly |
| Configuration | python-dotenv (`.env`, git-ignored) |
| Testing | pytest, `streamlit.testing.AppTest` |

## 6. Architecture
```
 reviews ─► [EXTRACT] prompt strategy ─► local LLM (Ollama) ─► parse ─► validate (+grounding) ─► repair loop ─► ReviewRecord / controlled failure
                                                                                                       │
            [AGGREGATE] pandas: sentiment %, aspect stats, complaints, trends, evidence quotes ◄─────────┘
                                                                                                       │
            [SUMMARISE] local LLM: grounded summary; every point cites review ids (validated) ◄─────────┘
                                                                                                       │
            [JUDGE]     separate local LLM: rubric 1–5 + deterministic checks ◄─────────────────────────┘
```
* **Provider-independent LLM layer:** the pipeline talks only to the `LLMClient` interface (`llm/base.py`); `llm/factory.py`
  builds a LangChain chat model for the configured provider or the offline `MockLLMClient`. The Ollama model gets its own
  context window (`OLLAMA_NUM_CTX`), timeout and output limit, and bypasses the cloud rate limiter.
* **Live and mock are physically separated:** `outputs/live/` vs `outputs/mock/`, with an `is_mock` flag on every record;
  live calls never fall back to mock.
* **Resumable everywhere:** extraction results are cached in JSONL keyed by review id + text hash; experiments resume by run id.

Details: [`docs/architecture.md`](docs/architecture.md).

```
├── app.py                      Streamlit entry point (st.navigation, six pages)
├── config.py                   settings from .env (local-first defaults, no secrets in code)
├── schemas/                    taxonomy · review_schema (ReviewRecord, grounding) · evaluation_schema
├── llm/                        base · factory · langchain_client (Ollama + optional cloud) · mock (offline fixtures)
├── prompts/                    common · zero_shot · few_shot · cot · strategies · repair · summarization · judge
├── pipeline/                   validation · repair · extraction · aggregation · summarization
├── evaluation/                 classification_metrics · prompt_comparison · summary_evaluation · visualizations
├── annotation/                 annotation_manager (sample, persistence, provenance) · annotation_guide.md
├── ui/                         the six dashboard pages + shared helpers and theme
├── scripts/                    prepare_dataset · build_annotation_sample · run_experiment · run_chain · export_d1 · export_d3 · build_reports
├── tests/                      offline test suite (no network, no LLM)
├── data/                       raw/ · processed/ (regenerated, git-ignored) · annotations/ (committed)
├── outputs/                    generated artefacts (git-ignored; live/ and mock/)
├── reports/                    curated final results, final report, presentation, screenshots
└── docs/                       architecture · experiment methodology · problem statement PDF
```

## 7. Dataset
* **Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), **Appliances** category
  (2,128,605 raw reviews). Citation and licence notes: [`data/README.md`](data/README.md).
* **Preparation** (`scripts/prepare_dataset.py`, deterministic, seed 42): clean text; keep reviews of 5–400 words; keep products
  with 200–5,000 valid reviews and a title (1,373 candidates); randomly pick 15 products; de-duplicate per user+product and per text;
  sample 100 reviews per product.
* **Verified result (regenerated for this submission):** **1,500 reviews, 15 products × 100, 1,500 unique review ids**; ratings
  1★ 150 · 2★ 55 · 3★ 64 · 4★ 145 · 5★ 1,086; review length 5–391 words (median 22); one empty review title, no other missing values.
* The star rating is used only for sampling and display and is **never shown to the LLM or used as ground truth**.
* **Evaluation sample:** 300 reviews stratified by rating (≈60 per rating), covering all 15 products; all 300 ids and texts match the dataset exactly.

## 8. Annotation methodology
The reference labels are in `data/annotations/annotations.json` (300 records, schema `Annotation`), following
[`annotation/annotation_guide.md`](annotation/annotation_guide.md): overall sentiment (4 classes), aspect categories with
sentiment, and complaint categories, labelled from the review text without the star rating.

**Provenance (stated honestly everywhere the labels are used):**

| | Records |
|---|---|
| Human label (annotator Akash) | **1** (`R3314d62a5df6`) |
| AI-assisted labels — generated by an AI model (Claude) from the review text only, with the project mentor's approval; 77 marked `DEBATABLE` | **299** |
| Flagged as unreadable and excluded from evaluation | 1 (`R5cfb12ef75b4`) |
| **Evaluable reference labels** | **299** (1 human + 298 AI-assisted) |

The split is recorded in `meta.annotation_method` and in each AI-assisted record's `notes` (prefix `AI-assisted label`).
The dashboard and every report show it next to the metrics. Because most labels are AI-assisted, **D2 scores measure agreement
with these reference labels, not with independent manual annotation.** No output of the evaluated models is ever written into the
label file. Re-labelling a review on the *Human Annotation* page replaces it with a human label.

## 9. Structured extraction pipeline
`ReviewExtractor` (`pipeline/extraction.py`) builds the strategy's messages, calls `generate_structured(...)` and returns a
`ReviewRecord` or a controlled `ExtractionFailure`. `extract_many` processes batches with a JSONL cache:
* **resumable** — an interrupted run continues where it stopped; successful reviews are never called again;
* **failed outputs are retried** on the next run;
* **provider errors stop the batch** (`ExtractionStopped`) instead of being stored as model failures, so an outage never looks like a wrong answer and completed results are kept.

**Repair loop** (`pipeline/repair.py`): capture the raw response → extract the JSON object (fences/prose stripped, truncation
detected) → deterministic local fixes (trailing commas, smart quotes; each logged) → normalise formatting only (`"Positive"` →
`"positive"`) → validate with Pydantic including evidence grounding → if invalid, send a repair prompt with the original output,
the schema, the review and the exact errors → retry up to `MAX_REPAIR_ATTEMPTS` (2) → otherwise a controlled failure.
Every attempt (raw output, errors, local fixes, tokens, latency) is stored in `extraction_metadata.attempts`.

**Exports** (`utils/export.py`, `scripts/export_d1.py`): records JSON (re-parsed and re-validated with evidence grounding after
writing), records CSV with a fixed column list, a long-format aspects CSV, a failures CSV, and a validation/repair/runtime report.

## 10. Pydantic schema
| Field | Type / constraint |
|---|---|
| `review_id`, `product_id`, `review_text` | identifiers and source text |
| `overall_sentiment` | `positive` · `negative` · `neutral` · `mixed` |
| `aspects[]` | list (may be empty) of **Aspect**: `aspect_name`, `aspect_category` (13), `sentiment` (positive/negative/neutral), `evidence` (near-verbatim quote, ≤300 chars), `confidence` 0–1 |
| `complaint_categories[]` | multi-label from 14 complaint categories (empty if none) |
| `confidence` | float 0–1 |
| `rationale` | optional, ≤400 characters — a concise justification, never a reasoning trace |
| `extraction_metadata` | strategy, provider, model, prompt version, `is_mock`, validation success, repair attempts, local fixes, latency, tokens, all attempts |

**Aspect categories:** performance, quality_durability, design_build, ease_of_use, installation_setup, noise, size_capacity,
compatibility, price_value, customer_service, shipping_packaging, safety, other.
**Complaint categories:** product_defect, poor_durability, performance_issue, noise_issue, installation_difficulty, usability_issue,
compatibility_fit_issue, poor_value, shipping_damage, missing_parts, not_as_described, customer_service_issue, safety_concern, other.
Every label has a written definition in `schemas/taxonomy.py`; a test keeps enums and definitions in sync.

## 11. Prompting strategies
All three share the system role, the extraction rules, the taxonomy definitions and LangChain `PydanticOutputParser` format
instructions (`prompts/common.py`), and return the **same schema**. Only the technique differs:

| Strategy | What is added |
|---|---|
| **Zero-shot** | nothing: instructions + definitions + review |
| **Few-shot** | 5 fixed **synthetic** demonstrations written by the authors (positive multi-aspect, negative multi-complaint, mixed, compatibility issue, neutral/no aspects). Not taken from the dataset, so they cannot leak into the evaluation set |
| **Chain-of-Thought** | a 7-step analysis procedure (split statements → identify aspects → map categories → decide sentiment incl. negation → check complaints → weigh overall → verify evidence), performed **silently**; only the JSON and a short rationale are returned and stored |

## 12. Evaluation methodology
* **Evaluation set:** the 299 evaluable reviews of the sample (one flagged review excluded and reported). The same review ids,
  order, model, temperature (0), output limit and repair limit are used for all three strategies; `config.json` records them.
* **Overall sentiment (primary task):** accuracy, **macro precision/recall/F1 (primary metric)**, weighted F1, per-class report,
  confusion matrix. Macro averages use every class present in the reference labels or the predictions — all four classes occur in
  the reference labels, so a class the model never predicts still counts with F1 = 0.
* **Complaints (multi-label):** micro/macro/weighted/sample F1, Hamming loss, subset accuracy. **Aspects:** category-level detection
  F1, aspect-sentiment macro-F1 on matched categories, strict joint (category:sentiment) F1.
* **Failures are not dropped:** a review without a valid output stays in the denominator and counts as wrong; coverage, validity,
  repair rate and failed-review counts are reported per strategy, plus latency and tokens.
* **Status labels:** `complete` only when every strategy predicted every review against all evaluable labels; `incomplete` after a
  stopped run; `provisional` with too few labels; `mock_demo` for offline fixtures.

Full details: [`docs/experiment_methodology.md`](docs/experiment_methodology.md).

## 13. Experimental results
<!-- RESULTS -->
_Results are added here from `reports/` once the local runs finish._
<!-- /RESULTS -->

## 14. Prompt chain (extract → aggregate → summarise)
1. **Extract** structured records for each product's reviews (cached, same pipeline as D1).
2. **Aggregate** with pandas (no LLM) — see §15.
3. **Summarise:** the LLM receives only the statistics and evidence quotes and returns `ProductSummaryContent` (overall opinion,
   appreciated points, complaints, frequent aspects, improvement areas, recommendations). Every appreciated/complaint point must cite
   review ids from the evidence; unknown ids fail validation and trigger repair. The prompt forbids calculating or inventing numbers.
4. **Judge** each summary with a separate local model — see §16.

Summaries keep the full aggregate (statistics, evidence quotes, review ids) so every claim can be traced back to reviews.
Generated summaries are **model output**, clearly separated from the reference labels and never treated as ground truth.

## 15. Aggregation methodology
`pipeline/aggregation.py` computes, per product: totals; positive/negative/neutral/mixed counts and percentages; per aspect category
the mentions, positive/negative/neutral counts, net sentiment ((pos − neg) / mentions) and the most frequent aspect names; complaint
counts and percentages and the number of reviews with complaints; a yearly aspect-sentiment trend; and diverse positive and negative
evidence quotes with their review ids. All numbers are computed here, never by the LLM.

## 16. LLM-as-a-Judge methodology
The judge (`llama3.2`, a different model family from the `qwen2.5` summariser, to reduce self-preference) receives the original reviews
(80 words each), the per-review extractions, the aggregated statistics and the summary. It applies an anchored 1–5 rubric for
**faithfulness, coverage, consistency and hallucination** (5 = no unsupported claims) and returns a validated `JudgeVerdict` with
justifications and a verbatim list of unsupported claims. Deterministic checks run alongside: percentages absent from the statistics,
coverage of the top aspects and complaints, validity of cited ids, and (in `scripts/export_d3.py`) a **citation-polarity check** —
reviews cited for a praised point should contain a positive aspect, reviews cited for a complaint a negative one.

> Judge scores are an automated model assessment, **not objective truth**. A 3B judge can be lenient, miss unsupported claims and
> misread statistics; its scores are reported next to the deterministic checks so they can be audited.

## 17. Installation
```bash
git clone https://github.com/RachitSMane/Prompt-engineering-mini-project-.git
cd Prompt-engineering-mini-project-
python -m venv .venv                      # Python 3.11 recommended
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py         # downloads ~340 MB once, builds data/processed/reviews.csv
copy .env.example .env                    # macOS/Linux: cp .env.example .env  (no API key needed)
```
Cloud providers (Gemini, OpenAI, Anthropic, Groq) are optional: install the matching LangChain package listed in
`requirements.txt` and set its key in `.env`. They are never used by default.

## 18. Ollama setup
1. Install Ollama from <https://ollama.com> and start it (the desktop app, or `ollama serve`).
2. Pull the models (once):
   ```bash
   ollama pull qwen2.5:3b     # extraction and summaries (~1.9 GB)
   ollama pull llama3.2       # LLM-as-a-Judge (~2.0 GB)
   ollama pull qwen2.5:7b     # optional, model comparison (~4.7 GB)
   ```
3. Check: `ollama list` and `curl http://127.0.0.1:11434/api/version`.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `qwen2.5:3b` | extraction and summary model (also the default when no `.env` exists) |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | `ollama` / `llama3.2` | separate judge model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | local endpoint; non-local addresses are rejected unless `OLLAMA_ALLOW_REMOTE=true` |
| `OLLAMA_NUM_CTX` | `8192` | context window (fits the few-shot prompt plus output) |
| `LLM_TIMEOUT_SECONDS` | `300` | per call; local CPU/GPU inference is slow |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` in `.env.example` | output cap; stops runaway generation (see §22) |
| `LLM_TEMPERATURE` | `0` | deterministic decoding |
| `MAX_REPAIR_ATTEMPTS` / `STRICT_EVIDENCE_GROUNDING` | `2` / `true` | repair loop and evidence check |

If Ollama is not running or a model is missing, the sidebar shows **Ollama not running** / **Model not pulled**, live buttons are
disabled with instructions, and the app never switches to a cloud provider. *Offline mock mode* (sidebar toggle) runs the pipeline on
deterministic fixtures, stored separately and always labelled as **not results**.

## 19. Running the application
```bash
streamlit run app.py          # http://localhost:8501
```
| Page | What it shows |
|---|---|
| **Overview** | dataset size, deliverable status, processed reviews, sentiment and aspect charts, recent activity |
| **Review Analysis** | run extraction on the dataset or an uploaded CSV with any strategy; JSON + repair logs; JSON/CSV download |
| **Prompt Experiments** | run/resume experiments; macro-F1 and all metrics, confusion matrices, per-class scores, latency, tokens; label provenance |
| **Product Intelligence** | per-product statistics, aspects, complaints, trends, evidence and generated summaries |
| **Summary Evaluation** | judge scores per dimension and per product, justifications, deterministic checks, inspection export |
| **Human Annotation** | browse/edit the 300 reference labels with provenance badges; import/export CSV and JSON |

## 20. Running experiments
```bash
python scripts/run_experiment.py --only-annotated            # D2: 3 strategies × 299 evaluable reviews
python scripts/run_experiment.py --resume <run_id> --only-annotated   # continue after an interruption
python scripts/run_experiment.py --reevaluate <run_id>       # recompute metrics from saved predictions (no LLM calls)
python scripts/export_d1.py --run <run_id>                   # D1: JSON/CSV exports + validation report
python scripts/run_chain.py --per-product 20 --extract-only  # D3 stage 1 (cached)
python scripts/run_chain.py --per-product 20 --judge         # D3 summaries + judge (re-uses cached extractions)
python scripts/export_d3.py --source dataset__<strategy>__qwen2.5-3b__v1
python scripts/build_reports.py --d2-run <run_id> --d3-source <source>   # curate into reports/ (+ recomputation check)
```
Add `--mock` to the run scripts for an offline pipeline demo (never a result).

## 21. Reproducing the results
The exact commands, settings, timings and decisions of the reported runs are in
[`reports/execution_log.md`](reports/execution_log.md). Every experiment directory stores `config.json` (settings snapshot, prompt
version, review ids, git commit) and per-strategy predictions with raw outputs and repair logs, so all metrics can be recomputed
without the LLM (`--reevaluate`, `scripts/build_reports.py` — see `reports/verification.md`). Decoding is deterministic
(temperature 0); re-running the same model on the same hardware reproduced the earlier smoke-test output exactly.

```bash
python -m pytest -q          # offline test suite (no network, no LLM)
```

## 22. Limitations
* **Reference labels are mostly AI-assisted** (298 of 299 evaluable); scores measure agreement with them, and there is no inter-annotator agreement.
* **Small local models:** `qwen2.5:3b` runs fully on a 4 GB GPU but produces shallow extractions (missed aspects/complaints); `qwen2.5:7b`
  is more accurate on spot checks but only partly fits in VRAM and is ~3–4× slower, so it was evaluated on a subset only.
* **Runaway generation:** the 3B model occasionally loops (repeating list items). The 1,024-token output cap truncates such outputs, which then
  go through the repair loop; some still fail validation and are counted as failures.
* Aspect matching is at category level; the rating-stratified sample does not reflect the natural (72% 5★) distribution, so accuracy describes the balanced set and macro-F1 is the headline metric.
* Evidence grounding uses fuzzy string matching; paraphrased evidence is rejected.
* LLM-as-a-Judge scores are model opinions from a 3B judge, not calibrated against human ratings.
* Product summaries use 20 reviews per product (runtime budget on local hardware), not all 100.

## 23. Future improvements
Human re-annotation of the reference labels (at least a double-annotated subset with Cohen's kappa); constrained JSON decoding
(Ollama structured outputs) to remove syntax failures; a larger local model or GPU for the full 7B comparison; aspect-name matching with
embeddings; dynamic few-shot example selection from a separate development pool; confidence calibration; human calibration of the judge.

## 24. Team and academic information
**PES University — Prompt Engineering Mini Project, 5th Semester, AIML**

| # | Name | SRN |
|---|---|---|
| 1 | Rachit S Mane | PES1UG24AM214 |
| 2 | P. Chinni Krishna Reddy | PES1UG24AM209 |
| 3 | Pranav V. Menon | PES1UG24AM198 |
| 4 | Akash R. | PES1UG24AM212 |

Problem statement and proposed solution: [`docs/ReviewIQ_Problem_Statement_and_Solution.pdf`](docs/ReviewIQ_Problem_Statement_and_Solution.pdf).
Dataset: Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952 (Amazon Reviews 2023).
