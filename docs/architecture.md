# System Architecture

## Layered design

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Streamlit UI (app.py, ui/*)            CLI scripts (scripts/*)           │
├──────────────────────────────────────────────────────────────────────────┤
│ evaluation/   classification_metrics · prompt_comparison ·               │
│               summary_evaluation (LLM-as-a-Judge) · visualizations       │
│ annotation/   annotation_manager (human ground truth)                    │
├──────────────────────────────────────────────────────────────────────────┤
│ pipeline/     extraction ──► aggregation ──► summarization                │
│               validation + repair (generic, used by all LLM stages)      │
├──────────────────────────────────────────────────────────────────────────┤
│ prompts/      zero_shot · few_shot · cot · repair · summarization · judge │
│ schemas/      taxonomy (labels) · review_schema · evaluation_schema       │
├──────────────────────────────────────────────────────────────────────────┤
│ llm/          LLMClient interface ─┬─ LangChainLLMClient (Gemini, OpenAI, │
│                                    │   Anthropic, Groq, Ollama)          │
│                                    └─ MockLLMClient (offline fixtures)   │
├──────────────────────────────────────────────────────────────────────────┤
│ config.py (.env)   utils/ (data_loader, export, storage, logging)        │
└──────────────────────────────────────────────────────────────────────────┘
```

Dependencies only point downwards. The pipeline never imports a provider SDK directly.
It only sees `LLMClient.invoke(messages, task=..., payload=...)`.

## Key design decisions

| Decision | Why |
|---|---|
| **Own `LLMClient` interface over LangChain chat models** | One switch (`LLM_PROVIDER`) changes the provider. The mock implements the same interface, so all code paths are testable offline. |
| **Raw text + explicit repair loop instead of `with_structured_output`** | The assignment requires *automatic repair of malformed outputs*. Provider-native structured output hides malformed responses, so we would have nothing to repair or measure. LangChain's `PydanticOutputParser` still supplies the format instructions. |
| **Validation context for grounding** | `LLMExtraction.model_validate(obj, context={"review_text": ...})` rejects evidence that is not a near-verbatim quote. This turns "do not invent aspects" from a request in the prompt into a check in code. |
| **One taxonomy module** | Prompts, Pydantic enums, the annotation guide and the annotation UI all use `schemas/taxonomy.py`. Gold labels and predictions are guaranteed to share one label space. |
| **Statistics computed in pandas** | LLMs are unreliable at counting. The summariser receives numbers and must copy them. A deterministic check flags any percentage that is not in the statistics. |
| **Citations validated** | Every summary point must cite review IDs from the evidence shown to the model. Unknown IDs fail validation and trigger repair. |
| **JSONL caches** | Each extraction is appended as it completes, so runs are resumable and an interrupted 900-call experiment loses nothing. A partially written last line is skipped on reload. |
| **Physical live/mock separation** | `outputs/live/` vs `outputs/mock/`. Mock run IDs start with `MOCK-`, every record carries `is_mock`, and reports carry a banner. |

## Data flow — extract → aggregate → summarise → judge

```
reviews.csv ─► ReviewInput ─► PromptStrategy.build_messages()
                                   │
                                   ▼
                       generate_structured()  ◄──────────────┐
                 LLM call → extract JSON → local fixes        │ repair prompt
                 → normalise → Pydantic validate (+grounding) │ (original output
                       │ valid                │ invalid ──────┘  + errors), ≤ N times
                       ▼                      ▼ (after N)
                 ReviewRecord           ExtractionFailure (controlled)
                       │
                       ▼  (ExtractionStore JSONL cache)
            aggregate_product()  — counts, %, aspect stats, complaints, trend, evidence quotes
                       │
                       ▼
            summarize_product()  — LLM, ProductSummaryContent with cited review IDs
                       │
                       ▼
            judge_summary()      — LLM-as-a-Judge, JudgeVerdict (1–5 × 4 dimensions)
                                   + deterministic_checks()
```

## The generic repair loop (`pipeline/repair.py`)

1. Call the LLM and keep the raw text (`AttemptLog.raw_output`).
2. `extract_json_candidate` strips code fences and surrounding prose. It uses a brace scanner that
   understands string literals, and it detects truncated output.
3. `apply_local_fixes` removes trailing commas and replaces smart quotes. These deterministic fixes
   need no LLM call and each one is logged.
4. The normaliser makes conservative format fixes (`"Positive"` → `"positive"`, unwraps `{"result": {...}}`).
   Semantically invalid values such as `"very good"` are deliberately **not** guessed.
5. Pydantic validation produces readable errors such as `aspects.2.evidence: Field required`.
6. If invalid, the **repair prompt** gets the task description, the schema, the source material,
   the original output and the error list. Go back to step 2.
7. After `MAX_REPAIR_ATTEMPTS` the loop returns a `StructuredResult` with `value=None` and an error.
   The extraction becomes an `ExtractionFailure`. Invalid data is never stored as a record.

The same function serves extraction (`LLMExtraction`), summaries (`ProductSummaryContent`)
and judge verdicts (`JudgeVerdict`).

## Offline mock mode

`llm/mock.py` returns JSON built by a keyword lexicon (sentence-level aspect keywords + polarity words with
negation handling). It deliberately corrupts a configurable share (`MOCK_MALFORMED_RATE`, default 15%)
of extraction responses in 7 realistic ways: prose wrapper, markdown fence, trailing comma, bad enum,
missing field, wrong type and truncation. This exercises every branch of the repair loop. It returns
identical output for all three strategies, so mock "metrics" cannot be mistaken for a strategy comparison.
