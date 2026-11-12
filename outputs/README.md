# outputs/

Generated artefacts. This folder is git-ignored apart from this file.

```
outputs/
├── live/                  real LLM results (local Ollama by default; created by live runs)
│   ├── extractions/       <source>__<strategy>__<model>__<prompt_version>.jsonl  (resumable cache)
│   ├── experiments/       <run_id>/config.json, predictions_<strategy>.jsonl, metrics.json, report.md
│   ├── summaries/         summaries.jsonl
│   └── judgements/        judgements.jsonl, inspection_sample.md
├── mock/                  OFFLINE fixture outputs (same layout) — NEVER experimental results
└── activity_log.jsonl     processing history shown on the dashboard (each row tagged live/mock)
```

To publish final live results, copy the relevant `report.md` / CSV files into `reports/`.
