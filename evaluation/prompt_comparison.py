"""Zero-shot vs few-shot vs CoT experiment runner.

Fairness controls: one client (same provider/model/temperature) for all
strategies, identical review set and order, identical preprocessing, schema and
label definitions (see prompts/common.py).

Each run lives in ``outputs/<live|mock>/experiments/<run_id>/``:
  config.json              settings snapshot, strategies, git commit, review ids
  predictions_<s>.jsonl    resumable per-strategy predictions (incl. raw outputs & repair logs)
  metrics.json             evaluation against human annotations + operational stats
  report.md                human-readable report
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from config import PROJECT_ROOT, Settings
from evaluation.classification_metrics import evaluate_predictions
from llm.base import LLMClient
from pipeline.extraction import ExtractionStore, ReviewExtractor
from prompts.common import PROMPT_VERSION
from prompts.strategies import STRATEGIES, get_strategy
from schemas.evaluation_schema import Annotation
from schemas.review_schema import ExtractionFailure, ReviewInput, ReviewRecord
from utils.storage import log_activity, read_json, slug, write_json_atomic

logger = logging.getLogger(__name__)

MOCK_BANNER = (
    "> **MOCK / OFFLINE RUN — NOT AN EXPERIMENTAL RESULT.** Predictions come from a deterministic "
    "keyword fixture, identical for every strategy. These numbers only demonstrate that the evaluation "
    "pipeline works and must not be reported as model performance."
)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def experiments_dir(settings: Settings, is_mock: bool) -> Path:
    return settings.paths.mode_dir(is_mock) / "experiments"


def new_run_id(model_name: str, is_mock: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{'MOCK-' if is_mock else ''}{stamp}_{slug(model_name)}"


def evaluation_status(is_mock: bool, n_evaluated: int, required: int) -> str:
    if is_mock:
        return "mock_demo"
    if n_evaluated == 0:
        return "awaiting_annotations"
    return "complete" if n_evaluated >= required else "provisional"


STATUS_TEXT = {
    "mock_demo": "Mock demo — not a result",
    "awaiting_annotations": "Awaiting human annotations — no classification metrics yet",
    "provisional": "Provisional — fewer than the required number of human-labelled reviews",
    "complete": "Complete — evaluated against the full human-labelled set",
}


def operational_stats(records: list[ReviewRecord], failures: list[ExtractionFailure]) -> dict:
    metas = [r.extraction_metadata for r in records] + [f.extraction_metadata for f in failures]
    n = len(metas)
    if n == 0:
        return {"n": 0}
    in_tok = [m.input_tokens for m in metas if m.input_tokens is not None]
    out_tok = [m.output_tokens for m in metas if m.output_tokens is not None]
    return {
        "n": n,
        "valid": len(records),
        "failed": len(failures),
        "validity_rate": round(len(records) / n, 4),
        "first_attempt_valid_rate": round(sum(m.validation_success and m.repair_attempts == 0 for m in metas) / n, 4),
        "needed_llm_repair": int(sum(m.repair_attempts > 0 for m in metas)),
        "mean_repair_attempts": round(sum(m.repair_attempts for m in metas) / n, 4),
        "local_fixes": int(sum(m.local_repairs for m in metas)),
        "mean_latency_s": round(sum(m.total_latency_s for m in metas) / n, 4),
        "total_latency_s": round(sum(m.total_latency_s for m in metas), 2),
        "total_input_tokens": int(sum(in_tok)) if in_tok else None,
        "total_output_tokens": int(sum(out_tok)) if out_tok else None,
        "mean_input_tokens": round(sum(in_tok) / len(in_tok), 1) if in_tok else None,
        "mean_output_tokens": round(sum(out_tok) / len(out_tok), 1) if out_tok else None,
    }


def run_experiment(
    settings: Settings,
    client: LLMClient,
    strategies: list[str],
    reviews: list[ReviewInput],
    gold: list[Annotation],
    run_id: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Run (or resume) an experiment and write predictions, metrics and report."""
    for s in strategies:
        get_strategy(s)
    run_id = run_id or new_run_id(client.model_name, client.is_mock)
    run_dir = experiments_dir(settings, client.is_mock) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config = read_json(config_path) or {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_mock": client.is_mock,
        "provider": client.provider,
        "model_name": client.model_name,
        "temperature": settings.llm_temperature,
        "prompt_version": PROMPT_VERSION,
        "strategies": strategies,
        "max_repair_attempts": settings.max_repair_attempts,
        "strict_evidence_grounding": settings.strict_evidence_grounding,
        "review_ids": [r.review_id for r in reviews],
        "git_commit": git_commit(),
        "settings": settings.public_dict(),
    }
    if config["model_name"] != client.model_name or config["is_mock"] != client.is_mock:
        raise ValueError(f"Run {run_id} was created with {config['model_name']}; refusing to mix models in one run.")
    config["strategies"] = list(dict.fromkeys(config["strategies"] + strategies))
    write_json_atomic(config_path, config)

    for name in strategies:
        store = ExtractionStore(run_dir / f"predictions_{name}.jsonl")
        extractor = ReviewExtractor(client, get_strategy(name), settings)
        extractor.extract_many(reviews, store=store, progress=(lambda i, n, _o, name=name: progress(name, i, n)) if progress else None)

    log_activity(settings.paths.activity_log, "prompt_experiment", client.is_mock, run_id=run_id, strategies=strategies, n_reviews=len(reviews))
    evaluate_run(run_dir, gold, settings.annotation_sample_size)
    return run_dir


def load_run_predictions(run_dir: Path, strategy: str) -> tuple[list[ReviewRecord], list[ExtractionFailure]]:
    store = ExtractionStore(run_dir / f"predictions_{strategy}.jsonl")
    return store.records, store.failures


def evaluate_run(run_dir: Path, gold: list[Annotation], required: int) -> dict:
    """(Re-)compute metrics for a run from saved predictions and current annotations."""
    config = read_json(run_dir / "config.json")
    run_ids = set(config["review_ids"])
    gold_in_run = [g for g in gold if g.review_id in run_ids]
    results: dict = {
        "run_id": config["run_id"],
        "is_mock": config["is_mock"],
        "model_name": config["model_name"],
        "provider": config["provider"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_gold_available": len(gold_in_run),
        "required_gold": required,
        "status": evaluation_status(config["is_mock"], len(gold_in_run), required),
        "strategies": {},
    }
    for name in config["strategies"]:
        records, failures = load_run_predictions(run_dir, name)
        preds: dict[str, ReviewRecord | None] = {r.review_id: r for r in records}
        preds.update({f.review_id: None for f in failures})
        entry = {"operational": operational_stats(records, failures)}
        if gold_in_run:
            entry["classification"] = evaluate_predictions(gold_in_run, preds)
        results["strategies"][name] = entry
    write_json_atomic(run_dir / "metrics.json", results)
    (run_dir / "report.md").write_text(render_report(config, results), encoding="utf-8")
    return results


def list_runs(settings: Settings, is_mock: bool) -> list[dict]:
    base = experiments_dir(settings, is_mock)
    runs = []
    if base.exists():
        for d in sorted(base.iterdir(), reverse=True):
            cfg = read_json(d / "config.json")
            if cfg:
                runs.append({"run_id": cfg["run_id"], "path": d, "model_name": cfg["model_name"], "strategies": cfg["strategies"], "n_reviews": len(cfg["review_ids"]), "created_at": cfg["created_at"]})
    return runs


# --------------------------------------------------------------------------- tables & report
def comparison_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, entry in metrics.get("strategies", {}).items():
        op = entry["operational"]
        cl = entry.get("classification", {})
        sent, comp, det, asent = (cl.get(k, {}) for k in ("overall_sentiment", "complaints", "aspect_detection", "aspect_sentiment"))
        rows.append(
            {
                "strategy": STRATEGIES[name].label if name in STRATEGIES else name,
                "n_gold": cl.get("n_evaluated"),
                "sentiment_accuracy": sent.get("accuracy"),
                "sentiment_macro_precision": sent.get("macro_precision"),
                "sentiment_macro_recall": sent.get("macro_recall"),
                "sentiment_macro_f1": sent.get("macro_f1"),
                "sentiment_weighted_f1": sent.get("weighted_f1"),
                "complaint_micro_f1": comp.get("micro_f1"),
                "complaint_macro_f1": comp.get("macro_f1_present_labels"),
                "complaint_hamming_loss": comp.get("hamming_loss"),
                "aspect_detection_micro_f1": det.get("micro_f1"),
                "aspect_sentiment_macro_f1": asent.get("macro_f1"),
                "aspect_joint_micro_f1": cl.get("aspect_joint", {}).get("micro_f1"),
                "validity_rate": op.get("validity_rate"),
                "first_attempt_valid_rate": op.get("first_attempt_valid_rate"),
                "needed_llm_repair": op.get("needed_llm_repair"),
                "mean_latency_s": op.get("mean_latency_s"),
                "mean_input_tokens": op.get("mean_input_tokens"),
                "mean_output_tokens": op.get("mean_output_tokens"),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_"
    df = df.astype(object).where(pd.notna(df), "—")
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def render_report(config: dict, metrics: dict) -> str:
    table = comparison_table(metrics)
    lines = [f"# Prompt Comparison Report — `{config['run_id']}`", ""]
    if config["is_mock"]:
        lines += [MOCK_BANNER, ""]
    lines += [
        f"**Evaluation status:** {STATUS_TEXT[metrics['status']]} "
        f"({metrics['n_gold_available']} / {metrics['required_gold']} human-labelled reviews in this run).",
        "",
        "## Configuration",
        "",
        f"- Provider / model: `{config['provider']}` / `{config['model_name']}` (temperature {config['temperature']})",
        f"- Prompt version: `{config['prompt_version']}` · strategies: {', '.join(config['strategies'])}",
        f"- Reviews in run: {len(config['review_ids'])} · max repair attempts: {config['max_repair_attempts']} · strict grounding: {config['strict_evidence_grounding']}",
        f"- Created: {config['created_at']} · git commit: `{config['git_commit']}`",
        "",
    ]
    sentiment_cols = ["strategy", "n_gold", "sentiment_accuracy", "sentiment_macro_precision", "sentiment_macro_recall", "sentiment_macro_f1", "sentiment_weighted_f1"]
    other_cols = ["strategy", "complaint_micro_f1", "complaint_macro_f1", "complaint_hamming_loss", "aspect_detection_micro_f1", "aspect_sentiment_macro_f1", "aspect_joint_micro_f1"]
    op_cols = ["strategy", "validity_rate", "first_attempt_valid_rate", "needed_llm_repair", "mean_latency_s", "mean_input_tokens", "mean_output_tokens"]
    if metrics["n_gold_available"]:
        lines += ["## Overall sentiment (4 classes)", "", _md_table(table[sentiment_cols]), "",
                  "## Complaints (multi-label) and aspects", "", _md_table(table[other_cols]), ""]
        for name, entry in metrics["strategies"].items():
            cl = entry.get("classification")
            if not cl:
                continue
            s = cl["overall_sentiment"]
            lines += [f"### {name}: per-class sentiment metrics", "", "| class | precision | recall | f1 | support |", "| --- | --- | --- | --- | --- |"]
            lines += [f"| {lab} | {v['precision']} | {v['recall']} | {v['f1-score']} | {v['support']} |" for lab, v in s["per_class"].items()]
            lines += ["", f"Confusion matrix (rows = human label, columns = prediction, order {s['labels']}): `{s['confusion_matrix']}`", ""]
    else:
        lines += ["## Classification metrics", "", "_Pending: no human annotations overlap with this run yet. Complete the annotation workflow, then re-evaluate._", ""]
    lines += ["## Reliability, latency and token usage", "", _md_table(table[op_cols]), ""]
    lines += [
        "## Methodology notes",
        "",
        "- Ground truth = human annotations from the project's annotation workflow (never model output).",
        "- Failed extractions are kept: wrong sentiment, empty aspect/complaint sets.",
        "- Aspects are matched by aspect category; aspect sentiment is scored on categories found in both gold and prediction.",
        "- Complaint macro-F1 is averaged over labels present in the gold set.",
        "",
    ]
    return "\n".join(lines)


def metrics_to_json(metrics: dict) -> str:
    return json.dumps(metrics, indent=2)
