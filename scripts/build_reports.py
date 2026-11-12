"""Curate final results from outputs/live/ into reports/ (tracked by git), with charts and a reproducibility check.

  python scripts/build_reports.py --d2-run <run_id> [--compare-run <run_id>] [--d1-run <run_id>] [--d3-source <source>]

Writes:
  reports/d2/<run_id>/   report.md, config.json, metrics.json, comparison_table.csv, per_class_metrics.csv,
                         confusion_matrices.csv, charts/*.html (+ .png when a local Chrome/Edge is available)
  reports/d1/            summary.md + validation_report_<strategy>.json   (from scripts/export_d1.py output)
  reports/d3/            d3_report.md, judgements.csv, aggregates.csv, ... (from scripts/export_d3.py output)
  reports/verification.md  metrics recomputed from the saved predictions vs. the stored metrics.json

No LLM calls are made.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.annotation_manager import AnnotationStore, load_sample  # noqa: E402
from config import PROJECT_ROOT, load_settings  # noqa: E402
from evaluation.prompt_comparison import comparison_table, evaluate_run, experiments_dir  # noqa: E402
from evaluation.visualizations import confusion_heatmap, grouped_strategy_bar, metric_comparison_bar, per_class_bar, strategy_bar  # noqa: E402
from prompts.strategies import STRATEGIES  # noqa: E402
from utils.storage import read_json  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium"]


def _browser() -> str | None:
    return next((b for b in BROWSERS if Path(b).exists()), None)


def save_chart(fig, path: Path, width: int = 900, height: int | None = None) -> None:
    """Self-contained HTML (works offline) and, if a local browser exists, a PNG rendered from it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Print styling (the dashboard theme is dark/transparent): white background, dark text, title above the legend.
    fig.update_layout(width=width, height=height or fig.layout.height or 420, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#1d2433", size=13), title=dict(x=0.01, y=0.97, font=dict(size=16, color="#111a33")),
                      margin=dict(t=90, l=60, r=30, b=60), legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#1d2433")))
    fig.update_xaxes(gridcolor="#e3e6ef", linecolor="#9aa3c0", tickfont=dict(color="#1d2433"), title_font=dict(color="#1d2433"))
    fig.update_yaxes(gridcolor="#e3e6ef", linecolor="#9aa3c0", tickfont=dict(color="#1d2433"), title_font=dict(color="#1d2433"))
    for tr in fig.data:
        if getattr(tr, "textfont", None) is not None and tr.type in ("bar",):
            tr.textfont.color = "#1d2433"
    html = path.with_suffix(".html")
    fig.write_html(html, include_plotlyjs=True, full_html=True, config={"displayModeBar": False})
    browser = _browser()
    if browser:
        h = int(fig.layout.height) + 40
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={width + 40},{h}",
                        "--virtual-time-budget=8000", f"--screenshot={path.with_suffix('.png')}", html.resolve().as_uri()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=False)


def d2(run_id: str, gold, required: int, settings) -> tuple[Path, dict, dict]:
    run_dir = experiments_dir(settings, run_id.startswith("MOCK-")) / run_id
    stored = read_json(run_dir / "metrics.json")
    recomputed = evaluate_run(run_dir, gold, required)  # rewrites metrics.json/report.md from saved predictions
    out = REPORTS / "d2" / run_id
    out.mkdir(parents=True, exist_ok=True)
    for f in ("report.md", "config.json", "metrics.json"):
        shutil.copy2(run_dir / f, out / f)
    table = comparison_table(recomputed)
    table.to_csv(out / "comparison_table.csv", index=False)
    per_class, cms = [], []
    for name, entry in recomputed["strategies"].items():
        cl = entry.get("classification", {}).get("overall_sentiment")
        if not cl:
            continue
        per_class += [{"strategy": name, "class": lab, **v} for lab, v in cl["per_class"].items()]
        cms += [{"strategy": name, "reference": t, "predicted": p, "count": cl["confusion_matrix"][i][j]}
                for i, t in enumerate(cl["labels"]) for j, p in enumerate(cl["labels"])]
    pd.DataFrame(per_class).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(cms).to_csv(out / "confusion_matrices.csv", index=False)
    charts = out / "charts"
    save_chart(strategy_bar(table, "sentiment_macro_f1", title="Overall-sentiment macro F1", y_title="macro F1", fmt=".3f"), charts / "macro_f1")
    save_chart(metric_comparison_bar(table, ["sentiment_accuracy", "sentiment_macro_f1", "complaint_micro_f1", "aspect_detection_micro_f1",
                                             "aspect_sentiment_macro_f1"], title="Metric comparison"), charts / "metric_comparison", width=1000)
    label_of = {n: STRATEGIES[n].label for n in STRATEGIES}
    names = [n for n in recomputed["strategies"] if recomputed["strategies"][n].get("classification")]
    save_chart(per_class_bar({label_of.get(n, n): recomputed["strategies"][n]["classification"]["overall_sentiment"]["per_class"] for n in names},
                             title="Per-class F1 (overall sentiment)"), charts / "per_class_f1")
    for n in names:
        cl = recomputed["strategies"][n]["classification"]["overall_sentiment"]
        save_chart(confusion_heatmap(cl["confusion_matrix"], cl["labels"], title=f"Confusion matrix — {label_of.get(n, n)}"), charts / f"confusion_{n}", width=520, height=440)
    save_chart(strategy_bar(table, "mean_latency_s", title="Mean latency per review", y_title="seconds"), charts / "latency")
    save_chart(grouped_strategy_bar(table, ["mean_input_tokens", "mean_output_tokens"], title="Mean tokens per review", y_title="tokens"), charts / "tokens")
    return out, stored or {}, recomputed


def _diff_metrics(stored: dict, recomputed: dict) -> list[str]:
    issues = []
    for name, entry in recomputed.get("strategies", {}).items():
        a = (stored.get("strategies", {}).get(name, {}).get("classification") or {}).get("overall_sentiment", {})
        b = (entry.get("classification") or {}).get("overall_sentiment", {})
        for k in ("accuracy", "macro_f1", "confusion_matrix"):
            if a.get(k) != b.get(k):
                issues.append(f"{name}.{k}: stored {a.get(k)} vs recomputed {b.get(k)}")
    return issues


def copy_tree(src: Path, dst: Path, patterns: tuple[str, ...]) -> list[Path]:
    copied = []
    for pat in patterns:
        for f in src.rglob(pat):
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied.append(target)
    return copied


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="Curate final results into reports/.")
    p.add_argument("--d2-run", required=True)
    p.add_argument("--compare-run", default=None, help="optional second run (e.g. another model) to curate as well")
    p.add_argument("--d1-run", default=None, help="run whose scripts/export_d1.py output to copy (default: --d2-run)")
    p.add_argument("--d3-source", default=None)
    args = p.parse_args()
    s = load_settings()
    sample = load_sample(s.paths.annotation_sample_csv)
    store = AnnotationStore(s.paths.annotations_json)
    gold = store.completed(sample["review_id"].tolist())
    required = store.progress(sample["review_id"].tolist())["evaluable"]

    lines = ["# Verification of curated results", "", "Metrics recomputed from the saved prediction files (no LLM calls) and compared with the stored `metrics.json`.", ""]
    for run_id in [r for r in (args.d2_run, args.compare_run) if r]:
        out, stored, recomputed = d2(run_id, gold, required, s)
        issues = _diff_metrics(stored, recomputed)
        lines += [f"## `{run_id}`", "", f"- status: **{recomputed['status']}** · reference labels in run: {recomputed['n_gold_available']} / {recomputed['required_gold']}",
                  f"- missing predictions: {recomputed.get('missing_predictions')}",
                  "- recomputation matches stored metrics: " + ("**yes**" if not issues else "**NO** — " + "; ".join(issues)), ""]
        print(f"D2 curated: {out}")

    is_mock = args.d2_run.startswith("MOCK-")
    d1_src = s.paths.mode_dir(is_mock) / "d1" / (args.d1_run or args.d2_run)
    if d1_src.exists():
        dst = REPORTS / "d1" / (args.d1_run or args.d2_run)
        copy_tree(d1_src, dst, ("summary.md", "validation_report.json"))
        lines += [f"- D1 export copied from `{d1_src.relative_to(PROJECT_ROOT)}`"]
    if args.d3_source:
        d3_src = s.paths.mode_dir(is_mock) / "d3" / args.d3_source
        if d3_src.exists():
            copy_tree(d3_src, REPORTS / "d3" / args.d3_source, ("*.md", "*.csv", "*.json"))
            lines += [f"- D3 export copied from `{d3_src.relative_to(PROJECT_ROOT)}`"]
        sanity = s.paths.mode_dir(is_mock) / "d3" / "judge_sanity" / args.d3_source
        if sanity.exists():
            copy_tree(sanity, REPORTS / "d3" / args.d3_source / "judge_sanity", ("*.json", "*.jsonl"))
            lines += [f"- Judge negative control copied from `{sanity.relative_to(PROJECT_ROOT)}`"]
    for f in REPORTS.rglob("*.json"):
        json.loads(f.read_text(encoding="utf-8"))
    for f in REPORTS.rglob("*.csv"):
        pd.read_csv(f)
    lines += ["", f"- All {len(list(REPORTS.rglob('*.json')))} JSON and {len(list(REPORTS.rglob('*.csv')))} CSV files under `reports/` parse successfully."]
    (REPORTS / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((REPORTS / "verification.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
