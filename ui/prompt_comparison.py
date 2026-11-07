"""Page 3 — Prompt Comparison: zero-shot vs few-shot vs CoT against human labels."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from annotation.annotation_manager import AnnotationStore, load_sample
from evaluation.prompt_comparison import (
    MOCK_BANNER,
    STATUS_TEXT,
    comparison_table,
    evaluate_run,
    list_runs,
    load_run_predictions,
    run_experiment,
)
from evaluation.visualizations import confusion_heatmap, metric_comparison_bar
from prompts.strategies import STRATEGIES
from ui.common import get_client, get_settings, llm_ready, mock_banner, require_llm_note, use_mock
from utils.data_loader import dataframe_to_inputs
from utils.export import dataframe_to_csv_bytes, records_to_dataframe
from utils.storage import read_json

s = get_settings()
st.title("⚖️ Prompt Comparison")
st.caption("Zero-shot vs Few-shot vs Chain-of-Thought: same model, same reviews, same schema and label definitions.")
mock_banner()

sample = load_sample(s.paths.annotation_sample_csv)
store = AnnotationStore(s.paths.annotations_json)
if sample.empty:
    st.error("No annotation sample. Build it on the **Data Annotation** page first.")
    st.stop()
gold = store.completed(sample["review_id"].tolist())
n_gold = len(gold)

st.progress(min(1.0, n_gold / s.annotation_sample_size), text=f"Human-labelled reviews: {n_gold} / {s.annotation_sample_size}")
if n_gold < s.annotation_sample_size:
    st.warning(
        f"**Evaluation pending:** the assignment requires {s.annotation_sample_size} manually labelled reviews; {n_gold} are done. "
        "Until then, any metrics are marked *provisional*. Label reviews on the **Data Annotation** page. "
        "No ground truth is ever generated automatically."
    )

# ---------------------------------------------------------------- run experiment
with st.expander("▶ Run a new experiment / resume a run", expanded=not list_runs(s, use_mock())):
    require_llm_note()
    strategies = st.multiselect("Strategies", list(STRATEGIES), default=list(STRATEGIES), format_func=lambda k: STRATEGIES[k].label)
    for k in strategies:
        st.caption(f"**{STRATEGIES[k].label}:** {STRATEGIES[k].description}")
    c1, c2 = st.columns(2)
    scope = c1.radio("Reviews", ["Human-annotated reviews only", "First N reviews of the annotation sample"], help="Classification metrics are only computed where human labels exist.")
    if scope.startswith("Human"):
        pool = sample[sample["review_id"].isin({g.review_id for g in gold})]
    else:
        pool = sample
    n = c2.number_input("Max reviews", 1, max(1, len(pool)), max(1, min(len(pool), 300)), disabled=pool.empty)
    existing = [r["run_id"] for r in list_runs(s, use_mock())]
    resume = st.selectbox("Run", ["(new run)"] + existing, help="Resuming re-uses saved predictions (no repeated API calls).")
    calls = int(n) * len(strategies)
    st.caption(f"Estimated LLM calls: ~{calls} (+ repairs). With the free-tier rate limit of {s.llm_requests_per_minute}/min that is ~{calls / max(1, s.llm_requests_per_minute):.0f} min. For long runs, `scripts/run_experiment.py` is more robust.")
    if st.button("Run experiment", type="primary", disabled=pool.empty or not strategies or not llm_ready()):
        client = get_client()
        if client:
            reviews = dataframe_to_inputs(pool.head(int(n)))
            bar = st.progress(0.0)
            try:
                run_dir = run_experiment(s, client, strategies, reviews, gold, run_id=None if resume == "(new run)" else resume,
                                         progress=lambda name, i, total: bar.progress(i / total, text=f"{STRATEGIES[name].label}: {i}/{total}"))
                st.success(f"Run saved: {run_dir.name}")
                st.session_state["selected_run"] = run_dir.name
            except Exception as exc:
                st.error(f"Experiment stopped: {exc}. Saved predictions are kept; resume the run to continue.")

# ---------------------------------------------------------------- results
runs = list_runs(s, use_mock())
if not runs:
    st.info("No experiment runs in this mode yet.")
    st.stop()

ids = [r["run_id"] for r in runs]
default = ids.index(st.session_state["selected_run"]) if st.session_state.get("selected_run") in ids else 0
run_id = st.selectbox("Experiment run", ids, index=default, format_func=lambda i: next(f"{r['run_id']} · {r['model_name']} · {r['n_reviews']} reviews · {', '.join(r['strategies'])}" for r in runs if r["run_id"] == i))
run = next(r for r in runs if r["run_id"] == run_id)
run_dir = run["path"]

c1, c2 = st.columns([1, 3])
if c1.button("🔄 Re-evaluate with current annotations", help="Recomputes metrics from saved predictions; no LLM calls."):
    evaluate_run(run_dir, gold, s.annotation_sample_size)
metrics = read_json(run_dir / "metrics.json") or evaluate_run(run_dir, gold, s.annotation_sample_size)
config = read_json(run_dir / "config.json")
if metrics["is_mock"]:
    st.error(MOCK_BANNER.replace("> ", ""))
status = metrics["status"]
(st.success if status == "complete" else st.warning)(f"**Status:** {STATUS_TEXT[status]} ({metrics['n_gold_available']} labelled reviews in this run; evaluated {metrics['evaluated_at']}).")

table = comparison_table(metrics)
has_cls = metrics["n_gold_available"] > 0

tab1, tab2, tab3, tab4 = st.tabs(["Classification metrics", "Confusion matrices", "Time & tokens", "Downloads"])
with tab1:
    if not has_cls:
        st.info("Classification metrics will appear once human annotations overlap with this run.")
    else:
        st.markdown("**Overall sentiment (4 classes)**")
        st.dataframe(table[["strategy", "n_gold", "sentiment_accuracy", "sentiment_macro_precision", "sentiment_macro_recall", "sentiment_macro_f1", "sentiment_weighted_f1"]], width="stretch", hide_index=True)
        st.plotly_chart(metric_comparison_bar(table, ["sentiment_accuracy", "sentiment_macro_f1", "sentiment_weighted_f1", "complaint_micro_f1", "complaint_macro_f1", "aspect_detection_micro_f1", "aspect_sentiment_macro_f1", "aspect_joint_micro_f1"]), width="stretch")
        st.markdown("**Complaints (multi-label) and aspects**")
        st.dataframe(table[["strategy", "complaint_micro_f1", "complaint_macro_f1", "complaint_hamming_loss", "aspect_detection_micro_f1", "aspect_sentiment_macro_f1", "aspect_joint_micro_f1"]], width="stretch", hide_index=True)
        strat = st.selectbox("Per-class detail for", list(metrics["strategies"]), format_func=lambda k: STRATEGIES.get(k).label if k in STRATEGIES else k)
        cl = metrics["strategies"][strat].get("classification", {})
        a, b = st.columns(2)
        a.markdown("Overall sentiment per class")
        a.dataframe(pd.DataFrame(cl["overall_sentiment"]["per_class"]).T, width="stretch")
        b.markdown("Complaint categories per label")
        b.dataframe(pd.DataFrame(cl["complaints"]["per_label"]).T.query("support > 0"), width="stretch")
with tab2:
    if not has_cls:
        st.info("Confusion matrices need human labels.")
    else:
        cols = st.columns(len(metrics["strategies"]))
        for col, (name, entry) in zip(cols, metrics["strategies"].items()):
            cm = entry["classification"]["overall_sentiment"]
            col.plotly_chart(confusion_heatmap(cm["confusion_matrix"], cm["labels"], STRATEGIES[name].label if name in STRATEGIES else name), width="stretch")
            asent = entry["classification"]["aspect_sentiment"]
            if asent.get("n"):
                col.plotly_chart(confusion_heatmap(asent["confusion_matrix"], asent["labels"], "Aspect sentiment (matched)"), width="stretch")
with tab3:
    st.dataframe(table[["strategy", "validity_rate", "first_attempt_valid_rate", "needed_llm_repair", "mean_latency_s", "mean_input_tokens", "mean_output_tokens"]], width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(table, x="strategy", y="mean_latency_s", title="Mean latency per review (s)", text_auto=".2f").update_layout(height=320), width="stretch")
    tok = table.melt(id_vars="strategy", value_vars=["mean_input_tokens", "mean_output_tokens"], var_name="tokens", value_name="mean").dropna()
    b.plotly_chart(px.bar(tok, x="strategy", y="mean", color="tokens", barmode="group", title="Mean tokens per review").update_layout(height=320), width="stretch")
    if metrics["is_mock"]:
        st.caption("Mock token counts are rough character-based estimates and latency is 0 — not measurements.")
with tab4:
    report = (run_dir / "report.md").read_text(encoding="utf-8") if (run_dir / "report.md").exists() else ""
    d = st.columns(3)
    d[0].download_button("Report (Markdown)", report, f"{run_id}_report.md", "text/markdown", width="stretch")
    d[1].download_button("Metrics (JSON)", json.dumps(metrics, indent=2), f"{run_id}_metrics.json", "application/json", width="stretch")
    d[2].download_button("Comparison table (CSV)", dataframe_to_csv_bytes(table), f"{run_id}_comparison.csv", "text/csv", width="stretch")
    for name in config["strategies"]:
        recs, _ = load_run_predictions(run_dir, name)
        st.download_button(f"Predictions — {name} (CSV)", dataframe_to_csv_bytes(records_to_dataframe(recs)), f"{run_id}_{name}_predictions.csv", "text/csv")
    with st.expander("Preview report"):
        st.markdown(report)
    with st.expander("Run configuration"):
        st.json({k: v for k, v in config.items() if k != "review_ids"}, expanded=False)
