"""Read-only dashboard section for the final D3 evaluation (judge v4 claim-level verification + negative control v3).

Rendered from the Summary Evaluation page. It only reads stored artefacts written by ``scripts/judge_v4.py report``;
it never calls a model. Deterministic checks and model-assisted (LLM) verdicts are labelled separately everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.storage import read_json

STATUS_ORDER = ["CONTRADICTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED", "UNVERIFIABLE", "SUPPORTED"]
METHOD_LABEL = {"deterministic": "deterministic check", "llm": "LLM (model-assisted)", "deterministic+llm": "deterministic + LLM",
                "none": "no verdict"}


@st.cache_data(show_spinner=False)
def _load(folder: str, mtime: float) -> tuple[dict, list[dict]]:
    d = Path(folder)
    rep = read_json(d / "report.json", default={}) or {}
    evals = [json.loads(x) for x in (d / "evaluations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (d / "evaluations.jsonl").exists() else []
    return rep, evals


def summary_status(mode_dir: Path, summary_id: str) -> dict | None:
    """The stored v4 evaluation of exactly this summary id (originals only; no score is transferred between ids)."""
    root = mode_dir / "d3" / "judge_v4"
    if not root.exists():
        return None
    for folder in sorted(p for p in root.iterdir() if (p / "report.json").exists()):
        _, evals = _load(str(folder), (folder / "report.json").stat().st_mtime)
        for e in evals:
            if e["summary_id"] == summary_id:
                return e
    return None


def render(mode_dir: Path, sources: list[str]) -> None:
    folder = next((mode_dir / "d3" / "judge_v4" / s for s in sources if (mode_dir / "d3" / "judge_v4" / s / "report.json").exists()), None)
    st.subheader("Final D3 evaluation: claim-level verification (judge v4)")
    if folder is None:
        st.info("No judge v4 results yet. Run `python scripts/judge_v4.py run --source <source>` and then `report`.",
                title="Not evaluated yet")
        return
    rep, evals = _load(str(folder), (folder / "report.json").stat().st_mtime)
    st.caption("Each summary is split into claims. Numbers, counts, percentages, majorities and aspect statements are checked "
               "**deterministically** against the statistics; each cited point is checked by the local **LLM** against the full "
               "text of its cited reviews only, and its quote must occur in those reviews. Scores are computed from the claim "
               "statuses by a fixed formula; the LLM never writes a score. v1-v3 results above are unchanged.")
    o = rep.get("originals", {})
    nc = rep.get("negative_control_v3", {})
    prim = nc.get("metrics", {}).get("primary", {}).get("combined", {})
    with st.container(horizontal=True):
        st.metric("Mean faithfulness (15 originals)", f"{o.get('mean_faithfulness', '–')}", border=True)
        st.metric("Claims evaluated", o.get("claims", "–"), border=True)
        st.metric("Contradicted / unsupported", f"{o.get('status_counts', {}).get('CONTRADICTED', 0)} / "
                                                f"{o.get('status_counts', {}).get('UNSUPPORTED', 0)}", border=True)
        st.metric("Negative control recall", f"{prim.get('recall', 0):.0%}" if prim.get("recall") is not None else "–", border=True)
        st.metric("Negative control specificity", f"{prim.get('specificity', 0):.0%}" if prim.get("specificity") is not None else "–", border=True)

    view = st.segmented_control("View", ["Summaries", "Claims", "Rubric (1-5)", "Negative control", "Completion status", "Run metadata"],
                                default="Summaries", key="v4_view")
    if view == "Summaries":
        rows = pd.DataFrame(o.get("per_product", []))
        if not rows.empty:
            st.markdown("**Original summaries**")
            st.dataframe(rows[["product_id", "total_claims", "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNVERIFIABLE",
                               "numeric_failed", "evidence_coverage", "faithfulness_score", "validity_status"]],
                         hide_index=True, alt="Judge v4 results per original summary")
        comp = pd.DataFrame(rep.get("corrected", {}).get("comparison", []))
        if not comp.empty:
            st.markdown("**Corrected copies** (separate objects with their own ids; labelled revised artefacts, never merged with originals)")
            st.dataframe(comp[["product_id", "correction_set", "corrected_summary_id", "original_score", "corrected_score",
                               "original_validity", "corrected_validity"]], hide_index=True, alt="Original versus corrected copies")
    elif view == "Claims":
        if not evals:
            st.info("No evaluation file found.")
            return
        labels = {e["summary_id"]: f"{e['product_id']} · {e['evaluated_object']}" + (" (revised copy)" if e["evaluated_object"] == "corrected" else "")
                  for e in evals}
        sid = st.selectbox("Summary", list(labels), format_func=labels.get, key="v4_summary")
        ev = next(e for e in evals if e["summary_id"] == sid)
        statuses = st.pills("Status", STATUS_ORDER, selection_mode="multi", default=STATUS_ORDER, key="v4_status")
        st.caption(f"Faithfulness {ev['faithfulness_score']} · {ev['validity_status']} · evidence coverage {ev['evidence_coverage']:.0%} · "
                   f"numeric claims {ev['numeric_claims']} ({ev['numeric_claims_failed']} failed)")
        rows = []
        for c in ev["claim_evaluations"]:
            if c["status"] not in statuses:
                continue
            rows.append({"status": c["status"], "field": c["claim"]["source_field"], "claim": c["claim"]["text"],
                         "check": METHOD_LABEL.get(c["method"], c["method"]),
                         "cited reviews": ", ".join(c["claim"]["cited_review_ids"]),
                         "evidence quote (verified)" : (c.get("llm_quote") or "") + (" ✓" if c.get("llm_quote_verified") else ""),
                         "numeric checks": " | ".join(f"{n['mention']}: {'pass' if n['passed'] else n['severity']} (expected {n['expected_value']})"
                                                      for n in c["numeric_checks"]),
                         "explanation": c["explanation"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Claim-level verdicts for the selected summary")
    elif view == "Negative control":
        man = nc.get("manifest", {})
        st.caption(f"{man.get('nc_version', 'nc-v3')}: {nc.get('cases', 0)} pre-registered cases (seed {man.get('seed')}, dataset SHA-256 "
                   f"{str(man.get('dataset_sha256', ''))[:12]}…, frozen before judging). Each case has one corrupted claim and one matched "
                   "correct claim. The 30 earlier nc-v2 copies and their v2/v3 results are unchanged (shown above).")
        m = nc.get("metrics", {})
        st.dataframe(pd.DataFrame([{"rule": r, "layer": layer, **{k: v[k] for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "specificity")}}
                                   for r, layers in m.items() for layer, v in layers.items()]), hide_index=True,
                     alt="Negative control confusion matrix and metrics by rule and layer")
        st.dataframe(pd.DataFrame([{"error type": t, "cases": v["cases"], "detected": v["detected"], "detection rate": v["detection_rate"],
                                    "95% CI": v["detection_wilson95"], "clean flagged (false positives)": v["clean_flagged"],
                                    "LLM-only detected": v["llm_only_detected"], "deterministic-only detected": v["deterministic_only_detected"]}
                                   for t, v in nc.get("by_type", {}).items()]), hide_index=True, alt="Detection by error type")
    elif view == "Rubric (1-5)":
        rub = read_json(folder / "rubric" / "rubric.json", default=None)
        if not rub:
            st.info("No rubric yet. Run `python scripts/d3_completion.py --source <source>`.")
            return
        st.caption(f"Anchored 1-5 rubric ({rub['rubric_version']}) derived from the claim results above; {rub['llm_calls']} LLM calls, "
                   f"invariant violations: {len(rub['invariant_violations'])}. Higher is better on every dimension. Anchors are not "
                   "calibrated against human ratings; hallucination counts LLM-flagged claims, which include false positives.")
        st.dataframe(pd.DataFrame([{"object": r["evaluated_object"], "summary": r["summary_id"], "product": r["product_id"],
                                    "faithfulness": r["faithfulness"], "coverage": r["coverage"], "consistency": r["consistency"],
                                    "hallucination": r["hallucination"], "score 0-100": r["faithfulness_score_0_100"],
                                    "missed themes": " ".join(r["coverage_detail"]["missed"])} for r in rub["rows"]]),
                     hide_index=True, alt="Derived rubric per summary")
        with st.expander("Anchors"):
            for k, v in rub["anchors"].items():
                st.markdown(f"**{k}** — {v}")
    elif view == "Completion status":
        status = read_json(folder.parent.parent / "completion" / folder.name / "status.json", default=None)
        if not status:
            st.info("No completion status yet. Run `python scripts/d3_completion.py --source <source> --verify`.")
            return
        st.caption(f"D3 completion criteria recomputed from the stored artefacts: {status['passed']} of {status['mandatory']} mandatory "
                   f"criteria pass. D3 is {'complete' if status['complete'] else 'NOT complete'}.")
        st.dataframe(pd.DataFrame([{"#": c["id"], "criterion": c["criterion"], "status": c["status"], "evidence": c["evidence"],
                                    "remaining": c["remaining"]} for c in status["criteria"]]), hide_index=True,
                     alt="D3 completion criteria")
    else:
        st.json({"judge config": rep.get("config"), "prompt_sha256": rep.get("prompt_sha256"), "llm_calls": {k: v for k, v in rep.get("llm_calls", {}).items() if k != "sessions"},
                 "reproducibility": {k: v for k, v in rep.get("reproducibility", {}).items() if k != "disagreements"},
                 "generated_at": rep.get("generated_at")}, expanded=False)
