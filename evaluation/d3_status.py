"""D3 completion status, recomputed from stored artefacts (read-only, no LLM).

Used by the Overview dashboard and by ``scripts/d3_completion.py``. The criteria are those of
``reports/proposed_revisions_2026-11-15/D3_INVESTIGATION_AND_COMPLETION_PLAN.md`` §14 plus the anchored 1-5 rubric of the
problem statement §4.6. D3 is reported as complete only if EVERY mandatory criterion passes; there is no override.
"""

from __future__ import annotations

import json
from pathlib import Path

from schemas.claim_schema import ClaimStatus, SummaryEvaluationV4
from utils.storage import read_json

DEFAULT_SOURCE = "dataset__few_shot__qwen2.5-3b__v1"
EXPECTED_PRODUCTS = 15
REQUIRED_CORR2 = ("B0BTY63SFY", "B001DHKBXU")


def paths(mode_dir: Path, source: str = DEFAULT_SOURCE) -> dict[str, Path]:
    """``mode_dir`` is outputs/<mode>; the label store lives in <project>/data/annotations (project = mode_dir/../..)."""
    v4 = mode_dir / "d3" / "judge_v4" / source
    gold = mode_dir / "d3" / "claim_gold" / source
    single = gold / "single_annotator"
    return {"v4_dir": v4, "evaluations": v4 / "evaluations.jsonl", "report": v4 / "report.json",
            "rubric": v4 / "rubric" / "rubric.json", "gold_dir": gold, "template": gold / "claim_gold_template.csv",
            "single_dir": single, "comparison": single / "comparison.json", "labels_csv": single / "human_labels.csv",
            "store": mode_dir.parent.parent / "data" / "annotations" / "d3_claim_validation.json",
            "verification": mode_dir / "d3" / "completion" / source / "verification.json"}


def load_evaluations(path: Path) -> list[SummaryEvaluationV4]:
    """Every row is re-validated: a stored score that contradicts its claims raises here."""
    if not path.exists():
        return []
    return [SummaryEvaluationV4.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _crit(cid: str, name: str, mandatory: bool, ok: bool | None, evidence: str, remaining: str = "") -> dict:
    status = "PASS" if ok else ("FAIL" if ok is False else "PARTIAL")
    return {"id": cid, "criterion": name, "mandatory": mandatory, "status": status, "evidence": evidence,
            "remaining": "" if ok else remaining}


def compute(mode_dir: Path, source: str = DEFAULT_SOURCE, n_summaries: int | None = None) -> dict:
    p = paths(mode_dir, source)
    crits: list[dict] = []
    try:
        evals = load_evaluations(p["evaluations"])
        load_error = ""
    except Exception as e:  # an inconsistent stored evaluation is itself a failure
        evals, load_error = [], f"{type(e).__name__}: {e}"[:300]
    orig = [e for e in evals if e.evaluated_object == "original"]
    rep = read_json(p["report"], default={}) or {}
    n_orig = len({e.summary_id for e in orig})

    point_claims = [ce for e in orig for ce in e.claim_evaluations if ce.claim.kind == "point"]
    cites_ok = all(any(f.check == "citation_valid" and f.outcome == "pass" for f in ce.findings) for ce in point_claims)
    n_ok = n_orig == EXPECTED_PRODUCTS and (n_summaries is None or n_summaries == EXPECTED_PRODUCTS)
    crits.append(_crit("C1", "Pipeline: 15 summaries, every cited review id valid", True, n_ok and cites_ok and bool(point_claims),
                       f"{n_orig} originals evaluated; {len(point_claims)} cited points, citations valid: {cites_ok}",
                       "re-run scripts/judge_v4.py report"))

    claims = [ce for e in orig for ce in e.claim_evaluations]
    unquoted = [ce.claim.claim_id for ce in claims if ce.llm_status_effective in
                (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.CONTRADICTED) and not ce.llm_quote_verified]
    numeric = [ce for ce in claims if ce.claim.is_numeric]
    numeric_checked = all(ce.numeric_checks or ce.findings for ce in numeric)
    crits.append(_crit("C2", "Claim-level audit: every claim has a status; LLM support/contradiction only with a verified quote; "
                             "numbers checked deterministically", True,
                       bool(claims) and not unquoted and numeric_checked,
                       f"{len(claims)} claims; {len(numeric)} statistic statements (numbers in the text), all with deterministic checks: "
                       f"{numeric_checked}; {sum(1 for ce in claims if ce.numeric_checks)} claims carry numeric checks in total; "
                       f"LLM verdicts without verified quote counted as support: {len(unquoted)}",
                       "fix claim evaluations"))

    rubric = read_json(p["rubric"], default=None)
    rub_rows = (rubric or {}).get("rows", [])
    rub_orig = [r for r in rub_rows if r["evaluated_object"] == "original"]
    rub_problems = (rubric or {}).get("invariant_violations", None)
    crits.append(_crit("C3", "Consistency by construction: stored evaluations re-validate; rubric invariants hold", True,
                       not load_error and bool(evals) and rub_problems == [],
                       (load_error or f"{len(evals)} evaluations re-validated on load") + f"; rubric invariant violations: {rub_problems}",
                       "run scripts/d3_completion.py"))

    dims_ok = len(rub_orig) == EXPECTED_PRODUCTS and all(r.get(k) is not None for r in rub_orig
                                                         for k in ("faithfulness", "coverage", "consistency", "hallucination"))
    crits.append(_crit("C4", "Anchored 1-5 rubric (faithfulness, coverage, consistency, hallucination) for all 15 originals", True,
                       dims_ok, f"{len(rub_orig)} rubric rows (derived from claims, no LLM scores)", "run scripts/d3_completion.py"))

    hv = human_validation_status(p, {ce.claim.claim_id for ce in claims})
    crits.append(_crit("C5", "Single-annotator human validation: every one of the claims labelled exactly once by one named "
                             "person (valid fields, template/claim/evidence hashes match, ids match judge v4) and judge v4 compared "
                             "with those exact labels", True, hv["ok"], hv["evidence"],
                       "label the remaining claims on Summary Evaluation > Human claim validation, then run "
                       "scripts/d3_completion.py --verify (generates the comparison)"))

    nc = rep.get("negative_control_v3", {})
    comb = nc.get("metrics", {}).get("primary", {}).get("combined", {})
    nc_ok = nc.get("cases", 0) > 0 and comb.get("specificity") is not None and comb.get("recall") is not None
    crits.append(_crit("C6", "Pre-registered negative control with clean controls (detection and false-positive rates)", True, nc_ok,
                       f"{nc.get('cases', 0)} cases; recall {comb.get('recall')}, specificity {comb.get('specificity')}, "
                       f"precision {comb.get('precision')}", "run scripts/judge_v4.py report"))

    repro = rep.get("reproducibility", {})
    crits.append(_crit("C7", "Reproducibility: repeat-run agreement reported, prompt hash and config stored", True,
                       repro.get("pairs", 0) > 0 and bool(rep.get("prompt_sha256")),
                       f"{repro.get('same_status')}/{repro.get('pairs')} identical statuses; prompt sha {str(rep.get('prompt_sha256'))[:12]}",
                       "run scripts/judge_v4.py repro"))

    corr = [e for e in evals if e.evaluated_object == "corrected" and e.summary_id.endswith("-corr2")]
    have = {e.product_id for e in corr}
    crits.append(_crit("C8", "Corrected copies (-corr2) of B0BTY63SFY and B001DHKBXU evaluated separately", True,
                       all(x in have for x in REQUIRED_CORR2), f"corr2 evaluated for {sorted(have)}",
                       "run scripts/build_corrected_summaries_v2.py and judge_v4.py run --objects corrected"))

    ver = read_json(p["verification"], default=None)
    ver_ok = bool(ver) and ver.get("verify_d3_final_passed") is True and ver.get("tests_failed") == 0 and ver.get("tests_errors") == 0
    crits.append(_crit("C9", "Artefacts consistent and no collateral change (verify_d3_final.py, full test suite)", True, ver_ok,
                       (f"verify_d3_final passed: {ver.get('verify_d3_final_passed')}; tests {ver.get('tests_passed')} passed / "
                        f"{ver.get('tests_failed')} failed / {ver.get('tests_skipped')} skipped ({ver.get('recorded_at')})") if ver
                       else "no recorded verification", "run scripts/d3_completion.py --verify"))

    mandatory = [c for c in crits if c["mandatory"]]
    complete = all(c["status"] == "PASS" for c in mandatory)
    return {"source": source, "complete": complete, "criteria": crits,
            "passed": sum(c["status"] == "PASS" for c in mandatory), "mandatory": len(mandatory),
            "remaining": [f"{c['id']} {c['criterion']}: {c['remaining']}" for c in mandatory if c["status"] != "PASS"],
            "n_original_evaluations": n_orig, "mean_faithfulness": rep.get("originals", {}).get("mean_faithfulness"),
            "human_validation": hv}


def badge_text(st: dict) -> str:
    if st["complete"]:
        return (f"D3 complete · judge v4 claim-level evaluation, single-annotator human validation and negative control "
                f"verified ({st['passed']}/{st['mandatory']} criteria)")
    missing = [c["id"] for c in st["criteria"] if c["mandatory"] and c["status"] != "PASS"]
    hv = st.get("human_validation") or {}
    if missing == ["C5"]:
        return (f"D3 judge v4 evaluated ({st['passed']}/{st['mandatory']} criteria) · pending: human claim validation "
                f"({hv.get('labelled', 0)}/{hv.get('total', '?')} claims labelled"
                + (", comparison not generated" if hv.get("labelled") and hv.get("labelled") == hv.get("total") else "") + ")")
    return f"D3 incomplete ({st['passed']}/{st['mandatory']} criteria) · pending: {', '.join(missing)}"


def human_validation_status(p: dict[str, Path], judge_claim_ids: set[str]) -> dict:
    """C5: valid complete single-annotator labels AND a comparison generated from exactly that label file."""
    from evaluation.claim_gold import load_template, sha256_file, validate_store

    if not p["template"].exists():
        return {"ok": False, "labelled": 0, "total": 0, "evidence": "claim template missing"}
    template = load_template(p["template"])
    tsha = sha256_file(p["template"])
    data = read_json(p["store"], default=None)
    v = validate_store(data, template, tsha, judge_claim_ids)
    comp = read_json(p["comparison"], default=None)
    store_sha = sha256_file(p["store"]) if p["store"].exists() else None
    comp_ok = bool(comp) and comp.get("status") == "complete" and comp.get("store_sha256") == store_sha         and comp.get("template_sha256") == tsha and comp.get("claims") == v["total"]
    ok = v["complete"] and comp_ok
    problems = list(v["problems"])
    ev = (f"{v['labelled']}/{v['total']} claims labelled by {v['annotator'] or 'nobody yet'} (single annotator); "
          f"store valid: {v['valid'] or problems == []}; comparison: "
          + ("generated from the current labels" if comp_ok else ("stale - labels changed after it was generated" if comp and comp.get("status") == "complete"
                                                                  else "not generated (needs all claims labelled)"))
          + (f"; problems: {problems[:3]}" if problems else ""))
    return {"ok": ok, "labelled": v["labelled"], "total": v["total"], "annotator": v["annotator"], "store_valid": not problems,
            "comparison_current": comp_ok, "evidence": ev}


def dumps(st: dict) -> str:
    return json.dumps(st, indent=1, ensure_ascii=False, default=str)
