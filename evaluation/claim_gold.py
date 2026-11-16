"""Single-annotator human claim validation for judge v4 (D3).

Methodology (decided by the project owner on 2026-11-16; supersedes the two-annotator / Cohen's-kappa criterion of
``reports/proposed_revisions_2026-11-15/D3_INVESTIGATION_AND_COMPLETION_PLAN.md`` §14.4, which stays unchanged as history):

* ONE named human labels each of the 239 claims of the 15 ORIGINAL summaries exactly once, blind to judge output, against
  the evidence in the frozen claim template (``outputs/live/d3/claim_gold/<source>/claim_gold_template.csv``).
* Labels: supported, partially_supported, unsupported, contradicted, unverifiable (+ optional explanation).
* After all 239 claims are labelled, judge v4 is compared with these labels (agreement, per-class precision/recall/F1,
  confusion matrix, disagreements). This measures agreement with ONE annotator. It is not inter-annotator reliability
  and not universal ground truth.

The store (``data/annotations/d3_claim_validation.json``) holds at most one current label per claim; changing a label is an
explicit revision whose previous value is kept in ``revisions``. Every label records the template, claim-text and evidence
hashes it was made against. Nothing in this module ever creates a label by itself.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluation.claim_checks import Stats
from evaluation.negative_control_v3 import wilson
from schemas.claim_schema import SummaryEvaluationV4
from utils.readonly import ensure_writable
from utils.storage import read_json, write_json_atomic

STORE_SCHEMA_VERSION = 2
METHODOLOGY = "single_human_annotator"
LABELS = ("supported", "partially_supported", "unsupported", "contradicted", "unverifiable")
FLAGGED = frozenset({"unsupported", "contradicted"})
EXPECTED_CLAIMS = 239
MAX_EXPLANATION = 2000
#: Informational only (proposed in the 2026-11-15 plan). They are reported, never used to decide completion.
PROPOSED_THRESHOLDS = {"flagged_recall_min": 0.7, "flagged_precision_min": 0.6}
TEMPLATE_COLUMNS = ["claim_id", "product_id", "summary_id", "claim_field", "claim_text", "cited_review_ids", "evidence",
                    "annotator", "label", "note"]
REQUIRED_LABEL_FIELDS = ("claim_id", "label", "explanation", "annotator", "labelled_at", "interface", "template_sha256",
                         "claim_text_sha256", "evidence_sha256")
_AI_NAME = re.compile(r"\b(ai|a\.i\.|claude|gpt|chatgpt|llm|llama|qwen|gemini|copilot|assistant|model|bot|auto\w*|machine|"
                      r"system|judge)\b", re.I)

LABEL_GUIDE = {
    "supported": "The evidence directly says what the claim says (reasonable paraphrase is fine).",
    "partially_supported": "Part of the claim is supported, or the claim is broader, stronger or more specific than the evidence.",
    "unsupported": "The evidence does not mention what the claim says.",
    "contradicted": "The evidence says the opposite, or gives a different number.",
    "unverifiable": "The claim cannot be checked against this evidence (e.g. generic advice), or the evidence is unreadable.",
}
GUIDE_NOTES = ("Statistic sentences (overall opinion, frequent aspects) are judged against the STATISTICS block. "
               "Improvement areas and recommendations are supported only if the statistics show complaints about that aspect. "
               "Cited points (appreciated / complaints) are judged against the full text of their cited reviews only.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise_label(x: str) -> str:
    return re.sub(r"[\s-]+", "_", (x or "").strip()).lower()


def is_ai_like_name(name: str) -> bool:
    return bool(_AI_NAME.search(name or ""))


class AnnotationInputError(ValueError):
    """Invalid annotation input (label, claim id, annotator, ...)."""


class DuplicateAnnotationError(ValueError):
    """The claim already has a label; changing it needs an explicit revision."""


# --------------------------------------------------------------------------- template (export is unchanged from 2026-11-16)
def evidence_for(ce, reviews: dict[str, dict], st: Stats) -> str:
    claim = ce.claim
    if claim.kind == "point" and claim.cited_review_ids:
        parts = []
        for rid in claim.cited_review_ids:
            r = reviews.get(rid)
            if r is None or r.get("product_id") != claim.product_id:
                parts.append(f"[{rid}] (review not found for this product)")
            else:
                rating = f"{r['rating']:.0f}/5" if r.get("rating") is not None else "unknown"
                title = f"{r.get('title')}. " if r.get("title") else ""
                parts.append(f"[{rid}] (rating {rating}) {title}{r.get('text', '')}")
        return "\n\n".join(parts)
    return "STATISTICS\n" + st.render()


def template_rows(originals: list[SummaryEvaluationV4], reviews: dict[str, dict], stats: dict[str, Stats]) -> list[dict]:
    rows = []
    for ev in originals:
        if ev.evaluated_object != "original":
            raise ValueError("the claim template covers original summaries only")
        for ce in ev.claim_evaluations:
            rows.append({"claim_id": ce.claim.claim_id, "product_id": ce.claim.product_id, "summary_id": ce.claim.summary_id,
                         "claim_field": ce.claim.source_field, "claim_text": ce.claim.text,
                         "cited_review_ids": " ".join(ce.claim.cited_review_ids),
                         "evidence": evidence_for(ce, reviews, stats[ce.claim.product_id]),
                         "annotator": "", "label": "", "note": ""})
    return rows


def write_template(path: Path, rows: list[dict], *, force: bool = False) -> None:
    """The exported template is frozen: it is never overwritten unless ``force`` is given explicitly."""
    ensure_writable(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists and is frozen (labels are hashed against it); pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEMPLATE_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def load_template(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def product_statistics(template: list[dict]) -> dict[str, str]:
    """product_id -> the deterministic STATISTICS block (taken from the product's statistic claims in the template)."""
    out: dict[str, str] = {}
    for r in template:
        if r["evidence"].startswith("STATISTICS") and r["product_id"] not in out:
            out[r["product_id"]] = r["evidence"]
    return out


# --------------------------------------------------------------------------- store
def empty_store(template_path: Path) -> dict:
    return {"schema_version": STORE_SCHEMA_VERSION, "methodology": METHODOLOGY,
            "provenance": "human-created claim labels entered by one named annotator (Streamlit page or CLI); never machine-filled",
            "annotator": None, "template_file": Path(template_path).name, "template_sha256": sha256_file(template_path),
            "created_at": now(), "labels": [], "revisions": [], "events": []}


class ClaimValidationStore:
    def __init__(self, path: Path, template_path: Path):
        self.path = Path(path)
        self.template_path = Path(template_path)
        self.template = load_template(self.template_path)
        self.by_id = {r["claim_id"]: r for r in self.template}
        self.order = [r["claim_id"] for r in self.template]
        self.template_sha256 = sha256_file(self.template_path)

    # -- reading
    def data(self) -> dict:
        d = read_json(self.path, default=None)
        return d if d else empty_store(self.template_path)

    def labels(self) -> dict[str, dict]:
        return {r["claim_id"]: r for r in self.data()["labels"]}

    def annotator(self) -> str | None:
        return self.data().get("annotator")

    def progress(self) -> dict:
        done = set(self.labels())
        remaining = [c for c in self.order if c not in done]
        return {"total": len(self.order), "completed": len(done & set(self.order)), "remaining": len(remaining),
                "next_unreviewed": remaining[0] if remaining else None, "complete": not remaining and bool(self.order)}

    # -- writing
    def _check_annotator(self, data: dict, annotator: str) -> str:
        name = (annotator or "").strip()
        if len(name) < 2:
            raise AnnotationInputError("enter your name (at least 2 characters) before saving")
        if is_ai_like_name(name):
            raise AnnotationInputError(f"annotator name {name!r} looks like an AI/automatic source; labels must come from a person")
        if data.get("annotator") and data["annotator"] != name:
            raise AnnotationInputError(f"this validation uses a single annotator ({data['annotator']!r}); {name!r} cannot add labels")
        return name

    def add(self, claim_id: str, label: str, explanation: str, annotator: str, *, revise: bool = False,
            interface: str = "streamlit") -> dict:
        """Store one human label. Re-reads the file first (safe across reruns/tabs); refuses duplicates unless ``revise``."""
        ensure_writable(self.path)
        if claim_id not in self.by_id:
            raise AnnotationInputError(f"unknown claim id {claim_id!r}")
        lab = normalise_label(label)
        if lab not in LABELS:
            raise AnnotationInputError(f"invalid label {label!r}; allowed: {', '.join(LABELS)}")
        expl = (explanation or "").strip()
        if len(expl) > MAX_EXPLANATION:
            raise AnnotationInputError(f"explanation longer than {MAX_EXPLANATION} characters")
        data = self.data()
        if data.get("template_sha256") != self.template_sha256:
            raise AnnotationInputError("the claim template changed since labelling started; refusing to mix label sets")
        name = self._check_annotator(data, annotator)
        row = self.by_id[claim_id]
        rec = {"claim_id": claim_id, "label": lab, "explanation": expl, "annotator": name, "labelled_at": now(),
               "interface": interface, "template_sha256": self.template_sha256,
               "claim_text_sha256": sha256_text(row["claim_text"]), "evidence_sha256": sha256_text(row["evidence"])}
        existing = next((r for r in data["labels"] if r["claim_id"] == claim_id), None)
        if existing is not None:
            if not revise:
                raise DuplicateAnnotationError(f"{claim_id} is already labelled '{existing['label']}'; tick 'revise' to change it")
            if existing["label"] == lab and existing["explanation"] == expl:
                return existing  # nothing changed: no revision entry
            data["revisions"].append({**existing, "superseded_at": rec["labelled_at"]})
            data["labels"] = [r for r in data["labels"] if r["claim_id"] != claim_id]
            rec["revision"] = existing.get("revision", 0) + 1
        data["annotator"] = name
        data["labels"].append(rec)
        data["updated_at"] = rec["labelled_at"]
        write_json_atomic(self.path, data)
        return rec

    def log_event(self, kind: str, detail: str = "") -> None:
        """Provenance events (e.g. the judge results were opened before the labelling was complete)."""
        ensure_writable(self.path)
        data = self.data()
        data["events"].append({"at": now(), "kind": kind, "detail": detail, "labels_at_time": len(data["labels"])})
        write_json_atomic(self.path, data)


# --------------------------------------------------------------------------- validation
def validate_store(data: dict | None, template: list[dict], template_sha256: str, judge_claim_ids: set[str] | None = None) -> dict:
    """Full integrity check of a label store against the frozen template (and the judge v4 claim ids)."""
    problems: list[str] = []
    by_id = {r["claim_id"]: r for r in template}
    tmpl_ids = [r["claim_id"] for r in template]
    if len(set(tmpl_ids)) != len(tmpl_ids):
        problems.append("template has duplicate claim ids")
    if judge_claim_ids is not None and set(tmpl_ids) != set(judge_claim_ids):
        problems.append(f"template claim ids differ from the judge v4 original claims ({len(set(tmpl_ids) ^ set(judge_claim_ids))} differ)")
    if not data:
        return {"valid": not problems, "complete": False, "labelled": 0, "total": len(tmpl_ids), "problems": problems,
                "annotator": None, "note": "no label store yet (labelling not started)"}
    if data.get("schema_version") != STORE_SCHEMA_VERSION or data.get("methodology") != METHODOLOGY:
        problems.append("store schema/methodology is not the single-annotator v2 format")
    if data.get("template_sha256") != template_sha256:
        problems.append("store was created against a different claim template")
    name = data.get("annotator")
    if data.get("labels") and (not name or is_ai_like_name(name)):
        problems.append("missing or AI-like annotator name")
    ids = [r.get("claim_id") for r in data.get("labels", [])]
    dup = [c for c, n in Counter(ids).items() if n > 1]
    if dup:
        problems.append(f"duplicate annotations for {len(dup)} claim(s): {dup[:5]}")
    for r in data.get("labels", []):
        cid = r.get("claim_id")
        missing = [f for f in REQUIRED_LABEL_FIELDS if f not in r]
        if missing:
            problems.append(f"{cid}: missing fields {missing}")
            continue
        if cid not in by_id:
            problems.append(f"{cid}: not in the claim template")
            continue
        if r["label"] not in LABELS:
            problems.append(f"{cid}: invalid label {r['label']!r}")
        if r["annotator"] != name:
            problems.append(f"{cid}: annotator {r['annotator']!r} differs from the store annotator")
        if r["template_sha256"] != template_sha256:
            problems.append(f"{cid}: labelled against a different template")
        if r["claim_text_sha256"] != sha256_text(by_id[cid]["claim_text"]) or r["evidence_sha256"] != sha256_text(by_id[cid]["evidence"]):
            problems.append(f"{cid}: claim text or evidence changed since it was labelled")
        if not isinstance(r["explanation"], str) or len(r["explanation"]) > MAX_EXPLANATION:
            problems.append(f"{cid}: invalid explanation")
    labelled = len(set(ids) & set(tmpl_ids))
    complete = not problems and labelled == len(tmpl_ids) == len(ids) and len(tmpl_ids) > 0
    return {"valid": not problems, "complete": complete, "labelled": labelled, "total": len(tmpl_ids), "problems": problems,
            "annotator": name}


# --------------------------------------------------------------------------- export
EXPORT_COLUMNS = ["claim_id", "product_id", "summary_id", "claim_field", "claim_text", "cited_review_ids", "label", "explanation",
                  "annotator", "labelled_at", "interface", "revision", "template_sha256", "claim_text_sha256", "evidence_sha256"]


def export_rows(data: dict | None, template: list[dict]) -> list[dict]:
    """Completed labels in template order (never includes judge output)."""
    labels = {r["claim_id"]: r for r in (data or {}).get("labels", [])}
    out = []
    for t in template:
        r = labels.get(t["claim_id"])
        if r is None:
            continue
        out.append({"claim_id": t["claim_id"], "product_id": t["product_id"], "summary_id": t["summary_id"],
                    "claim_field": t["claim_field"], "claim_text": t["claim_text"], "cited_review_ids": t["cited_review_ids"],
                    "label": r["label"], "explanation": r["explanation"], "annotator": r["annotator"], "labelled_at": r["labelled_at"],
                    "interface": r["interface"], "revision": r.get("revision", 0), "template_sha256": r["template_sha256"],
                    "claim_text_sha256": r["claim_text_sha256"], "evidence_sha256": r["evidence_sha256"]})
    return out


def export_csv_bytes(rows: list[dict]) -> bytes:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


# --------------------------------------------------------------------------- comparison with judge v4
def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return round((po - pe) / (1 - pe), 4)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) and p is not None and r is not None else None)
    rd = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"tp": tp, "fp": fp, "fn": fn, "precision": rd(p), "recall": rd(r), "f1": rd(f1),
            "precision_wilson95": wilson(tp, tp + fp), "recall_wilson95": wilson(tp, tp + fn)}


def compare(data: dict, template: list[dict], judge: dict[str, dict], *, store_sha256: str, template_sha256: str,
            thresholds: dict | None = None) -> dict:
    """Judge v4 vs the single human annotator. ``judge``: claim_id -> {status, method, explanation}. Metrics are computed
    only when every template claim has a valid human label (no partial peeking)."""
    v = validate_store(data, template, template_sha256, set(judge))
    res = {"methodology": METHODOLOGY, "status": "complete" if v["complete"] else "incomplete", "computed_at": now(),
           "annotator": v["annotator"], "labelled": v["labelled"], "total": v["total"], "validation_problems": v["problems"],
           "store_sha256": store_sha256, "template_sha256": template_sha256,
           "interpretation": "Agreement between judge v4 and ONE human annotator. This is not inter-annotator reliability and "
                             "the human labels are not universal ground truth.",
           "thresholds": thresholds or PROPOSED_THRESHOLDS,
           "thresholds_note": "informational (proposed in the 2026-11-15 plan); completion does not depend on meeting them"}
    if not v["complete"]:
        return res
    labels = {r["claim_id"]: r for r in data["labels"]}
    by_id = {t["claim_id"]: t for t in template}
    ids = [t["claim_id"] for t in template]
    h = [labels[c]["label"] for c in ids]
    j = [judge[c]["status"].lower() for c in ids]
    n = len(ids)
    agree = sum(a == b for a, b in zip(h, j))
    matrix = {hl: {jl: sum(1 for a, b in zip(h, j) if a == hl and b == jl) for jl in LABELS} for hl in LABELS}
    per_class = {}
    for lab in LABELS:
        tp = matrix[lab][lab]
        per_class[lab] = {**_prf(tp, sum(matrix[x][lab] for x in LABELS) - tp, sum(matrix[lab].values()) - tp),
                          "human_support": sum(matrix[lab].values()), "judge_predicted": sum(matrix[x][lab] for x in LABELS)}
    hf, jf = [x in FLAGGED for x in h], [x in FLAGGED for x in j]
    tp = sum(a and b for a, b in zip(hf, jf))
    fp = sum((not a) and b for a, b in zip(hf, jf))
    fn = sum(a and (not b) for a, b in zip(hf, jf))
    tn = n - tp - fp - fn
    flagged = {**_prf(tp, fp, fn), "tn": tn, "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
               "specificity_wilson95": wilson(tn, tn + fp)}
    by_method = {}
    for m in sorted({judge[c]["method"] for c in ids}):
        sel = [k for k, c in enumerate(ids) if judge[c]["method"] == m]
        by_method[m] = {"claims": len(sel), "agreement": round(sum(h[k] == j[k] for k in sel) / len(sel), 4),
                        "agreement_wilson95": wilson(sum(h[k] == j[k] for k in sel), len(sel))}
    by_field = {}
    for fld in sorted({by_id[c]["claim_field"] for c in ids}):
        sel = [k for k, c in enumerate(ids) if by_id[c]["claim_field"] == fld]
        by_field[fld] = {"claims": len(sel), "agreement": round(sum(h[k] == j[k] for k in sel) / len(sel), 4)}
    disagreements = [{"claim_id": c, "product_id": by_id[c]["product_id"], "claim_field": by_id[c]["claim_field"],
                      "claim_text": by_id[c]["claim_text"], "human_label": h[k], "human_explanation": labels[c]["explanation"],
                      "judge_status": j[k], "judge_method": judge[c]["method"], "judge_explanation": judge[c].get("explanation", "")}
                     for k, c in enumerate(ids) if h[k] != j[k]]
    thr = res["thresholds"]
    res.update({
        "claims": n, "exact_agreement": round(agree / n, 4), "exact_agreement_wilson95": wilson(agree, n),
        "cohen_kappa_judge_vs_annotator": cohen_kappa(h, j),
        "cohen_kappa_note": "chance-corrected agreement between the judge and the single annotator (not inter-annotator reliability)",
        "human_distribution": {lab: h.count(lab) for lab in LABELS}, "judge_distribution": {lab: j.count(lab) for lab in LABELS},
        "confusion_matrix": {"rows": "human label", "columns": "judge v4 status", "matrix": matrix},
        "per_class": per_class, "flagged_binary": flagged, "by_method": by_method, "by_field": by_field,
        "disagreements": disagreements,
        "meets_informational_thresholds": (flagged["recall"] is not None and flagged["precision"] is not None and
                                           flagged["recall"] >= thr["flagged_recall_min"] and flagged["precision"] >= thr["flagged_precision_min"]),
    })
    return res
