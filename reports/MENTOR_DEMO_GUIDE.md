# ReviewIQ — mentor demonstration guide (2026-11-14)

This guide replaces the 2026-11-12 `DEMO_GUIDE.md`, `DEMO_TALK_TRACK.md`, `DEMO_QA.md` and `DEMO_CHECKLIST.md`.
Those files are kept for history and carry a "superseded" notice. Current screenshots:
`reports/screenshots/final_2026-11-14/`. The demo shows saved, real results only. No model is called and nothing
is written.

## 1. Before the mentor arrives

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
.venv\Scripts\python.exe -m pytest -q -rs          # expected: 269 passed, 1 skipped (optional langchain_groq)
$env:REVIEWIQ_DEMO_READONLY = "true"
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

- Open http://127.0.0.1:8501 and click through all six pages once (warm-up; the first load takes 20–40 s).
- Check the amber banner **"DEMO MODE — READ ONLY"** on every page.
- Ollama does not need to run. If it runs, the only request is the availability check (`GET /api/tags`).
- Keep the sidebar toggle "Offline mock mode" **off**.

## 2. Demo flow (about 7 minutes)

| # | Page | Show | Say |
|---|---|---|---|
| 1 | **Overview** (45 s) | KPI row, sentiment and aspect charts, **Project deliverables** row | "D1 implemented. D2: experiments done and verified; the D2 card counts *approved reference labels* and shows independent human labels separately. D3: executed, with known evidence and judge limitations." |
| 2 | **Review Analysis** (90 s) | Source **Saved results (no LLM calls)**; filter product **B09WWM1WLS**, sentiment positive; open a record's *Validated JSON* and *Repair log*; the controlled-failures table; the download buttons | D1: a Pydantic-validated record with verbatim evidence. Invalid output goes through local fixes, then at most 2 LLM repairs, then a stored controlled failure. Exports are verified 27 / 27 |
| 3 | **Prompt Experiments** (90 s) | The **Primary evaluation** callout (299 reviews, flagged review excluded), strategy cards (few-shot best macro-F1 **0.603**), *Confusion matrices* tab; then the **Sensitivity analysis** panel (300 reviews, 0.632) and its "not the same evaluation" note | D2 primary: few-shot is best; zero-shot and CoT label many positive reviews neutral. The sensitivity analysis adds one garbled, low-confidence neutral review and gives the same ranking. Both are agreement with approved, mostly **AI-assisted** labels, independently recomputed |
| 4 | **Product Intelligence** (75 s) | Product **Napco (B09WWM1WLS)**; the statistics marked "no LLM involved"; the summary; expand a cited point | D3 chain: extract → aggregate (pandas only) → summarise with cited review ids |
| 5 | **Summary Evaluation** (2 min) | (a) **Evidence audit**: 13 / 15, then open *Separately labelled corrections*; (b) **Negative control**: the "all planted claims" row 18 / 30, the **missed 12** callout, the semantic-review caption (labelled model-assisted); (c) the **Judge v3** section: 12 / 15 contradictory verdicts, corrected B001DHKBXU scored identically, v3 negative control 5 / 30 | "The deterministic audit caught 2 bad summaries the judge missed. v2's claim list caught 18 of 30 planted errors; without repair pressure (v3) only 5 of 30, because the judge writes its criticism in prose and leaves the claim list empty. Neither judge is reliable, and we report that." |
| 6 | **Human Annotation** (60 s) | **D2 reference dataset** row (available 300 / 300, approved, awaiting, readiness); the provenance caption; then **Independent human annotations** (300 human-created, 300 completed, 0 pending, 0 flagged) and the blind tab | "The 300 reference annotations are mostly AI-assisted and mentor-approved. The reviewer approves them here, and the approval is recorded separately, without changing provenance. Separately, our team created 300 independent blind human annotations. Second-person verification was not part of our workflow, so they are human-created but not independently verified, and the reported D2 numbers do not use them." |

**Backup product** for pages 4–5: **B001ONHE8S** (Dishwasher Silverware Basket).

## 3. Statements to make (and not to make)

| Say | Never say |
|---|---|
| "D1 is complete and independently verified." | – |
| "D2 primary result: the original 299-review run; the 300-review approved-reference evaluation is a sensitivity analysis. Both use approved, mostly AI-assisted labels" | "the labels are human" / "human ground truth" / quoting 0.632 as the primary result |
| "D3 is implemented and executed; 2 of 15 summaries fail the evidence audit; the judge is not reliable." | "the judge confirms the summaries are faithful" |
| "The semantic review of the negative control is a model-assisted reading, not human ground truth." | "a human checked the judge" |
| "Corrected copies are separate; their v3 scores belong to the corrected ids only." | quoting an original's score for a corrected copy |

## 4. If something goes wrong

- **Page shows an exception:** reload once (Streamlit cache warm-up). If it persists, use the screenshots in
  `reports/screenshots/final_2026-11-14/`.
- **No banner:** the flag was not set in *that* PowerShell window. Stop the app, set
  `$env:REVIEWIQ_DEMO_READONLY = "true"`, and start again.
- **Port in use:** add `--server.port 8502`.
- **Mentor asks for live inference:** it is possible in normal mode, but it takes about 50 s per review on this
  laptop and would write to the result stores. Offer to show the logs and stored raw outputs instead.

## 5. Where the evidence is

| Question | File |
|---|---|
| Overall status and compliance matrix | `reports/DELIVERABLE_COMPLETION_REPORT.md` |
| D1 / D2 / D3 details | `reports/D1_COMPLETION_REPORT.md`, `reports/D2_COMPLETION_REPORT.md`, `reports/D3_COMPLETION_REPORT.md` |
| Judge v1 / v2 / v3 | `reports/D3_JUDGE_EVALUATION.md` |
| Negative control | `reports/D3_NEGATIVE_CONTROL.md` |
| Tests | `reports/FINAL_TEST_REPORT.md` |
| Viva answers | `reports/VIVA_EXPLANATION.md` |
