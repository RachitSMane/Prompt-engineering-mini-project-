# ReviewIQ — live demo guide (5–7 minutes)

> **Superseded figures (audit 2026-11-14).** This file was written for the 2026-11-12 state. Current results: judge v2 valid for 15 / 15 summaries (v1 5 / 15 preserved); negative control on 15 products with 30 planted claims, 18 / 30 detected and 12 missed (the judge is not reliable); evidence audit 13 / 15; D2 0 / 300 verified human labels; tests 269 passed, 1 skipped (final pass; judge v3 and the reference-approval workflow added). Use `MENTOR_DEMO_GUIDE.md` instead of this file.

> **Update 2026-11-13:** the Human Annotation page was rebuilt (blind labelling, verification, queue; the AI-assisted
> labels are in a read-only *legacy* tab), and Summary Evaluation now has a v1/v2 judge selector. The screenshots below
> predate these changes. Current status: `reports/DELIVERABLE_COMPLETION_REPORT.md`.

Everything below was verified on 2026-11-12 against the running app in read-only demo mode (Playwright walkthrough of all six
pages, no exceptions, `ollama ps` empty before and after, saved result files byte-identical before and after).
Screenshots of that run: `reports/screenshots/demo/01_overview.png` … `06_human_annotation.png`.

## 1. Start the app (Windows PowerShell)

```powershell
cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
$env:REVIEWIQ_DEMO_READONLY = "true"
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Then open **http://127.0.0.1:8501** in Chrome. These three lines are exactly what was run for the verification.

- `$env:REVIEWIQ_DEMO_READONLY = "true"` only lasts for that PowerShell window. A new window starts in normal mode.
- Demo mode does **not** need Ollama: no model is called, and the pages read saved files. If Ollama is running, the sidebar
  says "Local LLM ready"; that is only a status check (it lists installed models, it does not load one).
- **Warm-up:** the first page load after starting takes **18–38 s** (dataset and charts are cached on first use). Open every
  page once before the mentor arrives. After that, pages switch in a few seconds.

How to tell you are in demo mode: an amber **"DEMO MODE — READ ONLY. Saved experiment results are protected."** banner at
the top of every page, and a "Read-only demo · no LLM calls · no writes" row in the sidebar.

## 2. What demo mode does

| Action | Normal mode | Demo mode |
|---|---|---|
| Navigation, filters, tables, charts, JSON view, downloads | work | work (downloads are built in memory; no file is written) |
| Run extraction (Review Analysis, Product Intelligence) | calls `qwen2.5:3b` and appends to the extraction cache | disabled, with an explanation |
| Generate summary / summaries | calls the LLM and adds summaries | disabled |
| Judge all unjudged / Judge this summary / Re-judge | calls `llama3.2` and adds judgements | disabled |
| Re-evaluate (Prompt Experiments) | rewrites `metrics.json` and `report.md` | disabled |
| Run experiment | calls the LLM (after a confirmation checkbox) | disabled |
| Annotation Save / Save & next / Delete label / Import / Build sample | writes `annotations.json` | disabled |

The block is enforced in the backend, not just on the buttons. With the flag set:
- every result writer (`utils/storage.py` JSONL/JSON writers, annotation, export and dataset writers, experiment runs and
  metric re-evaluation) raises `ReadOnlyModeError`;
- the LLM client factory refuses to create any client, live or mock;
- the Ollama and mock clients refuse `invoke()` even if they were built some other way.

The code is in `utils/readonly.py`, and the behaviour is covered by `tests/test_demo_readonly.py` (31 tests).
`logs/pipeline.log` (diagnostic logging) is still written; it is not a result file.

## 3. Data shown: real, cached or mock

| What you see | Origin |
|---|---|
| 1,500 reviews, 15 products | Amazon Reviews 2023, Appliances (`data/processed/reviews.csv`) |
| 285 valid + 14 failed extraction records | **Real** local inference, Ollama `qwen2.5:3b`, few-shot, run 2026-11-11 (saved, shown from cache) |
| D2 metrics (3 strategies × 299 reviews) | **Real** inference, run `20261111-030912_qwen2.5-3b`, scored against the reference labels |
| 15 product summaries | **Real** `qwen2.5:3b` (saved) |
| 5 judgements + 10 recorded judge failures | **Real** `llama3.2` (saved) |
| Pandas statistics (percentages, aspect tables, charts) | Calculated from the saved records; no LLM |
| Mock data | Only when the sidebar toggle "Offline mock mode" is ON (reads `outputs/mock/`; pages then show a red/amber "Mock" banner). **Keep it OFF.** |

## 4. Demo sequence and selections (about 6 minutes)

| # | Page | Do | You should see |
|---|---|---|---|
| 1 | **Overview** (45 s) | Point at the KPI row and charts | 1,500 reviews · "285 processed (live)" · 15 of 15 products · positive 38.6% · negative 53.7% · aspect bars (performance 168 mentions, compatibility 100) · deliverables row: D2 299/299, D3 15 summaries / 5 judgements |
| 2 | **Review Analysis** (90 s) | Source defaults to **Saved results (no LLM calls)**. Product filter: **B09WWM1WLS**. Sentiment filter: **positive**. Click the first row, then open "Validated JSON" and "Repair log" | Green "Saved results · cached, not newly generated" callout naming `ollama/qwen2.5:3b` and "0 LLM calls" · KPIs 285 valid / 14 failures / 299 from cache · table shrinks 285 → 19 → 7 records · record **R6fae6e36a9fd** (sentiment positive, confidence 0.92, two aspects with verbatim evidence: quality durability negative "Construction of this item is definitely unpolished…", ease of use positive "Performance is awesome…") · Controlled failures table (14 rows) · 4 download buttons |
| 3 | **Prompt Experiments** (90 s) | Show strategy cards, then the "Confusion matrices" and "Efficiency" tabs | Status **Complete**, 299 labelled reviews · Few-shot marked "Best macro F1" **0.603** (0.6025) · zero-shot 0.438 · CoT 0.423 · the **Provenance** callout: 1 human / 298 AI-assisted labels |
| 4 | **Product Intelligence** (75 s) | Product dropdown → type **Napco** (B09WWM1WLS · 19 reviews). Scroll to "Product summary" and expand one cited point | 19 structured records, "no LLM involved" badge on the statistics · aspect chart · positive/negative verbatim evidence with review IDs · AI-generated summary whose points list cited review IDs; expanding one shows the full review text · generate buttons greyed out with the demo-mode explanation |
| 5 | **Summary Evaluation** (75 s) | **First set "Judge run" to v1** (the page now opens on v2, the fixed judge: 5 judged, 10 pending; see `reports/D3_JUDGE_EVALUATION.md`). Summary dropdown → **Napco…** (judged). Then pick **Whirlpool 3949238 Washer Lid Switch** | "5 of 15 summaries judged · **10 judge call(s) failed**" · mean scores F 3.80 / C 3.80 / K 3.00 / H 1.40 · Napco: F4 C4 K3 **H2** while the judge's own text says "does not introduce any unsupported claims" and none are flagged (the inconsistency to mention) · deterministic checks: percentages none, top-3 aspects 3/3, cited IDs belong to product: yes · Whirlpool 3949238 shows the amber **"Judge failed"** box |
| 6 | **Human Annotation** (30 s) | Show the progress row and the provenance callout; open the "Export / import / review" tab | 300 sample · 299 complete · 1 flagged · provenance "1 human, 299 AI-assisted" (the flagged review is counted there, hence 299 vs 298 on page 3) · Save/Delete disabled · "Export annotations (CSV)" download works |

**Backup product** (also judged successfully, cited IDs valid): **B001ONHE8S** Dishwasher Silverware Basket (16 records,
F4 C4 K3 H1). Avoid the default product on Product Intelligence (B001DHKBXU Whirlpool 3949238): its judge call failed. Use it
only to show what a recorded failure looks like.

## 5. Downloads verified (demo mode)

From Review Analysis (saved results), file names as offered by the browser:
`saved-dataset__few_shot__qwen2.5-3b__v1.json` (≈785 KB, 285 records), `….csv` (≈313 KB), `…_aspects.csv` (≈94 KB),
`…_failures.csv` (≈11 KB, 14 rows). Annotation page: `annotations.csv`. Summary Evaluation: judgements CSV/JSON and the
manual inspection sample. None of these writes into the project folder; the browser saves them to your Downloads folder.

## 6. Known limitations (say them; do not hide them)

1. **Reference labels are AI-assisted:** 298 of the 299 evaluable labels (1 human). D2 scores measure agreement with these
   labels, not with independent human annotators.
2. **Judge reliability:** `llama3.2` failed to produce schema-valid output for 10 of 15 summaries (kept as recorded failures,
   not re-sampled). In the 5 successful judgements, hallucination is scored 1–2 while 4 of them flag no unsupported claims,
   which contradicts itself.
3. **Negative control did not show discrimination on faithfulness:** 3 corrupted/original pairs could be scored (2 corrupted
   judgements failed). The corrupted summary got a lower *mean* score in 3/3 pairs, but a lower **faithfulness** score in
   **0/3**, and the invented-feature corruption was flagged in 0/3.
4. **Coverage:** only 299 of 1,500 reviews (the evaluation sample) were extracted; product pages therefore use 9–42 reviews
   per product, not 100.
5. **Speed:** on this laptop a few-shot extraction takes about 54 s per review (mean latency), so live inference in front of
   the mentor is impractical; that is why the demo uses saved results.
6. **Demo mode cannot show new inference.** That is by design.
