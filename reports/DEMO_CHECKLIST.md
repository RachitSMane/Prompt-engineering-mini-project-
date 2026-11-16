# ReviewIQ — pre-demo checklist and troubleshooting

> **Superseded figures (audit 2026-11-14).** This file was written for the 2026-11-12 state. Current results: judge v2 valid for 15 / 15 summaries (v1 5 / 15 preserved); negative control on 15 products with 30 planted claims, 18 / 30 detected and 12 missed (the judge is not reliable); evidence audit 13 / 15; D2 0 / 300 verified human labels; tests 269 passed, 1 skipped (final pass; judge v3 and the reference-approval workflow added). Use `MENTOR_DEMO_GUIDE.md` instead of this file.

## The day before
- [ ] Run the tests (no model calls, about 1 minute):
      `.venv\Scripts\python.exe -m pytest -q -rs` → expected **269 passed, 1 skipped** (the skip is the optional `langchain_groq`
      package, which is not used).
- [ ] Charge the laptop and keep it plugged in. Close heavy apps. Free RAM is not critical in demo mode (no model is loaded).
- [ ] Optional: copy `reports/screenshots/demo/*.png` somewhere you can open offline, as a fallback if the app won't start.

## 15 minutes before
- [ ] Open PowerShell and start the app in demo mode (exact, verified):
      ```powershell
      cd "C:\Users\Lenovo\Downloads\prompt mini project\Prompt-engineering-mini-project-"
      $env:REVIEWIQ_DEMO_READONLY = "true"
      .venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
      ```
- [ ] Open **http://127.0.0.1:8501** in Chrome. Zoom 90–100 %. (The verification screenshots were taken at 1440 px width.)
- [ ] Check the amber **"DEMO MODE — READ ONLY"** banner is at the top, and the sidebar shows "Read-only demo".
- [ ] Check the **"Offline mock mode"** toggle in the sidebar is **OFF**, and the page badge (top right) says **Live**, not Mock.
- [ ] **Warm-up:** click through all six pages once. The first load takes 18–38 s; after that it's a few seconds.
- [ ] Overview shows **1,500** reviews and **285 processed (live)**. If it says "mock" or shows other numbers, see Troubleshooting.
- [ ] Pre-select nothing. Streamlit keeps selections per browser tab, so a fresh tab starts from the defaults in `DEMO_GUIDE.md`.
- [ ] Have `DEMO_GUIDE.md` §4 (the selections) and `DEMO_QA.md` open on a second screen or printed.

## Do NOT during the demo
- Do not switch on **Offline mock mode**: pages would then show mock fixtures (clearly labelled, but not results).
- Do not start a second app instance without the flag to "quickly show inference". Live extraction takes about 40–55 s per
  review here, and new extraction files are included in the Overview numbers.
- Do not run `scripts/*.py` in the demo window.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No amber banner; Run/Generate buttons are enabled | The flag wasn't set in *this* PowerShell window | Stop the app (Ctrl+C), run `$env:REVIEWIQ_DEMO_READONLY = "true"`, start again. Check with `echo $env:REVIEWIQ_DEMO_READONLY` → `true` |
| `Port 8501 is already in use` | An earlier instance is still running | Use `--server.port 8502` and open that URL, or stop the old one: `Get-NetTCPConnection -LocalPort 8501 -State Listen \| % { Stop-Process -Id $_.OwningProcess }` |
| Page badge says **Mock**, Overview numbers differ | The mock toggle is on | Switch "Offline mock mode" off in the sidebar |
| Sidebar says "Ollama not running" | Ollama isn't started | **Harmless in demo mode**: all pages read saved files. (Starting Ollama is optional: `ollama serve`.) |
| A page shows grey placeholder boxes for a while | First render, or the laptop is busy | Wait 10–20 s; that's why you warm up. Pressing **R** in the browser reruns the page |
| `ModuleNotFoundError` or `python` not found | Wrong folder or venv | Run the `cd` line first; always use `.venv\Scripts\python.exe` (Python 3.11.9) |
| Red exception box on a page | Not seen in verification | Note the message, switch to the screenshot fallback in `reports/screenshots/demo/`, and say so honestly |
| Browser shows "Connection refused" | App stopped (the PowerShell window was closed) | Restart with the three commands above |

## After the demo
- Stop the app with **Ctrl+C** in its PowerShell window, or close the window. Closing it also clears the demo flag.
- Nothing needs restoring: demo mode cannot modify `outputs/`, `data/annotations/` or the reports. A test checks this
  (`tests/test_demo_readonly.py::test_real_results_unchanged_after_demo_session`).
