"""Capture screenshots of the RUNNING Streamlit app (all six pages) for the final report.

Start the app first, bound to localhost:
    streamlit run app.py --server.address 127.0.0.1 --server.port 8601 --server.headless true
then:
    python scripts/capture_screenshots.py --url http://127.0.0.1:8601

Optional tool dependency (not needed by the app): ``pip install playwright``. It drives the locally installed
Chrome (or Edge with --channel msedge), so no browser download is required. Screenshots are written to
reports/screenshots/<nn>_<page>.png. Nothing is mocked: the pages render whatever outputs/live/ contains.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ("01_overview", ""),
    ("02_review_analysis", "review_analysis"),
    ("03_prompt_experiments", "prompt_comparison"),
    ("04_product_intelligence", "product_insights"),
    ("05_summary_evaluation", "summary_evaluation"),
    ("06_human_annotation", "annotation_page"),
]


def wait_for_app(page, timeout_s: float = 90) -> None:
    """Wait until Streamlit has finished running the script (no running indicator, no spinner)."""
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout_s * 1000)
    main = '[data-testid="stMain"] [data-testid="stElementContainer"]'
    page.wait_for_selector(main, timeout=timeout_s * 1000)
    deadline = time.time() + timeout_s
    last, stable = -1, 0
    while time.time() < deadline and stable < 6:  # element count unchanged for ~6 s and nothing running
        n = page.locator(main).count()
        busy = (page.locator('[data-testid="stStatusWidget"] >> text=Running').count() + page.locator('[data-testid="stSpinner"]').count()
                + page.locator('[data-testid="stSkeleton"]').count())  # skeleton placeholders = element still loading
        stable = stable + 1 if (n == last and not busy) else 0
        last = n
        time.sleep(1)
    time.sleep(4)  # let Plotly charts finish drawing


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8601")
    p.add_argument("--channel", default="chrome", help="installed browser channel: chrome | msedge")
    p.add_argument("--out", default=str(ROOT / "reports" / "screenshots"))
    p.add_argument("--height", type=int, default=2000)
    args = p.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright is not installed: pip install playwright")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": args.height}, color_scheme="dark")
        for name, path in PAGES:
            page.goto(f"{args.url.rstrip('/')}/{path}", wait_until="domcontentloaded")
            wait_for_app(page)
            errors = page.locator('[data-testid="stException"]').count()
            target = out / f"{name}.png"
            page.screenshot(path=str(target), full_page=True)
            print(f"{target}  ({'EXCEPTION SHOWN ON PAGE' if errors else 'no exception'})")
        browser.close()


if __name__ == "__main__":
    main()
