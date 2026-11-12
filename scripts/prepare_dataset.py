"""Download (if needed) and prepare the review dataset.

Usage:  python scripts/prepare_dataset.py
No API key required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from utils.data_loader import prepare_dataset  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Download (~340 MB, once) and prepare the review subset. Settings come from .env (DATASET_*, RANDOM_SEED).")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading if raw files are missing")
    args = parser.parse_args()
    setup_logging()
    settings = load_settings()
    card = prepare_dataset(settings, download_if_missing=not args.no_download)
    print(json.dumps(card, indent=2))
    print(f"\nWrote {settings.paths.reviews_csv}")


if __name__ == "__main__":
    main()
