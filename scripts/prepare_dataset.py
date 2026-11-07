"""Download (if needed) and prepare the review dataset.

Usage:  python scripts/prepare_dataset.py
No API key required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from utils.data_loader import prepare_dataset  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    settings = load_settings()
    card = prepare_dataset(settings)
    print(json.dumps(card, indent=2))
    print(f"\nWrote {settings.paths.reviews_csv}")


if __name__ == "__main__":
    main()
