"""Create the reproducible 300-review annotation sample and a blank CSV template.

Usage:  python scripts/build_annotation_sample.py [--force]
The sample is NOT regenerated if it already exists (annotations refer to it),
unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.annotation_manager import build_annotation_sample, load_sample, save_sample, write_template  # noqa: E402
from config import load_settings  # noqa: E402
from utils.data_loader import load_reviews  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing sample")
    args = parser.parse_args()
    s = load_settings()
    if not s.paths.reviews_csv.exists():
        sys.exit("Run scripts/prepare_dataset.py first.")
    sample = build_annotation_sample(load_reviews(s.paths.reviews_csv), s.annotation_sample_size, s.random_seed)
    written = save_sample(sample, s.paths.annotation_sample_csv, force=args.force)
    sample = load_sample(s.paths.annotation_sample_csv)
    write_template(sample, s.paths.annotations / "annotation_template.csv")
    print(("Wrote" if written else "Kept existing"), s.paths.annotation_sample_csv, f"({len(sample)} reviews)")
    print("Rating distribution:", sample["rating"].value_counts().sort_index().to_dict())
    print("Products:", sample["product_id"].nunique())


if __name__ == "__main__":
    main()
