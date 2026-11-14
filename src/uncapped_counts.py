"""A1b: count papers per primary category and year *before* the 40k cap.

This is the same filter as ``src/preprocessing.py`` (primary category in
``CATEGORIES``, first-version year in ``YEAR_MIN``..``YEAR_MAX``) but without
``cap_per_category``.  It answers the class-imbalance question: how large was
each category on arXiv before we balanced the sample?

Run on Kaggle (the arXiv snapshot is attached as an input):

    python src/uncapped_counts.py --out /kaggle/working

Or locally:

    python src/uncapped_counts.py --path /path/to/arxiv-metadata-oai-snapshot.json --out results
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import CATEGORIES, MAX_PER_CATEGORY, YEAR_MIN, YEAR_MAX  # noqa: E402


def find_snapshot(path: str | None = None) -> str:
    """Return the arXiv JSON-lines snapshot path."""
    if path:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return path

    search_roots = ["/kaggle/input", str(REPO_ROOT / "data")]
    for root in search_roots:
        hits = glob.glob(f"{root}/**/arxiv-metadata-oai-snapshot.json", recursive=True)
        if hits:
            return sorted(hits)[0]

    raise FileNotFoundError(
        "Could not find arxiv-metadata-oai-snapshot.json. Attach the "
        "Cornell-University/arxiv dataset on Kaggle or pass --path."
    )


def count_uncapped(path: str) -> tuple[Counter, Counter, Counter, int, int]:
    """Stream the snapshot and count kept papers before any cap."""
    allowed = set(CATEGORIES)
    by_category: Counter = Counter()
    by_category_year: Counter = Counter()
    by_year: Counter = Counter()
    scanned = kept = skipped = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            try:
                record = json.loads(line)
                categories = record["categories"].split()
                if not categories or categories[0] not in allowed:
                    continue
                created = record["versions"][0]["created"]
                # The Cornell snapshot stores RFC-2822 dates, e.g.
                # "Fri, 01 May 2009 01:52:14 GMT"; pandas also accepts ISO dates.
                created_dt = pd.to_datetime(created, errors="coerce", utc=True)
                if pd.isna(created_dt):
                    raise ValueError(f"unparseable created date: {created!r}")
                year = int(created_dt.year)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                skipped += 1
                continue

            if not (YEAR_MIN <= year <= YEAR_MAX):
                continue

            category = categories[0]
            by_category[category] += 1
            by_category_year[(category, year)] += 1
            by_year[year] += 1
            kept += 1

    if kept == 0:
        raise RuntimeError(
            "No papers matched the category/year filter. Check that the input "
            "is the raw arXiv snapshot and that its version dates are parseable."
        )

    return by_category, by_category_year, by_year, scanned, skipped


def build_tables(by_category, by_category_year):
    """Turn the counters into tidy DataFrames."""
    category_df = pd.DataFrame(
        [(category, by_category.get(category, 0)) for category in CATEGORIES],
        columns=["primary_category", "papers"],
    )
    total = int(category_df["papers"].sum())
    category_df["share_pct"] = (category_df["papers"] / total * 100).round(2)
    category_df["after_cap"] = category_df["papers"].clip(upper=MAX_PER_CATEGORY)
    category_df["removed_by_cap"] = category_df["papers"] - category_df["after_cap"]

    year_df = pd.DataFrame(
        [
            (category, year, by_category_year.get((category, year), 0))
            for category in CATEGORIES
            for year in range(YEAR_MIN, YEAR_MAX + 1)
        ],
        columns=["primary_category", "year", "papers"],
    )
    pivot = (
        year_df.pivot(index="year", columns="primary_category", values="papers")
        .reindex(columns=CATEGORIES, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    pivot["total"] = pivot.sum(axis=1)
    return category_df, year_df, pivot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--out", default="results", help="folder for the CSV outputs")
    args = parser.parse_args()

    snapshot = find_snapshot(args.path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {snapshot}")
    by_category, by_category_year, by_year, scanned, skipped = count_uncapped(snapshot)
    category_df, year_df, pivot = build_tables(by_category, by_category_year)

    print(f"\nScanned lines: {scanned:,} | kept after category/year filter: "
          f"{int(category_df['papers'].sum()):,} | malformed/skipped: {skipped:,}")
    print("\nUncapped papers per primary category (before the 40,000 cap):")
    print(category_df.to_string(index=False))
    print("\nUncapped papers per category and year:")
    print(pivot.to_string())
    print("\nAll kept papers per year:")
    print(pd.Series(by_year, name="papers").sort_index().to_string())

    category_df.to_csv(out_dir / "uncapped_counts.csv", index=False)
    year_df.to_csv(out_dir / "uncapped_counts_by_year.csv", index=False)
    pivot.to_csv(out_dir / "uncapped_counts_pivot.csv")
    print(f"\nSaved CSVs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
