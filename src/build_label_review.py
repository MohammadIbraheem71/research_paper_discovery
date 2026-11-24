"""Generate the pooled Markdown review file and the blank decision CSV.

Owner: Ibraheem. No LLM is called; this only pools existing candidate ids with
real top-k FAISS results and writes a worksheet for a human to fill in.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.label_review import (
    DECISIONS_PATH,
    EVAL_PATH,
    POOL_MD_PATH,
    SYSTEM_TOP_K,
    build_pool,
    load_eval_items,
    md_cell,
)

CSV_FIELDS = [
    "question_id",
    "question",
    "arxiv_id",
    "title",
    "year",
    "category",
    "source",
    "abstract400",
    "relevant",
    "verifier",
]


def write_markdown(pools: list[dict], path: Path) -> None:
    """Write one pooled table per question, with no score or rank columns."""
    lines = [
        "# Pooled label review — PRELIMINARY (unverified labels)",
        "",
        "No relevance decision has been applied. `source` only says whether an "
        "id came from the existing candidate list, the system top 10, or both. "
        "Rows are sorted by arxiv id, so order carries no score or rank.",
        "",
    ]
    for pool in pools:
        lines.extend(
            [
                f"## {pool['question_id']}: {md_cell(pool['question'])}",
                "",
                f"Candidate ids: {pool['n_candidates']}; system top 10: "
                f"{pool['n_system']}; pooled rows: {len(pool['rows'])}.",
                "",
                "| question_id | arxiv_id | title | year | category | source | abstract (first 400 chars) |",
                "|---|---|---|---:|---|---|---|",
            ]
        )
        for row in pool["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row["question_id"]),
                        md_cell(row["arxiv_id"]),
                        md_cell(row["title"]),
                        md_cell(row["year"]),
                        md_cell(row["category"]),
                        md_cell(row["source"]),
                        md_cell(row["abstract400"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(pools: list[dict], path: Path) -> None:
    """Write one blank decision row per pooled (question_id, arxiv_id) pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pool in pools:
            for row in pool["rows"]:
                writer.writerow(
                    {
                        "question_id": row["question_id"],
                        "question": pool["question"],
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "year": row["year"],
                        "category": row["category"],
                        "source": row["source"],
                        "abstract400": row["abstract400"],
                        "relevant": "",
                        "verifier": "",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--md", type=Path, default=POOL_MD_PATH)
    parser.add_argument("--csv", type=Path, default=DECISIONS_PATH)
    parser.add_argument("--system-k", type=int, default=SYSTEM_TOP_K)
    args = parser.parse_args()

    pools = build_pool(load_eval_items(args.eval), system_k=args.system_k)
    write_markdown(pools, args.md)
    write_csv(pools, args.csv)
    total = sum(len(pool["rows"]) for pool in pools)
    print(f"questions: {len(pools)}")
    print(f"pooled rows: {total}")
    print(f"wrote: {args.md}")
    print(f"wrote: {args.csv}")
    print("relevant and verifier columns are blank; no labels were applied")


if __name__ == "__main__":
    main()
