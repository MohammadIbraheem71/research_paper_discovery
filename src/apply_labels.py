"""Apply human relevance decisions to eval/eval_retrieval.json.

Owner: Ibraheem. This script never decides relevance itself. It only copies the
ids marked ``y`` in ``eval/label_decisions.csv`` into ``relevant_ids``. It
refuses to run while any pooled row is missing, blank or malformed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.label_review import (
    DECISIONS_PATH,
    EVAL_PATH,
    SYSTEM_TOP_K,
    build_pool,
    load_eval_items,
)

VALID_RELEVANCE = {"y", "n", "?"}


def read_decisions(path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    """Read and validate the human-filled decision worksheet.

    Purpose:
        Turn the CSV into a keyed decision map without inferring any labels.
    Args:
        path: Path to ``label_decisions.csv``.
    Returns:
        Tuple of ``{(question_id, arxiv_id): row}`` and a list of error strings.
    Owner:
        Ibraheem.
    """
    required = {"question_id", "arxiv_id", "relevant", "verifier"}
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    errors: list[str] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            errors.append(
                "CSV is missing required columns: " + ", ".join(sorted(missing_columns))
            )
            return decisions, errors

        for line_number, row in enumerate(reader, start=2):
            qid = (row.get("question_id") or "").strip()
            arxiv_id = (row.get("arxiv_id") or "").strip()
            relevance = (row.get("relevant") or "").strip().lower()
            verifier = (row.get("verifier") or "").strip()

            blank = [
                name
                for name, value in (
                    ("question_id", qid),
                    ("arxiv_id", arxiv_id),
                    ("relevant", relevance),
                    ("verifier", verifier),
                )
                if not value
            ]
            if blank:
                errors.append(f"row {line_number}: blank " + ", ".join(blank))
                continue
            if relevance not in VALID_RELEVANCE:
                errors.append(
                    f"row {line_number}: invalid relevant={relevance!r}; "
                    f"expected y, n or ?"
                )
                continue

            key = (qid, arxiv_id)
            if key in decisions:
                errors.append(f"row {line_number}: duplicate question_id/arxiv_id {key}")
                continue
            decisions[key] = {
                "relevance": relevance,
                "verifier": verifier,
                "question": (row.get("question") or "").strip(),
            }
    return decisions, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DECISIONS_PATH)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--system-k", type=int, default=SYSTEM_TOP_K)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print what would change, but do not write the eval file",
    )
    args = parser.parse_args()

    decisions, errors = read_decisions(args.decisions)
    if errors:
        print("REFUSING TO APPLY LABELS — worksheet is not complete or valid:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(2)

    items = load_eval_items(args.eval)
    pools = build_pool(items, system_k=args.system_k)
    qid_to_question = {pool["question_id"]: pool["question"] for pool in pools}
    expected = {
        (pool["question_id"], row["arxiv_id"])
        for pool in pools
        for row in pool["rows"]
    }

    missing = sorted(expected - set(decisions))
    extra = sorted(set(decisions) - expected)
    for qid, arxiv_id in missing[:20]:
        errors.append(f"missing pooled row: {qid} / {arxiv_id}")
    if len(missing) > 20:
        errors.append(f"... and {len(missing) - 20} more missing pooled rows")
    for qid, arxiv_id in extra[:20]:
        errors.append(f"unexpected row: {qid} / {arxiv_id}")
    if len(extra) > 20:
        errors.append(f"... and {len(extra) - 20} more unexpected rows")
    for (qid, _arxiv_id), row in decisions.items():
        expected_question = qid_to_question.get(qid)
        if expected_question is None:
            errors.append(f"unknown question_id: {qid}")
        elif row["question"] and row["question"] != expected_question.strip():
            errors.append(f"question text changed for {qid}")

    if errors:
        print("REFUSING TO APPLY LABELS — worksheet is not complete or valid:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(2)

    updated: list[dict[str, Any]] = [dict(item) for item in items]
    zero_relevant: list[str] = []
    print("DRY RUN — no files written" if args.dry_run else "APPLYING LABELS")
    print(f"questions: {len(pools)} | pooled rows: {len(expected)}")
    print("question_id  pooled  y   n   ?  verified  zero_relevant")
    for index, pool in enumerate(pools):
        qid = pool["question_id"]
        rows = [decisions[(qid, row["arxiv_id"])] for row in pool["rows"]]
        counts = Counter(row["relevance"] for row in rows)
        relevant_ids = [
            row["arxiv_id"]
            for row in pool["rows"]
            if decisions[(qid, row["arxiv_id"])]["relevance"] == "y"
        ]
        fully_decided = len(rows) == len(pool["rows"]) and all(
            row["relevance"] and row["verifier"] for row in rows
        )
        if not relevant_ids:
            zero_relevant.append(qid)

        print(
            f"{qid:<12} {len(rows):>6}  {counts['y']:>2} {counts['n']:>3} "
            f"{counts['?']:>3}  {str(fully_decided):>8}  "
            f"{'YES' if not relevant_ids else '-'}"
        )

        updated[index]["relevant_ids"] = relevant_ids
        if fully_decided:
            updated[index]["needs_human_verification"] = False

    if zero_relevant:
        print("questions with zero relevant ids: " + ", ".join(zero_relevant))
    else:
        print("questions with zero relevant ids: none")

    if args.dry_run:
        print("DRY RUN COMPLETE — eval file unchanged")
        return

    target = Path(args.eval)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    print(f"wrote: {target}")


if __name__ == "__main__":
    main()
