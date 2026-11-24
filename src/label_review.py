"""Shared helpers for the human label-review workflow.

Owner: Ibraheem. This module never calls an LLM. It only loads the frozen
retrieval eval file and the real FAISS index so candidate ids and system
results can be pooled for offline human review.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src import rag
from src.settings_p2 import ROOT

EVAL_PATH = ROOT / "eval" / "eval_retrieval.json"
DECISIONS_PATH = ROOT / "eval" / "label_decisions.csv"
POOL_MD_PATH = ROOT / "results" / "label_review_pooled.md"
SYSTEM_TOP_K = 10


def question_id(index: int) -> str:
    """Stable id for an eval question, derived from its zero-based position.

    Purpose:
        Give reviewers a short id that does not depend on the question text.
    Args:
        index: Zero-based position in ``eval/eval_retrieval.json``.
    Returns:
        String such as ``q001``.
    Owner:
        Ibraheem.
    """
    if index < 0:
        raise ValueError(f"question index must be non-negative; got {index}")
    return f"q{index + 1:03d}"


def load_eval_items(path: str | Path = EVAL_PATH) -> list[dict[str, Any]]:
    """Load the retrieval eval questions without modifying them.

    Purpose:
        Read the frozen hand-label file for review-pool generation.
    Args:
        path: Path to ``eval_retrieval.json``.
    Returns:
        List of question dictionaries in their original order.
    Owner:
        Ibraheem.
    """
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return items


def md_cell(value: object) -> str:
    """Make one value safe for a Markdown table cell.

    Purpose:
        Keep abstracts readable without breaking pipe-delimited tables.
    Args:
        value: Value to render.
    Returns:
        Single-line, pipe-escaped text.
    Owner:
        Ibraheem.
    """
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text.replace("|", "\\|")


def build_pool(
    items: list[dict[str, Any]], *, system_k: int = SYSTEM_TOP_K
) -> list[dict[str, Any]]:
    """Pool existing candidate ids with the top-k system results per question.

    Purpose:
        Produce the exact review universe without asking an LLM to judge
        relevance and without exposing retrieval scores or ranks.
    Args:
        items: Questions from ``eval/eval_retrieval.json``.
        system_k: Number of top system results to include per question.
    Returns:
        One dict per question with ``question_id``, ``question`` and ``rows``.
        Each row contains only id, metadata, source and the first 400 abstract
        characters. Rows are sorted by arxiv id, so ordering carries no rank.
    Owner:
        Ibraheem.
    """
    if system_k <= 0:
        raise ValueError(f"system_k must be positive; got {system_k}")

    resources = rag.load_resources()
    docs_by_id = {doc["arxiv_id"]: doc for doc in resources["docs"]}
    pools: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        qid = question_id(index)
        question = str(item.get("question", ""))
        filters = item.get("filters") or {}
        candidate_ids = [str(value) for value in item.get("relevant_ids", [])]

        system_docs = rag.retrieve(
            question,
            k=system_k,
            year_min=filters.get("year_min"),
            year_max=filters.get("year_max"),
            category=filters.get("category"),
            topic=filters.get("topic"),
        )
        system_ids = [str(doc["arxiv_id"]) for doc in system_docs]

        candidate_set = set(candidate_ids)
        system_set = set(system_ids)
        rows: list[dict[str, Any]] = []
        for arxiv_id in sorted(candidate_set | system_set):
            doc = docs_by_id.get(arxiv_id)
            if doc is None:
                raise KeyError(
                    f"{qid}: pooled id {arxiv_id} is missing from papers.parquet"
                )
            if arxiv_id in candidate_set and arxiv_id in system_set:
                source = "both"
            elif arxiv_id in candidate_set:
                source = "candidate"
            else:
                source = "system"
            rows.append(
                {
                    "question_id": qid,
                    "arxiv_id": arxiv_id,
                    "title": str(doc.get("title", "")),
                    "year": int(doc.get("year", 0) or 0),
                    "category": str(doc.get("primary_category", "")),
                    "source": source,
                    "abstract400": str(doc.get("abstract", ""))[:400],
                }
            )

        pools.append(
            {
                "question_id": qid,
                "question": question,
                "rows": rows,
                "n_candidates": len(candidate_set),
                "n_system": len(system_set),
            }
        )
    return pools
