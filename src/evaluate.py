"""Retrieval, routing and ablation metrics.

Owners: Aliyan for retrieval/ablation helpers, Sohaib for router accuracy.
The two ranking metrics below are implemented (the tests assert exact values);
the remaining functions are offline stubs.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
import uuid

import pandas as pd

from src.settings_p2 import EVAL_RETRIEVAL_PATH

logger = logging.getLogger(__name__)


def hit_rate_at_k(results: list[list[str]], gold: list[list[str]], k: int) -> float:
    """Fraction of questions with at least one relevant id in the top k.

    Purpose:
        Measure retrieval recall at a fixed cutoff.
    Args:
        results: Ranked arxiv ids per question, e.g. ``[["a", "b"], ["c"]]``.
        gold: Relevant arxiv ids per question, e.g. ``[["a"], ["b"]]``.
        k: Cutoff, e.g. ``5``.
    Returns:
        float in [0, 1], e.g. ``0.5``.
    Owner:
        Aliyan.
    """
    if not results:
        return 0.0
    hits = sum(
        1 for ranked, relevant in zip(results, gold)
        if set(ranked[:k]) & set(relevant)
    )
    return hits / len(results)


def mrr(results: list[list[str]], gold: list[list[str]]) -> float:
    """Mean reciprocal rank of the first relevant result per question.

    Purpose:
        Reward putting the right paper near the top.
    Args:
        results: Ranked arxiv ids per question, e.g. ``[["a", "b"], ["c", "b"]]``.
        gold: Relevant arxiv ids per question, e.g. ``[["b"], ["b"]]``.
    Returns:
        float in [0, 1], e.g. ``0.75`` for ranks 1 and 2.
    Owner:
        Aliyan.
    """
    if not results:
        return 0.0
    total = 0.0
    for ranked, relevant in zip(results, gold):
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                total += 1.0 / rank
                break
    return total / len(results)


def router_accuracy(preds: list[str], labels: list[str]) -> dict:
    """Summarize route prediction accuracy and confusion.

    Purpose:
        Score the router against the hand-labelled eval_routing.json.
    Args:
        preds: Predicted routes, e.g. ``["trend", "literature"]``.
        labels: Gold routes, e.g. ``["trend", "trend"]``.
    Returns:
        dict, e.g. ``{"accuracy": 0.5, "confusion": {"trend": {"trend": 1}}}``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: router_accuracy")
    # TODO(Sohaib): compute accuracy and a nested route -> route -> count dict.
    return {"accuracy": 0.0, "confusion": {}}


def run_ablation(questions: list[str]) -> pd.DataFrame:
    """Run the retrieval ablations and return a tidy results table.

    Purpose:
        Compare variants with and without rewrite/grading components.
    Args:
        questions: Questions to replay, e.g. ``["what papers ..."]``.
    Returns:
        DataFrame with columns ``[question, variant, hit@5, rr, llm_calls]``, e.g.
        one row per question and variant.
    Owner:
        Aliyan.
    """
    from src import rag
    from src.graph import build_graph

    gold_by_question: dict[str, list[str]] = {}
    if EVAL_RETRIEVAL_PATH.exists():
        for item in json.loads(EVAL_RETRIEVAL_PATH.read_text(encoding="utf-8")):
            gold_by_question[str(item.get("question", ""))] = list(
                item.get("relevant_ids", [])
            )

    rows: list[dict] = []
    for question in questions:
        gold = gold_by_question.get(question, [])

        graph_state = build_graph().invoke(
            {
                "question": question,
                "thread_id": f"ablation-with-{uuid.uuid4()}",
                "retry_count": 0,
                "llm_calls": 0,
            }
        )
        with_ids = [doc["arxiv_id"] for doc in graph_state.get("docs", [])]
        rr = mrr([with_ids], [gold]) if gold else 0.0
        rows.append({
            "question": question,
            "variant": "with rewrite+grade",
            "hit@5": hit_rate_at_k([with_ids], [gold], 5) if gold else 0.0,
            "rr": rr,
            "llm_calls": int(graph_state.get("llm_calls", 0)),
        })

        without_ids = [
            doc["arxiv_id"]
            for doc in rag.retrieve(question, k=5)
        ]
        rr = mrr([without_ids], [gold]) if gold else 0.0
        rows.append({
            "question": question,
            "variant": "without rewrite+grade",
            "hit@5": hit_rate_at_k([without_ids], [gold], 5) if gold else 0.0,
            "rr": rr,
            "llm_calls": 0,
        })
    return pd.DataFrame(
        rows,
        columns=["question", "variant", "hit@5", "rr", "llm_calls"],
    )


def citation_sample_csv(n: int = 20, path: str | Path = "eval/citation_sample.csv") -> str:
    """Write a small human-review sample of answer citations.

    Purpose:
        Produce the manual citation-verification worksheet.
    Args:
        n: Number of rows to sample.
        path: Destination CSV path.
    Returns:
        str, e.g. ``"eval/citation_sample.csv"`` (no file is written by the stub).
    Owner:
        Aliyan.
    """
    from src import rag

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if not EVAL_RETRIEVAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing retrieval evaluation file: {EVAL_RETRIEVAL_PATH}"
        )
    eval_items = json.loads(EVAL_RETRIEVAL_PATH.read_text(encoding="utf-8"))
    sampled = eval_items[:n]
    for item in sampled:
        question = str(item.get("question", "")).strip()
        filters = item.get("filters", {})
        docs = rag.retrieve(
            question,
            k=5,
            year_min=filters.get("year_min"),
            year_max=filters.get("year_max"),
            category=filters.get("category"),
            topic=filters.get("topic"),
        )
        answer, citations = rag.generate_answer(question, docs)
        if citations:
            for citation in citations:
                rows.append({
                    "question": question,
                    "answer": answer,
                    "arxiv_id": citation["arxiv_id"],
                    "title": citation["title"],
                    "supported?": "",
                })
        else:
            rows.append({
                "question": question,
                "answer": answer,
                "arxiv_id": "",
                "title": "",
                "supported?": "",
            })
    pd.DataFrame(
        rows,
        columns=["question", "answer", "arxiv_id", "title", "supported?"],
    ).to_csv(destination, index=False)
    return str(destination)
