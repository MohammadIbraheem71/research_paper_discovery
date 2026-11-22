"""Retrieval, routing and ablation metrics.

Owners: Aliyan for retrieval/ablation helpers, Sohaib for router accuracy.
The two ranking metrics below are implemented (the tests assert exact values);
the remaining functions are offline stubs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

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
    if len(preds) != len(labels):
        raise ValueError(
            f"router_accuracy: preds ({len(preds)}) and labels ({len(labels)}) "
            "must have the same length"
        )
    if not labels:
        canonical = ["literature", "trend", "hybrid", "fallback"]
        return {
            "accuracy": 0.0,
            "confusion": {r: {c: 0 for c in canonical} for r in canonical},
        }

    canonical = ["literature", "trend", "hybrid", "fallback"]
    observed = sorted(set(labels) | set(preds))
    routes = [r for r in canonical if r in observed] + [
        r for r in observed if r not in canonical
    ]
    confusion = {r: {c: 0 for c in routes} for r in routes}
    correct = 0
    for pred, gold in zip(preds, labels):
        confusion.setdefault(gold, {c: 0 for c in routes})
        confusion[gold][pred] = confusion[gold].get(pred, 0) + 1
        if pred == gold:
            correct += 1
    return {
        "accuracy": correct / len(labels),
        "confusion": confusion,
    }


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
    logger.warning("STUB: run_ablation")
    # TODO(Aliyan): replay each question through build_graph() with components
    # disabled and score with hit_rate_at_k/mrr.
    return pd.DataFrame(
        [
            {"question": q, "variant": "stub", "hit@5": 0.0, "rr": 0.0, "llm_calls": 0}
            for q in questions
        ],
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
    logger.warning("STUB: citation_sample_csv")
    # TODO(Aliyan): sample n citations with question/answer/arxiv_id/title columns
    # and write them to path for hand labelling.
    return str(path)
