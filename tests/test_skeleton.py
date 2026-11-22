"""Offline, fast contract tests for the Phase 2 skeleton.

Teammates may update a test when they replace the stub it depends on, but must
never delete a test.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from src import contracts, evaluate, rag
from src.graph import run_question
from src.settings_p2 import EVAL_RETRIEVAL_PATH, EVAL_ROUTING_PATH, MAX_RETRIES

MODULES = [
    "src.contracts",
    "src.settings_p2",
    "src.llm",
    "src.rag",
    "src.topics",
    "src.trends",
    "src.classify",
    "src.nodes",
    "src.graph",
    "src.evaluate",
    "app",
]


def test_all_modules_import() -> None:
    for name in MODULES:
        assert importlib.import_module(name) is not None


def test_retrieve_returns_doc_shaped_dicts_and_honors_k_and_filters() -> None:
    docs = rag.retrieve("stub query", k=3, year_min=2021, category="cs.CL", topic=2)
    assert isinstance(docs, list) and len(docs) == 3
    expected = set(contracts.Doc.__annotations__)
    for doc in docs:
        assert set(doc) == expected
        assert doc["year"] >= 2021
        assert doc["primary_category"] == "cs.CL"
        assert doc["bertopic_topic"] == 2


def test_run_question_reaches_a_valid_answer_on_all_four_routes() -> None:
    cases = [
        ("Find papers about graph neural networks", "literature"),
        ("How has topic share changed over time?", "trend"),
        ("Show papers about CLIP and how their topic share trends over time", "hybrid"),
        ("What is the weather tomorrow?", "fallback"),
    ]
    for i, (question, expected_route) in enumerate(cases):
        state = run_question(question, thread_id=f"route-{i}")
        assert state["route"] == expected_route, (question, state.get("route"))
        assert isinstance(state.get("answer"), str) and state["answer"]


def test_retry_loop_runs_max_retries_then_flags_low_confidence() -> None:
    state = run_question("retry-test: find papers about GNNs", thread_id="retry")
    assert state["retry_count"] == MAX_RETRIES
    assert state["low_confidence"] is True


def test_trend_path_returns_trend_data_with_a_series() -> None:
    state = run_question("How has the share changed over time?", thread_id="trend")
    trend = state["trend_data"]
    assert set(trend) == set(contracts.TrendData.__annotations__)
    assert len(trend["series"]) > 0
    assert set(trend["series"][0]) == set(contracts.TrendPoint.__annotations__)


def test_ranking_metrics_on_a_tiny_hand_made_example() -> None:
    results = [["a", "b", "c"], ["x", "a", "y"]]
    gold = [["a"], ["a"]]
    assert evaluate.hit_rate_at_k(results, gold, k=1) == 0.5
    assert evaluate.hit_rate_at_k(results, gold, k=2) == 1.0
    assert evaluate.mrr(results, gold) == 0.75


def test_eval_json_files_parse() -> None:
    for path in (EVAL_ROUTING_PATH, EVAL_RETRIEVAL_PATH):
        assert Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(data, list) and data
