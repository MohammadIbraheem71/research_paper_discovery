"""LangGraph wiring for the four routes.

Owner: Ibraheem. Routes literature, trend, hybrid and fallback through the
teammates' real node implementations, with a SQLite checkpointer per thread.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src import nodes
from src.contracts import GraphState, Route
from src.settings_p2 import CHECKPOINT_DB_PATH, MAX_RETRIES


def _route_edge(state: GraphState) -> str:
    """Map the router's route key to a graph branch name."""
    route = state.get("route", "fallback")
    return route if route in {"literature", "trend", "hybrid", "fallback"} else "fallback"


def _grade_edge(state: GraphState) -> str:
    """Retry on a bad grade, otherwise route to the terminal answer node."""
    if state.get("grade") == "no" and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    if state.get("route") == "hybrid":
        return "trend_lookup"
    return "answer"


def get_checkpointer() -> Any:
    """Return a SQLite-backed checkpointer in the gitignored cache/ directory.

    Purpose:
        Persist GraphState per ``thread_id`` across runs.
    Args:
        None.
    Returns:
        Any, e.g. a ``SqliteSaver`` instance.
    Owner:
        Ibraheem.
    """
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # from_conn_string() is a context manager in this version; construct directly
    # so callers receive a BaseCheckpointSaver instance that LangGraph accepts.
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Compile the route graph.

    Purpose:
        Wire router -> literature/trend/hybrid/fallback branches with retries.
    Args:
        checkpointer: Optional LangGraph checkpointer; ``None`` disables persistence.
    Returns:
        Any, e.g. a compiled LangGraph ``Runnable``.
    Owner:
        Ibraheem.
    """
    builder = StateGraph(GraphState)
    for name, fn in (
        ("router", nodes.router_node),
        ("rewrite", nodes.query_rewrite_node),
        ("retrieve", nodes.retrieve_node),
        ("grade", nodes.grade_docs_node),
        ("answer", nodes.generate_answer_node),
        ("trend_lookup", nodes.trend_lookup_node),
        ("synthesize", nodes.synthesize_node),
        ("fallback", nodes.fallback_node),
    ):
        builder.add_node(name, fn)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _route_edge,
        {
            "literature": "rewrite",
            "trend": "trend_lookup",
            "hybrid": "rewrite",
            "fallback": "fallback",
        },
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _grade_edge,
        {"rewrite": "rewrite", "answer": "answer", "trend_lookup": "trend_lookup"},
    )
    builder.add_edge("trend_lookup", "synthesize")
    builder.add_edge("answer", END)
    builder.add_edge("synthesize", END)
    builder.add_edge("fallback", END)
    return builder.compile(checkpointer=checkpointer)


def run_question(question: str, thread_id: str) -> GraphState:
    """Run one question through the compiled graph.

    Purpose:
        Single entry point used by the Streamlit app, tests and notebooks.
    Args:
        question: User question.
        thread_id: Conversation id used by the checkpointer.
    Returns:
        GraphState, e.g. ``{"route": "trend", "answer": "...", "llm_calls": 1}``.
    Owner:
        Ibraheem.
    """
    graph = build_graph(get_checkpointer())
    initial = GraphState(
        question=question,
        thread_id=thread_id,
        retry_count=0,
        llm_calls=0,
    )
    return graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
