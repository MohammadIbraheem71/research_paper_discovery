"""LangGraph node functions.

Each node takes the full GraphState and returns only the keys it updates.
Owners are noted per function; stubs are deterministic and offline.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src import rag, trends
from src.contracts import Filters, GraphState, Route
from src.settings_p2 import TOP_K

logger = logging.getLogger(__name__)

TREND_WORDS = ("trend", "over time", "growing", "grew", "popular", "rising", "declining", "quarterly")
RETRIEVAL_WORDS = (
    "paper", "papers", "research", "find", "show", "related", "similar", "about",
    "summarize", "cite", "literature", "arxiv", "abstract", "work",
)


def _calls(state: GraphState, n: int = 1) -> int:
    """Return the accumulated LLM call count plus n."""
    return state.get("llm_calls", 0) + n


def _extract_filters(question: str) -> Filters:
    """Pull simple year/category filters out of a question (stub keyword rules)."""
    filters: Filters = {}
    years = re.findall(r"\b(20\d{2})\b", question)
    if len(years) >= 2:
        filters["year_min"], filters["year_max"] = int(years[0]), int(years[1])
    elif years:
        filters["year_min"] = int(years[0])
    for cat in ("cs.LG", "cs.CL", "cs.CV"):
        if cat.lower() in question.lower():
            filters["category"] = cat
            break
    return filters


def router_node(state: GraphState) -> GraphState:
    """Choose the route and extract metadata filters.

    Purpose:
        Send each question down the literature, trend, hybrid or fallback path.
    Args:
        state: GraphState containing ``question``.
    Returns:
        GraphState, e.g. ``{"route": "literature", "filters": {}, "llm_calls": 0}``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: router_node")
    # TODO(Sohaib): replace keyword rules with the real router LLM call.
    question = (state.get("question") or "").strip()
    low = question.lower()
    has_trend = any(w in low for w in TREND_WORDS)
    has_retrieval = any(w in low for w in RETRIEVAL_WORDS)
    if not question or not (has_trend or has_retrieval):
        route: Route = "fallback"
    elif has_trend and has_retrieval:
        route = "hybrid"
    elif has_trend:
        route = "trend"
    else:
        route = "literature"
    return GraphState(route=route, filters=_extract_filters(question), llm_calls=state.get("llm_calls", 0))


def fallback_node(state: GraphState) -> GraphState:
    """Answer off-topic or empty questions without calling an LLM.

    Purpose:
        Give a safe response when the router cannot find a useful path.
    Args:
        state: GraphState containing ``question``.
    Returns:
        GraphState, e.g. ``{"answer": "...", "citations": [], "low_confidence": True}``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: fallback_node")
    # TODO(Sohaib): write a fixed clarification message in the project's voice.
    return GraphState(
        answer="STUB: please ask about arXiv cs.LG/cs.CL/cs.V papers or their trends.",
        citations=[],
        low_confidence=True,
        llm_calls=state.get("llm_calls", 0),
    )


def synthesize_node(state: GraphState) -> GraphState:
    """Combine retrieval and trend evidence into one answer.

    Purpose:
        Merge the two hybrid branches before the final response.
    Args:
        state: GraphState with optional ``docs`` and ``trend_data``.
    Returns:
        GraphState, e.g. ``{"answer": "...", "citations": [], "llm_calls": 1}``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: synthesize_node")
    # TODO(Sohaib): call llm.llm_call with both docs and trend_data summarized.
    citations = [
        {"arxiv_id": d["arxiv_id"], "title": d["title"]} for d in state.get("docs", [])
    ]
    return GraphState(
        answer="STUB SYNTHESIS: retrieval and trend evidence go here.",
        citations=citations,
        low_confidence=state.get("grade") == "no",
        llm_calls=_calls(state),
    )


def query_rewrite_node(state: GraphState) -> GraphState:
    """Rewrite the question into a retrieval-friendly query.

    Purpose:
        Improve recall on the next retrieve step and record a retry.
    Args:
        state: GraphState containing ``question`` and optional ``retry_count``.
    Returns:
        GraphState, e.g. ``{"rewritten_query": "...", "retry_count": 1, "llm_calls": 1}``.
    Owner:
        Aliyan.
    """
    question = state.get("question") or ""
    if not question:
        return GraphState(
            rewritten_query="",
            retry_count=state.get("retry_count", 0) + 1,
            llm_calls=_calls(state),
        )
    system = (
        "Rewrite the question as a concise retrieval query. "
        "Return JSON with keys rewritten_query and filters. "
        "filters may contain year_min, year_max, category, topic."
    )
    try:
        from src.llm import llm_json

        result = llm_json(
            f"Question: {question}\nCurrent filters: {state.get('filters', {})}",
            system=system,
        )
        rewritten = str(result.get("rewritten_query") or question).strip()
        refined = result.get("filters")
        updates: dict = {}
        if isinstance(refined, dict):
            filters = dict(state.get("filters", {}))
            for key in ("year_min", "year_max", "topic"):
                if key in refined:
                    try:
                        filters[key] = int(refined[key])
                    except (TypeError, ValueError):
                        pass
            if isinstance(refined.get("category"), str):
                filters["category"] = refined["category"]
            updates["filters"] = filters
        return GraphState(
            rewritten_query=rewritten,
            retry_count=state.get("retry_count", 0) + 1,
            llm_calls=_calls(state),
            **updates,
        )
    except Exception as exc:
        logger.warning("Query rewrite unavailable: %s", exc)
        return GraphState(
            rewritten_query=question,
            retry_count=state.get("retry_count", 0) + 1,
            llm_calls=state.get("llm_calls", 0),
        )
    return GraphState(
        rewritten_query=f"{question} (stub rewrite)".strip(),
        retry_count=state.get("retry_count", 0) + 1,
        llm_calls=_calls(state),
    )


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve papers using the rewritten query and extracted filters.

    Purpose:
        Populate ``docs`` for grading and answer generation.
    Args:
        state: GraphState with ``rewritten_query`` and optional ``filters``.
    Returns:
        GraphState, e.g. ``{"docs": [Doc, ...], "llm_calls": 0}``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: retrieve_node")
    # TODO(Aliyan): call rag.retrieve(**filters) with state["rewritten_query"].
    filters = state.get("filters", {})
    docs = rag.retrieve(
        state.get("rewritten_query") or state.get("question") or "",
        k=TOP_K,
        year_min=filters.get("year_min"),
        year_max=filters.get("year_max"),
        category=filters.get("category"),
        topic=filters.get("topic"),
    )
    return GraphState(docs=docs, llm_calls=state.get("llm_calls", 0))


def grade_docs_node(state: GraphState) -> GraphState:
    """Grade whether the retrieved docs are good enough.

    Purpose:
        Drive the retry loop. The stub always fails ``retry-test`` questions so
        the loop can be exercised offline.
    Args:
        state: GraphState containing ``question`` and ``docs``.
    Returns:
        GraphState, e.g. ``{"grade": "yes", "llm_calls": 1}``.
    Owner:
        Aliyan.
    """
    question = (state.get("question") or "").lower()
    docs = state.get("docs", [])
    if "retry-test" in question:
        grade = "no"
    elif not docs:
        grade = "no"
    else:
        system = (
            "Grade whether the retrieved documents are relevant enough to answer "
            "the question. Return JSON with exactly one key: grade, where grade "
            "is \"yes\" or \"no\"."
        )
        doc_summary = "\n".join(
            f"[arXiv:{doc['arxiv_id']}] {doc['title']}: {doc['abstract'][:500]}"
            for doc in docs
        )
        try:
            from src.llm import llm_json

            result = llm_json(
                f"Question: {state.get('question') or ''}\n\nDocuments:\n{doc_summary}",
                system=system,
            )
            grade = "yes" if str(result.get("grade", "")).lower() == "yes" else "no"
        except Exception as exc:
            logger.warning("Document grading unavailable: %s", exc)
            grade = "no"
    return GraphState(grade=grade, llm_calls=_calls(state))


def generate_answer_node(state: GraphState) -> GraphState:
    """Produce the final grounded answer and its citations.

    Purpose:
        Terminate the literature/retry path with an answer contract payload.
    Args:
        state: GraphState with ``question``, ``docs`` and optional ``grade``.
    Returns:
        GraphState, e.g. ``{"answer": "...", "citations": [...],
        "low_confidence": False, "llm_calls": 2}``.
    Owner:
        Aliyan.
    """
    answer, citations = rag.generate_answer(
        state.get("question") or "", state.get("docs", [])
    )
    return GraphState(
        answer=answer,
        citations=citations,
        low_confidence=state.get("grade") == "no",
        llm_calls=_calls(state),
    )


def trend_lookup_node(state: GraphState) -> GraphState:
    """Wrap :func:`src.trends.trend_lookup` for the graph.

    Purpose:
        Attach a TrendData payload for trend and hybrid questions.
    Args:
        state: GraphState containing ``question``.
    Returns:
        GraphState, e.g. ``{"trend_data": TrendData(...), "llm_calls": 0}``.
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: trend_lookup_node")
    # TODO(Ibraheem): call compute_topic_share on real data/doc_topics.parquet once
    # it exists; the stub uses the tracked dummy share table.
    share_df = trends.compute_topic_share(pd.DataFrame())
    # Do not write llm_calls here: on the hybrid path this node runs in parallel
    # with query_rewrite_node, which owns that key in the same superstep.
    return GraphState(
        trend_data=trends.trend_lookup(state.get("question") or "", share_df),
    )
