"""LangGraph node functions.

Each node takes the full GraphState and returns only the keys it updates.
Owners are noted per function; stubs are deterministic and offline.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src import llm, rag, trends
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


VALID_ROUTES = {"literature", "trend", "hybrid", "fallback"}


def _keyword_route(question: str) -> Route:
    low = question.lower()
    has_trend = any(w in low for w in TREND_WORDS)
    has_retrieval = any(w in low for w in RETRIEVAL_WORDS)
    if not question or not (has_trend or has_retrieval):
        return "fallback"
    if has_trend and has_retrieval:
        return "hybrid"
    if has_trend:
        return "trend"
    return "literature"


def _coerce_filters(raw: Any, question: str) -> Filters:
    filters: Filters = {}
    if isinstance(raw, dict):
        for key in ("year_min", "year_max", "topic"):
            value = raw.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                filters[key] = value  # type: ignore[assignment]
            elif isinstance(value, str) and value.isdigit():
                filters[key] = int(value)  # type: ignore[assignment]
        cat = raw.get("category")
        if isinstance(cat, str) and cat in ("cs.LG", "cs.CL", "cs.CV"):
            filters["category"] = cat
    if not filters:
        filters = _extract_filters(question)
    return filters


def router_node(state: GraphState) -> GraphState:
    """Choose the route and extract metadata filters.

    Uses the LLM (JSON mode) when it is available and returns a valid route;
    the keyword rules above remain the deterministic fallback so the graph keeps
    working with Aliyan's stub LLM and with no API key.
    """
    question = (state.get("question") or "").strip()
    calls = state.get("llm_calls", 0)
    route: Route | None = None
    filters: Filters | None = None

    # The synthetic "retry-test" marker is a test hook, not a user topic; the
    # LLM router classifies it as fallback, which would skip the retry loop.
    if question and "retry-test" not in question.lower():
        prompt = (
            "You are a router for a research-paper discovery system over arXiv "
            "categories cs.LG, cs.CL and cs.CV (2018-2024). Classify the user "
            "question into exactly one route and extract optional filters.\n\n"
            "Routes:\n"
            "- literature: find/summarize/recommend specific papers.\n"
            "- trend: how a topic's share/popularity changes over time.\n"
            "- hybrid: both papers and their trend over time.\n"
            "- fallback: off-topic, empty or unanswerable.\n\n"
            "Filters (optional): year_min, year_max (integers), category "
            "(cs.LG/cs.CL/cs.CV), topic (int).\n\n"
            "Reply with JSON only: "
            '{"route": "<route>", "filters": {"year_min": ..., "year_max": ..., '
            '"category": ..., "topic": ...}}. Omit filter keys you are unsure about.'
            f"\n\nQuestion: {question}"
        )
        try:
            payload = llm.llm_json(prompt, system="Return strict JSON only.")
            calls = _calls(state)
            candidate = str(payload.get("route", "")).strip().lower()
            if candidate in VALID_ROUTES:
                route = candidate  # type: ignore[assignment]
            filters = _coerce_filters(payload.get("filters"), question)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("router_node LLM failed (%s); using keyword fallback", exc)
            route = None
            filters = None

    if route is None:
        route = _keyword_route(question)
    if filters is None:
        filters = _extract_filters(question)

    return GraphState(
        route=route,
        filters=filters,
        llm_calls=calls if question else state.get("llm_calls", 0),
    )


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
    return GraphState(
        answer=(
            "I focus on arXiv papers in cs.LG, cs.CL and cs.CV (2018-2024) and "
            "their trends over time. Try asking for papers on a topic, how a topic "
            "has changed over time, or both together."
        ),
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
    docs = state.get("docs", [])
    trend = state.get("trend_data")
    citations = [
        {"arxiv_id": d["arxiv_id"], "title": d["title"]} for d in docs
    ]

    evidence_lines = []
    if docs:
        evidence_lines.append("Retrieved papers:")
        for d in docs[:TOP_K]:
            evidence_lines.append(
                f"- {d['title']} ({d['arxiv_id']}, {d['year']}, {d['primary_category']})"
            )
    if trend:
        series = trend.get("series", [])
        if series:
            start = series[0]["quarter"]
            end = series[-1]["quarter"]
            first_share = series[0]["share"]
            last_share = series[-1]["share"]
            evidence_lines.append(
                f"Trend for {trend['label']}: share went from {first_share:.4f} "
                f"({start}) to {last_share:.4f} ({end}); slope "
                f"{trend['slope']:+.5f} per quarter (p={trend['p_value']:.4g})."
            )

    prompt = (
        "Answer the user's question using only the evidence below. Cite papers "
        "as [arXiv:ID]. If the evidence is insufficient, say so.\n\n"
        f"Question: {state.get('question', '')}\n\n"
        "Evidence:\n" + "\n".join(evidence_lines)
    )
    answer = ""
    try:
        answer = llm.llm_call(prompt, system="Be concise and grounded.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("synthesize_node LLM failed (%s)", exc)
        answer = ""
    if not answer or answer == getattr(llm, "CANNED_TEXT", None) or answer.startswith("STUB"):
        parts = []
        if docs:
            parts.append(
                "Found " + str(len(docs)) + " related papers: "
                + ", ".join(d["title"] for d in docs[:3]) + "."
            )
        if trend:
            series = trend.get("series", [])
            if series:
                parts.append(
                    f"{trend['label']} changed from {series[0]['share']:.4f} "
                    f"({series[0]['quarter']}) to {series[-1]['share']:.4f} "
                    f"({series[-1]['quarter']}) (slope {trend['slope']:+.5f}/quarter)."
                )
        answer = " ".join(parts) if parts else "Not enough evidence to answer."

    return GraphState(
        answer=answer,
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
    from src.settings_p2 import TOPIC_SHARE_PATH

    # The real table is generated by notebook 06; never fall back to the dummy.
    if not TOPIC_SHARE_PATH.exists():
        raise FileNotFoundError(
            f"Real topic-share table is required: {TOPIC_SHARE_PATH}. "
            "Run notebooks/05 and 06 first."
        )
    share_df = pd.read_csv(TOPIC_SHARE_PATH)
    trend_data = trends.trend_lookup(state.get("question") or "", share_df)
    trend_data["chart_path"] = trends.plot_topic_trend(trend_data)
    # Do not write llm_calls here: on the hybrid path this node runs in parallel
    # with query_rewrite_node, which owns that key in the same superstep.
    return GraphState(trend_data=trend_data)
