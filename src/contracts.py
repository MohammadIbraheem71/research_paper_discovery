"""Frozen Phase 2 data contracts.

Every Phase 2 module exchanges these types. Do not change this file without
telling Ibraheem first: the team's interfaces depend on it.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

Route = Literal["literature", "trend", "hybrid", "fallback"]


class Doc(TypedDict):
    """One retrieved paper."""

    arxiv_id: str
    title: str
    abstract: str
    year: int
    primary_category: str
    score: float
    bertopic_topic: Optional[int]


class Filters(TypedDict, total=False):
    """Optional metadata constraints applied at retrieval time."""

    year_min: int
    year_max: int
    category: str
    topic: int


class Citation(TypedDict):
    """A paper cited in a generated answer."""

    arxiv_id: str
    title: str


class TrendPoint(TypedDict):
    """One quarter of a topic-share time series."""

    quarter: str
    share: float
    count: int


class TrendData(TypedDict):
    """Everything needed to describe and plot one trending topic."""

    topic_id: int
    label: str
    start_quarter: str
    end_quarter: str
    series: list[TrendPoint]
    slope: float
    p_value: float
    chart_path: Optional[str]


class GraphState(TypedDict, total=False):
    """State carried through the LangGraph. Nodes return only the keys they update."""

    question: str
    thread_id: str
    route: Route
    filters: Filters
    rewritten_query: str
    docs: list[Doc]
    grade: Literal["yes", "no"]
    retry_count: int
    trend_data: TrendData
    answer: str
    citations: list[Citation]
    low_confidence: bool
    llm_calls: int
