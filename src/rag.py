"""Retrieval-augmented generation resources and operations.

Owner: Aliyan. Stubs are offline and need neither a FAISS index nor embeddings.
"""
from __future__ import annotations

import logging
from typing import Optional, TypedDict

import numpy as np

from src.contracts import Doc
from src.settings_p2 import EMBEDDING_DIM, TOP_K

logger = logging.getLogger(__name__)


class Resources(TypedDict):
    """Loaded retrieval assets. The stub uses ``None`` for real index objects."""

    index: Optional[object]
    docs: list[Doc]
    doc_ids: list[str]


def _dummy_doc(i: int, year_min: Optional[int], year_max: Optional[int],
               category: Optional[str], topic: Optional[int]) -> Doc:
    """Build one dummy Doc that visibly respects the requested filters."""
    year = year_min if year_min is not None else 2020
    if year_max is not None:
        year = min(year, year_max)
    return Doc(
        arxiv_id=f"dummy-{i:04d}",
        title=f"Stub paper {i}",
        abstract="Stub abstract used until the real retriever is implemented.",
        year=year,
        primary_category=category or "cs.LG",
        score=1.0 / (i + 1),
        bertopic_topic=topic,
    )


def load_resources() -> Resources:
    """Load the FAISS index, paper metadata and topic assignments.

    Purpose:
        Provide one cached entry point for all retrieval assets.
    Args:
        None.
    Returns:
        Resources, e.g. ``{"index": None, "docs": [Doc, ...], "doc_ids": [...]}``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: load_resources")
    # TODO(Aliyan): load data/embeddings.npy + data/doc_topics.parquet and build
    # the FAISS index; cache the result on module level.
    docs = [_dummy_doc(i, None, None, None, None) for i in range(TOP_K)]
    return Resources(index=None, docs=docs, doc_ids=[d["arxiv_id"] for d in docs])


def build_index(embeddings: np.ndarray) -> object:
    """Build a FAISS index from an (n, EMBEDDING_DIM) float array.

    Purpose:
        Turn full-corpus embeddings into a searchable index.
    Args:
        embeddings: Array of shape ``(n, 384)``.
    Returns:
        object, e.g. ``None`` in the stub.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: build_index")
    # TODO(Aliyan): create faiss.IndexFlatIP / IndexFlatL2 with EMBEDDING_DIM and add
    # the vectors; assert embeddings.shape[1] == EMBEDDING_DIM.
    assert embeddings.ndim == 2
    return None


def embed_query(text: str) -> np.ndarray:
    """Embed a query with the same model used for the corpus.

    Purpose:
        Map free text into the shared 384-dimensional embedding space.
    Args:
        text: Query string.
    Returns:
        np.ndarray of shape ``(384,)``, dtype float32, e.g. ``np.zeros(384)``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: embed_query")
    # TODO(Aliyan): encode with all-MiniLM-L6-v2 and L2-normalize.
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def retrieve(
    query: str,
    k: int = TOP_K,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    category: Optional[str] = None,
    topic: Optional[int] = None,
) -> list[Doc]:
    """Return up to k papers matching a query and metadata filters.

    Purpose:
        Top-k retrieval with hard year/category/topic constraints.
    Args:
        query: Free-text query.
        k: Maximum number of papers.
        year_min: Inclusive lower year bound.
        year_max: Inclusive upper year bound.
        category: Exact primary_category.
        topic: Exact BERTopic topic id.
    Returns:
        list[Doc], e.g. ``[Doc(arxiv_id='dummy-0000', year=2020, ...)]`` of length k.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: retrieve")
    # TODO(Aliyan): embed_query -> FAISS search -> hydrate Doc rows -> apply filters
    # before truncating to k (filtering after truncation can return fewer than k).
    return [_dummy_doc(i, year_min, year_max, category, topic) for i in range(k)]


def generate_answer(question: str, docs: list[Doc]) -> tuple[str, list[Citation]]:
    """Produce a grounded answer and its citations.

    Purpose:
        Turn retrieved papers into a short answer with traceable sources.
    Args:
        question: The user's question.
        docs: Retrieved papers.
    Returns:
        tuple[str, list[Citation]], e.g. ``("STUB...", [Citation(...), ...])``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: generate_answer")
    # TODO(Aliyan): build a grounded prompt from docs, call llm.llm_call and parse
    # the cited arxiv_ids into Citation dicts.
    from src.contracts import Citation

    citations = [Citation(arxiv_id=d["arxiv_id"], title=d["title"]) for d in docs]
    return "STUB ANSWER: replace me with a grounded answer.", citations
