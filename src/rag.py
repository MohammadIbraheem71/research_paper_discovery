"""Retrieval-augmented generation resources and operations.

Owner: Aliyan. Retrieval uses a real FAISS index built over data/embeddings.npy;
missing assets are fatal so the demo and graph can never return dummy papers.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional, TypedDict

import numpy as np
import pandas as pd

from src.contracts import Doc
from src.settings_p2 import (
    DOC_TOPICS_PATH,
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    ROOT,
    TOP_K,
)

logger = logging.getLogger(__name__)


class Resources(TypedDict):
    """Loaded retrieval assets: a real FAISS index, papers and topic ids."""

    index: object
    docs: list[Doc]
    doc_ids: list[str]


@lru_cache(maxsize=1)
def load_resources() -> Resources:
    """Load the FAISS index, paper metadata and topic assignments.

    Purpose:
        Provide one cached entry point for all retrieval assets.
    Args:
        None.
    Returns:
        Resources with a real FAISS index, all papers and their topic ids.
    Owner:
        Aliyan.
    """
    papers_path = ROOT / "data" / "papers.parquet"
    required = {
        "embeddings": EMBEDDINGS_PATH,
        "papers": papers_path,
        "doc_topics": DOC_TOPICS_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Real retrieval assets are required; dummy papers are never used. Missing: "
            + "; ".join(missing)
        )

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    papers = pd.read_parquet(papers_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"embeddings.npy has shape {embeddings.shape}; expected (n, {EMBEDDING_DIM})")
    if len(papers) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata/embedding row mismatch: {len(papers)} papers vs "
            f"{embeddings.shape[0]} embeddings"
        )

    topics = pd.read_parquet(DOC_TOPICS_PATH)
    if len(topics) != len(papers):
        raise ValueError(
            f"Topic/metadata row mismatch: {len(topics)} topics vs {len(papers)} papers"
        )

    topic_column = next(
        (column for column in ("bertopic_topic", "topic_id", "topic")
         if column in topics.columns),
        None,
    )
    if topic_column is None:
        raise ValueError(
            f"{DOC_TOPICS_PATH} has no topic column; expected one of "
            "bertopic_topic, topic_id, topic"
        )
    docs: list[Doc] = []
    for i, row in papers.reset_index(drop=True).iterrows():
        topic_value = None
        if topic_column is not None:
            value = topics.iloc[i][topic_column]
            topic_value = None if pd.isna(value) else int(value)
        year_value = int(row["year"]) if "year" in row and pd.notna(row["year"]) else 0
        arxiv_id = str(row.get("id", row.get("arxiv_id", f"row-{i}")))
        docs.append(Doc(
            arxiv_id=arxiv_id,
            title=str(row.get("title", "")),
            abstract=str(row.get("abstract", "")),
            year=year_value,
            primary_category=str(row.get("primary_category", "")),
            score=0.0,
            bertopic_topic=topic_value,
        ))
    index = build_index(np.asarray(embeddings, dtype=np.float32))
    print(
        f"Retrieval resources loaded: FAISS ntotal={index.ntotal}, "
        f"embeddings={tuple(embeddings.shape)}, doc_topics rows={len(topics)}, "
        f"papers={len(papers)}"
    )
    return Resources(index=index, docs=docs, doc_ids=[doc["arxiv_id"] for doc in docs])


def build_index(embeddings: np.ndarray) -> object:
    """Build a FAISS index from an (n, EMBEDDING_DIM) float array.

    Purpose:
        Turn full-corpus embeddings into a searchable index.
    Args:
        embeddings: Array of shape ``(n, 384)``.
    Returns:
        object, a populated ``faiss.IndexFlatIP``.
    Owner:
        Aliyan.
    """
    import faiss

    assert embeddings.ndim == 2
    assert embeddings.shape[1] == EMBEDDING_DIM
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    return index


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
    from sentence_transformers import SentenceTransformer

    model = getattr(embed_query, "_model", None)
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embed_query._model = model
    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Query embedding has shape {vector.shape}; expected ({EMBEDDING_DIM},)")
    return vector


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
        list[Doc], at most k real papers from the FAISS index.
    Owner:
        Aliyan.
    """
    resources = load_resources()
    if resources["index"] is None:
        raise RuntimeError(
            "FAISS index is unavailable; real retrieval assets are required"
        )

    index = resources["index"]
    docs = resources["docs"]
    query_vector = embed_query(query).reshape(1, -1)
    candidate_count = min(index.ntotal, max(k * 20, k))
    scores, indices = index.search(query_vector, candidate_count)

    matched: list[Doc] = []
    for score, row_index in zip(scores[0], indices[0]):
        if row_index < 0:
            continue
        doc = dict(docs[int(row_index)])
        if year_min is not None and doc["year"] < year_min:
            continue
        if year_max is not None and doc["year"] > year_max:
            continue
        if category is not None and doc["primary_category"] != category:
            continue
        if topic is not None and doc["bertopic_topic"] != topic:
            continue
        doc["score"] = float(score)
        matched.append(Doc(**doc))
        if len(matched) == k:
            break
    return matched


def generate_answer(question: str, docs: list[Doc]) -> tuple[str, list[Citation]]:
    """Produce a grounded answer and its citations.

    Purpose:
        Turn retrieved papers into a short answer with traceable sources.
    Args:
        question: The user's question.
        docs: Retrieved papers.
    Returns:
        tuple[str, list[Citation]], the grounded answer and its real citations.
    Owner:
        Aliyan.
    """
    from src.contracts import Citation
    if not docs:
        return "not enough evidence", []

    context = "\n\n".join(
        f"[arXiv:{doc['arxiv_id']}] {doc['title']}\n"
        f"Year: {doc['year']}; Category: {doc['primary_category']}\n"
        f"{doc['abstract']}"
        for doc in docs
    )
    system = (
        "Answer only from the retrieved paper abstracts. Cite claims with "
        "[arXiv:ID]. If the retrieved abstracts do not support an answer, "
        "say exactly: not enough evidence"
    )
    prompt = f"Question: {question}\n\nRetrieved documents:\n{context}"
    try:
        from src.llm import llm_call

        answer = llm_call(prompt, system=system).strip()
    except Exception as exc:
        logger.warning("LLM answer unavailable: %s", exc)
        return "not enough evidence", []

    cited_ids = set(re.findall(r"\[arXiv:([^\]]+)\]", answer))
    citations = [
        Citation(arxiv_id=doc["arxiv_id"], title=doc["title"])
        for doc in docs
        if doc["arxiv_id"] in cited_ids
    ]
    if not citations:
        return "not enough evidence", []
    return answer, citations
