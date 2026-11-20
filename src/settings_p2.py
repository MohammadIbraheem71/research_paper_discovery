"""Phase 2 constants and paths.

Phase 1 settings in ``src/config.py`` are frozen; new Phase 2 values live here.
"""
from __future__ import annotations

import os
from pathlib import Path

MAX_RETRIES = 2
TOP_K = 5
EMBEDDING_DIM = 384  # mirrors the frozen src.config.EMBEDDING_DIM
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

ROOT = Path(__file__).resolve().parents[1]

TOPIC_SHARE_PATH = ROOT / "results" / "topic_share.csv"
TOPIC_SHARE_DUMMY_PATH = ROOT / "results" / "topic_share_dummy.csv"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npy"
DOC_TOPICS_PATH = ROOT / "data" / "doc_topics.parquet"
EVAL_RETRIEVAL_PATH = ROOT / "eval" / "eval_retrieval.json"
EVAL_ROUTING_PATH = ROOT / "eval" / "eval_routing.json"
CACHE_DIR = ROOT / "cache"
CHECKPOINT_DB_PATH = CACHE_DIR / "phase2_checkpoints.sqlite"
