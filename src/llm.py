"""LLM access with a small on-disk cache and a retry/backoff seam.

Owner: Aliyan. Stubs only: no API key or network access is needed to import or
call these functions yet.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from src.settings_p2 import CACHE_DIR, LLM_MODEL

logger = logging.getLogger(__name__)

CANNED_TEXT = "STUB ANSWER: replace me with a real LLM call."
CANNED_JSON: dict = {"answer": CANNED_TEXT, "confidence": 0.0}


def _cache_key(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Hash the full request so identical prompts share one cache entry."""
    payload = json.dumps(
        {"model": LLM_MODEL, "system": system, "prompt": prompt, "json_mode": json_mode},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(prompt: str, system: str = "", json_mode: bool = False) -> Path:
    """Path of the JSON cache entry for a request, inside the gitignored cache/ dir."""
    return CACHE_DIR / f"llm_{_cache_key(prompt, system, json_mode)}.json"


def call_with_retry(
    prompt: str,
    system: str = "",
    json_mode: bool = False,
    max_retries: int = 2,
    backoff_seconds: float = 1.0,
) -> str:
    """Retry wrapper around the real provider call.

    Purpose:
        Keep retry/backoff policy in one place for every LLM request.
    Args:
        prompt: User prompt.
        system: Optional system instruction.
        json_mode: Whether the provider must return JSON.
        max_retries: Number of retries after the first attempt.
        backoff_seconds: Base delay, doubled after each failed attempt.
    Returns:
        str, e.g. ``"STUB ANSWER: replace me with a real LLM call."``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: call_with_retry")
    # TODO(Aliyan): implement the provider call inside this loop with exponential
    # backoff; raise after max_retries failures. Cache successful responses via
    # _cache_path(). Do not put the API key in source control.
    for attempt in range(max_retries + 1):
        try:
            return llm_call(prompt, system=system, json_mode=json_mode)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(backoff_seconds * (2**attempt))
    return CANNED_TEXT  # pragma: no cover - loop above always returns or raises


def llm_call(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Return one LLM completion as text.

    Purpose:
        Single entry point for text completions; reads/writes the prompt-hash cache.
    Args:
        prompt: User prompt.
        system: Optional system instruction.
        json_mode: Ask the provider for JSON output; still returned as a string.
    Returns:
        str, e.g. ``"STUB ANSWER: replace me with a real LLM call."``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: llm_call")
    # TODO(Aliyan): replace with google-generativeai call using LLM_MODEL, routed
    # through call_with_retry(); keep the cache lookup below.
    path = _cache_path(prompt, system, json_mode)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = CANNED_TEXT
    path.write_text(json.dumps({"text": text}), encoding="utf-8")
    return text


def llm_json(prompt: str, system: str = "") -> dict:
    """Return one LLM completion parsed into a dictionary.

    Purpose:
        Structured LLM output for filters, rewrite decisions and grades.
    Args:
        prompt: User prompt.
        system: Optional system instruction.
    Returns:
        dict, e.g. ``{"answer": "STUB ...", "confidence": 0.0}``.
    Owner:
        Aliyan.
    """
    logger.warning("STUB: llm_json")
    # TODO(Aliyan): call llm_call(..., json_mode=True) and json.loads the result,
    # validating the shape before returning.
    return dict(CANNED_JSON)
