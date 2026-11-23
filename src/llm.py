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
import os
import urllib.request

from dotenv import load_dotenv

from src.settings_p2 import CACHE_DIR, LLM_MODEL

logger = logging.getLogger(__name__)
load_dotenv()

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
    """Call a configured Gemini or Groq provider with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            groq_key = os.getenv("GROQ_API_KEY")
            if not gemini_key and not groq_key:
                raise RuntimeError(
                    "Missing LLM API key: set GEMINI_API_KEY or GROQ_API_KEY in .env"
                )
            if groq_key:
                payload = {
                    "model": LLM_MODEL,
                    "messages": (
                        ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}]
                    ),
                    "temperature": 0.2,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                request = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
            else:
                import google.generativeai as genai

                genai.configure(api_key=gemini_key)
                generation_config = (
                    {"response_mime_type": "application/json"}
                    if json_mode
                    else None
                )
                model = genai.GenerativeModel(
                    model_name=LLM_MODEL,
                    system_instruction=system or None,
                )
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                text = response.text
            path = _cache_path(prompt, system, json_mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text}), encoding="utf-8")
            return text
        except Exception as exc:
            if attempt == max_retries:
                raise
            time.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError("LLM retry loop exhausted")


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
    """Return a provider completion, using the disk cache first."""
    path = _cache_path(prompt, system, json_mode)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and isinstance(cached.get("text"), str):
            return cached["text"]
    return call_with_retry(
        prompt,
        system=system,
        json_mode=json_mode,
        max_retries=max(0, int(os.getenv("LLM_MAX_RETRIES", "2"))),
    )


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
    text = llm_call(prompt, system=system, json_mode=True)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {text[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be a JSON object")
    return parsed
