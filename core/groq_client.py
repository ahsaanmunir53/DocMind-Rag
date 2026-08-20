"""
Groq client.

One place that knows the endpoint, the credentials, and - importantly - the
current model IDs. Groq retires models on a schedule, and a hard-coded name
scattered across six files becomes six outages on the same morning.

Model roles, defaults chosen from Groq's own migration guidance:

  TEXT_MODEL   openai/gpt-oss-120b   main answering / rewriting
  FAST_MODEL   openai/gpt-oss-20b    reranking, classification, short calls
  VISION_MODEL qwen/qwen3.6-27b      the only multimodal model on Groq

Every one is overridable from .env, so a future deprecation is a config
change rather than a code change.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from django.conf import settings

logger = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Vision limits published by Groq. Exceeding either returns a 400.
MAX_IMAGES_PER_REQUEST = 5
MAX_REQUEST_BYTES = 18 * 1024 * 1024      # 20MB ceiling, kept under with headroom


class GroqError(Exception):
    """A problem worth showing a person."""

    def __init__(self, message: str, *, status: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def api_key() -> str:
    return (getattr(settings, "GROQ_API_KEY", "") or "").strip()


def configured() -> bool:
    return bool(api_key())


def model_for(role: str) -> str:
    return {
        "text": getattr(settings, "GROQ_TEXT_MODEL", "openai/gpt-oss-120b"),
        "fast": getattr(settings, "GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
        "vision": getattr(settings, "GROQ_VISION_MODEL", "qwen/qwen3.6-27b"),
    }.get(role, getattr(settings, "GROQ_TEXT_MODEL", "openai/gpt-oss-120b"))


def _raise(resp: httpx.Response) -> None:
    status = resp.status_code
    detail = ""
    try:
        detail = (resp.json().get("error", {}) or {}).get("message", "")[:300]
    except Exception:
        detail = resp.text[:300]

    # 401 and 403 mean different things and need different fixes, and Groq
    # says which in the body. The old branch computed `detail` and then threw
    # it away, so every one of these looked like a bad key.
    if status == 401:
        raise GroqError(
            "Groq rejected the API key (401). It is missing, mistyped, or has "
            "been deleted from console.groq.com. " + (f"Groq said: {detail}" if detail else ""),
            status=status,
        )
    if status == 403:
        raise GroqError(
            "Groq accepted the key but refused this request (403) - usually the "
            "account cannot use this model. " + (f"Groq said: {detail}" if detail else ""),
            status=status,
        )
    if status == 404:
        raise GroqError(
            f"That model is not available on your Groq account. {detail} "
            "Groq retires models periodically - set GROQ_TEXT_MODEL to a current one.",
            status=status,
        )
    if status == 413:
        raise GroqError("The request was too large for Groq. Try fewer or smaller images.",
                        status=status)
    if status == 429:
        raise GroqError("Groq free-tier rate limit reached. Waiting will clear it.",
                        status=status, retryable=True)
    if status >= 500:
        raise GroqError(f"Groq is having trouble ({status}). {detail}",
                        status=status, retryable=True)
    raise GroqError(f"Groq error {status}. {detail}", status=status)


def chat(
    messages: List[Dict[str, Any]],
    *,
    role: str = "text",
    model: Optional[str] = None,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    json_mode: bool = False,
    retries: int = 2,
    timeout: float = 120.0,
) -> str:
    """One completion. Returns text, or raises GroqError."""
    if not configured():
        raise GroqError("not_configured")

    payload: Dict[str, Any] = {
        "model": model or model_for(role),
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }

    last: Optional[GroqError] = None
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            last = GroqError(
                f"Could not reach Groq ({exc.__class__.__name__}). Check the connection.",
                retryable=True,
            )
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last from exc

        # Some models reject JSON mode; retry once without it rather than fail
        if resp.status_code == 400 and json_mode and "response_format" in resp.text:
            payload.pop("response_format", None)
            json_mode = False
            continue

        if resp.status_code >= 400:
            try:
                _raise(resp)
            except GroqError as exc:
                last = exc
                if exc.retryable and attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise GroqError("Groq returned an unexpected response shape.")

    raise last or GroqError("Groq request failed.")


def chat_json(messages: List[Dict[str, Any]], **kwargs) -> Any:
    """Completion that must parse as JSON, with one corrective retry."""
    kwargs.setdefault("json_mode", True)
    raw = chat(messages, **kwargs)
    try:
        return _loads(raw)
    except ValueError:
        fixed = chat(
            messages
            + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content": "Reply again with ONLY the JSON object."},
            ],
            **kwargs,
        )
        return _loads(fixed)


def _loads(raw: str) -> Any:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # models sometimes emit a reasoning preamble before the object
    start = min(
        [i for i in (s.find("{"), s.find("[")) if i != -1] or [-1]
    )
    if start == -1:
        raise ValueError("no JSON found")
    end = max(s.rfind("}"), s.rfind("]"))
    if end <= start:
        raise ValueError("truncated JSON")
    return json.loads(s[start : end + 1])


def image_part(png_bytes: bytes, mime: str = "image/png") -> Dict[str, Any]:
    """Wrap raw image bytes as a data-URL content part."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def fits_budget(images: List[bytes]) -> bool:
    """Base64 inflates by ~4/3; check before sending rather than eating a 400."""
    approx = sum(len(b) for b in images) * 4 // 3
    return approx < MAX_REQUEST_BYTES and len(images) <= MAX_IMAGES_PER_REQUEST
