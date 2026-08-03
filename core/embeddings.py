"""
Embeddings: turn text into a vector of numbers so we can measure similarity.

Two backends, chosen by settings.EMBEDDING_BACKEND:

  "local"  -> a deterministic hashing embedder. No API key, works offline,
              same text always gives the same vector. Good enough to build and
              test the whole pipeline. NOT semantically smart (it can't tell
              that "car" and "automobile" are related) — that's what the real
              backend is for.

  "openai" -> real semantic embeddings via OpenAI's API (needs OPENAI_API_KEY).
              Swap to this and retrieval quality jumps, with zero other changes.
"""
from __future__ import annotations

import hashlib
import math
from typing import List

from django.conf import settings


def _local_embed(text: str, dim: int) -> List[float]:
    """
    A tiny, dependency-free embedder. It hashes overlapping word n-grams into a
    fixed-length vector, then L2-normalizes. Deterministic and fast; it captures
    exact word overlap (so keyword-ish retrieval works) but not deep meaning.
    """
    vec = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vec
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 7) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _openai_embed_batch(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI  # imported lazily so the app runs without the package

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts -> list of vectors. Used for document chunks."""
    if not texts:
        return []
    if settings.EMBEDDING_BACKEND == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("EMBEDDING_BACKEND=openai but OPENAI_API_KEY is not set.")
        # OpenAI accepts batches; chunk to be safe on very large uploads
        out: List[List[float]] = []
        for i in range(0, len(texts), 100):
            out.extend(_openai_embed_batch(texts[i : i + 100]))
        return out
    return [_local_embed(t, settings.EMBEDDING_DIM) for t in texts]


def embed_query(text: str) -> List[float]:
    """Embed a single query -> one vector. Used when the user asks a question."""
    return embed_texts([text])[0]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors (used for SQLite fallback search)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
