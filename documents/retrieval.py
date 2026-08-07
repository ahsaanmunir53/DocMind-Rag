"""
Hybrid retrieval.

The old pipeline did one thing: embed the question, cosine-compare against
every chunk, return the top K. With the built-in hashing embedder that is
close to keyword matching with extra steps, and it fails in both directions -
it misses paraphrases, and it cannot reliably find an exact term like an
invoice number that the embedder has smeared across a hash bucket.

This module runs two retrievers and fuses them:

  BM25    exact and near-exact wording, document-length normalised. Finds
          "IRSA trust policy" or "clause 7.3" reliably.
  Vector  paraphrase and concept matching. Finds "how do we handle refunds"
          against a passage that never uses the word refund.

Fusion is Reciprocal Rank Fusion, which combines rankings rather than raw
scores - so a BM25 score of 14.2 and a cosine of 0.83 never have to be made
commensurable, which they cannot honestly be.

An optional rerank pass then asks a small model to order the survivors by
actual relevance. That step moves answer quality more cheaply than anything
else here, because precision at 5 matters far more than recall at 50.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from django.conf import settings
from django.db.models import Q

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# BM25 constants. k1 controls term-frequency saturation, b the length penalty.
K1 = 1.5
B = 0.75

CANDIDATE_LIMIT = 600      # lexical candidates pulled from the database
VECTOR_SCAN_LIMIT = 60000  # beyond this, move to Postgres + pgvector
RRF_K = 60                 # standard damping constant

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "its", "this",
    "that", "these", "those", "as", "at", "by", "from", "what", "which", "who",
    "whom", "how", "when", "where", "why", "do", "does", "did", "can", "could",
    "should", "would", "will", "shall", "may", "might", "i", "you", "we",
    "they", "he", "she", "me", "my", "our", "your", "their", "about", "into",
    "there", "here", "then", "than", "so", "not", "no", "any", "all", "some",
}

TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_.']*", re.I)


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    toks = [t.lower().strip(".-_'") for t in TOKEN.findall(text or "")]
    toks = [t for t in toks if len(t) > 1 or t.isdigit()]
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS]


@dataclass
class Hit:
    chunk_id: int
    text: str
    document_id: int
    document_title: str
    chunk_index: int
    page_number: Optional[int]
    kind: str                 # "text" | "figure"
    score: float
    lexical_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    figure_id: Optional[int] = None


# ------------------------------------------------------------------- BM25

def bm25_candidates(query: str, chunk_qs, limit: int = CANDIDATE_LIMIT) -> List:
    """Narrow the corpus with SQL before scoring in Python.

    Scoring every chunk in the database would defeat the point. An OR of
    icontains on the query terms cuts a 40,000-chunk corpus to a few hundred
    rows, and BM25 then ranks only those.
    """
    terms = tokenize(query)[:12]
    if not terms:
        return []
    q = Q()
    matched = False
    for t in terms:
        if len(t) >= 3:
            q |= Q(text__icontains=t)
            matched = True
    if not matched:
        q = Q(text__icontains=terms[0])
    return list(chunk_qs.filter(q)[:limit])


def bm25_rank(query: str, chunks: Sequence, corpus_avg_len: Optional[float] = None) -> List:
    """Score candidates with BM25 and return them ordered."""
    terms = tokenize(query)
    if not terms or not chunks:
        return []

    docs_tokens = [tokenize(c.text, keep_stopwords=True) for c in chunks]
    lengths = [len(t) or 1 for t in docs_tokens]
    avgdl = corpus_avg_len or (sum(lengths) / len(lengths))
    n = len(chunks)

    df: Counter = Counter()
    counters = []
    unique_terms = set(terms)
    for toks in docs_tokens:
        counter = Counter(toks)
        counters.append(counter)
        for t in unique_terms:
            if counter.get(t):
                df[t] += 1

    scored = []
    for i, chunk in enumerate(chunks):
        counter = counters[i]
        dl = lengths[i]
        score = 0.0
        for t in terms:
            f = counter.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avgdl))
        if score > 0:
            scored.append((score, i, chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored]


# ----------------------------------------------------------------- vector

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def vector_rank(query_vec: Sequence[float], chunk_qs, top: int = 40) -> List:
    """Cosine ranking. Uses pgvector when available, numpy when not."""
    if not query_vec:
        return []

    if settings.USING_PGVECTOR:
        from pgvector.django import CosineDistance

        return list(
            chunk_qs.exclude(embedding=None)
            .order_by(CosineDistance("embedding", list(query_vec)))[:top]
        )

    rows = list(
        chunk_qs.exclude(embedding=None).values_list("id", "embedding")[:VECTOR_SCAN_LIMIT]
    )
    if not rows:
        return []

    ids = [r[0] for r in rows]
    best_ids: List[int]
    if np is not None:
        try:
            mat = np.asarray([r[1] for r in rows], dtype=np.float32)
            qv = np.asarray(query_vec, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = (mat @ qv) / (norms * (float(np.linalg.norm(qv)) or 1.0))
            order = np.argsort(-sims)[:top]
            best_ids = [ids[int(i)] for i in order]
        except Exception:  # ragged vectors, dimension drift after a backend switch
            best_ids = _cosine_fallback(rows, query_vec, top)
    else:
        best_ids = _cosine_fallback(rows, query_vec, top)

    by_id = {c.id: c for c in chunk_qs.filter(id__in=best_ids)}
    return [by_id[i] for i in best_ids if i in by_id]


def _cosine_fallback(rows, query_vec, top) -> List[int]:
    scored = []
    for cid, vec in rows:
        if not vec or len(vec) != len(query_vec):
            continue
        scored.append((_cosine(query_vec, vec), cid))
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:top]]


# ------------------------------------------------------------------- fusion

def reciprocal_rank_fusion(*rankings: Iterable, k: int = RRF_K):
    """Combine ranked lists without pretending their scores are comparable."""
    scores: Dict[int, float] = {}
    objs: Dict[int, object] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k + rank)
            objs[item.id] = item

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(objs[i], s) for i, s in ordered]


# ------------------------------------------------------------------ rerank

RERANK_SYSTEM = (
    "You rank document passages by how well they answer a question. "
    'Return ONLY {"order": [passage numbers, most relevant first]}. '
    "Include only passages that genuinely help; drop the rest. Never invent numbers."
)


def llm_rerank(question: str, hits: List[Hit], keep: int) -> List[Hit]:
    """Reorder with a small model. Falls back silently to the fused order."""
    from core import groq_client as groq

    if not groq.configured() or len(hits) <= keep:
        return hits[:keep]

    listing = "\n\n".join(
        f"[{i}] (page {h.page_number or '?'}) {h.text[:700]}"
        for i, h in enumerate(hits[:20], start=1)
    )
    try:
        data = groq.chat_json(
            [
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{listing}"},
            ],
            role="fast",
            max_tokens=300,
            temperature=0.0,
        )
        order = data.get("order") if isinstance(data, dict) else data
        if not isinstance(order, list):
            return hits[:keep]
        picked: List[Hit] = []
        seen = set()
        for n in order:
            try:
                idx = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(hits) and idx not in seen:
                seen.add(idx)
                picked.append(hits[idx])
        for i, h in enumerate(hits):
            if len(picked) >= keep:
                break
            if i not in seen:
                picked.append(h)
        return picked[:keep]
    except Exception as exc:  # pragma: no cover
        logger.info("rerank skipped: %s", exc)
        return hits[:keep]


# -------------------------------------------------------------- entry point

def search(
    query: str,
    chunk_qs,
    query_vec: Optional[Sequence[float]] = None,
    top_k: int = 6,
    use_rerank: bool = True,
) -> List[Hit]:
    """Hybrid search over a chunk queryset. Returns ranked Hits."""
    chunk_qs = chunk_qs.select_related("document")

    lexical = bm25_rank(query, bm25_candidates(query, chunk_qs))[:40]
    vector = vector_rank(query_vec, chunk_qs, top=40) if query_vec else []

    if not lexical and not vector:
        return []

    fused = reciprocal_rank_fusion(lexical, vector)
    lex_pos = {c.id: i + 1 for i, c in enumerate(lexical)}
    vec_pos = {c.id: i + 1 for i, c in enumerate(vector)}

    hits: List[Hit] = []
    for chunk, score in fused[: max(top_k * 4, 20)]:
        hits.append(
            Hit(
                chunk_id=chunk.id,
                text=chunk.text,
                document_id=chunk.document_id,
                document_title=chunk.document.title,
                chunk_index=chunk.index,
                page_number=getattr(chunk, "page_number", None),
                kind=getattr(chunk, "kind", "text"),
                figure_id=getattr(chunk, "figure_id", None),
                score=score,
                lexical_rank=lex_pos.get(chunk.id),
                vector_rank=vec_pos.get(chunk.id),
            )
        )

    if use_rerank:
        hits = llm_rerank(query, hits, top_k)
    return hits[:top_k]
