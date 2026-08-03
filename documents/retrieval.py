"""
Retrieval: given a question, find the most relevant chunks for a user.

Two paths, same result shape:
  - pgvector (Postgres): uses the database's cosine-distance operator to do the
    nearest-neighbour search in SQL — fast, scales to lots of chunks.
  - SQLite fallback: loads the user's chunks and ranks them with cosine
    similarity in Python — simple, fine for dev and modest data.

Scope is always limited to the requesting user's own documents (and optionally a
single document), so users can never retrieve from each other's files.
"""
from __future__ import annotations

from typing import List, Optional

from django.conf import settings

from core.embeddings import cosine_similarity, embed_query
from core.llm import RetrievedChunk

from .models import Chunk


def retrieve(user, question: str, document_id: Optional[int] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    top_k = top_k or settings.TOP_K
    query_vec = embed_query(question)

    base = Chunk.objects.filter(document__owner=user, document__status="ready")
    if document_id is not None:
        base = base.filter(document_id=document_id)

    if settings.USING_PGVECTOR:
        from pgvector.django import CosineDistance

        rows = (
            base.annotate(distance=CosineDistance("embedding", query_vec))
            .order_by("distance")
            .select_related("document")[:top_k]
        )
        return [
            RetrievedChunk(
                text=c.text,
                document_title=c.document.title,
                chunk_index=c.index,
                score=round(1.0 - float(c.distance), 4),  # distance -> similarity
            )
            for c in rows
        ]

    # SQLite fallback: rank in Python
    scored = []
    for c in base.select_related("document").iterator():
        if not c.embedding:
            continue
        scored.append((cosine_similarity(query_vec, c.embedding), c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        RetrievedChunk(
            text=c.text,
            document_title=c.document.title,
            chunk_index=c.index,
            score=round(float(score), 4),
        )
        for score, c in scored[:top_k]
    ]
