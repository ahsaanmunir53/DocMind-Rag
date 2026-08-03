"""
Document ingestion pipeline: file -> text -> chunks -> embeddings -> DB.

This is called either synchronously (CELERY_ENABLED=False) or from a Celery
task (CELERY_ENABLED=True). Same function either way.
"""
from __future__ import annotations

from django.db import transaction

from core.embeddings import embed_texts
from core.textutils import chunk_text, extract_text

from .models import Chunk, Document


def process_document(document_id: int) -> None:
    """Extract text, chunk it, embed the chunks, and store them."""
    try:
        doc = Document.objects.get(pk=document_id)
    except Document.DoesNotExist:
        return

    doc.status = Document.Status.PROCESSING
    doc.error = ""
    doc.save(update_fields=["status", "error", "updated_at"])

    try:
        text = extract_text(doc.file.path)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text found in this file.")

        vectors = embed_texts(chunks)

        with transaction.atomic():
            doc.chunks.all().delete()  # idempotent: re-processing replaces old chunks
            Chunk.objects.bulk_create(
                [
                    Chunk(document=doc, index=i, text=chunk, embedding=vec)
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                ]
            )
            doc.num_chunks = len(chunks)
            doc.status = Document.Status.READY
            doc.save(update_fields=["num_chunks", "status", "updated_at"])
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        doc.status = Document.Status.FAILED
        doc.error = str(exc)[:2000]
        doc.save(update_fields=["status", "error", "updated_at"])
