"""
The ingestion pipeline.

    read file -> pages -> figures -> vision pass -> chunks -> embeddings -> ready

Written as a streaming pipeline with commits at each stage, for two reasons.

Memory. A 900-page PDF never exists in memory as one string. Pages are read
one at a time, chunked as they arrive, and flushed to the database in batches.

Visibility. Status and page counters are updated as work completes, so the UI
can show real progress. A silent five-minute spinner is indistinguishable from
a hang, and users kill it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction

from core import pdf as pdfio
from core import vision
from core.embeddings import embed_texts
from documents.chunking import chunk_pages, chunk_plain_text, estimate_tokens
from documents.models import Chunk, Document, Figure, Page

logger = logging.getLogger(__name__)

PAGE_FLUSH = 40          # write pages in batches of this many
CHUNK_FLUSH = 200        # ...and chunks
EMBED_BATCH = 128


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _set(doc: Document, **fields):
    for k, v in fields.items():
        setattr(doc, k, v)
    doc.save(update_fields=list(fields.keys()) + ["updated_at"])


def _extract_docx(path: str) -> str:
    from docx import Document as Docx

    d = Docx(path)
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def process_document(document_id: int) -> None:
    """Run the whole pipeline for one document. Safe to re-run."""
    doc = Document.objects.get(id=document_id)
    path = doc.file.path

    try:
        _set(doc, status=Document.Status.EXTRACTING, error="", stage_detail="Opening file",
             pages_done=0)

        doc.sha256 = sha256_of(path)
        doc.size_bytes = os.path.getsize(path)

        # a re-upload of identical bytes is a no-op worth catching early
        twin = (
            Document.objects.filter(owner=doc.owner, sha256=doc.sha256,
                                    status=Document.Status.READY)
            .exclude(id=doc.id)
            .first()
        )
        if twin:
            logger.info("document %s duplicates %s", doc.id, twin.id)

        ext = os.path.splitext(path)[1].lower()
        doc.chunks.all().delete()
        doc.pages.all().delete()
        doc.figures.all().delete()

        if ext == ".pdf":
            _ingest_pdf(doc, path)
        else:
            _ingest_flat(doc, path, ext)

        _set(doc, status=Document.Status.READY, stage_detail="",
             pages_done=doc.pages_total or doc.pages_done)

    except Exception as exc:  # noqa: BLE001 - surface anything to the user
        logger.exception("processing failed for document %s", document_id)
        Document.objects.filter(id=document_id).update(
            status=Document.Status.FAILED,
            error=f"{exc.__class__.__name__}: {exc}"[:900],
            stage_detail="",
        )


# ------------------------------------------------------------------- PDF

def _ingest_pdf(doc: Document, path: str) -> None:
    stats = pdfio.quick_stats(path)
    if stats.get("encrypted"):
        raise ValueError("This PDF is password-protected. Remove the password and re-upload.")

    doc.page_count = stats["page_count"]
    doc.pages_total = stats["page_count"]
    doc.is_scanned = stats["is_scanned"]
    _set(doc, page_count=doc.page_count, pages_total=doc.pages_total,
         is_scanned=doc.is_scanned, stage_detail=f"Reading {doc.page_count} pages")

    page_rows: List[Page] = []
    all_figures = []
    stored: List[Page] = []

    for content in pdfio.iter_pages(path, extract_figures=settings.DETECT_FIGURES):
        page_rows.append(
            Page(
                document=doc,
                number=content.number,
                text=content.text,
                width=content.width,
                height=content.height,
                has_text_layer=content.has_text_layer,
            )
        )
        for fc in content.figures:
            all_figures.append(fc)

        if len(page_rows) >= PAGE_FLUSH:
            Page.objects.bulk_create(page_rows)
            stored.extend(page_rows)
            page_rows = []
            Document.objects.filter(id=doc.id).update(pages_done=content.number)

    if page_rows:
        Page.objects.bulk_create(page_rows)
        stored.extend(page_rows)
    Document.objects.filter(id=doc.id).update(pages_done=doc.pages_total)

    if doc.is_scanned:
        _set(doc, stage_detail="No text layer found - reading figures instead")

    figure_rows = _persist_figures(doc, all_figures)

    _set(doc, status=Document.Status.INDEXING, stage_detail="Building the search index")
    _index(doc, figure_rows, pdf_path=path)


def _persist_figures(doc: Document, candidates) -> List[Figure]:
    if not candidates:
        _set(doc, num_figures=0)
        return []

    pdfio.mark_repeated(candidates, doc.page_count or 1)
    useful = [f for f in candidates if not (f.repeated and f.guess == "letterhead")]

    _set(doc, status=Document.Status.ANALYSING,
         stage_detail=f"Analysing {len(useful)} figures")

    described = vision.describe_figures(useful)

    rows: List[Figure] = []
    signature_found = False
    for fc, info in zip(useful, described):
        fig = Figure(
            document=doc,
            page_number=fc.page_number,
            kind=info["kind"] if info["kind"] in dict(Figure.Kind.choices) else Figure.Kind.OTHER,
            caption=info["caption"],
            ocr_text=info["text"],
            labels=info["labels"],
            has_signature=info["has_signature"],
            has_stamp=info["has_stamp"],
            is_decorative=info["is_decorative"],
            confidence=info["confidence"],
            analysed_by=info["analysed_by"],
            x0=fc.bbox[0], y0=fc.bbox[1], x1=fc.bbox[2], y1=fc.bbox[3],
            source=fc.source,
            sha1=fc.sha1,
        )
        fig.save()
        if fc.png:
            fig.image.save(f"{doc.id}_p{fc.page_number}_{fig.id}.png",
                           ContentFile(fc.png), save=True)
        rows.append(fig)
        signature_found = signature_found or fig.has_signature

    _set(doc, num_figures=len(rows), has_signatures=signature_found)
    return rows


# ------------------------------------------------------- non-PDF sources

def _ingest_flat(doc: Document, path: str, ext: str) -> None:
    if ext in (".docx", ".doc"):
        text = _extract_docx(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()

    Page.objects.create(document=doc, number=1, text=text, has_text_layer=bool(text.strip()))
    _set(doc, page_count=1, pages_total=1, pages_done=1,
         status=Document.Status.INDEXING, stage_detail="Building the search index")
    _index(doc, [], pdf_path=None)


# ---------------------------------------------------------------- indexing

def _index(doc: Document, figures: List[Figure], pdf_path) -> None:
    """Chunk, embed and store. Figures are indexed alongside the text."""
    pages = doc.pages.all().order_by("number").iterator(chunk_size=50)

    pending: List[Chunk] = []
    index = 0
    total = 0

    def flush():
        nonlocal pending
        if not pending:
            return
        vectors = embed_texts([c.text for c in pending])
        for chunk, vec in zip(pending, vectors):
            chunk.embedding = vec
        Chunk.objects.bulk_create(pending)
        pending = []

    for tc in chunk_pages(pages):
        pending.append(
            Chunk(
                document=doc,
                index=index,
                text=tc.text,
                kind=Chunk.Kind.TEXT,
                page_number=tc.page_start,
                page_end=tc.page_end,
                token_estimate=tc.token_estimate,
            )
        )
        index += 1
        total += 1
        if len(pending) >= CHUNK_FLUSH:
            flush()

    # figures become retrievable passages: without this, a chart contributes
    # no words and cannot be found by any query
    for fig in figures:
        if fig.is_decorative:
            continue
        text = vision.searchable_text(fig)
        if len(text.strip()) < 25:
            continue
        pending.append(
            Chunk(
                document=doc,
                index=index,
                text=text,
                kind=Chunk.Kind.FIGURE,
                page_number=fig.page_number,
                page_end=fig.page_number,
                figure=fig,
                token_estimate=estimate_tokens(text),
            )
        )
        index += 1
        total += 1
        if len(pending) >= CHUNK_FLUSH:
            flush()

    flush()
    _set(doc, num_chunks=total)
