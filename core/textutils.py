"""
Extract text from an uploaded file and split it into overlapping chunks.

Chunking matters for RAG: embeddings work best on smallish, self-contained
passages. We split on ~CHUNK_SIZE characters at sentence/paragraph boundaries
where possible, and overlap consecutive chunks by CHUNK_OVERLAP characters so a
sentence that straddles a boundary isn't lost to either side.
"""
from __future__ import annotations

import os
from typing import List

from django.conf import settings


def extract_text(file_path: str) -> str:
    """Pull plain text out of a PDF or a .txt/.md file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    # plain text formats
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _clean(text: str) -> str:
    # collapse excessive whitespace but keep paragraph breaks
    lines = [ln.strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> List[str]:
    """
    Split text into overlapping chunks. Tries to break on paragraph, then
    sentence, then whitespace boundaries near the target size so chunks read
    naturally instead of cutting mid-word.
    """
    size = size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
    text = _clean(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # prefer a clean boundary within the last ~20% of the window
            window_start = max(start + int(size * 0.8), start + 1)
            boundary = -1
            for sep in ("\n\n", ". ", ".\n", "\n", " "):
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    boundary = idx + len(sep)
                    break
            if boundary != -1:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
