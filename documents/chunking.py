"""
Chunking.

Two changes from the original character-window splitter.

Page provenance. Chunks carry the page they came from, so an answer can say
"page 42" instead of "part 137". On a 600-page contract that is the difference
between a citation and a shrug.

Structure awareness. Headings and list boundaries are better break points than
a character count. A chunk that starts mid-clause retrieves badly, because the
subject of the sentence is in the previous chunk.

Sizing is in estimated tokens rather than characters. Characters mislead badly
on tables and code, where a 1000-character window can be 400 tokens or 900.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from django.conf import settings

# English prose runs about 4 characters per token. Good enough for budgeting,
# and it costs nothing - a real tokenizer would be another dependency.
CHARS_PER_TOKEN = 4

HEADING = re.compile(
    r"^\s*("
    r"(?:chapter|section|article|clause|appendix|annex|part|schedule)\s+[\dIVXA-Z][\w.\-]*"
    r"|\d+(?:\.\d+)*\s+[A-Z][^\n]{2,80}"
    r"|[A-Z][A-Z \d\-&/,']{6,80}"
    r")\s*$",
    re.I | re.M,
)


@dataclass
class TextChunk:
    text: str
    page_start: int
    page_end: int
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def clean(text: str) -> str:
    """Normalise whitespace and repair words broken across a line break."""
    text = text.replace("\u00ad", "")                       # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # de-hyphenate
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    out: List[str] = []
    blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _split_blocks(text: str) -> List[str]:
    """Break into paragraph-ish blocks, keeping headings attached to what follows."""
    parts = re.split(r"\n\s*\n", text)
    blocks: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 40 and HEADING.match(p) is None:
            blocks.append(p)
        else:
            # a heading joins the next block rather than standing alone
            if blocks and len(blocks[-1]) < 80 and HEADING.match(blocks[-1] or ""):
                blocks[-1] = blocks[-1] + "\n" + p
            else:
                blocks.append(p)
    return blocks


def _split_long_block(block: str, max_chars: int) -> List[str]:
    """Split an oversized block on sentence boundaries where possible."""
    if len(block) <= max_chars:
        return [block]
    pieces: List[str] = []
    start = 0
    n = len(block)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window_start = max(start + int(max_chars * 0.6), start + 1)
            cut = -1
            for sep in (". ", ".\n", "; ", "\n", " "):
                idx = block.rfind(sep, window_start, end)
                if idx != -1:
                    cut = idx + len(sep)
                    break
            if cut > start:
                end = cut
        pieces.append(block[start:end].strip())
        start = end
    return [p for p in pieces if p]


def chunk_pages(
    pages: Iterable,
    target_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> Iterator[TextChunk]:
    """Stream chunks from an iterable of objects with .number and .text.

    Generator rather than list: a 900-page document is chunked and written to
    the database incrementally, so peak memory stays flat.
    """
    target = target_tokens or getattr(settings, "CHUNK_TOKENS", 380)
    overlap = overlap_tokens if overlap_tokens is not None else getattr(
        settings, "CHUNK_OVERLAP_TOKENS", 60
    )
    max_chars = target * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN

    buffer = ""
    buf_start_page = None
    buf_end_page = None

    for page in pages:
        text = clean(getattr(page, "text", "") or "")
        if not text:
            continue
        page_no = getattr(page, "number", None)

        for block in _split_blocks(text):
            for piece in _split_long_block(block, max_chars):
                if buf_start_page is None:
                    buf_start_page = page_no
                candidate = (buffer + "\n\n" + piece).strip() if buffer else piece

                if len(candidate) <= max_chars:
                    buffer = candidate
                    buf_end_page = page_no
                    continue

                if buffer:
                    yield TextChunk(buffer, buf_start_page, buf_end_page or buf_start_page,
                                    estimate_tokens(buffer))
                    tail = buffer[-overlap_chars:] if overlap_chars else ""
                    # start the next window on a word boundary
                    if tail:
                        space = tail.find(" ")
                        tail = tail[space + 1 :] if space != -1 else tail
                    buffer = (tail + "\n\n" + piece).strip() if tail else piece
                    buf_start_page = page_no
                    buf_end_page = page_no
                else:
                    buffer = piece
                    buf_start_page = buf_end_page = page_no

    if buffer.strip():
        yield TextChunk(buffer, buf_start_page or 1, buf_end_page or buf_start_page or 1,
                        estimate_tokens(buffer))


def chunk_plain_text(text: str, target_tokens: Optional[int] = None) -> List[TextChunk]:
    """Chunk a non-paginated source (txt, md, docx) as a single page."""

    class _P:
        number = 1

        def __init__(self, t):
            self.text = t

    return list(chunk_pages([_P(text)], target_tokens))
