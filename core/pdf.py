"""
PDF engine.

Two jobs the old pypdf path could not do:

  1. Stream a large PDF page by page instead of loading the whole text into
     memory. A 900-page file is processed in constant memory.

  2. Find the things that are not text - embedded photos, scanned stamps,
     and vector diagrams drawn with line primitives (which contain no image
     object at all, so image-only extractors miss every chart in the file).

Each figure is cropped, rendered to PNG, measured, and given a first-pass
classification from cheap pixel statistics. The vision model in vision.py
then confirms or corrects that. Doing the arithmetic first means we only
spend an API call on regions that are plausibly meaningful.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # PyMuPDF renamed its import; support both
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf


# --------------------------------------------------------------- thresholds

MIN_FIGURE_PT = 42.0          # ignore anything smaller than this on a side
MIN_PAGE_AREA_RATIO = 0.008   # ...or smaller than 0.8% of the page
MAX_PAGE_AREA_RATIO = 0.97    # ...or effectively the whole page (background)
MIN_VECTOR_PATHS = 6          # a "drawing" needs this many strokes to count
VECTOR_MERGE_GAP = 14.0       # points; paths closer than this join one cluster
MAX_FIGURES_PER_DOC = 80      # hard cap so a pathological file can't stall us
RENDER_ZOOM = 2.0             # crop render scale (≈144 dpi)
MAX_RENDER_PX = 1400          # longest side after render

SIGNATURE_CUES = re.compile(
    r"\b(signature|signed|sign here|authori[sz]ed signator|for and on behalf|"
    r"witness|initials?|/s/|stamp|seal)\b",
    re.I,
)
CAPTION_CUES = re.compile(
    r"\b(fig(?:ure)?\.?\s*\d+|table\s*\d+|chart\s*\d+|diagram\s*\d+|exhibit\s*[A-Z0-9]+|"
    r"appendix\s*[A-Z0-9]+|scheme\s*\d+)\b",
    re.I,
)


@dataclass
class FigureCandidate:
    page_number: int                 # 1-based
    bbox: Tuple[float, float, float, float]
    source: str                      # "raster" | "vector"
    png: bytes = b""
    width_pt: float = 0.0
    height_pt: float = 0.0
    ink_ratio: float = 0.0           # share of non-background pixels
    colour_count: int = 0
    aspect: float = 0.0
    nearby_text: str = ""
    caption_hint: str = ""
    guess: str = "unknown"
    signature_score: float = 0.0
    sha1: str = ""
    repeated: bool = False           # same artwork on many pages -> letterhead
    extra: dict = field(default_factory=dict)


@dataclass
class PageContent:
    number: int                      # 1-based
    text: str
    width: float
    height: float
    has_text_layer: bool
    figures: List[FigureCandidate] = field(default_factory=list)


# ------------------------------------------------------------------ helpers

def _rects_overlap_or_near(a, b, gap: float) -> bool:
    return not (
        a.x1 + gap < b.x0
        or b.x1 + gap < a.x0
        or a.y1 + gap < b.y0
        or b.y1 + gap < a.y0
    )


def _merge_rects(rects: List, gap: float) -> List:
    """Union-merge rectangles that touch or nearly touch."""
    out: List = []
    for r in rects:
        merged = False
        for i, existing in enumerate(out):
            if _rects_overlap_or_near(existing, r, gap):
                out[i] = existing | r
                merged = True
                break
        if merged:
            # one merge can make two clusters adjacent; settle by re-running
            changed = True
            while changed:
                changed = False
                for i in range(len(out)):
                    for j in range(i + 1, len(out)):
                        if _rects_overlap_or_near(out[i], out[j], gap):
                            out[i] = out[i] | out[j]
                            out.pop(j)
                            changed = True
                            break
                    if changed:
                        break
        else:
            out.append(pymupdf.Rect(r))
    return out


def _significant(rect, page_rect) -> bool:
    w, h = rect.width, rect.height
    if w < MIN_FIGURE_PT or h < MIN_FIGURE_PT:
        return False
    page_area = page_rect.width * page_rect.height or 1.0
    ratio = (w * h) / page_area
    return MIN_PAGE_AREA_RATIO <= ratio <= MAX_PAGE_AREA_RATIO


def _render_crop(page, rect) -> bytes:
    """Render a page region to PNG, capped so huge figures stay manageable."""
    zoom = RENDER_ZOOM
    longest = max(rect.width, rect.height) * zoom
    if longest > MAX_RENDER_PX:
        zoom *= MAX_RENDER_PX / longest
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect, alpha=False)
    return pix.tobytes("png")


def _pixel_stats(png: bytes) -> Tuple[float, int]:
    """Ink coverage and rough colour count, straight from the pixmap.

    Ink coverage separates a signature (a few dark strokes on white) from a
    photograph (dense, varied) without needing a model.
    """
    try:
        pix = pymupdf.Pixmap(io.BytesIO(png))
    except Exception:  # pragma: no cover
        return 0.0, 0
    if pix.n >= 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    step = max(1, (pix.width * pix.height) // 20000)  # sample, don't scan all
    samples = pix.samples
    n = pix.n
    total = 0
    ink = 0
    colours = set()

    for i in range(0, pix.width * pix.height, step):
        off = i * n
        if off + n > len(samples):
            break
        r, g, b = samples[off], samples[off + 1], samples[off + 2] if n >= 3 else (
            samples[off], samples[off], samples[off]
        )
        total += 1
        if (r + g + b) / 3 < 235:
            ink += 1
        if len(colours) < 4096:
            colours.add((r >> 4, g >> 4, b >> 4))

    return (ink / total if total else 0.0), len(colours)


def _classify_cheap(fc: FigureCandidate) -> Tuple[str, float]:
    """First-pass guess from geometry and pixels alone.

    Returns (guess, signature_score). The vision model gets the final say,
    but this decides which candidates are worth an API call at all.
    """
    aspect = fc.aspect
    ink = fc.ink_ratio
    colours = fc.colour_count
    near = fc.nearby_text or ""
    cue = bool(SIGNATURE_CUES.search(near))

    sig = 0.0
    if 1.4 <= aspect <= 9.0:
        sig += 0.30
    if 0.01 <= ink <= 0.30:
        sig += 0.30
    if colours <= 40:
        sig += 0.20
    if cue:
        sig += 0.35
    if fc.height_pt <= 130:
        sig += 0.10
    sig = min(sig, 1.0)

    if sig >= 0.75:
        return "signature", sig
    if fc.source == "vector":
        return ("chart" if colours > 24 else "diagram"), sig
    if colours > 900 and ink > 0.55:
        return "photo", sig
    if 0.85 <= aspect <= 1.2 and colours <= 60 and ink < 0.35:
        return "logo", sig
    if colours <= 120 and ink < 0.45:
        return "diagram", sig
    return "image", sig


def _nearby_text(page, rect, margin: float = 46.0) -> str:
    band = pymupdf.Rect(
        max(0, rect.x0 - margin),
        max(0, rect.y0 - margin),
        min(page.rect.x1, rect.x1 + margin),
        min(page.rect.y1, rect.y1 + margin),
    )
    try:
        txt = page.get_textbox(band) or ""
    except Exception:  # pragma: no cover
        return ""
    return re.sub(r"\s+", " ", txt).strip()[:600]


# ------------------------------------------------------------ figure finder

def _raster_figures(page, page_index: int) -> List[FigureCandidate]:
    out: List[FigureCandidate] = []
    seen = set()
    try:
        images = page.get_images(full=True)
    except Exception:  # pragma: no cover
        return out

    for info in images:
        xref = info[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if not _significant(rect, page.rect):
                continue
            out.append(
                FigureCandidate(
                    page_number=page_index + 1,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    source="raster",
                )
            )
    return out


def _vector_figures(page, page_index: int) -> List[FigureCandidate]:
    """Charts and diagrams drawn as vector paths carry no image object.

    Any extractor that only walks /XObject images returns nothing for them,
    which is why flowcharts and bar charts usually go missing. We cluster the
    drawing primitives instead and treat a dense cluster as one figure.
    """
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover
        return []
    if len(drawings) < MIN_VECTOR_PATHS:
        return []

    rects = []
    for d in drawings:
        r = d.get("rect")
        if r is None or r.is_infinite:
            continue
        # A perfectly horizontal or vertical stroke has zero height or width,
        # which PyMuPDF reports as an "empty" rect. Those are exactly the axis
        # lines and flowchart connectors that define a diagram, so give them a
        # hairline thickness rather than throwing them away.
        if r.width <= 0 or r.height <= 0:
            if r.width <= 0 and r.height <= 0:
                continue                       # a genuine point, not a stroke
            r = pymupdf.Rect(r.x0, r.y0, max(r.x1, r.x0 + 0.6), max(r.y1, r.y0 + 0.6))
        # skip specks
        if r.width < 3 and r.height < 3:
            continue
        if r.width > page.rect.width * 0.985 and r.height < 3:
            continue
        if r.height > page.rect.height * 0.985 and r.width < 3:
            continue
        rects.append(r)

    if len(rects) < MIN_VECTOR_PATHS:
        return []

    gap = max(VECTOR_MERGE_GAP, page.rect.width * 0.055)
    clusters = _merge_rects(rects, gap)
    out = []
    for c in clusters:
        members = sum(1 for r in rects if _rects_overlap_or_near(c, r, gap))
        if members < MIN_VECTOR_PATHS:
            continue
        if not _significant(c, page.rect):
            continue
        out.append(
            FigureCandidate(
                page_number=page_index + 1,
                bbox=(c.x0, c.y0, c.x1, c.y1),
                source="vector",
                extra={"path_count": members},
            )
        )
    return out


def _dedupe(figures: List[FigureCandidate]) -> List[FigureCandidate]:
    """Drop a vector cluster that merely traces a raster image already found."""
    kept: List[FigureCandidate] = []
    for f in figures:
        fr = pymupdf.Rect(*f.bbox)
        clash = False
        for k in kept:
            kr = pymupdf.Rect(*k.bbox)
            inter = fr & kr
            if inter.is_empty:
                continue
            overlap = (inter.width * inter.height) / max(
                1.0, min(fr.width * fr.height, kr.width * kr.height)
            )
            if overlap > 0.7:
                clash = True
                break
        if not clash:
            kept.append(f)
    return kept


# -------------------------------------------------------------- public API

def iter_pages(path: str, extract_figures: bool = True) -> Iterator[PageContent]:
    """Yield one page at a time. Memory stays flat regardless of file size."""
    doc = pymupdf.open(path)
    budget = MAX_FIGURES_PER_DOC
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            try:
                text = page.get_text("text") or ""
            except Exception:  # pragma: no cover
                text = ""

            content = PageContent(
                number=i + 1,
                text=text,
                width=page.rect.width,
                height=page.rect.height,
                has_text_layer=bool(text.strip()),
            )

            if extract_figures and budget > 0:
                found = _dedupe(_raster_figures(page, i) + _vector_figures(page, i))
                for fc in found[:budget]:
                    rect = pymupdf.Rect(*fc.bbox)
                    try:
                        fc.png = _render_crop(page, rect)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("crop render failed p%s: %s", i + 1, exc)
                        continue
                    fc.sha1 = hashlib.sha1(fc.png).hexdigest()
                    fc.width_pt = rect.width
                    fc.height_pt = rect.height
                    fc.aspect = rect.width / rect.height if rect.height else 0.0
                    fc.ink_ratio, fc.colour_count = _pixel_stats(fc.png)
                    fc.nearby_text = _nearby_text(page, rect)
                    caption = CAPTION_CUES.search(fc.nearby_text)
                    fc.caption_hint = caption.group(0) if caption else ""
                    fc.guess, fc.signature_score = _classify_cheap(fc)
                    content.figures.append(fc)
                budget -= len(content.figures)

            yield content
            page = None
    finally:
        doc.close()


def mark_repeated(figures: List[FigureCandidate], page_count: int) -> None:
    """Flag artwork that appears on most pages - letterheads, watermarks.

    Without this, a company logo in the header becomes N "figures" in an
    N-page document and drowns out the real content.
    """
    if page_count < 3:
        return
    counts: dict = {}
    for f in figures:
        if f.sha1:
            counts[f.sha1] = counts.get(f.sha1, 0) + 1
    threshold = max(3, int(page_count * 0.45))
    for f in figures:
        if f.sha1 and counts.get(f.sha1, 0) >= threshold:
            f.repeated = True
            if f.guess in ("logo", "image", "unknown"):
                f.guess = "letterhead"


def quick_stats(path: str) -> dict:
    """Cheap metadata read without processing the body."""
    doc = pymupdf.open(path)
    try:
        # Check this first: loading any page of a locked file raises, and the
        # raw error ("document closed or encrypted") tells a user nothing.
        if doc.is_encrypted and doc.needs_pass:
            return {"page_count": 0, "title": "", "author": "",
                    "is_scanned": False, "encrypted": True}
        pages = doc.page_count
        meta = doc.metadata or {}
        sample = ""
        for i in range(min(3, pages)):
            sample += doc.load_page(i).get_text("text") or ""
        return {
            "page_count": pages,
            "title": (meta.get("title") or "").strip(),
            "author": (meta.get("author") or "").strip(),
            "is_scanned": len(sample.strip()) < 60 and pages > 0,
            "encrypted": bool(doc.is_encrypted),
        }
    finally:
        doc.close()


def extract_page_text(path: str, page_number: int) -> str:
    doc = pymupdf.open(path)
    try:
        if 1 <= page_number <= doc.page_count:
            return doc.load_page(page_number - 1).get_text("text") or ""
        return ""
    finally:
        doc.close()
