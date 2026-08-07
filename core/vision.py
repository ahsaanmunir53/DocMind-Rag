"""
Vision layer.

Sends cropped figures to Groq's multimodal model and gets back structured
descriptions: what the figure is, what it says, what its labels are, and
whether it contains a handwritten signature or a stamp.

Two things make this affordable on a free tier:

  * Batching. The model accepts up to 5 images per request, so figures go up
    five at a time rather than one call each.
  * Pre-filtering. pdf.py has already measured every candidate and thrown out
    letterheads, hairlines and page furniture, so we only pay for regions
    that carry information.

If no key is configured the geometry-based guess from pdf.py stands on its
own. The pipeline still works; the labels are just coarser.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from core import groq_client as groq

logger = logging.getLogger(__name__)

KINDS = [
    "signature", "stamp_or_seal", "chart", "diagram", "flowchart", "table_image",
    "photo", "screenshot", "logo", "letterhead", "handwriting", "formula",
    "map", "floor_plan", "id_document", "barcode_or_qr", "other",
]

SYSTEM = """You analyse figures cropped out of PDF documents. You are precise and you never invent detail that is not visible.

For each image you are given, return an object with:
  index          the image's position in this batch, starting at 1
  kind           one of: """ + ", ".join(KINDS) + """
  caption        one sentence describing what it shows, in plain language
  text           any text legibly readable inside the image, verbatim; "" if none
  labels         array of short labels, axis names, node names or legend entries
  has_signature  true only if a handwritten signature mark is present
  has_stamp      true only if an official stamp, seal or embossed mark is present
  is_decorative  true if it is a logo, letterhead, border or page furniture carrying no information
  confidence     0.0 to 1.0

Rules:
- A signature is a handwritten personal mark. A typed name is NOT a signature.
- Read numbers off charts only if the value labels are actually printed.
- If the crop is too small or blurred to read, say so in caption and set confidence low.
- Never guess a person's identity from a signature. Describe it, do not name it.

Return ONLY: {"figures": [ ... ]}"""


def _batch_prompt(batch) -> List[Dict]:
    parts: List[Dict] = [{
        "type": "text",
        "text": (
            f"Analyse these {len(batch)} figures. Context lines found near each in the "
            "document are listed below; use them only as a hint.\n\n"
            + "\n".join(
                f"{i}. page {f.page_number}"
                + (f", nearby text: \"{f.nearby_text[:180]}\"" if f.nearby_text else "")
                for i, f in enumerate(batch, 1)
            )
        ),
    }]
    for f in batch:
        parts.append(groq.image_part(f.png))
    return parts


def describe_figures(figures: List, on_progress=None) -> List[Dict]:
    """Classify figures in batches. Always returns one result per figure."""
    results: List[Dict] = [_fallback(f) for f in figures]
    if not figures or not groq.configured():
        return results

    size = groq.MAX_IMAGES_PER_REQUEST
    for start in range(0, len(figures), size):
        batch = figures[start : start + size]
        pngs = [f.png for f in batch]

        while len(batch) > 1 and not groq.fits_budget(pngs):
            batch = batch[:-1]           # shed the last image until it fits
            pngs = [f.png for f in batch]
        if not groq.fits_budget(pngs):
            logger.info("figure at page %s too large for one request; keeping heuristic",
                        batch[0].page_number)
            continue

        try:
            data = groq.chat_json(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _batch_prompt(batch)},
                ],
                role="vision",
                max_tokens=1600,
                temperature=0.1,
            )
        except groq.GroqError as exc:
            logger.warning("vision batch failed (%s); heuristics retained", exc)
            continue

        items = data.get("figures") if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < len(batch):
                continue
            results[start + idx] = _merge(batch[idx], item)

        if on_progress:
            on_progress(min(start + size, len(figures)), len(figures))

    return results


def _fallback(f) -> Dict:
    """What we know without the model - geometry and pixels only."""
    return {
        "kind": f.guess,
        "caption": _describe_offline(f),
        "text": "",
        "labels": [],
        "has_signature": f.signature_score >= 0.75,
        "has_stamp": False,
        "is_decorative": f.repeated or f.guess in ("logo", "letterhead"),
        "confidence": 0.35 if f.guess != "unknown" else 0.15,
        "analysed_by": "heuristics",
    }


def _describe_offline(f) -> str:
    where = f"page {f.page_number}"
    size = f"{int(f.width_pt)}×{int(f.height_pt)}pt"
    if f.guess == "signature":
        return f"Probable signature on {where} ({size}), based on stroke density and nearby wording."
    if f.guess == "letterhead":
        return f"Repeated artwork on {where} - most likely a letterhead or watermark."
    if f.guess in ("chart", "diagram"):
        paths = f.extra.get("path_count")
        detail = f" built from {paths} vector paths" if paths else ""
        return f"Vector {f.guess} on {where} ({size}){detail}."
    return f"{f.guess.title()} on {where} ({size})."


def _merge(f, item: Dict) -> Dict:
    kind = str(item.get("kind", "") or f.guess).strip()
    if kind not in KINDS:
        kind = f.guess if f.guess in KINDS else "other"

    has_sig = bool(item.get("has_signature"))
    # The model is authoritative on what a signature looks like, but if the
    # page literally says "Signature:" next to a low-ink mark, trust that too.
    if not has_sig and f.signature_score >= 0.85:
        has_sig = True
    if has_sig and kind not in ("signature", "handwriting"):
        kind = "signature"

    labels = item.get("labels") or []
    if not isinstance(labels, list):
        labels = []

    return {
        "kind": kind,
        "caption": str(item.get("caption", "") or "").strip()[:600],
        "text": str(item.get("text", "") or "").strip()[:4000],
        "labels": [str(x)[:80] for x in labels[:24]],
        "has_signature": has_sig,
        "has_stamp": bool(item.get("has_stamp")),
        "is_decorative": bool(item.get("is_decorative")) or f.repeated,
        "confidence": _clamp(item.get("confidence")),
        "analysed_by": "vision",
    }


def _clamp(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def searchable_text(figure_row) -> str:
    """Turn a figure into a passage the retriever can match against.

    Without this, asking "what does the flowchart on page 4 show" finds
    nothing, because a diagram contributes no words to the text layer.
    """
    bits = [f"[{figure_row.kind.replace('_', ' ')} on page {figure_row.page_number}]"]
    if figure_row.caption:
        bits.append(figure_row.caption)
    if figure_row.labels:
        bits.append("Labels: " + ", ".join(figure_row.labels))
    if figure_row.ocr_text:
        bits.append("Text in figure: " + figure_row.ocr_text)
    if figure_row.has_signature:
        bits.append("This figure contains a handwritten signature.")
    if figure_row.has_stamp:
        bits.append("This figure contains a stamp or official seal.")
    return "\n".join(bits)
