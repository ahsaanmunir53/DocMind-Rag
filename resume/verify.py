"""Fabrication detection.

The tailoring prompt forbids inventing anything. A prompt is a request, not a
guarantee, and the failure it guards against is exactly the kind a model makes
willingly: the posting asks for SAP, the CV never mentions SAP, and "SAP" turns
up in a bullet because it fits the sentence.

So the output gets checked against the input. Anything material in the tailored
CV that cannot be traced to the source is surfaced to the user before they send
it anywhere. The check is deliberately noisy in one direction - it would rather
flag a rephrasing than miss an invention.

Answers supplied by the user during the gap interview count as source material,
because the user asserted them.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set

from resume import synonyms

# Capitalised words that are ordinary sentence starters, not proper nouns
_SENTENCE_WORDS = {
    "the", "a", "an", "i", "we", "my", "our", "this", "that", "these", "those",
    "and", "or", "but", "for", "with", "from", "to", "in", "on", "at", "by",
    "as", "of", "led", "built", "managed", "developed", "delivered", "created",
    "designed", "improved", "reduced", "increased", "maintained", "supported",
    "coordinated", "implemented", "deployed", "owned", "ran", "wrote", "drove",
    "achieved", "streamlined", "automated", "migrated", "collaborated",
    "partnered", "produced", "prepared", "analysed", "analyzed", "reviewed",
    "monitored", "trained", "mentored", "handled", "processed", "tracked",
    "resolved", "negotiated", "sourced", "consolidated", "standardised",
    "standardized", "conducted", "oversaw", "established", "introduced",
}

# Things worth flagging when new: acronyms, product names, versioned tools
_PROPER = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[+#.][A-Za-z0-9]+)*)\b")
_ACRONYM = re.compile(r"\b([A-Z]{2,}[0-9]*)\b")
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)\s*(%|percent|k\b|m\b|bn\b|million|billion)?")


def _norm_num(raw: str, unit: str = "") -> str:
    n = raw.replace(",", "").rstrip(".")
    return f"{n}{(unit or '').lower().strip()}"


def _is_year(raw: str) -> bool:
    n = raw.replace(",", "")
    return n.isdigit() and len(n) == 4 and 1950 <= int(n) <= 2100


def _numbers(text: str) -> Set[str]:
    """Figures that make a claim. Years are excluded - employment dates are
    already checked verbatim, so reporting them twice is noise."""
    out = set()
    for m in _NUMBER.finditer(text or ""):
        raw, unit = m.group(1), m.group(2) or ""
        if _is_year(raw) and not unit:
            continue
        out.add(_norm_num(raw, unit))
    return out


def _proper_nouns(text: str) -> Set[str]:
    out = set()
    for m in _PROPER.finditer(text or ""):
        w = m.group(1)
        if w.lower() in _SENTENCE_WORDS or len(w) < 2:
            continue
        out.add(w.lower())
    for m in _ACRONYM.finditer(text or ""):
        out.add(m.group(1).lower())
    return out


def _flatten(tailored: Dict) -> str:
    parts: List[str] = [tailored.get("summary", "")]
    parts += list(tailored.get("skills", []))
    for job in tailored.get("experience", []):
        parts += [job.get("company", ""), job.get("title", ""), job.get("dates", "")]
        parts += job.get("bullets", [])
    for proj in tailored.get("projects", []):
        parts.append(proj.get("name", ""))
        parts += proj.get("bullets", [])
    return "\n".join(p for p in parts if p)


def _entries(tailored: Dict) -> List[Dict]:
    return [
        {
            "company": (j.get("company") or "").strip(),
            "title": (j.get("title") or "").strip(),
            "dates": (j.get("dates") or "").strip(),
        }
        for j in tailored.get("experience", [])
    ]


def check(tailored: Dict, source_text: str, extra_sources: Iterable[str] = ()) -> Dict:
    """Compare a tailored CV against everything the user actually supplied.

    Returns a report with a `clean` flag and a list of findings. Each finding
    carries a severity: `critical` for a fabricated employment fact, `warning`
    for a new tool or number, `note` for softer drift.
    """
    source = "\n".join([source_text or "", *[s or "" for s in extra_sources]])
    src_low = source.lower()

    out_text = _flatten(tailored)
    findings: List[Dict] = []

    # --- employment facts must survive verbatim ---
    for entry in _entries(tailored):
        for field in ("company", "title", "dates"):
            val = entry[field]
            if not val:
                continue
            if val.lower() not in src_low:
                findings.append({
                    "severity": "critical",
                    "kind": field,
                    "value": val,
                    "message": f"{field.title()} \"{val}\" does not appear in your CV.",
                })

    # --- numbers cannot be conjured ---
    src_nums = _numbers(source)
    for num in sorted(_numbers(out_text) - src_nums):
        findings.append({
            "severity": "warning",
            "kind": "number",
            "value": num,
            "message": f"The figure \"{num}\" is not in your CV. Confirm it or remove it.",
        })

    # --- new named tools and systems ---
    src_props = _proper_nouns(source)
    src_alias = synonyms.expand_text(source)
    for word in sorted(_proper_nouns(out_text) - src_props):
        if word in src_alias:
            continue
        if re.search(rf"\b{re.escape(word)}", src_low):
            continue
        if any(word in s for s in src_props):
            continue
        findings.append({
            "severity": "warning",
            "kind": "term",
            "value": word,
            "message": f"\"{word}\" appears in the tailored CV but not in yours.",
        })

    # --- skills list additions ---
    for skill in tailored.get("skills", []):
        s = str(skill).strip().lower()
        if not s or s in src_low:
            continue
        if synonyms.matches(s, set(re.findall(r"[a-z][a-z0-9+#./-]+", src_low))):
            continue
        findings.append({
            "severity": "warning",
            "kind": "skill",
            "value": skill,
            "message": f"Skill \"{skill}\" is not evidenced in your CV.",
        })

    order = {"critical": 0, "warning": 1, "note": 2}
    findings.sort(key=lambda f: (order.get(f["severity"], 3), f["value"]))

    counts = {
        "critical": sum(1 for f in findings if f["severity"] == "critical"),
        "warning": sum(1 for f in findings if f["severity"] == "warning"),
        "note": sum(1 for f in findings if f["severity"] == "note"),
    }

    return {
        "clean": not findings,
        "findings": findings[:40],
        "counts": counts,
        "checked": {
            "numbers": len(_numbers(out_text)),
            "terms": len(_proper_nouns(out_text)),
            "entries": len(_entries(tailored)),
        },
        "summary": _summary(counts),
    }


def _summary(counts: Dict[str, int]) -> str:
    if not any(counts.values()):
        return "Every fact in the tailored CV traces back to something you supplied."
    bits = []
    if counts["critical"]:
        bits.append(f"{counts['critical']} employment detail(s) not found in your CV")
    if counts["warning"]:
        bits.append(f"{counts['warning']} new term(s) or figure(s)")
    return "Check before sending: " + ", ".join(bits) + "."
