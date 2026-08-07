"""
ATS scoring and gap analysis.

The score is computed in Python, not asked of a model. Two reasons: a model
gives a different number each time you ask, and a number nobody can explain is
worth nothing to the person trying to improve it. Every point here traces to a
rule you can read.

Five components, weighted by how much each actually affects an outcome:

  Format safety   35   a CV the parser mangles never gets read at all
  Keyword match   30   how ranking works in practice
  Content quality 20   quantified, verb-led bullets
  Structure       10   sections the parser recognises
  Contact          5   reachability

Format is weighted highest deliberately. Perfect keywords in a two-column
layout still lose to a plain CV that parses cleanly.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Set, Tuple

from resume import synonyms
from resume.parsing import ACTION_VERBS, QUANTIFIED, ParsedResume

WEIGHTS = {"format": 35, "keywords": 30, "content": 20, "structure": 10, "contact": 5}

SEVERITY_COST = {"critical": 14, "warning": 6, "note": 2}

REQUIRED_SECTIONS = ["experience", "education", "skills"]

STOP = {
    "and", "or", "the", "a", "an", "to", "of", "in", "for", "with", "on", "at",
    "by", "from", "as", "is", "are", "be", "will", "you", "your", "we", "our",
    "this", "that", "have", "has", "must", "should", "can", "able", "who",
    "role", "job", "work", "team", "years", "year", "experience", "strong",
    "good", "excellent", "including", "etc", "plus", "using", "use", "well",
    "new", "all", "any", "more", "other", "across", "within", "their", "them",

    "it", "its", "they", "he", "she", "his", "her", "him", "us", "me", "my",
    "was", "were", "been", "being", "do", "does", "did", "done", "not", "no",
    "but", "than", "then", "if", "when", "where", "while", "which", "what",
    "how", "why", "may", "might", "would", "could", "shall", "such", "both",
    "per", "via", "also", "into", "under", "over", "out", "up", "one", "two",
    "three", "each", "every", "some", "most", "very", "least",

    "apply", "applying", "application", "applications", "applicant",
    "applicants", "candidate", "candidates", "shortlist", "shortlisted",
    "interview", "interviews", "interviewed", "contact", "contacted",
    "submit", "submitted", "submission", "send", "sent", "email", "vacancy",
    "vacancies", "position", "positions", "post", "posting", "hiring", "hire",
    "recruit", "recruitment", "cv", "resume", "cover", "letter", "deadline",
    "closing", "consider", "considered", "note", "kindly", "please", "below",
    "above", "following", "further", "details", "detail", "attach", "attached",
    "reference", "subject", "line", "link", "portal", "website", "online",
    "successful", "unsuccessful", "selection", "selected", "process",

    "gender", "female", "male", "women", "men", "race", "religion", "ethnic",
    "ethnicity", "diversity", "inclusion", "inclusive", "disability",
    "disabled", "orientation", "nationality", "age", "marital", "equal",
    "opportunity", "employer", "regardless", "encouraged", "encourage",
    "committed", "commitment", "background", "backgrounds",

    "relevant", "related", "various", "ability", "abilities", "high", "higher",
    "minimum", "maximum", "required", "require", "requires", "requirement",
    "requirements", "responsibilities", "responsibility", "duties", "duty",
    "task", "tasks", "qualification", "qualifications", "desirable",
    "preferred", "essential", "only", "ensure", "ensuring", "provide",
    "provided", "providing", "through", "specified", "based", "upon",
    "willing", "willingness", "expected", "expect",

    "announcement", "title", "station", "location", "department", "grade",
    "duration", "salary", "type", "category",

    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
}

# Phrases that mark a hard requirement in a job description
HARD_CUES = re.compile(
    r"\b(required|must have|must be|essential|mandatory|minimum|at least|"
    r"proven|demonstrated|hands-on)\b",
    re.I,
)
NICE_CUES = re.compile(r"\b(nice to have|preferred|desirable|bonus|plus|advantage)\b", re.I)

YEARS = re.compile(r"(\d+)\+?\s*(?:-\s*\d+\s*)?year", re.I)

# Lines about the application process or equal-opportunity policy state no
# requirement. Counting their words drowns the real ones.
BOILERPLATE_LINE = re.compile(
    r"(only\s+short-?listed|will\s+be\s+contacted|how\s+to\s+apply|"
    r"to\s+apply|please\s+(send|submit|apply|note)|kindly\s+(send|submit|note)|"
    r"closing\s+date|deadline|last\s+date|no\s+phone\s+calls|"
    r"equal\s+opportunit|does\s+not\s+discriminate|regardless\s+of\s+(race|sex|gender)|"
    r"qualified\s+(applicants|candidates)|women\s+are\s+encouraged|"
    r"applications?\s+(received|submitted|will|must|should)|"
    r"incomplete\s+applications?|canvassing|"
    r"@[\w.-]+\.\w+|https?://)",
    re.I,
)


def strip_boilerplate(job_description: str) -> str:
    """Drop application-process and policy lines from a posting."""
    kept = [
        ln for ln in (job_description or "").splitlines()
        if not BOILERPLATE_LINE.search(ln)
    ]
    return "\n".join(kept)


def _terms(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9+#./\-]{1,}", text or "")
    return [t.lower().strip(".-/") for t in raw if len(t) > 1]


def _phrases(text: str) -> Set[str]:
    """Single words plus two-word pairs, so 'machine learning' survives."""
    toks = [t for t in _terms(text) if t not in STOP]
    out = set(toks)
    for a, b in zip(toks, toks[1:]):
        out.add(f"{a} {b}")
    return out


def extract_requirements(job_description: str) -> Dict:
    """Pull requirements out of a JD without a model.

    Bullet lines carry the requirements in almost every posting; cue words
    separate must-have from nice-to-have.
    """
    body = strip_boilerplate(job_description)

    lines = [ln.strip(" \t•-–—*·") for ln in body.splitlines()]
    lines = [ln for ln in lines if ln]

    must: List[str] = []
    nice: List[str] = []
    for ln in lines:
        if len(ln) < 8 or len(ln) > 260:
            continue
        if NICE_CUES.search(ln):
            nice.append(ln)
        elif HARD_CUES.search(ln) or ln.endswith((".", ";")) is False and len(ln) < 160:
            must.append(ln)

    counts = Counter(t for t in _terms(body) if t not in STOP and len(t) > 2)
    counts = Counter({t: c for t, c in counts.items() if c > 1 or len(t) > 3})
    keywords = [w for w, _ in counts.most_common(40)]

    years = [int(m) for m in YEARS.findall(body)]

    title = ""
    for ln in lines[:6]:
        if 4 < len(ln) < 80 and not ln.endswith((".", ":")):
            title = ln
            break

    return {
        "title_guess": title,
        "must_have": must[:25],
        "nice_to_have": nice[:15],
        "keywords": keywords,
        "years_required": max(years) if years else None,
        "keyword_weights": dict(counts.most_common(40)),
    }


def keyword_coverage(resume_text: str, requirements: Dict) -> Dict:
    """Which JD keywords appear in the CV, weighted by how often the JD repeats them."""
    have = _phrases(resume_text)
    weights = requirements.get("keyword_weights") or {}
    keywords = requirements.get("keywords") or []

    matched, missing = [], []
    total_w = 0.0
    got_w = 0.0

    for kw in keywords:
        w = float(weights.get(kw, 1))
        total_w += w
        if kw in have or any(kw in h for h in have if len(h) > len(kw)):
            matched.append(kw)
            got_w += w
        elif synonyms.matches(kw, have):
            matched.append(kw)
            got_w += w
        else:
            missing.append({"term": kw, "weight": w})

    missing.sort(key=lambda m: -m["weight"])
    pct = (got_w / total_w * 100) if total_w else 0.0
    return {
        "matched": matched,
        "missing": missing[:30],
        "coverage_percent": round(pct, 1),
        "matched_count": len(matched),
        "total_count": len(keywords),
    }


def _format_score(parsed: ParsedResume) -> Tuple[float, List[str]]:
    penalty = 0
    notes = []
    for issue in parsed.issues:
        cost = SEVERITY_COST.get(issue.severity, 2)
        penalty += cost
        notes.append(f"-{cost} {issue.title}")
    score = max(0.0, 100.0 - penalty)
    if not parsed.issues:
        notes.append("No parsing risks found")
    return score, notes


def _content_score(parsed: ParsedResume) -> Tuple[float, List[str]]:
    bullets = parsed.bullets
    notes = []
    if not bullets:
        return 25.0, ["No bullet points found - experience reads as a wall of text"]

    quant = sum(1 for b in bullets if QUANTIFIED.search(b))
    strong = sum(
        1 for b in bullets if b.split() and b.split()[0].lower().rstrip(":,.") in ACTION_VERBS
    )
    long_bullets = sum(1 for b in bullets if len(b.split()) > 42)

    quant_pct = quant / len(bullets)
    strong_pct = strong / len(bullets)

    score = 40.0 + quant_pct * 35.0 + strong_pct * 25.0
    if long_bullets > len(bullets) * 0.3:
        score -= 10
        notes.append(f"{long_bullets} bullets run past 40 words")

    notes.append(f"{quant}/{len(bullets)} bullets carry a number")
    notes.append(f"{strong}/{len(bullets)} bullets open with an action verb")
    return max(0.0, min(100.0, score)), notes


def _structure_score(parsed: ParsedResume) -> Tuple[float, List[str]]:
    found = set(parsed.sections.keys())
    have = [s for s in REQUIRED_SECTIONS if s in found]
    missing = [s for s in REQUIRED_SECTIONS if s not in found]
    score = 100.0 * len(have) / len(REQUIRED_SECTIONS)

    notes = []
    if missing:
        notes.append("Missing recognisable heading: " + ", ".join(missing))
    unusual = [s.split(":", 1)[1] for s in found if s.startswith("other:")]
    if unusual:
        score -= min(20, 5 * len(unusual))
        notes.append("Unrecognised headings: " + ", ".join(unusual[:4]))
    if not notes:
        notes.append("All standard sections present")
    return max(0.0, min(100.0, score)), notes


def _contact_score(parsed: ParsedResume) -> Tuple[float, List[str]]:
    c = parsed.contact
    checks = [("name", c.get("name")), ("email", c.get("email")), ("phone", c.get("phone"))]
    present = [k for k, v in checks if v]
    score = 100.0 * len(present) / len(checks)
    notes = []
    absent = [k for k, v in checks if not v]
    if absent:
        notes.append("Not detected: " + ", ".join(absent))
    if c.get("linkedin"):
        score = min(100.0, score + 5)
    else:
        notes.append("No LinkedIn URL found")
    if not notes:
        notes.append("Contact block complete")
    return score, notes


def score(parsed: ParsedResume, requirements: Dict | None = None) -> Dict:
    """Full ATS score. Pass requirements to include keyword matching."""
    fmt, fmt_notes = _format_score(parsed)
    content, content_notes = _content_score(parsed)
    structure, structure_notes = _structure_score(parsed)
    contact, contact_notes = _contact_score(parsed)

    if requirements:
        cov = keyword_coverage(parsed.raw_text, requirements)
        keywords = cov["coverage_percent"]
        kw_notes = [f"{cov['matched_count']} of {cov['total_count']} job keywords present"]
    else:
        cov = None
        keywords = 0.0
        kw_notes = ["No job description supplied - keyword match not assessed"]

    components = {
        "format": {"score": round(fmt, 1), "weight": WEIGHTS["format"], "notes": fmt_notes},
        "keywords": {"score": round(keywords, 1), "weight": WEIGHTS["keywords"], "notes": kw_notes},
        "content": {"score": round(content, 1), "weight": WEIGHTS["content"], "notes": content_notes},
        "structure": {"score": round(structure, 1), "weight": WEIGHTS["structure"], "notes": structure_notes},
        "contact": {"score": round(contact, 1), "weight": WEIGHTS["contact"], "notes": contact_notes},
    }

    if requirements:
        total = sum(c["score"] * c["weight"] for c in components.values()) / 100.0
    else:
        # without a JD, redistribute the keyword weight rather than score a zero
        usable = {k: v for k, v in components.items() if k != "keywords"}
        w = sum(c["weight"] for c in usable.values())
        total = sum(c["score"] * c["weight"] for c in usable.values()) / w

    return {
        "overall": round(total, 1),
        "band": _band(total),
        "components": components,
        "coverage": cov,
        "assessed_against_job": bool(requirements),
    }


def _band(total: float) -> str:
    if total >= 85:
        return "strong"
    if total >= 70:
        return "solid"
    if total >= 55:
        return "needs work"
    return "at risk"


def gap_analysis(parsed: ParsedResume, requirements: Dict) -> Dict:
    """What is missing, and which gaps are worth fixing first."""
    cov = keyword_coverage(parsed.raw_text, requirements)
    resume_terms = _phrases(parsed.raw_text)

    unmet: List[Dict] = []
    for line in requirements.get("must_have", []):
        line_terms = {t for t in _terms(line) if t not in STOP and len(t) > 2}
        if not line_terms:
            continue
        direct = line_terms & resume_terms
        loose = {t for t in line_terms - direct if synonyms.matches(t, resume_terms)}
        hit = len(direct | loose) / len(line_terms)
        if hit < 0.4:
            unmet.append({"requirement": line, "coverage": round(hit * 100)})

    critical = [i for i in parsed.issues if i.severity == "critical"]

    priorities: List[str] = []
    if critical:
        priorities.append(
            f"Fix {len(critical)} formatting problem(s) first - "
            f"{critical[0].title.lower()} can stop the CV being read at all."
        )
    if cov["missing"]:
        top = ", ".join(m["term"] for m in cov["missing"][:6])
        priorities.append(f"Work these job terms into your wording where true: {top}.")
    if unmet:
        priorities.append(
            f"{len(unmet)} stated requirement(s) have no matching evidence in the CV."
        )
    stats = parsed.stats
    if stats.get("bullets") and stats.get("quantified_bullets", 0) < stats["bullets"] * 0.4:
        priorities.append("Fewer than 40% of your bullets carry a number - add scale or result.")

    return {
        "missing_keywords": cov["missing"],
        "matched_keywords": cov["matched"],
        "coverage_percent": cov["coverage_percent"],
        "unmet_requirements": unmet[:12],
        "format_blockers": [
            {"title": i.title, "detail": i.detail, "fix": i.fix} for i in critical
        ],
        "priorities": priorities,
    }
