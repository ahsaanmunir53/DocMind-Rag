"""The gap interview.

Diagnosis without a remedy is the flaw in every free CV checker: it reports
that twenty requirements have no supporting evidence and stops, as though the
user could not have written the evidence down had anyone asked.

Usually they could. The experience is real and the CV is just thin. So for each
gap the tool asks a direct question, and the answers become new bullets. The
facts come from the user, which is what keeps the result honest - nothing is
inferred on their behalf.

Questions are generated deterministically when no model is available, and
sharpened by the model when there is one.
"""

from __future__ import annotations

import re
from typing import Dict, List

from core import groq_client as groq
from resume import synonyms

MAX_QUESTIONS = 12

_SYSTEM = """You interview a candidate to uncover experience their CV failed to mention.

You are given: a job's requirements, and the parts of them the CV does not evidence.

Write one question per gap. Rules:
- Ask whether they have done the thing, and ask for the scale or outcome in the same breath.
- Never assume they have the experience. "Have you ever..." not "Describe how you...".
- Plain language. No jargon the posting did not use.
- One sentence. Under 25 words.
- Skip any gap that is a personal attribute rather than experience.

Return ONLY this JSON:
{"questions": [{"requirement": "the gap, short", "question": "the question", "hint": "what a good answer includes, under 12 words"}]}"""

# Most specific first: "inventory management" is about volume, not headcount.
_ROLE_HINTS = [
    (r"\b(erp|crm|sap|oracle|software|platform|system)\b", "Which product, and for how long?"),
    (r"\b(inventory|stock|sku|volume|throughput|transaction|record)", "What sort of volume?"),
    (r"\b(budget|cost|spend|procure|purchas|tender|contract)", "Roughly what value did you handle?"),
    (r"\b(vendor|supplier|client|customer|stakeholder)", "How many, and how senior?"),
    (r"\b(report|dashboard|analys|analyz|forecast)", "Who read them, and how often?"),
    (r"\b(audit|complian|regulat|policy|standard)", "Which standard or regulator?"),
    (r"\b(process|procedure|sop|workflow|improve)\b", "What changed as a result?"),
    (r"\b(team|staff|supervis|mentor|train|people)", "How many people, and for how long?"),
    (r"\b(manage|lead|led|head)\b", "What did you own, and at what scale?"),
]


def _hint_for(text: str) -> str:
    low = text.lower()
    for pattern, hint in _ROLE_HINTS:
        if re.search(pattern, low):
            return hint
    return "Include a number if you can."


_LEAD_IN = re.compile(
    r"^(?:"
    r"(?:a\s+)?minimum(?:\s+of)?\s+\d+\+?\s*(?:-\s*\d+\s*)?years?(?:\s+of)?|"
    r"at\s+least\s+\d+\+?\s*years?(?:\s+of)?|"
    r"\d+\+?\s*(?:-\s*\d+\s*)?years?(?:\s+of)?|"
    r"(?:the\s+)?(?:proven\s+|demonstrated\s+|hands-on\s+|strong\s+|excellent\s+|solid\s+|good\s+)*"
    r"(?:ability|able|capacity|capability)\s+(?:to|in|with)|"
    r"(?:proven|demonstrated|hands-on|strong|excellent|solid|good|working|practical|extensive|relevant|related)"
    r")\s+",
    re.I,
)

_FILLER = re.compile(r"^(experience|knowledge|understanding|exposure|background)\s+(in|of|with)\s+", re.I)

# Credentials are a yes-or-no fact already on the CV, not experience to
# describe. Asking "have you worked on a high school diploma" is nonsense.
_CREDENTIAL = re.compile(
    r"\b(diploma|degree|bachelor|master|phd|matric|intermediate|fsc|bsc|msc|"
    r"certification|certificate|licence|license|graduate|graduation|"
    r"fluency|fluent|native speaker|driving licen[cs]e|nationality|citizen)\b",
    re.I,
)


def _shorten(req: str) -> str:
    """Strip the recruiter preamble so the question is about the actual work."""
    s = re.sub(r"\s+", " ", str(req)).strip(" .;:-")
    for _ in range(3):
        before = s
        s = _LEAD_IN.sub("", s)
        s = _FILLER.sub("", s)
        if s == before:
            break
    return s.strip(" .;:-")[:120]


def _decap(s: str) -> str:
    """Lower the first letter unless it opens an acronym like ERP or SAP."""
    if len(s) > 1 and s[1].isupper():
        return s
    return s[0].lower() + s[1:] if s else s


def _fallback_questions(gaps: List[str]) -> List[Dict]:
    out = []
    for gap in gaps[:MAX_QUESTIONS]:
        if _CREDENTIAL.search(gap):
            continue
        short = _shorten(gap)
        if not short or len(short) < 6:
            continue
        out.append({
            "requirement": short,
            "question": f"Have you worked on {_decap(short)}? If yes, what did you do?",
            "hint": _hint_for(short),
        })
    return out


def _as_text(item) -> str:
    """gap_analysis stores unmet requirements as dicts and missing keywords as
    dicts too. Accept either shape, plus a plain string."""
    if isinstance(item, dict):
        return str(item.get("requirement") or item.get("term") or "").strip()
    return str(item or "").strip()


def build_questions(requirements: Dict, gap_analysis: Dict) -> List[Dict]:
    """One question per unevidenced requirement, model-sharpened when possible."""
    gaps = [_as_text(g) for g in (gap_analysis.get("unmet_requirements") or [])]
    gaps = [g for g in gaps if len(g) > 5]

    if not gaps:
        gaps = [_as_text(m) for m in (gap_analysis.get("missing_keywords") or [])]
        gaps = [g for g in gaps if len(g) > 3]

    gaps = gaps[:MAX_QUESTIONS]
    if not gaps:
        return []

    if not groq.configured():
        return _fallback_questions(gaps)

    listed = "\n".join(f"- {g}" for g in gaps)
    user = (
        f"=== JOB TITLE ===\n{requirements.get('title_guess', 'Unknown')}\n\n"
        f"=== REQUIREMENTS THE CV DOES NOT EVIDENCE ===\n{listed}\n\n"
        "Write one question per gap."
    )
    try:
        data = groq.chat_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            role="fast",
            max_tokens=1400,
            temperature=0.3,
        )
    except groq.GroqError:
        return _fallback_questions(gaps)

    out = []
    for item in (data.get("questions") or [])[:MAX_QUESTIONS]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        out.append({
            "requirement": str(item.get("requirement", "")).strip()[:120],
            "question": q[:240],
            "hint": str(item.get("hint", "")).strip()[:90] or _hint_for(q),
        })
    return out or _fallback_questions(gaps)


def answers_to_evidence(questions: List[Dict], answers: Dict[str, str]) -> List[Dict]:
    """Pair each answered question with its requirement.

    Blank and dismissive answers are dropped - "no" is a legitimate response
    and must not become a bullet.
    """
    negatives = {"no", "n/a", "na", "none", "nope", "never", "-", "no experience"}
    evidence = []
    for idx, q in enumerate(questions):
        raw = (answers.get(str(idx)) or answers.get(idx) or "").strip()
        if not raw or raw.lower() in negatives or len(raw) < 4:
            continue
        evidence.append({
            "requirement": q.get("requirement", ""),
            "question": q.get("question", ""),
            "answer": raw[:800],
        })
    return evidence


def evidence_text(evidence: List[Dict]) -> str:
    """The block handed to the tailoring prompt as user-asserted fact."""
    if not evidence:
        return ""
    lines = ["=== EXPERIENCE THE CANDIDATE CONFIRMED IN INTERVIEW ==="]
    lines.append(
        "Treat every statement below as fact supplied by the candidate. "
        "You may write bullets from it. Do not embellish it."
    )
    for e in evidence:
        lines.append(f"\nQ: {e['question']}\nA: {e['answer']}")
    return "\n".join(lines)


def coverage(evidence: List[Dict], requirements: Dict) -> Dict:
    """How much of the gap the interview actually closed."""
    answered = " ".join(e["answer"] for e in evidence).lower()
    tokens = set(re.findall(r"[a-z][a-z0-9+#./-]+", answered))

    keywords = requirements.get("keywords") or []
    newly = [k for k in keywords if synonyms.matches(k, tokens)]

    return {
        "answered": len(evidence),
        "terms_added": sorted(set(newly))[:30],
        "term_count": len(set(newly)),
    }
