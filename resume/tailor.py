"""
CV tailoring.

The model's job is narrow on purpose: reword what is already there so the
truthful version of it matches the language of the posting. It is explicitly
forbidden from adding experience, tools, employers or numbers.

That restriction is the whole product. A CV that claims Kubernetes because
the job asked for Kubernetes gets its owner through the filter and destroyed
in the interview, and there is no version of that which is a good outcome.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from core import groq_client as groq
from resume.parsing import ParsedResume

logger = logging.getLogger(__name__)

SYSTEM = """You tailor an existing CV to a specific job description. You are a careful editor, not a ghostwriter.

ABSOLUTE RULES - breaking any of these makes the output useless:
- Never invent an employer, job title, date, degree, certification, tool or number.
- Never claim a skill the CV does not already evidence.
- Never change dates, company names, or job titles.
- You may only REPHRASE what is present, REORDER for relevance, and SURFACE things buried in the CV.
- If the CV genuinely lacks something the job requires, do NOT paper over it. Report it in "not_evidenced".
- An INTERVIEW ANSWERS block, when present, is fact the candidate stated about themselves. Write bullets from it
  exactly as reported, keeping their numbers. Do not scale, round or embellish what they said.

WHAT GOOD LOOKS LIKE:
- Bullets open with a strong past-tense verb.
- Where the CV already contains a number, keep it and make it prominent.
- Use the job description's own vocabulary when it describes the same thing the CV already describes. "Built CI pipelines" becomes "Built CI/CD pipelines" only if the CV shows CD too.
- Keep each bullet under 32 words.
- British or American spelling: match whatever the original CV uses.

Return ONLY this JSON object:
{
  "summary": "3-4 line professional summary aimed at this role, built only from CV facts",
  "skills": ["reordered and regrouped skills, most relevant first - no additions"],
  "experience": [
    {"company": "as written in the CV", "title": "as written", "dates": "as written",
     "bullets": ["rewritten bullets"]}
  ],
  "projects": [{"name": "", "bullets": [""]}],
  "changes": [{"what": "short description of the edit", "why": "which job requirement it serves"}],
  "not_evidenced": ["job requirements the CV does not support - stated plainly"],
  "honesty_note": "one line confirming nothing was invented"
}"""


def _cv_payload(parsed: ParsedResume) -> str:
    lines = ["=== CURRENT CV ==="]
    for key in ("summary", "experience", "projects", "skills", "education",
                "certifications", "awards"):
        body = parsed.sections.get(key)
        if body:
            lines.append(f"\n## {key.upper()}")
            lines.extend(body[:120])
    if not any(parsed.sections.get(k) for k in ("experience", "summary")):
        lines.append(parsed.raw_text[:6000])
    return "\n".join(lines)[:16000]


def tailor(
    parsed: ParsedResume,
    job_description: str,
    requirements: Dict,
    evidence: str = "",
) -> Dict:
    """Produce a tailored CV structure. Raises GroqError if the model is unreachable.

    `evidence` carries anything the user confirmed during the gap interview.
    It is source material of exactly the same standing as the CV itself: the
    user asserted it, so writing a bullet from it is reporting, not inventing.
    """
    must = "\n".join(f"- {m}" for m in requirements.get("must_have", [])[:18])
    keywords = ", ".join(requirements.get("keywords", [])[:40])

    user = (
        f"{_cv_payload(parsed)}\n\n"
        f"=== TARGET JOB ===\n{job_description[:8000]}\n\n"
        f"=== KEY REQUIREMENTS ===\n{must}\n\n"
        f"=== TERMS THE POSTING USES ===\n{keywords}\n\n"
    )
    if evidence:
        user += f"{evidence[:6000]}\n\n"
    user += (
        "Rewrite the CV for this role, obeying every rule. Anything neither the CV "
        "nor the interview answers support goes in not_evidenced, never into the bullets."
    )

    data = groq.chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        role="text",
        max_tokens=3000,
        temperature=0.25,
    )
    return _normalise(data, parsed)


def _normalise(data: Dict, parsed: ParsedResume) -> Dict:
    if not isinstance(data, dict):
        return {}

    def strlist(v, cap=40):
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v[:cap] if str(x).strip()]

    experience = []
    for item in (data.get("experience") or [])[:12]:
        if not isinstance(item, dict):
            continue
        experience.append({
            "company": str(item.get("company", "")).strip()[:120],
            "title": str(item.get("title", "")).strip()[:120],
            "dates": str(item.get("dates", "")).strip()[:60],
            "bullets": strlist(item.get("bullets"), 10),
        })

    projects = []
    for item in (data.get("projects") or [])[:8]:
        if not isinstance(item, dict):
            continue
        projects.append({
            "name": str(item.get("name", "")).strip()[:120],
            "bullets": strlist(item.get("bullets"), 6),
        })

    changes = []
    for item in (data.get("changes") or [])[:30]:
        if isinstance(item, dict):
            changes.append({
                "what": str(item.get("what", "")).strip()[:220],
                "why": str(item.get("why", "")).strip()[:220],
            })

    return {
        "summary": str(data.get("summary", "")).strip()[:900],
        "skills": strlist(data.get("skills"), 60),
        "experience": experience,
        "projects": projects,
        "education": parsed.sections.get("education", [])[:20],
        "certifications": parsed.sections.get("certifications", [])[:20],
        "changes": changes,
        "not_evidenced": strlist(data.get("not_evidenced"), 15),
        "honesty_note": str(data.get("honesty_note", "")).strip()[:300],
    }


def rendered_text(tailored: Dict, contact: Dict) -> str:
    """Flatten the tailored structure for re-scoring and .txt export."""
    parts: List[str] = []
    if contact.get("name"):
        parts.append(contact["name"])
    line = " | ".join(
        v for v in (contact.get("email"), contact.get("phone"), contact.get("location"),
                    contact.get("linkedin"), contact.get("github")) if v
    )
    if line:
        parts.append(line)

    if tailored.get("summary"):
        parts += ["", "SUMMARY", tailored["summary"]]
    if tailored.get("skills"):
        parts += ["", "SKILLS", ", ".join(tailored["skills"])]

    if tailored.get("experience"):
        parts += ["", "EXPERIENCE"]
        for job in tailored["experience"]:
            head = " — ".join(x for x in (job.get("title"), job.get("company")) if x)
            if job.get("dates"):
                head = f"{head} ({job['dates']})" if head else job["dates"]
            parts.append(head)
            parts += [f"- {b}" for b in job.get("bullets", [])]

    if tailored.get("projects"):
        parts += ["", "PROJECTS"]
        for p in tailored["projects"]:
            parts.append(p.get("name", ""))
            parts += [f"- {b}" for b in p.get("bullets", [])]

    for key, heading in (("education", "EDUCATION"), ("certifications", "CERTIFICATIONS")):
        if tailored.get(key):
            parts += ["", heading] + list(tailored[key])

    return "\n".join(p for p in parts if p is not None)
