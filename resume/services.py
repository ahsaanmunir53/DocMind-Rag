"""Resume pipeline orchestration."""

from __future__ import annotations

import logging

from django.core.files.base import ContentFile

from core import groq_client as groq
from resume import ats, export, interview, parsing, tailor, verify
from resume.models import Resume, Tailoring

logger = logging.getLogger(__name__)


def _parsed_from(resume: Resume) -> parsing.ParsedResume:
    """Rebuild the dataclass from stored JSON without re-reading the file."""
    p = parsing.ParsedResume(
        raw_text=resume.raw_text,
        contact=resume.contact,
        sections=resume.sections,
        skills=resume.skills,
        page_count=resume.page_count,
        word_count=resume.word_count,
    )
    p.bullets = parsing._collect_bullets(resume.sections)
    p.issues = [
        parsing.FormatIssue(**i) for i in (resume.format_report.get("issues") or [])
    ]
    p.stats = resume.format_report.get("stats", {})
    return p


def process_resume(resume_id: int) -> None:
    resume = Resume.objects.get(id=resume_id)
    try:
        resume.status = Resume.Status.PARSING
        resume.save(update_fields=["status"])

        parsed = parsing.parse(resume.file.path, resume.file.name)

        resume.raw_text = parsed.raw_text[:400_000]
        resume.contact = parsed.contact
        resume.sections = {k: v[:200] for k, v in parsed.sections.items()}
        resume.skills = parsed.skills
        resume.page_count = parsed.page_count
        resume.word_count = parsed.word_count
        resume.format_report = {
            "issues": [i.__dict__ for i in parsed.issues],
            "stats": parsed.stats,
        }
        resume.base_score = ats.score(parsed)
        resume.status = Resume.Status.READY
        resume.error = ""
        resume.save()
    except Exception as exc:  # noqa: BLE001
        logger.exception("resume parse failed %s", resume_id)
        Resume.objects.filter(id=resume_id).update(
            status=Resume.Status.FAILED,
            error=f"{exc.__class__.__name__}: {exc}"[:800],
        )


def process_tailoring(tailoring_id: int) -> None:
    job = Tailoring.objects.select_related("resume").get(id=tailoring_id)
    try:
        job.status = Tailoring.Status.WORKING
        job.save(update_fields=["status"])

        parsed = _parsed_from(job.resume)
        requirements = ats.extract_requirements(job.job_description)

        job.requirements = requirements
        job.before_score = ats.score(parsed, requirements)
        job.gap_analysis = ats.gap_analysis(parsed, requirements)

        if not job.job_title and requirements.get("title_guess"):
            job.job_title = requirements["title_guess"][:255]

        if not groq.configured():
            # Without a key the audit, score and gap analysis still stand -
            # only the rewriting needs a model.
            if not job.questions:
                job.questions = interview.build_questions(requirements, job.gap_analysis)

            evidence = interview.answers_to_evidence(job.questions, job.answers or {})
            job.evidence = evidence
            job.status = Tailoring.Status.READY
            job.tailored = {}
            job.change_log = [{
                "what": "Rewriting skipped",
                "why": "No GROQ_API_KEY configured. Scoring, format audit, gap "
                       "analysis and interview questions are computed locally "
                       "and are complete.",
            }]

            if evidence:
                # The answers cannot be turned into prose without a model, but
                # they are still real evidence. Rescoring against the CV plus
                # the answers shows what writing them in would be worth.
                answered = parsing.ParsedResume(
                    raw_text=parsed.raw_text + "\n" + "\n".join(e["answer"] for e in evidence),
                    contact=parsed.contact,
                    sections=parsed.sections,
                    skills=parsed.skills,
                    page_count=parsed.page_count,
                    word_count=parsed.word_count,
                )
                answered.bullets = parsed.bullets
                answered.issues = parsed.issues
                answered.stats = parsed.stats
                job.after_score = ats.score(answered, requirements)
                job.change_log.append({
                    "what": f"Rescored with {len(evidence)} interview answer(s)",
                    "why": "This is the score your CV would reach once these answers are "
                           "written into it. Add a key to have that done for you.",
                })

            job.save()
            return

        if not job.questions:
            job.questions = interview.build_questions(requirements, job.gap_analysis)

        evidence = interview.answers_to_evidence(job.questions, job.answers or {})
        job.evidence = evidence

        tailored = tailor.tailor(
            parsed, job.job_description, requirements,
            evidence=interview.evidence_text(evidence),
        )
        job.tailored = tailored
        job.change_log = tailored.get("changes", [])

        job.fabrication = verify.check(
            tailored,
            job.resume.raw_text,
            extra_sources=[e["answer"] for e in evidence],
        )

        text = tailor.rendered_text(tailored, job.resume.contact)
        after = parsing.ParsedResume(
            raw_text=text,
            contact=job.resume.contact,
            sections=_sections_from_tailored(tailored),
            skills=tailored.get("skills", []),
            page_count=1,
            word_count=len(text.split()),
        )
        after.bullets = _bullets_from_tailored(tailored)
        after.issues = []          # the export is clean by construction
        after.stats = {"bullets": len(after.bullets)}
        job.after_score = ats.score(after, requirements)

        stem = _stem(job)
        docx_bytes = export.build_docx(tailored, job.resume.contact)
        job.docx_file.save(f"{stem}.docx", ContentFile(docx_bytes), save=False)
        try:
            pdf_bytes = export.build_pdf(tailored, job.resume.contact)
            job.pdf_file.save(f"{stem}.pdf", ContentFile(pdf_bytes), save=False)
        except Exception:  # noqa: BLE001
            # A PDF failure must not cost the user the DOCX, which is the
            # format ATS vendors parse most reliably anyway.
            logger.exception("pdf export failed for tailoring %s", tailoring_id)
        job.txt_file.save(f"{stem}.txt", ContentFile(text.encode("utf-8")), save=False)

        job.status = Tailoring.Status.READY
        job.error = ""
        job.save()

    except groq.GroqError as exc:
        Tailoring.objects.filter(id=tailoring_id).update(
            status=Tailoring.Status.FAILED, error=str(exc)[:800]
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("tailoring failed %s", tailoring_id)
        Tailoring.objects.filter(id=tailoring_id).update(
            status=Tailoring.Status.FAILED,
            error=f"{exc.__class__.__name__}: {exc}"[:800],
        )


def _stem(job: Tailoring) -> str:
    name = (job.resume.contact.get("name") or "cv").replace(" ", "_")
    role = (job.job_title or "role").replace(" ", "_")
    safe = "".join(c for c in f"{name}_{role}" if c.isalnum() or c in "_-")
    return safe[:70] or "tailored_cv"


def _sections_from_tailored(t: dict) -> dict:
    sections = {}
    if t.get("summary"):
        sections["summary"] = [t["summary"]]
    if t.get("skills"):
        sections["skills"] = [", ".join(t["skills"])]
    exp = []
    for job in t.get("experience", []):
        head = " ".join(x for x in (job.get("title"), job.get("company"), job.get("dates")) if x)
        exp.append(head)
        exp += [f"- {b}" for b in job.get("bullets", [])]
    if exp:
        sections["experience"] = exp
    if t.get("education"):
        sections["education"] = list(t["education"])
    if t.get("certifications"):
        sections["certifications"] = list(t["certifications"])
    if t.get("projects"):
        proj = []
        for p in t["projects"]:
            proj.append(p.get("name", ""))
            proj += [f"- {b}" for b in p.get("bullets", [])]
        sections["projects"] = proj
    return sections


def _bullets_from_tailored(t: dict) -> list:
    out = []
    for job in t.get("experience", []):
        out += job.get("bullets", [])
    for p in t.get("projects", []):
        out += p.get("bullets", [])
    return out
