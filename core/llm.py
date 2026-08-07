"""
Answer generation.

Given a question and the passages hybrid retrieval selected, write an answer
that stays inside them. The system prompt does most of the work here: the
difference between a useful RAG app and a confident liar is almost entirely
whether the model will say "that is not in these documents".

Citations are by page, because that is what a person can actually go and
check. Figures cite the page and say what kind of figure it was, so an answer
drawn from a chart reads as an answer drawn from a chart.

Backends:
  groq   - the default now. Free tier, current models.
  echo   - no key needed. Assembles a readable stub from the retrieved text so
           retrieval can be tested end to end before any key exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    document_title: str
    chunk_index: int
    score: float
    page_number: Optional[int] = None
    kind: str = "text"
    figure_id: Optional[int] = None


SYSTEM_PROMPT = """You answer questions using ONLY the document excerpts provided.

Rules:
- If the excerpts do not contain the answer, say so plainly: "That isn't in the documents you've uploaded." Then say what related information IS there, if any. Never fill the gap from general knowledge.
- Cite with the bracketed numbers, like [1] or [2][4], immediately after the claim they support.
- Some excerpts describe figures - charts, diagrams, signatures, stamps. When you use one, say what it is: "the flowchart on page 4 [3]".
- If excerpts disagree, say so and cite both rather than silently picking one.
- Quote exact wording only when the precise phrasing matters, and keep quotes short.
- Be direct and concise. No preamble, no restating the question."""


def _context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        where = f'"{c.document_title}"'
        if c.page_number:
            where += f", page {c.page_number}"
        marker = " [FIGURE]" if c.kind == "figure" else ""
        blocks.append(f"[{i}] (from {where}){marker}\n{c.text}")
    return "\n\n".join(blocks)


def _echo_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "**[demo mode]** Nothing in your documents matched that question. "
            "Set GROQ_API_KEY in .env for real answers - retrieval is already live."
        )
    top = chunks[0]
    preview = top.text.strip().replace("\n", " ")
    if len(preview) > 420:
        preview = preview[:420] + "…"
    where = f"page {top.page_number}" if top.page_number else f"part {top.chunk_index + 1}"
    figures = sum(1 for c in chunks if c.kind == "figure")
    extra = f" {figures} of them describe figures." if figures else ""
    return (
        "**[demo mode - retrieval is real, the wording is a stub]**\n\n"
        f"Best match for *{question}* is in \"{top.document_title}\", {where}:\n\n"
        f"> {preview}\n\n"
        f"{len(chunks)} passages were retrieved and ranked.{extra} "
        "Add GROQ_API_KEY to get a written answer synthesised across all of them."
    )


def _groq_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    from core import groq_client as groq

    try:
        return groq.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Document excerpts:\n\n{_context(chunks)}\n\nQuestion: {question}",
                },
            ],
            role="text",
            max_tokens=1200,
            temperature=0.2,
        )
    except groq.GroqError as exc:
        if str(exc) == "not_configured":
            return _echo_answer(question, chunks)
        return f"_The answering model is unavailable right now: {exc}_"


def _anthropic_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Document excerpts:\n\n{_context(chunks)}\n\nQuestion: {question}",
        }],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    backend = settings.LLM_BACKEND
    if backend == "groq":
        return _groq_answer(question, chunks)
    if backend == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            return _echo_answer(question, chunks)
        return _anthropic_answer(question, chunks)
    return _echo_answer(question, chunks)


CONDENSE_SYSTEM = (
    "Rewrite the follow-up question as a standalone question that keeps every "
    "detail needed to search a document set. Resolve pronouns from the history. "
    "Return only the rewritten question, nothing else."
)


def condense_question(question: str, history: List[dict]) -> str:
    """Turn "and what about the second one?" into something searchable.

    Retrieval sees only the query string, so an unresolved pronoun retrieves
    noise no matter how good the index is.
    """
    if not history:
        return question
    from core import groq_client as groq

    if not groq.configured():
        return question

    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-6:])
    try:
        out = groq.chat(
            [
                {"role": "system", "content": CONDENSE_SYSTEM},
                {"role": "user", "content": f"History:\n{convo}\n\nFollow-up: {question}"},
            ],
            role="fast",
            max_tokens=120,
            temperature=0.0,
        )
        cleaned = out.strip().strip('"')
        return cleaned if 3 < len(cleaned) < 400 else question
    except Exception:  # pragma: no cover
        return question
