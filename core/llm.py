"""
The answering LLM: given a question + the most relevant document chunks,
produce an answer grounded in those chunks (this is the "Generation" in RAG).

Two backends, chosen by settings.LLM_BACKEND:

  "echo"      -> no API key. Returns a readable stub answer assembled from the
                 retrieved chunks, so you can watch retrieval work end-to-end
                 before spending a cent. It clearly labels itself as demo output.

  "anthropic" -> real Claude answers (needs ANTHROPIC_API_KEY). The prompt tells
                 the model to answer ONLY from the provided context and to say so
                 when the answer isn't in the documents — the discipline that
                 makes RAG trustworthy instead of hallucinatory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from django.conf import settings


@dataclass
class RetrievedChunk:
    text: str
    document_title: str
    chunk_index: int
    score: float


SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the provided "
    "document excerpts. If the answer is not contained in the excerpts, say you "
    "could not find it in the documents rather than guessing. When you use a fact, "
    "cite the source like [1], [2] matching the numbered excerpts. Be concise."
)


def _build_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (from \"{c.document_title}\", part {c.chunk_index + 1})\n{c.text}")
    return "\n\n".join(lines)


def _echo_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "**[demo mode]** No relevant text was found in your documents for that "
            "question. (Set LLM_BACKEND=anthropic with an API key for real answers.)"
        )
    top = chunks[0]
    preview = top.text.strip().replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "…"
    cites = ", ".join(f"[{i}]" for i in range(1, len(chunks) + 1))
    return (
        f"**[demo mode — retrieval is real, wording is a stub]**\n\n"
        f"Your question: *{question}*\n\n"
        f"The most relevant passage I found {cites} says:\n\n"
        f"> {preview}\n\n"
        f"Turn on the real model (LLM_BACKEND=anthropic) to get a written answer "
        f"synthesized across all {len(chunks)} retrieved passages."
    )


def _anthropic_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    import anthropic  # lazy import

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    context = _build_context(chunks)
    user_msg = (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the excerpts above, citing sources like [1]."
    )
    resp = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _openai_compatible_answer(question: str, chunks: List[RetrievedChunk], *, api_key: str, base_url: str | None, model: str) -> str:
    """
    Works for any OpenAI-compatible chat API — used for both Groq and OpenAI.
    Groq just points base_url at Groq's endpoint; everything else is identical.
    """
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key, base_url=base_url)
    context = _build_context(chunks)
    user_msg = (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the excerpts above, citing sources like [1]."
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content or ""


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    backend = settings.LLM_BACKEND

    if backend == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set.")
        return _anthropic_answer(question, chunks)

    if backend == "groq":
        if not settings.GROQ_API_KEY:
            raise RuntimeError("LLM_BACKEND=groq but GROQ_API_KEY is not set.")
        return _openai_compatible_answer(
            question, chunks,
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model=settings.LLM_MODEL,
        )

    if backend == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("LLM_BACKEND=openai but OPENAI_API_KEY is not set.")
        return _openai_compatible_answer(
            question, chunks,
            api_key=settings.OPENAI_API_KEY,
            base_url=None,  # default OpenAI endpoint
            model=settings.LLM_MODEL,
        )

    return _echo_answer(question, chunks)
