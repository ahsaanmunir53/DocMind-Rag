# DocuMind — AI Document Q&A

> **Chat with your PDFs.** Upload a document, ask questions in plain English, and
> get answers grounded in the content — with the exact source passages cited.
> A production-shaped **Retrieval-Augmented Generation (RAG)** app built on
> **Django + Django REST Framework**, with a 3D landing page.

![stack](https://img.shields.io/badge/Django-6-092E20) ![drf](https://img.shields.io/badge/DRF-REST_API-red) ![rag](https://img.shields.io/badge/RAG-pgvector-7c7cff) ![llm](https://img.shields.io/badge/LLM-Groq_·_Claude_·_OpenAI-a78bfa)

## What it demonstrates

- **Real RAG pipeline** — extraction → chunking → embeddings → vector search → grounded generation with citations. Not an API wrapper.
- **Solid Django** — custom apps, models with a DB-adaptive vector field, DRF API, session auth, file uploads, admin, migrations.
- **Production patterns** — Celery for async embedding, pgvector for scalable search, environment-driven config, WhiteNoise static serving, gunicorn, a Render blueprint.
- **A polished 3D landing page** (Three.js) so it *looks* like a product, not a class assignment.

## Try it in 2 minutes (no API keys, no external services)

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser        # optional: for /admin
python manage.py runserver
```

Open **http://127.0.0.1:8000** → you land on the 3D page → **Get started** →
create an account → upload a PDF → ask a question.

It runs out of the box on **SQLite + a local embedder + a stub answerer**, so
nothing external is required to see it work. Then flip env vars to go real.

## Architecture

```
Upload ─▶ Django/DRF saves file ─▶ (Celery) process:
                                    extract text (pypdf)
                                    ─▶ chunk with overlap
                                    ─▶ embed each chunk
                                    ─▶ store vectors (pgvector / JSON)
                                    ─▶ mark "ready"

Ask ─▶ embed question ─▶ vector search (top-K, per-user scope)
                       ─▶ send chunks + question to the LLM
                       ─▶ grounded answer + cited sources ─▶ chat history
```

| Layer | Module |
|-------|--------|
| Embeddings (local / OpenAI) | `core/embeddings.py` |
| PDF/text extraction + chunker | `core/textutils.py` |
| Answer generation (echo / Groq / Claude / OpenAI) | `core/llm.py` |
| Ingestion pipeline | `documents/services.py` |
| Vector retrieval (pgvector SQL / Python cosine) | `documents/retrieval.py` |
| Ask endpoint (retrieve → generate → save) | `chat/views.py` |
| 3D landing, signup, app pages | `templates/`, `config/views.py` |

## Everything is swappable by one env var

| Set in `.env` | Turns on |
|---|---|
| `LLM_BACKEND=groq` + `GROQ_API_KEY` | Fast, cheap answers (Llama 3.3 70B) |
| `LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY` | Claude answers (highest quality) |
| `EMBEDDING_BACKEND=openai` + `OPENAI_API_KEY` | Real semantic embeddings |
| `DATABASE_URL=postgres://…` | Postgres + **pgvector** |
| `CELERY_ENABLED=True` + Redis | Async processing (`celery -A config worker -l info`) |

## REST API (session auth; browsable at `/api-auth/`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET/POST | `/api/documents/` | list / upload documents |
| GET/DELETE | `/api/documents/{id}/` | detail (poll status) / delete |
| POST | `/api/chat/ask/` | `{question, document?, conversation?}` |
| GET | `/api/chat/conversations/` | list conversations |
| GET | `/api/chat/conversations/{id}/` | conversation + messages |

## Deploy (free tier)

A `render.yaml` blueprint is included. Push to GitHub → Render → **New → Blueprint**.
Set `GROQ_API_KEY` in the dashboard (never commit it), optionally attach a free
Postgres and set `DATABASE_URL` (enable the `vector` extension once). Note: free
web services sleep after ~15 min idle (first request is slow), and free uploads
are ephemeral — attach a disk or object storage for persistence.

## Tech stack

Python · Django 6 · Django REST Framework · PostgreSQL + pgvector · Celery + Redis ·
Groq / Anthropic / OpenAI · Three.js · WhiteNoise · gunicorn

## Honest notes

- The **local embedder** matches on word overlap (great for dev); switch to OpenAI for deep semantic retrieval.
- The **echo** LLM returns the top passage so you can see retrieval working without a key — real answers need `LLM_BACKEND=groq` (or anthropic/openai).
- Scanned (image-only) PDFs need OCR, which isn't included.

---

Built by **Ahsaan Munir** — MERN & Django developer.
