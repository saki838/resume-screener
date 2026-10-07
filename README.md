# AI Resume Screening & Ranking System

Ingests a folder of resumes, applies hard Python + AI eligibility rules, scores eligible
candidates on a 100-point explainable model, enriches with public GitHub activity, and
writes a ranked shortlist with evidence for every decision.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: GITHUB_TOKEN, ANTHROPIC_API_KEY

python main.py --input ./resumes --output ./output/results.json
pytest                          # 68 tests
python scripts/benchmark.py     # concurrency speed-up measurement
```

Outputs:

- `output/results.json`: batch summary, ranked candidates (score breakdown, evidence, penalties, GitHub, strengths/concerns), rejected candidates with reasons, failed/duplicate files. It is a JSON object (not a bare list) so the batch summary travels with the candidates.
- `output/candidates.json`: the same candidates as a bare list in exactly the assignment's example shapes (section 7 for ranked, section 3 for rejected).
- `output/report.html`: self-contained report; click a candidate for evidence, strengths, concerns and GitHub details.
- `output/report.md`: the same shortlist as a Markdown table.
- `docs/Value_Proposition.pdf`: value document for the company, users and author (`python docs/build_value_doc.py` regenerates it from results.json).

### About the shipped results

- Generated with **live GitHub enrichment**: 35 of 35 eligible candidates with a GitHub URL were enriched (repos + public events). The 4 others have no usable GitHub URL on their resume and are recorded as `not_provided`, without penalty.
- Generated in **deterministic mode** (no LLM key was used). Setting `ANTHROPIC_API_KEY` switches to hybrid scoring.
- To reproduce: put a `GITHUB_TOKEN` (no scopes needed) in `.env` and run the command above. Without a token the anonymous 60 requests/hour limit can be hit; the run then records `rate_limited` per candidate and, by default, withholds GitHub points from everyone so the ranking stays comparable.

CLI flags: `--no-llm`, `--no-github`, `--top N`, `-v`.

### Hybrid (LLM) mode: free and paid options

The LLM is optional. `LLM_PROVIDER=auto` picks the first one configured in `.env`:

| Option | Cost | Setup in `.env` | Notes |
|---|---|---|---|
| **Google Gemini** (recommended free option) | Free tier, no card | `GEMINI_API_KEY=...` (key from https://aistudio.google.com/apikey) | Defaults to `gemini-3.8-flash`, 1 request at a time, 6 s apart (~4 min for 39 resumes). The newest model is often overloaded (HTTP 503) and older ones can be closed to new accounts (HTTP 404): run `python scripts/check_llm_models.py` to see which model answers right now, and put the suggested `LLM_MODEL=` line in `.env`. |
| **Groq** (free option, reliable) | Free tier, no card | `GROQ_API_KEY=...` (key from https://console.groq.com/keys) and `LLM_PROVIDER=groq` | Defaults to `openai/gpt-oss-120b` with structured JSON output. Its free limit is 8K tokens/minute and 200K tokens/day, so resumes are sent ~30 s apart (~20 min for 39), which fits one day's budget. |
| **Ollama** (local) | Free, no key, data never leaves your PC | `LLM_PROVIDER=ollama` and `LLM_MODEL=<model you pulled>` | Install from https://ollama.com, run `ollama pull <model>` first. Needs a reasonably strong PC; small models judge less accurately. |
| Other OpenAI-compatible APIs (Groq, OpenRouter, ...) | Many have free models | `LLM_BASE_URL=...`, `LLM_MODEL=...`, `LLM_API_KEY=...` | Any endpoint that serves `/chat/completions`. |
| Anthropic Claude | Paid | `ANTHROPIC_API_KEY=...` | Uses a forced tool call for structured output. |

**Privacy:** free tiers (for example Gemini's) may let the provider use submitted
content to improve its products. Resumes are personal data, so for real candidates use
a paid tier or a local model (Ollama).

Then run:

```bash
python main.py --input ./resumes --output ./output/results.json
python docs/build_value_doc.py      # refresh the PDF
```

The console should show `Mode: hybrid (...)`, and each candidate gets `llm_ai_depth`,
`llm_status` and an LLM-written summary, strengths and concerns (the rule score stays
in `rule_ai_depth`). Keep `GITHUB_TOKEN` in `.env` too so GitHub scores stay complete.

Free-tier safety:

- **Pacing and 429s:** requests are spaced (`LLM_MIN_INTERVAL_S`), and a 429 is retried
  after the server's own retry hint.
- **Overload (HTTP 503 "high demand") and timeouts:** retried after longer pauses
  (10 s, 20 s, 40 s, 60 s; `LLM_BUSY_BACKOFF_S`). With Gemini the request also sets
  `reasoning_effort: low`, and replies get up to 120 s (`LLM_TIMEOUT_S`).
- **Configuration errors stop at once:** HTTP 401/403/404 (bad key, no access, model
  unavailable to the account) are never retried; the run stops calling the LLM with a
  "check .env" message. A 400 is not retried for that resume.
- **Quota and outage breakers:** if the quota is exhausted, or the provider fails for 3
  resumes in a row (`LLM_MAX_CONSECUTIVE_FAILURES`), the remaining calls are skipped
  instead of hammering the API.
- **Fairness:** if only part of the batch got an LLM score, everyone is ranked on the
  rule score (`LLM_PARTIAL_POLICY=neutralize`) and the batch summary shows a warning.
- **Token budget protection:** every call caps generated tokens (`LLM_MAX_OUTPUT_TOKENS`,
  which includes a reasoning model's thinking) and trims long resumes
  (`LLM_MAX_RESUME_CHARS`; 8,000 characters on Groq). Harmless deviations in the reply
  (five strengths instead of four, a score of 41) are repaired rather than paid for
  again, and an unusable reply is retried only once. A *daily*-limit 429 stops the run
  at once. The console prints the tokens billed per run, and
  `python scripts/check_llm_models.py <model> resumes/<file>.pdf` measures the cost of
  one real resume before a full run.
- **Cache:** successful answers are cached in `.cache/llm`, so re-running later only
  sends the missing calls.

## Project layout

```
main.py                     CLI
api.py                      FastAPI wrapper (POST /screen, GET /results, GET /health)
scripts/benchmark.py        sequential vs bounded-concurrent timings
scripts/check_llm_models.py which LLM models answer right now for your key
docs/                       value document (PDF) and its generator
src/resume_screener/
  config.py                 env settings, category caps, penalties, all point tables
  lexicon.py                skill + AI-signal patterns (tunable, no logic)
  ingest.py                 file discovery, PDF/DOCX/TXT extraction, link annotations
  parsing.py                sections, project/job entries, name/email/GitHub
  evidence.py               where each signal appears (applied vs listed)
  eligibility.py            hard filter (rules only)
  scoring.py                deterministic 100-point model + explanations
  github.py                 GitHub enrichment (cached, bounded, rate-limit aware)
  llm.py                    optional LLM adapter (structured output, retries, cache)
  pipeline.py               orchestration, dedupe, ranking, batch summary
  report.py                 JSON / HTML / Markdown / console output
tests/                      eligibility, scoring, parsing, GitHub, LLM adapter (mocked SDK),
                            API, real PDF/DOCX fixtures, pipeline resilience
```

## Design Decisions

### Core principle: applied evidence vs. listed claims

Every technology match is tagged with *where* it appears. A term inside an experience,
project or research entry is evidence of use. The same term in a skills list or
summary is a claim. Applied evidence earns full points, listed claims earn partial
points, and coursework/certificates earn nothing. This one distinction drives the
filter, the score, and the explanations, and it is why a well-written resume with
buzzwords cannot outrank one that shows real work.

### Parsing

- **Two PDF extractors, best one wins.** pdfplumber collapses words on some PDFs in
  this set ("BuiltanAI-poweredplatform"); pdfium scrambles reading order on some
  design-tool exports. Both run, and the output with more recognisable sections
  (and no collapsed words) is kept.
- **Hyperlink annotations are read.** Many resumes show only the word "GitHub"; the
  URL lives in the PDF link annotation. Annotations take priority over visible text.
- **Layout-agnostic sections.** Headings are matched against an alias table
  ("Technical Skills", "Key Projects", "Internship / Training", ...), and applied
  sections are split into project/job entries. When unsure, entries are merged
  rather than split, because a fragment would be wrongly penalised as "no detail".
- **Failures are isolated per file.** Corrupt, empty, unsupported and duplicate files
  (same bytes or same email) are recorded, never fatal.

### Filtering strategy (hard rules, no LLM)

A candidate is eligible only if both of these hold:

1. **Python evidence.** "Python" appears in skills/experience/projects/summary, or a
   Python-only framework (FastAPI, Django, PyTorch, LangGraph...) is used in a
   project or job. Python only under coursework/certifications does not count.
2. **Applied AI evidence.** At least one project/job/publication entry contains
   LLM, agentic or retrieval signals.
   - AI terms that appear only in skills/summary do not qualify.
   - "AI-assisted development", Copilot and similar *tooling* phrases are stripped first.
   - Generic words ("data retrieval", "chunked uploads") count as RAG only when the
     same entry also involves an LLM.
   - A bare "Agent" in a title counts only if the bullets back it up.

JavaScript/Java/React/Next.js never cause rejection by themselves. Classic ML/DL/CV
projects are accepted as AI (configurable with `ACCEPT_CLASSIC_ML_AS_AI=false`) but
capped low in ranking.

### Scoring strategy (100 points, every point traceable)

| Category | Cap | How |
|---|---|---|
| AI / Agentic / RAG depth | 40 | Each AI entry is classified as deep, moderate, thin wrapper, no detail, or classic ML. Points come from the union of depth categories across substantive entries: agentic 8, retrieval 7, evaluation 4, state/memory 3, data pipeline 3, backend integration 3, training 3. Breadth adds up to 6, AI in a real job adds 3. |
| Python & backend | 30 | Python, FastAPI, async, PostgreSQL, Redis and others, scored as (applied, listed) points. Non-Python backend points are scaled by how many entries actually use Python, so a Node.js backend can't max this category. |
| Cloud / deploy / full stack | 15 | GCP weighted highest, then Docker, K8s, AWS/Azure, CI/CD, and deployment evidence. React/Next.js count fully only alongside a Python backend (end-to-end). |
| GitHub | 10 | See below. |
| Engineering depth | 5 | 1 point each for applied testing, caching, queues, observability, concurrency, failure handling, architecture. |

**Penalties:**

- **Thin LLM wrapper** (an LLM call with no retrieval, agent, state, eval, data or
  backend logic): −5 each, capped at −15.
- **AI project with no implementation detail** (tutorial-style, guided labs): −3 each,
  capped at −6.
- **Classic-ML-only profiles:** AI depth is capped at 15/40, so strong Python without
  real AI work cannot reach the top.

Every candidate gets a project-level verdict list with an evidence line, applied
penalties, strengths, and concerns.

### LLM usage (optional, hybrid)

When an LLM is configured, each eligible resume is sent to the model with the Pydantic
`LLMAssessment` schema as the required output format. Anthropic uses a forced tool
call; the OpenAI-compatible adapter (Gemini free tier, Ollama, Groq, OpenRouter) uses
JSON-schema `response_format` and still accepts a tool call or fenced JSON in reply. The model
returns per-project verdicts with evidence, a 0–40 AI-depth score, a summary,
strengths and concerns.

- The final AI depth is `(1−b)·rule + b·LLM` (`LLM_BLEND`, default 0.5). The LLM
  cannot move a candidate more than half the category on its own, and the rule score
  is always shown alongside it.
- The LLM never decides eligibility. If it disagrees with the rules (no meaningful AI
  project), a "verify" concern is added for a human to check.
- Provider code is isolated in two small adapters behind one `LLMClient` protocol
  (`AnthropicClient`, `OpenAICompatibleClient`); provider presets live in config. The tool schema
  has all `$ref`s inlined so it is self-contained. Retries with backoff, bounded
  concurrency and an on-disk cache keyed by (model, prompt version, resume text) make
  runs reproducible. (The current Anthropic SDK no longer accepts `temperature`, so
  stability relies on the forced schema, the explicit rubric and the cache.)
- The adapter is tested through the real SDK against a mocked HTTP endpoint: request
  shape, schema-violating output (retried, then clean failure), a 529 overload followed
  by success, and cache hits.
- Any per-resume failure falls back to the rule score and is recorded in `llm_status`.

**The shipped `results.json` was generated in deterministic mode** (no API key in the
build environment).

### GitHub scoring

One request per user (`/users/{u}/repos?sort=pushed`), owned non-fork repos only, plus
an optional second request (`/users/{u}/events/public`) for recent public commits.
The events call is on by default only when `GITHUB_TOKEN` is set, because two calls per
user would exceed the anonymous 60/hour quota for this dataset.

- **Activity (0–5):** the higher of two tiers:
  - days since the last push: ≤30d → 5, ≤90d → 4, ≤180d → 2, ≤365d → 1;
  - public commits in the last 90 days (events): 20+ → 5, 8+ → 4, 3+ → 2, 1+ → 1.
- **Repos (0–5):** maintained repos (pushed within a year; 1/2/5+ → 1/2/3 points) plus
  Python/AI-relevant repos (Python language or AI terms in name/description/topics;
  1/3+ → 1/2 points).

Handling of failures and limits:

- Results are cached in-run (each user fetched once) and on disk (24h TTL).
- Concurrency is bounded.
- Once a rate limit is hit, the remaining lookups are skipped instead of retried.
- 404, timeout and rate limit are recorded per candidate. A missing GitHub never
  affects eligibility.
- Rejected candidates are not enriched, to save quota. `GITHUB_TOKEN` is read from
  the environment only.
- **Fairness under partial failure.** If a rate limit means only some candidates were
  enriched, those candidates would gain up to 10 points the others could not. By
  default (`GITHUB_PARTIAL_POLICY=neutralize`) GitHub points are withheld from everyone
  in that batch: fetched data stays visible, a concern is added, and the batch summary
  carries a warning. `keep` scores whoever was enriched.

### Concurrency and measured speed-up

Parsing runs in a thread pool (`PARSE_WORKERS`). GitHub and LLM calls run as asyncio
tasks behind semaphores (`GITHUB_CONCURRENCY`, `LLM_CONCURRENCY`).
`python scripts/benchmark.py` measures sequential vs bounded-concurrent runs; the
network stages use a mock API with fixed latency, so the numbers need no quota.
Results in the build sandbox:

| Stage | Sequential | Bounded concurrent | Speed-up |
|---|---|---|---|
| Parse 50 resumes (real PDFs) | 9.36s (1 worker) | 9.18s (4 workers) | 1.0x |
| GitHub: 43 users @ 0.3s latency | 12.96s (concurrency 1) | 2.72s (concurrency 5) | 4.8x |
| LLM: 50 calls @ 0.5s latency | 25.04s (concurrency 1) | 6.52s (concurrency 4) | 3.8x |

The I/O-bound stages, which dominate real runs, get the speed-up. Parsing did not
speed up because the sandbox has a single CPU and PDF text extraction is CPU-bound;
on a multi-core machine a process pool would be the next step.

## If I Had More Time

1. **Calibrate against human labels.** Have two reviewers score ~20 resumes and tune
   the point tables to maximise rank correlation. Track that as a regression test.
2. **Use the LLM for structured extraction too.** Extract project boundaries,
   ownership and technologies as structured JSON, with the rules running on that
   output. Heuristic entry splitting is the weakest link today.
3. **Richer GitHub signal.** Check that repos linked on the resume exist and match
   their claims (language, README, commits by the candidate), not just account
   activity.
4. **Reviewer workflow.** Extend the HTML report into a review screen with an
   override/feedback loop that records why a human changed a decision, and parse
   with a process pool on multi-core machines.

## Known limitations

- Section and entry detection is heuristic. Unusual layouts can merge two projects
  into one entry (scores stay reasonable, but per-project verdicts get coarser).
- Keyword evidence cannot verify *quality* or truthfulness of claims. That is what
  the optional LLM pass and human review are for.
- Scanned image-only PDFs are reported as unreadable (no OCR).
- This is a decision-support tool for a first pass. A human should review the
  shortlist and any rejection before it affects a candidate.
