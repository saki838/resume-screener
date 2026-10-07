import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resume_screener.config import Settings  # noqa: E402
from resume_screener.eligibility import check_eligibility  # noqa: E402
from resume_screener.evidence import analyze_entries, skill_locations  # noqa: E402
from resume_screener.parsing import parse_resume  # noqa: E402
from resume_screener.scoring import score_candidate  # noqa: E402


_ENV_PREFIXES = ("LLM_", "GITHUB_", "GEMINI_", "ANTHROPIC_", "CACHE_", "PARSE_", "ACCEPT_")


@pytest.fixture(autouse=True)
def _isolate_from_user_env(monkeypatch):
    """Tests must not see a developer's .env (real keys, models, limits)."""
    import os

    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path):
    return Settings(github_enabled=False, llm_provider="none", cache_enabled=False, cache_dir=tmp_path)


def analyze(text: str, settings: Settings, links=None):
    r = parse_resume("x.txt", "sha", text, links or [])
    locs, entries = skill_locations(r), analyze_entries(r)
    el = check_eligibility(r, locs, entries, settings)
    sc = score_candidate(r, locs, entries, el, settings) if el.eligible else None
    return r, el, sc


# --- synthetic resumes --------------------------------------------------------
AGENTIC = """Asha Rao
asha@example.com | github.com/asharao
SKILLS
Python, FastAPI, PostgreSQL, Redis, LangGraph, Docker, GCP
EXPERIENCE
AI Engineer Intern | Acme Labs | Jan 2025 – Present
• Built a LangGraph multi-agent workflow with tool calling and checkpointed state in Redis.
• Implemented a RAG pipeline: document ingestion, chunking, embeddings in pgvector and hybrid search.
• Exposed it via an async FastAPI service on GCP Cloud Run with Docker; added retries and rate limiting.
• Built an evaluation suite of 200 cases tracking answer accuracy and hallucination rate, run with pytest in CI.
PROJECTS
Support Copilot | Python, LangChain, FAISS | 2024
• Retrieval-augmented assistant over 5k support tickets with embeddings and re-ranking; caching layer cut latency 40%.
EDUCATION
B.Tech CSE, 2025
"""

WRAPPER = """Ravi Kumar
ravi@example.com
SKILLS
Python, Flask, OpenAI
PROJECTS
AI Poem Generator | Python, OpenAI API | 2024
• Sends the user's topic to the OpenAI GPT-4 API and displays the generated poem on a page for users to read.
Todo App | Python, Flask | 2023
• Simple CRUD todo list web app with login and a SQLite database, styled with Bootstrap for mobile use.
EDUCATION
B.Tech, 2025
"""

JS_ONLY = """Neha Shah
neha@example.com
SKILLS
JavaScript, TypeScript, React, Next.js, Node.js, MongoDB
EXPERIENCE
Frontend Developer | Webco | 2024 – Present
• Built 40+ React components; leveraged AI-assisted development tools and GitHub Copilot to speed up delivery.
PROJECTS
Shop | React, Node.js | 2023
• E-commerce store with cart, payments and an admin dashboard used by 300 merchants daily.
"""
