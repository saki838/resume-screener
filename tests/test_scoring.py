from conftest import AGENTIC, WRAPPER, analyze


def test_scores_respect_category_caps(settings):
    _, _, sc = analyze(AGENTIC, settings)
    b = sc.breakdown
    assert b.ai_project_depth <= 40 and b.python_backend <= 30
    assert b.cloud_fullstack <= 15 and b.engineering_depth <= 5
    assert 0 <= b.total <= 100


def test_deep_agentic_candidate_beats_wrapper_candidate(settings):
    _, _, deep = analyze(AGENTIC, settings)
    _, _, thin = analyze(WRAPPER, settings)
    assert deep.breakdown.ai_project_depth >= 30
    assert thin.breakdown.ai_project_depth <= 5
    assert deep.breakdown.total > thin.breakdown.total + 30


def test_thin_wrapper_penalty_is_applied_and_explained(settings):
    _, _, sc = analyze(WRAPPER, settings)
    assert any(p.verdict == "thin_wrapper" for p in sc.projects)
    assert sc.penalties and sc.penalties[0].startswith("-5")
    assert any("Thin LLM wrapper" in c for c in sc.concerns)


def test_applied_evidence_outscores_skills_list(settings):
    listed = """Ivy Chen
SKILLS
Python, FastAPI, PostgreSQL, Redis, Docker, GCP
PROJECTS
Notes Agent | Python | 2025
• Agent that answers questions over notes using an LLM with retrieval over embeddings and per-user memory.
"""
    used = listed.replace(
        "• Agent that",
        "• Async FastAPI service with PostgreSQL and Redis, Dockerized and deployed on GCP.\n• Agent that")
    _, _, a = analyze(listed, settings)
    _, _, b = analyze(used, settings)
    assert b.breakdown.python_backend > a.breakdown.python_backend
    assert b.breakdown.cloud_fullstack > a.breakdown.cloud_fullstack


def test_strong_python_with_only_classic_ml_stays_low_on_ai(settings):
    text = """Omar Ali
SKILLS
Python, FastAPI, PostgreSQL, Redis, asyncio, Docker, GCP
EXPERIENCE
Backend Engineer | Fintech Co | 2023 – Present
• Built async FastAPI services on PostgreSQL and Redis with Celery queues, pytest suites and Prometheus monitoring.
• Deployed with Docker on GCP; added retries, idempotency keys and circuit breakers for payment callbacks.
PROJECTS
Churn Model | Python, scikit-learn | 2022
• Trained an XGBoost classifier on 50k rows with feature engineering, reaching 0.86 AUC on a holdout set.
"""
    _, el, sc = analyze(text, settings)
    _, _, top = analyze(AGENTIC, settings)
    assert el.ai_kind == "classic_ml"
    assert sc.breakdown.ai_project_depth <= 15
    assert "classic ML only" in sc.concerns[0]
    assert sc.breakdown.total < top.breakdown.total


def test_framework_only_in_skills_is_flagged(settings):
    text = AGENTIC.replace("Python, FastAPI, PostgreSQL, Redis, LangGraph", "Python, CrewAI, LangGraph")
    _, _, sc = analyze(text, settings)
    assert any("CrewAI" in c for c in sc.concerns)
