from conftest import AGENTIC, JS_ONLY, WRAPPER, analyze

from resume_screener.config import Settings


def test_python_plus_agentic_project_is_eligible(settings):
    _, el, _ = analyze(AGENTIC, settings)
    assert el.eligible and el.ai_kind == "genai"


def test_js_react_only_profile_rejected_and_copilot_is_not_ai_evidence(settings):
    _, el, _ = analyze(JS_ONLY, settings)
    assert not el.eligible
    assert any("No evidence of Python" in r for r in el.rejection_reasons)
    assert any("No AI/agentic" in r for r in el.rejection_reasons)


def test_javascript_does_not_cause_rejection_when_python_and_ai_present(settings):
    text = AGENTIC.replace("Python, FastAPI", "JavaScript, React, Next.js, Java, Python, FastAPI")
    _, el, _ = analyze(text, settings)
    assert el.eligible


def test_ai_terms_only_in_skills_list_are_not_enough(settings):
    text = """Sam Lee
SKILLS
Python, Django, LangChain, RAG, LangGraph
PROJECTS
Blog Platform | Python, Django | 2024
• Multi-author blog with comments, tags and PostgreSQL full-text search, deployed on a VPS with Nginx.
"""
    _, el, _ = analyze(text, settings)
    assert not el.eligible
    assert "only in skills" in el.rejection_reasons[0]


def test_python_only_in_coursework_does_not_count(settings):
    text = """Kim Park
SKILLS
Java, Spring Boot
PROJECTS
Chat Assistant | Java, Spring AI, OpenAI | 2024
• Java chatbot using an LLM with retrieval over product docs stored in a vector database with embeddings.
CERTIFICATIONS
Python for Everybody - Coursera
"""
    _, el, _ = analyze(text, settings)
    assert not el.eligible
    assert "coursework" in el.rejection_reasons[0]


def test_python_implied_by_framework_in_project(settings):
    text = """Lee Wong
SKILLS
LangGraph
PROJECTS
Research Agent | FastAPI, LangGraph | 2025
• Tool-calling agent built with LangGraph and served via FastAPI with an LLM planner and memory per session.
"""
    _, el, _ = analyze(text, settings)
    assert el.eligible
    assert any("FastAPI" in e for e in el.python_evidence)


def test_classic_ml_policy_is_configurable(tmp_path):
    text = """Ana Gomez
SKILLS
Python
PROJECTS
Crop Disease Detection | Python, TensorFlow | 2024
• Trained a CNN on 10k leaf images reaching 94% accuracy, with data augmentation and a Streamlit demo app.
"""
    accept = Settings(llm_provider="none", cache_enabled=False, cache_dir=tmp_path)
    strict = Settings(llm_provider="none", cache_enabled=False, cache_dir=tmp_path, accept_classic_ml_as_ai=False)
    assert analyze(text, accept)[1].ai_kind == "classic_ml"
    assert not analyze(text, strict)[1].eligible


def test_wrapper_project_still_passes_hard_filter(settings):
    # Hard filter is about presence of AI work; depth is the ranking model's job.
    _, el, _ = analyze(WRAPPER, settings)
    assert el.eligible
