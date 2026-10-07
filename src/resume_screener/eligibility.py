"""Hard eligibility filter. Pure rules, no LLM: predictable and unit-testable.

Eligible  <=>  Python evidence  AND  applied AI evidence.

* Python evidence: "Python" in skills/experience/projects/summary, or a
  Python-only framework (FastAPI, Django, PyTorch, LangGraph, ...) used in a
  project/job. Python appearing only under coursework/certifications does not count.
* AI evidence: a project, job or publication entry containing LLM / agentic /
  retrieval signals (or, if enabled, classic ML/DL/CV). AI keywords that appear
  only in a skills list or summary are a claim, not evidence, and do not qualify.

Other stacks (JavaScript, Java, React, Next.js) never cause a rejection by themselves.
"""
from __future__ import annotations

from . import lexicon as lx
from .config import Settings
from .evidence import EntryAnalysis, SkillLocations, snippet
from .models import EligibilityResult, ParsedResume, SectionKind


def check_eligibility(
    resume: ParsedResume,
    locs: SkillLocations,
    entries: list[EntryAnalysis],
    settings: Settings,
) -> EligibilityResult:
    reasons: list[str] = []

    # ---- Python -------------------------------------------------------------
    python_evidence: list[str] = []
    for kind in (SectionKind.EXPERIENCE, SectionKind.PROJECTS, SectionKind.RESEARCH,
                 SectionKind.SKILLS, SectionKind.SUMMARY, SectionKind.HEADER, SectionKind.OTHER):
        body = resume.sections.get(kind)
        if body and (line := snippet(body, lx.SKILLS["Python"])):
            python_evidence.append(f"[{kind.value}] {line}")
    implied = sorted(s for s in lx.PYTHON_IMPLIED if locs.has_applied(s))
    if implied:
        python_evidence.append("Python frameworks used in projects/work: " + ", ".join(implied))

    if not python_evidence:
        if "Python" in locs.weak:
            reasons.append("Python appears only in coursework/certifications, not as a working skill")
        else:
            stack = sorted(locs.all_matched & {"JavaScript", "TypeScript", "React", "Next.js", "Node.js",
                                               "Java", "Spring Boot", "C++"})
            reasons.append("No evidence of Python stack" + (f" (profile is {', '.join(stack)})" if stack else ""))

    # ---- AI / agentic ------------------------------------------------------------
    genai = [e for e in entries if e.is_genai]
    classic = [e for e in entries if e.is_classic_ml]
    ai_evidence: list[str] = []
    ai_kind = "none"
    if genai:
        ai_kind = "genai"
        for e in genai[:3]:
            cats = sorted(e.signals & {"llm", "agentic", "retrieval"})
            ai_evidence.append(f"[{e.entry.section.value}] {e.entry.title[:90]} ({', '.join(cats)})")
    elif classic and settings.accept_classic_ml_as_ai:
        ai_kind = "classic_ml"
        ai_evidence = [f"[{e.entry.section.value}] {e.entry.title[:90]} (classic ML/DL)" for e in classic[:3]]
    else:
        listed_ai = sorted(
            locs.listed & {"LangChain", "LangGraph", "LlamaIndex", "Google ADK", "CrewAI", "AutoGen",
                           "MCP", "OpenAI API", "Gemini", "RAG", "Vector DB", "Embeddings",
                           "Hugging Face", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow"}
        )
        if classic and not settings.accept_classic_ml_as_ai:
            reasons.append("Only classic ML projects; no LLM/RAG/agentic project evidence")
        elif listed_ai:
            reasons.append("AI/LLM terms appear only in skills/summary (" + ", ".join(listed_ai)
                           + "); no AI project or work evidence")
        else:
            reasons.append("No AI/agentic project evidence")

    return EligibilityResult(
        eligible=not reasons,
        rejection_reasons=reasons,
        python_evidence=python_evidence[:4],
        ai_evidence=ai_evidence,
        ai_kind=ai_kind,
    )
