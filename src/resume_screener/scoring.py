"""Deterministic, explainable 100-point scoring (GitHub is added in the pipeline).

Every point awarded maps to a named signal, and the location of that signal
(project/job vs. skills list) decides how many points it is worth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config as cfg
from . import lexicon as lx
from .config import Settings
from .evidence import EntryAnalysis, SkillLocations, applied_text, snippet
from .models import EligibilityResult, ParsedResume, ProjectAssessment, ScoreBreakdown, SectionKind

READABLE = {
    "agentic": "agents/tool-calling",
    "retrieval": "retrieval/RAG",
    "state_memory": "state/memory",
    "evaluation": "evaluation/metrics",
    "data_pipeline": "data processing",
    "backend": "backend integration",
    "training": "training/fine-tuning",
    "llm": "LLM calls",
    "classic_ml": "classic ML/DL",
}
SUBSTANTIVE = {"deep", "moderate"}
_DATE_SPAN = re.compile(r"[(\[]?\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*)?"
                        r"(?:19|20)\d{2}\s*(?:[-–—]\s*(?:(?:[a-z]+\.?\s*)?(?:19|20)\d{2}|present|current))?[)\]]?",
                        re.I)


def short_title(title: str, width: int = 70) -> str:
    """Display title: first header line, dates and link words removed."""
    t = title.split(" / ")[0]
    t = _DATE_SPAN.sub("", t)
    t = re.sub(r"\b(?:github|live|demo|link\s*\d*)\b(?=\s*(?:\||$))", "", t, flags=re.I)
    t = re.sub(r"\s*[|–—-]\s*(?=[|–—-]|$)", "", re.sub(r"\s{2,}", " ", t)).strip(" |–—-•")
    return t if len(t) <= width else t[: width - 1].rsplit(" ", 1)[0] + "…"
NO_DETAIL_CHARS = 120


@dataclass
class ScoreResult:
    breakdown: ScoreBreakdown
    rule_ai_depth: float
    projects: list[ProjectAssessment]
    penalties: list[str]
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    project_summary: str | None = None
    matched_skills: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AI / agentic project depth (40)
# ---------------------------------------------------------------------------
def classify_entry(a: EntryAnalysis) -> str:
    if a.is_genai:
        if not a.depth_signals:
            return "thin_wrapper"     # an LLM call and nothing around it
        if a.detail_chars < NO_DETAIL_CHARS:
            return "no_detail"
        return "deep" if len(a.depth_signals) >= 4 else "moderate"
    return "no_detail" if a.detail_chars < NO_DETAIL_CHARS else "classic_ml"


def _evidence_line(a: EntryAnalysis) -> str:
    for cat in ("agentic", "retrieval", "evaluation", "llm", "classic_ml"):
        if cat in a.signals and (line := snippet(a.entry.body, lx.AI_SIGNALS[cat])):
            return line
    return (a.entry.body.split("\n")[0] or a.entry.title)[:150]


def _tiered(value: int, table: dict[int, float]) -> float:
    eligible = [k for k in table if value >= k]
    return table[max(eligible)] if eligible else 0.0


def score_ai_depth(entries: list[EntryAnalysis], locs: SkillLocations, settings: Settings
                   ) -> tuple[float, list[ProjectAssessment], list[str]]:
    pen = settings.penalties
    projects: list[ProjectAssessment] = []
    verdicts: list[tuple[EntryAnalysis, str]] = []
    for a in entries:
        if not a.is_ai:
            continue
        verdict = classify_entry(a)
        verdicts.append((a, verdict))
        projects.append(ProjectAssessment(
            title=a.entry.title[:120], section=a.entry.section.value,
            signals=sorted(READABLE.get(s, s) for s in a.signals if s in READABLE),
            verdict=verdict, evidence=_evidence_line(a),
        ))

    substantive = [a for a, v in verdicts if v in SUBSTANTIVE and a.is_genai]
    wrappers = [a for a, v in verdicts if v == "thin_wrapper"]
    no_detail = [a for a, v in verdicts if v == "no_detail"]
    classic = [a for a, v in verdicts if a.is_classic_ml and v != "no_detail"]
    penalties: list[str] = []

    if substantive:
        cats = set().union(*(a.depth_signals for a in substantive))
        score = sum(cfg.AI_DEPTH_CATEGORY_POINTS[c] for c in cats)
        score += _tiered(len(substantive), cfg.AI_BREADTH_POINTS)
        if any(a.entry.section == SectionKind.EXPERIENCE for a in substantive):
            score += cfg.AI_PROFESSIONAL_BONUS
        skills_only = [f for f in lx.GENAI_FRAMEWORKS if f in locs.listed and f not in locs.applied]
        score += min(len(skills_only) * cfg.AI_SKILLS_ONLY_FRAMEWORK_POINTS, cfg.AI_SKILLS_ONLY_FRAMEWORK_MAX)
        if classic:
            score += cfg.AI_CLASSIC_ML_BONUS
    elif classic:
        cats = set().union(*(a.depth_signals for a in classic))
        score = 6 + min(2 * (len(classic) - 1), 4) + 2 * len(cats & {"evaluation", "data_pipeline", "training"})
        score = min(score, pen.classic_ml_only_cap)
    elif wrappers:
        score = cfg.AI_WRAPPER_ONLY_BASE
    else:
        score = 0.0

    if wrappers:
        p = min(len(wrappers) * pen.thin_wrapper_each, pen.thin_wrapper_max)
        score -= p
        penalties.append(f"-{p}: thin LLM-wrapper project(s) with no retrieval/agent/state/eval/backend logic: "
                         + "; ".join(short_title(a.entry.title, 60) for a in wrappers))
    if no_detail:
        p = min(len(no_detail) * pen.no_detail_each, pen.no_detail_max)
        score -= p
        penalties.append(f"-{p}: AI project(s) listed without implementation detail: "
                         + "; ".join(short_title(a.entry.title, 60) for a in no_detail))

    return max(0.0, min(float(score), settings.weights.ai_project_depth)), projects, penalties


# ---------------------------------------------------------------------------
# Python & backend (30), cloud/full-stack (15), engineering depth (5)
# ---------------------------------------------------------------------------
def _points(table: dict[str, tuple[float, float]], locs: SkillLocations) -> float:
    total = 0.0
    for key, (applied, listed) in table.items():
        names = key.split("|")
        if any(locs.has_applied(n) for n in names):
            total += applied
        elif any(n in locs.listed for n in names):
            total += listed
    return total


def python_entry_count(resume: ParsedResume) -> int:
    patterns = [lx.SKILLS["Python"], *(lx.SKILLS[f] for f in lx.PYTHON_IMPLIED)]
    return sum(1 for e in resume.entries if any(lx.compiled(p).search(e.text) for p in patterns))


def score_python_backend(resume: ParsedResume, locs: SkillLocations, settings: Settings) -> float:
    python_only = {"Python": cfg.PYTHON_BACKEND_POINTS["Python"]}
    backend = {k: v for k, v in cfg.PYTHON_BACKEND_POINTS.items() if k != "Python"}
    centrality = cfg.PYTHON_CENTRALITY_MULTIPLIER[min(python_entry_count(resume), 2)]
    score = _points(python_only, locs) + centrality * _points(backend, locs)
    if "Python" not in locs.all_matched and any(locs.has_applied(f) for f in lx.PYTHON_IMPLIED):
        score += cfg.PYTHON_IMPLIED_ONLY_POINTS
    exp = resume.sections.get(SectionKind.EXPERIENCE, "")
    if exp and (lx.compiled(lx.SKILLS["Python"]).search(exp)
                or any(lx.compiled(lx.SKILLS[f]).search(exp) for f in ("FastAPI", "Django", "Flask"))):
        score += cfg.PYTHON_PROFESSIONAL_BONUS
    return min(score, settings.weights.python_backend)


def score_cloud_fullstack(resume: ParsedResume, locs: SkillLocations, settings: Settings) -> float:
    score = _points(cfg.CLOUD_POINTS, locs)
    if lx.compiled(lx.DEPLOYMENT_SIGNAL).search(applied_text(resume)):
        score += cfg.CLOUD_DEPLOYED_POINTS
    frontend_applied = locs.has_applied("React") or locs.has_applied("Next.js")
    python_backend_applied = any(locs.has_applied(s) for s in ("Python", "FastAPI", "Django", "Flask"))
    if frontend_applied and python_backend_applied:
        score += cfg.FULLSTACK_END_TO_END_POINTS
    elif frontend_applied:
        score += cfg.FULLSTACK_FRONTEND_ONLY_POINTS
    return min(score, settings.weights.cloud_fullstack)


def score_engineering_depth(resume: ParsedResume, settings: Settings) -> tuple[float, list[str]]:
    found = sorted(lx.find_terms(applied_text(resume), lx.ENGINEERING_SIGNALS))
    return min(len(found) * cfg.ENGINEERING_POINTS_PER_SIGNAL, settings.weights.engineering_depth), found


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def _summary(projects: list[ProjectAssessment], eligibility: EligibilityResult) -> str | None:
    if not projects:
        return None
    order = {"deep": 0, "moderate": 1, "classic_ml": 2, "thin_wrapper": 3, "no_detail": 4}
    ranked = sorted(projects, key=lambda p: (order.get(p.verdict, 9), -len(p.signals)))
    best = ranked[0]
    lead = short_title(best.title).split(" | ")[0]
    signals = [s for s in best.signals if s != READABLE["llm"]] or best.signals
    text = f"{lead} ({best.verdict.replace('_', ' ')}: {', '.join(signals)}). {best.evidence}"
    others = len([p for p in projects if p.verdict in SUBSTANTIVE]) - (1 if best.verdict in SUBSTANTIVE else 0)
    if others > 0:
        text += f" (+{others} more substantive AI project{'s' if others > 1 else ''})"
    if eligibility.ai_kind == "classic_ml":
        text = "Classic ML only: " + text
    return text


def _strengths_concerns(resume: ParsedResume, locs: SkillLocations, projects: list[ProjectAssessment],
                        eligibility: EligibilityResult, eng: list[str]) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    concerns: list[str] = []
    deep = [p for p in projects if p.verdict == "deep"]
    sub_signals = set().union(*(set(p.signals) for p in projects if p.verdict in SUBSTANTIVE)) if projects else set()
    if deep:
        strengths.append(f"Deep AI system: {short_title(deep[0].title)}")
    if READABLE["agentic"] in sub_signals:
        strengths.append("Agentic / tool-calling work shown in projects or jobs")
    if READABLE["retrieval"] in sub_signals:
        strengths.append("Retrieval / RAG implementation evidence")
    if READABLE["evaluation"] in sub_signals:
        strengths.append("Measures AI quality (evals, benchmarks or accuracy metrics)")
    exp = resume.sections.get(SectionKind.EXPERIENCE, "")
    if exp and lx.compiled(lx.SKILLS["Python"]).search(exp):
        strengths.append("Python used in professional work")
    if locs.has_applied("FastAPI"):
        strengths.append("FastAPI backend in projects/work")
    if locs.has_applied("GCP") or locs.has_applied("Docker"):
        strengths.append("Cloud/containerised deployment (" + ", ".join(
            s for s in ("GCP", "Docker", "Kubernetes") if locs.has_applied(s)) + ")")
    if len(eng) >= 4:
        strengths.append("Engineering depth: " + ", ".join(eng[:4]).replace("_", " "))

    if eligibility.ai_kind == "classic_ml":
        concerns.append("No LLM/RAG/agentic project — classic ML only")
    for p in projects:
        if p.verdict == "thin_wrapper":
            concerns.append(f"Thin LLM wrapper: {short_title(p.title, 60)}")
    skills_only = [f for f in lx.GENAI_FRAMEWORKS + ["RAG", "Vector DB"]
                   if f in locs.listed and f not in locs.applied]
    if skills_only:
        concerns.append("Listed in skills but not shown in any project: " + ", ".join(skills_only))
    if "Python" in locs.listed and "Python" not in locs.applied:
        concerns.append("Python appears only in skills/summary; no project names it explicitly")
    missing = [s for s in ("FastAPI", "Async", "PostgreSQL", "Redis") if not locs.has_any(s)]
    if missing:
        concerns.append("No evidence of " + ", ".join(missing))
    if not any(locs.has_any(s) for s in ("GCP", "AWS", "Azure", "Docker")):
        concerns.append("No cloud/Docker deployment evidence")
    return strengths[:5], concerns[:5]


def score_candidate(resume: ParsedResume, locs: SkillLocations, entries: list[EntryAnalysis],
                    eligibility: EligibilityResult, settings: Settings) -> ScoreResult:
    ai, projects, penalties = score_ai_depth(entries, locs, settings)
    py = score_python_backend(resume, locs, settings)
    cloud = score_cloud_fullstack(resume, locs, settings)
    eng_score, eng = score_engineering_depth(resume, settings)
    strengths, concerns = _strengths_concerns(resume, locs, projects, eligibility, eng)
    return ScoreResult(
        breakdown=ScoreBreakdown(ai_project_depth=round(ai, 1), python_backend=round(py, 1),
                                 cloud_fullstack=round(cloud, 1), engineering_depth=round(eng_score, 1)),
        rule_ai_depth=round(ai, 1), projects=projects, penalties=penalties,
        strengths=strengths, concerns=concerns,
        project_summary=_summary(projects, eligibility),
        matched_skills=matched_skills(locs),
    )


def matched_skills(locs: SkillLocations) -> list[str]:
    """Applied skills first (used in projects/work), then skills-list-only ones."""
    order = list(lx.SKILLS)
    applied = sorted(locs.applied, key=order.index)
    listed = sorted(locs.listed - locs.applied, key=order.index)
    return applied + listed
