"""Locate *where* each technology/AI signal appears.

The core idea of the screener: a term in a project or job bullet is evidence of
use; the same term in a skills list is only a claim. Everything downstream
(eligibility, scoring, explanations) works off this distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import lexicon as lx
from .models import Entry, ParsedResume, SectionKind

LISTED_SECTIONS = {SectionKind.SKILLS, SectionKind.SUMMARY, SectionKind.HEADER, SectionKind.OTHER}
WEAK_SECTIONS = {SectionKind.EDUCATION, SectionKind.CERTIFICATIONS}


@dataclass
class SkillLocations:
    applied: set[str] = field(default_factory=set)   # used in experience/projects/research
    listed: set[str] = field(default_factory=set)    # skills list / summary / header
    weak: set[str] = field(default_factory=set)      # only coursework / certificates

    def has_applied(self, skill: str) -> bool:
        return skill in self.applied

    def has_any(self, skill: str) -> bool:
        return skill in self.applied or skill in self.listed

    @property
    def all_matched(self) -> set[str]:
        return self.applied | self.listed


@dataclass
class EntryAnalysis:
    entry: Entry
    signals: set[str]

    @property
    def is_genai(self) -> bool:
        return bool(self.signals & set(lx.GENAI_CATEGORIES))

    @property
    def is_classic_ml(self) -> bool:
        return "classic_ml" in self.signals and not self.is_genai

    @property
    def is_ai(self) -> bool:
        return self.is_genai or self.is_classic_ml

    @property
    def depth_signals(self) -> set[str]:
        return self.signals & set(lx.DEPTH_CATEGORIES)

    @property
    def detail_chars(self) -> int:
        return len(self.entry.body.strip())


def clean_for_matching(text: str) -> str:
    """Drop phrases about AI *tooling the candidate used* (Copilot, AI-assisted dev)."""
    return lx.AI_TOOLING_NOISE.sub(" ", text)


def skill_locations(resume: ParsedResume) -> SkillLocations:
    locs = SkillLocations()
    for kind, body in resume.sections.items():
        found = lx.find_terms(clean_for_matching(body), lx.SKILLS)
        if kind.is_applied:
            locs.applied |= found
        elif kind in WEAK_SECTIONS:
            locs.weak |= found
        else:
            locs.listed |= found
    locs.weak -= locs.applied | locs.listed
    return locs


def entry_signals(text: str, body: str | None = None) -> set[str]:
    signals = lx.find_terms(clean_for_matching(text), lx.AI_SIGNALS)
    if "agentic_weak" in signals:
        signals.discard("agentic_weak")
        if body is None or lx.compiled(lx.AI_SIGNALS["agentic_weak"]).search(clean_for_matching(body)):
            signals.add("agentic")
    if "retrieval_weak" in signals:
        signals.discard("retrieval_weak")
        if signals & {"llm", "retrieval"}:
            signals.add("retrieval")
    return signals


def analyze_entries(resume: ParsedResume) -> list[EntryAnalysis]:
    return [EntryAnalysis(entry=e, signals=entry_signals(e.text, e.body)) for e in resume.entries]


def snippet(text: str, pattern: str, width: int = 150) -> str | None:
    """First line in `text` matching `pattern`, trimmed for the explanation output."""
    rx = lx.compiled(pattern)
    for line in text.split("\n"):
        if rx.search(line):
            line = line.strip(" •●-*\t")
            return line if len(line) <= width else line[: width - 1] + "…"
    return None


def applied_text(resume: ParsedResume) -> str:
    return "\n".join(body for kind, body in resume.sections.items() if kind.is_applied)
