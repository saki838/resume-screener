"""Typed data structures shared across the pipeline (and the JSON output schema)."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Status(str, Enum):
    ELIGIBLE = "eligible"
    REJECTED = "rejected"
    FAILED = "failed"          # unreadable / malformed / unsupported
    DUPLICATE = "duplicate"


class SectionKind(str, Enum):
    HEADER = "header"          # text before the first recognised heading (name, contacts)
    SUMMARY = "summary"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    RESEARCH = "research"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"
    OTHER = "other"

    @property
    def is_applied(self) -> bool:
        """Sections where a technology counts as *used*, not merely listed."""
        return self in {SectionKind.EXPERIENCE, SectionKind.PROJECTS, SectionKind.RESEARCH}


class Entry(BaseModel):
    """One project / job / publication inside an applied section."""

    section: SectionKind
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"


class ParsedResume(BaseModel):
    file: str
    sha256: str
    text: str
    sections: dict[SectionKind, str] = Field(default_factory=dict)
    entries: list[Entry] = Field(default_factory=list)
    name: str | None = None
    email: str | None = None
    github_username: str | None = None
    github_note: str | None = None
    links: list[str] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)
    ai_kind: str = "none"   # "genai" | "classic_ml" | "none"


class ScoreBreakdown(BaseModel):
    ai_project_depth: float = 0
    python_backend: float = 0
    cloud_fullstack: float = 0
    github: float = 0
    engineering_depth: float = 0

    @property
    def total(self) -> float:
        return round(
            self.ai_project_depth + self.python_backend + self.cloud_fullstack
            + self.github + self.engineering_depth, 1
        )


class ProjectAssessment(BaseModel):
    title: str
    section: str
    signals: list[str]
    verdict: str            # "deep" | "moderate" | "thin_wrapper" | "no_detail" | "classic_ml"
    evidence: str


class GitHubResult(BaseModel):
    username: str | None = None
    status: str = "not_provided"   # ok | not_provided | not_found | rate_limited | error | disabled
    detail: str | None = None
    public_repos: int | None = None
    maintained_repos: int | None = None
    relevant_repos: list[str] = Field(default_factory=list)
    last_push_days_ago: int | None = None
    recent_commits_90d: int | None = None
    activity_points: float = 0
    repo_points: float = 0

    @property
    def score(self) -> float:
        return round(self.activity_points + self.repo_points, 1)


class LLMProject(BaseModel):
    title: str
    verdict: Literal["deep", "moderate", "thin_wrapper", "tutorial", "not_ai"]
    evidence: str = Field(description="Short quote/paraphrase from the resume supporting the verdict")


class LLMAssessment(BaseModel):
    """Structured output requested from the LLM (its JSON schema is sent as the tool schema)."""

    projects: list[LLMProject] = Field(default_factory=list, max_length=8)
    has_meaningful_ai_project: bool
    ai_depth_score: int = Field(ge=0, le=40, description="0-40 per the rubric, penalties already applied")
    score_rationale: str = Field(description="1-2 sentences explaining the score")
    project_summary: str = Field(description="One evidence-backed sentence on the strongest AI work")
    strengths: list[str] = Field(default_factory=list, max_length=4)
    concerns: list[str] = Field(default_factory=list, max_length=4)


class CandidateResult(BaseModel):
    rank: int | None = None
    file: str
    candidate_name: str | None
    email: str | None = None
    status: Status
    eligible: bool = False
    total_score: float | None = None
    score_breakdown: ScoreBreakdown | None = None
    rule_ai_depth: float | None = None       # deterministic AI depth before LLM blend
    llm_ai_depth: float | None = None
    penalties: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    project_summary: str | None = None
    projects: list[ProjectAssessment] = Field(default_factory=list)
    eligibility_evidence: dict[str, list[str]] = Field(default_factory=dict)
    github: GitHubResult | None = None
    github_summary: str | None = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    llm_status: str = "not_used"
    error: str | None = None
    duplicate_of: str | None = None

    @computed_field  # same value as candidate_name; key used in the assignment's eligibility example
    @property
    def candidate(self) -> str | None:
        return self.candidate_name


class BatchSummary(BaseModel):
    total_files: int = 0
    parsed_successfully: int = 0
    eligible: int = 0
    rejected: int = 0
    failed_or_unreadable: int = 0
    duplicates: int = 0
    github_enriched: int = 0
    github_failed: int = 0
    llm_mode: str = "deterministic"
    llm_succeeded: int = 0
    llm_failed: int = 0
    llm_tokens_used: int | None = None   # tokens billed by the provider in this run (None if unknown)
    elapsed_seconds: float = 0
    warnings: list[str] = Field(default_factory=list)


class ScreeningReport(BaseModel):
    batch_summary: BatchSummary
    ranked_candidates: list[CandidateResult]
    rejected_candidates: list[CandidateResult]
    failed_files: list[CandidateResult]
