"""Runtime configuration.

Everything tunable (weights, thresholds, model name, concurrency, credentials)
lives here so business logic never reads env vars or hard-codes numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:  # optional: load .env from the current folder, then the project root (first value wins)
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_opt_int(name: str) -> int | None:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def _env_opt_float(name: str) -> float | None:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Weights:
    """Category caps for the 100-point model (must sum to 100)."""

    ai_project_depth: int = 40
    python_backend: int = 30
    cloud_fullstack: int = 15
    github: int = 10
    engineering_depth: int = 5


@dataclass(frozen=True)
class Penalties:
    thin_wrapper_each: int = 5          # per LLM-wrapper "AI project"
    thin_wrapper_max: int = 15          # spec: 5-15 points
    no_detail_each: int = 3             # AI project listed without implementation detail
    no_detail_max: int = 6
    classic_ml_only_cap: int = 15       # AI depth cap when no LLM/agentic/RAG work exists


@dataclass(frozen=True)
class Settings:
    weights: Weights = field(default_factory=Weights)
    penalties: Penalties = field(default_factory=Penalties)

    # Eligibility policy
    accept_classic_ml_as_ai: bool = field(
        default_factory=lambda: _env_bool("ACCEPT_CLASSIC_ML_AS_AI", True)
    )
    min_text_chars: int = 200  # below this a resume is treated as unreadable (likely scanned)
    parse_workers: int = field(default_factory=lambda: _env_int("PARSE_WORKERS", 4))

    # GitHub enrichment
    github_enabled: bool = field(default_factory=lambda: _env_bool("GITHUB_ENABLED", True))
    github_token: str | None = field(default_factory=lambda: os.getenv("GITHUB_TOKEN") or None)
    github_timeout_s: float = field(default_factory=lambda: _env_float("GITHUB_TIMEOUT_S", 10.0))
    github_concurrency: int = field(default_factory=lambda: _env_int("GITHUB_CONCURRENCY", 5))
    # Events doubles API calls; by default only used when a token raises the quota.
    # If some lookups were rate-limited, GitHub points would only reach some candidates.
    # "neutralize" withholds GitHub points from everyone in that batch (data still shown);
    # "keep" scores whoever was enriched.
    github_partial_policy: str = field(default_factory=lambda: os.getenv("GITHUB_PARTIAL_POLICY", "neutralize"))
    github_events_mode: str = field(default_factory=lambda: os.getenv("GITHUB_USE_EVENTS", "auto"))

    # LLM (optional; deterministic scoring is always computed)
    # LLM provider: auto | anthropic | gemini | groq | ollama | openai_compatible | none
    # "auto" picks the first configured: ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, LLM_BASE_URL.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    llm_model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or None)
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or None)
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    # None = use the provider preset (free tiers need low concurrency and pacing)
    llm_concurrency: int | None = field(default_factory=lambda: _env_opt_int("LLM_CONCURRENCY"))
    llm_max_retries: int | None = field(default_factory=lambda: _env_opt_int("LLM_MAX_RETRIES"))
    llm_min_interval_s: float | None = field(default_factory=lambda: _env_opt_float("LLM_MIN_INTERVAL_S"))
    llm_timeout_s: float | None = field(default_factory=lambda: _env_opt_float("LLM_TIMEOUT_S"))
    # Base wait before retrying an overloaded provider (HTTP 5xx / timeout); doubles each try, max 60s.
    llm_busy_backoff_s: float | None = field(default_factory=lambda: _env_opt_float("LLM_BUSY_BACKOFF_S"))
    # Stop calling the LLM after this many resumes in a row failed every retry (provider down).
    llm_max_consecutive_failures: int | None = field(
        default_factory=lambda: _env_opt_int("LLM_MAX_CONSECUTIVE_FAILURES"))
    # final_ai_depth = (1 - blend) * rule_score + blend * llm_score
    llm_blend: float = field(default_factory=lambda: _env_float("LLM_BLEND", 0.5))
    # If only some LLM calls succeed, "neutralize" ranks everyone on the rule score
    # (LLM output stays visible); "keep" blends wherever the LLM succeeded.
    llm_partial_policy: str = field(default_factory=lambda: os.getenv("LLM_PARTIAL_POLICY", "neutralize"))
    llm_max_resume_chars: int | None = field(default_factory=lambda: _env_opt_int("LLM_MAX_RESUME_CHARS"))
    # Cap on generated tokens per call (includes a reasoning model's "thinking" tokens).
    llm_max_output_tokens: int | None = field(default_factory=lambda: _env_opt_int("LLM_MAX_OUTPUT_TOKENS"))

    # Disk cache for GitHub/LLM responses (speeds up re-runs; safe to delete)
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))
    github_cache_ttl_s: int = field(default_factory=lambda: _env_int("GITHUB_CACHE_TTL_S", 86400))

    @property
    def github_use_events(self) -> bool:
        mode = self.github_events_mode.strip().lower()
        if mode == "auto":
            return bool(self.github_token)
        return mode in {"1", "true", "yes", "on"}

    @property
    def resolved_llm_provider(self) -> str | None:
        p = (self.llm_provider or "auto").strip().lower()
        if p == "none":
            return None
        if p != "auto":
            return p
        if self.anthropic_api_key:
            return "anthropic"
        if self.gemini_api_key:
            return "gemini"
        if self.groq_api_key:
            return "groq"
        if self.llm_base_url:
            return "openai_compatible"
        return None

    def llm_option(self, name: str):
        """Explicit setting if given, else the provider preset, else the global default."""
        explicit = getattr(self, f"llm_{name}", None)
        if explicit is not None:
            return explicit
        preset = LLM_PRESETS.get(self.resolved_llm_provider or "", {})
        return preset.get(name, LLM_DEFAULTS[name])

    @property
    def llm_enabled(self) -> bool:
        p = self.resolved_llm_provider
        if p == "anthropic":
            return bool(self.anthropic_api_key)
        if p == "gemini":
            return bool(self.gemini_api_key or self.llm_api_key)
        if p == "groq":
            return bool(self.groq_api_key or self.llm_api_key)
        if p in ("ollama", "openai_compatible"):
            return bool(self.llm_option("base_url") and self.llm_option("model"))
        return False


LLM_DEFAULTS = {"model": None, "base_url": None, "concurrency": 4, "max_retries": 2, "min_interval_s": 0.0,
                "timeout_s": 60.0, "busy_backoff_s": 10.0, "max_consecutive_failures": 3, "extra_body": {},
                "max_resume_chars": 12000, "max_output_tokens": 3000}
LLM_PRESETS: dict[str, dict] = {
    "anthropic": {"model": "claude-sonnet-5-5"},
    # Free tier (Google AI Studio key). Free limits are roughly 10-15 requests/minute,
    # so calls are serialised and spaced; 429s are retried using the server's retry hint.
    "gemini": {"model": "gemini-3.8-flash",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "concurrency": 1, "max_retries": 4, "min_interval_s": 6.0, "timeout_s": 120.0,
               # Less thinking = faster replies and fewer timeouts; supported by Gemini's
               # OpenAI-compatible endpoint, and plenty for a rubric-based judgment.
               "extra_body": {"reasoning_effort": "low"}},
    # Free tier (key from console.groq.com/keys). gpt-oss-120b: 30 req/min but only 8K tokens/min
    # and 200K tokens/day, so ~1 resume every 30 s; 39 resumes fit in one day's budget.
    "groq": {"model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1",
             "concurrency": 1, "max_retries": 4, "min_interval_s": 30.0, "timeout_s": 90.0,
             "extra_body": {"reasoning_effort": "low"},
             # Tight daily token budget: shorter input (most resumes are < 8,000 characters).
             "max_resume_chars": 8000, "max_output_tokens": 2500},
    # Local and free, no key: `ollama pull <model>` and set LLM_MODEL to it.
    "ollama": {"base_url": "http://localhost:11434/v1", "concurrency": 1, "max_retries": 1},
    # Any other OpenAI-compatible endpoint (Groq, OpenRouter, vLLM...): set LLM_BASE_URL,
    # LLM_MODEL and LLM_API_KEY.
    "openai_compatible": {},
}


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)


# ---------------------------------------------------------------------------
# Signal point tables. (applied_points, listed_points): "applied" = used in an
# experience/project/research entry; "listed" = skills list / summary only.
# Category totals are capped by Weights, so these can over-sum safely.
# ---------------------------------------------------------------------------
AI_DEPTH_CATEGORY_POINTS: dict[str, float] = {
    "agentic": 8,         # agents, tool calling, LangGraph/ADK/CrewAI, MCP, multi-agent
    "retrieval": 7,       # RAG, embeddings, vector search, hybrid search
    "evaluation": 4,      # evals, benchmarks, accuracy metrics, guardrails
    "state_memory": 3,    # state graphs, memory, checkpoints, sessions
    "data_pipeline": 3,   # ingestion, parsing, chunking, structured extraction
    "backend": 3,         # AI wired into a real backend (FastAPI, DB, streaming)
    "training": 3,        # fine-tuning / training
}
AI_BREADTH_POINTS: dict[int, float] = {1: 3, 2: 5, 3: 6}  # substantive GenAI entries (3 = 3+)
AI_PROFESSIONAL_BONUS = 3          # substantive GenAI work inside a job/internship
AI_SKILLS_ONLY_FRAMEWORK_POINTS = 1  # per GenAI framework listed but never shown in use
AI_SKILLS_ONLY_FRAMEWORK_MAX = 2
AI_CLASSIC_ML_BONUS = 2            # classic ML on top of GenAI work
AI_WRAPPER_ONLY_BASE = 4           # some LLM integration exists, but every one is a thin wrapper

PYTHON_BACKEND_POINTS: dict[str, tuple[float, float]] = {
    "Python": (8, 4),
    "FastAPI": (5, 2),
    "Async": (4, 1.5),
    "PostgreSQL": (4, 1.5),
    "Redis": (4, 1.5),
    "Django|Flask": (3, 1),
    "Pydantic": (1.5, 0.5),
    "SQLAlchemy": (1, 0.5),
    "Celery": (1.5, 0.5),
    "REST APIs": (1.5, 0.5),
    "WebSockets/SSE": (1, 0.5),
    "MySQL|MongoDB": (1, 0.5),
}
PYTHON_IMPLIED_ONLY_POINTS = 3     # no literal "Python", but Python-only frameworks used
PYTHON_PROFESSIONAL_BONUS = 3      # Python used in a job/internship
# Backend signals (FastAPI aside) are language-agnostic in text, so a Node.js backend with
# Postgres/Redis would otherwise max this category. Scale non-Python points by how many
# project/job entries actually use Python (or a Python-only framework).
PYTHON_CENTRALITY_MULTIPLIER: dict[int, float] = {0: 0.6, 1: 0.8, 2: 1.0}  # 2 = 2+ entries

CLOUD_POINTS: dict[str, tuple[float, float]] = {
    "GCP": (4, 1.5),
    "Docker": (3, 1),
    "Kubernetes": (2, 1),
    "AWS|Azure": (2, 1),
    "CI/CD": (1.5, 0.5),
}
CLOUD_DEPLOYED_POINTS = 2          # "deployed", "production", "live demo" in projects/work
FULLSTACK_END_TO_END_POINTS = 3    # React/Next.js used together with a Python backend
FULLSTACK_FRONTEND_ONLY_POINTS = 1

ENGINEERING_POINTS_PER_SIGNAL = 1  # testing, caching, queues, observability, ... (cap 5)

# GitHub (cap 10): activity by days since last push to an owned, non-fork repo.
GITHUB_ACTIVITY_POINTS: list[tuple[int, float]] = [(30, 5), (90, 4), (180, 2), (365, 1)]
GITHUB_MAINTAINED_WINDOW_DAYS = 365
GITHUB_MAINTAINED_POINTS: list[tuple[int, float]] = [(5, 3), (2, 2), (1, 1)]  # (min repos, points)
GITHUB_RELEVANT_POINTS: list[tuple[int, float]] = [(3, 2), (1, 1)]
GITHUB_COMMIT_POINTS: list[tuple[int, float]] = [(20, 5), (8, 4), (3, 2), (1, 1)]  # public commits, 90d
