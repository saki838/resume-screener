"""Optional LLM judgment of AI-project depth, behind a small provider adapter.

Role of the LLM (deliberately narrow):
  * It never decides eligibility (rules do).
  * It independently scores AI/agentic project depth (0-40) with per-project
    verdicts and evidence, which is blended with the rule score.
  * It writes the human-readable summary/strengths/concerns.

Swapping providers = implementing `LLMClient.assess`. Output is validated
against a Pydantic schema; on any failure the caller falls back to rules only.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import logging
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings
from .models import LLMAssessment

log = logging.getLogger(__name__)
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a senior engineer screening resumes for an SDE internship that needs strong \
Python and hands-on AI/agentic systems experience (LLM apps, RAG, tool-calling agents, multi-agent \
workflows, evaluation).

Judge ONLY the depth of the candidate's AI/agentic project work, using evidence written in the resume.
Score ai_depth_score from 0 to 40:
- 32-40: multiple real AI systems with clear engineering: agents with tools/state/orchestration, \
retrieval pipelines (chunking, embeddings, hybrid search), evaluation, production use or measured results.
- 22-31: at least one substantial AI system with several of those elements and concrete details.
- 12-21: AI projects with some workflow or retrieval but thin detail, or mostly classic ML.
- 1-11: only LLM API wrappers, tutorial-style or course projects, vague claims.
- 0: no AI project.
Rules:
- A project that only sends a prompt to an LLM API and shows the response is a "thin_wrapper"; deduct \
5-15 points for such projects being presented as AI work.
- Projects listed without implementation details or ownership evidence are "tutorial"; deduct points.
- A framework named only in a skills list earns nothing; reward evidence of how it was used.
- Do not reward writing quality, buzzwords or claimed metrics without supporting detail.
- Classic ML/CV projects count as AI but should not score above ~15 without LLM/agentic work.
Every verdict must cite short evidence from the resume. Do not invent facts.
Return your answer by calling the `record_assessment` tool."""

SYSTEM_PROMPT_JSON = SYSTEM_PROMPT.replace(
    "Return your answer by calling the `record_assessment` tool.",
    "Return ONLY a JSON object with keys: projects (list of {title, verdict, evidence}; verdict is one of "
    "deep, moderate, thin_wrapper, tutorial, not_ai), has_meaningful_ai_project (boolean), "
    "ai_depth_score (integer 0-40), score_rationale, project_summary, strengths (max 4 strings), "
    "concerns (max 4 strings). No markdown, no extra text.",
)


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    """HTTP 429. `retry_after` is the server's hint in seconds, when it gives one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMProviderBusy(LLMError):
    """HTTP 5xx (e.g. 503 "model is experiencing high demand") or a timeout: transient,
    worth retrying after a longer pause than an ordinary error."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaExhausted(LLMRateLimited):
    """429 on a *daily* limit (tokens/requests per day): nothing will succeed until it resets."""


class LLMBadOutput(LLMError):
    """The model answered, but not with a usable assessment (no JSON / wrong shape)."""


class LLMBadRequest(LLMError):
    """HTTP 400 for one request: retrying the same request won't help."""


class LLMConfigError(LLMError):
    """HTTP 401/403/404: wrong key, no access, or a model that doesn't exist / isn't
    available to this account. No request will succeed until the settings change."""


FATAL_STATUSES = (401, 403, 404)


def _status_of(exc: Exception) -> int | None:
    """HTTP status for our own errors and for SDK errors (e.g. anthropic.NotFoundError)."""
    if isinstance(exc, LLMConfigError):
        return 404
    if isinstance(exc, LLMBadRequest):
        return 400
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def parse_retry_after(headers, body: str) -> float | None:
    raw = headers.get("retry-after") if headers is not None else None
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    m = re.search(r'retry(?:Delay| in)["\s:]*"?(\d+(?:\.\d+)?)\s*s', body or "", re.I)
    if m:
        return float(m.group(1))
    # Groq style: "Please try again in 7.5s" / "in 450ms" / "in 1m2.5s"
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)", body or "", re.I)
    if m:
        seconds = float(m.group(2)) / (1000 if m.group(3).lower() == "ms" else 1)
        return seconds + 60 * int(m.group(1) or 0)
    return None


def tool_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with every $ref inlined (no $defs), so the tool schema is
    self-contained for any provider's validator."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            out = {}
            for k, v in node.items():
                if k == "properties":  # keys here are field names (a field may be called "title")
                    out[k] = {name: resolve(sub) for name, sub in v.items()}
                elif k != "title":     # the JSON-Schema "title" annotation keyword
                    out[k] = resolve(v)
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class LLMClient(Protocol):
    name: str

    async def assess(self, resume_text: str) -> LLMAssessment: ...


class AnthropicClient:
    """Anthropic Messages API with a forced tool call for structured output."""

    def __init__(self, settings: Settings, http_client=None):
        from anthropic import AsyncAnthropic  # imported lazily: optional dependency

        self.settings = settings
        self.model = settings.llm_option("model")
        self.name = f"anthropic:{self.model}"
        kwargs = {"http_client": http_client} if http_client is not None else {}
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_option("timeout_s"),
                                      max_retries=0, **kwargs)  # retries handled uniformly in CachedLLM
        self._tool = {
            "name": "record_assessment",
            "description": "Record the structured assessment of the candidate's AI project depth.",
            "input_schema": tool_schema(LLMAssessment),
        }

    async def assess(self, resume_text: str) -> LLMAssessment:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            # No temperature: current SDKs no longer accept it. Run-to-run stability comes
            # from the forced tool schema, the explicit rubric, and the on-disk cache.
            system=SYSTEM_PROMPT,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": "record_assessment"},
            messages=[{"role": "user", "content": f"<resume>\n{resume_text}\n</resume>"}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return LLMAssessment.model_validate(block.input)
        raise LLMError("model returned no tool_use block")


_SCHEMA_KEYS = {"type", "properties", "required", "items", "enum", "description",
                "minimum", "maximum", "minItems", "maxItems"}


def portable_schema(model: type[BaseModel]) -> dict:
    """Inlined schema reduced to keywords every OpenAI-compatible provider accepts
    (some reject `title`, `default` or `anyOf`)."""
    def clean(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "properties":
                    out[k] = {name: clean(sub) for name, sub in v.items()}
                elif k in _SCHEMA_KEYS:
                    out[k] = clean(v)
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(tool_schema(model))


_VERDICTS = {"deep", "moderate", "thin_wrapper", "tutorial", "not_ai"}


def coerce_assessment(data) -> dict:
    """Repair harmless deviations instead of rejecting a whole (paid-for) answer:
    over-long lists are trimmed, the score is clamped to 0-40, unknown verdicts are
    dropped. Only a missing score is fatal."""
    if not isinstance(data, dict):
        raise LLMBadOutput("reply JSON is not an object")
    if "ai_depth_score" not in data:
        raise LLMBadOutput("reply has no ai_depth_score")
    try:
        score = int(round(float(data["ai_depth_score"])))
    except (TypeError, ValueError) as exc:
        raise LLMBadOutput(f"ai_depth_score is not a number: {data['ai_depth_score']!r}") from exc
    score = max(0, min(40, score))

    def str_list(value, limit):
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in (value or []) if str(v).strip()][:limit] if isinstance(value, list) else []

    projects = []
    for item in data.get("projects") or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
        if verdict in _VERDICTS:
            projects.append({"title": str(item.get("title", ""))[:200], "verdict": verdict,
                             "evidence": str(item.get("evidence", ""))[:400]})
    meaningful = data.get("has_meaningful_ai_project")
    if isinstance(meaningful, str):
        meaningful = meaningful.strip().lower() in {"true", "yes", "1"}
    if not isinstance(meaningful, bool):
        meaningful = score >= 12
    return {
        "projects": projects[:8], "has_meaningful_ai_project": meaningful, "ai_depth_score": score,
        "score_rationale": str(data.get("score_rationale") or "")[:600],
        "project_summary": str(data.get("project_summary") or "")[:600],
        "strengths": str_list(data.get("strengths"), 4), "concerns": str_list(data.get("concerns"), 4),
    }


def extract_json(text: str) -> dict:
    """JSON object from a model reply, tolerating ```json fences or surrounding prose."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMBadOutput("model reply contained no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMBadOutput(f"model reply is not valid JSON: {exc}") from exc


class OpenAICompatibleClient:
    """Chat Completions API with JSON-schema structured output. Works with Gemini's
    OpenAI-compatible endpoint (free tier), Ollama (local, free), Groq, OpenRouter, vLLM."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.provider = settings.resolved_llm_provider or "openai_compatible"
        self.model = settings.llm_option("model")
        base_url = (settings.llm_option("base_url") or "").rstrip("/")
        if not base_url or not self.model:
            raise LLMError("LLM_BASE_URL and LLM_MODEL are required for this provider")
        provider_key = {"gemini": settings.gemini_api_key, "groq": settings.groq_api_key}.get(self.provider)
        api_key = provider_key or settings.llm_api_key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.name = f"{self.provider}:{self.model}"
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers,
                                         timeout=settings.llm_option("timeout_s"), transport=transport)
        self._extra_body = dict(settings.llm_option("extra_body") or {})
        self._max_tokens = settings.llm_option("max_output_tokens")
        self.tokens_used = 0  # provider-reported total over this run
        self._schema = portable_schema(LLMAssessment)

    async def assess(self, resume_text: str) -> LLMAssessment:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {"role": "user", "content": f"<resume>\n{resume_text}\n</resume>"},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "record_assessment", "schema": self._schema}},
            **self._extra_body,
        }
        if self._max_tokens:
            body["max_tokens"] = int(self._max_tokens)
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise LLMProviderBusy(f"{type(exc).__name__}: no reply within "
                                  f"{self.settings.llm_option('timeout_s'):.0f}s") from exc
        if resp.status_code >= 500:
            raise LLMProviderBusy(f"HTTP {resp.status_code}: {resp.text[:200]}",
                                  parse_retry_after(resp.headers, resp.text))
        if resp.status_code == 429:
            hint = parse_retry_after(resp.headers, resp.text)
            if re.search(r"per day|perday|\(TPD\)|\(RPD\)", resp.text, re.I):
                raise LLMQuotaExhausted(f"HTTP 429 daily limit: {resp.text[:200]}", hint)
            raise LLMRateLimited(f"HTTP 429: {resp.text[:200]}", hint)
        if resp.status_code in FATAL_STATUSES:
            raise LLMConfigError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 400:
            if re.search(r"does not match the expected schema|failed to generate json|json_validate_failed",
                         resp.text, re.I):
                # The model produced invalid JSON this time (Groq best-effort mode): worth retrying.
                raise LLMError(f"HTTP 400 (invalid JSON generated, retrying): {resp.text[:200]}")
            raise LLMBadRequest(f"HTTP 400: {resp.text[:300]}")
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBadOutput("unexpected response shape") from exc
        usage = payload.get("usage") or {}
        if isinstance(usage.get("total_tokens"), int):
            self.tokens_used += usage["total_tokens"]
        try:
            if message.get("tool_calls"):  # some providers answer via a function call instead
                data = json.loads(message["tool_calls"][0]["function"]["arguments"])
            else:
                content = message.get("content")
                if isinstance(content, list):  # content parts
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                data = extract_json(content)
            return LLMAssessment.model_validate(coerce_assessment(data))
        except LLMBadOutput:
            raise
        except (ValueError, KeyError, IndexError, TypeError) as exc:  # includes pydantic ValidationError
            raise LLMBadOutput(f"{type(exc).__name__}: {str(exc)[:150]}") from exc


class CachedLLM:
    """Retries with backoff (honouring 429 retry hints), bounded concurrency, request
    pacing for free tiers, a quota breaker, and an on-disk cache keyed by
    (model, prompt version, resume text) so re-runs are free and reproducible."""

    def __init__(self, inner: LLMClient, settings: Settings):
        self.inner = inner
        self.name = inner.name
        self.settings = settings
        self.max_retries = int(settings.llm_option("max_retries"))
        self.min_interval = float(settings.llm_option("min_interval_s"))
        self._sem = asyncio.Semaphore(max(1, int(settings.llm_option("concurrency"))))
        self._pace_lock = asyncio.Lock()
        self._next_slot = 0.0
        self.busy_backoff = float(settings.llm_option("busy_backoff_s"))
        self.max_consecutive_failures = int(settings.llm_option("max_consecutive_failures"))
        self._consecutive_failures = 0
        self._quota_exhausted: str | None = None
        self._dir = settings.cache_dir / "llm"

    @property
    def tokens_used(self) -> int | None:
        return getattr(self.inner, "tokens_used", None)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}|{PROMPT_VERSION}|{text}".encode()).hexdigest()[:32]

    async def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._pace_lock:
            loop = asyncio.get_running_loop()
            wait = self._next_slot - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_slot = loop.time() + self.min_interval

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        hint = getattr(exc, "retry_after", None)
        if hint is not None:
            return min(hint, 65.0)
        if isinstance(exc, LLMProviderBusy):  # overload: give the provider real time to recover
            return min(self.busy_backoff * 2 ** attempt, 60.0)
        return min(2 ** attempt, 8)

    async def assess(self, resume_text: str) -> LLMAssessment:
        text = resume_text[: int(self.settings.llm_option("max_resume_chars"))]
        path = self._dir / f"{self._key(text)}.json"
        if self.settings.cache_enabled and path.exists():
            try:
                return LLMAssessment.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError):
                pass
        if self._quota_exhausted:
            raise LLMError(f"skipped: {self._quota_exhausted}")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._sem:
                    await self._pace()
                    result = await self.inner.assess(text)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - provider errors, timeouts, schema violations
                last_exc = exc
                log.warning("LLM attempt %d/%d failed: %s: %s", attempt + 1, self.max_retries + 1,
                            type(exc).__name__, (str(exc) or "(no message)")[:200])
                if isinstance(exc, LLMQuotaExhausted):
                    self._quota_exhausted = "daily LLM quota used up (resets within 24 h); re-run then"
                    log.warning("Stopping LLM calls for this run: %s", self._quota_exhausted)
                    raise LLMError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
                if isinstance(exc, LLMBadOutput) and attempt >= 1:
                    break  # each retry costs a full request's tokens; one retry is enough
                status = _status_of(exc)
                if status in FATAL_STATUSES:
                    self._quota_exhausted = (f"LLM configuration error (HTTP {status}): check the API key and "
                                             "LLM_MODEL in .env, then re-run")
                    log.warning("Stopping LLM calls for this run: %s", self._quota_exhausted)
                    raise LLMError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
                if status == 400:
                    break  # same request would fail again; skip retries for this resume
                if attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay(exc, attempt))
        if last_exc is not None:
            self._consecutive_failures += 1
            if isinstance(last_exc, LLMRateLimited):
                # Still rate-limited after every retry: the quota is spent. Stop calling.
                self._quota_exhausted = "LLM rate limit/quota exhausted earlier in this run"
            elif self._consecutive_failures >= self.max_consecutive_failures:
                self._quota_exhausted = (f"LLM provider failed for {self._consecutive_failures} resumes in a row "
                                         "(likely overloaded or down); re-run later")
            if self._quota_exhausted:
                log.warning("Stopping LLM calls for this run: %s", self._quota_exhausted)
            raise LLMError(f"{type(last_exc).__name__}: {(str(last_exc) or '(no message)')[:200]}") from last_exc
        self._consecutive_failures = 0

        if self.settings.cache_enabled:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                path.write_text(result.model_dump_json(), encoding="utf-8")
            except OSError:
                pass
        return result


def get_llm_client(settings: Settings) -> CachedLLM | None:
    if not settings.llm_enabled:
        return None
    provider = settings.resolved_llm_provider
    model = str(settings.llm_option("model") or "")
    if (provider == "groq" and model.startswith("gemini")) or (provider == "gemini" and not model.startswith("gemini")):
        log.warning("LLM_MODEL=%s does not look like a %s model; clear LLM_MODEL in .env to use the "
                    "provider default", model, provider)
    try:
        if provider == "anthropic":
            return CachedLLM(AnthropicClient(settings), settings)
        if provider in ("gemini", "groq", "ollama", "openai_compatible"):
            return CachedLLM(OpenAICompatibleClient(settings), settings)
    except (ImportError, LLMError) as exc:
        log.warning("LLM disabled (%s); continuing with deterministic scoring", exc)
        return None
    log.warning("Unknown LLM_PROVIDER=%s; continuing with deterministic scoring", settings.llm_provider)
    return None


def blend_ai_depth(rule_score: float, llm: LLMAssessment, settings: Settings) -> float:
    b = min(max(settings.llm_blend, 0.0), 1.0)
    return round((1 - b) * rule_score + b * llm.ai_depth_score, 1)

