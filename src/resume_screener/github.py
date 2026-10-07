"""Lightweight public-GitHub enrichment (max 10 points, never a hard requirement).

GET /users/{u}/repos?sort=pushed (owned repos) gives recency of activity,
maintained repos and Python/AI-relevant repos. Optionally (default: only when a
GITHUB_TOKEN is set, to stay inside the 60 req/h anonymous quota) a second call to
/users/{u}/events/public adds recent public commits. Failures are recorded on the
candidate, never raised.

* In-run cache: each username is fetched once even if it appears in several resumes.
* Optional disk cache (TTL) so re-runs don't burn the 60 req/h anonymous quota.
* Once GitHub reports a rate limit, remaining lookups are skipped immediately
  instead of hammering the API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import config as cfg
from . import lexicon as lx
from .config import Settings
from .models import GitHubResult

log = logging.getLogger(__name__)
API = "https://api.github.com"
RELEVANT_RE = re.compile(
    "|".join([lx.AI_SIGNALS["llm"], lx.AI_SIGNALS["agentic"], lx.AI_SIGNALS["retrieval"],
              lx.AI_SIGNALS["agentic_weak"], lx.AI_SIGNALS["classic_ml"], r"(?<![a-z])(?:ai|ml)(?![a-z])"]),
    re.I,
)


def _tier(value: int, table: list[tuple[int, float]], higher_is_better: bool = True) -> float:
    for threshold, points in table:
        if (value >= threshold) if higher_is_better else (value <= threshold):
            return points
    return 0.0


def count_recent_commits(events: list[dict], now: datetime, days: int = 90) -> int:
    """Commits pushed in public PushEvents within `days`. Newer API payloads may omit
    commit lists, so each PushEvent counts at least once."""
    total = 0
    for ev in events:
        if ev.get("type") != "PushEvent" or not ev.get("created_at"):
            continue
        created = datetime.fromisoformat(ev["created_at"].replace("Z", "+00:00"))
        if (now - created).days <= days:
            payload = ev.get("payload") or {}
            total += max(1, int(payload.get("distinct_size") or payload.get("size") or len(payload.get("commits") or [])))
    return total


def score_repos(username: str, repos: list[dict], now: datetime | None = None,
                events: list[dict] | None = None) -> GitHubResult:
    """Pure function: repo list (+ optional public events) in GitHub API shape -> scored result."""
    now = now or datetime.now(timezone.utc)
    owned = [r for r in repos if not r.get("fork") and not r.get("archived")]

    def age_days(r: dict) -> int | None:
        ts = r.get("pushed_at") or r.get("updated_at")
        if not ts:
            return None
        return (now - datetime.fromisoformat(ts.replace("Z", "+00:00"))).days

    ages = [a for a in (age_days(r) for r in owned) if a is not None]
    last_push = min(ages) if ages else None
    maintained = [r for r in owned if (a := age_days(r)) is not None and a <= cfg.GITHUB_MAINTAINED_WINDOW_DAYS]
    relevant = [
        r["name"] for r in owned
        if r.get("language") == "Python"
        or RELEVANT_RE.search(" ".join([r.get("name", "").replace("-", " ").replace("_", " "),
                                        r.get("description") or "", " ".join(r.get("topics") or [])]))
    ]
    activity = _tier(last_push, cfg.GITHUB_ACTIVITY_POINTS, higher_is_better=False) if last_push is not None else 0
    commits = count_recent_commits(events, now) if events is not None else None
    if commits is not None:  # recency of the last push OR volume of recent commits, whichever is higher
        activity = max(activity, _tier(commits, cfg.GITHUB_COMMIT_POINTS))
    repo_pts = _tier(len(maintained), cfg.GITHUB_MAINTAINED_POINTS) + _tier(len(relevant), cfg.GITHUB_RELEVANT_POINTS)
    return GitHubResult(
        username=username, status="ok", public_repos=len(owned), maintained_repos=len(maintained),
        relevant_repos=relevant[:8], last_push_days_ago=last_push, recent_commits_90d=commits,
        activity_points=activity, repo_points=min(repo_pts, 5),
    )


def summarize(gh: GitHubResult | None) -> str:
    if gh is None or gh.status == "not_provided":
        return "No GitHub profile on resume" + (f" ({gh.detail})" if gh and gh.detail else "")
    if gh.status != "ok":
        return f"GitHub enrichment unavailable for '{gh.username}': {gh.status}" + (f" ({gh.detail})" if gh.detail else "")
    recency = "never pushed" if gh.last_push_days_ago is None else f"last push {gh.last_push_days_ago}d ago"
    if gh.recent_commits_90d is not None:
        n = gh.recent_commits_90d
        recency += f", {n} public commit{'s' if n != 1 else ''} in 90d"
    return (f"{gh.public_repos} owned public repo{'s' if gh.public_repos != 1 else ''}, "
            f"{gh.maintained_repos} updated in the last year, "
            f"{len(gh.relevant_repos)} Python/AI-relevant; {recency}.")


class GitHubClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "resume-screener"}
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self._client = httpx.AsyncClient(base_url=API, headers=headers,
                                         timeout=settings.github_timeout_s, transport=transport)
        self._sem = asyncio.Semaphore(settings.github_concurrency)
        self._memo: dict[str, asyncio.Task[GitHubResult]] = {}
        self._rate_limited_until: float | None = None
        self._cache_dir = settings.cache_dir / "github"

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def enrich(self, username: str | None, note: str | None = None) -> GitHubResult:
        if not username:
            return GitHubResult(status="not_provided", detail=note)
        if not self.settings.github_enabled:
            return GitHubResult(username=username, status="disabled")
        key = username.lower()
        if key not in self._memo:  # in-run dedupe, also for concurrent callers
            self._memo[key] = asyncio.ensure_future(self._fetch(username))
        result = await self._memo[key]
        return result.model_copy(update={"detail": result.detail or note})

    # -- internals -----------------------------------------------------------
    def _cache_path(self, username: str) -> Path:
        return self._cache_dir / f"{username.lower()}.json"

    def _read_cache(self, username: str) -> dict | None:
        if not self.settings.cache_enabled:
            return None
        path = self._cache_path(username)
        try:
            if time.time() - path.stat().st_mtime < self.settings.github_cache_ttl_s:
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return None

    def _write_cache(self, username: str, data: dict) -> None:
        if not self.settings.cache_enabled:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            keep = ("name", "fork", "archived", "language", "description", "topics", "pushed_at", "updated_at")
            data = {"repos": [{k: r.get(k) for k in keep} for r in data["repos"]], "events": data["events"]}
            self._cache_path(username).write_text(json.dumps(data), encoding="utf-8")
        except OSError as exc:
            log.debug("GitHub cache write failed: %s", exc)

    async def _fetch(self, username: str) -> GitHubResult:
        cached = self._read_cache(username)
        if isinstance(cached, dict) and isinstance(cached.get("repos"), list):
            return score_repos(username, cached["repos"], events=cached.get("events")).model_copy(
                update={"detail": "from cache"})
        async with self._sem:
            # Checked after acquiring the semaphore so queued lookups see a limit hit by earlier ones.
            if self._rate_limited_until and time.time() < self._rate_limited_until:
                return GitHubResult(username=username, status="rate_limited",
                                    detail="skipped: GitHub rate limit already hit in this run")
            try:
                resp = await self._client.get(f"/users/{username}/repos",
                                              params={"per_page": 100, "sort": "pushed", "type": "owner"})
            except httpx.HTTPError as exc:
                return GitHubResult(username=username, status="error", detail=f"{type(exc).__name__}: {exc}"[:200])

        if resp.status_code == 404:
            return GitHubResult(username=username, status="not_found", detail="profile does not exist or is private")
        if resp.status_code in (403, 429) and (resp.headers.get("x-ratelimit-remaining") == "0"
                                               or resp.status_code == 429):
            reset = resp.headers.get("x-ratelimit-reset")
            self._rate_limited_until = float(reset) if reset and reset.isdigit() else time.time() + 3600
            hint = "" if self.settings.github_token else "; set GITHUB_TOKEN for 5,000 req/h"
            return GitHubResult(username=username, status="rate_limited", detail=f"HTTP {resp.status_code}{hint}")
        if resp.status_code != 200:
            return GitHubResult(username=username, status="error", detail=f"HTTP {resp.status_code}")
        try:
            repos = resp.json()
            assert isinstance(repos, list)
        except (ValueError, AssertionError):
            return GitHubResult(username=username, status="error", detail="unexpected response body")
        events = await self._fetch_events(username) if self.settings.github_use_events else None
        self._write_cache(username, {"repos": repos, "events": events})
        return score_repos(username, repos, events=events)

    async def _fetch_events(self, username: str) -> list[dict] | None:
        """Second, optional call. Any failure just drops the commit signal."""
        try:
            async with self._sem:
                resp = await self._client.get(f"/users/{username}/events/public", params={"per_page": 100})
            if resp.status_code == 200 and isinstance(body := resp.json(), list):
                return [{"type": e.get("type"), "created_at": e.get("created_at"),
                         "payload": {k: (e.get("payload") or {}).get(k) for k in ("size", "distinct_size")}}
                        for e in body]
        except (httpx.HTTPError, ValueError):
            pass
        return None
