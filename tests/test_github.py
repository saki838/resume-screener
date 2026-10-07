import asyncio
from datetime import datetime, timezone

import httpx

from resume_screener.config import Settings
from resume_screener.github import GitHubClient, score_repos

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def repo(name, days_ago, language="Python", fork=False, description=""):
    ts = datetime.fromtimestamp(NOW.timestamp() - days_ago * 86400, tz=timezone.utc).isoformat()
    return {"name": name, "pushed_at": ts, "language": language, "fork": fork, "description": description}


def test_score_repos_activity_and_relevance():
    repos = [repo("rag-assistant", 3), repo("agent-lab", 40), repo("site", 100, "JavaScript"),
             repo("old", 900, "C"), repo("forked-llm", 1, fork=True)]
    gh = score_repos("u", repos, now=NOW)
    assert gh.last_push_days_ago == 3 and gh.activity_points == 5     # forks ignored
    assert gh.maintained_repos == 3 and gh.public_repos == 4
    assert set(gh.relevant_repos) == {"rag-assistant", "agent-lab"}
    assert gh.score <= 10


def test_inactive_profile_scores_low():
    gh = score_repos("u", [repo("hello-world", 800, "HTML")], now=NOW)
    assert gh.score == 0


def _client(tmp_path, handler):
    s = Settings(cache_enabled=False, cache_dir=tmp_path, github_token=None, llm_provider="none")
    return GitHubClient(s, transport=httpx.MockTransport(handler))


def test_rate_limit_is_recorded_and_later_lookups_skipped(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"})

    async def run():
        async with _client(tmp_path, handler) as c:
            return await c.enrich("alice"), await c.enrich("bob")

    a, b = asyncio.run(run())
    assert a.status == b.status == "rate_limited"
    assert "skipped" in b.detail and len(calls) == 1


def test_not_found_network_error_and_dedupe(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if "ghost" in request.url.path:
            return httpx.Response(404)
        if "flaky" in request.url.path:
            raise httpx.ConnectTimeout("timeout")
        return httpx.Response(200, json=[repo("x", 1)])

    async def run():
        async with _client(tmp_path, handler) as c:
            return await asyncio.gather(c.enrich("ghost"), c.enrich("flaky"), c.enrich("Dev"), c.enrich("dev"))

    ghost, flaky, dev1, dev2 = asyncio.run(run())
    assert ghost.status == "not_found" and flaky.status == "error"
    assert dev1.status == dev2.status == "ok"
    assert len([c for c in calls if "dev" in c.lower()]) == 1  # in-run cache, case-insensitive


def test_missing_username_is_not_a_failure(tmp_path):
    async def run():
        async with _client(tmp_path, lambda r: httpx.Response(500)) as c:
            return await c.enrich(None, "GitHub mentioned but no profile URL found")

    gh = asyncio.run(run())
    assert gh.status == "not_provided" and gh.score == 0


def test_public_events_raise_activity_when_repo_push_is_old():
    events = [{"type": "PushEvent", "created_at": "2026-09-20T00:00:00Z", "payload": {"size": 12}},
              {"type": "PushEvent", "created_at": "2026-02-01T00:00:00Z", "payload": {"size": 50}},  # >90d
              {"type": "WatchEvent", "created_at": "2026-09-25T00:00:00Z", "payload": {}}]
    without = score_repos("u", [repo("x", 200)], now=NOW)
    with_ev = score_repos("u", [repo("x", 200)], now=NOW, events=events)
    assert with_ev.recent_commits_90d == 12
    assert with_ev.activity_points == 4 > without.activity_points


def test_disk_cache_avoids_second_network_call(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=[repo("agent-lab", 5)])

    s = Settings(cache_enabled=True, cache_dir=tmp_path, github_token=None, llm_provider="none")

    async def run():
        async with GitHubClient(s, transport=httpx.MockTransport(handler)) as c:
            return await c.enrich("alice")

    first, second = asyncio.run(run()), asyncio.run(run())  # second run: new client, same disk cache
    assert first.status == second.status == "ok" and second.detail == "from cache"
    assert len(calls) == 1
