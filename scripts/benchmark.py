"""Measure the speed-up from bounded concurrency.

    python scripts/benchmark.py --input ./resumes

1. Parsing: the real resumes, thread pool with 1 vs N workers.
2. GitHub enrichment: one lookup per username found in the resumes, against a mock
   GitHub API with fixed simulated latency (so the benchmark needs no network or
   quota), concurrency 1 vs N.
3. LLM calls: the same idea with a fake model that takes a fixed time per call.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resume_screener.config import Settings  # noqa: E402
from resume_screener.github import GitHubClient  # noqa: E402
from resume_screener.ingest import discover_files  # noqa: E402
from resume_screener.llm import CachedLLM  # noqa: E402
from resume_screener.models import LLMAssessment  # noqa: E402
from resume_screener.pipeline import _parse_one  # noqa: E402


def bench_parse(files, settings, workers):
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda f: _parse_one(f, settings), files))
    return time.perf_counter() - t, results


async def bench_github(usernames, concurrency, latency, tmp):
    async def handler(request):
        await asyncio.sleep(latency)
        return httpx.Response(200, json=[{"name": "repo", "pushed_at": "2026-09-01T00:00:00Z",
                                          "language": "Python", "fork": False}])

    s = Settings(cache_enabled=False, cache_dir=tmp, github_token=None, github_concurrency=concurrency,
                 github_events_mode="false", llm_provider="none")
    t = time.perf_counter()
    async with GitHubClient(s, transport=httpx.MockTransport(handler)) as client:
        await asyncio.gather(*(client.enrich(u) for u in usernames))
    return time.perf_counter() - t


class SlowFakeLLM:
    name = "fake"

    def __init__(self, latency):
        self.latency = latency

    async def assess(self, text):
        await asyncio.sleep(self.latency)
        return LLMAssessment(ai_depth_score=20, has_meaningful_ai_project=True, score_rationale="-",
                             project_summary="-")


async def bench_llm(texts, concurrency, latency, tmp):
    s = Settings(cache_enabled=False, cache_dir=tmp, llm_concurrency=concurrency, llm_provider="none")
    llm = CachedLLM(SlowFakeLLM(latency), s)
    t = time.perf_counter()
    await asyncio.gather(*(llm.assess(x) for x in texts))
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("resumes"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--github-concurrency", type=int, default=5)
    ap.add_argument("--llm-concurrency", type=int, default=4)
    ap.add_argument("--github-latency", type=float, default=0.3)
    ap.add_argument("--llm-latency", type=float, default=0.5)
    args = ap.parse_args()
    tmp = Path(".cache/benchmark")
    settings = Settings(cache_enabled=False, cache_dir=tmp, llm_provider="none")
    files = discover_files(args.input)

    seq_parse, parsed = bench_parse(files, settings, 1)
    par_parse, _ = bench_parse(files, settings, args.workers)
    users = sorted({p.resume.github_username for p in parsed if p.resume and p.resume.github_username})
    texts = [p.resume.text for p in parsed if p.resume]
    seq_gh = asyncio.run(bench_github(users, 1, args.github_latency, tmp))
    par_gh = asyncio.run(bench_github(users, args.github_concurrency, args.github_latency, tmp))
    seq_llm = asyncio.run(bench_llm(texts, 1, args.llm_latency, tmp))
    par_llm = asyncio.run(bench_llm(texts, args.llm_concurrency, args.llm_latency, tmp))

    rows = [
        (f"Parse {len(files)} resumes (real PDFs)", f"1 worker", f"{args.workers} workers", seq_parse, par_parse),
        (f"GitHub: {len(users)} users @ {args.github_latency}s latency", "concurrency 1",
         f"concurrency {args.github_concurrency}", seq_gh, par_gh),
        (f"LLM: {len(texts)} calls @ {args.llm_latency}s latency", "concurrency 1",
         f"concurrency {args.llm_concurrency}", seq_llm, par_llm),
    ]
    print("| Stage | Sequential | Bounded concurrent | Speed-up |\n|---|---|---|---|")
    for name, a, b, t1, t2 in rows:
        print(f"| {name} | {t1:.2f}s ({a}) | {t2:.2f}s ({b}) | {t1 / t2:.1f}x |")


if __name__ == "__main__":
    main()
