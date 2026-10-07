"""End-to-end screening run.

parse (thread pool) -> dedupe -> eligibility (rules) -> deterministic score
-> [GitHub + optional LLM, concurrently, bounded] -> combine -> rank.

Each resume is isolated: any exception becomes a FAILED record for that file.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Settings, get_settings
from .eligibility import check_eligibility
from .evidence import EntryAnalysis, SkillLocations, analyze_entries, skill_locations
from .github import GitHubClient, summarize
from .ingest import SUPPORTED_EXTENSIONS, discover_files, extract_text, sha256_of
from .llm import CachedLLM, blend_ai_depth, get_llm_client
from .models import (BatchSummary, CandidateResult, EligibilityResult, GitHubResult, LLMAssessment,
                     ParsedResume, ScreeningReport, Status)
from .parsing import parse_resume
from .scoring import ScoreResult, matched_skills, score_candidate

log = logging.getLogger(__name__)


@dataclass
class _Parsed:
    path: Path
    resume: ParsedResume | None = None
    error: str | None = None


def _parse_one(path: Path, settings: Settings) -> _Parsed:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return _Parsed(path, error=f"unsupported file type '{path.suffix or 'none'}'")
    try:
        sha = sha256_of(path)
        text, links = extract_text(path)
        if len(text) < settings.min_text_chars:
            return _Parsed(path, error=f"no extractable text ({len(text)} chars) - scanned image or empty file?")
        return _Parsed(path, resume=parse_resume(str(path.name), sha, text, links))
    except Exception as exc:  # noqa: BLE001 - one malformed file must not stop the batch
        log.warning("Failed to parse %s: %s", path.name, exc)
        return _Parsed(path, error=f"{type(exc).__name__}: {str(exc)[:200]}")


def _failed(path: Path, error: str) -> CandidateResult:
    return CandidateResult(file=path.name, candidate_name=None, status=Status.FAILED, error=error)


def _rejected(r: ParsedResume, locs: SkillLocations, el: EligibilityResult) -> CandidateResult:
    gh = (GitHubResult(username=r.github_username, status="skipped", detail="not eligible; not enriched")
          if r.github_username else GitHubResult(status="not_provided", detail=r.github_note))
    return CandidateResult(
        file=r.file, candidate_name=r.name, email=r.email, status=Status.REJECTED, eligible=False,
        rejection_reasons=el.rejection_reasons, matched_skills=matched_skills(locs),
        eligibility_evidence={"python": el.python_evidence, "ai": el.ai_evidence},
        github=gh,
        github_summary=(f"GitHub profile '{r.github_username}' found; not enriched because the candidate "
                        "is not eligible" if r.github_username else summarize(gh)),
    )


def _combine(r: ParsedResume, el: EligibilityResult, sc: ScoreResult, gh: GitHubResult,
             llm: LLMAssessment | None, llm_status: str, settings: Settings) -> CandidateResult:
    breakdown = sc.breakdown.model_copy(update={"github": min(gh.score, settings.weights.github)})
    strengths, concerns, summary, llm_ai = list(sc.strengths), list(sc.concerns), sc.project_summary, None
    if llm is not None:
        llm_ai = float(llm.ai_depth_score)
        breakdown.ai_project_depth = blend_ai_depth(sc.rule_ai_depth, llm, settings)
        summary = llm.project_summary or summary
        strengths = list(dict.fromkeys(llm.strengths + strengths))[:6]
        concerns = list(dict.fromkeys(llm.concerns + concerns))[:6]
        if not llm.has_meaningful_ai_project:
            concerns.insert(0, "LLM review found no meaningful AI project despite passing rule filter - verify")
    if gh.status not in ("ok", "not_provided", "disabled"):
        concerns.append(f"GitHub not scored ({gh.status})")
    return CandidateResult(
        file=r.file, candidate_name=r.name, email=r.email, status=Status.ELIGIBLE, eligible=True,
        total_score=breakdown.total, score_breakdown=breakdown, rule_ai_depth=sc.rule_ai_depth,
        llm_ai_depth=llm_ai, penalties=sc.penalties, matched_skills=sc.matched_skills,
        project_summary=summary, projects=sc.projects,
        eligibility_evidence={"python": el.python_evidence, "ai": el.ai_evidence},
        github=gh, github_summary=summarize(gh), strengths=strengths, concerns=concerns,
        llm_status=llm_status,
    )


def _apply_github_partial_policy(ranked: list[CandidateResult], settings: Settings) -> list[str]:
    """Rate limits make enrichment partial: some candidates get up to 10 points that others
    could not. Unless configured otherwise, withhold GitHub points from everyone."""
    statuses = [c.github.status for c in ranked if c.github]
    limited, ok = statuses.count("rate_limited"), statuses.count("ok")
    if not limited:
        return []
    if ok == 0 or settings.github_partial_policy == "keep":
        return [f"GitHub enrichment incomplete: {limited} lookups rate-limited "
                f"({ok} succeeded); set GITHUB_TOKEN and re-run for comparable GitHub scores."]
    for c in ranked:
        if c.github and c.github.status == "ok" and c.score_breakdown.github:
            withheld = c.score_breakdown.github
            c.score_breakdown.github = 0
            c.total_score = c.score_breakdown.total
            c.concerns.append(f"GitHub points ({withheld}) withheld: enrichment was incomplete for this "
                              "batch, so GitHub scores would not be comparable")
    return [f"GitHub enrichment incomplete: {limited} lookups rate-limited, {ok} succeeded. GitHub points "
            "were withheld from all candidates to keep the ranking fair (GITHUB_PARTIAL_POLICY=neutralize). "
            "Set GITHUB_TOKEN and re-run for full GitHub scoring."]


def _apply_llm_partial_policy(ranked: list[CandidateResult], settings: Settings) -> list[str]:
    """If only some LLM calls succeeded (e.g. a free-tier quota ran out), blended and
    rule-only AI scores would be mixed in one ranking. By default rank everyone on the
    rule score; the LLM output stays in each record and the cache makes a re-run cheap."""
    ok = [c for c in ranked if c.llm_status == "ok"]
    failed = [c for c in ranked if c.llm_status.startswith("failed")]
    if not ok or not failed:
        return []
    if settings.llm_partial_policy == "keep":
        return [f"LLM assessment incomplete: {len(failed)} of {len(ok) + len(failed)} calls failed; "
                "those candidates use the rule score only."]
    for c in ok:
        c.score_breakdown.ai_project_depth = c.rule_ai_depth
        c.total_score = c.score_breakdown.total
        c.llm_status = "ok (not used for ranking: batch incomplete)"
        c.concerns.append("LLM score shown but not used for ranking: LLM assessment was incomplete for this batch")
    return [f"LLM assessment incomplete: {len(failed)} of {len(ok) + len(failed)} calls failed, so all candidates "
            "are ranked on the rule-based AI score (LLM_PARTIAL_POLICY=neutralize). Successful answers are "
            "cached: re-run later to complete the batch without repeating them."]


async def _llm_assess(llm: CachedLLM | None, text: str) -> tuple[LLMAssessment | None, str]:
    if llm is None:
        return None, "not_used"
    try:
        return await llm.assess(text), "ok"
    except Exception as exc:  # noqa: BLE001 - fall back to deterministic scoring for this resume
        return None, f"failed ({str(exc)[:120]}); deterministic score used"


async def run_screening(
    input_dir: Path,
    settings: Settings | None = None,
    llm_client: CachedLLM | None | bool = True,
    github_transport: httpx.AsyncBaseTransport | None = None,
) -> ScreeningReport:
    """llm_client: True = build from settings, None/False = disabled, or an injected client (tests)."""
    settings = settings or get_settings()
    started = time.perf_counter()
    llm = get_llm_client(settings) if llm_client is True else (llm_client or None)

    files = discover_files(input_dir)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max(1, settings.parse_workers)) as pool:
        parsed = await asyncio.gather(*(loop.run_in_executor(pool, _parse_one, f, settings) for f in files))

    failed: list[CandidateResult] = []
    rejected: list[CandidateResult] = []
    to_score: list[tuple[ParsedResume, SkillLocations, list[EntryAnalysis], EligibilityResult]] = []
    seen_hash: dict[str, str] = {}
    seen_email: dict[str, str] = {}
    duplicates = 0

    for item in parsed:  # sequential, in sorted file order -> deterministic duplicate handling
        if item.resume is None:
            failed.append(_failed(item.path, item.error or "unknown error"))
            continue
        r = item.resume
        original = seen_hash.get(r.sha256) or (seen_email.get(r.email) if r.email else None)
        if original:
            duplicates += 1
            failed.append(CandidateResult(file=r.file, candidate_name=r.name, email=r.email,
                                          status=Status.DUPLICATE, duplicate_of=original,
                                          error=f"duplicate of {original} (same file content or email)"))
            continue
        seen_hash[r.sha256] = r.file
        if r.email:
            seen_email[r.email] = r.file
        try:
            locs, entries = skill_locations(r), analyze_entries(r)
            el = check_eligibility(r, locs, entries, settings)
        except Exception as exc:  # noqa: BLE001
            failed.append(_failed(item.path, f"analysis error: {exc}"))
            continue
        if el.eligible:
            to_score.append((r, locs, entries, el))
        else:
            rejected.append(_rejected(r, locs, el))

    async with GitHubClient(settings, transport=github_transport) as gh_client:
        async def process(r: ParsedResume, locs, entries, el) -> CandidateResult:
            try:
                sc = score_candidate(r, locs, entries, el, settings)
                gh, (assessment, llm_status) = await asyncio.gather(
                    gh_client.enrich(r.github_username, r.github_note), _llm_assess(llm, r.text))
                return _combine(r, el, sc, gh, assessment, llm_status, settings)
            except Exception as exc:  # noqa: BLE001
                log.exception("Scoring failed for %s", r.file)
                return CandidateResult(file=r.file, candidate_name=r.name, status=Status.FAILED,
                                       error=f"scoring error: {exc}")

        scored = await asyncio.gather(*(process(*args) for args in to_score))

    ranked = [c for c in scored if c.status == Status.ELIGIBLE]
    failed += [c for c in scored if c.status == Status.FAILED]
    warnings = _apply_github_partial_policy(ranked, settings) + _apply_llm_partial_policy(ranked, settings)
    ranked.sort(key=lambda c: (-(c.total_score or 0), -c.score_breakdown.ai_project_depth,
                               -c.score_breakdown.python_backend, c.candidate_name or ""))
    for i, c in enumerate(ranked, 1):
        c.rank = i
    rejected.sort(key=lambda c: c.file)

    gh_statuses = [c.github.status for c in ranked if c.github]
    summary = BatchSummary(
        total_files=len(files),
        parsed_successfully=len([p for p in parsed if p.resume is not None]),
        eligible=len(ranked), rejected=len(rejected),
        failed_or_unreadable=len([p for p in parsed if p.resume is None]),
        duplicates=duplicates,
        github_enriched=gh_statuses.count("ok"),
        github_failed=len([s for s in gh_statuses if s not in ("ok", "not_provided", "disabled")]),
        llm_mode=f"hybrid ({llm.name}, blend={settings.llm_blend})" if llm else "deterministic (no LLM configured)",
        llm_succeeded=len([c for c in ranked if c.llm_status.startswith("ok")]),
        llm_failed=len([c for c in ranked if c.llm_status.startswith("failed")]),
        llm_tokens_used=llm.tokens_used if llm else None,
        elapsed_seconds=round(time.perf_counter() - started, 2),
        warnings=warnings,
    )
    return ScreeningReport(batch_summary=summary, ranked_candidates=ranked,
                           rejected_candidates=rejected, failed_files=failed)
