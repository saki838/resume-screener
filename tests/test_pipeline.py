import asyncio
import shutil

from conftest import AGENTIC, JS_ONLY, WRAPPER

from resume_screener.config import Settings
from resume_screener.llm import CachedLLM
from resume_screener.models import LLMAssessment, Status
from resume_screener.pipeline import run_screening


def _settings(tmp_path, **kw):
    return Settings(github_enabled=False, llm_provider="none", cache_enabled=False,
                    cache_dir=tmp_path / "cache", llm_max_retries=0, **kw)


def _folder(tmp_path):
    d = tmp_path / "resumes"
    d.mkdir()
    (d / "a_agentic.txt").write_text(AGENTIC, encoding="utf-8")
    (d / "b_wrapper.txt").write_text(WRAPPER, encoding="utf-8")
    (d / "c_js.txt").write_text(JS_ONLY, encoding="utf-8")
    shutil.copy(d / "a_agentic.txt", d / "d_copy.txt")          # duplicate content
    (d / "e_corrupt.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    (d / "f_empty.txt").write_text("   ", encoding="utf-8")
    (d / "g_photo.png").write_bytes(b"\x89PNG")
    return d


def test_batch_survives_bad_files_and_ranks(tmp_path):
    report = asyncio.run(run_screening(_folder(tmp_path), _settings(tmp_path), llm_client=None))
    s = report.batch_summary
    assert s.total_files == 7
    assert (s.eligible, s.rejected, s.duplicates, s.failed_or_unreadable) == (2, 1, 1, 3)
    assert [c.candidate_name for c in report.ranked_candidates] == ["Asha Rao", "Ravi Kumar"]
    assert report.ranked_candidates[0].rank == 1
    assert report.rejected_candidates[0].rejection_reasons
    statuses = {c.file: c.status for c in report.failed_files}
    assert statuses["d_copy.txt"] == Status.DUPLICATE
    assert statuses["e_corrupt.pdf"] == statuses["g_photo.png"] == Status.FAILED


class FakeLLM:
    name = "fake"

    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    async def assess(self, text):
        if self.fail_on and self.fail_on in text:
            raise TimeoutError("model timed out")
        return LLMAssessment(ai_depth_score=20, has_meaningful_ai_project=True, score_rationale="ok",
                             project_summary="LLM summary", strengths=["LLM strength"], concerns=[])


def test_llm_blend_and_per_resume_failure_fallback(tmp_path):
    settings = _settings(tmp_path, llm_blend=0.5, llm_partial_policy="keep")
    llm = CachedLLM(FakeLLM(fail_on="Ravi Kumar"), settings)
    report = asyncio.run(run_screening(_folder(tmp_path), settings, llm_client=llm))
    by_name = {c.candidate_name: c for c in report.ranked_candidates}

    ok = by_name["Asha Rao"]
    assert ok.llm_status == "ok" and ok.llm_ai_depth == 20
    assert ok.score_breakdown.ai_project_depth == round((ok.rule_ai_depth + 20) / 2, 1)
    assert ok.project_summary == "LLM summary"

    failed = by_name["Ravi Kumar"]
    assert failed.llm_status.startswith("failed")
    assert failed.score_breakdown.ai_project_depth == failed.rule_ai_depth
    assert report.batch_summary.llm_failed == 1 and report.batch_summary.llm_succeeded == 1


def test_disabled_github_is_not_counted_as_failure(tmp_path):
    report = asyncio.run(run_screening(_folder(tmp_path), _settings(tmp_path), llm_client=None))
    assert report.batch_summary.github_failed == 0
    assert all("GitHub not scored" not in c for r in report.ranked_candidates for c in r.concerns)


def test_partial_github_enrichment_is_neutralized(tmp_path):
    import httpx

    d = tmp_path / "resumes"
    d.mkdir()
    (d / "a.txt").write_text(AGENTIC, encoding="utf-8")                                   # github.com/asharao
    (d / "b.txt").write_text(WRAPPER + "\ngithub.com/ravik\n", encoding="utf-8")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if "asharao" in request.url.path:
            return httpx.Response(200, json=[{"name": "rag-agent", "pushed_at": "2026-10-01T00:00:00Z",
                                              "language": "Python", "fork": False}])
        return httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"})

    def run(policy):
        s = Settings(github_enabled=True, llm_provider="none", cache_enabled=False, cache_dir=tmp_path / "c",
                     github_token=None, github_concurrency=1, github_partial_policy=policy)
        return asyncio.run(run_screening(d, s, llm_client=None, github_transport=httpx.MockTransport(handler)))

    fair = run("neutralize")
    asha = next(c for c in fair.ranked_candidates if c.candidate_name == "Asha Rao")
    assert asha.github.status == "ok" and asha.score_breakdown.github == 0
    assert asha.total_score == asha.score_breakdown.total
    assert any("withheld" in c for c in asha.concerns) and fair.batch_summary.warnings

    kept = run("keep")
    asha_kept = next(c for c in kept.ranked_candidates if c.candidate_name == "Asha Rao")
    assert asha_kept.score_breakdown.github > 0 and kept.batch_summary.warnings


def test_spec_shaped_outputs(tmp_path):
    from resume_screener.report import to_spec_list

    report = asyncio.run(run_screening(_folder(tmp_path), _settings(tmp_path), llm_client=None))
    rows = to_spec_list(report)
    ranked_keys = {"rank", "candidate_name", "eligible", "total_score", "score_breakdown", "matched_skills",
                   "project_summary", "github_summary", "strengths", "concerns"}
    assert set(rows[0]) == ranked_keys and rows[0]["rank"] == 1
    assert set(rows[0]["score_breakdown"]) == {"ai_project_depth", "python_backend", "cloud_fullstack",
                                               "github", "engineering_depth"}
    rejected = [r for r in rows if not r["eligible"]]
    assert rejected and set(rejected[0]) == {"candidate", "eligible", "rejection_reasons", "matched_skills"}
    # the full report also carries the "candidate" key from the eligibility example
    assert report.model_dump()["rejected_candidates"][0]["candidate"] == "Neha Shah"


def test_partial_llm_results_are_neutralized_by_default(tmp_path):
    settings = _settings(tmp_path, llm_blend=0.5)
    llm = CachedLLM(FakeLLM(fail_on="Ravi Kumar"), settings)
    report = asyncio.run(run_screening(_folder(tmp_path), settings, llm_client=llm))
    asha = next(c for c in report.ranked_candidates if c.candidate_name == "Asha Rao")
    assert asha.llm_ai_depth == 20                                    # LLM output kept for review
    assert asha.score_breakdown.ai_project_depth == asha.rule_ai_depth  # but not used for ranking
    assert asha.total_score == asha.score_breakdown.total
    assert report.batch_summary.warnings and report.batch_summary.llm_succeeded == 1
