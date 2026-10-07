"""OpenAI-compatible adapter (Gemini free tier, Ollama, Groq, OpenRouter) against a mocked endpoint."""
import asyncio
import json

import httpx

from resume_screener.config import Settings
from resume_screener.llm import (CachedLLM, LLMError, OpenAICompatibleClient, extract_json, get_llm_client,
                                 parse_retry_after)

VALID = {
    "projects": [{"title": "Agentic RAG", "verdict": "deep", "evidence": "LangGraph supervisor"}],
    "has_meaningful_ai_project": True, "ai_depth_score": 31, "score_rationale": "ok",
    "project_summary": "Built an agentic RAG system.", "strengths": ["Agents"], "concerns": [],
}


def completion(content=None, tool_args=None):
    msg = {"role": "assistant", "content": content}
    if tool_args is not None:
        msg["tool_calls"] = [{"id": "c1", "type": "function",
                              "function": {"name": "record_assessment", "arguments": json.dumps(tool_args)}}]
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": msg}]}


def gemini_settings(tmp_path, **kw):
    base = dict(llm_provider="auto", anthropic_api_key=None, gemini_api_key="free-key", cache_dir=tmp_path,
                cache_enabled=False, llm_min_interval_s=0.0)
    base.update(kw)
    return Settings(**base)


def test_auto_picks_gemini_with_free_tier_presets(tmp_path):
    s = gemini_settings(tmp_path, llm_min_interval_s=None)
    assert s.resolved_llm_provider == "gemini" and s.llm_enabled
    assert s.llm_option("model") == "gemini-3.8-flash"
    assert s.llm_option("concurrency") == 1 and s.llm_option("min_interval_s") == 6.0
    assert get_llm_client(s).name == "gemini:gemini-3.8-flash"
    # explicit settings beat presets; anthropic key takes priority in auto mode
    assert gemini_settings(tmp_path, llm_model="gemini-3.1-flash-lite").llm_option("model") == "gemini-3.1-flash-lite"
    assert gemini_settings(tmp_path, anthropic_api_key="k").resolved_llm_provider == "anthropic"
    assert not Settings(llm_provider="auto", anthropic_api_key=None, gemini_api_key=None, llm_base_url=None,
                        cache_dir=tmp_path).llm_enabled


def test_ollama_needs_a_model_but_no_key(tmp_path):
    assert not Settings(llm_provider="ollama", cache_dir=tmp_path).llm_enabled
    s = Settings(llm_provider="ollama", llm_model="qwen3:8b", cache_dir=tmp_path)
    assert s.llm_enabled and s.llm_option("base_url") == "http://localhost:11434/v1"


def test_request_shape_and_plain_json_reply(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    client = OpenAICompatibleClient(gemini_settings(tmp_path), transport=httpx.MockTransport(handler))
    result = asyncio.run(client.assess("resume text"))
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert seen["auth"] == "Bearer free-key"
    body = seen["body"]
    assert body["model"] == "gemini-3.8-flash"
    assert body["response_format"]["type"] == "json_schema"
    schema_text = json.dumps(body["response_format"]["json_schema"]["schema"])
    assert "$ref" not in schema_text and '"default"' not in schema_text
    assert "resume text" in body["messages"][1]["content"]
    assert result.ai_depth_score == 31


def test_fenced_json_and_tool_call_replies(tmp_path):
    replies = [completion(content="Sure!\n```json\n" + json.dumps(VALID) + "\n```"),
               completion(content=None, tool_args=VALID)]

    def handler(request):
        return httpx.Response(200, json=replies.pop(0))

    client = OpenAICompatibleClient(gemini_settings(tmp_path), transport=httpx.MockTransport(handler))
    assert asyncio.run(client.assess("a")).ai_depth_score == 31
    assert asyncio.run(client.assess("b")).ai_depth_score == 31


def test_429_uses_retry_hint_then_succeeds(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "quota"}})
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    s = gemini_settings(tmp_path)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)
    assert asyncio.run(llm.assess("resume")).ai_depth_score == 31 and len(calls) == 2


def test_exhausted_quota_stops_further_calls(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "Quota exceeded. Please retry in 0s."}})

    s = gemini_settings(tmp_path, llm_max_retries=1)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        out = []
        for text in ("one", "two", "three"):
            try:
                await llm.assess(text)
            except LLMError as exc:
                out.append(str(exc))
        return out

    errors = asyncio.run(run())
    assert len(errors) == 3 and "skipped" in errors[1] and "skipped" in errors[2]
    assert len(calls) == 2  # first resume: 1 try + 1 retry; the others are not sent


def test_pacing_spaces_requests(tmp_path):
    import time

    def handler(request):
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    s = gemini_settings(tmp_path, llm_min_interval_s=0.2)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        t = time.perf_counter()
        await asyncio.gather(*(llm.assess(x) for x in ("a", "b", "c")))
        return time.perf_counter() - t

    assert asyncio.run(run()) >= 0.38  # 3 calls, 2 gaps of 0.2s


def test_helpers():
    assert extract_json('noise {"a": 1} trailing') == {"a": 1}
    assert parse_retry_after({}, '{"retryDelay": "37s"}') == 37.0
    assert parse_retry_after({"retry-after": "5"}, "") == 5.0
    try:
        extract_json("no json here")
        raise AssertionError
    except LLMError:
        pass


def test_gemini_sends_low_reasoning_effort_but_other_providers_do_not(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    gem = OpenAICompatibleClient(gemini_settings(tmp_path), transport=httpx.MockTransport(handler))
    oll = OpenAICompatibleClient(Settings(llm_provider="ollama", llm_model="m", cache_dir=tmp_path),
                                 transport=httpx.MockTransport(handler))
    asyncio.run(gem.assess("a"))
    asyncio.run(oll.assess("b"))
    assert bodies[0].get("reasoning_effort") == "low"
    assert "reasoning_effort" not in bodies[1]
    assert gemini_settings(tmp_path).llm_option("timeout_s") == 120.0


def test_503_overload_is_retried_with_backoff_then_succeeds(tmp_path, caplog):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(503, json=[{"error": {"code": 503, "message": "This model is currently "
                                                         "experiencing high demand.", "status": "UNAVAILABLE"}}])
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    s = gemini_settings(tmp_path, llm_busy_backoff_s=0.01)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)
    with caplog.at_level("WARNING"):
        assert asyncio.run(llm.assess("resume")).ai_depth_score == 31
    assert len(calls) == 3
    assert "LLMProviderBusy: HTTP 503" in caplog.text and "attempt 1/5" in caplog.text


def test_timeout_is_named_in_logs_and_status(tmp_path, caplog):
    def handler(request):
        raise httpx.ReadTimeout("")  # httpx timeouts often carry an empty message

    s = gemini_settings(tmp_path, llm_max_retries=0)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)
    with caplog.at_level("WARNING"):
        try:
            asyncio.run(llm.assess("resume"))
            raise AssertionError("expected LLMError")
        except LLMError as exc:
            assert "LLMProviderBusy" in str(exc) and "no reply within 120s" in str(exc)
    assert "failed: LLMProviderBusy: ReadTimeout" in caplog.text  # never a blank "failed:" line


def test_provider_down_stops_after_consecutive_failures(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "high demand"}})

    s = gemini_settings(tmp_path, llm_max_retries=1, llm_busy_backoff_s=0.0, llm_max_consecutive_failures=3)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        msgs = []
        for i in range(6):
            try:
                await llm.assess(f"resume {i}")
            except LLMError as exc:
                msgs.append(str(exc))
        return msgs

    msgs = asyncio.run(run())
    assert len(calls) == 6                      # 3 resumes x (1 try + 1 retry), then calls stop
    assert all("skipped" in m for m in msgs[3:]) and "in a row" in msgs[3]


def test_a_success_resets_the_failure_streak(tmp_path):
    plan = [503, 503, 200, 503, 503, 200]       # alternating: never 3 failed resumes in a row

    def handler(request):
        code = plan.pop(0)
        if code == 200:
            return httpx.Response(200, json=completion(content=json.dumps(VALID)))
        return httpx.Response(503, json={"error": {"message": "busy"}})

    s = gemini_settings(tmp_path, llm_max_retries=0, llm_busy_backoff_s=0.0, llm_max_consecutive_failures=3)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        ok = 0
        for i in range(6):
            try:
                await llm.assess(f"r{i}")
                ok += 1
            except LLMError:
                pass
        return ok

    assert asyncio.run(run()) == 2 and llm._quota_exhausted is None


def test_404_model_unavailable_stops_immediately_without_retries(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json=[{"error": {"code": 404, "message": "This model models/gemini-2.5-flash "
                                                     "is no longer available to new users."}}])

    s = gemini_settings(tmp_path, llm_busy_backoff_s=0.0)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        msgs = []
        for i in range(4):
            try:
                await llm.assess(f"resume {i}")
            except LLMError as exc:
                msgs.append(str(exc))
        return msgs

    msgs = asyncio.run(run())
    assert len(calls) == 1                                  # no retries, no further resumes sent
    assert "LLMConfigError" in msgs[0] and all("configuration error" in m for m in msgs[1:])


def test_400_is_not_retried_but_next_resume_still_runs(tmp_path):
    plan = [400, 200]

    def handler(request):
        code = plan.pop(0)
        if code == 400:
            return httpx.Response(400, json={"error": {"message": "invalid argument"}})
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    s = gemini_settings(tmp_path)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        try:
            await llm.assess("first")
            raise AssertionError("expected failure")
        except LLMError as exc:
            assert "LLMBadRequest" in str(exc)
        return await llm.assess("second")

    assert asyncio.run(run()).ai_depth_score == 31 and plan == []  # exactly 2 calls in total


def test_model_checker_reports_each_model_and_picks_first_working(tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "check_llm_models", Path(__file__).resolve().parents[1] / "scripts" / "check_llm_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def handler(request):
        model = json.loads(request.content)["model"]
        if model == "old-model":
            return httpx.Response(404, json={"error": {"message": "no longer available to new users"}})
        if model == "busy-model":
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    results = asyncio.run(mod.check(["old-model", "busy-model", "good-model"],
                                    transport=httpx.MockTransport(handler), pause_s=0))
    assert [(m, ok) for m, ok, _ in results] == [("old-model", False), ("busy-model", False), ("good-model", True)]
    out = capsys.readouterr().out
    assert "LLMConfigError" in out and "LLMProviderBusy" in out and "OK " in out


def groq_settings(tmp_path, **kw):
    base = dict(llm_provider="groq", groq_api_key="gsk-free", gemini_api_key="gem-key-should-not-be-sent",
                cache_dir=tmp_path, cache_enabled=False, llm_min_interval_s=0.0)
    base.update(kw)
    return Settings(**base)


def test_groq_preset_and_auto_order(tmp_path):
    s = groq_settings(tmp_path, llm_min_interval_s=None)
    assert s.llm_enabled and s.llm_option("model") == "openai/gpt-oss-120b"
    assert s.llm_option("min_interval_s") == 30.0 and s.llm_option("concurrency") == 1
    auto = Settings(llm_provider="auto", groq_api_key="g", cache_dir=tmp_path)
    assert auto.resolved_llm_provider == "groq"
    # with both keys present, auto prefers Gemini; LLM_PROVIDER=groq selects Groq explicitly
    both = Settings(llm_provider="auto", groq_api_key="g", gemini_api_key="k", cache_dir=tmp_path)
    assert both.resolved_llm_provider == "gemini"


def test_groq_request_uses_groq_key_url_model_and_low_reasoning(tmp_path):
    seen = {}

    def handler(request):
        seen.update(url=str(request.url), auth=request.headers.get("authorization"),
                    body=json.loads(request.content))
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    client = OpenAICompatibleClient(groq_settings(tmp_path), transport=httpx.MockTransport(handler))
    assert asyncio.run(client.assess("resume")).ai_depth_score == 31
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk-free"                       # never the Gemini key
    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["response_format"]["type"] == "json_schema"


def test_groq_schema_mismatch_400_is_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": "Generated JSON does not match the expected "
                                                                  "schema. Please adjust your prompt.",
                                                       "code": "json_validate_failed"}})
        return httpx.Response(200, json=completion(content=json.dumps(VALID)))

    s = groq_settings(tmp_path)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)
    assert asyncio.run(llm.assess("resume")).ai_depth_score == 31 and len(calls) == 2


def test_groq_style_retry_hints():
    assert parse_retry_after({}, "Rate limit reached. Please try again in 7.5s.") == 7.5
    assert parse_retry_after({}, "Please try again in 450ms") == 0.45
    assert parse_retry_after({}, "Please try again in 1m2.5s") == 62.5


def test_mismatched_model_name_is_warned(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        get_llm_client(groq_settings(tmp_path, llm_model="gemini-2.5-flash"))
    assert "does not look like a groq model" in caplog.text


# ---------------- token budget protection ----------------
from resume_screener.llm import LLMBadOutput, coerce_assessment  # noqa: E402


def test_coerce_repairs_harmless_deviations_instead_of_rejecting():
    raw = {"ai_depth_score": "45.6", "strengths": ["a", "b", "c", "d", "e"], "concerns": "single string",
           "projects": [{"title": "X", "verdict": "Thin Wrapper", "evidence": "e"},
                        {"title": "Y", "verdict": "excellent", "evidence": "e"}, "not a dict"]}
    fixed = coerce_assessment(raw)
    assert fixed["ai_depth_score"] == 40 and len(fixed["strengths"]) == 4 and fixed["concerns"] == ["single string"]
    assert [p["verdict"] for p in fixed["projects"]] == ["thin_wrapper"]       # unknown verdict dropped
    assert fixed["has_meaningful_ai_project"] is True                         # inferred from the score
    for bad in ({}, {"ai_depth_score": "lots"}, ["not", "an", "object"]):
        try:
            coerce_assessment(bad)
            raise AssertionError("expected LLMBadOutput")
        except LLMBadOutput:
            pass


def test_requests_cap_output_tokens_and_trim_long_resumes(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={**completion(content=json.dumps(VALID)),
                                         "usage": {"total_tokens": 3210}})

    s = groq_settings(tmp_path)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)
    asyncio.run(llm.assess("x" * 20000))
    assert bodies[0]["max_tokens"] == 2500
    assert len(bodies[0]["messages"][1]["content"]) < 8100          # 8,000 chars + the <resume> tags
    assert llm.tokens_used == 3210
    gem = OpenAICompatibleClient(gemini_settings(tmp_path), transport=httpx.MockTransport(handler))
    asyncio.run(gem.assess("short"))
    assert bodies[1]["max_tokens"] == 3000


def test_bad_output_is_retried_only_once(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=completion(content="I cannot produce JSON, sorry."))

    s = groq_settings(tmp_path)  # 4 retries allowed for network errors...
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)
    try:
        asyncio.run(llm.assess("resume"))
        raise AssertionError("expected LLMError")
    except LLMError as exc:
        assert "LLMBadOutput" in str(exc)
    assert len(calls) == 2  # ...but a bad answer only gets one retry


def test_daily_quota_429_stops_immediately(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "3600"}, json={"error": {"message":
               "Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per day (TPD): "
               "Limit 200000, Used 199870, Requested 4210. Please try again in 9m12s."}})

    s = groq_settings(tmp_path, llm_busy_backoff_s=0.0)
    llm = CachedLLM(OpenAICompatibleClient(s, transport=httpx.MockTransport(handler)), s)

    async def run():
        out = []
        for i in range(3):
            try:
                await llm.assess(f"r{i}")
            except LLMError as exc:
                out.append(str(exc))
        return out

    msgs = asyncio.run(run())
    assert len(calls) == 1                                   # no 65 s waits, no retries
    assert "LLMQuotaExhausted" in msgs[0] and all("daily LLM quota" in m for m in msgs[1:])


def test_console_reports_tokens(capsys):
    from resume_screener.models import BatchSummary, ScreeningReport
    from resume_screener.report import print_console

    rep = ScreeningReport(batch_summary=BatchSummary(llm_succeeded=3, llm_failed=1, llm_tokens_used=12000),
                          ranked_candidates=[], rejected_candidates=[], failed_files=[])
    print_console(rep)
    assert "tokens billed this run: 12,000 (~4,000 per successful resume)" in capsys.readouterr().out
