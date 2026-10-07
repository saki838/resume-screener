"""Exercise the real Anthropic SDK code path against a mocked HTTP endpoint."""
import asyncio
import json

try:  # newer anthropic SDKs ship their own fork of httpx; mocks must come from the same package
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from resume_screener.config import Settings
from resume_screener.llm import AnthropicClient, CachedLLM, LLMError, blend_ai_depth, tool_schema
from resume_screener.models import LLMAssessment

VALID = {
    "projects": [{"title": "Agentic RAG", "verdict": "deep", "evidence": "LangGraph supervisor + hybrid search"}],
    "has_meaningful_ai_project": True, "ai_depth_score": 34, "score_rationale": "Real agent system.",
    "project_summary": "Built a LangGraph multi-agent RAG system.", "strengths": ["Agents"], "concerns": [],
}


def _settings(tmp_path, **kw):
    return Settings(anthropic_api_key="test-key", llm_provider="anthropic", cache_dir=tmp_path,
                    llm_max_retries=1, **kw)


def _message(tool_input):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "record_assessment", "input": tool_input}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


def _client(tmp_path, handler, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicClient(_settings(tmp_path, **kw), http_client=http)


def test_tool_schema_is_self_contained():
    schema = tool_schema(LLMAssessment)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["properties"]["projects"]["items"]["properties"]["verdict"]["enum"]


def test_request_shape_and_structured_output(tmp_path):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=_message(VALID))

    result = asyncio.run(_client(tmp_path, handler).assess("resume text"))
    body = seen["body"]
    assert seen["key"] == "test-key"
    assert body["tool_choice"] == {"type": "tool", "name": "record_assessment"}
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert "resume text" in body["messages"][0]["content"]
    assert result.ai_depth_score == 34 and result.projects[0].verdict == "deep"


def test_invalid_output_is_retried_then_fails_cleanly(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=_message({**VALID, "ai_depth_score": 99}))  # violates le=40

    llm = CachedLLM(_client(tmp_path, handler), _settings(tmp_path, cache_enabled=False))
    try:
        asyncio.run(llm.assess("resume"))
        raise AssertionError("expected LLMError")
    except LLMError:
        pass
    assert len(calls) == 2  # 1 try + 1 retry


def test_server_error_then_success_and_cache_hit(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(529 if len(calls) == 1 else 200,
                              json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
                              if len(calls) == 1 else _message(VALID))

    llm = CachedLLM(_client(tmp_path, handler), _settings(tmp_path))
    first = asyncio.run(llm.assess("same resume"))
    second = asyncio.run(llm.assess("same resume"))  # served from disk cache
    assert first.ai_depth_score == second.ai_depth_score == 34
    assert len(calls) == 2


def test_blend(tmp_path):
    a = LLMAssessment.model_validate(VALID)
    assert blend_ai_depth(20, a, Settings(llm_blend=0.5, cache_dir=tmp_path)) == 27.0
    assert blend_ai_depth(20, a, Settings(llm_blend=0.0, cache_dir=tmp_path)) == 20.0


def _all_required_fields_defined(node):
    if isinstance(node, dict):
        props = node.get("properties", {})
        for req in node.get("required", []):
            assert req in props, f"required field {req!r} missing from properties"
        for v in node.values():
            _all_required_fields_defined(v)
    elif isinstance(node, list):
        for v in node:
            _all_required_fields_defined(v)


def test_schemas_keep_fields_named_title_and_define_every_required_field():
    from resume_screener.llm import portable_schema

    for schema in (tool_schema(LLMAssessment), portable_schema(LLMAssessment)):
        item = schema["properties"]["projects"]["items"]
        assert "title" in item["properties"]          # a field called "title" must survive
        assert "title" not in schema                   # the annotation keyword is stripped
        _all_required_fields_defined(schema)
