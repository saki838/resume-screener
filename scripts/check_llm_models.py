"""Find which LLM model works for your key *right now* (one tiny request per model).

    python scripts/check_llm_models.py                      # Gemini candidates, best first
    python scripts/check_llm_models.py gemini-3.5-flash ... # or your own list
    python scripts/check_llm_models.py openai/gpt-oss-20b resumes/candidate_07.pdf
                                                            # measure tokens for one real resume

Uses the provider and key from .env. Prints OK / the error for each model, then the
line to put in .env. Free-tier models can be overloaded (HTTP 503) at busy times and
some older models are closed to new accounts (HTTP 404), so it is worth checking.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resume_screener.config import get_settings  # noqa: E402
from resume_screener.llm import OpenAICompatibleClient  # noqa: E402

# Preference order: strongest Flash models first, then the lighter (usually less busy) Lite models.
GEMINI_CANDIDATES = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest",
]
GROQ_CANDIDATES = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
PROBE = ("Python developer. Built a LangGraph multi-agent RAG assistant with FastAPI, PostgreSQL, "
         "pgvector embeddings and an evaluation suite of 50 test cases.")


async def check(models: list[str], transport=None, pause_s: float = 6.0, timeout_s: float = 60.0,
                probe: str = PROBE) -> list[tuple[str, bool, str]]:
    results = []
    for i, model in enumerate(models):
        if i:
            await asyncio.sleep(pause_s)  # stay under free-tier requests-per-minute limits
        settings = get_settings(llm_model=model, llm_timeout_s=timeout_s)
        started = time.perf_counter()
        try:
            client = OpenAICompatibleClient(settings, transport=transport)
            reply = await client.assess(probe[: int(settings.llm_option("max_resume_chars"))])
            tokens = f", {client.tokens_used:,} tokens" if client.tokens_used else ""
            results.append((model, True, f"OK in {time.perf_counter() - started:.1f}s "
                                         f"(score {reply.ai_depth_score}/40{tokens})"))
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            detail = " ".join(str(exc).split())[:110] or "(no message)"
            results.append((model, False, f"{type(exc).__name__}: {detail}"))
        model_, ok, msg = results[-1]
        print(f"  {'OK ' if ok else 'ERR'}  {model_:<28} {msg}", flush=True)
    return results


def main() -> int:
    settings = get_settings()
    provider = settings.resolved_llm_provider
    if provider not in ("gemini", "groq", "ollama", "openai_compatible"):
        print("No OpenAI-compatible LLM configured. Put GROQ_API_KEY or GEMINI_API_KEY "
              "(or LLM_BASE_URL + LLM_MODEL) in .env.")
        return 2
    args = sys.argv[1:]
    files = [a for a in args if Path(a).suffix.lower() in (".pdf", ".docx", ".txt") and Path(a).is_file()]
    defaults = {"gemini": GEMINI_CANDIDATES, "groq": GROQ_CANDIDATES}
    models = [a for a in args if a not in files] or defaults.get(provider) or [settings.llm_option("model")]
    probe = PROBE
    if files:  # measure with a real resume: tokens per resume = how much of the daily budget a run needs
        from resume_screener.ingest import extract_text
        probe = extract_text(Path(files[0]))[0]
        print(f"Using the text of {files[0]} as the test input.")
    print(f"Checking {len(models)} model(s) on provider '{provider}' (one request each)...\n")
    results = asyncio.run(check(models, probe=probe))
    working = [m for m, ok, _ in results if ok]
    print()
    if working:
        print("Put this line in .env (replace the existing LLM_MODEL= line):\n")
        print(f"    LLM_MODEL={working[0]}\n")
        return 0
    print("No model worked right now. If you saw HTTP 503 (high demand), wait 30-60 minutes and try again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
