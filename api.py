"""Optional FastAPI wrapper around the same pipeline.

    uvicorn api:app --reload
    POST /screen   {"input_dir": "resumes"}   -> runs a batch, returns the report
    GET  /results                              -> last report (also persisted to output/results.json)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from resume_screener.config import get_settings  # noqa: E402
from resume_screener.models import ScreeningReport  # noqa: E402
from resume_screener.pipeline import run_screening  # noqa: E402
from resume_screener.report import write_json  # noqa: E402

app = FastAPI(title="Resume Screener", version="1.0")
OUTPUT = Path("output/results.json")
_state: dict[str, ScreeningReport] = {}
_lock = asyncio.Lock()  # one batch at a time keeps GitHub/LLM usage bounded


class ScreenRequest(BaseModel):
    input_dir: str = "resumes"
    use_llm: bool = True
    use_github: bool = True


@app.post("/screen", response_model=ScreeningReport)
async def screen(req: ScreenRequest) -> ScreeningReport:
    path = Path(req.input_dir)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"input_dir not found: {req.input_dir}")
    overrides = {} if req.use_llm else {"llm_provider": "none"}
    if not req.use_github:
        overrides["github_enabled"] = False
    async with _lock:
        report = await run_screening(path, get_settings(**overrides))
        _state["last"] = report
        write_json(report, OUTPUT)
    return report


@app.get("/results", response_model=ScreeningReport)
async def results() -> ScreeningReport:
    if "last" in _state:
        return _state["last"]
    if OUTPUT.exists():
        return ScreeningReport.model_validate_json(OUTPUT.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="No results yet; POST /screen first")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
