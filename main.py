"""CLI entry point.

    python main.py --input ./resumes --output ./output/results.json
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resume_screener.config import get_settings  # noqa: E402
from resume_screener.pipeline import run_screening  # noqa: E402
from resume_screener.report import (print_console, write_html, write_json, write_markdown,  # noqa: E402
                                    write_spec_list)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # e.g. a cp1252 Windows console or redirected output
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="AI resume screening & ranking")
    parser.add_argument("--input", type=Path, default=Path("resumes"), help="folder of resumes (pdf/docx/txt)")
    parser.add_argument("--output", type=Path, default=Path("output/results.json"), help="results JSON path")
    parser.add_argument("--no-llm", action="store_true", help="deterministic scoring only")
    parser.add_argument("--no-github", action="store_true", help="skip GitHub enrichment")
    parser.add_argument("--top", type=int, default=10, help="rows to print in the console summary")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    overrides = {}
    if args.no_llm:
        overrides["llm_provider"] = "none"
    if args.no_github:
        overrides["github_enabled"] = False
    settings = get_settings(**overrides)

    try:
        report = asyncio.run(run_screening(args.input, settings))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_json(report, args.output)
    write_markdown(report, args.output.with_name("report.md"))
    write_html(report, args.output.with_name("report.html"))
    write_spec_list(report, args.output.with_name("candidates.json"))
    print_console(report, args.top)
    names = ", ".join(str(args.output.with_name(n)) for n in ("candidates.json", "report.md", "report.html"))
    print(f"Wrote {args.output}, {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
