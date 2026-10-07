"""Output writers: results.json (machine-readable), report.md and a console table."""
from __future__ import annotations

import json
from pathlib import Path

from .models import ScreeningReport


def write_json(report: ScreeningReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")


def to_spec_list(report: ScreeningReport) -> list[dict]:
    """Every screened candidate as a bare list, in the exact shapes of the assignment's
    examples: section 7 for ranked candidates, section 3 for rejected ones."""
    ranked = [{
        "rank": c.rank, "candidate_name": c.candidate_name, "eligible": True,
        "total_score": c.total_score, "score_breakdown": c.score_breakdown.model_dump(),
        "matched_skills": c.matched_skills, "project_summary": c.project_summary,
        "github_summary": c.github_summary, "strengths": c.strengths, "concerns": c.concerns,
    } for c in report.ranked_candidates]
    rejected = [{
        "candidate": c.candidate_name, "eligible": False,
        "rejection_reasons": c.rejection_reasons, "matched_skills": c.matched_skills,
    } for c in report.rejected_candidates]
    return ranked + rejected


def write_spec_list(report: ScreeningReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_spec_list(report), indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(report: ScreeningReport, path: Path, top: int = 15) -> None:
    s = report.batch_summary
    lines = [
        "# Resume Screening Report", "",
        f"**{s.total_files}** files · **{s.parsed_successfully}** parsed · **{s.eligible}** eligible · "
        f"**{s.rejected}** rejected · **{s.failed_or_unreadable}** unreadable · **{s.duplicates}** duplicates  ",
        f"Scoring mode: {s.llm_mode} · GitHub enriched: {s.github_enriched} (failed: {s.github_failed})", "",
        *[f"> **Warning:** {w}" for w in s.warnings], "",
        "## Ranked shortlist", "",
        "| # | Candidate | Total | AI (40) | Py/Backend (30) | Cloud (15) | GitHub (10) | Eng (5) | Strongest AI work |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report.ranked_candidates[:top]:
        b = c.score_breakdown
        summary = (c.project_summary or "").split(". ")[0].replace("|", "/")[:90]
        lines.append(f"| {c.rank} | {c.candidate_name} | **{c.total_score}** | {b.ai_project_depth} | "
                     f"{b.python_backend} | {b.cloud_fullstack} | {b.github} | {b.engineering_depth} | {summary} |")
    lines += ["", "## Rejected", ""]
    for c in report.rejected_candidates:
        lines.append(f"- **{c.candidate_name}** ({c.file}): " + "; ".join(c.rejection_reasons))
    if report.failed_files:
        lines += ["", "## Failed / duplicate files", ""]
        lines += [f"- {c.file}: {c.error}" for c in report.failed_files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console(report: ScreeningReport, top: int = 10) -> None:
    s = report.batch_summary
    print(f"\nProcessed {s.total_files} files in {s.elapsed_seconds}s | parsed {s.parsed_successfully} | "
          f"eligible {s.eligible} | rejected {s.rejected} | failed {s.failed_or_unreadable} | "
          f"duplicates {s.duplicates}")
    print(f"Mode: {s.llm_mode} | GitHub ok: {s.github_enriched}, failed: {s.github_failed}")
    if s.llm_tokens_used is not None:
        sent = s.llm_succeeded + s.llm_failed
        per = f" (~{s.llm_tokens_used // max(1, s.llm_succeeded):,} per successful resume)" if s.llm_succeeded else ""
        print(f"LLM: {s.llm_succeeded} ok, {s.llm_failed} failed of {sent}; tokens billed this run: "
              f"{s.llm_tokens_used:,}{per}")
    for w in s.warnings:
        print(f"WARNING: {w}")
    print()
    print(f"{'#':>3}  {'Candidate':<28}{'Total':>6}{'AI':>6}{'Py':>6}{'Cloud':>6}{'GH':>5}{'Eng':>5}")
    for c in report.ranked_candidates[:top]:
        b = c.score_breakdown
        print(f"{c.rank:>3}  {(c.candidate_name or '?')[:27]:<28}{c.total_score:>6}{b.ai_project_depth:>6}"
              f"{b.python_backend:>6}{b.cloud_fullstack:>6}{b.github:>5}{b.engineering_depth:>5}")
    print()


_CATS = [("ai_project_depth", "AI depth", 40, "#1F3A5F"), ("python_backend", "Python & backend", 30, "#2A9D8F"),
         ("cloud_fullstack", "Cloud / full stack", 15, "#E9C46A"), ("github", "GitHub", 10, "#F4A261"),
         ("engineering_depth", "Eng. depth", 5, "#8D99AE")]


def write_html(report: ScreeningReport, path: Path, top: int = 20) -> None:
    """Small self-contained HTML report: shortlist with stacked score bars, rejections, failures."""
    from html import escape as e

    s = report.batch_summary
    legend = "".join(f'<span><i style="background:{c}"></i>{e(lab)} ({cap})</span>' for _, lab, cap, c in _CATS)
    rows = []
    for c in report.ranked_candidates[:top]:
        b = c.score_breakdown.model_dump()
        bar = "".join(f'<div title="{e(lab)}: {b[k]}" style="width:{b[k]}%;background:{col}"></div>'
                      for k, lab, _, col in _CATS)
        details = (
            f"<p><b>Strongest AI work:</b> {e(c.project_summary or '-')}</p>"
            f"<p><b>Strengths:</b> {e('; '.join(c.strengths) or '-')}</p>"
            f"<p><b>Concerns:</b> {e('; '.join(c.concerns) or '-')}</p>"
            + (f"<p><b>Penalties:</b> {e('; '.join(c.penalties))}</p>" if c.penalties else "")
            + f"<p><b>GitHub:</b> {e(c.github_summary or '-')}</p>"
            f"<p><b>Skills (applied first):</b> {e(', '.join(c.matched_skills))}</p>"
        )
        rows.append(
            f"<tr><td>{c.rank}</td><td><details><summary>{e(c.candidate_name or c.file)}</summary>{details}"
            f"</details></td><td class=n>{c.total_score}</td><td><div class=bar>{bar}</div></td>"
            + "".join(f"<td class=n>{b[k]}</td>" for k, *_ in _CATS) + "</tr>")
    rejected = "".join(f"<li><b>{e(c.candidate_name or c.file)}</b> ({e(c.file)}): "
                       f"{e('; '.join(c.rejection_reasons))}</li>" for c in report.rejected_candidates)
    failed = "".join(f"<li>{e(c.file)}: {e(c.error or '')}</li>" for c in report.failed_files) or "<li>None</li>"
    head_cells = "".join(f"<th>{e(lab)}</th>" for _, lab, *_ in _CATS)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Resume Screening Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1b2430;max-width:1200px}}
h1{{color:#1F3A5F;margin-bottom:4px}} .muted{{color:#5B6573}}
.tiles{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}
.tile{{background:#F4F1EA;padding:10px 16px;border-radius:6px;min-width:110px}}
.tile b{{display:block;font-size:22px;color:#1F3A5F}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:6px 8px;border-bottom:1px solid #e3e7ec;
text-align:left;vertical-align:top}} th{{background:#1F3A5F;color:#fff;font-weight:600}} td.n{{text-align:right}}
.bar{{display:flex;width:220px;height:14px;background:#eef1f4;border-radius:3px;overflow:hidden}}
.legend span{{margin-right:14px;font-size:13px}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:4px}}
details p{{margin:4px 0;font-size:13px}} summary{{cursor:pointer;font-weight:600}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Resume Screening Report</h1>
<div class=muted>Scoring mode: {e(s.llm_mode)} &middot; GitHub enriched: {s.github_enriched}
(not scored: {s.github_failed}) &middot; run time {s.elapsed_seconds}s</div>
{''.join(f'<p style="background:#FFF4E5;border-left:4px solid #F4A261;padding:8px 12px">{e(w)}</p>' for w in s.warnings)}
<div class=tiles><div class=tile><b>{s.total_files}</b>files</div><div class=tile><b>{s.parsed_successfully}</b>parsed</div>
<div class=tile><b>{s.eligible}</b>eligible</div><div class=tile><b>{s.rejected}</b>rejected</div>
<div class=tile><b>{s.failed_or_unreadable}</b>unreadable</div><div class=tile><b>{s.duplicates}</b>duplicates</div></div>
<h2>Ranked shortlist</h2><p class=legend>{legend}</p><p class=muted>Click a name for evidence, strengths and concerns.</p>
<div class=wrap><table><tr><th>#</th><th>Candidate</th><th>Total</th><th>Breakdown</th>{head_cells}</tr>
{''.join(rows)}</table></div>
<h2>Rejected</h2><ul>{rejected or '<li>None</li>'}</ul>
<h2>Failed / duplicate files</h2><ul>{failed}</ul>
</body></html>"""
    path.write_text(html, encoding="utf-8")
