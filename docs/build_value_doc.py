"""Builds docs/Value_Proposition.pdf from output/results.json (re-run after each screening).

    python docs/build_value_doc.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "results.json"
OUT = ROOT / "docs" / "Value_Proposition.pdf"
TMP = ROOT / "docs" / "_charts"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic",
                              boldItalic="Sans-Bold")

NAVY = colors.HexColor("#1F3A5F")
TEAL = colors.HexColor("#2A9D8F")
SAND = colors.HexColor("#F4F1EA")
GREY = colors.HexColor("#5B6573")
LINE = colors.HexColor("#D9DEE5")
CAT_COLORS = ["#1F3A5F", "#2A9D8F", "#E9C46A", "#F4A261", "#8D99AE"]

S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=26, leading=32, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=18, textColor=GREY),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=16, leading=21, textColor=NAVY,
                         spaceBefore=4, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=TEAL,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.6, leading=14, textColor=colors.black,
                           spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8.2, leading=11, textColor=GREY),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=11.5),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.6, leading=11.5, textColor=NAVY),
    "head": ParagraphStyle("head", fontName="Sans-Bold", fontSize=8.6, leading=11, textColor=colors.white),
    "stat": ParagraphStyle("stat", fontName="Sans-Bold", fontSize=20, leading=24, textColor=NAVY,
                           alignment=TA_CENTER),
    "statl": ParagraphStyle("statl", fontName="Sans", fontSize=8, leading=10, textColor=GREY,
                            alignment=TA_CENTER),
    "callout": ParagraphStyle("callout", fontName="Sans", fontSize=9.6, leading=14, textColor=NAVY),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def table(rows: list[list], widths: list[float], header: bool = True, zebra: bool = True) -> Table:
    data = [[c if not isinstance(c, str) else P(c, "head" if header and i == 0 else "cell")
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), SAND))
    t.setStyle(TableStyle(style))
    return t


def callout(text: str) -> Table:
    t = Table([[P(text, "callout")]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F4F2")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def stat_tiles(stats: list[tuple[str, str]]) -> Table:
    cells = [[P(v, "stat"), ] for v, _ in stats]
    row1 = [P(v, "stat") for v, _ in stats]
    row2 = [P(label, "statl") for _, label in stats]
    w = 170 * mm / len(stats)
    t = Table([row1, row2], colWidths=[w] * len(stats))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SAND),
                           ("LINEAFTER", (0, 0), (-2, -1), 1, colors.white),
                           ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, 1), (-1, 1), 10)]))
    del cells
    return t


def flow_diagram() -> Table:
    steps = [("Ingest", "PDF / DOCX / TXT,\ndual extractor,\nlink annotations"),
             ("Parse", "sections, projects,\nname, email,\nGitHub user"),
             ("Filter", "rules only:\nPython + applied\nAI evidence"),
             ("Score", "100-pt model,\npenalties,\nevidence lines"),
             ("Enrich", "GitHub activity,\noptional LLM\njudgment"),
             ("Rank", "JSON + report,\nreasons for every\ndecision")]
    top = [P(f"<b>{i}. {name}</b>", "cellb") for i, (name, _) in enumerate(steps, 1)]
    bottom = [P(desc.replace("\n", "<br/>"), "small") for _, desc in steps]
    t = Table([top, bottom], colWidths=[170 * mm / len(steps)] * len(steps))
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, -1), SAND),
             ("LINEABOVE", (0, 0), (-1, 0), 2.5, TEAL), ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------- charts
def chart_top10(ranked: list[dict]) -> Path:
    top = ranked[:10][::-1]
    names = [f"#{c['rank']} {c['candidate_name']}" for c in top]
    keys = ["ai_project_depth", "python_backend", "cloud_fullstack", "github", "engineering_depth"]
    labels = ["AI depth (40)", "Python & backend (30)", "Cloud / full stack (15)", "GitHub (10)",
              "Engineering depth (5)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=200)
    left = [0] * len(top)
    for k, lab, col in zip(keys, labels, CAT_COLORS):
        vals = [c["score_breakdown"][k] for c in top]
        ax.barh(names, vals, left=left, color=col, label=lab, height=0.66)
        left = [a + b for a, b in zip(left, vals)]
    for y, total in enumerate(left):
        ax.text(total + 1, y, f"{total:.1f}", va="center", fontsize=7.5, color="#1F3A5F")
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=3, frameon=False)
    ax.set_xlabel("Score (out of 100)", fontsize=8)
    fig.tight_layout()
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "top10.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_distribution(ranked: list[dict]) -> Path:
    bins = ["0-29", "30-49", "50-69", "70-89", "90-100"]
    edges = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 101)]
    counts = [sum(lo <= c["total_score"] < hi for c in ranked) for lo, hi in edges]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
    ax.bar(bins, counts, color="#2A9D8F")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=7.5)
    ax.set_title("Eligible candidates by score band", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = TMP / "dist.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_outcomes(summary: dict) -> Path:
    vals = [summary["eligible"], summary["rejected"], summary["failed_or_unreadable"] + summary["duplicates"]]
    labels = [f"Eligible ({vals[0]})", f"Rejected ({vals[1]})", f"Failed/duplicate ({vals[2]})"]
    pairs = [(v, lab, col) for v, lab, col in zip(vals, labels, ["#2A9D8F", "#F4A261", "#8D99AE"]) if v]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
    ax.pie([p[0] for p in pairs], labels=[p[1] for p in pairs], colors=[p[2] for p in pairs],
           startangle=90, wedgeprops={"width": 0.42}, textprops={"fontsize": 7.5})
    ax.set_title(f"Outcome for {summary['total_files']} resumes", fontsize=8.5)
    fig.tight_layout()
    path = TMP / "outcomes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- pages
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(GREY)
    if doc.page > 1:
        canvas.drawString(20 * mm, 12 * mm, "AI Resume Screening & Ranking System: Value Document")
        canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.restoreState()


def build() -> Path:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    s = data["batch_summary"]
    ranked, rejected = data["ranked_candidates"], data["rejected_candidates"]
    n_tests = sum(f.read_text(encoding="utf-8").count("\ndef test_") for f in (ROOT / "tests").glob("test_*.py"))
    gh_ok, gh_failed = s["github_enriched"], s["github_failed"]
    if s.get("warnings") and gh_ok:
        gh_line = (f"GitHub enrichment was incomplete because of API rate limits ({gh_ok} of {gh_ok + gh_failed} "
                   "lookups succeeded), so GitHub points were withheld from everyone to keep the ranking fair")
    elif gh_failed and not gh_ok:
        gh_line = ("GitHub lookups were rate-limited in the build environment and recorded per candidate, "
                   "not counted as failures of the batch")
    elif gh_ok:
        gh_line = f"{gh_ok} profile{'s' if gh_ok != 1 else ''} enriched with live GitHub data"
    else:
        gh_line = "GitHub enrichment was not used in this run"

    reasons = Counter()
    for c in rejected:
        for r in c["rejection_reasons"]:
            if "Python" in r:
                reasons["No Python evidence"] += 1
            elif "only in skills" in r:
                reasons["AI terms only in skills list"] += 1
            elif "AI" in r:
                reasons["No AI project evidence"] += 1
    penalised = [c for c in ranked if c["penalties"]]
    classic = [c for c in ranked if any("classic ML only" in x for x in c["concerns"])]

    story: list = []
    # ---- cover --------------------------------------------------------------
    story += [Spacer(1, 38 * mm), P("AI Resume Screening<br/>&amp; Ranking System", "title"), Spacer(1, 6),
              P("How the system creates value for the company, for its users, and for me as its builder",
                "subtitle"),
              Spacer(1, 14 * mm),
              stat_tiles([(str(s["total_files"]), "resumes processed"),
                          (str(s["eligible"]), "eligible &amp; ranked"),
                          (str(s["rejected"]), "rejected with reasons"),
                          (f"{s['elapsed_seconds']:.0f}s", "end-to-end run"),
                          (str(n_tests), "automated tests")]),
              Spacer(1, 14 * mm),
              P("Prepared by <b>MD Nawaj Saqlain Rizvi</b>", "body"),
              P("SDE Intern Coding Assignment submission · October 2026", "small"),
              Spacer(1, 30 * mm),
              P("Figures in this document are generated from <i>output/results.json</i> of the run on the "
                "provided 50-resume dataset (scoring mode: " + s["llm_mode"] + "). Re-run "
                "<i>docs/build_value_doc.py</i> after a new screening to refresh them.", "small"),
              PageBreak()]

    # ---- 1. summary ---------------------------------------------------------
    story += [P("1. Executive summary", "h1"),
              P("Hiring teams receive far more resumes than they can read carefully, and the first pass is "
                "where the most time is spent and the least consistency is achieved. This system automates "
                "that first pass for an SDE internship that needs strong Python and real AI/agentic "
                "experience. It reads a folder of resumes, removes profiles that do not meet the hard "
                "requirements, scores the rest on a transparent 100-point model, enriches them with public "
                "GitHub activity, and returns a ranked shortlist in which <b>every score and every rejection "
                "comes with the evidence behind it</b>."),
              P("The guiding design choice is simple: a technology mentioned inside a project or job counts "
                "as evidence of use, while the same word in a skills list is only a claim. That one rule is "
                "what keeps a well-written but shallow resume from outranking one that shows real work."),
              Spacer(1, 4), flow_diagram(), Spacer(1, 8),
              callout(f"<b>On the provided dataset:</b> {s['total_files']} resumes processed in "
                      f"{s['elapsed_seconds']:.0f} seconds with {s['failed_or_unreadable']} failures. "
                      f"{s['eligible']} candidates ranked, {s['rejected']} rejected with explicit reasons, "
                      f"{len(penalised)} candidates penalised for thin or undocumented AI projects, and "
                      f"{gh_line if gh_line.startswith('GitHub') else gh_line[0].lower() + gh_line[1:]}."),
              Spacer(1, 6),
              P("2. Value to the company", "h1"),
              P("The company's goals are to hire well, hire quickly, and be able to justify its decisions. "
                "The table maps each goal to what the system does and where that is visible in this build.")]
    story.append(table([
        ["Business outcome", "How the system delivers it", "Evidence in this build"],
        [P("<b>Faster first-pass screening</b>", "cell"),
         "Processes a whole folder in one run; reviewers only read the shortlist and the flagged edge cases.",
         f"{s['total_files']} resumes in {s['elapsed_seconds']:.0f}s. If a reviewer spends ~3 minutes per "
         f"resume (an assumption, not a measurement), that is ~{s['total_files'] * 3 / 60:.1f} hours of "
         "first-pass reading moved to the machine."],
        [P("<b>Better shortlists</b>", "cell"),
         "Rewards demonstrated AI systems (agents, retrieval, state, evaluation) over keywords; penalises "
         "thin LLM wrappers and tutorial projects; caps classic-ML-only profiles.",
         f"{len(penalised)} candidates penalised for wrapper/no-detail projects; {len(classic)} "
         "classic-ML-only profiles capped at 15/40 AI points."],
        [P("<b>Consistency and fairness</b>", "cell"),
         "The same rules apply to every resume in the same order. Writing style, formatting and "
         "buzzwords earn nothing on their own.",
         "Eligibility is pure rules (no LLM), covered by unit tests on edge cases such as "
         "'AI-assisted development' and skills-only claims."],
        [P("<b>Fair even when services fail</b>", "cell"),
         "If an external service fails for only some candidates, the system does not let that luck "
         "decide the ranking.",
         "When GitHub rate limits hit part of the batch, GitHub points are withheld from everyone and a "
         "warning is recorded; the fetched data stays visible."],
        [P("<b>Auditability</b>", "cell"),
         "Every decision is traceable: matched skills, the project lines that triggered points, applied "
         "penalties, and rejection reasons.",
         "results.json stores evidence lines, per-project verdicts, the rule score and LLM score "
         "separately, and GitHub status."],
        [P("<b>Controlled cost and risk</b>", "cell"),
         "Runs fully free in deterministic mode. The LLM is optional (paid Claude, the free Gemini tier, "
         "or a local model), used only on eligible resumes, cached, and limited to half of one category.",
         "Bounded concurrency (measured 4.8x faster GitHub enrichment and 3.8x faster LLM calls than "
         "sequential), disk cache, rate-limit detection, and per-resume fallback when a call fails."],
        [P("<b>Easy to adopt and change</b>", "cell"),
         "CLI or REST API; policy is in configuration (weights, thresholds, classic-ML policy, model "
         "name), so it can be tuned for other roles without touching logic.",
         "config.py and lexicon.py hold all numbers and term lists; the FastAPI endpoints wrap the "
         "same pipeline."],
    ], [32 * mm, 72 * mm, 66 * mm]))
    story.append(Spacer(1, 8))

    # ---- 3. users -----------------------------------------------------------
    story += [P("3. Value to the users", "h1"),
              P("The system has two kinds of users: the people who run and read the screening "
                "(recruiters, hiring managers, interviewers), and the candidates whose resumes it reads. "
                "It is designed to serve both."),
              P("Recruiters and hiring managers", "h2")]
    story.append(table([
        ["Need", "What the system gives them"],
        ["\"Why is this person ranked #3?\"",
         "A score breakdown across five categories, the strongest AI project with a one-line evidence quote, "
         "strengths, and concerns such as \"No evidence of Redis\" or \"Thin LLM wrapper: AI Poem Generator\"."],
        ["\"Why was this person rejected?\"",
         "Plain-language reasons, e.g. \"No evidence of Python stack (profile is JavaScript, React, "
         "Next.js)\" or \"AI terms appear only in skills/summary\". This makes a quick human double-check easy."],
        ["Interview preparation",
         "Concerns double as interview questions: probe the skills listed but never shown in a project, or "
         "ask how a 'deep' project handled evaluation and failure cases."],
        ["Trust in automation",
         "Rule and LLM scores are shown side by side; if the LLM disagrees with the rules, the candidate is "
         "flagged for manual verification rather than silently moved."],
        ["Working with messy input",
         "Corrupt, empty, unsupported and duplicate files are listed separately instead of stopping the run."],
        ["Reading the results",
         "A self-contained HTML report: score bars per category, and a click on any name shows the evidence, "
         "strengths, concerns and GitHub details. JSON and Markdown versions are produced in the same run."],
    ], [45 * mm, 125 * mm]))
    story += [P("Candidates", "h2"),
              P("Candidates benefit from being judged on what they have built. A student with a real "
                "LangGraph or RAG project and a plain resume is not out-scored by someone with a polished "
                "skills section. Mixed stacks are never penalised: JavaScript, Java or React experience does "
                "not cause rejection when the Python and AI requirements are met. A missing or private GitHub "
                "profile never affects eligibility, and a GitHub API failure is recorded instead of counted "
                "against them."),
              P("4. Value to me", "h1"),
              P("I am a final-year B.Tech CSE student at KIIT, currently an AI/ML Engineer Intern, aiming for "
                "roles in GenAI and agentic systems. This project is useful to me in four concrete ways."),
              ]
    story.append(table([
        ["What it demonstrates", "Where it shows in the code"],
        ["<b>Understanding of what real AI systems are</b>: the difference between an LLM call and an "
         "agentic or retrieval system with state, tools and evaluation.",
         "The AI-depth model scores agentic, retrieval, state, evaluation and data-pipeline signals and "
         "penalises thin wrappers; the LLM prompt encodes the same rubric."],
        ["<b>Production engineering, not just a demo</b>: failure isolation, bounded concurrency, caching, "
         "rate-limit handling, configuration separated from logic.",
         "pipeline.py, github.py and llm.py; per-file and per-call fallbacks; retries with backoff; "
         "structured output validated by Pydantic."],
        ["<b>Engineering judgment under ambiguity</b>: turning a loose brief into explicit, testable "
         "rules, and finding real data problems before they became scoring bugs.",
         "Dual PDF extractors (collapsed words, scrambled order), hyperlink-annotation parsing for GitHub, "
         f"and {n_tests} tests on the decisions that matter."],
        ["<b>Testing that catches real bugs</b>: exercising integrations through the real SDK, not "
         "just mocks of my own code.",
         "The LLM adapter test caught that the current Anthropic SDK rejects the <i>temperature</i> "
         "parameter, a bug that would have silently disabled hybrid scoring."],
    ], [85 * mm, 85 * mm]))
    story += [Spacer(1, 6),
              P("Beyond this assignment, it is a portfolio piece I can walk through in interviews, and a base "
                "I can extend (calibration against human labels, LLM-based structured extraction) to keep "
                "building skills in exactly the area I am targeting."),
              PageBreak()]

    # ---- 5. results ---------------------------------------------------------
    story += [P("5. Results on the provided dataset", "h1"),
              P(f"Scoring mode for this run: <b>{s['llm_mode']}</b>. {gh_line}."),
              Image(str(chart_top10(ranked)), width=170 * mm, height=170 * mm * 4.3 / 7.2),
              P("Top 10 eligible candidates by score, broken down by category.", "small"), Spacer(1, 6)]
    side = Table([[Image(str(chart_outcomes(s)), width=82 * mm, height=82 * mm * 2.5 / 3.4),
                   Image(str(chart_distribution(ranked)), width=82 * mm, height=82 * mm * 2.5 / 3.4)]],
                 colWidths=[85 * mm, 85 * mm])
    story.append(side)
    rows = [["Rejection reason", "Candidates"]] + [[k, str(v)] for k, v in reasons.most_common()]
    story += [Spacer(1, 6), KeepTogether([P("Why candidates were rejected", "h2"),
                                          table(rows, [120 * mm, 50 * mm]),
                                          P("A candidate can have more than one reason.", "small")])]
    story.append(PageBreak())

    top5 = [["#", "Candidate", "Score", "Strongest AI work (from the generated summary)"]]
    for c in ranked[:5]:
        summ = (c["project_summary"] or "").split(". ")[0]
        top5.append([str(c["rank"]), c["candidate_name"], f"{c['total_score']}", summ])
    story += [P("Top of the shortlist", "h2"), table(top5, [8 * mm, 38 * mm, 14 * mm, 110 * mm]),
              P("6. Responsible use and limitations", "h1"),
              P("This is a <b>decision-support tool for the first pass</b>, not a hiring decision maker. "
                "A human should review the shortlist and any rejection before it affects a candidate."),
              table([
                  ["Limitation", "Mitigation"],
                  ["Section and project detection is heuristic; unusual layouts can merge two projects.",
                   "Merging is preferred over splitting so nobody is penalised for a parsing fragment; "
                   "per-project verdicts are visible for review."],
                  ["Keywords cannot verify that claims are true or high quality.",
                   "Optional LLM judgment with evidence, GitHub signals, and concerns that feed interview "
                   "questions."],
                  ["Scanned image-only PDFs have no text layer.",
                   "Reported as unreadable rather than silently scored as zero."],
                  ["Weights reflect one role's priorities.",
                   "All weights and thresholds live in configuration; calibration against human labels is "
                   "the first planned improvement."],
                  ["Automated screening can encode bias.",
                   "No personal attributes (name, college, location) are used in scoring; only "
                   "technical evidence is scored, and every decision is explainable."],
              ], [80 * mm, 90 * mm]),
              P("7. Next steps", "h1"),
              table([
                  ["Improvement", "Value"],
                  ["Calibrate weights against ~20 human-scored resumes", "Measurable agreement with reviewers"],
                  ["LLM-based structured extraction of projects", "More accurate project-level verdicts"],
                  ["Verify linked repos against resume claims", "Stronger evidence of ownership"],
                  ["Reviewer UI with override feedback loop", "Faster review and continuous improvement"],
              ], [100 * mm, 70 * mm])]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="AI Resume Screening System: Value Document", author="MD Nawaj Saqlain Rizvi")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return OUT


if __name__ == "__main__":
    print(build())
