"""Integration tests on real PDF and DOCX files built from synthetic resumes."""
import asyncio

import docx
from conftest import AGENTIC
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from resume_screener.config import Settings
from resume_screener.ingest import extract_text
from resume_screener.pipeline import run_screening
from resume_screener.report import write_html

BODY = AGENTIC.replace("github.com/asharao", "GitHub")  # visible text hides the URL


def make_pdf(path, text, link=None):
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in text.split("\n"):
        c.drawString(40, y, line)
        if link and line.strip() == "asha@example.com | GitHub":
            c.linkURL(link, (150, y - 2, 200, y + 10))  # hyperlink annotation only
        y -= 14
    c.save()


def make_docx(path, text):
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    d.save(str(path))


def test_pdf_text_and_hidden_github_link(tmp_path):
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf, BODY, link="https://github.com/asharao")
    text, links = extract_text(pdf)
    assert "LangGraph multi-agent workflow" in text
    assert "https://github.com/asharao" in links


def test_pdf_and_docx_through_full_pipeline(tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    make_pdf(folder / "a.pdf", BODY, link="https://github.com/asharao")
    make_docx(folder / "b.docx", BODY.replace("Asha Rao", "Bina Rao").replace("asha@", "bina@"))
    s = Settings(github_enabled=False, llm_provider="none", cache_enabled=False, cache_dir=tmp_path / "c")
    report = asyncio.run(run_screening(folder, s, llm_client=None))
    names = {c.candidate_name: c for c in report.ranked_candidates}
    assert set(names) == {"Asha Rao", "Bina Rao"}
    assert names["Asha Rao"].github.username == "asharao"            # read from the annotation
    assert names["Asha Rao"].total_score == names["Bina Rao"].total_score  # same content, same score

    html = tmp_path / "report.html"
    write_html(report, html)
    out = html.read_text(encoding="utf-8")
    assert "Asha Rao" in out and "Ranked shortlist" in out


def test_txt_resumes_in_windows_and_utf8_encodings(tmp_path):
    text = "Café Rao – AI Engineer • “quoted”\nSKILLS\nPython"
    for name, data in {"cp1252.txt": text.encode("cp1252"),          # Notepad on Windows
                       "utf8.txt": text.encode("utf-8"),
                       "utf8_bom.txt": b"\xef\xbb\xbf" + text.encode("utf-8")}.items():
        path = tmp_path / name
        path.write_bytes(data)
        decoded, _ = extract_text(path)
        assert "Café Rao – AI Engineer • “quoted”" in decoded, name
        assert "\ufffd" not in decoded and not decoded.startswith("\ufeff")
