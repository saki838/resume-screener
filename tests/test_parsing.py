from resume_screener.parsing import extract_github, extract_name, heading_kind, parse_resume
from resume_screener.models import SectionKind


def test_github_prefers_hyperlink_annotation_over_visible_text():
    user, note = extract_github("LinkedIn | GitHub : raj-github", ["https://github.com/rthumar0903"])
    assert user == "rthumar0903" and note is None


def test_github_edge_cases():
    assert extract_github("portfolio: jane-doe.github.io", [])[0] == "jane-doe"
    assert extract_github("x", ["https://github.com/"])[0] is None
    assert "no username" in extract_github("x", ["https://github.com/"])[1]
    assert extract_github("no links here", []) == (None, None)


def test_name_variants():
    assert extract_name("Prathamesh\nPatil\n+91 73490 41840", None, "f") == "Prathamesh Patil"
    assert extract_name("P.V. Jayavardhana Sai Email: pv@x.com", "pv@x.com", "f") == "P.V. Jayavardhana Sai"
    assert extract_name("Atul Kumar Dehradun, Uttarakhand, India", "atulofficial02@gmail.com", "f") == "Atul Kumar"
    assert extract_name("saif@x.com\n+91 999", "saif.ali@x.com", "f") == "Saif Ali"


def test_heading_detection():
    assert heading_kind("TECHNICAL SKILLS") == SectionKind.SKILLS
    assert heading_kind("Research (Under Review)") == SectionKind.RESEARCH
    assert heading_kind("CERTIFICATIONS & PROFESSIONAL DEVELOPMENT") == SectionKind.CERTIFICATIONS
    assert heading_kind("Software Engineer Intern") is None
    assert heading_kind("• Projects delivered on time") is None


def test_entries_split_on_project_headers():
    text = """PROJECTS
Alpha | Python | 2024
• did a thing with a reasonably long description that wraps onto another line in the
pdf because it is long
Beta | FastAPI | 2025
• did another thing
"""
    r = parse_resume("x", "s", text, [])
    assert [e.title.split(" |")[0] for e in r.entries] == ["Alpha", "Beta"]
