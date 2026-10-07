"""Turn raw resume text into sections, entries and contact fields.

Layout-agnostic heuristics: headings are matched against an alias table, and
project/job entries are split on "header-looking" lines inside applied sections.
Imperfect splits degrade gracefully (two projects merged into one entry), which
is safer than over-splitting a single project into fragments.
"""
from __future__ import annotations

import re
from collections import Counter

from .models import Entry, ParsedResume, SectionKind

HEADING_ALIASES: dict[SectionKind, list[str]] = {
    SectionKind.SUMMARY: ["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE SUMMARY", "PROFILE", "OBJECTIVE",
                          "OBJECTIVES", "PROFILE OBJECTIVE", "CAREER OBJECTIVE", "RESUME SUMMARY", "ABOUT",
                          "ABOUT ME", "PROFESSIONAL PROFILE"],
    SectionKind.SKILLS: ["SKILLS", "TECHNICAL SKILLS", "SKILLS SUMMARY", "TOOLS", "PROGRAMMING LANGUAGES",
                         "FRAMEWORKS & LIBRARIES", "TECHNICAL EXPERTISE", "CORE COMPETENCIES", "TECH STACK",
                         "KEY SKILLS", "SKILLS & TOOLS", "TECHNOLOGIES", "SKILLS AND TOOLS",
                         "PROGRAMMING LANGUAGES & WEB TECHNOLOGIES", "SKILLS LANGUAGES TOOLS"],
    SectionKind.EXPERIENCE: ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "INTERNSHIPS",
                             "INTERNSHIP", "INTERNSHIP EXPERIENCE", "INTERNSHIP / TRAINING", "EMPLOYMENT",
                             "EMPLOYMENT HISTORY", "WORK HISTORY", "RELEVANT EXPERIENCE", "INDUSTRY EXPERIENCE"],
    SectionKind.PROJECTS: ["PROJECTS", "KEY PROJECTS", "SELECTED PROJECTS", "PROJECT EXPERIENCE",
                           "PERSONAL PROJECTS", "ACADEMIC PROJECTS", "TECHNICAL PROJECTS", "OPEN SOURCE",
                           "HACKATHON EXPERIENCE", "PROJECT", "SIDE PROJECTS", "MAJOR PROJECTS"],
    SectionKind.RESEARCH: ["RESEARCH", "PUBLICATIONS", "PUBLICATION", "RESEARCH EXPERIENCE",
                           "RESEARCH UNDER REVIEW", "RESEARCH & PUBLICATIONS"],
    SectionKind.EDUCATION: ["EDUCATION", "RELEVANT COURSEWORK", "COURSEWORK", "ACADEMIC BACKGROUND",
                            "EDUCATIONAL QUALIFICATION", "EDUCATIONAL QUALIFICATIONS", "ACADEMICS"],
    SectionKind.CERTIFICATIONS: ["CERTIFICATIONS", "CERTIFICATION", "CERTIFICATIONS & ACHIEVEMENTS",
                                 "ACHIEVEMENTS", "KEY ACHIEVEMENTS", "AWARDS", "HONORS", "HONOURS",
                                 "EXTRA CURRICULAR ACTIVITIES", "EXTRACURRICULAR ACTIVITIES",
                                 "LEADERSHIP", "POSITIONS OF RESPONSIBILITY", "ACTIVITIES", "LANGUAGES",
                                 "LANGUAGES KNOWN", "INTERESTS", "HOBBIES", "ONLINE CODING PROFILES",
                                 "KEY ACHIEVEMENTS & METRICS", "COURSES", "TRAINING", "LINKS",
                                 "ACHIEVEMENTS & CERTIFICATIONS", "AWARDS & ACHIEVEMENTS"],
}
_HEADING_LOOKUP = {alias: kind for kind, aliases in HEADING_ALIASES.items() for alias in aliases}
_ALIASES_LONGEST_FIRST = sorted(_HEADING_LOOKUP, key=len, reverse=True)

BULLET_RE = re.compile(r"^\s*(?:[•●◦▪■►➢✓\-\*–·○◆]|\d{1,2}[.)])\s*")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\bpresent\b|\bcurrent\b|\bongoing\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
GITHUB_URL_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))", re.I)
GITHUB_IO_RE = re.compile(r"([A-Za-z0-9-]{1,39})\.github\.io", re.I)
GITHUB_RESERVED = {"features", "topics", "settings", "orgs", "about", "pricing", "login", "join",
                   "explore", "marketplace", "sponsors", "collections", "trending", "apps", "site"}


def _normalize_heading(line: str) -> str:
    line = re.sub(r"\(.*?\)", " ", line)
    line = re.sub(r"[^A-Za-z&/ ]", " ", line)
    return re.sub(r"\s+", " ", line).strip().upper()


def heading_kind(line: str) -> SectionKind | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 45 or BULLET_RE.match(stripped):
        return None
    normalized = _normalize_heading(stripped)
    kind = _HEADING_LOOKUP.get(normalized)
    if kind is None and stripped.isupper():
        # "CERTIFICATIONS & PROFESSIONAL DEVELOPMENT" -> CERTIFICATIONS (all-caps lines only)
        for alias in _ALIASES_LONGEST_FIRST:
            if normalized.startswith(alias + " "):
                return _HEADING_LOOKUP[alias]
    return kind


def split_sections(text: str) -> list[tuple[SectionKind, str]]:
    """Ordered (kind, body) chunks. A kind can occur more than once."""
    chunks: list[tuple[SectionKind, list[str]]] = [(SectionKind.HEADER, [])]
    for line in text.split("\n"):
        kind = heading_kind(line)
        if kind is not None:
            chunks.append((kind, []))
        else:
            chunks[-1][1].append(line)
    return [(kind, "\n".join(lines).strip()) for kind, lines in chunks if "\n".join(lines).strip()]


def _is_bullet(line: str) -> bool:
    return bool(BULLET_RE.match(line))


def split_entries(kind: SectionKind, body: str) -> list[Entry]:
    """Split an applied section into project/job entries.

    A non-bullet line starts a new entry when it looks like a header: it carries
    a date or a ' | ' tech-stack separator, or it follows a bullet line that did
    not wrap (short previous line) and does not itself start in lowercase.
    Consecutive header lines are merged (company line + role line + stack line).
    """
    lines = [ln.rstrip() for ln in body.split("\n") if ln.strip()]
    if not lines:
        return []
    bullet_lens = [len(ln) for ln in lines if _is_bullet(ln)]
    wrap_width = (sorted(bullet_lens)[len(bullet_lens) // 2] if bullet_lens else 90) * 0.8

    entries: list[tuple[list[str], list[str]]] = []  # (title_lines, body_lines)
    prev_was_header = False
    prev_line = ""
    for line in lines:
        stripped = line.strip()
        is_header = False
        if not _is_bullet(stripped):
            looks_titled = bool(DATE_RE.search(stripped)) or " | " in stripped or "|" in stripped
            starts_lower = stripped[:1].islower()
            prev_wrapped = len(prev_line) >= wrap_width and not prev_was_header
            is_header = not entries or (not starts_lower and (
                looks_titled or (not prev_wrapped and len(stripped) < 90)))
        if is_header:
            if entries and prev_was_header:
                entries[-1][0].append(stripped)         # multi-line header
            else:
                entries.append(([stripped], []))
        else:
            if not entries:
                entries.append(([stripped], []))
            else:
                entries[-1][1].append(stripped)
        prev_was_header = is_header
        prev_line = stripped

    # A header block with (almost) no body is usually a sub-heading ("Project 1: X" followed by
    # "Role: Developer" + bullets): fold it into the next entry rather than keep a fragment.
    merged: list[tuple[list[str], list[str]]] = []
    for title, body_lines in entries:
        if merged and len("".join(merged[-1][1])) < 60:
            prev_title, prev_body = merged.pop()
            title, body_lines = prev_title + prev_body + title, body_lines
        merged.append((title, body_lines))
    return [Entry(section=kind, title=" / ".join(t)[:200], body="\n".join(b)) for t, b in merged]


def extract_name(text: str, email: str | None, fallback: str) -> str:
    lines = [ln.strip() for ln in text.split("\n")[:8] if ln.strip()]
    candidates: list[str] = []
    for i, line in enumerate(lines):
        segment = re.split(r"\s[|•·—]\s|\s{2,}|\+\d|\d|[\w.+-]+@|\b(?:e-?mail|phone|mobile|tel)\b",
                           line, flags=re.I)[0].strip(" ,|:")
        cut_at_comma = "," in segment
        segment = segment.split(",")[0].strip()
        if not segment or re.search(r"https?:|www\.|linkedin|github", segment, re.I):
            continue
        if heading_kind(segment):
            break
        tokens = segment.split()
        if 1 <= len(tokens) <= 5 and all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", t) for t in tokens):
            # Names split over two lines ("Prathamesh" / "Patil").
            if len(tokens) == 1 and i + 1 < len(lines) and re.fullmatch(r"[A-Za-z][A-Za-z.'-]+", lines[i + 1]):
                tokens.append(lines[i + 1])
            if cut_at_comma and len(tokens) >= 3 and email:
                # "Atul Kumar Dehradun, Uttarakhand" -> trailing tokens are a location.
                local = email.split("@")[0].lower()
                hits = [i for i, t in enumerate(tokens) if t.lower() in local]
                if hits:
                    tokens = tokens[: max(2, hits[-1] + 1)]
            if len(tokens) >= 2:
                candidates.append(" ".join(tokens))
                break
    if candidates:
        name = candidates[0]
        return name.title() if name.isupper() or name.islower() else name
    if email:
        local = re.sub(r"\d+", "", email.split("@")[0])
        parts = [p for p in re.split(r"[._-]", local) if p]
        if parts:
            return " ".join(p.capitalize() for p in parts)
    return fallback


def extract_github(text: str, links: list[str]) -> tuple[str | None, str | None]:
    """Return (username, note). Hyperlink annotations take priority over visible text."""
    found: list[str] = []
    for source in [*links, text]:
        for m in GITHUB_URL_RE.finditer(source):
            user = m.group(1)
            if user.lower() not in GITHUB_RESERVED:
                found.append(user)
    if found:
        # Most frequent username (case-insensitive); project links reinforce the profile.
        counts = Counter(u.lower() for u in found)
        top = counts.most_common(1)[0][0]
        return next(u for u in found if u.lower() == top), None
    for source in [*links, text]:
        m = GITHUB_IO_RE.search(source)
        if m:
            return m.group(1), "username inferred from github.io portfolio URL"
    if any("github.com" in link.lower() for link in links) or re.search(r"github\.com/?\s", text, re.I):
        return None, "GitHub link present but contains no username"
    if re.search(r"\bgithub\b", text, re.I):
        return None, "GitHub mentioned but no profile URL found"
    return None, None


def parse_resume(file: str, sha256: str, text: str, links: list[str]) -> ParsedResume:
    chunks = split_sections(text)
    sections: dict[SectionKind, str] = {}
    entries: list[Entry] = []
    for kind, body in chunks:
        sections[kind] = (sections.get(kind, "") + "\n" + body).strip()
        if kind.is_applied:
            for entry in split_entries(kind, body):
                # Some exports repeat whole sections; never double-count an entry.
                if not any(e.title == entry.title and e.body == entry.body for e in entries):
                    entries.append(entry)

    emails = EMAIL_RE.findall(text)
    email = emails[0].lower() if emails else None
    gh_user, gh_note = extract_github(text, links)
    stem = file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return ParsedResume(
        file=file, sha256=sha256, text=text, sections=sections, entries=entries,
        name=extract_name(text, email, stem), email=email,
        github_username=gh_user, github_note=gh_note, links=links,
    )
