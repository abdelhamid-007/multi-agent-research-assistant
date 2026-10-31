"""Break a report draft into sections, sentences, citations and references."""
import re
from dataclasses import dataclass, field

from app.evaluation.evidence import strip_citations

REQUIRED_SECTIONS = [
    "Executive Summary",
    "Research Findings",
    "Business Opportunities",
    "Technical Analysis",
    "Risks",
    "Recommendations",
    "Conclusion",
    "Sources",
]

_ALIASES = {
    "executive summary": "Executive Summary", "summary": "Executive Summary",
    "research findings": "Research Findings", "findings": "Research Findings",
    "key findings": "Research Findings",
    "business opportunities": "Business Opportunities", "opportunities": "Business Opportunities",
    "technical analysis": "Technical Analysis", "analysis": "Technical Analysis",
    "risks": "Risks", "risk": "Risks", "risks and mitigations": "Risks",
    "recommendations": "Recommendations",
    "conclusion": "Conclusion", "conclusions": "Conclusion",
    "sources": "Sources", "references": "Sources",
}

URL_RE = re.compile(r"https?://[^\s\)\]>\"'`,<]+")
_MARKER = re.compile(r"\[(\d+)\]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*\*\*(.+?)\*\*:?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MIN_SENTENCE_WORDS = 4


def canonical_section(title):
    return _ALIASES.get(re.sub(r"[#*_`:]", "", title or "").strip().lower().lstrip("0123456789.) ").strip())


@dataclass
class Sentence:
    text: str                   # plain text without citation markers
    cited_ids: list[int]
    section: str | None


@dataclass
class ParsedDraft:
    present: list[str] = field(default_factory=list)          # canonical sections, in order of appearance
    section_lines: dict = field(default_factory=dict)          # section -> raw lines
    items: dict = field(default_factory=dict)                  # section -> plain bullet items / sentences
    sentences: list[Sentence] = field(default_factory=list)    # all sentences outside Sources
    body_words: int = 0
    source_entries: list = field(default_factory=list)         # (id or None, url) in the Sources section
    urls: list[str] = field(default_factory=list)              # every URL anywhere in the draft
    marker_ids: set = field(default_factory=set)               # [n] numbers used anywhere


def _plain(line):
    line = re.sub(r"[*`]+", "", line)
    return re.sub(r"\s+", " ", line.replace("|", " ")).strip()


def _heading(line):
    m = _HEADING.match(line) or _BOLD_LINE.match(line)
    return m.group(1) if m else None


def parse_draft(draft):
    parsed = ParsedDraft(urls=URL_RE.findall(draft or ""))
    current = None

    for line in (draft or "").splitlines():
        title = _heading(line)
        if title is not None:
            section = canonical_section(title)
            if section:
                current = section
                if section not in parsed.present:
                    parsed.present.append(section)
                    parsed.section_lines[section] = []
                    parsed.items[section] = []
                continue
            if line.lstrip().startswith("#"):
                continue  # an unknown (sub)heading: keep the current section
        if not line.strip():
            continue
        if current:
            parsed.section_lines[current].append(line)

        if current == "Sources":
            m = re.search(r"\[(\d+)\]", line) or re.match(r"\s*(\d+)[.)]", line)
            url = URL_RE.search(line)
            if url:
                parsed.source_entries.append((int(m.group(1)) if m else None, url.group(0)))
            if m:
                parsed.marker_ids.add(int(m.group(1)))
            continue
        if _TABLE_SEP.match(line):
            continue

        bullet = _BULLET.match(line)
        text = _plain(bullet.group(1) if bullet else line)
        if not text:
            continue
        pieces = _SENTENCE_SPLIT.split(text)
        for piece in pieces:
            ids = [int(n) for n in _MARKER.findall(piece)]
            parsed.marker_ids.update(ids)
            plain = re.sub(r"\s+", " ", strip_citations(piece)).strip()
            if len(plain.split()) >= MIN_SENTENCE_WORDS:
                parsed.sentences.append(Sentence(plain, ids, current))
            parsed.body_words += len(plain.split())
        if current:
            cleaned = [re.sub(r"\s+", " ", strip_citations(p)).strip() for p in pieces]
            parsed.items[current].extend([" ".join(cleaned)] if bullet else cleaned)
    return parsed


def section_words(parsed, section):
    return sum(len(strip_citations(_plain(line)).split()) for line in parsed.section_lines.get(section, []))
