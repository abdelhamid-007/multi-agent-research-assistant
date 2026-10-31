"""Turn the Writer's markdown-style draft into structured report sections.

Pure functions only (no ReportLab), so they are easy to test.
"""
import re
from collections import OrderedDict

# Final PDF sections, in order. Each maps to the headings the Writer may use.
SECTION_ORDER = [
    "Executive Summary",
    "Research Findings",
    "Analysis",
    "Business Opportunities",
    "Recommendations",
    "Conclusion",
    "References",
]

_ALIASES = {
    "executive summary": "Executive Summary",
    "summary": "Executive Summary",
    "research findings": "Research Findings",
    "findings": "Research Findings",
    "key findings": "Research Findings",
    "analysis": "Analysis",
    "technical analysis": "Analysis",
    "risks": "Analysis",
    "risk": "Analysis",
    "risks and mitigations": "Analysis",
    "business opportunities": "Business Opportunities",
    "opportunities": "Business Opportunities",
    "recommendations": "Recommendations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "sources": "References",
    "references": "References",
}

# Headings whose own title should be kept as a sub-heading inside "Analysis".
_KEEP_AS_SUBHEADING = {"technical analysis", "risks", "risk", "risks and mitigations"}

URL_RE = re.compile(r"https?://[^\s\)\]>\"'`,<]+")

_HASH_HEADING = re.compile(r"^\s*(#{1,6})\s*(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*(?:\d+[.)]\s*)?\*\*(.+?)\*\*:?\s*$")
_PLAIN_NUMBERED_HEADING = re.compile(r"^\s*\d+[.)]\s+([A-Za-z][A-Za-z &]+?):?\s*$")
_BULLET = re.compile(r"^(\s*)[-*•+]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def clean_title(text):
    """Strip markdown noise and numbering from a heading."""
    text = re.sub(r"[#*_`]", "", text)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    return text.strip().rstrip(":").strip()


def match_section(text):
    """Return the canonical section name for a heading, or None."""
    return _ALIASES.get(clean_title(text).lower())


def _heading_title(line):
    """If the line is a heading, return (title, is_hash_heading), else None."""
    m = _HASH_HEADING.match(line)
    if m:
        return clean_title(m.group(2)), len(m.group(1))
    m = _BOLD_LINE.match(line)
    if m:
        return clean_title(m.group(1)), 0
    m = _PLAIN_NUMBERED_HEADING.match(line)
    if m and match_section(m.group(1)):
        return clean_title(m.group(1)), 0
    return None


def _split_row(line):
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def parse_blocks(lines):
    """Parse lines of one section into blocks.

    Block shapes:
      ("para", text) ("sub", text) ("bullet", level, text)
      ("num", number, text) ("table", rows)
    """
    blocks = []
    para = []
    table = []

    def flush_para():
        if para:
            blocks.append(("para", " ".join(para)))
            para.clear()

    def flush_table():
        if table:
            rows = [r for r in table if not _TABLE_SEP.match(r)]
            parsed = [_split_row(r) for r in rows]
            if parsed:
                blocks.append(("table", parsed))
            table.clear()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_para()
            table.append(stripped)
            continue
        flush_table()

        if not stripped:
            flush_para()
            continue
        if _RULE.match(stripped):
            flush_para()
            continue

        heading = _heading_title(line)
        if heading and heading[0]:
            flush_para()
            blocks.append(("sub", heading[0]))
            continue

        m = _BULLET.match(line)
        if m:
            flush_para()
            level = min(len(m.group(1).replace("\t", "  ")) // 2, 2)
            blocks.append(("bullet", level, m.group(2).strip()))
            continue

        m = _NUMBERED.match(line)
        if m:
            flush_para()
            blocks.append(("num", m.group(2), m.group(3).strip()))
            continue

        # Indented continuation of the previous list item.
        if line.startswith(("  ", "\t")) and blocks and blocks[-1][0] in ("bullet", "num") and not para:
            last = blocks[-1]
            blocks[-1] = last[:-1] + (last[-1] + " " + stripped,)
            continue

        para.append(stripped)

    flush_para()
    flush_table()
    return blocks


def parse_report(draft):
    """Split a draft into canonical sections.

    Returns OrderedDict[section_name -> list of blocks]. Sections that were
    absent from the draft are simply not present in the result.
    """
    raw = OrderedDict()  # section -> list of raw lines
    current = None
    preamble = []

    for line in (draft or "").splitlines():
        heading = _heading_title(line)
        if heading and heading[0]:
            title, level = heading
            section = match_section(title)
            if section:
                current = section
                raw.setdefault(section, [])
                if title.lower() in _KEEP_AS_SUBHEADING:
                    raw[section].append(f"### {title}")
                continue
            # A top-level report title before any section: ignore as a title.
            if level == 1 and current is None:
                continue
        if current is None:
            preamble.append(line)
        else:
            raw[current].append(line)

    # Text before the first heading is kept only if there is no summary at all.
    if preamble and "".join(preamble).strip() and "Executive Summary" not in raw:
        raw["Executive Summary"] = preamble

    parsed = OrderedDict()
    for name in SECTION_ORDER:
        if name in raw:
            blocks = parse_blocks(raw[name])
            if blocks:
                parsed[name] = blocks
    return parsed


def extract_references(section_blocks, research=""):
    """Collect references without inventing any.

    Only URLs present in the draft's Sources section are used, and (when
    research text is given) only those that also appear in the research.
    Returns (references, from_research_fallback) where references is a list of
    (label_number_or_None, description, url).
    """

    def norm(u):
        return u.rstrip(".;:*_/").lower()

    research_urls = {norm(u) for u in URL_RE.findall(research or "")}
    refs, seen = [], set()

    for block in section_blocks or []:
        kind = block[0]
        text = block[-1] if kind != "table" else " ".join(" ".join(r) for r in block[1])
        number = block[1] if kind == "num" else None
        lead = re.match(r"\s*\[(\d+)\]\s*", text)  # "- [3] Title - URL" keeps source number 3
        if lead:
            number, text = lead.group(1), text[lead.end():]
        for url in URL_RE.findall(text):
            url = url.rstrip(".;:*_")
            key = norm(url)
            if key in seen:
                continue
            if research and key not in research_urls:
                continue  # unverifiable -> never printed as a reference
            seen.add(key)
            desc = text.replace(url, "").strip(" -–—:*()[]<>")
            refs.append((number, desc if desc != url else "", url))

    if refs:
        return refs, False

    # Fallback: sources actually retrieved by the research agent.
    for url in URL_RE.findall(research or ""):
        url = url.rstrip(".;:*_")
        if norm(url) not in seen:
            seen.add(norm(url))
            refs.append((None, "", url))
    return refs, bool(refs)
