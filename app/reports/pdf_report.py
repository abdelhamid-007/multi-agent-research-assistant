"""Professional PDF export for the finished research report (ReportLab)."""
import logging
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.reports.markdown_parser import (
    SECTION_ORDER,
    extract_references,
    parse_report,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#2E86AB")
LIGHT = colors.HexColor("#EEF3F8")
GREY = colors.HexColor("#5B6770")
WARN = colors.HexColor("#B3261E")

MISSING_TEXT = "This section was not included in the final report."

# Characters the standard PDF fonts cannot draw -> safe replacements.
_CHAR_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    "→": "->", "←": "<-", "⇒": "=>",
    "≥": ">=", "≤": "<=", "≈": "~", "≠": "!=",
    " ": " ", " ": " ", " ": " ", "​": "",
    "【": "[", "】": "]", "〔": "[", "〕": "]",
    "［": "[", "］": "]", "（": "(", "）": ")",
    "✓": "v", "✔": "v", "✗": "x", "×": "x",
}


def slugify(text, max_len=60):
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "research-report"


def make_output_path(topic, output_dir=None, now=None):
    """output/<topic-slug>_<timestamp>.pdf, never overwriting an old file."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{slugify(topic)}_{stamp}.pdf"
    n = 2
    while path.exists():
        path = out_dir / f"{slugify(topic)}_{stamp}_{n}.pdf"
        n += 1
    return path


def pdf_safe(text):
    """Make text drawable with Helvetica (WinAnsi); never raises."""
    text = text or ""
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def inline_markup(text):
    """Markdown inline -> ReportLab paragraph markup (escaped, link-aware)."""
    t = escape(pdf_safe(text))
    t = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?![*\w])", r"<i>\1</i>", t)

    def link(m):
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".;:":
            trail = url[-1] + trail
            url = url[:-1]
        return f'<link href="{url}" color="#2E86AB">{url}</link>{trail}'

    return re.sub(r"https?://[^\s<]+", link, t)


def build_styles():
    base = getSampleStyleSheet()
    s = {}
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName="Helvetica", fontSize=10.5,
        leading=15, spaceAfter=7, alignment=TA_LEFT,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18,
        leading=22, textColor=NAVY, spaceBefore=4, spaceAfter=4, keepWithNext=1,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
        leading=16, textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=18, bulletIndent=6, spaceAfter=3,
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=s["body"], fontSize=9, leading=12, spaceAfter=0,
    )
    s["cellhead"] = ParagraphStyle(
        "cellhead", parent=s["cell"], fontName="Helvetica-Bold", textColor=colors.white,
    )
    s["ref"] = ParagraphStyle(
        "ref", parent=s["body"], fontSize=9, leading=12, leftIndent=24,
        firstLineIndent=-24, wordWrap="CJK", spaceAfter=4,
    )
    s["missing"] = ParagraphStyle(
        "missing", parent=s["body"], fontName="Helvetica-Oblique", textColor=GREY,
    )
    s["note"] = ParagraphStyle(
        "note", parent=s["body"], fontSize=9, textColor=GREY, leading=12,
    )
    s["cover_label"] = ParagraphStyle(
        "cover_label", parent=s["body"], fontName="Helvetica-Bold", fontSize=11,
        textColor=ACCENT, spaceAfter=10,
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title", parent=s["body"], fontName="Helvetica-Bold", fontSize=30,
        leading=36, textColor=NAVY, spaceAfter=18,
    )
    s["cover_meta"] = ParagraphStyle(
        "cover_meta", parent=s["body"], fontSize=12, leading=18, textColor=GREY,
        spaceAfter=4,
    )
    s["cover_warn"] = ParagraphStyle(
        "cover_warn", parent=s["body"], fontName="Helvetica-Bold", fontSize=11,
        leading=15, textColor=WARN, spaceBefore=18,
    )
    return s


def _table(rows, styles, width):
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]
    data = []
    for i, row in enumerate(rows):
        st = styles["cellhead"] if i == 0 else styles["cell"]
        data.append([Paragraph(inline_markup(c), st) for c in row])
    tbl = Table(data, colWidths=[width / cols] * cols, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D0DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    tbl.setStyle(TableStyle(style))
    return [tbl, Spacer(1, 8)]


def _block_to_flowables(block, styles, width):
    kind = block[0]
    if kind == "para":
        return [Paragraph(inline_markup(block[1]), styles["body"])]
    if kind == "sub":
        return [Paragraph(inline_markup(block[1]), styles["h2"])]
    if kind == "bullet":
        style = ParagraphStyle(
            "b%d" % block[1], parent=styles["bullet"], leftIndent=18 + 14 * block[1],
            bulletIndent=6 + 14 * block[1],
        )
        bullet = "•" if block[1] == 0 else "-"
        return [Paragraph(inline_markup(block[2]), style, bulletText=bullet)]
    if kind == "num":
        return [Paragraph(inline_markup(block[2]), styles["bullet"], bulletText=f"{block[1]}.")]
    if kind == "table":
        return _table(block[1], styles, width)
    return []


def _safe_flowables(block, styles, width):
    """Convert one block; on any formatting error fall back to plain text."""
    try:
        flowables = _block_to_flowables(block, styles, width)
        for f in flowables:  # force markup parsing now, not at build time
            if isinstance(f, Paragraph):
                f.wrap(width, 1000)
        return flowables
    except Exception as exc:  # noqa: BLE001 - never let one block kill the PDF
        logger.warning("PDF: block %s failed (%s); using plain text", block[0], exc)
        text = block[-1] if block[0] != "table" else " | ".join(" / ".join(r) for r in block[1])
        return [Paragraph(escape(pdf_safe(str(text))), styles["body"])]


def _section_flowables(name, blocks, styles, width, refs_info=None):
    out = [Paragraph(name, styles["h1"]),
           HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=8)]
    if name == "References":
        refs, fallback = refs_info
        if not refs:
            out.append(Paragraph("No verifiable source URLs were available for this report.",
                                 styles["missing"]))
            return out
        if fallback:
            out.append(Paragraph(
                "The report did not list its own sources. The URLs below are the "
                "sources retrieved during the research step.", styles["note"]))
        for i, (num, desc, url) in enumerate(refs, 1):
            label = num or i
            text = f"[{label}] " + (inline_markup(desc) + " - " if desc else "") + inline_markup(url)
            try:
                p = Paragraph(text, styles["ref"])
                p.wrap(width, 1000)
            except Exception:  # noqa: BLE001
                p = Paragraph(escape(pdf_safe(f"[{label}] {url}")), styles["ref"])
            out.append(p)
        return out

    if not blocks:
        out.append(Paragraph(MISSING_TEXT, styles["missing"]))
        return out
    for block in blocks:
        out.extend(_safe_flowables(block, styles, width))
    return out


def _cover(topic, status_line, warn, styles, today):
    items = [
        Spacer(1, 6 * cm),
        Paragraph("RESEARCH REPORT", styles["cover_label"]),
        Paragraph(escape(pdf_safe(topic)), styles["cover_title"]),
        HRFlowable(width="40%", thickness=3, color=ACCENT, hAlign="LEFT", spaceAfter=14),
        Paragraph(today, styles["cover_meta"]),
        Paragraph(escape(pdf_safe(status_line)), styles["cover_meta"]),
        Paragraph("Generated by the LangGraph Multi-Agent Research Assistant", styles["cover_meta"]),
    ]
    if warn:
        items.append(Paragraph(escape(pdf_safe(warn)), styles["cover_warn"]))
    items.append(PageBreak())
    return items


def _status_text(state):
    approved = bool(state.get("approved"))
    revisions = state.get("revision_count", 0)
    if approved:
        return f"Reviewer status: APPROVED (after {revisions} revision(s))", None
    status = state.get("final_status", "unknown")
    return (
        f"Reviewer status: NOT APPROVED ({status}; {revisions} revision(s))",
        "DRAFT - this report was not approved by the reviewer. Treat its content as unverified.",
    )


def _draw_cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 2.2 * cm, w, 2.2 * cm, stroke=0, fill=1)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 2.5 * cm, w, 0.3 * cm, stroke=0, fill=1)
    canvas.restoreState()


def _page_decorator(topic):
    short = pdf_safe(topic if len(topic) <= 70 else topic[:67] + "...")

    def draw(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, h - 1.3 * cm, short)
        canvas.setStrokeColor(colors.HexColor("#C5D0DB"))
        canvas.line(2 * cm, h - 1.5 * cm, w - 2 * cm, h - 1.5 * cm)
        canvas.line(2 * cm, 1.5 * cm, w - 2 * cm, 1.5 * cm)
        canvas.drawRightString(w - 2 * cm, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def _build(path, topic, sections, refs_info, state):
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.4 * cm, bottomMargin=2.2 * cm,
        title=pdf_safe(topic), author="LangGraph Multi-Agent Research Assistant",
        subject="Research Report",
    )
    width = A4[0] - 4 * cm
    status_line, warn = _status_text(state)
    story = _cover(topic, status_line, warn, styles, datetime.now().strftime("%B %d, %Y"))

    first = True
    for name in SECTION_ORDER:
        if not first:
            story.append(Spacer(1, 14))
        first = False
        flow = _section_flowables(name, sections.get(name), styles, width, refs_info)
        # keep heading + first block together; long sections flow across pages
        story.extend([KeepTogether(flow[:3])] + flow[3:] if len(flow) > 3 else flow)

    decorate = _page_decorator(topic)
    doc.build(story, onFirstPage=_draw_cover, onLaterPages=decorate)


def _build_plain(path, topic, draft):
    """Last-resort PDF: plain escaped text, no markup, no tables."""
    styles = build_styles()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=pdf_safe(topic))
    story = [Paragraph(escape(pdf_safe(topic)), styles["h1"]), Spacer(1, 10)]
    for chunk in re.split(r"\n\s*\n", draft or "(empty report)"):
        story.append(Paragraph(escape(pdf_safe(chunk)).replace("\n", "<br/>"), styles["body"]))
    doc.build(story)


def export_report_pdf(state, output_dir=None):
    """Export the final report in `state` to a PDF. Returns the Path or None.

    Never raises: failures are logged and None is returned so the research
    workflow's result is unaffected.
    """
    topic = (state.get("question") or "Research Report").strip()
    draft = state.get("draft") or ""

    try:
        path = make_output_path(topic, output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF: cannot create output path: %s", exc)
        return None

    if not draft.strip():
        logger.error("PDF: no report draft available; nothing to export")
        return None

    try:
        sections = parse_report(draft)
        missing = [n for n in SECTION_ORDER if n not in sections]
        if missing:
            logger.warning("PDF: sections missing from report: %s", ", ".join(missing))
        refs_info = extract_references(sections.get("References"), state.get("research", ""))
        _build(path, topic, sections, refs_info, state)
        logger.info("PDF: report exported to %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF: formatted export failed (%s); trying plain fallback", exc)

    try:
        _build_plain(path, topic, draft)
        logger.warning("PDF: plain-text fallback exported to %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF: fallback export failed too: %s", exc)
        path.unlink(missing_ok=True)
        return None
