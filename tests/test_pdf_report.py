import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.reports import pdf_report
from app.reports.markdown_parser import extract_references, parse_report
from app.reports.pdf_report import export_report_pdf, make_output_path, slugify

URL1 = "https://example.com/agents"
URL2 = "https://example.org/report?a=1&b=2"

DRAFT = f"""# Agent Trends

## 1. Executive Summary
Agents are **growing fast** [1]. Visit {URL1}.

## 2. Research Findings
- Finding one [1]
  - nested point
- Finding two with *emphasis*

| Trend | Evidence |
|-------|----------|
| Orchestration | Strong [1] |
| Governance | Mixed |

## 3. Business Opportunities
1. Consulting
2. Tooling

## 4. Technical Analysis
Uses cost‑optimization → better results ≥ 90% <tag> & more.

## 5. Risks
Some risk.

## 6. Recommendations
**Do this now**
Pilot a small project.

## 7. Conclusion
Done.

## 8. Sources
1. Example Agents - {URL1}
2. Invented - https://invented.example/x
"""

RESEARCH = f"Source: {URL1}\nOther: {URL2}"


class ParserTests(unittest.TestCase):
    def test_sections_and_mapping(self):
        sections = parse_report(DRAFT)
        self.assertEqual(
            list(sections),
            ["Executive Summary", "Research Findings", "Analysis",
             "Business Opportunities", "Recommendations", "Conclusion", "References"],
        )
        kinds = [b[0] for b in sections["Research Findings"]]
        self.assertIn("table", kinds)
        self.assertIn("bullet", kinds)
        analysis_subs = [b[1] for b in sections["Analysis"] if b[0] == "sub"]
        self.assertEqual(analysis_subs, ["Technical Analysis", "Risks"])

    def test_missing_sections_are_absent(self):
        sections = parse_report("## Executive Summary\nOnly this.")
        self.assertEqual(list(sections), ["Executive Summary"])

    def test_empty_draft(self):
        self.assertEqual(len(parse_report("")), 0)

    def test_references_never_include_unverified_urls(self):
        sections = parse_report(DRAFT)
        refs, fallback = extract_references(sections["References"], RESEARCH)
        urls = [r[2] for r in refs]
        self.assertEqual(urls, [URL1])
        self.assertFalse(fallback)

    def test_references_fall_back_to_research_urls_only(self):
        refs, fallback = extract_references([], RESEARCH)
        self.assertTrue(fallback)
        self.assertEqual({r[2] for r in refs}, {URL1, URL2})

    def test_no_references_when_nothing_real(self):
        self.assertEqual(extract_references([], "no urls here"), ([], False))


class HelperTests(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(
            slugify("Latest AI agent development trends in 2026!"),
            "latest-ai-agent-development-trends-in-2026",
        )
        self.assertEqual(slugify("???"), "research-report")

    def test_output_path_unique(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = make_output_path("Topic", d)
            p1.write_bytes(b"x")
            p2 = make_output_path("Topic", d, now=None)
            self.assertNotEqual(p1, p2)

    def test_pdf_safe_replaces_unsupported_chars(self):
        out = pdf_report.pdf_safe("a‑b → c ≥ \U0001f600")
        self.assertIn("a-b -> c >=", out)
        self.assertEqual(pdf_report.pdf_safe("fact【1】"), "fact[1]")
        out.encode("cp1252")


class ExportTests(unittest.TestCase):
    def state(self, **kw):
        s = {"question": "Latest AI agent development trends in 2026",
             "draft": DRAFT, "research": RESEARCH, "approved": True,
             "revision_count": 1, "final_status": "approved"}
        s.update(kw)
        return s

    def test_exports_valid_multi_section_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            path = export_report_pdf(self.state(), d)
            self.assertIsNotNone(path)
            self.assertTrue(path.name.startswith("latest-ai-agent-development-trends-in-2026_"))
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            self.assertGreater(data.count(b"/Type /Page\n") + data.count(b"/Type /Page "), 2)

    def test_unapproved_and_missing_sections_still_export(self):
        with tempfile.TemporaryDirectory() as d:
            path = export_report_pdf(
                self.state(draft="## Executive Summary\nShort.", approved=False,
                           final_status="max_revisions_reached"), d)
            self.assertIsNotNone(path)
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_empty_draft_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(export_report_pdf(self.state(draft=""), d))
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_formatting_failure_falls_back_to_plain_pdf(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(pdf_report, "_build", side_effect=RuntimeError("boom")):
            path = export_report_pdf(self.state(), d)
            self.assertIsNotNone(path)
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_total_failure_returns_none_without_raising(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(pdf_report, "_build", side_effect=RuntimeError("boom")), \
             patch.object(pdf_report, "_build_plain", side_effect=RuntimeError("boom2")):
            self.assertIsNone(export_report_pdf(self.state(), d))

    def test_malformed_markup_does_not_break_export(self):
        bad = "## Executive Summary\n**bold *italic** <b> & </i> broken `code"
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNotNone(export_report_pdf(self.state(draft=bad), d))


if __name__ == "__main__":
    unittest.main()
