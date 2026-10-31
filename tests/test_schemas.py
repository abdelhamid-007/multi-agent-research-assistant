import unittest

from pydantic import ValidationError

from app.errors import StateValidationError, StructuredOutputError
from app.schemas import (
    AnalysisResult,
    Finding,
    ResearchFindings,
    ReviewFeedback,
    ReviewIssue,
    Source,
    WriterOutput,
    cited_ids,
    require_analysis,
    require_research,
)
from app.structured import extract_json, invoke_structured
from tests.fakes import ANALYSIS_JSON, FakeLLM, RESEARCH_JSON, URL, issue

import json
from unittest.mock import patch


def src(i=1, **kw):
    data = {"id": i, "title": f"Title {i}", "url": f"https://site{i}.com/p"}
    data.update(kw)
    return Source(**data)


def findings(**kw):
    data = dict(summary="s", sources=[src(1), src(2)],
                findings=[Finding(claim="A claim", source_ids=[1])])
    data.update(kw)
    return ResearchFindings(**data)


class SourceTests(unittest.TestCase):
    def test_rejects_bad_urls(self):
        for bad in ("", "not a url", "ftp://x.com/a", "javascript:alert(1)", "https://"):
            with self.assertRaises(ValidationError, msg=bad):
                src(url=bad)

    def test_label_and_domain(self):
        s = src(url="https://www.example.com/x", publisher="ACME", published_date="2026-01-02")
        self.assertEqual(s.domain, "example.com")
        self.assertEqual(s.label(), "Title 1 - ACME, 2026-01-02")
        self.assertEqual(src(url="https://www.example.com/x").label(), "Title 1 - example.com")

    def test_snippet_is_cleaned_and_clipped(self):
        s = src(snippet="a  b\n" + "x" * 2000)
        self.assertLessEqual(len(s.snippet), 700)
        self.assertTrue(s.snippet.startswith("a b "))

    def test_id_must_be_positive(self):
        with self.assertRaises(ValidationError):
            src(0)


class ResearchFindingsTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(sorted(findings().source_map()), [1, 2])

    def test_unknown_source_id_rejected(self):
        with self.assertRaisesRegex(ValidationError, "unknown source ids"):
            findings(findings=[Finding(claim="A claim", source_ids=[9])])

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            findings(sources=[src(1), src(1)])

    def test_requires_sources_and_findings(self):
        with self.assertRaises(ValidationError):
            findings(sources=[])
        with self.assertRaises(ValidationError):
            findings(findings=[])

    def test_finding_needs_a_source(self):
        with self.assertRaises(ValidationError):
            Finding(claim="A claim", source_ids=[])

    def test_to_text_keeps_sources_when_trimmed(self):
        big = findings(findings=[Finding(claim="C" * 3000, source_ids=[1])])
        text = big.to_text(max_chars=500)
        self.assertIn("https://site1.com/p", text)
        self.assertIn("https://site2.com/p", text)
        self.assertLess(len(text), 800)


class AnalysisTests(unittest.TestCase):
    def test_valid_and_text(self):
        a = AnalysisResult.model_validate_json(ANALYSIS_JSON)
        self.assertIn("Risks:", a.to_text())
        self.assertIn("Assumptions", a.to_text())

    def test_empty_lists_rejected(self):
        data = json.loads(ANALYSIS_JSON)
        data["risks"] = ["  ", ""]  # cleaned to empty -> invalid
        with self.assertRaises(ValidationError):
            AnalysisResult(**data)

    def test_bad_complexity_rejected(self):
        data = json.loads(ANALYSIS_JSON)
        data["implementation_complexity"] = "extreme"
        with self.assertRaises(ValidationError):
            AnalysisResult(**data)


class ReviewTests(unittest.TestCase):
    def test_revise_requires_issue(self):
        with self.assertRaisesRegex(ValidationError, "at least one issue"):
            ReviewFeedback(decision="REVISE", score=5)

    def test_approved_cannot_have_major_issue(self):
        with self.assertRaisesRegex(ValidationError, "major"):
            ReviewFeedback(decision="APPROVED", score=9, issues=[ReviewIssue(**issue())])

    def test_approved_needs_minimum_score(self):
        with self.assertRaisesRegex(ValidationError, "score"):
            ReviewFeedback(decision="APPROVED", score=6)

    def test_approved_with_minor_issue_ok(self):
        r = ReviewFeedback(decision="APPROVED", score=8,
                           issues=[ReviewIssue(**issue(severity="minor"))])
        self.assertTrue(r.approved)

    def test_score_range(self):
        for bad in (-1, 11):
            with self.assertRaises(ValidationError):
                ReviewFeedback(decision="APPROVED", score=bad)

    def test_case_insensitive_enums(self):
        r = ReviewFeedback.model_validate({
            "decision": "revise", "score": 4,
            "issues": [{"category": "Citations", "severity": "MAJOR", "problem": "No cites"}],
        })
        self.assertEqual((r.decision, r.issues[0].category, r.issues[0].severity),
                         ("REVISE", "citations", "major"))

    def test_to_text_is_numbered(self):
        r = ReviewFeedback(decision="REVISE", score=4, summary="needs work",
                           issues=[ReviewIssue(**issue("First"), ), ReviewIssue(**issue("Second"))],
                           suggestions=["Be brief"])
        text = r.to_text()
        self.assertIn("DECISION: REVISE", text)
        self.assertIn("SCORE: 4/10", text)
        self.assertIn("1. [COMPLETENESS|major] Risks - First -> Expand it", text)
        self.assertIn("2. [COMPLETENESS|major]", text)
        self.assertIn("- Be brief", text)


class WriterAndHelpersTests(unittest.TestCase):
    def test_writer_output_validation(self):
        with self.assertRaises(ValidationError):
            WriterOutput(title="t", markdown="too short")
        out = WriterOutput(title="t", markdown="word " * 40)
        self.assertEqual(out.word_count, 40)

    def test_cited_ids(self):
        self.assertEqual(cited_ids("a [3] b [1][3] c [12]"), [1, 3, 12])
        self.assertEqual(cited_ids(""), [])


class StateAccessTests(unittest.TestCase):
    def test_missing_field(self):
        with self.assertRaisesRegex(StateValidationError, "research_data"):
            require_research({}, "Analysis")

    def test_accepts_model_and_dict(self):
        model = findings()
        self.assertIs(require_research({"research_data": model}, "X"), model)
        self.assertEqual(
            len(require_research({"research_data": model.model_dump()}, "X").sources), 2)

    def test_invalid_dict(self):
        with self.assertRaisesRegex(StateValidationError, "invalid"):
            require_research({"research_data": {"summary": "x"}}, "Writer")
        with self.assertRaises(StateValidationError):
            require_analysis({"analysis_data": {"risks": []}}, "Writer")


class StructuredOutputTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(json.loads(extract_json('```json\n{"a": 1}\n```')), {"a": 1})
        self.assertEqual(json.loads(extract_json('Sure! {"a": {"b": 2}} done')), {"a": {"b": 2}})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_repairs_invalid_output_once(self):
        llm = FakeLLM(["not json", ANALYSIS_JSON])
        with patch("app.retry._sleep"):
            result = invoke_structured(llm, "PROMPT", AnalysisResult, "Analysis")
        self.assertEqual(result.technical_feasibility, "Feasible with existing frameworks.")
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("rejected", llm.prompts[1])

    def test_raises_after_failed_repair(self):
        llm = FakeLLM(['{"summary": ""}'])
        with self.assertRaises(StructuredOutputError) as ctx:
            invoke_structured(llm, "PROMPT", ResearchFindings, "Research")
        self.assertEqual(ctx.exception.agent, "Research")
        self.assertEqual(len(llm.prompts), 2)

    def test_cross_field_validator_triggers_repair(self):
        calls = []

        def validate(obj):
            calls.append(obj)
            if len(calls) == 1:
                raise ValueError("source ids do not exist")

        llm = FakeLLM([ANALYSIS_JSON])
        invoke_structured(llm, "P", AnalysisResult, "Analysis", validate=validate)
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("source ids do not exist", llm.prompts[1])


if __name__ == "__main__":
    unittest.main()
