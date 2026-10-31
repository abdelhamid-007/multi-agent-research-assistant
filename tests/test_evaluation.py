"""Automated evaluation: scoring, validation, critical failures, Reviewer integration."""
import json
import unittest

from pydantic import ValidationError

from app.agents.reviewer import combine_reviews, review_from_evaluation, reviewer_agent
from app.errors import StateValidationError
from app.evaluation import evaluate_report
from app.evaluation.checks import classify_sentence
from app.evaluation.evidence import EvidenceIndex, numbers, overlap, stem, tokens
from app.evaluation.report_parser import canonical_section, parse_draft
from app.evaluation.scoring import MIN_DIMENSION_SCORE, PASS_SCORE, WEIGHTS
from app.history import _evaluation_dict
from app.runner import build_initial_state, summarize_update
from app.schemas import (
    DIMENSIONS, DimensionScore, EvalFlag, EvaluationResult, Finding, ResearchFindings,
    ResearchTask, ReviewFeedback, Source, TaskReport,
)
from app.state_validation import validate_state
from tests.eval_fixtures import (
    ANALYSIS, GOOD_BODY, RESEARCH, SOURCES, fabricated_report, report, without_citations,
)
from tests.fakes import APPROVED, REVISE, WRITER_BODY, FakeLLM, Fakes, review_json


def evaluate(draft, research=RESEARCH, analysis=ANALYSIS, revision=0):
    return evaluate_report(draft, research, analysis, revision)


def codes(evaluation, severity=None):
    return {f.code for f in evaluation.flags if severity in (None, f.severity)}


class HighQualityReportTests(unittest.TestCase):
    def test_high_quality_report_passes_with_a_high_score(self):
        ev = evaluate(report())
        self.assertTrue(ev.passed)
        self.assertEqual(ev.blocking_reasons, [])
        self.assertGreaterEqual(ev.overall_score, 95)
        self.assertEqual(ev.critical_failures, [])
        for d in ev.dimensions:
            self.assertGreaterEqual(d.score, 90, d.name)

    def test_all_five_dimensions_have_0_to_100_scores_and_the_required_weights(self):
        ev = evaluate(report())
        self.assertEqual([d.name for d in ev.dimensions], list(DIMENSIONS))
        self.assertEqual({d.name: d.weight for d in ev.dimensions}, {
            "factual_accuracy": 0.30, "citation_reliability": 0.25, "research_coverage": 0.20,
            "logical_structure": 0.15, "recommendations": 0.10})
        self.assertTrue(all(0 <= d.score <= 100 for d in ev.dimensions))

    def test_overall_is_the_weighted_sum(self):
        ev = evaluate(report(without_citations()))
        expected = sum(d.score * d.weight for d in ev.dimensions)
        self.assertAlmostEqual(ev.overall_score, expected, delta=0.06)

    def test_summary_is_clear_and_does_not_claim_truth(self):
        ev = evaluate(report())
        self.assertIn("meets the approval criteria", ev.summary)
        self.assertIn("Strongest:", ev.summary)
        self.assertIn("do not prove the claims are true", ev.summary)
        self.assertIn("not proof that claims are true", ev.to_text())
        self.assertEqual(ev.method, "deterministic evidence-support checks")


class MissingCitationsTests(unittest.TestCase):
    def setUp(self):
        self.ev = evaluate(report(without_citations()))

    def test_report_without_citations_cannot_pass(self):
        self.assertFalse(self.ev.passed)
        self.assertIn("no_citations", codes(self.ev, "major"))
        self.assertTrue(any("no citations" in r for r in self.ev.blocking_reasons))

    def test_citation_score_is_low_and_statistics_are_flagged_uncited(self):
        self.assertLess(self.ev.dimension("citation_reliability").score, 50)
        self.assertIn("uncited_statistic", codes(self.ev))
        self.assertLess(self.ev.overall_score, PASS_SCORE)

    def test_it_is_not_a_critical_failure_but_still_blocks(self):
        self.assertEqual(self.ev.critical_failures, [])
        self.assertTrue(any(f.blocking for f in self.ev.flags))

    def test_partially_cited_report_is_penalised_but_not_blocked_for_citations_alone(self):
        partial = GOOD_BODY.replace("Open standards reduce integration effort [3].",
                                    "Open standards reduce integration effort.")
        ev = evaluate(report(partial))
        self.assertIn("missing_citations", codes(ev))
        self.assertLess(ev.dimension("citation_reliability").score, 100)
        self.assertNotIn("no_citations", codes(ev))


class FabricatedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.ev = evaluate(fabricated_report())

    def test_fabricated_citation_url_and_misattribution_are_critical(self):
        self.assertEqual(
            codes(self.ev, "critical"),
            {"fabricated_citation", "fabricated_reference", "misattributed_reference"})
        messages = " ".join(f.message for f in self.ev.critical_failures)
        self.assertIn("[9]", messages)
        self.assertIn("made-up.example", messages)
        self.assertIn("Reference [2] lists the URL of source [1]", messages)

    def test_fabrication_prevents_approval_even_with_a_high_overall_score(self):
        self.assertGreaterEqual(self.ev.overall_score, PASS_SCORE)  # the score alone would pass
        self.assertFalse(self.ev.passed)
        self.assertTrue(any("critical failure" in r for r in self.ev.blocking_reasons))

    def test_citation_reliability_drops(self):
        self.assertLess(self.ev.dimension("citation_reliability").score, 60)

    def test_unverified_source_cited_is_flagged(self):
        sources = [s.model_copy(update={"verified": False}) if s.id == 1 else s for s in SOURCES]
        research = RESEARCH.model_copy(update={"sources": sources})
        self.assertIn("unverified_source_cited", codes(evaluate(report(), research)))

    def test_url_variants_of_a_real_source_are_not_fabricated(self):
        draft = report() + "\nMore: https://www.example.com/agents/?utm_source=x."
        self.assertNotIn("fabricated_reference", codes(evaluate(draft)))


class UnsupportedClaimTests(unittest.TestCase):
    def test_invented_statistic_is_a_critical_failure_that_blocks_approval(self):
        draft = report(GOOD_BODY.replace("grew 45% in 2025 [1]. Orchestration", "grew 73% in 2025 [1]. Orchestration"))
        ev = evaluate(draft)
        self.assertEqual(codes(ev, "critical"), {"unsupported_critical_claim"})
        self.assertIn("73", ev.critical_failures[0].message)
        self.assertIn("grew 73%", ev.critical_failures[0].evidence)
        self.assertGreaterEqual(ev.overall_score, PASS_SCORE)
        self.assertFalse(ev.passed)

    def test_uncited_invented_statistic_is_also_critical(self):
        ev = evaluate(report(GOOD_BODY + " Budgets for agents rose by 300% last year."))
        self.assertIn("unsupported_critical_claim", codes(ev, "critical"))

    def test_statistic_from_another_source_is_miscited_and_blocks(self):
        ev = evaluate(report(GOOD_BODY.replace("added audit logging [2]", "added audit logging [1]")))
        self.assertEqual(codes(ev, "major") & {"miscited_statistic"}, {"miscited_statistic"})
        self.assertFalse(ev.passed)
        self.assertEqual(ev.critical_failures, [])

    def test_qualitative_claim_with_no_support_is_flagged(self):
        draft = report(GOOD_BODY.replace(
            "## Business Opportunities",
            "Quantum networking will replace every datacenter protocol soon.\n\n## Business Opportunities"))
        ev = evaluate(draft)
        self.assertIn("unsupported_claim", codes(ev, "major"))
        self.assertLess(ev.dimension("factual_accuracy").score, 100)

    def test_numbers_are_checked_against_source_text_not_llm_findings(self):
        finding = Finding(claim="Agent budgets rose 300% last year", evidence="300% rise", source_ids=[1])
        research = RESEARCH.model_copy(update={"findings": RESEARCH.findings + [finding]})
        ev = evaluate(report(GOOD_BODY + " Agent budgets rose 300% last year [1]."), research)
        self.assertIn("unsupported_critical_claim", codes(ev, "critical"))  # not in any source snippet


class IncompleteReportTests(unittest.TestCase):
    DRAFT = ("## Executive Summary\nAdoption of AI agents is growing quickly across industries today [1].\n\n"
             "## Conclusion\nGovernance is a priority for most buyers and teams [2].")

    def test_incomplete_report_is_blocked_and_scored_low(self):
        ev = evaluate(report(self.DRAFT))
        self.assertFalse(ev.passed)
        self.assertIn("incomplete_report", codes(ev, "major"))
        incomplete = next(f for f in ev.flags if f.code == "incomplete_report")
        self.assertTrue(incomplete.blocking)
        for section in ("Research Findings", "Business Opportunities", "Technical Analysis", "Risks", "Recommendations"):
            self.assertIn(section, incomplete.message)
        self.assertLess(ev.dimension("logical_structure").score, 60)
        self.assertEqual(ev.dimension("recommendations").score, 0)

    def test_incomplete_report_also_loses_research_coverage(self):
        ev = evaluate(report(self.DRAFT))
        self.assertLess(ev.dimension("research_coverage").score, MIN_DIMENSION_SCORE)
        self.assertTrue(any("below 40" in r for r in ev.blocking_reasons))

    def test_empty_draft_does_not_crash(self):
        ev = evaluate("")
        self.assertFalse(ev.passed)
        self.assertIn("no_verifiable_claims", codes(ev))

    def test_short_report_and_thin_sections_are_flagged(self):
        ev = evaluate(report("## Executive Summary\nShort.\n\n## Conclusion\nDone."))
        self.assertIn("report_too_short", codes(ev))
        self.assertIn("thin_section", codes(ev))

    def test_sections_out_of_order_is_a_minor_flag(self):
        body = GOOD_BODY.replace("## Conclusion", "## Zzz").split("## Zzz")
        reordered = "## Conclusion\n" + body[1].strip() + "\n\n" + body[0]
        ev = evaluate(report(reordered))
        self.assertIn("sections_out_of_order", codes(ev, "minor"))


class CriticalFailuresPreventApprovalTests(unittest.TestCase):
    def dimensions(self, score=100.0):
        return [DimensionScore(name=n, score=score, weight=w) for n, w in WEIGHTS.items()]

    def test_critical_flags_always_block(self):
        flag = EvalFlag(code="x", severity="critical", dimension="factual_accuracy", message="m")
        self.assertTrue(flag.blocking)

    def test_a_passing_evaluation_cannot_contain_a_critical_failure(self):
        flag = EvalFlag(code="fabricated_reference", severity="critical",
                        dimension="citation_reliability", message="fake url")
        with self.assertRaisesRegex(ValidationError, "cannot pass"):
            EvaluationResult(dimensions=self.dimensions(), overall_score=100, flags=[flag], passed=True)

    def test_a_passing_evaluation_cannot_have_blocking_reasons(self):
        with self.assertRaisesRegex(ValidationError, "cannot pass"):
            EvaluationResult(dimensions=self.dimensions(), overall_score=100, passed=True,
                             blocking_reasons=["something"])

    def test_a_failed_evaluation_must_say_why(self):
        with self.assertRaisesRegex(ValidationError, "blocking reasons"):
            EvaluationResult(dimensions=self.dimensions(), overall_score=100, passed=False)

    def test_overall_must_match_the_weighted_scores(self):
        with self.assertRaisesRegex(ValidationError, "does not match"):
            EvaluationResult(dimensions=self.dimensions(80), overall_score=50, passed=False,
                             blocking_reasons=["x"])

    def test_dimensions_must_be_exactly_the_five(self):
        with self.assertRaisesRegex(ValidationError, "dimensions must be exactly"):
            EvaluationResult(dimensions=self.dimensions()[:4], overall_score=80, passed=False,
                             blocking_reasons=["x"])

    def test_each_critical_scenario_yields_a_failed_evaluation(self):
        scenarios = {
            "fabricated": fabricated_report(),
            "invented statistic": report(GOOD_BODY.replace("grew 45%", "grew 99%")),
        }
        for name, draft in scenarios.items():
            ev = evaluate(draft)
            self.assertTrue(ev.critical_failures, name)
            self.assertFalse(ev.passed, name)

    def test_combine_reviews_never_approves_a_failed_evaluation(self):
        failed = evaluate(fabricated_report())
        llm_says_approved = ReviewFeedback.model_validate_json(APPROVED)
        combined = combine_reviews(llm_says_approved, failed)
        self.assertEqual(combined.decision, "REVISE")
        self.assertFalse(combined.approved)

    def test_review_from_evaluation_lists_the_critical_problems_first(self):
        review = review_from_evaluation(evaluate(fabricated_report()))
        self.assertEqual(review.decision, "REVISE")
        self.assertEqual(review.issues[0].category, "citations")
        self.assertIn("[9]", " ".join(i.problem for i in review.issues))
        self.assertLess(review.score, 7)

    def test_review_from_evaluation_handles_failure_by_score_alone(self):
        ev = evaluate(report(without_citations()))
        review = review_from_evaluation(ev)
        self.assertTrue(review.issues)


class ReviewerIntegrationTests(unittest.TestCase):
    @staticmethod
    def state_with(draft, **extra):
        state = build_initial_state("How are AI agents evolving?")
        state.update(research_data=RESEARCH, analysis_data=ANALYSIS, draft=draft, **extra)
        return state

    def review(self, draft, *replies, **extra):
        with Fakes(list(replies) or [APPROVED]) as fakes:
            return reviewer_agent(self.state_with(draft, **extra)), fakes

    def test_passing_report_goes_to_the_llm_and_can_be_approved(self):
        update, fakes = self.review(report(), review_json("APPROVED", 9))
        self.assertTrue(update["approved"])
        self.assertEqual(len(fakes.reviewer.prompts), 1)
        self.assertIn("Automated checks (deterministic, already PASSED", fakes.reviewer.prompts[0])
        self.assertTrue(update["latest_evaluation"].passed)

    def test_the_llm_score_is_capped_by_the_evaluation(self):
        update, _ = self.review(report(without_citations().replace("Prefer", "Prefer")), review_json("APPROVED", 10))
        evaluation = update["latest_evaluation"]
        self.assertLessEqual(update["latest_review"].score, int(evaluation.overall_score // 10))

    def test_llm_can_still_reject_a_report_that_passes_the_checks(self):
        update, _ = self.review(report(), REVISE)
        self.assertFalse(update["approved"])
        self.assertTrue(update["latest_evaluation"].passed)
        self.assertEqual(update["final_status"], "revision_required")

    def test_failed_evaluation_skips_the_llm_and_overrides_an_approval(self):
        update, fakes = self.review(fabricated_report(), APPROVED)
        self.assertFalse(update["approved"])
        self.assertEqual(fakes.reviewer.prompts, [])
        self.assertEqual(update["latest_review"].decision, "REVISE")
        self.assertTrue(update["latest_evaluation"].critical_failures)
        self.assertIn("EVALUATION:", update["feedback"])
        self.assertIn("[9]", update["feedback"])

    def test_evaluation_is_stored_in_state_and_linked_to_the_draft(self):
        update, _ = self.review(report(), APPROVED)
        self.assertEqual(len(update["evaluation_history"]), 1)
        record = update["revision_history"][-1]
        self.assertIs(record.evaluation, update["latest_evaluation"])
        self.assertEqual(record.review, update["latest_review"])

    def test_evaluation_history_accumulates_across_reviews(self):
        first, _ = self.review(report(without_citations()))
        second, _ = self.review(report(), APPROVED, **{
            "evaluation_history": first["evaluation_history"], "review_history": first["review_history"],
            "revision_history": first["revision_history"], "feedback": first["feedback"], "revision_count": 1})
        history = second["evaluation_history"]
        self.assertEqual([e.revision for e in history], [0, 1])
        self.assertFalse(history[0].passed)
        self.assertTrue(history[1].passed)
        self.assertGreater(history[1].overall_score, history[0].overall_score)

    def test_feedback_for_the_writer_contains_the_evaluation_summary(self):
        update, _ = self.review(report(without_citations()))
        self.assertIn("EVALUATION:", update["feedback"])
        self.assertIn("NOT APPROVABLE", update["feedback"])
        self.assertIn("Citation reliability", update["feedback"])

    def test_requires_research_and_draft(self):
        with self.assertRaises(StateValidationError):
            reviewer_agent({"question": "q", "draft": "x"})
        with self.assertRaisesRegex(StateValidationError, "no draft"):
            reviewer_agent(self.state_with(""))


class ImprovementAcrossRevisionsTests(unittest.TestCase):
    def test_writer_gets_the_evaluation_and_the_score_improves(self):
        from app.workflows.graph import research_graph

        bad = WRITER_BODY + " Adoption grew 73% last year [1]."   # a figure that is in no source
        with Fakes([APPROVED], writer_replies=[bad, WRITER_BODY]) as fakes:
            result = research_graph.invoke(build_initial_state("q about AI agents"),
                                           config={"recursion_limit": 40})
        self.assertIn("EVALUATION:", fakes.writer.prompts[1])
        self.assertIn("appear in none of the retrieved sources", fakes.writer.prompts[1])
        scores = [e.overall_score for e in result["evaluation_history"]]
        self.assertEqual(len(scores), 2)
        self.assertFalse(result["evaluation_history"][0].passed)
        self.assertTrue(result["approved"])
        self.assertEqual(validate_state(result), [])


class StateValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.workflows.graph import research_graph
        with Fakes([REVISE, APPROVED]):
            cls.good = research_graph.invoke(build_initial_state("q about AI agents"),
                                             config={"recursion_limit": 40})

    def broken(self, **changes):
        return validate_state({**self.good, **changes})

    def test_a_real_run_is_consistent(self):
        self.assertEqual(validate_state(self.good), [])
        self.assertEqual(len(self.good["evaluation_history"]), 2)

    def test_approval_requires_a_passing_evaluation(self):
        failed = evaluate(fabricated_report())
        history = self.good["evaluation_history"][:-1] + [failed]
        problems = self.broken(latest_evaluation=failed, evaluation_history=history)
        self.assertTrue(any("evaluation did not pass" in p for p in problems))
        self.assertTrue(any("evaluation did not pass" in p for p in self.broken(latest_evaluation=None)))

    def test_evaluation_history_rules(self):
        first, last = self.good["evaluation_history"]
        self.assertTrue(any("not strictly increasing" in p for p in self.broken(evaluation_history=[last, first])))
        self.assertTrue(any("entries in 'review_history'" in p for p in self.broken(evaluation_history=[last])))
        self.assertTrue(any("not the last entry" in p for p in self.broken(latest_evaluation=first)))
        self.assertTrue(any("'evaluation_history[0]'" in p for p in self.broken(evaluation_history=[{"x": 1}, last])))


class ParserAndHelperTests(unittest.TestCase):
    def test_section_aliases(self):
        self.assertEqual(canonical_section("## 4. Technical Analysis:"), "Technical Analysis")
        self.assertEqual(canonical_section("References"), "Sources")
        self.assertIsNone(canonical_section("Random Title"))

    def test_parse_draft(self):
        parsed = parse_draft(report())
        self.assertEqual(parsed.present[0], "Executive Summary")
        self.assertEqual(parsed.present[-1], "Sources")
        self.assertEqual({1, 2, 3}, parsed.marker_ids)
        self.assertEqual([n for n, _ in parsed.source_entries], [1, 2, 3])
        self.assertEqual(len(parsed.items["Recommendations"]), 3)

    def test_unknown_subheadings_keep_the_current_section(self):
        parsed = parse_draft("## Research Findings\n### Detail\nAgents are growing fast today [1].\n")
        self.assertEqual(parsed.sentences[0].section, "Research Findings")

    def test_numbers_and_statistics(self):
        text = "Up 1,445% in 2026, $3.5 billion, 5 steps, 12 agents, 7.50x, [3]"
        self.assertEqual(numbers(text, statistics_only=True), ["1445", "3.5", "12", "7.5"])
        self.assertIn("2026", numbers(text))
        self.assertNotIn("3", numbers(text))  # citation markers are ignored

    def test_stemming_and_overlap(self):
        self.assertEqual(stem("agents"), stem("agent"))
        self.assertEqual(stem("adopted"), stem("adoption"))
        self.assertEqual(overlap(tokens("agents adopted widely"), tokens("widespread adoption of agent")), 2 / 3)
        self.assertEqual(overlap(set(), {"a"}), 1.0)

    def test_sentence_classification(self):
        ev = EvidenceIndex(RESEARCH)
        sentences = {s.text: s for s in parse_draft(report()).sentences}
        grew = sentences["Enterprise adoption of AI agents grew 45% in 2025."]
        self.assertEqual(classify_sentence(grew, ev)[0], "supported")
        self.assertEqual(classify_sentence(type(grew)("Adoption grew 45% in 2025 overall.", [], None), ev)[0], "uncited_stat")
        self.assertEqual(classify_sentence(type(grew)("Adoption grew 45% in 2025 overall.", [2], None), ev)[0], "miscited")
        self.assertEqual(classify_sentence(type(grew)("Adoption grew 88% in 2025 overall.", [1], None), ev)[0],
                         "unsupported_critical")
        self.assertEqual(classify_sentence(type(grew)("Mars colonies will host datacenters soon.", [1], None), ev)[0], "weak")
        self.assertEqual(classify_sentence(type(grew)("Mars colonies will host datacenters soon.", [], None), ev)[0],
                         "unsupported")
        self.assertEqual(classify_sentence(type(grew)("Mars colonies are fake.", [9], None), ev)[0], "unsupported")


class CoverageAndRecommendationTests(unittest.TestCase):
    def test_report_ignoring_the_research_has_low_coverage(self):
        draft = report("""## Executive Summary
Quantum networking will transform telecommunications in unexpected ways for operators and users alike [1].

## Research Findings
Photonic switching may become common in metropolitan backbones over the coming decade [1].

## Business Opportunities
Fiber vendors could package photonic switches for carriers that modernise their networks [1].

## Technical Analysis
Integration requires careful calibration of optical components and software control planes today.

## Risks
Supply chains for photonic components remain narrow, concentrated and slow to scale for vendors.

## Recommendations
- Pilot photonic switching in one metropolitan backbone segment.
- Track component suppliers every quarter for pricing changes.
- Review telecom standards for photonic interoperability annually.

## Conclusion
Photonic networking is promising but still early for most operators and their suppliers.""")
        ev = evaluate(draft)
        self.assertLess(ev.dimension("research_coverage").score, 70)
        self.assertIn("finding_not_covered", codes(ev))

    def test_failed_research_task_is_noted_but_does_not_penalise_the_report(self):
        ok = TaskReport(task_id=1, aspect="Adoption", query="q1", status="ok", n_sources=2, n_findings=2)
        ok2 = TaskReport(task_id=2, aspect="Governance", query="q2", status="ok", n_sources=1, n_findings=1)
        bad = TaskReport(task_id=3, aspect="Costs", query="q3", status="failed", error="down")
        research = RESEARCH.model_copy(update={
            "task_reports": [ok, ok2, bad],
            "sources": [s.model_copy(update={"found_by": [1] if s.id == 1 else [2]}) for s in SOURCES]})
        ev = evaluate(report(), research)
        self.assertIn("research_gap", codes(ev, "minor"))
        self.assertGreaterEqual(ev.dimension("research_coverage").score, 95)
        self.assertTrue(ev.passed)

    def test_uncovered_research_aspect_is_flagged(self):
        reports = [TaskReport(task_id=i, aspect=a, query=f"q{i}", status="ok")
                   for i, a in ((1, "Adoption"), (2, "Quantum networking"))]
        sources = [s.model_copy(update={"found_by": [1]}) for s in SOURCES]
        sources[0] = sources[0].model_copy(update={"found_by": [1]})
        research = RESEARCH.model_copy(update={"task_reports": reports, "sources": sources})
        ev = evaluate(report(), research)
        self.assertIn("aspect_not_covered", codes(ev))

    def test_missing_recommendations_section_scores_zero(self):
        draft = report(GOOD_BODY.split("## Recommendations")[0] + "## Conclusion\nAdoption is growing quickly [1].")
        ev = evaluate(draft)
        self.assertEqual(ev.dimension("recommendations").score, 0)
        self.assertIn("no_recommendations", codes(ev))

    def test_vague_or_few_recommendations_score_lower(self):
        vague = GOOD_BODY.replace(
            "- Pilot a governed multi-agent workflow using existing frameworks.\n"
            "- Add audit logging before scaling agent usage.\n"
            "- Prefer open standards to limit vendor lock-in.",
            "- More thinking about everything is nice to have.\n- Colorful dashboards everywhere.")
        ev = evaluate(report(vague))
        self.assertLess(ev.dimension("recommendations").score, 60)
        self.assertEqual({"few_recommendations", "vague_recommendations"} & codes(ev),
                         {"few_recommendations", "vague_recommendations"})


class ReportingTests(unittest.TestCase):
    def test_runner_summary_mentions_the_evaluation_score(self):
        review = ReviewFeedback.model_validate_json(REVISE)
        text = summarize_update("reviewer", {"approved": False, "latest_review": review,
                                             "latest_evaluation": evaluate(report())}, {})
        self.assertIn("evaluation 100/100", text)

    def test_history_entry_is_json_serialisable(self):
        entry = _evaluation_dict(evaluate(fabricated_report()))
        self.assertFalse(entry["passed"])
        self.assertEqual(len(entry["dimensions"]), 5)
        self.assertTrue(any(f["severity"] == "critical" for f in entry["flags"]))
        json.dumps(entry)
        self.assertIsNone(_evaluation_dict(None))


if __name__ == "__main__":
    unittest.main()
