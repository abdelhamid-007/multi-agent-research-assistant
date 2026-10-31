"""Each agent consumes and produces validated, structured data."""
import json
import unittest
from unittest.mock import patch

from app.agents.analysis_agent import analysis_agent
from app.agents.research_agent import build_sources, research_agent
from app.agents.reviewer import reviewer_agent
from app.agents.writer import assemble_report, strip_sources_section, writer_agent
from app.errors import AgentError, StateValidationError, StructuredOutputError
from app.runner import build_initial_state, friendly_error
from app.schemas import AnalysisResult, ResearchFindings, RevisionRecord
from tests.fakes import (
    ANALYSIS_JSON, APPROVED, REVISE, RESEARCH_JSON, SEARCH_RESULTS, URL, URL2,
    WRITER_BODY, FakeLLM, Fakes, issue, review_json,
)


def state_after_research():
    state = build_initial_state("How are AI agents evolving?")
    with Fakes([APPROVED]):
        state.update(research_agent(state))
    return state


def state_after_analysis():
    state = state_after_research()
    with Fakes([APPROVED]):
        state.update(analysis_agent(state))
    return state


def state_after_writer(body=WRITER_BODY):
    state = state_after_analysis()
    with Fakes([APPROVED], writer_replies=[body]):
        state.update(writer_agent(state))
    return state


class BuildSourcesTests(unittest.TestCase):
    def test_metadata_comes_from_search_results(self):
        sources = build_sources(SEARCH_RESULTS)
        self.assertEqual([s.id for s in sources], [1, 2])
        self.assertEqual(sources[0].published_date, "2026-03-01")
        self.assertEqual(sources[0].title, "Agent trends")
        self.assertIsNone(sources[1].published_date)

    def test_publisher_taken_when_available(self):
        results = [{"url": "https://a.com/x", "title": "T", "content": "c", "publisher": "The Times"}]
        self.assertEqual(build_sources(results)[0].publisher, "The Times")

    def test_skips_duplicates_and_unusable_results(self):
        results = {"results": [
            {"url": URL, "title": "A", "content": "x"},
            {"url": URL, "title": "dup", "content": "x"},
            {"url": "", "title": "no url"},
            {"url": "not-a-url", "title": "bad"},
            "garbage",
            {"url": URL2, "title": "B", "content": "y"},
        ]}
        self.assertEqual([s.url for s in build_sources(results)], [URL, URL2])
        self.assertEqual([s.id for s in build_sources(results)], [1, 2])

    def test_handles_odd_inputs(self):
        for odd in (None, "text", 42, {}, {"results": None}, []):
            self.assertEqual(build_sources(odd), [])

    def test_caps_source_count(self):
        many = [{"url": f"https://s{i}.com/", "title": f"T{i}", "content": "c"} for i in range(30)]
        self.assertEqual(len(build_sources(many)), 8)


class ResearchAgentTests(unittest.TestCase):
    def test_saves_structured_evidence_with_source_metadata(self):
        state = state_after_research()
        data = state["research_data"]
        self.assertIsInstance(data, ResearchFindings)
        self.assertEqual([s.url for s in data.sources], [URL, URL2])
        self.assertEqual(data.sources[0].published_date, "2026-03-01")
        self.assertEqual(len(data.findings), 2)
        # legacy string view is derived and contains the real URLs
        self.assertIn(URL, state["research"])
        self.assertEqual(state["execution_history"], ["research"])

    def test_llm_cannot_add_sources(self):
        llm_reply = json.loads(RESEARCH_JSON)
        llm_reply["sources"] = [{"id": 9, "title": "Invented", "url": "https://invented.example"}]
        state = build_initial_state("q")
        with Fakes([APPROVED], research_llm=FakeLLM([json.dumps(llm_reply)])):
            data = research_agent(state)["research_data"]
        self.assertNotIn("invented", data.to_text())

    def test_unknown_source_id_is_repaired(self):
        bad = json.loads(RESEARCH_JSON)
        bad["findings"][0]["source_ids"] = [7]
        llm = FakeLLM([json.dumps(bad), RESEARCH_JSON])
        with Fakes([APPROVED], research_llm=llm):
            data = research_agent(build_initial_state("q"))["research_data"]
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("unknown source ids", llm.prompts[1])
        self.assertEqual(data.findings[0].source_ids, [1])

    def test_no_sources_is_a_clear_error(self):
        with Fakes([APPROVED], search={"results": []}):
            with self.assertRaisesRegex(AgentError, "no usable sources"):
                research_agent(build_initial_state("q"))

    def test_invalid_llm_output_raises_structured_error(self):
        with Fakes([APPROVED], research_llm=FakeLLM(["I cannot do that"])):
            with self.assertRaises(StructuredOutputError):
                research_agent(build_initial_state("q"))

    def test_empty_question(self):
        with self.assertRaisesRegex(AgentError, "question is empty"):
            research_agent({"question": "  "})


class AnalysisAgentTests(unittest.TestCase):
    def test_receives_structured_research(self):
        state = state_after_research()
        llm = FakeLLM([ANALYSIS_JSON])
        with Fakes([APPROVED], analysis_llm=llm):
            update = analysis_agent(state)
        prompt = llm.prompts[0]
        self.assertIn("Agent adoption is growing", prompt)  # a finding
        self.assertIn(URL, prompt)  # a source
        self.assertIn("[sources: 1;", prompt)
        self.assertIsInstance(update["analysis_data"], AnalysisResult)
        self.assertIn("Governance tooling", update["analysis"])

    def test_requires_research_data(self):
        with self.assertRaises(StateValidationError):
            analysis_agent({"question": "q", "research": "plain text only"})

    def test_invalid_output_raises(self):
        with Fakes([APPROVED], analysis_llm=FakeLLM(['{"risks": []}'])):
            with self.assertRaises(StructuredOutputError):
                analysis_agent(state_after_research())


class WriterAgentTests(unittest.TestCase):
    def test_sources_section_built_from_real_sources_only(self):
        state = state_after_writer()
        out = state["writer_output"]
        self.assertEqual(out.cited_source_ids, [1, 2])
        self.assertEqual(state["draft"], out.markdown)
        self.assertIn(f"- [1] Agent trends - example.com, 2026-03-01 - {URL}", out.markdown)
        self.assertIn(f"- [2] Governance report - example.org - {URL2}", out.markdown)
        self.assertEqual(state["revision_count"], 0)
        self.assertFalse(state["draft_reviewed"])
        self.assertEqual(len(state["revision_history"]), 1)
        self.assertIsNone(state["revision_history"][0].review)

    def test_llm_written_sources_section_is_replaced(self):
        body = WRITER_BODY + "\n\n## Sources\n1. https://invented.example/fake"
        state = state_after_writer(body)
        self.assertNotIn("invented.example", state["draft"])
        self.assertEqual(state["draft"].count("## Sources"), 1)

    def test_only_cited_sources_are_listed(self):
        body = WRITER_BODY.replace("[2]", "")
        state = state_after_writer(body)
        self.assertEqual(state["writer_output"].cited_source_ids, [1])
        self.assertNotIn(URL2, state["draft"])

    def test_unknown_citation_gets_one_correction(self):
        state = state_after_analysis()
        wrong = WRITER_BODY + " Extra claim [9]."
        llm_replies = [wrong, WRITER_BODY]
        with Fakes([APPROVED], writer_replies=llm_replies) as fakes:
            update = writer_agent(state)
        self.assertEqual(len(fakes.writer.prompts), 2)
        self.assertIn("unknown source numbers [9]", fakes.writer.prompts[1])
        self.assertEqual(update["writer_output"].warnings, [])

    def test_persistent_unknown_citation_is_stripped_with_warning(self):
        state = state_after_analysis()
        wrong = WRITER_BODY + " Extra claim [9]."
        with Fakes([APPROVED], writer_replies=[wrong]):
            out = writer_agent(state)["writer_output"]
        self.assertNotIn("[9]", out.markdown)
        self.assertTrue(any("unknown source" in w for w in out.warnings))

    def test_no_citations_is_warned_not_crashed(self):
        state = state_after_analysis()
        uncited = "## Executive Summary\n" + "Plain claim without any citation. " * 5
        with Fakes([APPROVED], writer_replies=[uncited]):
            out = writer_agent(state)["writer_output"]
        self.assertIn("The report cites no sources.", out.warnings)
        self.assertIn("No sources were cited.", out.markdown)

    def test_unusable_output_raises(self):
        with Fakes([APPROVED], writer_replies=["ok [1]"]):
            with self.assertRaises(StructuredOutputError):
                writer_agent(state_after_analysis())

    def test_requires_structured_inputs(self):
        with self.assertRaises(StateValidationError):
            writer_agent({"question": "q", "research": "x", "analysis": "y"})
        with self.assertRaises(StateValidationError):
            writer_agent({k: v for k, v in state_after_analysis().items() if k != "analysis_data"})

    def test_revision_receives_feedback_and_keeps_history(self):
        state = state_after_writer()
        state["feedback"] = "DECISION: REVISE\n1. [ACCURACY|major] Fix claim X"
        state["review_history"] = [state["feedback"]]
        with Fakes([APPROVED]) as fakes:
            update = writer_agent(state)
        self.assertIn("Fix claim X", fakes.writer.prompts[0])
        self.assertEqual(update["revision_count"], 1)
        self.assertEqual([r.revision for r in update["revision_history"]], [0, 1])

    def test_strip_and_assemble_helpers(self):
        self.assertEqual(strip_sources_section("Body\n\n**Sources**\n- x"), "Body")
        self.assertEqual(strip_sources_section("No such section"), "No such section")
        md, used, warnings = assemble_report("Fact [1].", {1: build_sources(SEARCH_RESULTS)[0]})
        self.assertEqual((used, warnings), ([1], []))
        self.assertTrue(md.endswith(f"- [1] Agent trends - example.com, 2026-03-01 - {URL}"))


class ReviewerAgentTests(unittest.TestCase):
    def review(self, state, *replies):
        with Fakes(list(replies)) as fakes:
            return reviewer_agent(state), fakes

    def test_structured_approval(self):
        update, _ = self.review(state_after_writer(), APPROVED)
        self.assertTrue(update["approved"])
        self.assertEqual(update["latest_review"].score, 9)
        self.assertEqual(update["final_status"], "approved")
        self.assertTrue(update["draft_reviewed"])
        self.assertIn("DECISION: APPROVED", update["feedback"])
        self.assertEqual(update["review_history"], [update["feedback"]])

    def test_revise_with_issues_and_suggestions(self):
        reply = json.dumps({"decision": "REVISE", "score": 4, "summary": "weak",
                            "issues": [issue("Claim unsupported", "accuracy")],
                            "suggestions": ["Cut the fluff"]})
        update, _ = self.review(state_after_writer(), reply)
        review = update["latest_review"]
        self.assertFalse(update["approved"])
        self.assertEqual(update["final_status"], "revision_required")
        self.assertEqual(review.issues[0].category, "accuracy")
        self.assertEqual(review.suggestions, ["Cut the fluff"])
        self.assertIn("1. [ACCURACY|major]", update["feedback"])

    def test_review_is_attached_to_the_draft_it_judged(self):
        update, _ = self.review(state_after_writer(), REVISE)
        record = update["revision_history"][-1]
        self.assertIsInstance(record, RevisionRecord)
        self.assertEqual(record.review.decision, "REVISE")
        self.assertEqual(len(update["revision_history"]), 1)

    def test_invented_url_overrides_llm_approval(self):
        state = state_after_writer(WRITER_BODY + "\nSee https://invented.example/x [1].")
        # the writer keeps URLs the model wrote in the body; the reviewer must catch it
        update, _ = self.review(state, APPROVED)
        review = update["latest_review"]
        self.assertFalse(update["approved"])
        self.assertEqual(review.issues[0].category, "citations")
        self.assertIn("invented.example", review.issues[0].problem)
        self.assertLess(review.score, 7)

    def test_unknown_citation_marker_overrides_approval(self):
        state = state_after_writer()
        state["draft"] += "\nExtra [8]."
        update, _ = self.review(state, APPROVED)
        self.assertFalse(update["approved"])
        self.assertIn("[8]", update["feedback"])

    def test_inconsistent_llm_decision_is_repaired(self):
        bad = review_json("APPROVED", 4)  # approved with a failing score
        update, fakes = self.review(state_after_writer(), bad, APPROVED)
        self.assertEqual(len(fakes.reviewer.prompts), 2)
        self.assertIn("score of at least", fakes.reviewer.prompts[1])
        self.assertTrue(update["approved"])

    def test_unparseable_reply_raises_clear_error(self):
        with Fakes(["Looks good to me!"]):
            with self.assertRaisesRegex(StructuredOutputError, "Reviewer"):
                reviewer_agent(state_after_writer())

    def test_requires_draft_and_research(self):
        state = state_after_writer()
        state["draft"] = ""
        with self.assertRaisesRegex(StateValidationError, "no draft"):
            reviewer_agent(state)
        with self.assertRaises(StateValidationError):
            reviewer_agent({"question": "q", "draft": "text"})

    def test_prompt_includes_previous_feedback_and_evidence(self):
        state = state_after_writer()
        state["feedback"] = "PREVIOUS: add detail"
        _, fakes = self.review(state, APPROVED)
        prompt = fakes.reviewer.prompts[0]
        self.assertIn("PREVIOUS: add detail", prompt)
        self.assertIn(URL, prompt)


class ErrorMessageTests(unittest.TestCase):
    def test_agent_errors_are_shown_as_is(self):
        msg = friendly_error(AgentError("Research", "the web search returned no usable sources"))
        self.assertIn("Research: the web search returned no usable sources", msg)


if __name__ == "__main__":
    unittest.main()
