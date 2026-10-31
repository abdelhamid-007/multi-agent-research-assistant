import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app import history
from app.runner import (
    AGENTS,
    AgentTracker,
    build_initial_state,
    friendly_error,
    summarize_update,
)
from app.ui import components
from tests.fakes import APPROVED, REVISE, URL, Fakes, RESEARCH_JSON

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class AttrDict(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class RunnerHelperTests(unittest.TestCase):
    def test_initial_state_has_all_keys(self):
        s = build_initial_state("topic", 2)
        self.assertEqual(s["max_revisions"], 2)
        self.assertFalse(s["draft_reviewed"])
        self.assertEqual(s["final_status"], "in_progress")

    def test_tracker_follows_the_workflow(self):
        t = AgentTracker()
        t.start()
        self.assertEqual(t.status["supervisor"][0], "running")
        t.on_event("supervisor", {"next_agent": "research"}, {})
        self.assertEqual(t.status["research"][0], "running")
        self.assertEqual(t.status["supervisor"][0], "done")
        t.on_event("research", {"research": f"x {URL}"}, {})
        self.assertEqual(t.status["research"][0], "done")
        self.assertIn("1 source", t.status["research"][1])
        self.assertEqual(t.status["supervisor"][0], "running")
        t.on_event("supervisor", {"next_agent": "finish", "final_status": "approved"}, {})
        self.assertIsNone(t.current)
        self.assertEqual(sum(s == "running" for s, _ in t.status.values()), 0)

    def test_tracker_failure_marks_current_agent(self):
        t = AgentTracker()
        t.start()
        t.on_event("supervisor", {"next_agent": "writer"}, {"draft": ""})
        t.fail("boom")
        self.assertEqual(t.status["writer"][0], "error")

    def test_reviewer_summary(self):
        legacy = "DECISION: REVISE\nISSUES:\n1. [COMPLETENESS] Risks - add detail -> expand"
        text = summarize_update("reviewer", {"approved": False, "feedback": legacy}, {})
        self.assertIn("REVISE", text)
        self.assertIn("add detail", text)

    def test_structured_summaries(self):
        from app.schemas import ResearchFindings, ReviewFeedback
        from tests.fakes import RESEARCH_JSON, SEARCH_RESULTS, review_json, REVISE
        from app.agents.research_agent import build_sources
        import json

        data = ResearchFindings(sources=build_sources(SEARCH_RESULTS),
                                **json.loads(RESEARCH_JSON))
        self.assertIn("2 source URL(s), 2 finding(s)",
                      summarize_update("research", {"research_data": data}, {}))
        review = ReviewFeedback.model_validate_json(REVISE)
        text = summarize_update("reviewer", {"approved": False, "latest_review": review}, {})
        self.assertIn("score 5/10", text)
        self.assertIn("Add detail", text)

    def test_friendly_errors(self):
        self.assertIn(".env", friendly_error(ValueError("GROQ_API_KEY not found!")))
        self.assertIn("rate limit", friendly_error(RuntimeError("Error code: 429")).lower())
        self.assertIn("failed", friendly_error(KeyError("x")))


class HistoryTests(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "h.json"
            with patch.object(history, "HISTORY_PATH", path):
                self.assertEqual(history.load_history(), [])
                e = history.make_entry({"question": "q", "approved": True, "draft": "d"}, None, 1.234)
                history.add_entry(e)
                self.assertEqual(history.load_history()[0]["topic"], "q")
                path.write_text("{not json", encoding="utf-8")
                self.assertEqual(history.load_history(), [])


class DuplicateSubmissionTests(unittest.TestCase):
    def test_second_click_while_running_is_ignored(self):
        ss = AttrDict(running=True, topic_input="another topic", error="", result="keep")
        with patch.object(components.st, "session_state", ss):
            components.start_clicked()
        self.assertEqual(ss["result"], "keep")
        self.assertEqual(ss.get("active_topic"), None)

    def test_short_topic_rejected(self):
        ss = AttrDict(running=False, topic_input="  hi ", error="")
        with patch.object(components.st, "session_state", ss):
            components.start_clicked()
        self.assertFalse(ss["running"])
        self.assertIn("at least", ss["error"])

    def test_valid_topic_arms_run(self):
        ss = AttrDict(running=False, topic_input=" Solid topic ", error="x", result="old")
        with patch.object(components.st, "session_state", ss):
            components.start_clicked()
        self.assertTrue(ss["running"])
        self.assertEqual(ss["active_topic"], "Solid topic")
        self.assertIsNone(ss["result"])


class StreamlitWorkflowTests(unittest.TestCase):
    """Drive the real Streamlit script with the real graph, fake LLMs."""

    def run_app(self, reviewer_replies, topic="Latest AI agent development trends in 2026",
                writer_reply=None, research_llm=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)

        fakes = Fakes(
            reviewer_replies,
            writer_replies=[writer_reply] if writer_reply else None,
            research_llm=research_llm,
            extra=[
                patch("app.history.HISTORY_PATH", tmp / "history.json"),
                patch("app.reports.pdf_report.DEFAULT_OUTPUT_DIR", tmp / "pdf"),
            ],
        )
        fakes.__enter__()
        self.addCleanup(fakes.__exit__, None, None, None)

        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        at.text_input(key="topic_input").set_value(topic)
        at.form_submit_button[0].click().run()
        return at, tmp

    def test_full_flow_topic_to_approved_report_and_pdf(self):
        at, tmp = self.run_app([REVISE, APPROVED])

        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertFalse(at.session_state["running"])
        result = at.session_state["result"]
        self.assertTrue(result["approved"])
        self.assertEqual(result["revisions"], 1)
        self.assertEqual(result["status"], "approved")

        # every agent finished, none left running or in error
        statuses = {a: s for a, (s, _) in at.session_state["agent_status"].items()}
        self.assertEqual(set(statuses), set(AGENTS))
        self.assertTrue(all(s == "done" for s in statuses.values()), statuses)

        # PDF created in the output folder and offered for download
        pdf = Path(result["pdf_path"])
        self.assertTrue(pdf.exists() and pdf.read_bytes().startswith(b"%PDF"))
        self.assertEqual(len(at.get("download_button")), 1)

        # report preview + metrics shown, history persisted
        self.assertTrue(any("Agent adoption is growing" in m.value for m in at.markdown))
        saved = json.loads((tmp / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["topic"], "Latest AI agent development trends in 2026")
        # controls are enabled again
        self.assertFalse(at.form_submit_button[0].disabled)

    def test_unapproved_report_is_flagged(self):
        at, _ = self.run_app([REVISE])  # reviewer never approves
        self.assertFalse(at.exception)
        result = at.session_state["result"]
        self.assertFalse(result["approved"])
        self.assertEqual(result["status"], "max_revisions_reached")
        self.assertEqual(len(at.warning), 1)

    def test_short_topic_shows_error_and_does_not_run(self):
        at, _ = self.run_app([APPROVED], topic="ai")
        self.assertFalse(at.session_state["running"])
        self.assertIsNone(at.session_state["result"])
        self.assertEqual(len(at.error), 1)

    def test_workflow_failure_is_shown_gracefully(self):
        class Boom:
            def invoke(self, prompt):
                raise RuntimeError("Error code: 429 rate_limit_exceeded")

        at, _ = self.run_app([APPROVED], research_llm=Boom())
        self.assertFalse(at.exception)
        self.assertFalse(at.session_state["running"])
        self.assertIn("rate limit", at.session_state["error"].lower())
        self.assertEqual(at.session_state["agent_status"]["research"][0], "error")
        self.assertFalse(at.form_submit_button[0].disabled)


if __name__ == "__main__":
    unittest.main()
