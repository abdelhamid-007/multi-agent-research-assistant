import unittest
from unittest.mock import patch

from app.errors import StateValidationError
from app.runner import build_initial_state, stream_research
from app.schemas import ReviewFeedback, RevisionRecord
from app.state_validation import ensure_valid_state, validate_state
from tests.fakes import APPROVED, REVISE, Fakes, review_json


def run_graph(reviewer_replies):
    from app.workflows.graph import research_graph

    with Fakes(reviewer_replies):
        return research_graph.invoke(
            build_initial_state("q about AI agents"), config={"recursion_limit": 40})


class ValidStateTests(unittest.TestCase):
    def test_initial_state_is_valid(self):
        self.assertEqual(validate_state(build_initial_state("q")), [])

    def test_approved_run_is_valid(self):
        self.assertEqual(validate_state(run_graph([REVISE, APPROVED])), [])

    def test_unapproved_run_is_valid(self):
        result = run_graph([REVISE])
        self.assertEqual(result["final_status"], "max_revisions_reached")
        self.assertEqual(validate_state(result), [])

    def test_every_intermediate_state_is_valid(self):
        with Fakes([REVISE, APPROVED]):
            for node, _, state in stream_research("q about AI agents"):
                self.assertEqual(validate_state(state), [], f"after {node}")

    def test_dict_form_models_are_accepted(self):
        state = dict(run_graph([APPROVED]))
        for key in ("research_data", "analysis_data", "writer_output", "latest_review"):
            state[key] = state[key].model_dump()
        state["revision_history"] = [r.model_dump() for r in state["revision_history"]]
        self.assertEqual(validate_state(state), [])


class InvalidStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good = run_graph([REVISE, APPROVED])

    def broken(self, **changes):
        state = dict(self.good)
        state.update(changes)
        return validate_state(state)

    def assertProblem(self, problems, text):
        self.assertTrue(any(text in p for p in problems), f"{text!r} not in {problems}")

    def test_invalid_structured_fields(self):
        self.assertProblem(self.broken(research_data={"summary": "x"}),
                           "'research_data' is not a valid ResearchFindings")
        self.assertProblem(self.broken(latest_review={"decision": "MAYBE", "score": 5}),
                           "'latest_review' is not a valid ReviewFeedback")

    def test_text_without_structured_data(self):
        self.assertProblem(self.broken(research_data=None), "'research' text exists")
        self.assertProblem(self.broken(analysis_data=None), "'analysis' text exists")
        self.assertProblem(self.broken(writer_output=None), "'writer_output' is missing")

    def test_draft_mismatch(self):
        self.assertProblem(self.broken(draft=self.good["draft"] + " edited"),
                           "differs from 'writer_output.markdown'")

    def test_approved_without_approving_review(self):
        revise = ReviewFeedback.model_validate_json(REVISE)
        self.assertProblem(self.broken(latest_review=revise), "latest review is not an approval")
        self.assertProblem(self.broken(latest_review=None), "latest review is not an approval")
        self.assertProblem(self.broken(draft_reviewed=False), "was not reviewed")

    def test_status_contradictions(self):
        self.assertProblem(self.broken(final_status="bogus"), "unknown value 'bogus'")
        self.assertProblem(self.broken(final_status="revision_required"), "'final_status' is")
        self.assertProblem(self.broken(approved=False), "'final_status' is 'approved' but")

    def test_bad_next_agent(self):
        self.assertProblem(self.broken(next_agent="planner"), "'next_agent'")

    def test_revision_count_rules(self):
        self.assertProblem(self.broken(revision_count=-1), "non-negative")
        self.assertProblem(self.broken(revision_count=9), "exceeds 'max_revisions'")
        self.assertProblem(self.broken(revision_count=0), "latest revision record is 1")

    def test_revision_history_rules(self):
        first, last = self.good["revision_history"]
        self.assertProblem(self.broken(revision_history=[last, first]), "not strictly increasing")
        stale = RevisionRecord(revision=1, draft="old text", review=last.review)
        self.assertProblem(self.broken(revision_history=[first, stale]), "does not hold the current draft")
        self.assertProblem(self.broken(review_history=["only one"]), "reviewed revision record(s)")
        unreviewed = RevisionRecord(revision=0, draft=first.draft, review=None)
        self.assertProblem(self.broken(revision_history=[unreviewed, last]), "never reviewed")
        self.assertProblem(self.broken(revision_history=[{"revision": -1, "draft": "x"}]),
                           "'revision_history[0]'")

    def test_ensure_valid_state_raises_with_all_problems(self):
        state = dict(self.good, final_status="bogus", next_agent="planner")
        with self.assertRaises(StateValidationError) as ctx:
            ensure_valid_state(state, "Supervisor")
        self.assertEqual(ctx.exception.agent, "Supervisor")
        self.assertIn("'final_status'", str(ctx.exception))
        self.assertIn("'next_agent'", str(ctx.exception))

    def test_ensure_valid_state_returns_state_when_ok(self):
        self.assertIs(ensure_valid_state(self.good), self.good)


class RunnerHookTests(unittest.TestCase):
    def test_inconsistency_is_logged_not_fatal(self):
        with Fakes([APPROVED]), \
             patch("app.runner.validate_state", return_value=["something is off"]):
            with self.assertLogs("app.runner", level="WARNING") as logs:
                nodes = [n for n, _, _ in stream_research("q about AI agents")]
        self.assertEqual(nodes[-1], "supervisor")  # the run still completed
        self.assertTrue(any("something is off" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
