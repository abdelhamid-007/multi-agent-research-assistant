import unittest

from app.agents.supervisor import choose_next_agent


class SupervisorRoutingTests(unittest.TestCase):
    def base_state(self):
        return {
            "approved": False,
            "revision_count": 0,
            "research": "",
            "analysis": "",
            "draft": "",
            "feedback": "",
            "draft_reviewed": False,
        }

    def ready_state(self, **kw):
        state = self.base_state()
        state.update(research="Research", analysis="Analysis", draft="Draft")
        state.update(kw)
        return state

    def test_routes_to_research_first(self):
        self.assertEqual(choose_next_agent(self.base_state())[0], "research")

    def test_routes_to_analysis_after_research(self):
        state = self.base_state()
        state["research"] = "Research findings"
        self.assertEqual(choose_next_agent(state)[0], "analysis")

    def test_routes_to_writer_after_analysis(self):
        state = self.base_state()
        state.update(research="Research", analysis="Analysis")
        self.assertEqual(choose_next_agent(state)[0], "writer")

    def test_routes_to_reviewer_after_draft(self):
        self.assertEqual(choose_next_agent(self.ready_state())[0], "reviewer")

    def test_routes_back_to_writer_after_revision_feedback(self):
        state = self.ready_state(draft_reviewed=True, feedback="Add evidence")
        self.assertEqual(choose_next_agent(state)[0], "writer")

    def test_revised_draft_goes_to_reviewer_even_if_feedback_kept(self):
        # Writer keeps old feedback but resets draft_reviewed.
        state = self.ready_state(
            draft_reviewed=False, feedback="Add evidence", revision_count=1
        )
        self.assertEqual(choose_next_agent(state)[0], "reviewer")

    def test_last_revision_is_reviewed_before_limit_applies(self):
        state = self.ready_state(draft_reviewed=False, revision_count=3)
        self.assertEqual(choose_next_agent(state)[0], "reviewer")

    def test_approval_finishes_before_revision_limit(self):
        state = self.ready_state(approved=True, revision_count=3)
        next_agent, _, status = choose_next_agent(state)
        self.assertEqual((next_agent, status), ("finish", "approved"))

    def test_revision_limit_finishes_as_not_approved(self):
        state = self.ready_state(draft_reviewed=True, revision_count=3)
        next_agent, _, status = choose_next_agent(state)
        self.assertEqual(next_agent, "finish")
        self.assertEqual(status, "max_revisions_reached")

    def test_custom_max_revisions(self):
        state = self.ready_state(
            draft_reviewed=True, revision_count=1, max_revisions=1
        )
        self.assertEqual(choose_next_agent(state)[0], "finish")


if __name__ == "__main__":
    unittest.main()
