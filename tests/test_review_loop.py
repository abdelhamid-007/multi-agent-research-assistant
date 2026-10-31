import unittest

from app.runner import build_initial_state
from tests.fakes import APPROVED, REVISE, URL, WRITER_BODY, Fakes, issue, review_json


class GraphLoopTests(unittest.TestCase):
    def run_graph(self, reviewer_replies, **kw):
        from app.workflows.graph import research_graph

        with Fakes(reviewer_replies, **kw) as fakes:
            result = research_graph.invoke(
                build_initial_state("q about AI agents"),
                config={"recursion_limit": 40},
            )
        return result, fakes

    def test_never_approved_reviews_every_draft_and_reports_failure(self):
        result, _ = self.run_graph([REVISE])

        self.assertFalse(result["approved"])
        self.assertEqual(result["final_status"], "max_revisions_reached")
        self.assertEqual(result["revision_count"], 3)
        self.assertEqual(len(result["review_history"]), 4)  # initial + 3 revisions
        self.assertEqual(result["execution_history"][-2:], ["reviewer", "supervisor"])
        self.assertIn("Add detail", result["feedback"])

    def test_revision_history_links_each_draft_to_its_review(self):
        result, _ = self.run_graph([REVISE, APPROVED])

        records = result["revision_history"]
        self.assertEqual([r.revision for r in records], [0, 1])
        self.assertEqual([r.review.decision for r in records], ["REVISE", "APPROVED"])
        self.assertTrue(all(r.draft for r in records))
        self.assertEqual(result["latest_review"].decision, "APPROVED")

    def test_approval_on_second_review(self):
        result, _ = self.run_graph([REVISE, APPROVED])

        self.assertTrue(result["approved"])
        self.assertEqual(result["final_status"], "approved")
        self.assertEqual(result["revision_count"], 1)

    def test_writer_receives_full_feedback(self):
        long_issue = issue("detail " * 400 + "END_OF_FEEDBACK")
        _, fakes = self.run_graph([review_json("REVISE", 4, [long_issue]), APPROVED])

        self.assertIn("END_OF_FEEDBACK", fakes.writer.prompts[1])  # revision prompt
        self.assertNotIn("LATEST reviewer feedback", fakes.writer.prompts[0])

    def test_invented_url_blocks_approval_even_if_the_llm_would_approve(self):
        body = WRITER_BODY + "\nSee https://invented.example/x for more."
        result, fakes = self.run_graph([APPROVED], writer_replies=[body])
        first = result["revision_history"][0]
        self.assertEqual(first.review.decision, "REVISE")
        self.assertEqual(first.evaluation.critical_failures[0].code, "fabricated_reference")
        self.assertIn("not one of the verified sources", result["review_history"][0])
        self.assertEqual(fakes.reviewer.prompts, [])  # the deterministic gate decided; no LLM call needed
        self.assertFalse(result["approved"])
        self.assertEqual(result["final_status"], "max_revisions_reached")

    def test_evaluation_history_is_kept_across_revisions(self):
        result, _ = self.run_graph([REVISE, APPROVED])
        self.assertEqual([e.revision for e in result["evaluation_history"]], [0, 1])
        self.assertEqual(result["latest_evaluation"], result["evaluation_history"][-1])
        self.assertTrue(all(r.evaluation is not None for r in result["revision_history"]))
        self.assertTrue(result["latest_evaluation"].passed)

    def test_sources_section_is_generated_from_real_sources(self):
        result, _ = self.run_graph([APPROVED])
        self.assertIn("## Sources", result["draft"])
        self.assertIn(f"- [1] Agent trends - example.com, 2026-03-01 - {URL}", result["draft"])


if __name__ == "__main__":
    unittest.main()
