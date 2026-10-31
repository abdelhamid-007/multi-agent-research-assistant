import logging

from app.config import max_research_tasks
from app.research.planner import plan_research_tasks

logger = logging.getLogger(__name__)

DEFAULT_MAX_REVISIONS = 3


def choose_next_agent(state):
    """Choose the next node from the current workflow state.

    Returns (next_agent, reason, final_status). final_status is only set
    when next_agent == "finish".
    """
    max_revisions = state.get("max_revisions", DEFAULT_MAX_REVISIONS)
    revisions = state.get("revision_count", 0)

    if state.get("approved", False):
        return "finish", "Report approved.", "approved"

    if not state.get("research"):
        return "research", "Research is missing.", None

    if not state.get("analysis"):
        return "analysis", "Analysis is missing.", None

    if not state.get("draft"):
        return "writer", "Initial draft is missing.", None

    # Every draft, including the last permitted revision, must be reviewed
    # before the revision limit is allowed to end the run.
    if not state.get("draft_reviewed", False):
        return "reviewer", "Current draft has not been reviewed yet.", None

    if revisions >= max_revisions:
        return (
            "finish",
            f"Maximum revisions reached ({revisions}/{max_revisions}) "
            "and the last draft was not approved.",
            "max_revisions_reached",
        )

    return (
        "writer",
        f"Reviewer requested improvements "
        f"(revision {revisions + 1}/{max_revisions}).",
        None,
    )


def supervisor_agent(state):
    print("\n🧠 Supervisor is deciding...")

    next_agent, reason, final_status = choose_next_agent(state)

    print("Next agent:", next_agent)
    print("Reason:", reason)
    logger.info("Supervisor -> %s | %s", next_agent, reason)

    update = {
        "next_agent": next_agent,
        "execution_history": state.get("execution_history", []) + ["supervisor"],
    }

    if next_agent == "research" and not state.get("research_tasks"):
        # Dynamic task generation: one parallel worker per research task.
        tasks = plan_research_tasks(
            state.get("question", ""), state.get("max_research_tasks") or max_research_tasks()
        )
        update["research_tasks"] = tasks
        for t in tasks:
            logger.info("Supervisor: research task %d [%s] -> %r", t.id, t.aspect, t.query)

    if final_status:
        update["final_status"] = final_status
        if final_status != "approved":
            logger.warning(
                "Finishing WITHOUT approval (status=%s). "
                "Last reviewer feedback:\n%s",
                final_status,
                state.get("feedback", ""),
            )

    return update
