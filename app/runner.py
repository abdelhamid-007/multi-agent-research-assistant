"""UI-independent helpers for running the existing LangGraph workflow.

Used by the Streamlit app (and available to the CLI). It does not change the
graph, agents or state; it only streams the compiled graph node by node.
"""
import logging
import re

from app.config import max_research_tasks as default_research_tasks
from app.config import research_concurrency
from app.errors import AgentError
from app.state_validation import validate_state

logger = logging.getLogger(__name__)

AGENTS = ["supervisor", "research", "analysis", "writer", "reviewer"]
RECURSION_LIMIT = 40
DEFAULT_MAX_REVISIONS = 3

IDLE, RUNNING, DONE, ERROR = "idle", "running", "done", "error"


def build_initial_state(question, max_revisions=DEFAULT_MAX_REVISIONS, max_research_tasks=None):
    """Fresh ResearchState for one run."""
    return {
        "question": question,

        "research_tasks": [],
        "worker_results": [],
        "max_research_tasks": max_research_tasks or default_research_tasks(),

        "research_data": None,
        "analysis_data": None,
        "writer_output": None,
        "latest_review": None,
        "latest_evaluation": None,
        "evaluation_history": [],
        "revision_history": [],

        "research": "",
        "analysis": "",

        "draft": "",
        "feedback": "",
        "review_history": [],
        "draft_reviewed": False,

        "approved": False,
        "revision_count": 0,
        "max_revisions": max_revisions,

        "next_agent": "",
        "execution_history": [],
        "final_status": "in_progress",
    }


def graph_config(concurrency=None, recursion_limit=RECURSION_LIMIT):
    """LangGraph run config, including the parallel-research concurrency limit."""
    return {
        "recursion_limit": recursion_limit,
        "max_concurrency": concurrency or research_concurrency(),
    }


def apply_update(state, update):
    """Merge a node's update into an accumulated state dict.

    Mirrors the graph's own reducer: `worker_results` is appended to, every
    other key is replaced.
    """
    for key, value in (update or {}).items():
        if key == "worker_results":
            state[key] = list(state.get(key) or []) + list(value)
        else:
            state[key] = value


def stream_research(question, max_revisions=DEFAULT_MAX_REVISIONS,
                    max_research_tasks=None, concurrency=None):
    """Run the existing graph, yielding (node, update, state) after each node.

    `state` is the accumulated state so far (a single dict that is mutated
    in place; copy it if you need a snapshot). Parallel research workers
    appear as separate `research_worker` events.
    """
    # Imported lazily so a missing API key surfaces as a catchable error
    # at run time instead of crashing the importing app.
    from app.workflows.graph import research_graph

    state = build_initial_state(question, max_revisions, max_research_tasks)
    chunks = research_graph.stream(
        state,
        config=graph_config(concurrency),
        stream_mode="updates",
    )
    for chunk in chunks:
        for node, update in chunk.items():
            apply_update(state, update)
            # Diagnostic only: an inconsistent state is logged, never fatal.
            for problem in validate_state(state):
                logger.warning("State check after '%s': %s", node, problem)
            yield node, update or {}, state


def _clip(text, limit=220):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def summarize_update(node, update, state):
    """One short, human-readable line describing what a node just produced."""
    if node == "supervisor":
        nxt = update.get("next_agent", "")
        if nxt == "finish":
            return f"Workflow finished ({update.get('final_status', 'done')})."
        return f"Routed work to the {nxt} agent."

    if node == "research_worker":
        result = (update.get("worker_results") or [None])[0]
        if result is None:
            return ""
        label = f"Task {result.task.id} ({result.task.aspect})"
        if result.status == "failed":
            return f"{label} FAILED after {result.duration_s:.1f}s: {_clip(result.error, 120)}"
        return (f"{label}: {len(result.sources)} source(s), {len(result.findings)} finding(s) "
                f"in {result.duration_s:.1f}s.")

    if node == "research":
        data = update.get("research_data")
        if data is not None:
            failed = sum(1 for r in data.task_reports if r.status == "failed")
            tasks = (f" From {len(data.task_reports)} task(s)"
                     + (f", {failed} failed." if failed else ".")) if data.task_reports else ""
            return (f"{len(data.sources)} source URL(s), {len(data.findings)} finding(s)."
                    + tasks + " " + _clip(data.summary, 120))
        urls = set(re.findall(r"https?://[^\s\)\]>\"'`,]+", update.get("research", "")))
        return f"{len(urls)} source URL(s). " + _clip(update.get("research", ""), 160)

    if node == "analysis":
        data = update.get("analysis_data")
        if data is not None:
            return (f"{len(data.opportunities)} opportunity(ies), {len(data.risks)} risk(s), "
                    f"complexity {data.implementation_complexity}.")
        return _clip(update.get("analysis", ""))

    if node == "writer":
        kind = (
            f"Revision {update.get('revision_count', 0)} written"
            if update.get("revision_count", 0) > 0
            else "Initial draft written"
        )
        return f"{kind} ({len(update.get('draft', ''))} characters)."

    if node == "reviewer":
        verdict = "APPROVED" if update.get("approved") else "REVISE"
        review = update.get("latest_review")
        evaluation = update.get("latest_evaluation")
        if review is not None:
            first = review.issues[0].problem if review.issues else ""
            scores = (f"score {review.score}/10, evaluation {evaluation.overall_score:.0f}/100"
                      if evaluation is not None else f"score {review.score}/10")
            return f"Decision: {verdict} ({scores}). " + _clip(first, 130)
        feedback = update.get("feedback", "")
        issue = re.search(r"^\s*1\.\s*(.+)$", feedback, re.MULTILINE)
        if update.get("approved"):
            return "Decision: APPROVED."
        return f"Decision: {verdict}. " + (_clip(issue.group(1), 170) if issue else "")

    return ""


class AgentTracker:
    """Tracks per-agent status/latest result from the graph's node updates."""

    def __init__(self):
        self.status = {a: (IDLE, "Waiting.") for a in AGENTS}
        self.current = None

    def start(self):
        self._mark_running("supervisor", "Deciding the next step…")

    def _mark_running(self, agent, detail):
        # Only one agent is running at a time.
        for name, (st, det) in self.status.items():
            if st == RUNNING:
                self.status[name] = (DONE, det)
        self.status[agent] = (RUNNING, detail)
        self.current = agent

    def on_event(self, node, update, state):
        """Record a finished node and mark whichever agent runs next."""
        if node == "research_worker":
            # Parallel workers report progress on the Research card.
            done = len(state.get("worker_results") or [])
            total = len(state.get("research_tasks") or []) or done
            failed = sum(1 for r in state.get("worker_results") or [] if r.status == "failed")
            detail = f"Research tasks finished: {done}/{total}" + (f" ({failed} failed)" if failed else "")
            self.status["research"] = (RUNNING, detail)
            return

        if node not in self.status:
            return  # an unknown node must never break the run display

        self.status[node] = (DONE, summarize_update(node, update, state))

        if node == "supervisor":
            nxt = update.get("next_agent")
            if nxt in AGENTS:
                if nxt == "writer" and state.get("draft"):
                    detail = (
                        f"Revising the report "
                        f"({state.get('revision_count', 0) + 1}/{state.get('max_revisions', 3)})…"
                    )
                elif nxt == "research" and state.get("research_tasks"):
                    detail = f"Running {len(state['research_tasks'])} research task(s) in parallel…"
                else:
                    detail = "Working…"
                self._mark_running(nxt, detail)
            else:
                self.current = None
        else:
            self._mark_running("supervisor", "Deciding the next step…")

    def fail(self, message):
        agent = self.current or "supervisor"
        self.status[agent] = (ERROR, _clip(message, 200))
        self.current = None


def friendly_error(exc):
    """Translate common failures into a message a user can act on."""
    text = str(exc) if isinstance(exc, AgentError) else f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "groq_api_key" in low or "tavily_api_key" in low or ("api_key" in low and "not found" in low):
        return ("An API key is missing. Add GROQ_API_KEY and TAVILY_API_KEY to the "
                ".env file in the project folder, then restart the app.")
    if "tokens per day" in low or "(tpd)" in low:
        wait = re.search(r"try again in ([0-9hms.]+)", text)
        return ("The model provider's daily token quota is used up"
                + (f" (it resets in about {wait.group(1)})" if wait else "")
                + ". Wait for the reset or use a higher usage tier, then try again.")
    if "413" in low or "429" in low or "rate_limit" in low or "rate limit" in low:
        return ("The model provider's rate limit or token limit was hit. "
                "Wait a minute and try again, or lower the revision limit.")
    if "401" in low or "invalid api key" in low or "authentication" in low:
        return "An API key was rejected. Check the keys in your .env file."
    if "recursion" in low:
        return "The workflow exceeded its step limit and was stopped."
    if "connect" in low or "timeout" in low or "timed out" in low:
        return "A network request failed or timed out. Check your connection and retry."
    if isinstance(exc, AgentError):
        return f"{text}. Try again; if it keeps happening, check the logs."
    return f"The research workflow failed: {text}"
