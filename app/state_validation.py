"""Whole-state consistency checks for the shared LangGraph state.

Agents already validate the data they receive. These checks look at the
state as a whole and catch contradictions between fields (for example
`approved=True` without an approving review).
"""
from pydantic import ValidationError

from app.errors import StateValidationError
from app.schemas import (
    AnalysisResult,
    EvaluationResult,
    ResearchFindings,
    ReviewFeedback,
    ResearchTask,
    RevisionRecord,
    WorkerResult,
    WriterOutput,
)

VALID_STATUSES = {"in_progress", "revision_required", "approved", "max_revisions_reached"}
VALID_NEXT_AGENTS = {"", "research", "analysis", "writer", "reviewer", "finish"}

_STRUCTURED = {
    "research_data": ResearchFindings,
    "analysis_data": AnalysisResult,
    "writer_output": WriterOutput,
    "latest_review": ReviewFeedback,
}


def _as_model(value, model):
    """Return the model instance, or None if absent; raise ValueError if invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, model):
        return value
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise ValueError(str(exc).splitlines()[0]) from exc


def validate_state(state):
    """Return a list of human-readable problems (empty list = consistent)."""
    problems = []
    models = {}

    for key, model in _STRUCTURED.items():
        try:
            models[key] = _as_model(state.get(key), model)
        except ValueError as exc:
            problems.append(f"'{key}' is not a valid {model.__name__}: {exc}")
            models[key] = None

    history = []
    for i, record in enumerate(state.get("revision_history") or []):
        try:
            history.append(_as_model(record, RevisionRecord))
        except ValueError as exc:
            problems.append(f"'revision_history[{i}]' is not a valid RevisionRecord: {exc}")

    tasks = []
    for i, task in enumerate(state.get("research_tasks") or []):
        try:
            tasks.append(_as_model(task, ResearchTask))
        except ValueError as exc:
            problems.append(f"'research_tasks[{i}]' is not a valid ResearchTask: {exc}")
    if len({t.id for t in tasks}) != len(tasks):
        problems.append("'research_tasks' contains duplicate task ids")

    task_ids = {t.id for t in tasks}
    seen_workers = []
    for i, result in enumerate(state.get("worker_results") or []):
        try:
            seen_workers.append(_as_model(result, WorkerResult))
        except ValueError as exc:
            problems.append(f"'worker_results[{i}]' is not a valid WorkerResult: {exc}")
    worker_ids = [r.task.id for r in seen_workers]
    if len(set(worker_ids)) != len(worker_ids):
        problems.append("'worker_results' contains two results for the same task")
    if tasks and set(worker_ids) - task_ids:
        problems.append("'worker_results' has a result for a task that was never planned")

    evaluation = None
    try:
        evaluation = _as_model(state.get("latest_evaluation"), EvaluationResult)
    except ValueError as exc:
        problems.append(f"'latest_evaluation' is not a valid EvaluationResult: {exc}")
    evaluations = []
    for i, item in enumerate(state.get("evaluation_history") or []):
        try:
            evaluations.append(_as_model(item, EvaluationResult))
        except ValueError as exc:
            problems.append(f"'evaluation_history[{i}]' is not a valid EvaluationResult: {exc}")
    eval_revisions = [e.revision for e in evaluations]
    if eval_revisions != sorted(set(eval_revisions)):
        problems.append(f"'evaluation_history' revisions are not strictly increasing: {eval_revisions}")
    if evaluations and evaluation is not None and evaluations[-1].revision != evaluation.revision:
        problems.append("'latest_evaluation' is not the last entry of 'evaluation_history'")
    if len(evaluations) != len(state.get("review_history") or []):
        problems.append(f"{len(evaluations)} evaluation(s) but {len(state.get('review_history') or [])} "
                        "entries in 'review_history'")

    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)
    if not isinstance(revision_count, int) or revision_count < 0:
        problems.append(f"'revision_count' must be a non-negative integer, got {revision_count!r}")
    elif revision_count > max_revisions:
        problems.append(f"'revision_count' ({revision_count}) exceeds 'max_revisions' ({max_revisions})")

    status = state.get("final_status")
    if status is not None and status not in VALID_STATUSES:
        problems.append(f"'final_status' has unknown value {status!r}")

    next_agent = state.get("next_agent", "")
    if next_agent not in VALID_NEXT_AGENTS:
        problems.append(f"'next_agent' has unknown value {next_agent!r}")

    # text views must agree with the structured data they are derived from
    if state.get("research") and models["research_data"] is None:
        problems.append("'research' text exists but 'research_data' is missing")
    if state.get("analysis") and models["analysis_data"] is None:
        problems.append("'analysis' text exists but 'analysis_data' is missing")
    writer = models["writer_output"]
    if state.get("draft") and writer is None:
        problems.append("'draft' exists but 'writer_output' is missing")
    if writer is not None and state.get("draft") and writer.markdown != state["draft"]:
        problems.append("'draft' differs from 'writer_output.markdown'")

    # decisions
    review = models["latest_review"]
    if state.get("approved"):
        if review is None or not review.approved:
            problems.append("'approved' is True but the latest review is not an approval")
        if not state.get("draft_reviewed"):
            problems.append("'approved' is True but the current draft was not reviewed")
        if evaluation is None or not evaluation.passed:
            problems.append("'approved' is True but the latest automated evaluation did not pass")
        if status is not None and status not in ("approved",):
            problems.append(f"'approved' is True but 'final_status' is {status!r}")
    if status == "approved" and not state.get("approved"):
        problems.append("'final_status' is 'approved' but 'approved' is False")

    # revision history
    numbers = [r.revision for r in history]
    if numbers != sorted(set(numbers)):
        problems.append(f"'revision_history' revisions are not strictly increasing: {numbers}")
    if history and state.get("draft"):
        if history[-1].revision != revision_count:
            problems.append(
                f"latest revision record is {history[-1].revision} but 'revision_count' is {revision_count}"
            )
        if history[-1].draft != state["draft"]:
            problems.append("latest revision record does not hold the current draft")
    reviewed = sum(1 for r in history if r.review is not None)
    if history and reviewed != len(state.get("review_history") or []):
        problems.append(
            f"{reviewed} reviewed revision record(s) but {len(state.get('review_history') or [])} "
            "entries in 'review_history'"
        )
    if any(r.review is None for r in history[:-1]):
        problems.append("an earlier draft in 'revision_history' was never reviewed")

    return problems


def ensure_valid_state(state, where="state"):
    """Raise StateValidationError listing every problem, if there are any."""
    problems = validate_state(state)
    if problems:
        raise StateValidationError(where, "; ".join(problems))
    return state
