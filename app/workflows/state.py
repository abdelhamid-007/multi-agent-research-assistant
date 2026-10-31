
import operator
from typing import Annotated, Optional, TypedDict

from app.schemas import (
    AnalysisResult,
    EvaluationResult,
    ResearchFindings,
    ResearchTask,
    ReviewFeedback,
    RevisionRecord,
    WorkerResult,
    WriterOutput,
)


class ResearchState(TypedDict, total=False):

    question: str

    # Parallel research: the supervisor plans tasks, one worker runs per task,
    # and results are appended (operator.add) as the workers finish.
    research_tasks: list[ResearchTask]
    worker_results: Annotated[list[WorkerResult], operator.add]
    max_research_tasks: int

    # Structured data exchanged between agents (validated Pydantic models).
    research_data: Optional[ResearchFindings]
    analysis_data: Optional[AnalysisResult]
    writer_output: Optional[WriterOutput]
    latest_review: Optional[ReviewFeedback]
    # Automated evaluation of each draft; the history is kept across revisions.
    latest_evaluation: Optional[EvaluationResult]
    evaluation_history: list[EvaluationResult]
    # Every draft with the review that judged it, oldest first.
    revision_history: list[RevisionRecord]

    # Text renderings of the structured data (used by prompts, UI, PDF, CLI).
    research: str
    analysis: str

    draft: str
    feedback: str
    review_history: list[str]

    # True once the current draft has been seen by the reviewer.
    # The writer resets it to False; the reviewer sets it to True.
    draft_reviewed: bool

    approved: bool
    revision_count: int
    max_revisions: int

    # in_progress | revision_required | approved | max_revisions_reached
    final_status: str

    next_agent: str
    execution_history: list[str]
