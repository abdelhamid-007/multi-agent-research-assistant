
import logging

from app.errors import StateValidationError
from app.evaluation import evaluate_report
from app.evaluation.scoring import LABELS
from app.llm import llm
from app.schemas import (
    APPROVAL_MIN_SCORE,
    AnalysisResult,
    ReviewFeedback,
    ReviewIssue,
    RevisionRecord,
    require_research,
)
from app.structured import invoke_structured

logger = logging.getLogger(__name__)

MAX_EVALUATION_ISSUES = 8

_FORMAT = """{
  "decision": "APPROVED or REVISE",
  "score": 0-10 integer,
  "summary": "one sentence verdict",
  "issues": [
    {"category": "accuracy | completeness | structure | citations",
     "severity": "major | minor",
     "section": "report section name",
     "problem": "the exact problem or claim",
     "suggestion": "the required fix"}
  ],
  "suggestions": ["optional general improvements"]
}"""

_CATEGORY = {
    "factual_accuracy": "accuracy",
    "citation_reliability": "citations",
    "research_coverage": "completeness",
    "logical_structure": "structure",
    "recommendations": "completeness",
}

_FIX = {
    "fabricated_citation": "Remove the marker or cite one of the numbered sources from the research.",
    "fabricated_reference": "Remove the URL; references are generated from the numbered sources.",
    "misattributed_reference": "Cite the source number that matches the claim.",
    "unsupported_critical_claim": "Delete the figure, or replace it with a figure that appears in a numbered source.",
    "miscited_statistic": "Cite the source that actually contains the figure.",
    "uncited_statistic": "Add the citation of the source that contains the figure.",
    "unsupported_claim": "Remove the claim or restate what the cited sources say.",
    "no_citations": "Add inline [n] citations for the factual claims.",
    "missing_citations": "Add inline [n] citations to the uncited factual claims.",
    "incomplete_report": "Add every required section.",
    "report_too_short": "Expand the report with the research findings.",
    "no_recommendations": "Add a Recommendations section with at least three actions.",
}


def _analysis(state):
    value = state.get("analysis_data")
    if value is None or isinstance(value, AnalysisResult):
        return value
    try:
        return AnalysisResult.model_validate(value)
    except Exception:  # noqa: BLE001 - analysis only informs the recommendations check
        return None


def review_from_evaluation(evaluation):
    """A REVISE review built purely from a failed automated evaluation."""
    order = {"critical": 0, "major": 1}
    flags = sorted((f for f in evaluation.flags if f.severity in order), key=lambda f: order[f.severity])
    issues = [
        ReviewIssue(
            category=_CATEGORY[f.dimension], severity="major", section=LABELS[f.dimension],
            problem=f.message + (f' - "{f.evidence}"' if f.evidence else ""),
            suggestion=_FIX.get(f.code, "Fix this using only the numbered sources."),
        )
        for f in flags[:MAX_EVALUATION_ISSUES]
    ]
    if len(flags) > MAX_EVALUATION_ISSUES:
        issues[-1] = issues[-1].model_copy(update={
            "problem": issues[-1].problem + f" (+{len(flags) - MAX_EVALUATION_ISSUES} more flagged problem(s))"})
    if not issues:  # failed on score alone
        worst = min(evaluation.dimensions, key=lambda d: d.score)
        issues = [ReviewIssue(
            category=_CATEGORY[worst.name], severity="major", section=LABELS[worst.name],
            problem="; ".join(evaluation.blocking_reasons),
            suggestion=f"Improve {LABELS[worst.name].lower()} first: " + "; ".join(worst.notes),
        )]
    return ReviewFeedback(
        decision="REVISE", score=min(APPROVAL_MIN_SCORE - 1, int(evaluation.overall_score // 10)),
        summary=evaluation.summary, issues=issues,
    )


def combine_reviews(llm_review, evaluation):
    """The LLM review, with its score capped by the evaluation. Never approves a failed evaluation."""
    if not evaluation.passed:
        return review_from_evaluation(evaluation)
    return ReviewFeedback(
        decision=llm_review.decision,
        score=min(llm_review.score, int(evaluation.overall_score // 10)),
        summary=llm_review.summary,
        issues=llm_review.issues,
        suggestions=llm_review.suggestions,
    )


def reviewer_agent(state):

    print("\n🔎 Reviewer Agent is checking the report...")

    research = require_research(state, "Reviewer")
    draft = state.get("draft") or ""
    if not draft.strip():
        raise StateValidationError("Reviewer", "there is no draft to review")

    attempt = state.get("revision_count", 0)
    logger.info("Reviewer: reviewing draft after %d revision(s)", attempt)

    # Step 1: deterministic evaluation against the verified sources.
    evaluation = evaluate_report(draft, research, _analysis(state), revision=attempt)

    if not evaluation.passed:
        # A blocking failure decides the outcome; the LLM call would not change it.
        logger.info("Reviewer: automated evaluation blocks approval (%s); skipping the LLM review",
                    "; ".join(evaluation.blocking_reasons))
        review = review_from_evaluation(evaluation)
    else:
        # Step 2: the LLM gives a qualitative second opinion (advisory, not verification).
        previous = state.get("feedback", "")
        previous_block = (
            f"""
    Your feedback on the PREVIOUS draft (check whether each issue is now
    fixed; do not invent new minor demands):
    {previous[:2000]}
    """
            if previous
            else ""
        )

        prompt = f"""
    You are a strict but fair report quality reviewer.

    Original question:
    {state['question']}

    Research evidence (the only allowed source of facts; [n] are source numbers):
    {research.to_text(max_chars=4000)}

    Report to review (complete, not truncated):
    {draft}

    Automated checks (deterministic, already PASSED - do not repeat them):
    {evaluation.summary}
    {previous_block}
    These automated checks only compare the report with the retrieved sources.
    Look for what they cannot: claims that misrepresent what a cited source
    says, logical gaps or contradictions, and recommendations that do not
    follow from the findings.

    Approval policy:
    - Report at most 5 issues, most important first.
    - APPROVED requires a score of at least {APPROVAL_MIN_SCORE} and NO major issues.
    - REVISE requires at least one issue. Style preferences are not major.
    - Be specific: name the section and the exact claim to change.

    Reply with ONLY a JSON object in exactly this format:
    {_FORMAT}
    """
        review = combine_reviews(invoke_structured(llm, prompt, ReviewFeedback, "Reviewer"), evaluation)

    # Hard guarantee: a failed evaluation (including any critical failure) never yields approval.
    if review.approved and not evaluation.passed:
        raise StateValidationError("Reviewer", "an approval was produced for a failed evaluation")

    text = review.to_text() + "\n\n" + evaluation.to_text()
    logger.info(
        "Reviewer decision (after %d revision(s)): %s, score %d/10, evaluation %.1f/100, %d issue(s)\n%s",
        attempt, review.decision, review.score, evaluation.overall_score, len(review.issues), text,
    )
    print("\nReviewer decision:")
    print(text)

    history = list(state.get("revision_history", []))
    if history and history[-1].review is None:
        history[-1] = history[-1].model_copy(update={"review": review, "evaluation": evaluation})
    else:
        history.append(RevisionRecord(revision=attempt, draft=draft, review=review, evaluation=evaluation))

    return {
        "feedback": text,
        "review_history": state.get("review_history", []) + [text],
        "latest_review": review,
        "latest_evaluation": evaluation,
        "evaluation_history": state.get("evaluation_history", []) + [evaluation],
        "revision_history": history,
        "approved": review.approved,
        "draft_reviewed": True,

        "final_status": "approved" if review.approved else "revision_required",

        "execution_history": state.get("execution_history", []) + ["reviewer"],
    }
