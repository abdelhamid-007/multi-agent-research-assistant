
import logging
import re

from pydantic import ValidationError

from app.errors import StructuredOutputError
from app.llm import llm
from app.retry import invoke_llm
from app.schemas import (
    CITATION_RE,
    RevisionRecord,
    WriterOutput,
    cited_ids,
    require_analysis,
    require_research,
)

logger = logging.getLogger(__name__)

MIN_BODY_WORDS = 20

_SOURCES_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*)?(?:\d+[.)]\s*)?(?:sources|references)\s*:?\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_sources_section(text):
    """Drop any Sources/References section the LLM wrote itself."""
    match = _SOURCES_HEADING.search(text or "")
    return (text[: match.start()] if match else text or "").rstrip()


def build_sources_section(source_map, ids):
    """Sources list generated from real Source objects, never from the LLM."""
    if not ids:
        return "## Sources\nNo sources were cited."
    lines = [f"- [{i}] {source_map[i].label()} - {source_map[i].url}" for i in ids]
    return "## Sources\n" + "\n".join(lines)


def assemble_report(body, source_map):
    """Validate citations in `body` and append the generated Sources section.

    Returns (markdown, cited_ids, warnings). Citation markers that point to
    a non-existent source are removed and reported as warnings.
    """
    warnings = []
    body = strip_sources_section(body)
    invalid = [i for i in cited_ids(body) if i not in source_map]
    if invalid:
        body = CITATION_RE.sub(lambda m: "" if int(m.group(1)) in invalid else m.group(0), body)
        warnings.append(f"Removed citation(s) to unknown source(s): {invalid}")
    used = cited_ids(body)
    if not used:
        warnings.append("The report cites no sources.")
    return f"{body}\n\n{build_sources_section(source_map, used)}", used, warnings


def writer_agent(state):

    print("\n✍️ Writer Agent is working...")

    research = require_research(state, "Writer")
    analysis = require_analysis(state, "Writer")
    source_map = research.source_map()

    is_revision = bool(state.get("draft"))
    revision_count = state.get("revision_count", 0) + (1 if is_revision else 0)
    feedback = state.get("feedback", "")

    if is_revision:
        print("♻️ Revising report...")
        logger.info(
            "Writer: revision %d/%d. Feedback being addressed:\n%s",
            revision_count, state.get("max_revisions", 3), feedback,
        )
    else:
        print("📝 Creating initial report...")
        logger.info("Writer: creating initial draft")

    revision_block = ""
    if is_revision:
        # Only the review before the latest one, capped, to stay inside
        # the model's token-per-minute budget.
        earlier = state.get("review_history", [])[-2:-1]
        earlier_text = earlier[0][:1500] if earlier else "(none)"
        revision_block = f"""
    Previous draft (revise this, do not start over unless necessary):
    {strip_sources_section(state.get('draft', ''))}

    LATEST reviewer feedback (address EVERY numbered issue):
    {feedback}

    Earlier reviewer feedback (do not reintroduce these problems):
    {earlier_text}
    """

    prompt = f"""
    You are a professional business report writer.

    Question:
    {state['question']}

    Research evidence (the ONLY source of facts):
    {research.to_text(max_chars=4000)}

    Analysis:
    {analysis.to_text()[:2500]}
    {revision_block}
    Your task:
    Write or improve a professional business report.

    Include these sections as markdown headings:
    Executive Summary, Research Findings, Business Opportunities,
    Technical Analysis, Risks, Recommendations, Conclusion.

    Rules:
    - Maximum 700 words. Be concise.
    - Support factual claims with inline citations like [1] or [2][3] using
      ONLY the source numbers listed above. Never cite a number that is not
      listed, and never write URLs yourself.
    - Do NOT write a Sources or References section; it is added automatically.
    - Clearly label assumptions and opinions as such (e.g. "Assumption:").
    - Remove or soften any claim the research does not support.
    - If reviewer feedback exists, fix every numbered issue in the report
      itself. Output only the full revised report, no commentary.
    """

    body = invoke_llm(llm, prompt, "Writer LLM call").content

    # One repair attempt if the model cited unknown sources or none at all.
    problems = [i for i in cited_ids(body) if i not in source_map]
    if problems or not cited_ids(body):
        reason = (f"It cited unknown source numbers {problems}." if problems
                  else "It contained no [n] citations.")
        logger.warning("Writer: %s Asking for one correction.", reason)
        body = invoke_llm(
            llm,
            f"{prompt}\n\nYour previous report was rejected. {reason} "
            f"Valid source numbers: {sorted(source_map)}. "
            "Rewrite the full report with correct [n] citations only.",
            "Writer LLM call (correction)",
        ).content

    if len(strip_sources_section(body).split()) < MIN_BODY_WORDS:
        raise StructuredOutputError(
            "Writer", f"the model returned an unusable report (fewer than {MIN_BODY_WORDS} words)")

    markdown, used, warnings = assemble_report(body, source_map)
    for w in warnings:
        logger.warning("Writer: %s", w)

    try:
        output = WriterOutput(
            title=state["question"],
            markdown=markdown,
            cited_source_ids=used,
            revision=revision_count,
            warnings=warnings,
        )
    except ValidationError as exc:
        raise StructuredOutputError("Writer", f"the model returned an unusable report: {exc}") from exc

    logger.info(
        "Writer: produced %s (%d words, %d source(s) cited), revision_count=%d",
        "revised draft" if is_revision else "initial draft",
        output.word_count, len(used), revision_count,
    )

    return {
        "draft": output.markdown,
        "writer_output": output,
        "revision_count": revision_count,
        "revision_history": state.get("revision_history", [])
        + [RevisionRecord(revision=revision_count, draft=output.markdown)],

        # The new draft has not been reviewed yet.
        "draft_reviewed": False,
        "approved": False,
        "final_status": "in_progress",

        # feedback is deliberately kept; the reviewer overwrites it.

        "execution_history": state.get("execution_history", []) + ["writer"],
    }
