"""Supervisor-side research task planning.

The plan only decides WHAT to look up (aspects and search queries). It never
contains findings; all evidence still comes from the search tool.
"""
import logging

from app.llm import llm
from app.schemas import ResearchPlan, ResearchTask
from app.structured import invoke_structured

logger = logging.getLogger(__name__)

# Deterministic fallback: generic aspects of any topic.
_DEFAULT_ASPECTS = [
    ("Overview and key developments", "{q}"),
    ("Market adoption and trends", "{q} market adoption statistics trends"),
    ("Technology and technical approaches", "{q} technology architecture approaches"),
    ("Risks, challenges and limitations", "{q} risks challenges limitations"),
    ("Business opportunities and use cases", "{q} business opportunities use cases"),
    ("Regulation, governance and standards", "{q} regulation governance standards"),
    ("Case studies and real-world examples", "{q} case studies examples"),
    ("Future outlook and predictions", "{q} future outlook predictions"),
]

_FORMAT = """{
  "tasks": [
    {"id": 1, "aspect": "short label of the aspect", "query": "specific web search query"}
  ]
}"""


def default_tasks(question, count):
    """The first `count` generic aspects, built from the question text."""
    question = (question or "").strip()
    return [
        ResearchTask(id=i, aspect=aspect, query=template.format(q=question))
        for i, (aspect, template) in enumerate(_DEFAULT_ASPECTS[:max(1, count)], 1)
    ]


def plan_research_tasks(question, max_tasks):
    """Return 1..max_tasks distinct research tasks for the question.

    Uses one LLM call; if it fails or returns an unusable plan, falls back to
    the deterministic aspects so planning can never block the run.
    """
    question = (question or "").strip()
    if max_tasks <= 1:
        return default_tasks(question, 1)

    prompt = f"""
    You are a research planner for a team of parallel web researchers.

    Question:
    {question}

    Split the research into {max_tasks} DISTINCT aspects that together cover
    the question well (for example: current state, adoption/market, technology,
    risks, opportunities). Give each aspect one specific web search query
    that would find evidence for it. Do not answer the question and do not
    state any facts: only plan the searches.

    Reply with ONLY a JSON object in exactly this format:
    {_FORMAT}
    """
    try:
        plan = invoke_structured(llm, prompt, ResearchPlan, "Supervisor")
    except Exception as exc:  # noqa: BLE001 - planning must never fail the run
        logger.warning("Supervisor: research planning failed (%s); using default aspects", exc)
        return default_tasks(question, max_tasks)

    tasks = [
        ResearchTask(id=i, aspect=t.aspect, query=t.query)
        for i, t in enumerate(plan.tasks[:max_tasks], 1)
    ]
    logger.info("Supervisor: planned %d research task(s)", len(tasks))
    return tasks
