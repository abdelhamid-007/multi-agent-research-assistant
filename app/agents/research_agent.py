
import logging
import time
from typing import Optional

from pydantic import BaseModel

from app.errors import AgentError
from app.llm import llm
from app.research.aggregation import merge_results
from app.retry import retry_call
from app.schemas import (
    Finding,
    ResearchTask,
    Source,
    WorkerResult,
    require_worker_results,
)
from app.structured import invoke_structured
from app.tools.web_search import search_web

logger = logging.getLogger(__name__)

MAX_SOURCES = 8  # per worker

_DATE_KEYS = ("published_date", "published", "date", "publication_date")
_PUBLISHER_KEYS = ("publisher", "source", "site_name", "author")

_FORMAT = """{
  "summary": "2-4 sentence overview of what the sources say about the research focus",
  "findings": [
    {"claim": "one specific finding",
     "evidence": "short supporting detail taken from the source",
     "source_ids": [1],
     "confidence": "high | medium | low"}
  ],
  "conflicts": ["description of any conflicting information, or leave the list empty"]
}"""


class _ResearchDraft(BaseModel):
    """What the LLM contributes; sources are attached by code."""

    summary: str
    findings: list[Finding]
    conflicts: list[str] = []


def _first(item, keys) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _result_items(results):
    return (
        (results.get("results") or [])
        if isinstance(results, dict)
        else results if isinstance(results, list)
        else []
    )


def build_sources(results):
    """Turn raw search results into validated, numbered Source objects.

    All metadata comes from the search tool; unusable results are skipped.
    """
    sources, seen = [], set()
    for item in _result_items(results):
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url or url.lower() in seen:
            continue
        try:
            source = Source(
                id=len(sources) + 1,
                title=item.get("title") or url,
                url=url,
                publisher=_first(item, _PUBLISHER_KEYS),
                published_date=_first(item, _DATE_KEYS),
                snippet=item.get("content") or "",
                score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            )
        except ValueError as exc:
            logger.warning("Research: skipping unusable search result %r: %s", url, exc)
            continue
        seen.add(url.lower())
        sources.append(source)
        if len(sources) >= MAX_SOURCES:
            break
    return sources


def run_research_task(task, question):
    """Search, build sources and extract findings for ONE task.

    Returns an ok WorkerResult (without timing) or raises. Everything the
    result contains comes from the search tool or from an LLM constrained to
    cite the numbered sources.
    """
    results = retry_call(search_web, task.query, label=f"Web search [{task.aspect}]")
    sources = build_sources(results)
    skipped = max(len(_result_items(results)) - len(sources), 0)
    if not sources:
        raise AgentError(
            "Research",
            "the web search returned no usable sources, so there is no evidence to build a report on",
        )

    source_block = "\n\n".join(
        f"[{s.id}] {s.title}\n    publisher: {s.publisher or s.domain}"
        f"{'; published: ' + s.published_date if s.published_date else ''}\n"
        f"    url: {s.url}\n    content: {s.snippet}"
        for s in sources
    )

    prompt = f"""
    You are a professional research agent.

    Overall question:
    {question}

    Research focus: {task.aspect}
    (search query used: {task.query})

    Numbered sources (the ONLY evidence you may use):
    {source_block}

    Extract the findings that are relevant to the research focus and the
    overall question. Every finding must cite at least one source by its
    number in "source_ids". Do not use knowledge outside the sources. Never
    invent facts, numbers or sources. Report conflicting information in
    "conflicts".

    Reply with ONLY a JSON object in exactly this format:
    {_FORMAT}
    """

    def build(draft):
        return WorkerResult(
            task=task, status="ok", sources=sources, findings=draft.findings,
            summary=draft.summary, conflicts=draft.conflicts, skipped_results=skipped,
        )

    draft = invoke_structured(llm, prompt, _ResearchDraft, "Research", validate=build)
    return build(draft)


def research_worker(payload):
    """Parallel worker node (receives a Send payload, not the full state).

    Never raises: a failure becomes a `failed` WorkerResult so the other
    workers' results are kept.
    """
    task = payload["task"]
    if not isinstance(task, ResearchTask):
        task = ResearchTask.model_validate(task)
    question = payload.get("question", "")

    print(f"\n🔍 Research worker {task.id} ({task.aspect}) is working...")
    logger.info("Research worker %d [%s] started: %r", task.id, task.aspect, task.query)
    started = time.perf_counter()

    try:
        result = run_research_task(task, question)
        result = result.model_copy(update={"duration_s": round(time.perf_counter() - started, 3)})
        logger.info(
            "Research worker %d [%s] finished in %.2fs: %d source(s), %d finding(s), %d skipped result(s)",
            task.id, task.aspect, result.duration_s, len(result.sources),
            len(result.findings), result.skipped_results,
        )
    except Exception as exc:  # noqa: BLE001 - isolate the failure to this worker
        duration = round(time.perf_counter() - started, 3)
        error = str(exc) or type(exc).__name__
        logger.warning("Research worker %d [%s] FAILED after %.2fs: %s", task.id, task.aspect, duration, error)
        logger.debug("Research worker %d traceback", task.id, exc_info=True)
        result = WorkerResult(task=task, status="failed", error=error, duration_s=duration)

    return {"worker_results": [result]}


def research_agent(state):
    """Aggregate the parallel workers' results into ResearchFindings.

    If no workers ran (older callers, or planning disabled) it researches the
    question itself as a single task.
    """
    print("\n🔍 Research Agent is working...")

    question = (state.get("question") or "").strip()
    if not question:
        raise AgentError("Research", "the research question is empty")

    results = require_worker_results(state, "Research")
    if results:
        research_data = merge_results(question, results)
    else:
        task = ResearchTask(id=1, aspect="Overview", query=question)
        started = time.perf_counter()
        single = run_research_task(task, question)  # errors propagate unchanged
        single = single.model_copy(update={"duration_s": round(time.perf_counter() - started, 3)})
        research_data = merge_results(question, [single])

    logger.info(
        "Research: %d source(s), %d finding(s), %d conflict(s)",
        len(research_data.sources), len(research_data.findings), len(research_data.conflicts),
    )

    return {
        "research_data": research_data,
        "research": research_data.to_text(),
        "execution_history": state.get("execution_history", []) + ["research"],
    }
