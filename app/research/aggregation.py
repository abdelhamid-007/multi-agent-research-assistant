"""Merge the results of parallel research workers into one ResearchFindings.

Rules: nothing is invented. Sources keep their search-tool metadata, findings
keep their original citations (remapped to the merged source numbers), and
failed tasks are reported, never papered over.
"""
import logging
import re
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlparse

from app.errors import AgentError
from app.schemas import Finding, ResearchFindings, TaskReport

logger = logging.getLogger(__name__)

MAX_TOTAL_SOURCES = 12
SIMILARITY_THRESHOLD = 0.85
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "ref_src")


def normalize_url(url):
    """Key for URL de-duplication: ignores scheme, www, fragment, trailing slash, tracking params."""
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parsed.query)
        if not k.lower().startswith(_TRACKING_PARAMS)
    ))
    return f"{host}{path}" + (f"?{query}" if query else "")


def normalize_text(text):
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (text or "").lower())).strip()


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _similar(a, b):
    """Near-identical wording AND identical numbers (10% and 12% are different facts)."""
    if sorted(_NUMBER_RE.findall(a)) != sorted(_NUMBER_RE.findall(b)):
        return False
    a, b = normalize_text(a), normalize_text(b)
    return a == b or SequenceMatcher(None, a, b).ratio() >= SIMILARITY_THRESHOLD


def merge_results(question, results, max_sources=MAX_TOTAL_SOURCES):
    """Aggregate WorkerResults into a validated ResearchFindings.

    Raises AgentError if no worker succeeded. Results are processed in task
    order, so the output does not depend on which worker finished first.
    """
    results = sorted(results, key=lambda r: r.task.id)
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        reasons = "; ".join(f"[{r.task.aspect}] {r.error}" for r in results) or "no research tasks ran"
        raise AgentError("Research", f"all {len(results)} research task(s) failed: {reasons}")

    # ---- sources: merge duplicates by normalized URL, keep all metadata
    merged, order, local = {}, [], {}   # key -> Source; first-seen order; (task, local id) -> key
    occurrences = 0
    for r in ok:
        for src in r.sources:
            occurrences += 1
            key = normalize_url(src.url)
            local[(r.task.id, src.id)] = key
            if key not in merged:
                merged[key] = src.model_copy(update={"found_by": [r.task.id]})
                order.append(key)
                continue
            have = merged[key]
            merged[key] = have.model_copy(update={
                "found_by": sorted(set(have.found_by) | {r.task.id}),
                "publisher": have.publisher or src.publisher,
                "published_date": have.published_date or src.published_date,
                "snippet": have.snippet or src.snippet,
                "score": max(have.score, src.score) if None not in (have.score, src.score)
                         else (have.score if src.score is None else src.score),
                "verified": have.verified and src.verified,
            })
    duplicate_sources = occurrences - len(merged)

    # sources found by several tasks first, then first-seen order; cap the total
    ranked = sorted(order, key=lambda k: (-len(merged[k].found_by), order.index(k)))[:max_sources]
    new_id = {key: n for n, key in enumerate(ranked, 1)}
    sources = [merged[key].model_copy(update={"id": new_id[key]}) for key in ranked]
    dropped_sources = len(order) - len(ranked)
    if dropped_sources:
        logger.info("Research: dropped %d source(s) over the limit of %d", dropped_sources, max_sources)

    # ---- findings: remap citations, drop uncitable, merge near-duplicates
    findings, duplicate_findings = [], 0
    for r in ok:
        for f in r.findings:
            ids = sorted({new_id[local[(r.task.id, i)]] for i in f.source_ids
                          if local[(r.task.id, i)] in new_id})
            if not ids:
                continue  # every cited source was dropped; never keep an uncited claim
            candidate = f.model_copy(update={"source_ids": ids})
            twin = next((k for k, kept in enumerate(findings) if _similar(kept.claim, candidate.claim)), None)
            if twin is None:
                findings.append(candidate)
                continue
            duplicate_findings += 1
            kept = findings[twin]
            findings[twin] = kept.model_copy(update={
                "source_ids": sorted(set(kept.source_ids) | set(ids)),
                "confidence": max(kept.confidence, candidate.confidence, key=_CONFIDENCE_RANK.get),
                "evidence": max(kept.evidence, candidate.evidence, key=len),
            })
    if not findings:
        raise AgentError("Research", "the research tasks returned no findings that cite a usable source")

    conflicts, seen = [], set()
    for r in ok:
        for c in r.conflicts:
            if normalize_text(c) not in seen:
                seen.add(normalize_text(c))
                conflicts.append(c)

    summary = " ".join(f"[{r.task.aspect}] {r.summary}" for r in ok if r.summary) or "See findings."

    reports = [
        TaskReport(
            task_id=r.task.id, aspect=r.task.aspect, query=r.task.query, status=r.status,
            n_sources=len(r.sources), n_findings=len(r.findings),
            skipped_results=r.skipped_results, duration_s=r.duration_s, error=r.error,
        )
        for r in results
    ]

    data = ResearchFindings(
        question=question, summary=summary, sources=sources, findings=findings,
        conflicts=conflicts, task_reports=reports,
        duplicate_sources_removed=duplicate_sources, duplicate_findings_removed=duplicate_findings,
    )
    logger.info(
        "Research aggregate: %d/%d task(s) ok -> %d source(s) (%d duplicate URL(s) removed), "
        "%d finding(s) (%d duplicate(s) merged)",
        len(ok), len(results), len(sources), duplicate_sources, len(findings), duplicate_findings,
    )
    return data
