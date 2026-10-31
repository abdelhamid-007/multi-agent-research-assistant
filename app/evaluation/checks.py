"""The five deterministic dimension checks.

Each check returns (DimensionScore, flags). Nothing here calls an LLM, and
"supported" always means "supported by the retrieved sources", never "true".
"""
from app.evaluation.evidence import EvidenceIndex, numbers, overlap, tokens
from app.evaluation.report_parser import REQUIRED_SECTIONS, URL_RE, section_words
from app.evaluation.scoring import (
    CLAIM_CREDIT,
    FINDING_COVERED_OVERLAP,
    MAX_REPORT_WORDS,
    MIN_REPORT_WORDS,
    SOURCE_UTILIZATION_TARGET,
    SUPPORT_OVERLAP,
    THIN_SECTION_WORDS,
    UNCITED_OVERLAP,
    WEIGHTS,
)
from app.research.aggregation import normalize_url
from app.schemas import DimensionScore, EvalFlag

_ACTION_VERBS = frozenset("""
adopt align allocate analyze apply assess audit automate avoid begin build calibrate choose
collaborate combine conduct configure consider create define deploy design develop document
embed enable enforce ensure establish evaluate expand explore focus formalize identify implement
improve integrate introduce invest launch limit maintain measure mitigate monitor negotiate
pilot plan prepare prioritize protect publish reduce require review run scale secure select
set standardize start streamline test track train transition update use validate verify
add prefer favor keep make offer provide reuse strengthen support put take seek compare
prioritise standardise
""".split())


def _clip(text, n=160):
    return text if len(text) <= n else text[: n - 1] + "…"


def _score(name, value, notes=()):
    return DimensionScore(name=name, score=round(max(0.0, min(100.0, value)), 1),
                          weight=WEIGHTS[name], notes=list(notes))


def _norm_url(url):
    return normalize_url(url.rstrip(".,;:*_)"))


# ------------------------------------------------------------ factual accuracy

def classify_sentence(sentence, evidence):
    """Return (status, detail) for one factual sentence."""
    known = [i for i in sentence.cited_ids if i in evidence.source_ids]
    stats = numbers(sentence.text, statistics_only=True)

    if stats:
        missing_everywhere = [n for n in stats if n not in evidence.all_numbers]
        if missing_everywhere:
            return "unsupported_critical", ", ".join(missing_everywhere)
        cited_numbers = evidence.numbers_for(known)
        if known and all(n in cited_numbers for n in stats):
            return "supported", ""
        return ("miscited" if known else "uncited_stat"), ", ".join(stats)

    claim_tokens = tokens(sentence.text)
    if known:
        share = overlap(claim_tokens, evidence.tokens_for(known))
        return ("supported" if share >= SUPPORT_OVERLAP else "weak"), f"{share:.0%}"
    if sentence.cited_ids:  # cites only sources that do not exist
        return "unsupported", "cites no real source"
    share = overlap(claim_tokens, evidence.all_tokens)
    return ("uncited_supported" if share >= UNCITED_OVERLAP else "unsupported"), f"{share:.0%}"


_STATUS_FLAGS = {
    "unsupported_critical": ("critical", "unsupported_critical_claim",
                             "Figure(s) {d} appear in none of the retrieved sources"),
    "miscited": ("major", "miscited_statistic",
                 "Figure(s) {d} are not in the cited source(s), only in other sources"),
    "uncited_stat": ("major", "uncited_statistic", "Statistic(s) {d} are not cited"),
    "unsupported": ("major", "unsupported_claim",
                    "Claim has no support in the retrieved sources ({d})"),
    "weak": ("minor", "weak_support", "Cited source(s) only weakly support this claim ({d} word overlap)"),
    "uncited_supported": ("minor", "uncited_claim", "Claim matches the sources but carries no citation"),
}


# A statistic must be both present in the sources and attributed to the right one.
_BLOCKING_STATUSES = frozenset({"unsupported_critical", "miscited", "uncited_stat"})


def select_factual_sentences(parsed):
    """Sentences checked for support: research findings, anything cited, any statistic."""
    return [
        s for s in parsed.sentences
        if s.section == "Research Findings" or s.cited_ids or numbers(s.text, statistics_only=True)
    ]


def check_factual_accuracy(parsed, evidence):
    checked = select_factual_sentences(parsed)
    flags, credit = [], 0.0
    statuses = {}
    for s in checked:
        status, detail = classify_sentence(s, evidence)
        statuses[id(s)] = status
        credit += CLAIM_CREDIT[status]
        if status in _STATUS_FLAGS:
            severity, code, template = _STATUS_FLAGS[status]
            flags.append(EvalFlag(code=code, severity=severity, dimension="factual_accuracy",
                                  message=template.format(d=detail), evidence=_clip(s.text),
                                  blocking=status in _BLOCKING_STATUSES))

    if not checked:
        flags.append(EvalFlag(code="no_verifiable_claims", severity="major", dimension="factual_accuracy",
                              message="The report contains no factual claims that can be checked against sources"))
        return _score("factual_accuracy", 0, ["no checkable claims"]), flags, checked, statuses

    supported = sum(1 for s in checked if statuses[id(s)] == "supported")
    notes = [f"{supported}/{len(checked)} claims fully supported by their cited sources"]
    return _score("factual_accuracy", 100 * credit / len(checked), notes), flags, checked, statuses


# -------------------------------------------------------- citation reliability

def check_citation_reliability(parsed, research, checked):
    by_id = research.source_map()
    verified_urls = {_norm_url(s.url): s.id for s in research.sources}
    flags, fabricated = [], 0

    for n in sorted(parsed.marker_ids - set(by_id)):
        fabricated += 1
        flags.append(EvalFlag(code="fabricated_citation", severity="critical", dimension="citation_reliability",
                              message=f"Citation [{n}] does not match any retrieved source"))

    for url in dict.fromkeys(_norm_url(u) for u in parsed.urls):
        if url not in verified_urls:
            fabricated += 1
            flags.append(EvalFlag(code="fabricated_reference", severity="critical", dimension="citation_reliability",
                                  message=f"URL is not one of the verified sources: {url}"))

    for number, url in parsed.source_entries:
        if number in by_id and _norm_url(url) in verified_urls and verified_urls[_norm_url(url)] != number:
            fabricated += 1
            flags.append(EvalFlag(code="misattributed_reference", severity="critical", dimension="citation_reliability",
                                  message=f"Reference [{number}] lists the URL of source [{verified_urls[_norm_url(url)]}]"))

    cited = {i for i in parsed.marker_ids if i in by_id}
    penalty = 0.0
    for i in sorted(cited):
        if not by_id[i].verified:
            penalty += 0.25
            flags.append(EvalFlag(code="unverified_source_cited", severity="major", dimension="citation_reliability",
                                  message=f"Source [{i}] was not verified as a search-tool result"))
    listed = {n for n, _ in parsed.source_entries if n in by_id}
    if cited and not parsed.source_entries:
        penalty += 0.2
        flags.append(EvalFlag(code="missing_sources_section", severity="major", dimension="citation_reliability",
                              message="The report cites sources but has no Sources section listing them"))
    elif cited - listed:
        flags.append(EvalFlag(code="source_not_listed", severity="minor", dimension="citation_reliability",
                              message=f"Cited source(s) missing from the Sources list: {sorted(cited - listed)}"))
    if listed - cited:
        flags.append(EvalFlag(code="listed_source_not_cited", severity="minor", dimension="citation_reliability",
                              message=f"Listed but never cited in the text: {sorted(listed - cited)}"))

    cited_claims = sum(1 for s in checked if any(i in by_id for i in s.cited_ids))
    rate = cited_claims / len(checked) if checked else 0.0
    if checked and not cited_claims:
        flags.append(EvalFlag(code="no_citations", severity="major", dimension="citation_reliability", blocking=True,
                              message="The report contains no citations to the retrieved sources"))
    elif checked and cited_claims < len(checked):
        flags.append(EvalFlag(code="missing_citations", severity="major", dimension="citation_reliability",
                              message=f"{len(checked) - cited_claims} of {len(checked)} factual claims carry no citation"))

    integrity = max(0.0, 1.0 - 0.5 * fabricated - penalty)
    notes = [f"{cited_claims}/{len(checked)} factual claims cited",
             f"{fabricated} fabricated or misattributed reference(s)"]
    return _score("citation_reliability", 100 * (0.6 * rate + 0.4 * integrity), notes), flags


# ------------------------------------------------------------ research coverage

def check_research_coverage(parsed, research, evidence):
    body_tokens = set().union(*(tokens(s.text) for s in parsed.sentences)) if parsed.sentences else set()
    cited = {i for i in parsed.marker_ids if i in evidence.source_ids}
    flags, parts = [], []   # parts: (weight, value, note)

    covered, uncovered = 0, []
    for f in research.findings:
        t = tokens(f.claim)
        if overlap(t, body_tokens) >= FINDING_COVERED_OVERLAP:
            covered += 1
        else:
            uncovered.append(f)
    share = covered / len(research.findings)
    parts.append((0.5, share, f"{covered}/{len(research.findings)} research findings reflected in the report"))
    for f in [f for f in uncovered if f.confidence == "high"][:3]:
        flags.append(EvalFlag(code="finding_not_covered", severity="minor", dimension="research_coverage",
                              message="A high-confidence research finding is missing from the report",
                              evidence=_clip(f.claim)))

    ok_tasks = [r for r in research.task_reports if r.status == "ok"]
    if len(ok_tasks) >= 2:
        source_tasks = {s.id: set(s.found_by) for s in research.sources}
        reached = {t for i in cited for t in source_tasks.get(i, ())}
        done = 0
        for r in ok_tasks:
            if r.task_id in reached or overlap(tokens(r.aspect), body_tokens) >= 0.5:
                done += 1
            else:
                flags.append(EvalFlag(code="aspect_not_covered", severity="minor", dimension="research_coverage",
                                      message=f"Research aspect '{r.aspect}' is not reflected in the report"))
        parts.append((0.25, done / len(ok_tasks), f"{done}/{len(ok_tasks)} research aspects covered"))

    utilisation = len(cited) / len(research.sources)
    parts.append((0.25, min(1.0, utilisation / SOURCE_UTILIZATION_TARGET),
                  f"{len(cited)}/{len(research.sources)} sources cited"))

    failed = [r.aspect for r in research.task_reports if r.status == "failed"]
    if failed:
        flags.append(EvalFlag(code="research_gap", severity="minor", dimension="research_coverage",
                              message="Not researched (task failed): " + "; ".join(failed)))

    total = sum(w for w, _, _ in parts)
    value = 100 * sum(w * v for w, v, _ in parts) / total
    return _score("research_coverage", value, [n for _, _, n in parts]), flags


# ------------------------------------------------------------ logical structure

def check_logical_structure(parsed):
    flags, present = [], parsed.present
    missing = [s for s in REQUIRED_SECTIONS if s not in present]
    section_share = (len(REQUIRED_SECTIONS) - len(missing)) / len(REQUIRED_SECTIONS)
    if missing:
        flags.append(EvalFlag(code="incomplete_report", severity="major", dimension="logical_structure",
                              blocking=True,
                              message="Required section(s) missing: " + ", ".join(missing)))

    order = [REQUIRED_SECTIONS.index(s) for s in present]
    in_order = order == sorted(order)
    if not in_order:
        flags.append(EvalFlag(code="sections_out_of_order", severity="minor", dimension="logical_structure",
                              message="Sections are not in the expected order"))

    body = parsed.body_words
    if body < MIN_REPORT_WORDS:
        length = body / MIN_REPORT_WORDS
        flags.append(EvalFlag(code="report_too_short", severity="major", dimension="logical_structure",
                              message=f"Report body is only {body} words (minimum {MIN_REPORT_WORDS})"))
    elif body > MAX_REPORT_WORDS:
        length = max(0.5, 1 - (body - MAX_REPORT_WORDS) / 1000)
        flags.append(EvalFlag(code="report_too_long", severity="minor", dimension="logical_structure",
                              message=f"Report body is {body} words (limit about {MAX_REPORT_WORDS})"))
    else:
        length = 1.0

    content = [s for s in present if s != "Sources"]
    thin = [s for s in content if section_words(parsed, s) < THIN_SECTION_WORDS]
    for s in thin:
        flags.append(EvalFlag(code="thin_section", severity="minor", dimension="logical_structure",
                              message=f"Section '{s}' has fewer than {THIN_SECTION_WORDS} words"))
    full_share = (len(content) - len(thin)) / len(content) if content else 0.0

    value = 100 * (0.55 * section_share + 0.15 * (1.0 if in_order else 0.6) + 0.15 * length + 0.15 * full_share)
    notes = [f"{len(REQUIRED_SECTIONS) - len(missing)}/{len(REQUIRED_SECTIONS)} required sections present",
             f"{body} words"]
    return _score("logical_structure", value, notes), flags


# -------------------------------------------------------------- recommendations

def _is_actionable(item):
    words = [w.strip(".,;:()").lower() for w in item.split()]
    if not words:
        return False
    first = words[0]
    return (first in _ACTION_VERBS or first.rstrip("s") in _ACTION_VERBS
            or any(w in ("should", "must", "need", "needs", "recommend") for w in words[:6]))


def check_recommendations(parsed, research, analysis):
    items = [i for i in parsed.items.get("Recommendations", []) if len(i.split()) >= 3]
    if not items:
        flag = EvalFlag(code="no_recommendations", severity="major", dimension="recommendations",
                        message="The report has no usable Recommendations section")
        return _score("recommendations", 0, ["no recommendations"]), [flag]

    reference = tokens(" ".join(f"{f.claim} {f.evidence}" for f in research.findings))
    if analysis is not None:
        reference |= tokens(" ".join(analysis.recommendations + analysis.opportunities
                                     + analysis.risks + analysis.key_considerations))
    actionable = sum(_is_actionable(i) for i in items)
    grounded = sum(len(tokens(i) & reference) >= 2 for i in items)

    flags = []
    if len(items) < 3:
        flags.append(EvalFlag(code="few_recommendations", severity="minor", dimension="recommendations",
                              message=f"Only {len(items)} recommendation(s); at least 3 are expected"))
    if actionable < len(items):
        flags.append(EvalFlag(code="vague_recommendations", severity="minor", dimension="recommendations",
                              message=f"{len(items) - actionable} recommendation(s) are not phrased as actions"))
    if grounded < len(items):
        flags.append(EvalFlag(code="ungrounded_recommendations", severity="minor", dimension="recommendations",
                              message=f"{len(items) - grounded} recommendation(s) do not connect to the research or analysis"))

    value = 100 * (0.35 * min(1.0, len(items) / 3) + 0.35 * actionable / len(items) + 0.30 * grounded / len(items))
    notes = [f"{len(items)} recommendation(s)", f"{actionable} actionable", f"{grounded} grounded in the research"]
    return _score("recommendations", value, notes), flags
