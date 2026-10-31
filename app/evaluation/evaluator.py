"""Automated, deterministic evaluation of one report draft."""
import logging

from app.evaluation.checks import (
    check_citation_reliability,
    check_factual_accuracy,
    check_logical_structure,
    check_recommendations,
    check_research_coverage,
)
from app.evaluation.evidence import EvidenceIndex
from app.evaluation.report_parser import parse_draft
from app.evaluation.scoring import LABELS, MIN_DIMENSION_SCORE, PASS_SCORE
from app.schemas import DIMENSIONS, EvaluationResult

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def evaluate_report(draft, research, analysis=None, revision=0):
    """Score a draft against the verified research and return an EvaluationResult.

    `research` is the validated ResearchFindings (its sources are the only
    allowed references). `analysis` is optional and only used to judge whether
    recommendations are grounded.
    """
    parsed = parse_draft(draft)
    evidence = EvidenceIndex(research)

    factual, flags, checked, _ = check_factual_accuracy(parsed, evidence)
    citations, cite_flags = check_citation_reliability(parsed, research, checked)
    coverage, cov_flags = check_research_coverage(parsed, research, evidence)
    structure, struct_flags = check_logical_structure(parsed)
    recs, rec_flags = check_recommendations(parsed, research, analysis)

    scores = {d.name: d for d in (factual, citations, coverage, structure, recs)}
    dimensions = [scores[name] for name in DIMENSIONS]
    overall = round(sum(d.score * d.weight for d in dimensions), 1)
    all_flags = sorted(flags + cite_flags + cov_flags + struct_flags + rec_flags,
                       key=lambda f: _SEVERITY_ORDER[f.severity])

    reasons = []
    critical = [f for f in all_flags if f.severity == "critical"]
    if critical:
        kinds = sorted({f.code.replace("_", " ") for f in critical})
        reasons.append(f"{len(critical)} critical failure(s): {', '.join(kinds)}")
    reasons += [f.message for f in all_flags if f.blocking and f.severity != "critical"]
    low = [d for d in dimensions if d.score < MIN_DIMENSION_SCORE]
    if low:
        reasons.append("dimension(s) below " + f"{MIN_DIMENSION_SCORE:.0f}: "
                       + ", ".join(f"{LABELS[d.name]} ({d.score:.0f})" for d in low))
    if overall < PASS_SCORE:
        reasons.append(f"overall score {overall:.1f} is below the pass mark of {PASS_SCORE:.0f}")

    result = EvaluationResult(
        revision=revision, dimensions=dimensions, overall_score=overall, flags=all_flags,
        passed=not reasons, blocking_reasons=reasons,
        summary=build_summary(overall, dimensions, all_flags, reasons),
    )
    logger.info(
        "Evaluation (revision %d): overall %.1f/100 %s | %s | %d critical, %d major, %d minor flag(s)",
        revision, overall, "PASSED" if result.passed else "FAILED",
        ", ".join(f"{LABELS[d.name]} {d.score:.0f}" for d in dimensions),
        len(critical), sum(f.severity == "major" for f in all_flags),
        sum(f.severity == "minor" for f in all_flags),
    )
    return result


def build_summary(overall, dimensions, flags, reasons):
    best = max(dimensions, key=lambda d: d.score)
    worst = min(dimensions, key=lambda d: d.score)
    counts = {sev: sum(f.severity == sev for f in flags) for sev in _SEVERITY_ORDER}
    verdict = "meets the approval criteria" if not reasons else "cannot be approved"
    parts = [
        f"Overall {overall:.0f}/100: the report {verdict}.",
        f"Strongest: {LABELS[best.name]} ({best.score:.0f}); weakest: {LABELS[worst.name]} ({worst.score:.0f}).",
        f"Flags: {counts['critical']} critical, {counts['major']} major, {counts['minor']} minor.",
    ]
    if reasons:
        parts.append("Blocking: " + "; ".join(reasons) + ".")
    parts.append("Scores measure support by the retrieved sources using deterministic checks; "
                 "they do not prove the claims are true.")
    return " ".join(parts)
