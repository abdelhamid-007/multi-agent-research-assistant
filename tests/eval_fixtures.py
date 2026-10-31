import re
from app.agents.writer import assemble_report, build_sources_section
from app.schemas import AnalysisResult, Finding, ResearchFindings, Source

SOURCES = [
    Source(id=1, title="AI agent adoption survey", url="https://example.com/agents", published_date="2026-03-01",
           snippet="Enterprise adoption of AI agents grew 45% in 2025. Orchestration layers coordinate multiple specialised agents. Cost controls are a growing concern."),
    Source(id=2, title="Governance report", url="https://example.org/governance",
           snippet="Governance is now a top priority. 60% of surveyed firms added audit logging for agents. Vendor lock-in worries buyers."),
    Source(id=3, title="Platform comparison", url="https://example.net/platforms",
           snippet="Open standards reduce integration effort. Teams report faster pilots when they reuse existing frameworks."),
]
FINDINGS = [
    Finding(claim="Enterprise adoption of AI agents grew 45% in 2025", evidence="adoption grew 45% in 2025", source_ids=[1], confidence="high"),
    Finding(claim="Multi-agent orchestration layers are common", evidence="Orchestration layers coordinate multiple agents", source_ids=[1]),
    Finding(claim="Governance is a top priority, with 60% of firms adding audit logging", evidence="60% of surveyed firms added audit logging", source_ids=[2], confidence="high"),
    Finding(claim="Open standards reduce integration effort", evidence="faster pilots with existing frameworks", source_ids=[3]),
]
RESEARCH = ResearchFindings(question="How are AI agents evolving?", summary="Agents are spreading.", sources=SOURCES, findings=FINDINGS)
ANALYSIS = AnalysisResult(
    key_considerations=["Adoption is rising"], opportunities=["Governance and audit tooling"],
    technical_feasibility="Feasible with current frameworks.", risks=["Vendor lock-in"],
    recommendations=["Pilot a governed multi-agent workflow", "Add audit logging", "Prefer open standards"],
)

GOOD_BODY = """## Executive Summary
Enterprise adoption of AI agents grew 45% in 2025 [1]. Governance is a top priority for buyers [2]. This report recommends a governed, standards-based pilot.

## Research Findings
Enterprise adoption of AI agents grew 45% in 2025 [1]. Orchestration layers coordinate multiple specialised agents [1]. Governance is a top priority, and 60% of surveyed firms added audit logging [2]. Open standards reduce integration effort [3].

## Business Opportunities
Audit logging and governance tooling address a priority that buyers already rank highly [2]. Standards-based integration kits can shorten pilots for teams that reuse existing frameworks [3].

## Technical Analysis
Orchestration layers make multi-agent systems feasible with current frameworks [1]. Complexity is medium because teams must add cost controls and audit logging from the start.

## Risks
Vendor lock-in worries buyers [2]. Cost controls remain a growing concern as agent usage expands [1].

## Recommendations
- Pilot a governed multi-agent workflow using existing frameworks.
- Add audit logging before scaling agent usage.
- Prefer open standards to limit vendor lock-in.

## Conclusion
Adoption is growing quickly [1], governance is a priority [2], and open standards help teams move faster [3]."""


def report(body=GOOD_BODY):
    """Body plus the generated Sources section, exactly as the Writer assembles it."""
    return assemble_report(body, RESEARCH.source_map())[0]


def without_citations(body=GOOD_BODY):
    return re.sub(r"\s?\[\d+\]", "", body)


def fabricated_report():
    """A hand-built draft (bypassing the Writer's assembly, which would strip bad markers):
    cites a non-existent source [9], links an invented URL, and mislabels a reference."""
    body = GOOD_BODY.replace("[3]", "[9]") + "\nFurther detail: https://made-up.example/paper."
    sources = (
        "## Sources\n"
        "- [1] AI agent adoption survey - example.com - https://example.com/agents\n"
        "- [2] Governance report - example.org - https://example.com/agents\n"   # source 2 listed with source 1's URL
        "- [9] Invented study - made-up.example - https://made-up.example/paper"
    )
    return f"{body}\n\n{sources}"
