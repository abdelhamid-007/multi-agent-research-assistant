"""Shared fakes: scripted LLMs and canned, valid structured agent output."""
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

URL = "https://example.com/agents"
URL2 = "https://example.org/report?a=1"

SEARCH_RESULTS = {
    "results": [
        {"url": URL, "title": "Agent trends", "content": "Agents are growing fast.",
         "published_date": "2026-03-01", "score": 0.9},
        {"url": URL2, "title": "Governance report", "content": "Governance matters.",
         "score": 0.7},
    ]
}

RESEARCH_JSON = json.dumps({
    "summary": "AI agents are growing and governance is a priority.",
    "findings": [
        {"claim": "Agent adoption is growing", "evidence": "Growing fast.",
         "source_ids": [1], "confidence": "high"},
        {"claim": "Governance is a priority", "evidence": "Governance matters.",
         "source_ids": [2], "confidence": "medium"},
    ],
    "conflicts": [],
})

ANALYSIS_JSON = json.dumps({
    "key_considerations": ["Adoption is rising"],
    "opportunities": ["Governance tooling"],
    "technical_feasibility": "Feasible with existing frameworks.",
    "risks": ["Vendor lock-in"],
    "implementation_complexity": "medium",
    "recommendations": ["Pilot a small agent team"],
    "assumptions": ["Budget is available"],
})

WRITER_BODY = """## Executive Summary
Agent adoption is growing fast [1]. Governance is a priority for organizations that deploy agents [2]. This report summarizes the evidence and recommends a cautious pilot.

## Research Findings
Agent adoption is growing, and agents are growing fast across industries [1]. Governance matters, and governance is a priority for teams that run agents in production [2].

## Business Opportunities
Governance tooling is a business opportunity because governance matters to buyers [2]. Teams that adopt agents early can gain an advantage while adoption keeps growing [1].

## Technical Analysis
Existing frameworks make agents feasible to build today. Implementation complexity is medium and needs monitoring and governance controls from the start.

## Risks
Vendor lock-in is a risk when teams adopt a single agent platform. Weak governance can expose organizations to compliance problems.

## Recommendations
- Pilot a small agent team and measure the results.
- Establish governance controls before scaling agents.
- Review vendor lock-in risk and keep the tooling portable.

## Conclusion
Agents are here to stay, adoption is growing fast [1], and governance is a priority [2]."""


PLAN_JSON = json.dumps({"tasks": [
    {"id": 1, "aspect": "Adoption", "query": "ai agents adoption"},
    {"id": 2, "aspect": "Technology", "query": "ai agents technology"},
    {"id": 3, "aspect": "Risks", "query": "ai agents risks"},
]})


def review_json(decision="APPROVED", score=9, issues=None, summary="ok"):
    return json.dumps({
        "decision": decision, "score": score, "summary": summary,
        "issues": issues or [], "suggestions": [],
    })


def issue(problem="Add detail", category="completeness", severity="major"):
    return {"category": category, "severity": severity, "section": "Risks",
            "problem": problem, "suggestion": "Expand it"}


APPROVED = review_json()
REVISE = review_json("REVISE", 5, [issue()])


class FakeLLM:
    """Returns scripted replies (the last one repeats); records prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return SimpleNamespace(content=reply)


class Fakes:
    """Patches every agent's LLM and the search tool for one test."""

    def __init__(self, reviewer_replies, writer_replies=None, research_llm=None,
                 analysis_llm=None, search=SEARCH_RESULTS, extra=(), planner_llm=None):
        self.planner = planner_llm or FakeLLM([PLAN_JSON])
        self.research = research_llm or FakeLLM([RESEARCH_JSON])
        self.analysis = analysis_llm or FakeLLM([ANALYSIS_JSON])
        self.writer = FakeLLM(writer_replies or [WRITER_BODY])
        self.reviewer = FakeLLM(reviewer_replies)
        self.search = search
        self.extra = extra

    def __enter__(self):
        self.stack = ExitStack()
        for target, value in [
            ("app.research.planner.llm", self.planner),
            ("app.agents.research_agent.llm", self.research),
            ("app.agents.analysis_agent.llm", self.analysis),
            ("app.agents.writer.llm", self.writer),
            ("app.agents.reviewer.llm", self.reviewer),
        ]:
            self.stack.enter_context(patch(target, value))
        search_patch = (patch("app.agents.research_agent.search_web", side_effect=self.search)
                        if callable(self.search)
                        else patch("app.agents.research_agent.search_web", return_value=self.search))
        self.stack.enter_context(search_patch)
        self.stack.enter_context(patch("app.retry._sleep"))
        for p in self.extra:
            self.stack.enter_context(p)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)
