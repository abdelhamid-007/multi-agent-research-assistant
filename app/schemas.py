"""Pydantic models for the data exchanged between agents.

Each model validates on construction and can render itself as compact text
for prompts. The legacy string fields in the graph state are derived from
these models, so older consumers (UI, PDF export, CLI) keep working.
"""
import re
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from app.errors import StateValidationError

APPROVAL_MIN_SCORE = 7
SNIPPET_LIMIT = 700

Decision = Literal["APPROVED", "REVISE"]
Category = Literal["accuracy", "completeness", "structure", "citations"]
Severity = Literal["major", "minor"]
Confidence = Literal["high", "medium", "low"]


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


# ----------------------------------------------------------------- research

class Source(BaseModel):
    """One retrieved web source. Metadata comes from the search tool only."""

    id: int = Field(ge=1)
    title: str = Field(min_length=1)
    url: str
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    snippet: str = ""
    score: Optional[float] = None
    # Verification: the URL was returned by the search tool and passed URL
    # validation. `found_by` lists the research tasks that returned it.
    verified: bool = True
    found_by: list[int] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def _http_url(cls, v):
        v = (v or "").strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"not a valid http(s) URL: {v!r}")
        return v

    @field_validator("title", "publisher", "published_date", mode="before")
    @classmethod
    def _strip(cls, v):
        return _clean(v) if isinstance(v, str) else v

    @field_validator("snippet", mode="before")
    @classmethod
    def _clip_snippet(cls, v):
        return _clean(v)[:SNIPPET_LIMIT] if isinstance(v, str) else ""

    @property
    def domain(self):
        return urlparse(self.url).netloc.removeprefix("www.")

    def label(self):
        """'Title - Publisher/domain, date'."""
        origin = self.publisher or self.domain
        date = f", {self.published_date}" if self.published_date else ""
        return f"{self.title} - {origin}{date}"


class Finding(BaseModel):
    claim: str = Field(min_length=3)
    evidence: str = ""
    source_ids: list[int] = Field(min_length=1)
    confidence: Confidence = "medium"

    @field_validator("claim", "evidence", mode="before")
    @classmethod
    def _strip(cls, v):
        return _clean(v) if isinstance(v, str) else v


class ResearchFindings(BaseModel):
    question: str = ""
    summary: str = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)
    findings: list[Finding] = Field(min_length=1)
    conflicts: list[str] = Field(default_factory=list)
    # Parallel research bookkeeping (empty for a single-task run).
    task_reports: list["TaskReport"] = Field(default_factory=list)
    duplicate_sources_removed: int = 0
    duplicate_findings_removed: int = 0

    @model_validator(mode="after")
    def _consistent(self):
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate source ids")
        unknown = sorted({i for f in self.findings for i in f.source_ids} - set(ids))
        if unknown:
            raise ValueError(f"findings cite unknown source ids: {unknown}")
        return self

    def source_map(self):
        return {s.id: s for s in self.sources}

    def to_text(self, max_chars=None):
        """Sources first (always kept), then findings (trimmed if needed)."""
        src = ["Sources:"] + [f"[{s.id}] {s.label()} - {s.url}" for s in self.sources]
        body = [f"Summary: {self.summary}"]
        failed = [r.aspect for r in self.task_reports if r.status == "failed"]
        if failed:
            body.append("Coverage gaps (research task failed, not covered): " + "; ".join(failed))
        body.append("Findings:")
        for i, f in enumerate(self.findings, 1):
            ids = ",".join(str(n) for n in f.source_ids)
            line = f"{i}. {f.claim} [sources: {ids}; confidence: {f.confidence}]"
            if f.evidence:
                line += f" Evidence: {f.evidence}"
            body.append(line)
        if self.conflicts:
            body.append("Conflicting information:")
            body += [f"- {c}" for c in self.conflicts]
        text = "\n".join(src) + "\n" + "\n".join(body)
        if max_chars and len(text) > max_chars:
            keep = max_chars - len("\n".join(src)) - 20
            text = "\n".join(src) + "\n" + "\n".join(body)[: max(keep, 200)] + " ..."
        return text


# ------------------------------------------------- parallel research tasks

class ResearchTask(BaseModel):
    """One aspect of the topic, researched by one parallel worker."""

    id: int = Field(ge=1)
    aspect: str = Field(min_length=1)
    query: str = Field(min_length=1)

    @field_validator("aspect", "query", mode="before")
    @classmethod
    def _strip(cls, v):
        return _clean(v) if isinstance(v, str) else v


class ResearchPlan(BaseModel):
    """What the planning LLM returns: distinct tasks (ids are assigned by code)."""

    tasks: list[ResearchTask] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct(self):
        queries = [t.query.lower() for t in self.tasks]
        if len(set(queries)) != len(queries):
            raise ValueError("research task queries must be distinct")
        return self


class WorkerResult(BaseModel):
    """Outcome of one worker. A failed worker carries an error, never findings."""

    task: ResearchTask
    status: Literal["ok", "failed"]
    sources: list[Source] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""
    conflicts: list[str] = Field(default_factory=list)
    skipped_results: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.status == "failed":
            if not self.error:
                raise ValueError("a failed worker result needs an error message")
            if self.sources or self.findings:
                raise ValueError("a failed worker result cannot carry sources or findings")
        else:
            if not self.sources or not self.findings:
                raise ValueError("an ok worker result needs sources and findings")
            unknown = sorted({i for f in self.findings for i in f.source_ids} - {s.id for s in self.sources})
            if unknown:
                raise ValueError(f"findings cite unknown source ids: {unknown}")
        return self


class TaskReport(BaseModel):
    task_id: int
    aspect: str
    query: str
    status: Literal["ok", "failed"]
    n_sources: int = 0
    n_findings: int = 0
    skipped_results: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None


ResearchFindings.model_rebuild()


# ----------------------------------------------------------------- analysis

class AnalysisResult(BaseModel):
    key_considerations: list[str] = Field(min_length=1)
    opportunities: list[str] = Field(min_length=1)
    technical_feasibility: str = Field(min_length=1)
    risks: list[str] = Field(min_length=1)
    implementation_complexity: Literal["low", "medium", "high"] = "medium"
    recommendations: list[str] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("key_considerations", "opportunities", "risks",
                     "recommendations", "assumptions", mode="before")
    @classmethod
    def _clean_items(cls, v):
        if isinstance(v, list):
            return [_clean(x) for x in v if isinstance(x, str) and _clean(x)]
        return v

    def to_text(self):
        def block(title, items):
            return [f"{title}:"] + [f"- {x}" for x in items] if items else []

        lines = block("Key considerations", self.key_considerations)
        lines += block("Business opportunities", self.opportunities)
        lines += [f"Technical feasibility: {self.technical_feasibility}"]
        lines += block("Risks", self.risks)
        lines += [f"Implementation complexity: {self.implementation_complexity}"]
        lines += block("Recommendations", self.recommendations)
        lines += block("Assumptions (not backed by sources)", self.assumptions)
        return "\n".join(lines)


# ------------------------------------------------------------------- writer

CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_ids(markdown):
    """Citation numbers used as [n] in the text, in sorted order."""
    return sorted({int(n) for n in CITATION_RE.findall(markdown or "")})


class WriterOutput(BaseModel):
    title: str = Field(min_length=1)
    markdown: str = Field(min_length=50)  # full report including Sources
    cited_source_ids: list[int] = Field(default_factory=list)
    revision: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    @property
    def word_count(self):
        return len(self.markdown.split())


# ----------------------------------------------------------------- reviewer

class ReviewIssue(BaseModel):
    category: Category
    severity: Severity = "major"
    section: str = ""
    problem: str = Field(min_length=3)
    suggestion: str = ""

    @field_validator("category", "severity", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("section", "problem", "suggestion", mode="before")
    @classmethod
    def _strip(cls, v):
        return _clean(v) if isinstance(v, str) else v


class ReviewFeedback(BaseModel):
    decision: Decision
    score: int = Field(ge=0, le=10)
    summary: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @field_validator("decision", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _consistent(self):
        if self.decision == "REVISE" and not self.issues:
            raise ValueError("a REVISE decision must list at least one issue")
        if self.decision == "APPROVED":
            if any(i.severity == "major" for i in self.issues):
                raise ValueError("an APPROVED decision cannot contain major issues")
            if self.score < APPROVAL_MIN_SCORE:
                raise ValueError(
                    f"an APPROVED decision needs a score of at least {APPROVAL_MIN_SCORE}"
                )
        return self

    @property
    def approved(self):
        return self.decision == "APPROVED"

    def to_text(self):
        """Numbered, human- and writer-readable feedback."""
        lines = [f"DECISION: {self.decision}", f"SCORE: {self.score}/10"]
        if self.summary:
            lines.append(f"SUMMARY: {self.summary}")
        lines.append("ISSUES:")
        if self.issues:
            for n, i in enumerate(self.issues, 1):
                where = f"{i.section} - " if i.section else ""
                fix = f" -> {i.suggestion}" if i.suggestion else ""
                lines.append(f"{n}. [{i.category.upper()}|{i.severity}] {where}{i.problem}{fix}")
        else:
            lines.append("none")
        if self.suggestions:
            lines.append("SUGGESTIONS:")
            lines += [f"- {s}" for s in self.suggestions]
        return "\n".join(lines)


# --------------------------------------------------------------- evaluation

Dimension = Literal[
    "factual_accuracy", "citation_reliability", "research_coverage",
    "logical_structure", "recommendations",
]
FlagSeverity = Literal["critical", "major", "minor"]
DIMENSIONS = (
    "factual_accuracy", "citation_reliability", "research_coverage",
    "logical_structure", "recommendations",
)


class EvalFlag(BaseModel):
    """One finding of the automated evaluation (a problem in the report)."""

    code: str
    severity: FlagSeverity
    dimension: Dimension
    message: str = Field(min_length=1)
    evidence: str = ""
    blocking: bool = False  # a blocking flag prevents approval; critical flags always block

    @model_validator(mode="after")
    def _critical_blocks(self):
        if self.severity == "critical":
            self.blocking = True
        return self


class DimensionScore(BaseModel):
    name: Dimension
    score: float = Field(ge=0, le=100)
    weight: float = Field(gt=0, le=1)
    notes: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Deterministic quality evaluation of one draft.

    Scores measure how well the report is supported by the retrieved sources.
    They do not prove that the claims are true.
    """

    revision: int = Field(default=0, ge=0)
    dimensions: list[DimensionScore]
    overall_score: float = Field(ge=0, le=100)
    flags: list[EvalFlag] = Field(default_factory=list)
    passed: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    summary: str = ""
    method: str = "deterministic evidence-support checks"

    @model_validator(mode="after")
    def _consistent(self):
        names = [d.name for d in self.dimensions]
        if sorted(names) != sorted(DIMENSIONS):
            raise ValueError(f"dimensions must be exactly {list(DIMENSIONS)}, got {names}")
        if abs(sum(d.weight for d in self.dimensions) - 1.0) > 1e-6:
            raise ValueError("dimension weights must sum to 1")
        expected = sum(d.score * d.weight for d in self.dimensions)
        if abs(expected - self.overall_score) > 0.06:
            raise ValueError(f"overall_score {self.overall_score} does not match the weighted "
                             f"dimension scores ({expected:.1f})")
        if self.passed and (self.blocking_reasons or any(f.blocking for f in self.flags)):
            raise ValueError("an evaluation with blocking flags or critical failures cannot pass")
        if not self.passed and not self.blocking_reasons:
            raise ValueError("a failed evaluation must state its blocking reasons")
        return self

    @property
    def critical_failures(self):
        return [f for f in self.flags if f.severity == "critical"]

    def dimension(self, name):
        return next(d for d in self.dimensions if d.name == name)

    def to_text(self, max_flags=6):
        """Compact text for the Writer's feedback and the CLI."""
        lines = [f"EVALUATION: {self.overall_score:.0f}/100 - {'PASSED' if self.passed else 'NOT APPROVABLE'}"
                 f" (revision {self.revision})"]
        lines += [f"- {d.name.replace('_', ' ').capitalize()}: {d.score:.0f}/100 ({d.weight:.0%})"
                  for d in self.dimensions]
        if self.critical_failures:
            lines.append(f"CRITICAL FAILURES: {len(self.critical_failures)}")
        if self.blocking_reasons:
            lines.append("BLOCKING: " + "; ".join(self.blocking_reasons))
        order = {"critical": 0, "major": 1, "minor": 2}
        for f in sorted(self.flags, key=lambda f: order[f.severity])[:max_flags]:
            lines.append(f"[{f.severity}] {f.message}")
        if len(self.flags) > max_flags:
            lines.append(f"... and {len(self.flags) - max_flags} more flag(s)")
        lines.append("Note: scores measure support by the retrieved sources, not proof that claims are true.")
        return "\n".join(lines)


class RevisionRecord(BaseModel):
    """A draft with the review and automated evaluation that judged it."""

    revision: int = Field(ge=0)
    draft: str
    review: Optional[ReviewFeedback] = None
    evaluation: Optional[EvaluationResult] = None


# ------------------------------------------------------- state access helpers

def _require(state, key, model, agent):
    value = state.get(key)
    if value is None:
        raise StateValidationError(agent, f"required field '{key}' is missing from the workflow state")
    if isinstance(value, model):
        return value
    try:
        return model.model_validate(value)
    except Exception as exc:  # noqa: BLE001
        raise StateValidationError(agent, f"field '{key}' is invalid: {exc}") from exc


def require_research(state, agent):
    return _require(state, "research_data", ResearchFindings, agent)


def require_analysis(state, agent):
    return _require(state, "analysis_data", AnalysisResult, agent)


def require_worker_results(state, agent):
    results = state.get("worker_results") or []
    try:
        return [r if isinstance(r, WorkerResult) else WorkerResult.model_validate(r) for r in results]
    except Exception as exc:  # noqa: BLE001
        raise StateValidationError(agent, f"field 'worker_results' is invalid: {exc}") from exc
