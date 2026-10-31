# LangGraph Multi-Agent Research Assistant

A Python prototype that runs a research → analysis → writing → review workflow using LangGraph, with a Streamlit web interface and PDF report export.

## Agents
- **Supervisor**: deterministic routing based on workflow state.
- **Research**: retrieves web search results with Tavily and summarizes them with source URLs.
- **Analysis**: analyzes the collected research.
- **Writer**: creates or revises a business report, citing only URLs found in the research.
- **Reviewer**: checks factual accuracy, completeness, structure and citations, then approves or requests revisions.

`app/agents/planner.py` is an unused standalone experiment; it is not part of the active graph.

## How the workflow runs
1. Supervisor routes to Research, then Analysis, then Writer.
2. Every draft goes to the Reviewer, including the last permitted revision.
3. If the Reviewer returns `REVISE`, the Writer receives the complete numbered feedback and revises the report.
4. The run ends when the report is approved, or when the revision limit (default 3) is reached with the last draft still unapproved.
5. After the workflow, the final report is exported to PDF in `output/`.

Final statuses: `approved` or `max_revisions_reached`. An unapproved report is never reported as a success, and its PDF cover carries a "not approved" warning.

## Parallel research
The Supervisor plans several research tasks (aspects of the topic, each with a search query) and LangGraph's `Send` API fans them out to `research_worker` nodes that run at the same time. The Research agent then aggregates the results:

```
supervisor -> research_worker x N (parallel) -> research (aggregate) -> supervisor
```
- Each worker searches, builds sources from the search results and extracts findings on its own. A worker never raises: a failure becomes a `failed` result, the other workers' results are kept, and the gap is noted for the Analysis and Writer agents. If every worker fails, the run stops with a clear error.
- Duplicate URLs (ignoring `www`, scheme, fragments, trailing slashes and tracking parameters) and near-identical findings (same wording and the same numbers) are merged. Citations are remapped to the merged source numbers, and each source records which tasks found it (`found_by`) and that it came from the search tool (`verified`).
- Each worker logs its start, duration, and source/finding counts.

Settings (environment variables, also available as function arguments):

| Variable | Meaning | Default |
|---|---|---|
| `RESEARCH_MAX_TASKS` | research tasks the supervisor plans (1-8) | 4 |
| `RESEARCH_MAX_CONCURRENCY` | workers allowed to run at once (1-8) | 4 |

Parallel research uses roughly one LLM call per task, so on a free Groq tier it spends the token budget faster. Set `RESEARCH_MAX_TASKS=1` for a single search.

## Automated evaluation
Every draft is scored by deterministic checks (`app/evaluation/`) before the LLM reviewer sees it. The scores measure how well the report is **supported by the retrieved sources**; they do not prove that the claims are true.

| Dimension | Weight | What is checked |
|---|---|---|
| Factual accuracy | 30% | each factual claim against the cited sources: statistics must appear in the source text, other claims must overlap with it |
| Citation reliability | 25% | share of factual claims that are cited; fabricated `[n]` markers, URLs and mismatched references |
| Research coverage | 20% | research findings, research aspects and sources reflected in the report |
| Logical structure | 15% | the 8 required sections, their order, length, thin sections |
| Recommendations | 10% | number, actionability and grounding in the research and analysis |

Each dimension is scored 0-100 and combined into a weighted overall score. A draft **cannot be approved** if it has any of:
- a critical failure: a fabricated citation or URL, a mismatched reference, or a statistic that appears in no source;
- another blocking flag: no citations at all, a miscited or uncited statistic, or a missing required section;
- an overall score below 70, or any dimension below 40.

The Reviewer runs the evaluation first. A failing draft is sent back to the Writer with the problems listed, without an LLM call. A passing draft goes to the LLM for a qualitative second opinion, its score is capped by the evaluation, and approval needs both. Results are kept in the state (`latest_evaluation`, `evaluation_history`, and on each `RevisionRecord`), printed by the CLI and shown in the web UI.

Limits: it only knows the source snippets the search tool returned (up to 700 characters each), so a true claim from a longer page can be flagged as unsupported. Word-overlap checks can miss a paraphrase that changes the meaning, which is what the LLM review is for.

## Structured agent communication
Agents exchange validated Pydantic models (`app/schemas.py`) stored in the graph state:

| State field | Model | Produced by |
|---|---|---|
| `research_data` | `ResearchFindings` (`Source`s with title, URL, publisher, date; `Finding`s that cite source ids) | Research |
| `analysis_data` | `AnalysisResult` | Analysis |
| `writer_output` | `WriterOutput` (markdown, cited source ids, warnings) | Writer |
| `latest_review` | `ReviewFeedback` (decision, score 0-10, issues, suggestions) | Reviewer |
| `revision_history` | `RevisionRecord`s linking every draft to the review of it | Writer + Reviewer |

- Source metadata comes from the search results, never from the LLM. The Writer cites sources as `[n]`, and the **Sources section is generated from the real `Source` objects**, so references cannot be invented.
- LLM replies are parsed as JSON and validated (`app/structured.py`). An invalid reply gets one repair attempt, then a clear `AgentError`.
- Each agent validates the data it receives and raises `StateValidationError` if it is missing or malformed.
- The Reviewer approves only with a score of 7 or more and no major issues, and its deterministic citation checks can override an LLM approval.
- `validate_state()` checks the whole shared state for contradictions (for example `approved` without an approving review); the runner logs any problems after every node.
- The plain-text fields (`research`, `analysis`, `draft`, `feedback`, `review_history`) are still filled in for the UI, PDF export and CLI.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your own `GROQ_API_KEY` and `TAVILY_API_KEY` values to `.env`.

## Run

From the project folder (`multi-agent-research-assistant`).

**Web interface**

```powershell
streamlit run streamlit_app.py
```

Open http://localhost:8501, enter a topic and click **Start research**. The page shows each agent's live status and latest result, then the report preview, a **Download PDF** button, and a history of past runs in the sidebar.

**Command line**

```powershell
python -m app.main
```

Runs the built-in example topic, prints the report and status, and saves the PDF.

If emoji print as garbled text in the Windows console, set `$env:PYTHONUTF8=1` first.

## PDF reports
- Saved in `output/` as `<topic-slug>_<timestamp>.pdf`; existing files are never overwritten.
- Cover page, Executive Summary, Research Findings, Analysis, Business Opportunities, Recommendations, Conclusion, and References, with tables, headings, bullets and page numbers.
- References contain only URLs that appear in the report's Sources and in the retrieved research. Nothing is invented. A missing section is labelled as not included.
- If formatting fails, a plain-text PDF is written instead; export failures never break the workflow.

## Project layout
```
streamlit_app.py            Streamlit entry point
app/main.py                 CLI entry point
app/agents/                 supervisor, research, analysis, writer, reviewer
app/workflows/              graph.py (LangGraph) and state.py
app/schemas.py              Pydantic models for agent data
app/structured.py           validated JSON from LLM replies
app/state_validation.py     whole-state consistency checks
app/research/               task planner and result aggregation for parallel research
app/config.py               parallel research settings
app/evaluation/             automated report evaluation (scoring, checks, evaluator)
app/errors.py               AgentError types
app/runner.py               streams the graph for the UI; agent status tracking
app/reports/                markdown parser and ReportLab PDF export
app/ui/components.py        Streamlit UI functions
app/history.py              JSON history of past runs (output/history.json)
app/retry.py                retry with backoff for LLM and search calls
tests/                      unit and workflow tests
```

## Tests

```powershell
python -m unittest tests.test_supervisor_routing tests.test_schemas tests.test_state_validation tests.test_agent_communication tests.test_parallel_research tests.test_evaluation tests.test_review_loop tests.test_pdf_report tests.test_streamlit_app tests.test_retry
```

The tests use fake LLMs and need no API keys. They cover routing, the revision loop, PDF export, the Streamlit flow (topic → agents → approval → PDF download) and retry behaviour.

## Reliability
- LLM and search calls retry transient failures (rate limits, 5xx, timeouts) up to 4 attempts with exponential backoff, honouring the provider's "try again in Xs" hint. Bad keys and invalid requests fail immediately.
- Revisions, reviewer decisions and retries are logged.

## Limitations
- The Reviewer is an LLM-based quality check. It sees only the retrieved research, so it cannot verify facts beyond it. A deterministic check does reject cited URLs that are not in the research.
- Research and report prompts are truncated to fit the Groq free tier (8000 tokens per minute). Very long research may be cut off.
- In the web UI, closing or refreshing the page mid-run abandons that run, and a run that fails midway keeps no partial draft.
- The UI has no login; history is a shared local file, so it is intended for local use.
- Never commit `.env`, API keys, or `.venv`.
