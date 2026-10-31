"""Streamlit UI building blocks for the AI Research Assistant.

All Streamlit calls live here; workflow logic is in app.runner,
history in app.history and PDF export in app.reports.
"""
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.history import add_entry, load_history, make_entry
from app.reports import export_report_pdf
from app.runner import (
    AGENTS,
    DEFAULT_MAX_REVISIONS,
    DONE,
    ERROR,
    IDLE,
    RUNNING,
    AgentTracker,
    friendly_error,
    stream_research,
    summarize_update,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MIN_TOPIC_LENGTH = 5
MAX_TOPIC_LENGTH = 300

AGENT_META = {
    "supervisor": ("Supervisor", ":material/hub:"),
    "research": ("Research", ":material/search:"),
    "analysis": ("Analysis", ":material/analytics:"),
    "writer": ("Writer", ":material/edit_note:"),
    "reviewer": ("Reviewer", ":material/fact_check:"),
    "research_worker": ("Research worker", ":material/travel_explore:"),
}
STATUS_BADGE = {
    IDLE: ":gray-badge[Idle]",
    RUNNING: ":orange-badge[Running]",
    DONE: ":green-badge[Done]",
    ERROR: ":red-badge[Error]",
}


# ---------------------------------------------------------------- state ----

def fresh_agent_status():
    return AgentTracker().status


def init_session_state():
    defaults = {
        "running": False,
        "active_topic": "",
        "max_revisions": DEFAULT_MAX_REVISIONS,
        "agent_status": fresh_agent_status(),
        "activity": [],
        "result": None,
        "error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def start_clicked():
    """Form callback: validate and arm a run. Ignored if one is running."""
    ss = st.session_state
    if ss.running:
        return  # duplicate submission guard
    topic = (ss.get("topic_input") or "").strip()
    if len(topic) < MIN_TOPIC_LENGTH:
        ss.error = f"Please enter a research topic (at least {MIN_TOPIC_LENGTH} characters)."
        return
    ss.update(
        running=True,
        active_topic=topic[:MAX_TOPIC_LENGTH],
        error="",
        result=None,
        activity=[],
        agent_status=fresh_agent_status(),
    )


# -------------------------------------------------------------- sidebar ----

def _key_status():
    load_dotenv(PROJECT_DIR / ".env")
    return {k: bool(os.getenv(k)) for k in ("GROQ_API_KEY", "TAVILY_API_KEY")}


def _open_history_entry(entry):
    st.session_state.result = entry
    st.session_state.error = ""


def render_sidebar():
    running = st.session_state.running
    with st.sidebar:
        st.header("AI Research Assistant")
        st.caption("Supervisor · Research · Analysis · Writer · Reviewer")

        st.subheader("Settings")
        st.slider(
            "Maximum revisions",
            0, 3, key="max_revisions", disabled=running,
            help="How many times the writer may revise the report after review.",
        )

        st.subheader("API keys")
        for name, ok in _key_status().items():
            st.markdown(f"{':green[:material/check_circle:]' if ok else ':red[:material/cancel:]'} {name}")
        st.caption("Keys are read from the `.env` file and never displayed.")

        st.subheader("Research history")
        history = load_history()
        if not history:
            st.caption("No research runs yet.")
            return
        index = st.selectbox(
            "Past runs",
            range(len(history)),
            format_func=lambda i: (
                f"{history[i]['created']} · {history[i]['topic'][:38]}"
                f"{'…' if len(history[i]['topic']) > 38 else ''}"
                f" · {'approved' if history[i]['approved'] else 'not approved'}"
            ),
            disabled=running,
            key="history_index",
        )
        st.button(
            "Open report", icon=":material/folder_open:", width="stretch",
            disabled=running, on_click=_open_history_entry, args=(history[index],),
        )


# ---------------------------------------------------------- topic input ----

def render_topic_form():
    running = st.session_state.running
    with st.form("research_form", border=False):
        st.text_input(
            "Research topic",
            key="topic_input",
            placeholder="e.g. Latest AI agent development trends in 2026",
            max_chars=MAX_TOPIC_LENGTH,
            disabled=running,
        )
        st.form_submit_button(
            "Research in progress…" if running else "Start research",
            icon=":material/hourglass_top:" if running else ":material/play_arrow:",
            type="primary",
            disabled=running,
            on_click=start_clicked,
        )
    if st.session_state.error:
        st.error(st.session_state.error, icon=":material/error:")


# --------------------------------------------------------- agent monitor ----

class AgentMonitor:
    """Five status cards that can be updated live while the graph runs."""

    def __init__(self):
        columns = st.columns(len(AGENTS))
        self._slots = {}
        for column, agent in zip(columns, AGENTS):
            with column, st.container(border=True, height=230):
                self._slots[agent] = st.empty()

    def render_all(self, statuses):
        for agent in AGENTS:
            status, detail = statuses[agent]
            label, icon = AGENT_META[agent]
            with self._slots[agent].container():
                st.markdown(f"**{icon} {label}**")
                st.markdown(STATUS_BADGE[status])
                st.caption(detail)


# ------------------------------------------------------------- the run ----

def run_research(monitor):
    """Execute the existing graph, updating the monitor live.

    Stores the outcome (or a friendly error) in session state; the caller
    reruns the script afterwards so the controls are re-enabled.
    """
    ss = st.session_state
    topic, max_revisions = ss.active_topic, ss.max_revisions
    tracker = AgentTracker()
    tracker.start()
    monitor.render_all(tracker.status)

    started = time.time()
    final_state = None
    activity = []

    with st.status(f"Researching: {topic}", expanded=True) as status:
        try:
            for node, update, state in stream_research(topic, max_revisions):
                tracker.on_event(node, update, state)
                ss.agent_status = dict(tracker.status)
                monitor.render_all(tracker.status)

                label = AGENT_META.get(node, (node.replace("_", " ").title(),))[0]
                line = f"**{label}** — {summarize_update(node, update, state)}"
                activity.append(line)
                st.markdown(f":material/check: {line}")
                if tracker.current:
                    status.update(label=f"Running: {AGENT_META[tracker.current][0]} agent…")
                final_state = dict(state)

            if not final_state or not final_state.get("draft"):
                raise RuntimeError("The workflow ended without producing a report.")

            status.update(label="Creating PDF report…")
            pdf_path = export_report_pdf(final_state)
            entry = make_entry(final_state, pdf_path, time.time() - started)
            add_entry(entry)
            ss.result = entry

            approved = entry["approved"]
            status.update(
                label="Research complete — report approved" if approved
                else "Research finished — report NOT approved",
                state="complete",
            )
        except Exception as exc:  # noqa: BLE001 - show any failure in the UI
            message = friendly_error(exc)
            tracker.fail(message)
            ss.agent_status = dict(tracker.status)
            monitor.render_all(tracker.status)
            ss.error = message
            status.update(label="Research failed", state="error")
            st.error(message, icon=":material/error:")

    ss.activity = activity
    ss.running = False


def render_activity():
    activity = st.session_state.activity
    if activity and not st.session_state.running:
        with st.expander("Agent activity log"):
            for line in activity:
                st.markdown(f"- {line}")


# ---------------------------------------------------------------- report ----

def _read_pdf(path):
    try:
        return Path(path).read_bytes() if path else None
    except OSError:
        return None


def render_evaluation(evaluation):
    """Automated evaluation: per-dimension scores and the flags found."""
    verdict = "passed" if evaluation["passed"] else "not approvable"
    with st.expander(f"Automated evaluation: {evaluation['overall']:.0f}/100 ({verdict})"):
        st.caption(evaluation["summary"])
        st.table([
            {"Dimension": d["name"].replace("_", " ").capitalize(),
             "Score": f"{d['score']:.0f}/100", "Weight": f"{d['weight']:.0%}",
             "Details": "; ".join(d["notes"])}
            for d in evaluation["dimensions"]
        ])
        for flag in evaluation["flags"][:15]:
            evidence = f" — _{flag['evidence']}_" if flag["evidence"] else ""
            st.markdown(f"- **{flag['severity']}** · {flag['message']}{evidence}".replace("$", "\\$"))


def render_report(result):
    st.subheader("Final report")
    approved = result["approved"]

    cols = st.columns(4)
    cols[0].metric("Status", "Approved" if approved else "Not approved")
    cols[1].metric("Revisions", result["revisions"])
    cols[2].metric("Reviews", result["reviews"])
    cols[3].metric("Time", f"{result['elapsed']}s")

    if not approved:
        st.warning(
            "The reviewer did not approve this report "
            f"({result['status'].replace('_', ' ')}). Treat it as an unverified draft.",
            title="Not approved",
        )

    pdf_bytes = _read_pdf(result.get("pdf_path"))
    if pdf_bytes:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=Path(result["pdf_path"]).name,
            mime="application/pdf",
            icon=":material/download:",
            type="primary",
            on_click="ignore",
            key=f"download_{result['id']}",
        )
    else:
        st.info("The PDF for this report is not available.", icon=":material/picture_as_pdf:")

    with st.container(border=True):
        # "$" would otherwise start LaTeX math in st.markdown.
        st.markdown(result["draft"].replace("$", "\\$"))

    if result.get("evaluation"):
        render_evaluation(result["evaluation"])

    if result.get("feedback"):
        with st.expander("Final reviewer feedback"):
            st.markdown(result["feedback"].replace("$", "\\$"))
