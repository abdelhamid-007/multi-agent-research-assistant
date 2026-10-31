"""Streamlit web interface for the AI Research Assistant.

Launch with:  streamlit run streamlit_app.py
"""
import streamlit as st

from app.ui.components import (
    AgentMonitor,
    init_session_state,
    render_activity,
    render_report,
    render_sidebar,
    render_topic_form,
    run_research,
)

st.set_page_config(page_title="AI Research Assistant", page_icon=":material/biotech:", layout="wide")

init_session_state()
render_sidebar()

st.title("AI Research Assistant")
st.caption("A supervised team of AI agents researches, analyses, writes and reviews a report for you.")

render_topic_form()

st.subheader("Agent monitor")
monitor = AgentMonitor()
monitor.render_all(st.session_state.agent_status)

if st.session_state.running:
    run_research(monitor)
    st.rerun()  # re-enable the controls and show the finished report

render_activity()

if st.session_state.result:
    render_report(st.session_state.result)
