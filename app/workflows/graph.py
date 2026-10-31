
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.workflows.state import ResearchState

from app.agents.supervisor import supervisor_agent
from app.agents.research_agent import research_agent, research_worker
from app.agents.analysis_agent import analysis_agent
from app.agents.writer import writer_agent
from app.agents.reviewer import reviewer_agent


builder = StateGraph(ResearchState)


# Register agents
builder.add_node("supervisor", supervisor_agent)
builder.add_node("research", research_agent)          # aggregates the workers
builder.add_node("research_worker", research_worker)  # parallel, one per task
builder.add_node("analysis", analysis_agent)
builder.add_node("writer", writer_agent)
builder.add_node("reviewer", reviewer_agent)


# Start
builder.add_edge(START, "supervisor")


# Supervisor routing function
def route_supervisor(state):

    next_agent = state["next_agent"]

    # Fan out: one parallel research worker per planned task.
    if next_agent == "research" and state.get("research_tasks"):
        return [
            Send("research_worker", {"task": task, "question": state["question"]})
            for task in state["research_tasks"]
        ]

    return next_agent


# Conditional routing
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "research": "research",
        "analysis": "analysis",
        "writer": "writer",
        "reviewer": "reviewer",
        "finish": END
    }
)


# Fan in: the research agent runs once, after every worker has finished
builder.add_edge("research_worker", "research")

# Every agent returns control to supervisor
builder.add_edge("research", "supervisor")
builder.add_edge("analysis", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")


# Compile
research_graph = builder.compile()
