"""Graph construction.

```
START → planner → plan_review (HITL) → researcher ─┐
                                          ▲        │
                                          │   (more_queries?)
                                          └────────┤
                                                   ▼
                                              synthesizer → END
```
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deep_research_agent.nodes import (
    more_queries,
    plan_review,
    planner,
    researcher,
    synthesizer,
)
from deep_research_agent.state import ResearchState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile the research graph.

    `checkpointer` is optional — pass `AsyncSqliteSaver(...)` (or any
    other persistence layer) to enable thread-scoped resumption,
    interrupt(), and time-travel. Without it the graph runs to
    completion in-process.
    """
    builder = (
        StateGraph(ResearchState)
        .add_node("planner", planner)
        .add_node("plan_review", plan_review)
        .add_node("researcher", researcher)
        .add_node("synthesizer", synthesizer)
        .add_edge(START, "planner")
        .add_edge("planner", "plan_review")
        .add_edge("plan_review", "researcher")
        .add_conditional_edges(
            "researcher", more_queries, ["researcher", "synthesizer"]
        )
        .add_edge("synthesizer", END)
    )
    return builder.compile(checkpointer=checkpointer)
