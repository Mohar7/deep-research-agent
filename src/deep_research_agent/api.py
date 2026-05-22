"""FastAPI wrapper around the research graph.

Endpoints:
- `POST /research`               start a new research thread
- `POST /research/{tid}/resume`  resume after the HITL interrupt
- `GET  /research/{tid}/state`   inspect current state for the thread
- `GET  /research/{tid}/stream`  server-sent events of state updates

The graph runs against an `AsyncSqliteSaver` so threads survive process
restarts. Configure the SQLite path via `SQLITE_PATH` env var.
"""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from deep_research_agent.config import get_settings
from deep_research_agent.graph import build_graph
from deep_research_agent.observability import langfuse_callbacks

logger = logging.getLogger(__name__)


# ---------- Schemas ----------


class StartResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    thread_id: str | None = Field(
        default=None,
        description="Optional. Auto-generated if omitted.",
    )


class StartResearchResponse(BaseModel):
    thread_id: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Present when the graph paused for human input. Resume by "
            "POSTing the user's decision to /research/{tid}/resume."
        ),
    )


class ResumeRequest(BaseModel):
    # Free-form payload — whatever the interrupt() call expects.
    # For our HITL gate: {"plan": [...edited list...]} or {"plan": null}.
    payload: dict[str, Any]


# ---------- App ----------


# `compiled_graph` is set at startup so we only build the graph once.
_compiled_graph = None
_checkpointer_cm: contextlib.AbstractAsyncContextManager | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _compiled_graph, _checkpointer_cm
    settings = get_settings()
    logger.info("Starting research agent; sqlite=%s", settings.sqlite_path)

    _checkpointer_cm = AsyncSqliteSaver.from_conn_string(settings.sqlite_path)
    checkpointer = await _checkpointer_cm.__aenter__()
    _compiled_graph = build_graph(checkpointer=checkpointer)
    try:
        yield
    finally:
        if _checkpointer_cm is not None:
            await _checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(
    title="deep-research-agent",
    description="LangGraph research agent with HITL course-correction.",
    version="0.1.0",
    lifespan=lifespan,
)


def _graph():
    if _compiled_graph is None:
        raise HTTPException(503, "Graph not initialized yet")
    return _compiled_graph


def _config_for(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": langfuse_callbacks(),
    }


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the interrupt() value from a graph result, if any."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    # LangGraph's Interrupt is a NamedTuple-like with .value attr.
    return getattr(first, "value", first)


# ---------- Routes ----------


@app.post("/research", response_model=StartResearchResponse)
async def start_research(req: StartResearchRequest) -> StartResearchResponse:
    """Kick off research on `req.topic`. Returns when the graph either
    completes or pauses at the HITL gate."""
    thread_id = req.thread_id or str(uuid.uuid4())
    graph = _graph()
    result = await graph.ainvoke({"topic": req.topic}, _config_for(thread_id))
    return StartResearchResponse(
        thread_id=thread_id,
        state=_safe_state(result),
        interrupt=_interrupt_payload(result),
    )


@app.post("/research/{thread_id}/resume", response_model=StartResearchResponse)
async def resume_research(thread_id: str, req: ResumeRequest) -> StartResearchResponse:
    """Resume a thread that paused at `interrupt()`."""
    graph = _graph()
    result = await graph.ainvoke(
        Command(resume=req.payload), _config_for(thread_id)
    )
    return StartResearchResponse(
        thread_id=thread_id,
        state=_safe_state(result),
        interrupt=_interrupt_payload(result),
    )


@app.get("/research/{thread_id}/state")
async def get_thread_state(thread_id: str) -> dict[str, Any]:
    """Return the latest checkpointed state for `thread_id`."""
    graph = _graph()
    snapshot = await graph.aget_state(_config_for(thread_id))
    if not snapshot.values:
        raise HTTPException(404, "No such thread")
    return _safe_state(snapshot.values)


@app.get("/research/{thread_id}/stream")
async def stream_research(thread_id: str) -> EventSourceResponse:
    """Server-sent events of state updates for an in-flight thread."""
    graph = _graph()

    async def _events() -> AsyncGenerator[dict[str, str], None]:
        async for mode, chunk in graph.astream(
            None, _config_for(thread_id), stream_mode=["updates"]
        ):
            yield {"event": mode, "data": json.dumps(_safe_state(chunk), default=str)}

    return EventSourceResponse(_events())


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "deep-research-agent",
        "docs": "/docs",
        "version": "0.1.0",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


# ---------- Helpers ----------


def _safe_state(state: dict[str, Any] | Any) -> dict[str, Any]:
    """Pydantic models in state aren't JSON-serializable until we dump them."""
    if state is None:
        return {}
    out: dict[str, Any] = {}
    for key, value in dict(state).items():
        if hasattr(value, "model_dump"):
            out[key] = value.model_dump()
        elif isinstance(value, list) and value and hasattr(value[0], "model_dump"):
            out[key] = [v.model_dump() for v in value]
        else:
            out[key] = value
    return out
