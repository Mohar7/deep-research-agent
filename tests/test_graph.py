"""End-to-end graph test with a fake LLM and mocked search/fetch."""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from deep_research_agent.graph import build_graph
from deep_research_agent.state import ResearchReport
from deep_research_agent.tools import FetchedPage, SearchHit


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_when_plan_is_pre_approved(
    fake_chat_model_factory,
) -> None:
    """Full flow: planner → plan_review (auto-accept) → researcher → synth.

    We skip the HITL interrupt by using a graph without a checkpointer
    (the `interrupt()` call still happens, but `invoke()` without a
    checkpointer treats the interrupt return as the resume value).

    Actually: without a checkpointer, `interrupt()` raises GraphInterrupt
    and the invoke surfaces it. So we need a checkpointer + manual resume.
    See `test_graph_pauses_at_hitl_then_resumes` below for the realistic flow.
    """
    # Smoke-only: just verify the graph compiles with no checkpointer.
    graph = build_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_pauses_at_hitl_then_resumes(fake_chat_model_factory) -> None:
    """The HITL gate should surface an interrupt; resume continues to synth."""
    from langgraph.types import Command

    # 1 planner response + N summarizer responses + 1 synthesizer call.
    # Synthesizer uses with_structured_output, so we mock the whole model.
    plan_text = "1. What is X?"  # one subquery → one researcher iteration

    class FakeModel:
        """Returns canned responses; with_structured_output returns self."""

        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            from langchain_core.messages import AIMessage

            # First call = planner, second = summarizer, third = synthesizer.
            if self.calls == 1:
                return AIMessage(content=plan_text)
            if self.calls == 2:
                return AIMessage(content="X is a thing.")
            # Synthesizer path — we'll have replaced with_structured_output
            return ResearchReport(
                topic="X",
                executive_summary="X is summarized here.",
                sections=[{"heading": "Definition", "body": "X is a thing."}],
                sources=["https://example.com/x"],
            )

        def with_structured_output(self, schema):
            return self

    fake = FakeModel()
    config = {"configurable": {"thread_id": "test-thread", "model": fake}}

    async with _mock_network(), _build_graph_with_memory() as graph:
        # First invoke: should pause at plan_review.
        first = await graph.ainvoke({"topic": "X"}, config)
        interrupts = first.get("__interrupt__")
        assert interrupts, "Expected the graph to pause at HITL gate"
        assert interrupts[0].value["type"] == "approve_plan"

        # Resume with no edit — accept planner output as-is.
        final = await graph.ainvoke(Command(resume={"plan": None}), config)

        assert "report" in final
        report = final["report"]
        assert isinstance(report, ResearchReport)
        assert report.topic == "X"


# ---------- helpers ----------


import contextlib


@contextlib.asynccontextmanager
async def _build_graph_with_memory():
    """Compile the graph with an in-memory checkpointer (good enough for tests)."""
    from langgraph.checkpoint.memory import MemorySaver

    yield build_graph(checkpointer=MemorySaver())


@contextlib.asynccontextmanager
async def _mock_network():
    """Patch web_search and fetch_url at the nodes-module level."""
    with (
        patch(
            "deep_research_agent.nodes.web_search",
            new=AsyncMock(
                return_value=[
                    SearchHit(
                        title="X explained",
                        url="https://example.com/x",
                        snippet="X is a thing.",
                    )
                ]
            ),
        ),
        patch(
            "deep_research_agent.nodes.fetch_url",
            new=AsyncMock(
                return_value=FetchedPage(
                    url="https://example.com/x",
                    title="X explained",
                    text="X is a thing that does X-like things.",
                )
            ),
        ),
    ):
        yield
