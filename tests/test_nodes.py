"""Unit tests for nodes — fake LLM, no network for graph."""

from __future__ import annotations

import pytest

from deep_research_agent.nodes import _parse_numbered_list, planner, synthesizer
from deep_research_agent.state import Finding, ResearchReport


def test_parse_numbered_list_handles_various_bullet_styles() -> None:
    text = """
    1. First query about async patterns
    2) Second query
    - Third query bullet
    * Fourth query star
    Some leading prose to ignore? Actually keep — only list lines start with bullets.
    """
    parsed = _parse_numbered_list(text)
    assert "First query about async patterns" in parsed
    assert "Second query" in parsed
    assert "Third query bullet" in parsed
    assert "Fourth query star" in parsed
    # Prose lines without a bullet are kept as-is (parser is lenient).
    assert len(parsed) >= 4


def test_parse_numbered_list_drops_empty_lines() -> None:
    assert _parse_numbered_list("\n\n   \n") == []


@pytest.mark.asyncio
async def test_planner_produces_plan_from_topic(fake_chat_model_factory) -> None:
    fake = fake_chat_model_factory(["1. Query A\n2. Query B\n3. Query C"])
    state = {"topic": "async python performance"}
    config = {"configurable": {"model": fake}}

    result = await planner(state, config)

    assert result["plan"] == ["Query A", "Query B", "Query C"]
    assert result["original_plan"] == ["Query A", "Query B", "Query C"]
    assert result["iteration"] == 0


@pytest.mark.asyncio
async def test_synthesizer_returns_empty_report_on_no_findings(
    fake_chat_model_factory,
) -> None:
    # Even with no findings, synthesizer must produce a structured report
    # (the LLM shouldn't even be called in this case).
    fake = fake_chat_model_factory([])
    state = {"topic": "anything", "findings": []}
    config = {"configurable": {"model": fake}}

    result = await synthesizer(state, config)

    report = result["report"]
    assert isinstance(report, ResearchReport)
    assert report.topic == "anything"
    assert "No findings" in report.executive_summary
    assert report.sources == []


@pytest.mark.asyncio
async def test_synthesizer_populates_sources_from_findings(
    fake_chat_model_factory,
) -> None:
    """If the LLM omits sources, the node should fill them from findings."""

    # Structured-output fake: we mock the model so that with_structured_output
    # returns a report missing sources, and verify the node patches it.
    class FakeStructuredModel:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            return ResearchReport(
                topic="",  # intentionally empty — node should fill from state
                executive_summary="summary",
                sections=[{"heading": "h", "body": "b"}],
                sources=[],  # intentionally empty — node should fill from findings
            )

    findings = [
        Finding(query="q", url="https://a.com", title="A", summary="..."),
        Finding(query="q", url="https://b.com", title="B", summary="..."),
        Finding(query="q", url="https://a.com", title="A dup", summary="..."),
    ]
    state = {"topic": "the topic", "findings": findings}
    config = {"configurable": {"model": FakeStructuredModel()}}

    result = await synthesizer(state, config)

    report = result["report"]
    assert report.topic == "the topic"
    # Sources deduplicated, order preserved.
    assert report.sources == ["https://a.com", "https://b.com"]
