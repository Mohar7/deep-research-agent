"""Test fixtures.

The agent is driven by an LLM, which we mock with a FakeListChatModel
to keep tests deterministic and offline.
"""

from __future__ import annotations

import os

import pytest

# Stub env so config.get_settings() works without a real .env.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")


@pytest.fixture
def fake_search_results():
    """A canned set of SearchHits with stable shape."""
    from deep_research_agent.tools import SearchHit

    return [
        SearchHit(
            title="Async Python — official docs",
            url="https://example.com/async",
            snippet="asyncio is a library to write concurrent code...",
        ),
        SearchHit(
            title="Coroutines explained",
            url="https://example.com/coroutines",
            snippet="A coroutine is a specialized version of a generator...",
        ),
    ]


@pytest.fixture
def fake_fetched_page():
    from deep_research_agent.tools import FetchedPage

    return FetchedPage(
        url="https://example.com/async",
        title="Async Python",
        text="asyncio is part of Python's standard library and enables \
            structured concurrency via the async/await syntax.",
    )


class FakeChatResponses:
    """Iterator over canned LLM responses.

    Pattern:
        model = FakeChatModel([
            "1. First subquery\n2. Second subquery",  # planner
            "Summary of source 1",                     # researcher
            ...
        ])
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def next_response(self) -> str:
        return next(self._responses)


@pytest.fixture
def fake_chat_model_factory():
    """Returns a factory that produces a fake BaseChatModel."""

    def _factory(responses: list[str]):
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        return FakeListChatModel(responses=responses)

    return _factory
