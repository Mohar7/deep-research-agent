"""Tools the researcher node calls.

Two tools:
- `web_search(query, max_results)` — DuckDuckGo via the `ddgs` package.
  No API key required, so cloners can demo the agent immediately.
- `fetch_url(url)` — async httpx fetch + readability extraction.
  Returns clean main-text, not the raw HTML.

Both tools are pure-Python; they return data, and the researcher node
turns that into `Finding` objects. We do NOT decorate them with
`@tool` from LangChain because we're not letting the LLM call them
directly — the researcher node calls them deterministically. Keeping
them as plain async functions makes them trivial to unit-test.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from ddgs import DDGS
from readability import Document

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchHit:
    """One search result. Mirrors what DDGS returns, minus the noise."""

    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class FetchedPage:
    """Cleaned page content."""

    url: str
    title: str
    text: str


async def web_search(query: str, max_results: int = 5) -> list[SearchHit]:
    """Return up to `max_results` web search hits for `query`.

    DDGS is sync; we run it in a thread so we don't block the event loop.
    """

    def _run() -> list[SearchHit]:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            SearchHit(
                title=item.get("title", ""),
                url=item.get("href") or item.get("url", ""),
                snippet=item.get("body") or item.get("snippet", ""),
            )
            for item in raw
        ]

    return await asyncio.to_thread(_run)


async def fetch_url(url: str, *, timeout_s: float = 15.0) -> FetchedPage:
    """Fetch `url` and return readability-extracted main text.

    Raises `httpx.HTTPError` on network failures (caller decides what to do).
    Truncates the extracted text to ~8000 chars to keep LLM prompts bounded.
    """
    async with httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=True,
        headers={"User-Agent": "deep-research-agent/0.1 (+research bot)"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        html = response.text

    # readability is sync; offload it.
    def _extract() -> tuple[str, str]:
        doc = Document(html)
        return doc.short_title(), doc.summary(html_partial=True)

    title, summary_html = await asyncio.to_thread(_extract)

    # Strip tags from the readability HTML — we want plain text in prompts.
    from bs4 import BeautifulSoup

    text = BeautifulSoup(summary_html, "lxml").get_text(separator="\n", strip=True)

    if len(text) > 8000:
        text = text[:8000] + "\n\n[…truncated]"

    return FetchedPage(url=url, title=title, text=text)


__all__ = ["FetchedPage", "SearchHit", "fetch_url", "web_search"]
