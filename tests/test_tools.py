"""Unit tests for tools.py — mocked network via unittest.mock."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from deep_research_agent.tools import SearchHit, fetch_url


def _mock_httpx_response(text: str, status_code: int = 200) -> httpx.Response:
    """Build a real Response object so fetch_url's response.raise_for_status()
    and .text behave like the real thing."""
    return httpx.Response(
        status_code,
        text=text,
        request=httpx.Request("GET", "https://example.com/article"),
    )


@pytest.mark.asyncio
async def test_fetch_url_extracts_main_text() -> None:
    html = """
    <html>
      <head><title>Test Article</title></head>
      <body>
        <nav>Site nav garbage</nav>
        <article>
          <h1>Real Heading</h1>
          <p>This is the actual body of the article that we care about.</p>
          <p>A second paragraph with substantive content for extraction.</p>
        </article>
        <footer>Footer garbage</footer>
      </body>
    </html>
    """
    with patch(
        "httpx.AsyncClient.get",
        new=AsyncMock(return_value=_mock_httpx_response(html)),
    ):
        page = await fetch_url("https://example.com/article")

    assert page.url == "https://example.com/article"
    assert "actual body" in page.text


@pytest.mark.asyncio
async def test_fetch_url_truncates_long_pages() -> None:
    huge = "<html><body><article>" + ("word " * 5000) + "</article></body></html>"
    with patch(
        "httpx.AsyncClient.get",
        new=AsyncMock(return_value=_mock_httpx_response(huge)),
    ):
        page = await fetch_url("https://example.com/huge")

    assert page.text.endswith("[…truncated]")


@pytest.mark.asyncio
async def test_fetch_url_raises_on_http_error() -> None:
    err_response = _mock_httpx_response("not found", status_code=404)
    with patch(
        "httpx.AsyncClient.get", new=AsyncMock(return_value=err_response)
    ), pytest.raises(httpx.HTTPStatusError):
        await fetch_url("https://example.com/missing")


def test_searchhit_is_lightweight_dataclass() -> None:
    a = SearchHit(title="t", url="u", snippet="s")
    b = SearchHit(title="t", url="u", snippet="s")
    assert a == b
