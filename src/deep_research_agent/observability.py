"""Langfuse tracing setup.

Opt-in: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_HOST` env vars, then `LANGFUSE_ENABLED=true`. The CallbackHandler
attaches to every LLM and tool call via the LangChain callbacks system.

Usage:
    from deep_research_agent.observability import langfuse_callbacks
    await graph.ainvoke(state, config={"callbacks": langfuse_callbacks()})
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def langfuse_callbacks() -> list:
    """Return a list of LangChain callbacks for tracing.

    Returns `[]` if Langfuse isn't configured / enabled, so callers can
    unconditionally pass the result without branching.
    """
    if os.getenv("LANGFUSE_ENABLED", "false").lower() not in ("true", "1", "yes"):
        return []

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_ENABLED=true but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "are missing; tracing disabled."
        )
        return []

    try:
        from langfuse.langchain import CallbackHandler  # langfuse>=3
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # type: ignore[no-redef]
        except ImportError:
            logger.warning("langfuse not installed; tracing disabled.")
            return []

    handler = CallbackHandler()
    logger.info("Langfuse tracing enabled")
    return [handler]


__all__ = ["langfuse_callbacks"]
