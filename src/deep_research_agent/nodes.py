"""Graph nodes: planner, researcher, synthesizer, + control helpers.

Pattern: every node is an async function that takes the state and a
RunnableConfig, returns a partial state dict. Config gives us the
thread_id (for traces) and the language model (injected via config so
tests can swap in a fake without monkey-patching).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from deep_research_agent.config import get_settings
from deep_research_agent.state import Finding, ResearchReport, ResearchState
from deep_research_agent.tools import fetch_url, web_search

logger = logging.getLogger(__name__)


# ---------- Model factory ----------


def get_model(config: RunnableConfig | None = None) -> BaseChatModel:
    """Return the chat model.

    Tests inject a fake via `config["configurable"]["model"]`; production
    falls back to `init_chat_model(settings.model_name)`.
    """
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    settings = get_settings()
    return init_chat_model(
        settings.model_name,
        temperature=settings.model_temperature,
    )


# ---------- Planner ----------


PLANNER_SYSTEM = """\
You are a research planner. Given a topic, output a numbered list of 3-5 \
focused subqueries that, together, would let someone write a thorough \
report. Each subquery must be specific enough to type into a search \
engine and get useful hits. Output ONLY the numbered list, one query per \
line, no preamble.
"""


async def planner(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Turn a topic into a list of subqueries."""
    topic = state["topic"]
    model = get_model(config)
    response = await model.ainvoke(
        [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=topic)]
    )
    raw = response.content if isinstance(response.content, str) else str(response.content)
    plan = _parse_numbered_list(raw)
    logger.info("Planner produced %d subqueries", len(plan))
    return {"plan": plan, "original_plan": plan, "iteration": 0}


def _parse_numbered_list(text: str) -> list[str]:
    """Pull lines that look like `1. ...` or `- ...` out of `text`."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering / bullets.
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        else:
            # Numeric prefix?
            if line[0].isdigit():
                for sep in (". ", ") ", " - "):
                    idx = line.find(sep)
                    if 0 < idx <= 3:
                        line = line[idx + len(sep) :].strip()
                        break
        if line:
            out.append(line)
    return out


# ---------- HITL gate ----------


def plan_review(state: ResearchState) -> dict[str, Any]:
    """Pause for the user to approve / edit the plan.

    `interrupt()` halts the graph and surfaces `{ "plan": [...] }` to the
    caller. The caller resumes with either:
        Command(resume={"plan": [...edited list...]})   -> overrides
        Command(resume={"plan": None})                  -> keeps planner output
    """
    decision: dict[str, Any] = interrupt(
        {
            "type": "approve_plan",
            "topic": state["topic"],
            "proposed_plan": state["original_plan"],
            "instructions": (
                "Return {'plan': [...]} to override, or {'plan': None} to "
                "accept the proposed plan as-is."
            ),
        }
    )
    edited_plan = decision.get("plan") if isinstance(decision, dict) else None
    if edited_plan:
        return {"plan": list(edited_plan), "original_plan": list(edited_plan)}
    return {}  # accept planner output unchanged


# ---------- Researcher ----------


SUMMARIZER_SYSTEM = """\
You will receive a search query and the full text of a web page. In \
2-4 sentences, summarize ONLY the parts of the page that directly \
answer the query. If the page is off-topic, output the single word: \
SKIP.
"""


async def researcher(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Process ONE subquery: search → fetch top results → summarize each.

    Pops the query off `plan` and appends results to `findings`. The
    conditional edge `more_queries?` decides whether to loop or move on.
    """
    settings = get_settings()
    plan = list(state.get("plan", []))
    if not plan:
        return {"iteration": state.get("iteration", 0) + 1}

    query = plan.pop(0)
    logger.info("Researching: %s", query)

    try:
        hits = await web_search(query, max_results=settings.results_per_query)
    except Exception:
        logger.exception("web_search failed for %r", query)
        hits = []

    findings = await _summarize_hits(query, hits, get_model(config))

    return {
        "plan": plan,
        "findings": findings,
        "iteration": state.get("iteration", 0) + 1,
    }


async def _summarize_hits(
    query: str, hits: list, model: BaseChatModel
) -> list[Finding]:
    """Fetch + summarize hits concurrently. Drops failures and SKIPs."""

    async def _one(hit) -> Finding | None:
        try:
            page = await fetch_url(hit.url)
        except (httpx.HTTPError, ValueError):
            return None
        prompt = (
            f"QUERY: {query}\n\n"
            f"URL: {page.url}\n"
            f"TITLE: {page.title}\n\n"
            f"PAGE TEXT:\n{page.text}"
        )
        try:
            resp = await model.ainvoke(
                [
                    SystemMessage(content=SUMMARIZER_SYSTEM),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception:
            logger.exception("summarizer LLM failed for %s", page.url)
            return None
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip()
        if text.upper().startswith("SKIP"):
            return None
        return Finding(query=query, url=page.url, title=page.title, summary=text)

    results = await asyncio.gather(*[_one(h) for h in hits], return_exceptions=False)
    return [r for r in results if r is not None]


def more_queries(state: ResearchState) -> str:
    """Loop while we have unconsumed plan items and budget left."""
    settings = get_settings()
    if state.get("plan") and state.get("iteration", 0) < settings.max_iterations + len(
        state.get("original_plan", [])
    ):
        return "researcher"
    return "synthesizer"


# ---------- Synthesizer ----------


SYNTHESIZER_SYSTEM = """\
You will receive a research topic and a list of source-grounded \
findings. Produce a structured report with: \

1. A 2-3 sentence executive summary. \
2. 2-4 thematic sections, each with a heading and a 1-2 paragraph body. \

Cite sources inline by URL where useful. Be concise; do not invent \
facts that aren't in the findings.
"""


async def synthesizer(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Turn accumulated findings into a `ResearchReport`."""
    topic = state["topic"]
    findings = state.get("findings", [])

    if not findings:
        report = ResearchReport(
            topic=topic,
            executive_summary="No findings were produced — the search returned no usable sources.",
            sections=[],
            sources=[],
        )
        return {"report": report}

    model = get_model(config)
    structured = model.with_structured_output(ResearchReport)

    findings_text = "\n\n".join(
        f"[{i + 1}] {f.title or f.url}\n    URL: {f.url}\n    {f.summary}"
        for i, f in enumerate(findings)
    )

    report = await structured.ainvoke(
        [
            SystemMessage(content=SYNTHESIZER_SYSTEM),
            HumanMessage(
                content=f"TOPIC: {topic}\n\nFINDINGS:\n{findings_text}"
            ),
        ]
    )

    # Make sure the topic and sources are filled even if the model omits them.
    if not report.topic:
        report.topic = topic
    if not report.sources:
        seen: set[str] = set()
        sources = []
        for f in findings:
            if f.url not in seen:
                sources.append(f.url)
                seen.add(f.url)
        report.sources = sources

    return {"report": report}
