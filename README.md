# deep-research-agent

> LangGraph agent that researches a topic by planning subqueries, searching the web, and synthesizing a source-grounded report. With human-in-the-loop course-correction and SQLite-checkpointed persistence.

[![CI](https://github.com/Mohar7/deep-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohar7/deep-research-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![LangGraph 1.x](https://img.shields.io/badge/LangGraph-1.x-1c3d5a.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What it does

Give it a topic. The agent:

1. **Plans** 3–5 focused subqueries (LLM).
2. **Pauses** for you to approve or rewrite the plan (`interrupt()`).
3. **Researches** each subquery in turn: web search → fetch top results → summarize each into a `Finding`.
4. **Synthesizes** the findings into a structured `ResearchReport` with citations.

Every step is checkpointed, so threads survive process restarts and can be resumed days later.

## Why LangGraph?

A plain `function_calling` loop would conflate planning, searching, and writing. LangGraph lets each step be its own node with its own prompt and validation — and `interrupt()` gives you a clean place to insert a human without rewriting the agent. Persistence is the other reason: `AsyncSqliteSaver` snapshots state after every node, so `thread_id` is enough to inspect or resume any run.

## Architecture

```
START
  │
  ▼
┌──────────┐    plan: list[str]
│ planner  │  ───────────────────► state
└────┬─────┘
     ▼
┌──────────────┐  interrupt({"type": "approve_plan", ...})
│ plan_review  │  ◄─────────────────────── caller resumes with Command(resume={"plan": ...})
└────┬─────────┘
     ▼
┌──────────────┐   pop one query → web_search → fetch_url → LLM summarize
│ researcher   │   appends Finding[] to state.findings
└────┬─────────┘
     │
     ▼
  more_queries?  ── yes ──► researcher (loop)
     │
     no
     ▼
┌──────────────┐  with_structured_output(ResearchReport)
│ synthesizer  │
└────┬─────────┘
     ▼
    END
```

**Files**

| File | Purpose |
|---|---|
| `src/deep_research_agent/state.py` | `TypedDict` state + `Finding` / `ResearchReport` Pydantic models |
| `src/deep_research_agent/tools.py` | `web_search` (DuckDuckGo via `ddgs`), `fetch_url` (httpx + readability) |
| `src/deep_research_agent/nodes.py` | `planner`, `plan_review` (interrupt), `researcher`, `synthesizer` |
| `src/deep_research_agent/graph.py` | `build_graph(checkpointer)` — wires it all |
| `src/deep_research_agent/api.py` | FastAPI wrapper: `/research`, `/research/{id}/resume`, `/research/{id}/stream` |
| `src/deep_research_agent/observability.py` | Optional Langfuse `CallbackHandler` |

## Quick start

```bash
git clone https://github.com/Mohar7/deep-research-agent.git
cd deep-research-agent
uv sync

cp .env.example .env
# edit .env — set OPENAI_API_KEY (or ANTHROPIC_API_KEY and MODEL_NAME=anthropic:claude-haiku-4-5)

uv run uvicorn deep_research_agent.api:app --reload
# http://localhost:8000/docs
```

### One-shot CLI use

```python
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from deep_research_agent.graph import build_graph

async def main():
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-1"}}

    # Run until the HITL gate.
    result = await graph.ainvoke({"topic": "async Python performance pitfalls"}, config)
    print("Proposed plan:", result["__interrupt__"][0].value["proposed_plan"])

    # Accept as-is. Pass {"plan": [...]} to rewrite.
    final = await graph.ainvoke(Command(resume={"plan": None}), config)
    print(final["report"].model_dump_json(indent=2))

asyncio.run(main())
```

### HTTP API

```bash
# Start a research thread
curl -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"topic": "async Python performance pitfalls"}'

# Response:
# {
#   "thread_id": "f3a1...",
#   "state": {"topic": "...", "plan": [...], "original_plan": [...]},
#   "interrupt": {
#     "type": "approve_plan",
#     "proposed_plan": ["What are common async/await mistakes?", ...]
#   }
# }

# Approve the plan as-is (or pass an edited list)
curl -X POST http://localhost:8000/research/$THREAD_ID/resume \
  -H 'Content-Type: application/json' \
  -d '{"payload": {"plan": null}}'
```

## Observability

Set `LANGFUSE_ENABLED=true` plus the three Langfuse env vars and every LLM/tool call is traced:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

The agent attaches a `CallbackHandler` to its config; no other wiring needed. Traces show: planner prompt + completion, every search query, every fetch, every summarizer call, the synthesizer's structured-output call, latencies, tokens, costs.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `openai:gpt-4o-mini` | `init_chat_model` syntax; can be `anthropic:...`, `google_genai:...` etc. |
| `MODEL_TEMPERATURE` | `0.0` | |
| `MAX_ITERATIONS` | `3` | Hard cap on researcher loops beyond the plan length |
| `RESULTS_PER_QUERY` | `5` | Hits to fetch per subquery |
| `SQLITE_PATH` | `./checkpoints.sqlite` | Path to the checkpointer DB |
| `LANGFUSE_ENABLED` | `false` | Opt-in observability |

## Tests

```bash
uv run pytest -v          # 11 tests, ~0.5s, no network, no LLM calls
uv run ruff check src/ tests/
uv run mypy src/
```

The graph test (`tests/test_graph.py::test_graph_pauses_at_hitl_then_resumes`) walks the full planner → HITL → researcher → synthesizer flow with mocked web search, mocked fetch, and a fake LLM — proves that the `interrupt()` + `Command(resume=...)` round-trip works without spending a cent on real API calls.

## CI

GitHub Actions runs on every push and PR: ruff, ruff-format, mypy (`continue-on-error` until baseline cleanup), pytest with coverage, all on Python 3.12 with uv caching.

## License

MIT — see [LICENSE](LICENSE).
