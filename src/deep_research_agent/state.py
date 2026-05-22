"""Graph state schema.

The state accumulates as the agent works:
1. Planner writes a list of subqueries → `plan`.
2. Researcher pops one query at a time, searches, fetches, writes a
   Finding → appended to `findings`.
3. Synthesizer turns `findings` into a structured `report`.

`plan` is consumed in-place by the researcher (mutated via state delta).
`findings` accumulates with `operator.add`.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """One source-grounded finding."""

    query: str = Field(description="Subquery that produced this finding")
    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Page title")
    summary: str = Field(description="2-4 sentence summary of relevant content")


class ResearchReport(BaseModel):
    """Final structured output."""

    topic: str
    executive_summary: str = Field(description="2-3 sentence TL;DR")
    sections: list[dict[str, str]] = Field(
        default_factory=list,
        description="Each item: {'heading': str, 'body': str}",
    )
    sources: list[str] = Field(
        default_factory=list, description="Deduplicated URLs cited in the report"
    )


class ResearchState(TypedDict, total=False):
    """LangGraph state.

    `total=False` so partial dicts returned by nodes type-check cleanly.
    """

    # ---- Input ----
    topic: str

    # ---- Planner output ----
    # Remaining subqueries the researcher still needs to address.
    plan: list[str]
    # The plan as originally proposed (kept for the HITL approval payload
    # and for tracing — `plan` mutates as queries are consumed).
    original_plan: list[str]

    # ---- Researcher output (accumulated) ----
    findings: Annotated[list[Finding], operator.add]

    # ---- Loop control ----
    iteration: int

    # ---- Final output ----
    report: ResearchReport
