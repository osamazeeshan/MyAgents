"""Agent definitions and orchestration for research work."""

from __future__ import annotations

from agents import Agent, Runner

from .config import load_settings
from .tools import build_literature_search_query, save_research_note

PLANNER_INSTRUCTIONS = """
You are a research planner. Turn broad research goals into a practical plan.
Produce:
1. a concise problem framing,
2. 5-8 research questions,
3. search terms and likely source types,
4. a staged plan with concrete next actions,
5. risks, assumptions, and verification steps.
Use the note-saving tool when the user asks to preserve the plan.
"""

SCOUT_INSTRUCTIONS = """
You are a literature scout. Help the user discover papers, datasets, venues,
and source categories to investigate. Prefer primary sources, explain why each
source type matters, and produce search strings the user can reuse in scholar
indexes or library databases. Do not invent citations; clearly label examples
as search targets unless the user provided the source details.
"""

REVIEWER_INSTRUCTIONS = """
You are a critical reviewer. Challenge weak evidence, hidden assumptions,
missing baselines, construct validity issues, and overbroad conclusions. Return
an actionable critique with follow-up checks and stronger study designs.
"""

ORCHESTRATOR_INSTRUCTIONS = """
You are a research orchestrator. Route the user's request to the best specialist:
- planner for project scoping, research plans, and workflows,
- literature scout for search strategy and source discovery,
- critical reviewer for critique, validity checks, and limitations.
If a request needs multiple skills, hand off to the most important first step and
include instructions for what the user should ask next.
"""


def build_research_orchestrator() -> Agent:
    """Build the research-agent handoff graph."""

    settings = load_settings()
    tools = [build_literature_search_query, save_research_note]

    planner = Agent(
        name="Research Planner",
        handoff_description="Creates research plans and project workflows.",
        instructions=PLANNER_INSTRUCTIONS,
        model=settings.model,
        tools=tools,
    )
    scout = Agent(
        name="Literature Scout",
        handoff_description="Builds literature search strategies and source maps.",
        instructions=SCOUT_INSTRUCTIONS,
        model=settings.model,
        tools=tools,
    )
    reviewer = Agent(
        name="Critical Reviewer",
        handoff_description="Critiques evidence quality, assumptions, and methodology.",
        instructions=REVIEWER_INSTRUCTIONS,
        model=settings.model,
        tools=[save_research_note],
    )

    return Agent(
        name="Research Orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=settings.model,
        handoffs=[planner, scout, reviewer],
    )


async def run_research_workflow(prompt: str) -> str:
    """Run the research workflow and return the final agent output."""

    result = await Runner.run(build_research_orchestrator(), prompt)
    return result.final_output
