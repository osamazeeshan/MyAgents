"""Agent definitions and orchestration for research work."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from agents import (
    Agent,
    Runner,
    WebSearchTool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from .config import ResearchAgentSettings, load_settings
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

TOPIC_SCOUT_INSTRUCTIONS = """
You are a conference topic scout for machine learning, computer vision, AI, and
LLM research. Use web search to inspect recent accepted-paper lists, proceedings,
workshop pages, calls for papers, and official conference programs from top
venues such as NeurIPS, ICML, ICLR, AAAI, CVPR, ECCV, ICCV, and WACV.

Rules:
- Focus on the current year and at most one year before it, unless the user
  explicitly asks for a wider discovery window.
- Prefer primary sources: conference/proceedings sites, OpenReview, CVF, PMLR,
  AAAI proceedings, and official workshop pages.
- Return topics, not just paper titles. Group near-duplicates into coherent
  research topics.
- Include the venues and years where each topic appears.
- Add 1-3 representative papers or workshops only when you can verify them.
- Do not invent citations, paper titles, authors, or URLs.
- End with a numbered selection menu that the user can choose from.
"""

PAPER_SCOUT_INSTRUCTIONS = """
You are the same conference topic scout, now doing a focused paper search for
the topic selected by the user. Use web search to find relevant papers in this
area.

Rules:
- Papers must generally be from the last 5-6 years relative to the supplied
  current date. Include older papers only in a short "foundational exceptions"
  section when they are essential context.
- Prefer papers from top peer-reviewed venues and reputable preprint servers
  when the preprint is influential or tied to a top venue.
- Include title, authors when available, venue/source, year, URL or DOI, and a
  one-sentence reason the paper is relevant.
- Group papers by subtheme, method, benchmark/dataset, and application area.
- Clearly separate verified papers from search leads that still need checking.
"""

METHODOLOGY_REVIEWER_INSTRUCTIONS = """
You are Reviewer A, a rigorous methodology and evidence-quality reviewer. Write
an in-depth critical literature review of the provided papers. Focus on study
design, baselines, datasets, evaluation metrics, ablations, reproducibility,
statistical validity, leakage/contamination risks, and whether claims are
supported by evidence. Identify contradictions and gaps across papers.
"""

SYNTHESIS_REVIEWER_INSTRUCTIONS = """
You are Reviewer B, a field-synthesis reviewer. Write an in-depth critical
literature review of the provided papers. Focus on how the field has evolved,
main research clusters, theoretical assumptions, practical limitations, open
problems, promising directions, and how a new researcher should position a
project in this area. Identify underexplored questions and risky hype.
"""

CONFERENCE_VENUES = (
    "NeurIPS, ICML, ICLR, AAAI, CVPR, ECCV, ICCV, WACV, and closely related "
    "top-tier workshops or proceedings"
)


def configure_model_provider(settings: ResearchAgentSettings) -> None:
    """Configure the Agents SDK for OpenAI or OpenAI-compatible local models."""

    if settings.disable_tracing:
        set_tracing_disabled(True)

    if settings.use_chat_completions:
        set_default_openai_api("chat_completions")

    if settings.base_url:
        set_default_openai_client(
            AsyncOpenAI(
                api_key=settings.api_key or "local",
                base_url=settings.base_url,
            )
        )


def _build_web_search_tool(settings: ResearchAgentSettings) -> WebSearchTool | None:
    """Return a hosted web-search tool when the active provider supports it."""

    if settings.uses_local_model or settings.use_chat_completions:
        return None
    return WebSearchTool(search_context_size="high")


def build_research_orchestrator() -> Agent:
    """Build the research-agent handoff graph."""

    settings = load_settings()
    configure_model_provider(settings)
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


def build_conference_topic_scout() -> Agent:
    """Build the web-enabled scout that discovers recent conference topics."""

    settings = load_settings()
    configure_model_provider(settings)
    tools = [build_literature_search_query, save_research_note]
    web_search = _build_web_search_tool(settings)
    if web_search is not None:
        tools.insert(0, web_search)

    return Agent(
        name="Conference Topic Scout",
        handoff_description=(
            "Searches the web for recent top-conference research topics and papers."
        ),
        instructions=TOPIC_SCOUT_INSTRUCTIONS,
        model=settings.model,
        tools=tools,
    )


def build_methodology_reviewer() -> Agent:
    """Build the first LLM reviewer for methodological critique."""

    settings = load_settings()
    configure_model_provider(settings)
    return Agent(
        name="Methodology Reviewer",
        instructions=METHODOLOGY_REVIEWER_INSTRUCTIONS,
        model=settings.model,
        tools=[save_research_note],
    )


def build_synthesis_reviewer() -> Agent:
    """Build the second LLM reviewer for field-level synthesis."""

    settings = load_settings()
    configure_model_provider(settings)
    return Agent(
        name="Field Synthesis Reviewer",
        instructions=SYNTHESIS_REVIEWER_INSTRUCTIONS,
        model=settings.model,
        tools=[save_research_note],
    )


def _current_and_previous_year(current_date: datetime | None = None) -> tuple[int, int]:
    current_date = current_date or datetime.now(timezone.utc)
    return current_date.year, current_date.year - 1


def _web_search_availability_note(settings: ResearchAgentSettings) -> str:
    if settings.uses_local_model or settings.use_chat_completions:
        return (
            "\n\nNote: hosted WebSearchTool is unavailable for local/chat-completions "
            "providers. Produce reusable search queries and ask the user to run "
            "with the OpenAI Responses API provider for automatic web search."
        )
    return ""


async def run_research_workflow(prompt: str) -> str:
    """Run the research workflow and return the final agent output."""

    result = await Runner.run(build_research_orchestrator(), prompt)
    return result.final_output


async def discover_recent_conference_topics(prompt: str = "") -> str:
    """Search for recent top-conference research topics and return a menu."""

    settings = load_settings()
    current_year, previous_year = _current_and_previous_year()
    focus = (
        prompt.strip()
        or "broad AI, machine learning, LLM, and computer vision research"
    )
    scout_prompt = f"""
Current date: {datetime.now(timezone.utc).date().isoformat()}.
Find research topics from {CONFERENCE_VENUES} for {previous_year} and {current_year} only.
User focus or constraints: {focus}

Return:
1. A short method note describing sources searched.
2. A grouped list of all high-signal recent topics you found, with venue/year evidence.
3. A numbered menu of topics for the user to select.
{_web_search_availability_note(settings)}
""".strip()
    result = await Runner.run(
        build_conference_topic_scout(), scout_prompt, max_turns=20
    )
    return result.final_output


async def search_papers_for_topic(
    selected_topic: str, discovery_context: str = ""
) -> str:
    """Use the topic scout to find recent papers for the selected topic."""

    settings = load_settings()
    current_year, _ = _current_and_previous_year()
    earliest_year = current_year - 6
    prompt = f"""
Current date: {datetime.now(timezone.utc).date().isoformat()}.
Selected topic: {selected_topic}
Discovery context from the previous step:
{discovery_context}

Search for relevant papers on this selected topic. Keep papers within {earliest_year}-{current_year}
where possible; use older foundational exceptions sparingly and label them.

Return a structured bibliography grouped by subtheme, and include direct source links.
{_web_search_availability_note(settings)}
""".strip()
    result = await Runner.run(
        build_conference_topic_scout().clone(instructions=PAPER_SCOUT_INSTRUCTIONS),
        prompt,
        max_turns=20,
    )
    return result.final_output


async def review_selected_topic(selected_topic: str, paper_context: str) -> str:
    """Run two LLM reviewers over the selected topic and paper set."""

    review_prompt = f"""
Selected topic: {selected_topic}
Recent paper set and notes:
{paper_context}

Write a critical, evidence-grounded literature review. Use only the supplied
paper set unless explicitly marking an item as a suggested follow-up search.
""".strip()

    methodology_result, synthesis_result = await asyncio.gather(
        Runner.run(build_methodology_reviewer(), review_prompt, max_turns=12),
        Runner.run(build_synthesis_reviewer(), review_prompt, max_turns=12),
    )

    return (
        "# Critical Literature Review\n\n"
        "## Reviewer A: Methodology and Evidence Quality\n\n"
        f"{methodology_result.final_output}\n\n"
        "## Reviewer B: Field Synthesis and Research Directions\n\n"
        f"{synthesis_result.final_output}"
    )


async def run_interactive_conference_literature_review(prompt: str = "") -> str:
    """Run topic discovery, user selection, paper search, and two-agent review."""

    topics = await discover_recent_conference_topics(prompt)
    print(topics)
    selected_topic = input(
        "\nSelect a topic by number or paste a topic name: "
    ).strip()
    if not selected_topic:
        raise ValueError("A topic selection is required to continue.")

    paper_context = await search_papers_for_topic(selected_topic, topics)
    print("\n# Focused Paper Search\n")
    print(paper_context)

    return await review_selected_topic(selected_topic, paper_context)
