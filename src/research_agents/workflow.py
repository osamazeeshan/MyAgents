"""Agent definitions and orchestration for research work."""

from __future__ import annotations

import asyncio
import re
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
from .tools import (
    build_literature_search_query,
    save_research_note,
    search_verified_recent_papers,
    search_verified_recent_papers_markdown,
)

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
You are a grounded paper-search and bibliography agent for the topic selected by
user. You are given verified paper records from external scholarly indexes.

Non-negotiable citation rules:
- Do not add any paper that is not present in the verified paper records unless
  you first verify it with an available search tool and include its returned URL,
  DOI, or arXiv ID.
- Never invent paper titles, authors, venues, URLs, DOIs, or arXiv IDs.
- If the verified records are sparse, say so and propose search strings instead
  of fabricating missing literature.
- Keep papers within the requested 5-6 year window unless an older work is
  explicitly marked as a foundational exception.

Output rules:
- First show the full verified paper list, preserving every URL/DOI/arXiv ID.
- Then group papers by subtheme, method, benchmark/dataset, and application.
- For each paper, add a short relevance note grounded in its title, venue, year,
  and abstract snippet when present.
- End with coverage gaps and additional exact search queries to verify manually.
"""

METHODOLOGY_REVIEWER_INSTRUCTIONS = """
You are Reviewer A, a rigorous methodology and evidence-quality reviewer. Write
an in-depth critical literature review of the provided verified papers. Focus on
study design, baselines, datasets, evaluation metrics, ablations,
reproducibility, statistical validity, leakage/contamination risks, and whether
claims are supported by evidence. Include complete analysis, critical analysis,
limitations, and future directions. Do not cite or discuss papers absent from
the verified paper list unless you clearly label them as follow-up search leads.
"""

SYNTHESIS_REVIEWER_INSTRUCTIONS = """
You are Reviewer B, a field-synthesis reviewer. Write an in-depth critical
literature review of the provided verified papers. Focus on how the field has
evolved, main research clusters, theoretical assumptions, practical limitations,
open problems, promising directions, and how a new researcher should position a
project in this area. Include complete analysis, critical analysis, limitations,
and future directions. Do not cite or discuss papers absent from the verified
paper list unless you clearly label them as follow-up search leads.
"""

CONFERENCE_VENUES = (
    "NeurIPS, ICML, ICLR, AAAI, CVPR, ECCV, ICCV, WACV, and closely related "
    "top-tier workshops or proceedings"
)

NUMBERED_TOPIC_PATTERN = re.compile(r"^\s*(\d{1,3})[.)]\s+(.+?)\s*$")
MAX_FOCUSED_PAPERS = 20


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
    tools = [
        build_literature_search_query,
        save_research_note,
        search_verified_recent_papers,
    ]

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
    tools = [
        build_literature_search_query,
        save_research_note,
        search_verified_recent_papers,
    ]
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


def _extract_numbered_topics(discovery_context: str) -> dict[int, str]:
    """Extract the final numbered topic menu from a topic-discovery response."""

    lines = discovery_context.splitlines()

    # Prefer the explicit section title requested in the topic-scout prompt. This
    # lets menu items include wrapped evidence lines or blank spacing without
    # confusing earlier numbered prose with selectable topics.
    for index, line in enumerate(lines):
        if "topic selection menu" not in line.lower():
            continue
        menu: dict[int, str] = {}
        for menu_line in lines[index + 1 :]:
            if menu and menu_line.lstrip().startswith("#"):
                break
            match = NUMBERED_TOPIC_PATTERN.match(menu_line)
            if match:
                menu[int(match.group(1))] = _normalize_menu_topic(match.group(2))
        if menu:
            return menu

    blocks: list[dict[int, str]] = []
    current_block: dict[int, str] = {}
    for line in lines:
        match = NUMBERED_TOPIC_PATTERN.match(line)
        if match:
            current_block[int(match.group(1))] = _normalize_menu_topic(match.group(2))
            continue
        if current_block and not line.strip():
            blocks.append(current_block)
            current_block = {}
    if current_block:
        blocks.append(current_block)

    if not blocks:
        return {}

    # The prompt asks for the selection menu at the end. Prefer the last sizeable
    # numbered block, and fall back to the last numbered block if only one item was
    # found. This avoids accidentally treating earlier explanatory numbering as the
    # user-selectable menu.
    for block in reversed(blocks):
        if len(block) >= 2:
            return block
    return blocks[-1]


def _normalize_menu_topic(raw_topic: str) -> str:
    """Remove common Markdown adornments while preserving venue/year evidence."""

    topic = raw_topic.strip()
    topic = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", topic)
    topic = topic.replace("**", "").replace("__", "").strip()
    return topic


def resolve_topic_selection(selection: str, discovery_context: str) -> str:
    """Resolve a numeric menu selection to the actual topic text."""

    selection = selection.strip()
    if not selection:
        raise ValueError("A topic selection is required to continue.")

    if selection.isdigit():
        topics_by_number = _extract_numbered_topics(discovery_context)
        selected_number = int(selection)
        if selected_number not in topics_by_number:
            available = ", ".join(str(number) for number in sorted(topics_by_number))
            raise ValueError(
                f"Topic number {selected_number} was not found in the selection menu"
                + (f". Available numbers: {available}." if available else ".")
            )
        return topics_by_number[selected_number]

    return selection


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
- A short method note describing sources searched.
- A grouped, unnumbered list of all high-signal recent topics you found, with venue/year evidence.
- A final section titled "Topic selection menu". This must be the only numbered
  list in the answer; number each selectable topic consecutively as 1., 2., 3.,
  and so on.
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
    verified_papers = search_verified_recent_papers_markdown(
        topic=selected_topic,
        start_year=earliest_year,
        end_year=current_year,
        max_results=MAX_FOCUSED_PAPERS,
    )
    prompt = f"""
Current date: {datetime.now(timezone.utc).date().isoformat()}.
Selected topic: {selected_topic}
Discovery context from the previous step:
{discovery_context}

The following paper records were retrieved from external scholarly indexes.
Use these as the authoritative source of truth. Do not add papers that are not
in this verified list unless you verify them with the paper-search tool and
include a returned URL, DOI, or arXiv ID.

{verified_papers}

Return the full verified paper list first, then organize and annotate it. Keep
the final bibliography to no more than {MAX_FOCUSED_PAPERS} papers, prioritizing
the most relevant records with high citation counts and recent publication years.
{_web_search_availability_note(settings)}
""".strip()
    result = await Runner.run(
        build_conference_topic_scout().clone(instructions=PAPER_SCOUT_INSTRUCTIONS),
        prompt,
        max_turns=20,
    )
    return (
        f"{verified_papers}\n\n"
        "---\n\n"
        "# Paper Scout Organization\n\n"
        f"{result.final_output}"
    )


async def review_selected_topic(selected_topic: str, paper_context: str) -> str:
    """Run two LLM reviewers over the selected topic and paper set."""

    review_prompt = f"""
Selected topic: {selected_topic}
Recent paper set and notes:
{paper_context}

Write a critical, evidence-grounded literature review with explicit sections for
complete analysis, critical analysis, limitations, and future directions. Use
only the supplied verified paper set unless explicitly marking an item as a
suggested follow-up search.
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
    selection = input(
        "\nSelect a topic by number or paste a topic name: "
    ).strip()
    selected_topic = resolve_topic_selection(selection, topics)
    if selected_topic != selection:
        print(f"\nSelected topic {selection}: {selected_topic}")

    paper_context = await search_papers_for_topic(selected_topic, topics)
    print("\n# Focused Paper Search\n")
    print(paper_context)

    return await review_selected_topic(selected_topic, paper_context)
