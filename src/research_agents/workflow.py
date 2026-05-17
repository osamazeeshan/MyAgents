"""Agent definitions and orchestration for research work."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    handoff,
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

CONFERENCE_REVIEW_FOLLOW_UP_INSTRUCTIONS = """
You are a conference literature-review follow-up assistant. The user has just
completed a two-reviewer conference literature review with verified paper
records. Answer follow-up questions conversationally while staying grounded in
the supplied selected topic, paper context, and review context. Do not invent
new citations. If the user asks for more literature, clearly label suggestions
as follow-up search leads unless they are present in the supplied verified
paper set. When the user wants to read a complete paper, find code/datasets,
or prepare a reproduction repository, explain that the interactive workflow can
collect the required inputs step by step, optionally create a GitHub repository,
clone existing code when the user supplies a URL, and scaffold/test dummy data
when no user dataset is available yet.
"""

PAPER_READING_INSTRUCTIONS = """
You are a paper-reading assistant. Help the user deeply read one selected paper
from a verified paper set. Work only from the paper details, URL/PDF/text, and
review context supplied by the workflow. If the full paper text is unavailable,
say exactly what is missing and ask the user to provide a PDF URL, abstract, or
text excerpt. Produce a structured reading with: problem, contributions, method,
math/algorithm details, experiments, datasets, results, limitations,
reproducibility checklist, and questions to discuss with the user. Do not
invent paper content not present in the supplied context.
"""

ARTIFACT_SCOUT_INSTRUCTIONS = """
You are a code-and-dataset scout for research papers. Search for official and
credible unofficial implementation repositories, project pages, model cards,
datasets, benchmark leaderboards, data licenses, and setup instructions for the
selected paper. Prefer official paper/project/GitHub/Hugging Face/Papers with
Code/dataset-homepage links. Clearly separate verified artifacts from search
leads and unknowns. End by asking the user which code URL, dataset URL, and
local repository path should be used for reproduction.
"""

REPRODUCTION_PLANNER_INSTRUCTIONS = """
You are a reproduction repository planner and implementation guide. Given the
selected paper, paper-reading notes, code/dataset artifact notes, and the local
repository preparation result, create an actionable implementation plan. If an
existing codebase was cloned, explain the repo layout to inspect, environment
setup, data download steps, smoke tests, and experiment commands. If a scaffold
was created because no code was available, use the generated dummy dataset,
baseline module, and pytest smoke test as the first runnable checkpoint before
proposing a minimal clean-room implementation structure, module responsibilities,
pseudocode, tests, and an incremental coding checklist. Ask the user for
confirmation before each next implementation step.
"""

PAPER_CODING_AGENT_INSTRUCTIONS = """
You are a coding agent for research-paper implementation. The user supplies a
paper identifier or title, implementation constraints, and optional idea prompts.
Your job is to turn that into an actionable coding workspace plan.

Rules:
- Treat arXiv IDs, DOIs, URLs, and titles as paper identifiers to verify or ask
  the user to verify; do not invent paper details when the identifier is
  ambiguous.
- When web search is available, use it to ground paper metadata, official code,
  datasets, and project pages. Prefer official paper pages, arXiv, DOI landing
  pages, Papers with Code, GitHub, Hugging Face, and dataset homepages.
- Design for implementation: break the method into modules, data contracts,
  tests, smoke commands, and incremental checkpoints.
- Include a section called "LLM idea loop" with safe, testable variants,
  ablations, promptable hypotheses, and metrics the user can ask an LLM to try.
- Clearly separate confirmed facts, assumptions, and open questions.
- If the user gives links or ideas, incorporate them as optional experiments,
  not as verified paper claims unless supported by the provided context.
"""

CONFERENCE_VENUES = (
    "NeurIPS, ICML, ICLR, AAAI, CVPR, ECCV, ICCV, WACV, and closely related "
    "top-tier workshops or proceedings"
)

NUMBERED_TOPIC_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(\d{1,3})[.)]\s+(.+?)(?:\*\*)?\s*$"
)
QUERY_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?\*?\*?Query\*?\*?\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
MAX_FOCUSED_PAPERS = 20
REPRODUCTION_REPOS_DIR = Path("reproduction_repos")
EXIT_COMMANDS = {"", "q", "quit", "exit"}
FOLLOW_UP_REQUEST_PATTERN = re.compile(
    r"^\s*(?:"
    r"can|could|would|will|please|refine|expand|explain|compare|contrast|"
    r"summarize|list|show|tell|what|which|who|when|where|why|how"
    r")\b|\?\s*$",
    re.IGNORECASE,
)
FOLLOW_UP_INVITATION_PATTERN = re.compile(
    r"(?:would you like|do you want|shall i|should i|want me to|"
    r"ask me|tell me|select|choose|pick|refine|expand).{0,240}\?\s*$",
    re.IGNORECASE | re.DOTALL,
)
PAPER_READING_REQUEST_PATTERN = re.compile(
    r"\b(?:read|deep dive|full paper|complete paper|walk through|discuss)\b",
    re.IGNORECASE,
)
ARTIFACT_REQUEST_PATTERN = re.compile(
    r"\b(?:code|repo|repository|github|dataset|data set|artifacts?|implementation)\b",
    re.IGNORECASE,
)
REPRODUCTION_REQUEST_PATTERN = re.compile(
    r"\b(?:create|clone|scaffold|implement|reproduce|reproduction)\b.*\b(?:repo|repository|code|paper)\b",
    re.IGNORECASE,
)


def build_handoff_tool_name(agent_name: str) -> str:
    """Return a stable function-call-safe handoff tool name for a display name.

    The OpenAI function-calling APIs only allow letters, digits, and underscores
    in tool names. Agent display names may contain spaces, punctuation, or other
    special characters, so keep those names for users while exposing a sanitized
    handoff tool name to the model.
    """

    normalized = re.sub(r"\W+", "_", agent_name, flags=re.ASCII).strip("_").lower()
    if not normalized:
        normalized = "agent"
    if normalized[0].isdigit():
        normalized = f"agent_{normalized}"
    return f"transfer_to_{normalized}"


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
        handoffs=[
            handoff(planner, tool_name_override=build_handoff_tool_name(planner.name)),
            handoff(scout, tool_name_override=build_handoff_tool_name(scout.name)),
            handoff(reviewer, tool_name_override=build_handoff_tool_name(reviewer.name)),
        ],
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


def build_conference_review_follow_up_agent() -> Agent:
    """Build an agent for post-review conference-literature follow-ups."""

    settings = load_settings()
    configure_model_provider(settings)
    return Agent(
        name="Conference Review Follow-up Assistant",
        instructions=CONFERENCE_REVIEW_FOLLOW_UP_INSTRUCTIONS,
        model=settings.model,
        tools=[save_research_note],
    )


def build_paper_reading_agent() -> Agent:
    """Build an agent for grounded full-paper reading and discussion."""

    settings = load_settings()
    configure_model_provider(settings)
    return Agent(
        name="Paper Reading Assistant",
        instructions=PAPER_READING_INSTRUCTIONS,
        model=settings.model,
        tools=[save_research_note],
    )


def build_artifact_scout_agent() -> Agent:
    """Build an agent that searches for implementation code and datasets."""

    settings = load_settings()
    configure_model_provider(settings)
    tools = [save_research_note]
    web_search = _build_web_search_tool(settings)
    if web_search is not None:
        tools.insert(0, web_search)
    return Agent(
        name="Code and Dataset Scout",
        instructions=ARTIFACT_SCOUT_INSTRUCTIONS,
        model=settings.model,
        tools=tools,
    )


def build_reproduction_planner_agent() -> Agent:
    """Build an agent that guides reproduction repository work."""

    settings = load_settings()
    configure_model_provider(settings)
    return Agent(
        name="Reproduction Repository Planner",
        instructions=REPRODUCTION_PLANNER_INSTRUCTIONS,
        model=settings.model,
        tools=[save_research_note],
    )


def build_paper_coding_agent() -> Agent:
    """Build an agent for paper-to-code implementation planning."""

    settings = load_settings()
    configure_model_provider(settings)
    tools = [save_research_note, search_verified_recent_papers]
    web_search = _build_web_search_tool(settings)
    if web_search is not None:
        tools.insert(0, web_search)
    return Agent(
        name="Paper Coding Agent",
        instructions=PAPER_CODING_AGENT_INSTRUCTIONS,
        model=settings.model,
        tools=tools,
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

    # Prefer explicit menu/selection headings. Some model outputs use a numbered
    # heading such as "3. Select a Topic..." followed by blank-line-separated
    # menu entries, so blank lines alone cannot delimit menu blocks.
    for index, line in enumerate(lines):
        heading = line.lower()
        if "topic selection menu" not in heading and "select a topic" not in heading:
            continue
        menu = _extract_numbered_topic_block(lines[index + 1 :])
        if menu:
            return menu

    blocks = _extract_numbered_topic_blocks(lines)
    if not blocks:
        return {}

    # The prompt asks for the selection menu at the end. Prefer the longest
    # sequential numbered block, using the later block as a tie-breaker, so earlier
    # numbered prose does not mask the actual user-selectable menu.
    return max(enumerate(blocks), key=lambda item: (len(item[1]), item[0]))[1]


def _extract_numbered_topic_block(lines: list[str]) -> dict[int, str]:
    """Return the best topic block from the supplied lines."""

    blocks = _extract_numbered_topic_blocks(lines)
    if not blocks:
        return {}
    return max(enumerate(blocks), key=lambda item: (len(item[1]), item[0]))[1]


def _extract_numbered_topic_blocks(lines: list[str]) -> list[dict[int, str]]:
    """Collect sequential numbered topic blocks and adjacent Query lines."""

    blocks: list[dict[int, str]] = []
    current_block: dict[int, str] = {}
    current_number: int | None = None

    for line in lines:
        if current_block and line.lstrip().startswith("#"):
            blocks.append(current_block)
            current_block = {}
            current_number = None
            continue

        match = NUMBERED_TOPIC_PATTERN.match(line)
        if match:
            number = int(match.group(1))
            if current_block and number <= max(current_block):
                blocks.append(current_block)
                current_block = {}
            current_block[number] = _normalize_menu_topic(match.group(2))
            current_number = number
            continue

        query_match = QUERY_LINE_PATTERN.match(line)
        if query_match and current_number in current_block:
            query = _normalize_query_text(query_match.group(1))
            if query:
                current_block[current_number] = query

    if current_block:
        blocks.append(current_block)

    return blocks


def _normalize_menu_topic(raw_topic: str) -> str:
    """Remove common Markdown adornments while preserving venue/year evidence."""

    topic = raw_topic.strip()
    topic = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", topic)
    topic = topic.replace("**", "").replace("__", "").strip()
    return topic.rstrip(" :")


def _normalize_query_text(raw_query: str) -> str:
    """Normalize a displayed search query from a topic menu."""

    query = raw_query.strip().rstrip()
    query = query.strip('"“”')
    return query.strip()


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


def looks_like_follow_up_request(user_text: str) -> bool:
    """Return True when text is likely a conversational follow-up, not a topic."""

    return bool(FOLLOW_UP_REQUEST_PATTERN.search(user_text.strip()))


def output_invites_follow_up(output: str) -> bool:
    """Return True when the model's final lines appear to ask for more input."""

    return bool(FOLLOW_UP_INVITATION_PATTERN.search(output.strip()[-1000:]))


def looks_like_paper_reading_request(user_text: str) -> bool:
    """Return True when the user wants to read or discuss a full paper."""

    return bool(PAPER_READING_REQUEST_PATTERN.search(user_text.strip()))


def looks_like_artifact_request(user_text: str) -> bool:
    """Return True when the user wants paper code or dataset artifacts."""

    return bool(ARTIFACT_REQUEST_PATTERN.search(user_text.strip()))


def looks_like_reproduction_request(user_text: str) -> bool:
    """Return True when the user wants repository preparation or implementation."""

    return bool(REPRODUCTION_REQUEST_PATTERN.search(user_text.strip()))


async def answer_conference_topic_follow_up(
    user_query: str, discovery_context: str
) -> str:
    """Answer a conversational follow-up during topic discovery."""

    follow_up_prompt = f"""
The user is in the conference-topic discovery step. Continue the conversation
instead of starting paper search or review. Answer the user's relevant question
directly, using the existing discovery context below as grounding. If the user
asks to refine or expand topics, provide the requested refinement and then end
with an updated numbered topic-selection menu when useful.

Existing discovery context:
{discovery_context}

User follow-up:
{user_query}
""".strip()
    result = await Runner.run(
        build_conference_topic_scout(), follow_up_prompt, max_turns=12
    )
    return result.final_output


def format_conference_review_follow_up_prompt(
    question: str, selected_topic: str, paper_context: str, review_context: str
) -> str:
    """Build a grounded prompt for post-review follow-up questions."""

    return f"""
The user is asking a follow-up after an interactive conference literature
review. Keep the conversation open and answer directly from the supplied
context.

Selected topic:
{selected_topic}

Verified paper search and organization context:
{paper_context}

Two-reviewer literature-review context:
{review_context}

User follow-up question:
{question}
""".strip()


async def answer_conference_review_follow_up(
    question: str, selected_topic: str, paper_context: str, review_context: str
) -> str:
    """Answer a user follow-up after the two-reviewer conference review."""

    result = await Runner.run(
        build_conference_review_follow_up_agent(),
        format_conference_review_follow_up_prompt(
            question, selected_topic, paper_context, review_context
        ),
        max_turns=12,
    )
    return result.final_output


def format_paper_reading_prompt(
    paper_request: str,
    paper_source: str,
    selected_topic: str,
    paper_context: str,
    review_context: str,
) -> str:
    """Build a grounded full-paper reading prompt."""

    return f"""
The user wants to read and discuss a complete paper from the literature-review
workflow. Use only supplied context and explicitly identify missing full-text
information.

Selected topic:
{selected_topic}

User-selected paper or reading goal:
{paper_request}

User-provided paper URL/text/PDF details:
{paper_source}

Verified paper search context:
{paper_context}

Review context:
{review_context}
""".strip()


async def read_selected_paper(
    paper_request: str,
    paper_source: str,
    selected_topic: str,
    paper_context: str,
    review_context: str,
) -> str:
    """Have the paper-reading agent analyze one selected paper."""

    result = await Runner.run(
        build_paper_reading_agent(),
        format_paper_reading_prompt(
            paper_request, paper_source, selected_topic, paper_context, review_context
        ),
        max_turns=16,
    )
    return result.final_output


def format_artifact_scout_prompt(
    paper_request: str, selected_topic: str, paper_context: str, reading_context: str
) -> str:
    """Build a prompt for finding code and dataset artifacts."""

    return f"""
Find implementation code and datasets for the selected paper. Search only for
artifacts related to this paper/topic and label confidence clearly.

Selected topic:
{selected_topic}

User-selected paper:
{paper_request}

Verified paper search context:
{paper_context}

Paper-reading notes, if any:
{reading_context}
""".strip()


async def scout_code_and_datasets(
    paper_request: str, selected_topic: str, paper_context: str, reading_context: str
) -> str:
    """Find code repositories and datasets for one selected paper."""

    result = await Runner.run(
        build_artifact_scout_agent(),
        format_artifact_scout_prompt(
            paper_request, selected_topic, paper_context, reading_context
        ),
        max_turns=16,
    )
    return result.final_output


def _safe_repo_name(name: str) -> str:
    """Normalize a user-supplied repository directory name."""

    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-._")
    return safe or "paper-reproduction"


def _github_repo_creator_plugin_script() -> Path:
    """Return the repo-local GitHub plugin script path."""

    return (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "github-repo-creator"
        / "scripts"
        / "create_github_repo.py"
    )


def _create_github_repository_with_plugin(
    repo_name: str, description: str, private: bool, owner: str
) -> str:
    """Create a GitHub repository through the repo-local plugin helper."""

    script = _github_repo_creator_plugin_script()
    if not script.exists():
        raise FileNotFoundError(f"GitHub repo creator plugin script not found: {script}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
        output_path = Path(output_file.name)

    cmd = [
        sys.executable,
        str(script),
        repo_name,
        "--description",
        description or "Research paper reproduction workspace",
        "--visibility",
        "private" if private else "public",
        "--output",
        str(output_path),
    ]
    if owner:
        cmd.extend(["--owner", owner])

    try:
        if sys.stdin.isatty():
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(
            "GitHub repository creation plugin failed" + (f": {stderr}" if stderr else ".")
        ) from exc
    finally:
        output_path.unlink(missing_ok=True)

    html_url = data.get("html_url")
    if not isinstance(html_url, str) or not html_url:
        raise RuntimeError("GitHub repository creation plugin did not return html_url.")
    return html_url


def _create_github_repository_with_token(
    repo_name: str, description: str, private: bool, owner: str, token: str
) -> str:
    """Create a GitHub repository directly with a token fallback."""

    payload = {
        "name": repo_name,
        "description": description or "Research paper reproduction workspace",
        "private": private,
        "auto_init": False,
    }
    if owner:
        endpoint = f"https://api.github.com/orgs/{owner.strip()}/repos"
    else:
        endpoint = "https://api.github.com/user/repos"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ResearchAgent/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - requires live GitHub API
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub repository creation failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - requires network failure
        raise RuntimeError(f"GitHub repository creation failed: {exc.reason}") from exc

    html_url = data.get("html_url")
    if not isinstance(html_url, str) or not html_url:
        raise RuntimeError("GitHub repository creation succeeded but no html_url was returned.")
    return html_url


def create_github_repository(
    repo_name: str,
    *,
    description: str = "",
    private: bool = False,
    owner: str = "",
    token: str | None = None,
) -> str:
    """Create a GitHub repository and return its browser URL.

    Repository creation is delegated to the repo-local GitHub plugin so an
    interactive terminal can ask the user to log in with ``gh auth login`` when
    necessary. Supplying ``owner`` creates an organization repo; leaving it blank
    creates a repository for the authenticated user.
    """

    safe_name = _safe_repo_name(repo_name)
    if token:
        return _create_github_repository_with_token(
            safe_name, description, private, owner, token
        )

    try:
        return _create_github_repository_with_plugin(
            safe_name, description, private, owner
        )
    except FileNotFoundError:
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if github_token:
            return _create_github_repository_with_token(
                safe_name, description, private, owner, github_token
            )
        raise


def _write_dummy_dataset_scaffold(repo_path: Path, repo_title: str) -> None:
    """Write a runnable clean-room Python scaffold with dummy data and tests."""

    package_dir = repo_path / "src" / "reproduction_baseline"
    package_dir.mkdir(parents=True, exist_ok=True)
    (repo_path / "tests").mkdir(exist_ok=True)
    (repo_path / "data").mkdir(exist_ok=True)

    (repo_path / "README.md").write_text(
        f"# {repo_title}\n\n"
        "Clean-room research reproduction scaffold. It includes a tiny dummy "
        "classification dataset, a baseline implementation, and pytest checks "
        "so agents can validate code before replacing the placeholder with the "
        "paper-specific method.\n\n"
        "## Quick start\n\n"
        "```bash\n"
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -e .[test]\n"
        "pytest\n"
        "python -m reproduction_baseline data/dummy_dataset.csv\n"
        "```\n\n"
        "## Next steps\n\n"
        "1. Replace or extend `data/dummy_dataset.csv` with the user-provided dataset.\n"
        "2. Implement the paper method incrementally in `src/reproduction_baseline/`.\n"
        "3. Add regression tests for preprocessing, training, and evaluation.\n",
        encoding="utf-8",
    )
    (repo_path / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = [\"hatchling>=1.25\"]\n"
        "build-backend = \"hatchling.build\"\n\n"
        "[project]\n"
        f"name = \"{_safe_repo_name(repo_title).lower()}\"\n"
        "version = \"0.1.0\"\n"
        "description = \"Clean-room research reproduction scaffold\"\n"
        "requires-python = \">=3.10\"\n\n"
        "[project.optional-dependencies]\n"
        "test = [\"pytest>=8\"]\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "packages = [\"src/reproduction_baseline\"]\n",
        encoding="utf-8",
    )
    (repo_path / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.DS_Store\n",
        encoding="utf-8",
    )
    (repo_path / "data" / "dummy_dataset.csv").write_text(
        "feature_a,feature_b,label\n"
        "0.10,1.20,negative\n"
        "0.30,1.00,negative\n"
        "0.80,0.40,positive\n"
        "0.90,0.20,positive\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text(
        "\"\"\"Minimal baseline utilities for a paper reproduction scaffold.\"\"\"\n\n"
        "from .baseline import DatasetSummary, load_dataset, majority_label\n\n"
        "__all__ = [\"DatasetSummary\", \"load_dataset\", \"majority_label\"]\n",
        encoding="utf-8",
    )
    (package_dir / "baseline.py").write_text(
        "\"\"\"Dummy-dataset baseline used to smoke-test reproduction work.\"\"\"\n\n"
        "from __future__ import annotations\n\n"
        "import csv\n"
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n\n\n"
        "@dataclass(frozen=True)\n"
        "class DatasetSummary:\n"
        "    rows: int\n"
        "    labels: tuple[str, ...]\n"
        "    majority: str\n\n\n"
        "def load_dataset(path: str | Path) -> list[dict[str, str]]:\n"
        "    with Path(path).open(newline=\"\", encoding=\"utf-8\") as handle:\n"
        "        return list(csv.DictReader(handle))\n\n\n"
        "def majority_label(rows: list[dict[str, str]], label_column: str = \"label\") -> DatasetSummary:\n"
        "    if not rows:\n"
        "        raise ValueError(\"dataset must contain at least one row\")\n"
        "    counts: dict[str, int] = {}\n"
        "    for row in rows:\n"
        "        label = row[label_column]\n"
        "        counts[label] = counts.get(label, 0) + 1\n"
        "    majority = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]\n"
        "    return DatasetSummary(rows=len(rows), labels=tuple(sorted(counts)), majority=majority)\n\n\n"
        "def main() -> None:\n"
        "    import argparse\n\n"
        "    parser = argparse.ArgumentParser(description=\"Summarize a reproduction dataset.\")\n"
        "    parser.add_argument(\"dataset\", type=Path)\n"
        "    args = parser.parse_args()\n"
        "    print(majority_label(load_dataset(args.dataset)))\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n",
        encoding="utf-8",
    )
    (repo_path / "tests" / "test_baseline.py").write_text(
        "from pathlib import Path\n\n"
        "from reproduction_baseline import load_dataset, majority_label\n\n\n"
        "def test_dummy_dataset_majority_label() -> None:\n"
        "    dataset = Path(__file__).resolve().parents[1] / \"data\" / \"dummy_dataset.csv\"\n"
        "    rows = load_dataset(dataset)\n"
        "    summary = majority_label(rows)\n\n"
        "    assert summary.rows == 4\n"
        "    assert summary.labels == (\"negative\", \"positive\")\n"
        "    assert summary.majority == \"negative\"\n",
        encoding="utf-8",
    )


def _commit_scaffold(repo_path: Path) -> None:
    """Create an initial local commit for generated scaffolds."""

    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ResearchAgent",
            "-c",
            "user.email=researchagent@example.com",
            "commit",
            "-m",
            "Initial reproduction scaffold",
        ],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _add_github_remote(repo_path: Path, github_url: str) -> str:
    """Add a GitHub remote and return the remote name used."""

    remotes = subprocess.run(
        ["git", "remote"], cwd=repo_path, check=True, capture_output=True, text=True
    ).stdout.split()
    remote_name = "origin" if "origin" not in remotes else "github"
    if remote_name not in remotes:
        subprocess.run(
            ["git", "remote", "add", remote_name, github_url],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    return remote_name


def prepare_reproduction_repository(
    repo_name: str,
    code_url: str = "",
    dataset_url: str = "",
    github_repo: str = "",
    github_private: bool = False,
    github_owner: str = "",
    github_create_func: Callable[..., str] | None = None,
) -> str:
    """Clone existing code or create a tested scaffold, then optionally create GitHub."""

    REPRODUCTION_REPOS_DIR.mkdir(parents=True, exist_ok=True)
    repo_path = REPRODUCTION_REPOS_DIR / _safe_repo_name(repo_name)
    if repo_path.exists() and any(repo_path.iterdir()):
        raise FileExistsError(f"Repository path already exists and is not empty: {repo_path}")

    code_url = code_url.strip()
    dataset_url = dataset_url.strip()
    github_repo = github_repo.strip()
    actions: list[str] = []
    if code_url:
        subprocess.run(
            ["git", "clone", code_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        actions.append(f"Cloned existing implementation from {code_url}.")
    else:
        repo_path.mkdir(parents=True, exist_ok=True)
        _write_dummy_dataset_scaffold(repo_path, _safe_repo_name(repo_name))
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True, text=True)
        _commit_scaffold(repo_path)
        actions.append(
            "Created a new clean-room scaffold repository with dummy data, baseline code, and tests."
        )

    if dataset_url:
        (repo_path / "DATASET.md").write_text(
            f"# Dataset\n\nSelected dataset URL or instructions:\n\n{dataset_url}\n",
            encoding="utf-8",
        )
        actions.append("Saved dataset notes in DATASET.md.")

    if github_repo:
        create_func = github_create_func or create_github_repository
        github_url = create_func(
            github_repo,
            description=f"Research reproduction workspace for {repo_name}",
            private=github_private,
            owner=github_owner,
        )
        remote_name = _add_github_remote(repo_path, github_url)
        actions.append(
            f"Created GitHub repository {github_url} and added it as remote '{remote_name}'."
        )
        actions.append(
            "Push when ready with: "
            f"git -C {repo_path} push -u {remote_name} HEAD:main"
        )

    return "\n".join([*actions, f"Local path: {repo_path}"])


def format_reproduction_prompt(
    selected_topic: str,
    paper_request: str,
    paper_context: str,
    reading_context: str,
    artifact_context: str,
    repo_result: str,
) -> str:
    """Build a prompt for reproduction planning after repo preparation."""

    return f"""
The user approved repository preparation for reproducing a paper. Create the
next-step plan and ask for confirmation before any implementation step.

Selected topic:
{selected_topic}

Selected paper:
{paper_request}

Verified paper context:
{paper_context}

Paper-reading notes:
{reading_context}

Code and dataset artifact notes:
{artifact_context}

Repository preparation result:
{repo_result}
""".strip()


async def plan_reproduction_repository(
    selected_topic: str,
    paper_request: str,
    paper_context: str,
    reading_context: str,
    artifact_context: str,
    repo_result: str,
) -> str:
    """Create a user-confirmable plan for repo-based reproduction work."""

    result = await Runner.run(
        build_reproduction_planner_agent(),
        format_reproduction_prompt(
            selected_topic,
            paper_request,
            paper_context,
            reading_context,
            artifact_context,
            repo_result,
        ),
        max_turns=12,
    )
    return result.final_output


def format_paper_coding_prompt(
    paper_identifier: str, implementation_goal: str = "", idea_context: str = ""
) -> str:
    """Build a prompt for the web coding workspace agent."""

    goal = implementation_goal.strip() or "Create an implementation plan."
    ideas = idea_context.strip() or "No extra idea prompts supplied yet."
    return f"""
The user opened the coding workspace for a research-paper implementation.
Produce a practical coding plan, not just a literature summary.

Paper ID or title:
{paper_identifier}

Implementation goal and constraints:
{goal}

Ideas, links, variants, or LLM experiment prompts to consider:
{ideas}

Return:
1. Paper identity and verification status.
2. Implementation assumptions and missing inputs.
3. Proposed repository/modules/classes/functions.
4. Data, training, evaluation, and smoke-test commands.
5. Incremental coding checkpoints.
6. LLM idea loop with safe variants, ablations, and metrics.
""".strip()


async def run_paper_coding_agent(
    paper_identifier: str, implementation_goal: str = "", idea_context: str = ""
) -> str:
    """Run the paper coding agent for a paper identifier or title."""

    result = await Runner.run(
        build_paper_coding_agent(),
        format_paper_coding_prompt(paper_identifier, implementation_goal, idea_context),
        max_turns=16,
    )
    return result.final_output


async def _run_paper_reading_sequence(
    initial_request: str,
    selected_topic: str,
    paper_context: str,
    review_context: str,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> tuple[str, str]:
    """Collect user input and run the paper-reading agent."""

    paper_request = input_func(
        "Which paper should I read? Enter a paper number/title, or press Enter to use your request: "
    ).strip() or initial_request
    paper_source = input_func(
        "Paste a PDF URL, paper URL, abstract, or text excerpt for grounding (Enter to use verified context only): "
    ).strip()

    output_func("\n# Paper Reading Assistant\n")
    reading = await read_selected_paper(
        paper_request, paper_source, selected_topic, paper_context, review_context
    )
    output_func(reading)
    return paper_request, reading


async def _run_artifact_scout_sequence(
    initial_request: str,
    selected_topic: str,
    paper_context: str,
    reading_context: str,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> tuple[str, str]:
    """Collect user input and search for code and dataset artifacts."""

    paper_request = input_func(
        "Which paper should I find code/data for? Enter a paper number/title, or press Enter to use your request: "
    ).strip() or initial_request

    output_func("\n# Code and Dataset Scout\n")
    artifacts = await scout_code_and_datasets(
        paper_request, selected_topic, paper_context, reading_context
    )
    output_func(artifacts)
    return paper_request, artifacts


async def _run_reproduction_sequence(
    initial_request: str,
    selected_topic: str,
    paper_context: str,
    reading_context: str,
    artifact_context: str,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> str:
    """Ask for approval at each repo-preparation step, then plan reproduction."""

    paper_request = input_func(
        "Which paper should the repo reproduce? Enter a paper number/title, or press Enter to use your request: "
    ).strip() or initial_request
    code_url = input_func(
        "Existing code URL to clone (press Enter if no implementation is available): "
    ).strip()
    dataset_url = input_func(
        "Dataset URL or setup notes (press Enter to use the dummy dataset scaffold): "
    ).strip()
    repo_name = input_func(
        "Local repo directory name under reproduction_repos/: "
    ).strip()
    github_repo = input_func(
        "GitHub repo name to create (press Enter to skip GitHub creation): "
    ).strip()
    github_owner = ""
    github_private = False
    if github_repo:
        github_owner = input_func(
            "GitHub organization/user owner (press Enter for the authenticated user): "
        ).strip()
        github_private = input_func(
            "Make the GitHub repo private? [y/N]: "
        ).strip().lower() in {"y", "yes"}
    if not repo_name:
        repo_name = _safe_repo_name(paper_request)

    summary = (
        f"Prepare repo '{repo_name}'"
        + (f" by cloning {code_url}" if code_url else " as a new scaffold with dummy data/tests")
        + (f" with dataset notes from {dataset_url}" if dataset_url else "")
        + (f" and create GitHub repo {github_repo}" if github_repo else "")
    )
    confirmation = input_func(f"Confirm this step? {summary} [y/N]: ").strip().lower()
    if confirmation not in {"y", "yes"}:
        return "Repository preparation cancelled by user before any local changes."

    repo_result = prepare_reproduction_repository(
        repo_name,
        code_url,
        dataset_url,
        github_repo=github_repo,
        github_private=github_private,
        github_owner=github_owner,
    )
    output_func("\n# Reproduction Repository Preparation\n")
    output_func(repo_result)

    plan = await plan_reproduction_repository(
        selected_topic,
        paper_request,
        paper_context,
        reading_context,
        artifact_context,
        repo_result,
    )
    output_func("\n# Reproduction Implementation Plan\n")
    output_func(plan)
    return f"{repo_result}\n\n{plan}"


async def run_research_workflow(prompt: str) -> str:
    """Run the research workflow and return the final agent output."""

    result = await Runner.run(build_research_orchestrator(), prompt)
    return result.final_output


async def run_interactive_research_workflow(
    prompt: str,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> str:
    """Run a persistent research-agent conversation in the terminal."""

    agent = build_research_orchestrator()
    session = SQLiteSession(f"research-agents-{uuid4().hex}")
    outputs: list[str] = []

    result = await Runner.run(agent, prompt, session=session)
    outputs.append(result.final_output)
    output_func(result.final_output)

    while True:
        follow_up = input_func(
            "\nAsk a follow-up, or press Enter to exit: "
        ).strip()
        if follow_up.lower() in EXIT_COMMANDS:
            break

        result = await Runner.run(agent, follow_up, session=session)
        outputs.append(result.final_output)
        output_func(f"\n{result.final_output}")

    return "\n\n".join(outputs)


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

    selected_topic = selected_topic.strip()
    if selected_topic.isdigit():
        if not discovery_context.strip():
            raise ValueError(
                "Numeric topic selections require the topic-discovery context. "
                "Paste a topic name or rerun the interactive conference-review workflow."
            )
        selected_topic = resolve_topic_selection(selected_topic, discovery_context)

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


async def run_interactive_conference_literature_review(
    prompt: str = "",
    *,
    keep_conversation_open: bool = False,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> str:
    """Run topic discovery, paper search, review, and optional follow-ups."""

    topics = await discover_recent_conference_topics(prompt)
    output_func(topics)

    while True:
        selection = input_func(
            "\nSelect a topic by number, paste a topic name, "
            "or ask a follow-up question (Enter to exit): "
        ).strip()
        if selection.lower() in EXIT_COMMANDS:
            return "Topic selection cancelled before paper search."

        if selection.isdigit() or not looks_like_follow_up_request(selection):
            selected_topic = resolve_topic_selection(selection, topics)
            if selected_topic != selection:
                output_func(f"\nSelected topic {selection}: {selected_topic}")
            break

        follow_up_answer = await answer_conference_topic_follow_up(selection, topics)
        topics = f"{topics}\n\n# Follow-up: {selection}\n\n{follow_up_answer}"
        output_func("\n# Topic Scout Follow-up\n")
        output_func(follow_up_answer)

    paper_context = await search_papers_for_topic(selected_topic, topics)
    output_func("\n# Focused Paper Search\n")
    output_func(paper_context)

    review = await review_selected_topic(selected_topic, paper_context)
    if not keep_conversation_open:
        return review

    outputs = [review]
    latest_reading_context = ""
    latest_artifact_context = ""
    output_func("\n# Two-Reviewer Critical Literature Review\n")
    output_func(review)

    while True:
        follow_up = input_func(
            "\nAsk a follow-up, request 'read paper', 'find code/data', "
            "or 'create repo' (Enter to exit): "
        ).strip()
        if follow_up.lower() in EXIT_COMMANDS:
            break

        review_context = "\n\n".join(outputs)
        if looks_like_reproduction_request(follow_up):
            reproduction = await _run_reproduction_sequence(
                follow_up,
                selected_topic,
                paper_context,
                latest_reading_context,
                latest_artifact_context,
                input_func,
                output_func,
            )
            outputs.append(f"# Reproduction request: {follow_up}\n\n{reproduction}")
            continue

        if looks_like_artifact_request(follow_up):
            paper_request, artifacts = await _run_artifact_scout_sequence(
                follow_up,
                selected_topic,
                paper_context,
                latest_reading_context or review_context,
                input_func,
                output_func,
            )
            latest_artifact_context = artifacts
            outputs.append(f"# Code/data scout for: {paper_request}\n\n{artifacts}")
            continue

        if looks_like_paper_reading_request(follow_up):
            paper_request, reading = await _run_paper_reading_sequence(
                follow_up,
                selected_topic,
                paper_context,
                review_context,
                input_func,
                output_func,
            )
            latest_reading_context = reading
            outputs.append(f"# Paper reading for: {paper_request}\n\n{reading}")
            continue

        follow_up_answer = await answer_conference_review_follow_up(
            follow_up, selected_topic, paper_context, review_context
        )
        outputs.append(f"# Follow-up: {follow_up}\n\n{follow_up_answer}")
        output_func(f"\n{follow_up_answer}")

    return "\n\n".join(outputs)
