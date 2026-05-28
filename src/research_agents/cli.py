"""Command-line interface for the research-agent workflow."""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import format_local_model_presets
from .local_setup import ensure_first_run_local_model


PLAN_ONLY_TEXT = """Research-agent workflow preview:
1. Research Orchestrator decides which specialist should handle the request.
2. Research Planner scopes the problem and proposes a staged plan.
3. Literature Scout creates search terms, source categories, and reusable queries.
4. Critical Reviewer flags weak evidence, assumptions, and validation gaps.
5. Tools can save markdown notes or format literature-search queries.

Conference literature-review workflow preview (--conference-review):
1. Conference Topic Scout searches recent top AI/ML/CV venues from the current year
   and one year prior.
2. The CLI prints a single numbered topic menu and prompts you to select a topic.
3. You can ask follow-up questions before selecting, such as refining or expanding clusters.
4. If you enter a menu number, the workflow resolves it to that topic before searching.
5. The workflow verifies up to 20 recent/high-citation papers through scholarly indexes.
6. Two reviewer agents independently produce analysis, limitations, and future directions.
7. In interactive mode, follow-ups can read a paper, find code/data, create or
   clone a reproduction repo, optionally create a GitHub repo through the local
   GitHub plugin, and scaffold dummy-data tests before implementation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a research-agent workflow.")
    parser.add_argument("prompt", nargs="?", help="Research question or task to work on.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the local workflow preview without calling the OpenAI API.",
    )
    parser.add_argument(
        "--list-local-models",
        action="store_true",
        help="Print recommended local model presets for a 16GB Mac M2 and exit.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help=(
            "Keep a conversational session open after the first answer or "
            "after a conference review so you can ask follow-up questions."
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch ResearchAgent, the web interface for the research agents.",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="Host to bind when using --web (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8765,
        help="Port to bind when using --web (default: 8765).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open ResearchAgent in the default browser when using --web.",
    )
    parser.add_argument(
        "--skip-local-setup",
        action="store_true",
        help="Skip first-run local system detection and Ollama model activation.",
    )
    parser.add_argument(
        "--conference-review",
        action="store_true",
        help=(
            "Run the interactive top-conference topic discovery, focused paper "
            "search, and two-reviewer literature review workflow."
        ),
    )
    args = parser.parse_args()
    if not args.prompt and not (args.list_local_models or args.conference_review or args.web):
        parser.error(
            "prompt is required unless --list-local-models, --conference-review, or --web is used"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.list_local_models:
        print(format_local_model_presets())
        return

    if not args.skip_local_setup and not args.plan_only:
        setup = ensure_first_run_local_model(verbose=True)
        if setup.activated or setup.attempted:
            print(setup.message, file=sys.stderr)

    if args.web:
        from .web import run_server

        run_server(
            args.web_host,
            args.web_port,
            open_browser=args.open_browser,
            skip_local_setup=True,
        )
        return

    if args.plan_only:
        print(PLAN_ONLY_TEXT)
        print(f"Prompt: {args.prompt}")
        return

    if args.conference_review:
        from .workflow import run_interactive_conference_literature_review

        output = asyncio.run(
            run_interactive_conference_literature_review(
                args.prompt or "", keep_conversation_open=args.interactive
            )
        )
        if not args.interactive:
            print("\n# Two-Reviewer Critical Literature Review\n")
            print(output)
        return

    from .workflow import run_interactive_research_workflow, run_research_workflow

    if args.interactive or sys.stdin.isatty():
        asyncio.run(run_interactive_research_workflow(args.prompt))
        return

    output = asyncio.run(run_research_workflow(args.prompt))
    print(output)


if __name__ == "__main__":
    main()
