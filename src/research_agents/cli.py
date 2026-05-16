"""Command-line interface for the research-agent workflow."""

from __future__ import annotations

import argparse
import asyncio

from .config import format_local_model_presets


PLAN_ONLY_TEXT = """Research-agent workflow preview:
1. Research Orchestrator decides which specialist should handle the request.
2. Research Planner scopes the problem and proposes a staged plan.
3. Literature Scout creates search terms, source categories, and reusable queries.
4. Critical Reviewer flags weak evidence, assumptions, and validation gaps.
5. Tools can save markdown notes or format literature-search queries.

Conference literature-review workflow preview (--conference-review):
1. Conference Topic Scout searches recent top AI/ML/CV venues from the current year
   and one year prior.
2. The CLI prints a numbered topic menu and prompts you to select a topic.
3. The same scout searches papers in the selected area from roughly the last 5-6 years.
4. Two reviewer agents independently produce methodology and field-synthesis literature reviews.
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
        "--conference-review",
        action="store_true",
        help=(
            "Run the interactive top-conference topic discovery, focused paper "
            "search, and two-reviewer literature review workflow."
        ),
    )
    args = parser.parse_args()
    if not args.prompt and not (args.list_local_models or args.conference_review):
        parser.error(
            "prompt is required unless --list-local-models or --conference-review is used"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.list_local_models:
        print(format_local_model_presets())
        return

    if args.plan_only:
        print(PLAN_ONLY_TEXT)
        print(f"Prompt: {args.prompt}")
        return

    if args.conference_review:
        from .workflow import run_interactive_conference_literature_review

        output = asyncio.run(
            run_interactive_conference_literature_review(args.prompt or "")
        )
        print("\n# Two-Reviewer Critical Literature Review\n")
        print(output)
        return

    from .workflow import run_research_workflow

    output = asyncio.run(run_research_workflow(args.prompt))
    print(output)


if __name__ == "__main__":
    main()
