"""Command-line interface for the research-agent workflow."""

from __future__ import annotations

import argparse
import asyncio


PLAN_ONLY_TEXT = """Research-agent workflow preview:
1. Research Orchestrator decides which specialist should handle the request.
2. Research Planner scopes the problem and proposes a staged plan.
3. Literature Scout creates search terms, source categories, and reusable queries.
4. Critical Reviewer flags weak evidence, assumptions, and validation gaps.
5. Tools can save markdown notes or format literature-search queries.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a research-agent workflow.")
    parser.add_argument("prompt", help="Research question or task to work on.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the local workflow preview without calling the OpenAI API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan_only:
        print(PLAN_ONLY_TEXT)
        print(f"Prompt: {args.prompt}")
        return

    from .workflow import run_research_workflow

    output = asyncio.run(run_research_workflow(args.prompt))
    print(output)


if __name__ == "__main__":
    main()
