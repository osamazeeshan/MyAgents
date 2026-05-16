"""Research-agent starter kit."""

from importlib import import_module

__all__ = [
    "build_research_orchestrator",
    "discover_recent_conference_topics",
    "run_interactive_conference_literature_review",
    "run_research_workflow",
    "search_papers_for_topic",
]


def __getattr__(name: str):
    if name in __all__:
        workflow = import_module("research_agents.workflow")
        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
