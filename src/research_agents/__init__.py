"""Research-agent starter kit."""

from importlib import import_module

__all__ = ["build_research_orchestrator", "run_research_workflow"]


def __getattr__(name: str):
    if name in __all__:
        workflow = import_module("research_agents.workflow")
        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
