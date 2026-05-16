"""Runtime configuration for the research agents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchAgentSettings:
    """Settings shared by the agent workflow."""

    model: str = "gpt-4.1"
    notes_dir: Path = Path("research_notes")


def load_settings() -> ResearchAgentSettings:
    """Load settings from environment variables."""

    return ResearchAgentSettings(
        model=os.getenv("RESEARCH_AGENTS_MODEL", ResearchAgentSettings.model),
        notes_dir=Path(os.getenv("RESEARCH_AGENTS_NOTES_DIR", "research_notes")),
    )
