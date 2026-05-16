"""Tool functions that research agents can call."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from agents import function_tool

from .config import load_settings

_SAFE_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(value: str) -> str:
    slug = _SAFE_SLUG_PATTERN.sub("-", value.strip().lower()).strip("-._")
    return slug or "research-note"


@function_tool
def save_research_note(title: str, body: str) -> str:
    """Save a markdown research note locally and return its file path."""

    settings = load_settings()
    settings.notes_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = settings.notes_dir / f"{timestamp}-{_slugify(title)}.md"
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return str(path)


@function_tool
def build_literature_search_query(topic: str, method: str = "broad") -> str:
    """Create a reusable literature-search query for a research topic."""

    topic = topic.strip()
    if method == "systematic":
        return f'("{topic}" OR related terminology) AND (review OR meta-analysis OR benchmark OR dataset)'
    if method == "recent":
        return f'("{topic}") AND (2024 OR 2025 OR 2026) AND (paper OR preprint OR proceedings)'
    return f'("{topic}") AND (survey OR benchmark OR framework OR evaluation OR evidence)'
