import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agents.web import build_home_page, format_memory_augmented_prompt


def test_home_page_contains_persistent_memory_layout() -> None:
    html = build_home_page()

    assert "YourResearchGuide" in html
    assert "Saved conversations" in html
    assert "Agent launchpads" in html
    assert "localStorage" in html
    assert "STORAGE_KEY" in html
    assert "yourresearchguide.conversations.v1" in html
    assert "memory: memoryContext()" in html


def test_memory_augmented_prompt_includes_memory_and_latest_request() -> None:
    prompt = format_memory_augmented_prompt("What should I read next?", "User: Topic A\nAgent: Read Paper B")

    assert "Saved conversation memory:" in prompt
    assert "User: Topic A" in prompt
    assert "Latest user request:" in prompt
    assert prompt.endswith("What should I read next?")


def test_memory_augmented_prompt_leaves_empty_memory_unchanged() -> None:
    assert format_memory_augmented_prompt("Fresh question", "") == "Fresh question"
