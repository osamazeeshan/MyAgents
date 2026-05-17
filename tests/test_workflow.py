import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agents.tools import search_verified_recent_papers_markdown
from research_agents.workflow import (
    looks_like_follow_up_request,
    output_invites_follow_up,
    resolve_topic_selection,
)


DISCOVERY_CONTEXT_WITH_QUERY_MENU = '''
3. **Select a Topic to Explore Further** (run via OpenAI API with search queries):

   1. **Multimodal Reasoning with LLMs and Vision Transformers**
      *Query*: "Multimodal reasoning LLM vision transformers ICCV 2025 CVPR 2026"

   2. **LLM-Driven Visual Task Pipelines**
      *Query*: "LLM visual task pipelines CVPR 2025 ICML 2025"

   3. **Vision-Language Pre-training for Agents**
      *Query*: "Vision-language pre-training agents ICLR 2025 AAAI 2025"

   4. **Safety and Ethics in LLM-Visual Agents**
      *Query*: "LLM safety visual agents NeurIPS 2025 ICML 2025"

   5. **Video Understanding via Multimodal LLMs**
      *Query*: "Video understanding multimodal LLMs ECCV 2025 CVPR 2026"

   6. **Cross-Modal Few-Shot Learning**
      *Query*: "Cross-modal few-shot learning WACV 2026 NeurIPS 2025"
'''


def test_numeric_selection_uses_adjacent_query_text() -> None:
    assert (
        resolve_topic_selection("5", DISCOVERY_CONTEXT_WITH_QUERY_MENU)
        == "Video understanding multimodal LLMs ECCV 2025 CVPR 2026"
    )


def test_numeric_selection_reports_available_numbers() -> None:
    with pytest.raises(ValueError, match="Available numbers: 1, 2, 3, 4, 5, 6"):
        resolve_topic_selection("9", DISCOVERY_CONTEXT_WITH_QUERY_MENU)


def test_verified_paper_search_rejects_unresolved_numeric_topic() -> None:
    with pytest.raises(ValueError, match="numeric menu selection"):
        search_verified_recent_papers_markdown("5", 2020, 2026)


def test_follow_up_request_detection_for_refinement_prompts() -> None:
    assert looks_like_follow_up_request("expand on vision-language models")
    assert looks_like_follow_up_request("Can you compare federated learning clusters?")
    assert not looks_like_follow_up_request("Vision-language pre-training agents")


def test_output_follow_up_invitation_detection() -> None:
    output = (
        "Here are the clusters. Would you like to refine any section or "
        "expand on specific clusters (e.g., federated learning, "
        "vision-language models)?"
    )
    assert output_invites_follow_up(output)


def test_conference_review_follow_up_prompt_includes_review_context() -> None:
    from research_agents.workflow import format_conference_review_follow_up_prompt

    prompt = format_conference_review_follow_up_prompt(
        "What should I read first?", "Topic A", "Paper context", "Review context"
    )

    assert "What should I read first?" in prompt
    assert "Topic A" in prompt
    assert "Paper context" in prompt
    assert "Review context" in prompt


def test_interactive_conference_review_stays_open_for_followups(monkeypatch) -> None:
    import asyncio
    import research_agents.workflow as workflow

    async def fake_discover(prompt: str = "") -> str:
        return "# Topic selection menu\n1. Topic A"

    async def fake_search(selected_topic: str, discovery_context: str = "") -> str:
        return f"papers for {selected_topic}"

    async def fake_review(selected_topic: str, paper_context: str) -> str:
        return f"review for {selected_topic} with {paper_context}"

    async def fake_follow_up(
        question: str, selected_topic: str, paper_context: str, review_context: str
    ) -> str:
        assert question == "What are the gaps?"
        assert selected_topic == "Topic A"
        assert paper_context == "papers for Topic A"
        assert "review for Topic A" in review_context
        return "follow-up answer"

    monkeypatch.setattr(workflow, "discover_recent_conference_topics", fake_discover)
    monkeypatch.setattr(workflow, "search_papers_for_topic", fake_search)
    monkeypatch.setattr(workflow, "review_selected_topic", fake_review)
    monkeypatch.setattr(workflow, "answer_conference_review_follow_up", fake_follow_up)

    inputs = iter(["1", "What are the gaps?", ""])
    outputs: list[str] = []

    transcript = asyncio.run(
        workflow.run_interactive_conference_literature_review(
            "",
            keep_conversation_open=True,
            input_func=lambda prompt: next(inputs),
            output_func=outputs.append,
        )
    )

    assert "review for Topic A" in transcript
    assert "# Follow-up: What are the gaps?" in transcript
    assert "follow-up answer" in transcript
    assert any("Two-Reviewer Critical Literature Review" in item for item in outputs)
