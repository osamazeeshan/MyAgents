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


def test_follow_up_detection_for_paper_to_code_requests() -> None:
    from research_agents.workflow import (
        looks_like_artifact_request,
        looks_like_paper_reading_request,
        looks_like_reproduction_request,
    )

    assert looks_like_paper_reading_request("read the complete paper with me")
    assert looks_like_artifact_request("find the code and dataset")
    assert looks_like_reproduction_request("create a repo to implement this paper")


def test_prepare_reproduction_repository_creates_scaffold(tmp_path, monkeypatch) -> None:
    import research_agents.workflow as workflow

    monkeypatch.setattr(workflow, "REPRODUCTION_REPOS_DIR", tmp_path)

    result = workflow.prepare_reproduction_repository(
        "My Paper Repo!", dataset_url="https://example.com/dataset"
    )

    repo_path = tmp_path / "My-Paper-Repo"
    assert "Created a new clean-room scaffold repository" in result
    assert (repo_path / "README.md").exists()
    assert (repo_path / "src").is_dir()
    assert (repo_path / "tests").is_dir()
    assert (repo_path / "DATASET.md").read_text(encoding="utf-8").strip().endswith(
        "https://example.com/dataset"
    )


def test_interactive_conference_review_supports_paper_to_repo_path(monkeypatch, tmp_path) -> None:
    import asyncio
    import research_agents.workflow as workflow

    async def fake_discover(prompt: str = "") -> str:
        return "# Topic selection menu\n1. Topic A"

    async def fake_search(selected_topic: str, discovery_context: str = "") -> str:
        return f"papers for {selected_topic}"

    async def fake_review(selected_topic: str, paper_context: str) -> str:
        return f"review for {selected_topic} with {paper_context}"

    async def fake_read(
        paper_request: str,
        paper_source: str,
        selected_topic: str,
        paper_context: str,
        review_context: str,
    ) -> str:
        assert paper_request == "Paper A"
        assert paper_source == "https://example.com/paper.pdf"
        return "reading notes"

    async def fake_artifacts(
        paper_request: str, selected_topic: str, paper_context: str, reading_context: str
    ) -> str:
        assert paper_request == "Paper A"
        assert "reading notes" in reading_context
        return "artifact notes"

    async def fake_plan(
        selected_topic: str,
        paper_request: str,
        paper_context: str,
        reading_context: str,
        artifact_context: str,
        repo_result: str,
    ) -> str:
        assert paper_request == "Paper A"
        assert "reading notes" in reading_context
        assert "artifact notes" in artifact_context
        assert "Local path" in repo_result
        return "implementation plan"

    monkeypatch.setattr(workflow, "REPRODUCTION_REPOS_DIR", tmp_path)
    monkeypatch.setattr(workflow, "discover_recent_conference_topics", fake_discover)
    monkeypatch.setattr(workflow, "search_papers_for_topic", fake_search)
    monkeypatch.setattr(workflow, "review_selected_topic", fake_review)
    monkeypatch.setattr(workflow, "read_selected_paper", fake_read)
    monkeypatch.setattr(workflow, "scout_code_and_datasets", fake_artifacts)
    monkeypatch.setattr(workflow, "plan_reproduction_repository", fake_plan)

    inputs = iter(
        [
            "1",
            "read paper",
            "Paper A",
            "https://example.com/paper.pdf",
            "find code and dataset",
            "Paper A",
            "create repo to implement this paper",
            "Paper A",
            "",
            "https://example.com/dataset",
            "paper-a-repro",
            "yes",
            "",
        ]
    )
    outputs: list[str] = []

    transcript = asyncio.run(
        workflow.run_interactive_conference_literature_review(
            "",
            keep_conversation_open=True,
            input_func=lambda prompt: next(inputs),
            output_func=outputs.append,
        )
    )

    assert "# Paper reading for: Paper A" in transcript
    assert "# Code/data scout for: Paper A" in transcript
    assert "# Reproduction request: create repo to implement this paper" in transcript
    assert (tmp_path / "paper-a-repro" / "README.md").exists()


def test_agentarium_home_page_includes_name_and_modes() -> None:
    from research_agents.web import APP_NAME, build_home_page

    html = build_home_page()

    assert APP_NAME in html
    assert "Ask the agent crew" in html
    assert "Discover conference topics" in html
    assert "Review selected topic" in html
    assert "Conference follow-up" in html


def test_agentarium_requires_prompt_for_research_api() -> None:
    import asyncio
    import pytest
    from research_agents.web import handle_api_request

    with pytest.raises(ValueError, match="'prompt' is required"):
        asyncio.run(handle_api_request("/api/research", {"prompt": ""}))


def test_agentarium_research_api_delegates_to_workflow(monkeypatch) -> None:
    import asyncio
    import research_agents.workflow as workflow
    from research_agents.web import handle_api_request

    async def fake_run_research_workflow(prompt: str) -> str:
        assert prompt == "Map the field"
        return "mapped"

    monkeypatch.setattr(workflow, "run_research_workflow", fake_run_research_workflow)

    result = asyncio.run(handle_api_request("/api/research", {"prompt": "Map the field"}))

    assert result == {"output": "mapped"}


def test_agentarium_conference_review_api_chains_paper_search_and_review(monkeypatch) -> None:
    import asyncio
    import research_agents.workflow as workflow
    from research_agents.web import handle_api_request

    async def fake_search(topic: str, discovery_context: str = "") -> str:
        assert topic == "Agent benchmarks"
        assert discovery_context == "menu"
        return "papers"

    async def fake_review(topic: str, paper_context: str) -> str:
        assert topic == "Agent benchmarks"
        assert paper_context == "papers"
        return "review"

    monkeypatch.setattr(workflow, "search_papers_for_topic", fake_search)
    monkeypatch.setattr(workflow, "review_selected_topic", fake_review)

    result = asyncio.run(
        handle_api_request(
            "/api/conference/review",
            {"topic": "Agent benchmarks", "discovery_context": "menu"},
        )
    )

    assert result == {"paper_context": "papers", "review": "review"}
