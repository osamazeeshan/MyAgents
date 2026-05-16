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
