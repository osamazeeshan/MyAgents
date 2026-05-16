"""Runtime configuration for the research agents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_API_KEY = "ollama"

LOCAL_MODEL_PRESETS: dict[str, dict[str, str]] = {
    "fast-small": {
        "model": "llama3.2:3b",
        "memory": "low",
        "use_case": "Fast planning, simple routing, and quick drafts.",
    },
    "balanced": {
        "model": "qwen3:8b",
        "memory": "medium",
        "use_case": "Best default balance for local agent reasoning on a 16GB M2 Mac.",
    },
    "balanced-alt": {
        "model": "llama3.1:8b",
        "memory": "medium",
        "use_case": "Strong general-purpose fallback with broad tool-calling compatibility.",
    },
    "coding": {
        "model": "qwen2.5-coder:7b",
        "memory": "medium",
        "use_case": "Code search, implementation notes, and developer research.",
    },
    "reasoning": {
        "model": "deepseek-r1:7b",
        "memory": "medium",
        "use_case": "Slow but useful for step-by-step critique and hard trade-off analysis.",
    },
    "compact-reasoning": {
        "model": "phi4-mini",
        "memory": "low-medium",
        "use_case": "Efficient reasoning when you want lower memory pressure.",
    },
    "multimodal-small": {
        "model": "gemma3:4b",
        "memory": "low-medium",
        "use_case": "Lightweight general model; useful if your local server exposes Gemma 3.",
    },
    "quality-large": {
        "model": "gemma3:12b",
        "memory": "high",
        "use_case": "Higher-quality local synthesis; use quantized weights and close other apps.",
    },
    "mistral": {
        "model": "mistral:7b",
        "memory": "medium",
        "use_case": "Reliable instruction-following fallback for OpenAI-compatible local servers.",
    },
}


@dataclass(frozen=True)
class ResearchAgentSettings:
    """Settings shared by the agent workflow."""

    model: str = "gpt-4.1"
    notes_dir: Path = Path("research_notes")
    base_url: str | None = None
    api_key: str | None = None
    use_chat_completions: bool = False
    disable_tracing: bool = False

    @property
    def uses_local_model(self) -> bool:
        """Return whether the settings point at an OpenAI-compatible local endpoint."""

        return self.base_url is not None


def resolve_model_name(model_or_preset: str) -> str:
    """Resolve a model preset name to the provider-specific model identifier."""

    return LOCAL_MODEL_PRESETS.get(model_or_preset, {}).get("model", model_or_preset)


def format_local_model_presets() -> str:
    """Return a human-readable table of recommended local model presets."""

    rows = [
        "Preset | Model | Memory | Best for",
        "--- | --- | --- | ---",
    ]
    for preset, details in LOCAL_MODEL_PRESETS.items():
        rows.append(
            f"{preset} | {details['model']} | {details['memory']} | {details['use_case']}"
        )
    return "\n".join(rows)


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> ResearchAgentSettings:
    """Load settings from environment variables."""

    provider = os.getenv("RESEARCH_AGENTS_PROVIDER", "openai").strip().lower()
    local_mode = provider in {"local", "ollama", "lmstudio", "llama.cpp", "llamacpp"}
    model_from_env = os.getenv("RESEARCH_AGENTS_MODEL", ResearchAgentSettings.model)
    model = resolve_model_name(model_from_env)

    if local_mode:
        return ResearchAgentSettings(
            model=model,
            notes_dir=Path(os.getenv("RESEARCH_AGENTS_NOTES_DIR", "research_notes")),
            base_url=os.getenv("RESEARCH_AGENTS_BASE_URL", DEFAULT_LOCAL_BASE_URL),
            api_key=os.getenv("RESEARCH_AGENTS_API_KEY", DEFAULT_LOCAL_API_KEY),
            use_chat_completions=True,
            disable_tracing=_truthy(os.getenv("RESEARCH_AGENTS_DISABLE_TRACING", "1")),
        )

    return ResearchAgentSettings(
        model=model,
        notes_dir=Path(os.getenv("RESEARCH_AGENTS_NOTES_DIR", "research_notes")),
        base_url=os.getenv("RESEARCH_AGENTS_BASE_URL") or None,
        api_key=os.getenv("RESEARCH_AGENTS_API_KEY") or None,
        use_chat_completions=_truthy(os.getenv("RESEARCH_AGENTS_USE_CHAT_COMPLETIONS")),
        disable_tracing=_truthy(os.getenv("RESEARCH_AGENTS_DISABLE_TRACING")),
    )
