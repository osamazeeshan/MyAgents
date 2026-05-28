import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agents.local_setup import (
    LocalSystemSpec,
    ensure_first_run_local_model,
    recommend_local_model,
)


def test_recommend_local_model_selects_safe_small_model_for_low_memory() -> None:
    spec = LocalSystemSpec("Linux", "x86_64", 4, 7.5)

    recommendation = recommend_local_model(spec)

    assert recommendation.preset == "fast-small"
    assert recommendation.model == "llama3.2:3b"


def test_recommend_local_model_selects_balanced_for_16gb_system() -> None:
    spec = LocalSystemSpec("Darwin", "arm64", 8, 16.0)

    recommendation = recommend_local_model(spec)

    assert recommendation.preset == "balanced"
    assert recommendation.model == "qwen3:8b"


def test_first_run_setup_activates_recommended_local_model_when_ollama_missing(
    tmp_path, monkeypatch
) -> None:
    import research_agents.local_setup as local_setup

    monkeypatch.setattr(local_setup, "LOCAL_SETUP_SENTINEL", tmp_path / "setup")
    monkeypatch.setattr(
        local_setup,
        "detect_system_spec",
        lambda: LocalSystemSpec("Linux", "x86_64", 8, 16.0),
    )
    monkeypatch.setattr(local_setup.shutil, "which", lambda name: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_AGENTS_PROVIDER", raising=False)
    monkeypatch.delenv("RESEARCH_AGENTS_MODEL", raising=False)
    monkeypatch.delenv("RESEARCH_AGENTS_BASE_URL", raising=False)
    monkeypatch.delenv("RESEARCH_AGENTS_API_KEY", raising=False)

    result = ensure_first_run_local_model(verbose=False)

    assert result.activated is True
    assert result.recommendation.model == "qwen3:8b"
    assert "Ollama is not installed" in result.message
    assert "RESEARCH_AGENTS_PROVIDER" in local_setup.os.environ
    assert local_setup.os.environ["RESEARCH_AGENTS_MODEL"] == "balanced"


def test_first_run_setup_preserves_explicit_hosted_configuration(
    tmp_path, monkeypatch
) -> None:
    import research_agents.local_setup as local_setup

    monkeypatch.setattr(local_setup, "LOCAL_SETUP_SENTINEL", tmp_path / "setup")
    monkeypatch.setattr(
        local_setup,
        "detect_system_spec",
        lambda: LocalSystemSpec("Linux", "x86_64", 8, 16.0),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("RESEARCH_AGENTS_PROVIDER", raising=False)
    monkeypatch.delenv("RESEARCH_AGENTS_MODEL", raising=False)

    result = ensure_first_run_local_model(verbose=False)

    assert result.attempted is False
    assert result.activated is False
    assert (
        result.message
        == "Existing hosted or explicit model configuration was preserved."
    )
    assert "RESEARCH_AGENTS_PROVIDER" not in local_setup.os.environ
