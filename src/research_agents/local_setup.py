"""First-run local model setup helpers."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_LOCAL_API_KEY,
    DEFAULT_LOCAL_BASE_URL,
    LOCAL_MODEL_PRESETS,
)

LOCAL_SETUP_SENTINEL = Path.home() / ".research_agents" / "local_model_setup"
LOCAL_SETUP_ENV = "RESEARCH_AGENTS_AUTO_LOCAL_SETUP"
DEFAULT_SETUP_TIMEOUT_SECONDS = 60 * 30


@dataclass(frozen=True)
class LocalSystemSpec:
    """Small hardware summary used for local-model recommendation."""

    os_name: str
    machine: str
    cpu_count: int
    memory_gb: float | None

    @property
    def memory_label(self) -> str:
        """Return a human-readable memory label."""

        if self.memory_gb is None:
            return "unknown RAM"
        return f"{self.memory_gb:.1f} GB RAM"


@dataclass(frozen=True)
class LocalModelRecommendation:
    """Recommended local model preset and backing provider model."""

    preset: str
    model: str
    reason: str


@dataclass(frozen=True)
class LocalSetupResult:
    """Outcome from the local first-run setup step."""

    spec: LocalSystemSpec
    recommendation: LocalModelRecommendation
    attempted: bool
    activated: bool
    downloaded: bool
    message: str


def _falsey(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"0", "false", "no", "off"}


def detect_system_spec() -> LocalSystemSpec:
    """Inspect local CPU architecture, CPU count, and physical memory."""

    return LocalSystemSpec(
        os_name=platform.system() or "Unknown OS",
        machine=platform.machine() or "unknown architecture",
        cpu_count=os.cpu_count() or 1,
        memory_gb=_detect_memory_gb(),
    )


def _detect_memory_gb() -> float | None:
    """Best-effort physical memory detection using the standard library."""

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            physical_pages = os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            page_size = physical_pages = None
        if isinstance(page_size, int) and isinstance(physical_pages, int):
            return (page_size * physical_pages) / (1024**3)

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        try:
            return int(result.stdout.strip()) / (1024**3)
        except ValueError:
            return None

    return None


def recommend_local_model(spec: LocalSystemSpec) -> LocalModelRecommendation:
    """Choose the largest built-in local preset likely to fit the machine."""

    memory_gb = spec.memory_gb
    if memory_gb is None:
        preset = "fast-small"
        reason = "RAM could not be detected, so the safest small preset was selected."
    elif memory_gb < 8:
        preset = "fast-small"
        reason = "Less than 8 GB RAM is best matched with a small 3B model."
    elif memory_gb < 14:
        preset = "compact-reasoning"
        reason = "8-14 GB RAM should use a compact model to avoid memory pressure."
    elif memory_gb < 32:
        preset = "balanced"
        reason = "14-32 GB RAM can usually run an 8B balanced local model."
    else:
        preset = "quality-large"
        reason = "32+ GB RAM can usually run a higher-quality 12B local model."

    model = LOCAL_MODEL_PRESETS[preset]["model"]
    return LocalModelRecommendation(preset=preset, model=model, reason=reason)


def ensure_first_run_local_model(
    *, force: bool = False, verbose: bool = True
) -> LocalSetupResult:
    """Detect local specs, download the recommended Ollama model, and activate it.

    The setup is intentionally conservative: explicit provider/model/API settings are
    respected, and users can disable this step with ``RESEARCH_AGENTS_AUTO_LOCAL_SETUP=0``.
    """

    spec = detect_system_spec()
    recommendation = recommend_local_model(spec)

    if _falsey(os.getenv(LOCAL_SETUP_ENV)):
        return LocalSetupResult(
            spec,
            recommendation,
            attempted=False,
            activated=False,
            downloaded=False,
            message="Automatic local model setup is disabled.",
        )

    provider_was_explicit = "RESEARCH_AGENTS_PROVIDER" in os.environ
    model_was_explicit = "RESEARCH_AGENTS_MODEL" in os.environ
    hosted_credentials_exist = bool(os.getenv("OPENAI_API_KEY"))
    should_activate_local = force or not (
        provider_was_explicit or model_was_explicit or hosted_credentials_exist
    )

    if not should_activate_local:
        return LocalSetupResult(
            spec,
            recommendation,
            attempted=False,
            activated=False,
            downloaded=False,
            message="Existing hosted or explicit model configuration was preserved.",
        )

    if not force and LOCAL_SETUP_SENTINEL.exists():
        if not model_was_explicit:
            _activate_local_environment(recommendation)
        return LocalSetupResult(
            spec,
            recommendation,
            attempted=False,
            activated=True,
            downloaded=False,
            message="Local model setup was already completed.",
        )

    if shutil.which("ollama") is None:
        if should_activate_local and not model_was_explicit:
            _activate_local_environment(recommendation)
        return LocalSetupResult(
            spec,
            recommendation,
            attempted=False,
            activated=should_activate_local,
            downloaded=False,
            message=(
                "Ollama is not installed. Install Ollama, then run "
                f"`ollama pull {recommendation.model}`."
            ),
        )

    downloaded = _ollama_model_is_installed(recommendation.model)
    if not downloaded:
        if verbose:
            print(
                "Detected "
                f"{spec.os_name} {spec.machine}, {spec.cpu_count} CPUs, "
                f"{spec.memory_label}. Pulling local model "
                f"{recommendation.model} ({recommendation.preset})...",
                flush=True,
            )
        _pull_ollama_model(recommendation.model)
        downloaded = True

    if should_activate_local and not model_was_explicit:
        _activate_local_environment(recommendation)

    _write_setup_sentinel(spec, recommendation)
    return LocalSetupResult(
        spec,
        recommendation,
        attempted=True,
        activated=should_activate_local,
        downloaded=downloaded,
        message=f"Local model ready: {recommendation.model} ({recommendation.preset}).",
    )


def _activate_local_environment(recommendation: LocalModelRecommendation) -> None:
    """Point this process at the recommended local Ollama model."""

    os.environ.setdefault("RESEARCH_AGENTS_PROVIDER", "ollama")
    os.environ.setdefault("RESEARCH_AGENTS_MODEL", recommendation.preset)
    os.environ.setdefault("RESEARCH_AGENTS_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    os.environ.setdefault("RESEARCH_AGENTS_API_KEY", DEFAULT_LOCAL_API_KEY)
    os.environ.setdefault("RESEARCH_AGENTS_DISABLE_TRACING", "1")


def _ollama_model_is_installed(model: str) -> bool:
    """Return whether Ollama already has the requested model tag."""

    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

    installed_names = {
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.split()
    }
    return model in installed_names


def _pull_ollama_model(model: str) -> None:
    """Download an Ollama model, streaming progress to the user's terminal."""

    subprocess.run(
        ["ollama", "pull", model],
        check=True,
        timeout=int(
            os.getenv("RESEARCH_AGENTS_OLLAMA_PULL_TIMEOUT", DEFAULT_SETUP_TIMEOUT_SECONDS)
        ),
    )


def _write_setup_sentinel(
    spec: LocalSystemSpec, recommendation: LocalModelRecommendation
) -> None:
    """Persist first-run setup metadata so later starts are fast."""

    LOCAL_SETUP_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SETUP_SENTINEL.write_text(
        "\n".join(
            [
                f"preset={recommendation.preset}",
                f"model={recommendation.model}",
                f"os={spec.os_name}",
                f"machine={spec.machine}",
                f"cpu_count={spec.cpu_count}",
                f"memory_gb={spec.memory_gb if spec.memory_gb is not None else 'unknown'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
