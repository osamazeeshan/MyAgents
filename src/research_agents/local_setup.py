"""First-run local model setup helpers."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_LOCAL_API_KEY,
    DEFAULT_LOCAL_BASE_URL,
    LOCAL_MODEL_PRESETS,
)

LOCAL_SETUP_SENTINEL = Path.home() / ".research_agents" / "local_model_setup"
LOCAL_SETUP_ENV = "RESEARCH_AGENTS_AUTO_LOCAL_SETUP"
OLLAMA_INSTALL_ENV = "RESEARCH_AGENTS_AUTO_INSTALL_OLLAMA"
DEFAULT_SETUP_TIMEOUT_SECONDS = 60 * 30
DEFAULT_OLLAMA_INSTALL_TIMEOUT_SECONDS = 60 * 10
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"


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
    """Outcome from the local setup step."""

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
    """Best-effort physical memory detection using local OS commands."""

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            physical_pages = os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            page_size = physical_pages = None
        if isinstance(page_size, int) and isinstance(physical_pages, int):
            return (page_size * physical_pages) / (1024**3)

    os_name = platform.system()
    if os_name == "Darwin":
        return _memory_from_command(["sysctl", "-n", "hw.memsize"])
    if os_name == "Linux":
        meminfo_memory = _memory_from_linux_meminfo()
        if meminfo_memory is not None:
            return meminfo_memory
    if os_name == "Windows":
        return _memory_from_command(
            ["wmic", "computersystem", "get", "totalphysicalmemory", "/value"],
            parse_value=lambda output: output.partition("=")[2].strip(),
        )

    return None


def _memory_from_linux_meminfo() -> float | None:
    """Read Linux physical memory from /proc/meminfo when available."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                return int(parts[1]) / (1024**2)
    except (OSError, IndexError, ValueError):
        return None
    return None


def _memory_from_command(
    command: list[str], *, parse_value: Callable[[str], str] | None = None
) -> float | None:
    """Run an OS hardware command and parse byte output as GiB."""

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    raw_value = result.stdout.strip()
    if parse_value is not None:
        raw_value = parse_value(raw_value)
    try:
        return int(raw_value) / (1024**3)
    except ValueError:
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
    """Detect specs, install/download the recommended model, and activate it.

    The setup is conservative: explicit provider/model/API settings are respected,
    and users can disable this step with ``RESEARCH_AGENTS_AUTO_LOCAL_SETUP=0``.
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

    if not model_was_explicit:
        _activate_local_environment(recommendation)

    if verbose:
        print(
            "Detected "
            f"{spec.os_name} {spec.machine}, {spec.cpu_count} CPUs, "
            f"{spec.memory_label}. Selected {recommendation.model} "
            f"({recommendation.preset}): {recommendation.reason}",
            flush=True,
        )

    if not _ensure_ollama_available(verbose=verbose):
        return LocalSetupResult(
            spec,
            recommendation,
            attempted=True,
            activated=True,
            downloaded=False,
            message=(
                "Could not install or find Ollama automatically. Install Ollama, then run "
                f"`ollama pull {recommendation.model}`; the app has activated the "
                f"{recommendation.preset} preset for local mode."
            ),
        )

    _ensure_ollama_server(verbose=verbose)
    try:
        downloaded = _download_recommended_model(recommendation, verbose=verbose)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return LocalSetupResult(
            spec,
            recommendation,
            attempted=True,
            activated=True,
            downloaded=False,
            message=(
                f"Failed to download {recommendation.model}: {exc}. "
                f"Run `ollama pull {recommendation.model}` and restart the app."
            ),
        )
    _write_setup_sentinel(spec, recommendation)
    return LocalSetupResult(
        spec,
        recommendation,
        attempted=True,
        activated=True,
        downloaded=downloaded,
        message=f"Local model ready and active: {recommendation.model} ({recommendation.preset}).",
    )


def _activate_local_environment(recommendation: LocalModelRecommendation) -> None:
    """Point this process at the recommended local Ollama model."""

    os.environ.setdefault("RESEARCH_AGENTS_PROVIDER", "ollama")
    os.environ.setdefault("RESEARCH_AGENTS_MODEL", recommendation.preset)
    os.environ.setdefault("RESEARCH_AGENTS_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    os.environ.setdefault("RESEARCH_AGENTS_API_KEY", DEFAULT_LOCAL_API_KEY)
    os.environ.setdefault("RESEARCH_AGENTS_DISABLE_TRACING", "1")


def _ensure_ollama_available(*, verbose: bool) -> bool:
    """Return true when the ollama command exists, installing it if possible."""

    if shutil.which("ollama") is not None:
        return True
    if _falsey(os.getenv(OLLAMA_INSTALL_ENV)):
        return False

    installer = _ollama_install_command()
    if installer is None:
        return False

    if verbose:
        print(
            "Ollama was not found. Installing Ollama before downloading the model...",
            flush=True,
        )
    try:
        subprocess.run(
            installer,
            check=True,
            timeout=int(
                os.getenv(
                    "RESEARCH_AGENTS_OLLAMA_INSTALL_TIMEOUT",
                    DEFAULT_OLLAMA_INSTALL_TIMEOUT_SECONDS,
                )
            ),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return shutil.which("ollama") is not None


def _ollama_install_command() -> list[str] | None:
    """Return a non-interactive Ollama install command for the current system."""

    os_name = platform.system()
    if os_name == "Darwin" and shutil.which("brew") is not None:
        return ["brew", "install", "ollama"]
    if os_name == "Linux" and shutil.which("curl") is not None:
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    if os_name == "Windows" and shutil.which("winget") is not None:
        return [
            "winget",
            "install",
            "Ollama.Ollama",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    return None


def _ensure_ollama_server(*, verbose: bool) -> None:
    """Start Ollama in the background when the local API is not answering yet."""

    if _ollama_api_is_running():
        return
    if verbose:
        print("Starting the local Ollama server before model download...", flush=True)
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return
    for _ in range(10):
        time.sleep(0.5)
        if _ollama_api_is_running():
            return


def _ollama_api_is_running() -> bool:
    """Return whether the default Ollama API is reachable."""

    try:
        with urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=2):
            return True
    except (OSError, urllib.error.URLError):
        return False


def _download_recommended_model(
    recommendation: LocalModelRecommendation, *, verbose: bool
) -> bool:
    """Pull the selected Ollama model if it is not already installed."""

    if _ollama_model_is_installed(recommendation.model):
        return False
    if verbose:
        print(
            f"Downloading {recommendation.model} with `ollama pull` before startup...",
            flush=True,
        )
    _pull_ollama_model(recommendation.model)
    return True


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
        line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()
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
    """Persist setup metadata for users to inspect."""

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
