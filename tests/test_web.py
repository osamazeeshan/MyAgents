import http.client
import re
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agents.web import (
    build_home_page,
    create_server,
    format_memory_augmented_prompt,
)


def test_home_page_contains_persistent_memory_layout() -> None:
    html = build_home_page()

    assert "ResearchAgent" in html
    assert "Recent" in html
    assert "Saved conversations" not in html
    assert "Agent launchpads" in html
    assert 'id="modelChoice"' in html
    assert "model: selectedModel()" in html
    assert "localStorage" in html
    assert "STORAGE_KEY" in html
    assert "researchagent.conversations.v1" in html
    assert "memory: memoryContext()" in html
    assert 'href="/favicon.ico"' in html
    assert 'id="agentRunning"' in html
    assert "agent-running-logo" in html
    assert "setAgentRunning(true)" in html
    assert "setAgentRunning(false)" in html
    assert ".conversation-list { display: grid; align-content: start; gap: 8px; overflow-y: auto;" in html
    assert ".suggestions { display: grid; gap: 12px; overflow-y: auto;" in html
    assert ".launchpad-panel { overflow: hidden; padding: 22px; gap: 14px; }" in html
    assert ".help-icon::after" in html
    assert "top: calc(100% + 9px)" in html
    assert ".workspace.card { background: transparent; border-color: transparent; box-shadow: none; backdrop-filter: none; padding: 0; }" in html
    assert ".modebar { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 7px; margin-bottom: 12px; }" in html
    assert "font-size: 12px; line-height: 1.15; white-space: nowrap;" in html
    assert "Choose a model before running agents. Local presets use your configured OpenAI-compatible provider." in html
    assert 'id="provider"' not in html
    assert 'id="notes"' not in html


def test_home_page_script_escapes_newline_sequences_for_browser_parsing() -> None:
    html = build_home_page()

    assert r"+ ']\n' + m.text" in html
    assert r".join('\n\n')" in html
    assert r"+ '\n\nAgent is thinking…'" in html
    assert r"codingGoal ? '\n' + codingGoal" in html
    assert "looksLikeCodingRequest(prompt)" in html


def test_home_page_inline_script_is_valid_javascript(tmp_path: Path) -> None:
    html = build_home_page()
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts
    script_path = tmp_path / "home-page.js"
    script_path.write_text("\n".join(scripts), encoding="utf-8")

    subprocess.run(["node", "--check", str(script_path)], check=True)


def test_memory_augmented_prompt_includes_memory_and_latest_request() -> None:
    prompt = format_memory_augmented_prompt(
        "What should I read next?", "User: Topic A\nAgent: Read Paper B"
    )

    assert "Saved conversation memory:" in prompt
    assert "User: Topic A" in prompt
    assert "Latest user request:" in prompt
    assert prompt.endswith("What should I read next?")


def test_memory_augmented_prompt_leaves_empty_memory_unchanged() -> None:
    assert format_memory_augmented_prompt("Fresh question", "") == "Fresh question"


def test_favicon_route_returns_svg_icon() -> None:
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection(
            server.server_address[0], server.server_address[1], timeout=2
        )
        conn.request("GET", "/favicon.ico")
        response = conn.getresponse()
        body = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 200
    assert response.getheader("Content-Type") == "image/svg+xml; charset=utf-8"
    assert "<svg" in body


def test_home_page_includes_paper_coding_workspace() -> None:
    html = build_home_page()

    assert "Paper Coding Agent" in html
    assert "Paper coding agent" in html
    assert 'data-mode="coding"' in html
    assert 'data-mode="research"' in html
    assert 'id="codingWindow"' in html
    assert 'id="paperIdentifier"' in html
    assert 'id="codingGoal"' in html
    assert 'id="ideaStream"' in html
    assert 'id="codingConsole"' in html
    assert "qwen2.5-coder:7b" in html
    assert "coding mode will not force an unavailable model" in html
    assert "preferCodingModel" not in html
    assert "/api/coding/implement" in html
    assert "state.mode === 'coding'" in html
    assert "codingWindow').classList.toggle('visible'" in html
