import http.client
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agents.web import build_home_page, create_server, format_memory_augmented_prompt


def test_home_page_contains_persistent_memory_layout() -> None:
    html = build_home_page()

    assert "YourResearchGuide" in html
    assert "Saved conversations" in html
    assert "Agent launchpads" in html
    assert "localStorage" in html
    assert "STORAGE_KEY" in html
    assert "yourresearchguide.conversations.v1" in html
    assert "memory: memoryContext()" in html
    assert 'href="/favicon.ico"' in html
    assert 'id=\"agentRunning\"' in html
    assert 'agent-running-logo' in html
    assert 'setAgentRunning(true)' in html
    assert 'setAgentRunning(false)' in html
    assert ".conversation-list { display: grid; gap: 10px; overflow-y: auto;" in html
    assert ".suggestions { display: grid; gap: 12px; overflow-y: auto;" in html


def test_home_page_script_escapes_newline_sequences_for_browser_parsing() -> None:
    html = build_home_page()

    assert r"+ ']\n' + m.text" in html
    assert r".join('\n\n')" in html
    assert r"+ '\n\nAgent is thinking…'" in html


def test_memory_augmented_prompt_includes_memory_and_latest_request() -> None:
    prompt = format_memory_augmented_prompt("What should I read next?", "User: Topic A\nAgent: Read Paper B")

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
        conn = http.client.HTTPConnection(server.server_address[0], server.server_address[1], timeout=2)
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
