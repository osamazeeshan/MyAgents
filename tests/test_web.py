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
    assert ".actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; margin-top: 12px; }" in html
    assert ".primary, .secondary, .memory-pill { min-height: 42px; border-radius: 16px; padding: 10px 12px; font-size: 13px; line-height: 1.15; font-weight: 800; display: inline-flex; align-items: center; justify-content: center; text-align: center; white-space: nowrap; }" in html
    assert ".primary, .secondary { width: 112px; flex: 0 0 112px; cursor: pointer; }" in html
    assert ".memory-pill { color: var(--accent); border: 1px solid rgba(124,247,212,.4); margin-left: auto; min-width: 170px; }" in html
    assert ".agent-running.visible { display: inline-flex; flex-basis: 100%; justify-content: center; }" in html
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


def test_home_page_includes_generated_code_interface_controls() -> None:
    html = build_home_page()

    assert 'id="openCodeInterface"' in html
    assert 'id="codeInterface"' in html
    assert 'id="fileTree"' in html
    assert 'id="codeEditor"' in html
    assert 'id="runDummy"' in html
    assert 'id="publishGithub"' in html
    assert 'id="linkGithub"' in html
    assert 'id="createPullRequest"' in html
    assert 'id="viewPullRequest"' in html
    assert ".code-interface-actions { display: grid; grid-auto-flow: column; grid-auto-columns: max-content; gap: 8px; justify-content: end; overflow-x: auto; white-space: nowrap; }" in html
    assert 'publishWorkspaceToGithub' in html
    assert 'linkWorkspaceToGithub' in html
    assert 'createWorkspacePullRequest' in html
    assert '/api/coding/files' in html
    assert '/api/coding/file' in html
    assert '/api/coding/save' in html
    assert '/api/coding/run' in html
    assert '/api/coding/publish' in html
    assert '/api/coding/link' in html
    assert '/api/coding/create-pr' in html
    assert 'state.currentWorkspace = data.workspace' in html


def test_coding_workspace_file_api_reads_saves_and_runs_dummy_data(tmp_path, monkeypatch) -> None:
    import asyncio
    import research_agents.workflow as workflow
    from research_agents.web import handle_api_request

    monkeypatch.chdir(tmp_path)
    workspace_text = workflow.prepare_paper_coding_environment("Interface Smoke")
    workspace = re.search(r"Created local workspace: (.+)", workspace_text).group(1)

    tree_result = asyncio.run(
        handle_api_request("/api/coding/files", {"workspace": workspace})
    )
    assert tree_result["workspace"].endswith("Interface-Smoke-coding-lab")
    assert any(node["name"] == "src" for node in tree_result["tree"])

    file_result = asyncio.run(
        handle_api_request(
            "/api/coding/file",
            {"workspace": workspace, "path": "src/reproduction_baseline/baseline.py"},
        )
    )
    assert "def majority_label" in file_result["content"]

    edited = file_result["content"] + "\n# browser edit\n"
    save_result = asyncio.run(
        handle_api_request(
            "/api/coding/save",
            {
                "workspace": workspace,
                "path": "src/reproduction_baseline/baseline.py",
                "content": edited,
            },
        )
    )
    assert save_result["saved"] is True
    assert "# browser edit" in (
        tmp_path
        / "reproduction_repos"
        / "Interface-Smoke-coding-lab"
        / "src"
        / "reproduction_baseline"
        / "baseline.py"
    ).read_text(encoding="utf-8")

    run_result = asyncio.run(
        handle_api_request("/api/coding/run", {"workspace": workspace})
    )
    assert run_result["returncode"] == 0
    assert "DatasetSummary" in run_result["output"]


def test_coding_workspace_link_api_enables_manual_github_connection(tmp_path, monkeypatch) -> None:
    import asyncio
    import subprocess
    import research_agents.workflow as workflow
    from research_agents.web import handle_api_request

    monkeypatch.chdir(tmp_path)
    workspace_text = workflow.prepare_paper_coding_environment("Manual Link Smoke")
    workspace = re.search(r"Created local workspace: (.+)", workspace_text).group(1)
    bare_repo = tmp_path / "manual-link.git"
    subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True)

    link_result = asyncio.run(
        handle_api_request(
            "/api/coding/link",
            {"workspace": workspace, "repo_url": str(bare_repo) + ".git"},
        )
    )

    assert link_result["linked"] is True
    assert link_result["create_pr_enabled"] is True
    assert link_result["html_url"] == str(bare_repo)
    assert link_result["remote_name"] == "origin"
    remote_url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=Path(workspace),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_url == str(bare_repo)


def test_coding_workspace_publish_failure_points_to_link_button(tmp_path, monkeypatch) -> None:
    import asyncio
    import research_agents.workflow as workflow
    from research_agents.web import handle_api_request

    monkeypatch.chdir(tmp_path)
    workspace_text = workflow.prepare_paper_coding_environment("Publish Failure")
    workspace = re.search(r"Created local workspace: (.+)", workspace_text).group(1)

    def fake_create_github_repository(*args: object, **kwargs: object) -> str:
        raise RuntimeError("GitHub repository creation plugin failed.")

    monkeypatch.setattr(
        workflow, "create_github_repository", fake_create_github_repository
    )

    publish_result = asyncio.run(
        handle_api_request(
            "/api/coding/publish",
            {"workspace": workspace, "repo": "publish-failure", "private": True},
        )
    )

    assert publish_result["published"] is False
    assert publish_result["create_pr_enabled"] is False
    assert "Link GitHub" in publish_result["message"]
    assert "Link GitHub" in publish_result["manual_link_hint"]


def test_coding_workspace_publish_api_creates_remote_and_pushes(tmp_path, monkeypatch) -> None:
    import asyncio
    import subprocess
    import research_agents.workflow as workflow
    import research_agents.web as web
    from research_agents.web import handle_api_request

    monkeypatch.chdir(tmp_path)
    workspace_text = workflow.prepare_paper_coding_environment("Publish Smoke")
    workspace = re.search(r"Created local workspace: (.+)", workspace_text).group(1)
    bare_repo = tmp_path / "publish-smoke.git"
    subprocess.run(["git", "init", "--bare", str(bare_repo)], check=True)

    def fake_create_github_repository(
        repo_name: str,
        *,
        description: str = "",
        private: bool = False,
        owner: str = "",
        token: str | None = None,
    ) -> str:
        assert repo_name == "publish-smoke"
        assert "Publish-Smoke-coding-lab" in description
        assert private is True
        assert owner == "example-org"
        assert token is None
        return str(bare_repo)

    monkeypatch.setattr(
        workflow, "create_github_repository", fake_create_github_repository
    )
    monkeypatch.setattr(web.time, "time", lambda: 1234567890)

    publish_result = asyncio.run(
        handle_api_request(
            "/api/coding/publish",
            {
                "workspace": workspace,
                "repo": "publish-smoke",
                "private": True,
                "owner": "example-org",
            },
        )
    )

    assert publish_result["published"] is True
    assert publish_result["linked"] is True
    assert publish_result["create_pr_enabled"] is True
    assert publish_result["html_url"] == str(bare_repo)

    pr_result = asyncio.run(
        handle_api_request(
            "/api/coding/create-pr",
            {"workspace": workspace, "repo_url": publish_result["html_url"]},
        )
    )

    assert pr_result["pushed"] is True
    assert pr_result["pull_request_created"] is False
    assert pr_result["html_url"] == str(bare_repo)
    assert pr_result["branch"] == "researchagent-coding-workspace-1234567890"
    assert "push -u origin researchagent-coding-workspace-1234567890" in pr_result["push_command"]
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        cwd=bare_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert refs == ["main", "researchagent-coding-workspace-1234567890"]
