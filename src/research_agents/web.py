"""Web interface for the research-agent workflow."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .config import format_local_model_presets, load_settings

APP_NAME = "Agentarium"
APP_TAGLINE = "A glasshouse for research agents, paper scouts, and critical reviewers."
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 1_000_000

AGENT_SUGGESTIONS = [
    {
        "name": "Research Orchestrator",
        "role": "Routes each request to the best specialist agent.",
        "try": "Build a source map for evaluating retrieval-augmented generation.",
    },
    {
        "name": "Research Planner",
        "role": "Turns broad goals into staged research plans and validation checks.",
        "try": "Create a 4-week study plan for benchmarking local LLM agents.",
    },
    {
        "name": "Literature Scout",
        "role": "Finds search terms, venue targets, paper clusters, and source strategies.",
        "try": "Suggest scholar queries for multimodal agent evaluation.",
    },
    {
        "name": "Critical Reviewer",
        "role": "Challenges assumptions, weak evidence, and missing baselines.",
        "try": "Critique this claim: agent benchmarks prove general autonomy.",
    },
    {
        "name": "Conference Review Crew",
        "role": "Discovers recent top-conference topics, verifies papers, and runs two reviewers.",
        "try": "LLM agents, multimodal models, and computer vision",
    },
]


def build_home_page() -> str:
    """Return the single-page web app HTML."""

    suggestions = "\n".join(
        f"""
        <button class=\"suggestion\" data-prompt=\"{html.escape(agent['try'], quote=True)}\">
          <strong>{html.escape(agent['name'])}</strong>
          <span>{html.escape(agent['role'])}</span>
          <em>Try: {html.escape(agent['try'])}</em>
        </button>
        """.strip()
        for agent in AGENT_SUGGESTIONS
    )
    local_models = html.escape(format_local_model_presets())
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{APP_NAME} · Research Agents</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #070817;
      --panel: rgba(20, 24, 50, 0.72);
      --panel-strong: rgba(31, 37, 75, 0.92);
      --text: #f8fbff;
      --muted: #aeb9d8;
      --accent: #7cf7d4;
      --accent-2: #b58cff;
      --hot: #ff80c5;
      --border: rgba(255,255,255,0.16);
      --shadow: 0 24px 90px rgba(0, 0, 0, 0.46);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(124,247,212,.18), transparent 34rem),
        radial-gradient(circle at 82% 12%, rgba(181,140,255,.26), transparent 32rem),
        radial-gradient(circle at 48% 92%, rgba(255,128,197,.14), transparent 30rem),
        var(--bg);
      color: var(--text);
    }}
    .shell {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 38px 0 46px; }}
    header {{ display: grid; grid-template-columns: 1.15fr .85fr; gap: 24px; align-items: stretch; }}
    .hero, .card {{
      border: 1px solid var(--border);
      background: var(--panel);
      backdrop-filter: blur(18px);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 34px; position: relative; overflow: hidden; }}
    .hero::after {{
      content: ""; position: absolute; inset: auto -10% -42% 36%; height: 260px;
      background: linear-gradient(90deg, transparent, rgba(124,247,212,.28), transparent);
      transform: rotate(-8deg); filter: blur(20px);
    }}
    .eyebrow {{ color: var(--accent); letter-spacing: .18em; text-transform: uppercase; font-size: 12px; font-weight: 800; }}
    h1 {{ font-size: clamp(42px, 8vw, 86px); line-height: .9; margin: 16px 0; letter-spacing: -.07em; }}
    .tagline {{ color: var(--muted); font-size: 20px; line-height: 1.5; max-width: 760px; }}
    .name-note {{ margin-top: 20px; color: #d9e1ff; }}
    .name-note strong {{ color: var(--accent); }}
    .status {{ padding: 24px; display: grid; gap: 14px; }}
    .pill {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
    .pill b {{ color: var(--text); }}
    main {{ display: grid; grid-template-columns: 360px 1fr; gap: 24px; margin-top: 24px; }}
    .card {{ padding: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .suggestions {{ display: grid; gap: 12px; }}
    .suggestion {{
      text-align: left; border: 1px solid var(--border); border-radius: 18px; padding: 14px;
      color: var(--text); background: rgba(255,255,255,.055); cursor: pointer; transition: .2s ease;
    }}
    .suggestion:hover {{ transform: translateY(-2px); border-color: rgba(124,247,212,.7); }}
    .suggestion strong, .suggestion span, .suggestion em {{ display: block; }}
    .suggestion span {{ color: var(--muted); margin: 6px 0; font-size: 13px; line-height: 1.35; }}
    .suggestion em {{ color: var(--accent); font-style: normal; font-size: 12px; }}
    .workspace {{ min-height: 680px; display: flex; flex-direction: column; }}
    .modebar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
    .mode {{ border: 1px solid var(--border); border-radius: 999px; padding: 10px 14px; color: var(--muted); background: transparent; cursor: pointer; }}
    .mode.active {{ color: #06120f; background: var(--accent); border-color: var(--accent); font-weight: 800; }}
    textarea, input {{
      width: 100%; border: 1px solid var(--border); background: rgba(5,8,24,.86); color: var(--text);
      border-radius: 18px; padding: 14px 16px; font: inherit; outline: none;
    }}
    textarea {{ min-height: 130px; resize: vertical; }}
    textarea:focus, input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 4px rgba(124,247,212,.12); }}
    .actions {{ display: flex; gap: 12px; align-items: center; margin: 12px 0 18px; }}
    .primary, .secondary {{ border: 0; border-radius: 16px; padding: 13px 18px; font-weight: 800; cursor: pointer; }}
    .primary {{ color: #06120f; background: linear-gradient(135deg, var(--accent), #e6ff8a); }}
    .secondary {{ color: var(--text); background: rgba(255,255,255,.09); border: 1px solid var(--border); }}
    .hint {{ color: var(--muted); font-size: 13px; }}
    .output {{
      flex: 1; overflow: auto; white-space: pre-wrap; line-height: 1.55; border: 1px solid var(--border);
      background: rgba(3, 6, 20, .72); border-radius: 22px; padding: 18px;
    }}
    .output .empty {{ color: var(--muted); }}
    .conference-fields {{ display: none; gap: 10px; margin-bottom: 12px; }}
    .conference-fields.visible {{ display: grid; }}
    details {{ margin-top: 16px; color: var(--muted); }}
    pre {{ overflow: auto; background: rgba(0,0,0,.3); padding: 12px; border-radius: 12px; }}
    @media (max-width: 900px) {{ header, main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <header>
      <section class=\"hero\">
        <div class=\"eyebrow\">Web console suggestion</div>
        <h1>{APP_NAME}</h1>
        <p class=\"tagline\">{APP_TAGLINE}</p>
        <p class=\"name-note\"><strong>Name idea:</strong> {APP_NAME} suggests a living observatory where multiple agents grow ideas, cross-pollinate evidence, and turn research questions into grounded next steps.</p>
      </section>
      <aside class=\"card status\" id=\"status\">
        <h2>Runtime</h2>
        <div class=\"pill\"><span>Provider</span><b id=\"provider\">loading…</b></div>
        <div class=\"pill\"><span>Model</span><b id=\"model\">loading…</b></div>
        <div class=\"pill\"><span>Notes</span><b id=\"notes\">loading…</b></div>
        <div class=\"hint\">Tip: use <code>RESEARCH_AGENTS_PROVIDER=ollama</code> and <code>RESEARCH_AGENTS_MODEL=balanced</code> for local model testing.</div>
      </aside>
    </header>
    <main>
      <aside class=\"card\">
        <h2>Agent launchpads</h2>
        <div class=\"suggestions\">{suggestions}</div>
        <details><summary>Local model presets</summary><pre>{local_models}</pre></details>
      </aside>
      <section class=\"card workspace\">
        <div class=\"modebar\">
          <button class=\"mode active\" data-mode=\"research\">Ask the agent crew</button>
          <button class=\"mode\" data-mode=\"discover\">Discover conference topics</button>
          <button class=\"mode\" data-mode=\"review\">Review selected topic</button>
          <button class=\"mode\" data-mode=\"followup\">Conference follow-up</button>
        </div>
        <div class=\"conference-fields\" id=\"conferenceFields\">
          <input id=\"topic\" placeholder=\"Selected topic (required for review/follow-up)\" />
          <textarea id=\"context\" placeholder=\"Discovery, paper, or review context. Agentarium stores the latest output here automatically.\"></textarea>
        </div>
        <textarea id=\"prompt\" placeholder=\"Ask a research question, describe a topic, or paste a follow-up…\"></textarea>
        <div class=\"actions\">
          <button class=\"primary\" id=\"run\">Run agents</button>
          <button class=\"secondary\" id=\"clear\">Clear</button>
          <span class=\"hint\" id=\"busy\"></span>
        </div>
        <div class=\"output\" id=\"output\"><span class=\"empty\">Your agent transcript will appear here.</span></div>
      </section>
    </main>
  </div>
  <script>
    const state = {{ mode: 'research', lastDiscovery: '', lastPaperContext: '', lastReview: '' }};
    const $ = (id) => document.getElementById(id);
    const output = $('output');
    function setOutput(text) {{ output.textContent = text || 'No output returned.'; }}
    function appendOutput(label, text) {{ output.textContent = `${{output.textContent}}\n\n# ${{label}}\n\n${{text}}`.trim(); }}
    async function postJSON(path, body) {{
      const res = await fetch(path, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(body) }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }}
    document.querySelectorAll('.mode').forEach(btn => btn.addEventListener('click', () => {{
      document.querySelectorAll('.mode').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.mode = btn.dataset.mode;
      $('conferenceFields').classList.toggle('visible', state.mode !== 'research' && state.mode !== 'discover');
      $('prompt').placeholder = state.mode === 'discover' ? 'Describe the domain to scan across recent top conferences…' : 'Ask a research question, describe a topic, or paste a follow-up…';
    }}));
    document.querySelectorAll('.suggestion').forEach(btn => btn.addEventListener('click', () => {{ $('prompt').value = btn.dataset.prompt; }}));
    $('clear').addEventListener('click', () => {{ $('prompt').value = ''; $('topic').value = ''; $('context').value = ''; setOutput('Your agent transcript will appear here.'); }});
    $('run').addEventListener('click', async () => {{
      const prompt = $('prompt').value.trim();
      const topic = $('topic').value.trim();
      const context = $('context').value.trim() || state.lastReview || state.lastPaperContext || state.lastDiscovery;
      $('busy').textContent = 'Agents are thinking…'; $('run').disabled = true;
      try {{
        if (state.mode === 'research') {{
          const data = await postJSON('/api/research', {{ prompt }}); setOutput(data.output);
        }} else if (state.mode === 'discover') {{
          const data = await postJSON('/api/conference/discover', {{ prompt }}); state.lastDiscovery = data.output; $('context').value = data.output; setOutput(data.output);
        }} else if (state.mode === 'review') {{
          const data = await postJSON('/api/conference/review', {{ topic: topic || prompt, discovery_context: context }});
          state.lastPaperContext = data.paper_context; state.lastReview = data.review; $('context').value = `${{data.paper_context}}\n\n${{data.review}}`; setOutput(`${{data.paper_context}}\n\n${{data.review}}`);
        }} else {{
          const data = await postJSON('/api/conference/follow-up', {{ question: prompt, selected_topic: topic, paper_context: state.lastPaperContext || context, review_context: state.lastReview || context }});
          appendOutput(`Follow-up: ${{prompt}}`, data.output);
        }}
      }} catch (err) {{ setOutput(`Error: ${{err.message}}`); }}
      finally {{ $('busy').textContent = ''; $('run').disabled = false; }}
    }});
    fetch('/api/health').then(r => r.json()).then(data => {{ $('provider').textContent = data.provider; $('model').textContent = data.model; $('notes').textContent = data.notes_dir; }});
  </script>
</body>
</html>"""


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body: str) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' is required")
    return value.strip()


async def handle_api_request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Route a JSON API request to the appropriate agent workflow."""

    if path == "/api/research":
        from .workflow import run_research_workflow

        return {"output": await run_research_workflow(_require_text(payload, "prompt"))}

    if path == "/api/conference/discover":
        from .workflow import discover_recent_conference_topics

        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        return {"output": await discover_recent_conference_topics(prompt)}

    if path == "/api/conference/review":
        from .workflow import review_selected_topic, search_papers_for_topic

        topic = _require_text(payload, "topic")
        discovery_context = payload.get("discovery_context", "")
        if not isinstance(discovery_context, str):
            raise ValueError("'discovery_context' must be a string")
        paper_context = await search_papers_for_topic(topic, discovery_context)
        review = await review_selected_topic(topic, paper_context)
        return {"paper_context": paper_context, "review": review}

    if path == "/api/conference/follow-up":
        from .workflow import answer_conference_review_follow_up

        output = await answer_conference_review_follow_up(
            _require_text(payload, "question"),
            _require_text(payload, "selected_topic"),
            _require_text(payload, "paper_context"),
            _require_text(payload, "review_context"),
        )
        return {"output": output}

    raise KeyError(path)


class AgentariumRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Agentarium app."""

    server_version = "AgentariumHTTP/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logging.getLogger(__name__).info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            _html_response(self, build_home_page())
            return
        if path == "/api/health":
            settings = load_settings()
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "app": APP_NAME,
                    "tagline": APP_TAGLINE,
                    "model": settings.model,
                    "provider": "local" if settings.uses_local_model else "openai",
                    "notes_dir": str(settings.notes_dir),
                    "agents": AGENT_SUGGESTIONS,
                },
            )
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            _json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request body too large"})
            return

        try:
            raw_body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            result = asyncio.run(handle_api_request(path, payload))
        except KeyError:
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {path}"})
        except (json.JSONDecodeError, ValueError) as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - preserves useful errors for the browser
            logging.getLogger(__name__).exception("Agent workflow failed")
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            _json_response(self, HTTPStatus.OK, result)


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create, but do not start, the Agentarium HTTP server."""

    return ThreadingHTTPServer((host, port), AgentariumRequestHandler)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, open_browser: bool = False) -> None:
    """Run the Agentarium web server until interrupted."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = create_server(host, port)
    url = f"http://{host}:{server.server_port}"
    print(f"{APP_NAME} is ready at {url}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Agentarium.")
    finally:
        server.server_close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse web-server command-line arguments."""

    parser = argparse.ArgumentParser(description=f"Run {APP_NAME}, the research-agent web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT}).")
    parser.add_argument("--open", action="store_true", help="Open the web UI in your default browser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console entry point for the Agentarium web UI."""

    args = parse_args(argv)
    run_server(args.host, args.port, open_browser=args.open)
