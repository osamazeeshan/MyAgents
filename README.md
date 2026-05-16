# My Research Agents

A small Python starter kit for building OpenAI-powered agents for research work. It uses the OpenAI Agents SDK so you can define specialist agents, give them tools, and run a coordinated research workflow from the command line.

## What is included

- **Research Planner**: decomposes a research question into concrete search, reading, and synthesis steps.
- **Literature Scout**: identifies useful source types, keywords, and inclusion/exclusion criteria.
- **Critical Reviewer**: challenges assumptions, flags weak evidence, and suggests follow-up checks.
- **Research Orchestrator**: routes your request to the right specialist agent.
- **Local notes tool**: saves structured notes to `research_notes/` so your work is not trapped in chat history.
- **Conference literature-review workflow**: searches recent top AI/ML/CV venues, asks you to select a topic, verifies recent papers through scholarly indexes, and sends the grounded paper set to two independent reviewer agents for critical literature review.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="sk-..."
research-agents "Map the current debates around retrieval-augmented generation evaluation."
```

### Interactive top-conference literature review

Run the conference-review workflow when you want an agent to search recent topics from venues such as NeurIPS, ICML, ICLR, AAAI, CVPR, ECCV, ICCV, and WACV. The topic-discovery step is limited to the current year and one year before it; on May 16, 2026, that means 2026 and 2025. The discovery output ends with one numbered topic-selection menu. When you enter a number, the CLI resolves it to the matching topic before searching the web and scholarly indexes. The focused paper search then returns no more than 20 recent and/or highly cited verified papers from roughly the last 5-6 years, and two LLM reviewer agents produce independent critical analyses, limitations, and possible future directions.

```bash
research-agents --conference-review "LLM agents, multimodal models, and computer vision"
```

For automatic topic discovery web search, use the default OpenAI Responses API configuration. Focused paper search is grounded by external scholarly index records, and the reviewer prompts forbid adding citations that are not present in the verified list. If scholarly index APIs are unreachable in your environment, the workflow will say so instead of fabricating papers.

### Run against a local model on a Mac M2 with 16GB RAM

The agents can connect to any OpenAI-compatible local server, including Ollama, LM Studio, or llama.cpp. For your Mac M2 with 16GB RAM, start with 3B-8B models for speed and stability; 12B models can work if you use a quantized build and close memory-heavy apps.

Recommended local models to try:

| Preset | Ollama model | Memory | Best for |
| --- | --- | --- | --- |
| `fast-small` | `llama3.2:3b` | Low | Fast planning, simple routing, and quick drafts. |
| `balanced` | `qwen3:8b` | Medium | Best default balance for local agent reasoning on a 16GB M2 Mac. |
| `balanced-alt` | `llama3.1:8b` | Medium | Strong general-purpose fallback with broad tool-calling compatibility. |
| `coding` | `qwen2.5-coder:7b` | Medium | Code search, implementation notes, and developer research. |
| `reasoning` | `deepseek-r1:7b` | Medium | Slower, but useful for step-by-step critique and hard trade-off analysis. |
| `compact-reasoning` | `phi4-mini` | Low-medium | Efficient reasoning when you want lower memory pressure. |
| `multimodal-small` | `gemma3:4b` | Low-medium | Lightweight general model if your local server exposes Gemma 3. |
| `quality-large` | `gemma3:12b` | High | Higher-quality local synthesis; use quantized weights and close other apps. |
| `mistral` | `mistral:7b` | Medium | Reliable instruction-following fallback for OpenAI-compatible local servers. |

Example with Ollama:

```bash
ollama pull qwen3:8b
ollama serve
export RESEARCH_AGENTS_PROVIDER=ollama
export RESEARCH_AGENTS_MODEL=balanced
research-agents "Create a research plan for evaluating local LLM agents."
```

Example with LM Studio or llama.cpp:

```bash
export RESEARCH_AGENTS_PROVIDER=local
export RESEARCH_AGENTS_BASE_URL="http://localhost:1234/v1"
export RESEARCH_AGENTS_API_KEY="local"
export RESEARCH_AGENTS_MODEL="qwen3:8b"
research-agents "Create a source map for local LLM evaluation."
```

List all built-in local model presets from the CLI:

```bash
research-agents --list-local-models
```

To preview the workflow without calling the API:

```bash
research-agents --plan-only "Your research question"
```

## Project layout

```text
src/research_agents/
  cli.py        # Command-line entry point
  config.py     # Runtime settings
  tools.py      # Custom tools available to agents
  workflow.py   # Agent definitions and orchestration
research_notes/ # Created at runtime when notes are saved
```

## Customizing the agents

1. Edit `src/research_agents/workflow.py` to change agent instructions, models, handoffs, or the conference-review workflow.
2. Add safe local tools in `src/research_agents/tools.py` with `@function_tool`.
3. Keep research outputs in `research_notes/` or point `RESEARCH_AGENTS_NOTES_DIR` to another folder.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RESEARCH_AGENTS_MODEL` | `gpt-4.1` | Model used by the sample agents. Can be a local preset such as `balanced` when `RESEARCH_AGENTS_PROVIDER=ollama`. |
| `RESEARCH_AGENTS_PROVIDER` | `openai` | Set to `ollama`, `local`, `lmstudio`, or `llama.cpp` to use an OpenAI-compatible local endpoint. |
| `RESEARCH_AGENTS_BASE_URL` | `http://localhost:11434/v1` in local mode | Base URL for an OpenAI-compatible local model server. |
| `RESEARCH_AGENTS_API_KEY` | `ollama` in local mode | API key placeholder or real key for the configured endpoint. |
| `RESEARCH_AGENTS_USE_CHAT_COMPLETIONS` | unset | Force the Agents SDK to use the Chat Completions API shape for non-default endpoints. |
| `RESEARCH_AGENTS_DISABLE_TRACING` | `1` in local mode | Disable hosted tracing export when running entirely locally. |
| `RESEARCH_AGENTS_NOTES_DIR` | `research_notes` | Directory where the note-saving tool writes markdown files. |

## Suggested research workflow

1. Ask for a research plan and key terms.
2. Use the plan to gather papers, datasets, and primary sources.
3. Paste abstracts or excerpts back into the agents for comparison and critique.
4. Ask the Critical Reviewer to identify methodological weaknesses.
5. Ask for a final synthesis with open questions and next actions.

## Safety and quality notes

- Treat agent output as a draft, not a source of truth.
- Verify claims against primary sources before citing them.
- Avoid pasting confidential or unpublished material unless your data-handling policy allows it.
