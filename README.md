# My Research Agents

A small Python starter kit for building OpenAI-powered agents for research work. It uses the OpenAI Agents SDK so you can define specialist agents, give them tools, and run a coordinated research workflow from the command line.

## What is included

- **Research Planner**: decomposes a research question into concrete search, reading, and synthesis steps.
- **Literature Scout**: identifies useful source types, keywords, and inclusion/exclusion criteria.
- **Critical Reviewer**: challenges assumptions, flags weak evidence, and suggests follow-up checks.
- **Research Orchestrator**: routes your request to the right specialist agent.
- **Local notes tool**: saves structured notes to `research_notes/` so your work is not trapped in chat history.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="sk-..."
research-agents "Map the current debates around retrieval-augmented generation evaluation."
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

1. Edit `src/research_agents/workflow.py` to change agent instructions, models, or handoffs.
2. Add safe local tools in `src/research_agents/tools.py` with `@function_tool`.
3. Keep research outputs in `research_notes/` or point `RESEARCH_AGENTS_NOTES_DIR` to another folder.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RESEARCH_AGENTS_MODEL` | `gpt-4.1` | Model used by the sample agents. |
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
