---
name: github-repo-creator
description: Create GitHub repositories for reproduction workspaces. Use when a user asks to create, publish, or connect a GitHub repo, especially when login may be required.
---

# GitHub Repo Creator

Use `scripts/create_github_repo.py` to create repositories on GitHub.

## Authentication behavior

1. If `GITHUB_TOKEN` or `GH_TOKEN` is set, the script uses the GitHub REST API directly.
2. If no token is set and the GitHub CLI (`gh`) is installed, the script checks `gh auth status`.
3. When run in an interactive terminal and `gh` is not logged in, the script runs `gh auth login --hostname github.com --web` so GitHub can ask the user to log in.
4. In non-interactive runs without a token or existing `gh` login, the script exits with a clear message telling the caller to run `gh auth login` or set `GITHUB_TOKEN`/`GH_TOKEN`.

## Usage

```bash
python plugins/github-repo-creator/scripts/create_github_repo.py my-repo \
  --description "Paper reproduction workspace" \
  --visibility private
```

For organization-owned repositories:

```bash
python plugins/github-repo-creator/scripts/create_github_repo.py my-repo --owner my-org
```

Use `--output result.json` when another workflow needs machine-readable output. The JSON contains `html_url`.
