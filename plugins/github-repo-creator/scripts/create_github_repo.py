#!/usr/bin/env python3
"""Create GitHub repositories with token or interactive GitHub CLI login."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def normalize_repo_name(name: str) -> str:
    normalized = "".join(char if char.isalnum() or char in ".-_" else "-" for char in name.strip())
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("repository name cannot be empty")
    return normalized[:100]


def create_with_token(repo_name: str, description: str, private: bool, owner: str, token: str) -> str:
    endpoint = f"https://api.github.com/orgs/{owner}/repos" if owner else "https://api.github.com/user/repos"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "name": repo_name,
                "description": description,
                "private": private,
                "auto_init": False,
            }
        ).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "github-repo-creator-plugin/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc.reason}") from exc

    html_url = data.get("html_url")
    if not isinstance(html_url, str) or not html_url:
        raise RuntimeError("GitHub API response did not include html_url")
    return html_url


def gh_is_logged_in() -> bool:
    return subprocess.run(
        ["gh", "auth", "status", "--hostname", "github.com"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    ).returncode == 0


def ensure_gh_login() -> None:
    if gh_is_logged_in():
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "GitHub login is required. Run `gh auth login --hostname github.com --web` "
            "or set GITHUB_TOKEN/GH_TOKEN before creating a repository."
        )
    print("GitHub login is required before creating a repository.", file=sys.stderr)
    subprocess.run(["gh", "auth", "login", "--hostname", "github.com", "--web"], check=True)


def create_with_gh(repo_name: str, description: str, private: bool, owner: str) -> str:
    if shutil.which("gh") is None:
        raise RuntimeError(
            "GitHub CLI (`gh`) is not installed. Install it and run `gh auth login`, "
            "or set GITHUB_TOKEN/GH_TOKEN."
        )
    ensure_gh_login()
    endpoint = f"orgs/{owner}/repos" if owner else "user/repos"
    cmd = [
        "gh",
        "api",
        endpoint,
        "--method",
        "POST",
        "--field",
        f"name={repo_name}",
        "--field",
        f"description={description}",
        "--field",
        f"private={str(private).lower()}",
        "--field",
        "auto_init=false",
        "--jq",
        ".html_url",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    html_url = result.stdout.strip()
    if not html_url:
        raise RuntimeError("GitHub CLI did not return a repository URL")
    return html_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GitHub repository.")
    parser.add_argument("name", help="Repository name to create.")
    parser.add_argument("--description", default="Research reproduction workspace")
    parser.add_argument("--owner", default="", help="Organization owner; omit for authenticated user.")
    parser.add_argument(
        "--visibility",
        choices=("public", "private"),
        default="public",
        help="Repository visibility (default: public).",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo_name = normalize_repo_name(args.name)
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        private = args.visibility == "private"
        if token:
            html_url = create_with_token(repo_name, args.description, private, args.owner.strip(), token)
        else:
            html_url = create_with_gh(repo_name, args.description, private, args.owner.strip())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = {"html_url": html_url, "name": repo_name}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
