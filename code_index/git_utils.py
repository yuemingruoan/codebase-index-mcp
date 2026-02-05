from __future__ import annotations

import subprocess

from .errors import GitError
from .paths import normalize_repo_path


def _run_git(repo_path: str, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", repo_path, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def is_git_repo(repo_path: str) -> bool:
    repo_path = normalize_repo_path(repo_path)
    try:
        output = _run_git(repo_path, ["rev-parse", "--is-inside-work-tree"])
    except GitError:
        return False
    return output.strip() == "true"


def get_repo_root(repo_path: str) -> str:
    repo_path = normalize_repo_path(repo_path)
    return _run_git(repo_path, ["rev-parse", "--show-toplevel"]).strip()


def list_tracked_files(repo_root: str) -> list[str]:
    output = _run_git(repo_root, ["ls-files"])
    return [line for line in output.splitlines() if line]


def get_head_commit(repo_root: str) -> str:
    return _run_git(repo_root, ["rev-parse", "HEAD"]).strip()
