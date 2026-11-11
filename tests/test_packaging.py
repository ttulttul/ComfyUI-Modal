"""Packaging guards for the explicit .gitignore whitelist."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_ignored(paths: list[str]) -> list[str]:
    """Return the repo-relative paths git would ignore."""
    result = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def test_runtime_python_modules_are_not_git_ignored() -> None:
    """Every shipped Python module must be whitelisted, or fresh clones break.

    The .gitignore ignores everything (`*`) and whitelists files one by one,
    so a new module is silently left out of commits unless it is added there.
    """
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("git checkout not available")
    module_paths = sorted(
        str(path.relative_to(REPO_ROOT))
        for pattern in ("*.py", "remote/*.py")
        for path in REPO_ROOT.glob(pattern)
    )

    assert _git_ignored(module_paths) == []
