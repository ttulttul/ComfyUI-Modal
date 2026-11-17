"""Tests for the files the Modal image build takes from the extension repo."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from conftest import PACKAGE_NAME


@pytest.fixture(scope="session")
def cloud_image_env_module(extension_package: object) -> Any:
    """Return the Modal image filtering module."""
    return importlib.import_module(f"{PACKAGE_NAME}.cloud_image_env")


@pytest.mark.parametrize(
    "relative_path",
    [
        ".cache",
        ".cache/strings/739b7f44.txt",
        ".git/HEAD",
        ".venv/bin/python",
        ".agents/notes.md",
        ".gitignore",
        "remote/__pycache__/mapped_execution.cpython-312.pyc",
        "logs/run.log",
    ],
)
def test_repo_image_filter_skips_dot_paths_caches_and_logs(
    cloud_image_env_module: Any,
    relative_path: str,
) -> None:
    """Runtime caches written into the extension directory never enter the image."""
    assert cloud_image_env_module._should_ignore_repo_path(Path(relative_path))


@pytest.mark.parametrize(
    "relative_path",
    ["__init__.py", "remote/mapped_execution.py", "web/modal_toggle.js", "llm_profiles.json"],
)
def test_repo_image_filter_keeps_runtime_sources(
    cloud_image_env_module: Any,
    relative_path: str,
) -> None:
    """Source files the worker imports stay in the image."""
    assert not cloud_image_env_module._should_ignore_repo_path(Path(relative_path))


def test_repo_image_filter_judges_absolute_paths_from_the_repo_root(
    cloud_image_env_module: Any,
) -> None:
    """Dot-prefixed parents of the repo, such as a worktree under ~/.t3, do not hide sources."""
    repo_root = cloud_image_env_module._REPO_ROOT

    assert not cloud_image_env_module._should_ignore_repo_path(repo_root / "__init__.py")
    assert cloud_image_env_module._should_ignore_repo_path(repo_root / ".cache" / "strings" / "x.txt")
