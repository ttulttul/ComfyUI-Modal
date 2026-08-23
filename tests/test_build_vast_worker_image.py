"""Tests for safe Vast worker image build configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _module() -> Any:
    """Load the build script as a testable module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_vast_worker_image.py"
    spec = importlib.util.spec_from_file_location("build_vast_worker_image", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Vast image build script.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_tag_requires_versioned_non_latest_reference() -> None:
    """Long-lived rentals must not silently change runtime image identity."""
    module = _module()

    assert module._validate_tag("ghcr.io/example/worker:v0.4.0") == (
        "ghcr.io/example/worker:v0.4.0"
    )
    with pytest.raises(ValueError, match="explicit version"):
        module._validate_tag("ghcr.io/example/worker")
    with pytest.raises(ValueError, match="mutable latest"):
        module._validate_tag("ghcr.io/example/worker:latest")


def test_parser_accepts_explicit_comfyui_source_root(tmp_path: Path) -> None:
    """Standalone checkouts need a direct source-root override for image builds."""
    module = _module()

    arguments = module._parser().parse_args(
        [
            "--tag",
            "ghcr.io/example/worker:v1",
            "--comfyui-root",
            str(tmp_path),
        ]
    )

    assert arguments.comfyui_root == tmp_path


def test_default_tag_uses_pyproject_version_and_repository_owner(
    tmp_path: Path,
) -> None:
    """The normal push command should need no manually duplicated version tag."""
    module = _module()
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "modal-sync"
version = "1.2.3"

[project.urls]
Repository = "https://github.com/Example-Owner/ComfyUI-Modal"
""".strip(),
        encoding="utf-8",
    )

    assert module._resolve_image_tag(
        None,
        owner=None,
        tag_template=module.DEFAULT_TAG_TEMPLATE,
        repo_root=tmp_path,
    ) == "ghcr.io/example-owner/comfy-modal-worker:v1.2.3"


def test_default_tag_accepts_owner_and_template_overrides(tmp_path: Path) -> None:
    """Non-default organizations and registries remain configurable without --tag."""
    module = _module()
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "modal-sync"
version = "2.0.0rc1"
""".strip(),
        encoding="utf-8",
    )

    assert module._resolve_image_tag(
        None,
        owner="Container-Org",
        tag_template="registry.example/{owner}/vast-worker:{version}",
        repo_root=tmp_path,
    ) == "registry.example/container-org/vast-worker:2.0.0rc1"


def test_parser_allows_push_without_explicit_tag() -> None:
    """The documented one-line push command must parse without --tag."""
    module = _module()

    arguments = module._parser().parse_args(["--push"])

    assert arguments.push is True
    assert arguments.tag is None


def test_docker_build_targets_vast_x86_64_platform() -> None:
    """Apple Silicon builders must produce the architecture used by Vast workers."""
    module = _module()

    command = module._docker_build_command(
        "ghcr.io/example/worker:v1",
        "runtime-fingerprint",
    )

    assert command[:5] == (
        "docker",
        "build",
        "--pull",
        "--platform",
        "linux/amd64",
    )
    assert "comfy.remote.runtime-fingerprint=runtime-fingerprint" in command
