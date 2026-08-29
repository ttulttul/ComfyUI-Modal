"""Tests for safe Vast worker image build configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
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


def test_build_stops_before_docker_when_expected_source_changed(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Automatic publication must compare startup identity before exporting context."""
    module = _module()
    expected = "a" * 64
    actual = "b" * 64
    settings = SimpleNamespace(
        comfyui_root=tmp_path,
        custom_nodes_dir=tmp_path,
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        module,
        "build_remote_runtime_identity",
        lambda **_kwargs: SimpleNamespace(fingerprint=actual),
    )
    monkeypatch.setattr(
        module,
        "export_worker_dependency_image_context",
        lambda **_kwargs: pytest.fail("build context was exported after source drift"),
    )

    with pytest.raises(RuntimeError, match="Restart ComfyUI"):
        module.build_image(
            "ghcr.io/example/worker:v1",
            push=True,
            expected_fingerprint=expected,
        )

    assert capsys.readouterr().out.strip() == (
        f"{module.SOURCE_FINGERPRINT_RESULT_PREFIX}{actual}"
    )


def test_docker_build_targets_vast_x86_64_platform() -> None:
    """Apple Silicon builders must produce the architecture used by Vast workers."""
    module = _module()

    command = module._docker_build_command(
        "ghcr.io/example/worker:v1",
        "runtime-fingerprint",
    )

    assert command[:4] == (
        "docker",
        "build",
        "--platform",
        "linux/amd64",
    )
    assert "comfy.remote.runtime-fingerprint=runtime-fingerprint" in command
    assert "--pull" not in command


def test_dependency_tag_uses_stable_fingerprint_in_same_repository() -> None:
    """Source builds should share a registry base keyed only by dependencies."""
    module = _module()

    assert module._dependency_image_tag(
        "registry.example:5443/team/worker:v1",
        "cafebabe" * 8,
    ) == "registry.example:5443/team/worker:deps-cafebabecafebabe"


def test_run_can_inherit_stdout_for_live_docker_progress(monkeypatch: Any) -> None:
    """Push output must not be buffered until the publication process exits."""
    module = _module()
    observed: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        """Record subprocess output routing and report success."""
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=None)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert (
        module._run(
            ("docker", "push", "example/worker:v1"),
            capture_stdout=False,
        )
        == ""
    )
    assert observed["stdout"] is None


def test_build_publishes_dependency_base_then_small_source_overlay(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A missing dependency base should be pushed once and referenced by digest."""
    module = _module()
    identity = SimpleNamespace(fingerprint="a" * 64, manifest={})
    settings = SimpleNamespace(comfyui_root=tmp_path, custom_nodes_dir=tmp_path)
    calls: list[tuple[tuple[str, ...], bytes | None, bool]] = []
    overlay_inputs: dict[str, Any] = {}
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        module,
        "build_remote_runtime_identity",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        module,
        "remote_runtime_dependency_fingerprint",
        lambda _identity: "b" * 64,
    )
    monkeypatch.setattr(module, "_pull_current_dependency", lambda *_args: False)
    monkeypatch.setattr(
        module,
        "export_worker_dependency_image_context",
        lambda **_kwargs: b"dependency-context",
    )

    def overlay_context(**kwargs: Any) -> bytes:
        """Record the immutable base used by the source overlay."""
        overlay_inputs.update(kwargs)
        return b"source-context"

    monkeypatch.setattr(module, "export_worker_source_overlay_context", overlay_context)

    def fake_run(
        command: Any,
        *,
        input_payload: bytes | None = None,
        capture_stdout: bool = True,
    ) -> str:
        """Record Docker calls and synthesize repository digests."""
        normalized = tuple(command)
        calls.append((normalized, input_payload, capture_stdout))
        if normalized[:3] == ("docker", "image", "inspect"):
            repository = module._image_repository(normalized[-1])
            return f"{repository}@sha256:" + "c" * 64
        return ""

    monkeypatch.setattr(module, "_run", fake_run)

    result = module.build_image("ghcr.io/example/worker:v1", push=True)

    dependency_tag = "ghcr.io/example/worker:deps-" + "b" * 16
    dependency_digest = "ghcr.io/example/worker@sha256:" + "c" * 64
    assert result == dependency_digest
    assert overlay_inputs["dependency_image"] == dependency_digest
    assert calls[0][0] == module._docker_dependency_build_command(
        dependency_tag,
        "b" * 64,
    )
    assert calls[0][1:] == (b"dependency-context", False)
    assert ("docker", "push", dependency_tag) in [call[0] for call in calls]
    overlay_build = next(call for call in calls if call[1] == b"source-context")
    assert "comfy.remote.runtime-fingerprint=" + "a" * 64 in overlay_build[0]
    assert overlay_build[2] is False
