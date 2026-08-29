"""Tests for workflow-triggered Vast worker image publication."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class FakeProcess:
    """Expose deterministic merged subprocess output and an exit status."""

    def __init__(self, output: str, return_code: int = 0) -> None:
        """Initialize the readable output and return code."""
        self.stdout = io.StringIO(output)
        self.return_code = return_code

    def wait(self) -> int:
        """Return the configured process status."""
        return self.return_code


def test_builder_streams_progress_and_returns_digest(
    vast_image_build_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Build output should reach workflow status before the digest is returned."""
    digest = "ghcr.io/example/worker@sha256:" + "a" * 64
    fingerprint = "f" * 64
    observed: dict[str, Any] = {}

    def fake_popen(command: Any, **kwargs: Any) -> FakeProcess:
        """Record safe process inputs and return a successful build."""
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return FakeProcess(
            "\x1b[32mBuilding layer 1/3\x1b[0m\n"
            f"Pushing sha256:{'d' * 64}\n"
            f"COMFY_MODAL_VAST_SOURCE_FINGERPRINT={fingerprint}\n"
            f"COMFY_MODAL_VAST_IMAGE={digest}\n"
        )

    monkeypatch.setattr(vast_image_build_module.subprocess, "Popen", fake_popen)
    vast_image_build_module._BUILT_IMAGES_BY_FINGERPRINT.clear()
    comfyui_root = tmp_path / "ComfyUI"
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=comfyui_root,
        modal_gpu="RTX-PRO-6000",
        environment={"PATH": "/usr/bin"},
    )
    statuses: list[str] = []

    result = builder.build_and_push(fingerprint, status_callback=statuses.append)

    assert result == digest
    assert observed["command"] == (
        sys.executable,
        "scripts/build_vast_worker_image.py",
        "--push",
        "--expected-fingerprint",
        fingerprint,
    )
    assert observed["cwd"] == tmp_path
    assert observed["env"]["COMFYUI_ROOT"] == str(comfyui_root)
    assert observed["env"]["COMFY_MODAL_GPU"] == "RTX-PRO-6000"
    assert observed["env"]["BUILDKIT_PROGRESS"] == "plain"
    assert statuses == [
        "Building the current Vast worker image",
        "Vast image build: Building layer 1/3",
        "Vast image build: Pushing sha256:[redacted]",
        "Published the current Vast worker image",
    ]


def test_builder_reuses_current_published_image_without_building(
    vast_image_build_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Matching registry metadata should avoid both publication and GPU rental delay."""
    fingerprint = "f" * 64
    image = "ghcr.io/example/worker@sha256:" + "a" * 64
    monkeypatch.setattr(
        vast_image_build_module,
        "published_image_metadata",
        lambda _image: SimpleNamespace(
            runtime_fingerprint=fingerprint,
            immutable_image=image,
        ),
    )
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=None,
    )
    monkeypatch.setattr(
        vast_image_build_module.VastWorkerImageBuilder,
        "build_and_push",
        lambda *_args, **_kwargs: pytest.fail("current image was rebuilt"),
    )
    statuses: list[str] = []

    result = builder.ensure_published_image(
        image,
        fingerprint,
        status_callback=statuses.append,
    )

    assert result == image
    assert statuses == [
        "Checking the published Vast worker image",
        "Published Vast worker image is current",
    ]


def test_builder_rebuilds_stale_published_image(
    vast_image_build_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A mismatched registry label should publish before capacity is requested."""
    expected = "a" * 64
    replacement = "ghcr.io/example/worker@sha256:" + "c" * 64
    monkeypatch.setattr(
        vast_image_build_module,
        "published_image_metadata",
        lambda _image: SimpleNamespace(
            runtime_fingerprint="b" * 64,
            immutable_image="ghcr.io/example/worker@sha256:" + "b" * 64,
        ),
    )
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=None,
    )
    requested: list[str] = []

    def fake_build(
        self: Any,
        fingerprint: str,
        *,
        status_callback: Any,
    ) -> str:
        """Record the expected source identity and return a replacement."""
        del self, status_callback
        requested.append(fingerprint)
        return replacement

    monkeypatch.setattr(
        vast_image_build_module.VastWorkerImageBuilder,
        "build_and_push",
        fake_build,
    )
    statuses: list[str] = []

    result = builder.ensure_published_image(
        "ghcr.io/example/worker:v1",
        expected,
        status_callback=statuses.append,
    )

    assert result == replacement
    assert requested == [expected]
    assert "stale; rebuilding before requesting capacity" in statuses[-1]


def test_builder_publishes_missing_configured_image(
    vast_image_build_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A first publication should be automatic when the configured tag is absent."""
    expected = "a" * 64
    replacement = "ghcr.io/example/worker@sha256:" + "d" * 64

    def missing_image(_image: str) -> Any:
        """Report a registry tag that has not been published yet."""
        raise vast_image_build_module.VastImageNotFoundError("not found")

    monkeypatch.setattr(
        vast_image_build_module,
        "published_image_metadata",
        missing_image,
    )
    monkeypatch.setattr(
        vast_image_build_module.VastWorkerImageBuilder,
        "build_and_push",
        lambda _self, _fingerprint, *, status_callback: replacement,
    )
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=None,
    )
    statuses: list[str] = []

    result = builder.ensure_published_image(
        "ghcr.io/example/worker:v1",
        expected,
        status_callback=statuses.append,
    )

    assert result == replacement
    assert "missing; building before requesting capacity" in statuses[-1]


def test_builder_failure_requires_the_documented_manual_command(
    vast_image_build_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Missing local build prerequisites should cancel with recovery guidance."""
    monkeypatch.setattr(
        vast_image_build_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(
            "Fatal Python error: init_fs_encoding\n"
            "ModuleNotFoundError: No module named 'encodings'\n"
            "Current thread 0x00000001 (most recent call first):\n"
            "<no Python frame>\n",
            2,
        ),
    )
    vast_image_build_module._BUILT_IMAGES_BY_FINGERPRINT.clear()
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=None,
        environment={"PATH": "/usr/bin"},
    )

    with pytest.raises(
        vast_image_build_module.VastWorkerImageBuildError,
        match=r"python.* scripts/build_vast_worker_image\.py --push",
    ) as raised:
        builder.build_and_push("e" * 64)

    assert "ModuleNotFoundError: No module named 'encodings'" in str(raised.value)
    assert "<no Python frame> Run" not in str(raised.value)


def test_builder_rejects_source_changed_after_comfyui_started(
    vast_image_build_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A newly built source identity must not be adopted under an old fingerprint."""
    expected = "a" * 64
    actual = "b" * 64
    monkeypatch.setattr(
        vast_image_build_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(
            f"COMFY_MODAL_VAST_SOURCE_FINGERPRINT={actual}\n"
            "RuntimeError: Local runtime source changed after ComfyUI started\n",
            1,
        ),
    )
    vast_image_build_module._BUILT_IMAGES_BY_FINGERPRINT.clear()
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=None,
        environment={"PATH": "/usr/bin"},
    )

    with pytest.raises(
        vast_image_build_module.VastWorkerImageBuildError,
        match="no manual image build is needed",
    ) as raised:
        builder.build_and_push(expected)

    assert expected[:12] in str(raised.value)
    assert actual[:12] in str(raised.value)
