"""Tests for workflow-triggered Vast worker image publication."""

from __future__ import annotations

import io
import sys
from pathlib import Path
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
    observed: dict[str, Any] = {}

    def fake_popen(command: Any, **kwargs: Any) -> FakeProcess:
        """Record safe process inputs and return a successful build."""
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return FakeProcess(
            "\x1b[32mBuilding layer 1/3\x1b[0m\n"
            f"Pushing sha256:{'d' * 64}\n"
            f"COMFY_MODAL_VAST_IMAGE={digest}\n"
        )

    monkeypatch.setattr(vast_image_build_module.subprocess, "Popen", fake_popen)
    vast_image_build_module._BUILT_IMAGES_BY_FINGERPRINT.clear()
    comfyui_root = tmp_path / "ComfyUI"
    builder = vast_image_build_module.VastWorkerImageBuilder(
        repo_root=tmp_path,
        comfyui_root=comfyui_root,
        environment={"PATH": "/usr/bin"},
    )
    statuses: list[str] = []

    result = builder.build_and_push("f" * 64, status_callback=statuses.append)

    assert result == digest
    assert observed["command"] == (
        sys.executable,
        "scripts/build_vast_worker_image.py",
        "--push",
    )
    assert observed["cwd"] == tmp_path
    assert observed["env"]["COMFYUI_ROOT"] == str(comfyui_root)
    assert observed["env"]["BUILDKIT_PROGRESS"] == "plain"
    assert statuses == [
        "Building the current Vast worker image",
        "Vast image build: Building layer 1/3",
        "Vast image build: Pushing sha256:[redacted]",
        "Published the current Vast worker image",
    ]


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
