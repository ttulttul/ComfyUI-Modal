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
