"""Tests for automatic Modal SDK startup installation."""

from __future__ import annotations

import importlib.machinery
import logging
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_pin_matches_declared_remote_extra(modal_sdk_module: Any) -> None:
    """The runtime installer and project metadata should require the same Modal release."""
    project_metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project_metadata["project"]["optional-dependencies"]["remote"] == [
        modal_sdk_module.MODAL_PACKAGE_SPEC
    ]


def test_remote_mode_installs_and_reimports_missing_modal_sdk(
    modal_sdk_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote startup should install a missing SDK and retry the import immediately."""
    importable_results = iter([False, True])
    install_modal_sdk = Mock(return_value=True)
    invalidate_caches = Mock()
    mock_logger = Mock(spec=logging.Logger)
    monkeypatch.setattr(
        modal_sdk_module,
        "_modal_sdk_is_importable",
        lambda: next(importable_results),
    )
    monkeypatch.setattr(modal_sdk_module, "_install_modal_sdk", install_modal_sdk)
    monkeypatch.setattr(modal_sdk_module.importlib, "invalidate_caches", invalidate_caches)
    monkeypatch.setattr(modal_sdk_module, "logger", mock_logger)

    assert modal_sdk_module.ensure_modal_sdk_available("remote") is True

    install_modal_sdk.assert_called_once_with()
    invalidate_caches.assert_called_once_with()
    mock_logger.warning.assert_called_once()
    mock_logger.info.assert_called_once()


def test_local_mode_does_not_install_missing_modal_sdk(
    modal_sdk_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local startup should preserve the dependency-free development path."""
    install_modal_sdk = Mock(return_value=True)
    monkeypatch.setattr(modal_sdk_module, "_modal_sdk_is_importable", lambda: False)
    monkeypatch.setattr(modal_sdk_module, "_install_modal_sdk", install_modal_sdk)

    assert modal_sdk_module.ensure_modal_sdk_available("local") is False

    install_modal_sdk.assert_not_called()


def test_uv_install_command_targets_running_python(
    modal_sdk_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preferred uv installer must write into the interpreter running ComfyUI."""
    monkeypatch.setattr(modal_sdk_module.shutil, "which", lambda name: "/usr/local/bin/uv")

    assert modal_sdk_module._build_install_command() == [
        "/usr/local/bin/uv",
        "pip",
        "install",
        "--python",
        modal_sdk_module.sys.executable,
        modal_sdk_module.MODAL_PACKAGE_SPEC,
    ]


def test_pip_install_command_is_used_when_uv_is_unavailable(
    modal_sdk_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap should fall back to the current interpreter's pip module."""
    pip_spec = importlib.machinery.ModuleSpec("pip", loader=None)
    monkeypatch.setattr(modal_sdk_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(modal_sdk_module.importlib.util, "find_spec", lambda name: pip_spec)

    assert modal_sdk_module._build_install_command() == [
        modal_sdk_module.sys.executable,
        "-m",
        "pip",
        "install",
        modal_sdk_module.MODAL_PACKAGE_SPEC,
    ]


def test_installer_runs_pinned_command_with_timeout(
    modal_sdk_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic installation should be bounded and use the exact pinned package spec."""
    command = ["/usr/local/bin/uv", "pip", "install", "modal==1.4.2"]
    observed_kwargs: dict[str, Any] = {}

    def fake_run(
        observed_command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Record the installer invocation and return a successful process result."""
        assert observed_command == command
        observed_kwargs.update(kwargs)
        return subprocess.CompletedProcess(observed_command, 0, stdout="installed", stderr="")

    mock_logger = Mock(spec=logging.Logger)
    monkeypatch.setattr(modal_sdk_module, "_build_install_command", lambda: command)
    monkeypatch.setattr(modal_sdk_module.subprocess, "run", fake_run)
    monkeypatch.setattr(modal_sdk_module, "logger", mock_logger)

    assert modal_sdk_module._install_modal_sdk() is True

    assert observed_kwargs == {
        "capture_output": True,
        "check": False,
        "text": True,
        "timeout": modal_sdk_module._INSTALL_TIMEOUT_SECONDS,
    }
    mock_logger.info.assert_any_call(
        "Automatic Modal SDK installation completed successfully for %s.",
        modal_sdk_module.MODAL_PACKAGE_SPEC,
    )


def test_installer_failure_is_logged_and_preserves_startup(
    modal_sdk_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed package install should log an actionable error and return the fallback signal."""
    command = ["/usr/local/bin/uv", "pip", "install", "modal==1.4.2"]
    failed_process = subprocess.CompletedProcess(command, 2, stdout="", stderr="network error")
    mock_logger = Mock(spec=logging.Logger)
    monkeypatch.setattr(modal_sdk_module, "_build_install_command", lambda: command)
    monkeypatch.setattr(
        modal_sdk_module.subprocess,
        "run",
        lambda *args, **kwargs: failed_process,
    )
    monkeypatch.setattr(modal_sdk_module, "logger", mock_logger)

    assert modal_sdk_module._install_modal_sdk() is False

    mock_logger.error.assert_called_once_with(
        "Automatic Modal SDK installation failed with exit status %d. Retry manually: %s",
        2,
        "/usr/local/bin/uv pip install modal==1.4.2",
    )
