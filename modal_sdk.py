"""Startup bootstrap for the optional Modal Python SDK."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shlex
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

MODAL_PACKAGE_SPEC = "modal==1.4.2"
_INSTALL_TIMEOUT_SECONDS = 300


def _remote_execution_requested(execution_mode: str | None = None) -> bool:
    """Return whether startup configuration requests Modal-backed execution."""
    resolved_mode = execution_mode
    if resolved_mode is None:
        resolved_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    return resolved_mode.strip().lower() == "remote"


def _build_install_command() -> list[str] | None:
    """Build an installer command targeting the current Python interpreter."""
    uv_executable = shutil.which("uv")
    if uv_executable is not None:
        return [
            uv_executable,
            "pip",
            "install",
            "--python",
            sys.executable,
            MODAL_PACKAGE_SPEC,
        ]
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", MODAL_PACKAGE_SPEC]
    return None


def _log_installer_output(completed_process: subprocess.CompletedProcess[str]) -> None:
    """Log captured installer output at debug level for troubleshooting."""
    if completed_process.stdout.strip():
        logger.debug("Modal SDK installer stdout:\n%s", completed_process.stdout.strip())
    if completed_process.stderr.strip():
        logger.debug("Modal SDK installer stderr:\n%s", completed_process.stderr.strip())


def _install_modal_sdk() -> bool:
    """Install the pinned Modal SDK into the interpreter running ComfyUI."""
    command = _build_install_command()
    if command is None:
        logger.error(
            "Cannot install required Modal SDK %s automatically because neither uv nor pip "
            "is available for Python interpreter %s.",
            MODAL_PACKAGE_SPEC,
            sys.executable,
        )
        return False

    command_text = shlex.join(command)
    logger.info("Running automatic Modal SDK installation: %s", command_text)
    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Automatic Modal SDK installation timed out after %d seconds: %s",
            _INSTALL_TIMEOUT_SECONDS,
            command_text,
        )
        return False
    except OSError as exc:
        logger.error("Unable to start automatic Modal SDK installation %s: %s", command_text, exc)
        return False

    _log_installer_output(completed_process)
    if completed_process.returncode != 0:
        logger.error(
            "Automatic Modal SDK installation failed with exit status %d. Retry manually: %s",
            completed_process.returncode,
            command_text,
        )
        return False
    logger.info("Automatic Modal SDK installation completed successfully for %s.", MODAL_PACKAGE_SPEC)
    return True


def _modal_sdk_is_importable() -> bool:
    """Return whether the Modal SDK imports without a missing-module error."""
    try:
        importlib.import_module("modal")
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            logger.error(
                "The Modal SDK is installed but cannot import because module %s is missing.",
                exc.name,
            )
        return False
    return True


def ensure_modal_sdk_available(execution_mode: str | None = None) -> bool:
    """Install and import the pinned Modal SDK when remote execution needs it."""
    if _modal_sdk_is_importable():
        logger.debug("Modal SDK is already available to Python interpreter %s.", sys.executable)
        return True
    if not _remote_execution_requested(execution_mode):
        logger.debug("Modal SDK is unavailable, but local execution mode does not require it.")
        return False

    logger.warning(
        "Modal SDK is missing from Python interpreter %s; installing %s automatically.",
        sys.executable,
        MODAL_PACKAGE_SPEC,
    )
    if not _install_modal_sdk():
        return False

    importlib.invalidate_caches()
    if not _modal_sdk_is_importable():
        logger.error(
            "Modal SDK %s was installed but is still not importable in Python interpreter %s.",
            MODAL_PACKAGE_SPEC,
            sys.executable,
        )
        return False
    logger.info("Modal SDK %s is ready for remote execution.", MODAL_PACKAGE_SPEC)
    return True
