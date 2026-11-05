"""Tests for Modal container logging of sibling project modules."""

from __future__ import annotations

import importlib
import io
import logging
from pathlib import Path
import sys
from typing import Any, Iterator
import uuid

import pytest

from conftest import PACKAGE_NAME


@pytest.fixture
def cloud_runtime_logging_module(extension_package: object) -> Any:
    """Return the cloud runtime logging helpers."""
    del extension_package
    return importlib.import_module(f"{PACKAGE_NAME}.cloud_runtime_logging")


@pytest.fixture
def module_logger_names() -> Iterator[list[str]]:
    """Yield unique logger names and remove their handlers afterwards."""
    names = [f"test_cloud_module_{uuid.uuid4().hex}" for _ in range(2)]
    yield names
    for name in names:
        module_logger = logging.getLogger(name)
        for handler in list(module_logger.handlers):
            module_logger.removeHandler(handler)
        module_logger.setLevel(logging.NOTSET)
        module_logger.propagate = True


def test_cloud_module_loggers_emit_info_to_current_stdout(
    cloud_runtime_logging_module: Any,
    module_logger_names: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INFO from sibling modules must reach the stdout that exists at emit time."""
    cloud_runtime_logging_module.configure_cloud_module_loggers(module_logger_names)
    replaced_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", replaced_stdout)

    logging.getLogger(module_logger_names[0]).info(
        "Finished remote invocation completion in %.3fs", 1.5
    )

    output = replaced_stdout.getvalue()
    assert "Finished remote invocation completion in 1.500s" in output
    assert f"INFO [{module_logger_names[0]}]" in output


def test_cloud_module_loggers_are_idempotent_and_isolated(
    cloud_runtime_logging_module: Any,
    module_logger_names: list[str],
) -> None:
    """Repeat configuration must not duplicate output or leak to the root logger."""
    module = cloud_runtime_logging_module
    first = module.configure_cloud_module_loggers(module_logger_names)
    second = module.configure_cloud_module_loggers(module_logger_names)

    assert first == second == tuple(module_logger_names)
    handlers = [logging.getLogger(name).handlers for name in module_logger_names]
    assert all(len(module_handlers) == 1 for module_handlers in handlers)
    assert handlers[0][0] is handlers[1][0]
    for name in module_logger_names:
        module_logger = logging.getLogger(name)
        assert module_logger.level == logging.INFO
        assert module_logger.propagate is False


def test_cloud_module_loggers_skip_the_entrypoint_logger(
    cloud_runtime_logging_module: Any,
) -> None:
    """The entrypoint logger keeps its own handler and is never double-configured."""
    module = cloud_runtime_logging_module
    entrypoint_name = module._TARGET_LOGGER.name

    configured = module.configure_cloud_module_loggers([entrypoint_name])

    assert configured == ()


def test_cloud_project_module_names_lists_flat_modules(
    cloud_runtime_logging_module: Any,
    tmp_path: Path,
) -> None:
    """Every shipped top-level module except the package marker is configured."""
    for file_name in ("cloud_durable_invocation.py", "settings.py", "__init__.py"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")

    names = cloud_runtime_logging_module.cloud_project_module_names(tmp_path)

    assert names == ("cloud_durable_invocation", "settings")


def test_shipped_cloud_module_names_come_from_the_repo_copy(
    cloud_runtime_logging_module: Any,
) -> None:
    """Module discovery must not depend on where Modal places the entrypoint file."""
    names = cloud_runtime_logging_module.shipped_cloud_module_names()

    assert "cloud_durable_invocation" in names
    assert "cloud_session_bridge" in names
    assert "comfyui_modal_sync_cloud" in names
    assert "__init__" not in names
