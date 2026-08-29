"""Timestamped logging and phase timing for the cloud worker runtime."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_TARGET_LOGGER: logging.Logger = logger
_HANDLER_NAME = "comfyui-modal-sync-cloud-timestamped"


def _build_cloud_log_formatter() -> logging.Formatter:
    """Return the default formatter for remote Modal-Sync logs with timestamps."""
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d +%(relativeCreated)07.0fms %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def configure_cloud_runtime_logging(
    target_logger: logging.Logger,
    handler_name: str,
) -> logging.Logger:
    """Configure and retain the stable entrypoint logger used by cloud helpers."""
    global _TARGET_LOGGER
    global _HANDLER_NAME
    _TARGET_LOGGER = target_logger
    _HANDLER_NAME = handler_name
    target_logger.setLevel(logging.INFO)
    for existing_handler in target_logger.handlers:
        if getattr(existing_handler, "name", "") == handler_name:
            return target_logger

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(handler_name)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_build_cloud_log_formatter())
    target_logger.addHandler(handler)
    target_logger.propagate = False
    return target_logger


def _is_modal_container_runtime() -> bool:
    """Return whether the current process is executing inside a Modal container."""
    return os.getenv("MODAL_IS_REMOTE") == "1" or bool(os.getenv("MODAL_TASK_ID"))


def _cloud_formatter() -> logging.Formatter:
    """Return the configured formatter used for cloud phase trace lines."""
    for existing_handler in _TARGET_LOGGER.handlers:
        if getattr(existing_handler, "name", "") == _HANDLER_NAME:
            formatter = existing_handler.formatter
            if formatter is not None:
                return formatter
    return _build_cloud_log_formatter()


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Emit an info line and mirror it to stdout inside Modal containers."""
    if not _is_modal_container_runtime():
        _TARGET_LOGGER.info(message, *args)
        return

    record = _TARGET_LOGGER.makeRecord(
        _TARGET_LOGGER.name,
        logging.INFO,
        __file__,
        0,
        message,
        args,
        exc_info=None,
    )
    print(_cloud_formatter().format(record), file=sys.stdout, flush=True)


@contextmanager
def _timed_phase(phase: str, **fields: Any) -> Iterator[None]:
    """Log a start/finish pair with elapsed time for a named execution phase."""
    field_suffix = ""
    if fields:
        rendered_fields = " ".join(f"{key}={value}" for key, value in fields.items())
        field_suffix = f" {rendered_fields}"
    phase_started_at = time.perf_counter()
    _emit_cloud_info("Starting %s%s", phase, field_suffix)
    try:
        yield
    finally:
        _emit_cloud_info(
            "Finished %s in %.3fs%s",
            phase,
            time.perf_counter() - phase_started_at,
            field_suffix,
        )
