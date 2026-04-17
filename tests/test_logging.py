"""Tests for Modal-Sync logging configuration."""

from __future__ import annotations

import logging
from typing import Any


def test_extension_installs_timestamped_logger_handler(extension_package: Any) -> None:
    """The extension should install a dedicated timestamped handler for its logger hierarchy."""
    handler_name = extension_package._EXTENSION_HANDLER_NAME
    extension_logger = logging.getLogger(extension_package._EXTENSION_LOGGER_NAME)

    matching_handlers = [
        handler for handler in extension_logger.handlers if getattr(handler, "name", "") == handler_name
    ]

    assert len(matching_handlers) == 1
    assert extension_logger.propagate is False
    formatter = matching_handlers[0].formatter
    assert formatter is not None
    assert "%(asctime)s" in formatter._fmt
    assert "%(relativeCreated)" in formatter._fmt
