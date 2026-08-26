"""Tests for bounded controller-side remote staging processes."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest


def test_staging_process_aborts_remote_after_no_progress(
    staging_process_module: Any,
) -> None:
    """A silent remote stager must be terminated and explicitly cancelled."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    aborted: list[bool] = []

    with pytest.raises(
        staging_process_module.RemoteStagingProcessError,
        match="produced no progress",
    ):
        staging_process_module.consume_staging_process(
            process,
            {},
            lambda _payload, _event: None,
            provider_label="Test",
            abort_remote=lambda: aborted.append(True),
            timeout_seconds=0.05,
        )

    assert process.poll() is not None
    assert aborted == [True]


def test_staging_process_returns_result_after_progress(
    staging_process_module: Any,
) -> None:
    """Valid progress and terminal result envelopes should pass through intact."""
    script = (
        "print('{\"kind\":\"progress\",\"stage\":\"download\"}');"
        "print('{\"kind\":\"result\",\"results\":[{\"profile_id\":\"p\"}]}')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    observed: list[dict[str, Any]] = []

    results = staging_process_module.consume_staging_process(
        process,
        {"component_id": "llm"},
        lambda _payload, event: observed.append(dict(event)),
        provider_label="Test",
        abort_remote=lambda: pytest.fail("successful staging must not abort"),
        timeout_seconds=1.0,
    )

    assert observed[0]["stage"] == "download"
    assert results == [{"profile_id": "p"}]


def test_invalid_staging_timeout_still_cleans_up_process(
    staging_process_module: Any,
) -> None:
    """A bad timeout setting must not strand the subprocess it was meant to bound."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    aborted: list[bool] = []

    with pytest.raises(ValueError, match="must be positive"):
        staging_process_module.consume_staging_process(
            process,
            {},
            lambda _payload, _event: None,
            provider_label="Test",
            abort_remote=lambda: aborted.append(True),
            timeout_seconds=0,
        )

    assert process.poll() is not None
    assert aborted == [True]
