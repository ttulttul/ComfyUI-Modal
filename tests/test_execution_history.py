"""Tests for persistent runtime observations used by the scheduler."""

from __future__ import annotations

from typing import Any


def test_history_returns_median_recent_runtime_per_environment(
    execution_history_module: Any,
    tmp_path: Any,
) -> None:
    """Cost estimates should resist one slow outlier and stay host-specific."""
    module = execution_history_module
    history = module.ExecutionHistory.for_user_directory(tmp_path)
    for index, elapsed in enumerate((10.0, 12.0, 100.0), start=1):
        history.record(
            module.ExecutionObservation(
                component_signature="component",
                environment_id="host-one",
                provider="ssh_docker",
                elapsed_seconds=elapsed,
                recorded_at_epoch=float(index),
            )
        )
    history.record(
        module.ExecutionObservation(
            component_signature="component",
            environment_id="host-two",
            provider="ssh_docker",
            elapsed_seconds=3.0,
            recorded_at_epoch=4.0,
        )
    )

    estimates = history.estimates("component", ("host-one", "host-two"))

    assert estimates["host-one"].execution_seconds == 12.0
    assert estimates["host-one"].sample_count == 3
    assert estimates["host-two"].execution_seconds == 3.0


def test_history_retains_only_the_recent_sample_window(
    execution_history_module: Any,
    tmp_path: Any,
) -> None:
    """Long-lived ComfyUI installations should keep timing storage bounded."""
    module = execution_history_module
    history = module.ExecutionHistory.for_user_directory(tmp_path)
    for index in range(30):
        history.record(
            module.ExecutionObservation(
                component_signature="bounded",
                environment_id="host",
                provider="ssh_docker",
                elapsed_seconds=float(index),
                recorded_at_epoch=float(index),
            )
        )

    estimate = history.estimates("bounded", ("host",))["host"]

    assert estimate.sample_count == 20
    assert estimate.execution_seconds == 19.5
