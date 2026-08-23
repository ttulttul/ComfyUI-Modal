"""Tests for the direct Vast worker executor boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_payload_instance_identity_accepts_concrete_environment(
    vast_executor_module: Any,
) -> None:
    """Queue-time environment identities should route to exact leases."""
    assert (
        vast_executor_module._payload_instance_id(
            {"execution_environment_id": "vast:node-17:9001"}
        )
        == 9001
    )
    assert vast_executor_module._payload_instance_id({"vast_instance_id": 9}) == 9


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"execution_environment_id": "vast:profile"},
        {"vast_instance_id": 0},
    ],
)
def test_payload_instance_identity_rejects_unresolved_profile(
    vast_executor_module: Any,
    payload: dict[str, Any],
) -> None:
    """Execution cannot begin before a profile resolves to a concrete lease."""
    with pytest.raises(
        vast_executor_module.VastRemoteInvocationError,
        match="instance identity",
    ):
        vast_executor_module._payload_instance_id(payload)


def test_payload_retention_defaults_to_twenty_four_hours(
    vast_executor_module: Any,
) -> None:
    """Legacy or defensive payloads should retain the planned long cooldown."""
    assert vast_executor_module._payload_retention_seconds({}) == 24 * 3600
    assert (
        vast_executor_module._payload_retention_seconds(
            {"vast_idle_retention_seconds": 60}
        )
        == 60
    )


def test_runtime_rejects_unregistered_or_draining_lease(
    tmp_path: Path,
    vast_executor_module: Any,
    vast_leases_module: Any,
    vast_runtime_module: Any,
) -> None:
    """The executor must never infer or reconnect to an unowned instance."""

    class Activity:
        """Unused activity manager."""

    client = vast_executor_module.VastExecutorClient(
        registry=vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path),
        activity_manager=Activity(),
        runtime_configuration=vast_runtime_module.VastRuntimeConfiguration(
            image="worker",
            runtime_fingerprint="a" * 64,
        ),
        user_directory=tmp_path,
    )

    with pytest.raises(
        vast_executor_module.VastRemoteInvocationError,
        match="not present in the local registry",
    ):
        client._runtime({"vast_instance_id": 9001})
