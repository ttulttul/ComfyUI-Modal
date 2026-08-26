"""Tests for the direct Vast worker executor boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def test_vast_cancel_targets_owned_stager_before_worker_request(
    vast_executor_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A staging cancellation must terminate its transport and exact remote owner."""
    client = vast_executor_module.VastExecutorClient(
        registry=SimpleNamespace(),
        activity_manager=SimpleNamespace(),
        runtime_configuration=SimpleNamespace(),
        user_directory=tmp_path,
    )
    invocation_id = "RIV_staging"
    runner = SimpleNamespace()
    process = SimpleNamespace()
    terminated: list[Any] = []
    targeted: list[tuple[Any, str]] = []
    monkeypatch.setattr(
        vast_executor_module,
        "terminate_staging_transport",
        terminated.append,
    )
    client._cancel_remote_stager = (  # type: ignore[method-assign]
        lambda active_runner, owner_id, **_kwargs: (
            targeted.append((active_runner, owner_id)) or True
        )
    )
    with vast_executor_module._ACTIVE_VAST_STAGERS_LOCK:
        vast_executor_module._ACTIVE_VAST_STAGERS[invocation_id] = (
            runner,
            "vast:RIV_staging:owner",
            process,
        )
    try:
        cancelled = client.cancel({}, invocation_id)
    finally:
        with vast_executor_module._ACTIVE_VAST_STAGERS_LOCK:
            vast_executor_module._ACTIVE_VAST_STAGERS.pop(invocation_id, None)

    assert cancelled is True
    assert terminated == [process]
    assert targeted == [(runner, "vast:RIV_staging:owner")]
