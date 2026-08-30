"""Tests for the direct Vast worker executor boundary."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _CompletedRelayProcess:
    """Expose in-memory streams for one completed Vast relay command."""

    def __init__(self, stdout: bytes, stderr: bytes, returncode: int) -> None:
        """Initialize deterministic subprocess-compatible state."""
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode

    def poll(self) -> int:
        """Return the completed relay status."""
        return self.returncode

    def wait(self, timeout: float) -> int:
        """Return the completed relay status without blocking."""
        del timeout
        return self.returncode

    def terminate(self) -> None:
        """Reject unexpected termination of an already-completed relay."""
        raise AssertionError("completed relay should not be terminated")

    def kill(self) -> None:
        """Reject unexpected killing of an already-completed relay."""
        raise AssertionError("completed relay should not be killed")


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


def test_invoke_stream_surfaces_structured_worker_oom(
    vast_executor_module: Any,
    remote_protocol_module: Any,
    tmp_path: Path,
) -> None:
    """The controller should show cgroup OOM evidence as a terminal resource error."""
    error_frame = remote_protocol_module.encode_json_frame(
        remote_protocol_module.RemoteFrameKind.ERROR,
        {
            "error_type": "WorkerOutOfMemoryError",
            "message": (
                "Vast worker was killed because the instance ran out of host RAM "
                "(container cgroup OOM; instance RAM limit 88.1 GiB)."
            ),
        },
    )
    process = _CompletedRelayProcess(
        error_frame,
        b"Welcome to vast.ai.\nHave fun!.\n",
        70,
    )
    runner = SimpleNamespace(popen=lambda _arguments: process)
    client = vast_executor_module.VastExecutorClient(
        registry=SimpleNamespace(),
        activity_manager=SimpleNamespace(),
        runtime_configuration=SimpleNamespace(),
        user_directory=tmp_path,
    )

    with pytest.raises(
        vast_executor_module.VastRemoteResourceError,
        match="ran out of host RAM.*88.1 GiB",
    ):
        list(
            client._invoke_stream(
                runner,
                {"invocation_id": "RIV_oom"},
                b"inputs",
            )
        )


def test_invoke_stream_discards_vast_login_greeting_without_postmortem(
    vast_executor_module: Any,
    tmp_path: Path,
) -> None:
    """Vast's generic SSH greeting must never be presented as the crash cause."""
    process = _CompletedRelayProcess(
        b"",
        (
            b"Welcome to vast.ai. If authentication fails, try again after a few "
            b"seconds, and double check your ssh key.\nHave fun!.\n"
        ),
        0,
    )
    runner = SimpleNamespace(popen=lambda _arguments: process)
    client = vast_executor_module.VastExecutorClient(
        registry=SimpleNamespace(),
        activity_manager=SimpleNamespace(),
        runtime_configuration=SimpleNamespace(),
        user_directory=tmp_path,
    )

    with pytest.raises(
        vast_executor_module.VastRemoteTransportError,
        match="No structured worker postmortem was available",
    ) as raised:
        list(
            client._invoke_stream(
                runner,
                {"invocation_id": "RIV_legacy"},
                b"inputs",
            )
        )

    assert "Welcome to vast.ai" not in str(raised.value)
    assert "authentication fails" not in str(raised.value)
    assert "SSH diagnostic" not in str(raised.value)


def test_consume_stream_does_not_retry_evidenced_oom(
    vast_executor_module: Any,
    tmp_path: Path,
) -> None:
    """Restarting the same worker cannot recover from an undersized RAM limit."""
    restart_count = 0

    class Runtime:
        """Record worker lifecycle calls made by the executor."""

        def ensure_worker(self) -> None:
            """Represent an already-ready worker."""

        def restart_worker(self) -> None:
            """Record an unexpected resource-error recovery attempt."""
            nonlocal restart_count
            restart_count += 1

    client = vast_executor_module.VastExecutorClient(
        registry=SimpleNamespace(),
        activity_manager=SimpleNamespace(),
        runtime_configuration=SimpleNamespace(),
        user_directory=tmp_path,
    )
    runtime = Runtime()
    client._runtime = lambda _payload: (  # type: ignore[method-assign]
        SimpleNamespace(instance_id=42),
        SimpleNamespace(),
        runtime,
    )
    client._ensure_llm_profiles_staged = (  # type: ignore[method-assign]
        lambda _lease, _runner, _payload: None
    )

    def invoke(*_arguments: Any) -> Any:
        """Raise the terminal error when the stream is first consumed."""
        raise vast_executor_module.VastRemoteResourceError("container cgroup OOM")
        yield

    client._invoke_stream = invoke  # type: ignore[method-assign]

    with pytest.raises(
        vast_executor_module.VastRemoteResourceError,
        match="container cgroup OOM",
    ):
        client._consume_stream(
            {"invocation_id": "RIV_oom", "vast_instance_id": 42},
            b"inputs",
        )

    assert restart_count == 0
