"""Tests for direct Vast runtime launch and readiness management."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest


class FakeResult:
    """Minimal SSH result used by the runtime manager."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Serialize one JSON standard output object."""
        self.stdout_text = json.dumps(payload)


class FakeRunner:
    """Return sequenced runtime-info and record supervisor calls."""

    def __init__(self, infos: list[dict[str, Any]]) -> None:
        """Configure runtime states returned by successive probes."""
        self.infos = infos
        self.calls: list[tuple[str, ...]] = []
        self.call_options: list[dict[str, Any]] = []

    def run(self, argv: Any, **kwargs: Any) -> FakeResult:
        """Record the command and return its simulated result."""
        arguments = tuple(argv)
        self.calls.append(arguments)
        self.call_options.append(dict(kwargs))
        if arguments[-1] == "runtime-info":
            return FakeResult(self.infos.pop(0))
        return FakeResult({"ok": True})


def test_launch_spec_starts_worker_and_watchdog_without_exporting_api_key(
    vast_models_module: Any,
    vast_runtime_module: Any,
) -> None:
    """The launch command should persist only fixed non-secret runtime settings."""
    configuration = vast_runtime_module.VastRuntimeConfiguration(
        image="ghcr.io/example/worker@sha256:abc",
        runtime_fingerprint="a" * 64,
    )
    profile = vast_models_module.VastResourceProfile(
        profile_id="17",
        profile_name="default",
    )

    launch = configuration.launch_spec(profile, "managed-label")
    payload = launch.to_api_payload()

    assert payload["image"] == configuration.image
    assert payload["runtype"] == "ssh_direct"
    assert "remote.vast_supervisor start" in payload["onstart"]
    assert "CONTAINER_API_KEY" not in payload["onstart"]
    assert "VAST_API_KEY" not in payload["onstart"]
    assert "\nCOMFY_VAST_ENV\n" in payload["onstart"]
    assert "\n COMFY_VAST_ENV\n" not in payload["onstart"]
    assert payload["env"]["COMFY_MODAL_LLM_EXECUTION_TARGET"] == "vast"
    assert "COMFY_MODAL_RUNTIME_FINGERPRINT" not in payload["env"]
    assert "COMFY_MODAL_RUNTIME_FINGERPRINT" not in payload["onstart"]


def test_configuration_requires_explicit_published_image(
    vast_runtime_module: Any,
) -> None:
    """Do not rent against an invented or unpublished default image tag."""
    with pytest.raises(RuntimeError, match="COMFY_MODAL_VAST_IMAGE"):
        vast_runtime_module.VastRuntimeConfiguration.from_environment(
            "a" * 64,
            environment={},
        )


def test_runtime_manager_starts_missing_socket_then_verifies_fingerprint(
    vast_runtime_module: Any,
) -> None:
    """A present image can idempotently recover its supervised worker."""
    fingerprint = "a" * 64
    runner = FakeRunner(
        [
            {
                "runtime_fingerprint": fingerprint,
                "worker_socket_ready": False,
            },
            {
                "runtime_fingerprint": fingerprint,
                "worker_socket_ready": True,
            },
        ]
    )
    manager = vast_runtime_module.VastRuntimeManager(
        runner=runner,
        configuration=vast_runtime_module.VastRuntimeConfiguration(
            image="worker",
            runtime_fingerprint=fingerprint,
            readiness_poll_seconds=0.001,
        ),
        sleep=lambda _seconds: None,
    )

    info = manager.ensure_worker()

    assert info["worker_socket_ready"] is True
    assert any("remote.vast_supervisor" in call for call in runner.calls)
    runtime_info_options = [
        options
        for call, options in zip(runner.calls, runner.call_options, strict=True)
        if call[-1] == "runtime-info"
    ]
    assert all(options["transport_attempts"] == 1 for options in runtime_info_options)


def test_runtime_manager_rejects_image_fingerprint_drift(
    vast_runtime_module: Any,
) -> None:
    """Never execute against a stale or incorrectly published worker image."""
    manager = vast_runtime_module.VastRuntimeManager(
        runner=FakeRunner(
            [{"runtime_fingerprint": "b" * 64, "worker_socket_ready": True}]
        ),
        configuration=vast_runtime_module.VastRuntimeConfiguration(
            image="worker",
            runtime_fingerprint="a" * 64,
        ),
    )

    with pytest.raises(Exception, match="fingerprint mismatch"):
        manager.ensure_worker()


def test_runtime_manager_accepts_same_protocol_fingerprint_drift(
    vast_runtime_module: Any,
) -> None:
    """Controller-only source changes may reuse the same protocol-compatible image."""
    manager = vast_runtime_module.VastRuntimeManager(
        runner=FakeRunner(
            [
                {
                    "protocol_version": vast_runtime_module.REMOTE_PROTOCOL_VERSION,
                    "runtime_fingerprint": "b" * 64,
                    "worker_socket_ready": True,
                }
            ]
        ),
        configuration=vast_runtime_module.VastRuntimeConfiguration(
            image="worker",
            runtime_fingerprint="a" * 64,
        ),
    )

    info = manager.ensure_worker()

    assert info["worker_socket_ready"] is True


def test_runtime_manager_logs_changed_ssh_readiness_diagnostics(
    vast_runtime_module: Any,
    caplog: Any,
) -> None:
    """SSH startup failures should be visible before the readiness timeout."""

    class FailingRunner:
        """Raise the same provider handshake diagnostic on each probe."""

        def run(self, argv: Any, **kwargs: Any) -> FakeResult:
            """Simulate Vast closing SSH during key exchange."""
            del argv, kwargs
            raise vast_runtime_module.VastSshError(
                "kex_exchange_identification: Connection closed by remote host"
            )

    current_time = [0.0]

    def advance(seconds: float) -> None:
        """Advance the deterministic monotonic readiness clock."""
        current_time[0] += seconds

    manager = vast_runtime_module.VastRuntimeManager(
        runner=FailingRunner(),
        configuration=vast_runtime_module.VastRuntimeConfiguration(
            image="worker",
            runtime_fingerprint="a" * 64,
            startup_timeout_seconds=2.0,
            readiness_poll_seconds=1.0,
        ),
        monotonic=lambda: current_time[0],
        sleep=advance,
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(TimeoutError, match="Connection closed by remote host"):
            manager.ensure_worker()

    readiness_messages = [
        record.message
        for record in caplog.records
        if "Vast worker readiness probe" in record.message
    ]
    assert readiness_messages == [
        "Vast worker readiness probe attempt=1 failed: "
        "kex_exchange_identification: Connection closed by remote host"
    ]


def test_runtime_manager_validates_instance_before_ssh_probe(
    vast_runtime_module: Any,
) -> None:
    """A vanished Vast contract must stop readiness before stale SSH is attempted."""

    class RecordingRunner:
        """Fail the test if worker readiness reaches the SSH transport."""

        def __init__(self) -> None:
            """Initialize the SSH invocation counter."""
            self.calls = 0

        def run(self, argv: Any, **kwargs: Any) -> FakeResult:
            """Record the forbidden SSH call."""
            del argv, kwargs
            self.calls += 1
            return FakeResult({})

    runner = RecordingRunner()

    def reject_missing_instance() -> None:
        """Simulate Vast removing the contract from the live inventory."""
        raise RuntimeError("Vast instance no longer exists")

    manager = vast_runtime_module.VastRuntimeManager(
        runner=runner,
        configuration=vast_runtime_module.VastRuntimeConfiguration(
            image="worker",
            runtime_fingerprint="a" * 64,
        ),
        instance_validator=reject_missing_instance,
    )

    with pytest.raises(RuntimeError, match="no longer exists"):
        manager.ensure_worker()

    assert runner.calls == 0
