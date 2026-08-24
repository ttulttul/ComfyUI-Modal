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

    def run(self, argv: Any, **kwargs: Any) -> FakeResult:
        """Record the command and return its simulated result."""
        del kwargs
        arguments = tuple(argv)
        self.calls.append(arguments)
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
