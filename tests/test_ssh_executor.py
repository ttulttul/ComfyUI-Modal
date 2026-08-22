"""Tests for SSH worker invocation and local artifact restoration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def test_ssh_stream_materializes_remote_output_artifacts(
    ssh_executor_module: Any,
    remote_modal_app_module: Any,
    output_artifacts_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """SSH results should restore output files before deserializing node outputs."""
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    packed_result = output_artifacts_module.pack_remote_execution_result(
        output_artifacts_module.RemoteExecutionResult(
            outputs=b"serialized outputs",
            artifacts=(
                output_artifacts_module.RemoteOutputArtifact(
                    relative_path="images/result.png",
                    payload=b"png bytes",
                ),
            ),
            completed_epoch=1_700_000_000,
        )
    )
    settings = SimpleNamespace(
        comfyui_root=tmp_path,
        app_name="test-app",
    )
    client = ssh_executor_module.SshDockerExecutorClient(
        registry=SimpleNamespace(),
        repo_root=tmp_path,
        settings=settings,
    )
    manager = SimpleNamespace()
    spec = SimpleNamespace()
    monkeypatch.setattr(client, "_runtime", lambda _payload: (manager, spec))
    monkeypatch.setattr(
        client,
        "_invoke_stream",
        lambda *_args: iter(({"kind": "result", "outputs": packed_result},)),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_consume_remote_payload_stream",
        lambda _payload, stream: next(stream)["outputs"],
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_local_comfy_output_directory",
        lambda _settings: output_directory,
    )

    outputs = client._consume_stream({"component_id": "1"}, b"inputs")

    assert outputs == b"serialized outputs"
    materialized = list(output_directory.rglob("*.png"))
    assert len(materialized) == 1
    assert materialized[0].read_bytes() == b"png bytes"


def test_ssh_transport_failure_restarts_worker_and_retries_once(
    ssh_executor_module: Any,
    remote_modal_app_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A broken relay should recover the exact worker slot with the stable request."""
    stopped: list[int] = []
    manager = SimpleNamespace(
        controller=SimpleNamespace(host=SimpleNamespace(environment_id="host")),
        stop_worker=lambda worker_index: stopped.append(worker_index),
    )
    spec = SimpleNamespace(worker_index=2)
    client = ssh_executor_module.SshDockerExecutorClient(
        registry=SimpleNamespace(),
        repo_root=tmp_path,
        settings=SimpleNamespace(comfyui_root=tmp_path, app_name="app"),
    )
    attempts = 0

    def invoke(*_args: Any) -> Any:
        """Fail the first relay attempt and complete the second."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ssh_executor_module.SshRemoteTransportError("connection lost")
        return iter(({"kind": "result", "outputs": b"serialized"},))

    monkeypatch.setattr(client, "_runtime", lambda _payload: (manager, spec))
    monkeypatch.setattr(client, "_invoke_stream", invoke)
    monkeypatch.setattr(
        remote_modal_app_module,
        "_consume_remote_payload_stream",
        lambda _payload, stream: next(stream)["outputs"],
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_materialize_remote_execution_result",
        lambda response, settings: response,
    )

    result = client._consume_stream({"component_id": "1"}, b"inputs")

    assert result == b"serialized"
    assert attempts == 2
    assert stopped == [2]
