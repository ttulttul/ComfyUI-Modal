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
