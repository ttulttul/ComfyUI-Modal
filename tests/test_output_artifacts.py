"""Tests for downloading files produced inside remote ComfyUI executions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_output_snapshot_collects_new_and_replaced_regular_files(
    output_artifacts_module: Any,
    tmp_path: Path,
) -> None:
    """Only files changed after the pre-execution snapshot should be transferred."""
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    unchanged_path = output_directory / "unchanged.png"
    replaced_path = output_directory / "replaced.txt"
    unchanged_path.write_bytes(b"unchanged")
    replaced_path.write_bytes(b"before")

    snapshot = output_artifacts_module.snapshot_output_directory(output_directory)
    replaced_path.write_bytes(b"after-with-a-different-size")
    nested_path = output_directory / "videos" / "result.mp4"
    nested_path.parent.mkdir()
    nested_path.write_bytes(b"encoded-video")

    artifacts = output_artifacts_module.collect_output_artifacts(snapshot)

    assert [(artifact.relative_path, artifact.payload) for artifact in artifacts] == [
        ("replaced.txt", b"after-with-a-different-size"),
        ("videos/result.mp4", b"encoded-video"),
    ]


def test_remote_execution_result_round_trip_and_legacy_compatibility(
    output_artifacts_module: Any,
) -> None:
    """Artifact envelopes should be binary-safe and accept old output-only results."""
    result = output_artifacts_module.RemoteExecutionResult(
        outputs=b"serialized-node-outputs",
        artifacts=(
            output_artifacts_module.RemoteOutputArtifact(
                relative_path="video/render.mp4",
                payload=b"\x00\x01compressed-video",
            ),
        ),
        completed_epoch=1_786_123_456,
    )

    packed = output_artifacts_module.pack_remote_execution_result(result)
    unpacked = output_artifacts_module.unpack_remote_execution_result(packed)

    assert unpacked == result
    legacy_result = output_artifacts_module.unpack_remote_execution_result(b"legacy")
    assert legacy_result.outputs == b"legacy"

    corrupted = bytearray(packed)
    corrupted[-1] ^= 1
    with pytest.raises(
        output_artifacts_module.RemoteOutputArtifactError,
        match="SHA256",
    ):
        output_artifacts_module.unpack_remote_execution_result(corrupted)


def test_remote_execution_result_rejects_unsafe_artifact_paths(
    output_artifacts_module: Any,
) -> None:
    """A remote result must never be able to escape the local output directory."""
    result = output_artifacts_module.RemoteExecutionResult(
        outputs=b"[]",
        artifacts=(
            output_artifacts_module.RemoteOutputArtifact(
                relative_path="../outside.mp4",
                payload=b"video",
            ),
        ),
        completed_epoch=1,
    )

    with pytest.raises(
        output_artifacts_module.RemoteOutputArtifactError,
        match="Unsafe",
    ):
        output_artifacts_module.pack_remote_execution_result(result)


def test_materialize_remote_outputs_prefixes_names_and_avoids_overwrite(
    output_artifacts_module: Any,
    tmp_path: Path,
) -> None:
    """Downloads should retain subdirectories and never overwrite local data."""
    output_directory = tmp_path / "output"
    first_result = output_artifacts_module.RemoteExecutionResult(
        outputs=b"[]",
        artifacts=(
            output_artifacts_module.RemoteOutputArtifact(
                relative_path="videos/render.mp4",
                payload=b"first-video",
            ),
        ),
        completed_epoch=1_786_123_456,
    )

    first_paths = output_artifacts_module.materialize_remote_output_artifacts(
        first_result,
        output_directory=output_directory,
        app_name="comfy-modal-sync-ez7utL15S7Y",
    )
    replay_paths = output_artifacts_module.materialize_remote_output_artifacts(
        first_result,
        output_directory=output_directory,
        app_name="comfy-modal-sync-ez7utL15S7Y",
    )
    collision_result = output_artifacts_module.RemoteExecutionResult(
        outputs=b"[]",
        artifacts=(
            output_artifacts_module.RemoteOutputArtifact(
                relative_path="videos/render.mp4",
                payload=b"second-video",
            ),
        ),
        completed_epoch=1_786_123_456,
    )
    collision_paths = output_artifacts_module.materialize_remote_output_artifacts(
        collision_result,
        output_directory=output_directory,
        app_name="comfy-modal-sync-ez7utL15S7Y",
    )

    expected_path = (
        output_directory
        / "videos"
        / "remote-ez7utL15S7Y-786123456-render.mp4"
    )
    assert first_paths == (expected_path,)
    assert replay_paths == (expected_path,)
    assert expected_path.read_bytes() == b"first-video"
    assert collision_paths[0].name == "remote-ez7utL15S7Y-786123456-render-2.mp4"
    assert collision_paths[0].read_bytes() == b"second-video"
    assert len(list((output_directory / "videos").iterdir())) == 2


def test_materialize_remote_outputs_rejects_local_symlink_escape(
    output_artifacts_module: Any,
    tmp_path: Path,
) -> None:
    """An existing local output symlink must not redirect a download elsewhere."""
    output_directory = tmp_path / "output"
    outside_directory = tmp_path / "outside"
    output_directory.mkdir()
    outside_directory.mkdir()
    (output_directory / "videos").symlink_to(
        outside_directory,
        target_is_directory=True,
    )
    result = output_artifacts_module.RemoteExecutionResult(
        outputs=b"[]",
        artifacts=(
            output_artifacts_module.RemoteOutputArtifact(
                relative_path="videos/render.mp4",
                payload=b"encoded-video",
            ),
        ),
        completed_epoch=1_786_123_456,
    )

    with pytest.raises(
        output_artifacts_module.RemoteOutputArtifactError,
        match="escapes",
    ):
        output_artifacts_module.materialize_remote_output_artifacts(
            result,
            output_directory=output_directory,
            app_name="comfy-modal-sync-ez7utL15S7Y",
        )
    assert list(outside_directory.iterdir()) == []


def test_cloud_payload_capture_bundles_new_remote_outputs(
    modal_cloud_module: Any,
    output_artifacts_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cloud execution wrapper should snapshot and return newly generated files."""
    output_directory = tmp_path / "remote-output"
    output_directory.mkdir()
    (output_directory / "old.png").write_bytes(b"old")
    monkeypatch.setattr(
        modal_cloud_module,
        "_remote_comfy_output_directory",
        lambda: output_directory,
    )

    def execute_once() -> bytes:
        """Create one encoded file as a remote output node would."""
        (output_directory / "render.mp4").write_bytes(b"encoded-video")
        return b"serialized-outputs"

    packed = modal_cloud_module._execute_payload_with_output_capture(
        {
            "capture_remote_outputs": True,
            "clear_remote_session": True,
            "component_id": "component-1",
        },
        execute_once,
    )
    result = output_artifacts_module.unpack_remote_execution_result(packed)

    assert result.outputs == b"serialized-outputs"
    artifact_values = [
        (artifact.relative_path, artifact.payload)
        for artifact in result.artifacts
    ]
    assert artifact_values == [("render.mp4", b"encoded-video")]
    assert result.completed_epoch is not None


def test_cloud_stream_returns_output_artifact_bundle(
    modal_cloud_module: Any,
    output_artifacts_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production progress-stream path should return captured output files."""
    output_directory = tmp_path / "remote-output"
    output_directory.mkdir()
    monkeypatch.setattr(
        modal_cloud_module,
        "_remote_comfy_output_directory",
        lambda: output_directory,
    )

    def execute_subgraph(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        status_callback: Any = None,
    ) -> bytes:
        """Emulate a remote output node inside the streamed executor."""
        del payload, kwargs_payload, status_callback
        (output_directory / "render.mp4").write_bytes(b"encoded-video")
        return b"serialized-outputs"

    monkeypatch.setattr(
        modal_cloud_module,
        "execute_subgraph_locally",
        execute_subgraph,
    )
    events = list(
        modal_cloud_module._stream_remote_payload_events(
            {
                "payload_kind": "subgraph",
                "component_id": "component-1",
                "capture_remote_outputs": True,
            },
            b"{}",
        )
    )
    result = output_artifacts_module.unpack_remote_execution_result(
        events[-1]["outputs"]
    )

    assert events[-1]["kind"] == "result"
    assert result.outputs == b"serialized-outputs"
    assert [(item.relative_path, item.payload) for item in result.artifacts] == [
        ("render.mp4", b"encoded-video")
    ]


def test_local_remote_result_materializer_returns_node_outputs(
    remote_modal_app_module: Any,
    output_artifacts_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The local boundary should download files before returning node outputs."""
    output_directory = tmp_path / "local-output"
    result = output_artifacts_module.RemoteExecutionResult(
        outputs=b"serialized-outputs",
        artifacts=(
            output_artifacts_module.RemoteOutputArtifact(
                relative_path="render.mp4",
                payload=b"encoded-video",
            ),
        ),
        completed_epoch=1_786_123_456,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_local_comfy_output_directory",
        lambda settings: output_directory,
    )

    outputs = remote_modal_app_module._materialize_remote_execution_result(
        output_artifacts_module.pack_remote_execution_result(result),
        settings=SimpleNamespace(app_name="comfy-modal-sync-ez7utL15S7Y"),
    )

    assert outputs == b"serialized-outputs"
    assert (
        output_directory
        / "remote-ez7utL15S7Y-gpu-rtx-pro-6000-786123456-render.mp4"
    ).read_bytes() == b"encoded-video"
