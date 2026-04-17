"""Tests for prompt rewriting and asset sync integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class _FakeRemoteModelNode:
    """Fake node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteSamplerNode:
    """Fake node that consumes a model and produces a transportable latent."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteClipNode:
    """Fake node that produces a non-transportable CLIP output."""

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_IS_LIST = (False,)


class _FakeLocalSinkNode:
    """Fake local node used to verify downstream rewiring."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


def test_rewrite_groups_connected_remote_nodes_into_single_proxy(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Connected remote nodes should collapse into one proxy-backed component."""
    model_path = tmp_path / "weights.safetensors"
    model_path.write_bytes(b"weights")
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()
    (custom_nodes_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=custom_nodes_dir,
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteModel": _FakeRemoteModelNode,
                "RemoteSampler": _FakeRemoteSamplerNode,
                "LocalSink": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {"model_name": str(model_path)},
            "_meta": {"title": "Model"},
        },
        "2": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0]},
            "_meta": {"title": "Sampler"},
        },
        "3": {
            "class_type": "LocalSink",
            "inputs": {"latent": ["2", 0]},
            "_meta": {"title": "Sink"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
        extra_data={"extra_pnginfo": {"workflow": workflow}},
    )

    assert set(rewritten_prompt) == {"1", "3"}
    rewritten_node = rewritten_prompt["1"]
    payload = rewritten_node["inputs"]["original_node_data"]
    assert rewritten_node["class_type"].startswith("ModalUniversalExecutor_")
    assert payload["payload_kind"] == "subgraph"
    assert payload["subgraph_prompt"]["1"]["inputs"]["model_name"].startswith("/assets/")
    assert payload["execute_node_ids"] == ["2"]
    assert payload["boundary_inputs"] == []
    assert payload["boundary_outputs"] == [
        {
            "proxy_output_name": "2_latent",
            "node_id": "2",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
        }
    ]
    assert rewritten_prompt["3"]["inputs"]["latent"] == ["1", 0]
    assert summary.remote_node_ids == ["1", "2"]
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {"1": ["1", "2"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "1"}
    assert len(summary.synced_assets) == 1


def test_rewrite_rejects_non_transportable_remote_inputs(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Remote component boundaries should reject non-transportable local inputs."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()
    (custom_nodes_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=custom_nodes_dir,
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteConsumer": _FakeRemoteSamplerNode,
                "ModelSource": _FakeRemoteModelNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": True}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "ModelSource",
            "inputs": {},
            "_meta": {"title": "Model Source"},
        },
        "2": {
            "class_type": "RemoteConsumer",
            "inputs": {"model": ["1", 0]},
            "_meta": {"title": "Remote Consumer"},
        },
    }

    try:
        api_intercept_module.rewrite_prompt_for_modal(
            prompt=prompt,
            workflow=workflow,
            sync_engine=sync_engine,
            settings=settings,
            nodes_module=fake_nodes_module,
        )
    except api_intercept_module.ModalPromptValidationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ModalPromptValidationError to be raised.")

    assert "input 'model'" in message
    assert "type 'MODEL'" in message
    assert "cannot cross the current local/remote boundary" in message


def test_rewrite_rejects_non_transportable_remote_outputs(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Remote component boundaries should reject non-transportable local downstream edges."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()
    (custom_nodes_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=custom_nodes_dir,
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteClip": _FakeRemoteClipNode,
                "LocalConsumer": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteClip",
            "inputs": {},
            "_meta": {"title": "Remote Clip"},
        },
        "2": {
            "class_type": "LocalConsumer",
            "inputs": {"clip": ["1", 0]},
            "_meta": {"title": "Local Consumer"},
        },
    }

    try:
        api_intercept_module.rewrite_prompt_for_modal(
            prompt=prompt,
            workflow=workflow,
            sync_engine=sync_engine,
            settings=settings,
            nodes_module=fake_nodes_module,
        )
    except api_intercept_module.ModalPromptValidationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ModalPromptValidationError to be raised.")

    assert "exports node 1 (RemoteClip) output index 0 of type 'CLIP'" in message
    assert "cannot cross the current local/remote boundary" in message


def test_emit_modal_status_targets_prompt_client(
    api_intercept_module: Any,
) -> None:
    """Modal status events should preserve prompt and component metadata for the UI."""

    class FakePromptServer:
        """Capture websocket events emitted by the queue route."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[str, dict[str, Any], str | None]] = []

        def send_sync(self, event: str, data: dict[str, Any], sid: str | None) -> None:
            """Record an emitted websocket message."""
            self.messages.append((event, data, sid))

    prompt_server = FakePromptServer()
    api_intercept_module._emit_modal_status(
        prompt_server=prompt_server,
        phase="setup",
        client_id="client-1",
        prompt_id="prompt-1",
        node_ids=["4", "5"],
        component_node_ids_by_representative={"4": ["4", "5"]},
    )

    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "setup",
                "prompt_id": "prompt-1",
                "node_ids": ["4", "5"],
                "components": [
                    {
                        "representative_node_id": "4",
                        "node_ids": ["4", "5"],
                    }
                ],
            },
            "client-1",
        )
    ]
