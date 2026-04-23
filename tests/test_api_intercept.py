"""Tests for prompt rewriting and asset sync integration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class _FakeRemoteModelNode:
    """Fake node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)


class _FakeCheckpointLoaderSimpleNode:
    """Fake root loader node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteSamplerNode:
    """Fake node that consumes a model and produces a transportable latent."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)


class _FakeLatentSourceNode:
    """Fake node that produces a transportable LATENT output."""

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


class _FakeRemoteConditioningNode:
    """Fake node that produces a non-transportable CONDITIONING output."""

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteImageNode:
    """Fake remote node that produces a transportable IMAGE output."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteImageConsumerNode:
    """Fake remote node that consumes IMAGE and produces IMAGE."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteModelAndImageNode:
    """Fake remote node that produces both MODEL and IMAGE outputs."""

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "image")
    OUTPUT_IS_LIST = (False, False)


class _FakeRemoteModelAndImageConsumerNode:
    """Fake remote node that consumes MODEL and IMAGE and produces IMAGE."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


class _FakePromptListNode:
    """Fake upstream node that represents a prompt-list producer."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)


class _FakeModalMapInputNode:
    """Fake Modal map marker node that passes a wildcard value through."""

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteStringEchoNode:
    """Fake remote node that echoes a string output."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)


class _FakeLocalStringSinkNode:
    """Fake local node used to consume remote STRING outputs."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)


def test_queue_prompt_json_includes_resolved_modal_metadata(
    api_intercept_module: Any,
    monkeypatch: Any,
) -> None:
    """Successful queue responses should include resolved remote nodes and component membership."""

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def __init__(self) -> None:
            """Initialize captured queue items."""
            self.items: list[tuple[Any, ...]] = []

        def put(self, item: tuple[Any, ...]) -> None:
            """Record one queued prompt item."""
            self.items.append(item)

    class FakePromptServer:
        """Minimal PromptServer double for queue-response tests."""

        def __init__(self) -> None:
            """Initialize queue state."""
            self.number = 0
            self.prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeExecutionModule:
        """Minimal execution module exposing prompt validation."""

        SENSITIVE_EXTRA_DATA_KEYS: tuple[str, ...] = ()

        @staticmethod
        async def validate_prompt(
            prompt_id: str,
            prompt: dict[str, Any],
            partial_execution_targets: Any,
        ) -> tuple[bool, None, list[str], list[Any]]:
            """Accept the supplied prompt with one fake execution target."""
            return True, None, ["1"], []

    monkeypatch.setattr(api_intercept_module, "_get_execution_module", lambda: FakeExecutionModule)
    prompt_server = FakePromptServer()

    response = asyncio.run(
        api_intercept_module._queue_prompt_json(
            prompt_server,
            {
                "prompt_id": "prompt-1",
                "prompt": {"1": {"class_type": "Anything", "inputs": {}}},
                "extra_data": {},
            },
            modal_response_payload={
                "modal_remote_node_ids": ["1", "2"],
                "modal_components": [
                    {
                        "representative_node_id": "1",
                        "node_ids": ["1", "2"],
                    }
                ],
            },
        )
    )

    response_payload = json.loads(response.text)
    assert response_payload["prompt_id"] == "prompt-1"
    assert response_payload["modal_remote_node_ids"] == ["1", "2"]
    assert response_payload["modal_components"] == [
        {
            "representative_node_id": "1",
            "node_ids": ["1", "2"],
        }
    ]


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
    assert "prompt_id" not in payload
    assert payload["component_node_ids"] == ["1", "2"]
    assert payload["subgraph_prompt"]["1"]["inputs"]["model_name"].startswith("/assets/")
    assert payload["execute_node_ids"] == ["2"]
    assert "requires_volume_reload" not in payload
    assert "volume_reload_marker" not in payload
    assert "uploaded_volume_paths" not in payload
    assert payload["terminate_container_on_error"] is True
    assert payload["boundary_inputs"] == []
    assert payload["boundary_outputs"] == [
        {
            "proxy_output_name": "2_latent",
            "node_id": "2",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
            "preview_target_node_ids": [],
        }
    ]
    assert rewritten_prompt["3"]["inputs"]["latent"] == ["1", 0]
    assert summary.remote_node_ids == ["1", "2"]
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {"1": ["1", "2"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "1"}
    assert len(summary.synced_assets) == 1
    assert summary.synced_assets[0].uploaded is True


def test_rewrite_strips_prompt_id_from_cache_safe_proxy_payload(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Cache-safe remote proxies should not bake prompt_id into original_node_data inputs."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
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
            "inputs": {},
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

    rewritten_prompt, _summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
        extra_data={"prompt_id": "prompt-1", "client_id": "client-1"},
    )

    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert "prompt_id" not in payload


def test_rewrite_records_local_preview_targets_for_remote_boundary_images(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Boundary IMAGE outputs should remember direct local PreviewImage consumers."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteImage": _FakeRemoteImageNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 9, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteImage",
            "inputs": {},
            "_meta": {"title": "Remote Image"},
        },
        "9": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["1", 0]},
            "_meta": {"title": "Preview"},
        },
    }

    rewritten_prompt, _summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["boundary_outputs"] == [
        {
            "proxy_output_name": "1_image",
            "node_id": "1",
            "output_index": 0,
            "io_type": "IMAGE",
            "is_list": False,
            "preview_target_node_ids": ["9"],
        }
    ]


def test_rewrite_splits_remote_chain_across_transportable_edges(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Transportable remote-to-remote edges should become ordered proxy boundaries."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteImage": _FakeRemoteImageNode,
                "RemoteImageConsumer": _FakeRemoteImageConsumerNode,
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
            "class_type": "RemoteImage",
            "inputs": {},
            "_meta": {"title": "Remote Image"},
        },
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 0]},
            "_meta": {"title": "Remote Image Consumer"},
        },
        "3": {
            "class_type": "LocalSink",
            "inputs": {"image": ["2", 0]},
            "_meta": {"title": "Local Sink"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", "2", "3"}
    assert summary.remote_component_ids == ["1", "2"]
    assert summary.component_node_ids_by_representative == {"1": ["1"], "2": ["2"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "2"}

    first_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    second_payload = rewritten_prompt["2"]["inputs"]["original_node_data"]

    assert first_payload["component_node_ids"] == ["1"]
    assert first_payload["boundary_inputs"] == []
    assert first_payload["boundary_outputs"] == [
        {
            "proxy_output_name": "1_image",
            "node_id": "1",
            "output_index": 0,
            "io_type": "IMAGE",
            "is_list": False,
            "preview_target_node_ids": [],
        }
    ]

    assert second_payload["component_node_ids"] == ["2"]
    assert second_payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "IMAGE",
            "targets": [{"node_id": "2", "input_name": "image"}],
        }
    ]
    assert second_payload["boundary_outputs"] == [
        {
            "proxy_output_name": "2_image",
            "node_id": "2",
            "output_index": 0,
            "io_type": "IMAGE",
            "is_list": False,
            "preview_target_node_ids": [],
        }
    ]
    assert rewritten_prompt["2"]["inputs"]["remote_input_0"] == ["1", 0]
    assert rewritten_prompt["3"]["inputs"]["image"] == ["2", 0]


def test_rewrite_reports_parallel_component_stages(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Prompt rewrites should report best-effort concurrent stages for independent remote components."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteImage": _FakeRemoteImageNode,
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
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {"class_type": "RemoteImage", "inputs": {}, "_meta": {"title": "Remote A"}},
        "2": {"class_type": "RemoteImage", "inputs": {}, "_meta": {"title": "Remote B"}},
        "3": {"class_type": "LocalSink", "inputs": {"image": ["1", 0]}, "_meta": {"title": "Sink A"}},
        "4": {"class_type": "LocalSink", "inputs": {"image": ["2", 0]}, "_meta": {"title": "Sink B"}},
    }

    _rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert summary.component_execution_stages == [["1", "2"]]
    assert summary.component_dependency_ids_by_representative == {"1": [], "2": []}
    assert summary.mapped_component_ids == []
    assert summary.estimated_max_parallel_requests == 2
    assert summary.max_parallel_requests_upper_bound == 2


def test_rewrite_reports_mapped_parallelism_upper_bound(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Mapped components should warm only the single container needed for one in-process mapped run."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
        max_containers=5,
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "PromptList": _FakePromptListNode,
                "ModalMapInput": _FakeModalMapInputNode,
                "RemoteStringEcho": _FakeRemoteStringEchoNode,
                "LocalStringSink": _FakeLocalStringSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {"class_type": "PromptList", "inputs": {}, "_meta": {"title": "Prompt List"}},
        "2": {"class_type": "ModalMapInput", "inputs": {"value": ["1", 0]}, "_meta": {"title": "Map"}},
        "3": {"class_type": "RemoteStringEcho", "inputs": {"text": ["2", 0]}, "_meta": {"title": "Echo"}},
        "4": {"class_type": "LocalStringSink", "inputs": {"text": ["3", 0]}, "_meta": {"title": "Sink"}},
    }

    _rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert summary.component_execution_stages == [["2"]]
    assert summary.mapped_component_ids == ["2"]
    assert summary.estimated_max_parallel_requests == 1
    assert summary.max_parallel_requests_upper_bound == 1


def test_rewrite_uses_one_request_wide_volume_reload_marker_across_components(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """All components in one rewritten prompt should share one reload marker and decision."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteImage": _FakeRemoteImageNode,
                "RemoteImageConsumer": _FakeRemoteImageConsumerNode,
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
            "class_type": "RemoteImage",
            "inputs": {},
            "_meta": {"title": "Remote Image"},
        },
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 0]},
            "_meta": {"title": "Remote Image Consumer"},
        },
        "3": {
            "class_type": "LocalSink",
            "inputs": {"image": ["2", 0]},
            "_meta": {"title": "Local Sink"},
        },
    }

    uploaded_asset = sync_engine_module.SyncedAsset(
        local_path=tmp_path / "uploaded.bin",
        remote_path="/assets/uploaded.bin",
        sha256="uploaded",
        uploaded=True,
    )

    def fake_sync_component_prompt_inputs(
        *,
        component: Any,
        rewritten_prompt: dict[str, Any],
        sync_engine: Any,
        status_callback: Any = None,
    ) -> tuple[dict[str, Any], list[Any]]:
        if component.representative_node_id == "1":
            return {"1": rewritten_prompt["1"]}, []
        return {
            "2": {
                "class_type": rewritten_prompt["2"]["class_type"],
                "inputs": {"image_path": uploaded_asset.remote_path},
                "_meta": rewritten_prompt["2"]["_meta"],
            }
        }, [uploaded_asset]

    monkeypatch.setattr(
        api_intercept_module,
        "_sync_component_prompt_inputs",
        fake_sync_component_prompt_inputs,
    )

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    first_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    second_payload = rewritten_prompt["2"]["inputs"]["original_node_data"]

    assert summary.remote_component_ids == ["1", "2"]
    assert summary.synced_assets == [uploaded_asset]
    assert "requires_volume_reload" not in first_payload
    assert "requires_volume_reload" not in second_payload
    assert "volume_reload_marker" not in first_payload
    assert "volume_reload_marker" not in second_payload
    assert "uploaded_volume_paths" not in first_payload
    assert "uploaded_volume_paths" not in second_payload
    assert summary.requires_volume_reload is True
    assert isinstance(summary.volume_reload_marker, str)
    assert summary.volume_reload_marker
    assert summary.uploaded_volume_paths == [uploaded_asset.remote_path]


def test_rewrite_merges_cyclic_coarse_components_back_into_single_proxy(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A cyclic quotient between coarse groups should collapse back into one remote proxy."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteModelAndImage": _FakeRemoteModelAndImageNode,
                "RemoteImageConsumer": _FakeRemoteImageConsumerNode,
                "RemoteModelAndImageConsumer": _FakeRemoteModelAndImageConsumerNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteModelAndImage",
            "inputs": {},
            "_meta": {"title": "Remote Model And Image"},
        },
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 1]},
            "_meta": {"title": "Remote Image Consumer"},
        },
        "3": {
            "class_type": "RemoteModelAndImageConsumer",
            "inputs": {"model": ["1", 0], "image": ["2", 0]},
            "_meta": {"title": "Remote Model And Image Consumer"},
        },
        "4": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["3", 0]},
            "_meta": {"title": "Preview"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", "4"}
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {"1": ["1", "2", "3"]}
    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["component_node_ids"] == ["1", "2", "3"]
    assert payload["boundary_inputs"] == []
    assert payload["boundary_outputs"] == [
        {
            "proxy_output_name": "3_image",
            "node_id": "3",
            "output_index": 0,
            "io_type": "IMAGE",
            "is_list": False,
            "preview_target_node_ids": ["4"],
        }
    ]
    assert rewritten_prompt["4"]["inputs"]["images"] == ["1", 0]


def test_rewrite_marks_modal_map_boundary_as_mapped_subgraph(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A remote component fed through ModalMapInput should rewrite to a mapped payload."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "PromptList": _FakePromptListNode,
                "ModalMapInput": _FakeModalMapInputNode,
                "RemoteStringEcho": _FakeRemoteStringEchoNode,
                "LocalStringSink": _FakeLocalStringSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 5, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "PromptList",
            "inputs": {},
            "_meta": {"title": "Prompt List"},
        },
        "2": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["1", 0]},
            "_meta": {"title": "Map Input"},
        },
        "3": {
            "class_type": "RemoteStringEcho",
            "inputs": {"text": ["2", 0]},
            "_meta": {"title": "Remote Echo"},
        },
        "5": {
            "class_type": "RemoteStringEcho",
            "inputs": {"text": ["3", 0]},
            "_meta": {"title": "Remote Echo 2"},
        },
        "4": {
            "class_type": "LocalStringSink",
            "inputs": {"text": ["5", 0]},
            "_meta": {"title": "Local Sink"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", "2", "4"}
    assert summary.remote_component_ids == ["2"]
    payload = rewritten_prompt["2"]["inputs"]["original_node_data"]
    assert payload["payload_kind"] == "mapped_subgraph"
    assert payload["component_node_ids"] == ["2", "3", "5"]
    assert payload["mapped_input"] == {
        "proxy_input_name": "remote_input_0",
        "io_type": "STRING",
    }
    assert payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "STRING",
            "targets": [{"node_id": "2", "input_name": "value"}],
        }
    ]
    assert payload["static_to_mapped_boundaries"] == []
    assert payload["static_phase"] == {
        "component_node_ids": [],
        "subgraph_prompt": {},
        "boundary_inputs": [],
        "boundary_outputs": [],
        "execute_node_ids": [],
    }
    assert payload["mapped_phase"] == {
        "component_node_ids": ["2", "3", "5"],
        "subgraph_prompt": {
            "2": prompt["2"],
            "3": prompt["3"],
            "5": prompt["5"],
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "STRING",
                "targets": [{"node_id": "2", "input_name": "value"}],
            }
        ],
        "boundary_outputs": [
            {
                "proxy_output_name": "5_text",
                "node_id": "5",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "preview_target_node_ids": [],
                "mapped_output": True,
            }
        ],
        "execute_node_ids": ["5"],
    }
    assert rewritten_prompt["4"]["inputs"]["text"] == ["2", 0]


def test_rewrite_marks_local_modal_map_source_as_mapped_subgraph(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A local ModalMapInput feeding a remote node should still rewrite to mapped remote execution."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "PromptList": _FakePromptListNode,
                "ModalMapInput": _FakeModalMapInputNode,
                "RemoteStringEcho": _FakeRemoteStringEchoNode,
                "LocalStringSink": _FakeLocalStringSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "PromptList",
            "inputs": {},
            "_meta": {"title": "Prompt List"},
        },
        "2": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["1", 0]},
            "_meta": {"title": "Map Input"},
        },
        "3": {
            "class_type": "RemoteStringEcho",
            "inputs": {"text": ["2", 0]},
            "_meta": {"title": "Remote Echo"},
        },
        "4": {
            "class_type": "LocalStringSink",
            "inputs": {"text": ["3", 0]},
            "_meta": {"title": "Local Sink"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", "2", "3", "4"}
    assert summary.remote_component_ids == ["3"]
    payload = rewritten_prompt["3"]["inputs"]["original_node_data"]
    assert payload["payload_kind"] == "mapped_subgraph"
    assert payload["component_node_ids"] == ["3"]
    assert payload["mapped_input"] == {
        "proxy_input_name": "remote_input_0",
        "io_type": "STRING",
    }
    assert payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "*",
            "targets": [{"node_id": "3", "input_name": "text"}],
        }
    ]
    assert payload["mapped_phase"] == {
        "component_node_ids": ["3"],
        "subgraph_prompt": {
            "3": prompt["3"],
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "*",
                "targets": [{"node_id": "3", "input_name": "text"}],
            }
        ],
        "boundary_outputs": [
            {
                "proxy_output_name": "3_text",
                "node_id": "3",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "preview_target_node_ids": [],
                "mapped_output": True,
            }
        ],
        "execute_node_ids": ["3"],
    }
    with modal_executor_module._MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        warmup_context = modal_executor_module._MODAL_MAP_WARMUP_CONTEXTS["2"]
    assert warmup_context.mapped_io_type == "STRING"
    assert warmup_context.execution_payload["component_id"] == "3"
    assert rewritten_prompt["4"]["inputs"]["text"] == ["3", 0]


def test_rewrite_supports_mapped_branch_that_shares_non_transportable_upstream_with_unmapped_sibling(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Mapped execution should separate static and per-item execute targets within one coarse component."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteModel": _FakeRemoteModelNode,
                "RemoteSampler": _FakeRemoteSamplerNode,
                "LatentSource": _FakeLatentSourceNode,
                "ModalMapInput": _FakeModalMapInputNode,
                "LocalSink": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
            {"id": 5, "properties": {"is_modal_remote": False}},
            {"id": 6, "properties": {"is_modal_remote": True}},
            {"id": 7, "properties": {"is_modal_remote": True}},
            {"id": 8, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {},
            "_meta": {"title": "Shared Model"},
        },
        "2": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Single Latent"},
        },
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["2", 0]},
            "_meta": {"title": "Unmapped Sampler"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"image": ["3", 0]},
            "_meta": {"title": "Local Sink 1"},
        },
        "5": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Batch Latent Source"},
        },
        "6": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["5", 0]},
            "_meta": {"title": "Map Input"},
        },
        "7": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["6", 0]},
            "_meta": {"title": "Mapped Sampler"},
        },
        "8": {
            "class_type": "LocalSink",
            "inputs": {"image": ["7", 0]},
            "_meta": {"title": "Local Sink 2"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", "1__mapped", "2", "4", "5", "8"}
    assert summary.remote_component_ids == ["1", "1__mapped"]
    assert summary.component_node_ids_by_representative == {
        "1": ["1", "3"],
        "1__mapped": ["6", "7"],
    }
    assert summary.component_dependency_ids_by_representative == {
        "1": [],
        "1__mapped": ["1"],
    }
    assert summary.component_execution_stages == [["1"], ["1__mapped"]]
    assert summary.mapped_component_ids == ["1__mapped"]

    static_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    mapped_payload = rewritten_prompt["1__mapped"]["inputs"]["original_node_data"]
    static_execution_payload = modal_executor_module._rehydrate_proxy_payload(
        static_payload,
        unique_id="1",
    )
    mapped_execution_payload = modal_executor_module._rehydrate_proxy_payload(
        mapped_payload,
        unique_id="1__mapped",
    )

    assert static_payload["payload_kind"] == "subgraph"
    assert static_payload["component_node_ids"] == ["1", "3"]
    assert static_payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "LATENT",
            "targets": [{"node_id": "3", "input_name": "latent"}],
        }
    ]
    assert static_payload["boundary_outputs"] == [
        {
            "proxy_output_name": "3_latent",
            "node_id": "3",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
            "preview_target_node_ids": [],
        },
        {
            "proxy_output_name": "static_input_0",
            "node_id": "1",
            "output_index": 0,
            "io_type": "MODEL",
            "is_list": False,
            "preview_target_node_ids": [],
            "session_output": True,
        },
    ]
    assert static_payload["execute_node_ids"] == ["1", "3"]
    assert "remote_session" not in static_payload
    assert static_execution_payload["remote_session"]["owner_component_id"] == "1"

    assert mapped_payload["payload_kind"] == "subgraph"
    assert mapped_payload["component_node_ids"] == ["6", "7"]
    assert mapped_payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_1",
            "io_type": "LATENT",
            "targets": [{"node_id": "6", "input_name": "value"}],
        },
        {
            "proxy_input_name": "static_input_0",
            "io_type": "MODEL",
            "targets": [{"node_id": "7", "input_name": "model"}],
            "source_signature": api_intercept_module._boundary_source_signature(
                prompt,
                api_intercept_module.LinkedOutputRef(node_id="1", output_index=0),
            ),
        },
    ]
    assert mapped_payload["boundary_outputs"] == [
        {
            "proxy_output_name": "7_latent",
            "node_id": "7",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
            "preview_target_node_ids": [],
        }
    ]
    assert mapped_payload["execute_node_ids"] == ["7"]
    assert "clear_remote_session" not in mapped_payload
    assert mapped_payload["mapped_progress_display_node_id"] == "1"
    assert mapped_execution_payload["clear_remote_session"] is True
    assert (
        mapped_execution_payload["remote_session"]["session_id"]
        == static_execution_payload["remote_session"]["session_id"]
    )
    assert rewritten_prompt["1__mapped"]["inputs"]["static_input_0"] == ["1", 1]
    assert rewritten_prompt["4"]["inputs"]["image"] == ["1", 0]
    assert rewritten_prompt["8"]["inputs"]["image"] == ["1__mapped", 0]


def test_rewrite_stamps_snapshot_profile_on_split_static_and_mapped_payloads(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Split static and mapped payloads should inherit the same loader snapshot profile."""
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=True,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "CheckpointLoaderSimple": _FakeCheckpointLoaderSimpleNode,
                "RemoteSampler": _FakeRemoteSamplerNode,
                "LatentSource": _FakeLatentSourceNode,
                "ModalMapInput": _FakeModalMapInputNode,
                "LocalSink": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
            {"id": 5, "properties": {"is_modal_remote": False}},
            {"id": 6, "properties": {"is_modal_remote": True}},
            {"id": 7, "properties": {"is_modal_remote": True}},
            {"id": 8, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base.safetensors"},
            "_meta": {"title": "Checkpoint"},
        },
        "2": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Single Latent"},
        },
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["2", 0]},
            "_meta": {"title": "Unmapped Sampler"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"image": ["3", 0]},
            "_meta": {"title": "Local Sink 1"},
        },
        "5": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Batch Latent Source"},
        },
        "6": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["5", 0]},
            "_meta": {"title": "Map Input"},
        },
        "7": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["6", 0]},
            "_meta": {"title": "Mapped Sampler"},
        },
        "8": {
            "class_type": "LocalSink",
            "inputs": {"image": ["7", 0]},
            "_meta": {"title": "Local Sink 2"},
        },
    }

    rewritten_prompt, _ = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    static_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    mapped_payload = rewritten_prompt["1__mapped"]["inputs"]["original_node_data"]

    assert static_payload["snapshot_profile_key"].startswith("loader-profile:")
    assert mapped_payload["snapshot_profile_key"] == static_payload["snapshot_profile_key"]


def test_rewrite_splits_unmapped_remote_siblings_that_share_non_transportable_upstream(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Unmapped remote execute siblings should become ordered proxies when they only share remote-only upstream state."""
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
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteModel": _FakeRemoteModelNode,
                "RemoteSampler": _FakeRemoteSamplerNode,
                "LatentSource": _FakeLatentSourceNode,
                "LocalSink": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
            {"id": 5, "properties": {"is_modal_remote": False}},
            {"id": 6, "properties": {"is_modal_remote": True}},
            {"id": 7, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {},
            "_meta": {"title": "Shared Model"},
        },
        "2": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Latent A"},
        },
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["2", 0]},
            "_meta": {"title": "Sampler A"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"image": ["3", 0]},
            "_meta": {"title": "Sink A"},
        },
        "5": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Latent B"},
        },
        "6": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["5", 0]},
            "_meta": {"title": "Sampler B"},
        },
        "7": {
            "class_type": "LocalSink",
            "inputs": {"image": ["6", 0]},
            "_meta": {"title": "Sink B"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"2", "3", "4", "5", "6", "7"}
    assert summary.remote_component_ids == ["3", "6"]
    assert summary.component_node_ids_by_representative == {
        "3": ["1", "3"],
        "6": ["6"],
    }
    assert summary.component_dependency_ids_by_representative == {
        "3": [],
        "6": ["3"],
    }
    assert summary.component_execution_stages == [["3"], ["6"]]
    assert summary.rewritten_node_id_map["1"] == "3"
    assert summary.rewritten_node_id_map["3"] == "3"
    assert summary.rewritten_node_id_map["6"] == "6"

    first_payload = rewritten_prompt["3"]["inputs"]["original_node_data"]
    second_payload = rewritten_prompt["6"]["inputs"]["original_node_data"]
    first_execution_payload = modal_executor_module._rehydrate_proxy_payload(
        first_payload,
        unique_id="3",
    )
    second_execution_payload = modal_executor_module._rehydrate_proxy_payload(
        second_payload,
        unique_id="6",
    )

    assert first_payload["payload_kind"] == "subgraph"
    assert first_payload["component_node_ids"] == ["1", "3"]
    assert first_payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "LATENT",
            "targets": [{"node_id": "3", "input_name": "latent"}],
        }
    ]
    assert first_payload["boundary_outputs"][0] == {
        "proxy_output_name": "3_latent",
        "node_id": "3",
        "output_index": 0,
        "io_type": "LATENT",
        "is_list": False,
        "preview_target_node_ids": [],
    }
    assert first_payload["boundary_outputs"][1]["node_id"] == "1"
    assert first_payload["boundary_outputs"][1]["output_index"] == 0
    assert first_payload["boundary_outputs"][1]["io_type"] == "MODEL"
    assert first_payload["boundary_outputs"][1]["session_output"] is True
    assert first_payload["execute_node_ids"] == ["3"]
    assert "remote_session" not in first_payload
    assert first_execution_payload["remote_session"]["owner_component_id"] == "1"

    assert second_payload["payload_kind"] == "subgraph"
    assert second_payload["component_node_ids"] == ["6"]
    assert second_payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_1",
            "io_type": "LATENT",
            "targets": [{"node_id": "6", "input_name": "latent"}],
        },
        {
            "proxy_input_name": first_payload["boundary_outputs"][1]["proxy_output_name"],
            "io_type": "MODEL",
            "targets": [{"node_id": "6", "input_name": "model"}],
            "source_signature": api_intercept_module._boundary_source_signature(
                prompt,
                api_intercept_module.LinkedOutputRef(node_id="1", output_index=0),
            ),
        },
    ]
    assert second_payload["boundary_outputs"] == [
        {
            "proxy_output_name": "6_latent",
            "node_id": "6",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
            "preview_target_node_ids": [],
        }
    ]
    assert second_payload["execute_node_ids"] == ["6"]
    assert "clear_remote_session" not in second_payload
    assert second_execution_payload["clear_remote_session"] is True
    assert (
        second_execution_payload["remote_session"]["session_id"]
        == first_execution_payload["remote_session"]["session_id"]
    )
    assert rewritten_prompt["6"]["inputs"][first_payload["boundary_outputs"][1]["proxy_output_name"]] == ["3", 1]
    assert rewritten_prompt["4"]["inputs"]["image"] == ["3", 0]
    assert rewritten_prompt["7"]["inputs"]["image"] == ["6", 0]


def test_boundary_source_signature_changes_with_upstream_prompt_structure(
    api_intercept_module: Any,
) -> None:
    """Non-transportable boundary provenance should change when the upstream prompt changes."""
    source = api_intercept_module.LinkedOutputRef(node_id="2", output_index=0)
    base_prompt = {
        "1": {
            "class_type": "CheckpointLoader",
            "inputs": {"ckpt_name": "base.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {"model": ["1", 0], "strength_model": 0.8},
        },
    }
    changed_prompt = {
        "1": {
            "class_type": "CheckpointLoader",
            "inputs": {"ckpt_name": "base.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {"model": ["1", 0], "strength_model": 0.5},
        },
    }

    first_signature = api_intercept_module._boundary_source_signature(base_prompt, source)
    second_signature = api_intercept_module._boundary_source_signature(base_prompt, source)
    changed_signature = api_intercept_module._boundary_source_signature(changed_prompt, source)

    assert first_signature == second_signature
    assert changed_signature != first_signature


def test_extract_remote_node_ids_recurses_into_nested_subgraph_workflows(
    api_intercept_module: Any,
    settings_module: Any,
) -> None:
    """Modal marker extraction should find nodes nested inside saved subgraph metadata."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )

    workflow = {
        "nodes": [
            {
                "id": 100,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 11, "properties": {"is_modal_remote": True}},
                        {"id": 12, "properties": {"is_modal_remote": False}},
                    ]
                },
            }
        ]
    }

    assert api_intercept_module.extract_remote_node_ids(workflow, settings) == {"11"}
    assert api_intercept_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"100"},
    ) == {"100"}
    assert api_intercept_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"100:11"},
    ) == {"100:11"}


def test_extract_remote_node_ids_maps_subgraph_container_to_descendant_prompt_nodes(
    api_intercept_module: Any,
    settings_module: Any,
) -> None:
    """A marked subgraph container should remote its expanded descendant prompt nodes."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )

    workflow = {
        "nodes": [
            {
                "id": 24,
                "properties": {"is_modal_remote": True},
                "subgraph": {
                    "nodes": [
                        {"id": 23, "properties": {"is_modal_remote": False}},
                        {"id": 25, "properties": {"is_modal_remote": False}},
                    ]
                },
            }
        ]
    }

    assert api_intercept_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"24:23", "24:25", "99"},
    ) == {"24:23", "24:25"}


def test_rewrite_rejects_non_transportable_remote_inputs(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Remote nodes should absorb a single non-transportable upstream dependency automatically."""
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

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert list(rewritten_prompt) == ["1"]
    assert summary.remote_node_ids == ["1", "2"]
    assert summary.remote_component_ids == ["1"]


def test_rewrite_detects_remote_marker_inside_nested_subgraph_workflow(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Prompt rewrite should honor Modal markers found inside nested subgraph metadata."""
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
                "LocalConsumer": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {
                "id": 99,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 1, "properties": {"is_modal_remote": False}},
                        {"id": 2, "properties": {"is_modal_remote": True}},
                    ]
                },
            },
            {"id": 3, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "99": {
            "class_type": "RemoteConsumer",
            "inputs": {},
            "_meta": {"title": "Subgraph Container"},
        },
        "4": {
            "class_type": "LocalConsumer",
            "inputs": {"latent": ["99", 0]},
            "_meta": {"title": "Local Consumer"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert list(rewritten_prompt) == ["99", "4"]
    assert rewritten_prompt["4"]["inputs"]["latent"] == ["99", 0]
    assert summary.remote_node_ids == ["99"]
    assert summary.remote_component_ids == ["99"]


def test_rewrite_detects_marked_inner_subgraph_prompt_node_ids(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A marked nested workflow node should resolve to its composed prompt id."""
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
                "RemoteConsumer": _FakeRemoteSamplerNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {
                "id": 24,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 23, "properties": {"is_modal_remote": True}},
                    ]
                },
            },
            {"id": 30, "properties": {"is_modal_remote": True}},
        ]
    }
    prompt = {
        "30": {
            "class_type": "RemoteClip",
            "inputs": {},
            "_meta": {"title": "Remote VAE Source"},
        },
        "24:23": {
            "class_type": "RemoteConsumer",
            "inputs": {"clip": ["30", 0]},
            "_meta": {"title": "Nested Remote Consumer"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert list(rewritten_prompt) == ["24:23"]
    assert summary.remote_node_ids == ["24:23", "30"]
    assert summary.remote_component_ids == ["24:23"]


def test_rewrite_auto_expands_upstream_non_transportable_dependencies(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Marked remote nodes should absorb upstream non-transportable producers automatically."""
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
                "ModelSource": _FakeRemoteModelNode,
                "ConditioningSource": _FakeRemoteConditioningNode,
                "RemoteConsumer": _FakeRemoteSamplerNode,
                "LocalConsumer": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "ModelSource",
            "inputs": {},
            "_meta": {"title": "Model Source"},
        },
        "2": {
            "class_type": "ConditioningSource",
            "inputs": {},
            "_meta": {"title": "Conditioning Source"},
        },
        "3": {
            "class_type": "RemoteConsumer",
            "inputs": {
                "model": ["1", 0],
                "conditioning": ["2", 0],
            },
            "_meta": {"title": "Remote Consumer"},
        },
        "4": {
            "class_type": "LocalConsumer",
            "inputs": {"latent": ["3", 0]},
            "_meta": {"title": "Local Consumer"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", "4"}
    assert summary.remote_node_ids == ["1", "2", "3"]
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {"1": ["1", "2", "3"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "1", "3": "1"}
    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["boundary_inputs"] == []
    assert payload["execute_node_ids"] == ["3"]
    assert rewritten_prompt["4"]["inputs"]["latent"] == ["1", 0]


def test_analyze_remote_node_selection_returns_nodes_to_mark_and_reasons(
    api_intercept_module: Any,
    settings_module: Any,
) -> None:
    """Dry-run analysis should surface the clicked node plus required upstream nodes."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "ModelSource": _FakeRemoteModelNode,
                "ConditioningSource": _FakeRemoteConditioningNode,
                "RemoteConsumer": _FakeRemoteSamplerNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {"class_type": "ModelSource", "inputs": {}, "_meta": {"title": "Model"}},
        "2": {
            "class_type": "ConditioningSource",
            "inputs": {},
            "_meta": {"title": "Conditioning"},
        },
        "3": {
            "class_type": "RemoteConsumer",
            "inputs": {"model": ["1", 0], "conditioning": ["2", 0]},
            "_meta": {"title": "Remote Consumer"},
        },
    }

    analysis = api_intercept_module.analyze_remote_node_selection(
        prompt=prompt,
        workflow=workflow,
        seed_workflow_node_paths=["3"],
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert analysis.requested_node_ids == ["3"]
    assert analysis.requested_workflow_node_paths == ["3"]
    assert analysis.current_remote_node_ids == []
    assert analysis.current_remote_workflow_node_paths == []
    assert analysis.resolved_remote_node_ids == ["1", "2", "3"]
    assert analysis.resolved_workflow_node_paths == ["1", "2", "3"]
    assert analysis.added_node_ids == ["1", "2", "3"]
    assert analysis.added_workflow_node_paths == ["1", "2", "3"]
    assert [(reason.node_id, reason.required_by_node_id) for reason in analysis.reasons] == [
        ("1", "3"),
        ("2", "3"),
    ]


def test_analyze_remote_node_selection_prefers_nested_workflow_paths(
    api_intercept_module: Any,
    settings_module: Any,
) -> None:
    """Nested prompt ids should map back to the specific inner workflow node path."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "ModelSource": _FakeRemoteModelNode,
                "RemoteConsumer": _FakeRemoteSamplerNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {
                "id": 24,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 23, "properties": {"is_modal_remote": True}},
                    ]
                },
            },
            {"id": 30, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "30": {"class_type": "ModelSource", "inputs": {}, "_meta": {"title": "Model"}},
        "24:23": {
            "class_type": "RemoteConsumer",
            "inputs": {"model": ["30", 0]},
            "_meta": {"title": "Nested Consumer"},
        },
    }

    analysis = api_intercept_module.analyze_remote_node_selection(
        prompt=prompt,
        workflow=workflow,
        seed_workflow_node_paths=["24:23"],
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert analysis.requested_node_ids == ["24:23"]
    assert analysis.current_remote_node_ids == ["24:23"]
    assert analysis.current_remote_workflow_node_paths == ["24:23"]
    assert analysis.resolved_remote_node_ids == ["24:23", "30"]
    assert analysis.resolved_workflow_node_paths == ["24:23", "30"]
    assert analysis.added_node_ids == ["30"]
    assert analysis.added_workflow_node_paths == ["30"]
    assert [(reason.node_id, reason.required_by_node_id) for reason in analysis.reasons] == [
        ("30", "24:23"),
    ]


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


def test_extract_remote_node_ids_prefers_nested_prompt_id_over_colliding_root_id(
    api_intercept_module: Any,
    settings_module: Any,
) -> None:
    """Nested Modal markers should resolve to their composed prompt ids when root ids collide."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )

    workflow = {
        "nodes": [
            {"id": 27, "properties": {"is_modal_remote": False}},
            {
                "id": 195,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 27, "properties": {"is_modal_remote": True}},
                    ]
                },
            },
        ]
    }

    assert api_intercept_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"27", "195:27", "222", "223"},
    ) == {"195:27"}


def test_rewrite_keeps_nested_remote_nodes_remote_when_root_ids_collide(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Nested remote markers should survive prompt-id collisions with root workflow nodes."""
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
                "RemoteConsumer": _FakeRemoteSamplerNode,
                "LocalConsumer": _FakeLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 27, "properties": {"is_modal_remote": False}},
            {"id": 222, "properties": {"is_modal_remote": True}},
            {
                "id": 195,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 27, "properties": {"is_modal_remote": True}},
                    ]
                },
            },
            {"id": 223, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "27": {
            "class_type": "LocalConsumer",
            "inputs": {},
            "_meta": {"title": "Root Local Consumer"},
        },
        "222": {
            "class_type": "RemoteClip",
            "inputs": {},
            "_meta": {"title": "Remote Clip Source"},
        },
        "195:27": {
            "class_type": "RemoteConsumer",
            "inputs": {"clip": ["222", 0]},
            "_meta": {"title": "Nested Remote Consumer"},
        },
        "223": {
            "class_type": "LocalConsumer",
            "inputs": {"latent": ["195:27", 0]},
            "_meta": {"title": "Local Sink"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"27", "195:27", "223"}
    assert summary.remote_node_ids == ["195:27", "222"]
    assert summary.remote_component_ids == ["195:27"]


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
        phase="executing",
        client_id="client-1",
        prompt_id="prompt-1",
        node_ids=["4", "5"],
        component_node_ids_by_representative={"4": ["4", "5"]},
        active_node_id="5",
        active_node_class_type="KSampler",
        active_node_role="sampling",
    )

    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["4", "5"],
                "active_node_id": "5",
                "active_node_class_type": "KSampler",
                "active_node_role": "sampling",
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
