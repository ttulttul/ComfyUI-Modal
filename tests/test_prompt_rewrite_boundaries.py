"""Tests for boundaries prompt rewriting behavior."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

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

def test_rewrite_colocates_remote_chain_across_large_transportable_edges(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Large transportable remote-to-remote values should remain inside one component."""
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

    assert set(rewritten_prompt) == {"1", "3", _artifact_finalizer_node_id(summary)}
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {"1": ["1", "2"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "1"}

    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]

    assert payload["component_node_ids"] == ["1", "2"]
    assert payload["boundary_inputs"] == []
    assert payload["boundary_outputs"] == [
        {
            "proxy_output_name": "2_image",
            "node_id": "2",
            "output_index": 0,
            "io_type": "IMAGE",
            "is_list": False,
            "preview_target_node_ids": [],
        }
    ]
    assert payload["execute_node_ids"] == ["2"]
    assert rewritten_prompt["3"]["inputs"]["image"] == ["1", 0]
    assert api_intercept_module._is_inexpensive_remote_boundary_type("IMAGE") is False
    assert api_intercept_module._is_inexpensive_remote_boundary_type("STRING") is True

def test_rewrite_keeps_non_returning_local_preview_taps_local(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Local preview branches should stay local even when a remote chain continues."""
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
                "PreviewImage": _FakePreviewImageNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": False}},
            {"id": 9, "properties": {"is_modal_remote": False}},
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
        "9": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["1", 0]},
            "_meta": {"title": "Interim Preview"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    materializer_node_id = next(
        node_id
        for node_id, prompt_node in rewritten_prompt.items()
        if prompt_node["class_type"]
        == api_intercept_module.MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID
    )
    assert set(rewritten_prompt) == {
        "1",
        "2",
        "3",
        "9",
        materializer_node_id,
        *summary.parallel_local_branch_node_ids,
        _artifact_finalizer_node_id(summary),
    }
    assert summary.remote_component_ids == ["1", "2"]
    assert summary.component_execution_stages == [["1"], ["2"]]
    assert summary.component_node_ids_by_representative == {
        "1": ["1"],
        "2": ["2"],
    }
    assert summary.rewritten_node_id_map == {"1": "1", "2": "2"}

    remote_payloads = [
        rewritten_node["inputs"]["original_node_data"]
        for rewritten_node in rewritten_prompt.values()
        if isinstance(rewritten_node.get("inputs"), dict)
        and "original_node_data" in rewritten_node["inputs"]
    ]
    assert remote_payloads
    assert all("9" not in payload["component_node_ids"] for payload in remote_payloads)
    assert all("9" not in payload["subgraph_prompt"] for payload in remote_payloads)
    assert all("9" not in payload["execute_node_ids"] for payload in remote_payloads)
    producer_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert producer_payload["boundary_outputs"] == [
        {
            "proxy_output_name": "1_image",
            "node_id": "1",
            "output_index": 0,
            "io_type": "IMAGE",
            "is_list": False,
            "preview_target_node_ids": [],
            "session_output": True,
        },
    ]
    assert rewritten_prompt["9"]["inputs"]["images"] == [materializer_node_id, 0]
    assert rewritten_prompt["3"]["inputs"]["image"] == ["2", 0]

def test_rewrite_keeps_unmarked_preview_subgraph_nodes_local(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Unmarked preview producer nodes must not execute remotely."""
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
                "RemoteSampler": _FakeRemoteSamplerNode,
                "LocalSink": _FakeLocalSinkNode,
                "PreviewImage": _FakePreviewImageNode,
                "VAEDecode": _FakeVAEDecodeNode,
                "VAEEncode": _FakeVAEEncodeNode,
                "VAELoader": _FakeVAELoaderNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": False}},
            {"id": 7, "properties": {"is_modal_remote": False}},
            {"id": 8, "properties": {"is_modal_remote": False}},
            {"id": 9, "properties": {"is_modal_remote": False}},
            {"id": 11, "properties": {"is_modal_remote": False}},
            {"id": 90, "properties": {"is_modal_remote": False}},
            {"id": 192, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "9": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "vae.safetensors"},
            "_meta": {"title": "VAE Loader"},
        },
        "1": {
            "class_type": "RemoteSampler",
            "inputs": {},
            "_meta": {"title": "Remote Sampler 1"},
        },
        "192": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["1", 0], "vae": ["9", 0]},
            "_meta": {"title": "VAE Decode Preview"},
        },
        "8": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["192", 0], "vae": ["9", 0]},
            "_meta": {"title": "Local VAE Encode"},
        },
        "7": {
            "class_type": "LocalSink",
            "inputs": {"image": ["8", 0]},
            "_meta": {"title": "Local Encoded Sink"},
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["1", 0], "vae": ["9", 0]},
            "_meta": {"title": "Local VAE Decode"},
        },
        "90": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["192", 0]},
            "_meta": {"title": "Preview"},
        },
        "2": {
            "class_type": "RemoteSampler",
            "inputs": {"latent": ["1", 0]},
            "_meta": {"title": "Remote Sampler 2"},
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

    materializer_node_id = next(
        node_id
        for node_id, prompt_node in rewritten_prompt.items()
        if prompt_node["class_type"]
        == api_intercept_module.MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID
    )
    assert set(rewritten_prompt) == {
        "1",
        "2",
        "3",
        "7",
        "8",
        "9",
        "11",
        "90",
        "192",
        materializer_node_id,
        *summary.parallel_local_branch_node_ids,
        _artifact_finalizer_node_id(summary),
    }
    assert summary.remote_component_ids == ["1", "2"]
    assert summary.component_execution_stages == [["1"], ["2"]]
    assert summary.component_node_ids_by_representative == {
        "1": ["1"],
        "2": ["2"],
    }
    assert summary.rewritten_node_id_map == {"1": "1", "2": "2"}

    remote_payloads = [
        rewritten_node["inputs"]["original_node_data"]
        for rewritten_node in rewritten_prompt.values()
        if isinstance(rewritten_node.get("inputs"), dict)
        and "original_node_data" in rewritten_node["inputs"]
    ]
    local_node_ids = {"7", "8", "9", "11", "90", "192"}
    assert len(remote_payloads) == 2
    for payload in remote_payloads:
        assert not (local_node_ids & set(payload["component_node_ids"]))
        assert not (local_node_ids & set(payload["subgraph_prompt"]))
        assert not (local_node_ids & set(payload["execute_node_ids"]))
    assert rewritten_prompt["192"]["inputs"]["samples"] == [
        materializer_node_id,
        0,
    ]
    assert rewritten_prompt["11"]["inputs"]["samples"] == [
        materializer_node_id,
        0,
    ]
    assert rewritten_prompt["90"]["inputs"]["images"] == ["192", 0]
    assert rewritten_prompt["3"]["inputs"]["image"] == ["2", 0]

def test_rewrite_keeps_local_branches_that_feed_remote_as_boundaries(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Local branches that later feed remote work are dependencies, not preview taps."""
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
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteImage",
            "inputs": {},
            "_meta": {"title": "Remote Image"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"image": ["1", 0]},
            "_meta": {"title": "Local Transform"},
        },
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["4", 0]},
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

    assert set(rewritten_prompt) == {
        "1",
        "2",
        "3",
        "4",
        _artifact_finalizer_node_id(summary),
    }
    assert summary.remote_component_ids == ["1", "2"]
    assert summary.component_node_ids_by_representative == {"1": ["1"], "2": ["2"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "2"}
    assert summary.sandwiched_local_node_ids == ["4"]
    assert rewritten_prompt["4"]["inputs"]["image"] == ["1", 0]
    assert rewritten_prompt["2"]["inputs"]["remote_input_0"] == ["4", 0]
    assert rewritten_prompt["3"]["inputs"]["image"] == ["2", 0]

    second_payload = rewritten_prompt["2"]["inputs"]["original_node_data"]
    assert second_payload["component_node_ids"] == ["2"]
    assert second_payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "IMAGE",
            "targets": [{"node_id": "2", "input_name": "image"}],
        }
    ]

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

    assert list(rewritten_prompt) == ["1", _artifact_finalizer_node_id(summary)]
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

    assert list(rewritten_prompt) == ["99", "4", _artifact_finalizer_node_id(summary)]
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

    assert list(rewritten_prompt) == ["24:23", _artifact_finalizer_node_id(summary)]
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

    assert set(rewritten_prompt) == {"1", "4", _artifact_finalizer_node_id(summary)}
    assert summary.remote_node_ids == ["1", "2", "3"]
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {"1": ["1", "2", "3"]}
    assert summary.rewritten_node_id_map == {"1": "1", "2": "1", "3": "1"}
    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["boundary_inputs"] == []
    assert payload["execute_node_ids"] == ["3"]
    assert rewritten_prompt["4"]["inputs"]["latent"] == ["1", 0]

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
    assert "cannot cross the current component boundary" in message

def test_rewrite_keeps_remote_noise_producer_with_remote_sampler(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """NOISE strategy objects should remain inside one remote component."""
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
                "RandomNoise": _FakeRemoteNoiseNode,
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
        "1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "2": {
            "class_type": "RemoteSampler",
            "inputs": {"noise": ["1", 0]},
        },
        "3": {
            "class_type": "LocalSink",
            "inputs": {"latent": ["2", 0]},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["component_node_ids"] == ["1", "2"]
    assert set(payload["subgraph_prompt"]) == {"1", "2"}
    assert payload["boundary_outputs"][0]["io_type"] == "LATENT"
    assert rewritten_prompt["3"]["inputs"]["latent"] == ["1", 0]
    assert summary.remote_component_ids == ["1"]
    assert api_intercept_module._is_transportable_output_type("NOISE") is False

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

    assert set(rewritten_prompt) == {
        "27",
        "195:27",
        "223",
        _artifact_finalizer_node_id(summary),
    }
    assert summary.remote_node_ids == ["195:27", "222"]
    assert summary.remote_component_ids == ["195:27"]

