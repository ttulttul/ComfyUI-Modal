"""Tests for core prompt rewriting behavior."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_rewrite_remote_mode_rejects_local_sync_backend(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Remote execution must not queue payloads whose synced assets only exist in local mirror storage."""
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="remote",
        sync_custom_nodes=True,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine(
        volume=sync_engine_module.LocalMirrorVolume(settings.local_storage_root),
        settings=settings,
    )
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"RemoteImage": _FakeRemoteImageNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    with pytest.raises(api_intercept_module.ModalPromptValidationError) as exc_info:
        api_intercept_module.rewrite_prompt_for_modal(
            prompt={"1": {"class_type": "RemoteImage", "inputs": {}}},
            workflow={"nodes": [{"id": 1, "properties": {"is_modal_remote": True}}]},
            sync_engine=sync_engine,
            settings=settings,
            nodes_module=fake_nodes_module,
        )

    assert "requires asset sync to use the Modal volume backend" in str(exc_info.value)

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

    assert set(rewritten_prompt) == {"1", "3", _artifact_finalizer_node_id(summary)}
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

def test_rewrite_anchors_terminal_artifact_only_remote_node(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A terminal remote side-effect node should remain executable through the finalizer."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()
    (custom_nodes_dir / "__init__.py").write_text(
        "NODE_CLASS_MAPPINGS = {}\n",
        encoding="utf-8",
    )
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
                "RemoteArtifactWriter": _FakeRemoteArtifactWriterNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {"nodes": [{"id": 1, "properties": {"is_modal_remote": True}}]}
    prompt = {
        "1": {
            "class_type": "RemoteArtifactWriter",
            "inputs": {},
            "_meta": {"title": "Remote Artifact Writer"},
        }
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    finalizer_node_id = _artifact_finalizer_node_id(summary)
    assert set(rewritten_prompt) == {"1", finalizer_node_id}
    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["execute_node_ids"] == ["1"]
    assert payload["boundary_outputs"] == []

    proxy_class_type = rewritten_prompt["1"]["class_type"]
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_class_type]
    proxy_schema = proxy_class.GET_SCHEMA()
    assert [output.io_type for output in proxy_schema.outputs] == ["BOOLEAN"]
    assert [output.display_name for output in proxy_schema.outputs] == [
        modal_executor_module.MODAL_COMPONENT_COMPLETION_OUTPUT_NAME
    ]

    assert rewritten_prompt[finalizer_node_id] == {
        "class_type": modal_executor_module.MODAL_ARTIFACT_FINALIZER_NODE_ID,
        "inputs": {"components.component_0": ["1", 0]},
        "_meta": {"title": "Modal Artifact Finalizer"},
    }
    finalizer_class = fake_nodes_module.NODE_CLASS_MAPPINGS[
        modal_executor_module.MODAL_ARTIFACT_FINALIZER_NODE_ID
    ]
    assert finalizer_class.GET_SCHEMA().is_output_node is True
    assert finalizer_class.OUTPUT_NODE is True
    finalized_inputs, _hidden_inputs, _v3_data = (
        modal_executor_module.io.get_finalized_class_inputs(
            finalizer_class.INPUT_TYPES(),
            rewritten_prompt[finalizer_node_id]["inputs"],
        )
    )
    assert "components.component_0" in finalized_inputs["required"]

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

def test_rewrite_runs_terminal_save_video_as_remote_artifact_sink(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A terminal SaveVideo should encode remotely instead of importing raw VIDEO tensors."""
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
                "RemoteModel": _FakeRemoteModelNode,
                "RemoteSampler": _FakeRemoteSamplerNode,
                "VAELoader": _FakeVAELoaderNode,
                "VAEDecode": _FakeVAEDecodeNode,
                "VAEDecodeAudio": _FakeRemoteAudioNode,
                "CreateVideo": _FakeRemoteVideoNode,
                "SaveVideo": _FakeSaveVideoNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": node_id, "properties": {"is_modal_remote": node_id != 9}}
            for node_id in range(1, 10)
        ]
    }
    prompt = {
        "1": {"class_type": "RemoteModel", "inputs": {}},
        "2": {"class_type": "RemoteSampler", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "VAELoader", "inputs": {}},
        "4": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["2", 0], "vae": ["3", 0]},
        },
        "5": {"class_type": "VAELoader", "inputs": {}},
        "6": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["2", 0], "vae": ["5", 0]},
        },
        "7": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["4", 0], "audio": ["6", 0]},
        },
        "8": {"class_type": "SaveVideo", "inputs": {"video": ["7", 0]}},
        "9": {"class_type": "SaveVideo", "inputs": {"video": ["8", 0]}},
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert set(rewritten_prompt) == {"1", _artifact_finalizer_node_id(summary)}
    assert summary.remote_node_ids == [str(node_id) for node_id in range(1, 10)]
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {
        "1": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    }
    assert summary.rewritten_node_id_map == {
        str(node_id): "1"
        for node_id in range(1, 10)
    }
    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    assert payload["component_node_ids"] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    assert payload["execute_node_ids"] == ["8", "9"]
    assert payload["boundary_inputs"] == []
    assert payload["boundary_outputs"] == []

def test_rewrite_keeps_nonterminal_save_video_local(
    api_intercept_module: Any,
) -> None:
    """SaveVideo must stay local when its VIDEO output feeds additional local work."""
    prompt = {
        "1": {"class_type": "RemoteVideo", "inputs": {}},
        "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LocalVideoSink", "inputs": {"video": ["2", 0]}},
    }
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteVideo": _FakeRemoteVideoNode,
                "SaveVideo": _FakeSaveVideoNode,
                "LocalVideoSink": _FakeRemoteVideoNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    expanded = api_intercept_module._expand_remote_node_ids_for_terminal_video_sinks(
        prompt=prompt,
        remote_node_ids={"1"},
        nodes_module=fake_nodes_module,
    )

    assert expanded == {"1"}

def test_rewrite_allows_video_and_audio_across_remote_boundaries(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Current ComfyUI VIDEO and AUDIO values should pass boundary validation."""
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
                "CreateVideo": _FakeRemoteVideoNode,
                "LocalVideoSink": _FakeLocalSinkNode,
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
            "class_type": "CreateVideo",
            "inputs": {},
            "_meta": {"title": "Create Video"},
        },
        "2": {
            "class_type": "LocalVideoSink",
            "inputs": {"video": ["1", 0]},
            "_meta": {"title": "Local Video Sink"},
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
    assert payload["boundary_outputs"][0]["io_type"] == "VIDEO"
    assert rewritten_prompt["2"]["inputs"]["video"] == ["1", 0]
    assert summary.remote_node_ids == ["1"]
    assert api_intercept_module._is_transportable_output_type("AUDIO") is True

