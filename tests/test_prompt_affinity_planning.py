"""Tests for affinity prompt rewriting behavior."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

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

def test_rewrite_uses_one_request_wide_volume_reload_marker_across_components(
    api_intercept_module: Any,
    prompt_interception_module: Any,
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
                "RemoteString": _FakeRemoteStringEchoNode,
                "RemoteStringConsumer": _FakeRemoteStringEchoNode,
                "LocalSink": _FakeLocalStringSinkNode,
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
            "class_type": "RemoteString",
            "inputs": {},
            "_meta": {"title": "Remote String"},
        },
        "2": {
            "class_type": "RemoteStringConsumer",
            "inputs": {"text": ["1", 0]},
            "_meta": {"title": "Remote String Consumer"},
        },
        "3": {
            "class_type": "LocalSink",
            "inputs": {"text": ["2", 0]},
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
        request_cache: Any,
        status_callback: Any = None,
    ) -> tuple[dict[str, Any], list[Any]]:
        del sync_engine, request_cache, status_callback
        if component.representative_node_id == "1":
            return {"1": rewritten_prompt["1"]}, []
        return {
            "2": {
                "class_type": rewritten_prompt["2"]["class_type"],
                "inputs": {"text_path": uploaded_asset.remote_path},
                "_meta": rewritten_prompt["2"]["_meta"],
            }
        }, [uploaded_asset]

    monkeypatch.setattr(
        prompt_interception_module,
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

    assert set(rewritten_prompt) == {"1", "4", _artifact_finalizer_node_id(summary)}
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

def test_snapshot_profile_stamping_excludes_llm_phase_from_comfy_profile(
    api_intercept_module: Any,
    prompt_payload_metadata_module: Any,
) -> None:
    """A split LLM phase must not inherit the surrounding Comfy loader profile."""
    split_payload = {
        "split_proxy_payloads": [
            {
                "component_id": "251",
                "remote_worker_affinity_group": "comfy",
                "subgraph_prompt": {
                    "6": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": "minimax.safetensors"},
                    }
                },
            },
            {
                "component_id": "249:263",
                "remote_worker_affinity_group": "llm",
                "subgraph_prompt": {
                    "249:263": {"class_type": "ModalLLM", "inputs": {}}
                },
            },
            {
                "component_id": "172",
                "remote_worker_affinity_group": "comfy",
                "subgraph_prompt": {
                    "172": {"class_type": "SaveVideo", "inputs": {}}
                },
            },
        ]
    }
    settings = SimpleNamespace(
        enable_gpu_memory_snapshot=True,
        enable_loader_prewarm=True,
    )

    result = prompt_payload_metadata_module._attach_snapshot_profile_key(split_payload, settings)

    snapshot_profile_key = result["snapshot_profile_key"]
    phases = result["split_proxy_payloads"]
    assert snapshot_profile_key.startswith("loader-profile:")
    assert phases[0]["snapshot_profile_key"] == snapshot_profile_key
    assert "snapshot_profile_key" not in phases[1]
    assert phases[2]["snapshot_profile_key"] == snapshot_profile_key

