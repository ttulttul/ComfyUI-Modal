"""Tests for mapped prompt rewriting behavior."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_rewrite_splits_cyclic_remote_fanout_into_ordered_parallel_preview_phases(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Mixed local previews must not make coarse SCC merging reunify remote phases."""
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
                "ModalLLM": _FakeTextNode,
                "LocalSink": _FakeLocalSinkNode,
                "PreviewImage": _FakePreviewImageNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": node_id, "properties": {"is_modal_remote": node_id in {1, 2, 3, 4}}}
            for node_id in (1, 2, 3, 4, 5, 9, 10)
        ]
    }
    prompt = {
        "1": {"class_type": "RemoteImage", "inputs": {}},
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 0]},
        },
        "3": {
            "class_type": "ModalLLM",
            "inputs": {"image": ["2", 0]},
        },
        "4": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 0], "prompt": ["3", 0]},
        },
        "5": {"class_type": "LocalSink", "inputs": {"image": ["4", 0]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert summary.remote_component_ids == ["2", "3", "4"]
    assert summary.component_execution_stages == [["2"], ["3"], ["4"]]
    assert summary.component_node_ids_by_representative == {
        "2": ["1", "2"],
        "3": ["3"],
        "4": ["4"],
    }
    phase_payloads = [
        rewritten_prompt[phase_node_id]["inputs"]["original_node_data"]
        for phase_node_id in summary.remote_component_ids
    ]
    assert [payload["component_node_ids"] for payload in phase_payloads] == [
        ["1", "2"],
        ["3"],
        ["4"],
    ]
    assert [payload["remote_worker_affinity_group"] for payload in phase_payloads] == [
        "comfy",
        "llm",
        "comfy",
    ]
    assert len(summary.parallel_local_branch_node_ids) == 2
    materializer_node_ids = {
        node_id
        for node_id, prompt_node in rewritten_prompt.items()
        if prompt_node["class_type"]
        == api_intercept_module.MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID
    }
    assert len(materializer_node_ids) == 2
    assert rewritten_prompt["9"]["inputs"]["images"][0] in materializer_node_ids
    assert rewritten_prompt["10"]["inputs"]["images"][0] in materializer_node_ids
    assert rewritten_prompt["5"]["inputs"]["image"] == ["4", 0]

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

    assert set(rewritten_prompt) == {
        "1",
        "2",
        "4",
        _artifact_finalizer_node_id(summary),
    }
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
                "scheduler_is_list": True,
            }
        ],
        "execute_node_ids": ["5"],
    }
    assert rewritten_prompt["4"]["inputs"]["text"] == ["2", 0]

def test_rewrite_marks_local_modal_map_source_as_mapped_subgraph(
    api_intercept_module: Any,
    modal_executor_module: Any,
    proxy_payloads_module: Any,
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

    assert set(rewritten_prompt) == {
        "1",
        "2",
        "3",
        "4",
        _artifact_finalizer_node_id(summary),
    }
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
                "scheduler_is_list": True,
            }
        ],
        "execute_node_ids": ["3"],
    }
    with proxy_payloads_module._MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        warmup_context = proxy_payloads_module._MODAL_MAP_WARMUP_CONTEXTS["2"]
    assert warmup_context.mapped_io_type == "STRING"
    assert warmup_context.execution_payload["component_id"] == "3"
    assert rewritten_prompt["4"]["inputs"]["text"] == ["3", 0]

def test_rewrite_supports_mapped_branch_that_shares_non_transportable_upstream_with_unmapped_sibling(
    api_intercept_module: Any,
    prompt_payload_metadata_module: Any,
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

    assert set(rewritten_prompt) == {
        "1",
        "1__mapped",
        "2",
        "4",
        "5",
        "8",
        _artifact_finalizer_node_id(summary),
    }
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
            "source_signature": prompt_payload_metadata_module._boundary_source_signature(
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
    assert rewritten_prompt["1__mapped"]["inputs"]["remote_input_0"] == ["2", 0]
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
        modal_gpu="L40S",
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
    assert static_payload["modal_gpu"] == "L40S"
    assert mapped_payload["modal_gpu"] == "L40S"

def test_rewrite_keeps_unmapped_remote_siblings_without_local_reentry_together(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Ordinary remote execute siblings should remain one proxy without local re-entry."""
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

    assert set(rewritten_prompt) == {
        "1",
        "2",
        "4",
        "5",
        "7",
        _artifact_finalizer_node_id(summary),
    }
    assert summary.remote_component_ids == ["1"]
    assert summary.component_node_ids_by_representative == {
        "1": ["1", "3", "6"],
    }
    assert summary.component_dependency_ids_by_representative == {
        "1": [],
    }
    assert summary.component_execution_stages == [["1"]]
    assert summary.rewritten_node_id_map == {"1": "1", "3": "1", "6": "1"}

    payload = rewritten_prompt["1"]["inputs"]["original_node_data"]

    assert payload["payload_kind"] == "subgraph"
    assert payload["component_node_ids"] == ["1", "3", "6"]
    assert payload["boundary_inputs"] == [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "LATENT",
            "targets": [{"node_id": "3", "input_name": "latent"}],
        },
        {
            "proxy_input_name": "remote_input_1",
            "io_type": "LATENT",
            "targets": [{"node_id": "6", "input_name": "latent"}],
        },
    ]
    assert payload["boundary_outputs"] == [
        {
            "proxy_output_name": "3_latent",
            "node_id": "3",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
            "preview_target_node_ids": [],
        },
        {
            "proxy_output_name": "6_latent",
            "node_id": "6",
            "output_index": 0,
            "io_type": "LATENT",
            "is_list": False,
            "preview_target_node_ids": [],
        },
    ]
    assert payload["execute_node_ids"] == ["3", "6"]
    assert "remote_session" not in payload
    assert rewritten_prompt["4"]["inputs"]["image"] == ["1", 0]
    assert rewritten_prompt["7"]["inputs"]["image"] == ["1", 1]

