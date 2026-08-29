"""Tests for the workflow prompt mapping boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_extract_remote_node_ids_recurses_into_nested_subgraph_workflows(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
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

    assert remote_graph_analysis_module.extract_remote_node_ids(workflow, settings) == {"11"}
    assert remote_graph_analysis_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"100"},
    ) == {"100"}
    assert remote_graph_analysis_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"100:11"},
    ) == {"100:11"}

def test_extract_remote_node_ids_prefers_visible_toggle_over_stale_property(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
    settings_module: Any,
) -> None:
    """A restored disabled widget must prevent stale metadata from starting Modal."""
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
                "id": 9,
                "properties": {"is_modal_remote": True},
                "widgets_values_named": {"Run on Modal": False},
            },
            {
                "id": 10,
                "properties": {"is_modal_remote": False},
                "widgets_values_named": {"Run on Modal": True},
            },
        ]
    }

    assert remote_graph_analysis_module.extract_remote_node_ids(workflow, settings) == {"10"}

def test_extract_remote_node_ids_maps_subgraph_container_to_descendant_prompt_nodes(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
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

    assert remote_graph_analysis_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"24:23", "24:25", "99"},
    ) == {"24:23", "24:25"}

def test_extract_remote_node_ids_maps_defined_subgraph_nodes_through_instances(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
    settings_module: Any,
) -> None:
    """Markers in reusable subgraph definitions should map to every executable instance path."""
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
    subgraph_type = "4c314f31-ecda-4b08-ae98-faaba1bf613f"
    workflow = {
        "nodes": [
            {"id": 105, "type": subgraph_type, "properties": {"is_modal_remote": False}},
            {"id": 205, "type": subgraph_type, "properties": {"is_modal_remote": False}},
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": subgraph_type,
                    "nodes": [
                        {"id": 11, "type": "VAELoader", "properties": {"is_modal_remote": True}},
                        {
                            "id": 14,
                            "type": "SamplerCustomAdvanced",
                            "properties": {"is_modal_remote": True},
                        },
                        {"id": 107, "type": "ComfyMathExpression", "properties": {}},
                    ],
                }
            ]
        },
    }

    prompt_node_ids = {
        "105:11",
        "105:14",
        "105:107",
        "205:11",
        "205:14",
        "205:107",
        "300",
    }

    assert remote_graph_analysis_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids=prompt_node_ids,
    ) == {"105:11", "105:14", "205:11", "205:14"}
    assert api_intercept_module._extract_marked_workflow_node_paths(
        workflow,
        settings,
    ) == {"105:11", "105:14", "205:11", "205:14"}

def test_extract_remote_node_ids_maps_nested_defined_subgraph_instances(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
    settings_module: Any,
) -> None:
    """Nested reusable definitions should retain every instance ancestor in prompt ids."""
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
    outer_type = "outer-subgraph"
    inner_type = "inner-subgraph"
    workflow = {
        "nodes": [
            {"id": 105, "type": outer_type, "properties": {"is_modal_remote": False}},
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": outer_type,
                    "nodes": [
                        {"id": 7, "type": inner_type, "properties": {"is_modal_remote": False}},
                    ],
                },
                {
                    "id": inner_type,
                    "nodes": [
                        {"id": 11, "type": "VAELoader", "properties": {"is_modal_remote": True}},
                    ],
                },
            ]
        },
    }

    assert remote_graph_analysis_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"105:7:11", "300"},
    ) == {"105:7:11"}

def test_extract_remote_node_ids_prefers_nested_prompt_id_over_colliding_root_id(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
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

    assert remote_graph_analysis_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"27", "195:27", "222", "223"},
    ) == {"195:27"}

def test_workflow_ssh_metadata_preserves_probed_gpu_capabilities(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    execution_environments_module: Any,
    remote_configurations_module: Any,
    remote_hosts_module: Any,
) -> None:
    """Queued SSH workers must retain the GPU snapshot used for placement."""
    environment_module = execution_environments_module
    capabilities = environment_module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=48 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(
            environment_module.GpuCapability(
                "GPU-4090",
                "RTX 4090",
                24 * 1024**3,
            ),
        ),
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="lambda",
        display_name="Lambda",
        ssh_target="lambda",
        capabilities=capabilities,
        health=environment_module.EnvironmentHealth.READY,
        last_error="stale diagnostic",
    )
    configuration = remote_configurations_module.SshRemoteConfiguration(
        configuration_id="lambda",
        display_name="Lambda",
        host=host,
    )
    assignment = environment_module.ExecutionAssignment(
        environment_id="lambda",
        provider=environment_module.ExecutionProvider.SSH_DOCKER,
        predicted_cost_usd=0.0,
        predicted_completion_seconds=60.0,
        configuration_id="lambda",
    )
    execution_plan = api_intercept_module.ComponentExecutionPlan(
        assignments={"367": assignment},
        configurations_by_id={"lambda": configuration},
        ssh_hosts_by_id={"lambda": host},
    )

    metadata = execution_scheduling_module._configured_provider_metadata(
        execution_plan=execution_plan,
        assignment=assignment,
        vast_leases_by_environment={},
    )

    assert metadata is not None
    queued_host = remote_hosts_module.SshHostConfig.from_dict(
        metadata["ssh_host_config"]
    )
    assert queued_host.capabilities == capabilities
    assert queued_host.health is environment_module.EnvironmentHealth.UNKNOWN
    assert queued_host.last_error is None

