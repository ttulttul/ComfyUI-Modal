"""Tests for the remote graph analysis boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_remote_partition_preserves_dag_around_ssh_only_llm(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
) -> None:
    """Provider boundaries must not be undone by a coarse fanout cycle."""
    prompt = {
        "1": {"class_type": "RemoteImage", "inputs": {}},
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 0]},
        },
        "3": {
            "class_type": "ModalLLM",
            "inputs": {
                "image": ["2", 0],
                "model_profile": "huihui-qwen3.8-27b-abliterated-q2-k-gguf",
            },
        },
        "4": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["1", 0], "prompt": ["3", 0]},
        },
    }
    fake_nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteImage": _FakeRemoteImageNode,
            "RemoteImageConsumer": _FakeRemoteImageConsumerNode,
            "ModalLLM": _FakeTextNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    component_groups = remote_graph_analysis_module._remote_component_partition_groups(
        prompt,
        set(prompt),
        remote_graph_analysis_module._build_consumer_map(prompt),
        fake_nodes_module,
    )
    components = remote_graph_analysis_module._component_topological_order(
        prompt,
        component_groups,
    )

    assert components == [["1", "2"], ["3"], ["4"]]
    assert remote_graph_analysis_module._component_execution_stages(
        prompt,
        component_groups,
    ) == [["1"], ["3"], ["4"]]

def test_remote_partition_replicates_non_transportable_fanout_around_ssh_llm(
    api_intercept_module: Any,
    component_planning_module: Any,
    execution_scheduling_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A safe shared runtime producer should be rebuilt after an SSH-only phase."""
    prompt = {
        "1": {"class_type": "VAELoader", "inputs": {}},
        "2": {
            "class_type": "VAEDecode",
            "inputs": {"vae": ["1", 0]},
        },
        "3": {
            "class_type": "ModalLLM",
            "inputs": {
                "image": ["2", 0],
                "model_profile": "huihui-qwen3.8-27b-abliterated-q2-k-gguf",
            },
        },
        "4": {
            "class_type": "VAEDecode",
            "inputs": {"vae": ["1", 0], "prompt": ["3", 0]},
        },
    }
    pristine_prompt = json.loads(json.dumps(prompt))
    fake_nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "VAELoader": _FakeVAELoaderNode,
            "VAEDecode": _FakeVAEDecodeNode,
            "ModalLLM": _FakeTextNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    remote_node_ids = set(prompt)
    component_plans = component_planning_module._build_component_plans(
        prompt,
        remote_node_ids,
        fake_nodes_module,
    )
    component_planning_module.validate_remote_component_transport_compatibility(
        prompt,
        component_plans,
        fake_nodes_module,
    )
    assert [component.representative_node_id for component in component_plans] == [
        "1",
        "3",
        "4",
    ]
    replica_node_ids = {
        node_id
        for node_id in prompt
        if node_id.startswith(component_planning_module._REMOTE_REPLICA_NODE_PREFIX)
    }
    assert len(replica_node_ids) == 1
    replica_node_id = next(iter(replica_node_ids))
    assert prompt["4"]["inputs"]["vae"] == [replica_node_id, 0]
    assert prompt[replica_node_id] == prompt["1"]
    assert replica_node_id in component_plans[2].node_ids
    assert component_plans[0].boundary_inputs == []
    assert component_plans[0].boundary_outputs[0].io_type == "IMAGE"
    assert component_plans[2].boundary_inputs[0].io_type == "STRING"
    assert component_plans[2].boundary_outputs == []
    required_provider = execution_scheduling_module._component_required_provider(
        component_plans[1],
        prompt,
        {
            "huihui-qwen3.8-27b-abliterated-q2-k-gguf": SimpleNamespace(
                backend="llama_cpp_server"
            )
        },
    )
    assert required_provider.value == "ssh_docker"

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

    def assign_to_modal(*, components: list[Any], **_kwargs: Any) -> dict[str, Any]:
        """Keep the rewrite test independent of real provider availability."""
        return {
            component.representative_node_id: api_intercept_module.ExecutionAssignment(
                environment_id="modal:H200",
                provider=api_intercept_module.ExecutionProvider.MODAL,
                predicted_cost_usd=0.0,
                predicted_completion_seconds=1.0,
            )
            for component in components
        }

    monkeypatch.setattr(
        execution_scheduling_module,
        "_plan_component_execution_assignments",
        assign_to_modal,
    )
    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=pristine_prompt,
        workflow={
            "nodes": [
                {"id": node_id, "properties": {"is_modal_remote": True}}
                for node_id in range(1, 5)
            ]
        },
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert not any(
        node_id.startswith(component_planning_module._REMOTE_REPLICA_NODE_PREFIX)
        for node_id in rewritten_prompt
    )
    assert summary.component_node_ids_by_representative["4"] == ["4"]
    assert not any(
        node_id.startswith(component_planning_module._REMOTE_REPLICA_NODE_PREFIX)
        for node_id in summary.rewritten_node_id_map
    )
    downstream_payload = rewritten_prompt["4"]["inputs"]["original_node_data"]
    replica_payload_node_ids = {
        node_id
        for node_id in downstream_payload["subgraph_prompt"]
        if node_id.startswith(component_planning_module._REMOTE_REPLICA_NODE_PREFIX)
    }
    assert len(replica_payload_node_ids) == 1
    downstream_vae_input = downstream_payload["subgraph_prompt"]["4"]["inputs"][
        "vae"
    ]
    assert downstream_vae_input[0] in replica_payload_node_ids

def test_remote_partition_replicates_linked_model_loader_closure(
    api_intercept_module: Any,
    component_planning_module: Any,
) -> None:
    """A downstream provider phase should rebuild a linked loader chain, not sample."""
    prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "model.safetensors"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": "adapter.safetensors"},
        },
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["2", 0]},
        },
        "4": {
            "class_type": "ModalLLM",
            "inputs": {
                "image": ["3", 0],
                "model_profile": "huihui-qwen3.8-27b-abliterated-q2-k-gguf",
            },
        },
        "5": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["2", 0], "prompt": ["4", 0]},
        },
    }
    fake_nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "UNETLoader": _FakeRemoteModelNode,
            "LoraLoaderModelOnly": _FakeRemoteModelNode,
            "RemoteSampler": _FakeRemoteSamplerNode,
            "ModalLLM": _FakeTextNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    component_plans = component_planning_module._build_component_plans(
        prompt,
        set(prompt),
        fake_nodes_module,
    )

    assert [component.representative_node_id for component in component_plans] == [
        "1",
        "4",
        "5",
    ]
    replica_node_ids = sorted(
        node_id
        for node_id in prompt
        if node_id.startswith(component_planning_module._REMOTE_REPLICA_NODE_PREFIX)
    )
    assert len(replica_node_ids) == 2
    replica_loader_id = next(
        node_id
        for node_id in replica_node_ids
        if prompt[node_id]["class_type"] == "UNETLoader"
    )
    replica_lora_id = next(
        node_id
        for node_id in replica_node_ids
        if prompt[node_id]["class_type"] == "LoraLoaderModelOnly"
    )
    assert prompt[replica_lora_id]["inputs"]["model"] == [replica_loader_id, 0]
    assert prompt["5"]["inputs"]["model"] == [replica_lora_id, 0]
    assert replica_node_ids == sorted(
        set(component_plans[2].node_ids) - {"5"}
    )
    component_planning_module.validate_remote_component_transport_compatibility(
        prompt,
        component_plans,
        fake_nodes_module,
    )

def test_cross_provider_boundary_uses_transport_instead_of_remote_session(
    api_intercept_module: Any,
    component_planning_module: Any,
    execution_environments_module: Any,
) -> None:
    """Session-backed references must never cross provider storage boundaries."""
    source = api_intercept_module.LinkedOutputRef("1", 0)
    boundary_output = api_intercept_module.BoundaryOutputSpec(
        proxy_output_name="remote_output_0",
        source=source,
        io_type="IMAGE",
        is_list=False,
    )
    producer = api_intercept_module.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[boundary_output],
        execute_node_ids=["1"],
        contains_output_node=False,
    )
    consumer = api_intercept_module.RemoteComponentPlan(
        node_ids=["2"],
        representative_node_id="2",
        boundary_inputs=[
            api_intercept_module.BoundaryInputSpec(
                proxy_input_name="remote_input_0",
                source=source,
                io_type="IMAGE",
                targets=[api_intercept_module.InputTarget("2", "image")],
            )
        ],
        boundary_outputs=[],
        execute_node_ids=["2"],
        contains_output_node=False,
    )
    assignment_type = execution_environments_module.ExecutionAssignment
    provider_type = execution_environments_module.ExecutionProvider

    session_component_ids = (
        component_planning_module._mark_remote_to_remote_session_boundaries(
            {
                "1": {"class_type": "RemoteImage", "inputs": {}},
                "2": {
                    "class_type": "RemoteImageConsumer",
                    "inputs": {"image": ["1", 0]},
                },
            },
            [producer, consumer],
            SimpleNamespace(
                NODE_CLASS_MAPPINGS={
                    "RemoteImage": _FakeRemoteImageNode,
                    "RemoteImageConsumer": _FakeRemoteImageConsumerNode,
                }
            ),
            {
                "1": assignment_type(
                    "modal:H200",
                    provider_type.MODAL,
                    None,
                    0.0,
                ),
                "2": assignment_type(
                    "lambda",
                    provider_type.SSH_DOCKER,
                    0.0,
                    0.0,
                ),
            },
        )
    )

    assert session_component_ids == set()
    assert boundary_output.session_output is False
    assert boundary_output.session_consumer_node_ids == []

def test_non_modal_boundary_with_local_preview_uses_transport(
    api_intercept_module: Any,
    component_planning_module: Any,
    execution_environments_module: Any,
) -> None:
    """A Vast bridge with a local consumer must not require Modal shared storage."""
    source = api_intercept_module.LinkedOutputRef("1", 0)
    boundary_output = api_intercept_module.BoundaryOutputSpec(
        proxy_output_name="remote_output_0",
        source=source,
        io_type="IMAGE",
        is_list=False,
    )
    producer = api_intercept_module.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[boundary_output],
        execute_node_ids=["1"],
        contains_output_node=False,
    )
    consumer = api_intercept_module.RemoteComponentPlan(
        node_ids=["2"],
        representative_node_id="2",
        boundary_inputs=[
            api_intercept_module.BoundaryInputSpec(
                proxy_input_name="remote_input_0",
                source=source,
                io_type="IMAGE",
                targets=[api_intercept_module.InputTarget("2", "image")],
            )
        ],
        boundary_outputs=[],
        execute_node_ids=["2"],
        contains_output_node=False,
    )
    assignment_type = execution_environments_module.ExecutionAssignment
    provider_type = execution_environments_module.ExecutionProvider
    vast_assignment = assignment_type(
        "vast:profile:1234",
        provider_type.VAST,
        0.0,
        0.0,
    )

    session_component_ids = (
        component_planning_module._mark_remote_to_remote_session_boundaries(
            {
                "1": {"class_type": "RemoteImage", "inputs": {}},
                "2": {
                    "class_type": "RemoteImageConsumer",
                    "inputs": {"image": ["1", 0]},
                },
                "3": {
                    "class_type": "PreviewImage",
                    "inputs": {"images": ["1", 0]},
                },
            },
            [producer, consumer],
            SimpleNamespace(
                NODE_CLASS_MAPPINGS={
                    "RemoteImage": _FakeRemoteImageNode,
                    "RemoteImageConsumer": _FakeRemoteImageConsumerNode,
                    "PreviewImage": _FakePreviewImageNode,
                }
            ),
            {"1": vast_assignment, "2": vast_assignment},
        )
    )

    assert session_component_ids == set()
    assert boundary_output.session_output is False
    assert boundary_output.session_consumer_node_ids == []
    assert boundary_output.local_materializer_node_id is None

def test_transportable_list_boundary_preserves_scheduler_items(
    api_intercept_module: Any,
    component_planning_module: Any,
    execution_environments_module: Any,
) -> None:
    """Keep a same-host list output in ComfyUI instead of one bridge token."""
    source = api_intercept_module.LinkedOutputRef("1", 0)
    boundary_output = api_intercept_module.BoundaryOutputSpec(
        proxy_output_name="seed_list",
        source=source,
        io_type="INT",
        is_list=True,
    )
    producer = api_intercept_module.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[boundary_output],
        execute_node_ids=["1"],
        contains_output_node=False,
    )
    consumer = api_intercept_module.RemoteComponentPlan(
        node_ids=["2"],
        representative_node_id="2",
        boundary_inputs=[
            api_intercept_module.BoundaryInputSpec(
                proxy_input_name="seed",
                source=source,
                io_type="INT",
                targets=[api_intercept_module.InputTarget("2", "seed")],
            )
        ],
        boundary_outputs=[],
        execute_node_ids=["2"],
        contains_output_node=False,
    )
    assignment_type = execution_environments_module.ExecutionAssignment
    provider_type = execution_environments_module.ExecutionProvider
    lambda_assignment = assignment_type(
        "lambda",
        provider_type.SSH_DOCKER,
        0.0,
        0.0,
    )

    session_component_ids = (
        component_planning_module._mark_remote_to_remote_session_boundaries(
            {
                "1": {"class_type": "NextSeeds", "inputs": {}},
                "2": {
                    "class_type": "NextSeeds",
                    "inputs": {"seed": ["1", 0]},
                },
            },
            [producer, consumer],
            SimpleNamespace(NODE_CLASS_MAPPINGS={}),
            {"1": lambda_assignment, "2": lambda_assignment},
        )
    )

    assert session_component_ids == set()
    assert boundary_output.session_output is False
    assert boundary_output.session_consumer_node_ids == []

def test_split_phase_order_accounts_for_local_feedback_dependencies(
    api_intercept_module: Any,
    component_planning_module: Any,
) -> None:
    """Split phase ordering should treat local re-entry paths as real dependencies."""
    prompt = {
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["14", 0]},
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0]},
        },
        "14": {
            "class_type": "RemoteModel",
            "inputs": {},
        },
        "191": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["14", 0], "conditioning": ["358", 0]},
        },
        "357": {
            "class_type": "BetterGrok",
            "inputs": {"prompt_images": ["11", 0]},
        },
        "358": {
            "class_type": "RemoteTextEncode",
            "inputs": {"text": ["357", 1]},
        },
    }
    component_prompt = {
        "3": prompt["3"],
        "14": prompt["14"],
        "191": prompt["191"],
        "358": prompt["358"],
    }

    ordered_execute_node_ids = (
        component_planning_module._order_execute_node_ids_for_transportable_splits(
            prompt=prompt,
            component_prompt=component_prompt,
            component_node_ids={"3", "14", "191", "358"},
            execute_node_ids=["191", "3"],
        )
    )

    assert ordered_execute_node_ids == ["3", "191"]

def test_boundary_source_signature_changes_with_upstream_prompt_structure(
    api_intercept_module: Any,
    prompt_payload_metadata_module: Any,
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

    first_signature = prompt_payload_metadata_module._boundary_source_signature(base_prompt, source)
    second_signature = prompt_payload_metadata_module._boundary_source_signature(base_prompt, source)
    changed_signature = prompt_payload_metadata_module._boundary_source_signature(changed_prompt, source)

    assert first_signature == second_signature
    assert changed_signature != first_signature

