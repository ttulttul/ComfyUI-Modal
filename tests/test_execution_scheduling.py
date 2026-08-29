"""Tests for the execution scheduling boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_auto_placement_selects_every_eligible_prompt_node(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
) -> None:
    """Workflow auto placement should not require per-node remote toggles."""
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "ModalEndpointChat", "inputs": {}},
        "4": {"class_type": "VastAILeaseConfiguration", "inputs": {}},
        "5": {"class_type": "ModalRemoteConfiguration", "inputs": {}},
        "6": {"class_type": "VastRemoteConfiguration", "inputs": {}},
        "7": {"class_type": "SshRemoteConfiguration", "inputs": {}},
        "8": {"class_type": "RemoteExecutionConfigurator", "inputs": {}},
    }
    workflow = {
        "extra": {
            "remote_execution": {
                "policy": "automatic",
                "auto_place": True,
            }
        },
        "nodes": [],
    }

    selected = remote_graph_analysis_module.requested_remote_node_ids(
        prompt=prompt,
        workflow=workflow,
        settings=SimpleNamespace(marker_property="is_modal_remote"),
    )

    assert selected == {"1", "2"}

def test_disabled_auto_placement_preserves_explicit_markers(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
) -> None:
    """Manual workflows should continue honoring the existing node property."""
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    workflow = {
        "extra": {"remote_execution": {"policy": "self_hosted", "auto_place": False}},
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": True}},
        ],
    }

    selected = remote_graph_analysis_module.requested_remote_node_ids(
        prompt=prompt,
        workflow=workflow,
        settings=SimpleNamespace(marker_property="is_modal_remote"),
    )

    assert selected == {"2"}

def test_scheduler_refreshes_recent_ssh_capabilities_before_placement(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Automatic placement must not trust even recently persisted free VRAM."""
    registry = remote_hosts_module.RemoteHostRegistry.for_user_directory(tmp_path)
    previous_capabilities = execution_environments_module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=60 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        probed_at_epoch=2_000_000_000.0,
    )
    registry.replace_hosts(
        [
            remote_hosts_module.SshHostConfig(
                environment_id="freshened",
                display_name="Freshened",
                ssh_target="freshened",
                capabilities=previous_capabilities,
                health=execution_environments_module.EnvironmentHealth.READY,
            )
        ]
    )
    capabilities = execution_environments_module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=60 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        probed_at_epoch=1_700_000_000.0,
    )

    class FakeController:
        """Return current capability data for one configured host."""

        def __init__(self, host: Any) -> None:
            """Retain the probed host."""
            self.host = host

        def probe_capabilities(self) -> Any:
            """Return current capabilities."""
            return capabilities

    monkeypatch.setattr(
        execution_scheduling_module,
        "_ssh_host_registry",
        lambda _settings: registry,
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "SshDockerController",
        FakeController,
    )

    hosts = execution_scheduling_module._schedulable_ssh_hosts(SimpleNamespace())

    assert hosts[0].health.value == "ready"
    assert hosts[0].capabilities == capabilities

def test_modal_only_policy_rejects_ssh_only_llm_backend(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    tmp_path: Path,
) -> None:
    """A provider-specific backend must not be dispatched to Modal by policy."""
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["257"],
        representative_node_id="257",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["257"],
        contains_output_node=False,
    )

    with pytest.raises(
        api_intercept_module.ModalPromptValidationError,
        match="Modal-only execution cannot run SSH-only component",
    ):
        execution_scheduling_module._plan_component_execution_assignments(
            components=[component],
            prompt={
                "257": {
                    "class_type": "ModalLLM",
                    "inputs": {
                        "model_profile": (
                            "huihui-qwen3.8-27b-abliterated-q2-k-gguf"
                        )
                    },
                }
            },
            workflow={"extra": {"remote_execution": {"policy": "modal"}}},
            settings=SimpleNamespace(
                modal_gpu="H200",
                max_containers=1,
                local_storage_root=tmp_path,
            ),
        )

def test_automatic_policy_assigns_component_to_lower_cost_ready_host(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
    monkeypatch: Any,
) -> None:
    """Automatic policy should compare a compatible SSH host with Modal."""
    module = execution_environments_module
    capabilities = module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=60 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(module.GpuCapability("GPU-1", "GPU", 80 * 1024**3),),
        probed_at_epoch=2_000_000_000.0,
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="cheap-host",
        display_name="Cheap host",
        ssh_target="cheap-host",
        cost_usd_per_second=0.0001,
        capabilities=capabilities,
        health=module.EnvironmentHealth.READY,
    )
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["1"],
        contains_output_node=False,
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = execution_scheduling_module._plan_component_execution_assignments(
        components=[component],
        prompt={"1": {"class_type": "KSampler", "inputs": {"steps": 20}}},
        workflow={
            "extra": {
                "remote_execution": {
                    "policy": "automatic",
                    "auto_place": True,
                }
            }
        },
        settings=SimpleNamespace(modal_gpu="RTX-PRO-6000", max_containers=1),
    )

    assert assignments["1"].provider.value == "ssh_docker"
    assert assignments["1"].environment_id == "cheap-host"

def test_planner_recycles_idle_ssh_worker_before_cost_ranking(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
    monkeypatch: Any,
) -> None:
    """Resident managed-worker VRAM should not hide a cheaper host's capacity."""
    module = execution_environments_module
    gib = 1024**3
    occupied_capabilities = module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * gib,
        available_ram_bytes=48 * gib,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(
            module.GpuCapability(
                "GPU-4090",
                "RTX 4090",
                24 * gib,
                free_vram_bytes=11 * gib,
            ),
        ),
    )
    reclaimed_capabilities = replace(
        occupied_capabilities,
        gpus=(
            replace(
                occupied_capabilities.gpus[0],
                free_vram_bytes=23 * gib,
            ),
        ),
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="lambda",
        display_name="Lambda 4090",
        ssh_target="lambda",
        cost_usd_per_second=0.0,
        capabilities=occupied_capabilities,
        health=module.EnvironmentHealth.READY,
    )
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["1"],
        contains_output_node=False,
    )
    lifecycle_calls: list[str] = []

    class FakeController:
        """Expose one idle worker and its post-reclaim capability probe."""

        def __init__(self, configured_host: Any) -> None:
            """Retain the selected host."""
            assert configured_host.environment_id == "lambda"

        def remove_idle_managed_workers(self) -> tuple[str, ...]:
            """Report one safely recycled warm worker."""
            lifecycle_calls.append("remove")
            return ("comfy-remote-lambda-fingerprint-w0",)

        def probe_capabilities(self) -> Any:
            """Return free VRAM after the managed container stopped."""
            lifecycle_calls.append("probe")
            return reclaimed_capabilities

    monkeypatch.setattr(
        execution_scheduling_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "SshDockerController",
        FakeController,
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_ssh_host_registry",
        lambda _settings: None,
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = execution_scheduling_module._plan_component_execution_assignments(
        components=[component],
        prompt={"1": {"class_type": "KSampler", "inputs": {"steps": 20}}},
        workflow={
            "extra": {
                "remote_execution": {
                    "policy": "automatic",
                    "auto_place": True,
                    "minimum_vram_gb": 16,
                }
            }
        },
        settings=SimpleNamespace(modal_gpu="RTX-PRO-6000", max_containers=1),
    )

    assert lifecycle_calls == ["remove", "probe"]
    assert assignments["1"].provider is module.ExecutionProvider.SSH_DOCKER
    assert assignments["1"].environment_id == "lambda"

def test_planner_does_not_recycle_for_equal_cost_tie_break(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    execution_environments_module: Any,
) -> None:
    """A lexical tie alone must not discard a compatible worker's warm cache."""
    module = execution_environments_module
    actual = module.ExecutionAssignment(
        environment_id="ready-host",
        provider=module.ExecutionProvider.SSH_DOCKER,
        predicted_cost_usd=0.0,
        predicted_completion_seconds=60.0,
    )
    optimistic = replace(actual, environment_id="idle-host")

    assert not execution_scheduling_module._reclaim_improves_assignment(
        optimistic,
        actual,
        module.ComponentResourceRequirements(),
    )

def test_automatic_policy_rejects_zero_cost_host_for_oversized_model(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Model weights plus headroom must exclude a cheap host before cost ranking."""
    module = execution_environments_module
    model_path = tmp_path / "minimax_h3_bf16.safetensors"
    with model_path.open("wb") as model_file:
        model_file.truncate(66 * 1024**3)

    capabilities = module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=60 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(module.GpuCapability("GPU-4090", "RTX 4090", 24 * 1024**3),),
        probed_at_epoch=2_000_000_000.0,
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="lambda",
        display_name="Lambda 4090",
        ssh_target="lambda",
        cost_usd_per_second=0.0,
        capabilities=capabilities,
        health=module.EnvironmentHealth.READY,
    )
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["6"],
        representative_node_id="6",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["6"],
        contains_output_node=False,
    )
    prompt = {
        "6": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": str(model_path), "weight_dtype": "default"},
        }
    }
    workflow = {
        "extra": {
            "remote_execution": {
                "policy": "automatic",
                "auto_place": True,
            }
        }
    }
    settings = SimpleNamespace(
        modal_gpu="H200",
        max_containers=1,
        comfyui_root=None,
    )
    preferences = module.WorkflowExecutionPreferences.from_workflow(workflow)
    estimate = execution_scheduling_module._component_memory_estimate(
        component,
        prompt,
        preferences,
        settings,
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = execution_scheduling_module._plan_component_execution_assignments(
        components=[component],
        prompt=prompt,
        workflow=workflow,
        settings=settings,
    )

    assert estimate.model_asset_count == 1
    assert estimate.largest_model_bytes == 66 * 1024**3
    assert 24 * 1024**3 < estimate.minimum_vram_bytes < 96 * 1024**3
    assert estimate.minimum_ram_bytes == 70 * 1024**3
    assert assignments["6"].provider is module.ExecutionProvider.MODAL
    assert assignments["6"].environment_id == "modal:H200"

def test_planner_resolves_hugging_face_metadata_before_cost_ranking(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    execution_resource_estimates_module: Any,
    execution_environments_module: Any,
    remote_hosts_module: Any,
    llm_resolver_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Raw Hugging Face IDs must expose their VRAM floor before SSH placement."""
    module = execution_environments_module
    capabilities = module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=120 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(module.GpuCapability("GPU-4090", "RTX 4090", 24 * 1024**3),),
        probed_at_epoch=2_000_000_000.0,
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="lambda",
        display_name="Lambda 4090",
        ssh_target="lambda",
        cost_usd_per_second=0.0,
        capabilities=capabilities,
        health=module.EnvironmentHealth.READY,
    )
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["257"],
        representative_node_id="257",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["257"],
        contains_output_node=False,
    )
    prompt = {
        "257": {
            "class_type": "ModalLLM",
            "inputs": {"model_profile": "owner/large-model"},
        }
    }
    workflow = {
        "extra": {
            "remote_execution": {
                "policy": "automatic",
                "auto_place": True,
            }
        }
    }
    profile = SimpleNamespace(
        profile_id="hf-" + "d" * 64,
        artifact_bytes=55_563_006_216,
        estimated_vram_gb=67.9,
    )
    resolved_references: list[str] = []

    def resolve(model_reference: str, storage_root: Path) -> Any:
        """Return deterministic metadata without downloading model weights."""
        assert storage_root == tmp_path
        resolved_references.append(model_reference)
        return SimpleNamespace(profile=profile)

    settings = SimpleNamespace(
        modal_gpu="H200",
        max_containers=1,
        comfyui_root=None,
        local_storage_root=tmp_path,
    )
    monkeypatch.setattr(
        execution_resource_estimates_module,
        "resolve_model_profile",
        resolve,
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        execution_scheduling_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = execution_scheduling_module._plan_component_execution_assignments(
        components=[component],
        prompt=prompt,
        workflow=workflow,
        settings=settings,
    )

    assert resolved_references == ["owner/large-model"]
    assert assignments["257"].provider is module.ExecutionProvider.MODAL
    assert assignments["257"].environment_id == "modal:H200"
    assert "requires at least 67.90 GiB GPU VRAM" in assignments["257"].reasons

def test_planner_keeps_unmarked_llm_local_between_remote_text_nodes(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
) -> None:
    """Transportable text boundaries must not absorb a local LLM into Modal."""
    prompt = {
        "1": {"class_type": "RemoteTextSource", "inputs": {}},
        "2": {
            "class_type": "ModalLLM",
            "inputs": {"prompt": ["1", 0], "model_profile": "owner/model"},
        },
        "3": {
            "class_type": "RemoteTextConsumer",
            "inputs": {"prompt": ["2", 0]},
        },
    }
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
        ]
    }
    fake_nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteTextSource": _FakeTextNode,
            "ModalLLM": _FakeTextNode,
            "RemoteTextConsumer": _FakeTextNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    analysis = remote_graph_analysis_module.analyze_remote_node_selection(
        prompt,
        workflow,
        [],
        settings=SimpleNamespace(marker_property="is_modal_remote"),
        nodes_module=fake_nodes_module,
    )

    assert analysis.resolved_remote_node_ids == ["1", "3"]
    assert analysis.sandwiched_local_node_ids == ["2"]
    assert analysis.added_node_ids == []

def test_configured_non_modal_plan_omits_modal_status_gpu(
    api_intercept_module: Any,
) -> None:
    """Vast-only and SSH-only plans must not activate Modal status polling."""
    summary = api_intercept_module.RewriteSummary(
        execution_assignments_by_representative={
            "vast-component": api_intercept_module.ExecutionAssignment(
                environment_id="vast:instance-1",
                provider=api_intercept_module.ExecutionProvider.VAST,
                predicted_cost_usd=0.01,
                predicted_completion_seconds=20.0,
                configuration_id="vast-config",
            ),
            "ssh-component": api_intercept_module.ExecutionAssignment(
                environment_id="ssh-host",
                provider=api_intercept_module.ExecutionProvider.SSH_DOCKER,
                predicted_cost_usd=0.0,
                predicted_completion_seconds=15.0,
                configuration_id="ssh-config",
            ),
        },
        remote_configurations=[
            {
                "configuration_id": "vast-config",
                "provider": "vast",
                "display_name": "Vast pool",
            },
            {
                "configuration_id": "ssh-config",
                "provider": "ssh_docker",
                "display_name": "SSH host",
            },
        ],
    )

    assert api_intercept_module._selected_modal_gpus(summary, "B300") == []
    assert api_intercept_module._prompt_uses_remote_execution_configurator(
        {
            "99": {
                "class_type": "RemoteExecutionConfigurator",
                "inputs": {},
            }
        }
    )

def test_configured_modal_plan_reports_only_selected_modal_gpus(
    api_intercept_module: Any,
) -> None:
    """Status metadata should come from selected configurations, not legacy GPU state."""
    summary = api_intercept_module.RewriteSummary(
        execution_assignments_by_representative={
            "modal-component": api_intercept_module.ExecutionAssignment(
                environment_id="modal:modal-config:H200",
                provider=api_intercept_module.ExecutionProvider.MODAL,
                predicted_cost_usd=0.01,
                predicted_completion_seconds=10.0,
                configuration_id="modal-config",
            )
        },
        remote_configurations=[
            {
                "configuration_id": "modal-config",
                "provider": "modal",
                "display_name": "Modal H200",
                "gpu_type": "H200",
            }
        ],
    )

    assert api_intercept_module._selected_modal_gpus(summary, "B300") == ["H200"]

def test_planner_resolved_llm_profile_is_attached_to_matching_payload(
    api_intercept_module: Any,
    prompt_payload_metadata_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """The execution payload should carry metadata already resolved by planning."""
    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    payload = {
        "component_id": "llm",
        "subgraph_prompt": {
            "llm": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": profile.profile_id},
            }
        },
    }

    prompt_payload_metadata_module._attach_resolved_llm_profiles(
        payload,
        {profile.profile_id: profile},
        SimpleNamespace(local_storage_root=tmp_path),
    )

    entry = payload["resolved_llm_profiles"][profile.profile_id]
    assert entry["profile"] == profile.to_mapping()
    assert entry["security_scan_complete"] is True

def test_planner_attaches_next_distinct_affinity_as_speculative_prewarm_target(
    api_intercept_module: Any,
    prompt_affinity_planning_module: Any,
) -> None:
    """Each proxy should prepare only its nearest reachable future worker group."""
    rewritten_prompt = {
        "spec-a": {
            "class_type": "ModalProxy",
            "inputs": {
                "original_node_data": {
                    "component_id": "spec-a",
                    "prompt_id": "prompt-spec",
                    "modal_gpu": "RTX-PRO-6000",
                    "execution_provider": "modal",
                    "execution_environment_id": "modal:RTX-PRO-6000",
                    "remote_worker_affinity_group": "llm",
                    "subgraph_prompt": {"1": {"class_type": "ModalLLM", "inputs": {}}},
                }
            },
        },
        "spec-b": {
            "class_type": "ModalProxy",
            "inputs": {
                "upstream": ["spec-local", 0],
                "original_node_data": {
                    "component_id": "spec-b",
                    "prompt_id": "prompt-spec",
                    "modal_gpu": "RTX-PRO-6000",
                    "execution_provider": "modal",
                    "execution_environment_id": "modal:RTX-PRO-6000",
                    "remote_worker_affinity_group": "comfy",
                    "remote_local_gap_pool": True,
                    "snapshot_profile_key": "loader-profile:abc",
                    "subgraph_prompt": {
                        "2": {
                            "class_type": "UNETLoader",
                            "inputs": {"unet_name": "video-model.safetensors"},
                        }
                    },
                },
            },
        },
        "spec-local": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["spec-a", 0]},
        },
        "spec-c": {
            "class_type": "ModalProxy",
            "inputs": {
                "upstream": ["spec-b", 0],
                "original_node_data": {
                    "component_id": "spec-c",
                    "prompt_id": "prompt-spec",
                    "modal_gpu": "RTX-PRO-6000",
                    "execution_provider": "modal",
                    "execution_environment_id": "modal:RTX-PRO-6000",
                    "remote_worker_affinity_group": "llm",
                    "subgraph_prompt": {"3": {"class_type": "ModalLLM", "inputs": {}}},
                },
            },
        },
    }

    prompt_affinity_planning_module._configure_speculative_affinity_prewarm_payloads(
        rewritten_prompt=rewritten_prompt,
        execution_stages=[["spec-a", "spec-b"], ["spec-c"]],
    )

    first_payload = api_intercept_module.registered_proxy_execution_payload(
        "spec-a", rewritten_prompt["spec-a"]["inputs"]["original_node_data"]
    )
    second_payload = api_intercept_module.registered_proxy_execution_payload(
        "spec-b", rewritten_prompt["spec-b"]["inputs"]["original_node_data"]
    )
    third_payload = api_intercept_module.registered_proxy_execution_payload(
        "spec-c", rewritten_prompt["spec-c"]["inputs"]["original_node_data"]
    )

    first_target = first_payload["speculative_remote_prewarm_target"]
    second_target = second_payload["speculative_remote_prewarm_target"]
    assert first_target["component_id"] == "spec-b"
    assert first_target["remote_worker_affinity_group"] == "comfy"
    assert first_target["snapshot_profile_key"] == "loader-profile:abc"
    assert second_target["component_id"] == "spec-c"
    assert second_target["remote_worker_affinity_group"] == "llm"
    assert "speculative_remote_prewarm_target" not in third_payload

def test_planner_does_not_bridge_local_gap_keepalive_across_providers(
    api_intercept_module: Any,
    prompt_affinity_planning_module: Any,
) -> None:
    """A Modal producer must not retain a slot for an SSH continuation."""
    rewritten_prompt = {
        "modal-producer": {
            "class_type": "ModalProxy",
            "inputs": {
                "original_node_data": {
                    "component_id": "modal-producer",
                    "execution_provider": "modal",
                    "execution_environment_id": "modal:H200",
                }
            },
        },
        "local-gap": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["modal-producer", 0]},
        },
        "ssh-consumer": {
            "class_type": "ModalProxy",
            "inputs": {
                "source": ["local-gap", 0],
                "original_node_data": {
                    "component_id": "ssh-consumer",
                    "execution_provider": "ssh_docker",
                    "execution_environment_id": "lambda",
                },
            },
        },
    }

    prompt_affinity_planning_module._configure_local_gap_keepalive_payloads(
        rewritten_prompt=rewritten_prompt,
        remote_component_ids=["modal-producer", "ssh-consumer"],
        sandwiched_local_node_ids={"local-gap"},
    )

    for component_id in ("modal-producer", "ssh-consumer"):
        payload = api_intercept_module.registered_proxy_execution_payload(
            component_id,
            rewritten_prompt[component_id]["inputs"]["original_node_data"],
        )
        assert "remote_local_gap_pool" not in payload
        assert "keepalive_after_remote_component" not in payload
        assert "stop_local_gap_keepalive_before_remote_component" not in payload

def test_delete_modal_cache_dicts_deletes_configured_dicts(
    api_intercept_module: Any,
    modal_admin_ops_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Deleting Modal caches should clear and delete every configured cache Dict."""

    class FakeNotFoundError(Exception):
        """Stand-in for Modal object misses."""

    class FakeDictObject:
        """Minimal Modal Dict object used only for existence checks."""

        def __init__(self, name: str) -> None:
            """Store the configured Dict name."""
            self.name = name

        def delete(self, name: str) -> None:
            """Fail if the deprecated instance deletion path is used."""
            raise AssertionError(f"deprecated Dict.delete path used for {name}")

    class FakeDictObjects:
        """Minimal Modal Dict manager namespace."""

        @staticmethod
        def delete(name: str, allow_missing: bool = False) -> None:
            """Record a manager delete call."""
            assert allow_missing is True
            deleted.append(name)

    class FakeDict:
        """Minimal Modal Dict namespace."""

        objects = FakeDictObjects()

        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> FakeDictObject:
            """Return fake Dict objects, except for one missing cache."""
            assert create_if_missing is False
            if name == "app-interrupts":
                raise FakeNotFoundError(name)
            return FakeDictObject(name)

    class FakeModal:
        """Minimal Modal SDK double."""

        exception = SimpleNamespace(NotFoundError=FakeNotFoundError)
        Dict = FakeDict

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="remote",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
        interrupt_dict_name="app-interrupts",
        node_output_cache_dict_name="app-node-cache",
        session_bridge_dict_name="app-session-bridges",
        sync_index_dict_name="app-sync-index",
        snapshot_profile_dict_name="app-snapshot-profiles",
    )
    deleted: list[str] = []
    monkeypatch.setattr(modal_admin_ops_module, "modal", FakeModal)

    result = asyncio.run(api_intercept_module.delete_modal_cache_dicts(settings))

    assert result == {
        "deleted": [
            "app-node-cache",
            "app-session-bridges",
            "app-sync-index",
            "app-snapshot-profiles",
        ],
        "skipped": ["app-interrupts"],
    }
    assert deleted == result["deleted"]

def test_delete_modal_volume_deletes_configured_volume(
    api_intercept_module: Any,
    modal_admin_ops_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Deleting the Modal volume should target only the configured volume name."""

    class FakeVolumeObject:
        """Minimal Modal Volume object used only for existence checks."""

        def __init__(self, name: str) -> None:
            """Store the configured Volume name."""
            self.name = name

        def delete(self, name: str) -> None:
            """Fail if the deprecated instance deletion path is used."""
            raise AssertionError(f"deprecated Volume.delete path used for {name}")

    class FakeVolumeObjects:
        """Minimal Modal Volume manager namespace."""

        @staticmethod
        def delete(name: str, allow_missing: bool = False) -> None:
            """Record a manager delete call."""
            assert allow_missing is True
            deleted.append(name)

    class FakeVolume:
        """Minimal Modal Volume namespace."""

        objects = FakeVolumeObjects()

        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> FakeVolumeObject:
            """Return a fake Volume object."""
            assert create_if_missing is False
            return FakeVolumeObject(name)

    class FakeModal:
        """Minimal Modal SDK double."""

        exception = SimpleNamespace()
        Volume = FakeVolume

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="remote",
        sync_custom_nodes=False,
        volume_name="configured-volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )
    deleted: list[str] = []
    monkeypatch.setattr(modal_admin_ops_module, "modal", FakeModal)

    result = asyncio.run(api_intercept_module.delete_modal_volume(settings))

    assert result == {"deleted": ["configured-volume"], "skipped": []}
    assert deleted == ["configured-volume"]

def test_selected_vast_capacity_streams_setup_status(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    vast_models_module: Any,
) -> None:
    """Queue-time Vast acquisition should stream provider progress into the UI."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="vast-config",
        profile_name="Vast-pool",
        maximum_instances=1,
    )
    configuration = api_intercept_module.VastRemoteConfiguration(
        configuration_id="vast-config",
        display_name="Vast-pool",
        profile=profile,
    )
    configuration_set = api_intercept_module.RemoteConfigurationSet(
        configurations=(configuration,)
    )
    assignments = {
        "component-1": api_intercept_module.ExecutionAssignment(
            environment_id=profile.environment_id,
            provider=api_intercept_module.ExecutionProvider.VAST,
            predicted_cost_usd=0.02,
            predicted_completion_seconds=30.0,
            configuration_id="vast-config",
            capacity_slot_index=0,
        )
    }
    requirements = {
        "component-1": api_intercept_module.ComponentResourceRequirements(
            estimated_execution_seconds=30.0
        )
    }
    quote = SimpleNamespace(
        profile=profile,
        predicted_incremental_cost_usd=0.02,
    )
    status_events: list[tuple[str, int | None, int | None]] = []
    environment_status_events: list[
        tuple[str, str, int | None, int | None]
    ] = []

    class FakeVastService:
        """Emit representative readiness phases without renting an instance."""

        def acquire_sync(
            self,
            selected_quote: Any,
            *,
            slot: int,
            status_callback: Any,
        ) -> Any:
            """Emit image and runtime phases before returning a fake lease."""
            assert selected_quote is quote
            assert slot == 0
            status_callback("Vast.ai instance 42 is downloading the worker image")
            status_callback("Initializing Vast.ai worker")
            return SimpleNamespace(
                environment_id="vast:vast-config:42",
                idle_retention_seconds=3600.0,
            )

    leases = execution_scheduling_module._prepare_selected_vast_capacity(
        assignments=assignments,
        configuration_set=configuration_set,
        requirements_by_component=requirements,
        vast_quotes={("component-1", "vast-config"): quote},
        vast_service=FakeVastService(),
        status_callback=lambda message, current, total: status_events.append(
            (message, current, total)
        ),
        environment_status_callback=(
            lambda environment_id, message, current, total: (
                environment_status_events.append(
                    (environment_id, message, current, total)
                )
            )
        ),
    )

    assert list(leases) == ["vast:vast-config:42"]
    assert status_events == [
        ("Acquiring Vast.ai capacity 1 of 1", 0, 1),
        ("Vast.ai instance 42 is downloading the worker image", 0, 1),
        ("Initializing Vast.ai worker", 0, 1),
        ("Vast.ai capacity 1 of 1 is ready", 1, 1),
    ]
    assert environment_status_events == [
        ("vast:vast-config", "Acquiring Vast.ai capacity 1 of 1", None, None),
        (
            "vast:vast-config",
            "Vast.ai instance 42 is downloading the worker image",
            None,
            None,
        ),
        ("vast:vast-config", "Initializing Vast.ai worker", None, None),
        (
            "vast:vast-config:42",
            "Vast.ai worker ready; preparing remote assets next",
            None,
            None,
        ),
    ]
    assert assignments["component-1"].environment_id == "vast:vast-config:42"

def test_selected_vast_capacity_preserves_intentional_cancellation(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    vast_models_module: Any,
) -> None:
    """Cancelling capacity acquisition should not become a provider failure."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="vast-config",
        profile_name="Vast-pool",
        maximum_instances=1,
    )
    configuration = api_intercept_module.VastRemoteConfiguration(
        configuration_id="vast-config",
        display_name="Vast-pool",
        profile=profile,
    )
    assignment = api_intercept_module.ExecutionAssignment(
        environment_id=profile.environment_id,
        provider=api_intercept_module.ExecutionProvider.VAST,
        predicted_cost_usd=0.02,
        predicted_completion_seconds=30.0,
        configuration_id="vast-config",
        capacity_slot_index=0,
    )
    quote = SimpleNamespace(
        profile=profile,
        predicted_incremental_cost_usd=0.02,
    )

    class CancelledVastService:
        """Stop acquisition as though the user cancelled queue preparation."""

        def acquire_sync(self, selected_quote: Any, *, slot: int) -> Any:
            """Raise the prompt-scoped cancellation without a provider failure."""
            assert selected_quote is quote
            assert slot == 0
            raise api_intercept_module.SyncCancelledError(
                "Remote workflow preparation was cancelled."
            )

    with pytest.raises(
        api_intercept_module.SyncCancelledError,
        match="Remote workflow preparation was cancelled",
    ):
        execution_scheduling_module._prepare_selected_vast_capacity(
            assignments={"component-1": assignment},
            configuration_set=api_intercept_module.RemoteConfigurationSet(
                configurations=(configuration,)
            ),
            requirements_by_component={
                "component-1": api_intercept_module.ComponentResourceRequirements(
                    estimated_execution_seconds=30.0
                )
            },
            vast_quotes={("component-1", "vast-config"): quote},
            vast_service=CancelledVastService(),
        )

