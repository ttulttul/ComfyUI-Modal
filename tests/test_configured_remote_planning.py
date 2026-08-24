"""Tests for capacity planning from connected remote configuration nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _RemoteImageNode:
    """Minimal remote image producer used by rewrite integration coverage."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


class _LocalImageSinkNode:
    """Minimal local image consumer used by rewrite integration coverage."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


def _component(api_module: Any, node_id: str) -> Any:
    """Return one independent remote component plan."""
    return api_module.RemoteComponentPlan(
        node_ids=[node_id],
        representative_node_id=node_id,
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=[node_id],
        contains_output_node=False,
    )


def _vast_inputs(name: str, maximum_instances: int) -> dict[str, Any]:
    """Return minimal valid Vast configuration inputs."""
    return {
        "profile_name": name,
        "allocated_disk_gb": 200.0,
        "idle_retention_hours": 24.0,
        "maximum_instances": maximum_instances,
    }


def _capabilities(module: Any, name: str = "Configured GPU") -> Any:
    """Return deterministic scheduler capabilities."""
    return module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=128 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="managed",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(
            module.GpuCapability(
                uuid=f"gpu-{name}",
                name=name,
                total_vram_bytes=96 * 1024**3,
            ),
        ),
    )


def test_connected_modal_configurations_replace_global_single_gpu_choice(
    api_intercept_module: Any,
    remote_configuration_nodes_module: Any,
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Two Modal pools should both participate even under the legacy default policy."""
    api = api_intercept_module
    nodes = remote_configuration_nodes_module
    monkeypatch.setattr(api, "_execution_history", lambda _settings: None)
    monkeypatch.setitem(api._MODAL_GPU_COST_USD_PER_SECOND, "T4", 0.0)
    monkeypatch.setitem(api._MODAL_GPU_COST_USD_PER_SECOND, "H200", 0.0)
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {}},
        "10": {
            "class_type": nodes.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-t4",
                "gpu_type": "T4",
                "instance_count": 1,
            },
        },
        "11": {
            "class_type": nodes.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-h200",
                "gpu_type": "H200",
                "instance_count": 1,
            },
        },
        "99": {
            "class_type": nodes.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {
                "configuration_0": ["10", 0],
                "configuration_1": ["11", 0],
            },
        },
    }

    plan = api._plan_component_execution(
        components=[_component(api, "1"), _component(api, "2")],
        prompt=prompt,
        workflow={"extra": {"remote_execution": {"policy": "modal"}}},
        settings=settings_module.get_settings(),
    )

    assert plan.configuration_set is not None
    assert {assignment.configuration_id for assignment in plan.assignments.values()} == {
        "10",
        "11",
    }
    assert {assignment.environment_id for assignment in plan.assignments.values()} == {
        "modal:10:T4",
        "modal:11:H200",
    }


def test_vast_capacity_is_quoted_globally_and_acquired_only_after_assignment(
    api_intercept_module: Any,
    execution_environments_module: Any,
    remote_configuration_nodes_module: Any,
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Independent Vast pools and their capacity slots should survive into leases."""
    api = api_intercept_module
    environment_module = execution_environments_module
    nodes = remote_configuration_nodes_module
    events: list[tuple[Any, ...]] = []

    class FakeVastService:
        """Expose deterministic quotes and record post-plan acquisitions."""

        @classmethod
        def from_environment(cls, settings: Any, *, repo_root: Any) -> "FakeVastService":
            """Construct the fake service without credentials."""
            events.append(("service", settings, repo_root))
            return cls()

        def quote_best_profile_sync(
            self,
            profiles: Any,
            **requirements: Any,
        ) -> Any:
            """Return a lower incremental price for the broad pool."""
            profile = profiles[0]
            events.append(("quote", profile.profile_name, requirements))
            cost = 0.01 if profile.profile_name == "vast-broad" else 0.02
            return SimpleNamespace(
                profile=profile,
                predicted_incremental_cost_usd=cost,
                predicted_execution_seconds=requirements[
                    "predicted_execution_seconds"
                ],
            )

        def scheduling_state(self, quote: Any) -> Any:
            """Expose one virtual pool whose limit is applied by the caller."""
            return environment_module.EnvironmentSchedulingState(
                environment_id=f"vast:{quote.profile.profile_id}",
                provider=environment_module.ExecutionProvider.VAST,
                enabled=True,
                health=environment_module.EnvironmentHealth.READY,
                cost_usd_per_second=(
                    quote.predicted_incremental_cost_usd
                    / quote.predicted_execution_seconds
                ),
                capabilities=_capabilities(
                    environment_module,
                    quote.profile.profile_name,
                ),
            )

        def acquire_sync(self, quote: Any, *, slot: int = 0) -> Any:
            """Create one concrete lease after all assignments have succeeded."""
            profile = quote.profile
            events.append(("acquire", profile.profile_name, slot))
            instance_id = len(
                [event for event in events if event[0] == "acquire"]
            )
            profile_id = (
                profile.profile_id
                if slot == 0
                else f"{profile.profile_id}-slot-{slot}"
            )
            return SimpleNamespace(
                environment_id=f"vast:{profile_id}:{instance_id}",
                instance_id=instance_id,
                profile_id=profile_id,
                profile_name=profile.profile_name,
                gpu_name="Configured GPU",
                gpu_count=1,
                gpu_ram_mb=96 * 1024,
                hourly_cost_usd=1.0,
                idle_retention_seconds=24 * 3600,
            )

    monkeypatch.setattr(api, "VastService", FakeVastService)
    monkeypatch.setattr(api, "_execution_history", lambda _settings: None)
    prompt = {
        str(index): {"class_type": "KSampler", "inputs": {}}
        for index in range(1, 4)
    }
    prompt.update(
        {
            "20": {
                "class_type": nodes.VAST_REMOTE_CONFIGURATION_NODE_ID,
                "inputs": _vast_inputs("vast-broad", 2),
            },
            "21": {
                "class_type": nodes.VAST_REMOTE_CONFIGURATION_NODE_ID,
                "inputs": _vast_inputs("vast-premium", 1),
            },
            "99": {
                "class_type": nodes.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
                "inputs": {
                    "configuration_0": ["20", 0],
                    "configuration_1": ["21", 0],
                },
            },
        }
    )

    plan = api._plan_component_execution(
        components=[_component(api, str(index)) for index in range(1, 4)],
        prompt=prompt,
        workflow={"extra": {"remote_execution": {"policy": "automatic"}}},
        settings=settings_module.get_settings(),
    )

    acquisitions = [event for event in events if event[0] == "acquire"]
    assert acquisitions == [
        ("acquire", "vast-broad", 0),
        ("acquire", "vast-broad", 1),
        ("acquire", "vast-premium", 0),
    ]
    assert len(plan.vast_leases_by_environment) == 3
    assert len({assignment.environment_id for assignment in plan.assignments.values()}) == 3
    assert {assignment.configuration_id for assignment in plan.assignments.values()} == {
        "20",
        "21",
    }


def test_workflow_ssh_configuration_is_probed_and_carried_into_execution_metadata(
    api_intercept_module: Any,
    execution_environments_module: Any,
    remote_configuration_nodes_module: Any,
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Workflow SSH hosts should not depend on the installation-wide host registry."""
    api = api_intercept_module
    nodes = remote_configuration_nodes_module

    def probe(configuration: Any) -> Any:
        """Attach deterministic capabilities to the workflow host declaration."""
        return api.replace(
            configuration.host,
            capabilities=_capabilities(execution_environments_module, "SSH GPU"),
            health=execution_environments_module.EnvironmentHealth.READY,
        )

    monkeypatch.setattr(api, "_probe_workflow_ssh_configuration", probe)
    monkeypatch.setattr(api, "_execution_history", lambda _settings: None)
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "30": {
            "class_type": nodes.SSH_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "environment_id": "lambda",
                "display_name": "Lambda GPU",
                "ssh_target": "lambda",
                "cost_usd_per_hour": 1.0,
                "maximum_workers": 2,
            },
        },
        "99": {
            "class_type": nodes.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {"configuration_0": ["30", 0]},
        },
    }

    plan = api._plan_component_execution(
        components=[_component(api, "1")],
        prompt=prompt,
        workflow={"extra": {"remote_execution": {"policy": "modal"}}},
        settings=settings_module.get_settings(),
    )
    assignment = plan.assignments["1"]
    metadata = api._configured_provider_metadata(
        execution_plan=plan,
        assignment=assignment,
        vast_leases_by_environment={},
    )

    assert assignment.provider is execution_environments_module.ExecutionProvider.SSH_DOCKER
    assert assignment.environment_id == "lambda"
    assert assignment.configuration_id == "lambda"
    assert metadata is not None
    assert metadata["ssh_host_config"]["ssh_target"] == "lambda"
    assert metadata["ssh_host_config"]["capabilities"] is None
    assert metadata["ssh_host_config"]["health"] == "unknown"


def test_rewrite_stamps_selected_modal_configuration_on_proxy_payload(
    api_intercept_module: Any,
    remote_configuration_nodes_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """The configured GPU and count must reach the existing remote executor path."""
    api = api_intercept_module
    nodes = remote_configuration_nodes_module
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()
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
    fake_nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteImage": _RemoteImageNode,
            "LocalSink": _LocalImageSinkNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    prompt = {
        "1": {"class_type": "RemoteImage", "inputs": {}},
        "2": {"class_type": "LocalSink", "inputs": {"image": ["1", 0]}},
        "10": {
            "class_type": nodes.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-t4",
                "gpu_type": "T4",
                "instance_count": 2,
            },
        },
        "99": {
            "class_type": nodes.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {"configuration_0": ["10", 0]},
        },
    }
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
        ]
    }
    monkeypatch.setattr(api, "_execution_history", lambda _settings: None)

    rewritten, summary = api.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    payload = rewritten["1"]["inputs"]["original_node_data"]
    assert payload["execution_provider"] == "modal"
    assert payload["execution_environment_id"] == "modal:10:T4"
    assert payload["remote_configuration_id"] == "10"
    assert payload["remote_configuration_name"] == "modal-t4"
    assert payload["modal_gpu"] == "T4"
    assert payload["modal_max_containers"] == 2
    assert summary.remote_configurations == [
        {
            "configuration_id": "10",
            "display_name": "modal-t4",
            "provider": "modal",
            "gpu_type": "T4",
            "instance_count": 2,
        }
    ]
