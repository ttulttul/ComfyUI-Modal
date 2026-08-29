"""Tests for prompt rewriting and asset sync integration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest


def test_ssh_hostname_extracts_safe_runtime_badge_label(
    api_intercept_module: Any,
) -> None:
    """Planner UI metadata should show the host rather than an SSH user target."""
    assert api_intercept_module._ssh_hostname("worker@example.internal") == "example.internal"
    assert api_intercept_module._ssh_hostname("[2001:db8::17]") == "2001:db8::17"


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


class _FakeRemoteNoiseNode:
    """Fake node that produces a non-transportable NOISE strategy object."""

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    OUTPUT_IS_LIST = (False,)


class _FakeVAELoaderNode:
    """Fake VAE loader that produces a non-transportable VAE output."""

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    OUTPUT_IS_LIST = (False,)


class _FakeVAEDecodeNode:
    """Fake VAE decoder that consumes latent and VAE inputs and produces an image."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


class _FakeVAEEncodeNode:
    """Fake VAE encoder that consumes image and VAE inputs and produces a latent."""

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


class _FakeRemoteVideoNode:
    """Fake remote node that produces a transportable VIDEO output."""

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_IS_LIST = (False,)


class _FakeSaveVideoNode:
    """Fake terminal SaveVideo node that returns its VIDEO input."""

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True


class _FakeRemoteAudioNode:
    """Fake remote node that produces a transportable AUDIO output."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_IS_LIST = (False,)


class _FakeTextNode:
    """Fake text node used for local LLM boundary planning."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteImageConsumerNode:
    """Fake remote node that consumes IMAGE and produces IMAGE."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


class _FakePreviewImageNode:
    """Fake preview node that behaves like a terminal UI output."""

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = True


class _FakeRemoteArtifactWriterNode:
    """Fake terminal remote node that saves artifacts without ComfyUI outputs."""

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = False


def test_auto_placement_selects_every_eligible_prompt_node(
    api_intercept_module: Any,
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

    selected = api_intercept_module.requested_remote_node_ids(
        prompt=prompt,
        workflow=workflow,
        settings=SimpleNamespace(marker_property="is_modal_remote"),
    )

    assert selected == {"1", "2"}


def test_disabled_auto_placement_preserves_explicit_markers(
    api_intercept_module: Any,
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

    selected = api_intercept_module.requested_remote_node_ids(
        prompt=prompt,
        workflow=workflow,
        settings=SimpleNamespace(marker_property="is_modal_remote"),
    )

    assert selected == {"2"}


def test_remote_environment_routes_save_and_probe_hosts(
    api_intercept_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The ComfyUI API should persist and refresh credential-free SSH hosts."""

    class FakeRoutes:
        """Capture handlers by HTTP method and route."""

        def __init__(self) -> None:
            """Initialize an empty handler map."""
            self.handlers: dict[tuple[str, str], Any] = {}

        def _decorator(self, method: str, path: str) -> Any:
            """Return one route registration decorator."""

            def register(handler: Any) -> Any:
                """Store and return one handler."""
                self.handlers[(method, path)] = handler
                return handler

            return register

        def get(self, path: str) -> Any:
            """Register one GET route."""
            return self._decorator("GET", path)

        def put(self, path: str) -> Any:
            """Register one PUT route."""
            return self._decorator("PUT", path)

        def post(self, path: str) -> Any:
            """Register one POST route."""
            return self._decorator("POST", path)

    class FakeRequest:
        """Return one predefined JSON body."""

        def __init__(self, payload: dict[str, Any]) -> None:
            """Store the request body."""
            self.payload = payload
            self.query: dict[str, str] = {}

        async def json(self) -> dict[str, Any]:
            """Return the request body."""
            return self.payload

    registry = remote_hosts_module.RemoteHostRegistry.for_user_directory(tmp_path)
    routes = FakeRoutes()
    prompt_server = SimpleNamespace(routes=routes, prompt_queue=None)
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
        custom_nodes_dir=None,
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
    )

    class FakeController:
        """Return fixed capabilities without opening SSH."""

        def __init__(self, host: Any) -> None:
            """Retain the selected host."""
            self.host = host

        def probe_capabilities(self) -> Any:
            """Return fixed ready capabilities."""
            return capabilities

    monkeypatch.setattr(api_intercept_module, "_ROUTE_REGISTERED", False)
    monkeypatch.setattr(api_intercept_module, "_ssh_host_registry", lambda _settings: registry)
    monkeypatch.setattr(api_intercept_module, "SshDockerController", FakeController)
    monkeypatch.setattr(
        api_intercept_module,
        "_refresh_r2_storage_usage",
        lambda _storage: api_intercept_module.R2StorageUsage(
            size_bytes=7 * 1024**3,
            object_count=17,
        ),
    )
    unlock_requests: list[bool] = []
    monkeypatch.setattr(
        api_intercept_module,
        "request_macos_keychain_unlock",
        lambda: unlock_requests.append(True),
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_get_server_module",
        lambda: SimpleNamespace(PromptServer=SimpleNamespace(instance=prompt_server)),
    )

    api_intercept_module.setup_modal_queue_route(
        prompt_server=prompt_server,
        sync_engine=object(),
        settings=settings,
    )
    update = routes.handlers[("PUT", "/remote/environments")]
    update_response = asyncio.run(
        update(
            FakeRequest(
                {
                    "version": 1,
                    "hosts": [
                        {
                            "environment_id": "gpu-one",
                            "display_name": "GPU one",
                            "ssh_target": "gpu-one",
                        }
                    ],
                }
            )
        )
    )
    probe = routes.handlers[("POST", "/remote/environments/probe")]
    probe_response = asyncio.run(
        probe(FakeRequest({"environment_id": "gpu-one"}))
    )
    vast_verify = routes.handlers[("POST", "/remote/vast/verify")]
    vast_verify_response = asyncio.run(vast_verify(FakeRequest({})))
    r2_usage = routes.handlers[("POST", "/remote/storage/r2/usage")]
    r2_usage_response = asyncio.run(
        r2_usage(
            FakeRequest(
                {
                    "configuration_id": "385",
                    "display_name": "Shared R2",
                    "account_id": "a" * 32,
                    "bucket": "models",
                    "credential_id": "opaque-reference",
                    "jurisdiction": "eu",
                }
            )
        )
    )
    r2_unlock = routes.handlers[("POST", "/remote/storage/r2/keychain/unlock")]
    r2_unlock_response = asyncio.run(r2_unlock(FakeRequest({})))
    monkeypatch.setattr(
        api_intercept_module,
        "_refresh_r2_storage_usage",
        lambda _storage: (_ for _ in ()).throw(
            api_intercept_module.R2CredentialError(
                "The macOS login keychain must be unlocked.",
                code=api_intercept_module.R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
            )
        ),
    )
    r2_locked_response = asyncio.run(
        r2_usage(
            FakeRequest(
                {
                    "configuration_id": "385",
                    "display_name": "Shared R2",
                    "account_id": "a" * 32,
                    "bucket": "models",
                    "credential_id": "opaque-reference",
                    "jurisdiction": "eu",
                }
            )
        )
    )

    assert update_response.status == 200
    assert probe_response.status == 200
    assert vast_verify_response.status == 400
    assert r2_usage_response.status == 200
    assert r2_unlock_response.status == 200
    assert unlock_requests == [True]
    assert r2_locked_response.status == 423
    assert json.loads(r2_locked_response.text)["code"] == "keychain_unlock_required"
    assert json.loads(r2_usage_response.text)["storage_usage_bytes"] == 7 * 1024**3
    assert json.loads(r2_usage_response.text)["storage_object_count"] == 17
    assert ("GET", "/remote/vast/status") in routes.handlers
    assert ("POST", "/remote/vast/reap") in routes.handlers
    assert ("POST", "/remote/vast/destroy") in routes.handlers
    assert ("POST", "/modal/container_stop") in routes.handlers
    assert ("POST", "/remote/storage/r2/oauth/start") in routes.handlers
    assert ("GET", "/remote/storage/r2/oauth/callback") in routes.handlers
    assert ("POST", "/remote/storage/r2/credentials") in routes.handlers
    assert ("GET", "/remote/storage/r2/status") in routes.handlers
    assert ("POST", "/remote/storage/r2/usage") in routes.handlers
    assert ("POST", "/remote/storage/r2/keychain/unlock") in routes.handlers
    assert registry.get_host("gpu-one").health.value == "ready"
    assert registry.get_host("gpu-one").capabilities == capabilities


def test_scheduler_refreshes_recent_ssh_capabilities_before_placement(
    api_intercept_module: Any,
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

    monkeypatch.setattr(api_intercept_module, "_ssh_host_registry", lambda _settings: registry)
    monkeypatch.setattr(api_intercept_module, "SshDockerController", FakeController)

    hosts = api_intercept_module._schedulable_ssh_hosts(SimpleNamespace())

    assert hosts[0].health.value == "ready"
    assert hosts[0].capabilities == capabilities


def test_remote_partition_preserves_dag_around_ssh_only_llm(
    api_intercept_module: Any,
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

    component_groups = api_intercept_module._remote_component_partition_groups(
        prompt,
        set(prompt),
        api_intercept_module._build_consumer_map(prompt),
        fake_nodes_module,
    )
    components = api_intercept_module._component_topological_order(
        prompt,
        component_groups,
    )

    assert components == [["1", "2"], ["3"], ["4"]]
    assert api_intercept_module._component_execution_stages(
        prompt,
        component_groups,
    ) == [["1"], ["3"], ["4"]]


def test_remote_partition_replicates_non_transportable_fanout_around_ssh_llm(
    api_intercept_module: Any,
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
    component_plans = api_intercept_module._build_component_plans(
        prompt,
        remote_node_ids,
        fake_nodes_module,
    )
    api_intercept_module.validate_remote_component_transport_compatibility(
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
        if node_id.startswith(api_intercept_module._REMOTE_REPLICA_NODE_PREFIX)
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
    required_provider = api_intercept_module._component_required_provider(
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
        api_intercept_module,
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
        node_id.startswith(api_intercept_module._REMOTE_REPLICA_NODE_PREFIX)
        for node_id in rewritten_prompt
    )
    assert summary.component_node_ids_by_representative["4"] == ["4"]
    assert not any(
        node_id.startswith(api_intercept_module._REMOTE_REPLICA_NODE_PREFIX)
        for node_id in summary.rewritten_node_id_map
    )
    downstream_payload = rewritten_prompt["4"]["inputs"]["original_node_data"]
    replica_payload_node_ids = {
        node_id
        for node_id in downstream_payload["subgraph_prompt"]
        if node_id.startswith(api_intercept_module._REMOTE_REPLICA_NODE_PREFIX)
    }
    assert len(replica_payload_node_ids) == 1
    downstream_vae_input = downstream_payload["subgraph_prompt"]["4"]["inputs"][
        "vae"
    ]
    assert downstream_vae_input[0] in replica_payload_node_ids


def test_remote_partition_replicates_linked_model_loader_closure(
    api_intercept_module: Any,
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

    component_plans = api_intercept_module._build_component_plans(
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
        if node_id.startswith(api_intercept_module._REMOTE_REPLICA_NODE_PREFIX)
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
    api_intercept_module.validate_remote_component_transport_compatibility(
        prompt,
        component_plans,
        fake_nodes_module,
    )


def test_modal_only_policy_rejects_ssh_only_llm_backend(
    api_intercept_module: Any,
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
        api_intercept_module._plan_component_execution_assignments(
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


def test_cross_provider_boundary_uses_transport_instead_of_remote_session(
    api_intercept_module: Any,
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
        api_intercept_module._mark_remote_to_remote_session_boundaries(
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
        api_intercept_module._mark_remote_to_remote_session_boundaries(
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
        api_intercept_module._mark_remote_to_remote_session_boundaries(
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


def test_automatic_policy_assigns_component_to_lower_cost_ready_host(
    api_intercept_module: Any,
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
        api_intercept_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = api_intercept_module._plan_component_execution_assignments(
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
        api_intercept_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(api_intercept_module, "SshDockerController", FakeController)
    monkeypatch.setattr(
        api_intercept_module,
        "_ssh_host_registry",
        lambda _settings: None,
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = api_intercept_module._plan_component_execution_assignments(
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

    assert not api_intercept_module._reclaim_improves_assignment(
        optimistic,
        actual,
        module.ComponentResourceRequirements(),
    )


def test_automatic_policy_rejects_zero_cost_host_for_oversized_model(
    api_intercept_module: Any,
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
    estimate = api_intercept_module._component_memory_estimate(
        component,
        prompt,
        preferences,
        settings,
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = api_intercept_module._plan_component_execution_assignments(
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
    monkeypatch.setattr(llm_resolver_module, "resolve_model_profile", resolve)
    monkeypatch.setattr(
        api_intercept_module,
        "_schedulable_ssh_hosts",
        lambda _settings: (host,),
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_execution_history",
        lambda _settings: None,
    )

    assignments = api_intercept_module._plan_component_execution_assignments(
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

    analysis = api_intercept_module.analyze_remote_node_selection(
        prompt,
        workflow,
        [],
        settings=SimpleNamespace(marker_property="is_modal_remote"),
        nodes_module=fake_nodes_module,
    )

    assert analysis.resolved_remote_node_ids == ["1", "3"]
    assert analysis.sandwiched_local_node_ids == ["2"]
    assert analysis.added_node_ids == []


def _artifact_finalizer_node_id(summary: Any) -> str:
    """Return the finalizer id after asserting that prompt rewrite attached it."""
    finalizer_node_id = summary.artifact_finalizer_node_id
    assert isinstance(finalizer_node_id, str)
    assert finalizer_node_id
    return finalizer_node_id


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


def test_rewritten_prompt_diagnostics_reports_dependency_cycles(
    api_intercept_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewritten prompt diagnostics should name local dependency cycles before Comfy executes."""
    prompt = {
        "1": {
            "class_type": "ModalUniversalExecutor_a",
            "inputs": {"remote_input_0": ["2", 0]},
        },
        "2": {
            "class_type": "ModalUniversalExecutor_b",
            "inputs": {"remote_input_0": ["1", 0]},
        },
    }

    diagnostics = api_intercept_module._modal_rewritten_prompt_diagnostics(prompt)

    assert diagnostics["cycles"] == [["1", "2", "1"]]

    warning_messages: list[str] = []
    log_messages: list[str] = []

    def record_warning(message: str, *args: Any, **_kwargs: Any) -> None:
        """Record one warning log message."""
        warning_messages.append(message % args)

    def record_log(_level: int, message: str, *args: Any, **_kwargs: Any) -> None:
        """Record one generic log message."""
        log_messages.append(message % args)

    monkeypatch.setattr(api_intercept_module.logger, "warning", record_warning)
    monkeypatch.setattr(api_intercept_module.logger, "log", record_log)

    api_intercept_module._log_modal_rewritten_prompt_diagnostics(
        prompt_id="prompt-cycle",
        prompt=prompt,
        reason="test",
    )

    assert any("Modal rewritten prompt contains dependency cycle(s)" in item for item in warning_messages)
    assert any("prompt-cycle" in item for item in warning_messages)
    assert any("Modal rewritten prompt diagnostics" in item for item in log_messages)


def test_queue_prompt_json_logs_rewritten_modal_diagnostics_on_validation_failure(
    api_intercept_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation failures for Modal prompts should log the rewritten dependency graph."""

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def put(self, _item: tuple[Any, ...]) -> None:
            """Fail if an invalid prompt reaches the queue."""
            raise AssertionError("invalid prompt must not be queued")

    class FakePromptServer:
        """Minimal PromptServer double for validation-failure tests."""

        number = 0
        prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeExecutionModule:
        """Minimal execution module that rejects the prompt."""

        SENSITIVE_EXTRA_DATA_KEYS: tuple[str, ...] = ()

        @staticmethod
        async def validate_prompt(
            prompt_id: str,
            prompt: dict[str, Any],
            partial_execution_targets: Any,
        ) -> tuple[bool, dict[str, Any], list[str], dict[str, Any]]:
            """Reject the supplied prompt with a dependency-cycle shaped error."""
            del prompt_id, prompt, partial_execution_targets
            return (
                False,
                {
                    "type": "execution_error",
                    "message": "Dependency cycle detected",
                    "details": "",
                    "extra_info": {},
                },
                [],
                {},
            )

    prompt = {
        "1": {
            "class_type": "ModalUniversalExecutor_a",
            "inputs": {"remote_input_0": ["2", 0]},
        },
        "2": {
            "class_type": "ModalUniversalExecutor_b",
            "inputs": {"remote_input_0": ["1", 0]},
        },
    }

    diagnostic_calls: list[dict[str, Any]] = []

    def record_diagnostics(**kwargs: Any) -> None:
        """Record one rewritten-prompt diagnostics request."""
        diagnostic_calls.append(dict(kwargs))

    monkeypatch.setattr(api_intercept_module, "_get_execution_module", lambda: FakeExecutionModule)
    monkeypatch.setattr(
        api_intercept_module,
        "_log_modal_rewritten_prompt_diagnostics",
        record_diagnostics,
    )

    response = asyncio.run(
        api_intercept_module._queue_prompt_json(
            FakePromptServer(),
            {
                "prompt_id": "prompt-cycle",
                "prompt": prompt,
                "extra_data": {
                    "modal": {
                        "remote_component_ids": ["1", "2"],
                    }
                },
            },
        )
    )

    assert response.status == 400
    assert diagnostic_calls == [
        {
            "prompt_id": "prompt-cycle",
            "prompt": prompt,
            "reason": "comfy_validation_failure",
            "level": api_intercept_module.logging.WARNING,
        }
    ]


def test_split_phase_order_accounts_for_local_feedback_dependencies(
    api_intercept_module: Any,
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
        api_intercept_module._order_execute_node_ids_for_transportable_splits(
            prompt=prompt,
            component_prompt=component_prompt,
            component_node_ids={"3", "14", "191", "358"},
            execute_node_ids=["191", "3"],
        )
    )

    assert ordered_execute_node_ids == ["3", "191"]


def test_queue_prompt_route_does_not_warm_modal_at_queue_time(
    api_intercept_module: Any,
    remote_modal_app_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Accepting a queued prompt should not launch Modal warmup containers."""

    class FakeRoutes:
        """Capture aiohttp route registrations."""

        def __init__(self) -> None:
            """Initialize the route handler map."""
            self.handlers: dict[str, Any] = {}

        def post(self, path: str) -> Any:
            """Return a decorator that records one POST handler."""

            def register(handler: Any) -> Any:
                """Store the decorated handler unchanged."""
                self.handlers[path] = handler
                return handler

            return register

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def __init__(self) -> None:
            """Initialize captured queue items."""
            self.items: list[tuple[Any, ...]] = []

        def put(self, item: tuple[Any, ...]) -> None:
            """Record one queued prompt item."""
            self.items.append(item)

    class FakePromptServer:
        """Minimal PromptServer double with route registration."""

        def __init__(self) -> None:
            """Initialize routing and queue state."""
            self.number = 0
            self.routes = FakeRoutes()
            self.prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeRequest:
        """Minimal aiohttp request double."""

        async def json(self) -> dict[str, Any]:
            """Return one Modal-marked prompt request."""
            return {
                "prompt_id": "prompt-queue-warmup",
                "prompt": {"1": {"class_type": "RemoteImage", "inputs": {}}},
                "extra_data": {
                    "extra_pnginfo": {
                        "workflow": {
                            "extra": {"comfy_modal": {"gpu": "B300"}},
                            "nodes": [
                                {
                                    "id": 1,
                                    "type": "RemoteImage",
                                    "properties": {"is_modal_remote": True},
                                }
                            ]
                        }
                    }
                },
            }

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

    def fail_queue_time_warmup(*_args: Any, **_kwargs: Any) -> int:
        """Fail the test if queue handling tries to launch proactive warmup."""
        raise AssertionError("queue route must not schedule Modal warmup")

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
    )
    prompt_server = FakePromptServer()
    summary = api_intercept_module.RewriteSummary(
        remote_node_ids=["1"],
        remote_component_ids=["1"],
        component_node_ids_by_representative={"1": ["1"]},
        sandwiched_local_node_ids=["4"],
        component_execution_stages=[["1"]],
        estimated_max_parallel_requests=1,
        max_parallel_requests_upper_bound=1,
        execution_assignments_by_representative={
            "1": api_intercept_module.ExecutionAssignment(
                environment_id="modal:B300",
                provider=api_intercept_module.ExecutionProvider.MODAL,
                predicted_cost_usd=0.01,
                predicted_completion_seconds=10.0,
            )
        },
    )
    observed_rewrite_settings: list[Any] = []

    def capture_rewrite_settings(**kwargs: Any) -> tuple[dict[str, Any], Any]:
        """Capture the workflow-derived settings passed into prompt rewriting."""
        observed_rewrite_settings.append(kwargs["settings"])
        return kwargs["prompt"], summary

    monkeypatch.setattr(api_intercept_module, "_ROUTE_REGISTERED", False)
    monkeypatch.setattr(
        api_intercept_module,
        "_get_server_module",
        lambda: SimpleNamespace(PromptServer=SimpleNamespace(instance=prompt_server)),
    )
    monkeypatch.setattr(api_intercept_module, "_get_execution_module", lambda: FakeExecutionModule)
    monkeypatch.setattr(api_intercept_module, "_emit_modal_status", lambda **_kwargs: None)
    monkeypatch.setattr(
        api_intercept_module,
        "rewrite_prompt_for_modal",
        capture_rewrite_settings,
    )
    monkeypatch.setattr(remote_modal_app_module, "ensure_remote_warm_capacity", fail_queue_time_warmup)

    api_intercept_module.setup_modal_queue_route(
        prompt_server=prompt_server,
        sync_engine=object(),
        settings=settings,
    )
    response = asyncio.run(prompt_server.routes.handlers["/modal/queue_prompt"](FakeRequest()))

    response_payload = json.loads(response.text)
    assert response_payload["prompt_id"] == "prompt-queue-warmup"
    assert response_payload["modal_gpu"] == "B300"
    assert response_payload["modal_remote_node_ids"] == ["1"]
    assert response_payload["modal_sandwiched_local_node_ids"] == ["4"]
    assert observed_rewrite_settings[0].modal_gpu == "B300"
    queued_extra_data = prompt_server.prompt_queue.items[0][3]
    assert queued_extra_data["modal"]["gpu"] == "B300"
    assert queued_extra_data["extra_pnginfo"][
        api_intercept_module.MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY
    ] == "prompt-queue-warmup"
    assert len(prompt_server.prompt_queue.items) == 1


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


def test_remote_execution_configurator_identity_is_preserved(
    api_intercept_module: Any,
) -> None:
    """Queue-time UI events should address the exact serialized configurator node."""
    prompt = {
        "12": {"class_type": "KSampler", "inputs": {}},
        "99": {
            "class_type": "RemoteExecutionConfigurator",
            "inputs": {"configurations.configuration_0": ["20", 0]},
        },
    }

    assert (
        api_intercept_module._remote_execution_configurator_node_id(prompt)
        == "99"
    )
    assert api_intercept_module._remote_execution_configurator_node_id(
        {
            **prompt,
            "100": {
                "class_type": "RemoteExecutionConfigurator",
                "inputs": {},
            },
        }
    ) is None


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


def test_modal_prompt_rewrite_keeps_event_loop_responsive(
    api_intercept_module: Any,
    monkeypatch: Any,
) -> None:
    """Hashing and upload preparation should execute outside the ComfyUI event loop."""
    rewrite_started = threading.Event()
    release_rewrite = threading.Event()

    def blocking_rewrite(**kwargs: Any) -> tuple[dict[str, Any], Any]:
        """Hold one fake rewrite until the async test proves the loop is responsive."""
        rewrite_started.set()
        assert release_rewrite.wait(timeout=1.0)
        return kwargs["prompt"], api_intercept_module.RewriteSummary()

    monkeypatch.setattr(api_intercept_module, "rewrite_prompt_for_modal", blocking_rewrite)

    async def run_test() -> None:
        """Run the blocking rewrite and an independent event-loop callback together."""
        rewrite_task = asyncio.create_task(
            api_intercept_module.rewrite_prompt_for_modal_async(
                prompt={"1": {"class_type": "RemoteImage", "inputs": {}}},
                workflow=None,
            )
        )
        deadline = asyncio.get_running_loop().time() + 1.0
        while not rewrite_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("background rewrite did not start")
            await asyncio.sleep(0)

        loop_progress: list[str] = []
        asyncio.get_running_loop().call_soon(loop_progress.append, "responsive")
        await asyncio.sleep(0)
        assert loop_progress == ["responsive"]
        assert not rewrite_task.done()

        release_rewrite.set()
        await asyncio.wait_for(rewrite_task, timeout=1.0)

    asyncio.run(run_test())


def test_queue_prompt_route_without_remote_nodes_skips_modal_status_and_rewrite(
    api_intercept_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Prompts with no Modal-enabled workflow nodes should queue without Modal UI setup."""

    class FakeRoutes:
        """Capture aiohttp route registrations."""

        def __init__(self) -> None:
            """Initialize the route handler map."""
            self.handlers: dict[str, Any] = {}

        def post(self, path: str) -> Any:
            """Return a decorator that records one POST handler."""

            def register(handler: Any) -> Any:
                """Store the decorated handler unchanged."""
                self.handlers[path] = handler
                return handler

            return register

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def __init__(self) -> None:
            """Initialize captured queue items."""
            self.items: list[tuple[Any, ...]] = []

        def put(self, item: tuple[Any, ...]) -> None:
            """Record one queued prompt item."""
            self.items.append(item)

    class FakePromptServer:
        """Minimal PromptServer double with route registration."""

        def __init__(self) -> None:
            """Initialize routing and queue state."""
            self.number = 0
            self.routes = FakeRoutes()
            self.prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeRequest:
        """Minimal aiohttp request double."""

        async def json(self) -> dict[str, Any]:
            """Return one ordinary prompt request."""
            return {
                "prompt_id": "prompt-no-modal",
                "prompt": {"1": {"class_type": "LocalImage", "inputs": {}}},
                "extra_data": {"extra_pnginfo": {"workflow": {"nodes": []}}},
            }

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

    def fail_modal_status(*_args: Any, **_kwargs: Any) -> None:
        """Fail if the no-remote fast path emits Modal UI state."""
        raise AssertionError("no-remote prompts must not emit Modal status")

    def fail_rewrite(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
        """Fail if the no-remote fast path enters Modal prompt rewriting."""
        raise AssertionError("no-remote prompts must not be rewritten for Modal")

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
    )
    prompt_server = FakePromptServer()
    monkeypatch.setattr(api_intercept_module, "_ROUTE_REGISTERED", False)
    monkeypatch.setattr(
        api_intercept_module,
        "_get_server_module",
        lambda: SimpleNamespace(PromptServer=SimpleNamespace(instance=prompt_server)),
    )
    monkeypatch.setattr(api_intercept_module, "_get_execution_module", lambda: FakeExecutionModule)
    monkeypatch.setattr(api_intercept_module, "_emit_modal_status", fail_modal_status)
    monkeypatch.setattr(api_intercept_module, "rewrite_prompt_for_modal", fail_rewrite)

    api_intercept_module.setup_modal_queue_route(
        prompt_server=prompt_server,
        sync_engine=object(),
        settings=settings,
    )
    response = asyncio.run(prompt_server.routes.handlers["/modal/queue_prompt"](FakeRequest()))

    response_payload = json.loads(response.text)
    assert response_payload["prompt_id"] == "prompt-no-modal"
    assert "modal_remote_node_ids" not in response_payload
    assert len(prompt_server.prompt_queue.items) == 1


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


def test_component_local_reentry_dependency_detection(
    api_intercept_module: Any,
) -> None:
    """Boundary inputs that trace back to the same component require a split-capable proxy."""
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
            "_meta": {"title": "Remote Consumer"},
        },
    }
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["1", "2"],
        representative_node_id="1",
        boundary_inputs=[
            api_intercept_module.BoundaryInputSpec(
                proxy_input_name="remote_input_0",
                source=api_intercept_module.LinkedOutputRef(node_id="4", output_index=0),
                io_type="IMAGE",
                targets=[
                    api_intercept_module.InputTarget(
                        node_id="2",
                        input_name="image",
                    )
                ],
            )
        ],
        boundary_outputs=[],
        execute_node_ids=["1", "2"],
        contains_output_node=False,
        local_tap_node_ids=["9"],
    )

    assert api_intercept_module._component_has_local_reentry_dependency(
        prompt=prompt,
        component=component,
    )


def test_sandwiched_local_nodes_include_only_remote_reentry_paths(
    api_intercept_module: Any,
) -> None:
    """Planner warnings should cover local chains that leave and re-enter remote work."""
    prompt = {
        "1": {"class_type": "RemoteSource", "inputs": {}},
        "2": {"class_type": "LocalTransform", "inputs": {"value": ["1", 0]}},
        "3": {
            "class_type": "LocalTransform",
            "inputs": {"value": ["2", 0], "local_only": ["7", 0]},
        },
        "4": {"class_type": "RemoteSink", "inputs": {"value": ["3", 0]}},
        "5": {"class_type": "LocalPreview", "inputs": {"value": ["1", 0]}},
        "6": {"class_type": "RemoteSink", "inputs": {"value": ["8", 0]}},
        "7": {"class_type": "LocalSource", "inputs": {}},
        "8": {"class_type": "LocalSource", "inputs": {}},
        "9": {"class_type": "RemoteSink", "inputs": {"value": ["1", 1]}},
    }

    assert api_intercept_module._sandwiched_local_node_ids(
        prompt,
        {"1", "4", "6", "9"},
    ) == {"2", "3"}


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


def test_snapshot_profile_stamping_excludes_llm_phase_from_comfy_profile(
    api_intercept_module: Any,
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

    result = api_intercept_module._attach_snapshot_profile_key(split_payload, settings)

    snapshot_profile_key = result["snapshot_profile_key"]
    phases = result["split_proxy_payloads"]
    assert snapshot_profile_key.startswith("loader-profile:")
    assert phases[0]["snapshot_profile_key"] == snapshot_profile_key
    assert "snapshot_profile_key" not in phases[1]
    assert phases[2]["snapshot_profile_key"] == snapshot_profile_key


def test_planner_resolved_llm_profile_is_attached_to_matching_payload(
    api_intercept_module: Any,
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

    api_intercept_module._attach_resolved_llm_profiles(
        payload,
        {profile.profile_id: profile},
        SimpleNamespace(local_storage_root=tmp_path),
    )

    entry = payload["resolved_llm_profiles"][profile.profile_id]
    assert entry["profile"] == profile.to_mapping()
    assert entry["security_scan_complete"] is True


def test_planner_attaches_next_distinct_affinity_as_speculative_prewarm_target(
    api_intercept_module: Any,
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

    api_intercept_module._configure_speculative_affinity_prewarm_payloads(
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

    api_intercept_module._configure_local_gap_keepalive_payloads(
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


def test_extract_remote_node_ids_prefers_visible_toggle_over_stale_property(
    api_intercept_module: Any,
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

    assert api_intercept_module.extract_remote_node_ids(workflow, settings) == {"10"}


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


def test_extract_remote_node_ids_maps_defined_subgraph_nodes_through_instances(
    api_intercept_module: Any,
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

    assert api_intercept_module.extract_remote_node_ids(
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

    assert api_intercept_module.extract_remote_node_ids(
        workflow,
        settings,
        prompt_node_ids={"105:7:11", "300"},
    ) == {"105:7:11"}


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


def test_analyze_remote_node_selection_reports_local_reentry(
    api_intercept_module: Any,
    settings_module: Any,
) -> None:
    """Dry-run analysis should report local nodes between existing remote regions."""
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
                "RemoteString": _FakeRemoteStringEchoNode,
                "LocalString": _FakeLocalStringSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
        ]
    }
    prompt = {
        "1": {"class_type": "RemoteString", "inputs": {}},
        "2": {"class_type": "LocalString", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "RemoteString", "inputs": {"text": ["2", 0]}},
    }

    analysis = api_intercept_module.analyze_remote_node_selection(
        prompt=prompt,
        workflow=workflow,
        seed_workflow_node_paths=[],
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert analysis.resolved_remote_node_ids == ["1", "3"]
    assert analysis.sandwiched_local_node_ids == ["2"]


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
    assert "cannot cross the current component boundary" in message


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

    assert set(rewritten_prompt) == {
        "27",
        "195:27",
        "223",
        _artifact_finalizer_node_id(summary),
    }
    assert summary.remote_node_ids == ["195:27", "222"]
    assert summary.remote_component_ids == ["195:27"]


def test_emit_modal_status_targets_prompt_client(
    api_intercept_module: Any,
) -> None:
    """Modal status events should preserve prompt and component metadata for the UI."""
    api_intercept_module._MODAL_UI_EVENTS_BY_CLIENT.clear()

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
        configurator_node_id="99",
        modal_gpu="B300",
        component_node_ids_by_representative={"4": ["4", "5"]},
        active_node_id="5",
        active_node_class_type="KSampler",
        active_node_role="sampling",
        execution_environment_id="vast:48602895",
        remote_execution_assignments={
            "4": {"provider": "vast", "node_ids": ["4", "5"]}
        },
        remote_execution_configurations=[
            {"configuration_id": "vast-big", "display_name": "Vast Big"}
        ],
    )

    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["4", "5"],
                "configurator_node_id": "99",
                "modal_gpu": "B300",
                "active_node_id": "5",
                "active_node_class_type": "KSampler",
                "active_node_role": "sampling",
                "execution_environment_id": "vast:48602895",
                "components": [
                    {
                        "representative_node_id": "4",
                        "node_ids": ["4", "5"],
                    }
                ],
                "remote_execution_assignments": {
                    "4": {"provider": "vast", "node_ids": ["4", "5"]}
                },
                "remote_execution_configurations": [
                    {
                        "configuration_id": "vast-big",
                        "display_name": "Vast Big",
                    }
                ],
            },
            "client-1",
        )
    ]
    replay_events = api_intercept_module.modal_ui_events_for_client("client-1")
    assert replay_events == [
        {
            "event": "modal_status",
            "payload": {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["4", "5"],
                "configurator_node_id": "99",
                "modal_gpu": "B300",
                "active_node_id": "5",
                "active_node_class_type": "KSampler",
                "active_node_role": "sampling",
                "execution_environment_id": "vast:48602895",
                "components": [
                    {
                        "representative_node_id": "4",
                        "node_ids": ["4", "5"],
                    }
                ],
                "remote_execution_assignments": {
                    "4": {"provider": "vast", "node_ids": ["4", "5"]}
                },
                "remote_execution_configurations": [
                    {
                        "configuration_id": "vast-big",
                        "display_name": "Vast Big",
                    }
                ],
            },
            "updated_at": replay_events[0]["updated_at"],
        }
    ]


def test_environment_setup_status_callback_preserves_environment_identity(
    api_intercept_module: Any,
) -> None:
    """Environment setup updates should also retain prompt-wide progress."""
    prompt_updates: list[tuple[str, int | None, int | None]] = []
    environment_updates: list[tuple[str, str, int | None, int | None]] = []
    callback = api_intercept_module._environment_setup_status_callback(
        "vast:48602895",
        lambda message, current, total: prompt_updates.append(
            (message, current, total)
        ),
        lambda environment_id, message, current, total: environment_updates.append(
            (environment_id, message, current, total)
        ),
    )

    assert callback is not None
    callback("Uploading asset", 3, 10)

    assert prompt_updates == [("Uploading asset", 3, 10)]
    assert environment_updates == [
        ("vast:48602895", "Uploading asset", 3, 10)
    ]


def test_remote_environment_assets_are_prepared_in_parallel(
    api_intercept_module: Any,
) -> None:
    """Distinct environments should enter asset preparation concurrently."""
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    active_environment_count = 0
    maximum_active_environment_count = 0
    environment_events: list[tuple[str, str]] = []

    class FakeSyncEngine:
        """Block custom-node setup until both environment workers are active."""

        def __init__(self) -> None:
            """Record the environment-local preparation order."""
            self.calls: list[str] = []

        def preflight_r2_access(self, *, status_callback: Any) -> None:
            """Represent an environment without configured R2 backing."""
            del status_callback
            self.calls.append("r2_preflight")

        def sync_custom_nodes_directory(self, *, status_callback: Any) -> None:
            """Prove the second environment starts before the first can finish."""
            nonlocal active_environment_count, maximum_active_environment_count
            self.calls.append("custom_nodes")
            with state_lock:
                active_environment_count += 1
                maximum_active_environment_count = max(
                    maximum_active_environment_count,
                    active_environment_count,
                )
            status_callback("Uploading custom nodes", None, None)
            try:
                barrier.wait(timeout=2.0)
            finally:
                with state_lock:
                    active_environment_count -= 1

        def create_request_asset_cache(self, values: Any) -> object:
            """Consume the environment-local input plan and return a sentinel cache."""
            tuple(values)
            self.calls.append("asset_plan")
            return object()

        def sync_prompt_inputs(
            self,
            inputs: dict[str, Any],
            *,
            status_callback: Any,
            request_cache: object,
        ) -> tuple[dict[str, Any], list[Any]]:
            """Return one prepared prompt after publishing environment-local status."""
            assert request_cache is not None
            self.calls.append("prompt_assets")
            status_callback("Downloading prompt asset", 1, 1)
            return inputs, []

    components = [
        SimpleNamespace(representative_node_id="a", node_ids=["a"]),
        SimpleNamespace(representative_node_id="b", node_ids=["b"]),
    ]
    assignments = {
        "a": SimpleNamespace(
            environment_id="vast:big:1",
            provider=api_intercept_module.ExecutionProvider.VAST,
        ),
        "b": SimpleNamespace(
            environment_id="lambda",
            provider=api_intercept_module.ExecutionProvider.SSH_DOCKER,
        ),
    }
    vast_engine = FakeSyncEngine()
    ssh_engine = FakeSyncEngine()
    results = api_intercept_module._prepare_remote_environment_assets(
        components=components,
        assignments_by_component_id=assignments,
        sync_engines_by_environment={
            "vast:big:1": vast_engine,
            "lambda": ssh_engine,
        },
        rewritten_prompt={
            "a": {"class_type": "VAELoader", "inputs": {"vae": "a.safetensors"}},
            "b": {"class_type": "VAELoader", "inputs": {"vae": "b.safetensors"}},
        },
        sync_custom_nodes=True,
        status_callback=None,
        environment_status_callback=(
            lambda environment_id, message, _current, _total: environment_events.append(
                (environment_id, message)
            )
        ),
    )

    assert maximum_active_environment_count == 2
    assert list(results) == ["vast:big:1", "lambda"]
    assert results["vast:big:1"].component_prompts["a"]["a"]["inputs"] == {
        "vae": "a.safetensors"
    }
    assert results["lambda"].component_prompts["b"]["b"]["inputs"] == {
        "vae": "b.safetensors"
    }
    assert vast_engine.calls == [
        "r2_preflight",
        "custom_nodes",
        "asset_plan",
        "prompt_assets",
    ]
    assert ssh_engine.calls == vast_engine.calls
    for environment_id in results:
        messages = [
            message
            for event_environment_id, message in environment_events
            if event_environment_id == environment_id
        ]
        expected_completion = (
            "Ready for remote execution"
            if environment_id == "vast:big:1"
            else "Remote assets prepared; SSH runtime starts on dispatch"
        )
        assert messages == [
            "Preparing remote assets",
            "Uploading custom nodes",
            "Downloading prompt asset",
            expected_completion,
        ]


def test_remote_environment_asset_worker_failures_bubble_up(
    api_intercept_module: Any,
) -> None:
    """A failed environment worker must fail queue preparation with its cause."""

    class FailingSyncEngine:
        """Fail before prompt assets are considered prepared."""

        def preflight_r2_access(self, *, status_callback: Any) -> None:
            """Represent an environment without configured R2 backing."""
            del status_callback

        def sync_custom_nodes_directory(self, *, status_callback: Any) -> None:
            """Raise the representative environment-specific setup failure."""
            status_callback("Uploading custom nodes", None, None)
            raise OSError("remote storage is unavailable")

    with pytest.raises(OSError, match="remote storage is unavailable"):
        api_intercept_module._prepare_remote_environment_assets(
            components=[
                SimpleNamespace(representative_node_id="a", node_ids=["a"])
            ],
            assignments_by_component_id={
                "a": SimpleNamespace(
                    environment_id="vast:broken:1",
                    provider=api_intercept_module.ExecutionProvider.VAST,
                )
            },
            sync_engines_by_environment={
                "vast:broken:1": FailingSyncEngine()
            },
            rewritten_prompt={
                "a": {"class_type": "VAELoader", "inputs": {}}
            },
            sync_custom_nodes=True,
            status_callback=None,
            environment_status_callback=lambda *_args: None,
        )


def test_workflow_ssh_metadata_preserves_probed_gpu_capabilities(
    api_intercept_module: Any,
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

    metadata = api_intercept_module._configured_provider_metadata(
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


def test_modal_ui_event_replay_is_client_scoped(api_intercept_module: Any) -> None:
    """Refocus replay should only return events for the requesting ComfyUI client."""
    api_intercept_module._MODAL_UI_EVENTS_BY_CLIENT.clear()

    api_intercept_module.record_modal_ui_event(
        "modal_progress",
        {"prompt_id": "prompt-1", "node_id": "4", "value": 2.0, "max": 10.0},
        "client-1",
    )
    api_intercept_module.record_modal_ui_event(
        "modal_status",
        {"prompt_id": "prompt-2", "phase": "executing", "node_ids": ["9"]},
        "client-2",
    )

    replay_events = api_intercept_module.modal_ui_events_for_client("client-1")

    assert len(replay_events) == 1
    assert replay_events[0]["event"] == "modal_progress"
    assert replay_events[0]["payload"] == {
        "prompt_id": "prompt-1",
        "node_id": "4",
        "value": 2.0,
        "max": 10.0,
    }
    assert api_intercept_module.modal_ui_events_for_client(None) == []


def test_progress_state_route_is_queue_route_sibling(api_intercept_module: Any) -> None:
    """The frontend should have a stable sibling route for Modal UI replay."""
    assert api_intercept_module._progress_state_route_path("/modal/queue_prompt") == (
        "/modal/progress_state"
    )
    assert api_intercept_module._progress_state_route_path("/custom/modal") == (
        "/custom/modal/progress_state"
    )


def test_container_status_route_is_queue_route_sibling(api_intercept_module: Any) -> None:
    """The frontend should have a stable sibling route for active Modal containers."""
    assert api_intercept_module._container_status_route_path("/modal/queue_prompt") == (
        "/modal/container_status"
    )


def test_cancel_preparation_route_is_queue_route_sibling(
    api_intercept_module: Any,
) -> None:
    """Queue-time cancellation should use a stable sibling route."""
    assert api_intercept_module._cancel_preparation_route_path(
        "/modal/queue_prompt"
    ) == "/modal/cancel_preparation"
    assert api_intercept_module._cancel_preparation_route_path(
        "/custom/modal"
    ) == "/custom/modal/cancel_preparation"
    assert api_intercept_module._container_status_route_path("/custom/modal") == (
        "/custom/modal/container_status"
    )


def test_modal_reset_route_paths_are_queue_route_siblings(api_intercept_module: Any) -> None:
    """The frontend should have stable sibling routes for Modal maintenance actions."""
    assert api_intercept_module._delete_modal_caches_route_path("/modal/queue_prompt") == (
        "/modal/delete_caches"
    )
    assert api_intercept_module._delete_modal_volume_route_path("/modal/queue_prompt") == (
        "/modal/delete_volume"
    )
    assert api_intercept_module._delete_modal_caches_route_path("/custom/modal") == (
        "/custom/modal/delete_caches"
    )
    assert api_intercept_module._delete_modal_volume_route_path("/custom/modal") == (
        "/custom/modal/delete_volume"
    )


def test_delete_modal_cache_dicts_deletes_configured_dicts(
    api_intercept_module: Any,
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
    monkeypatch.setattr(api_intercept_module, "modal", FakeModal)

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
    monkeypatch.setattr(api_intercept_module, "modal", FakeModal)

    result = asyncio.run(api_intercept_module.delete_modal_volume(settings))

    assert result == {"deleted": ["configured-volume"], "skipped": []}
    assert deleted == ["configured-volume"]


def test_modal_interrupt_queue_bridge_exposes_active_remote_prompts(
    api_intercept_module: Any,
    remote_modal_app_module: Any,
) -> None:
    """Targeted ComfyUI interrupts should see prompts currently blocked on Modal work."""

    class FakePromptQueue:
        """Minimal ComfyUI prompt queue with no native running prompts."""

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return an empty running queue and one pending item."""
            return [], ["queued"]

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(prompt_queue=prompt_queue)
    cancellation_event = remote_modal_app_module.threading.Event()

    api_intercept_module._install_modal_interrupt_queue_bridge(prompt_server)
    with remote_modal_app_module._registered_active_remote_invocation(
        {"prompt_id": "prompt-1", "component_id": "component-1"},
        cancellation_event,
        None,
    ):
        running, queued = prompt_queue.get_current_queue()

    assert queued == ["queued"]
    assert [item[1] for item in running] == ["prompt-1"]


def test_remote_preparation_bridge_exposes_work_to_all_queue_views(
    api_intercept_module: Any,
) -> None:
    """Capacity acquisition should look active before the rewritten prompt is queued."""

    class FakePromptQueue:
        """Minimal native queue with every state method used by ComfyUI."""

        def __init__(self) -> None:
            """Initialize an empty native queue."""
            self.running: list[tuple[Any, ...]] = []
            self.queued: list[tuple[Any, ...]] = []

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return stable native queue state."""
            return list(self.running), list(self.queued)

        def get_current_queue_volatile(self) -> tuple[list[Any], list[Any]]:
            """Return volatile native queue state."""
            return list(self.running), list(self.queued)

        def get_tasks_remaining(self) -> int:
            """Count native running and pending prompts."""
            return len(self.running) + len(self.queued)

    prompt_queue = FakePromptQueue()
    queue_update_counts: list[int] = []
    prompt_server = SimpleNamespace(
        prompt_queue=prompt_queue,
        queue_updated=lambda: queue_update_counts.append(
            prompt_queue.get_tasks_remaining()
        ),
    )
    api_intercept_module._install_modal_interrupt_queue_bridge(prompt_server)

    registered = api_intercept_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-preparing",
        prompt={"1": {"class_type": "RemoteImage", "inputs": {}}},
        extra_data={"client_id": "client-1"},
    )

    assert registered is True
    assert prompt_queue.get_tasks_remaining() == 1
    assert [item[1] for item in prompt_queue.get_current_queue()[0]] == [
        "prompt-preparing"
    ]
    assert [item[1] for item in prompt_queue.get_current_queue_volatile()[0]] == [
        "prompt-preparing"
    ]
    preparation_item = prompt_queue.get_current_queue_volatile()[0][0]
    assert preparation_item[3]["client_id"] == "client-1"
    assert isinstance(preparation_item[3]["create_time"], int)
    assert preparation_item[3]["create_time"] > 0

    prompt_queue.queued.append((1, "prompt-preparing", {}, {}, []))
    assert prompt_queue.get_tasks_remaining() == 1
    assert [item[1] for item in prompt_queue.get_current_queue_volatile()[0]] == []

    api_intercept_module._clear_remote_preparation(
        prompt_server,
        "prompt-preparing",
    )

    assert prompt_queue.get_tasks_remaining() == 1
    assert queue_update_counts == [1, 1]


def test_queued_ssh_environment_ids_reads_earlier_prompt_assignments(
    api_intercept_module: Any,
) -> None:
    """Queue-time planning should recognize SSH hosts owned by earlier prompts."""

    def assignment(provider: str, environment_id: str) -> dict[str, str]:
        """Return minimal serialized placement metadata."""
        return {
            "provider": provider,
            "environment_id": environment_id,
        }
    prompt_server = SimpleNamespace(
        prompt_queue=SimpleNamespace(
            get_current_queue=lambda: (
                [
                    (
                        1,
                        "prompt-running",
                        {},
                        {
                            "remote_execution": {
                                "assignments": {
                                    "257": assignment("ssh_docker", "lambda")
                                }
                            }
                        },
                    )
                ],
                [
                    (
                        2,
                        "prompt-current",
                        {},
                        {
                            "remote_execution": {
                                "assignments": {
                                    "300": assignment("ssh_docker", "ignored")
                                }
                            }
                        },
                    ),
                    (
                        3,
                        "prompt-modal",
                        {},
                        {
                            "remote_execution": {
                                "assignments": {
                                    "400": assignment("modal", "modal:H100")
                                }
                            }
                        },
                    ),
                ],
            )
        )
    )

    environment_ids = api_intercept_module._queued_ssh_environment_ids(
        prompt_server,
        excluding_prompt_id="prompt-current",
    )

    assert environment_ids == frozenset({"lambda"})


def test_remote_preparation_bridge_clears_failed_submission(
    api_intercept_module: Any,
) -> None:
    """A rejected pre-queue prompt must not leave phantom active queue work."""

    class FakePromptQueue:
        """Minimal empty queue used to exercise preparation cleanup."""

        def get_current_queue_volatile(self) -> tuple[list[Any], list[Any]]:
            """Return empty native queue state."""
            return [], []

        def get_tasks_remaining(self) -> int:
            """Return the empty native queue count."""
            return 0

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(
        prompt_queue=prompt_queue,
        queue_updated=lambda: None,
    )
    api_intercept_module._install_modal_interrupt_queue_bridge(prompt_server)
    assert api_intercept_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-failed",
        prompt={},
        extra_data={},
    )
    assert prompt_queue.get_tasks_remaining() == 1

    api_intercept_module._clear_remote_preparation(prompt_server, "prompt-failed")

    assert prompt_queue.get_tasks_remaining() == 0
    assert prompt_queue.get_current_queue_volatile() == ([], [])


def test_remote_preparation_bridge_tracks_prompt_cancellation(
    api_intercept_module: Any,
) -> None:
    """Queue-time work should expose a prompt-scoped cancellation event."""

    class FakePromptQueue:
        """Provide the queue method required by the bridge."""

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return no native work."""
            return [], []

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(prompt_queue=prompt_queue)
    cancellation_event = api_intercept_module.threading.Event()
    api_intercept_module._install_modal_interrupt_queue_bridge(prompt_server)

    assert api_intercept_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-cancel",
        prompt={},
        extra_data={},
        cancellation_event=cancellation_event,
    )
    cancellations = getattr(
        prompt_queue,
        api_intercept_module._REMOTE_PREPARATION_CANCELLATIONS_ATTR,
    )
    assert cancellations["prompt-cancel"] is cancellation_event

    api_intercept_module._clear_remote_preparation(prompt_server, "prompt-cancel")

    assert "prompt-cancel" not in cancellations


def test_jobs_api_interrupt_cancels_remote_preparation(
    api_intercept_module: Any,
) -> None:
    """ComfyUI's normal Jobs API cancellation should stop remote setup."""

    class FakePromptQueue:
        """Provide queue and interruption methods used by the bridge."""

        def __init__(self) -> None:
            """Track whether native interruption was used."""
            self.native_interrupts: list[str] = []

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return no native work."""
            return [], []

        def interrupt_if_running(self, prompt_id: str) -> bool:
            """Record native fallback interruptions."""
            self.native_interrupts.append(prompt_id)
            return False

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(prompt_queue=prompt_queue)
    cancellation_event = api_intercept_module.threading.Event()
    api_intercept_module._install_modal_interrupt_queue_bridge(prompt_server)
    api_intercept_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-cancel",
        prompt={},
        extra_data={},
        cancellation_event=cancellation_event,
    )

    running, queued = prompt_queue.get_current_queue()
    assert [item[1] for item in running] == ["prompt-cancel"]
    assert queued == []
    assert prompt_queue.interrupt_if_running("prompt-cancel") is True
    assert cancellation_event.is_set()
    assert prompt_queue.native_interrupts == []
    assert prompt_queue.interrupt_if_running("native-prompt") is False
    assert prompt_queue.native_interrupts == ["native-prompt"]


def test_queue_bridge_releases_r2_writeback_reservations(
    api_intercept_module: Any,
    monkeypatch: Any,
) -> None:
    """Completed, deleted, and wiped prompts should release idle cache work."""

    class FakePromptQueue:
        """Model the native lifecycle methods wrapped by the remote queue bridge."""

        def __init__(self) -> None:
            """Initialize one running prompt and two queued prompts."""
            self.currently_running = {
                7: (0, "prompt-running", {}, {}, [], {}),
            }
            self.queue = [
                (1, "prompt-delete", {}, {}, [], {}),
                (2, "prompt-wipe", {}, {}, [], {}),
            ]

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return native running and queued snapshots."""
            return list(self.currently_running.values()), list(self.queue)

        def task_done(self, item_id: int, *_args: Any, **_kwargs: Any) -> None:
            """Remove one completed running prompt."""
            self.currently_running.pop(item_id)

        def delete_queue_item(self, predicate: Any) -> bool:
            """Delete the first queued item matching a predicate."""
            for index, item in enumerate(self.queue):
                if predicate(item):
                    self.queue.pop(index)
                    return True
            return False

        def wipe_queue(self) -> None:
            """Delete every queued prompt."""
            self.queue.clear()

    released: list[str] = []
    monkeypatch.setattr(
        api_intercept_module,
        "finish_r2_writeback_prompt",
        released.append,
    )
    prompt_queue = FakePromptQueue()
    api_intercept_module._install_modal_interrupt_queue_bridge(
        SimpleNamespace(prompt_queue=prompt_queue)
    )

    prompt_queue.task_done(7, {})
    assert prompt_queue.delete_queue_item(
        lambda item: item[1] == "prompt-delete"
    ) is True
    prompt_queue.wipe_queue()

    assert released == ["prompt-running", "prompt-delete", "prompt-wipe"]


def test_selected_vast_capacity_streams_setup_status(
    api_intercept_module: Any,
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

    leases = api_intercept_module._prepare_selected_vast_capacity(
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
        api_intercept_module._prepare_selected_vast_capacity(
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
