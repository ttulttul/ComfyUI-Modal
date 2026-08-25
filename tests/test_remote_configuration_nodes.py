"""Tests for workflow-scoped remote execution configuration nodes."""

from __future__ import annotations

from typing import Any

import pytest


def _vast_inputs(name: str, maximum_instances: int) -> dict[str, Any]:
    """Return a compact valid Vast configuration input mapping."""
    return {
        "profile_name": name,
        "allocated_disk_gb": 200.0,
        "idle_retention_hours": 24.0,
        "maximum_instances": maximum_instances,
    }


def test_configuration_nodes_expose_typed_outputs_and_autogrow_sink(
    remote_configuration_nodes_module: Any,
) -> None:
    """All provider nodes should connect to one typed variable-input configurator."""
    module = remote_configuration_nodes_module

    for node_class in (
        module.ModalConfiguration,
        module.VastConfiguration,
        module.SshConfiguration,
        module.R2StorageConfiguration,
    ):
        schema = node_class.define_schema()
        assert len(schema.outputs) == 1
        assert schema.outputs[0].io_type == module.REMOTE_CONFIGURATION_IO_TYPE
        assert schema.is_output_node is False

    configurator_schema = module.RemoteExecutionConfigurator.define_schema()
    assert configurator_schema.is_output_node is True
    assert len(configurator_schema.inputs) == 1
    assert configurator_schema.inputs[0].id == "configurations"
    assert configurator_schema.inputs[0].template.min == 1
    assert configurator_schema.inputs[0].template.max == 32
    assert (
        configurator_schema.outputs[0].io_type
        == module.REMOTE_CONFIGURATION_SET_IO_TYPE
    )


def test_compiler_collects_r2_storage_without_treating_it_as_capacity(
    remote_configuration_nodes_module: Any,
) -> None:
    """R2 should share the configurator socket but remain outside scheduler pools."""
    module = remote_configuration_nodes_module
    prompt = {
        "10": {
            "class_type": module.VAST_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": _vast_inputs("vast", 2),
        },
        "20": {
            "class_type": module.R2_STORAGE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "shared-r2",
                "account_id": "a" * 32,
                "bucket": "comfy-models",
                "credential_id": "opaque-reference",
                "jurisdiction": "eu",
                "write_back_mode": "async",
            },
        },
        "99": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {
                "configuration_0": ["10", 0],
                "configuration_1": ["20", 0],
            },
        },
    }

    configuration_set = module.compile_remote_configuration_set(prompt)

    assert configuration_set is not None
    assert len(configuration_set.capacity_configurations) == 1
    assert len(configuration_set.storage_configurations) == 1
    storage = configuration_set.storage_configurations[0]
    assert storage.storage_provider == "cloudflare_r2"
    assert storage.credential_id == "opaque-reference"
    safe_storage = storage.to_safe_dict()
    assert safe_storage == {
        "configuration_id": "20",
        "display_name": "shared-r2",
        "configuration_kind": "storage",
        "storage_provider": "cloudflare_r2",
        "account_id": "a" * 32,
        "bucket": "comfy-models",
        "jurisdiction": "eu",
        "key_prefix": "comfy-modal-cache/v1/blobs/sha256",
        "write_back_mode": "async",
    }
    assert "credential" not in str(safe_storage).casefold()


def test_configurator_rejects_storage_without_capacity(
    remote_configuration_nodes_module: Any,
) -> None:
    """A storage backing alone cannot schedule remote execution."""
    module = remote_configuration_nodes_module
    prompt = {
        "20": {
            "class_type": module.R2_STORAGE_CONFIGURATION_NODE_ID,
            "inputs": {
                "account_id": "a" * 32,
                "bucket": "comfy-models",
            },
        },
        "99": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {"configuration_0": ["20", 0]},
        },
    }

    with pytest.raises(ValueError, match="capacity configuration"):
        module.compile_remote_configuration_set(prompt)


def test_compiler_builds_two_modal_and_two_vast_configurations(
    remote_configuration_nodes_module: Any,
    execution_environments_module: Any,
) -> None:
    """Connected input order should produce all four independent capacity pools."""
    module = remote_configuration_nodes_module
    provider = execution_environments_module.ExecutionProvider
    prompt = {
        "10": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-h200",
                "gpu_type": "H200",
                "instance_count": 1,
            },
        },
        "11": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-b300",
                "gpu_type": "B300",
                "instance_count": 2,
            },
        },
        "20": {
            "class_type": module.VAST_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": _vast_inputs("vast-broad", 3),
        },
        "21": {
            "class_type": module.VAST_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": _vast_inputs("vast-cheap", 2),
        },
        "99": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {
                "configurations.configuration_3": ["21", 0],
                "configurations.configuration_1": ["11", 0],
                "configurations.configuration_0": ["10", 0],
                "configurations.configuration_2": ["20", 0],
            },
        },
    }

    configuration_set = module.compile_remote_configuration_set(prompt)

    assert configuration_set is not None
    assert [
        configuration.display_name
        for configuration in configuration_set.configurations
    ] == ["modal-h200", "modal-b300", "vast-broad", "vast-cheap"]
    assert [
        configuration.provider for configuration in configuration_set.configurations
    ] == [provider.MODAL, provider.MODAL, provider.VAST, provider.VAST]
    assert [
        configuration.capacity_limit
        for configuration in configuration_set.configurations
    ] == [1, 2, 3, 2]


def test_compiler_accepts_reconstructed_autogrow_input_mapping(
    remote_configuration_nodes_module: Any,
) -> None:
    """The runtime-style nested mapping should compile like queue-time paths."""
    module = remote_configuration_nodes_module
    prompt = {
        "10": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-t4",
                "gpu_type": "T4",
            },
        },
        "11": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "modal-h200",
                "gpu_type": "H200",
            },
        },
        "99": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {
                "configurations": {
                    "configuration_0": ["10", 0],
                    "configuration_1": ["11", 0],
                }
            },
        },
    }

    configuration_set = module.compile_remote_configuration_set(prompt)

    assert configuration_set is not None
    assert [
        configuration.display_name
        for configuration in configuration_set.configurations
    ] == ["modal-t4", "modal-h200"]


def test_compiler_builds_workflow_declared_ssh_host_without_credentials(
    remote_configuration_nodes_module: Any,
) -> None:
    """SSH configuration should contain portable scheduling fields only."""
    module = remote_configuration_nodes_module
    prompt = {
        "30": {
            "class_type": module.SSH_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "environment_id": "lambda",
                "display_name": "Lambda GPU",
                "ssh_target": "lambda",
                "cost_usd_per_hour": 1.25,
                "maximum_workers": 2,
                "reserve_vram_gb": 4.0,
                "tags": "owned, fast",
                "docker_env_file": "/etc/comfy-worker.env",
            },
        },
        "99": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {"configuration_0": ["30", 0]},
        },
    }

    configuration_set = module.compile_remote_configuration_set(prompt)

    assert configuration_set is not None
    configuration = configuration_set.configurations[0]
    assert configuration.configuration_id == "lambda"
    assert configuration.capacity_limit == 2
    assert configuration.host.ssh_target == "lambda"
    assert configuration.host.cost_usd_per_second == pytest.approx(1.25 / 3600)
    assert configuration.host.reserve_vram_bytes == 4 * 1024**3
    assert configuration.host.tags == frozenset({"owned", "fast"})
    assert "private_key" not in str(configuration.to_safe_dict()).lower()


def test_ssh_configuration_preserves_unknown_cost_as_unknown(
    remote_configuration_nodes_module: Any,
) -> None:
    """An unpriced owned host must not be silently interpreted as free."""
    configuration = remote_configuration_nodes_module.ssh_configuration_from_inputs(
        "node",
        {
            "environment_id": "owned-host",
            "display_name": "Owned host",
            "ssh_target": "owned-host",
            "cost_usd_per_hour": "Unknown",
        },
    )

    assert configuration.host.cost_usd_per_second is None


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        (
            {
                "90": {
                    "class_type": "RemoteExecutionConfigurator",
                    "inputs": {},
                }
            },
            "at least one connected",
        ),
        (
            {
                "90": {
                    "class_type": "RemoteExecutionConfigurator",
                    "inputs": {"configuration_0": ["missing", 0]},
                },
                "91": {
                    "class_type": "RemoteExecutionConfigurator",
                    "inputs": {"configuration_0": ["missing", 0]},
                },
            },
            "only one",
        ),
    ],
)
def test_compiler_rejects_invalid_configurator_topology(
    remote_configuration_nodes_module: Any,
    prompt: dict[str, Any],
    message: str,
) -> None:
    """Queue-time validation should fail before provider side effects."""
    with pytest.raises(ValueError, match=message):
        remote_configuration_nodes_module.compile_remote_configuration_set(prompt)


def test_configuration_set_rejects_duplicate_names(
    remote_configuration_nodes_module: Any,
) -> None:
    """Human-facing pool names should remain unambiguous across providers."""
    module = remote_configuration_nodes_module
    prompt = {
        "1": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "duplicate",
                "gpu_type": "H200",
            },
        },
        "2": {
            "class_type": module.VAST_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": _vast_inputs("Duplicate", 1),
        },
        "9": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {
                "configuration_0": ["1", 0],
                "configuration_1": ["2", 0],
            },
        },
    }

    with pytest.raises(ValueError, match="names must be unique"):
        module.compile_remote_configuration_set(prompt)


def test_configuration_set_rejects_duplicate_modal_targets(
    remote_configuration_nodes_module: Any,
) -> None:
    """Two logical pools must not fight over one GPU-specific Modal app limit."""
    module = remote_configuration_nodes_module
    prompt = {
        "1": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "first",
                "gpu_type": "H200",
            },
        },
        "2": {
            "class_type": module.MODAL_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "second",
                "gpu_type": "H200",
            },
        },
        "9": {
            "class_type": module.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {
                "configuration_0": ["1", 0],
                "configuration_1": ["2", 0],
            },
        },
    }

    with pytest.raises(ValueError, match="Each Modal GPU type"):
        module.compile_remote_configuration_set(prompt)


def test_extension_registers_all_remote_configuration_nodes(
    extension_package: Any,
    remote_configuration_nodes_module: Any,
) -> None:
    """The v3 extension entrypoint should expose the complete node family."""
    module = remote_configuration_nodes_module
    assert extension_package.ModalConfiguration is module.ModalConfiguration
    assert extension_package.VastConfiguration is module.VastConfiguration
    assert extension_package.SshConfiguration is module.SshConfiguration
    assert extension_package.R2StorageConfiguration is module.R2StorageConfiguration
    assert (
        extension_package.RemoteExecutionConfigurator
        is module.RemoteExecutionConfigurator
    )
