"""Tests for persistent SSH execution host configuration."""

from __future__ import annotations

import json
from typing import Any

import pytest


def test_registry_round_trips_hosts_without_credentials(
    tmp_path: Any,
    remote_hosts_module: Any,
) -> None:
    """Host configuration should persist atomically as versioned JSON."""
    registry = remote_hosts_module.RemoteHostRegistry.for_user_directory(tmp_path)
    host = remote_hosts_module.SshHostConfig(
        environment_id="studio-4090",
        display_name="Studio 4090",
        ssh_target="studio-gpu",
        cost_usd_per_second=0.00012,
        maximum_workers=2,
        tags=frozenset({"home", "interactive"}),
    )

    registry.replace_hosts([host])

    assert registry.load() == remote_hosts_module.RemoteExecutionConfig(hosts=(host,))
    payload = json.loads(registry.config_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert "password" not in json.dumps(payload).lower()
    assert "private_key" not in json.dumps(payload).lower()


def test_host_rejects_option_injection_in_ssh_target(remote_hosts_module: Any) -> None:
    """An SSH destination must not be accepted as a command-line option."""
    with pytest.raises(ValueError, match="option prefix"):
        remote_hosts_module.SshHostConfig(
            environment_id="unsafe",
            display_name="Unsafe",
            ssh_target="-oProxyCommand=bad",
        )


def test_host_requires_safe_absolute_remote_environment_file(
    remote_hosts_module: Any,
) -> None:
    """Docker environment-file paths must not escape through relative traversal."""
    with pytest.raises(ValueError, match="safe absolute path"):
        remote_hosts_module.SshHostConfig(
            environment_id="unsafe-env",
            display_name="Unsafe env",
            ssh_target="gpu-host",
            docker_env_file="../secrets.env",
        )


def test_registry_persists_probe_results(
    tmp_path: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
) -> None:
    """Capability probes should update only their matching host."""
    registry = remote_hosts_module.RemoteHostRegistry.for_user_directory(tmp_path)
    registry.replace_hosts(
        [
            remote_hosts_module.SshHostConfig(
                environment_id="gpu-one",
                display_name="GPU one",
                ssh_target="gpu-one",
            )
        ]
    )
    capabilities = execution_environments_module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=32,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=96 * 1024**3,
        available_disk_bytes=2 * 1024**4,
        docker_version="28.1.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
    )

    updated = registry.update_probe_result(
        "gpu-one",
        capabilities=capabilities,
        health=execution_environments_module.EnvironmentHealth.READY,
        last_error=None,
    )

    assert updated.capabilities == capabilities
    assert (
        registry.get_host("gpu-one").health
        is execution_environments_module.EnvironmentHealth.READY
    )


def test_scheduling_reserves_configured_vram(
    remote_hosts_module: Any,
    execution_environments_module: Any,
) -> None:
    """A host's local-use VRAM reserve must not be offered to remote work."""
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
        gpus=(module.GpuCapability("GPU-1", "GPU", 48 * 1024**3, 40 * 1024**3),),
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="reserved",
        display_name="Reserved",
        ssh_target="reserved",
        reserve_vram_bytes=8 * 1024**3,
        capabilities=capabilities,
        health=module.EnvironmentHealth.READY,
    )

    scheduling_capabilities = host.scheduling_state().capabilities

    assert scheduling_capabilities is not None
    assert scheduling_capabilities.gpus[0].total_vram_bytes == 40 * 1024**3
    assert scheduling_capabilities.gpus[0].free_vram_bytes == 32 * 1024**3


def test_scheduling_uses_discovered_cost_when_no_override_exists(
    remote_hosts_module: Any,
    execution_environments_module: Any,
) -> None:
    """Provisioning can publish host cost through the reserved Docker daemon label."""
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
        reported_cost_usd_per_second=0.00025,
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="reported-cost",
        display_name="Reported cost",
        ssh_target="reported-cost",
        capabilities=capabilities,
        health=module.EnvironmentHealth.READY,
    )

    assert host.scheduling_state().cost_usd_per_second == 0.00025
