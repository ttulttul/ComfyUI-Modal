"""Tests for SSH OCI worker image and GPU assignment behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def test_worker_indices_are_distributed_across_discovered_gpus(
    ssh_runtime_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
) -> None:
    """Parallel worker containers should deterministically select different GPUs."""
    gpu_type = execution_environments_module.GpuCapability
    capabilities = execution_environments_module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=32,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=120 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(
            gpu_type("GPU-one", "GPU one", 48 * 1024**3),
            gpu_type("GPU-two", "GPU two", 48 * 1024**3),
        ),
    )
    host = remote_hosts_module.SshHostConfig(
        environment_id="dual-gpu",
        display_name="Dual GPU",
        ssh_target="dual-gpu",
        capabilities=capabilities,
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(host=host),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )

    assert manager._gpu_arguments(0) == ("--gpus", "device=GPU-one")
    assert manager._gpu_arguments(1) == ("--gpus", "device=GPU-two")
    assert manager._gpu_arguments(2) == ("--gpus", "device=GPU-one")
