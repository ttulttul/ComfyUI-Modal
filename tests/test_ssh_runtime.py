"""Tests for SSH OCI worker image and GPU assignment behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def test_worker_build_loads_image_into_the_remote_daemon(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """BuildKit docker-container drivers must export the image for docker run."""
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def docker(arguments: tuple[str, ...], **kwargs: Any) -> None:
        """Record one remote Docker invocation."""
        calls.append((arguments, kwargs))

    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(
            host=SimpleNamespace(environment_id="gpu-host"),
            docker=docker,
        ),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(startup_timeout_seconds=900),
    )
    spec = SimpleNamespace(
        image_tag="comfy-remote:deadbeef",
        identity=SimpleNamespace(fingerprint="deadbeef"),
    )
    monkeypatch.setattr(manager, "_build_context", lambda runtime_spec: b"context")

    manager._build_image(spec)

    arguments, kwargs = calls[0]
    assert arguments[:3] == ("build", "--pull", "--load")
    assert kwargs["input_payload"] == b"context"


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


def test_stale_worker_reconciliation_only_replaces_the_same_slot(
    ssh_runtime_module: Any,
) -> None:
    """A new fingerprint should retire its superseded logical worker container."""
    current = SimpleNamespace(
        worker_index=1,
        container_name="comfy-remote-host-new-w1",
    )
    workers = (
        SimpleNamespace(worker_index=0, container_name="comfy-remote-host-old-w0"),
        SimpleNamespace(worker_index=1, container_name="comfy-remote-host-old-w1"),
        SimpleNamespace(worker_index=1, container_name=current.container_name),
    )
    removed: list[str] = []
    controller = SimpleNamespace(
        host=SimpleNamespace(environment_id="host"),
        list_managed_workers=lambda: workers,
        remove_managed_worker=removed.append,
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=controller,
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )

    manager._remove_stale_worker_containers(current)

    assert removed == ["comfy-remote-host-old-w1"]


def test_stop_all_workers_removes_only_controller_managed_names(
    ssh_runtime_module: Any,
) -> None:
    """Environment shutdown should delegate every ownership check to the controller."""
    workers = (
        SimpleNamespace(container_name="worker-one"),
        SimpleNamespace(container_name="worker-two"),
    )
    removed: list[str] = []
    controller = SimpleNamespace(
        list_managed_workers=lambda: workers,
        remove_managed_worker=removed.append,
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=controller,
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )

    result = manager.stop_all_workers()

    assert result == ("worker-one", "worker-two")
    assert removed == ["worker-one", "worker-two"]


def test_runtime_passes_only_remote_environment_file_path(
    ssh_runtime_module: Any,
) -> None:
    """Worker secrets should remain in an administrator-managed file on the host."""
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(
            host=SimpleNamespace(docker_env_file="/etc/comfy/worker.env")
        ),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )

    assert manager._environment_file_arguments() == (
        "--env-file",
        "/etc/comfy/worker.env",
    )
