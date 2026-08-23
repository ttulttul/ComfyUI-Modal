"""Tests for SSH OCI worker image and GPU assignment behavior."""

from __future__ import annotations

from pathlib import Path
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


def test_worker_context_includes_top_level_comfyui_python_modules(
    ssh_runtime_module: Any,
    tmp_path: Path,
) -> None:
    """The SSH image must contain modules imported by headless ComfyUI startup."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    repo_root.mkdir()
    comfyui_root.mkdir()
    (repo_root / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    for module_name in ("execution.py", "folder_paths.py", "nodes.py", "server.py"):
        (comfyui_root / module_name).write_text("VALUE = 1\n", encoding="utf-8")
    (comfyui_root / "requirements.txt").write_text(
        "aiohttp==3.13.3\n",
        encoding="utf-8",
    )
    (comfyui_root / "README.md").write_text("not runtime source\n", encoding="utf-8")
    comfy_package = comfyui_root / "comfy"
    comfy_package.mkdir()
    (comfy_package / "__init__.py").write_text("", encoding="utf-8")
    (comfy_package / "data.json").write_text("{}\n", encoding="utf-8")

    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(),
        repo_root=repo_root,
        settings=SimpleNamespace(comfyui_root=comfyui_root),
    )

    archive_paths = {
        archive_path for _, archive_path in manager._runtime_context_files()
    }

    assert {
        "comfyui/execution.py",
        "comfyui/folder_paths.py",
        "comfyui/nodes.py",
        "comfyui/server.py",
        "comfyui/requirements.txt",
        "comfyui/comfy/__init__.py",
    }.issubset(archive_paths)
    assert "comfyui/README.md" not in archive_paths
    assert "comfyui/comfy/data.json" not in archive_paths


def test_worker_dockerfile_disables_inherited_base_image_healthcheck(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """The socket worker must not inherit llama-server's HTTP health check."""
    monkeypatch.setattr(
        ssh_runtime_module,
        "select_remote_torch_build",
        lambda _gpu: SimpleNamespace(
            install_layers=(),
            validation_command=lambda: "true",
        ),
    )
    monkeypatch.setattr(ssh_runtime_module, "remote_apt_packages", lambda: ())
    monkeypatch.setattr(ssh_runtime_module, "remote_runtime_packages", lambda: ())
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_accelerator_packages",
        lambda _gpu: (),
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_accelerator_validation_command",
        lambda _gpu: "true",
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "custom_node_runtime_packages",
        lambda _path: (),
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(
            modal_gpu="RTX-PRO-6000",
            custom_nodes_dir=Path("/custom_nodes"),
        ),
    )
    spec = SimpleNamespace(identity=SimpleNamespace(fingerprint="deadbeef"))

    dockerfile = manager._dockerfile(spec)

    assert "HEALTHCHECK NONE" in dockerfile
    assert dockerfile.index("HEALTHCHECK NONE") < dockerfile.index("ENTRYPOINT")
    assert "LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}" in dockerfile
    assert "COMFY_MODAL_LLM_EXECUTION_TARGET=ssh_docker" in dockerfile


def test_worker_dockerfile_retries_large_accelerator_downloads(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """Large direct wheels should survive transient interrupted downloads."""
    monkeypatch.setattr(
        ssh_runtime_module,
        "select_remote_torch_build",
        lambda _gpu: SimpleNamespace(
            install_layers=(),
            validation_command=lambda: "true",
        ),
    )
    monkeypatch.setattr(ssh_runtime_module, "remote_apt_packages", lambda: ())
    monkeypatch.setattr(ssh_runtime_module, "remote_runtime_packages", lambda: ())
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_accelerator_packages",
        lambda _gpu: ("https://example.test/large-accelerator.whl",),
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_accelerator_validation_command",
        lambda _gpu: "true",
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "custom_node_runtime_packages",
        lambda _path: (),
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(
            modal_gpu="RTX-PRO-6000",
            custom_nodes_dir=Path("/custom_nodes"),
        ),
    )
    spec = SimpleNamespace(identity=SimpleNamespace(fingerprint="deadbeef"))

    dockerfile = manager._dockerfile(spec)

    assert (
        "RUN python -m pip install --no-cache-dir --resume-retries 20 "
        "--timeout 120 https://example.test/large-accelerator.whl"
    ) in dockerfile


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
