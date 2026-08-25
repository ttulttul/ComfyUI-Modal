"""Tests for SSH OCI worker image and GPU assignment behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def test_worker_lifecycle_reports_build_and_readiness_status(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """A rebuilt SSH worker should expose every material readiness transition."""
    statuses: list[str] = []
    spec = SimpleNamespace(
        image_tag="comfy-remote:deadbeef",
        storage_volume_name="comfy-remote-lambda",
    )
    controller = SimpleNamespace(
        host=SimpleNamespace(environment_id="lambda"),
        ensure_volume=lambda _name: None,
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=controller,
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    monkeypatch.setattr(manager, "runtime_spec", lambda _worker_index: spec)
    monkeypatch.setattr(manager, "_image_is_current", lambda _spec: False)
    monkeypatch.setattr(
        manager,
        "_build_image",
        lambda _spec, status_callback=None: None,
    )
    monkeypatch.setattr(manager, "_remove_stale_worker_containers", lambda _spec: None)
    monkeypatch.setattr(
        manager,
        "_container_is_current_and_running",
        lambda _spec: False,
    )
    monkeypatch.setattr(manager, "_replace_worker_container", lambda _spec: None)
    monkeypatch.setattr(manager, "_wait_until_ready", lambda _spec: None)

    result = manager.ensure_worker(status_callback=statuses.append)

    assert result is spec
    assert statuses == [
        "Checking SSH runtime environment=lambda",
        "Building SSH runtime environment=lambda image=comfy-remote:deadbeef",
        "Starting SSH worker environment=lambda",
        "Waiting for SSH worker environment=lambda",
        "Ready for remote execution",
    ]


def test_worker_build_loads_image_into_the_remote_daemon(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """BuildKit docker-container drivers must export the image for docker run."""
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def docker(arguments: tuple[str, ...], **kwargs: Any) -> Any:
        """Record one remote Docker invocation."""
        calls.append((arguments, kwargs))
        if arguments[:2] == ("image", "inspect"):
            return SimpleNamespace(returncode=1, stdout_text="")
        return None

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
    monkeypatch.setattr(
        manager,
        "_dependency_build_context",
        lambda: b"dependencies",
    )
    monkeypatch.setattr(
        manager,
        "_source_overlay_build_context",
        lambda runtime_spec: b"source-overlay",
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_runtime_dependency_fingerprint",
        lambda _identity: "cafebabe" * 8,
    )

    manager._build_image(spec)

    assert calls[0][0][:2] == ("image", "inspect")
    dependency_arguments, dependency_kwargs = calls[1]
    assert dependency_arguments[:6] == (
        "buildx",
        "build",
        "--builder",
        "default",
        "--pull",
        "--load",
    )
    assert dependency_kwargs["input_payload"] == b"dependencies"
    overlay_arguments, overlay_kwargs = calls[2]
    assert overlay_arguments[:5] == (
        "buildx",
        "build",
        "--builder",
        "default",
        "--load",
    )
    assert "--pull" not in overlay_arguments
    assert overlay_kwargs["input_payload"] == b"source-overlay"


def test_worker_build_reuses_retained_dependency_image(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """A code-only runtime change should build only the source overlay."""
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def docker(arguments: tuple[str, ...], **kwargs: Any) -> Any:
        """Report a current dependency image and record the overlay build."""
        calls.append((arguments, kwargs))
        if arguments[:2] == ("image", "inspect"):
            return SimpleNamespace(
                returncode=0,
                stdout_text="cafebabe" * 8,
            )
        return None

    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(
            host=SimpleNamespace(environment_id="gpu-host"),
            docker=docker,
        ),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(startup_timeout_seconds=900),
    )
    spec = SimpleNamespace(
        image_tag="comfy-remote:source-change",
        identity=SimpleNamespace(fingerprint="source-change"),
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_runtime_dependency_fingerprint",
        lambda _identity: "cafebabe" * 8,
    )
    def fail_dependency_build() -> bytes:
        """Fail if a retained dependency image is rebuilt."""
        raise AssertionError("dependency context should not be rebuilt")

    monkeypatch.setattr(manager, "_dependency_build_context", fail_dependency_build)
    monkeypatch.setattr(
        manager,
        "_source_overlay_build_context",
        lambda _runtime_spec: b"52MB-source-overlay",
    )

    manager._build_image(spec)

    assert len(calls) == 2
    assert calls[0][0][:2] == ("image", "inspect")
    assert calls[1][0][:5] == (
        "buildx",
        "build",
        "--builder",
        "default",
        "--load",
    )
    assert calls[1][1]["input_payload"] == b"52MB-source-overlay"


def test_conflicting_worker_launch_adopts_current_managed_winner(
    ssh_runtime_module: Any,
    ssh_docker_module: Any,
) -> None:
    """A concurrent launcher winning the Docker name race should be reused."""
    fingerprint = "ac12dff11abd460ecb642f3f0f19f3c5a8fe3d345f88f0d45e7b05d4036f8fa2"
    container_name = "comfy-remote-lambda-ac12dff11abd460e-w0"
    inspect_count = 0

    def docker(arguments: tuple[str, ...], **kwargs: Any) -> Any:
        """Return a name conflict followed by the concurrent winner's state."""
        nonlocal inspect_count
        if arguments[:2] == ("container", "inspect"):
            inspect_count += 1
            if inspect_count == 1:
                return ssh_docker_module.SshCommandResult(b"", b"not found", 1)
            payload = [
                {
                    "State": {"Running": True},
                    "Config": {
                        "Labels": {
                            "comfy.remote.runtime-fingerprint": fingerprint,
                            "comfy.remote.environment-id": "lambda",
                            "comfy.remote.worker-index": "0",
                        }
                    },
                }
            ]
            return ssh_docker_module.SshCommandResult(
                json.dumps(payload).encode(),
                b"",
                0,
            )
        if arguments[0] == "run":
            assert kwargs["check"] is False
            return ssh_docker_module.SshCommandResult(
                b"",
                b"Conflict. The container name is already in use.",
                125,
            )
        raise AssertionError(f"Unexpected Docker arguments: {arguments!r}")

    controller = SimpleNamespace(
        host=SimpleNamespace(
            environment_id="lambda",
            ssh_target="lambda",
            capabilities=SimpleNamespace(
                gpus=(SimpleNamespace(uuid="GPU-one"),),
                nvidia_container_runtime=True,
            ),
            docker_env_file=None,
        ),
        docker=docker,
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=controller,
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    spec = ssh_runtime_module.SshRuntimeSpec(
        identity=SimpleNamespace(fingerprint=fingerprint),
        image_tag="comfy-remote:ac12dff11abd460e",
        container_name=container_name,
        storage_volume_name="comfy-remote-lambda",
        worker_index=0,
    )

    manager._replace_worker_container(spec)

    assert inspect_count == 2


def test_current_worker_requires_environment_and_slot_ownership_labels(
    ssh_runtime_module: Any,
    ssh_docker_module: Any,
) -> None:
    """A matching runtime fingerprint alone must not authorize worker adoption."""
    fingerprint = "deadbeef" * 8
    payload = [
        {
            "State": {"Running": True},
            "Config": {
                "Labels": {
                    "comfy.remote.runtime-fingerprint": fingerprint,
                    "comfy.remote.environment-id": "different-host",
                    "comfy.remote.worker-index": "0",
                }
            },
        }
    ]
    controller = SimpleNamespace(
        host=SimpleNamespace(environment_id="lambda"),
        docker=lambda *_args, **_kwargs: ssh_docker_module.SshCommandResult(
            json.dumps(payload).encode(),
            b"",
            0,
        ),
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=controller,
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    spec = SimpleNamespace(
        identity=SimpleNamespace(fingerprint=fingerprint),
        container_name="comfy-remote-lambda-deadbeefdeadbeef-w0",
        worker_index=0,
    )

    assert manager._container_is_current_and_running(spec) is False


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
    tokenizer_directory = comfy_package / "text_encoders" / "qwen25_tokenizer"
    tokenizer_directory.mkdir(parents=True)
    (tokenizer_directory / "merges.txt").write_text("a b\n", encoding="utf-8")
    (tokenizer_directory / "vocab.json").write_text("{}\n", encoding="utf-8")

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
        "comfyui/comfy/data.json",
        "comfyui/comfy/text_encoders/qwen25_tokenizer/merges.txt",
        "comfyui/comfy/text_encoders/qwen25_tokenizer/vocab.json",
    }.issubset(archive_paths)
    assert "comfyui/README.md" not in archive_paths


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
    assert "from av.video.reformatter import ColorPrimaries" in dockerfile


def test_worker_dependency_base_is_separate_from_source_overlay(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """Frequent source edits must not invalidate heavyweight package layers."""
    monkeypatch.setattr(
        ssh_runtime_module,
        "select_remote_torch_build",
        lambda _gpu: SimpleNamespace(
            install_layers=(),
            validation_command=lambda: "true",
        ),
    )
    monkeypatch.setattr(ssh_runtime_module, "remote_apt_packages", lambda: ())
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_runtime_packages",
        lambda: ("runtime-package==1",),
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_accelerator_packages",
        lambda _gpu: ("accelerator-package==1",),
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
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_runtime_dependency_fingerprint",
        lambda _identity: "cafebabe" * 8,
    )
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(
            modal_gpu="RTX-PRO-6000",
            custom_nodes_dir=Path("/custom_nodes"),
        ),
    )
    spec = SimpleNamespace(identity=SimpleNamespace(fingerprint="source-fingerprint"))

    dependency_dockerfile = manager._dependency_dockerfile()
    overlay_dockerfile = manager._source_overlay_dockerfile(spec)

    assert "pip install --no-cache-dir runtime-package==1" in dependency_dockerfile
    assert "accelerator-package==1" in dependency_dockerfile
    assert "COPY repo" not in dependency_dockerfile
    assert "COPY comfyui" not in dependency_dockerfile
    assert overlay_dockerfile.startswith("FROM comfy-remote-deps:cafebabecafebabe\n")
    assert "COPY repo /opt/comfy-remote/repo" in overlay_dockerfile
    assert "COPY comfyui /opt/comfy-remote/ComfyUI" in overlay_dockerfile
    assert "apt-get" not in overlay_dockerfile
    assert "pip install" not in overlay_dockerfile
    assert "COMFY_MODAL_RUNTIME_FINGERPRINT=source-fingerprint" in overlay_dockerfile


def test_worker_dockerfile_exposes_triton_compiler_toolchain(
    ssh_runtime_module: Any,
    monkeypatch: Any,
) -> None:
    """Generated SSH and Vast images must expose the compiler Triton discovers."""
    monkeypatch.setattr(
        ssh_runtime_module,
        "select_remote_torch_build",
        lambda _gpu: SimpleNamespace(
            install_layers=(),
            validation_command=lambda: "true",
        ),
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_apt_packages",
        lambda: ("build-essential",),
    )
    monkeypatch.setattr(
        ssh_runtime_module,
        "remote_compiler_validation_command",
        lambda: "test -x /usr/bin/gcc",
    )
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

    assert "apt-get install -y --no-install-recommends build-essential" in dockerfile
    assert "CC=/usr/bin/gcc CXX=/usr/bin/g++" in dockerfile
    assert "RUN test -x /usr/bin/gcc" in dockerfile


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


def test_worker_launch_refuses_missing_gpu_capabilities(
    ssh_runtime_module: Any,
    ssh_docker_module: Any,
) -> None:
    """A missing probe must fail instead of silently launching a CPU container."""
    manager = ssh_runtime_module.SshRuntimeManager(
        controller=SimpleNamespace(
            host=SimpleNamespace(capabilities=None),
        ),
        repo_root=SimpleNamespace(),
        settings=SimpleNamespace(),
    )

    try:
        manager._gpu_arguments(0)
    except ssh_docker_module.SshDockerError as exc:
        assert "requires probed GPU capabilities" in str(exc)
    else:
        raise AssertionError("Missing GPU capabilities should fail worker launch.")


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
