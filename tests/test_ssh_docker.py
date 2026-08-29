"""Tests for SSH Docker discovery and named-volume storage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


class _FakeRunner:
    """Return deterministic results for remote capability commands."""

    def __init__(
        self,
        ssh_docker_module: Any,
        *,
        worker_processes: tuple[str, ...] = (
            "python -m remote.ssh_worker serve",
            "/app/llama-server --model /storage/model.gguf",
        ),
    ) -> None:
        """Initialize the fake with the module's result type."""
        self.module = ssh_docker_module
        self.calls: list[tuple[tuple[str, ...], bytes | None]] = []
        self.worker_processes = worker_processes

    def run(
        self,
        remote_argv: Any,
        *,
        input_payload: bytes | None = None,
        timeout_seconds: float | None = None,
        check: bool = True,
    ) -> Any:
        """Return one command-specific deterministic result."""
        del timeout_seconds
        command = tuple(remote_argv)
        self.calls.append((command, input_payload))
        stdout = b""
        returncode = 0
        if command[:2] == ("docker", "info"):
            stdout = json.dumps(
                {
                    "ServerVersion": "28.1.0",
                    "Runtimes": {"nvidia": {}, "runc": {}},
                    "SecurityOptions": ["name=seccomp"],
                    "Labels": ["comfy.remote.cost-usd-per-second=0.00025"],
                }
            ).encode()
        elif command == ("uname", "-m"):
            stdout = b"x86_64\n"
        elif command == ("uname", "-s"):
            stdout = b"Linux\n"
        elif command == ("getconf", "_NPROCESSORS_ONLN"):
            stdout = b"32\n"
        elif command == ("cat", "/proc/meminfo"):
            stdout = b"MemTotal: 131072000 kB\nMemAvailable: 98304000 kB\n"
        elif command == ("df", "-Pk", "/var/lib/docker"):
            stdout = b"Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/test 2000000 100 1900000 1% /var/lib/docker\n"
        elif command[0] == "nvidia-smi":
            stdout = b"GPU-123, NVIDIA RTX 6000 Ada, 49140, 48000, 8.9, 580.95\n"
        elif command[:3] == ("docker", "volume", "create"):
            stdout = f"{command[3]}\n".encode()
        elif command[:2] == ("docker", "ps"):
            stdout = (
                json.dumps(
                    {
                        "Names": "comfy-remote-gpu-host-deadbeef-w1",
                        "Image": "comfy-remote:deadbeef",
                        "State": "running",
                        "Status": "Up 3 minutes",
                        "Labels": (
                            "comfy.remote.environment-id=gpu-host,"
                            "comfy.remote.runtime-fingerprint=deadbeef,"
                            "comfy.remote.worker-index=1"
                        ),
                    }
                ).encode()
                + b"\n"
            )
        elif command[:3] == ("docker", "rm", "-f"):
            stdout = f"{command[3]}\n".encode()
        elif command[:2] == ("docker", "top"):
            stdout = (
                "COMMAND\n" + "\n".join(self.worker_processes) + "\n"
            ).encode()
        elif command[:2] == ("docker", "run"):
            if command[-2:] == ("-m", "remote.r2_materializer"):
                operation = json.loads(input_payload or b"{}")["operation"]
                stdout = json.dumps(
                    {"parts": []}
                    if operation == "upload"
                    else (
                        {"authorized": True}
                        if operation == "preflight"
                        else {"sha256": "a" * 64}
                    )
                ).encode()
            else:
                stdout = b""
        else:
            returncode = 1
        result = self.module.SshCommandResult(stdout, b"failed" if returncode else b"", returncode)
        if check and returncode:
            raise self.module.SshDockerError("fake command failed")
        return result


def _host(remote_hosts_module: Any) -> Any:
    """Return one deterministic SSH host configuration."""
    return remote_hosts_module.SshHostConfig(
        environment_id="gpu-host",
        display_name="GPU host",
        ssh_target="gpu-host-alias",
    )


def test_ssh_command_disables_configured_forwardings(ssh_docker_module: Any) -> None:
    """Controller sessions should not activate unrelated forwards from an SSH alias."""
    runner = ssh_docker_module.SshCommandRunner("gpu-host-alias")

    command = runner.command(("docker", "info"))

    clear_index = command.index("ClearAllForwardings=yes")
    assert command[clear_index - 1] == "-o"
    assert command[-3:] == ["--", "gpu-host-alias", "docker info"]


def test_probe_discovers_host_docker_memory_and_gpu(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
) -> None:
    """Capability probing should normalize all scheduler-visible resources."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )

    capabilities = controller.probe_capabilities()

    assert capabilities.architecture == "x86_64"
    assert capabilities.operating_system == "linux"
    assert capabilities.cpu_count == 32
    assert capabilities.total_ram_bytes == 131072000 * 1024
    assert capabilities.available_disk_bytes == 1900000 * 1024
    assert capabilities.nvidia_container_runtime is True
    assert capabilities.gpus[0].total_vram_bytes == 49140 * 1024**2
    assert capabilities.reported_cost_usd_per_second == 0.00025


def test_volume_upload_streams_payload_without_embedding_it_in_arguments(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
) -> None:
    """Volume uploads should use SSH stdin and an atomically published target."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )
    volume = ssh_docker_module.SshDockerVolumeBackend(
        controller,
        "comfy-remote-gpu-host",
    )

    volume.put_bytes(b"secret binary payload", "/assets/test.bin")

    upload_command, upload_payload = runner.calls[-1]
    assert upload_payload == b"secret binary payload"
    assert b"secret binary payload" not in " ".join(upload_command).encode()
    assert "COMFY_REMOTE_PATH=assets/test.bin" in upload_command


def test_volume_rejects_parent_traversal(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
) -> None:
    """Remote volume paths must stay beneath the storage mount."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )
    volume = ssh_docker_module.SshDockerVolumeBackend(controller, "safe-volume")

    with pytest.raises(ValueError, match="Unsafe remote volume path"):
        volume.put_bytes(b"payload", "../escape")


def test_put_file_requires_a_regular_local_file(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
    tmp_path: Path,
) -> None:
    """Uploading a missing local asset should fail before any remote mutation."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )
    volume = ssh_docker_module.SshDockerVolumeBackend(controller, "safe-volume")

    with pytest.raises(FileNotFoundError):
        volume.put_file(tmp_path / "missing.bin", "/assets/missing.bin")


def test_r2_materialization_keeps_signed_url_in_docker_stdin(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
    r2_cache_module: Any,
) -> None:
    """Signed R2 credentials should not appear in SSH or Docker arguments."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )
    image_preparations: list[bool] = []
    volume = ssh_docker_module.SshDockerVolumeBackend(
        controller,
        "safe-volume",
        materializer_image="comfy-remote:current",
        materializer_image_preparer=lambda: image_preparations.append(True),
    )
    signed_url = (
        "https://account.r2.cloudflarestorage.com/object?X-Amz-Signature=secret"
    )
    request = r2_cache_module.R2DownloadRequest(
        url=signed_url,
        allowed_host="account.r2.cloudflarestorage.com",
        sha256="a" * 64,
        size_bytes=1024,
    )

    volume.materialize_r2_file(request, "/assets/model.safetensors")
    volume.materialize_r2_file(request, "/assets/model.safetensors")

    command, input_payload = runner.calls[-1]
    assert command[-2:] == ("-m", "remote.r2_materializer")
    assert "secret" not in " ".join(command)
    assert input_payload is not None and b"secret" in input_payload
    assert image_preparations == [True]
    assert volume.exists("/assets/model.safetensors") is True


def test_r2_preflight_uses_worker_image_and_protected_stdin(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
    r2_cache_module: Any,
) -> None:
    """SSH worker authorization probes must not expose their signed URL in argv."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )
    volume = ssh_docker_module.SshDockerVolumeBackend(
        controller,
        "safe-volume",
        materializer_image="comfy-remote:current",
    )
    request = r2_cache_module.R2WorkerPreflightRequest(
        url="https://account.r2.cloudflarestorage.com/missing?signature=secret",
        allowed_host="account.r2.cloudflarestorage.com",
    )

    volume.preflight_r2_access(request)

    command, input_payload = runner.calls[-1]
    assert "secret" not in " ".join(command)
    assert input_payload is not None
    assert b'"operation": "preflight"' in input_payload
    assert b"secret" in input_payload


def test_r2_writeback_cancellation_removes_exact_helper_container(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
    r2_cache_module: Any,
    monkeypatch: Any,
) -> None:
    """Foreground work should stop both SSH and its named Docker upload helper."""

    class FakeProcess:
        """Remain active until the backend observes cancellation."""

        def __init__(self) -> None:
            """Initialize one running fake process."""
            self.returncode = 0
            self.terminated = False
            self.communicate_calls = 0

        def communicate(self, *_args: Any, **_kwargs: Any) -> tuple[bytes, bytes]:
            """Time out while active and finish after termination."""
            self.communicate_calls += 1
            if not self.terminated:
                raise subprocess.TimeoutExpired("ssh", 0.25)
            return b"", b""

        def terminate(self) -> None:
            """Record graceful termination."""
            self.terminated = True

        def kill(self) -> None:
            """Record forced termination."""
            self.terminated = True

    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )
    process = FakeProcess()
    monkeypatch.setattr(controller, "docker_popen", lambda _arguments: process)
    volume = ssh_docker_module.SshDockerVolumeBackend(
        controller,
        "safe-volume",
        materializer_image="comfy-remote:current",
    )
    plan = r2_cache_module.R2UploadPlan(
        key=f"cache/{'a' * 64}",
        sha256="a" * 64,
        size_bytes=1024,
        allowed_host="account.r2.cloudflarestorage.com",
        mode="single",
        urls=("https://account.r2.cloudflarestorage.com/object?secret=1",),
    )
    cancellation_checks = iter((False, False, True))

    with pytest.raises(InterruptedError, match="cancelled"):
        volume.upload_r2_file_cancellable(
            plan,
            "assets/model.safetensors",
            cancellation_check=lambda: next(cancellation_checks),
        )

    assert process.terminated is True
    remove_command = next(
        command
        for command, _payload in runner.calls
        if command[:3] == ("docker", "rm", "-f")
    )
    assert remove_command[3].startswith("comfy-r2-materializer-")


def test_managed_worker_status_and_removal_are_label_scoped(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
) -> None:
    """Lifecycle operations must inspect ownership before removing a container."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )

    workers = controller.list_managed_workers()
    controller.remove_managed_worker(workers[0].container_name)

    assert workers[0].worker_index == 1
    assert workers[0].state == "running"
    assert workers[0].runtime_fingerprint == "deadbeef"
    assert runner.calls[-1][0] == (
        "docker",
        "rm",
        "-f",
        "comfy-remote-gpu-host-deadbeef-w1",
    )


def test_managed_worker_removal_rejects_unowned_container(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
) -> None:
    """A caller cannot use the managed-worker API to remove arbitrary containers."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )

    with pytest.raises(ssh_docker_module.SshDockerError, match="is not managed"):
        controller.remove_managed_worker("unrelated-service")


def test_idle_managed_worker_can_be_recycled_to_release_resident_gpu_memory(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
) -> None:
    """A warm worker with only its server and resident model is reclaimable."""
    runner = _FakeRunner(ssh_docker_module)
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )

    removed = controller.remove_idle_managed_workers()

    assert removed == ("comfy-remote-gpu-host-deadbeef-w1",)
    assert runner.calls[-1][0] == (
        "docker",
        "rm",
        "-f",
        "comfy-remote-gpu-host-deadbeef-w1",
    )


@pytest.mark.parametrize(
    "active_process",
    (
        "python -m remote.ssh_worker client",
        "python -m remote.ssh_worker stage-profiles --model-reference owner/model",
    ),
)
def test_active_managed_worker_is_never_recycled(
    ssh_docker_module: Any,
    remote_hosts_module: Any,
    active_process: str,
) -> None:
    """Execution and model-staging relay processes make a worker non-reclaimable."""
    runner = _FakeRunner(
        ssh_docker_module,
        worker_processes=(
            "python -m remote.ssh_worker serve",
            active_process,
        ),
    )
    controller = ssh_docker_module.SshDockerController(
        _host(remote_hosts_module),
        runner=runner,
    )

    removed = controller.remove_idle_managed_workers()

    assert removed == ()
    assert not any(call[0][:3] == ("docker", "rm", "-f") for call in runner.calls)
