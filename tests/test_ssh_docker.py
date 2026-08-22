"""Tests for SSH Docker discovery and named-volume storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class _FakeRunner:
    """Return deterministic results for remote capability commands."""

    def __init__(self, ssh_docker_module: Any) -> None:
        """Initialize the fake with the module's result type."""
        self.module = ssh_docker_module
        self.calls: list[tuple[tuple[str, ...], bytes | None]] = []

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
        elif command[:2] == ("docker", "run"):
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
