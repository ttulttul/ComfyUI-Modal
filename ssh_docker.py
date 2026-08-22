"""SSH transport, Docker host discovery, and remote named-volume access."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

if __package__:
    from .execution_environments import EnvironmentCapabilities, GpuCapability
    from .remote_hosts import SshHostConfig
else:  # pragma: no cover - stable remote entrypoints may import modules top-level.
    from execution_environments import EnvironmentCapabilities, GpuCapability
    from remote_hosts import SshHostConfig

logger = logging.getLogger(__name__)

DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_VOLUME_HELPER_IMAGE = "busybox:1.37.0"
_MIB = 1024 * 1024


class SshDockerError(RuntimeError):
    """Raised when an SSH or remote Docker operation fails."""


@dataclass(frozen=True)
class SshCommandResult:
    """Capture one completed SSH command without exposing input payloads."""

    stdout: bytes
    stderr: bytes
    returncode: int

    @property
    def stdout_text(self) -> str:
        """Decode standard output as UTF-8 text."""
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        """Decode standard error as UTF-8 text."""
        return self.stderr.decode("utf-8", errors="replace")


@dataclass
class SshCommandRunner:
    """Run fixed remote commands through the system SSH client."""

    ssh_target: str
    connect_timeout_seconds: int = DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS
    command_timeout_seconds: float = DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS
    ssh_binary: str = "ssh"

    def command(self, remote_argv: Sequence[str]) -> list[str]:
        """Return the local SSH argv for one remote argv sequence."""
        if not remote_argv:
            raise ValueError("remote_argv must contain at least one command value.")
        remote_command = shlex.join([str(value) for value in remote_argv])
        return [
            self.ssh_binary,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_seconds}",
            "--",
            self.ssh_target,
            remote_command,
        ]

    def run(
        self,
        remote_argv: Sequence[str],
        *,
        input_payload: bytes | None = None,
        timeout_seconds: float | None = None,
        check: bool = True,
    ) -> SshCommandResult:
        """Execute an argv-safe remote command through SSH."""
        command = self.command(remote_argv)
        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=input_payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=(
                    self.command_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            )
        except FileNotFoundError as exc:
            raise SshDockerError(
                f"The SSH client {self.ssh_binary!r} is not installed."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SshDockerError(
                f"SSH command to {self.ssh_target!r} timed out after "
                f"{exc.timeout} seconds."
            ) from exc
        result = SshCommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
        logger.debug(
            "Finished SSH command target=%s command=%s returncode=%d elapsed_seconds=%.3f.",
            self.ssh_target,
            remote_argv[0],
            result.returncode,
            time.monotonic() - started_at,
        )
        if check and result.returncode != 0:
            error_text = result.stderr_text.strip() or result.stdout_text.strip()
            raise SshDockerError(
                f"SSH command {remote_argv[0]!r} failed on {self.ssh_target!r} "
                f"with exit status {result.returncode}: {error_text or 'no error output'}"
            )
        return result

    def popen(self, remote_argv: Sequence[str]) -> subprocess.Popen[bytes]:
        """Start one streaming SSH command with binary standard streams."""
        try:
            return subprocess.Popen(
                self.command(remote_argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SshDockerError(
                f"The SSH client {self.ssh_binary!r} is not installed."
            ) from exc


@dataclass
class SshDockerController:
    """Inspect and control one Docker daemon through an SSH destination."""

    host: SshHostConfig
    runner: SshCommandRunner | None = None

    def __post_init__(self) -> None:
        """Create the default SSH command runner when none was injected."""
        if self.runner is None:
            self.runner = SshCommandRunner(self.host.ssh_target)

    def probe_capabilities(self) -> EnvironmentCapabilities:
        """Discover host, Docker, RAM, disk, and NVIDIA GPU capabilities."""
        runner = self._runner()
        docker_info = self._docker_info()
        architecture = self._text_command(("uname", "-m"))
        operating_system = self._text_command(("uname", "-s")).lower()
        cpu_count = _parse_positive_int(
            self._text_command(("getconf", "_NPROCESSORS_ONLN")),
            "remote CPU count",
        )
        memory_values = self._memory_info()
        available_disk_bytes = self._available_disk_bytes()
        gpus = self._nvidia_gpus()
        runtimes = docker_info.get("Runtimes")
        runtime_names = set(runtimes) if isinstance(runtimes, Mapping) else set()
        cdi_directories = docker_info.get("CDISpecDirs")
        cdi_available = isinstance(cdi_directories, list) and bool(cdi_directories)
        security_options = docker_info.get("SecurityOptions")
        security_values = security_options if isinstance(security_options, list) else []
        docker_rootless = any("rootless" in str(value).lower() for value in security_values)
        server_version = str(
            docker_info.get("ServerVersion")
            or docker_info.get("ServerVersionRaw")
            or "unknown"
        ).strip()
        return EnvironmentCapabilities(
            architecture=architecture,
            operating_system=operating_system,
            cpu_count=cpu_count,
            total_ram_bytes=memory_values["MemTotal"] * 1024,
            available_ram_bytes=memory_values.get("MemAvailable", 0) * 1024,
            available_disk_bytes=available_disk_bytes,
            docker_version=server_version,
            docker_rootless=docker_rootless,
            nvidia_container_runtime=bool(gpus) and (
                "nvidia" in runtime_names or cdi_available
            ),
            gpus=tuple(gpus),
            probed_at_epoch=time.time(),
        )

    def ensure_volume(self, volume_name: str) -> str:
        """Create the named Docker volume when absent and return its name."""
        normalized_name = _validated_docker_object_name(volume_name)
        result = self._runner().run(("docker", "volume", "create", normalized_name))
        created_name = result.stdout_text.strip()
        if created_name != normalized_name:
            raise SshDockerError(
                f"Docker returned unexpected volume name {created_name!r}."
            )
        return normalized_name

    def docker(
        self,
        arguments: Sequence[str],
        *,
        input_payload: bytes | None = None,
        timeout_seconds: float | None = None,
        check: bool = True,
    ) -> SshCommandResult:
        """Run one remote Docker CLI operation."""
        return self._runner().run(
            ("docker", *arguments),
            input_payload=input_payload,
            timeout_seconds=timeout_seconds,
            check=check,
        )

    def docker_popen(self, arguments: Sequence[str]) -> subprocess.Popen[bytes]:
        """Start one streaming remote Docker CLI operation."""
        return self._runner().popen(("docker", *arguments))

    def _runner(self) -> SshCommandRunner:
        """Return the initialized SSH command runner."""
        if self.runner is None:
            raise RuntimeError("SSH command runner was not initialized.")
        return self.runner

    def _text_command(self, command: Sequence[str]) -> str:
        """Run one remote command and return stripped UTF-8 output."""
        output = self._runner().run(command).stdout_text.strip()
        if not output:
            raise SshDockerError(
                f"Remote capability command {command[0]!r} returned no output."
            )
        return output

    def _docker_info(self) -> Mapping[str, Any]:
        """Return Docker daemon information as a JSON mapping."""
        result = self.docker(("info", "--format", "{{json .}}"))
        try:
            payload = json.loads(result.stdout_text)
        except json.JSONDecodeError as exc:
            raise SshDockerError("Remote Docker returned invalid info JSON.") from exc
        if not isinstance(payload, Mapping):
            raise SshDockerError("Remote Docker info JSON must be an object.")
        return payload

    def _memory_info(self) -> dict[str, int]:
        """Return selected `/proc/meminfo` values in KiB."""
        result = self._runner().run(("cat", "/proc/meminfo"))
        values: dict[str, int] = {}
        for line in result.stdout_text.splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator or key not in {"MemTotal", "MemAvailable"}:
                continue
            value_text = raw_value.strip().split()[0]
            values[key] = _parse_positive_int(value_text, key)
        if "MemTotal" not in values:
            raise SshDockerError("Remote /proc/meminfo omitted MemTotal.")
        return values

    def _available_disk_bytes(self) -> int | None:
        """Return available bytes on Docker's root filesystem when measurable."""
        result = self._runner().run(
            ("df", "-Pk", "/var/lib/docker"),
            check=False,
        )
        if result.returncode != 0:
            result = self._runner().run(("df", "-Pk", "/"), check=False)
        lines = [line for line in result.stdout_text.splitlines() if line.strip()]
        if result.returncode != 0 or len(lines) < 2:
            return None
        fields = lines[-1].split()
        if len(fields) < 4:
            return None
        try:
            return int(fields[3]) * 1024
        except ValueError:
            return None

    def _nvidia_gpus(self) -> list[GpuCapability]:
        """Return NVIDIA GPU details, or an empty list on CPU-only hosts."""
        query_fields = (
            "uuid,name,memory.total,memory.free,compute_cap,driver_version"
        )
        result = self._runner().run(
            (
                "nvidia-smi",
                f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits",
            ),
            check=False,
        )
        compute_capability_included = result.returncode == 0
        if result.returncode != 0:
            query_fields = "uuid,name,memory.total,memory.free,driver_version"
            result = self._runner().run(
                (
                    "nvidia-smi",
                    f"--query-gpu={query_fields}",
                    "--format=csv,noheader,nounits",
                ),
                check=False,
            )
            compute_capability_included = False
        if result.returncode != 0:
            return []

        gpus: list[GpuCapability] = []
        for line in result.stdout_text.splitlines():
            fields = [field.strip() for field in line.split(",")]
            expected_fields = 6 if compute_capability_included else 5
            if len(fields) != expected_fields:
                logger.warning("Ignoring malformed nvidia-smi GPU row: %s", line)
                continue
            uuid_value, name, total_mib, free_mib = fields[:4]
            compute_capability = fields[4] if compute_capability_included else None
            driver_version = fields[5] if compute_capability_included else fields[4]
            gpus.append(
                GpuCapability(
                    uuid=uuid_value,
                    name=name,
                    total_vram_bytes=_parse_positive_int(total_mib, "GPU VRAM") * _MIB,
                    free_vram_bytes=_parse_positive_int(free_mib, "free GPU VRAM") * _MIB,
                    compute_capability=compute_capability,
                    driver_version=driver_version,
                )
            )
        return gpus


@dataclass
class SshDockerVolumeBackend:
    """Store content-addressed files in a named volume on an SSH Docker host."""

    remote_volume_epoch_scoped = True

    controller: SshDockerController
    volume_name: str
    helper_image: str = DEFAULT_VOLUME_HELPER_IMAGE
    _exists_cache: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and ensure the configured storage volume."""
        self.volume_name = self.controller.ensure_volume(self.volume_name)

    def exists(self, remote_path: str) -> bool:
        """Return whether one regular file exists in the remote volume."""
        normalized_path = _validated_volume_path(remote_path)
        cached = self._exists_cache.get(normalized_path)
        if cached is not None:
            return cached
        result = self.controller.docker(
            (
                "run",
                "--rm",
                "-e",
                f"COMFY_REMOTE_PATH={normalized_path}",
                "-v",
                f"{self.volume_name}:/storage",
                self.helper_image,
                "sh",
                "-ceu",
                'test -f "/storage/$COMFY_REMOTE_PATH"',
            ),
            check=False,
        )
        exists = result.returncode == 0
        self._exists_cache[normalized_path] = exists
        return exists

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload one local file into the named volume."""
        resolved_path = local_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Asset not found: {resolved_path}")
        self._put_payload(resolved_path.read_bytes(), remote_path)

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload bytes into the named volume."""
        self._put_payload(payload, remote_path)

    def _put_payload(self, payload: bytes, remote_path: str) -> None:
        """Atomically stream one payload into the remote named volume."""
        normalized_path = _validated_volume_path(remote_path)
        self.controller.docker(
            (
                "run",
                "--rm",
                "-i",
                "-e",
                f"COMFY_REMOTE_PATH={normalized_path}",
                "-v",
                f"{self.volume_name}:/storage",
                self.helper_image,
                "sh",
                "-ceu",
                (
                    'target="/storage/$COMFY_REMOTE_PATH"; '
                    'mkdir -p "$(dirname "$target")"; '
                    'temporary="$target.tmp.$$"; '
                    'trap \'rm -f "$temporary"\' EXIT; '
                    'cat > "$temporary"; chmod 600 "$temporary"; '
                    'mv -f "$temporary" "$target"; trap - EXIT'
                ),
            ),
            input_payload=payload,
            timeout_seconds=max(60.0, len(payload) / (4 * 1024 * 1024)),
        )
        self._exists_cache[normalized_path] = True


def _validated_volume_path(remote_path: str) -> str:
    """Return one safe path relative to the mounted storage root."""
    path = PurePosixPath(str(remote_path).lstrip("/"))
    if (
        not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in str(path)
    ):
        raise ValueError(f"Unsafe remote volume path {remote_path!r}.")
    return path.as_posix()


def _validated_docker_object_name(name: str) -> str:
    """Return a conservative Docker object name."""
    normalized = str(name).strip()
    if not normalized or not all(
        character.isalnum() or character in "_.-" for character in normalized
    ):
        raise ValueError(f"Unsafe Docker object name {name!r}.")
    return normalized


def _parse_positive_int(value: str, field_name: str) -> int:
    """Parse one positive integer returned by a remote capability command."""
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise SshDockerError(f"Invalid {field_name} value {value!r}.") from exc
    if parsed <= 0:
        raise SshDockerError(f"Invalid {field_name} value {value!r}.")
    return parsed
