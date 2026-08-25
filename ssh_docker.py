"""SSH transport, Docker host discovery, and remote named-volume access."""

from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

if __package__:
    from .execution_environments import EnvironmentCapabilities, GpuCapability
    from .r2_cache import R2DownloadRequest, R2UploadPlan, R2UploadResult
    from .remote_hosts import SshHostConfig
else:  # pragma: no cover - stable remote entrypoints may import modules top-level.
    from execution_environments import EnvironmentCapabilities, GpuCapability
    from r2_cache import R2DownloadRequest, R2UploadPlan, R2UploadResult
    from remote_hosts import SshHostConfig

logger = logging.getLogger(__name__)

DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_VOLUME_HELPER_IMAGE = "busybox:1.37.0"
_MIB = 1024 * 1024
_ACTIVE_MANAGED_WORKER_PROCESS_MARKERS = (
    "remote.ssh_worker client",
    "remote.ssh_worker stage-profiles",
)
_INACTIVE_CONTAINER_STATES = frozenset({"created", "dead", "exited"})


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


@dataclass(frozen=True)
class SshManagedWorkerStatus:
    """Describe one node-pack-managed Docker worker on an SSH host."""

    container_name: str
    image: str
    state: str
    status: str
    runtime_fingerprint: str | None
    worker_index: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible worker status record."""
        return {
            "container_name": self.container_name,
            "image": self.image,
            "state": self.state,
            "status": self.status,
            "runtime_fingerprint": self.runtime_fingerprint,
            "worker_index": self.worker_index,
        }


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
            "ClearAllForwardings=yes",
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
        reported_cost = _reported_cost_usd_per_second(docker_info)
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
            reported_cost_usd_per_second=reported_cost,
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

    def list_managed_workers(self) -> tuple[SshManagedWorkerStatus, ...]:
        """Return containers carrying this environment's ownership label."""
        result = self.docker(
            (
                "ps",
                "-a",
                "--filter",
                f"label=comfy.remote.environment-id={self.host.environment_id}",
                "--format",
                "{{json .}}",
            )
        )
        workers: list[SshManagedWorkerStatus] = []
        for line in result.stdout_text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SshDockerError(
                    "Docker returned invalid managed-worker status JSON."
                ) from exc
            if not isinstance(payload, Mapping):
                raise SshDockerError(
                    "Docker returned a non-object managed-worker status record."
                )
            workers.append(_managed_worker_status_from_payload(payload))
        return tuple(
            sorted(
                workers,
                key=lambda worker: (
                    worker.worker_index if worker.worker_index is not None else 10**9,
                    worker.container_name,
                ),
            )
        )

    def remove_managed_worker(self, container_name: str) -> None:
        """Remove one exact managed container after validating its ownership label."""
        normalized_name = _validated_docker_object_name(container_name)
        managed_names = {
            worker.container_name for worker in self.list_managed_workers()
        }
        if normalized_name not in managed_names:
            raise SshDockerError(
                f"Container {normalized_name!r} is not managed by environment "
                f"{self.host.environment_id!r}."
            )
        self.docker(("rm", "-f", normalized_name))

    def managed_worker_is_idle(self, worker: SshManagedWorkerStatus) -> bool:
        """Return whether a worker has no active execution or staging client."""
        normalized_state = worker.state.strip().lower()
        if normalized_state in _INACTIVE_CONTAINER_STATES:
            return True
        if normalized_state != "running":
            return False
        result = self.docker(
            ("top", worker.container_name, "-eo", "pid,args"),
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "Could not inspect managed SSH worker activity environment=%s "
                "container=%s: %s",
                self.host.environment_id,
                worker.container_name,
                result.stderr_text.strip() or "docker top failed",
            )
            return False
        process_rows = tuple(
            line.strip().lower()
            for line in result.stdout_text.splitlines()[1:]
            if line.strip()
        )
        return not any(
            marker in process_row
            for process_row in process_rows
            for marker in _ACTIVE_MANAGED_WORKER_PROCESS_MARKERS
        )

    def remove_idle_managed_workers(self) -> tuple[str, ...]:
        """Remove managed workers that are currently safe to recycle for capacity."""
        removed_names: list[str] = []
        for worker in self.list_managed_workers():
            if not self.managed_worker_is_idle(worker):
                continue
            self.docker(("rm", "-f", worker.container_name))
            removed_names.append(worker.container_name)
        return tuple(removed_names)

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
    materializer_image: str | None = None
    _exists_cache: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and ensure the configured storage volume."""
        self.volume_name = self.controller.ensure_volume(self.volume_name)
        if self.materializer_image is not None and (
            not self.materializer_image.strip()
            or any(
                character in self.materializer_image
                for character in ("\x00", "\n", "\r")
            )
        ):
            raise ValueError("SSH R2 materializer image must be a non-empty single line.")

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

    def materialize_r2_file(
        self,
        request: R2DownloadRequest,
        remote_path: str,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> None:
        """Download and verify one signed R2 object inside the worker image."""
        if cancellation_check is not None and cancellation_check():
            raise InterruptedError("SSH R2 download was cancelled.")
        normalized_path = _validated_volume_path(remote_path)
        self._run_r2_materializer(
            {
                "operation": "download",
                "storage_root": "/storage",
                "remote_path": normalized_path,
                "download": request.to_dict(),
            },
            size_bytes=request.size_bytes,
        )
        if cancellation_check is not None and cancellation_check():
            raise InterruptedError("SSH R2 download was cancelled.")
        self._exists_cache[normalized_path] = True

    def upload_r2_file(
        self,
        plan: R2UploadPlan,
        remote_path: str,
    ) -> R2UploadResult:
        """Upload one named-volume file through a signed R2 transfer plan."""
        normalized_path = _validated_volume_path(remote_path)
        result = self._run_r2_materializer(
            {
                "operation": "upload",
                "storage_root": "/storage",
                "remote_path": normalized_path,
                "upload": plan.to_dict(),
            },
            size_bytes=plan.size_bytes,
        )
        return R2UploadResult.from_dict(result)

    def _run_r2_materializer(
        self,
        request: Mapping[str, Any],
        *,
        size_bytes: int,
    ) -> Mapping[str, object]:
        """Run signed transfers with URLs supplied only through protected stdin."""
        if self.materializer_image is None:
            raise SshDockerError("SSH R2 backing requires a current worker image.")
        result = self.controller.docker(
            (
                "run",
                "--rm",
                "-i",
                "--entrypoint",
                "python",
                "-v",
                f"{self.volume_name}:/storage",
                self.materializer_image,
                "-m",
                "remote.r2_materializer",
            ),
            input_payload=json.dumps(request, sort_keys=True).encode("utf-8"),
            timeout_seconds=max(900.0, size_bytes / (2 * _MIB)),
        )
        try:
            payload = json.loads(result.stdout_text)
        except json.JSONDecodeError as exc:
            raise SshDockerError("SSH R2 materializer returned invalid JSON.") from exc
        if not isinstance(payload, Mapping):
            raise SshDockerError("SSH R2 materializer returned a non-object result.")
        return payload

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


def _parse_docker_label_text(value: str) -> dict[str, str]:
    """Parse Docker's comma-separated label rendering into a mapping."""
    labels: dict[str, str] = {}
    for item in value.split(","):
        key, separator, label_value = item.partition("=")
        if not separator or not key.strip():
            continue
        labels[key.strip()] = label_value.strip()
    return labels


def _reported_cost_usd_per_second(docker_info: Mapping[str, Any]) -> float | None:
    """Return an infrastructure-reported cost from a Docker daemon label."""
    raw_labels = docker_info.get("Labels")
    label_values = raw_labels if isinstance(raw_labels, list) else []
    labels = _parse_docker_label_text(",".join(str(value) for value in label_values))
    raw_cost = labels.get("comfy.remote.cost-usd-per-second")
    if raw_cost is None:
        return None
    try:
        cost = float(raw_cost)
    except ValueError:
        logger.warning("Ignoring invalid Docker host cost label %r.", raw_cost)
        return None
    if not math.isfinite(cost) or cost < 0:
        logger.warning("Ignoring invalid Docker host cost label %r.", raw_cost)
        return None
    return cost


def _managed_worker_status_from_payload(
    payload: Mapping[str, Any],
) -> SshManagedWorkerStatus:
    """Normalize one Docker `ps --format json` record."""
    labels = _parse_docker_label_text(str(payload.get("Labels") or ""))
    raw_worker_index = labels.get("comfy.remote.worker-index")
    try:
        worker_index = int(raw_worker_index) if raw_worker_index is not None else None
    except ValueError:
        worker_index = None
    return SshManagedWorkerStatus(
        container_name=str(payload.get("Names") or "").strip(),
        image=str(payload.get("Image") or "").strip(),
        state=str(payload.get("State") or "unknown").strip().lower(),
        status=str(payload.get("Status") or "unknown").strip(),
        runtime_fingerprint=labels.get("comfy.remote.runtime-fingerprint"),
        worker_index=worker_index,
    )


def _parse_positive_int(value: str, field_name: str) -> int:
    """Parse one positive integer returned by a remote capability command."""
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise SshDockerError(f"Invalid {field_name} value {value!r}.") from exc
    if parsed <= 0:
        raise SshDockerError(f"Invalid {field_name} value {value!r}.")
    return parsed
