"""Direct SSH command and filesystem storage adapters for Vast containers."""

from __future__ import annotations

import json
import logging
import os
import random
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Sequence

if __package__:
    from .huggingface_assets import HuggingFaceAssetSource
else:  # pragma: no cover - direct debugging imports.
    from huggingface_assets import HuggingFaceAssetSource

logger = logging.getLogger(__name__)

DEFAULT_VAST_SSH_CONNECT_TIMEOUT_SECONDS = 15
DEFAULT_VAST_SSH_COMMAND_TIMEOUT_SECONDS = 120.0
DEFAULT_VAST_SSH_RETRY_ATTEMPTS = 4
DEFAULT_VAST_SSH_RETRY_BASE_DELAY_SECONDS = 0.5
DEFAULT_VAST_SSH_RETRY_MAX_DELAY_SECONDS = 4.0
DEFAULT_VAST_STORAGE_ROOT = PurePosixPath("/storage")
_TRANSIENT_SSH_TRANSPORT_DIAGNOSTICS = (
    "connection closed by",
    "connection reset by",
    "connection refused",
    "connection timed out",
    "kex_exchange_identification:",
    "network is unreachable",
    "no route to host",
    "operation timed out",
    "remote host has disconnected",
    "ssh_exchange_identification:",
)
_ATOMIC_STDIN_WRITER = """\
import os
import pathlib
import shutil
import sys
import tempfile

expected_size = int(sys.argv[1])
target = pathlib.Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(
    prefix=f".{target.name}.",
    suffix=".tmp",
    dir=target.parent,
)
try:
    with os.fdopen(fd, "wb", buffering=0) as output:
        shutil.copyfileobj(sys.stdin.buffer, output, length=1048576)
        actual_size = output.tell()
        if actual_size != expected_size:
            raise ValueError(
                f"Incomplete SSH upload: expected {expected_size} bytes, "
                f"received {actual_size}."
            )
        output.flush()
        os.fsync(output.fileno())
    os.chmod(name, 0o600)
    os.replace(name, target)
except (OSError, ValueError):
    pathlib.Path(name).unlink(missing_ok=True)
    raise
"""


class VastSshError(RuntimeError):
    """Raised when a direct Vast SSH or filesystem operation fails."""


@dataclass(frozen=True)
class VastSshConnection:
    """Describe one Vast instance SSH endpoint without private credentials."""

    host: str
    port: int
    known_hosts_path: Path
    user: str = "root"
    identity_file: Path | None = None

    def __post_init__(self) -> None:
        """Validate endpoint and local trust-store paths."""
        if not self.host.strip() or any(
            character in self.host for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("Vast SSH host must be a non-empty single-line value.")
        if self.host.startswith("-"):
            raise ValueError("Vast SSH host must not begin with an option prefix.")
        if isinstance(self.port, bool) or not (1 <= self.port <= 65535):
            raise ValueError("Vast SSH port must be between 1 and 65535.")
        if not self.user.strip() or not all(
            character.isalnum() or character in "_.-" for character in self.user
        ):
            raise ValueError("Vast SSH user contains unsafe characters.")
        if not self.known_hosts_path.is_absolute():
            raise ValueError("Vast known_hosts_path must be absolute.")
        if self.identity_file is not None and not self.identity_file.is_absolute():
            raise ValueError("Vast SSH identity_file must be absolute when configured.")

    @property
    def target(self) -> str:
        """Return the OpenSSH user and host destination."""
        return f"{self.user}@{self.host}"


@dataclass(frozen=True)
class VastSshCommandResult:
    """Capture a completed direct SSH command."""

    stdout: bytes
    stderr: bytes
    returncode: int

    @property
    def stdout_text(self) -> str:
        """Decode standard output as replacement-safe UTF-8."""
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        """Decode standard error as replacement-safe UTF-8."""
        return self.stderr.decode("utf-8", errors="replace")


def _is_transient_ssh_transport_failure(result: VastSshCommandResult) -> bool:
    """Return whether OpenSSH reports a retryable connection-level failure."""
    if result.returncode != 255:
        return False
    diagnostic = f"{result.stderr_text}\n{result.stdout_text}".casefold()
    return any(
        marker in diagnostic for marker in _TRANSIENT_SSH_TRANSPORT_DIAGNOSTICS
    )


def _resolved_input_file(
    input_payload: bytes | None,
    input_file: Path | None,
) -> Path | None:
    """Validate mutually exclusive SSH input sources and resolve a file source."""
    if input_payload is not None and input_file is not None:
        raise ValueError("Vast SSH input_payload and input_file are mutually exclusive.")
    resolved = input_file.expanduser().resolve() if input_file is not None else None
    if resolved is not None and not resolved.is_file():
        raise FileNotFoundError(f"Vast SSH input file not found: {resolved}")
    return resolved


@dataclass
class VastSshRunner:
    """Run argv-safe commands inside one Vast instance container."""

    connection: VastSshConnection
    connect_timeout_seconds: int = DEFAULT_VAST_SSH_CONNECT_TIMEOUT_SECONDS
    command_timeout_seconds: float = DEFAULT_VAST_SSH_COMMAND_TIMEOUT_SECONDS
    retry_attempts: int = DEFAULT_VAST_SSH_RETRY_ATTEMPTS
    retry_base_delay_seconds: float = DEFAULT_VAST_SSH_RETRY_BASE_DELAY_SECONDS
    retry_max_delay_seconds: float = DEFAULT_VAST_SSH_RETRY_MAX_DELAY_SECONDS
    ssh_binary: str = "ssh"
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    random_unit: Callable[[], float] = field(default=random.random, repr=False)

    def __post_init__(self) -> None:
        """Create the dedicated trust store without weakening global SSH policy."""
        if self.connect_timeout_seconds <= 0 or self.command_timeout_seconds <= 0:
            raise ValueError("Vast SSH timeouts must be positive.")
        if isinstance(self.retry_attempts, bool) or self.retry_attempts <= 0:
            raise ValueError("Vast SSH retry attempts must be positive.")
        if (
            self.retry_base_delay_seconds <= 0
            or self.retry_max_delay_seconds < self.retry_base_delay_seconds
        ):
            raise ValueError("Vast SSH retry delays must be positive and ordered.")
        try:
            self.connection.known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection.known_hosts_path.touch(mode=0o600, exist_ok=True)
            os.chmod(self.connection.known_hosts_path, 0o600)
        except OSError as exc:
            raise VastSshError(
                f"Unable to initialize Vast SSH trust store at "
                f"{self.connection.known_hosts_path}."
            ) from exc

    def command(self, remote_argv: Sequence[str]) -> list[str]:
        """Return the local OpenSSH argv for one remote argv sequence."""
        if not remote_argv:
            raise ValueError("remote_argv must contain at least one command value.")
        normalized_argv = [str(value) for value in remote_argv]
        if any("\x00" in value for value in normalized_argv):
            raise ValueError("remote_argv must not contain null bytes.")
        command = [
            self.ssh_binary,
            "-p",
            str(self.connection.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.connection.known_hosts_path}",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_seconds}",
        ]
        if self.connection.identity_file is not None:
            command.extend(("-i", str(self.connection.identity_file)))
        command.extend(("--", self.connection.target, shlex.join(normalized_argv)))
        return command

    def run(
        self,
        remote_argv: Sequence[str],
        *,
        input_payload: bytes | None = None,
        input_file: Path | None = None,
        timeout_seconds: float | None = None,
        check: bool = True,
    ) -> VastSshCommandResult:
        """Execute one command with bounded transient transport retries."""
        resolved_input_file = _resolved_input_file(input_payload, input_file)
        started_at = time.monotonic()
        for attempt in range(1, self.retry_attempts + 1):
            result = self._run_attempt(
                remote_argv,
                input_payload=input_payload,
                input_file=resolved_input_file,
                timeout_seconds=timeout_seconds,
            )
            if _is_transient_ssh_transport_failure(result):
                if attempt >= self.retry_attempts:
                    raise self._command_error(
                        remote_argv,
                        result,
                        suffix=f" after {attempt} transport attempts",
                    )
                delay_seconds = self._retry_delay_seconds(attempt)
                logger.warning(
                    "Retrying transient Vast SSH transport failure host=%s "
                    "command=%s attempt=%d/%d delay_seconds=%.2f diagnostic=%s",
                    self.connection.host,
                    remote_argv[0],
                    attempt,
                    self.retry_attempts,
                    delay_seconds,
                    result.stderr_text.strip() or result.stdout_text.strip(),
                )
                self.sleep(delay_seconds)
                continue
            logger.debug(
                "Finished Vast SSH command host=%s command=%s returncode=%d "
                "attempts=%d elapsed_seconds=%.3f.",
                self.connection.host,
                remote_argv[0],
                result.returncode,
                attempt,
                time.monotonic() - started_at,
            )
            if check and result.returncode != 0:
                raise self._command_error(remote_argv, result)
            return result
        raise RuntimeError("Vast SSH retry loop ended without a result.")

    def _run_attempt(
        self,
        remote_argv: Sequence[str],
        *,
        input_payload: bytes | None,
        input_file: Path | None,
        timeout_seconds: float | None,
    ) -> VastSshCommandResult:
        """Execute one OpenSSH subprocess attempt, reopening streamed input files."""
        try:
            if input_file is None:
                completed = self._run_subprocess(
                    remote_argv,
                    input_payload=input_payload,
                    input_handle=None,
                    timeout_seconds=timeout_seconds,
                )
            else:
                with input_file.open("rb") as input_handle:
                    completed = self._run_subprocess(
                        remote_argv,
                        input_payload=None,
                        input_handle=input_handle,
                        timeout_seconds=timeout_seconds,
                    )
        except FileNotFoundError as exc:
            raise VastSshError(
                f"The SSH client {self.ssh_binary!r} is not installed."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise VastSshError(
                f"SSH command to Vast instance {self.connection.host!r} timed out "
                f"after {exc.timeout} seconds."
            ) from exc
        return VastSshCommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    def _retry_delay_seconds(self, failed_attempt: int) -> float:
        """Return exponential retry delay with bounded per-attempt jitter."""
        base_delay = min(
            self.retry_max_delay_seconds,
            self.retry_base_delay_seconds * (2 ** (failed_attempt - 1)),
        )
        jitter_unit = min(1.0, max(0.0, float(self.random_unit())))
        return base_delay * (0.75 + (0.5 * jitter_unit))

    @staticmethod
    def _command_error(
        remote_argv: Sequence[str],
        result: VastSshCommandResult,
        *,
        suffix: str = "",
    ) -> VastSshError:
        """Build one bounded, actionable SSH command failure."""
        detail = result.stderr_text.strip() or result.stdout_text.strip()
        return VastSshError(
            f"Vast SSH command {remote_argv[0]!r} failed with exit status "
            f"{result.returncode}{suffix}: {detail or 'no diagnostic output'}"
        )

    def _run_subprocess(
        self,
        remote_argv: Sequence[str],
        *,
        input_payload: bytes | None,
        input_handle: BinaryIO | None,
        timeout_seconds: float | None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run OpenSSH with either a small byte payload or a streamed file handle."""
        timeout = (
            self.command_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if input_handle is not None:
            return subprocess.run(
                self.command(remote_argv),
                stdin=input_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
        return subprocess.run(
            self.command(remote_argv),
            input=input_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )

    def popen(self, remote_argv: Sequence[str]) -> subprocess.Popen[bytes]:
        """Start one streaming direct SSH command."""
        try:
            return subprocess.Popen(
                self.command(remote_argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise VastSshError(
                f"The SSH client {self.ssh_binary!r} is not installed."
            ) from exc


@dataclass
class VastSshVolumeBackend:
    """Store content-addressed files directly on a Vast instance filesystem."""

    remote_volume_epoch_scoped = True

    runner: VastSshRunner
    storage_root: PurePosixPath = DEFAULT_VAST_STORAGE_ROOT
    _exists_cache: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require an absolute non-root storage location."""
        if not self.storage_root.is_absolute() or self.storage_root == PurePosixPath("/"):
            raise ValueError("Vast storage_root must be an absolute non-root path.")
        self.runner.run(("mkdir", "-p", str(self.storage_root)))

    def exists(self, remote_path: str) -> bool:
        """Return whether one regular file exists under the storage root."""
        normalized = _validated_storage_path(remote_path)
        cached = self._exists_cache.get(normalized)
        if cached is not None:
            return cached
        result = self.runner.run(
            ("test", "-f", str(self.storage_root / normalized)),
            check=False,
        )
        exists = result.returncode == 0
        self._exists_cache[normalized] = exists
        return exists

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Stream one local file through SSH and atomically publish it remotely."""
        resolved_path = local_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Asset not found: {resolved_path}")
        size_bytes = resolved_path.stat().st_size
        normalized = _validated_storage_path(remote_path)
        target = str(self.storage_root / normalized)
        self.runner.run(
            (
                "python",
                "-c",
                _ATOMIC_STDIN_WRITER,
                str(size_bytes),
                target,
            ),
            input_file=resolved_path,
            timeout_seconds=max(60.0, size_bytes / (4 * 1024 * 1024)),
        )
        self._exists_cache[normalized] = True

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Atomically upload bytes under the storage root."""
        self._put_payload(payload, remote_path)

    def _put_payload(self, payload: bytes, remote_path: str) -> None:
        """Stream bytes to an argv-safe remote Python atomic writer."""
        normalized = _validated_storage_path(remote_path)
        target = str(self.storage_root / normalized)
        self.runner.run(
            ("python", "-c", _ATOMIC_STDIN_WRITER, str(len(payload)), target),
            input_payload=payload,
            timeout_seconds=max(60.0, len(payload) / (4 * 1024 * 1024)),
        )
        self._exists_cache[normalized] = True

    def materialize_huggingface_file(
        self,
        source: HuggingFaceAssetSource,
        remote_path: str,
        *,
        token: str | None,
    ) -> bool:
        """Ask the Vast worker to fetch and verify one immutable Hugging Face file."""
        normalized = _validated_storage_path(remote_path)
        request_payload = json.dumps(
            {
                "remote_path": normalized,
                "source": source.to_dict(),
                "storage_root": str(self.storage_root),
                "token": token,
            },
            sort_keys=True,
        ).encode("utf-8")
        timeout_seconds = max(900.0, source.size_bytes / (2 * 1024 * 1024))
        try:
            self.runner.run(
                ("python", "-m", "remote.huggingface_materializer"),
                input_payload=request_payload,
                timeout_seconds=timeout_seconds,
            )
        except VastSshError as exc:
            logger.warning(
                "Vast Hugging Face materialization failed for %s; falling back to SSH upload: %s",
                source.display_reference,
                exc,
            )
            return False
        self._exists_cache[normalized] = True
        return True


def vast_connection_from_lease(
    *,
    ssh_host: str | None,
    ssh_port: int | None,
    user_directory: Path,
    identity_file: Path | None = None,
) -> VastSshConnection:
    """Build one direct connection from a ready lease's safe public fields."""
    if not ssh_host or ssh_port is None:
        raise VastSshError("Vast lease does not expose an SSH endpoint.")
    return VastSshConnection(
        host=ssh_host,
        port=ssh_port,
        known_hosts_path=(
            user_directory.expanduser().resolve()
            / "comfyui-modal"
            / "vast-known-hosts"
        ),
        identity_file=(identity_file.expanduser().resolve() if identity_file else None),
    )


def _validated_storage_path(remote_path: str) -> str:
    """Return one traversal-safe path relative to Vast storage."""
    path = PurePosixPath(str(remote_path).lstrip("/"))
    if (
        not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in str(path)
    ):
        raise ValueError(f"Unsafe Vast storage path {remote_path!r}.")
    return path.as_posix()


__all__ = [
    "DEFAULT_VAST_STORAGE_ROOT",
    "VastSshCommandResult",
    "VastSshConnection",
    "VastSshError",
    "VastSshRunner",
    "VastSshVolumeBackend",
    "vast_connection_from_lease",
]
