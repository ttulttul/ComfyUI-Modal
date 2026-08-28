"""Direct Vast worker launch settings, readiness, and watchdog state updates."""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

if __package__:
    from .remote.vast_watchdog import VastWatchdogSnapshot
    from .vast_leases import VastLeaseRecord
    from .vast_models import VastInstanceLaunchSpec, VastResourceProfile
    from .vast_ssh import VastSshError, VastSshRunner, VastSshVolumeBackend
else:  # pragma: no cover - direct debugging imports.
    from remote.vast_watchdog import VastWatchdogSnapshot
    from vast_leases import VastLeaseRecord
    from vast_models import VastInstanceLaunchSpec, VastResourceProfile
    from vast_ssh import VastSshError, VastSshRunner, VastSshVolumeBackend

logger = logging.getLogger(__name__)

VAST_IMAGE_ENV = "COMFY_MODAL_VAST_IMAGE"
DEFAULT_VAST_REMOTE_REPO_ROOT = PurePosixPath("/opt/comfy-remote/repo")
DEFAULT_VAST_REMOTE_COMFYUI_ROOT = PurePosixPath("/opt/comfy-remote/ComfyUI")
DEFAULT_VAST_REMOTE_STORAGE_ROOT = PurePosixPath("/storage")
DEFAULT_VAST_WATCHDOG_STATE_PATH = PurePosixPath(
    "/storage/comfy-vast-watchdog.json"
)


class VastRuntimeFingerprintDriftError(VastSshError):
    """Report that a published Vast worker does not match the local runtime."""

    def __init__(
        self,
        *,
        expected_fingerprint: str,
        actual_fingerprint: str,
        protocol_version: int,
    ) -> None:
        """Retain the compared identities for automated image replacement."""
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        self.protocol_version = protocol_version
        super().__init__(
            "Vast worker image source fingerprint drift: expected "
            f"{expected_fingerprint[:12]}, found "
            f"{actual_fingerprint[:12] or 'empty'}, protocol {protocol_version}."
        )


def _ssh_permission_repair_lines() -> tuple[str, ...]:
    """Return idempotent repairs for the SSH key path injected by Vast."""
    return (
        "chown root:root /root",
        "chmod go-w /root",
        "mkdir -p /root/.ssh",
        "chown root:root /root/.ssh",
        "chmod 700 /root/.ssh",
        "if [ -f /root/.ssh/authorized_keys ]; then",
        "  chown root:root /root/.ssh/authorized_keys",
        "  chmod 600 /root/.ssh/authorized_keys",
        "fi",
    )


@dataclass(frozen=True)
class VastRuntimeConfiguration:
    """Describe the digest-pinned worker image and direct runtime layout."""

    image: str
    runtime_fingerprint: str
    remote_repo_root: PurePosixPath = DEFAULT_VAST_REMOTE_REPO_ROOT
    remote_comfyui_root: PurePosixPath = DEFAULT_VAST_REMOTE_COMFYUI_ROOT
    remote_storage_root: PurePosixPath = DEFAULT_VAST_REMOTE_STORAGE_ROOT
    watchdog_state_path: PurePosixPath = DEFAULT_VAST_WATCHDOG_STATE_PATH
    startup_timeout_seconds: float = 900.0
    readiness_poll_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Validate launch and fingerprint values."""
        if not self.image.strip() or any(
            character in self.image for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("Vast runtime image must be a non-empty single-line value.")
        if len(self.runtime_fingerprint) != 64:
            raise ValueError("Vast runtime fingerprint must be a SHA-256 hex digest.")
        for field_name, path in (
            ("remote_repo_root", self.remote_repo_root),
            ("remote_comfyui_root", self.remote_comfyui_root),
            ("remote_storage_root", self.remote_storage_root),
            ("watchdog_state_path", self.watchdog_state_path),
        ):
            if not path.is_absolute() or path == PurePosixPath("/"):
                raise ValueError(f"{field_name} must be an absolute non-root path.")
        if self.startup_timeout_seconds <= 0 or self.readiness_poll_seconds <= 0:
            raise ValueError("Vast runtime readiness timing must be positive.")

    @classmethod
    def from_environment(
        cls,
        runtime_fingerprint: str,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "VastRuntimeConfiguration":
        """Resolve the required worker image without inventing an unpublished tag."""
        source = os.environ if environment is None else environment
        image = str(source.get(VAST_IMAGE_ENV) or "").strip()
        if not image:
            raise RuntimeError(
                f"Set {VAST_IMAGE_ENV} to a published ComfyUI-Modal Vast worker image "
                "before renting an instance."
            )
        return cls(image=image, runtime_fingerprint=runtime_fingerprint)

    def launch_spec(
        self,
        profile: VastResourceProfile,
        label: str,
    ) -> VastInstanceLaunchSpec:
        """Return one direct-SSH launch with worker and watchdog supervision."""
        environment = {
            "COMFY_MODAL_REMOTE_WORKER": "1",
            "COMFY_MODAL_LLM_EXECUTION_TARGET": "vast",
            "COMFYUI_ROOT": str(self.remote_comfyui_root),
            "COMFY_MODAL_COMFYUI_ROOT": str(self.remote_comfyui_root),
            "COMFY_MODAL_LOCAL_STORAGE_ROOT": str(self.remote_storage_root),
            "COMFY_MODAL_REMOTE_STORAGE_ROOT": str(self.remote_storage_root),
            "COMFY_MODAL_EXECUTION_MODE": "local",
            "PYTHONPATH": f"{self.remote_repo_root}:{self.remote_comfyui_root}",
        }
        exported_lines = "\n".join(
            f"{name}={value}" for name, value in sorted(environment.items())
        )
        supervisor_command = shlex.join(
            (
                "python",
                "-m",
                "remote.vast_supervisor",
                "start",
                "--watchdog-state",
                str(self.watchdog_state_path),
            )
        )
        onstart = "\n".join(
            (
                "set -e",
                *_ssh_permission_repair_lines(),
                "mkdir -p /run/comfy-remote /storage/logs",
                "cat >> /etc/environment <<'COMFY_VAST_ENV'",
                exported_lines,
                "COMFY_VAST_ENV",
                supervisor_command,
            )
        )
        return VastInstanceLaunchSpec(
            image=self.image,
            disk_gb=profile.allocated_disk_gb,
            label=label,
            onstart=onstart,
            environment=environment,
            rental_type=profile.rental_type,
        )


@dataclass
class VastRuntimeManager:
    """Ensure the direct worker process matches the selected runtime image."""

    runner: VastSshRunner
    configuration: VastRuntimeConfiguration
    fallback_runner: VastSshRunner | None = None
    fallback_selected: Callable[[], None] | None = None
    instance_validator: Callable[[], None] | None = None
    clock: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def ensure_worker(self) -> dict[str, Any]:
        """Wait for the supervised worker socket and exact runtime fingerprint."""
        started_at = self.monotonic()
        deadline = started_at + self.configuration.startup_timeout_seconds
        last_error: str | None = None
        last_logged_error: str | None = None
        attempt = 0
        while self.monotonic() < deadline:
            attempt += 1
            if self.instance_validator is not None:
                self.instance_validator()
            try:
                info = self.runtime_info()
            except (VastSshError, ValueError) as exc:
                last_error = str(exc)
            else:
                actual_fingerprint = str(info.get("runtime_fingerprint") or "")
                if actual_fingerprint != self.configuration.runtime_fingerprint:
                    actual_protocol = int(info.get("protocol_version") or 0)
                    raise VastRuntimeFingerprintDriftError(
                        expected_fingerprint=self.configuration.runtime_fingerprint,
                        actual_fingerprint=actual_fingerprint,
                        protocol_version=actual_protocol,
                    )
                if bool(info.get("worker_socket_ready")):
                    logger.info(
                        "Vast worker became ready after %d probe attempts (%.1fs).",
                        attempt,
                        self.monotonic() - started_at,
                    )
                    return info
                self._start_supervisor()
                last_error = "worker socket is not ready"
            if last_error != last_logged_error:
                logger.warning(
                    "Vast worker readiness probe attempt=%d failed: %s",
                    attempt,
                    last_error,
                )
                last_logged_error = last_error
            self.sleep(self.configuration.readiness_poll_seconds)
        raise TimeoutError(
            "Vast worker did not become ready within "
            f"{self.configuration.startup_timeout_seconds:.0f}s: "
            f"{last_error or 'no diagnostics'}."
        )

    def runtime_info(self) -> dict[str, Any]:
        """Return validated JSON from the direct worker image."""
        try:
            return self._runtime_info(self.runner)
        except VastSshError as primary_error:
            fallback_runner = self.fallback_runner
            if fallback_runner is None:
                raise
            try:
                payload = self._runtime_info(fallback_runner)
            except VastSshError as fallback_error:
                raise VastSshError(
                    "Vast direct and proxy SSH endpoints are both unavailable. "
                    f"Direct: {primary_error} Proxy: {fallback_error}"
                ) from fallback_error
            self.runner = fallback_runner
            self.fallback_runner = None
            if self.fallback_selected is not None:
                self.fallback_selected()
            logger.warning(
                "Vast direct SSH failed; continuing through the Vast SSH proxy."
            )
            return payload

    @staticmethod
    def _runtime_info(runner: VastSshRunner) -> dict[str, Any]:
        """Return validated runtime metadata through one exact SSH endpoint."""
        result = runner.run(
            ("python", "-m", "remote.ssh_worker", "runtime-info"),
            check=True,
            transport_attempts=1,
        )
        try:
            payload = json.loads(result.stdout_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Vast worker returned invalid runtime-info JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Vast worker runtime-info must be a JSON object.")
        return payload

    def restart_worker(self) -> None:
        """Restart only the managed worker while preserving watchdog and storage."""
        self.runner.run(
            (
                "python",
                "-m",
                "remote.vast_supervisor",
                "restart-worker",
                "--watchdog-state",
                str(self.configuration.watchdog_state_path),
            ),
            timeout_seconds=60.0,
        )
        self.ensure_worker()

    def update_watchdog(self, lease: VastLeaseRecord) -> None:
        """Atomically publish controller activity for the in-instance fail-safe."""
        if not math.isfinite(lease.idle_deadline_epoch):
            raise ValueError("Vast lease idle deadline must be finite.")
        snapshot = VastWatchdogSnapshot(
            instance_id=lease.instance_id,
            owner_label=lease.label,
            idle_deadline_epoch=lease.idle_deadline_epoch,
            active_invocations=lease.active_invocations,
            updated_at_epoch=self.clock(),
        )
        backend = VastSshVolumeBackend(
            self.runner,
            storage_root=self.configuration.remote_storage_root,
        )
        relative_path = self.configuration.watchdog_state_path.relative_to(
            self.configuration.remote_storage_root
        )
        backend.put_bytes(snapshot.to_json_bytes(), relative_path.as_posix())

    def _start_supervisor(self) -> None:
        """Idempotently start both direct managed processes."""
        self.runner.run(
            (
                "python",
                "-m",
                "remote.vast_supervisor",
                "start",
                "--watchdog-state",
                str(self.configuration.watchdog_state_path),
            ),
            timeout_seconds=60.0,
        )


__all__ = [
    "DEFAULT_VAST_REMOTE_COMFYUI_ROOT",
    "DEFAULT_VAST_REMOTE_REPO_ROOT",
    "DEFAULT_VAST_REMOTE_STORAGE_ROOT",
    "DEFAULT_VAST_WATCHDOG_STATE_PATH",
    "VAST_IMAGE_ENV",
    "VastRuntimeConfiguration",
    "VastRuntimeFingerprintDriftError",
    "VastRuntimeManager",
]
