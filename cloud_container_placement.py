"""Record where a Modal container runs and whether it booted or restored a snapshot.

Memory-snapshot misses are invisible without this: a restored container looks like
any other warm worker. Placement facts captured during the snapshot phase survive
inside the snapshot, so a restored container can report both where it runs now and
where its snapshot was taken.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Literal

logger = logging.getLogger(__name__)

StartupKind = Literal["boot", "snapshot-created", "restore"]

_DEFAULT_RESTORE_STATE_PATH = Path("/__modal/restore-state.json")
_NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=name,uuid,driver_version",
    "--format=csv,noheader",
)
_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class GpuIdentity:
    """Model, physical identity, and driver of the container's first GPU."""

    name: str = _UNKNOWN
    uuid: str = _UNKNOWN
    driver_version: str = _UNKNOWN


@dataclass(frozen=True)
class ContainerPlacement:
    """Where Modal scheduled one container and which bound function it serves."""

    task_id: str
    function_id: str
    region: str
    cloud_provider: str
    hostname: str
    gpu: GpuIdentity


@dataclass(frozen=True)
class SnapshotOrigin:
    """Placement and identity captured while Modal prepared a memory snapshot."""

    placement: ContainerPlacement
    captured_at: float
    worker_affinity_key: str
    snapshot_profile_key: str


@dataclass(frozen=True)
class ContainerStartup:
    """One container start: how it started, where it runs, and its snapshot origin."""

    kind: StartupKind
    placement: ContainerPlacement
    origin: SnapshotOrigin | None
    worker_affinity_key: str
    snapshot_profile_key: str
    gpu_snapshot_enabled: bool

    @property
    def snapshot_age_seconds(self) -> float | None:
        """Return seconds since the snapshot was captured, when known."""
        if self.origin is None:
            return None
        return max(0.0, time.time() - self.origin.captured_at)


_SNAPSHOT_ORIGIN: SnapshotOrigin | None = None


def _env(name: str) -> str:
    """Return a stripped environment value or the unknown marker."""
    return (os.environ.get(name) or "").strip() or _UNKNOWN


def _parse_nvidia_smi_line(output: str) -> GpuIdentity:
    """Parse the first CSV row of an ``nvidia-smi`` GPU query."""
    first_line = next((line for line in output.splitlines() if line.strip()), "")
    fields = [field.strip() for field in first_line.split(",")]
    if len(fields) < 3:
        return GpuIdentity()
    return GpuIdentity(name=fields[0], uuid=fields[1], driver_version=fields[2])


def current_gpu_identity() -> GpuIdentity:
    """Return the first GPU's model, UUID, and driver, or unknowns when unavailable."""
    try:
        completed = subprocess.run(
            _NVIDIA_SMI_QUERY,
            capture_output=True,
            check=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not query GPU identity with nvidia-smi: %s", exc)
        return GpuIdentity()
    return _parse_nvidia_smi_line(completed.stdout)


def _modal_io_manager_ids() -> tuple[str, str]:
    """Return Modal's (task_id, bound function_id) for this container when available."""
    try:
        from modal._runtime.container_io_manager import _ContainerIOManager
    except ImportError:
        return _env("MODAL_TASK_ID"), _UNKNOWN
    io_manager = getattr(_ContainerIOManager, "_singleton", None)
    task_id = str(getattr(io_manager, "task_id", "") or "") or _env("MODAL_TASK_ID")
    function_id = str(getattr(io_manager, "function_id", "") or "") or _UNKNOWN
    return task_id, function_id


def current_container_placement(gpu: GpuIdentity | None = None) -> ContainerPlacement:
    """Return where this container runs according to Modal and the host."""
    task_id, function_id = _modal_io_manager_ids()
    return ContainerPlacement(
        task_id=task_id,
        function_id=function_id,
        region=_env("MODAL_REGION"),
        cloud_provider=_env("MODAL_CLOUD_PROVIDER"),
        hostname=socket.gethostname() or _UNKNOWN,
        gpu=gpu if gpu is not None else current_gpu_identity(),
    )


def _restore_state_path() -> Path:
    """Return the file Modal writes when it restores a container from a snapshot."""
    try:
        from modal.config import config as modal_config
    except ImportError:
        return _DEFAULT_RESTORE_STATE_PATH
    configured_path = modal_config.get("restore_state_path")
    return Path(configured_path) if configured_path else _DEFAULT_RESTORE_STATE_PATH


def container_startup_kind(
    restore_state_path: Path | None = None,
    *,
    task_id: str | None = None,
    origin: SnapshotOrigin | None = None,
) -> StartupKind:
    """Classify how Modal started this container.

    Modal writes the restore-state file for every container that resumes from a
    snapshot, including the booting container that just created the snapshot and
    is resumed in place. A snapshot captured by this same task is therefore a boot
    that created a snapshot, not a restore of someone else's.
    """
    path = restore_state_path if restore_state_path is not None else _restore_state_path()
    if not path.exists():
        return "boot"
    if origin is not None and task_id is not None and origin.placement.task_id == task_id:
        return "snapshot-created"
    return "restore"


def record_snapshot_origin(
    *,
    worker_affinity_key: str,
    snapshot_profile_key: str,
    placement: ContainerPlacement | None = None,
) -> SnapshotOrigin:
    """Remember this container's placement so restores of its snapshot can report it."""
    global _SNAPSHOT_ORIGIN
    _SNAPSHOT_ORIGIN = SnapshotOrigin(
        placement=placement if placement is not None else current_container_placement(),
        captured_at=time.time(),
        worker_affinity_key=worker_affinity_key,
        snapshot_profile_key=snapshot_profile_key,
    )
    logger.info(
        "Captured Modal snapshot origin pool=%s function=%s task=%s region=%s cloud=%s host=%s gpu=%s gpu_uuid=%s driver=%s.",
        worker_affinity_key,
        _SNAPSHOT_ORIGIN.placement.function_id,
        _SNAPSHOT_ORIGIN.placement.task_id,
        _SNAPSHOT_ORIGIN.placement.region,
        _SNAPSHOT_ORIGIN.placement.cloud_provider,
        _SNAPSHOT_ORIGIN.placement.hostname,
        _SNAPSHOT_ORIGIN.placement.gpu.name,
        _SNAPSHOT_ORIGIN.placement.gpu.uuid,
        _SNAPSHOT_ORIGIN.placement.gpu.driver_version,
    )
    return _SNAPSHOT_ORIGIN


def describe_container_startup(
    *,
    worker_affinity_key: str,
    snapshot_profile_key: str,
    gpu_snapshot_enabled: bool,
    restore_state_path: Path | None = None,
    placement: ContainerPlacement | None = None,
) -> ContainerStartup:
    """Return this container's startup kind, placement, and snapshot origin."""
    current_placement = (
        placement if placement is not None else current_container_placement()
    )
    kind = container_startup_kind(
        restore_state_path,
        task_id=current_placement.task_id,
        origin=_SNAPSHOT_ORIGIN,
    )
    origin = _SNAPSHOT_ORIGIN if kind == "restore" else None
    return ContainerStartup(
        kind=kind,
        placement=current_placement,
        origin=origin,
        worker_affinity_key=worker_affinity_key,
        snapshot_profile_key=snapshot_profile_key,
        gpu_snapshot_enabled=gpu_snapshot_enabled,
    )


def _origin_fields(startup: ContainerStartup) -> str:
    """Format snapshot-origin fields for a restored container, or ``origin=none``."""
    origin = startup.origin
    if origin is None:
        return "origin=none"
    same_host = origin.placement.hostname == startup.placement.hostname
    same_gpu = origin.placement.gpu.uuid == startup.placement.gpu.uuid
    return (
        f"origin_task={origin.placement.task_id} origin_function={origin.placement.function_id} "
        f"origin_region={origin.placement.region} origin_host={origin.placement.hostname} "
        f"origin_gpu_uuid={origin.placement.gpu.uuid} origin_pool={origin.worker_affinity_key} "
        f"same_host={same_host} same_gpu={same_gpu} "
        f"snapshot_age_seconds={startup.snapshot_age_seconds:.0f}"
    )


def log_container_startup(startup: ContainerStartup) -> ContainerStartup:
    """Emit one greppable line describing how and where this container started."""
    placement = startup.placement
    logger.info(
        "Modal container startup kind=%s pool=%s function=%s task=%s region=%s cloud=%s host=%s gpu=%s gpu_uuid=%s driver=%s gpu_snapshot=%s profile=%s %s.",
        startup.kind,
        startup.worker_affinity_key,
        placement.function_id,
        placement.task_id,
        placement.region,
        placement.cloud_provider,
        placement.hostname,
        placement.gpu.name,
        placement.gpu.uuid,
        placement.gpu.driver_version,
        startup.gpu_snapshot_enabled,
        startup.snapshot_profile_key or None,
        _origin_fields(startup),
    )
    return startup


def reset_snapshot_origin_for_tests(origin: SnapshotOrigin | None = None) -> None:
    """Replace the recorded snapshot origin; intended for tests only."""
    global _SNAPSHOT_ORIGIN
    _SNAPSHOT_ORIGIN = replace(origin) if origin is not None else None
