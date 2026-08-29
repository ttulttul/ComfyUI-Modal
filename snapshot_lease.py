"""Cross-process, heartbeating file lease with owner-liveness validation."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Callable, Iterator, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)

_DEFAULT_LEASE_TIMEOUT_SECONDS = 7200.0
_LEASE_POLL_SECONDS = 2.0
_LEASE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_LEASE_HEARTBEAT_STALE_SECONDS = 300.0
LeaseWaitCallback = Callable[[str], None]

def _process_start_identity(pid: int) -> str | None:
    """Return the Linux process start tick used to reject PID reuse."""
    try:
        stat_record = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    fields_after_command = stat_record.rpartition(") ")[2].split()
    return fields_after_command[19] if len(fields_after_command) > 19 else None


def _lease_owner_payload(owner_id: str) -> dict[str, Any]:
    """Return an owner record that can be validated on this worker."""
    pid = os.getpid()
    return {
        "version": 1,
        "owner_id": owner_id,
        "host_id": socket.gethostname(),
        "pid": pid,
        "process_start": _process_start_identity(pid),
        "token": uuid4().hex,
        "acquired_at": time.time(),
    }


def _read_lease_owner(lease_path: Path) -> dict[str, Any] | None:
    """Read a snapshot lease owner, tolerating legacy text records."""
    try:
        raw_value = lease_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _local_lease_owner_is_alive(owner: Mapping[str, Any]) -> bool | None:
    """Return local owner liveness, or None for another worker host."""
    if str(owner.get("host_id") or "") != socket.gethostname():
        return None
    try:
        pid = int(owner.get("pid"))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    expected_start = str(owner.get("process_start") or "")
    actual_start = _process_start_identity(pid)
    if actual_start is None:
        if expected_start:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return not expected_start or actual_start == expected_start


def _lease_record_matches(lease_path: Path, owner: Mapping[str, Any]) -> bool:
    """Return whether a path still contains the observed owner token."""
    current = _read_lease_owner(lease_path)
    return bool(
        current
        and current.get("token")
        and current.get("token") == owner.get("token")
    )


def _remove_owned_lease(lease_path: Path, owner: Mapping[str, Any]) -> None:
    """Remove a lock only while its ownership token still matches."""
    if not _lease_record_matches(lease_path, owner):
        return
    try:
        lease_path.unlink()
    except FileNotFoundError:
        pass


def _heartbeat_snapshot_lease(
    lease_path: Path,
    owner: Mapping[str, Any],
    stop_event: threading.Event,
) -> None:
    """Refresh an active lease so another container never steals live work."""
    while not stop_event.wait(_LEASE_HEARTBEAT_SECONDS):
        if not _lease_record_matches(lease_path, owner):
            return
        try:
            lease_path.touch()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning(
                "Unable to heartbeat LLM staging lease %s: %s",
                lease_path,
                exc,
            )
            return


def _snapshot_lease_wait_timeout_seconds() -> float:
    """Return the configured maximum wait to acquire one snapshot lease."""
    timeout_seconds = float(
        os.getenv("COMFY_MODAL_LLM_STAGE_LEASE_TIMEOUT_SECONDS", "7200")
    )
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "COMFY_MODAL_LLM_STAGE_LEASE_TIMEOUT_SECONDS must be positive."
        )
    return timeout_seconds


def _snapshot_lease_stale_seconds(
    existing_owner: Mapping[str, Any] | None,
) -> float:
    """Return the age that proves a heartbeating or legacy lease abandoned."""
    if existing_owner is None or not existing_owner.get("token"):
        return _DEFAULT_LEASE_TIMEOUT_SECONDS
    raw_value = os.getenv(
        "COMFY_MODAL_LLM_STAGE_LEASE_HEARTBEAT_STALE_SECONDS",
        str(_DEFAULT_LEASE_HEARTBEAT_STALE_SECONDS),
    )
    stale_seconds = float(raw_value)
    if (
        not math.isfinite(stale_seconds)
        or stale_seconds <= _LEASE_HEARTBEAT_SECONDS * 2
    ):
        raise ValueError(
            "COMFY_MODAL_LLM_STAGE_LEASE_HEARTBEAT_STALE_SECONDS must exceed "
            f"{_LEASE_HEARTBEAT_SECONDS * 2:.0f} seconds."
        )
    return stale_seconds


def _remove_expired_snapshot_lease(
    lease_path: Path,
    existing_owner: Mapping[str, Any] | None,
) -> None:
    """Remove one expired lease without deleting a replacement owner."""
    logger.warning(
        "Removing expired LLM staging lease %s owner=%s.",
        lease_path,
        (existing_owner or {}).get("owner_id"),
    )
    if existing_owner is not None:
        _remove_owned_lease(lease_path, existing_owner)
        return
    try:
        lease_path.unlink()
    except FileNotFoundError:
        pass


def _report_snapshot_lease_wait(
    progress_callback: LeaseWaitCallback | None,
    model_label: str,
) -> None:
    """Publish the first wait state for a contended snapshot."""
    if progress_callback is None:
        return
    progress_callback(
        f"Waiting for another download of {model_label} to finish"
    )


def _wait_for_existing_snapshot_lease(
    lease_path: Path,
    *,
    started_at: float,
    timeout_seconds: float,
    progress_callback: LeaseWaitCallback | None,
    model_label: str,
    report_wait: bool,
) -> bool:
    """Wait once or reclaim an abandoned existing snapshot lease."""
    existing_owner = _read_lease_owner(lease_path)
    if (
        existing_owner is not None
        and _local_lease_owner_is_alive(existing_owner) is False
    ):
        logger.warning(
            "Removing abandoned LLM staging lease %s owner=%s.",
            lease_path,
            existing_owner.get("owner_id"),
        )
        _remove_owned_lease(lease_path, existing_owner)
        return False
    try:
        lease_age = time.time() - lease_path.stat().st_mtime
    except FileNotFoundError:
        return False
    if lease_age >= _snapshot_lease_stale_seconds(existing_owner):
        _remove_expired_snapshot_lease(lease_path, existing_owner)
        return False
    if time.monotonic() - started_at >= timeout_seconds:
        raise TimeoutError(
            f"Timed out waiting {timeout_seconds:.0f}s for model staging lease "
            f"{lease_path}."
        )
    if report_wait:
        _report_snapshot_lease_wait(progress_callback, model_label)
    time.sleep(_LEASE_POLL_SECONDS)
    return True


def _acquire_snapshot_lease(
    lease_path: Path,
    owner: Mapping[str, Any],
    *,
    progress_callback: LeaseWaitCallback | None,
    model_label: str,
) -> None:
    """Acquire one exclusive snapshot lease, waiting or reclaiming as needed."""
    timeout_seconds = _snapshot_lease_wait_timeout_seconds()
    started_at = time.monotonic()
    waiting_reported = False
    while True:
        try:
            descriptor = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            waited = _wait_for_existing_snapshot_lease(
                lease_path,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                progress_callback=progress_callback,
                model_label=model_label,
                report_wait=not waiting_reported,
            )
            waiting_reported = waiting_reported or waited
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as lease_file:
            json.dump(owner, lease_file, sort_keys=True, separators=(",", ":"))
        return


@contextmanager
def _snapshot_lease(
    snapshot_path: Path,
    *,
    progress_callback: LeaseWaitCallback | None = None,
    model_label: str,
    owner_id: str,
) -> Iterator[None]:
    """Serialize downloads with liveness-checked, heartbeating ownership."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path = snapshot_path.parent / f".{snapshot_path.name}.download.lock"
    owner = _lease_owner_payload(owner_id)
    _acquire_snapshot_lease(
        lease_path,
        owner,
        progress_callback=progress_callback,
        model_label=model_label,
    )
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_snapshot_lease,
        args=(lease_path, owner, heartbeat_stop),
        name="llm-snapshot-lease-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        _remove_owned_lease(lease_path, owner)


