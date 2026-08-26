"""CPU-side Hugging Face snapshot staging for Modal LLM profiles."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid4

if __package__:
    from .llm_profiles import LLMModelProfile, get_llm_profile, load_llm_profiles
else:  # pragma: no cover - the stable cloud entrypoint imports top-level modules.
    from llm_profiles import LLMModelProfile, get_llm_profile, load_llm_profiles

logger = logging.getLogger(__name__)

_COMPLETE_MARKER_FILENAME = ".comfy-modal-llm-complete.json"
_DEFAULT_LEASE_TIMEOUT_SECONDS = 7200.0
_LEASE_POLL_SECONDS = 2.0
_LEASE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_LEASE_HEARTBEAT_STALE_SECONDS = 300.0
_DEFAULT_MINIMUM_FREE_DISK_BYTES = 8 * 1024**3
_SNAPSHOT_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.model",
    "*.tiktoken",
    "added_tokens.json",
    "chat_template*",
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor*",
    "special_tokens_map.json",
    "tokenizer*",
    "video_preprocessor_config.json",
    "vocab*",
)
_TOKENIZER_ALLOW_PATTERNS = tuple(
    pattern for pattern in _SNAPSHOT_ALLOW_PATTERNS if "safetensors" not in pattern
)


@dataclass(frozen=True)
class StagedModelSnapshot:
    """Describe a model snapshot made visible on the shared Modal Volume."""

    profile_id: str
    repository: str
    revision: str
    path: str
    downloaded: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class LLMStagingProgress:
    """Describe one numeric Hugging Face snapshot-staging update."""

    stage: str
    message: str
    value: float | None = None
    maximum: float | None = None
    unit: str | None = None
    indeterminate: bool = False
    model_reference: str | None = None


StagingProgressCallback = Callable[[LLMStagingProgress], None]


@dataclass(frozen=True)
class ResolvedStagedModelProfile:
    """Describe one resolved immutable profile and its staged snapshot."""

    requested_reference: str
    profile_id: str
    repository: str
    revision: str
    backend: str
    quantization_method: str
    artifact_bytes: int
    manifest_path: str | None
    manifest_created: bool
    security_scan_complete: bool
    path: str
    downloaded: bool
    resolve_elapsed_seconds: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible staging metadata."""
        return asdict(self)


def _snapshot_tqdm_class(
    progress_callback: StagingProgressCallback,
) -> type[Any]:
    """Build a tqdm subclass that forwards Hugging Face file progress."""
    from tqdm.auto import tqdm

    class SnapshotProgressTqdm(tqdm):
        """Mirror snapshot_download's aggregate file bar to the caller."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the Hugging Face bar and report its initial state."""
            super().__init__(*args, **kwargs)
            self._report_progress()

        def update(self, n: float | None = 1) -> bool | None:
            """Forward every aggregate file-count update."""
            updated = super().update(n)
            self._report_progress()
            return updated

        def _report_progress(self) -> None:
            """Publish the current tqdm count with compact user-facing copy."""
            maximum = float(self.total) if self.total is not None else None
            value = float(self.n)
            description = str(
                getattr(self, "desc", None) or "Downloading model snapshot"
            ).strip()
            progress_callback(
                LLMStagingProgress(
                    stage="download",
                    message=description,
                    value=value,
                    maximum=maximum,
                    unit=str(getattr(self, "unit", None) or "files"),
                    indeterminate=maximum is None,
                )
            )

    return SnapshotProgressTqdm


def model_snapshot_path(storage_root: str | Path, profile: LLMModelProfile) -> Path:
    """Return the shared path, reusing a valid legacy profile path in place."""
    resolved_root = Path(storage_root).resolve()
    canonical_path = resolved_root / profile.storage_relative_path()
    if canonical_path.exists():
        return canonical_path
    model_root = resolved_root / "llm_models"
    for legacy_path in sorted(model_root.glob(f"*/{profile.revision}")):
        if legacy_path != canonical_path and _legacy_snapshot_matches(
            legacy_path,
            profile,
        ):
            logger.info(
                "Reusing legacy Modal LLM weight path repository=%s revision=%s "
                "path=%s.",
                profile.repository,
                profile.revision,
                legacy_path,
            )
            return legacy_path
    return canonical_path


def _marker_path(snapshot_path: Path) -> Path:
    """Return the completion marker path for one staged snapshot."""
    return snapshot_path / _COMPLETE_MARKER_FILENAME


def _required_model_paths(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> tuple[Path, ...]:
    """Return every required model artifact inside one snapshot."""
    model_filename = profile.backend_option("model_filename")
    if model_filename is None:
        return (snapshot_path / "config.json",)
    filenames = [str(model_filename)]
    mmproj_filename = profile.backend_option("mmproj_filename")
    if mmproj_filename is not None:
        filenames.append(str(mmproj_filename))
    return tuple(snapshot_path / filename for filename in filenames)


def _tokenizer_source(profile: LLMModelProfile) -> tuple[str, str] | None:
    """Return an optional separately pinned tokenizer snapshot source."""
    repository = profile.backend_option("tokenizer_repository")
    revision = profile.backend_option("tokenizer_revision")
    if repository is None and revision is None:
        return None
    normalized_repository = str(repository or "").strip()
    normalized_revision = str(revision or "").strip().lower()
    if "/" not in normalized_repository or len(normalized_revision) != 40:
        raise ValueError(
            f"Modal LLM profile {profile.profile_id!r} has an invalid separately "
            "pinned tokenizer source."
        )
    return normalized_repository, normalized_revision


def _snapshot_has_required_artifacts(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> bool:
    """Return whether model and optional tokenizer artifacts are complete."""
    if not all(
        required_path.is_file()
        for required_path in _required_model_paths(snapshot_path, profile)
    ):
        return False
    return (
        _tokenizer_source(profile) is None
        or (snapshot_path / "tokenizer_config.json").is_file()
    )


def _model_allow_patterns(profile: LLMModelProfile) -> tuple[str, ...]:
    """Return a bounded allowlist for this profile's primary repository."""
    model_filename = profile.backend_option("model_filename")
    if model_filename is None:
        return _SNAPSHOT_ALLOW_PATTERNS
    selected_filenames = [str(model_filename)]
    mmproj_filename = profile.backend_option("mmproj_filename")
    if mmproj_filename is not None:
        selected_filenames.append(str(mmproj_filename))
    return (*_TOKENIZER_ALLOW_PATTERNS, *selected_filenames)


def _read_marker(snapshot_path: Path) -> dict[str, Any] | None:
    """Return a valid completion marker, if present."""
    marker_path = _marker_path(snapshot_path)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def is_model_snapshot_staged(
    storage_root: str | Path, profile: LLMModelProfile
) -> bool:
    """Return whether the requested immutable snapshot is completely staged."""
    snapshot_path = model_snapshot_path(storage_root, profile)
    marker = _read_marker(snapshot_path)
    return bool(
        marker
        and marker.get("repository") == profile.repository
        and marker.get("revision") == profile.revision
        and _snapshot_has_required_artifacts(snapshot_path, profile)
    )


def _write_marker(snapshot_path: Path, profile: LLMModelProfile) -> None:
    """Atomically record that an immutable model snapshot is complete."""
    marker_path = _marker_path(snapshot_path)
    temporary_path = marker_path.with_suffix(".tmp")
    marker_payload = {
        "marker_version": 2,
        "profile_id": profile.profile_id,
        "repository": profile.repository,
        "revision": profile.revision,
        "completed_at_unix": time.time(),
    }
    temporary_path.write_text(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_path, marker_path)


def _legacy_snapshot_matches(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> bool:
    """Return whether an old profile-keyed directory has these exact weights."""
    marker = _read_marker(snapshot_path)
    return bool(
        marker
        and marker.get("repository") == profile.repository
        and marker.get("revision") == profile.revision
        and _snapshot_has_required_artifacts(snapshot_path, profile)
    )


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
    progress_callback: StagingProgressCallback | None,
    model_label: str,
) -> None:
    """Publish the first wait state for a contended snapshot."""
    if progress_callback is None:
        return
    progress_callback(
        LLMStagingProgress(
            stage="waiting_for_download",
            message=f"Waiting for another download of {model_label} to finish",
            indeterminate=True,
        )
    )


def _wait_for_existing_snapshot_lease(
    lease_path: Path,
    *,
    started_at: float,
    timeout_seconds: float,
    progress_callback: StagingProgressCallback | None,
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
    progress_callback: StagingProgressCallback | None,
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
    progress_callback: StagingProgressCallback | None = None,
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


def _snapshot_existing_bytes(snapshot_path: Path) -> int:
    """Return bytes already present for a resumable model snapshot."""
    total = 0
    if not snapshot_path.exists():
        return total
    for candidate in snapshot_path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.name.endswith(".lock"):
                total += candidate.stat().st_size
        except (FileNotFoundError, OSError):
            continue
    return total


def _minimum_free_disk_bytes() -> int:
    """Return the configured free-space reserve retained after staging."""
    raw_value = os.getenv("COMFY_MODAL_LLM_MIN_FREE_DISK_GB")
    if raw_value is None:
        return _DEFAULT_MINIMUM_FREE_DISK_BYTES
    value_gb = float(raw_value)
    if not math.isfinite(value_gb) or value_gb < 0:
        raise ValueError("COMFY_MODAL_LLM_MIN_FREE_DISK_GB must not be negative.")
    return int(value_gb * 1024**3)


def _preflight_snapshot_capacity(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> tuple[int, int, int]:
    """Require room for missing artifacts plus a post-download safety reserve."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(snapshot_path.parent).free
    existing_bytes = _snapshot_existing_bytes(snapshot_path)
    remaining_bytes = max(0, profile.artifact_bytes - existing_bytes)
    required_bytes = remaining_bytes + _minimum_free_disk_bytes()
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient disk space to stage LLM profile "
            f"{profile.profile_id!r}: {free_bytes / 1024**3:.2f} GiB free, "
            f"{remaining_bytes / 1024**3:.2f} GiB still required, and "
            f"{_minimum_free_disk_bytes() / 1024**3:.2f} GiB must remain free."
        )
    return free_bytes, remaining_bytes, required_bytes


def stage_model_profile(
    profile_id: str,
    storage_root: str | Path,
    *,
    profile: LLMModelProfile | None = None,
    snapshot_download: Callable[..., str] | None = None,
    progress_callback: StagingProgressCallback | None = None,
    model_reference: str | None = None,
    owner_id: str | None = None,
) -> StagedModelSnapshot:
    """Download one pinned curated or generated snapshot on a CPU worker."""
    profile = profile or get_llm_profile(profile_id, storage_root=storage_root)
    if profile.profile_id != profile_id:
        raise ValueError(
            f"Stage request id {profile_id!r} does not match profile "
            f"{profile.profile_id!r}."
        )
    model_label = model_reference or profile.profile_id

    def report(progress: LLMStagingProgress) -> None:
        """Attach the user-facing model reference to one staging event."""
        if progress_callback is not None:
            progress_callback(replace(progress, model_reference=model_label))

    snapshot_path = model_snapshot_path(storage_root, profile)
    started_at = time.perf_counter()
    report(
        LLMStagingProgress(
            stage="snapshot_check",
            message=f"Checking staged snapshot for {model_label}",
            indeterminate=True,
        )
    )
    if is_model_snapshot_staged(storage_root, profile):
        logger.info(
            "Reusing staged Modal LLM profile=%s revision=%s path=%s.",
            profile.profile_id,
            profile.revision,
            snapshot_path,
        )
        report(
            LLMStagingProgress(
                stage="cached",
                message=f"Using staged snapshot for {model_label}",
                value=1,
                maximum=1,
                unit="model",
            )
        )
        return StagedModelSnapshot(
            profile_id=profile.profile_id,
            repository=profile.repository,
            revision=profile.revision,
            path=str(snapshot_path),
            downloaded=False,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    with _snapshot_lease(
        snapshot_path,
        progress_callback=report,
        model_label=model_label,
        owner_id=owner_id or f"pid-{os.getpid()}",
    ):
        if is_model_snapshot_staged(storage_root, profile):
            report(
                LLMStagingProgress(
                    stage="cached",
                    message=(
                        f"Using snapshot staged by another worker for {model_label}"
                    ),
                    value=1,
                    maximum=1,
                    unit="model",
                )
            )
            return StagedModelSnapshot(
                profile_id=profile.profile_id,
                repository=profile.repository,
                revision=profile.revision,
                path=str(snapshot_path),
                downloaded=False,
                elapsed_seconds=time.perf_counter() - started_at,
            )
        free_bytes, remaining_bytes, _required_bytes = _preflight_snapshot_capacity(
            snapshot_path,
            profile,
        )
        report(
            LLMStagingProgress(
                stage="disk_check",
                message=(
                    f"Storage ready for {model_label}: "
                    f"{free_bytes / 1024**3:.1f} GiB free, "
                    f"{remaining_bytes / 1024**3:.1f} GiB remaining"
                ),
                value=free_bytes,
                maximum=free_bytes,
                unit="bytes",
            )
        )
        if snapshot_download is None:
            from huggingface_hub import (
                snapshot_download as huggingface_snapshot_download,
            )

            snapshot_download = huggingface_snapshot_download
        report(
            LLMStagingProgress(
                stage="download_preparing",
                message=f"Preparing download for {model_label}",
                indeterminate=True,
            )
        )
        snapshot_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Staging Modal LLM profile=%s repository=%s revision=%s to %s.",
            profile.profile_id,
            profile.repository,
            profile.revision,
            snapshot_path,
        )
        download_options: dict[str, Any] = {
            "repo_id": profile.repository,
            "revision": profile.revision,
            "local_dir": str(snapshot_path),
            "token": os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            "allow_patterns": _model_allow_patterns(profile),
        }
        if progress_callback is not None:
            download_options["tqdm_class"] = _snapshot_tqdm_class(report)
        resolved_path = Path(snapshot_download(**download_options)).resolve()
        if resolved_path != snapshot_path.resolve():
            raise RuntimeError(
                f"Hugging Face staged {profile.profile_id!r} at unexpected path "
                f"{resolved_path}; expected {snapshot_path.resolve()}."
            )
        tokenizer_source = _tokenizer_source(profile)
        if tokenizer_source is not None:
            tokenizer_repository, tokenizer_revision = tokenizer_source
            tokenizer_options: dict[str, Any] = {
                "repo_id": tokenizer_repository,
                "revision": tokenizer_revision,
                "local_dir": str(snapshot_path),
                "token": download_options["token"],
                "allow_patterns": _TOKENIZER_ALLOW_PATTERNS,
            }
            if progress_callback is not None:
                tokenizer_options["tqdm_class"] = _snapshot_tqdm_class(
                    report
                )
            tokenizer_path = Path(snapshot_download(**tokenizer_options)).resolve()
            if tokenizer_path != snapshot_path.resolve():
                raise RuntimeError(
                    f"Hugging Face staged tokenizer for {profile.profile_id!r} at "
                    f"unexpected path {tokenizer_path}; expected "
                    f"{snapshot_path.resolve()}."
                )
        if not _snapshot_has_required_artifacts(snapshot_path, profile):
            raise RuntimeError(
                f"Staged Modal LLM profile {profile.profile_id!r} is missing its "
                "required model or tokenizer artifacts."
            )
        _write_marker(snapshot_path, profile)
        report(
            LLMStagingProgress(
                stage="staged",
                message=f"Model snapshot ready for {model_label}",
                value=1,
                maximum=1,
                unit="model",
            )
        )
    result = StagedModelSnapshot(
        profile_id=profile.profile_id,
        repository=profile.repository,
        revision=profile.revision,
        path=str(snapshot_path),
        downloaded=True,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    logger.info("Completed Modal LLM staging: %s", asdict(result))
    return result


def _persist_supplied_profile(
    model_reference: str,
    storage_root: str | Path,
    supplied: Mapping[str, Any],
) -> tuple[LLMModelProfile, str | None, bool, bool]:
    """Validate planner metadata and persist a generated manifest remotely."""
    raw_profile = supplied.get("profile", supplied)
    if not isinstance(raw_profile, Mapping):
        raise ValueError(
            f"Resolved LLM profile for {model_reference!r} must contain an object."
        )
    profile = LLMModelProfile.from_mapping(raw_profile)
    _validate_supplied_profile_reference(model_reference, profile)
    if profile.source != "generated":
        return profile, None, False, True
    if __package__:
        from .llm_profiles import generated_profile_manifest_path
    else:  # pragma: no cover - stable cloud entrypoint imports top-level modules.
        from llm_profiles import generated_profile_manifest_path

    manifest_path = generated_profile_manifest_path(storage_root, profile.profile_id)
    manifest_payload = _supplied_profile_manifest_payload(
        model_reference,
        profile,
        supplied,
    )
    existing_scan = _existing_supplied_profile_scan(
        manifest_path,
        raw_profile,
    )
    if existing_scan is not None:
        return profile, str(manifest_path), False, existing_scan
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(f".{uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return profile, str(manifest_path), True, bool(
        manifest_payload["security_scan_complete"]
    )


def _supplied_profile_manifest_payload(
    model_reference: str,
    profile: LLMModelProfile,
    supplied: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the generated manifest persisted from trusted planner metadata."""
    security_scan_complete = supplied.get("security_scan_complete", False)
    if not isinstance(security_scan_complete, bool):
        raise ValueError(
            f"Resolved LLM profile for {model_reference!r} has an invalid "
            "security scan state."
        )
    return {
        "schema_version": profile.schema_version,
        "compatibility_policy_version": profile.compatibility_policy_version,
        "requested_reference": model_reference,
        "resolved_at_unix": time.time(),
        "security_scan_complete": security_scan_complete,
        "profile": profile.to_mapping(),
    }


def _existing_supplied_profile_scan(
    manifest_path: Path,
    raw_profile: Mapping[str, Any],
) -> bool | None:
    """Return an existing matching manifest's scan state, or None if absent."""
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"Resolved LLM manifest {manifest_path} is unreadable: {exc}"
            ) from exc
        if not isinstance(existing, Mapping) or existing.get("profile") != raw_profile:
            raise RuntimeError(
                f"Resolved LLM profile collision at {manifest_path}."
            )
        return bool(existing.get("security_scan_complete", False))
    return None


def _validate_supplied_profile_reference(
    model_reference: str,
    profile: LLMModelProfile,
) -> None:
    """Reject planner metadata that does not describe its requested reference."""
    curated_profile = load_llm_profiles().get(model_reference)
    if curated_profile is not None:
        if profile != curated_profile:
            raise ValueError(
                f"Resolved metadata for curated profile {model_reference!r} does "
                "not match the checked-in profile."
            )
        return
    if model_reference.startswith("hf-"):
        if profile.profile_id != model_reference:
            raise ValueError(
                f"Resolved metadata profile id {profile.profile_id!r} does not "
                f"match requested id {model_reference!r}."
            )
        return
    if profile.source != "generated":
        raise ValueError(
            f"Resolved metadata for Hugging Face reference {model_reference!r} "
            "must contain a generated profile."
        )
    if __package__:
        from .llm_resolver import HuggingFaceModelReference
    else:  # pragma: no cover - stable cloud entrypoint imports top-level modules.
        from llm_resolver import HuggingFaceModelReference

    parsed = HuggingFaceModelReference.parse(model_reference)
    if profile.repository != parsed.repository:
        raise ValueError(
            f"Resolved metadata repository {profile.repository!r} does not match "
            f"requested repository {parsed.repository!r}."
        )
    requested_revision = (parsed.requested_revision or "").lower()
    if len(requested_revision) == 40 and requested_revision != profile.revision:
        raise ValueError(
            f"Resolved metadata revision {profile.revision!r} does not match "
            f"requested revision {requested_revision!r}."
        )


def _resolve_profile_for_staging(
    model_reference: str,
    storage_root: str | Path,
    supplied_profile: Mapping[str, Any] | None = None,
) -> tuple[LLMModelProfile, str | None, bool, bool]:
    """Resolve one curated, generated, or Hugging Face model reference."""
    if supplied_profile is not None:
        return _persist_supplied_profile(
            model_reference,
            storage_root,
            supplied_profile,
        )
    curated_profiles = load_llm_profiles()
    if model_reference in curated_profiles:
        return curated_profiles[model_reference], None, False, True
    if model_reference.startswith("hf-"):
        return (
            get_llm_profile(model_reference, storage_root=storage_root),
            None,
            False,
            True,
        )
    if __package__:
        from .llm_resolver import resolve_model_profile
    else:  # pragma: no cover - stable cloud entrypoint imports top-level modules.
        from llm_resolver import resolve_model_profile

    resolved = resolve_model_profile(model_reference, storage_root)
    return (
        resolved.profile,
        resolved.manifest_path,
        resolved.manifest_created,
        resolved.security_scan_complete,
    )


def resolve_and_stage_model_references(
    model_references: list[str],
    storage_root: str | Path,
    *,
    progress_callback: StagingProgressCallback | None = None,
    resolved_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    owner_id: str | None = None,
) -> list[ResolvedStagedModelProfile]:
    """Resolve and stage model references on any CPU-backed remote worker."""
    results: list[ResolvedStagedModelProfile] = []
    for model_reference in model_references:
        supplied_profile = (resolved_profiles or {}).get(model_reference)
        if progress_callback is not None:
            is_hugging_face_reference = (
                "/" in model_reference and not model_reference.startswith("hf-")
            )
            progress_callback(
                LLMStagingProgress(
                    stage=(
                        "resolved_metadata"
                        if supplied_profile is not None
                        else "metadata"
                        if is_hugging_face_reference
                        else "profile_resolution"
                    ),
                    message=(
                        f"Using planner-resolved metadata for {model_reference}"
                        if supplied_profile is not None
                        else f"Inspecting Hugging Face metadata for {model_reference}"
                        if is_hugging_face_reference
                        else f"Resolving model profile {model_reference}"
                    ),
                    indeterminate=True,
                    model_reference=model_reference,
                )
            )
        resolve_started_at = time.perf_counter()
        (
            profile,
            manifest_path,
            manifest_created,
            scan_complete,
        ) = (
            _resolve_profile_for_staging(
                model_reference,
                storage_root,
                supplied_profile,
            )
            if supplied_profile is not None
            else _resolve_profile_for_staging(model_reference, storage_root)
        )
        resolve_elapsed_seconds = time.perf_counter() - resolve_started_at
        staged = stage_model_profile(
            profile.profile_id,
            storage_root,
            profile=profile,
            progress_callback=progress_callback,
            model_reference=model_reference,
            owner_id=owner_id,
        )
        results.append(
            ResolvedStagedModelProfile(
                requested_reference=model_reference,
                profile_id=staged.profile_id,
                repository=staged.repository,
                revision=staged.revision,
                backend=profile.backend,
                quantization_method=profile.quantization_method,
                artifact_bytes=profile.artifact_bytes,
                manifest_path=manifest_path,
                manifest_created=manifest_created,
                security_scan_complete=scan_complete,
                path=staged.path,
                downloaded=staged.downloaded,
                resolve_elapsed_seconds=resolve_elapsed_seconds,
                elapsed_seconds=staged.elapsed_seconds,
            )
        )
    return results


__all__ = [
    "LLMStagingProgress",
    "ResolvedStagedModelProfile",
    "StagedModelSnapshot",
    "is_model_snapshot_staged",
    "model_snapshot_path",
    "resolve_and_stage_model_references",
    "stage_model_profile",
]
