"""Asset synchronization helpers for Modal-backed execution."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .settings import ModalSyncSettings, get_settings

logger = logging.getLogger(__name__)
SyncStatusCallback = Callable[[str, int | None, int | None], None]

_SYNC_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".vae"}
_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "venv",
}
_SKIP_FILE_SUFFIXES = {
    ".log",
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".swp",
    ".tmp",
}

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when Modal SDK is unavailable.
    modal = None


def _emit_sync_status(
    status_callback: SyncStatusCallback | None,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Emit one human-readable sync status update when a callback is available."""
    if status_callback is None:
        return
    status_callback(message, current, total)


def _format_asset_upload_status(
    asset_name: str,
    *,
    item_index: int | None,
    total_items: int | None,
) -> str:
    """Return the global-status message for one asset upload."""
    if item_index is not None and total_items is not None and total_items > 1:
        return f"Uploading asset {item_index}/{total_items} to Modal: {asset_name}"
    return f"Uploading asset to Modal: {asset_name}"


class VolumeBackend(Protocol):
    """Minimal storage interface needed by the sync engine."""

    def exists(self, remote_path: str) -> bool:
        """Return whether the remote path already exists."""

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the remote storage backend."""

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload raw bytes into the remote storage backend."""


class SyncIndexBackend(Protocol):
    """Minimal metadata index interface used to deduplicate deterministic uploads."""

    def get(self, key: str) -> dict[str, Any] | None:
        """Return one stored sync record when it exists."""

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Persist one sync record under the provided key."""


@dataclass(frozen=True)
class SyncedAsset:
    """Description of a local asset mirrored into the remote storage root."""

    local_path: Path
    remote_path: str
    sha256: str
    uploaded: bool


@dataclass(frozen=True)
class _CustomNodesArchiveSpec:
    """Deterministic archive spec for one top-level custom_nodes payload slice."""

    entry_name: str
    display_name: str
    source_description: str
    files: tuple[Path, ...]
    sha256: str


@dataclass(frozen=True)
class _CustomNodesArchiveSyncResult:
    """Describe the sync result for one top-level custom_nodes archive."""

    entry_name: str
    display_name: str
    sha256: str
    remote_path: str
    uploaded: bool


@dataclass(frozen=True)
class _ContentAddressedSyncResult:
    """Describe the outcome of one content-addressed file sync decision."""

    remote_path: str
    uploaded: bool


def _modal_volume_worker_count() -> int:
    """Return the worker count used for local Modal volume SDK calls."""
    return 4


def _custom_nodes_sync_worker_count() -> int:
    """Return the worker count used to package and upload per-package custom_nodes archives."""
    return max(4, min(16, os.cpu_count() or 1))


_MODAL_VOLUME_EXECUTOR = ThreadPoolExecutor(max_workers=_modal_volume_worker_count())


class _ModalSdkCaller:
    """Shared retry and backoff helper for Modal SDK calls."""

    def __init__(self, *, target_kind: str) -> None:
        """Initialize shared retry bookkeeping for one Modal SDK target."""
        self._target_kind = target_kind
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_until_monotonic = 0.0
        self._rate_limit_backoff_seconds = 0.0

    def _resource_exhausted_error_types(self) -> tuple[type[BaseException], ...]:
        """Return the Modal SDK exception types that indicate transient rate limiting."""
        if modal is None:
            return ()
        exception_namespace = getattr(modal, "exception", None)
        if exception_namespace is None:
            return ()
        error_type = getattr(exception_namespace, "ResourceExhaustedError", None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            return (error_type,)
        return ()

    def _wait_for_shared_rate_limit_backoff(self) -> None:
        """Pause until the shared Modal backoff window expires."""
        while True:
            with self._rate_limit_lock:
                remaining_seconds = self._rate_limit_until_monotonic - time.monotonic()
            if remaining_seconds <= 0.0:
                return
            time.sleep(remaining_seconds)

    def _record_shared_rate_limit_backoff(self) -> float:
        """Increase and publish the shared Modal backoff window."""
        with self._rate_limit_lock:
            next_backoff_seconds = (
                0.25
                if self._rate_limit_backoff_seconds <= 0.0
                else min(self._rate_limit_backoff_seconds * 2.0, 8.0)
            )
            self._rate_limit_backoff_seconds = next_backoff_seconds
            self._rate_limit_until_monotonic = max(
                self._rate_limit_until_monotonic,
                time.monotonic() + next_backoff_seconds,
            )
            return next_backoff_seconds

    def _clear_shared_rate_limit_backoff_if_expired(self) -> None:
        """Reset the shared backoff after the cooldown window has fully elapsed."""
        with self._rate_limit_lock:
            if time.monotonic() >= self._rate_limit_until_monotonic:
                self._rate_limit_backoff_seconds = 0.0
                self._rate_limit_until_monotonic = 0.0

    def _run_sdk_call(self, callback: Any, *args: Any, **kwargs: Any) -> Any:
        """Run one Modal SDK call with shared retry and backoff semantics."""
        retryable_errors = self._resource_exhausted_error_types()
        max_attempts = 5
        for attempt_index in range(max_attempts):
            self._wait_for_shared_rate_limit_backoff()
            future = _MODAL_VOLUME_EXECUTOR.submit(callback, *args, **kwargs)
            try:
                result = future.result()
                self._clear_shared_rate_limit_backoff_if_expired()
                return result
            except retryable_errors:
                if attempt_index >= max_attempts - 1:
                    raise
                backoff_seconds = self._record_shared_rate_limit_backoff()
                logger.warning(
                    "Modal %s call %s hit rate limiting on attempt %d/%d; applying shared retry backoff of %.2fs.",
                    self._target_kind,
                    getattr(callback, "__name__", repr(callback)),
                    attempt_index + 1,
                    max_attempts,
                    backoff_seconds,
                )
        raise RuntimeError("Modal SDK call retry loop exited unexpectedly.")


class LocalMirrorVolume:
    """Simple filesystem-backed volume used for tests and dry runs."""

    def __init__(self, root: Path) -> None:
        """Initialize the local mirror volume root."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self, remote_path: str) -> bool:
        """Return whether a file already exists in the local mirror."""
        return self._resolve(remote_path).exists()

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Copy a local file into the mirror volume."""
        target = self._resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(local_path.read_bytes())

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Write bytes into the mirror volume."""
        target = self._resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def _resolve(self, remote_path: str) -> Path:
        """Resolve a remote storage path relative to the mirror root."""
        return self.root / remote_path.lstrip("/")


class LocalFileSyncIndex:
    """JSON-backed sync index used for local mirrors and tests."""

    def __init__(self, root: Path) -> None:
        """Initialize the on-disk metadata store."""
        self._index_path = root / "metadata" / "sync_index.json"
        self._lock = threading.Lock()
        self._records = self._load_records()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return one stored sync record when it exists."""
        with self._lock:
            payload = self._records.get(key)
            return dict(payload) if isinstance(payload, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Persist one sync record to the local metadata file."""
        with self._lock:
            self._records[key] = dict(value)
            self._save_records()

    def _load_records(self) -> dict[str, dict[str, Any]]:
        """Load the persisted sync index when available."""
        if not self._index_path.exists():
            return {}
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Sync index at %s is unreadable; rebuilding it from scratch.", self._index_path)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, dict)
        }

    def _save_records(self) -> None:
        """Write the current sync index to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(self._records, sort_keys=True), encoding="utf-8")


class ModalDictSyncIndex(_ModalSdkCaller):
    """Modal Dict-backed sync index for remote content-addressed uploads."""

    def __init__(self, dict_name: str) -> None:
        """Resolve a named Modal Dict lazily from the local SDK client."""
        if modal is None:
            raise RuntimeError("Modal SDK is required for ModalDictSyncIndex.")
        super().__init__(target_kind="dict")
        self._dict = modal.Dict.from_name(dict_name, create_if_missing=True)
        self._missing = object()
        self._cache: dict[str, object] = {}
        self._cache_lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return one stored sync record when it exists."""
        with self._cache_lock:
            cached_value = self._cache.get(key, self._missing)
        if cached_value is not self._missing:
            return dict(cached_value) if isinstance(cached_value, dict) else None
        payload = self._run_sdk_call(self._dict.get, key)
        normalized_payload = dict(payload) if isinstance(payload, dict) else None
        with self._cache_lock:
            self._cache[key] = dict(normalized_payload) if normalized_payload is not None else None
        return dict(normalized_payload) if normalized_payload is not None else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Persist one sync record to the shared Modal Dict."""
        normalized_value = dict(value)

        def write_record() -> None:
            self._dict[key] = normalized_value

        self._run_sdk_call(write_record)
        with self._cache_lock:
            self._cache[key] = dict(normalized_value)


class ModalVolumeBackend(_ModalSdkCaller):
    """Modal Volume-backed storage for real remote execution."""

    def __init__(self, volume_name: str) -> None:
        """Resolve a named Modal volume lazily from the local SDK client."""
        if modal is None:
            raise RuntimeError("Modal SDK is required for ModalVolumeBackend.")
        super().__init__(target_kind="volume")
        self._volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        self._exists_cache: dict[str, bool] = {}
        self._exists_cache_lock = threading.Lock()

    def exists(self, remote_path: str) -> bool:
        """Return whether a file already exists in the Modal volume."""
        with self._exists_cache_lock:
            cached_result = self._exists_cache.get(remote_path)
        if cached_result is not None:
            return cached_result
        try:
            exists = len(self._run_sdk_call(self._volume.listdir, remote_path, recursive=False)) > 0
        except modal.exception.NotFoundError:
            exists = False
        with self._exists_cache_lock:
            self._exists_cache[remote_path] = exists
        return exists

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the Modal volume."""
        def upload() -> None:
            with self._volume.batch_upload() as batch:
                batch.put_file(local_path, remote_path)

        self._run_sdk_call(upload)
        with self._exists_cache_lock:
            self._exists_cache[remote_path] = True

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload bytes into the Modal volume."""
        def upload() -> None:
            with self._volume.batch_upload() as batch:
                batch.put_file(io.BytesIO(payload), remote_path)

        self._run_sdk_call(upload)
        with self._exists_cache_lock:
            self._exists_cache[remote_path] = True


@dataclass
class ModalAssetSyncEngine:
    """Content-addressable storage sync engine for files and custom nodes."""

    volume: VolumeBackend
    settings: ModalSyncSettings
    sync_index: SyncIndexBackend | None = None
    _hash_cache: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    _path_resolution_cache: dict[str, str | None] = field(init=False, default_factory=dict)
    _hash_cache_dirty: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Load persistent metadata caches used to avoid repeated hashing work."""
        if self.sync_index is None:
            self.sync_index = LocalFileSyncIndex(self.settings.local_storage_root)
        self._hash_cache = self._load_hash_cache()

    @classmethod
    def from_environment(cls, settings: ModalSyncSettings | None = None) -> "ModalAssetSyncEngine":
        """Create a sync engine using the local mirror backend by default."""
        resolved_settings = settings or get_settings()
        backend: VolumeBackend
        sync_index: SyncIndexBackend
        if resolved_settings.execution_mode == "remote" and modal is not None:
            logger.info(
                "Using Modal volume backend %s and Modal sync index %s for remote asset sync.",
                resolved_settings.volume_name,
                resolved_settings.sync_index_dict_name,
            )
            backend = ModalVolumeBackend(resolved_settings.volume_name)
            sync_index = ModalDictSyncIndex(resolved_settings.sync_index_dict_name)
        else:
            if resolved_settings.execution_mode == "remote" and modal is None:
                logger.warning(
                    "Modal SDK is unavailable in remote execution mode; falling back to local mirror storage."
                )
            backend = LocalMirrorVolume(resolved_settings.local_storage_root)
            sync_index = LocalFileSyncIndex(resolved_settings.local_storage_root)
        return cls(volume=backend, settings=resolved_settings, sync_index=sync_index)

    def sync_file(
        self,
        local_path: Path,
        remote_folder: str = "/assets",
        *,
        status_callback: SyncStatusCallback | None = None,
        item_index: int | None = None,
        total_items: int | None = None,
    ) -> SyncedAsset:
        """Sync a file into content-addressable remote storage."""
        resolved_path = local_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Asset not found: {resolved_path}")

        sha256 = self._hash_file(resolved_path)
        proposed_remote_path = f"{remote_folder.rstrip('/')}/{sha256}_{resolved_path.name}"
        sync_result = self._sync_content_addressed_file(
            local_path=resolved_path,
            remote_path=proposed_remote_path,
            sync_key=self._asset_sync_index_key(sha256),
            source_description=str(resolved_path),
            status_callback=status_callback,
            upload_status_message=_format_asset_upload_status(
                resolved_path.name,
                item_index=item_index,
                total_items=total_items,
            ),
            status_current=item_index,
            status_total=total_items,
        )

        return SyncedAsset(
            local_path=resolved_path,
            remote_path=sync_result.remote_path,
            sha256=sha256,
            uploaded=sync_result.uploaded,
        )

    def sync_prompt_inputs(
        self,
        inputs: dict[str, Any],
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> tuple[dict[str, Any], list[SyncedAsset]]:
        """Rewrite file-like prompt inputs to mirrored storage paths."""
        synced_assets: list[SyncedAsset] = []
        sync_started_at = time.perf_counter()
        logger.info("Scanning prompt inputs for syncable assets.")
        syncable_asset_paths = self._collect_syncable_asset_paths(inputs)
        syncable_asset_index = 0

        def rewrite(value: Any) -> Any:
            nonlocal syncable_asset_index
            if isinstance(value, str):
                maybe_path = self._resolve_model_path(value)
                if maybe_path is not None:
                    syncable_asset_index += 1
                    synced_asset = self.sync_file(
                        maybe_path,
                        status_callback=status_callback,
                        item_index=syncable_asset_index,
                        total_items=len(syncable_asset_paths),
                    )
                    synced_assets.append(synced_asset)
                    return synced_asset.remote_path
                return value
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, dict):
                return {str(key): rewrite(item) for key, item in value.items()}
            return value

        rewritten_inputs = rewrite(inputs)
        logger.info(
            "Finished scanning prompt inputs in %.3fs with %d synced assets.",
            time.perf_counter() - sync_started_at,
            len(synced_assets),
        )
        return rewritten_inputs, synced_assets

    def sync_custom_nodes_directory(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> SyncedAsset | None:
        """Mirror custom_nodes as a manifest plus per-package archives when available."""
        custom_nodes_dir = self.settings.custom_nodes_dir
        if custom_nodes_dir is None or not custom_nodes_dir.exists():
            logger.info("No custom_nodes directory detected for mirroring.")
            return None

        sync_started_at = time.perf_counter()
        logger.info("Hashing custom_nodes directory at %s", custom_nodes_dir)
        directory_hash = self._hash_directory(custom_nodes_dir)
        logger.info(
            "Finished hashing custom_nodes directory in %.3fs with digest %s.",
            time.perf_counter() - sync_started_at,
            directory_hash,
        )
        manifest_sync_key = self._custom_nodes_manifest_sync_index_key(directory_hash)
        manifest_record = self._lookup_sync_record(manifest_sync_key)
        if manifest_record is not None:
            remote_path = str(manifest_record["remote_path"])
            logger.info(
                "Custom_nodes manifest already mirrored at %s after %.3fs total sync time.",
                remote_path,
                time.perf_counter() - sync_started_at,
            )
            return SyncedAsset(
                local_path=custom_nodes_dir,
                remote_path=remote_path,
                sha256=directory_hash,
                uploaded=False,
            )
        remote_path = self._custom_nodes_manifest_remote_path(directory_hash)

        archive_specs = self._custom_nodes_archive_specs(custom_nodes_dir)
        if not archive_specs:
            logger.info("Custom_nodes directory %s contained no syncable files.", custom_nodes_dir)
            return None

        if any(
            not self._cached_custom_nodes_archive_path(
                archive_spec.entry_name,
                archive_spec.sha256,
            ).exists()
            for archive_spec in archive_specs
        ):
            _emit_sync_status(status_callback, "Packaging custom nodes ZIP for Modal")
        _emit_sync_status(status_callback, "Uploading custom nodes ZIP to Modal")

        archive_results = self._sync_custom_nodes_archives_parallel(
            custom_nodes_dir=custom_nodes_dir,
            archive_specs=archive_specs,
        )
        uploaded = any(archive_result.uploaded for archive_result in archive_results)
        manifest_entries = [
            {
                "entry_name": archive_result.entry_name,
                "display_name": archive_result.display_name,
                "sha256": archive_result.sha256,
                "remote_path": archive_result.remote_path,
            }
            for archive_result in archive_results
        ]

        manifest_path = self._cached_custom_nodes_manifest_path(directory_hash)
        if manifest_path.exists():
            logger.info(
                "Reusing cached custom_nodes manifest %s for digest %s.",
                manifest_path,
                directory_hash,
            )
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "bundle_sha256": directory_hash,
                        "entries": manifest_entries,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        manifest_sync_result = self._sync_content_addressed_file(
            local_path=manifest_path,
            remote_path=remote_path,
            sync_key=manifest_sync_key,
            source_description=str(custom_nodes_dir),
            upload_status_message="Uploading custom nodes ZIP to Modal",
        )
        uploaded = uploaded or manifest_sync_result.uploaded

        logger.info(
            "Finished custom_nodes sync to %s in %.3fs total.",
            manifest_sync_result.remote_path,
            time.perf_counter() - sync_started_at,
        )
        return SyncedAsset(
            local_path=custom_nodes_dir,
            remote_path=manifest_sync_result.remote_path,
            sha256=directory_hash,
            uploaded=uploaded,
        )

    def _sync_content_addressed_file(
        self,
        *,
        local_path: Path,
        remote_path: str,
        sync_key: str,
        source_description: str,
        status_callback: SyncStatusCallback | None = None,
        upload_status_message: str | None = None,
        status_current: int | None = None,
        status_total: int | None = None,
    ) -> _ContentAddressedSyncResult:
        """Upload one deterministic file only when its digest is absent from the sync index."""
        existing_record = self._lookup_sync_record(sync_key)
        if existing_record is not None:
            indexed_remote_path = str(existing_record["remote_path"])
            logger.info(
                "Reusing mirrored asset at %s because sync index key %s already exists.",
                indexed_remote_path,
                sync_key,
            )
            return _ContentAddressedSyncResult(
                remote_path=indexed_remote_path,
                uploaded=False,
            )

        logger.info("Syncing %s to %s", source_description, remote_path)
        _emit_sync_status(
            status_callback,
            upload_status_message or f"Uploading {Path(source_description).name} to Modal",
            status_current,
            status_total,
        )
        self.volume.put_file(local_path, remote_path)
        self._store_sync_record(
            sync_key=sync_key,
            remote_path=remote_path,
            source_description=source_description,
        )
        return _ContentAddressedSyncResult(remote_path=remote_path, uploaded=True)

    def _sync_custom_nodes_archives_parallel(
        self,
        *,
        custom_nodes_dir: Path,
        archive_specs: list[_CustomNodesArchiveSpec],
    ) -> list[_CustomNodesArchiveSyncResult]:
        """Build and upload per-package custom_nodes archives in parallel."""
        max_workers = min(len(archive_specs), _custom_nodes_sync_worker_count())
        if max_workers <= 1:
            return [
                self._sync_custom_nodes_archive_spec(custom_nodes_dir, archive_spec)
                for archive_spec in archive_specs
            ]

        results_by_entry_name: dict[str, _CustomNodesArchiveSyncResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_by_entry_name: dict[str, Future[_CustomNodesArchiveSyncResult]] = {
                archive_spec.entry_name: executor.submit(
                    self._sync_custom_nodes_archive_spec,
                    custom_nodes_dir,
                    archive_spec,
                )
                for archive_spec in archive_specs
            }
            for entry_name, future in futures_by_entry_name.items():
                results_by_entry_name[entry_name] = future.result()
        return [
            results_by_entry_name[archive_spec.entry_name]
            for archive_spec in archive_specs
        ]

    def _sync_custom_nodes_archive_spec(
        self,
        custom_nodes_dir: Path,
        archive_spec: _CustomNodesArchiveSpec,
    ) -> _CustomNodesArchiveSyncResult:
        """Build and upload one per-package custom_nodes archive."""
        archive_path = self._cached_custom_nodes_archive_path(
            archive_spec.entry_name,
            archive_spec.sha256,
        )
        archive_remote_path = self._custom_nodes_archive_remote_path(
            archive_spec.entry_name,
            archive_spec.sha256,
        )
        if archive_path.exists():
            logger.info(
                "Reusing cached custom_nodes archive %s for entry=%s digest=%s.",
                archive_path,
                archive_spec.display_name,
                archive_spec.sha256,
            )
        else:
            archive_started_at = time.perf_counter()
            logger.info(
                "Creating custom_nodes archive for entry=%s from %d files.",
                archive_spec.display_name,
                len(archive_spec.files),
            )
            self._create_archive_from_files(
                custom_nodes_dir,
                list(archive_spec.files),
                archive_path,
            )
            logger.info(
                "Created custom_nodes archive %s for entry=%s in %.3fs.",
                archive_path,
                archive_spec.display_name,
                time.perf_counter() - archive_started_at,
            )

        entry_uploaded = self._sync_content_addressed_file(
            local_path=archive_path,
            remote_path=archive_remote_path,
            sync_key=self._custom_nodes_entry_sync_index_key(
                archive_spec.entry_name,
                archive_spec.sha256,
            ),
            source_description=archive_spec.source_description,
        )
        return _CustomNodesArchiveSyncResult(
            entry_name=archive_spec.entry_name,
            display_name=archive_spec.display_name,
            sha256=archive_spec.sha256,
            remote_path=entry_uploaded.remote_path,
            uploaded=entry_uploaded.uploaded,
        )

    def _lookup_sync_record(self, sync_key: str) -> dict[str, Any] | None:
        """Return one normalized sync-index record when the key is present."""
        assert self.sync_index is not None
        payload = self.sync_index.get(sync_key)
        if payload is None:
            return None
        remote_path = payload.get("remote_path")
        if not isinstance(remote_path, str) or not remote_path:
            logger.warning("Ignoring malformed sync-index record for key=%s payload=%s.", sync_key, payload)
            return None
        return dict(payload)

    def _store_sync_record(
        self,
        *,
        sync_key: str,
        remote_path: str,
        source_description: str,
    ) -> None:
        """Persist one normalized sync-index record."""
        assert self.sync_index is not None
        self.sync_index.put(
            sync_key,
            {
                "remote_path": remote_path,
                "source": source_description,
            },
        )

    def _asset_sync_index_key(self, sha256: str) -> str:
        """Return the sync-index key for one content-addressed asset digest."""
        return f"{self.settings.volume_name}:asset:{sha256}"

    def _custom_nodes_manifest_sync_index_key(self, directory_hash: str) -> str:
        """Return the sync-index key for one whole-tree custom_nodes manifest digest."""
        return f"{self.settings.volume_name}:custom_nodes_manifest:{directory_hash}"

    def _custom_nodes_entry_sync_index_key(self, entry_name: str, entry_hash: str) -> str:
        """Return the sync-index key for one top-level custom_nodes entry archive digest."""
        return (
            f"{self.settings.volume_name}:custom_nodes_entry:"
            f"{self._custom_nodes_entry_slug(entry_name)}:{entry_hash}"
        )


    def _resolve_model_path(self, value: str) -> Path | None:
        """Resolve a prompt string into a local model file path when possible."""
        if value in self._path_resolution_cache:
            cached = self._path_resolution_cache[value]
            return Path(cached) if cached is not None else None

        path = Path(value).expanduser()
        if path.suffix.lower() not in _SYNC_EXTENSIONS:
            self._path_resolution_cache[value] = None
            return None
        if path.is_file():
            resolved = path.resolve()
            self._path_resolution_cache[value] = str(resolved)
            return resolved

        if os.path.isabs(value):
            self._path_resolution_cache[value] = None
            return None

        try:
            import folder_paths
        except ModuleNotFoundError:
            folder_paths = None

        if folder_paths is not None:
            for folder_name in folder_paths.folder_names_and_paths:
                full_path = folder_paths.get_full_path(folder_name, value)
                if full_path is not None:
                    resolved = Path(full_path).resolve()
                    self._path_resolution_cache[value] = str(resolved)
                    return resolved

        if self.settings.comfyui_root is not None:
            candidate = self.settings.comfyui_root / value
            if candidate.is_file():
                resolved = candidate.resolve()
                self._path_resolution_cache[value] = str(resolved)
                return resolved

        self._path_resolution_cache[value] = None
        return None

    def _collect_syncable_asset_paths(self, value: Any) -> list[Path]:
        """Return all prompt asset paths that resolve to syncable local files."""
        collected_paths: list[Path] = []

        def visit(item: Any) -> None:
            if isinstance(item, str):
                resolved_path = self._resolve_model_path(item)
                if resolved_path is not None:
                    collected_paths.append(resolved_path)
                return
            if isinstance(item, list):
                for child in item:
                    visit(child)
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)

        visit(value)
        return collected_paths

    def _hash_file(self, path: Path) -> str:
        """Compute the SHA256 digest for a file."""
        resolved_path = path.resolve()
        stat_result = resolved_path.stat()
        cache_key = str(resolved_path)
        cache_entry = self._hash_cache.get(cache_key)
        if (
            cache_entry is not None
            and cache_entry.get("kind") == "file"
            and cache_entry.get("size") == stat_result.st_size
            and cache_entry.get("mtime_ns") == stat_result.st_mtime_ns
        ):
            return str(cache_entry["sha256"])

        digest = hashlib.sha256()
        with resolved_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
        self._hash_cache[cache_key] = {
            "kind": "file",
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "sha256": sha256,
        }
        self._mark_hash_cache_dirty()
        return sha256

    def _hash_directory(self, path: Path) -> str:
        """Compute a stable SHA256 digest for a directory tree."""
        hash_started_at = time.perf_counter()
        resolved_path = path.resolve()
        digest = hashlib.sha256()
        files = sorted(self._iter_files(resolved_path), key=lambda item: item.relative_to(resolved_path).as_posix())
        logger.info("Hashing %d files under %s", len(files), resolved_path)
        fingerprint = self._directory_fingerprint(resolved_path, files)
        cache_key = f"dir::{resolved_path}"
        cache_entry = self._hash_cache.get(cache_key)
        if (
            cache_entry is not None
            and cache_entry.get("kind") == "dir"
            and cache_entry.get("fingerprint") == fingerprint
        ):
            logger.info(
                "Reused cached directory hash for %s over %d files in %.3fs.",
                resolved_path,
                len(files),
                time.perf_counter() - hash_started_at,
            )
            return str(cache_entry["sha256"])

        for child in files:
            relative_path = child.relative_to(resolved_path).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._hash_file(child).encode("ascii"))
            digest.update(b"\0")
        sha256 = digest.hexdigest()
        self._hash_cache[cache_key] = {
            "kind": "dir",
            "fingerprint": fingerprint,
            "sha256": sha256,
        }
        self._mark_hash_cache_dirty()
        logger.info(
            "Computed directory hash for %s over %d files in %.3fs.",
            resolved_path,
            len(files),
            time.perf_counter() - hash_started_at,
        )
        return sha256

    def _hash_file_group(self, root: Path, files: list[Path]) -> str:
        """Compute a stable digest for a selected file subset rooted under one directory."""
        digest = hashlib.sha256()
        for child in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            relative_path = child.relative_to(root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._hash_file(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _create_archive(self, path: Path, archive_path: Path) -> Path:
        """Create a zip archive for the given directory tree."""
        files = sorted(self._iter_files(path), key=lambda item: item.relative_to(path).as_posix())
        return self._create_archive_from_files(path, files, archive_path)

    def _create_archive_from_files(
        self,
        root_path: Path,
        files: list[Path],
        archive_path: Path,
    ) -> Path:
        """Create a zip archive from a selected file list rooted under one directory."""
        archive_started_at = time.perf_counter()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Archiving %d files from %s into %s", len(files), root_path, archive_path)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for child in sorted(files, key=lambda item: item.relative_to(root_path).as_posix()):
                archive.write(child, arcname=child.relative_to(root_path))

        logger.info(
            "Finished archive build for %s in %.3fs.",
            root_path,
            time.perf_counter() - archive_started_at,
        )
        return archive_path

    def _cached_custom_nodes_archive_path(self, entry_name: str, entry_hash: str) -> Path:
        """Return the deterministic local path for one digest-keyed custom_nodes slice archive."""
        return (
            self.settings.local_storage_root
            / "custom_nodes_archives"
            / self._custom_nodes_entry_slug(entry_name)
            / f"{entry_hash}_{self.settings.custom_nodes_archive_name}"
        )

    def _cached_custom_nodes_manifest_path(self, directory_hash: str) -> Path:
        """Return the deterministic local path for a whole-tree custom_nodes manifest."""
        return (
            self.settings.local_storage_root
            / "custom_nodes_manifests"
            / f"{directory_hash}_custom_nodes_bundle_manifest.json"
        )

    def _custom_nodes_manifest_remote_path(self, directory_hash: str) -> str:
        """Return the remote storage path for a whole-tree custom_nodes manifest."""
        return f"/custom_nodes/manifests/{directory_hash}_custom_nodes_bundle_manifest.json"

    def _custom_nodes_archive_remote_path(self, entry_name: str, entry_hash: str) -> str:
        """Return the remote storage path for one content-addressed custom_nodes slice archive."""
        return (
            f"/custom_nodes/entries/{self._custom_nodes_entry_slug(entry_name)}/"
            f"{entry_hash}_{self.settings.custom_nodes_archive_name}"
        )

    def _custom_nodes_entry_slug(self, entry_name: str) -> str:
        """Return a filesystem-safe slug for one top-level custom_nodes entry name."""
        normalized_name = entry_name.strip() or "root_files"
        return "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in normalized_name
        )

    def _custom_nodes_archive_specs(self, custom_nodes_dir: Path) -> list[_CustomNodesArchiveSpec]:
        """Return deterministic archive specs for each top-level custom_nodes payload slice."""
        resolved_root = custom_nodes_dir.resolve()
        root_files: list[Path] = []
        archive_specs: list[_CustomNodesArchiveSpec] = []

        for child in sorted(resolved_root.iterdir(), key=lambda item: item.name):
            if child.name in _SKIP_DIRS:
                continue
            if child.is_file():
                if child.suffix.lower() in _SKIP_FILE_SUFFIXES:
                    continue
                root_files.append(child)
                continue
            if child.is_dir():
                files = sorted(
                    self._iter_files(child),
                    key=lambda item: item.relative_to(resolved_root).as_posix(),
                )
                if not files:
                    continue
                archive_specs.append(
                    _CustomNodesArchiveSpec(
                        entry_name=child.name,
                        display_name=child.name,
                        source_description=str(child),
                        files=tuple(files),
                        sha256=self._hash_directory(child),
                    )
                )

        if root_files:
            archive_specs.append(
                _CustomNodesArchiveSpec(
                    entry_name="root_files",
                    display_name="root files",
                    source_description=str(resolved_root),
                    files=tuple(root_files),
                    sha256=self._hash_file_group(resolved_root, root_files),
                )
            )

        return archive_specs

    def _directory_fingerprint(self, root: Path, files: list[Path]) -> str:
        """Return a metadata-only fingerprint for a directory tree."""
        digest = hashlib.sha256()
        for child in files:
            stat_result = child.stat()
            digest.update(child.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat_result.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat_result.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _hash_cache_path(self) -> Path:
        """Return the on-disk metadata cache path."""
        return self.settings.local_storage_root / "metadata" / "hash_cache.json"

    def _load_hash_cache(self) -> dict[str, dict[str, Any]]:
        """Load the persistent hash cache from disk when available."""
        cache_path = self._hash_cache_path()
        if not cache_path.exists():
            return {}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Hash cache at %s is unreadable; rebuilding it from scratch.", cache_path)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, dict)
        }

    def _mark_hash_cache_dirty(self) -> None:
        """Persist the hash cache after it changes."""
        self._hash_cache_dirty = True
        self._save_hash_cache()

    def _save_hash_cache(self) -> None:
        """Write the persistent hash cache to disk."""
        if not self._hash_cache_dirty:
            return
        cache_path = self._hash_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(self._hash_cache, sort_keys=True), encoding="utf-8")
        self._hash_cache_dirty = False

    def _iter_files(self, path: Path) -> list[Path]:
        """Yield files from a directory tree while skipping cache folders."""
        files: list[Path] = []
        for root, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            for filename in filenames:
                child = Path(root) / filename
                if child.suffix.lower() in _SKIP_FILE_SUFFIXES:
                    continue
                files.append(child)
        return files
