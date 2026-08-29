"""Local and Modal storage/index backends for asset synchronization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised without Modal SDK.
    modal = None


def _modal_volume_worker_count() -> int:
    """Return the worker count used for local Modal volume SDK calls."""
    return 4


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

        try:
            self._run_sdk_call(upload)
        except FileExistsError:
            logger.info(
                "Treating Modal volume upload for %s as successful because the content-addressed path already exists.",
                remote_path,
            )
        with self._exists_cache_lock:
            self._exists_cache[remote_path] = True

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload bytes into the Modal volume."""
        def upload() -> None:
            with self._volume.batch_upload() as batch:
                batch.put_file(io.BytesIO(payload), remote_path)

        try:
            self._run_sdk_call(upload)
        except FileExistsError:
            logger.info(
                "Treating Modal volume byte upload for %s as successful because the content-addressed path already exists.",
                remote_path,
            )
        with self._exists_cache_lock:
            self._exists_cache[remote_path] = True

