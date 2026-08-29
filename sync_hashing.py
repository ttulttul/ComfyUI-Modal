"""Persistent hashing and filesystem filtering for asset synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

if __package__:
    from .settings import ModalSyncSettings
    from .sync_protocols import CancellationCheck, SyncCancelledError
else:  # pragma: no cover - flat import inside the Modal container.
    from settings import ModalSyncSettings
    from sync_protocols import CancellationCheck, SyncCancelledError

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "__pycache__", "venv",
}
_SKIP_FILE_SUFFIXES = {".log", ".pyc", ".pyd", ".pyo", ".so", ".swp", ".tmp"}


@dataclass
class SyncHasher:
    """Own persistent content hashes and cooperative cancellation checks."""

    settings: ModalSyncSettings
    cancellation_check: CancellationCheck | None = None
    _hash_cache: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    _hash_cache_dirty: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Load the persistent hash cache once for this synchronization engine."""
        self._hash_cache = self._load_hash_cache()

    def _hash_file(self, path: Path) -> str:
        """Compute the SHA256 digest for a file."""
        self._raise_if_cancelled()
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
                self._raise_if_cancelled()
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

    def _raise_if_cancelled(self) -> None:
        """Abort queue-time synchronization when its prompt was cancelled."""
        if self.cancellation_check is not None and self.cancellation_check():
            raise SyncCancelledError("Remote workflow preparation was cancelled.")

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

