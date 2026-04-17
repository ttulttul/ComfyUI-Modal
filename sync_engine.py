"""Asset synchronization helpers for Modal-backed execution."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .settings import ModalSyncSettings, get_settings

logger = logging.getLogger(__name__)
_MODAL_VOLUME_EXECUTOR = ThreadPoolExecutor(max_workers=1)

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


class VolumeBackend(Protocol):
    """Minimal storage interface needed by the sync engine."""

    def exists(self, remote_path: str) -> bool:
        """Return whether the remote path already exists."""

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the remote storage backend."""

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload raw bytes into the remote storage backend."""


@dataclass(frozen=True)
class SyncedAsset:
    """Description of a local asset mirrored into the remote storage root."""

    local_path: Path
    remote_path: str
    sha256: str


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


class ModalVolumeBackend:
    """Modal Volume-backed storage for real remote execution."""

    def __init__(self, volume_name: str) -> None:
        """Resolve a named Modal volume lazily from the local SDK client."""
        if modal is None:
            raise RuntimeError("Modal SDK is required for ModalVolumeBackend.")
        self._volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    def _run_volume_call(self, callback: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a Modal SDK volume call in a worker thread outside the request event loop."""
        future = _MODAL_VOLUME_EXECUTOR.submit(callback, *args, **kwargs)
        return future.result()

    def exists(self, remote_path: str) -> bool:
        """Return whether a file already exists in the Modal volume."""
        try:
            return len(self._run_volume_call(self._volume.listdir, remote_path, recursive=False)) > 0
        except modal.exception.NotFoundError:
            return False

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the Modal volume."""
        def upload() -> None:
            with self._volume.batch_upload() as batch:
                batch.put_file(local_path, remote_path)

        self._run_volume_call(upload)

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload bytes into the Modal volume."""
        def upload() -> None:
            with self._volume.batch_upload() as batch:
                batch.put_file(io.BytesIO(payload), remote_path)

        self._run_volume_call(upload)


@dataclass
class ModalAssetSyncEngine:
    """Content-addressable storage sync engine for files and custom nodes."""

    volume: VolumeBackend
    settings: ModalSyncSettings
    _hash_cache: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    _path_resolution_cache: dict[str, str | None] = field(init=False, default_factory=dict)
    _hash_cache_dirty: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Load persistent metadata caches used to avoid repeated hashing work."""
        self._hash_cache = self._load_hash_cache()

    @classmethod
    def from_environment(cls, settings: ModalSyncSettings | None = None) -> "ModalAssetSyncEngine":
        """Create a sync engine using the local mirror backend by default."""
        resolved_settings = settings or get_settings()
        backend: VolumeBackend
        if resolved_settings.execution_mode == "remote" and modal is not None:
            logger.info(
                "Using Modal volume backend %s for remote asset sync.",
                resolved_settings.volume_name,
            )
            backend = ModalVolumeBackend(resolved_settings.volume_name)
        else:
            if resolved_settings.execution_mode == "remote" and modal is None:
                logger.warning(
                    "Modal SDK is unavailable in remote execution mode; falling back to local mirror storage."
                )
            backend = LocalMirrorVolume(resolved_settings.local_storage_root)
        return cls(volume=backend, settings=resolved_settings)

    def sync_file(self, local_path: Path, remote_folder: str = "/assets") -> SyncedAsset:
        """Sync a file into content-addressable remote storage."""
        resolved_path = local_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Asset not found: {resolved_path}")

        sha256 = self._hash_file(resolved_path)
        marker_path = f"/hashes/{sha256}.done"
        remote_path = f"{remote_folder.rstrip('/')}/{sha256}_{resolved_path.name}"

        if not self.volume.exists(marker_path):
            logger.info("Syncing %s to %s", resolved_path, remote_path)
            self.volume.put_file(resolved_path, remote_path)
            self.volume.put_bytes(
                json.dumps({"source": str(resolved_path), "remote_path": remote_path}).encode(
                    "utf-8"
                ),
                marker_path,
            )
        else:
            logger.debug("Asset already mirrored for sha=%s", sha256)

        return SyncedAsset(local_path=resolved_path, remote_path=remote_path, sha256=sha256)

    def sync_prompt_inputs(self, inputs: dict[str, Any]) -> tuple[dict[str, Any], list[SyncedAsset]]:
        """Rewrite file-like prompt inputs to mirrored storage paths."""
        synced_assets: list[SyncedAsset] = []
        sync_started_at = time.perf_counter()
        logger.info("Scanning prompt inputs for syncable assets.")

        def rewrite(value: Any) -> Any:
            if isinstance(value, str):
                maybe_path = self._resolve_model_path(value)
                if maybe_path is not None:
                    synced_asset = self.sync_file(maybe_path)
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

    def sync_custom_nodes_directory(self) -> SyncedAsset | None:
        """Zip and mirror the local custom_nodes directory when available."""
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
        archive_name = self.settings.custom_nodes_archive_name
        marker_path = f"/hashes/custom_nodes_{directory_hash}.done"
        remote_path = f"/custom_nodes/{directory_hash}_{archive_name}"

        if self.volume.exists(marker_path):
            logger.info(
                "Custom_nodes bundle already mirrored at %s after %.3fs total sync time.",
                remote_path,
                time.perf_counter() - sync_started_at,
            )
            return SyncedAsset(
                local_path=custom_nodes_dir,
                remote_path=remote_path,
                sha256=directory_hash,
            )

        archive_path = self._cached_custom_nodes_archive_path(directory_hash)
        if archive_path.exists():
            logger.info("Reusing cached custom_nodes archive %s for digest %s.", archive_path, directory_hash)
        else:
            archive_started_at = time.perf_counter()
            logger.info("Creating custom_nodes archive for %s", custom_nodes_dir)
            self._create_archive(custom_nodes_dir, archive_path)
            logger.info(
                "Created custom_nodes archive %s in %.3fs.",
                archive_path,
                time.perf_counter() - archive_started_at,
            )

        logger.info("Syncing custom_nodes bundle from %s to %s", custom_nodes_dir, remote_path)
        self.volume.put_file(archive_path, remote_path)
        self.volume.put_bytes(
            json.dumps({"source": str(custom_nodes_dir), "remote_path": remote_path}).encode(
                "utf-8"
            ),
            marker_path,
        )

        logger.info(
            "Finished custom_nodes sync to %s in %.3fs total.",
            remote_path,
            time.perf_counter() - sync_started_at,
        )
        return SyncedAsset(local_path=custom_nodes_dir, remote_path=remote_path, sha256=directory_hash)

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

    def _create_archive(self, path: Path, archive_path: Path) -> Path:
        """Create a zip archive for the given directory tree."""
        archive_started_at = time.perf_counter()
        files = sorted(self._iter_files(path), key=lambda item: item.relative_to(path).as_posix())
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Archiving %d files from %s into %s", len(files), path, archive_path)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for child in files:
                archive.write(child, arcname=child.relative_to(path))

        logger.info(
            "Finished archive build for %s in %.3fs.",
            path,
            time.perf_counter() - archive_started_at,
        )
        return archive_path

    def _cached_custom_nodes_archive_path(self, directory_hash: str) -> Path:
        """Return the deterministic local path for a digest-keyed custom_nodes archive."""
        return (
            self.settings.local_storage_root
            / "custom_nodes_archives"
            / f"{directory_hash}_{self.settings.custom_nodes_archive_name}"
        )

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
