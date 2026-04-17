"""Asset synchronization helpers for Modal-backed execution."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

        archive_started_at = time.perf_counter()
        logger.info("Creating custom_nodes archive for %s", custom_nodes_dir)
        archive_path = self._create_archive(custom_nodes_dir)
        logger.info(
            "Created custom_nodes archive %s in %.3fs.",
            archive_path,
            time.perf_counter() - archive_started_at,
        )
        try:
            logger.info("Syncing custom_nodes bundle from %s to %s", custom_nodes_dir, remote_path)
            self.volume.put_file(archive_path, remote_path)
            self.volume.put_bytes(
                json.dumps({"source": str(custom_nodes_dir), "remote_path": remote_path}).encode(
                    "utf-8"
                ),
                marker_path,
            )
        finally:
            archive_path.unlink(missing_ok=True)

        logger.info(
            "Finished custom_nodes sync to %s in %.3fs total.",
            remote_path,
            time.perf_counter() - sync_started_at,
        )
        return SyncedAsset(local_path=custom_nodes_dir, remote_path=remote_path, sha256=directory_hash)

    def _resolve_model_path(self, value: str) -> Path | None:
        """Resolve a prompt string into a local model file path when possible."""
        path = Path(value).expanduser()
        if path.suffix.lower() not in _SYNC_EXTENSIONS:
            return None
        if path.is_file():
            return path.resolve()

        if os.path.isabs(value):
            return None

        try:
            import folder_paths
        except ModuleNotFoundError:
            folder_paths = None

        if folder_paths is not None:
            for folder_name in folder_paths.folder_names_and_paths:
                full_path = folder_paths.get_full_path(folder_name, value)
                if full_path is not None:
                    return Path(full_path).resolve()

        if self.settings.comfyui_root is not None:
            candidate = self.settings.comfyui_root / value
            if candidate.is_file():
                return candidate.resolve()

        return None

    def _hash_file(self, path: Path) -> str:
        """Compute the SHA256 digest for a file."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _hash_directory(self, path: Path) -> str:
        """Compute a stable SHA256 digest for a directory tree."""
        hash_started_at = time.perf_counter()
        digest = hashlib.sha256()
        files = sorted(self._iter_files(path), key=lambda item: item.relative_to(path).as_posix())
        logger.info("Hashing %d files under %s", len(files), path)
        for child in files:
            relative_path = child.relative_to(path).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._hash_file(child).encode("ascii"))
            digest.update(b"\0")
        logger.info(
            "Computed directory hash for %s over %d files in %.3fs.",
            path,
            len(files),
            time.perf_counter() - hash_started_at,
        )
        return digest.hexdigest()

    def _create_archive(self, path: Path) -> Path:
        """Create a zip archive for the given directory tree."""
        archive_started_at = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            prefix="comfy-modal-custom-nodes-",
            suffix=".zip",
            delete=False,
        ) as handle:
            archive_path = Path(handle.name)

        files = sorted(self._iter_files(path), key=lambda item: item.relative_to(path).as_posix())
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
