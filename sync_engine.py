"""Asset synchronization helpers for Modal-backed execution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .settings import ModalSyncSettings, get_settings

logger = logging.getLogger(__name__)

_SYNC_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".vae"}
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


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


@dataclass
class ModalAssetSyncEngine:
    """Content-addressable storage sync engine for files and custom nodes."""

    volume: VolumeBackend
    settings: ModalSyncSettings

    @classmethod
    def from_environment(cls, settings: ModalSyncSettings | None = None) -> "ModalAssetSyncEngine":
        """Create a sync engine using the local mirror backend by default."""
        resolved_settings = settings or get_settings()
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

        return rewrite(inputs), synced_assets

    def sync_custom_nodes_directory(self) -> SyncedAsset | None:
        """Zip and mirror the local custom_nodes directory when available."""
        custom_nodes_dir = self.settings.custom_nodes_dir
        if custom_nodes_dir is None or not custom_nodes_dir.exists():
            logger.info("No custom_nodes directory detected for mirroring.")
            return None

        directory_hash = self._hash_directory(custom_nodes_dir)
        archive_name = self.settings.custom_nodes_archive_name
        marker_path = f"/hashes/custom_nodes_{directory_hash}.done"
        remote_path = f"/custom_nodes/{directory_hash}_{archive_name}"

        if self.volume.exists(marker_path):
            return SyncedAsset(
                local_path=custom_nodes_dir,
                remote_path=remote_path,
                sha256=directory_hash,
            )

        archive_path = self._create_archive(custom_nodes_dir)
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
        digest = hashlib.sha256()
        for child in sorted(self._iter_files(path), key=lambda item: item.relative_to(path).as_posix()):
            relative_path = child.relative_to(path).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._hash_file(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _create_archive(self, path: Path) -> Path:
        """Create a zip archive for the given directory tree."""
        with tempfile.NamedTemporaryFile(
            prefix="comfy-modal-custom-nodes-",
            suffix=".zip",
            delete=False,
        ) as handle:
            archive_path = Path(handle.name)

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for child in sorted(self._iter_files(path), key=lambda item: item.relative_to(path).as_posix()):
                archive.write(child, arcname=child.relative_to(path))

        return archive_path

    def _iter_files(self, path: Path) -> list[Path]:
        """Yield files from a directory tree while skipping cache folders."""
        files: list[Path] = []
        for root, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            for filename in filenames:
                files.append(Path(root) / filename)
        return files
