"""Custom-node archive, asset, manifest, and backfill synchronization."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Protocol
import zipfile

if __package__:
    from .settings import ModalSyncSettings
    from .sync_backends import _modal_volume_worker_count
    from .sync_hashing import _SKIP_DIRS, _SKIP_FILE_SUFFIXES
    from .sync_protocols import (
        CancellationCheck,
        SyncedAsset,
        SyncStatusCallback,
        _ContentAddressedSyncResult,
        _CustomNodeAssetSpec,
        _CustomNodeAssetSyncResult,
        _CustomNodesArchiveSpec,
        _CustomNodesArchiveSyncResult,
    )
    from .sync_r2_transfer import R2WriteBackCancelled, _R2_WRITE_BACK_COORDINATOR
else:  # pragma: no cover - flat import inside the Modal container.
    from settings import ModalSyncSettings
    from sync_backends import _modal_volume_worker_count
    from sync_hashing import _SKIP_DIRS, _SKIP_FILE_SUFFIXES
    from sync_protocols import (
        CancellationCheck,
        SyncedAsset,
        SyncStatusCallback,
        _ContentAddressedSyncResult,
        _CustomNodeAssetSpec,
        _CustomNodeAssetSyncResult,
        _CustomNodesArchiveSpec,
        _CustomNodesArchiveSyncResult,
    )
    from sync_r2_transfer import R2WriteBackCancelled, _R2_WRITE_BACK_COORDINATOR

logger = logging.getLogger(__name__)

_CUSTOM_NODES_MANIFEST_VERSION = 2
MODEL_FILE_EXTENSIONS = frozenset({
    ".bin", ".ckpt", ".engine", ".gguf", ".onnx", ".pt", ".pth",
    ".safetensors", ".vae",
})
_CUSTOM_NODE_ASSET_SUFFIXES = MODEL_FILE_EXTENSIONS


def _emit_sync_status(
    status_callback: SyncStatusCallback | None,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Emit one human-readable sync status update when available."""
    if status_callback is not None:
        status_callback(message, current, total)


def _custom_nodes_sync_worker_count() -> int:
    """Return the worker count used for parallel custom-node synchronization."""
    return max(4, min(16, os.cpu_count() or 1))


class CustomNodesSyncHost(Protocol):
    """Define the narrow engine services consumed by custom-node synchronization."""

    settings: ModalSyncSettings

    def _destination_label(self) -> str:
        """Return a user-facing synchronization destination."""

    def _hash_directory(self, path: Path) -> str:
        """Return the stable digest for one directory."""

    def _hash_file(self, path: Path) -> str:
        """Return the content digest for one file."""

    def _hash_file_group(self, root: Path, files: list[Path]) -> str:
        """Return the stable digest for selected files."""

    def _iter_files(self, path: Path) -> list[Path]:
        """Return syncable files below one directory."""

    def _lookup_sync_record(self, sync_key: str) -> dict[str, Any] | None:
        """Return an indexed synchronization record."""

    def _r2_writeback_enabled(self) -> bool:
        """Return whether R2 write-back is configured."""

    def _r2_writeback_job_prefix(self) -> tuple[str, str, str]:
        """Return the stable write-back namespace."""

    def _sync_content_addressed_file(
        self,
        **kwargs: Any,
    ) -> _ContentAddressedSyncResult:
        """Synchronize one content-addressed file."""

    def _sync_index_scope_prefix(self) -> str:
        """Return the active persistent-index scope."""


@dataclass
class CustomNodesSynchronizer:
    """Own custom-node synchronization state and archive orchestration."""

    _host: CustomNodesSyncHost
    _custom_nodes_sync_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _custom_nodes_sync_checked: bool = field(init=False, default=False)
    _custom_nodes_bundle_cache: SyncedAsset | None = field(init=False, default=None)

    def sync_custom_nodes_directory(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> SyncedAsset | None:
        """Mirror custom_nodes once per ComfyUI process and reuse that result afterward."""
        with self._custom_nodes_sync_lock:
            if self._custom_nodes_sync_checked:
                logger.info(
                    "Skipping custom_nodes rescan because this ComfyUI process already resolved it once."
                )
                return self._clone_cached_custom_nodes_bundle()

            bundle = self._sync_custom_nodes_directory_uncached(status_callback=status_callback)
            self._custom_nodes_bundle_cache = bundle
            self._custom_nodes_sync_checked = True
            return bundle

    def _clone_cached_custom_nodes_bundle(self) -> SyncedAsset | None:
        """Return the cached custom_nodes sync result as a no-upload reuse decision."""
        if self._custom_nodes_bundle_cache is None:
            return None
        return SyncedAsset(
            local_path=self._custom_nodes_bundle_cache.local_path,
            remote_path=self._custom_nodes_bundle_cache.remote_path,
            sha256=self._custom_nodes_bundle_cache.sha256,
            uploaded=False,
        )

    def _sync_custom_nodes_directory_uncached(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
        revisit_indexed_for_r2: bool = False,
        writeback_cancellation_check: CancellationCheck | None = None,
    ) -> SyncedAsset | None:
        """Mirror custom_nodes as a manifest plus per-package archives when available."""
        if writeback_cancellation_check is not None and writeback_cancellation_check():
            raise R2WriteBackCancelled("Custom-node R2 backfill yielded before scanning.")
        custom_nodes_dir = self._host.settings.custom_nodes_dir
        if custom_nodes_dir is None or not custom_nodes_dir.exists():
            logger.info("No custom_nodes directory detected for mirroring.")
            return None

        sync_started_at = time.perf_counter()
        logger.info("Hashing custom_nodes directory at %s", custom_nodes_dir)
        directory_hash = self._host._hash_directory(custom_nodes_dir)
        if writeback_cancellation_check is not None and writeback_cancellation_check():
            raise R2WriteBackCancelled("Custom-node R2 backfill yielded after hashing.")
        logger.info(
            "Finished hashing custom_nodes directory in %.3fs with digest %s.",
            time.perf_counter() - sync_started_at,
            directory_hash,
        )
        manifest_sync_key = self._custom_nodes_manifest_sync_index_key(directory_hash)
        manifest_record = self._host._lookup_sync_record(manifest_sync_key)
        if manifest_record is not None and (
            not self._host._r2_writeback_enabled() or not revisit_indexed_for_r2
        ):
            remote_path = str(manifest_record["remote_path"])
            if self._host._r2_writeback_enabled():
                self._schedule_custom_nodes_r2_backfill(directory_hash)
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
        if manifest_record is not None:
            logger.info(
                "Revisiting indexed custom_nodes entries during idle time so R2 "
                "write-back can backfill missing objects."
            )
        remote_path = self._custom_nodes_manifest_remote_path(directory_hash)

        archive_specs, asset_specs = self._custom_nodes_bundle_specs(custom_nodes_dir)
        if writeback_cancellation_check is not None and writeback_cancellation_check():
            raise R2WriteBackCancelled("Custom-node R2 backfill yielded after planning.")
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
            _emit_sync_status(
                status_callback,
                f"Packaging custom-node code for {self._host._destination_label()}",
            )
        _emit_sync_status(
            status_callback,
            f"Uploading custom-node code and assets to {self._host._destination_label()}",
        )

        archive_results = self._sync_custom_nodes_archives_parallel(
            custom_nodes_dir=custom_nodes_dir,
            archive_specs=archive_specs,
        )
        asset_results = self._sync_custom_node_assets_parallel(asset_specs)
        if writeback_cancellation_check is not None and writeback_cancellation_check():
            raise R2WriteBackCancelled("Custom-node R2 backfill yielded after packaging.")
        uploaded = any(
            result.uploaded for result in [*archive_results, *asset_results]
        )
        assets_by_entry: dict[str, list[dict[str, Any]]] = {}
        for asset_result in asset_results:
            assets_by_entry.setdefault(asset_result.entry_name, []).append(
                {
                    "relative_path": asset_result.relative_path,
                    "sha256": asset_result.sha256,
                    "size_bytes": asset_result.size_bytes,
                    "remote_path": asset_result.remote_path,
                }
            )
        manifest_entries = [
            {
                "entry_name": archive_result.entry_name,
                "display_name": archive_result.display_name,
                "sha256": archive_result.sha256,
                "remote_path": archive_result.remote_path,
                "assets": assets_by_entry.get(archive_result.entry_name, []),
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
                        "version": _CUSTOM_NODES_MANIFEST_VERSION,
                        "bundle_sha256": directory_hash,
                        "entries": manifest_entries,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        manifest_sync_result = self._host._sync_content_addressed_file(
            local_path=manifest_path,
            remote_path=remote_path,
            sha256=self._host._hash_file(manifest_path),
            sync_key=manifest_sync_key,
            source_description=str(custom_nodes_dir),
            upload_status_message=(
                f"Uploading custom-node manifest to {self._host._destination_label()}"
            ),
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

    def _schedule_custom_nodes_r2_backfill(self, directory_hash: str) -> None:
        """Defer indexed custom-node traversal until foreground workflows are idle."""
        if not self._host._r2_writeback_enabled():
            return
        job_key = (
            *self._host._r2_writeback_job_prefix(),
            "custom-nodes-backfill",
            directory_hash,
        )
        _R2_WRITE_BACK_COORDINATOR.submit(
            job_key,
            lambda cancellation_check: self._sync_custom_nodes_directory_uncached(
                revisit_indexed_for_r2=True,
                writeback_cancellation_check=cancellation_check,
            ),
        )

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

        archive_sha256 = self._host._hash_file(archive_path)
        entry_uploaded = self._host._sync_content_addressed_file(
            local_path=archive_path,
            remote_path=archive_remote_path,
            sha256=archive_sha256,
            sync_key=self._custom_nodes_entry_sync_index_key(
                archive_spec.entry_name,
                archive_spec.sha256,
            ),
            source_description=archive_spec.source_description,
        )
        return _CustomNodesArchiveSyncResult(
            entry_name=archive_spec.entry_name,
            display_name=archive_spec.display_name,
            sha256=archive_sha256,
            remote_path=entry_uploaded.remote_path,
            uploaded=entry_uploaded.uploaded,
        )

    def _sync_custom_node_assets_parallel(
        self,
        asset_specs: list[_CustomNodeAssetSpec],
    ) -> list[_CustomNodeAssetSyncResult]:
        """Upload package-owned model assets without embedding them in code ZIPs."""
        if not asset_specs:
            return []
        max_workers = min(len(asset_specs), _modal_volume_worker_count())
        if max_workers <= 1:
            return [self._sync_custom_node_asset_spec(spec) for spec in asset_specs]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._sync_custom_node_asset_spec, spec)
                for spec in asset_specs
            ]
            return [future.result() for future in futures]

    def _sync_custom_node_asset_spec(
        self,
        asset_spec: _CustomNodeAssetSpec,
    ) -> _CustomNodeAssetSyncResult:
        """Upload one package-owned model asset to its content-addressed path."""
        remote_path = self._custom_nodes_asset_remote_path(asset_spec)
        sync_result = self._host._sync_content_addressed_file(
            local_path=asset_spec.local_path,
            remote_path=remote_path,
            sha256=asset_spec.sha256,
            sync_key=self._custom_nodes_asset_sync_index_key(asset_spec.sha256),
            source_description=str(asset_spec.local_path),
        )
        return _CustomNodeAssetSyncResult(
            entry_name=asset_spec.entry_name,
            relative_path=asset_spec.relative_path,
            sha256=asset_spec.sha256,
            size_bytes=asset_spec.size_bytes,
            remote_path=sync_result.remote_path,
            uploaded=sync_result.uploaded,
        )

    def _custom_nodes_manifest_sync_index_key(self, directory_hash: str) -> str:
        """Return the sync-index key for one whole-tree custom_nodes manifest digest."""
        return (
            f"{self._host._sync_index_scope_prefix()}:custom_nodes_manifest:"
            f"v{_CUSTOM_NODES_MANIFEST_VERSION}:{directory_hash}"
        )

    def _custom_nodes_entry_sync_index_key(self, entry_name: str, entry_hash: str) -> str:
        """Return the sync-index key for one top-level custom_nodes entry archive digest."""
        return (
            f"{self._host._sync_index_scope_prefix()}:custom_nodes_entry:"
            f"{self._custom_nodes_entry_slug(entry_name)}:{entry_hash}"
        )

    def _custom_nodes_asset_sync_index_key(self, asset_hash: str) -> str:
        """Return the sync-index key for one package-owned model asset digest."""
        return f"{self._host._sync_index_scope_prefix()}:custom_nodes_asset:{asset_hash}"

    def _create_archive(self, path: Path, archive_path: Path) -> Path:
        """Create a zip archive for the given directory tree."""
        files = sorted(self._host._iter_files(path), key=lambda item: item.relative_to(path).as_posix())
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
            self._host.settings.local_storage_root
            / "custom_nodes_archives"
            / self._custom_nodes_entry_slug(entry_name)
            / f"{entry_hash}_{self._host.settings.custom_nodes_archive_name}"
        )

    def _cached_custom_nodes_manifest_path(self, directory_hash: str) -> Path:
        """Return the deterministic local path for a whole-tree custom_nodes manifest."""
        return (
            self._host.settings.local_storage_root
            / "custom_nodes_manifests"
            / f"{directory_hash}_custom_nodes_bundle_manifest_v{_CUSTOM_NODES_MANIFEST_VERSION}.json"
        )

    def _custom_nodes_manifest_remote_path(self, directory_hash: str) -> str:
        """Return the remote storage path for a whole-tree custom_nodes manifest."""
        return (
            f"/custom_nodes/manifests/{directory_hash}_custom_nodes_bundle_"
            f"manifest_v{_CUSTOM_NODES_MANIFEST_VERSION}.json"
        )

    def _custom_nodes_archive_remote_path(self, entry_name: str, entry_hash: str) -> str:
        """Return the remote storage path for one content-addressed custom_nodes slice archive."""
        return (
            f"/custom_nodes/entries/{self._custom_nodes_entry_slug(entry_name)}/"
            f"{entry_hash}_{self._host.settings.custom_nodes_archive_name}"
        )

    def _custom_nodes_asset_remote_path(self, asset_spec: _CustomNodeAssetSpec) -> str:
        """Return the content-addressed remote path for one custom-node model asset."""
        return (
            f"/custom_nodes/assets/{self._custom_nodes_entry_slug(asset_spec.entry_name)}/"
            f"{asset_spec.sha256}_{asset_spec.local_path.name}"
        )

    def _custom_nodes_entry_slug(self, entry_name: str) -> str:
        """Return a filesystem-safe slug for one top-level custom_nodes entry name."""
        normalized_name = entry_name.strip() or "root_files"
        return "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in normalized_name
        )

    def _custom_nodes_bundle_specs(
        self,
        custom_nodes_dir: Path,
    ) -> tuple[list[_CustomNodesArchiveSpec], list[_CustomNodeAssetSpec]]:
        """Split custom-node source archives from package-owned model assets."""
        resolved_root = custom_nodes_dir.resolve()
        root_files: list[Path] = []
        archive_specs: list[_CustomNodesArchiveSpec] = []
        asset_specs: list[_CustomNodeAssetSpec] = []

        for child in sorted(resolved_root.iterdir(), key=lambda item: item.name):
            if child.name in _SKIP_DIRS:
                continue
            if child.is_file():
                if child.suffix.lower() in _SKIP_FILE_SUFFIXES:
                    continue
                root_files.append(child)
                continue
            if child.is_dir():
                entry_files = sorted(
                    self._host._iter_files(child),
                    key=lambda item: item.relative_to(resolved_root).as_posix(),
                )
                code_files, entry_assets = self._partition_custom_node_files(entry_files)
                if not code_files:
                    continue
                archive_specs.append(
                    _CustomNodesArchiveSpec(
                        entry_name=child.name,
                        display_name=child.name,
                        source_description=str(child),
                        files=tuple(code_files),
                        sha256=self._host._hash_file_group(resolved_root, code_files),
                    )
                )
                asset_specs.extend(
                    self._custom_node_asset_specs(
                        resolved_root,
                        child.name,
                        entry_assets,
                    )
                )

        if root_files:
            code_files, root_assets = self._partition_custom_node_files(root_files)
        else:
            code_files, root_assets = [], []
        if code_files:
            archive_specs.append(
                _CustomNodesArchiveSpec(
                    entry_name="root_files",
                    display_name="root files",
                    source_description=str(resolved_root),
                    files=tuple(code_files),
                    sha256=self._host._hash_file_group(resolved_root, code_files),
                )
            )
            asset_specs.extend(
                self._custom_node_asset_specs(
                    resolved_root,
                    "root_files",
                    root_assets,
                )
            )

        return archive_specs, asset_specs

    def _custom_nodes_archive_specs(self, custom_nodes_dir: Path) -> list[_CustomNodesArchiveSpec]:
        """Return code-only archive specs for compatibility with existing callers."""
        archive_specs, _ = self._custom_nodes_bundle_specs(custom_nodes_dir)
        return archive_specs

    def _partition_custom_node_files(
        self,
        files: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        """Partition custom-node files into code resources and mounted model assets."""
        code_files: list[Path] = []
        asset_files: list[Path] = []
        for file_path in files:
            if file_path.suffix.lower() in _CUSTOM_NODE_ASSET_SUFFIXES:
                asset_files.append(file_path)
            else:
                code_files.append(file_path)
        return code_files, asset_files

    def _custom_node_asset_specs(
        self,
        custom_nodes_root: Path,
        entry_name: str,
        asset_files: list[Path],
    ) -> list[_CustomNodeAssetSpec]:
        """Build deterministic metadata for package-owned model assets."""
        return [
            _CustomNodeAssetSpec(
                entry_name=entry_name,
                local_path=asset_path,
                relative_path=asset_path.relative_to(custom_nodes_root).as_posix(),
                sha256=self._host._hash_file(asset_path),
                size_bytes=asset_path.stat().st_size,
            )
            for asset_path in asset_files
        ]
