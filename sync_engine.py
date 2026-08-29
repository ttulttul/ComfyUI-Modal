"""Asset synchronization helpers for Modal-backed execution."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__:
    from . import sync_backends as _sync_backends
    from .huggingface_assets import HuggingFaceAssetRegistry, HuggingFaceAssetSource
    from .huggingface_discovery import HuggingFaceAssetDiscovery
    from .r2_cache import (
        R2CacheClient,
        R2CacheError,
        R2DownloadRequest,
        R2UploadPlan,
        R2UploadResult,
        R2WorkerPreflightRequest,
    )
    from .settings import ModalSyncSettings, get_settings
    from .sync_backends import (
        LocalFileSyncIndex,
        LocalMirrorVolume,
        ModalDictSyncIndex,
        ModalVolumeBackend,
        _ModalSdkCaller,
        _modal_volume_worker_count,
    )
    from .sync_custom_nodes import CustomNodesSynchronizer, MODEL_FILE_EXTENSIONS
    from .sync_hashing import SyncHasher, _SKIP_DIRS, _SKIP_FILE_SUFFIXES
    from .sync_protocols import (
        AssetSyncRequestCache,
        CancellationCheck,
        CancellableHuggingFaceMaterializingBackend,
        CancellableR2WriteBackBackend,
        CancellableVolumeBackend,
        HuggingFaceMaterializingBackend,
        R2MaterializingBackend,
        R2WorkerPreflightBackend,
        SyncCancelledError,
        SyncIndexBackend,
        SyncedAsset,
        SyncStatusCallback,
        VolumeBackend,
        _ContentAddressedSyncResult,
        _ContentAddressedSyncSpec,
        _CustomNodeAssetSpec,
        _CustomNodeAssetSyncResult,
        _CustomNodesArchiveSpec,
        _CustomNodesArchiveSyncResult,
        _R2MaterializationOutcome,
        _format_indexed_asset_status,
    )
    from .sync_r2_transfer import (
        R2TransferManager,
        R2WriteBackCancelled,
        R2WriteBackCoordinator,
        _R2_WRITE_BACK_COORDINATOR,
        begin_r2_writeback_prompt,
        finish_r2_writeback_prompt,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    import sync_backends as _sync_backends
    from huggingface_assets import HuggingFaceAssetRegistry, HuggingFaceAssetSource
    from huggingface_discovery import HuggingFaceAssetDiscovery
    from r2_cache import (
        R2CacheClient,
        R2CacheError,
        R2DownloadRequest,
        R2UploadPlan,
        R2UploadResult,
        R2WorkerPreflightRequest,
    )
    from settings import ModalSyncSettings, get_settings
    from sync_backends import (
        LocalFileSyncIndex,
        LocalMirrorVolume,
        ModalDictSyncIndex,
        ModalVolumeBackend,
        _ModalSdkCaller,
        _modal_volume_worker_count,
    )
    from sync_custom_nodes import CustomNodesSynchronizer, MODEL_FILE_EXTENSIONS
    from sync_hashing import SyncHasher, _SKIP_DIRS, _SKIP_FILE_SUFFIXES
    from sync_protocols import (
        AssetSyncRequestCache,
        CancellationCheck,
        CancellableHuggingFaceMaterializingBackend,
        CancellableR2WriteBackBackend,
        CancellableVolumeBackend,
        HuggingFaceMaterializingBackend,
        R2MaterializingBackend,
        R2WorkerPreflightBackend,
        SyncCancelledError,
        SyncIndexBackend,
        SyncedAsset,
        SyncStatusCallback,
        VolumeBackend,
        _ContentAddressedSyncResult,
        _ContentAddressedSyncSpec,
        _CustomNodeAssetSpec,
        _CustomNodeAssetSyncResult,
        _CustomNodesArchiveSpec,
        _CustomNodesArchiveSyncResult,
        _R2MaterializationOutcome,
        _format_indexed_asset_status,
    )
    from sync_r2_transfer import (
        R2TransferManager,
        R2WriteBackCancelled,
        R2WriteBackCoordinator,
        _R2_WRITE_BACK_COORDINATOR,
        begin_r2_writeback_prompt,
        finish_r2_writeback_prompt,
    )

logger = logging.getLogger(__name__)
_SYNC_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".gguf", ".pt", ".vae"})
def resolve_model_path(
    value: str,
    *,
    comfyui_root: Path | None = None,
    extensions: frozenset[str] = _SYNC_EXTENSIONS,
) -> Path | None:
    """Resolve one prompt string to a local model file when possible."""
    path = Path(value).expanduser()
    if path.suffix.lower() not in extensions:
        return None
    if path.is_file():
        return path.absolute()
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
                return Path(full_path).absolute()

    if comfyui_root is not None:
        candidate = comfyui_root / value
        if candidate.is_file():
            return candidate.absolute()
    return None


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


@dataclass
class ModalAssetSyncEngine:
    """Content-addressable storage sync engine for files and custom nodes."""

    volume: VolumeBackend
    settings: ModalSyncSettings
    sync_index: SyncIndexBackend | None = None
    huggingface_asset_registry: HuggingFaceAssetRegistry | None = None
    huggingface_asset_discovery: HuggingFaceAssetDiscovery | None = None
    r2_cache: R2CacheClient | None = None
    cancellation_check: CancellationCheck | None = None
    r2_writeback_activity: Callable[[], AbstractContextManager[None]] | None = None
    _hasher: SyncHasher = field(init=False)
    _custom_nodes: CustomNodesSynchronizer = field(init=False)
    _r2_transfer: R2TransferManager = field(init=False)
    _path_resolution_cache: dict[str, str | None] = field(init=False, default_factory=dict)
    _sync_scope_prefix_cache: str | None = field(init=False, default=None)
    _sync_scope_prefix_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Load persistent metadata caches used to avoid repeated hashing work."""
        if self.sync_index is None:
            self.sync_index = LocalFileSyncIndex(self.settings.local_storage_root)
        self._hasher = SyncHasher(
            settings=self.settings,
            cancellation_check=self.cancellation_check,
        )
        self._custom_nodes = CustomNodesSynchronizer(self)
        self._r2_transfer = R2TransferManager(
            self,
            volume=self.volume,
            r2_cache=self.r2_cache,
            cancellation_check=self.cancellation_check,
            r2_writeback_activity=self.r2_writeback_activity,
        )

    def _hash_file(self, path: Path) -> str:
        """Return the cached content digest for one file."""
        return self._hasher._hash_file(path)

    def _hash_directory(self, path: Path) -> str:
        """Return the cached stable digest for one directory tree."""
        return self._hasher._hash_directory(path)

    def _hash_file_group(self, root: Path, files: list[Path]) -> str:
        """Return a stable digest for a selected file group."""
        return self._hasher._hash_file_group(root, files)

    def _iter_files(self, path: Path) -> list[Path]:
        """Return syncable files below a directory."""
        return self._hasher._iter_files(path)

    def _raise_if_cancelled(self) -> None:
        """Raise when queue-time synchronization has been cancelled."""
        self._hasher._raise_if_cancelled()

    def _destination_label(self) -> str:
        """Return a user-facing destination name for sync progress messages."""
        if self.settings.execution_mode == "ssh_docker":
            return "the self-hosted worker"
        if self.settings.execution_mode == "vast":
            return "the Vast.ai instance"
        return "Modal"

    def preflight_r2_access(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> None:
        """Validate effective worker-side R2 access before transfer."""
        self._r2_transfer.preflight_r2_access(status_callback=status_callback)

    def _materialize_r2_source(
        self,
        spec: _ContentAddressedSyncSpec,
        size_bytes: int,
    ) -> _R2MaterializationOutcome:
        """Delegate worker-side R2 read-through materialization."""
        return self._r2_transfer._materialize_r2_source(spec, size_bytes)

    def _schedule_r2_writeback(
        self,
        *,
        sha256: str,
        size_bytes: int,
        remote_path: str,
        force: bool = False,
    ) -> None:
        """Schedule one idle-gated R2 cache population job."""
        self._r2_transfer._schedule_r2_writeback(
            sha256=sha256,
            size_bytes=size_bytes,
            remote_path=remote_path,
            force=force,
        )

    def _r2_writeback_job_prefix(self) -> tuple[str, str, str]:
        """Return the write-back deduplication namespace."""
        return self._r2_transfer._r2_writeback_job_prefix()

    def _r2_writeback_enabled(self) -> bool:
        """Return whether R2 cache population is configured."""
        return self._r2_transfer._r2_writeback_enabled()

    def wait_for_r2_writebacks(self) -> None:
        """Wait for this process's queued R2 write-backs to finish."""
        self._r2_transfer.wait_for_r2_writebacks()


    @classmethod
    def from_environment(cls, settings: ModalSyncSettings | None = None) -> "ModalAssetSyncEngine":
        """Create a sync engine using the local mirror backend by default."""
        resolved_settings = settings or get_settings()
        backend: VolumeBackend
        sync_index: SyncIndexBackend
        if resolved_settings.execution_mode == "remote" and _sync_backends.modal is not None:
            logger.info(
                "Using Modal volume backend %s and Modal sync index %s for remote asset sync.",
                resolved_settings.volume_name,
                resolved_settings.sync_index_dict_name,
            )
            backend = ModalVolumeBackend(resolved_settings.volume_name)
            sync_index = ModalDictSyncIndex(resolved_settings.sync_index_dict_name)
        else:
            if resolved_settings.execution_mode == "remote" and _sync_backends.modal is None:
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
        self._raise_if_cancelled()
        source_path = local_path.expanduser().absolute()
        if not source_path.is_file():
            raise FileNotFoundError(f"Asset not found: {source_path}")
        resolved_path = source_path.resolve()

        sha256 = self._hash_file(resolved_path)
        proposed_remote_path = f"{remote_folder.rstrip('/')}/{sha256}_{source_path.name}"
        sync_key = self._asset_sync_index_key(sha256)
        huggingface_source = (
            self._huggingface_source_for_asset(
                source_path,
                sha256=sha256,
                status_callback=status_callback,
                item_index=item_index,
                total_items=total_items,
            )
            if self._lookup_sync_record(sync_key) is None
            else None
        )
        sync_result = self._sync_content_addressed_file(
            local_path=source_path,
            remote_path=proposed_remote_path,
            sha256=sha256,
            sync_key=sync_key,
            source_description=str(source_path),
            status_callback=status_callback,
            upload_status_message=_format_indexed_asset_status(
                source_path.name,
                action="Uploading asset",
                location=f"to {self._destination_label()}",
                item_index=item_index,
                total_items=total_items,
            ),
            status_current=item_index,
            status_total=total_items,
            huggingface_source=huggingface_source,
        )

        return SyncedAsset(
            local_path=source_path,
            remote_path=sync_result.remote_path,
            sha256=sha256,
            uploaded=sync_result.uploaded,
        )

    def sync_prompt_inputs(
        self,
        inputs: dict[str, Any],
        *,
        status_callback: SyncStatusCallback | None = None,
        request_cache: AssetSyncRequestCache | None = None,
    ) -> tuple[dict[str, Any], list[SyncedAsset]]:
        """Rewrite file-like prompt inputs to mirrored storage paths."""
        self._raise_if_cancelled()
        synced_assets: list[SyncedAsset] = []
        sync_started_at = time.perf_counter()
        logger.info("Scanning prompt inputs for syncable assets.")
        syncable_asset_paths = self._collect_syncable_asset_paths(inputs)
        syncable_asset_index = 0

        def rewrite(value: Any) -> Any:
            nonlocal syncable_asset_index
            self._raise_if_cancelled()
            if isinstance(value, str):
                maybe_path = self._resolve_model_path(value)
                if maybe_path is not None:
                    cached_asset = request_cache.get(maybe_path) if request_cache is not None else None
                    if cached_asset is not None:
                        synced_assets.append(cached_asset)
                        return cached_asset.remote_path
                    if request_cache is None:
                        syncable_asset_index += 1
                        item_index = syncable_asset_index
                        total_items = len(syncable_asset_paths)
                    else:
                        item_index, total_items = request_cache.progress(maybe_path)
                    synced_asset = self.sync_file(
                        maybe_path,
                        status_callback=status_callback,
                        item_index=item_index,
                        total_items=total_items,
                    )
                    if request_cache is not None:
                        request_cache.put(synced_asset)
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

    def create_request_asset_cache(
        self,
        prompt_input_values: Iterable[Any],
    ) -> AssetSyncRequestCache:
        """Plan unique syncable assets for one queued prompt in stable encounter order."""
        unique_paths: dict[Path, Path] = {}
        for prompt_input_value in prompt_input_values:
            for local_path in self._collect_syncable_asset_paths(prompt_input_value):
                unique_paths.setdefault(local_path.resolve(), local_path.absolute())
        return AssetSyncRequestCache(planned_paths=tuple(unique_paths.values()))





    def sync_custom_nodes_directory(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> SyncedAsset | None:
        """Synchronize custom nodes through the dedicated collaborator."""
        return self._custom_nodes.sync_custom_nodes_directory(
            status_callback=status_callback
        )

    def _sync_custom_nodes_directory_uncached(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
        revisit_indexed_for_r2: bool = False,
        writeback_cancellation_check: CancellationCheck | None = None,
    ) -> SyncedAsset | None:
        """Delegate an uncached custom-node synchronization pass."""
        return self._custom_nodes._sync_custom_nodes_directory_uncached(
            status_callback=status_callback,
            revisit_indexed_for_r2=revisit_indexed_for_r2,
            writeback_cancellation_check=writeback_cancellation_check,
        )

    def _custom_nodes_archive_specs(
        self,
        custom_nodes_dir: Path,
    ) -> list[_CustomNodesArchiveSpec]:
        """Return deterministic custom-node archive specifications."""
        return self._custom_nodes._custom_nodes_archive_specs(custom_nodes_dir)

    def _cached_custom_nodes_archive_path(
        self,
        entry_name: str,
        entry_hash: str,
    ) -> Path:
        """Return the cached archive path for one custom-node entry."""
        return self._custom_nodes._cached_custom_nodes_archive_path(
            entry_name,
            entry_hash,
        )

    def _custom_nodes_manifest_remote_path(self, directory_hash: str) -> str:
        """Return the remote custom-node manifest path."""
        return self._custom_nodes._custom_nodes_manifest_remote_path(directory_hash)

    def _custom_nodes_manifest_sync_index_key(self, directory_hash: str) -> str:
        """Return the scoped custom-node manifest index key."""
        return self._custom_nodes._custom_nodes_manifest_sync_index_key(directory_hash)

    def _custom_nodes_archive_remote_path(
        self,
        entry_name: str,
        entry_hash: str,
    ) -> str:
        """Return the remote path for one custom-node archive."""
        return self._custom_nodes._custom_nodes_archive_remote_path(
            entry_name,
            entry_hash,
        )

    def _create_archive_from_files(
        self,
        root_path: Path,
        files: list[Path],
        archive_path: Path,
    ) -> Path:
        """Create one deterministic custom-node archive."""
        return self._custom_nodes._create_archive_from_files(
            root_path,
            files,
            archive_path,
        )

    def _sync_custom_nodes_archive_spec(
        self,
        custom_nodes_dir: Path,
        spec: _CustomNodesArchiveSpec,
    ) -> _CustomNodesArchiveSyncResult:
        """Synchronize one custom-node archive specification."""
        return self._custom_nodes._sync_custom_nodes_archive_spec(
            custom_nodes_dir,
            spec,
        )

    def _sync_content_addressed_file(
        self,
        *,
        local_path: Path,
        remote_path: str,
        sha256: str,
        sync_key: str,
        source_description: str,
        status_callback: SyncStatusCallback | None = None,
        upload_status_message: str | None = None,
        status_current: int | None = None,
        status_total: int | None = None,
        huggingface_source: HuggingFaceAssetSource | None = None,
    ) -> _ContentAddressedSyncResult:
        """Materialize one deterministic file from the fastest available source."""
        self._raise_if_cancelled()
        spec = _ContentAddressedSyncSpec(
            local_path=local_path,
            remote_path=remote_path,
            sha256=sha256,
            sync_key=sync_key,
            source_description=source_description,
            status_callback=status_callback,
            upload_status_message=upload_status_message,
            status_current=status_current,
            status_total=status_total,
            huggingface_source=huggingface_source,
        )
        size_bytes = spec.local_path.stat().st_size
        existing_record = self._lookup_sync_record(spec.sync_key)
        if existing_record is not None:
            indexed_remote_path = str(existing_record["remote_path"])
            self._schedule_r2_writeback(
                sha256=spec.sha256,
                size_bytes=size_bytes,
                remote_path=indexed_remote_path,
            )
            logger.info(
                "Reusing mirrored asset at %s because sync index key %s already exists.",
                indexed_remote_path,
                spec.sync_key,
            )
            return _ContentAddressedSyncResult(
                remote_path=indexed_remote_path,
                uploaded=False,
            )

        adopted = self._adopt_existing_remote(spec, size_bytes)
        if adopted is not None:
            return adopted
        logger.info("Syncing %s to %s", spec.source_description, spec.remote_path)
        huggingface_result = self._materialize_huggingface_source(spec, size_bytes)
        if huggingface_result is not None:
            return huggingface_result
        r2_outcome = self._materialize_r2_source(spec, size_bytes)
        if r2_outcome.result is not None:
            return r2_outcome.result
        return self._upload_content_addressed_file(
            spec,
            size_bytes,
            force_r2_writeback=r2_outcome.refresh_required,
        )

    def _adopt_existing_remote(
        self,
        spec: _ContentAddressedSyncSpec,
        size_bytes: int,
    ) -> _ContentAddressedSyncResult | None:
        """Repair a lost local index from an existing persistent remote path."""
        if not self.volume.exists(spec.remote_path):
            return None
        self._store_sync_record(
            sync_key=spec.sync_key,
            remote_path=spec.remote_path,
            source_description=spec.source_description,
        )
        self._schedule_r2_writeback(
            sha256=spec.sha256,
            size_bytes=size_bytes,
            remote_path=spec.remote_path,
        )
        logger.info("Adopted existing content-addressed remote file at %s.", spec.remote_path)
        return _ContentAddressedSyncResult(remote_path=spec.remote_path, uploaded=False)

    def _materialize_huggingface_source(
        self,
        spec: _ContentAddressedSyncSpec,
        size_bytes: int,
    ) -> _ContentAddressedSyncResult | None:
        """Try an authoritative Hugging Face source before shared cache lookup."""
        source = spec.huggingface_source
        if source is None or not isinstance(self.volume, HuggingFaceMaterializingBackend):
            return None
        _emit_sync_status(
            spec.status_callback,
            _format_indexed_asset_status(
                spec.local_path.name,
                action="Downloading asset",
                location="from Hugging Face on Vast.ai",
                item_index=spec.status_current,
                total_items=spec.status_total,
            ),
            spec.status_current,
            spec.status_total,
        )
        if not self._invoke_huggingface_materializer(source, spec.remote_path):
            return None
        self._store_sync_record(
            sync_key=spec.sync_key,
            remote_path=spec.remote_path,
            source_description=source.display_reference,
        )
        self._schedule_r2_writeback(
            sha256=spec.sha256,
            size_bytes=size_bytes,
            remote_path=spec.remote_path,
        )
        logger.info(
            "Materialized %s directly from Hugging Face at %s.",
            source.display_reference,
            spec.remote_path,
        )
        return _ContentAddressedSyncResult(remote_path=spec.remote_path, uploaded=True)

    def _invoke_huggingface_materializer(
        self,
        source: HuggingFaceAssetSource,
        remote_path: str,
    ) -> bool:
        """Invoke the available Hub materializer and normalize cancellation."""
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if self.cancellation_check is None or not isinstance(
            self.volume,
            CancellableHuggingFaceMaterializingBackend,
        ):
            assert isinstance(self.volume, HuggingFaceMaterializingBackend)
            return self.volume.materialize_huggingface_file(
                source,
                remote_path,
                token=token,
            )
        try:
            return self.volume.materialize_huggingface_file_cancellable(
                source,
                remote_path,
                token=token,
                cancellation_check=self.cancellation_check,
            )
        except InterruptedError as exc:
            raise SyncCancelledError("Remote workflow preparation was cancelled.") from exc


    def _upload_content_addressed_file(
        self,
        spec: _ContentAddressedSyncSpec,
        size_bytes: int,
        *,
        force_r2_writeback: bool = False,
    ) -> _ContentAddressedSyncResult:
        """Use the established controller-to-worker upload and schedule write-back."""
        _emit_sync_status(
            spec.status_callback,
            spec.upload_status_message
            or (
                f"Uploading {Path(spec.source_description).name} to "
                f"{self._destination_label()}"
            ),
            spec.status_current,
            spec.status_total,
        )
        if self.cancellation_check is not None and isinstance(
            self.volume,
            CancellableVolumeBackend,
        ):
            try:
                self.volume.put_file_cancellable(
                    spec.local_path,
                    spec.remote_path,
                    cancellation_check=self.cancellation_check,
                )
            except InterruptedError as exc:
                raise SyncCancelledError(
                    "Remote workflow preparation was cancelled."
                ) from exc
        else:
            self.volume.put_file(spec.local_path, spec.remote_path)
        self._store_sync_record(
            sync_key=spec.sync_key,
            remote_path=spec.remote_path,
            source_description=spec.source_description,
        )
        self._schedule_r2_writeback(
            sha256=spec.sha256,
            size_bytes=size_bytes,
            remote_path=spec.remote_path,
            force=force_r2_writeback,
        )
        return _ContentAddressedSyncResult(remote_path=spec.remote_path, uploaded=True)






    def _huggingface_source_for_asset(
        self,
        local_path: Path,
        *,
        sha256: str,
        status_callback: SyncStatusCallback | None,
        item_index: int | None,
        total_items: int | None,
    ) -> HuggingFaceAssetSource | None:
        """Return registered or automatically discovered immutable provenance."""
        if not isinstance(self.volume, HuggingFaceMaterializingBackend):
            return None
        source = (
            self.huggingface_asset_registry.get(sha256)
            if self.huggingface_asset_registry is not None
            else None
        )
        if source is None and self.huggingface_asset_discovery is not None:
            _emit_sync_status(
                status_callback,
                _format_indexed_asset_status(
                    local_path.name,
                    action="Identifying Hugging Face source",
                    location="for Vast.ai",
                    item_index=item_index,
                    total_items=total_items,
                ),
                item_index,
                total_items,
            )
            source = self.huggingface_asset_discovery.discover(
                local_path,
                sha256=sha256,
            )
        if source is None:
            return None
        actual_size = local_path.stat().st_size
        if actual_size != source.size_bytes:
            logger.warning(
                "Ignoring Hugging Face provenance for %s because registered size %d "
                "does not match local size %d.",
                local_path,
                source.size_bytes,
                actual_size,
            )
            return None
        return source





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
        return f"{self._sync_index_scope_prefix()}:asset:{sha256}"




    def _sync_index_scope_prefix(self) -> str:
        """Return the active sync-index scope prefix for this storage backend."""
        cached_prefix = self._sync_scope_prefix_cache
        if cached_prefix is not None:
            return cached_prefix
        with self._sync_scope_prefix_lock:
            cached_prefix = self._sync_scope_prefix_cache
            if cached_prefix is not None:
                return cached_prefix
            if isinstance(self.volume, ModalVolumeBackend) or bool(
                getattr(self.volume, "remote_volume_epoch_scoped", False)
            ):
                cached_prefix = self._ensure_remote_volume_epoch_scope()
            else:
                cached_prefix = f"local:{self.settings.local_storage_root.resolve()}"
            self._sync_scope_prefix_cache = cached_prefix
            return cached_prefix

    def _ensure_remote_volume_epoch_scope(self) -> str:
        """Return a sync-index prefix tied to the active remote volume contents."""
        fixed_key = f"{self.settings.volume_name}:current_volume_epoch"
        current_record = self._lookup_sync_record(fixed_key)
        if current_record is not None:
            epoch = current_record.get("epoch")
            sentinel_path = current_record.get("sentinel_path")
            if (
                isinstance(epoch, str)
                and epoch
                and isinstance(sentinel_path, str)
                and sentinel_path
                and self.volume.exists(sentinel_path)
            ):
                return f"{self.settings.volume_name}:epoch:{epoch}"
            logger.warning(
                "Discarding stale remote sync-index volume epoch for volume=%s because sentinel %s is missing.",
                self.settings.volume_name,
                sentinel_path,
            )

        epoch = uuid.uuid4().hex
        sentinel_path = f"/sync_index_epochs/{epoch}.json"
        self.volume.put_bytes(
            json.dumps(
                {
                    "volume_name": self.settings.volume_name,
                    "epoch": epoch,
                },
                sort_keys=True,
            ).encode("utf-8"),
            sentinel_path,
        )
        assert self.sync_index is not None
        self.sync_index.put(
            fixed_key,
            {
                "epoch": epoch,
                "remote_path": sentinel_path,
                "sentinel_path": sentinel_path,
            },
        )
        logger.info(
            "Initialized remote sync-index volume epoch %s for volume=%s sentinel=%s.",
            epoch,
            self.settings.volume_name,
            sentinel_path,
        )
        return f"{self.settings.volume_name}:epoch:{epoch}"


    def _resolve_model_path(self, value: str) -> Path | None:
        """Resolve a prompt string into a local model file path when possible."""
        if value in self._path_resolution_cache:
            cached = self._path_resolution_cache[value]
            return Path(cached) if cached is not None else None
        resolved = resolve_model_path(
            value,
            comfyui_root=self.settings.comfyui_root,
        )
        self._path_resolution_cache[value] = (
            str(resolved) if resolved is not None else None
        )
        return resolved

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
