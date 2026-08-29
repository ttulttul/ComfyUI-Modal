"""Protocols and value objects shared by asset synchronization components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

if __package__:
    from .huggingface_assets import HuggingFaceAssetSource
    from .r2_cache import (
        R2DownloadRequest,
        R2UploadPlan,
        R2UploadResult,
        R2WorkerPreflightRequest,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from huggingface_assets import HuggingFaceAssetSource
    from r2_cache import (
        R2DownloadRequest,
        R2UploadPlan,
        R2UploadResult,
        R2WorkerPreflightRequest,
    )

SyncStatusCallback = Callable[[str, int | None, int | None], None]
CancellationCheck = Callable[[], bool]


def _format_indexed_asset_status(
    asset_name: str,
    *,
    action: str,
    location: str,
    item_index: int | None,
    total_items: int | None,
) -> str:
    """Return one asset status with optional request-wide item progress."""
    if item_index is not None and total_items is not None and total_items > 1:
        return f"{action} {item_index}/{total_items} {location}: {asset_name}"
    return f"{action} {location}: {asset_name}"


class VolumeBackend(Protocol):
    """Minimal storage interface needed by the sync engine."""

    def exists(self, remote_path: str) -> bool:
        """Return whether the remote path already exists."""

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the remote storage backend."""

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload raw bytes into the remote storage backend."""


@runtime_checkable
class HuggingFaceMaterializingBackend(Protocol):
    """Optional backend capability for direct verified Hugging Face acquisition."""

    def materialize_huggingface_file(
        self,
        source: HuggingFaceAssetSource,
        remote_path: str,
        *,
        token: str | None,
    ) -> bool:
        """Materialize one immutable file and report whether it succeeded."""


@runtime_checkable
class CancellableVolumeBackend(Protocol):
    """Optional backend capability for interruptible file uploads."""

    def put_file_cancellable(
        self,
        local_path: Path,
        remote_path: str,
        *,
        cancellation_check: CancellationCheck,
    ) -> None:
        """Upload a file while cooperatively observing cancellation."""


@runtime_checkable
class CancellableHuggingFaceMaterializingBackend(Protocol):
    """Optional backend capability for interruptible remote downloads."""

    def materialize_huggingface_file_cancellable(
        self,
        source: HuggingFaceAssetSource,
        remote_path: str,
        *,
        token: str | None,
        cancellation_check: CancellationCheck,
    ) -> bool:
        """Materialize one immutable file while observing cancellation."""


@runtime_checkable
class R2MaterializingBackend(Protocol):
    """Optional backend capability for signed R2 downloads and write-back."""

    def materialize_r2_file(
        self,
        request: R2DownloadRequest,
        remote_path: str,
        *,
        cancellation_check: CancellationCheck | None = None,
    ) -> None:
        """Download and verify one content-addressed R2 object."""

    def upload_r2_file(
        self,
        plan: R2UploadPlan,
        remote_path: str,
    ) -> R2UploadResult:
        """Upload one remote file through a controller-issued signed plan."""


@runtime_checkable
class CancellableR2WriteBackBackend(Protocol):
    """Optional backend capability for preemptible background R2 uploads."""

    def upload_r2_file_cancellable(
        self,
        plan: R2UploadPlan,
        remote_path: str,
        *,
        cancellation_check: CancellationCheck,
    ) -> R2UploadResult:
        """Upload one remote file while yielding promptly to workflow activity."""


@runtime_checkable
class R2WorkerPreflightBackend(Protocol):
    """Optional backend capability for testing effective worker-side R2 access."""

    def preflight_r2_access(
        self,
        request: R2WorkerPreflightRequest,
        *,
        cancellation_check: CancellationCheck | None = None,
    ) -> None:
        """Verify one read-only presigned request from the worker environment."""


class SyncCancelledError(RuntimeError):
    """Raised when queue-time asset preparation is cancelled."""


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


@dataclass
class AssetSyncRequestCache:
    """Deduplicate asset hashing, index lookup, and upload within one queued prompt."""

    planned_paths: tuple[Path, ...]
    _assets_by_path: dict[Path, SyncedAsset] = field(default_factory=dict)
    _positions_by_path: dict[Path, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Index stable request progress positions once."""
        self._positions_by_path = {
            path.resolve(): index
            for index, path in enumerate(self.planned_paths, start=1)
        }

    def get(self, local_path: Path) -> SyncedAsset | None:
        """Return a previously synced asset for this request when available."""
        return self._assets_by_path.get(local_path.resolve())

    def put(self, synced_asset: SyncedAsset) -> None:
        """Remember one request-scoped sync result by its resolved local path."""
        self._assets_by_path[synced_asset.local_path.resolve()] = synced_asset

    def progress(self, local_path: Path) -> tuple[int, int]:
        """Return the stable one-based progress position for a planned asset."""
        resolved_path = local_path.resolve()
        return self._positions_by_path[resolved_path], len(self.planned_paths)

    def synced_assets(self) -> tuple[SyncedAsset, ...]:
        """Return unique sync results in request planning order."""
        return tuple(
            self._assets_by_path[path.resolve()]
            for path in self.planned_paths
            if path.resolve() in self._assets_by_path
        )


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
class _CustomNodeAssetSpec:
    """Describe one package-owned model asset excluded from a code archive."""

    entry_name: str
    local_path: Path
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _CustomNodeAssetSyncResult:
    """Describe one content-addressed package asset and its upload result."""

    entry_name: str
    relative_path: str
    sha256: str
    size_bytes: int
    remote_path: str
    uploaded: bool


@dataclass(frozen=True)
class _ContentAddressedSyncResult:
    """Describe the outcome of one content-addressed file sync decision."""

    remote_path: str
    uploaded: bool


@dataclass(frozen=True)
class _ContentAddressedSyncSpec:
    """Collect one file's immutable identity, destination, and progress context."""

    local_path: Path
    remote_path: str
    sha256: str
    sync_key: str
    source_description: str
    status_callback: SyncStatusCallback | None
    upload_status_message: str | None
    status_current: int | None
    status_total: int | None
    huggingface_source: HuggingFaceAssetSource | None


@dataclass(frozen=True)
class _R2MaterializationOutcome:
    """Return an optional hit and whether failed cached bytes need replacement."""

    result: _ContentAddressedSyncResult | None
    refresh_required: bool = False
