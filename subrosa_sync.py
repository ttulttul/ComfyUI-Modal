"""Milestone no-op asset synchronization for Subrosa mock-worker execution."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

if __package__:
    from .settings import ModalSyncSettings
    from .sync_engine import ModalAssetSyncEngine
    from .sync_protocols import (
        AssetSyncRequestCache,
        CancellationCheck,
        SyncedAsset,
        SyncStatusCallback,
    )
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from settings import ModalSyncSettings
    from sync_engine import ModalAssetSyncEngine
    from sync_protocols import (
        AssetSyncRequestCache,
        CancellationCheck,
        SyncedAsset,
        SyncStatusCallback,
    )

logger = logging.getLogger(__name__)


class SubrosaNoOpVolumeBackend:
    """Pretend assets exist while the relay-backed R2 implementation is pending."""

    def exists(self, remote_path: str) -> bool:
        """Report every path present so the sync engine never uploads it."""
        del remote_path
        return True

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Skip file uploads during the mock-worker milestone."""
        logger.warning(
            "Skipping Subrosa asset upload during mock milestone local=%s remote=%s.",
            local_path,
            remote_path,
        )

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Skip byte uploads during the mock-worker milestone."""
        logger.warning(
            "Skipping Subrosa byte upload during mock milestone bytes=%d remote=%s.",
            len(payload),
            remote_path,
        )


class SubrosaNoOpSyncEngine(ModalAssetSyncEngine):
    """Preserve prompt paths and skip all asset work for the current mock pool."""

    def preflight_r2_access(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> None:
        """Skip R2 preflight until the Subrosa-managed asset backend is wired."""
        del status_callback

    def sync_custom_nodes_directory(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> SyncedAsset | None:
        """Skip custom-node bundling for the relay mock worker."""
        del status_callback
        return None

    def create_request_asset_cache(
        self,
        prompt_input_values: Iterable[Any],
    ) -> AssetSyncRequestCache:
        """Return an empty request cache without scanning local model paths."""
        del prompt_input_values
        return AssetSyncRequestCache(planned_paths=())

    def sync_prompt_inputs(
        self,
        inputs: dict[str, Any],
        *,
        status_callback: SyncStatusCallback | None = None,
        request_cache: AssetSyncRequestCache | None = None,
    ) -> tuple[dict[str, Any], list[SyncedAsset]]:
        """Return prompt inputs unchanged and report no transferred assets."""
        del status_callback, request_cache
        return inputs, []


def subrosa_noop_sync_engine(
    settings: ModalSyncSettings,
    cancellation_check: CancellationCheck | None,
) -> ModalAssetSyncEngine:
    """Build the explicit milestone sync engine for one Subrosa assignment."""
    return SubrosaNoOpSyncEngine(
        volume=SubrosaNoOpVolumeBackend(),
        settings=settings,
        cancellation_check=cancellation_check,
    )


__all__ = [
    "SubrosaNoOpSyncEngine",
    "SubrosaNoOpVolumeBackend",
    "subrosa_noop_sync_engine",
]
