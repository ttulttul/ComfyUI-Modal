"""Tests for asset syncing and custom_nodes archiving."""

from __future__ import annotations

import hashlib

import json

from dataclasses import replace

from pathlib import Path

import threading

import time

import types

from typing import Any

import zipfile

import pytest

def _r2_sync_settings(settings_module: Any, tmp_path: Path) -> Any:
    """Return minimal Vast-mode settings for R2 sync behavior tests."""
    return settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="vast",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )

class _R2BackfillVolume:
    """Retain worker paths and record uploads made through signed R2 plans."""

    def __init__(self, r2_cache_module: Any) -> None:
        """Initialize empty worker storage and upload history."""
        self.r2_cache_module = r2_cache_module
        self.paths: set[str] = set()
        self.worker_puts: list[str] = []
        self.r2_uploads: list[tuple[Any, str]] = []

    def exists(self, remote_path: str) -> bool:
        """Return whether ordinary sync previously published the worker path."""
        return remote_path in self.paths

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Record one ordinary controller-to-worker file upload."""
        assert local_path.is_file()
        self.paths.add(remote_path)
        self.worker_puts.append(remote_path)

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Record one ordinary controller-to-worker byte upload."""
        del payload
        self.paths.add(remote_path)
        self.worker_puts.append(remote_path)

    def materialize_r2_file(self, *args: Any, **kwargs: Any) -> None:
        """Reject downloads because backfill tests begin with an empty R2 cache."""
        del args, kwargs
        raise AssertionError("backfill should not download from R2")

    def upload_r2_file(self, plan: Any, remote_path: str) -> Any:
        """Record one worker-to-R2 upload and return successful part metadata."""
        self.r2_uploads.append((plan, remote_path))
        return self.r2_cache_module.R2UploadResult()

class _SynchronousBackfillCache:
    """Plan deterministic synchronous uploads for missing R2 objects."""

    write_back_mode = "sync"

    def __init__(self, r2_cache_module: Any) -> None:
        """Retain the R2 data models and every requested upload identity."""
        self.r2_cache_module = r2_cache_module
        self.requests: list[tuple[str, int, bool]] = []

    def prepare_upload(
        self,
        digest: str,
        size_bytes: int,
        *,
        force: bool = False,
    ) -> Any:
        """Return one single-part plan and record its immutable identity."""
        self.requests.append((digest, size_bytes, force))
        return self.r2_cache_module.R2UploadPlan(
            key=f"cache/{digest}",
            sha256=digest,
            size_bytes=size_bytes,
            allowed_host="account.r2.cloudflarestorage.com",
            mode="single",
            urls=("https://account.r2.cloudflarestorage.com/object?secret=1",),
        )

    def complete_upload(self, plan: Any, result: Any) -> None:
        """Validate successful completion of one planned upload."""
        assert plan.sha256
        assert result == self.r2_cache_module.R2UploadResult()

    def abort_upload(self, plan: Any) -> None:
        """Reject aborts because backfill test uploads must succeed."""
        del plan
        raise AssertionError("successful backfill should not abort")


__all__ = tuple(name for name in globals() if not name.startswith("__"))

