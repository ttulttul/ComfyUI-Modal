"""Tests for the sync r2 transfer boundary."""

from __future__ import annotations

from sync_engine_test_support import *  # noqa: F401,F403

def test_r2_worker_preflight_runs_before_transfer_and_reports_ip_filter_hint(
    settings_module: Any,
    sync_engine_module: Any,
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """A remote AccessDenied probe should become an actionable preparation failure."""
    events: list[str] = []

    class PreflightVolume:
        """Reject the read-only worker probe with a sanitized R2 diagnostic."""

        def exists(self, remote_path: str) -> bool:
            """Satisfy the storage protocol without accessing a worker path."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Reject asset mutation before a successful preflight."""
            del local_path, remote_path
            raise AssertionError("preflight must run before file transfer")

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Reject asset mutation before a successful preflight."""
            del payload, remote_path
            raise AssertionError("preflight must run before byte transfer")

        def preflight_r2_access(
            self,
            request: Any,
            *,
            cancellation_check: Any = None,
        ) -> None:
            """Record the protected request and report the worker policy denial."""
            del cancellation_check
            events.append(request.allowed_host)
            raise RuntimeError(
                "R2 materializer failed safely "
                "(category=http_client status=403 r2_code=AccessDenied)."
            )

    class PreflightCache:
        """Return one controller-approved read-only worker request."""

        write_back_mode = "off"

        def worker_preflight_request(self) -> Any:
            """Record controller validation before returning the signed probe."""
            events.append("controller")
            return r2_cache_module.R2WorkerPreflightRequest(
                url="https://account.r2.cloudflarestorage.com/missing?secret=1",
                allowed_host="account.r2.cloudflarestorage.com",
            )

    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=PreflightVolume(),
        settings=_r2_sync_settings(settings_module, tmp_path),
        r2_cache=PreflightCache(),
    )
    statuses: list[str] = []

    with pytest.raises(
        r2_cache_module.R2CacheError,
        match="Remove Client IP Address Filtering",
    ):
        engine.preflight_r2_access(
            status_callback=lambda message, _current, _total: statuses.append(message)
        )

    assert events == ["controller", "account.r2.cloudflarestorage.com"]
    assert statuses == [
        "Checking Cloudflare R2 access from the Vast.ai instance"
    ]

def test_r2_cache_hit_materializes_without_local_upload(
    settings_module: Any,
    sync_engine_module: Any,
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """An R2 hit should transfer directly to the worker and update its local index."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"r2-backed-model")
    sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    materializations: list[tuple[Any, str]] = []
    uploads: list[tuple[Path, str]] = []
    statuses: list[str] = []

    class R2Volume:
        """Record remote R2 and fallback transfer operations."""

        def exists(self, remote_path: str) -> bool:
            """Report an initially empty remote volume."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Record a fallback local upload."""
            uploads.append((local_path, remote_path))

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol-required byte uploads."""
            del payload, remote_path

        def materialize_r2_file(
            self,
            request: Any,
            remote_path: str,
            *,
            cancellation_check: Any = None,
        ) -> None:
            """Record one direct R2 materialization."""
            del cancellation_check
            materializations.append((request, remote_path))

        def upload_r2_file(self, plan: Any, remote_path: str) -> Any:
            """Reject write-back, which is disabled in this test."""
            del plan, remote_path
            raise AssertionError("write-back should be disabled")

    class R2Cache:
        """Return one deterministic cache hit."""

        write_back_mode = "off"

        def download_request(self, digest: str, size_bytes: int) -> Any:
            """Return a signed transfer request for the expected digest."""
            return r2_cache_module.R2DownloadRequest(
                url="https://account.r2.cloudflarestorage.com/object?secret=1",
                allowed_host="account.r2.cloudflarestorage.com",
                sha256=digest,
                size_bytes=size_bytes,
            )

    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=R2Volume(),
        settings=_r2_sync_settings(settings_module, tmp_path),
        r2_cache=R2Cache(),
    )

    result = engine.sync_file(
        asset_path,
        status_callback=lambda message, current, total: statuses.append(message),
    )

    assert result.sha256 == sha256
    assert result.uploaded is True
    assert len(materializations) == 1
    assert uploads == []
    assert statuses == ["Downloading asset from Cloudflare R2: model.safetensors"]

def test_local_upload_queues_r2_writeback_without_blocking(
    settings_module: Any,
    sync_engine_module: Any,
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """A cache miss should return before background R2 population completes."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"new-model")
    sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    remote_uploads: list[str] = []
    completed_plans: list[Any] = []
    writeback_started = threading.Event()
    allow_writeback = threading.Event()

    class R2Volume:
        """Record local worker upload and R2 write-back operations."""

        def exists(self, remote_path: str) -> bool:
            """Report empty worker storage."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Record the normal local-to-worker upload."""
            del local_path
            remote_uploads.append(remote_path)

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol-required byte uploads."""
            del payload, remote_path

        def materialize_r2_file(self, *args: Any, **kwargs: Any) -> None:
            """Reject a materialization on the forced cache miss."""
            del args, kwargs
            raise AssertionError("cache miss should not materialize")

        def upload_r2_file(self, plan: Any, remote_path: str) -> Any:
            """Return a successful single-part upload result."""
            assert remote_path == remote_uploads[0]
            writeback_started.set()
            assert allow_writeback.wait(timeout=2.0)
            return r2_cache_module.R2UploadResult()

    class R2Cache:
        """Expose one legacy sync setting that must migrate to background work."""

        write_back_mode = "sync"

        def download_request(self, digest: str, size_bytes: int) -> None:
            """Report an R2 miss."""
            del digest, size_bytes
            return None

        def prepare_upload(
            self,
            digest: str,
            size_bytes: int,
            *,
            force: bool = False,
        ) -> Any:
            """Return one single-part signed upload plan."""
            assert force is False
            return r2_cache_module.R2UploadPlan(
                key=f"cache/{digest}",
                sha256=digest,
                size_bytes=size_bytes,
                allowed_host="account.r2.cloudflarestorage.com",
                mode="single",
                urls=("https://account.r2.cloudflarestorage.com/object?secret=1",),
            )

        def complete_upload(self, plan: Any, result: Any) -> None:
            """Record successful controller completion."""
            assert result == r2_cache_module.R2UploadResult()
            completed_plans.append(plan)

        def abort_upload(self, plan: Any) -> None:
            """Reject an abort for a successful write-back."""
            del plan
            raise AssertionError("successful write-back should not abort")

    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=R2Volume(),
        settings=_r2_sync_settings(settings_module, tmp_path),
        r2_cache=R2Cache(),
    )

    result = engine.sync_file(asset_path)
    assert writeback_started.wait(timeout=2.0)
    assert completed_plans == []
    allow_writeback.set()
    engine.wait_for_r2_writebacks()

    assert result.sha256 == sha256
    assert len(remote_uploads) == 1
    assert len(completed_plans) == 1
    assert completed_plans[0].sha256 == sha256

def test_r2_writeback_coordinator_waits_for_foreground_prompt(
    sync_engine_module: Any,
) -> None:
    """Idle jobs must not start while a remote prompt holds foreground priority."""
    coordinator = sync_engine_module.R2WriteBackCoordinator(max_workers=1)
    started = threading.Event()
    completed = threading.Event()
    coordinator.begin_prompt("prompt-1")

    def run_writeback(cancellation_check: Any) -> None:
        """Record one coordinator-controlled cache operation."""
        assert cancellation_check() is False
        started.set()
        completed.set()

    assert coordinator.submit(("cache", "asset"), run_writeback) is True
    assert started.wait(timeout=0.05) is False

    coordinator.finish_prompt("prompt-1")

    assert completed.wait(timeout=2.0)
    assert coordinator.wait_for_idle(timeout_seconds=2.0) is True

def test_r2_writeback_coordinator_preempts_and_requeues_for_new_prompt(
    sync_engine_module: Any,
) -> None:
    """A newly preparing prompt should cancel and later resume active cache work."""
    coordinator = sync_engine_module.R2WriteBackCoordinator(max_workers=1)
    first_started = threading.Event()
    first_cancelled = threading.Event()
    completed = threading.Event()
    attempts = 0

    def run_writeback(cancellation_check: Any) -> None:
        """Yield the first attempt and complete the retried idle attempt."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_started.set()
            assert first_cancelled.wait(timeout=2.0)
            assert cancellation_check() is True
            raise sync_engine_module.R2WriteBackCancelled("foreground resumed")
        assert cancellation_check() is False
        completed.set()

    assert coordinator.submit(("cache", "asset"), run_writeback) is True
    assert first_started.wait(timeout=2.0)
    coordinator.begin_prompt("prompt-2")
    first_cancelled.set()
    assert completed.wait(timeout=0.05) is False

    coordinator.finish_prompt("prompt-2")

    assert completed.wait(timeout=2.0)
    assert attempts == 2
    assert coordinator.wait_for_idle(timeout_seconds=2.0) is True

def test_indexed_remote_payload_is_backfilled_to_r2(
    settings_module: Any,
    sync_engine_module: Any,
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """An indexed worker file should still populate a newly enabled R2 cache."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"previously-synced-model")
    sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    volume = _R2BackfillVolume(r2_cache_module)
    cache = _SynchronousBackfillCache(r2_cache_module)
    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=volume,
        settings=_r2_sync_settings(settings_module, tmp_path),
        r2_cache=cache,
    )
    indexed_remote_path = "/assets/indexed-model.safetensors"
    engine.sync_index.put(
        engine._asset_sync_index_key(sha256),
        {"remote_path": indexed_remote_path, "source": "previous sync"},
    )

    result = engine.sync_file(asset_path)
    engine.wait_for_r2_writebacks()

    assert result.uploaded is False
    assert result.remote_path == indexed_remote_path
    assert volume.worker_puts == []
    assert [(plan.sha256, path) for plan, path in volume.r2_uploads] == [
        (sha256, indexed_remote_path)
    ]
    assert cache.requests == [(sha256, asset_path.stat().st_size, False)]

def test_existing_remote_content_is_adopted_without_reupload(
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A lost local index should be repaired from a persistent remote volume."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"already-remote")

    class ExistingVolume:
        """Expose one already populated content-addressed path."""

        def exists(self, remote_path: str) -> bool:
            """Report the expected remote file as present."""
            del remote_path
            return True

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Reject an unnecessary duplicate upload."""
            del local_path, remote_path
            raise AssertionError("existing remote file should be adopted")

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol-required byte uploads."""
            del payload, remote_path

    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=ExistingVolume(),
        settings=_r2_sync_settings(settings_module, tmp_path),
    )

    first = engine.sync_file(asset_path)
    second = engine.sync_file(asset_path)

    assert first.uploaded is False
    assert second.uploaded is False
    assert first.remote_path == second.remote_path

def test_failed_r2_hit_is_replaced_after_fallback_upload(
    settings_module: Any,
    sync_engine_module: Any,
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """A same-size corrupt cache hit should trigger forced write-back after fallback."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"healthy-local-model")
    forced_values: list[bool] = []

    class CorruptR2Volume:
        """Fail cached materialization, then accept normal upload and replacement."""

        def exists(self, remote_path: str) -> bool:
            """Report empty worker storage."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Accept the established fallback transfer."""
            del local_path, remote_path

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol-required byte uploads."""
            del payload, remote_path

        def materialize_r2_file(self, *args: Any, **kwargs: Any) -> None:
            """Emulate a worker-side SHA mismatch."""
            del args, kwargs
            raise ValueError("R2 download digest mismatch")

        def upload_r2_file(self, plan: Any, remote_path: str) -> Any:
            """Accept the forced cache replacement."""
            del plan, remote_path
            return r2_cache_module.R2UploadResult()

    class CorruptR2Cache:
        """Return a hit and record whether replacement bypasses size reuse."""

        write_back_mode = "sync"

        def download_request(self, digest: str, size_bytes: int) -> Any:
            """Return one apparently valid same-size cache object."""
            return r2_cache_module.R2DownloadRequest(
                url="https://account.r2.cloudflarestorage.com/object?secret=1",
                allowed_host="account.r2.cloudflarestorage.com",
                sha256=digest,
                size_bytes=size_bytes,
            )

        def prepare_upload(
            self,
            digest: str,
            size_bytes: int,
            *,
            force: bool = False,
        ) -> Any:
            """Record and return one replacement plan."""
            forced_values.append(force)
            return r2_cache_module.R2UploadPlan(
                key=f"cache/{digest}",
                sha256=digest,
                size_bytes=size_bytes,
                allowed_host="account.r2.cloudflarestorage.com",
                mode="single",
                urls=("https://account.r2.cloudflarestorage.com/object?secret=2",),
            )

        def complete_upload(self, plan: Any, result: Any) -> None:
            """Accept successful forced replacement completion."""
            del plan, result

        def abort_upload(self, plan: Any) -> None:
            """Reject an abort in the successful path."""
            del plan
            raise AssertionError("replacement should not abort")

    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=CorruptR2Volume(),
        settings=_r2_sync_settings(settings_module, tmp_path),
        r2_cache=CorruptR2Cache(),
    )

    engine.sync_file(asset_path)
    engine.wait_for_r2_writebacks()

    assert forced_values == [True]

