"""Tests for the sync backends boundary."""

from __future__ import annotations

from sync_engine_test_support import *  # noqa: F401,F403

def test_remote_mode_uses_modal_volume_backend_when_sdk_is_available(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Remote execution mode should upload into a real Modal volume when the SDK is available."""

    class FakeBatch:
        """Capture files uploaded through Modal batch_upload."""

        def __init__(self, uploads: list[tuple[Any, str]]) -> None:
            """Store uploaded file references."""
            self.uploads = uploads

        def __enter__(self) -> "FakeBatch":
            """Return the active batch context."""
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            """Close the fake batch context."""
            return None

        def put_file(self, source: Any, remote_path: str) -> None:
            """Record the uploaded file or bytes payload."""
            self.uploads.append((source, remote_path))

    class FakeVolume:
        """Minimal Modal volume double."""

        def __init__(self) -> None:
            """Initialize the fake upload state."""
            self.paths: set[str] = set()
            self.uploads: list[tuple[Any, str]] = []

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Return a non-empty listing when the path has already been uploaded."""
            return [remote_path] if remote_path in self.paths else []

        def batch_upload(self) -> FakeBatch:
            """Return a fake batch uploader."""
            return FakeBatch(self.uploads)

    fake_volume = FakeVolume()

    class FakeModal:
        """Minimal modal SDK double that returns a stable volume handle."""

        class Dict:
            """Namespace for sync-index lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> dict[str, Any]:
                """Return a plain dict-backed sync index."""
                del name, create_if_missing
                return {}

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                return fake_volume

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="remote",
        sync_custom_nodes=True,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )

    engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    assert type(engine.volume).__name__ == "ModalVolumeBackend"
    assert type(engine.sync_index).__name__ == "ModalDictSyncIndex"

    asset_path = tmp_path / "encoder.safetensors"
    asset_path.write_bytes(b"weights")
    synced = engine.sync_file(asset_path)

    uploaded_remote_paths = [remote_path for _, remote_path in fake_volume.uploads]
    assert synced.remote_path in uploaded_remote_paths
    assert all(not remote_path.startswith("/hashes/") for remote_path in uploaded_remote_paths)

def test_modal_volume_backend_treats_missing_path_as_cache_miss(
    sync_engine_module: Any,
    monkeypatch: Any,
) -> None:
    """A missing Modal volume path should behave like a normal absent marker file."""

    class FakeNotFoundError(Exception):
        """Stand-in for modal.exception.NotFoundError."""

    class FakeVolume:
        """Minimal Modal volume double that raises on missing listdir."""

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Simulate Modal's missing-path behavior."""
            raise FakeNotFoundError(remote_path)

    class FakeModal:
        """Minimal modal SDK double exposing the exception namespace."""

        exception = type("FakeExceptionNamespace", (), {"NotFoundError": FakeNotFoundError})

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return a fake volume that always reports missing paths."""
                return FakeVolume()

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)
    backend = sync_engine_module.ModalVolumeBackend("volume")

    assert backend.exists("/hashes/missing.done") is False

def test_modal_volume_backend_caches_exists_results_and_uploaded_paths(
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Repeated existence checks for the same path should avoid repeated Modal metadata calls."""

    class FakeBatch:
        """Capture one uploaded path."""

        def __init__(self, volume: "FakeVolume") -> None:
            """Store the backing volume."""
            self.volume = volume

        def __enter__(self) -> "FakeBatch":
            """Return the active batch context."""
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            """Close the fake batch context."""
            return None

        def put_file(self, source: Any, remote_path: str) -> None:
            """Record the uploaded path as present."""
            self.volume.paths.add(remote_path)

    class FakeVolume:
        """Minimal Modal volume double with listdir accounting."""

        def __init__(self) -> None:
            """Initialize fake storage state."""
            self.paths: set[str] = set()
            self.listdir_calls = 0

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Return a listing for known paths while counting calls."""
            self.listdir_calls += 1
            return [remote_path] if remote_path in self.paths else []

        def batch_upload(self) -> FakeBatch:
            """Return a fake uploader."""
            return FakeBatch(self)

    fake_volume = FakeVolume()

    class FakeModal:
        """Minimal modal SDK double exposing a fake volume."""

        exception = type(
            "FakeExceptionNamespace",
            (),
            {
                "NotFoundError": FileNotFoundError,
            },
        )

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                return fake_volume

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)
    backend = sync_engine_module.ModalVolumeBackend("volume")

    assert backend.exists("/hashes/present.done") is False
    assert backend.exists("/hashes/present.done") is False
    assert fake_volume.listdir_calls == 1

    local_path = tmp_path / "marker.txt"
    local_path.write_text("marker", encoding="utf-8")
    backend.put_file(local_path, "/hashes/present.done")

    assert backend.exists("/hashes/present.done") is True
    assert fake_volume.listdir_calls == 1

def test_modal_volume_backend_ignores_file_exists_upload_race(
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Content-addressed Modal uploads should be idempotent when the path already exists remotely."""

    class FakeBatch:
        """Raise the same collision Modal raises when closing a duplicate upload batch."""

        def __enter__(self) -> "FakeBatch":
            """Return the active batch context."""
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            """Surface the duplicate remote file at batch close."""
            raise FileExistsError("/custom_nodes/entries/package/hash_custom_nodes_bundle.zip")

        def put_file(self, source: Any, remote_path: str) -> None:
            """Accept the upload request before the batch close detects the duplicate."""
            del source, remote_path

    class FakeVolume:
        """Minimal Modal volume double with duplicate-upload behavior."""

        def __init__(self) -> None:
            """Initialize fake storage state."""
            self.listdir_calls = 0

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Return a listing for known paths while counting calls."""
            del recursive
            self.listdir_calls += 1
            return [remote_path]

        def batch_upload(self) -> FakeBatch:
            """Return a fake uploader."""
            return FakeBatch()

    fake_volume = FakeVolume()

    class FakeModal:
        """Minimal modal SDK double exposing a duplicate-prone fake volume."""

        exception = type(
            "FakeExceptionNamespace",
            (),
            {
                "NotFoundError": FileNotFoundError,
            },
        )

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                del name, create_if_missing
                return fake_volume

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)
    backend = sync_engine_module.ModalVolumeBackend("volume")
    local_path = tmp_path / "bundle.zip"
    local_path.write_bytes(b"bundle")

    backend.put_file(local_path, "/custom_nodes/entries/package/hash_custom_nodes_bundle.zip")
    backend.put_bytes(b"{}", "/custom_nodes/manifests/hash_manifest.json")

    assert backend.exists("/custom_nodes/entries/package/hash_custom_nodes_bundle.zip") is True
    assert backend.exists("/custom_nodes/manifests/hash_manifest.json") is True
    assert fake_volume.listdir_calls == 0

def test_modal_volume_backend_retries_rate_limited_calls(
    sync_engine_module: Any,
    monkeypatch: Any,
) -> None:
    """Transient Modal rate limiting should back off and retry instead of failing immediately."""

    class FakeResourceExhaustedError(Exception):
        """Stand-in for modal.exception.ResourceExhaustedError."""

    class FakeVolume:
        """Minimal Modal volume double that rate limits once."""

        def __init__(self) -> None:
            """Initialize the listdir attempt counter."""
            self.listdir_calls = 0

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Raise once, then succeed."""
            self.listdir_calls += 1
            if self.listdir_calls == 1:
                raise FakeResourceExhaustedError("rate limited")
            return [remote_path]

    fake_volume = FakeVolume()

    class FakeModal:
        """Minimal modal SDK double exposing retryable error types."""

        exception = type(
            "FakeExceptionNamespace",
            (),
            {
                "NotFoundError": FileNotFoundError,
                "ResourceExhaustedError": FakeResourceExhaustedError,
            },
        )

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                return fake_volume

    monotonic_time = 100.0
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        """Return the controllable monotonic clock."""
        return monotonic_time

    def fake_sleep(seconds: float) -> None:
        """Advance the controllable monotonic clock instead of actually sleeping."""
        nonlocal monotonic_time
        sleep_calls.append(seconds)
        monotonic_time += seconds

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)
    monkeypatch.setattr(
        sync_engine_module._sync_backends,
        "time",
        types.SimpleNamespace(sleep=fake_sleep, monotonic=fake_monotonic),
    )
    backend = sync_engine_module.ModalVolumeBackend("volume")

    assert backend.exists("/hashes/present.done") is True
    assert fake_volume.listdir_calls == 2
    assert 0.25 in sleep_calls

def test_modal_volume_backend_applies_shared_rate_limit_backoff_across_calls(
    sync_engine_module: Any,
    monkeypatch: Any,
) -> None:
    """One rate-limited Modal volume call should publish a shared backoff window for later calls."""

    class FakeVolume:
        """Minimal Modal volume double that always succeeds."""

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Return one successful listing."""
            return [remote_path]

    fake_volume = FakeVolume()

    class FakeModal:
        """Minimal modal SDK double exposing retryable error types."""

        exception = type(
            "FakeExceptionNamespace",
            (),
            {
                "NotFoundError": FileNotFoundError,
                "ResourceExhaustedError": RuntimeError,
            },
        )

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                return fake_volume

    monotonic_time = 100.0
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        """Return the controllable monotonic clock."""
        return monotonic_time

    def fake_sleep(seconds: float) -> None:
        """Advance the controllable monotonic clock instead of actually sleeping."""
        nonlocal monotonic_time
        sleep_calls.append(seconds)
        monotonic_time += seconds

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)
    monkeypatch.setattr(
        sync_engine_module._sync_backends,
        "time",
        types.SimpleNamespace(sleep=fake_sleep, monotonic=fake_monotonic),
    )
    backend = sync_engine_module.ModalVolumeBackend("volume")

    assert backend._record_shared_rate_limit_backoff() == 0.25
    assert backend.exists("/hashes/second.done") is True
    assert 0.25 in sleep_calls

def test_cancellable_backend_interruption_becomes_sync_cancellation(
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """An interrupted transport should stop queue-time sync without indexing it."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"model")

    class CancellableVolume:
        """Expose an upload that reports prompt cancellation."""

        def exists(self, remote_path: str) -> bool:
            """Report empty storage."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Reject the non-cancellable path."""
            del local_path, remote_path
            raise AssertionError("cancellable upload path was not used")

        def put_file_cancellable(
            self,
            local_path: Path,
            remote_path: str,
            *,
            cancellation_check: Any,
        ) -> None:
            """Report that the active transport was interrupted."""
            del local_path, remote_path
            raise InterruptedError("cancelled")

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol-required byte writes."""
            del payload, remote_path

    settings = settings_module.ModalSyncSettings(
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
    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=CancellableVolume(),
        settings=settings,
        cancellation_check=lambda: False,
    )

    with pytest.raises(sync_engine_module.SyncCancelledError, match="cancelled"):
        engine._sync_content_addressed_file(
            local_path=asset_path,
            remote_path="assets/model.safetensors",
            sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            sync_key="test-key",
            source_description=str(asset_path),
        )

