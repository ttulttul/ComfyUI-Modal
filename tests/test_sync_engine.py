"""Tests for asset syncing and custom_nodes archiving."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def test_sync_file_deduplicates_by_hash(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Repeated file syncs should reuse the same remote path and marker."""
    monkeypatch.setattr(sync_engine_module, "modal", None)
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"model-bytes")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
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

    engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    first = engine.sync_file(asset_path)
    second = engine.sync_file(asset_path)

    assert first.remote_path == second.remote_path
    assert first.sha256 == second.sha256
    assert first.uploaded is True
    assert second.uploaded is False
    assert (settings.local_storage_root / first.remote_path.lstrip("/")).exists()
    assert (settings.local_storage_root / f"hashes/{first.sha256}.done").exists()


def test_sync_custom_nodes_directory_creates_archive(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The sync engine should archive and mirror a custom_nodes directory."""
    monkeypatch.setattr(sync_engine_module, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

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
        custom_nodes_dir=custom_nodes_dir,
    )

    engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    bundle = engine.sync_custom_nodes_directory()

    assert bundle is not None
    assert bundle.remote_path.startswith("/custom_nodes/")
    assert bundle.uploaded is True
    assert (settings.local_storage_root / bundle.remote_path.lstrip("/")).exists()


def test_hash_directory_ignores_virtualenv_and_bytecode_artifacts(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Ignored directories and file suffixes should not affect custom_nodes hashing."""
    monkeypatch.setattr(sync_engine_module, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

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
        custom_nodes_dir=custom_nodes_dir,
    )

    engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    baseline_hash = engine._hash_directory(custom_nodes_dir)

    ignored_venv_dir = custom_nodes_dir / ".venv" / "lib"
    ignored_venv_dir.mkdir(parents=True)
    (ignored_venv_dir / "ignored.py").write_text("print('ignored')\n", encoding="utf-8")
    ignored_cache_dir = package_dir / "__pycache__"
    ignored_cache_dir.mkdir()
    (ignored_cache_dir / "example.cpython-312.pyc").write_bytes(b"bytecode")
    (package_dir / "ignored.pyc").write_bytes(b"bytecode")
    (package_dir / "ignored.log").write_text("log noise\n", encoding="utf-8")

    ignored_hash = engine._hash_directory(custom_nodes_dir)

    assert ignored_hash == baseline_hash


def test_sync_file_reuses_cached_hash_for_unchanged_file(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Unchanged files should reuse the persisted file digest instead of re-reading the payload."""
    monkeypatch.setattr(sync_engine_module, "modal", None)
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"model-bytes")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
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

    first_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    first_sha = first_engine._hash_file(asset_path)

    second_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)

    def fail_open(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Expected cached hash lookup to avoid reopening the unchanged file.")

    monkeypatch.setattr(Path, "open", fail_open)
    second_sha = second_engine._hash_file(asset_path)

    assert second_sha == first_sha


def test_sync_custom_nodes_directory_reuses_cached_archive(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An unchanged custom_nodes tree should reuse its digest-keyed local archive."""
    monkeypatch.setattr(sync_engine_module, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

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
        custom_nodes_dir=custom_nodes_dir,
    )

    first_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    first_bundle = first_engine.sync_custom_nodes_directory()
    assert first_bundle is not None

    class NeverExistsVolume:
        """Volume double that forces archive reuse by reporting no remote markers."""

        def exists(self, remote_path: str) -> bool:
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            return None

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            return None

    second_engine = sync_engine_module.ModalAssetSyncEngine(
        volume=NeverExistsVolume(),
        settings=settings,
    )

    def fail_create_archive(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Expected cached custom_nodes archive to be reused.")

    monkeypatch.setattr(second_engine, "_create_archive", fail_create_archive)
    second_bundle = second_engine.sync_custom_nodes_directory()

    assert second_bundle is not None
    assert second_bundle.sha256 == first_bundle.sha256
    assert second_bundle.uploaded is True
    assert second_engine._cached_custom_nodes_archive_path(first_bundle.sha256).exists()


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

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                return fake_volume

    monkeypatch.setattr(sync_engine_module, "modal", FakeModal)

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

    asset_path = tmp_path / "encoder.safetensors"
    asset_path.write_bytes(b"weights")
    synced = engine.sync_file(asset_path)

    uploaded_remote_paths = [remote_path for _, remote_path in fake_volume.uploads]
    assert synced.remote_path in uploaded_remote_paths
    assert f"/hashes/{synced.sha256}.done" in uploaded_remote_paths


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

    monkeypatch.setattr(sync_engine_module, "modal", FakeModal)
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

    monkeypatch.setattr(sync_engine_module, "modal", FakeModal)
    backend = sync_engine_module.ModalVolumeBackend("volume")

    assert backend.exists("/hashes/present.done") is False
    assert backend.exists("/hashes/present.done") is False
    assert fake_volume.listdir_calls == 1

    local_path = tmp_path / "marker.txt"
    local_path.write_text("marker", encoding="utf-8")
    backend.put_file(local_path, "/hashes/present.done")

    assert backend.exists("/hashes/present.done") is True
    assert fake_volume.listdir_calls == 1


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

    sleep_calls: list[float] = []
    monkeypatch.setattr(sync_engine_module, "modal", FakeModal)
    monkeypatch.setattr(sync_engine_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    backend = sync_engine_module.ModalVolumeBackend("volume")

    assert backend.exists("/hashes/present.done") is True
    assert fake_volume.listdir_calls == 2
    assert sleep_calls == [0.25]
