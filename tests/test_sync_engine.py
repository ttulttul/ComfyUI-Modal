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
