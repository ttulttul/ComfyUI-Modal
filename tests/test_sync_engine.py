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


def test_sync_file_deduplicates_by_hash(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Repeated file syncs should reuse the same remote path and sync-index record."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    sync_index_path = settings.local_storage_root / "metadata" / "sync_index.json"
    assert sync_index_path.exists()
    assert first.remote_path in sync_index_path.read_text(encoding="utf-8")


def test_request_asset_cache_syncs_repeated_prompt_asset_once(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Repeated references across remote nodes should share one request sync decision."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    sync_calls: list[Path] = []
    original_sync_file = engine.sync_file

    def recording_sync_file(local_path: Path, *args: Any, **kwargs: Any) -> Any:
        """Record actual request sync decisions before delegating."""
        sync_calls.append(local_path.resolve())
        return original_sync_file(local_path, *args, **kwargs)

    monkeypatch.setattr(engine, "sync_file", recording_sync_file)
    request_cache = engine.create_request_asset_cache(
        ({"model": str(asset_path)}, {"second_model": str(asset_path)})
    )

    first_inputs, first_assets = engine.sync_prompt_inputs(
        {"model": str(asset_path)},
        request_cache=request_cache,
    )
    second_inputs, second_assets = engine.sync_prompt_inputs(
        {"second_model": str(asset_path)},
        request_cache=request_cache,
    )

    assert sync_calls == [asset_path.resolve()]
    assert first_inputs["model"] == second_inputs["second_model"]
    assert first_assets == second_assets
    assert request_cache.synced_assets() == (first_assets[0],)


def test_sync_file_emits_upload_status(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Uploading a new asset should emit a status message naming the uploaded file."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"model-bytes")
    observed_statuses: list[tuple[str, int | None, int | None]] = []

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
    engine.sync_file(
        asset_path,
        status_callback=lambda message, current, total: observed_statuses.append(
            (message, current, total)
        ),
        item_index=1,
        total_items=2,
    )

    assert observed_statuses == [("Uploading asset 1/2 to Modal: model.safetensors", 1, 2)]


def test_sync_file_uses_self_hosted_destination_in_ssh_mode(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """SSH uploads should not describe the destination as Modal."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"model-bytes")
    observed_statuses: list[tuple[str, int | None, int | None]] = []
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="ssh_docker",
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
    engine.sync_file(
        asset_path,
        status_callback=lambda message, current, total: observed_statuses.append(
            (message, current, total)
        ),
    )

    assert observed_statuses == [
        ("Uploading asset to the self-hosted worker: model.safetensors", None, None)
    ]


def test_vast_sync_materializes_registered_huggingface_asset_before_upload(
    settings_module: Any,
    sync_engine_module: Any,
    huggingface_assets_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A registered Vast asset should be fetched remotely with the local hash contract."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"hub-backed-model")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    source = huggingface_assets_module.HuggingFaceAssetSource(
        repo_id="owner/model",
        revision="a" * 40,
        filename="weights/model.safetensors",
        sha256=sha256,
        size_bytes=asset_path.stat().st_size,
    )
    registry = huggingface_assets_module.HuggingFaceAssetRegistry(
        tmp_path / "user" / "huggingface-assets.json"
    )
    registry.upsert(source)
    observed_materializations: list[tuple[Any, str, str | None]] = []
    observed_uploads: list[tuple[Path, str]] = []
    observed_statuses: list[tuple[str, int | None, int | None]] = []

    class MaterializingVolume:
        """Expose the optional remote acquisition capability under test."""

        def exists(self, remote_path: str) -> bool:
            """Report an initially empty remote store."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Record a fallback upload that should not be needed."""
            observed_uploads.append((local_path, remote_path))

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept sync-index sentinel bytes for protocol completeness."""
            del payload, remote_path

        def materialize_huggingface_file(
            self,
            registered_source: Any,
            remote_path: str,
            *,
            token: str | None,
        ) -> bool:
            """Record and accept one direct Hub materialization."""
            observed_materializations.append((registered_source, remote_path, token))
            return True

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
    monkeypatch.setenv("HF_TOKEN", "secret-hf-token")
    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=MaterializingVolume(),
        settings=settings,
        huggingface_asset_registry=registry,
    )

    result = engine.sync_file(
        asset_path,
        status_callback=lambda message, current, total: observed_statuses.append(
            (message, current, total)
        ),
        item_index=1,
        total_items=2,
    )

    assert observed_uploads == []
    assert observed_materializations == [
        (source, result.remote_path, "secret-hf-token")
    ]
    assert observed_statuses == [
        (
            "Downloading asset 1/2 from Hugging Face on Vast.ai: model.safetensors",
            1,
            2,
        )
    ]


def test_vast_sync_automatically_discovers_huggingface_asset(
    settings_module: Any,
    sync_engine_module: Any,
    huggingface_assets_module: Any,
    tmp_path: Path,
) -> None:
    """Vast sync should discover provenance without a user-run registration command."""
    asset_path = tmp_path / "automatic.safetensors"
    asset_path.write_bytes(b"automatically-discovered")
    source = huggingface_assets_module.HuggingFaceAssetSource(
        repo_id="owner/automatic",
        revision="9" * 40,
        filename="automatic.safetensors",
        sha256=huggingface_assets_module.sha256_file(asset_path),
        size_bytes=asset_path.stat().st_size,
    )
    statuses: list[str] = []
    discoveries: list[tuple[Path, str]] = []

    class DiscoveringVolume:
        """Accept direct Hub materialization and reject local upload use."""

        def exists(self, remote_path: str) -> bool:
            """Report no preexisting remote asset."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Fail if automatic discovery unexpectedly falls back to upload."""
            raise AssertionError(f"Unexpected upload of {local_path} to {remote_path}")

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol metadata writes."""
            del payload, remote_path

        def materialize_huggingface_file(
            self,
            registered_source: Any,
            remote_path: str,
            *,
            token: str | None,
        ) -> bool:
            """Accept the automatically discovered immutable source."""
            del remote_path, token
            assert registered_source == source
            return True

    class FakeDiscovery:
        """Return an automatically verified source for the local asset."""

        def discover(self, local_path: Path, *, sha256: str) -> Any:
            """Record discovery and return the expected source."""
            discoveries.append((local_path, sha256))
            return source

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
        volume=DiscoveringVolume(),
        settings=settings,
        huggingface_asset_registry=huggingface_assets_module.HuggingFaceAssetRegistry(
            tmp_path / "empty-registry.json"
        ),
        huggingface_asset_discovery=FakeDiscovery(),
    )

    engine.sync_file(
        asset_path,
        status_callback=lambda message, current, total: statuses.append(message),
    )

    assert discoveries == [(asset_path, source.sha256)]
    assert statuses == [
        "Identifying Hugging Face source for Vast.ai: automatic.safetensors",
        "Downloading asset from Hugging Face on Vast.ai: automatic.safetensors",
    ]


def test_resolve_model_path_preserves_huggingface_cache_symlink(
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Model resolution must retain a cache symlink long enough for provenance discovery."""
    cached_file = tmp_path / "cache" / "model.safetensors"
    cached_file.parent.mkdir()
    cached_file.write_bytes(b"cached")
    linked_file = tmp_path / "models" / "model.safetensors"
    linked_file.parent.mkdir()
    linked_file.symlink_to(cached_file)

    resolved = sync_engine_module.resolve_model_path(str(linked_file))

    assert resolved == linked_file
    assert resolved.resolve() == cached_file


def test_vast_sync_falls_back_to_upload_after_huggingface_failure(
    settings_module: Any,
    sync_engine_module: Any,
    huggingface_assets_module: Any,
    tmp_path: Path,
) -> None:
    """A remote Hub miss should retain the existing content-addressed upload behavior."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"fallback-model")
    source = huggingface_assets_module.HuggingFaceAssetSource(
        repo_id="owner/model",
        revision="b" * 40,
        filename="model.safetensors",
        sha256=huggingface_assets_module.sha256_file(asset_path),
        size_bytes=asset_path.stat().st_size,
    )
    registry = huggingface_assets_module.HuggingFaceAssetRegistry(
        tmp_path / "user" / "huggingface-assets.json"
    )
    registry.upsert(source)
    uploads: list[tuple[Path, str]] = []
    statuses: list[str] = []

    class FailingMaterializingVolume:
        """Decline remote acquisition and record the fallback upload."""

        def exists(self, remote_path: str) -> bool:
            """Report no cached files."""
            del remote_path
            return False

        def put_file(self, local_path: Path, remote_path: str) -> None:
            """Record the safe fallback path."""
            uploads.append((local_path, remote_path))

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            """Accept protocol metadata writes."""
            del payload, remote_path

        def materialize_huggingface_file(
            self,
            registered_source: Any,
            remote_path: str,
            *,
            token: str | None,
        ) -> bool:
            """Simulate an unavailable or unauthorized Hub source."""
            del registered_source, remote_path, token
            return False

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
        volume=FailingMaterializingVolume(),
        settings=settings,
        huggingface_asset_registry=registry,
    )

    result = engine.sync_file(
        asset_path,
        status_callback=lambda message, current, total: statuses.append(message),
    )

    assert uploads == [(asset_path.resolve(), result.remote_path)]
    assert statuses == [
        "Downloading asset from Hugging Face on Vast.ai: model.safetensors",
        "Uploading asset to the Vast.ai instance: model.safetensors",
    ]


def test_sync_custom_nodes_directory_creates_archive(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The sync engine should archive and mirror a custom_nodes directory."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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


def test_sync_custom_nodes_directory_only_checks_once_per_engine_lifetime(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The same sync engine should not rescan custom_nodes after the first successful sync."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    first_bundle = engine.sync_custom_nodes_directory()
    assert first_bundle is not None
    assert first_bundle.uploaded is True

    (package_dir / "new_file.py").write_text("print('new node')\n", encoding="utf-8")

    def fail_hash_directory(path: Path) -> str:
        """Fail the test when a second custom_nodes scan happens."""
        raise AssertionError(f"Did not expect custom_nodes to be rehashed: {path}")

    monkeypatch.setattr(engine, "_hash_directory", fail_hash_directory)

    second_bundle = engine.sync_custom_nodes_directory()

    assert second_bundle == sync_engine_module.SyncedAsset(
        local_path=first_bundle.local_path,
        remote_path=first_bundle.remote_path,
        sha256=first_bundle.sha256,
        uploaded=False,
    )


def test_sync_custom_nodes_directory_emits_packaging_and_upload_status(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Building a new custom_nodes archive should report packaging and upload stages."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    observed_statuses: list[tuple[str, int | None, int | None]] = []

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
    engine.sync_custom_nodes_directory(
        status_callback=lambda message, current, total: observed_statuses.append(
            (message, current, total)
        )
    )

    assert observed_statuses == [
        ("Packaging custom-node code for Modal", None, None),
        ("Uploading custom-node code and assets to Modal", None, None),
    ]


def test_sync_custom_nodes_directory_concurrent_calls_share_one_initial_sync(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Concurrent first-run calls should serialize behind one custom_nodes sync decision."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    original_sync = engine._custom_nodes._sync_custom_nodes_directory_uncached
    sync_call_count = 0
    sync_call_count_lock = threading.Lock()
    release_first_sync = threading.Event()

    def wrapped_sync(*, status_callback: Any = None) -> Any:
        """Count uncached sync executions and block the first one briefly."""
        nonlocal sync_call_count
        with sync_call_count_lock:
            sync_call_count += 1
        release_first_sync.wait(timeout=5.0)
        return original_sync(status_callback=status_callback)

    monkeypatch.setattr(
        engine._custom_nodes,
        "_sync_custom_nodes_directory_uncached",
        wrapped_sync,
    )

    results: list[Any] = [None, None]

    def run_sync(index: int) -> None:
        """Run one custom_nodes sync call and store the result."""
        results[index] = engine.sync_custom_nodes_directory()

    first_thread = threading.Thread(target=run_sync, args=(0,))
    second_thread = threading.Thread(target=run_sync, args=(1,))
    first_thread.start()
    time.sleep(0.1)
    second_thread.start()
    release_first_sync.set()
    first_thread.join(timeout=5.0)
    second_thread.join(timeout=5.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert sync_call_count == 1
    assert results[0] is not None
    assert results[1] is not None
    assert results[0].remote_path == results[1].remote_path
    assert results[0].sha256 == results[1].sha256
    assert results[0].uploaded is True
    assert results[1].uploaded is False


def test_hash_directory_ignores_virtualenv_and_bytecode_artifacts(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Ignored directories and file suffixes should not affect custom_nodes hashing."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
        """Volume double that forces archive reuse without any remote metadata probes."""

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

    monkeypatch.setattr(
        second_engine._custom_nodes,
        "_create_archive",
        fail_create_archive,
    )
    second_bundle = second_engine.sync_custom_nodes_directory()

    assert second_bundle is not None
    assert second_bundle.sha256 == first_bundle.sha256
    assert second_bundle.uploaded is False
    entry_hash = second_engine._custom_nodes_archive_specs(custom_nodes_dir)[0].sha256
    assert second_engine._cached_custom_nodes_archive_path("example", entry_hash).exists()


def test_sync_custom_nodes_separates_code_assets_and_nested_virtualenv(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Code ZIPs should exclude mounted model assets and nested virtual environments."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    model_path = package_dir / "checkpoints" / "model.pth"
    model_path.parent.mkdir()
    model_path.write_bytes(b"package-model-weights")
    venv_path = package_dir / ".venv" / "lib" / "site-packages" / "ignored.py"
    venv_path.parent.mkdir(parents=True)
    venv_path.write_text("raise RuntimeError('must not ship')\n", encoding="utf-8")
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
    manifest_path = settings.local_storage_root / bundle.remote_path.lstrip("/")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    [entry] = manifest["entries"]
    [asset] = entry["assets"]
    assert asset["relative_path"] == "example/checkpoints/model.pth"
    assert asset["size_bytes"] == len(b"package-model-weights")
    asset_path = settings.local_storage_root / asset["remote_path"].lstrip("/")
    assert asset_path.read_bytes() == b"package-model-weights"

    archive_path = settings.local_storage_root / entry["remote_path"].lstrip("/")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archived_names = archive.namelist()
    assert "example/__init__.py" in archived_names
    assert "example/checkpoints/model.pth" not in archived_names
    assert not any(".venv" in name for name in archived_names)


def test_sync_file_reuses_sync_index_record_for_existing_remote_payload(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An indexed deterministic asset should be reused without a second upload."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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

    class RecordingVolume:
        """Volume double that records any attempted uploads."""

        def __init__(self) -> None:
            """Initialize captured writes."""
            self.put_file_calls: list[tuple[Path, str]] = []

        def put_file(self, local_path: Path, remote_path: str) -> None:
            self.put_file_calls.append((local_path, remote_path))

        def put_bytes(self, payload: bytes, remote_path: str) -> None:
            raise AssertionError("Sync records should not be mirrored as marker files.")

        def exists(self, remote_path: str) -> bool:
            raise AssertionError("Sync should not probe volume metadata for indexed payloads.")

    volume = RecordingVolume()
    engine = sync_engine_module.ModalAssetSyncEngine(volume=volume, settings=settings)
    engine.sync_index.put(
        engine._asset_sync_index_key(engine._hash_file(asset_path)),
        {"remote_path": "/assets/existing_model.safetensors", "source": "existing"},
    )

    synced_asset = engine.sync_file(asset_path)

    assert synced_asset.uploaded is False
    assert synced_asset.remote_path == "/assets/existing_model.safetensors"
    assert volume.put_file_calls == []


def test_sync_custom_nodes_directory_reuses_indexed_remote_bundle(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An indexed hash-named remote custom_nodes bundle should be reused without rebuilding or reuploading."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
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
    directory_hash = engine._hash_directory(custom_nodes_dir)
    remote_path = engine._custom_nodes_manifest_remote_path(directory_hash)

    class RecordingVolume:
        """Volume double that records any attempted uploads."""

        def __init__(self) -> None:
            """Initialize captured writes."""
            self.put_file_calls: list[tuple[Path, str]] = []

        def put_file(self, local_path: Path, candidate_path: str) -> None:
            self.put_file_calls.append((local_path, candidate_path))

        def put_bytes(self, payload: bytes, candidate_path: str) -> None:
            raise AssertionError("Sync records should not be mirrored as marker files.")

        def exists(self, candidate_path: str) -> bool:
            raise AssertionError("Sync should not probe volume metadata for indexed bundles.")

    volume = RecordingVolume()
    engine = sync_engine_module.ModalAssetSyncEngine(volume=volume, settings=settings)
    engine.sync_index.put(
        engine._custom_nodes_manifest_sync_index_key(directory_hash),
        {"remote_path": remote_path, "source": str(custom_nodes_dir)},
    )

    def fail_create_archive(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Expected the indexed hash-named remote bundle to be reused.")

    monkeypatch.setattr(engine._custom_nodes, "_create_archive", fail_create_archive)
    bundle = engine.sync_custom_nodes_directory()

    assert bundle is not None
    assert bundle.sha256 == directory_hash
    assert bundle.remote_path == remote_path
    assert bundle.uploaded is False
    assert volume.put_file_calls == []


def test_sync_custom_nodes_directory_only_rebuilds_changed_top_level_archive(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Changing one custom_nodes package should only rebuild that package archive plus the manifest."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_a = custom_nodes_dir / "example_a"
    package_b = custom_nodes_dir / "example_b"
    package_a.mkdir(parents=True)
    package_b.mkdir(parents=True)
    (package_a / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    (package_b / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

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

    (package_b / "node.py").write_text("VALUE = 2\n", encoding="utf-8")

    class NeverExistsVolume:
        """Volume double that forces all deterministic uploads through without indexed cache hits."""

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
    rebuilt_entries: list[str] = []
    original_create_archive = second_engine._custom_nodes._create_archive_from_files

    def record_create_archive(root_path: Path, files: list[Path], archive_path: Path) -> Path:
        """Record which top-level package archive had to be rebuilt."""
        del root_path
        rebuilt_entries.append(files[0].relative_to(custom_nodes_dir).parts[0])
        return original_create_archive(custom_nodes_dir, files, archive_path)

    monkeypatch.setattr(
        second_engine._custom_nodes,
        "_create_archive_from_files",
        record_create_archive,
    )
    second_bundle = second_engine.sync_custom_nodes_directory()

    assert second_bundle is not None
    assert second_bundle.sha256 != first_bundle.sha256
    assert rebuilt_entries == ["example_b"]


def test_sync_custom_nodes_directory_builds_multiple_archives_in_parallel(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Fresh per-package custom_nodes archives should build in parallel."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_a = custom_nodes_dir / "example_a"
    package_b = custom_nodes_dir / "example_b"
    package_a.mkdir(parents=True)
    package_b.mkdir(parents=True)
    (package_a / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    (package_b / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

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
    original_create_archive = engine._custom_nodes._create_archive_from_files
    thread_ids: set[int] = set()
    started_count = 0
    started_lock = threading.Lock()
    overlap_event = threading.Event()

    def record_create_archive(root_path: Path, files: list[Path], archive_path: Path) -> Path:
        """Block briefly until two archive builds overlap so the test can observe parallel execution."""
        nonlocal started_count
        del root_path
        with started_lock:
            started_count += 1
            thread_ids.add(threading.get_ident())
            if started_count >= 2:
                overlap_event.set()
        assert overlap_event.wait(0.2), "Expected per-package archive builds to overlap."
        time.sleep(0.02)
        return original_create_archive(custom_nodes_dir, files, archive_path)

    monkeypatch.setattr(
        engine._custom_nodes,
        "_create_archive_from_files",
        record_create_archive,
    )
    bundle = engine.sync_custom_nodes_directory()

    assert bundle is not None
    assert len(thread_ids) >= 2


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


def test_remote_sync_index_discards_stale_volume_epoch_and_reuploads_missing_payloads(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A fresh volume should ignore stale sync-index records from an older volume epoch."""

    class FakeBatch:
        """Capture files uploaded through Modal batch_upload."""

        def __init__(self, volume: "FakeVolume") -> None:
            """Store the backing volume for uploaded-path tracking."""
            self.volume = volume

        def __enter__(self) -> "FakeBatch":
            """Return the active batch context."""
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            """Close the fake batch context."""
            return None

        def put_file(self, source: Any, remote_path: str) -> None:
            """Record the uploaded remote path as present in the fake volume."""
            del source
            self.volume.paths.add(remote_path)
            self.volume.uploads.append(remote_path)

    class FakeVolume:
        """Minimal Modal volume double with listdir accounting."""

        def __init__(self) -> None:
            """Initialize fake storage state."""
            self.paths: set[str] = set()
            self.uploads: list[str] = []

        def listdir(self, remote_path: str, recursive: bool = False) -> list[str]:
            """Return a listing for known paths."""
            del recursive
            return [remote_path] if remote_path in self.paths else []

        def batch_upload(self) -> FakeBatch:
            """Return a fake uploader."""
            return FakeBatch(self)

    fake_volume = FakeVolume()
    fake_dict: dict[str, Any] = {}

    class FakeModal:
        """Minimal Modal SDK double exposing both Volume and Dict backends."""

        exception = type(
            "FakeExceptionNamespace",
            (),
            {
                "NotFoundError": FileNotFoundError,
                "ResourceExhaustedError": RuntimeError,
            },
        )

        class Dict:
            """Namespace for sync-index lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> dict[str, Any]:
                """Return the shared fake dict store for any lookup."""
                del name, create_if_missing
                return fake_dict

        class Volume:
            """Namespace for volume lookups."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
                """Return the fake volume for any lookup."""
                del name, create_if_missing
                return fake_volume

    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", FakeModal)
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
    entry_hash = engine._hash_directory(package_dir)
    fake_dict[f"{settings.volume_name}:current_volume_epoch"] = {
        "epoch": "stale-epoch",
        "remote_path": "/sync_index_epochs/stale-epoch.json",
        "sentinel_path": "/sync_index_epochs/stale-epoch.json",
    }
    fake_dict[
        f"{settings.volume_name}:epoch:stale-epoch:custom_nodes_entry:example:{entry_hash}"
    ] = {
        "remote_path": engine._custom_nodes_archive_remote_path("example", entry_hash),
        "source": str(package_dir),
    }

    bundle = engine.sync_custom_nodes_directory()

    assert bundle is not None
    assert bundle.uploaded is True
    assert any(path.startswith("/sync_index_epochs/") for path in fake_volume.uploads)
    assert any(path.startswith("/custom_nodes/entries/example/") for path in fake_volume.uploads)
    assert fake_dict[f"{settings.volume_name}:current_volume_epoch"]["epoch"] != "stale-epoch"


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


def test_custom_node_archive_uses_archive_byte_digest_for_object_identity(
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """R2 identity should describe ZIP bytes rather than the source-tree digest."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    source_path = package_dir / "__init__.py"
    source_path.write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    engine = sync_engine_module.ModalAssetSyncEngine(
        volume=object(),
        settings=_r2_sync_settings(settings_module, tmp_path),
    )
    source_digest = engine._hash_file_group(custom_nodes_dir, [source_path])
    archive_spec = sync_engine_module._CustomNodesArchiveSpec(
        entry_name="example",
        display_name="example",
        source_description=str(package_dir),
        files=(source_path,),
        sha256=source_digest,
    )
    captured_identity: list[tuple[Path, str, str]] = []

    def sync_file(**kwargs: Any) -> Any:
        """Capture the exact local payload and identities selected for sync."""
        captured_identity.append(
            (kwargs["local_path"], kwargs["sha256"], kwargs["sync_key"])
        )
        return sync_engine_module._ContentAddressedSyncResult(
            remote_path=kwargs["remote_path"],
            uploaded=True,
        )

    monkeypatch.setattr(engine, "_sync_content_addressed_file", sync_file)

    result = engine._sync_custom_nodes_archive_spec(custom_nodes_dir, archive_spec)

    archive_path, payload_digest, sync_key = captured_identity[0]
    assert payload_digest == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert payload_digest != source_digest
    assert sync_key.endswith(f"custom_nodes_entry:example:{source_digest}")
    assert result.sha256 == payload_digest


def test_indexed_custom_node_manifest_backfills_archives_and_manifest_to_r2(
    settings_module: Any,
    sync_engine_module: Any,
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """Enabling R2 should revisit every object behind an indexed manifest."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    package_dir = custom_nodes_dir / "example"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "NODE_CLASS_MAPPINGS = {}\n",
        encoding="utf-8",
    )
    settings = replace(
        _r2_sync_settings(settings_module, tmp_path),
        sync_custom_nodes=True,
        custom_nodes_dir=custom_nodes_dir,
    )
    volume = _R2BackfillVolume(r2_cache_module)
    initial_engine = sync_engine_module.ModalAssetSyncEngine(
        volume=volume,
        settings=settings,
    )
    initial_bundle = initial_engine.sync_custom_nodes_directory()
    assert initial_bundle is not None
    assert initial_bundle.uploaded is True

    backfill_engine = sync_engine_module.ModalAssetSyncEngine(
        volume=volume,
        settings=settings,
        r2_cache=_SynchronousBackfillCache(r2_cache_module),
    )
    backfilled_bundle = backfill_engine.sync_custom_nodes_directory()
    backfill_engine.wait_for_r2_writebacks()

    assert backfilled_bundle is not None
    assert backfilled_bundle.uploaded is False
    assert len(volume.r2_uploads) == 2
    uploaded_paths = [remote_path for _, remote_path in volume.r2_uploads]
    assert any(remote_path.endswith(".zip") for remote_path in uploaded_paths)
    assert any(remote_path.endswith(".json") for remote_path in uploaded_paths)


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
