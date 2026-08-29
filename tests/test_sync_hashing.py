"""Tests for the sync hashing boundary."""

from __future__ import annotations

from sync_engine_test_support import *  # noqa: F401,F403

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

