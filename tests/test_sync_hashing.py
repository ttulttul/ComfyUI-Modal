"""Tests for the sync hashing boundary."""

from __future__ import annotations

import hashlib
import os

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



@pytest.fixture
def symlink_sync_engine(
    settings_module: Any, sync_engine_module: Any, monkeypatch: Any, tmp_path: Path,
) -> Any:
    """Build an offline engine with persistent caches and an empty staging root."""
    monkeypatch.setattr(sync_engine_module._sync_backends, "modal", None)
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()

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
    return engine


def test_symlinked_packages_match_archive_files(symlink_sync_engine: Any, tmp_path: Path) -> None:
    """Directory hashes must cover each logical alias included in package archives."""
    engine = symlink_sync_engine
    root = engine.settings.custom_nodes_dir
    package = tmp_path / "external_pack"
    package.mkdir()
    (package / "nodes.py").write_text("original", encoding="utf-8")
    (package / ".git").mkdir()
    (package / ".git" / "ignored.py").write_text("ignored", encoding="utf-8")
    (package / "ignored.pyc").write_bytes(b"ignored")
    for alias in ("first", "second"):
        (root / alias).symlink_to(package, target_is_directory=True)
    files = set(engine._iter_files(root))
    specs = engine._custom_nodes._custom_nodes_archive_specs(root)
    assert files == {root / "first" / "nodes.py", root / "second" / "nodes.py"}
    assert files == {file for spec in specs for file in spec.files}
    assert engine._hash_directory(root) != hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("mutation", ["edit", "add", "retarget"])
def test_symlinked_package_invalidates_manifest_cache(
    symlink_sync_engine: Any, tmp_path: Path, mutation: str,
) -> None:
    """A new engine must not reuse a manifest after linked sources change."""
    engine = symlink_sync_engine
    root = engine.settings.custom_nodes_dir
    package = tmp_path / "external_pack"
    package.mkdir()
    source = package / "nodes.py"
    source.write_text("original", encoding="utf-8")
    alias = root / "pack"
    alias.symlink_to(package, target_is_directory=True)
    first = engine.sync_custom_nodes_directory()
    assert first is not None
    if mutation == "edit":
        source.write_text("changed source", encoding="utf-8")
    elif mutation == "add":
        (root / "another").symlink_to(package, target_is_directory=True)
    else:
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        replacement_source = replacement / "nodes.py"
        replacement_source.write_text("replaced", encoding="utf-8")
        source_stat = source.stat()
        os.utime(replacement_source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
    restarted = type(engine).from_environment(engine.settings)
    second = restarted.sync_custom_nodes_directory()
    assert second is not None
    assert second.sha256 != first.sha256
    assert second.remote_path != first.remote_path
    assert second.uploaded


def test_symlink_directory_cycles_preserve_aliases(symlink_sync_engine: Any, tmp_path: Path) -> None:
    """Ancestor links stop traversal without suppressing independent aliases."""
    engine = symlink_sync_engine
    root = engine.settings.custom_nodes_dir
    package = tmp_path / "external_pack"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (nested / "nodes.py").write_text("source", encoding="utf-8")
    (nested / "back").symlink_to(package, target_is_directory=True)
    (package / "self").symlink_to(package, target_is_directory=True)
    for alias in ("first", "second"):
        (root / alias).symlink_to(package, target_is_directory=True)
    assert set(engine._iter_files(root)) == {
        root / alias / "nested" / "nodes.py" for alias in ("first", "second")
    }
    (root / "back_to_root").symlink_to(root, target_is_directory=True)
    specs = engine._custom_nodes._custom_nodes_archive_specs(root)
    assert {file for spec in specs for file in spec.files} == set(engine._iter_files(root))
    assert engine._hash_directory(root) == engine._hash_directory(root)
