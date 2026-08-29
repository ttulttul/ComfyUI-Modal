"""Tests for the sync custom nodes boundary."""

from __future__ import annotations

from sync_engine_test_support import *  # noqa: F401,F403

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

