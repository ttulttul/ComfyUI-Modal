"""Tests for the sync engine coordinator boundary."""

from __future__ import annotations

from sync_engine_test_support import *  # noqa: F401,F403

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

