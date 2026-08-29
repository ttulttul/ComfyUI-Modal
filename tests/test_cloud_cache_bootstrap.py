"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_dynamic_proxy_registration_sets_module_identity_on_new_and_cached_class(
    modal_executor_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary proxies should serialize a valid module after creation and cache reuse."""
    expected_module = "custom_nodes.ComfyUI-Modal"
    monkeypatch.setattr(
        modal_executor_module.ModalUniversalExecutor,
        "RELATIVE_PYTHON_MODULE",
        expected_module,
    )
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"OriginalNodeForModuleIdentity": _FakeOriginalNode},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    proxy_node_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNodeForModuleIdentity",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    _assert_node_module_identity(proxy_class, expected_module)

    proxy_class.RELATIVE_PYTHON_MODULE = None
    fake_nodes_module.NODE_CLASS_MAPPINGS.clear()
    fake_nodes_module.NODE_DISPLAY_NAME_MAPPINGS.clear()

    cached_proxy_node_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNodeForModuleIdentity",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )

    assert cached_proxy_node_id == proxy_node_id
    assert fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id] is proxy_class
    _assert_node_module_identity(proxy_class, expected_module)

def test_component_proxy_registration_sets_module_identity_on_new_and_cached_class(
    modal_executor_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Component proxies should serialize a valid module after creation and cache reuse."""
    expected_module = "custom_nodes.ComfyUI-Modal"
    monkeypatch.setattr(
        modal_executor_module.ModalUniversalExecutor,
        "RELATIVE_PYTHON_MODULE",
        expected_module,
    )
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    registration_kwargs = {
        "output_types": ("MODULE_IDENTITY_COMPONENT",),
        "output_names": ("value",),
        "output_is_list": (False,),
        "nodes_module": fake_nodes_module,
        "is_output_node": False,
    }

    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        **registration_kwargs
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    _assert_node_module_identity(proxy_class, expected_module)

    proxy_class.RELATIVE_PYTHON_MODULE = None
    fake_nodes_module.NODE_CLASS_MAPPINGS.clear()
    fake_nodes_module.NODE_DISPLAY_NAME_MAPPINGS.clear()

    cached_proxy_node_id = (
        modal_executor_module.ensure_modal_component_proxy_node_registered(
            **registration_kwargs
        )
    )

    assert cached_proxy_node_id == proxy_node_id
    assert fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id] is proxy_class
    _assert_node_module_identity(proxy_class, expected_module)

def test_component_proxy_cache_distinguishes_scheduler_list_outputs(
    modal_executor_module: Any,
) -> None:
    """Mapped list proxies must not reuse a cached scalar-output proxy class."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    scalar_proxy_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("LATENT",),
        output_names=("samples",),
        output_is_list=(False,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    list_proxy_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("LATENT",),
        output_names=("samples",),
        output_is_list=(True,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )

    assert scalar_proxy_id != list_proxy_id
    assert fake_nodes_module.NODE_CLASS_MAPPINGS[scalar_proxy_id].OUTPUT_IS_LIST == [False]
    assert fake_nodes_module.NODE_CLASS_MAPPINGS[list_proxy_id].OUTPUT_IS_LIST == [True]

def test_cache_friendly_proxy_payload_ignores_volatile_queue_metadata(
    modal_executor_module: Any,
    proxy_payloads_module: Any,
) -> None:
    """Identical proxy work should sanitize to one local cache surface across prompt runs."""
    first_payload = modal_executor_module.register_cache_friendly_proxy_payload(
        "node-9",
        {
            "payload_kind": "subgraph",
            "component_id": "component-9",
            "prompt_id": "prompt-1",
            "boundary_outputs": [],
            "execute_node_ids": ["12"],
            "extra_data": {
                "client_id": "client-1",
                "create_time": 1000,
                "modal": {"remote_component_ids": ["12"]},
            },
            "requires_volume_reload": True,
            "volume_reload_marker": "marker-1",
            "uploaded_volume_paths": ["/storage/assets/a.bin"],
        },
    )
    second_payload = modal_executor_module.register_cache_friendly_proxy_payload(
        "node-9",
        {
            "payload_kind": "subgraph",
            "component_id": "component-9",
            "prompt_id": "prompt-2",
            "boundary_outputs": [],
            "execute_node_ids": ["12"],
            "extra_data": {
                "client_id": "client-1",
                "create_time": 2000,
                "modal": {"remote_component_ids": ["12", "39"]},
            },
            "requires_volume_reload": False,
            "volume_reload_marker": None,
            "uploaded_volume_paths": [],
        },
    )

    assert first_payload == second_payload == {
        "payload_kind": "subgraph",
        "component_id": "component-9",
        "boundary_outputs": [],
        "execute_node_ids": ["12"],
        proxy_payloads_module._PROXY_CACHE_CONTEXT_ID_KEY: "node-9",
    }
    assert modal_executor_module._rehydrate_proxy_payload(
        first_payload,
        unique_id="node-9",
        prompt_id="prompt-1",
    ) == {
        "payload_kind": "subgraph",
        "component_id": "component-9",
        "prompt_id": "prompt-1",
        "boundary_outputs": [],
        "execute_node_ids": ["12"],
        "extra_data": {
            "client_id": "client-1",
            "create_time": 1000,
            "modal": {"remote_component_ids": ["12"]},
        },
        "requires_volume_reload": True,
        "volume_reload_marker": "marker-1",
        "uploaded_volume_paths": ["/storage/assets/a.bin"],
    }
    assert modal_executor_module._rehydrate_proxy_payload(
        second_payload,
        unique_id="node-9",
        prompt_id="prompt-2",
    ) == {
        "payload_kind": "subgraph",
        "component_id": "component-9",
        "prompt_id": "prompt-2",
        "boundary_outputs": [],
        "execute_node_ids": ["12"],
        "extra_data": {
            "client_id": "client-1",
            "create_time": 2000,
            "modal": {"remote_component_ids": ["12", "39"]},
        },
        "requires_volume_reload": False,
        "volume_reload_marker": None,
        "uploaded_volume_paths": [],
    }

def test_modal_cloud_reuses_extracted_custom_nodes_bundle(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The remote runtime should avoid re-extracting an unchanged custom_nodes bundle."""
    storage_root = tmp_path / "storage"
    bundle_path = storage_root / "custom_nodes" / "bundle.zip"
    bundle_path.parent.mkdir(parents=True)

    import zipfile

    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("example/__init__.py", "NODE_CLASS_MAPPINGS = {}\n")

    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(storage_root))
    modal_cloud_module.get_settings.cache_clear()
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    original_cache = dict(bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES)
    bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
    try:
        first_root = modal_cloud_module._extract_custom_nodes_bundle("/custom_nodes/bundle.zip")
        monkeypatch.setattr(
            bootstrap_owner.zipfile,
            "ZipFile",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Expected cached extraction root to be reused.")
            ),
        )
        second_root = modal_cloud_module._extract_custom_nodes_bundle("/custom_nodes/bundle.zip")
    finally:
        modal_cloud_module.get_settings.cache_clear()
        bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
        bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.update(original_cache)

    assert first_root is not None
    assert second_root == first_root

def test_modal_cloud_extracts_custom_nodes_manifest_with_multiple_archives(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The remote runtime should resolve manifest-based custom_nodes bundles into several archives."""
    storage_root = tmp_path / "storage"
    entry_a_path = storage_root / "custom_nodes" / "entries" / "example_a" / "hash_a_bundle.zip"
    entry_b_path = storage_root / "custom_nodes" / "entries" / "example_b" / "hash_b_bundle.zip"
    manifest_path = storage_root / "custom_nodes" / "manifests" / "bundle_manifest.json"
    entry_a_path.parent.mkdir(parents=True, exist_ok=True)
    entry_b_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    import zipfile

    with zipfile.ZipFile(entry_a_path, "w") as archive:
        archive.writestr("example_a/__init__.py", "NODE_CLASS_MAPPINGS = {}\n")
    with zipfile.ZipFile(entry_b_path, "w") as archive:
        archive.writestr("example_b/__init__.py", "NODE_CLASS_MAPPINGS = {}\n")
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "bundle_sha256": "bundle-hash",
                "entries": [
                    {
                        "entry_name": "example_a",
                        "display_name": "example_a",
                        "sha256": "hash-a",
                        "remote_path": "/custom_nodes/entries/example_a/hash_a_bundle.zip",
                    },
                    {
                        "entry_name": "example_b",
                        "display_name": "example_b",
                        "sha256": "hash-b",
                        "remote_path": "/custom_nodes/entries/example_b/hash_b_bundle.zip",
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(storage_root))
    modal_cloud_module.get_settings.cache_clear()
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    original_cache = dict(bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES)
    bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
    try:
        extraction_root = modal_cloud_module._extract_custom_nodes_bundle(
            "/custom_nodes/manifests/bundle_manifest.json"
        )
    finally:
        modal_cloud_module.get_settings.cache_clear()
        bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
        bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.update(original_cache)

    assert extraction_root is not None
    assert (extraction_root / "example_a" / "__init__.py").exists()
    assert (extraction_root / "example_b" / "__init__.py").exists()

def test_modal_cloud_links_version_two_custom_node_assets(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Version-two manifests should keep model assets on mounted storage."""
    storage_root = tmp_path / "storage"
    asset_sha256 = "a" * 64
    archive_path = storage_root / "custom_nodes/entries/example/code_bundle.zip"
    asset_path = storage_root / f"custom_nodes/assets/example/{asset_sha256}_model.pth"
    manifest_path = storage_root / "custom_nodes/manifests/bundle_manifest_v2.json"
    archive_path.parent.mkdir(parents=True)
    asset_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"mounted-model")
    import zipfile

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("example/__init__.py", "NODE_CLASS_MAPPINGS = {}\n")
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "bundle_sha256": "bundle-hash",
                "entries": [
                    {
                        "entry_name": "example",
                        "display_name": "example",
                        "sha256": "code-hash",
                        "remote_path": "/custom_nodes/entries/example/code_bundle.zip",
                        "assets": [
                            {
                                "relative_path": "example/checkpoints/model.pth",
                                "sha256": asset_sha256,
                                "size_bytes": len(b"mounted-model"),
                                "remote_path": (
                                    f"/custom_nodes/assets/example/{asset_sha256}_model.pth"
                                ),
                            }
                        ],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(storage_root))
    modal_cloud_module.get_settings.cache_clear()
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    original_cache = dict(bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES)
    bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
    try:
        extraction_root = modal_cloud_module._extract_custom_nodes_bundle(
            "/custom_nodes/manifests/bundle_manifest_v2.json"
        )
    finally:
        modal_cloud_module.get_settings.cache_clear()
        bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
        bootstrap_owner._EXTRACTED_CUSTOM_NODE_BUNDLES.update(original_cache)

    assert extraction_root is not None
    linked_asset = extraction_root / "example/checkpoints/model.pth"
    assert linked_asset.is_symlink()
    assert linked_asset.read_bytes() == b"mounted-model"

def test_local_fallback_links_version_two_custom_node_assets(
    remote_modal_app_module: Any,
    tmp_path: Path,
) -> None:
    """The non-Modal extractor should preserve version-two package asset links."""
    storage_root = tmp_path / "storage"
    extraction_root = tmp_path / "extracted"
    asset_sha256 = "b" * 64
    asset_path = storage_root / f"custom_nodes/assets/example/{asset_sha256}_model.pth"
    manifest_path = storage_root / "custom_nodes/manifests/bundle_manifest_v2.json"
    asset_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    extraction_root.mkdir()
    asset_path.write_bytes(b"local-mounted-model")
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": [
                    {
                        "entry_name": "example",
                        "assets": [
                            {
                                "relative_path": "example/checkpoints/model.pth",
                                "sha256": asset_sha256,
                                "size_bytes": len(b"local-mounted-model"),
                                "remote_path": (
                                    f"/custom_nodes/assets/example/{asset_sha256}_model.pth"
                                ),
                            }
                        ],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    remote_modal_app_module._materialize_local_custom_node_assets(
        manifest_path,
        storage_root,
        extraction_root,
    )

    linked_asset = extraction_root / "example/checkpoints/model.pth"
    assert linked_asset.is_symlink()
    assert linked_asset.read_bytes() == b"local-mounted-model"

def test_modal_cloud_readthrough_hydrates_custom_node_manifest_dependencies(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Read-through hydration should include archives and assets named by a manifest."""
    manifest_payload = json.dumps(
        {
            "version": 2,
            "entries": [
                {
                    "remote_path": "/custom_nodes/packages/hash_package.zip",
                    "assets": [
                        {"remote_path": "/custom_nodes/assets/hash_model.safetensors"}
                    ],
                }
            ],
        }
    ).encode()
    committed_files = {
        "custom_nodes/manifests/hash_manifest.json": manifest_payload,
        "custom_nodes/packages/hash_package.zip": b"package",
        "custom_nodes/assets/hash_model.safetensors": b"model",
    }

    class FakeVolume:
        """Modal Volume double backed by committed in-memory files."""

        def read_file(self, path: str) -> Iterator[bytes]:
            """Yield the requested committed file."""
            yield committed_files[path]

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    readthrough_root = tmp_path / "readthrough"
    _patch_cloud_storage_root(monkeypatch, modal_cloud_module, storage_root)
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    monkeypatch.setattr(
        bootstrap_owner,
        "_REMOTE_VOLUME_READTHROUGH_ROOT",
        readthrough_root,
    )

    hydrated_paths = modal_cloud_module._hydrate_missing_payload_volume_paths(
        FakeVolume(),
        {"custom_nodes_bundle": "/custom_nodes/manifests/hash_manifest.json"},
    )

    assert hydrated_paths == [
        readthrough_root / "custom_nodes" / "manifests" / "hash_manifest.json",
        readthrough_root / "custom_nodes" / "assets" / "hash_model.safetensors",
        readthrough_root / "custom_nodes" / "packages" / "hash_package.zip",
    ]

def test_modal_cloud_clears_each_owner_cache_before_volume_reload(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Volume preparation should clear every cache that can retain mounted files."""
    cleared: list[str] = []
    volume_reload_owner = _cloud_volume_reload_owner()
    monkeypatch.setattr(
        volume_reload_owner,
        "clear_cloud_prompt_execution_warm_caches",
        lambda: cleared.append("prompt-execution"),
    )
    monkeypatch.setattr(
        volume_reload_owner,
        "clear_cloud_session_bridge_warm_caches",
        lambda: cleared.append("session-bridge"),
    )
    monkeypatch.setattr(
        volume_reload_owner,
        "clear_comfy_bootstrap_warm_caches",
        lambda: cleared.append("comfy-bootstrap"),
    )

    modal_cloud_module._clear_warm_remote_caches()

    assert cleared == ["prompt-execution", "session-bridge", "comfy-bootstrap"]

def test_modal_cloud_loader_cache_reuses_and_clones_outputs(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Cached loader wrappers should avoid repeated loads and clone cached outputs on hits."""

    class FakeLoader:
        """Simple loader with one expensive method."""

        def __init__(self) -> None:
            """Initialize call counter state."""
            self.calls = 0

        def load(self, model_name: str, device: str = "default") -> tuple[_CloneableCacheValue]:
            """Return a cloneable payload while counting underlying loads."""
            self.calls += 1
            return (_CloneableCacheValue(f"{model_name}:{device}:{self.calls}"),)

    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    original_cache = dict(bootstrap_owner._LOADER_OUTPUT_CACHE)
    original_wrapped = set(bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES)
    original_metrics = dict(bootstrap_owner._LOADER_CACHE_METRICS)
    bootstrap_owner._LOADER_OUTPUT_CACHE.clear()
    bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES.clear()
    bootstrap_owner._LOADER_CACHE_METRICS.clear()
    bootstrap_owner._LOADER_CACHE_METRICS.update({"hit": 0, "miss": 0})
    try:
        modal_cloud_module._wrap_loader_method_with_cache(
            "FakeLoader",
            FakeLoader,
            "load",
            lambda kwargs: modal_cloud_module._serialize_loader_cache_key(kwargs),
        )
        loader = FakeLoader()
        first = loader.load("model.safetensors", device="cpu")[0]
        second = loader.load("model.safetensors", device="cpu")[0]
        third = loader.load("other.safetensors", device="cpu")[0]
        metrics_snapshot = modal_cloud_module._loader_cache_metric_snapshot()
    finally:
        bootstrap_owner._LOADER_OUTPUT_CACHE.clear()
        bootstrap_owner._LOADER_OUTPUT_CACHE.update(original_cache)
        bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES.clear()
        bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES.update(original_wrapped)
        bootstrap_owner._LOADER_CACHE_METRICS.clear()
        bootstrap_owner._LOADER_CACHE_METRICS.update(original_metrics)

    assert loader.calls == 2
    assert first.value == "model.safetensors:cpu:1"
    assert second.value == "model.safetensors:cpu:1"
    assert third.value == "other.safetensors:cpu:2"
    assert first is not second
    assert metrics_snapshot == {"hit": 1, "miss": 2}

def test_modal_cloud_installs_loader_cache_wrappers_for_builtin_loaders(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The runtime should patch the heavyweight built-in loaders once they are available."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "CheckpointLoader": type("CheckpointLoader", (), {"load_checkpoint": lambda self, config_name, ckpt_name: (config_name, ckpt_name)}),
            "CheckpointLoaderSimple": type("CheckpointLoaderSimple", (), {"load_checkpoint": lambda self, ckpt_name: (ckpt_name,)}),
            "UNETLoader": type("UNETLoader", (), {"load_unet": lambda self, unet_name, weight_dtype="default": (unet_name,)}),
            "CLIPLoader": type("CLIPLoader", (), {"load_clip": lambda self, clip_name, type="stable_diffusion", device="default": (clip_name,)}),
            "DualCLIPLoader": type("DualCLIPLoader", (), {"load_clip": lambda self, clip_name1, clip_name2, type, device="default": (clip_name1, clip_name2)}),
            "VAELoader": type("VAELoader", (), {"load_vae": lambda self, vae_name: (vae_name,)}),
            "unCLIPCheckpointLoader": type("unCLIPCheckpointLoader", (), {"load_checkpoint": lambda self, ckpt_name, output_vae=True, output_clip=True: (ckpt_name,)}),
            "ImageOnlyCheckpointLoader": type("ImageOnlyCheckpointLoader", (), {"load_checkpoint": lambda self, ckpt_name, output_vae=True, output_clip=True: (ckpt_name,)}),
        }
    )

    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    original_wrapped = set(bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES)
    bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES.clear()
    monkeypatch.setattr(bootstrap_owner, "_load_nodes_module", lambda: fake_nodes_module)
    try:
        modal_cloud_module._install_loader_cache_wrappers()
        installed_wrappers = set(bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES)
    finally:
        bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES.clear()
        bootstrap_owner._LOADER_CACHE_WRAPPED_CLASSES.update(original_wrapped)

    assert {
        "CheckpointLoader",
        "CheckpointLoaderSimple",
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
        "VAELoader",
        "unCLIPCheckpointLoader",
        "ImageOnlyCheckpointLoader",
    } <= installed_wrappers

def test_modal_cloud_node_cache_key_hashes_boundary_tensors(
    modal_cloud_module: Any,
) -> None:
    """Boundary tensors inside ComfyUI cache signatures should produce stable cache keys."""
    torch = pytest.importorskip("torch")

    first_tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    same_value_tensor = first_tensor.clone()
    different_tensor = first_tensor + 1
    signature = frozenset(
        {
            (
                12,
                frozenset(
                    {
                        (
                            4,
                            frozenset(
                                {
                                    (0, "latent_image"),
                                    (1, frozenset({("samples", first_tensor)})),
                                }
                            ),
                        )
                    }
                ),
            )
        }
    )
    same_signature = frozenset(
        {
            (
                12,
                frozenset(
                    {
                        (
                            4,
                            frozenset(
                                {
                                    (0, "latent_image"),
                                    (1, frozenset({("samples", same_value_tensor)})),
                                }
                            ),
                        )
                    }
                ),
            )
        }
    )
    different_signature = frozenset(
        {
            (
                12,
                frozenset(
                    {
                        (
                            4,
                            frozenset(
                                {
                                    (0, "latent_image"),
                                    (1, frozenset({("samples", different_tensor)})),
                                }
                            ),
                        )
                    }
                ),
            )
        }
    )

    first_key = modal_cloud_module._node_output_cache_key(signature)
    second_key = modal_cloud_module._node_output_cache_key(same_signature)
    different_key = modal_cloud_module._node_output_cache_key(different_signature)

    assert isinstance(first_key, str)
    assert first_key.startswith("NC_")
    assert second_key == first_key
    assert different_key != first_key

def test_modal_cloud_node_cache_key_rebuilds_input_signature_before_unhashable_conversion(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Distributed node caching should bypass ComfyUI's precomputed `Unhashable` data key."""
    torch = pytest.importorskip("torch")

    class FakeDynPrompt:
        """Minimal dynamic-prompt stub for input-signature reconstruction."""

        def __init__(self, node: dict[str, Any]) -> None:
            """Store one node payload under id `12`."""
            self._node = node

        def has_node(self, node_id: str) -> bool:
            """Return whether the requested node exists."""
            return str(node_id) == "12"

        def get_node(self, node_id: str) -> dict[str, Any]:
            """Return the stored node payload."""
            if not self.has_node(node_id):
                raise KeyError(node_id)
            return self._node

    class FakeUnhashable:
        """Stand-in for ComfyUI's `Unhashable` marker."""

    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    dynprompt = FakeDynPrompt(
        {
            "class_type": "FakeSampler",
            "inputs": {
                "latent_image": {"samples": tensor},
                "steps": 18,
            },
        }
    )
    cache_key_set = types.SimpleNamespace(
        dynprompt=dynprompt,
        is_changed_cache=types.SimpleNamespace(is_changed={"12": False}),
        get_ordered_ancestry=lambda current_dynprompt, node_id: ([], {}),
        include_node_id_in_input=lambda: False,
        get_data_key=lambda node_id: frozenset({("latent_image", FakeUnhashable())}),
    )

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_load_nodes_module",
        lambda: types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={"FakeSampler": type("FakeSampler", (), {})}
        ),
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_include_unique_id_in_input_signature",
        lambda class_type: False,
    )

    rebuilt_key = modal_cloud_module._node_output_cache_key_from_key_set_sync(cache_key_set, "12")
    direct_bad_key = modal_cloud_module._node_output_cache_key(cache_key_set.get_data_key("12"))

    assert isinstance(rebuilt_key, str)
    assert rebuilt_key.startswith("NC_")
    assert direct_bad_key is None

def test_modal_cloud_node_cache_key_uses_boundary_source_signature_for_unhashable_inputs(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Boundary-fed runtime objects should hash by stable source provenance instead of object identity."""

    class FakeDynPrompt:
        """Minimal dynamic prompt wrapper backed by one mutable prompt dict."""

        def __init__(self, prompt: dict[str, Any]) -> None:
            """Store the prompt used by the cache-key rebuild."""
            self._prompt = prompt

        def has_node(self, node_id: str) -> bool:
            """Return whether the requested node exists in the prompt."""
            return str(node_id) in self._prompt

        def get_node(self, node_id: str) -> dict[str, Any]:
            """Return the stored node payload."""
            return self._prompt[str(node_id)]

    class FakeModelPatcher:
        """Stand-in for ComfyUI's unhashable `ModelPatcher` runtime object."""

    prompt_one = {
        "39": {
            "class_type": "FakeSampler",
            "inputs": {},
        }
    }
    prompt_two = {
        "39": {
            "class_type": "FakeSampler",
            "inputs": {},
        }
    }
    prompt_three = {
        "39": {
            "class_type": "FakeSampler",
            "inputs": {},
        }
    }
    boundary_spec_one = [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "MODEL",
            "source_signature": "SRC_same_model",
            "targets": [{"node_id": "39", "input_name": "model"}],
        }
    ]
    boundary_spec_three = [
        {
            "proxy_input_name": "remote_input_0",
            "io_type": "MODEL",
            "source_signature": "SRC_other_model",
            "targets": [{"node_id": "39", "input_name": "model"}],
        }
    ]
    modal_cloud_module._apply_boundary_inputs(
        prompt_one,
        boundary_spec_one,
        {"remote_input_0": FakeModelPatcher()},
    )
    modal_cloud_module._apply_boundary_inputs(
        prompt_two,
        boundary_spec_one,
        {"remote_input_0": FakeModelPatcher()},
    )
    modal_cloud_module._apply_boundary_inputs(
        prompt_three,
        boundary_spec_three,
        {"remote_input_0": FakeModelPatcher()},
    )

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_load_nodes_module",
        lambda: types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={"FakeSampler": type("FakeSampler", (), {})}
        ),
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_include_unique_id_in_input_signature",
        lambda class_type: False,
    )

    def cache_key_for(prompt: dict[str, Any]) -> str | None:
        """Build one distributed cache key for the prepared prompt."""
        cache_key_set = types.SimpleNamespace(
            dynprompt=FakeDynPrompt(prompt),
            is_changed_cache=types.SimpleNamespace(is_changed={"39": False}),
            get_ordered_ancestry=lambda current_dynprompt, node_id: ([], {}),
            include_node_id_in_input=lambda: False,
            get_data_key=lambda node_id: None,
        )
        return modal_cloud_module._node_output_cache_key_from_key_set_sync(cache_key_set, "39")

    first_key = cache_key_for(prompt_one)
    second_key = cache_key_for(prompt_two)
    different_key = cache_key_for(prompt_three)

    assert isinstance(first_key, str)
    assert first_key.startswith("NC_")
    assert second_key == first_key
    assert different_key != first_key

def test_modal_cloud_node_cache_key_treats_nested_two_item_lists_as_data(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Nested conditioning-style two-item lists should not be mistaken for prompt links."""

    class FakeDynPrompt:
        """Minimal dynamic prompt wrapper for one data-only node."""

        def has_node(self, node_id: str) -> bool:
            """Return whether the requested node exists."""
            return str(node_id) == "545"

        def get_node(self, node_id: str) -> dict[str, Any]:
            """Return a node with nested list data shaped like Comfy conditioning."""
            if not self.has_node(node_id):
                raise KeyError(node_id)
            return {
                "class_type": "AnythingToMarkdown",
                "inputs": {
                    "anything": [
                        [
                            {"pooled_output": "summary"},
                            [["not-a-node-id"], ["not-an-output-index"]],
                        ]
                    ]
                },
            }

    cache_key_set = types.SimpleNamespace(
        dynprompt=FakeDynPrompt(),
        is_changed_cache=types.SimpleNamespace(is_changed={"545": False}),
        get_ordered_ancestry=lambda current_dynprompt, node_id: ([], {}),
        include_node_id_in_input=lambda: False,
        get_data_key=lambda node_id: None,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_load_nodes_module",
        lambda: types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={"AnythingToMarkdown": type("AnythingToMarkdown", (), {})}
        ),
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_include_unique_id_in_input_signature",
        lambda class_type: False,
    )

    cache_key = modal_cloud_module._node_output_cache_key_from_key_set_sync(cache_key_set, "545")

    assert isinstance(cache_key, str)
    assert cache_key.startswith("NC_")

def test_modal_cloud_collects_custom_node_runtime_packages(
    modal_cloud_module: Any,
    tmp_path: Path,
) -> None:
    """The Modal image should install requirements declared by bundled custom nodes."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    first_package = custom_nodes_dir / "first"
    second_package = custom_nodes_dir / "second"
    first_package.mkdir(parents=True)
    second_package.mkdir(parents=True)
    (first_package / "requirements.txt").write_text(
        "\n".join(
            [
                "# comments are ignored",
                "omegaconf==2.3.0",
                "hydra-core==1.3.2 # inline comment",
                "--extra-index-url https://example.invalid/simple",
                "-r extra-requirements.txt",
            ]
        ),
        encoding="utf-8",
    )
    (first_package / "extra-requirements.txt").write_text(
        "diffusers>=0.37.0\n",
        encoding="utf-8",
    )
    (second_package / "requirements.txt").write_text(
        "omegaconf==2.3.0\nsoundfile\n",
        encoding="utf-8",
    )

    packages = modal_cloud_module._custom_node_runtime_packages(custom_nodes_dir)

    assert packages == (
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
        "diffusers>=0.37.0",
        "soundfile",
    )

def test_modal_cloud_returns_no_custom_node_packages_without_requirements(
    modal_cloud_module: Any,
    tmp_path: Path,
) -> None:
    """Custom-node package installation should be skipped when no requirements exist."""
    custom_nodes_dir = tmp_path / "custom_nodes"
    (custom_nodes_dir / "example").mkdir(parents=True)

    assert modal_cloud_module._custom_node_runtime_packages(custom_nodes_dir) == ()

def test_modal_cloud_restores_runtime_pins_after_custom_node_packages(
    modal_cloud_module: Any,
) -> None:
    """Custom requirements must not remain authoritative for shared runtime packages."""

    class RecordingImage:
        """Record Modal image package layers."""

        def __init__(self) -> None:
            """Initialize an empty call record."""
            self.calls: list[tuple[str, ...]] = []

        def pip_install(self, *packages: str, **_options: Any) -> "RecordingImage":
            """Record one package installation layer."""
            self.calls.append(packages)
            return self

    image = RecordingImage()

    result = modal_cloud_module._install_custom_node_packages(
        image,
        ("numpy==1.26.4", "opencv-python-headless==4.11.0.86"),
    )

    assert result is image
    assert image.calls[0] == (
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
    )
    assert "numpy==2.3.5" in image.calls[1]
    assert "opencv-python-headless==4.13.0.92" in image.calls[1]

def test_modal_cloud_missing_prompt_node_class_reports_unavailable_bundle_path(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Missing-class diagnostics should distinguish absent payload metadata from unavailable storage."""
    fake_nodes_module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes_module)

    with pytest.raises(modal_cloud_module.RemoteSubgraphExecutionError) as exc_info:
        modal_cloud_module._ensure_prompt_node_classes_registered(
            component_id="component-1",
            prompt={"2": {"class_type": "KSamplerLoraSigmaInverse", "inputs": {}}},
            custom_nodes_root=None,
            custom_nodes_bundle_path="/custom_nodes/manifest.json",
        )

    message = str(exc_info.value)
    assert "custom_nodes_bundle='/custom_nodes/manifest.json'" in message
    assert "not available in Modal worker storage" in message

def test_modal_cloud_retries_custom_node_import_for_missing_prompt_class(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A missing prompt class should trigger one external custom-node import retry."""
    import_calls: list[str] = []
    custom_nodes_root = tmp_path / "custom_nodes_bundle"
    custom_nodes_root.mkdir()

    class RegisteredLaterNode:
        """Node type registered by the retry import."""

    async def fake_init_external_custom_nodes() -> None:
        """Register the custom class when the retry import runs."""
        import_calls.append("init")
        fake_nodes_module.NODE_CLASS_MAPPINGS["KSamplerLoraSigmaInverse"] = RegisteredLaterNode

    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        init_external_custom_nodes=fake_init_external_custom_nodes,
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes_module)
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    monkeypatch.setattr(
        bootstrap_owner,
        "_register_custom_nodes_root",
        lambda path: import_calls.append(str(path)),
    )
    monkeypatch.setattr(
        bootstrap_owner,
        "_install_loader_cache_wrappers",
        lambda: import_calls.append("wrappers"),
    )

    resolved_mapping = modal_cloud_module._ensure_prompt_node_classes_registered(
        component_id="component-1",
        prompt={"2": {"class_type": "KSamplerLoraSigmaInverse", "inputs": {}}},
        custom_nodes_root=custom_nodes_root,
    )

    assert resolved_mapping["KSamplerLoraSigmaInverse"] is RegisteredLaterNode
    assert import_calls == [str(custom_nodes_root), "init", "wrappers"]

def test_modal_cloud_commits_after_each_genuine_triton_cache_miss(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated new specializations for one profile must each become durable."""
    signal = {"size": 10}
    runtime_module = types.ModuleType("modal_llm_runtime")
    runtime_module.triton_compile_miss_signal_size = lambda: signal["size"]
    runtime_module.triton_compile_listener_engine_pids = lambda: (1234,)
    monkeypatch.setitem(sys.modules, "modal_llm_runtime", runtime_module)
    payload = {
        "payload_kind": "subgraph",
        "execute_node_ids": ["llm"],
        "subgraph_prompt": {
            "llm": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "same-profile"},
            }
        },
    }

    class FakeVolume:
        """Count explicit compile-cache commits."""

        def __init__(self) -> None:
            """Initialize an empty commit count."""
            self.commits = 0

        def commit(self) -> None:
            """Record one explicit commit."""
            self.commits += 1

    volume = FakeVolume()
    checkpoint = modal_cloud_module._llm_compile_miss_checkpoint(payload)
    assert checkpoint is not None
    assert (
        modal_cloud_module._commit_actual_llm_compile_cache(checkpoint, volume)
        is False
    )

    signal["size"] = 20
    assert modal_cloud_module._commit_actual_llm_compile_cache(checkpoint, volume)
    second_checkpoint = modal_cloud_module._llm_compile_miss_checkpoint(payload)
    signal["size"] = 35
    assert modal_cloud_module._commit_actual_llm_compile_cache(
        second_checkpoint,
        volume,
    )
    assert volume.commits == 2

def test_modal_cloud_does_not_claim_cache_hits_without_engine_listener(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing EngineCore telemetry must skip commits without reporting a hit."""
    runtime_module = types.ModuleType("modal_llm_runtime")
    runtime_module.triton_compile_miss_signal_size = lambda: 0
    runtime_module.triton_compile_listener_engine_pids = lambda: ()
    monkeypatch.setitem(sys.modules, "modal_llm_runtime", runtime_module)
    checkpoint = modal_cloud_module._LLMCompileMissCheckpoint(
        profiles=("profile",),
        signal_size=0,
        listener_engine_pids=(),
    )

    class FakeVolume:
        """Reject any commit without listener evidence."""

        def commit(self) -> None:
            """Fail if missing telemetry causes a commit."""
            raise AssertionError("listener-less execution cannot commit")

    with caplog.at_level(logging.WARNING):
        committed = modal_cloud_module._commit_actual_llm_compile_cache(
            checkpoint,
            FakeVolume(),
        )

    assert committed is False
    assert "no live vLLM EngineCore" in caplog.text
    assert "every Triton lookup hit" not in caplog.text

def test_modal_cloud_reloads_compile_cache_before_restored_runtime(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Restored workers should refresh the cache Volume before importing runtimes."""
    calls: list[str] = []

    class FakeVolume:
        """Record one reload."""

        def reload(self) -> None:
            """Record the refresh."""
            calls.append("reload")

    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append("runtime"),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_restored_runtime(FakeVolume())

    assert calls == ["reload", "runtime", "execution"]

def test_get_hourly_modal_app_billing_filters_and_caches_gpu_app(
    modal_billing_module: Any,
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hourly billing should quietly select and cache one GPU app/environment."""
    settings = modal_billing_module.get_settings()
    selected_settings = modal_billing_module.settings_for_modal_gpu(
        settings,
        "B300",
    )
    app_name = modal_billing_module.modal_deployment_app_name(selected_settings)
    interval_start = datetime(2026, 8, 19, 7, 0, tzinfo=timezone.utc)
    report_calls: list[dict[str, Any]] = []

    def workspace_billing_report(**kwargs: Any) -> list[dict[str, Any]]:
        """Return matching and unrelated hourly billing rows."""
        report_calls.append(kwargs)
        return [
            {
                "object_id": "ap-selected",
                "description": app_name,
                "environment_name": "main",
                "interval_start": interval_start,
                "cost": Decimal("0.11799566"),
            },
            {
                "object_id": "ap-other-env",
                "description": app_name,
                "environment_name": "dev",
                "interval_start": interval_start,
                "cost": Decimal("9.99"),
            },
            {
                "object_id": "ap-other-app",
                "description": "unrelated",
                "environment_name": "main",
                "interval_start": interval_start,
                "cost": Decimal("8.88"),
            },
        ]

    original_import_module = modal_billing_module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        """Supply the public Modal billing and environment SDK surfaces."""
        if name == "modal._object":
            return types.SimpleNamespace(_get_environment_name=lambda _environment: "main")
        if name == "modal.environments":
            return types.SimpleNamespace(ensure_env=lambda environment: environment or "main")
        if name == "modal.billing":
            return types.SimpleNamespace(
                workspace_billing_report=workspace_billing_report
            )
        if name == "modal.exception":
            return types.SimpleNamespace(Error=RuntimeError)
        return original_import_module(name)

    monkeypatch.setattr(modal_billing_module, "modal", object())
    monkeypatch.setattr(
        modal_billing_module.importlib,
        "import_module",
        fake_import_module,
    )
    modal_billing_module._MODAL_HOURLY_BILLING_CACHE.clear()
    modal_billing_module._MODAL_HOURLY_BILLING_ERROR_CACHE.clear()
    caplog.set_level(logging.INFO, logger=modal_billing_module.__name__)
    now = datetime(2026, 8, 19, 8, 15, tzinfo=timezone.utc)

    first_status = asyncio.run(
        modal_billing_module.get_hourly_modal_app_billing(
            "B300",
            settings,
            now=now,
        )
    )
    cached_status = asyncio.run(
        modal_billing_module.get_hourly_modal_app_billing(
            "B300",
            settings,
            now=now + timedelta(minutes=30),
        )
    )

    assert first_status is cached_status
    assert first_status.app_id == "ap-selected"
    assert first_status.app_name == app_name
    assert first_status.modal_gpu == "B300"
    assert first_status.environment_name == "main"
    assert first_status.app_cost_usd_before_credits == Decimal("0.11799566")
    assert first_status.has_usage is True
    assert first_status.interval_start == interval_start
    assert first_status.interval_end == datetime(
        2026,
        8,
        19,
        8,
        tzinfo=timezone.utc,
    )
    assert first_status.next_refresh_at == datetime(
        2026,
        8,
        19,
        9,
        10,
        tzinfo=timezone.utc,
    )
    assert report_calls == [
        {
            "start": interval_start,
            "end": datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc),
            "resolution": "h",
        }
    ]
    assert "Fetched Modal hourly billing" not in caplog.text

def test_get_hourly_modal_app_billing_caches_report_failures(
    modal_billing_module: Any,
    monkeypatch: Any,
) -> None:
    """Fast status polling should attempt a failed billing interval only once."""
    report_calls = 0

    def workspace_billing_report(**_kwargs: Any) -> list[dict[str, Any]]:
        """Simulate a workspace that cannot expose billing reports."""
        nonlocal report_calls
        report_calls += 1
        raise RuntimeError("billing access denied")

    original_import_module = modal_billing_module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        """Supply a failing public Modal billing report."""
        if name == "modal._object":
            return types.SimpleNamespace(_get_environment_name=lambda _environment: "main")
        if name == "modal.environments":
            return types.SimpleNamespace(ensure_env=lambda environment: environment or "main")
        if name == "modal.billing":
            return types.SimpleNamespace(
                workspace_billing_report=workspace_billing_report
            )
        if name == "modal.exception":
            return types.SimpleNamespace(Error=RuntimeError)
        return original_import_module(name)

    monkeypatch.setattr(modal_billing_module, "modal", object())
    monkeypatch.setattr(
        modal_billing_module.importlib,
        "import_module",
        fake_import_module,
    )
    modal_billing_module._MODAL_HOURLY_BILLING_CACHE.clear()
    modal_billing_module._MODAL_HOURLY_BILLING_ERROR_CACHE.clear()
    now = datetime(2026, 8, 19, 8, 15, tzinfo=timezone.utc)

    for _attempt in range(2):
        with pytest.raises(
            modal_billing_module.ModalBillingStatusError,
            match="billing access denied",
        ):
            asyncio.run(
                modal_billing_module.get_hourly_modal_app_billing(
                    "L4",
                    now=now,
                )
            )

    assert report_calls == 1

def test_modal_cloud_emits_restored_node_cache_events(
    modal_cloud_module: Any,
) -> None:
    """Persisted node-cache restores should surface one cached-node marker per restored node."""
    observed_updates: list[dict[str, Any]] = []

    modal_cloud_module._emit_restored_node_cache_events(observed_updates.append, ["7", "9"])

    assert observed_updates == [
        {
            "event_type": "node_cached",
            "node_id": "7",
            "display_node_id": "7",
            "real_node_id": "7",
        },
        {
            "event_type": "node_cached",
            "node_id": "9",
            "display_node_id": "9",
            "real_node_id": "9",
        },
    ]

def test_modal_cloud_initializes_remote_comfy_runtime_once_per_custom_node_root(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """The remote runtime should load built-in extras once and custom bundles per extracted root."""
    init_calls: list[tuple[Any, ...]] = []
    folder_path_calls: list[tuple[str, str, bool]] = []
    original_nodes_module = sys.modules.get("nodes")
    original_execution_module = sys.modules.get("execution")

    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    async def fake_init_extra_nodes(init_custom_nodes: bool = True, init_api_nodes: bool = True) -> None:
        init_calls.append(("extra", init_custom_nodes, init_api_nodes))

    async def fake_init_external_custom_nodes() -> None:
        init_calls.append(("external",))

    fake_nodes_module.init_extra_nodes = fake_init_extra_nodes
    fake_nodes_module.init_external_custom_nodes = fake_init_external_custom_nodes

    fake_folder_paths_module = types.SimpleNamespace(
        add_model_folder_path=lambda folder_name, full_folder_path, is_default=False: folder_path_calls.append(
            (folder_name, full_folder_path, is_default)
        )
    )

    monkeypatch.setitem(sys.modules, "nodes", fake_nodes_module)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths_module)

    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    original_base_initialized = bootstrap_owner._COMFY_RUNTIME_BASE_INITIALIZED
    original_custom_node_roots = set(bootstrap_owner._COMFY_RUNTIME_CUSTOM_NODE_ROOTS)
    bootstrap_owner._COMFY_RUNTIME_BASE_INITIALIZED = False
    bootstrap_owner._COMFY_RUNTIME_CUSTOM_NODE_ROOTS.clear()
    try:
        custom_nodes_root = tmp_path / "custom_nodes"
        custom_nodes_root.mkdir()

        modal_cloud_module._ensure_comfy_runtime_initialized(None)
        modal_cloud_module._ensure_comfy_runtime_initialized(custom_nodes_root)
        modal_cloud_module._ensure_comfy_runtime_initialized(custom_nodes_root)
    finally:
        bootstrap_owner._COMFY_RUNTIME_BASE_INITIALIZED = original_base_initialized
        bootstrap_owner._COMFY_RUNTIME_CUSTOM_NODE_ROOTS.clear()
        bootstrap_owner._COMFY_RUNTIME_CUSTOM_NODE_ROOTS.update(original_custom_node_roots)
        loaded_execution_module = sys.modules.get("execution")
        if original_execution_module is None:
            sys.modules.pop("execution", None)
        elif loaded_execution_module is not None and original_nodes_module is not None:
            loaded_execution_module.nodes = original_nodes_module

    assert init_calls == [("extra", False, True), ("external",)]
    assert folder_path_calls == [("custom_nodes", str(custom_nodes_root), True)]

def test_modal_cloud_uses_comfy_prompt_executor_cache_defaults(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The remote worker should mirror ComfyUI's prompt executor cache configuration."""
    fake_args = types.SimpleNamespace(
        cache_lru=0,
        cache_ram=4.0,
        cache_none=False,
    )
    fake_cli_args_module = types.SimpleNamespace(args=fake_args)
    fake_execution_module = types.SimpleNamespace(
        CacheType=types.SimpleNamespace(
            CLASSIC="classic",
            LRU="lru",
            RAM_PRESSURE="ram-pressure",
            NONE="none",
        )
    )
    monkeypatch.setitem(sys.modules, "comfy.cli_args", fake_cli_args_module)

    cache_type, cache_args = modal_cloud_module._prompt_executor_cache_config(fake_execution_module)

    assert cache_type == "ram-pressure"
    assert cache_args == {"lru": 0, "ram": 4.0}

def test_modal_cloud_uses_current_comfy_ram_pressure_cache_defaults(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Current list-valued cache arguments should use ComfyUI's calculated RAM thresholds."""
    fake_args = types.SimpleNamespace(
        cache_classic=False,
        cache_lru=0,
        cache_ram=[],
        cache_none=False,
    )
    fake_cli_args_module = types.SimpleNamespace(args=fake_args)
    fake_model_management_module = types.SimpleNamespace(total_ram=64 * 1024)
    fake_execution_module = types.SimpleNamespace(
        CacheType=types.SimpleNamespace(
            CLASSIC="classic",
            LRU="lru",
            RAM_PRESSURE="ram-pressure",
            NONE="none",
        )
    )
    monkeypatch.setitem(sys.modules, "comfy.cli_args", fake_cli_args_module)
    monkeypatch.setitem(
        sys.modules,
        "comfy.model_management",
        fake_model_management_module,
    )

    cache_type, cache_args = modal_cloud_module._prompt_executor_cache_config(fake_execution_module)

    assert cache_type == "ram-pressure"
    assert cache_args == {"lru": 0, "ram": 6.4, "ram_inactive": 64.0}

def test_modal_cloud_sync_cache_reads_use_current_get_local_api(
    modal_cloud_module: Any,
) -> None:
    """Synchronous progress paths should avoid creating current cache get coroutines."""
    cache_entry = types.SimpleNamespace(outputs=[[512, 512]])

    class FakeOutputsCache:
        """Current cache double with async get and synchronous get_local."""

        async def get(self, node_id: str) -> Any:
            """Fail if the synchronous compatibility path calls async get."""
            del node_id
            raise AssertionError("async get should not be created")

        def get_local(self, node_id: str) -> Any:
            """Return the already materialized local cache entry."""
            assert node_id == "115"
            return cache_entry

    resolved_entry = modal_cloud_module._prompt_executor_cache_get_sync(
        FakeOutputsCache(),
        "115",
    )

    assert resolved_entry is cache_entry

def test_modal_cloud_reuses_prompt_executor_for_same_cache_scope(
    modal_cloud_module: Any,
    tmp_path: Path,
) -> None:
    """Warm-container subgraph runs should reuse one PromptExecutor per cache scope."""

    class FakePromptExecutor:
        """Simple PromptExecutor double that records how many instances were created."""

        instances_created = 0

        def __init__(self, server: Any, cache_type: Any = False, cache_args: Any = None) -> None:
            """Capture initialization state for later assertions."""
            type(self).instances_created += 1
            self.server = server
            self.cache_type = cache_type
            self.cache_args = cache_args
            self.status_messages = [("stale", {})]
            self.success = False
            self.history_result = {"stale": True}

    fake_execution_module = types.SimpleNamespace(PromptExecutor=FakePromptExecutor)
    first_server = types.SimpleNamespace(client_id="first", last_node_id="node-1")
    second_server = types.SimpleNamespace(client_id="second", last_node_id="node-2")

    prompt_execution_owner = _cloud_prompt_execution_owner()
    original_states = dict(prompt_execution_owner._PROMPT_EXECUTOR_STATES)
    prompt_execution_owner._PROMPT_EXECUTOR_STATES.clear()
    try:
        first_state = modal_cloud_module._get_or_create_prompt_executor_state(
            execution=fake_execution_module,
            prompt_server=first_server,
            cache_type="classic",
            cache_args={"lru": 0, "ram": 0.0},
            custom_nodes_root=tmp_path / "bundle-a",
        )
        modal_cloud_module._reset_prompt_executor_request_state(first_state.executor, first_server)
        second_state = modal_cloud_module._get_or_create_prompt_executor_state(
            execution=fake_execution_module,
            prompt_server=second_server,
            cache_type="classic",
            cache_args={"lru": 0, "ram": 0.0},
            custom_nodes_root=tmp_path / "bundle-a",
        )
        modal_cloud_module._reset_prompt_executor_request_state(second_state.executor, second_server)
    finally:
        prompt_execution_owner._PROMPT_EXECUTOR_STATES.clear()
        prompt_execution_owner._PROMPT_EXECUTOR_STATES.update(original_states)

    assert FakePromptExecutor.instances_created == 1
    assert first_state is second_state
    assert second_state.executor.server is second_server
    assert second_state.executor.status_messages == []
    assert second_state.executor.success is True
    assert second_state.executor.history_result == {}
    assert second_server.client_id is None
    assert second_server.last_node_id is None

def test_modal_cloud_separates_prompt_executor_cache_scopes_by_custom_nodes_root(
    modal_cloud_module: Any,
    tmp_path: Path,
) -> None:
    """Different custom-node bundle roots should not share a PromptExecutor cache scope."""

    class FakePromptExecutor:
        """Simple PromptExecutor double used to count cache-scope creations."""

        instances_created = 0

        def __init__(self, server: Any, cache_type: Any = False, cache_args: Any = None) -> None:
            """Capture initialization state for later assertions."""
            type(self).instances_created += 1
            self.server = server
            self.cache_type = cache_type
            self.cache_args = cache_args
            self.status_messages = []
            self.success = True
            self.history_result = {}

    fake_execution_module = types.SimpleNamespace(PromptExecutor=FakePromptExecutor)

    prompt_execution_owner = _cloud_prompt_execution_owner()
    original_states = dict(prompt_execution_owner._PROMPT_EXECUTOR_STATES)
    prompt_execution_owner._PROMPT_EXECUTOR_STATES.clear()
    try:
        first_state = modal_cloud_module._get_or_create_prompt_executor_state(
            execution=fake_execution_module,
            prompt_server=types.SimpleNamespace(client_id=None, last_node_id=None),
            cache_type="classic",
            cache_args={"lru": 0, "ram": 0.0},
            custom_nodes_root=tmp_path / "bundle-a",
        )
        second_state = modal_cloud_module._get_or_create_prompt_executor_state(
            execution=fake_execution_module,
            prompt_server=types.SimpleNamespace(client_id=None, last_node_id=None),
            cache_type="classic",
            cache_args={"lru": 0, "ram": 0.0},
            custom_nodes_root=tmp_path / "bundle-b",
        )
    finally:
        prompt_execution_owner._PROMPT_EXECUTOR_STATES.clear()
        prompt_execution_owner._PROMPT_EXECUTOR_STATES.update(original_states)

    assert FakePromptExecutor.instances_created == 2
    assert first_state is not second_state

def test_modal_cloud_awaits_async_node_output_cache_writes(
    modal_cloud_module: Any,
) -> None:
    """Persisted cache writes should use Modal's non-blocking Dict interface."""
    observed_writes: list[tuple[str, dict[str, Any]]] = []

    class AsyncPut:
        """Expose the callable shape used by Modal's synchronized methods."""

        async def aio(self, cache_key: str, record: dict[str, Any]) -> None:
            """Record one asynchronous cache write."""
            observed_writes.append((cache_key, record))

    class AsyncCacheStore:
        """Reject synchronous assignment while exposing an asynchronous put method."""

        put = AsyncPut()

        def __setitem__(self, cache_key: str, record: dict[str, Any]) -> None:
            """Fail if persistence regresses to Modal's blocking operator interface."""
            del cache_key, record
            raise AssertionError("cache write must use put.aio")

    record = {"version": 1, "outputs_zlib": b"payload"}

    asyncio.run(
        modal_cloud_module._node_output_cache_store_put(
            AsyncCacheStore(),
            "NC_example",
            record,
        )
    )

    assert observed_writes == [("NC_example", record)]

def test_modal_cloud_restores_persisted_node_cache_across_prompt_executor_instances(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """A fresh PromptExecutor cache should round-trip persisted node outputs through Modal Dict state."""
    monkeypatch.setitem(sys.modules, "torchsde", types.ModuleType("torchsde"))
    modal_cloud_module._ensure_comfy_runtime_initialized(None)
    import comfy_execution.caching as comfy_caching

    execution = modal_cloud_module._load_execution_module()
    nodes_module = modal_cloud_module._load_nodes_module()
    cache_store: dict[str, Any] = {}
    prompt = {
        "node_1": {
            "class_type": "PersistentCacheNode",
            "inputs": {"value": 4},
            "_meta": {},
        }
    }

    _PersistentCacheNode.invocation_count = 0
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "PersistentCacheNode", _PersistentCacheNode)
    monkeypatch.setitem(
        comfy_caching.nodes.NODE_CLASS_MAPPINGS,
        "PersistentCacheNode",
        _PersistentCacheNode,
    )
    monkeypatch.setitem(
        nodes_module.NODE_DISPLAY_NAME_MAPPINGS,
        "PersistentCacheNode",
        "PersistentCacheNode",
    )
    cache_entry = execution.CacheEntry(ui={"output": {"value": [5]}}, outputs=[[5]])
    first_executor = execution.PromptExecutor(
        modal_cloud_module._NullPromptServer(),
        cache_type=execution.CacheType.CLASSIC,
        cache_args={"lru": 0, "ram": 0.0},
    )
    restored_first = asyncio.run(
        modal_cloud_module._restore_persisted_node_output_cache_entries(
            execution,
            first_executor,
            prompt_id="prompt-a",
            prompt=copy.deepcopy(prompt),
            cache_store=cache_store,
        )
    )
    asyncio.run(
        modal_cloud_module._await_maybe(
            first_executor.caches.outputs.set("node_1", cache_entry)
        )
    )
    persisted_nodes = asyncio.run(
        modal_cloud_module._persist_node_output_cache_entries(
            first_executor,
            prompt=copy.deepcopy(prompt),
            cache_store=cache_store,
        )
    )

    second_executor = execution.PromptExecutor(
        modal_cloud_module._NullPromptServer(),
        cache_type=execution.CacheType.CLASSIC,
        cache_args={"lru": 0, "ram": 0.0},
    )
    restored_cache_keys_by_node_id: dict[str, str] = {}
    restored_second = asyncio.run(
        modal_cloud_module._restore_persisted_node_output_cache_entries(
            execution,
            second_executor,
            prompt_id="prompt-b",
            prompt=copy.deepcopy(prompt),
            cache_store=cache_store,
            restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
        )
    )
    restored_entry = asyncio.run(
        modal_cloud_module._await_maybe(second_executor.caches.outputs.get("node_1"))
    )

    assert restored_first == []
    assert persisted_nodes == ["node_1"]
    assert restored_second == ["node_1"]
    assert restored_cache_keys_by_node_id == {"node_1": next(iter(cache_store))}
    assert list(cache_store) and all(key.startswith("NC_") for key in cache_store)
    assert restored_entry == cache_entry

def test_modal_cloud_installs_persisted_cache_restore_after_live_set_prompt(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Persisted-cache restore should run after PromptExecutor prepares the active outputs cache."""

    class FakeOutputsCache:
        """Minimal outputs-cache stub with a mutable cache-key-set marker."""

        def __init__(self) -> None:
            """Initialize the fake cache-key-set marker."""
            self.cache_key_set = None

        async def set_prompt(self, dynprompt: Any, node_ids: Any, is_changed_cache: Any) -> None:
            """Simulate ComfyUI assigning the live cache-key set during prompt setup."""
            del dynprompt, node_ids, is_changed_cache
            self.cache_key_set = "live-cache-key-set"

    outputs_cache = FakeOutputsCache()
    executor = types.SimpleNamespace(caches=types.SimpleNamespace(outputs=outputs_cache))
    observed_events: list[tuple[str, Any]] = []

    async def fake_restore(
        execution: Any,
        prepared_outputs_cache: Any,
        *,
        prompt: dict[str, Any],
        cache_store: Any,
        required_materialized_node_ids: Any = None,
        restored_cache_keys_by_node_id: dict[str, str] | None = None,
    ) -> list[str]:
        """Record the cache-key-set marker visible at restore time."""
        del execution, required_materialized_node_ids
        if restored_cache_keys_by_node_id is not None:
            restored_cache_keys_by_node_id["12"] = "NC_example"
        observed_events.append(
            (
                "restore",
                prepared_outputs_cache.cache_key_set,
                tuple(sorted(prompt)),
                cache_store,
            )
        )
        return ["12"]

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_restore_persisted_node_output_cache_entries_into_prepared_cache",
        fake_restore,
    )

    restore_state = (
        modal_cloud_module._install_prompt_executor_persisted_cache_restore(
            object(),
            executor,
            component_id="component-1",
            prompt={"12": {"class_type": "PersistentCacheNode", "inputs": {}}},
            cache_store={"NC_example": {"version": 1}},
        )
    )

    try:
        asyncio.run(outputs_cache.set_prompt(object(), ["12"], object()))
    finally:
        restore_state.restore_original_method()

    assert restore_state.restored_node_ids == ["12"]
    assert restore_state.restored_cache_keys_by_node_id == {"12": "NC_example"}
    assert observed_events == [
        ("restore", "live-cache-key-set", ("12",), {"NC_example": {"version": 1}})
    ]
    assert outputs_cache.set_prompt.__func__ is FakeOutputsCache.set_prompt

def test_modal_cloud_skips_rewriting_restored_distributed_cache_entries(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Persist should skip distributed cache entries that were restored unchanged this run."""
    monkeypatch.setitem(sys.modules, "torchsde", types.ModuleType("torchsde"))
    modal_cloud_module._ensure_comfy_runtime_initialized(None)

    execution = modal_cloud_module._load_execution_module()
    cache_entry = execution.CacheEntry(ui={"output": {"value": [5]}}, outputs=[[5]])
    cache_key = "NC_existing"
    cache_store: dict[str, Any] = {cache_key: {"version": 1, "outputs_zlib": b"old"}}
    observed_logs: list[tuple[Any, ...]] = []

    class FakeOutputsCache:
        """Minimal outputs cache stub for persist-phase tests."""

        def __init__(self) -> None:
            """Populate one persistent cache entry."""
            self.cache_key_set = object()

        async def get(self, node_id: str) -> Any:
            """Return the prepared cache entry for the target node only."""
            if node_id == "node_1":
                return cache_entry
            return None

    executor = types.SimpleNamespace(caches=types.SimpleNamespace(outputs=FakeOutputsCache()))

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_key_from_key_set_sync",
        lambda cache_key_set, node_id: cache_key if node_id == "node_1" else None,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_emit_cloud_info",
        lambda message, *args: observed_logs.append((message, *args)),
    )

    persisted_nodes = asyncio.run(
        modal_cloud_module._persist_node_output_cache_entries(
            executor,
            prompt={"node_1": {"class_type": "PersistentCacheNode", "inputs": {"value": 4}}},
            cache_store=cache_store,
            restored_cache_keys_by_node_id={"node_1": cache_key},
        )
    )

    assert persisted_nodes == []
    assert cache_store == {cache_key: {"version": 1, "outputs_zlib": b"old"}}
    assert observed_logs[-1] == (
        "Node output cache write node=%s key_prefix=%s result=skip reason=restored-hit",
        "node_1",
        "NC_existing",
    )

def test_modal_cloud_skips_restored_hit_before_cache_entry_serialization(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Persist should avoid cache-entry serialization entirely for restored-hit keys."""
    cache_entry = object()

    class FakeOutputsCache:
        """Minimal outputs cache stub for persist fast-path tests."""

        def __init__(self) -> None:
            """Expose a cache-key-set marker for sync key generation."""
            self.cache_key_set = object()

        def get(self, node_id: str) -> Any:
            """Return a placeholder cache entry for the target node."""
            if node_id == "node_1":
                return cache_entry
            return None

    executor = types.SimpleNamespace(caches=types.SimpleNamespace(outputs=FakeOutputsCache()))
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_key_from_key_set_sync",
        lambda cache_key_set, node_id: "NC_existing" if node_id == "node_1" else None,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_serialize_node_output_cache_entry",
        lambda cache_entry, max_bytes: (_ for _ in ()).throw(
            AssertionError("restored-hit should skip before serialization")
        ),
    )

    persisted_nodes = asyncio.run(
        modal_cloud_module._persist_node_output_cache_entries(
            executor,
            prompt={"node_1": {"class_type": "PersistentCacheNode", "inputs": {"value": 4}}},
            cache_store={},
            restored_cache_keys_by_node_id={"node_1": "NC_existing"},
        )
    )

    assert persisted_nodes == []

def test_modal_cloud_restores_persisted_node_cache_entries_in_parallel(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Prepared-cache restore should overlap independent distributed lookups."""
    in_flight_gets = 0
    max_in_flight_gets = 0
    restored_values: dict[str, Any] = {}

    class FakeOutputsCache:
        """Minimal outputs cache stub for restore concurrency tests."""

        def __init__(self) -> None:
            """Expose the cache-key-set marker read by restore."""
            self.cache_key_set = object()

        async def get(self, node_id: str) -> Any:
            """Return any previously restored entry."""
            return restored_values.get(node_id)

        async def set(self, node_id: str, cache_entry: Any) -> None:
            """Record one restored cache entry."""
            restored_values[node_id] = cache_entry

    async def fake_key_from_key_set(cache_key_set: Any, node_id: str) -> str:
        """Yield briefly so multiple node lookups can queue together."""
        del cache_key_set
        await asyncio.sleep(0)
        return f"NC_{node_id}"

    async def fake_store_get(cache_store: Any, cache_key: str) -> Any:
        """Track how many distributed cache reads overlap."""
        nonlocal in_flight_gets, max_in_flight_gets
        del cache_store
        in_flight_gets += 1
        max_in_flight_gets = max(max_in_flight_gets, in_flight_gets)
        await asyncio.sleep(0.01)
        in_flight_gets -= 1
        return {"cache_key": cache_key}

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_key_from_key_set_async",
        fake_key_from_key_set,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_store_get",
        fake_store_get,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_deserialize_node_output_cache_entry",
        lambda execution, record: record,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(), "_emit_cloud_info", lambda *args: None
    )

    outputs_cache = FakeOutputsCache()
    restored_node_ids = asyncio.run(
        modal_cloud_module._restore_persisted_node_output_cache_entries_into_prepared_cache(
            object(),
            outputs_cache,
            prompt={
                "node_1": {"class_type": "PersistentCacheNode", "inputs": {"value": 1}},
                "node_2": {"class_type": "PersistentCacheNode", "inputs": {"value": 2}},
                "node_3": {"class_type": "PersistentCacheNode", "inputs": {"value": 3}},
            },
            cache_store={},
        )
    )

    assert restored_node_ids == ["node_1", "node_2", "node_3"]
    assert max_in_flight_gets >= 2
    assert restored_values == {
        "node_1": {"cache_key": "NC_node_1"},
        "node_2": {"cache_key": "NC_node_2"},
        "node_3": {"cache_key": "NC_node_3"},
    }

def test_modal_cloud_creates_default_custom_nodes_dir_when_missing(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The remote runtime should create an empty default custom_nodes directory for ComfyUI."""
    comfyui_root = tmp_path / "comfyui"
    comfyui_root.mkdir()
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    monkeypatch.setattr(bootstrap_owner, "_REMOTE_COMFYUI_ROOT", comfyui_root)
    monkeypatch.setattr(bootstrap_owner, "_LOCAL_COMFYUI_ROOT", tmp_path / "missing-local")

    custom_nodes_dir = modal_cloud_module._ensure_default_custom_nodes_dir()

    assert custom_nodes_dir == comfyui_root / "custom_nodes"
    assert custom_nodes_dir is not None
    assert custom_nodes_dir.exists()
    assert custom_nodes_dir.is_dir()
