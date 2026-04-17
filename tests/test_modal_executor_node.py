"""Tests for dynamic Modal proxy nodes and local execution fallback."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


class _FakeOriginalNode:
    """Simple fake legacy node for proxy signature mirroring."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "count")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "run"

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        """Return a tuple that exposes the inputs for verification."""
        return (kwargs["value"], 1)


def test_dynamic_proxy_node_preserves_output_signature(
    modal_executor_module: Any,
) -> None:
    """Dynamic Modal proxies should mirror the original output count and names."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )

    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]
    schema = proxy_class.GET_SCHEMA()

    assert schema.node_id == proxy_id
    assert [output.display_name for output in schema.outputs] == ["image", "count"]
    assert [output.io_type for output in schema.outputs] == ["IMAGE", "INT"]


def test_proxy_execution_uses_injected_remote_client(
    modal_executor_module: Any,
) -> None:
    """Proxy execution should delegate to the configured remote client."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]

    class FakeClient:
        """Test client that returns deterministic outputs."""

        def execute_payload(self, payload: dict[str, Any], kwargs: dict[str, Any]) -> tuple[str, int]:
            """Return values derived from the proxied node payload."""
            return (f"{payload['class_type']}::{kwargs['value']}", 3)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = proxy_class.execute(original_node_data={"class_type": "OriginalNode"}, value="payload")
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("OriginalNode::payload", 3)


def test_local_remote_app_executes_original_node(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback remote app should execute a mapped legacy node."""
    payload = remote_modal_app_module.execute_node_locally(
        node_data={"class_type": "OriginalNode"},
        kwargs_payload='{"value": "hello"}',
        node_mapping={"OriginalNode": _FakeOriginalNode},
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == ("hello", 1)


def test_stable_modal_cloud_entry_imports_without_modal_sdk(
    modal_cloud_module: Any,
) -> None:
    """The stable Modal cloud module should stay importable when modal is unavailable."""
    assert modal_cloud_module.__name__ == "comfyui_modal_sync_cloud"
    assert hasattr(modal_cloud_module, "RemoteEngine")


def test_modal_cloud_ignores_heavy_comfyui_paths(
    modal_cloud_module: Any,
) -> None:
    """The Modal cloud module should skip heavyweight ComfyUI runtime artifacts."""
    from pathlib import Path

    assert modal_cloud_module._should_ignore_comfyui_path(Path("models/checkpoint.safetensors"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("output/run/output.png"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("__pycache__/execution.pyc"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("execution.py"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("comfy/model_management.py"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("comfy/ldm/models/diffusion/ddpm.py"))


def test_modal_cloud_installs_comfyui_runtime_packages(
    modal_cloud_module: Any,
) -> None:
    """The Modal cloud image should include the core packages ComfyUI imports at runtime."""
    packages = set(modal_cloud_module._comfyui_runtime_packages())

    assert "psutil" in packages
    assert "torchsde" in packages
    assert "transformers" in packages
    assert "sentencepiece" in packages
    assert "aiohttp" in packages
    assert "opencv-python-headless" in packages
    assert "comfy-kitchen>=0.2.7" in packages


def test_modal_cloud_pins_cu128_pytorch_stack(
    modal_cloud_module: Any,
) -> None:
    """The Modal cloud image should pin the PyTorch stack to the CUDA 12.8 wheel index."""
    packages = modal_cloud_module._comfyui_torch_packages()

    assert packages == (
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
    )
    assert modal_cloud_module._PYTORCH_CUDA_INDEX_URL == "https://download.pytorch.org/whl/cu128"


def test_modal_cloud_builds_snapshot_enabled_cls_options(
    modal_cloud_module: Any,
) -> None:
    """The remote engine should default to CPU memory snapshots and optional GPU snapshots."""
    base_settings = types.SimpleNamespace(
        remote_storage_root="/storage",
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
    )

    options = modal_cloud_module._remote_engine_cls_options(base_settings, "volume", "image")

    assert options["enable_memory_snapshot"] is True
    assert "experimental_options" not in options
    assert options["volumes"] == {"/storage": "volume"}

    gpu_snapshot_settings = types.SimpleNamespace(
        remote_storage_root="/storage",
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=True,
    )
    gpu_snapshot_options = modal_cloud_module._remote_engine_cls_options(
        gpu_snapshot_settings,
        "volume",
        "image",
    )
    assert gpu_snapshot_options["experimental_options"] == {"enable_gpu_snapshot": True}


def test_remote_modal_requires_deployed_app_by_default(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote mode should reject slow ephemeral fallback unless explicitly enabled."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    class FakeModal:
        """Minimal modal SDK double with deployed lookup failure types."""

        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Simulate a missing deployed app."""
                raise FakeLookupError("not deployed")

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK", "false")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        try:
            remote_modal_app_module._invoke_modal_payload_blocking(
                {"component_id": "component-1"},
                b"{}",
            )
        except remote_modal_app_module.ModalRemoteInvocationError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ModalRemoteInvocationError to be raised.")
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert "requires a deployed Modal app" in message
    assert "COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true" in message


def test_modal_cloud_initializes_remote_comfy_runtime_once_per_custom_node_root(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """The remote runtime should load built-in extras once and custom bundles per extracted root."""
    init_calls: list[tuple[Any, ...]] = []
    folder_path_calls: list[tuple[str, str, bool]] = []

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

    original_base_initialized = modal_cloud_module._COMFY_RUNTIME_BASE_INITIALIZED
    original_custom_node_roots = set(modal_cloud_module._COMFY_RUNTIME_CUSTOM_NODE_ROOTS)
    modal_cloud_module._COMFY_RUNTIME_BASE_INITIALIZED = False
    modal_cloud_module._COMFY_RUNTIME_CUSTOM_NODE_ROOTS.clear()
    try:
        custom_nodes_root = tmp_path / "custom_nodes"
        custom_nodes_root.mkdir()

        modal_cloud_module._ensure_comfy_runtime_initialized(None)
        modal_cloud_module._ensure_comfy_runtime_initialized(custom_nodes_root)
        modal_cloud_module._ensure_comfy_runtime_initialized(custom_nodes_root)
    finally:
        modal_cloud_module._COMFY_RUNTIME_BASE_INITIALIZED = original_base_initialized
        modal_cloud_module._COMFY_RUNTIME_CUSTOM_NODE_ROOTS.clear()
        modal_cloud_module._COMFY_RUNTIME_CUSTOM_NODE_ROOTS.update(original_custom_node_roots)

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


def test_modal_cloud_materializes_synced_asset_paths(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote asset references should resolve to absolute files under the storage root."""
    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", "/storage")
    modal_cloud_module.get_settings.cache_clear()
    try:
        assert modal_cloud_module._materialize_remote_asset_path("/assets/model.safetensors") == (
            "/storage/assets/model.safetensors"
        )
        assert modal_cloud_module._rewrite_modal_asset_references(
            {"clip_name": "/assets/model.safetensors", "nested": ["/assets/other.pt", 3]}
        ) == {
            "clip_name": "/storage/assets/model.safetensors",
            "nested": ["/storage/assets/other.pt", 3],
        }
    finally:
        modal_cloud_module.get_settings.cache_clear()


def test_modal_cloud_accepts_absolute_asset_paths_in_folder_lookup(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Patched folder lookups should return already-materialized absolute asset paths."""
    remote_storage_root = tmp_path / "storage"
    asset_path = remote_storage_root / "assets" / "clip.safetensors"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"clip")

    fake_folder_paths_module = types.SimpleNamespace(
        get_full_path=lambda folder_name, filename: None,
        get_full_path_or_raise=lambda folder_name, filename: (_ for _ in ()).throw(
            FileNotFoundError(filename)
        ),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths_module)
    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(remote_storage_root))
    modal_cloud_module.get_settings.cache_clear()
    try:
        with modal_cloud_module._patched_folder_paths_absolute_lookup():
            resolved = fake_folder_paths_module.get_full_path(
                "text_encoders",
                "/assets/clip.safetensors",
            )
            assert resolved == str(asset_path)
            assert (
                fake_folder_paths_module.get_full_path_or_raise(
                    "text_encoders",
                    "/assets/clip.safetensors",
                )
                == str(asset_path)
            )
    finally:
        modal_cloud_module.get_settings.cache_clear()


def test_modal_cloud_force_imports_comfyui_utils_package(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """ComfyUI's utils package should override a shadowing non-package module."""
    package_root = tmp_path / "comfyui"
    utils_dir = package_root / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "__init__.py").write_text("SENTINEL = 'comfy-utils'\n", encoding="utf-8")

    shadow_module = types.ModuleType("utils")
    shadow_module.__file__ = str(tmp_path / "utils.py")
    monkeypatch.setitem(sys.modules, "utils", shadow_module)

    modal_cloud_module._force_import_package_from_root("utils", package_root)

    imported_module = sys.modules["utils"]
    assert getattr(imported_module, "SENTINEL", None) == "comfy-utils"
    assert list(getattr(imported_module, "__path__", [])) == [str(utils_dir)]


class _BoundarySourceNode:
    """Simple source node used for subgraph execution tests."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal V1 input schema."""
        return {"required": {"value": ("INT",)}}

    def run(self, value: int) -> tuple[int]:
        """Increment the boundary input."""
        return (value + 1,)


class _BoundarySinkNode:
    """Simple downstream node used for subgraph execution tests."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal V1 input schema."""
        return {"required": {"value": ("INT",)}}

    def run(self, value: int) -> tuple[int]:
        """Double the upstream value."""
        return (value * 2,)


def test_local_remote_app_executes_subgraph_payload(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback remote app should execute rewritten subgraph payloads."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "BoundarySource",
                    "inputs": {"value": 0},
                    "_meta": {},
                },
                "remote_2": {
                    "class_type": "BoundarySink",
                    "inputs": {"value": ["remote_1", 0]},
                    "_meta": {},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_0",
                    "targets": [{"node_id": "remote_1", "input_name": "value"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "remote_2_value",
                    "node_id": "remote_2",
                    "output_index": 0,
                    "io_type": "INT",
                    "is_list": False,
                }
            ],
            "execute_node_ids": ["remote_2"],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload='{"remote_input_0": 4}',
        node_mapping={
            "BoundarySource": _BoundarySourceNode,
            "BoundarySink": _BoundarySinkNode,
        },
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (10,)
