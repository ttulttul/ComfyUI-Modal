"""Tests for dynamic Modal proxy nodes and local execution fallback."""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import logging

import pytest


class _FakeOriginalNode:
    """Simple fake legacy node for proxy signature mirroring."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "count")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "run"

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        """Return a tuple that exposes the inputs for verification."""
        return (kwargs["value"], 1)


class _CloneableCacheValue:
    """Simple cloneable object used for loader cache tests."""

    def __init__(self, value: str) -> None:
        """Store an identifying value for later clone assertions."""
        self.value = value

    def clone(self) -> "_CloneableCacheValue":
        """Return a fresh object carrying the same value."""
        return _CloneableCacheValue(self.value)


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


def test_modal_cloud_installs_timestamped_logger_handler(
    modal_cloud_module: Any,
) -> None:
    """The cloud runtime should install its own timestamped logger handler."""
    matching_handlers = [
        handler
        for handler in modal_cloud_module.logger.handlers
        if getattr(handler, "name", "") == modal_cloud_module._CLOUD_HANDLER_NAME
    ]

    assert len(matching_handlers) == 1
    assert modal_cloud_module.logger.propagate is False
    assert modal_cloud_module.logger.level == logging.INFO
    assert matching_handlers[0].stream is sys.stdout
    formatter = matching_handlers[0].formatter
    assert isinstance(formatter, logging.Formatter)
    assert "%(asctime)s" in formatter._fmt
    assert "%(relativeCreated)" in formatter._fmt


def test_modal_cloud_mirrors_phase_logs_to_stdout_in_modal_runtime(
    modal_cloud_module: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Timed cloud phases should write directly to stdout inside Modal containers."""
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")

    with modal_cloud_module._timed_phase("phase_under_test", component="component-1"):
        pass

    captured = capsys.readouterr()
    assert "Starting phase_under_test component=component-1" in captured.out
    assert "Finished phase_under_test in " in captured.out
    assert "component=component-1" in captured.out


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
    original_cache = dict(modal_cloud_module._EXTRACTED_CUSTOM_NODE_BUNDLES)
    modal_cloud_module._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
    try:
        first_root = modal_cloud_module._extract_custom_nodes_bundle("/custom_nodes/bundle.zip")
        monkeypatch.setattr(
            modal_cloud_module.zipfile,
            "ZipFile",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Expected cached extraction root to be reused.")
            ),
        )
        second_root = modal_cloud_module._extract_custom_nodes_bundle("/custom_nodes/bundle.zip")
    finally:
        modal_cloud_module.get_settings.cache_clear()
        modal_cloud_module._EXTRACTED_CUSTOM_NODE_BUNDLES.clear()
        modal_cloud_module._EXTRACTED_CUSTOM_NODE_BUNDLES.update(original_cache)

    assert first_root is not None
    assert second_root == first_root


def test_modal_cloud_traces_remote_node_execution_spans(
    modal_cloud_module: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """The tracing prompt server should emit per-node timing lines."""
    prompt = {
        "7": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {}},
    }
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    server = modal_cloud_module._TracingPromptServer("component-1", prompt)

    server.send_sync("executing", {"node": "7"}, None)
    server.send_sync("executed", {"node": "7"}, None)
    server.send_sync("executing", {"node": "2"}, None)
    server.send_sync("execution_success", {"prompt_id": "component-1"}, None)

    captured = capsys.readouterr()
    assert "Remote node 7 class_type=UNETLoader role=model_load started" in captured.out
    assert "Remote node 7 class_type=UNETLoader role=model_load finished in " in captured.out
    assert "Remote node 2 class_type=KSampler role=sampling started" in captured.out
    assert "Remote node 2 class_type=KSampler role=sampling finished in " in captured.out


def test_modal_cloud_installs_headless_prompt_server_instance(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote custom-node init should get a minimal PromptServer.instance shim."""
    fake_prompt_server_class = type("PromptServer", (), {})
    fake_server_module = types.SimpleNamespace(PromptServer=fake_prompt_server_class)

    monkeypatch.setitem(sys.modules, "server", fake_server_module)
    modal_cloud_module._ensure_headless_prompt_server_instance()

    instance = fake_prompt_server_class.instance
    assert instance is not None
    assert hasattr(instance, "routes")
    assert hasattr(instance, "app")
    assert instance.supports == ["custom_nodes_from_web"]
    assert instance.client_id is None
    assert instance.last_node_id is None

    instance.add_on_prompt_handler("handler")
    assert instance.on_prompt_handlers == ["handler"]


def test_modal_cloud_streams_progress_and_result_events(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The cloud runtime should stream progress envelopes before the final result."""
    progress_callbacks: list[dict[str, Any]] = []

    def fake_execute_subgraph_locally(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        status_callback: Any = None,
    ) -> bytes:
        if status_callback is not None:
            status_callback(
                {
                    "phase": "executing",
                    "active_node_id": "7",
                    "active_node_class_type": "UNETLoader",
                    "active_node_role": "model_load",
                }
            )
            status_callback({"phase": "execution_success"})
        progress_callbacks.append({"component_id": payload["component_id"], "kwargs": kwargs_payload})
        return b"serialized-outputs"

    monkeypatch.setattr(modal_cloud_module, "execute_subgraph_locally", fake_execute_subgraph_locally)

    events = list(
        modal_cloud_module._stream_remote_payload_events(
            {"payload_kind": "subgraph", "component_id": "component-1"},
            b"{}",
        )
    )

    assert progress_callbacks == [{"component_id": "component-1", "kwargs": b"{}"}]
    assert events == [
        {
            "kind": "progress",
            "phase": "executing",
            "active_node_id": "7",
            "active_node_class_type": "UNETLoader",
            "active_node_role": "model_load",
        },
        {
            "kind": "progress",
            "phase": "execution_success",
        },
        {
            "kind": "result",
            "outputs": b"serialized-outputs",
        },
    ]


def test_modal_cloud_streams_tensor_safe_progress_and_result_events(
    modal_cloud_module: Any,
    monkeypatch: Any,
    serialization_module: Any,
) -> None:
    """Streamed Modal events should serialize stray tensor payloads before yielding them."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    def fake_execute_subgraph_locally(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        status_callback: Any = None,
    ) -> tuple[Any, ...]:
        if status_callback is not None:
            status_callback(
                {
                    "phase": "executing",
                    "active_node_id": "7",
                    "preview_tensor": tensor,
                }
            )
        return (tensor,)

    monkeypatch.setattr(modal_cloud_module, "execute_subgraph_locally", fake_execute_subgraph_locally)

    events = list(
        modal_cloud_module._stream_remote_payload_events(
            {"payload_kind": "subgraph", "component_id": "component-1"},
            b"{}",
        )
    )

    assert events[0]["kind"] == "progress"
    assert events[0]["phase"] == "executing"
    assert torch.equal(
        serialization_module.deserialize_value(events[0]["preview_tensor"]),
        tensor,
    )

    assert events[1]["kind"] == "result"
    decoded_outputs = serialization_module.deserialize_node_outputs(events[1]["outputs"])
    assert len(decoded_outputs) == 1
    assert torch.equal(decoded_outputs[0], tensor)


def test_modal_cloud_only_reloads_volume_for_requests_with_new_uploads(
    modal_cloud_module: Any,
) -> None:
    """Steady-state requests should skip Modal volume reload when queue-time sync uploaded nothing."""

    assert modal_cloud_module._should_reload_modal_volume({"requires_volume_reload": True}) is True
    assert modal_cloud_module._should_reload_modal_volume({"requires_volume_reload": False}) is False
    assert modal_cloud_module._should_reload_modal_volume({}) is True


def test_modal_cloud_retries_volume_reload_after_clearing_warm_state(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal volume reload should retry once after unloading warm caches when open files block it."""

    class FakeVolume:
        """Simple Modal volume double that fails once before succeeding."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Raise on the first call and succeed on the second."""
            self.reload_calls += 1
            if self.reload_calls == 1:
                raise RuntimeError("there are open files preventing the operation")

    prepare_calls: list[str] = []
    monkeypatch.setattr(
        modal_cloud_module,
        "_prepare_for_modal_volume_reload",
        lambda: prepare_calls.append("prepared"),
    )

    volume = FakeVolume()
    modal_cloud_module._reload_modal_volume_for_request(volume, "component-1")

    assert volume.reload_calls == 2
    assert prepare_calls == ["prepared"]


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

    original_cache = dict(modal_cloud_module._LOADER_OUTPUT_CACHE)
    original_wrapped = set(modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES)
    modal_cloud_module._LOADER_OUTPUT_CACHE.clear()
    modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES.clear()
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
    finally:
        modal_cloud_module._LOADER_OUTPUT_CACHE.clear()
        modal_cloud_module._LOADER_OUTPUT_CACHE.update(original_cache)
        modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES.clear()
        modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES.update(original_wrapped)

    assert loader.calls == 2
    assert first.value == "model.safetensors:cpu:1"
    assert second.value == "model.safetensors:cpu:1"
    assert third.value == "other.safetensors:cpu:2"
    assert first is not second


def test_modal_cloud_installs_loader_cache_wrappers_for_builtin_loaders(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The runtime should patch the heavyweight built-in loaders once they are available."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "UNETLoader": type("UNETLoader", (), {"load_unet": lambda self, unet_name, weight_dtype="default": (unet_name,)}),
            "CLIPLoader": type("CLIPLoader", (), {"load_clip": lambda self, clip_name, type="stable_diffusion", device="default": (clip_name,)}),
            "VAELoader": type("VAELoader", (), {"load_vae": lambda self, vae_name: (vae_name,)}),
        }
    )

    original_wrapped = set(modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES)
    modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES.clear()
    monkeypatch.setattr(modal_cloud_module, "_load_nodes_module", lambda: fake_nodes_module)
    try:
        modal_cloud_module._install_loader_cache_wrappers()
        installed_wrappers = set(modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES)
    finally:
        modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES.clear()
        modal_cloud_module._LOADER_CACHE_WRAPPED_CLASSES.update(original_wrapped)

    assert {"UNETLoader", "CLIPLoader", "VAELoader"} <= installed_wrappers


def test_modal_cloud_ignores_heavy_comfyui_paths(
    modal_cloud_module: Any,
) -> None:
    """The Modal cloud module should skip heavyweight ComfyUI runtime artifacts."""
    from pathlib import Path

    assert modal_cloud_module._should_ignore_comfyui_path(Path("models/checkpoint.safetensors"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("custom_nodes/example/__init__.py"))
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
    assert "alembic" in packages
    assert "pydantic-settings" in packages
    assert "spandrel" in packages
    assert "kornia" in packages


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
        modal_gpu="L40S",
        scaledown_window_seconds=600,
        min_containers=0,
    )

    options = modal_cloud_module._remote_engine_cls_options(base_settings, "volume", "image")

    assert options["enable_memory_snapshot"] is True
    assert "experimental_options" not in options
    assert options["gpu"] == "L40S"
    assert options["volumes"] == {"/storage": "volume"}
    assert options["scaledown_window"] == 600
    assert options["min_containers"] == 0

    gpu_snapshot_settings = types.SimpleNamespace(
        remote_storage_root="/storage",
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=True,
        modal_gpu="A100",
        scaledown_window_seconds=900,
        min_containers=1,
    )
    gpu_snapshot_options = modal_cloud_module._remote_engine_cls_options(
        gpu_snapshot_settings,
        "volume",
        "image",
    )
    assert gpu_snapshot_options["experimental_options"] == {"enable_gpu_snapshot": True}
    assert gpu_snapshot_options["gpu"] == "A100"
    assert gpu_snapshot_options["scaledown_window"] == 900
    assert gpu_snapshot_options["min_containers"] == 1


def test_modal_cloud_prewarms_snapshot_state_without_gpu_runtime_by_default(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """CPU-only snapshot prewarm should avoid full ComfyUI runtime initialization."""
    calls: list[str] = []

    monkeypatch.setattr(
        modal_cloud_module,
        "_ensure_comfyui_support_packages",
        lambda: calls.append("support"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(f"runtime:{custom_nodes_root}"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_snapshot_state(
        types.SimpleNamespace(enable_gpu_memory_snapshot=False)
    )

    assert calls == ["support"]


def test_modal_cloud_prewarms_snapshot_state_fully_for_gpu_snapshots(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """GPU snapshot prewarm should initialize the remote runtime before snapshot capture."""
    calls: list[str] = []

    monkeypatch.setattr(
        modal_cloud_module,
        "_ensure_comfyui_support_packages",
        lambda: calls.append("support"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(f"runtime:{custom_nodes_root}"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_snapshot_state(
        types.SimpleNamespace(enable_gpu_memory_snapshot=True)
    )

    assert calls == ["support", "runtime:None", "execution"]


def test_modal_cloud_prewarms_restored_runtime(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Post-restore prewarm should fully initialize the request-serving runtime."""
    calls: list[str] = []

    monkeypatch.setattr(
        modal_cloud_module,
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(f"runtime:{custom_nodes_root}"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_restored_runtime()

    assert calls == ["runtime:None", "execution"]


def test_remote_modal_auto_deploys_missing_app_by_default(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote mode should auto-deploy the stable Modal app on first lookup failure."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []

    class FakeExecuteMethod:
        """Minimal Modal method handle that records remote calls."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Return deterministic remote bytes."""
            return b"remote-response"

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance."""

        execute_payload = FakeExecuteMethod()

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(self, *, name: str | None = None, environment_name: str | None = None, **_: Any) -> "FakeApp":
            """Record the deploy request and mark the deployment available."""
            deploy_calls.append((name, environment_name))
            FakeModal.deployed = True
            return self

    class FakeModal:
        """Minimal modal SDK double with deployed lookup failure types."""

        deployed = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a deployed class after the first auto-deploy."""
                if not FakeModal.deployed:
                    raise FakeLookupError("not deployed")
                return lambda: FakeRemoteEngine()

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(
        remote_modal_app_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    remote_modal_app_module.get_settings.cache_clear()
    remote_modal_app_module._MODAL_AUTO_DEPLOYED_APPS.clear()
    try:
        response = remote_modal_app_module._invoke_modal_payload_blocking(
            {"component_id": "component-1"},
            b"{}",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        remote_modal_app_module._MODAL_AUTO_DEPLOYED_APPS.clear()

    assert response == b"remote-response"
    assert deploy_calls == [("comfy-modal-sync", None)]


def test_remote_modal_consumes_streamed_progress_and_result(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """The local Modal client should forward streamed progress events into the UI websocket."""

    class FakePromptServer:
        """Capture websocket events emitted by streamed remote progress."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[str, dict[str, Any], str | None]] = []

        def send_sync(self, event: str, data: dict[str, Any], sid: str | None) -> None:
            """Record one emitted websocket message."""
            self.messages.append((event, data, sid))

    prompt_server = FakePromptServer()
    monkeypatch.setattr(remote_modal_app_module, "_lookup_local_prompt_server", lambda: prompt_server)

    payload = {
        "prompt_id": "prompt-1",
        "component_id": "component-1",
        "component_node_ids": ["7", "8"],
        "extra_data": {"client_id": "client-1"},
    }
    result = remote_modal_app_module._consume_remote_payload_stream(
        payload,
        iter(
            [
                {
                    "kind": "progress",
                    "phase": "executing",
                    "active_node_id": "7",
                    "active_node_class_type": "UNETLoader",
                    "active_node_role": "model_load",
                },
                {
                    "kind": "progress",
                    "phase": "execution_success",
                },
                {
                    "kind": "result",
                    "outputs": b"serialized-outputs",
                },
            ]
        ),
    )

    assert result == b"serialized-outputs"
    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["7", "8"],
                "active_node_id": "7",
                "active_node_class_type": "UNETLoader",
                "active_node_role": "model_load",
            },
            "client-1",
        ),
        (
            "modal_status",
            {
                "phase": "execution_success",
                "prompt_id": "prompt-1",
                "node_ids": ["7", "8"],
            },
            "client-1",
        ),
    ]


def test_remote_modal_consumes_streamed_tensor_result_payload(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local stream consumer should accept JSON-safe serialized tensor outputs."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(5, dtype=torch.float32)

    result = remote_modal_app_module._consume_remote_payload_stream(
        {
            "prompt_id": "prompt-1",
            "component_id": "component-1",
            "component_node_ids": ["7"],
            "extra_data": {"client_id": "client-1"},
        },
        iter(
            [
                {
                    "kind": "result",
                    "outputs": [serialization_module.serialize_value(tensor)],
                },
            ]
        ),
    )

    decoded_outputs = serialization_module.deserialize_node_outputs(result)
    assert len(decoded_outputs) == 1
    assert torch.equal(decoded_outputs[0], tensor)


def test_remote_modal_requires_manual_deploy_when_auto_deploy_disabled(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote mode should fail clearly when auto-deploy and ephemeral fallback are both disabled."""

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
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "false")
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

    assert "requires a deployed Modal app or a successful first-run auto-deploy" in message
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

    original_states = dict(modal_cloud_module._PROMPT_EXECUTOR_STATES)
    modal_cloud_module._PROMPT_EXECUTOR_STATES.clear()
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
        modal_cloud_module._PROMPT_EXECUTOR_STATES.clear()
        modal_cloud_module._PROMPT_EXECUTOR_STATES.update(original_states)

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

    original_states = dict(modal_cloud_module._PROMPT_EXECUTOR_STATES)
    modal_cloud_module._PROMPT_EXECUTOR_STATES.clear()
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
        modal_cloud_module._PROMPT_EXECUTOR_STATES.clear()
        modal_cloud_module._PROMPT_EXECUTOR_STATES.update(original_states)

    assert FakePromptExecutor.instances_created == 2
    assert first_state is not second_state


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


def test_modal_cloud_creates_default_custom_nodes_dir_when_missing(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The remote runtime should create an empty default custom_nodes directory for ComfyUI."""
    comfyui_root = tmp_path / "comfyui"
    comfyui_root.mkdir()
    monkeypatch.setattr(modal_cloud_module, "_REMOTE_COMFYUI_ROOT", comfyui_root)
    monkeypatch.setattr(modal_cloud_module, "_LOCAL_COMFYUI_ROOT", tmp_path / "missing-local")

    custom_nodes_dir = modal_cloud_module._ensure_default_custom_nodes_dir()

    assert custom_nodes_dir == comfyui_root / "custom_nodes"
    assert custom_nodes_dir is not None
    assert custom_nodes_dir.exists()
    assert custom_nodes_dir.is_dir()


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
