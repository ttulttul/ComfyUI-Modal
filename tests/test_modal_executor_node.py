"""Tests for dynamic Modal proxy nodes and local execution fallback."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
import threading
import time
import types
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import logging
from typing import Any, Iterator

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
    assert proxy_class.INPUT_IS_LIST is True


def test_proxy_execution_uses_injected_remote_client(
    modal_executor_module: Any,
) -> None:
    """Proxy execution should delegate to the configured remote client asynchronously."""
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

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str, int]:
            """Return values derived from the proxied node payload."""
            return (f"{payload['class_type']}::{kwargs['value']}", 3)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = asyncio.run(
            proxy_class.execute(original_node_data={"class_type": "OriginalNode"}, value="payload")
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("OriginalNode::payload", 3)


def test_proxy_execution_normalizes_input_is_list_kwargs(
    modal_executor_module: Any,
) -> None:
    """Dynamic Modal proxies should unwrap singleton INPUT_IS_LIST wrappers before remote execution."""
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
    observed_kwargs: dict[str, Any] = {}

    class FakeClient:
        """Test client that records normalized proxy kwargs."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str, int]:
            """Capture the kwargs forwarded by the proxy."""
            observed_kwargs["payload_kind"] = payload["payload_kind"]
            observed_kwargs.update(kwargs)
            return ("ok", 1)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = asyncio.run(
            proxy_class.execute(
                original_node_data=[{"payload_kind": "subgraph", "class_type": "OriginalNode"}],
                scalar_value=[3],
                mapped_value=["a", "b", "c"],
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("ok", 1)
    assert observed_kwargs == {
        "payload_kind": "subgraph",
        "scalar_value": 3,
        "mapped_value": ["a", "b", "c"],
    }


def test_proxy_execution_wraps_sync_remote_clients(
    modal_executor_module: Any,
) -> None:
    """Async proxy execution should still support legacy sync remote clients."""
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

    class FakeSyncClient:
        """Legacy client that only exposes the blocking execution method."""

        def execute_payload(self, payload: dict[str, Any], kwargs: dict[str, Any]) -> tuple[str, int]:
            """Return values derived from the proxied node payload."""
            return (f"sync::{payload['class_type']}::{kwargs['value']}", 4)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeSyncClient())
    try:
        result = asyncio.run(
            proxy_class.execute(original_node_data={"class_type": "OriginalNode"}, value="payload")
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("sync::OriginalNode::payload", 4)


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


def test_remote_modal_call_worker_count_allows_parallel_blocking_calls(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """The local Modal dispatch pool should have more than one worker."""
    monkeypatch.setattr(remote_modal_app_module.os, "cpu_count", lambda: 8)
    remote_modal_app_module.get_settings.cache_clear()

    assert remote_modal_app_module._remote_modal_call_worker_count() == 8


def test_remote_modal_call_worker_count_honors_max_container_limit(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """The local Modal dispatch pool should scale up to the configured max container count."""
    monkeypatch.setattr(remote_modal_app_module.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("COMFY_MODAL_MAX_CONTAINERS", "12")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        assert remote_modal_app_module._remote_modal_call_worker_count() == 12
    finally:
        remote_modal_app_module.get_settings.cache_clear()


def test_ensure_remote_warm_capacity_deduplicates_prompt_slots(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Prompt-scoped proactive warmup should only schedule each target slot once."""
    submitted_tasks: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        """Minimal executor that records submitted warmup jobs."""

        def submit(self, fn: Any, *args: Any) -> None:
            """Capture one scheduled warmup task without running it."""
            submitted_tasks.append((fn, args))

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_REMOTE_MODAL_WARMUP_EXECUTOR", FakeExecutor())
    remote_modal_app_module.get_settings.cache_clear()
    with remote_modal_app_module._PROMPT_WARMUP_STATES_LOCK:
        remote_modal_app_module._PROMPT_WARMUP_STATES.clear()
        remote_modal_app_module._PROMPT_WARMUP_STATE_ORDER = None

    try:
        warmup_request = {"prompt_id": "prompt-1", "component_id": "component-1"}
        first_target = remote_modal_app_module.ensure_remote_warm_capacity(
            warmup_request,
            warmup_target=2,
            reason="queue_time",
        )
        second_target = remote_modal_app_module.ensure_remote_warm_capacity(
            warmup_request,
            warmup_target=2,
            reason="queue_time_repeat",
        )
        third_target = remote_modal_app_module.ensure_remote_warm_capacity(
            warmup_request,
            warmup_target=4,
            reason="runtime_top_up",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert first_target == 2
    assert second_target == 2
    assert third_target == 4
    assert len(submitted_tasks) == 4
    assert [args[1] for _fn, args in submitted_tasks] == [0, 1, 2, 3]


def test_register_exact_component_parallelism_refines_prompt_target(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped fan-out should raise the prompt-wide warmup target once exact item count is known."""
    monkeypatch.setenv("COMFY_MODAL_MAX_CONTAINERS", "6")
    remote_modal_app_module.get_settings.cache_clear()
    with remote_modal_app_module._PROMPT_WARMUP_STATES_LOCK:
        remote_modal_app_module._PROMPT_WARMUP_STATES.clear()
        remote_modal_app_module._PROMPT_WARMUP_STATE_ORDER = None

    try:
        payload = {
            "prompt_id": "prompt-2",
            "component_id": "component-a",
            "extra_data": {
                "modal": {
                    "component_execution_stages": [["component-a", "component-b"], ["component-c"]],
                    "mapped_component_ids": ["component-a"],
                    "estimated_max_parallel_requests": 2,
                }
            },
        }
        refined_target = remote_modal_app_module._register_exact_component_parallelism(payload, 5)
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert refined_target == 6


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
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": "7",
                    "display_node_id": "7",
                    "value": 3,
                    "max": 10,
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
            "event_type": "node_progress",
            "node_id": "7",
            "display_node_id": "7",
            "value": 3,
            "max": 10,
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


def test_modal_cloud_skips_reload_when_uploaded_paths_are_already_visible(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Visible immutable uploaded paths should not force an extra Modal volume reload."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    uploaded_path = storage_root / "assets" / "hash_model.safetensors"
    uploaded_path.parent.mkdir(parents=True, exist_ok=True)
    uploaded_path.write_bytes(b"weights")
    recorded_markers: list[str] = []

    monkeypatch.setattr(
        modal_cloud_module,
        "get_settings",
        lambda: types.SimpleNamespace(remote_storage_root=str(storage_root)),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_record_modal_volume_reload_marker",
        lambda marker: recorded_markers.append(marker),
    )

    payload = {
        "requires_volume_reload": True,
        "volume_reload_marker": "marker-1",
        "uploaded_volume_paths": ["/assets/hash_model.safetensors"],
    }

    assert modal_cloud_module._should_reload_modal_volume(payload) is False
    assert recorded_markers == ["marker-1"]


def test_modal_cloud_schedules_container_exit_for_remote_failures(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Unhandled remote execution failures should retire the current Modal container."""
    scheduled_exits: list[tuple[float, int]] = []
    original_flag = modal_cloud_module._CONTAINER_TERMINATION_SCHEDULED
    monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", False)
    monkeypatch.setattr(modal_cloud_module, "_is_modal_container_runtime", lambda: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_schedule_process_exit",
        lambda delay_seconds, exit_code: scheduled_exits.append((delay_seconds, exit_code)),
    )
    try:
        scheduled = modal_cloud_module._maybe_schedule_container_termination_on_error(
            {"component_id": "component-1", "terminate_container_on_error": True},
            RuntimeError("boom"),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is True
    assert scheduled_exits == [(modal_cloud_module._REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS, 1)]


def test_modal_cloud_does_not_schedule_container_exit_for_interruptions(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Expected interruption-style failures should not tear down the warm Modal container."""
    scheduled_exits: list[tuple[float, int]] = []
    original_flag = modal_cloud_module._CONTAINER_TERMINATION_SCHEDULED
    monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", False)
    monkeypatch.setattr(modal_cloud_module, "_is_modal_container_runtime", lambda: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_schedule_process_exit",
        lambda delay_seconds, exit_code: scheduled_exits.append((delay_seconds, exit_code)),
    )
    try:
        scheduled = modal_cloud_module._maybe_schedule_container_termination_on_error(
            {"component_id": "component-1", "terminate_container_on_error": True},
            modal_cloud_module.RemoteSubgraphExecutionError(
                "Remote subgraph execution was interrupted."
            ),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is False
    assert scheduled_exits == []


def test_modal_cloud_skips_duplicate_reload_markers_in_same_container(
    modal_cloud_module: Any,
) -> None:
    """One container should reload a given uploaded-asset marker only once."""

    class FakeVolume:
        """Simple Modal volume double that tracks reload calls."""

        def __init__(self) -> None:
            """Initialize the reload counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Record one reload attempt."""
            self.reload_calls += 1

    original_marker_queue = modal_cloud_module._MODAL_VOLUME_RELOAD_MARKERS
    original_marker_set = set(modal_cloud_module._MODAL_VOLUME_RELOAD_MARKER_SET)
    modal_cloud_module._MODAL_VOLUME_RELOAD_MARKERS = None
    modal_cloud_module._MODAL_VOLUME_RELOAD_MARKER_SET.clear()
    try:
        payload = {"requires_volume_reload": True, "volume_reload_marker": "marker-1"}
        assert modal_cloud_module._should_reload_modal_volume(payload) is True

        volume = FakeVolume()
        modal_cloud_module._reload_modal_volume_for_request(
            volume,
            "component-1",
            reload_marker="marker-1",
        )

        assert volume.reload_calls == 1
        assert modal_cloud_module._should_reload_modal_volume(payload) is False
    finally:
        modal_cloud_module._MODAL_VOLUME_RELOAD_MARKERS = original_marker_queue
        modal_cloud_module._MODAL_VOLUME_RELOAD_MARKER_SET.clear()
        modal_cloud_module._MODAL_VOLUME_RELOAD_MARKER_SET.update(original_marker_set)


def test_modal_cloud_retries_volume_reload_after_clearing_warm_state(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal volume reload should retry after unloading warm caches when open files block it."""

    class FakeVolume:
        """Simple Modal volume double that fails twice before succeeding."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Raise on the first two calls and succeed on the third."""
            self.reload_calls += 1
            if self.reload_calls < 3:
                raise RuntimeError("there are open files preventing the operation")

    prepare_calls: list[str] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        modal_cloud_module,
        "_prepare_for_modal_volume_reload",
        lambda: prepare_calls.append("prepared"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_sleep_before_modal_volume_reload_retry",
        lambda delay_seconds: sleep_calls.append(delay_seconds),
    )

    volume = FakeVolume()
    modal_cloud_module._reload_modal_volume_for_request(volume, "component-1")

    assert volume.reload_calls == 3
    assert prepare_calls == ["prepared", "prepared"]
    assert sleep_calls == [0.25, 0.5]


def test_modal_cloud_raises_after_exhausting_open_file_reload_retries(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal volume reload should surface persistent open-file errors after bounded retries."""

    class FakeVolume:
        """Simple Modal volume double that always fails with open files."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Always fail with the same open-file reload error."""
            self.reload_calls += 1
            raise RuntimeError("there are open files preventing the operation")

    prepare_calls: list[str] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        modal_cloud_module,
        "_prepare_for_modal_volume_reload",
        lambda: prepare_calls.append("prepared"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_sleep_before_modal_volume_reload_retry",
        lambda delay_seconds: sleep_calls.append(delay_seconds),
    )

    volume = FakeVolume()
    with pytest.raises(RuntimeError, match="open files"):
        modal_cloud_module._reload_modal_volume_for_request(volume, "component-1")

    assert volume.reload_calls == len(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS
    )
    assert prepare_calls == ["prepared"] * (
        len(modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS) - 1
    )
    assert sleep_calls == list(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS[1:]
    )


def test_modal_cloud_proceeds_when_referenced_volume_paths_are_already_visible(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Persistent open-file reload errors may be ignored when the payload's mounted files are already visible."""

    class FakeVolume:
        """Simple Modal volume double that always fails with open files."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Always fail with the same open-file reload error."""
            self.reload_calls += 1
            raise RuntimeError("there are open files preventing the operation")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    asset_path = storage_root / "assets" / "hash_model.safetensors"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"weights")
    bundle_path = storage_root / "custom_nodes" / "hash_bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(b"bundle")

    recorded_markers: list[str] = []
    prepare_calls: list[str] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        modal_cloud_module,
        "get_settings",
        lambda: types.SimpleNamespace(remote_storage_root=str(storage_root)),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_prepare_for_modal_volume_reload",
        lambda: prepare_calls.append("prepared"),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_sleep_before_modal_volume_reload_retry",
        lambda delay_seconds: sleep_calls.append(delay_seconds),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_record_modal_volume_reload_marker",
        lambda marker: recorded_markers.append(marker),
    )

    payload = {
        "custom_nodes_bundle": "/custom_nodes/hash_bundle.zip",
        "subgraph_prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "/assets/hash_model.safetensors"},
            }
        },
    }

    volume = FakeVolume()
    modal_cloud_module._reload_modal_volume_for_request(
        volume,
        "component-1",
        reload_marker="marker-1",
        payload=payload,
    )

    assert volume.reload_calls == len(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS
    )
    assert recorded_markers == ["marker-1"]
    assert prepare_calls == ["prepared"] * (
        len(modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS) - 1
    )
    assert sleep_calls == list(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS[1:]
    )


def test_modal_cloud_logs_volume_reload_diagnostics_for_open_file_retries(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Open-file retries should log which uploaded and referenced volume paths matter."""

    class FakeVolume:
        """Simple Modal volume double that always fails with open files."""

        def reload(self) -> None:
            """Always fail with the same open-file reload error."""
            raise RuntimeError("there are open files preventing the operation")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    uploaded_path = storage_root / "assets" / "missing_model.safetensors"
    logged_messages: list[tuple[str, tuple[Any, ...]]] = []

    monkeypatch.setattr(
        modal_cloud_module,
        "get_settings",
        lambda: types.SimpleNamespace(remote_storage_root=str(storage_root)),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_prepare_for_modal_volume_reload",
        lambda: None,
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_sleep_before_modal_volume_reload_retry",
        lambda delay_seconds: None,
    )
    monkeypatch.setattr(
        modal_cloud_module.logger,
        "info",
        lambda message, *args: logged_messages.append((message, args)),
    )

    payload = {
        "uploaded_volume_paths": ["/assets/missing_model.safetensors"],
        "subgraph_prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "/assets/missing_model.safetensors"},
            }
        },
    }

    with pytest.raises(RuntimeError, match="open files"):
        modal_cloud_module._reload_modal_volume_for_request(
            FakeVolume(),
            "component-1",
            payload=payload,
        )

    assert any(
        "Modal volume reload diagnostics for component=%s context=%s uploaded_paths=%s referenced_paths=%s visible_uploaded=%s visible_referenced=%s."
        in message
        and args[0] == "component-1"
        and args[1] == "open_files_retry"
        and args[2] == [str(uploaded_path)]
        and args[3] == [str(uploaded_path)]
        and args[4] is False
        and args[5] is False
        for message, args in logged_messages
    )


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
        modal_cloud_module,
        "_load_nodes_module",
        lambda: types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={"FakeSampler": type("FakeSampler", (), {})}
        ),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_include_unique_id_in_input_signature",
        lambda class_type: False,
    )

    rebuilt_key = modal_cloud_module._node_output_cache_key_from_key_set_sync(cache_key_set, "12")
    direct_bad_key = modal_cloud_module._node_output_cache_key(cache_key_set.get_data_key("12"))

    assert isinstance(rebuilt_key, str)
    assert rebuilt_key.startswith("NC_")
    assert direct_bad_key is None


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


def test_modal_cloud_prepares_warm_container_for_request(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Warmup requests should prime volume visibility and extracted custom nodes without executing a payload."""
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(modal_cloud_module, "_should_reload_modal_volume", lambda payload: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_reload_modal_volume_for_request",
        lambda volume, component_id, reload_marker=None, payload=None: calls.append(
            ("reload", component_id, reload_marker, payload.get("uploaded_volume_paths"))
        ),
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_extract_custom_nodes_bundle",
        lambda bundle_path: Path("/tmp/extracted-bundle") if bundle_path else None,
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_register_custom_nodes_root",
        lambda custom_nodes_root: calls.append(("register", custom_nodes_root)),
    )
    monkeypatch.setenv("MODAL_TASK_ID", "task-123")

    result = modal_cloud_module._prepare_warm_container_for_request(
        object(),
        {
            "component_id": "component-1",
            "volume_reload_marker": "marker-1",
            "uploaded_volume_paths": ["/storage/example.bin"],
            "custom_nodes_bundle": "custom_nodes_bundle.zip",
            "warmup_slot_index": 2,
        },
    )

    assert calls == [
        ("reload", "component-1", "marker-1", ["/storage/example.bin"]),
        ("register", Path("/tmp/extracted-bundle")),
    ]
    assert result == {
        "component_id": "component-1",
        "task_id": "task-123",
        "warmup_slot_index": 2,
        "reloaded_volume": True,
    }


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


def test_load_modal_cloud_module_reloads_stale_partial_module(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Stale partially imported cloud modules should be discarded and reloaded."""
    original_module = sys.modules.get(remote_modal_app_module._MODAL_CLOUD_MODULE_NAME)
    stale_module = types.SimpleNamespace(app=None)
    sys.modules[remote_modal_app_module._MODAL_CLOUD_MODULE_NAME] = stale_module

    loaded_module = types.SimpleNamespace(app="fresh-app")

    class FakeLoader:
        """Populate the fresh replacement module during exec."""

        def create_module(self, spec: Any) -> None:
            """Use the default module creation path."""
            del spec
            return None

        def exec_module(self, module: Any) -> None:
            """Install the expected deployable app onto the reloaded module."""
            module.app = loaded_module.app

    monkeypatch.setattr(
        remote_modal_app_module.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: importlib.util.spec_from_loader(
            remote_modal_app_module._MODAL_CLOUD_MODULE_NAME,
            FakeLoader(),
        ),
    )
    try:
        reloaded_module = remote_modal_app_module._load_modal_cloud_module()
    finally:
        sys.modules.pop(remote_modal_app_module._MODAL_CLOUD_MODULE_NAME, None)
        if original_module is not None:
            sys.modules[remote_modal_app_module._MODAL_CLOUD_MODULE_NAME] = original_module

    assert reloaded_module is not stale_module
    assert getattr(reloaded_module, "app", None) == "fresh-app"


def test_load_modal_cloud_module_clears_failed_import_from_sys_modules(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Failed cloud-module imports should not leave a poisoned cache entry behind."""
    original_module = sys.modules.get(remote_modal_app_module._MODAL_CLOUD_MODULE_NAME)

    class FakeLoader:
        """Raise during module execution to simulate a partial import failure."""

        def create_module(self, spec: Any) -> None:
            """Use the default module creation path."""
            del spec
            return None

        def exec_module(self, module: Any) -> None:
            """Fail while the module is being initialized."""
            module.app = None
            raise RuntimeError("boom")

    monkeypatch.setattr(
        remote_modal_app_module.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: importlib.util.spec_from_loader(
            remote_modal_app_module._MODAL_CLOUD_MODULE_NAME,
            FakeLoader(),
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="boom"):
            remote_modal_app_module._load_modal_cloud_module()
        assert remote_modal_app_module._MODAL_CLOUD_MODULE_NAME not in sys.modules
    finally:
        sys.modules.pop(remote_modal_app_module._MODAL_CLOUD_MODULE_NAME, None)
        if original_module is not None:
            sys.modules[remote_modal_app_module._MODAL_CLOUD_MODULE_NAME] = original_module


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
                    "event_type": "node_progress",
                    "node_id": "7",
                    "display_node_id": "7",
                    "value": 4,
                    "max": 20,
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
            "modal_progress",
            {
                "prompt_id": "prompt-1",
                "node_id": "7",
                "display_node_id": "7",
                "value": 4.0,
                "max": 20.0,
            },
            "client-1",
        ),
        (
            "modal_status",
            {
                "phase": "finalizing",
                "prompt_id": "prompt-1",
                "node_ids": ["7", "8"],
                "status_message": "Receiving Modal outputs",
            },
            "client-1",
        ),
    ]


def test_remote_modal_consumes_streamed_executed_outputs_and_previews(
    remote_modal_app_module: Any,
    monkeypatch: Any,
    serialization_module: Any,
) -> None:
    """The local Modal client should forward streamed executed outputs and previews."""
    from PIL import Image
    from protocol import BinaryEventTypes

    class FakePromptServer:
        """Capture websocket events emitted by streamed remote UI output updates."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[Any, Any, str | None]] = []

        def send_sync(self, event: Any, data: Any, sid: str | None) -> None:
            """Record one emitted websocket message."""
            self.messages.append((event, data, sid))

    preview_buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(preview_buffer, format="PNG")

    prompt_server = FakePromptServer()
    monkeypatch.setattr(remote_modal_app_module, "_lookup_local_prompt_server", lambda: prompt_server)

    payload = {
        "prompt_id": "prompt-1",
        "component_id": "component-1",
        "component_node_ids": ["7"],
        "extra_data": {"client_id": "client-1"},
    }
    result = remote_modal_app_module._consume_remote_payload_stream(
        payload,
        iter(
            [
                {
                    "kind": "progress",
                    "event_type": "executed",
                    "node_id": "7",
                    "display_node_id": "7",
                    "output": {
                        "images": [
                            {
                                "filename": "preview.png",
                                "subfolder": "",
                                "type": "temp",
                            }
                        ]
                    },
                },
                {
                    "kind": "progress",
                    "event_type": "preview",
                    "node_id": "7",
                    "display_node_id": "7",
                    "parent_node_id": None,
                    "real_node_id": "7",
                    "image_type": "PNG",
                    "image_bytes": serialization_module.serialize_value(preview_buffer.getvalue()),
                    "max_size": 256,
                },
                {
                    "kind": "result",
                    "outputs": b"serialized-outputs",
                },
            ]
        ),
    )

    assert result == b"serialized-outputs"
    assert prompt_server.messages[0] == (
        "executed",
        {
            "prompt_id": "prompt-1",
            "node": "7",
            "display_node": "7",
            "output": {
                "images": [
                    {
                        "filename": "preview.png",
                        "subfolder": "",
                        "type": "temp",
                    }
                ]
            },
        },
        "client-1",
    )

    preview_event, preview_payload, preview_sid = prompt_server.messages[1]
    preview_image, preview_metadata = preview_payload
    assert preview_event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA
    assert preview_sid == "client-1"
    assert preview_image[0] == "PNG"
    assert preview_image[2] == 256
    assert preview_image[1].size == (2, 2)
    assert preview_metadata == {
        "node_id": "7",
        "prompt_id": "prompt-1",
        "display_node_id": "7",
        "real_node_id": "7",
    }


def test_remote_modal_consumes_streamed_boundary_output_preview_targets(
    remote_modal_app_module: Any,
    monkeypatch: Any,
    serialization_module: Any,
) -> None:
    """A streamed remote boundary IMAGE should synthesize local PreviewImage executed events."""
    torch = pytest.importorskip("torch")
    image_tensor = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    class FakePromptServer:
        """Capture websocket events emitted by boundary preview synthesis."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[Any, Any, str | None]] = []

        def send_sync(self, event: Any, data: Any, sid: str | None) -> None:
            """Record one emitted websocket message."""
            self.messages.append((event, data, sid))

    class FakePreviewImage:
        """Minimal PreviewImage double that returns deterministic UI payloads."""

        def save_images(self, images: Any) -> dict[str, Any]:
            """Return a fake UI payload for the supplied image tensor."""
            assert torch.equal(images, image_tensor)
            return {
                "ui": {
                    "images": [
                        {
                            "filename": "temp_preview.png",
                            "subfolder": "",
                            "type": "temp",
                        }
                    ]
                }
            }

    prompt_server = FakePromptServer()
    monkeypatch.setattr(remote_modal_app_module, "_lookup_local_prompt_server", lambda: prompt_server)
    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(PreviewImage=FakePreviewImage))

    payload = {
        "prompt_id": "prompt-1",
        "component_id": "component-1",
        "component_node_ids": ["7"],
        "extra_data": {"client_id": "client-1"},
    }
    result = remote_modal_app_module._consume_remote_payload_stream(
        payload,
        iter(
            [
                {
                    "kind": "progress",
                    "event_type": "boundary_output",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "IMAGE",
                    "is_list": False,
                    "preview_target_node_ids": ["9"],
                    "value": serialization_module.serialize_value(image_tensor),
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
            "executed",
            {
                "prompt_id": "prompt-1",
                "node": "9",
                "display_node": "9",
                "output": {
                    "images": [
                        {
                            "filename": "temp_preview.png",
                            "subfolder": "",
                            "type": "temp",
                        }
                    ]
                },
            },
            "client-1",
        )
    ]


def test_modal_cloud_tracing_prompt_server_emits_numeric_node_progress(
    modal_cloud_module: Any,
) -> None:
    """The cloud tracing prompt server should forward active-node numeric progress updates."""
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {"7": {"class_type": "KSampler", "inputs": {}}},
        status_callback=observed_updates.append,
    )

    server.send_sync("executing", {"node": "7"}, None)
    server.send_sync(
        "progress_state",
        {
            "prompt_id": "component-1",
            "nodes": {
                "7": {
                    "node_id": "7",
                    "display_node_id": "7",
                    "real_node_id": "7",
                    "state": "running",
                    "value": 5,
                    "max": 20,
                }
            },
        },
        None,
    )

    assert observed_updates[0]["phase"] == "executing"
    assert observed_updates[1] == {
        "event_type": "node_progress",
        "node_id": "7",
        "display_node_id": "7",
        "real_node_id": "7",
        "value": 5.0,
        "max": 20.0,
    }


def test_modal_cloud_tracing_prompt_server_ignores_trivial_node_progress(
    modal_cloud_module: Any,
) -> None:
    """The cloud tracing prompt server should ignore 0/1 progress updates from non-progress nodes."""
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {"18": {"class_type": "CLIPTextEncode", "inputs": {}}},
        status_callback=observed_updates.append,
    )

    server.send_sync("executing", {"node": "18"}, None)
    server.send_sync(
        "progress_state",
        {
            "prompt_id": "component-1",
            "nodes": {
                "18": {
                    "node_id": "18",
                    "display_node_id": "18",
                    "real_node_id": "18",
                    "state": "running",
                    "value": 0,
                    "max": 1,
                }
            },
        },
        None,
    )

    assert observed_updates == [
        {
            "phase": "executing",
            "active_node_id": "18",
            "active_node_class_type": "CLIPTextEncode",
            "active_node_role": "conditioning",
        }
    ]


def test_modal_cloud_tracing_prompt_server_emits_executed_outputs(
    modal_cloud_module: Any,
) -> None:
    """The cloud tracing prompt server should stream node UI outputs as executed events."""
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {"7": {"class_type": "PreviewImage", "inputs": {}}},
        status_callback=observed_updates.append,
    )

    server.send_sync(
        "executed",
        {
            "node": "7",
            "display_node": "7",
            "output": {"images": [{"filename": "preview.png"}]},
        },
        None,
    )

    assert observed_updates == [
        {
            "event_type": "executed",
            "node_id": "7",
            "display_node_id": "7",
            "output": {"images": [{"filename": "preview.png"}]},
        }
    ]


def test_modal_cloud_tracing_prompt_server_emits_boundary_image_outputs(
    modal_cloud_module: Any,
) -> None:
    """The cloud tracing prompt server should stream configured boundary IMAGE outputs once cached."""
    torch = pytest.importorskip("torch")
    image_tensor = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {
            "7": {"class_type": "VAEDecode", "inputs": {}},
            "8": {"class_type": "OtherNode", "inputs": {}},
        },
        status_callback=observed_updates.append,
    )
    cache_entries = {
        "7": types.SimpleNamespace(outputs=[image_tensor]),
    }
    server.configure_boundary_output_stream(
        boundary_outputs=[
            {
                "node_id": "7",
                "output_index": 0,
                "io_type": "IMAGE",
                "is_list": False,
                "preview_target_node_ids": ["9"],
            }
        ],
        lookup_cache_entry=lambda node_id: cache_entries.get(node_id),
    )

    server.send_sync("executing", {"node": "7"}, None)
    server.send_sync("executing", {"node": "8"}, None)

    assert observed_updates[0]["phase"] == "executing"
    assert observed_updates[1] == {
        "event_type": "boundary_output",
        "node_id": "7",
        "output_index": 0,
        "io_type": "IMAGE",
        "is_list": False,
        "preview_target_node_ids": ["9"],
        "value": image_tensor,
    }
    assert observed_updates[2]["phase"] == "executing"


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


def test_invoke_remote_engine_propagates_local_interrupt_to_modal(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """The local proxy should propagate ComfyUI interrupts to the remote Modal call."""

    class FakeInterrupt(Exception):
        """Stand-in for ComfyUI's InterruptProcessingException."""

    observed_cancellation_events: list[threading.Event] = []
    interrupt_checks = iter([False, True, True])

    def fake_blocking_invoke(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: threading.Event | None = None,
    ) -> bytes:
        assert cancellation_event is not None
        observed_cancellation_events.append(cancellation_event)
        while not cancellation_event.is_set():
            time.sleep(0.01)
        raise RuntimeError("remote interrupted")

    def fake_local_processing_interrupted() -> bool:
        return next(interrupt_checks, True)

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_invoke_modal_payload_blocking", fake_blocking_invoke)
    monkeypatch.setattr(
        remote_modal_app_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_raise_local_interrupt",
        lambda: (_ for _ in ()).throw(FakeInterrupt()),
    )

    with pytest.raises(FakeInterrupt):
        remote_modal_app_module.invoke_remote_engine(
            {"component_id": "component-1", "payload_kind": "subgraph"},
            b"{}",
        )

    assert len(observed_cancellation_events) == 1
    assert observed_cancellation_events[0].is_set()


def test_invoke_remote_engine_async_propagates_local_interrupt_to_modal(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """The async local proxy should also propagate ComfyUI interrupts to the remote Modal call."""

    class FakeInterrupt(Exception):
        """Stand-in for ComfyUI's InterruptProcessingException."""

    observed_cancellation_events: list[threading.Event] = []
    interrupt_checks = iter([False, True, True])

    def fake_blocking_invoke(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: threading.Event | None = None,
    ) -> bytes:
        assert cancellation_event is not None
        observed_cancellation_events.append(cancellation_event)
        while not cancellation_event.is_set():
            time.sleep(0.01)
        raise RuntimeError("remote interrupted")

    def fake_local_processing_interrupted() -> bool:
        return next(interrupt_checks, True)

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_invoke_modal_payload_blocking", fake_blocking_invoke)
    monkeypatch.setattr(
        remote_modal_app_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_raise_local_interrupt",
        lambda: (_ for _ in ()).throw(FakeInterrupt()),
    )

    with pytest.raises(FakeInterrupt):
        asyncio.run(
            remote_modal_app_module.invoke_remote_engine_async(
                {"component_id": "component-1", "payload_kind": "subgraph"},
                b"{}",
            )
        )

    assert len(observed_cancellation_events) == 1
    assert observed_cancellation_events[0].is_set()


def test_remote_modal_interrupt_callback_writes_shared_control_flag(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """The local interrupt callback should write to the shared Modal Dict control store."""

    class FakeInterruptStore:
        """Simple Modal Dict double that records written interruption flags."""

        def __init__(self) -> None:
            """Initialize captured writes."""
            self.put_calls: list[tuple[str, Any]] = []

        def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
            """Record one interrupt flag write."""
            self.put_calls.append((key, value))
            return True

    interrupt_store = FakeInterruptStore()

    class FakeModalDict:
        """Minimal modal.Dict shim that returns the fake interrupt store."""

        @staticmethod
        def from_name(
            name: str,
            *,
            environment_name: str | None = None,
            create_if_missing: bool = False,
            client: Any | None = None,
        ) -> FakeInterruptStore:
            return interrupt_store

    monkeypatch.setattr(
        remote_modal_app_module,
        "modal",
        types.SimpleNamespace(Dict=FakeModalDict),
    )
    monkeypatch.setenv("COMFY_MODAL_INTERRUPT_DICT_NAME", "shared-interrupts")
    remote_modal_app_module.get_settings.cache_clear()
    remote_modal_app_module._MODAL_INTERRUPT_DICTS.clear()
    try:
        callback = remote_modal_app_module._build_remote_interrupt_callback(
            object(),
            {"prompt_id": "prompt-1", "component_id": "component-2"},
        )
        assert callback is not None
        callback()
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        remote_modal_app_module._MODAL_INTERRUPT_DICTS.clear()

    assert len(interrupt_store.put_calls) == 1
    interrupt_key, interrupt_value = interrupt_store.put_calls[0]
    assert interrupt_key == "prompt-1:component-2"
    assert isinstance(interrupt_value["requested_at"], float)


def test_invoke_remote_engine_payload_stream_detects_local_interrupt_without_outer_sync(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """The blocking streamed Modal bridge should propagate interrupts without relying on the outer wrapper loop."""
    cancellation_event = threading.Event()
    remote_release_event = threading.Event()
    interrupt_calls: list[str] = []
    interrupt_checks = iter([False, True, True])

    def fake_local_processing_interrupted() -> bool:
        """Report a local interrupt after the first poll interval."""
        return next(interrupt_checks, True)

    def fake_interrupt_remote_call() -> None:
        """Record the propagated remote interrupt and let the fake stream finish."""
        interrupt_calls.append("interrupt")
        remote_release_event.set()

    def fake_stream_events() -> Iterator[dict[str, Any]]:
        """Block until the local bridge requests cancellation, then yield one final result."""
        while not remote_release_event.is_set():
            time.sleep(0.01)
        yield {
            "kind": "result",
            "outputs": serialization_module.serialize_node_outputs(("done",)),
        }

    class FakeStreamMethod:
        """Minimal Modal stream method shim."""

        def remote_gen(self, payload: dict[str, Any], kwargs_payload: bytes) -> Iterator[dict[str, Any]]:
            """Return the fake delayed stream for this request."""
            del payload, kwargs_payload
            return fake_stream_events()

    monkeypatch.setattr(
        remote_modal_app_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_build_remote_interrupt_callback",
        lambda remote_engine, payload: fake_interrupt_remote_call,
    )

    response = remote_modal_app_module._invoke_remote_engine_payload(
        types.SimpleNamespace(execute_payload_stream=FakeStreamMethod()),
        {
            "component_id": "component-1",
            "payload_kind": "subgraph",
            "prompt_id": "prompt-1",
            "component_node_ids": ["1"],
            "extra_data": {"client_id": "client-1"},
        },
        b"{}",
        cancellation_event,
    )

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert cancellation_event.is_set()
    assert interrupt_calls == ["interrupt"]


def test_invoke_mapped_remote_engine_async_runs_explicit_mapped_phase_items(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote execution should run the explicit mapped phase once per item in order."""
    observed_calls: list[tuple[str, dict[str, Any]]] = []
    progress_updates: list[dict[str, Any]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str]:
        assert payload["payload_kind"] == "subgraph"
        assert payload["suppress_status_stream"] is True
        observed_calls.append((str(payload["component_id"]), dict(hydrated_inputs)))
        return (f"done:{hydrated_inputs['remote_input_0']}",)

    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_updates.append(kwargs),
    )
    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "6",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_0", "io_type": "STRING"},
        "boundary_outputs": [
            {
                "proxy_output_name": "7_text",
                "node_id": "7",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            }
        ],
        "static_to_mapped_boundaries": [],
        "static_phase": {
            "component_node_ids": [],
            "subgraph_prompt": {},
            "boundary_inputs": [],
            "boundary_outputs": [],
            "execute_node_ids": [],
        },
        "mapped_phase": {
            "component_node_ids": ["7"],
            "subgraph_prompt": {
                "7": {
                    "class_type": "RemoteStringEcho",
                    "inputs": {"text": ["remote_input_0", 0]},
                }
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_0",
                    "io_type": "STRING",
                    "targets": [{"node_id": "7", "input_name": "text"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "7_text",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["7"],
        },
        "extra_data": {"client_id": "client-1"},
    }
    response = asyncio.run(
        remote_modal_app_module._invoke_mapped_remote_engine_async(
            payload,
            serialization_module.serialize_node_inputs(
                {"remote_input_0": ["a", "b", "c", "d"]}
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["done:a", "done:b", "done:c", "done:d"],
    )
    assert observed_calls == [
        ("6::item:0", {"remote_input_0": "a"}),
        ("6::item:1", {"remote_input_0": "b"}),
        ("6::item:2", {"remote_input_0": "c"}),
        ("6::item:3", {"remote_input_0": "d"}),
    ]
    assert progress_updates[0]["value"] == 0.0
    assert progress_updates[0].get("lane_id") is None
    assert progress_updates[-1]["value"] == 4.0


def test_invoke_mapped_remote_engine_async_executes_static_branch_once(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote execution should run the explicit static phase once and inject its bridge outputs."""
    observed_execute_node_ids: list[tuple[str, tuple[str, ...]]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str, ...]:
        observed_execute_node_ids.append(
            (
                str(payload["component_id"]),
                tuple(str(node_id) for node_id in payload.get("execute_node_ids", [])),
            )
        )
        if str(payload["component_id"]).endswith("::static"):
            assert tuple(payload.get("execute_node_ids", [])) == ("1", "3")
            assert [output["proxy_output_name"] for output in payload.get("boundary_outputs", [])] == [
                "3_text",
                "static_input_0",
            ]
            return ("static-output", "shared-model")

        assert tuple(payload.get("execute_node_ids", [])) == ("7",)
        assert hydrated_inputs["static_input_0"] == "shared-model"
        assert [output["proxy_output_name"] for output in payload.get("boundary_outputs", [])] == ["7_text"]
        return (f"mapped:{hydrated_inputs['remote_input_1']}",)

    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "1",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_1", "io_type": "STRING"},
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "1",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "7", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["1", "3"],
            "subgraph_prompt": {
                "1": {"class_type": "RemoteModel", "inputs": {}},
                "3": {"class_type": "RemoteSampler", "inputs": {"model": ["1", 0]}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "3_text",
                    "node_id": "3",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "1",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                },
            ],
            "execute_node_ids": ["1", "3"],
        },
        "mapped_phase": {
            "component_node_ids": ["6", "7"],
            "subgraph_prompt": {
                "6": {"class_type": "ModalMapInput", "inputs": {"value": ["remote_input_1", 0]}},
                "7": {
                    "class_type": "RemoteSampler",
                    "inputs": {"model": ["static_input_0", 0], "latent": ["6", 0]},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_1",
                    "io_type": "STRING",
                    "targets": [{"node_id": "6", "input_name": "value"}],
                },
                {
                    "proxy_input_name": "static_input_0",
                    "io_type": "MODEL",
                    "targets": [{"node_id": "7", "input_name": "model"}],
                },
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "7_text",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["7"],
        },
        "boundary_outputs": [
            {
                "proxy_output_name": "3_text",
                "node_id": "3",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": False,
            },
            {
                "proxy_output_name": "7_text",
                "node_id": "7",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            },
        ],
        "execute_node_ids": ["3", "7"],
        "static_execute_node_ids": ["1", "3"],
        "mapped_execute_node_ids": ["7"],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_mapped_remote_engine_async(
            payload,
            serialization_module.serialize_node_inputs(
                {"remote_input_1": ["a", "b"]}
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        "static-output",
        ["mapped:a", "mapped:b"],
    )
    assert observed_execute_node_ids[0] == ("1::static", ("1", "3"))
    assert observed_execute_node_ids[1:] == [
        ("1::item:0", ("7",)),
        ("1::item:1", ("7",)),
    ]


def test_modal_cloud_execute_mapped_subgraph_payload_injects_static_bridges(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The cloud runtime should execute the static phase once and feed its outputs into each mapped item."""
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        custom_nodes_root: Any,
        status_callback: Any = None,
        cancellation_event: Any = None,
        interrupt_store: Any = None,
        interrupt_flag_key: Any = None,
    ) -> tuple[str, ...]:
        observed_calls.append(
            (
                str(payload["component_id"]),
                tuple(str(node_id) for node_id in payload.get("execute_node_ids", [])),
                dict(hydrated_inputs),
            )
        )
        if str(payload["component_id"]).endswith("::static"):
            return ("static-output", "shared-model")
        return (f"mapped:{hydrated_inputs['remote_input_1']}:{hydrated_inputs['static_input_0']}",)

    monkeypatch.setattr(
        modal_cloud_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "cloud-1",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_1", "io_type": "STRING"},
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "1",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "7", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["1", "3"],
            "subgraph_prompt": {
                "1": {"class_type": "RemoteModel", "inputs": {}},
                "3": {"class_type": "RemoteSampler", "inputs": {"model": ["1", 0]}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "3_text",
                    "node_id": "3",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "1",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                },
            ],
            "execute_node_ids": ["1", "3"],
        },
        "mapped_phase": {
            "component_node_ids": ["6", "7"],
            "subgraph_prompt": {
                "6": {"class_type": "ModalMapInput", "inputs": {"value": ["remote_input_1", 0]}},
                "7": {
                    "class_type": "RemoteSampler",
                    "inputs": {"model": ["static_input_0", 0], "latent": ["6", 0]},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_1",
                    "io_type": "STRING",
                    "targets": [{"node_id": "6", "input_name": "value"}],
                },
                {
                    "proxy_input_name": "static_input_0",
                    "io_type": "MODEL",
                    "targets": [{"node_id": "7", "input_name": "model"}],
                },
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "7_text",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["7"],
        },
        "boundary_outputs": [
            {
                "proxy_output_name": "3_text",
                "node_id": "3",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": False,
            },
            {
                "proxy_output_name": "7_text",
                "node_id": "7",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            },
        ],
        "extra_data": {"client_id": "client-1"},
    }

    outputs = modal_cloud_module._execute_mapped_subgraph_payload(
        payload,
        {"remote_input_1": ["a", "b"]},
        None,
    )

    assert outputs == (
        "static-output",
        ["mapped:a:shared-model", "mapped:b:shared-model"],
    )
    assert observed_calls == [
        ("cloud-1::static", ("1", "3"), {}),
        ("cloud-1::item:0", ("7",), {"static_input_0": "shared-model", "remote_input_1": "a"}),
        ("cloud-1::item:1", ("7",), {"static_input_0": "shared-model", "remote_input_1": "b"}),
    ]


def test_modal_cloud_execute_mapped_subgraph_payload_preserves_assigned_lane_id(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped cloud progress should keep the caller-assigned lane id instead of collapsing to `0`."""
    observed_progress_events: list[dict[str, Any]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        custom_nodes_root: Any,
        status_callback: Any = None,
        cancellation_event: Any = None,
        interrupt_store: Any = None,
        interrupt_flag_key: Any = None,
    ) -> tuple[str, ...]:
        del custom_nodes_root, cancellation_event, interrupt_store, interrupt_flag_key
        if status_callback is not None:
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": "12",
                    "display_node_id": "12",
                    "real_node_id": "12",
                    "value": 3.0,
                    "max": 9.0,
                }
            )
        return (f"mapped:{hydrated_inputs['remote_input_1']}",)

    monkeypatch.setattr(
        modal_cloud_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "cloud-2",
        "prompt_id": "prompt-1",
        "mapped_progress_lane_id": "3",
        "mapped_input": {"proxy_input_name": "remote_input_1", "io_type": "STRING"},
        "static_to_mapped_boundaries": [],
        "mapped_phase": {
            "component_node_ids": ["12", "39"],
            "subgraph_prompt": {
                "39": {
                    "class_type": "RemoteSampler",
                    "inputs": {"latent": ["remote_input_1", 0]},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_1",
                    "io_type": "STRING",
                    "targets": [{"node_id": "39", "input_name": "latent"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "39_text",
                    "node_id": "39",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["39"],
        },
        "boundary_outputs": [
            {
                "proxy_output_name": "39_text",
                "node_id": "39",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            },
        ],
    }

    outputs = modal_cloud_module._execute_mapped_subgraph_payload(
        payload,
        {"remote_input_1": ["a"]},
        None,
        status_callback=lambda event: observed_progress_events.append(dict(event)),
    )

    assert outputs == (["mapped:a"],)
    assert any(
        event.get("event_type") == "node_progress"
        and event.get("real_node_id") == "12"
        and event.get("lane_id") == "3"
        for event in observed_progress_events
    )
    assert any(
        event.get("event_type") == "node_progress"
        and event.get("clear") is True
        and event.get("lane_id") == "3"
        for event in observed_progress_events
    )


def test_invoke_implicitly_mapped_subgraph_async_zips_batched_boundary_inputs(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Ordinary remote subgraphs should fan out when multiple boundary inputs arrive batched."""
    observed_inputs: list[dict[str, Any]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        assert payload["payload_kind"] == "subgraph"
        assert payload["suppress_status_stream"] is True
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        observed_inputs.append(hydrated_inputs)
        return serialization_module.serialize_node_outputs(
            (f"{hydrated_inputs['remote_input_0']}:{hydrated_inputs['remote_input_1']}",)
        )

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "12",
        "prompt_id": "prompt-1",
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "latent_image": ["remote_input_0", 0],
                    "seed": ["remote_input_1", 0],
                },
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "LATENT",
                "targets": [{"node_id": "12", "input_name": "latent_image"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [{"node_id": "12", "io_type": "STRING", "is_list": False}],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": ["latent-a", "latent-b"],
                    "remote_input_1": [10, 11],
                }
            ),
        )
    )

    assert observed_inputs == [
        {"remote_input_0": "latent-a", "remote_input_1": 10},
        {"remote_input_0": "latent-b", "remote_input_1": 11},
    ]
    assert serialization_module.deserialize_node_outputs(response) == (
        ["latent-a:10", "latent-b:11"],
    )


def test_implicitly_mapped_subgraph_shared_model_keeps_unbatched_sampler_single_run(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """A shared MODEL with mixed batch-size INT seeds should run sampler 4 once and sampler 12 four times."""
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append((str(payload["component_id"]), execute_node_ids, hydrated_inputs))

        if execute_node_ids == ("4",):
            return serialization_module.serialize_node_outputs(("sampler-4",))
        if execute_node_ids == ("12",):
            return serialization_module.serialize_node_outputs((f"sampler-12:{hydrated_inputs['remote_input_0']}",))
        raise AssertionError(f"Unexpected execute nodes for implicit mapped regression: {execute_node_ids!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["4", "12", "17"],
        "execute_node_ids": ["4", "12"],
        "subgraph_prompt": {
            "17": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            "4": {"class_type": "KSampler", "inputs": {"model": ["17", 0], "seed": 0}},
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": ["17", 0], "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "4", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "4", "io_type": "STRING", "is_list": False},
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11, 12, 13],
                    "remote_input_1": [28],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        "sampler-4",
        ["sampler-12:10", "sampler-12:11", "sampler-12:12", "sampler-12:13"],
    )
    assert observed_calls == [
        ("17::static", ("4",), {"remote_input_1": [28]}),
        ("17::item:0", ("12",), {"remote_input_0": 10, "remote_input_1": [28]}),
        ("17::item:1", ("12",), {"remote_input_0": 11, "remote_input_1": [28]}),
        ("17::item:2", ("12",), {"remote_input_0": 12, "remote_input_1": [28]}),
        ("17::item:3", ("12",), {"remote_input_0": 13, "remote_input_1": [28]}),
    ]


@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("remote_modal_app_module",),
        ("modal_cloud_module",),
    ],
)
def test_trim_subgraph_payload_to_required_nodes_drops_unrelated_mapped_branch(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Static or per-item sub-runs should exclude unrelated nodes from the mapped sibling branch."""
    target_module = request.getfixturevalue(module_fixture_name)
    payload = {
        "component_id": "1::static",
        "component_node_ids": ["1", "2", "3", "7"],
        "subgraph_prompt": {
            "1": {"class_type": "LoadDiffusionModel", "inputs": {}},
            "2": {"class_type": "ModalMapInput", "inputs": {"value": ["remote_input_1", 0]}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "steps": 20}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "latent_image": ["2", 0]}},
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_1",
                "targets": [{"node_id": "2", "input_name": "value"}],
            }
        ],
        "boundary_outputs": [
            {"node_id": "3", "output_index": 0, "io_type": "LATENT", "is_list": False},
        ],
        "execute_node_ids": ["3"],
        "mapped_execute_node_ids": ["7"],
        "static_execute_node_ids": ["3"],
    }

    trimmed_payload = target_module._trim_subgraph_payload_to_required_nodes(payload)

    assert trimmed_payload["component_node_ids"] == ["1", "3"]
    assert list(trimmed_payload["subgraph_prompt"].keys()) == ["1", "3"]
    assert trimmed_payload["boundary_inputs"] == []
    assert trimmed_payload["boundary_outputs"] == payload["boundary_outputs"]
    assert trimmed_payload["execute_node_ids"] == ["3"]
    assert trimmed_payload["mapped_execute_node_ids"] == []
    assert trimmed_payload["static_execute_node_ids"] == ["3"]


@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("remote_modal_app_module",),
        ("modal_cloud_module",),
    ],
)
def test_trim_subgraph_payload_to_required_nodes_drops_stale_execute_targets(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Trimmed subgraph payloads should ignore execute targets that are absent from the current prompt."""
    target_module = request.getfixturevalue(module_fixture_name)
    payload = {
        "component_id": "1::static",
        "component_node_ids": ["1", "3"],
        "subgraph_prompt": {
            "1": {"class_type": "LoadDiffusionModel", "inputs": {}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "steps": 20}},
        },
        "boundary_inputs": [],
        "boundary_outputs": [
            {"node_id": "3", "output_index": 0, "io_type": "LATENT", "is_list": False},
        ],
        "execute_node_ids": ["3", "5"],
        "mapped_execute_node_ids": [],
        "static_execute_node_ids": ["3", "5"],
    }

    trimmed_payload = target_module._trim_subgraph_payload_to_required_nodes(payload)

    assert trimmed_payload["component_node_ids"] == ["1", "3"]
    assert list(trimmed_payload["subgraph_prompt"].keys()) == ["1", "3"]
    assert trimmed_payload["execute_node_ids"] == ["3"]
    assert trimmed_payload["mapped_execute_node_ids"] == []
    assert trimmed_payload["static_execute_node_ids"] == ["3"]


def test_consume_remote_payload_stream_suppresses_status_but_keeps_boundary_previews(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped per-item remote calls should suppress status chatter but still forward previews and lane progress."""
    progress_calls: list[dict[str, Any]] = []
    status_calls: list[dict[str, Any]] = []
    preview_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_calls.append(kwargs),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_modal_status",
        lambda **kwargs: status_calls.append(kwargs),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_preview_boundary_output",
        lambda **kwargs: preview_calls.append(kwargs),
    )

    payload = {
        "component_id": "6::item:0",
        "prompt_id": "prompt-1",
        "component_node_ids": ["6", "7"],
        "extra_data": {"client_id": "client-1"},
        "suppress_status_stream": True,
        "mapped_progress_lane_id": "1",
        "mapped_progress_display_node_id": "6",
        "map_item_index": 0,
    }
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "node_progress",
                "node_id": "7",
                "value": 1.0,
                "max": 4.0,
            },
            {
                "kind": "progress",
                "event_type": "boundary_output",
                "node_id": "7",
                "preview_target_node_ids": ["9"],
                "value": serialization_module.serialize_value(["preview"]),
            },
            {
                "kind": "progress",
                "phase": "executing",
                "active_node_id": "7",
            },
            {
                "kind": "result",
                "outputs": serialization_module.serialize_node_outputs(("done",)),
            },
        ]
    )

    response = remote_modal_app_module._consume_remote_payload_stream(payload, stream_events)

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert progress_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "7",
            "value": 1.0,
            "max_value": 4.0,
            "display_node_id": "7",
            "real_node_id": None,
            "lane_id": "1",
            "clear": False,
            "item_index": 0,
            "aggregate_only": False,
        }
    ]
    assert status_calls == []
    assert preview_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "preview_target_node_ids": ["9"],
            "image_value": ["preview"],
        }
    ]


def test_consume_remote_payload_stream_keeps_static_execute_node_progress_when_status_is_suppressed(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Static sub-runs should still forward real execute-node progress under suppressed status streams."""
    progress_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_calls.append(kwargs),
    )

    payload = {
        "component_id": "1::static",
        "prompt_id": "prompt-1",
        "component_node_ids": ["1", "2", "12", "4"],
        "execute_node_ids": ["12", "4"],
        "boundary_outputs": [
            {"node_id": "4", "output_index": 0, "io_type": "LATENT", "is_list": False}
        ],
        "extra_data": {"client_id": "client-1"},
        "suppress_status_stream": True,
    }
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "node_progress",
                "node_id": "1",
                "display_node_id": "1",
                "real_node_id": "12",
                "value": 5.0,
                "max": 20.0,
            },
            {
                "kind": "progress",
                "event_type": "node_progress",
                "node_id": "2",
                "display_node_id": "2",
                "real_node_id": "2",
                "value": 1.0,
                "max": 10.0,
            },
            {
                "kind": "result",
                "outputs": serialization_module.serialize_node_outputs(("done",)),
            },
        ]
    )

    response = remote_modal_app_module._consume_remote_payload_stream(payload, stream_events)

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert progress_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "1",
            "value": 5.0,
            "max_value": 20.0,
            "display_node_id": "1",
            "real_node_id": "12",
            "lane_id": None,
            "clear": False,
            "item_index": None,
            "aggregate_only": False,
        }
    ]


def test_consume_remote_payload_stream_clears_static_execute_node_progress_on_suppressed_completion(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Suppressed static sub-runs should emit an explicit clear for lane-less node progress on completion."""
    progress_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_calls.append(kwargs),
    )

    payload = {
        "component_id": "1::static",
        "prompt_id": "prompt-1",
        "component_node_ids": ["1", "2", "12", "4"],
        "execute_node_ids": ["12", "4"],
        "boundary_outputs": [
            {"node_id": "4", "output_index": 0, "io_type": "LATENT", "is_list": False}
        ],
        "extra_data": {"client_id": "client-1"},
        "suppress_status_stream": True,
    }
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "node_progress",
                "node_id": "1",
                "display_node_id": "1",
                "real_node_id": "12",
                "value": 5.0,
                "max": 20.0,
            },
            {
                "kind": "progress",
                "phase": "execution_success",
            },
            {
                "kind": "result",
                "outputs": serialization_module.serialize_node_outputs(("done",)),
            },
        ]
    )

    response = remote_modal_app_module._consume_remote_payload_stream(payload, stream_events)

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert progress_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "1",
            "value": 5.0,
            "max_value": 20.0,
            "display_node_id": "1",
            "real_node_id": "12",
            "lane_id": None,
            "clear": False,
            "item_index": None,
            "aggregate_only": False,
        },
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "1",
            "value": 0.0,
            "max_value": 1.0,
            "display_node_id": "1",
            "real_node_id": "12",
            "clear": True,
        },
    ]


def test_consume_remote_payload_stream_filters_static_sibling_ui_events_from_mapped_items(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped item streams should not forward executed or preview events for static sibling nodes."""
    executed_calls: list[dict[str, Any]] = []
    preview_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_executed_output",
        lambda **kwargs: executed_calls.append(kwargs),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_preview_image",
        lambda **kwargs: preview_calls.append(kwargs),
    )

    payload = {
        "component_id": "6::item:0",
        "prompt_id": "prompt-1",
        "component_node_ids": ["3", "6", "7"],
        "execute_node_ids": ["7"],
        "boundary_outputs": [
            {"node_id": "7", "output_index": 0, "io_type": "IMAGE", "is_list": False}
        ],
        "extra_data": {"client_id": "client-1"},
        "suppress_status_stream": True,
        "mapped_progress_lane_id": "0",
        "mapped_progress_display_node_id": "6",
        "map_item_index": 0,
    }
    preview_bytes = serialization_module.serialize_value(b"preview-bytes")
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "executed",
                "node_id": "3",
                "display_node_id": "3",
                "output": serialization_module.serialize_value({"images": ["static"]}),
            },
            {
                "kind": "progress",
                "event_type": "preview",
                "node_id": "3",
                "display_node_id": "3",
                "image_type": "PNG",
                "image_bytes": preview_bytes,
            },
            {
                "kind": "progress",
                "event_type": "executed",
                "node_id": "7",
                "display_node_id": "7",
                "output": serialization_module.serialize_value({"images": ["mapped"]}),
            },
            {
                "kind": "progress",
                "event_type": "preview",
                "node_id": "7",
                "display_node_id": "7",
                "image_type": "PNG",
                "image_bytes": preview_bytes,
            },
            {
                "kind": "result",
                "outputs": serialization_module.serialize_node_outputs(("done",)),
            },
        ]
    )

    response = remote_modal_app_module._consume_remote_payload_stream(payload, stream_events)

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert executed_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "7",
            "display_node_id": "7",
            "output_payload": {"images": ["mapped"]},
        }
    ]
    assert preview_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "7",
            "display_node_id": "7",
            "parent_node_id": None,
            "real_node_id": None,
            "image_type": "PNG",
            "image_bytes": b"preview-bytes",
            "max_size": None,
        }
    ]


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


def test_modal_cloud_class_options_do_not_use_deprecated_concurrency_flag(
    modal_cloud_module: Any,
) -> None:
    """The deployed Modal class options should avoid deprecated concurrency flags."""
    fake_settings = types.SimpleNamespace(
        modal_gpu="A100",
        remote_storage_root="/vol/data",
        scaledown_window_seconds=60,
        min_containers=0,
        max_containers=4,
        buffer_containers=1,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
    )

    options = modal_cloud_module._remote_engine_cls_options(
        fake_settings,
        vol=object(),
        image=object(),
    )

    assert "allow_concurrent_inputs" not in options
    assert options["max_containers"] == 4
    assert options["buffer_containers"] == 1
    module_source = Path(modal_cloud_module.__file__).read_text(encoding="utf-8")
    assert "@modal.concurrent(max_inputs=1)" in module_source


def test_modal_cloud_registered_execution_clears_shared_interrupt_flags(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote execution registration should clear stale shared interrupt flags on entry and exit."""

    class FakeInterruptFlags:
        """Simple Modal Dict double that records cleared keys."""

        def __init__(self) -> None:
            """Initialize captured pop calls."""
            self.pop_calls: list[tuple[str, Any]] = []

        def pop(self, key: str, default: Any = None) -> Any:
            """Record one cleared interrupt flag."""
            self.pop_calls.append((key, default))
            return None

    interrupt_flags = FakeInterruptFlags()
    monkeypatch.setattr(modal_cloud_module, "modal", object())
    monkeypatch.setattr(modal_cloud_module, "interrupt_flags", interrupt_flags, raising=False)

    with modal_cloud_module._registered_remote_execution(
        {"prompt_id": "prompt-1", "component_id": "component-2"}
    ) as execution_control:
        assert execution_control.interrupt_flag_key == "prompt-1:component-2"

    assert interrupt_flags.pop_calls == [
        ("prompt-1:component-2", None),
        ("prompt-1:component-2", None),
    ]


def test_modal_cloud_interrupt_monitor_consumes_shared_cancel_flag(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The remote interrupt monitor should trip when the shared Modal Dict flag appears."""

    class FakeInterruptFlags:
        """Simple Modal Dict double that exposes one shared cancel flag."""

        def __init__(self) -> None:
            """Initialize the backing key set."""
            self.keys = {"prompt-1:component-2"}
            self.contains_calls = 0
            self.pop_calls: list[tuple[str, Any]] = []

        def contains(self, key: str) -> bool:
            """Report whether the shared interrupt flag exists."""
            self.contains_calls += 1
            return key in self.keys

        def pop(self, key: str, default: Any = None) -> Any:
            """Remove the shared interrupt flag once consumed."""
            self.pop_calls.append((key, default))
            self.keys.discard(key)
            return None

    interrupt_calls: list[str] = []
    cancellation_event = threading.Event()
    monkeypatch.setitem(
        sys.modules,
        "nodes",
        types.SimpleNamespace(interrupt_processing=lambda: interrupt_calls.append("interrupt")),
    )

    with modal_cloud_module._temporary_remote_interrupt_monitor(
        "component-2",
        cancellation_event,
        interrupt_store=FakeInterruptFlags(),
        interrupt_flag_key="prompt-1:component-2",
    ):
        deadline = time.time() + 1.0
        while not interrupt_calls and time.time() < deadline:
            time.sleep(0.01)

    assert interrupt_calls == ["interrupt"]
    assert cancellation_event.is_set()


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


class _PersistentCacheNode:
    """Simple node used to verify persisted node-cache reuse across prompt runs."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"
    invocation_count = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal V1 schema needed for cache-key generation."""
        return {"required": {"value": ("INT",)}}

    def run(self, value: int) -> tuple[int]:
        """Count real executions so persisted cache hits are visible to the test."""
        type(self).invocation_count += 1
        return (value + 1,)


def test_modal_cloud_serializes_only_small_transport_safe_node_outputs(
    modal_cloud_module: Any,
) -> None:
    """Persisted node-cache records should keep small tensor outputs and skip oversized ones."""
    import torch

    execution = modal_cloud_module._load_execution_module()
    small_entry = execution.CacheEntry(ui=None, outputs=[[torch.zeros((8,), dtype=torch.float32)]])
    large_entry = execution.CacheEntry(ui=None, outputs=[[torch.zeros((512,), dtype=torch.float32)]])

    small_record = modal_cloud_module._serialize_node_output_cache_entry(
        small_entry,
        max_bytes=1024,
    )
    large_record = modal_cloud_module._serialize_node_output_cache_entry(
        large_entry,
        max_bytes=1024,
    )

    assert small_record is not None
    assert large_record is None


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
    first_executor.caches.outputs.set("node_1", cache_entry)
    persisted_nodes = modal_cloud_module._persist_node_output_cache_entries(
        first_executor,
        prompt=copy.deepcopy(prompt),
        cache_store=cache_store,
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
    restored_entry = second_executor.caches.outputs.get("node_1")

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
        restored_cache_keys_by_node_id: dict[str, str] | None = None,
    ) -> list[str]:
        """Record the cache-key-set marker visible at restore time."""
        del execution
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
        modal_cloud_module,
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

        def get(self, node_id: str) -> Any:
            """Return the prepared cache entry for the target node only."""
            if node_id == "node_1":
                return cache_entry
            return None

    executor = types.SimpleNamespace(caches=types.SimpleNamespace(outputs=FakeOutputsCache()))

    monkeypatch.setattr(
        modal_cloud_module,
        "_node_output_cache_key_from_key_set_sync",
        lambda cache_key_set, node_id: cache_key if node_id == "node_1" else None,
    )
    monkeypatch.setattr(
        modal_cloud_module,
        "_emit_cloud_info",
        lambda message, *args: observed_logs.append((message, *args)),
    )

    persisted_nodes = modal_cloud_module._persist_node_output_cache_entries(
        executor,
        prompt={"node_1": {"class_type": "PersistentCacheNode", "inputs": {"value": 4}}},
        cache_store=cache_store,
        restored_cache_keys_by_node_id={"node_1": cache_key},
    )

    assert persisted_nodes == []
    assert cache_store == {cache_key: {"version": 1, "outputs_zlib": b"old"}}
    assert observed_logs[-1] == (
        "Node output cache write node=%s key_prefix=%s result=skip reason=restored-hit",
        "node_1",
        "NC_existing",
    )


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


def test_modal_cloud_summarizes_suspicious_wrapped_prompt_inputs(
    modal_cloud_module: Any,
) -> None:
    """Remote failure diagnostics should flag remaining singleton-list prompt wrappers."""
    prompt = {
        "12": {
            "class_type": "ExampleNode",
            "inputs": {
                "steps": [20],
                "latent": ["7", [0]],
                "ok_link": ["8", 0],
            },
        }
    }

    findings = modal_cloud_module._summarize_suspicious_prompt_inputs(prompt)

    assert findings == [
        "12.steps=[20]",
        "12.latent=['7', [0]]",
    ]


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


class _PrimitiveEchoNode:
    """Simple node used to verify primitive widget coercion."""

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("steps", "cfg", "enabled", "label")
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return one primitive input of each V1 widget type."""
        return {
            "required": {
                "steps": ("INT",),
                "cfg": ("FLOAT",),
                "enabled": ("BOOLEAN",),
                "label": ("STRING",),
            }
        }

    def run(
        self,
        steps: int,
        cfg: float,
        enabled: bool,
        label: str,
    ) -> tuple[int, float, bool, str]:
        """Echo primitive inputs after asserting their coerced Python types."""
        assert isinstance(steps, int)
        assert isinstance(cfg, float)
        assert isinstance(enabled, bool)
        assert isinstance(label, str)
        return (steps, cfg, enabled, label)


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


def test_local_remote_app_normalizes_wrapped_subgraph_link_indexes(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback runner should canonicalize singleton-list prompt link indexes."""
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
                    "inputs": {"value": [[["remote_1", [0]]]]},
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
                    "output_index": [[0]],
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


def test_local_remote_app_normalizes_wrapped_scalar_prompt_inputs(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback runner should unwrap singleton-list scalar prompt inputs."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "BoundarySource",
                    "inputs": {"value": [4]},
                    "_meta": {},
                }
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "remote_1_value",
                    "node_id": "remote_1",
                    "output_index": 0,
                    "io_type": "INT",
                    "is_list": False,
                }
            ],
            "execute_node_ids": [["remote_1"]],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload="{}",
        node_mapping={
            "BoundarySource": _BoundarySourceNode,
        },
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (5,)


@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("remote_modal_app_module",),
        ("modal_cloud_module",),
    ],
)
def test_apply_boundary_inputs_normalizes_wrapped_scalar_values(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Boundary input hydration should unwrap singleton-list scalar wrappers before PromptExecutor sees them."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "BoundarySource",
            "inputs": {"value": 0},
            "_meta": {},
        }
    }

    target_module._apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=[
            {
                "proxy_input_name": "remote_input_0",
                "targets": [{"node_id": "remote_1", "input_name": "value"}],
            }
        ],
        hydrated_inputs={"remote_input_0": [4]},
    )

    assert prompt["remote_1"]["inputs"]["value"] == 4


@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("remote_modal_app_module",),
        ("modal_cloud_module",),
    ],
)
def test_validate_prompt_input_shapes_rejects_list_on_primitive_socket(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Prepared remote prompts should fail early when primitive widget inputs still carry raw lists."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "BoundarySource",
            "inputs": {"value": [4, 5]},
            "_meta": {},
        }
    }

    with pytest.raises(target_module.RemoteSubgraphExecutionError, match="input_name='value'"):
        target_module._validate_prompt_input_shapes(
            prompt,
            {"BoundarySource": _BoundarySourceNode},
        )


@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("remote_modal_app_module",),
        ("modal_cloud_module",),
    ],
)
def test_coerce_prompt_primitive_input_values_matches_comfyui_semantics(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Remote runtimes should coerce primitive prompt literals the same way ComfyUI does."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "PrimitiveEcho",
            "inputs": {
                "steps": 18.0,
                "cfg": 5,
                "enabled": 1,
                "label": 7,
            },
            "_meta": {},
        }
    }

    target_module._coerce_prompt_primitive_input_values(
        prompt,
        {"PrimitiveEcho": _PrimitiveEchoNode},
    )

    assert prompt["remote_1"]["inputs"] == {
        "steps": 18,
        "cfg": 5.0,
        "enabled": True,
        "label": "7",
    }
    assert isinstance(prompt["remote_1"]["inputs"]["steps"], int)
    assert isinstance(prompt["remote_1"]["inputs"]["cfg"], float)
    assert isinstance(prompt["remote_1"]["inputs"]["enabled"], bool)
    assert isinstance(prompt["remote_1"]["inputs"]["label"], str)


def test_local_remote_app_coerces_primitive_widget_literals_before_execution(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback runner should coerce primitive widget literals before executing nodes."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-primitive-coercion",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "PrimitiveEcho",
                    "inputs": {
                        "steps": 18.0,
                        "cfg": 5,
                        "enabled": 1,
                        "label": 7,
                    },
                    "_meta": {},
                }
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "steps",
                    "node_id": "remote_1",
                    "output_index": 0,
                    "io_type": "INT",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "cfg",
                    "node_id": "remote_1",
                    "output_index": 1,
                    "io_type": "FLOAT",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "enabled",
                    "node_id": "remote_1",
                    "output_index": 2,
                    "io_type": "BOOLEAN",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "label",
                    "node_id": "remote_1",
                    "output_index": 3,
                    "io_type": "STRING",
                    "is_list": False,
                },
            ],
            "execute_node_ids": ["remote_1"],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload="{}",
        node_mapping={"PrimitiveEcho": _PrimitiveEchoNode},
    )

    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (18, 5.0, True, "7")


@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("remote_modal_app_module",),
        ("modal_cloud_module",),
    ],
)
def test_format_prompt_executor_error_payload_includes_node_context(
    request: Any,
    module_fixture_name: str,
) -> None:
    """PromptExecutor failure formatting should surface the failing node and current inputs."""
    target_module = request.getfixturevalue(module_fixture_name)

    message = target_module._format_prompt_executor_error_payload(
        {
            "exception_message": "int() argument must be a string, a bytes-like object or a real number, not 'list'",
            "node_id": "12",
            "node_type": "KSampler",
            "current_inputs": [{"input_name": "steps", "value": [4, 5]}],
        }
    )

    assert "node_id='12'" in message
    assert "node_type='KSampler'" in message
    assert "current_inputs=" in message
