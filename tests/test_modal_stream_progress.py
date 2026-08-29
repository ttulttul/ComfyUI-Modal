"""Regression tests for streamed Modal progress forwarding."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE_NAME = "codex_modal_sync_testpkg"


def _clear_test_package_modules() -> None:
    """Remove any transient test-package modules from the interpreter cache."""
    module_names = [
        module_name
        for module_name in sys.modules
        if module_name == TEST_PACKAGE_NAME or module_name.startswith(f"{TEST_PACKAGE_NAME}.")
    ]
    for module_name in module_names:
        del sys.modules[module_name]


def _load_repo_module(module_name: str, file_path: Path) -> types.ModuleType:
    """Load one repository module under the transient test package."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create an import spec for {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_modal_app_module() -> types.ModuleType:
    """Load `remote.modal_app` with package-relative imports resolved locally."""
    _clear_test_package_modules()

    package_module = types.ModuleType(TEST_PACKAGE_NAME)
    package_module.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
    sys.modules[TEST_PACKAGE_NAME] = package_module

    remote_package_name = f"{TEST_PACKAGE_NAME}.remote"
    remote_package = types.ModuleType(remote_package_name)
    remote_package.__path__ = [str(REPO_ROOT / "remote")]  # type: ignore[attr-defined]
    sys.modules[remote_package_name] = remote_package

    return _load_repo_module(
        f"{remote_package_name}.modal_app",
        REPO_ROOT / "remote" / "modal_app.py",
    )


def _loaded_payload_stream_module() -> types.ModuleType:
    """Return the payload-stream owner loaded with the transient package."""
    return sys.modules[f"{TEST_PACKAGE_NAME}.remote.payload_stream"]


def test_mapped_stream_progress_preserves_real_node_id(monkeypatch: Any) -> None:
    """Mapped lane progress should target the real executing node, not the representative."""
    modal_app = _load_modal_app_module()
    payload_stream = _loaded_payload_stream_module()
    mapped_execution = sys.modules[
        f"{TEST_PACKAGE_NAME}.remote.mapped_execution"
    ]
    emitted_progress: list[dict[str, Any]] = []

    def capture_progress(**kwargs: Any) -> None:
        """Record forwarded local progress events for assertions."""
        emitted_progress.append(kwargs)

    monkeypatch.setattr(payload_stream, "_emit_local_modal_progress", capture_progress)
    monkeypatch.setattr(
        mapped_execution, "_emit_local_modal_progress", capture_progress
    )

    payload = {
        "prompt_id": "prompt-1",
        "component_id": "component-1",
        "component_node_ids": ["component-1", "node-a", "node-b"],
        "extra_data": {"client_id": "client-1"},
        "mapped_progress_lane_id": "7",
        "map_item_index": 3,
    }
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "node_progress",
                "node_id": "component-1",
                "display_node_id": "component-1",
                "real_node_id": "node-b",
                "value": 4,
                "max": 9,
            },
            {
                "kind": "result",
                "outputs": [1],
            },
        ]
    )

    with mapped_execution._MAPPED_PROGRESS_NODE_IDS_LOCK:
        mapped_execution._MAPPED_PROGRESS_NODE_IDS.clear()

    modal_app._consume_remote_payload_stream(payload, stream_events)

    assert emitted_progress == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "component-1",
            "value": 4.0,
            "max_value": 9.0,
            "display_node_id": "component-1",
            "real_node_id": "node-b",
            "lane_id": "7",
            "clear": False,
            "item_index": 3,
            "aggregate_only": False,
        }
    ]
    with mapped_execution._MAPPED_PROGRESS_NODE_IDS_LOCK:
        assert mapped_execution._MAPPED_PROGRESS_NODE_IDS[
            ("prompt-1", "component-1", "7")
        ] == "node-b"

    emitted_progress.clear()
    modal_app._clear_local_mapped_lane_progress(
        payload,
        lane_index=7,
        item_index=3,
    )
    assert emitted_progress == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "node-b",
            "value": 0.0,
            "max_value": 1.0,
            "display_node_id": "node-b",
            "lane_id": "7",
            "clear": True,
            "item_index": 3,
        }
    ]


def test_llm_stream_progress_preserves_stage_and_token_metrics(
    monkeypatch: Any,
) -> None:
    """LLM labels, TTFT, and token rate should reach the local websocket."""
    modal_app = _load_modal_app_module()
    payload_stream = _loaded_payload_stream_module()
    emitted_progress: list[dict[str, Any]] = []
    monkeypatch.setattr(
        payload_stream,
        "_emit_local_modal_progress",
        lambda **kwargs: emitted_progress.append(kwargs),
    )
    payload = {
        "prompt_id": "prompt-llm",
        "component_id": "llm-node",
        "component_node_ids": ["llm-node"],
        "extra_data": {"client_id": "client-1"},
    }

    modal_app._consume_remote_payload_stream(
        payload,
        iter(
            [
                {
                    "kind": "progress",
                    "event_type": "node_progress",
                    "node_id": "llm-node",
                    "value": 12,
                    "max": 64,
                    "stage": "generating",
                    "message": "Generating",
                    "unit": "tokens",
                    "time_to_first_token_seconds": 2.5,
                    "tokens_per_second": 9.75,
                },
                {"kind": "result", "outputs": ["done"]},
            ]
        ),
    )

    assert emitted_progress == [
        {
            "prompt_id": "prompt-llm",
            "client_id": "client-1",
            "node_id": "llm-node",
            "value": 12.0,
            "max_value": 64.0,
            "display_node_id": "llm-node",
            "real_node_id": None,
            "lane_id": None,
            "clear": False,
            "item_index": None,
            "aggregate_only": False,
            "stage": "generating",
            "message": "Generating",
            "unit": "tokens",
            "time_to_first_token_seconds": 2.5,
            "tokens_per_second": 9.75,
        }
    ]


def test_stream_progress_reports_exact_modal_container_identity(monkeypatch: Any) -> None:
    """A Modal task id should follow its component progress into the node badge."""
    modal_app = _load_modal_app_module()
    payload_stream = _loaded_payload_stream_module()
    emitted_progress: list[dict[str, Any]] = []
    monkeypatch.setattr(
        payload_stream,
        "_emit_local_modal_progress",
        lambda **kwargs: emitted_progress.append(kwargs),
    )
    payload = {
        "prompt_id": "prompt-location",
        "component_id": "node-162",
        "component_node_ids": ["node-162"],
        "execution_provider": "modal",
        "execution_environment_id": "modal:B300",
        "extra_data": {"client_id": "client-1"},
    }

    modal_app._consume_remote_payload_stream(
        payload,
        iter(
            [
                {"kind": "remote_logs", "task_id": "ta-01K3MODAL"},
                {
                    "kind": "progress",
                    "event_type": "node_progress",
                    "node_id": "node-162",
                    "value": 1,
                    "max": 8,
                },
                {"kind": "result", "outputs": ["done"]},
            ]
        ),
    )

    assert emitted_progress[0]["execution_provider"] == "modal"
    assert emitted_progress[0]["execution_environment_id"] == "modal:B300"
    assert emitted_progress[0]["execution_location"] == "ta-01K3MODAL"


def test_stream_status_reports_exact_modal_container_identity(monkeypatch: Any) -> None:
    """Node status should expose a Modal task id before numeric progress begins."""
    modal_app = _load_modal_app_module()
    payload_stream = _loaded_payload_stream_module()
    emitted_status: list[dict[str, Any]] = []
    monkeypatch.setattr(
        payload_stream,
        "_emit_local_modal_status",
        lambda **kwargs: emitted_status.append(kwargs),
    )
    payload = {
        "prompt_id": "prompt-location",
        "component_id": "node-162",
        "component_node_ids": ["node-162"],
        "execution_provider": "modal",
        "execution_environment_id": "modal:B300",
        "extra_data": {"client_id": "client-1"},
    }

    modal_app._consume_remote_payload_stream(
        payload,
        iter(
            [
                {"kind": "remote_logs", "task_id": "ta-01K3MODAL"},
                {
                    "kind": "progress",
                    "event_type": "status",
                    "phase": "executing",
                    "active_node_id": "node-162",
                },
                {"kind": "result", "outputs": ["done"]},
            ]
        ),
    )

    assert emitted_status[0]["execution_provider"] == "modal"
    assert emitted_status[0]["execution_environment_id"] == "modal:B300"
    assert emitted_status[0]["execution_location"] == "ta-01K3MODAL"
