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


def test_mapped_stream_progress_preserves_real_node_id(monkeypatch: Any) -> None:
    """Mapped lane progress should target the real executing node, not the representative."""
    modal_app = _load_modal_app_module()
    emitted_progress: list[dict[str, Any]] = []

    def capture_progress(**kwargs: Any) -> None:
        """Record forwarded local progress events for assertions."""
        emitted_progress.append(kwargs)

    monkeypatch.setattr(modal_app, "_emit_local_modal_progress", capture_progress)

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

    with modal_app._MAPPED_PROGRESS_NODE_IDS_LOCK:
        modal_app._MAPPED_PROGRESS_NODE_IDS.clear()

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
            "item_index": 3,
        }
    ]
    with modal_app._MAPPED_PROGRESS_NODE_IDS_LOCK:
        assert modal_app._MAPPED_PROGRESS_NODE_IDS[("prompt-1", "component-1", "7")] == "node-b"

    emitted_progress.clear()
    modal_app._clear_local_mapped_lane_progress(payload, lane_index=7, item_index=3)
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
