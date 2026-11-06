"""Mapped items must execute instead of reusing outputs from a shared session."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeSessionStore:
    """Session store that already holds an earlier item's node output."""

    def __init__(self) -> None:
        """Start with no recorded lookups."""
        self.lookups: list[Any] = []

    def get_output(self, ref: Any) -> str:
        """Return the stale value an earlier item stored for this node."""
        self.lookups.append(ref)
        return "earlier-item-output"

    def put_bridge_output(self, session_handle: Any, **kwargs: Any) -> Any:
        """Pretend to publish a bridge output into the session."""
        return SimpleNamespace(session_id=session_handle.session_id)


@pytest.fixture
def cloud_prompt_execution_module(modal_cloud_module: Any) -> Any:
    """Return the flat-imported cloud prompt execution module."""
    return importlib.import_module("cloud_prompt_execution")


def _payload(map_item_index: int | None) -> dict[str, Any]:
    """Return a payload whose only boundary output is session-backed."""
    payload: dict[str, Any] = {
        "component_id": "1__mapped::item:5",
        "boundary_outputs": [
            {"node_id": "9", "output_index": 0, "io_type": "IMAGE", "session_output": True}
        ],
    }
    if map_item_index is not None:
        payload["map_item_index"] = map_item_index
    return payload


def _short_circuit(module: Any, store: _FakeSessionStore, payload: dict[str, Any]) -> Any:
    """Invoke the short-circuit with every other precondition satisfied."""
    stats = module._RemoteSessionBridgeResolutionStats()
    stats.input_ref_count = 5
    return module._short_circuit_restored_session_output_subgraph(
        payload=payload,
        hydrated_inputs={},
        session_handle=SimpleNamespace(session_id="shared-session"),
        resolution_stats=stats,
    )


def _install_cloud_doubles(module: Any, store: _FakeSessionStore, monkeypatch: Any) -> None:
    """Route the cloud module's session and bridge helpers to test doubles."""
    monkeypatch.setattr(module, "remote_session_store", lambda: store)
    monkeypatch.setattr(
        module,
        "_build_remote_session_bridge_record",
        lambda **kwargs: SimpleNamespace(bridge_key="RSB_test"),
    )
    monkeypatch.setattr(module, "_store_remote_session_bridge_record", lambda record: None)
    monkeypatch.setattr(module, "_store_remote_session_bridge_value", lambda key, value: None)


def test_cloud_mapped_item_never_reuses_shared_session_output(
    cloud_prompt_execution_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapped item must run even when the shared session already holds its node output."""
    store = _FakeSessionStore()
    _install_cloud_doubles(cloud_prompt_execution_module, store, monkeypatch)

    result = _short_circuit(cloud_prompt_execution_module, store, _payload(map_item_index=5))

    assert result is None
    assert store.lookups == []


def test_cloud_seed_phase_still_short_circuits(
    cloud_prompt_execution_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static seed phases keep reusing outputs that are identical for every lane."""
    store = _FakeSessionStore()
    _install_cloud_doubles(cloud_prompt_execution_module, store, monkeypatch)

    result = _short_circuit(cloud_prompt_execution_module, store, _payload(map_item_index=None))

    assert result is not None and len(result) == 1
    assert len(store.lookups) == 1


def test_local_mapped_item_never_reuses_shared_session_output(
    local_execution_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host-side fallback applies the same rule as the cloud executor."""
    store = _FakeSessionStore()
    monkeypatch.setattr(
        local_execution_module._host_session_bridge, "_REMOTE_SESSION_STORE", store
    )

    result = _short_circuit(local_execution_module, store, _payload(map_item_index=0))

    assert result is None
    assert store.lookups == []
