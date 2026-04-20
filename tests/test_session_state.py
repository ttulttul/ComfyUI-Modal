"""Tests for prompt-scoped remote session state helpers."""

from __future__ import annotations

from typing import Any

import pytest


def test_remote_session_store_round_trips_value_refs(session_state_module: Any) -> None:
    """Stored remote-only outputs should round-trip through opaque session refs."""
    handle = session_state_module.RemoteSessionHandle(
        session_id="session-1",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    store = session_state_module.InMemoryRemoteSessionStore()

    ref = store.put_output(
        handle,
        node_id="node-7",
        output_index=0,
        value={"remote": "value"},
    )

    assert store.get_output(ref) == {"remote": "value"}
    assert store.resolve_value(ref.to_payload()) == {"remote": "value"}


def test_remote_session_store_clear_session_invalidates_refs(session_state_module: Any) -> None:
    """Clearing a prompt-scoped session should invalidate every ref stored inside it."""
    handle = session_state_module.RemoteSessionHandle(
        session_id="session-2",
        prompt_id="prompt-2",
        owner_component_id="component-2",
    )
    store = session_state_module.InMemoryRemoteSessionStore()
    ref = store.put_output(
        handle,
        node_id="node-9",
        output_index=1,
        value="secret",
    )

    store.clear_session(handle)

    with pytest.raises(session_state_module.RemoteSessionStateError):
        store.get_output(ref)
