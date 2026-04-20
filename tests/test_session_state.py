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


def test_remote_session_store_logs_reuse_resolution_and_cleanup(
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """Session observability logs should cover reuse, ref resolution, and cleanup."""
    handle = session_state_module.RemoteSessionHandle(
        session_id="session-3",
        prompt_id="prompt-3",
        owner_component_id="component-3",
    )
    store = session_state_module.InMemoryRemoteSessionStore()
    observed_messages: list[str] = []

    def capture_log(message: str, *args: Any, **kwargs: Any) -> None:
        """Record one formatted session log line."""
        del kwargs
        observed_messages.append(message % args if args else message)

    monkeypatch.setattr(session_state_module.logger, "info", capture_log)

    store.ensure_session(handle)
    store.ensure_session(handle)
    ref = store.put_output(
        handle,
        node_id="node-11",
        output_index=0,
        value="payload",
    )
    assert store.resolve_value(ref.to_payload()) == "payload"
    store.clear_session(handle)

    assert any("Created remote session session_id=session-3" in message for message in observed_messages)
    assert any("Reusing remote session session_id=session-3" in message for message in observed_messages)
    assert any(
        "Stored remote session value session_id=session-3 node_id=node-11 output_index=0" in message
        for message in observed_messages
    )
    assert any(
        "Resolving remote session ref session_id=session-3 node_id=node-11 output_index=0" in message
        for message in observed_messages
    )
    assert any(
        "Resolved remote session value session_id=session-3 node_id=node-11 output_index=0" in message
        for message in observed_messages
    )
    assert any("Cleared remote session session_id=session-3" in message for message in observed_messages)
