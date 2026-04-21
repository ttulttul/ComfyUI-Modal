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


def test_stable_session_bridge_key_ignores_transient_session_ids(
    session_state_module: Any,
) -> None:
    """Durable bridge keys should stay stable across prompt-scoped session id changes."""
    first_key = session_state_module.stable_session_bridge_key(
        producer_payload={"component_id": "1"},
        producer_inputs={
            "model": session_state_module.RemoteSessionBridgeRef(
                bridge_key="RSB_upstream",
                node_id="2",
                output_index=0,
                session_id="session-a",
            ).to_payload()
        },
        node_id="3",
        output_index=0,
    )
    second_key = session_state_module.stable_session_bridge_key(
        producer_payload={"component_id": "1"},
        producer_inputs={
            "model": session_state_module.RemoteSessionBridgeRef(
                bridge_key="RSB_upstream",
                node_id="2",
                output_index=0,
                session_id="session-b",
            ).to_payload()
        },
        node_id="3",
        output_index=0,
    )

    assert first_key == second_key


def test_remote_session_store_resolves_bridge_refs_via_replay_callback(
    session_state_module: Any,
) -> None:
    """Durable bridge refs should rehydrate into the target session when the source is gone."""
    store = session_state_module.InMemoryRemoteSessionStore()
    target_handle = session_state_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = session_state_module.RemoteSessionBridgeRef(
        bridge_key="RSB_bridge",
        node_id="node-7",
        output_index=0,
        session_id="session-source",
    )
    observed_bridge_keys: list[str] = []

    def bridge_resolver(ref: Any) -> str:
        """Populate the target session and return the replayed value."""
        observed_bridge_keys.append(ref.bridge_key)
        store.put_output(
            target_handle,
            node_id=ref.node_id,
            output_index=ref.output_index,
            value="rehydrated-value",
        )
        return "rehydrated-value"

    assert (
        store.resolve_value_with_bridges(
            bridge_ref.to_payload(),
            target_session_handle=target_handle,
            bridge_resolver=bridge_resolver,
        )
        == "rehydrated-value"
    )
    assert observed_bridge_keys == ["RSB_bridge"]
    assert (
        store.resolve_value_with_bridges(
            bridge_ref.to_payload(),
            target_session_handle=target_handle,
            bridge_resolver=bridge_resolver,
        )
        == "rehydrated-value"
    )
    assert observed_bridge_keys == ["RSB_bridge"]
