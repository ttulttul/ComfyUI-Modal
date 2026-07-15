"""Tests for durable remote invocation and oversized bridge payload handling."""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any

import pytest


def test_cloud_invocation_replays_completed_result_without_reexecution(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client retry should receive the first completed result without rerunning work."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    monkeypatch.setattr(modal_cloud_module, "invocation_records", None, raising=False)
    monkeypatch.setattr(modal_cloud_module, "_REMOTE_INVOCATION_STORE", invocation_store)
    serialized_outputs = modal_cloud_module.serialize_node_outputs(("completed",))
    execution_count = 0

    def execute_once() -> bytes:
        """Return one result while tracking actual executions."""
        nonlocal execution_count
        execution_count += 1
        return serialized_outputs

    payload = {"invocation_id": "RIV_retry", "component_id": "component-1"}

    first_result = modal_cloud_module._execute_with_durable_invocation(
        payload,
        execute_once,
    )
    retry_result = modal_cloud_module._execute_with_durable_invocation(
        payload,
        execute_once,
    )

    assert first_result == serialized_outputs
    assert retry_result == serialized_outputs
    assert execution_count == 1
    assert invocation_store.get_record("RIV_retry").state == "completed"


def test_cloud_invocation_offloads_large_result_and_retries_failures(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Large results should use object storage and failed attempts should remain retryable."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    object_store = modal_cloud_module.FileDurableObjectStore(tmp_path)
    monkeypatch.setattr(modal_cloud_module, "invocation_records", None, raising=False)
    monkeypatch.setattr(modal_cloud_module, "_REMOTE_INVOCATION_STORE", invocation_store)
    monkeypatch.setattr(modal_cloud_module, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setenv("COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES", "1")
    modal_cloud_module.get_settings.cache_clear()
    payload = {"invocation_id": "RIV_large", "component_id": "component-1"}
    serialized_outputs = modal_cloud_module.serialize_node_outputs(("large-result",))
    attempt_count = 0

    def execute_once() -> bytes:
        """Fail the first attempt and complete the second."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ValueError("transient execution failure")
        return serialized_outputs

    try:
        with pytest.raises(ValueError, match="transient"):
            modal_cloud_module._execute_with_durable_invocation(payload, execute_once)
        failed_record = invocation_store.get_record("RIV_large")
        assert failed_record.state == "failed"

        result = modal_cloud_module._execute_with_durable_invocation(payload, execute_once)
        completed_record = invocation_store.get_record("RIV_large")
    finally:
        modal_cloud_module.get_settings.cache_clear()

    assert result == serialized_outputs
    assert completed_record.state == "completed"
    assert completed_record.attempt == 2
    assert completed_record.result_inline is None
    assert completed_record.result_object is not None
    assert object_store.get(completed_record.result_object) == serialized_outputs


def test_cloud_stream_commits_durable_outputs_on_consumer_thread(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Streaming execution should batch worker writes into one consumer commit."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    commit_thread_ids: list[int | None] = []
    object_store = modal_cloud_module.FileDurableObjectStore(
        tmp_path,
        commit_callback=lambda: commit_thread_ids.append(threading.current_thread().ident),
    )
    serialized_outputs = modal_cloud_module.serialize_node_outputs(("large-result",))

    def execute_node_locally(payload: Any, kwargs_payload: Any) -> bytes:
        """Write worker-produced bridge state before returning the node result."""
        del payload, kwargs_payload
        object_store.put("bridge_outputs", b"bridge-result")
        return serialized_outputs

    monkeypatch.setattr(modal_cloud_module, "invocation_records", None, raising=False)
    monkeypatch.setattr(modal_cloud_module, "_REMOTE_INVOCATION_STORE", invocation_store)
    monkeypatch.setattr(modal_cloud_module, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setattr(modal_cloud_module, "execute_node_locally", execute_node_locally)
    monkeypatch.setenv("COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES", "1")
    modal_cloud_module.get_settings.cache_clear()
    consumer_thread_id = threading.current_thread().ident

    try:
        events = list(
            modal_cloud_module._stream_remote_payload_events(
                {"invocation_id": "RIV_stream", "component_id": "component-1"},
                b"inputs",
            )
        )
    finally:
        modal_cloud_module.get_settings.cache_clear()

    completed_record = invocation_store.get_record("RIV_stream")
    assert events == [{"kind": "result", "outputs": serialized_outputs}]
    assert completed_record.state == "completed"
    assert completed_record.result_object is not None
    assert object_store.get(completed_record.result_object) == serialized_outputs
    assert commit_thread_ids == [consumer_thread_id]


def test_cloud_invocation_rejects_duplicate_active_attempt(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping deliveries of one invocation must not run the payload twice."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    invocation_store.put_record(
        modal_cloud_module.RemoteInvocationRecord(
            invocation_id="RIV_running",
            state="running",
            attempt=1,
            created_at=time.time(),
            updated_at=time.time(),
        )
    )
    monkeypatch.setattr(modal_cloud_module, "invocation_records", None, raising=False)
    monkeypatch.setattr(modal_cloud_module, "_REMOTE_INVOCATION_STORE", invocation_store)

    with pytest.raises(
        modal_cloud_module.RemoteInvocationInProgressError,
        match="already running",
    ):
        modal_cloud_module._execute_with_durable_invocation(
            {"invocation_id": "RIV_running"},
            lambda: b"must-not-run",
        )


def test_cloud_canary_echo_reports_binary_transport(
    modal_cloud_module: Any,
) -> None:
    """The dependency-light canary should round-trip tensors through binary RPC bytes."""
    import torch

    value = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    kwargs_payload = modal_cloud_module.serialize_node_inputs({"value": value})

    response = modal_cloud_module._execute_canary_payload(
        {"payload_kind": "canary", "component_id": "canary-echo"},
        kwargs_payload,
        cancellation_event=threading.Event(),
        interrupt_store=None,
        interrupt_flag_key=None,
    )
    restored_value, metadata = modal_cloud_module.deserialize_node_outputs(response)

    assert kwargs_payload.startswith(b"CMODALB1")
    assert response.startswith(b"CMODALB1")
    assert torch.equal(restored_value, value)
    assert metadata["transport_kind"] == "binary"
    assert metadata["component_id"] == "canary-echo"


def test_cloud_canary_observes_cancellation_before_work(
    modal_cloud_module: Any,
) -> None:
    """A canary should fail immediately when its local cancellation event is set."""
    cancellation_event = threading.Event()
    cancellation_event.set()

    with pytest.raises(
        modal_cloud_module.RemoteCanaryInterruptedError,
        match="interrupted",
    ):
        modal_cloud_module._execute_canary_payload(
            {
                "payload_kind": "canary",
                "component_id": "canary-cancel",
                "canary_delay_seconds": 1.0,
            },
            modal_cloud_module.serialize_node_inputs({"value": "unused"}),
            cancellation_event=cancellation_event,
            interrupt_store=None,
            interrupt_flag_key=None,
        )


def test_cloud_canary_consumes_shared_interrupt_flag(
    modal_cloud_module: Any,
) -> None:
    """The live canary should consume the same Modal Dict flag as real prompts."""

    class InterruptStore(dict[str, Any]):
        """Minimal Modal Dict-like interrupt store for the canary helper."""

        def contains(self, key: str) -> bool:
            """Return whether one interrupt key is present."""
            return key in self

    cancellation_event = threading.Event()
    interrupt_store = InterruptStore({"prompt-1:component-1": {"requested_at": time.time()}})

    with pytest.raises(
        modal_cloud_module.RemoteCanaryInterruptedError,
        match="interrupted",
    ):
        modal_cloud_module._execute_canary_payload(
            {
                "payload_kind": "canary",
                "component_id": "component-1",
                "canary_delay_seconds": 1.0,
            },
            modal_cloud_module.serialize_node_inputs({"value": "unused"}),
            cancellation_event=cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key="prompt-1:component-1",
        )

    assert cancellation_event.is_set()
    assert "prompt-1:component-1" not in interrupt_store


def test_cloud_canary_barrier_coordinates_shared_store_members(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel canaries should rendezvous through their shared Modal Dict markers."""
    barrier_store: dict[str, Any] = {
        "CANARY_BARRIER:barrier-1:member-b": {"ready_at": time.time()}
    }
    monkeypatch.setattr(modal_cloud_module, "invocation_records", barrier_store, raising=False)

    released_at = modal_cloud_module._wait_for_canary_barrier(
        {
            "barrier_id": "barrier-1",
            "member_id": "member-a",
            "members": ["member-a", "member-b"],
            "timeout_seconds": 1.0,
        },
        component_id="canary-a",
        cancellation_event=threading.Event(),
        interrupt_store=None,
        interrupt_flag_key=None,
    )

    assert released_at is not None
    assert "CANARY_BARRIER:barrier-1:member-a" in barrier_store


def test_cloud_large_bridge_inputs_and_output_use_object_store(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Oversized bridge state should keep only content-addressed refs in metadata."""
    import torch

    object_store = modal_cloud_module.FileDurableObjectStore(tmp_path)
    monkeypatch.setattr(modal_cloud_module, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setenv("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", "1")
    modal_cloud_module.get_settings.cache_clear()
    latent = {"samples": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)}
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-object-backed",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )

    try:
        record = modal_cloud_module._build_remote_session_bridge_record(
            payload={"component_id": "component-1"},
            hydrated_inputs={"latent": latent},
            node_id="node-1",
            output_index=0,
            io_type="LATENT",
            output_value=latent,
        )
        restored_value = (
            modal_cloud_module._restore_serialized_remote_session_bridge_value(
                record,
                target_session_handle=target_handle,
            )
        )
        restored_inputs = (
            modal_cloud_module._deserialize_remote_session_bridge_producer_inputs(
                record
            )
        )
    finally:
        modal_cloud_module._REMOTE_SESSION_STORE.clear_session(target_handle)
        modal_cloud_module.get_settings.cache_clear()

    assert record.producer_inputs == {}
    assert record.producer_inputs_object is not None
    assert record.serialized_output is None
    assert record.serialized_output_object is not None
    assert torch.equal(restored_value["samples"], latent["samples"])
    assert torch.equal(restored_inputs["latent"]["samples"], latent["samples"])


def test_local_client_attaches_stable_invocation_id(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every deployed and recovered call should carry the same client idempotency key."""
    observed_payloads: list[dict[str, Any]] = []

    def invoke_payload(
        remote_engine: Any,
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: Any,
    ) -> bytes:
        """Capture the payload passed to the Modal handle."""
        del remote_engine, kwargs_payload, cancellation_event
        observed_payloads.append(payload)
        return b"result"

    monkeypatch.setattr(
        remote_modal_app_module,
        "_invoke_remote_engine_payload",
        invoke_payload,
    )
    monkeypatch.setattr(remote_modal_app_module, "_modal_lookup_error_types", lambda: ())
    payload = {"component_id": "component-1", "payload_kind": "subgraph"}

    result = remote_modal_app_module._invoke_remote_engine_payload_with_recovery(
        object(),
        payload,
        b"serialized-inputs",
        None,
    )

    assert result == b"result"
    assert observed_payloads[0]["invocation_id"].startswith("RIV_")
    assert "invocation_id" not in payload


def test_local_fallback_restores_object_backed_bridge_output(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The local fallback should use the same large bridge object format as Modal."""
    import torch

    object_store = remote_modal_app_module.FileDurableObjectStore(tmp_path)
    monkeypatch.setattr(remote_modal_app_module, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setenv("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", "1")
    remote_modal_app_module.get_settings.cache_clear()
    latent = {"samples": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)}
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="local-object-backed",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )

    try:
        record = remote_modal_app_module._build_remote_session_bridge_record(
            payload={"component_id": "component-1"},
            hydrated_inputs={"latent": latent},
            node_id="node-1",
            output_index=0,
            io_type="LATENT",
            output_value=latent,
        )
        restored_value = (
            remote_modal_app_module._restore_serialized_remote_session_bridge_value(
                record,
                target_session_handle=target_handle,
            )
        )
    finally:
        remote_modal_app_module._REMOTE_SESSION_STORE.clear_session(target_handle)
        remote_modal_app_module.get_settings.cache_clear()

    assert record.producer_inputs_object is not None
    assert record.serialized_output_object is not None
    assert torch.equal(restored_value["samples"], latent["samples"])
