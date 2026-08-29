"""Tests for durable remote invocation and oversized bridge payload handling."""

from __future__ import annotations

import hashlib
import importlib
import logging
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import pytest


def _cloud_durable_invocation_owner() -> Any:
    """Return the module that owns cloud durable-invocation mutable state."""
    return importlib.import_module("cloud_durable_invocation")


def _cloud_session_bridge_owner() -> Any:
    """Return the module that owns cloud session-bridge mutable state."""
    return importlib.import_module("cloud_session_bridge")


def _patch_cloud_durable_state(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
) -> None:
    """Patch mutable durable-invocation state at its extracted owner."""
    monkeypatch.setattr(_cloud_durable_invocation_owner(), name, value)


def test_remote_worker_affinity_separates_llm_and_comfy_pools(
    remote_modal_app_module: Any,
) -> None:
    """LLM phases must not reuse a warm worker holding ordinary Comfy models."""
    common_payload = {
        "component_id": "component-1",
        "remote_local_gap_pool": True,
    }

    comfy_key = remote_modal_app_module._remote_worker_affinity_key(
        {**common_payload, "remote_worker_affinity_group": "comfy"}
    )
    llm_key = remote_modal_app_module._remote_worker_affinity_key(
        {**common_payload, "remote_worker_affinity_group": "llm"}
    )

    assert comfy_key == "worker-pool:comfy:slot:0"
    assert llm_key == "worker-pool:llm:slot:0"
    assert comfy_key != llm_key


@pytest.mark.parametrize(
    "module_fixture_name",
    ["modal_cloud_module", "host_session_bridge_module"],
)
def test_modal_durable_store_reads_committed_object_without_volume_reload(
    module_fixture_name: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale mounted object should be fetched through Volume.read_file()."""
    runtime_module = request.getfixturevalue(module_fixture_name)
    payload = b"committed-in-another-container"
    digest = hashlib.sha256(payload).hexdigest()
    object_ref = runtime_module.DurableObjectRef(
        object_path=f"bridge_inputs/{digest[:2]}/{digest}.bin",
        sha256=digest,
        size_bytes=len(payload),
    )

    class FakeVolume:
        """Expose direct reads while rejecting any accidental mounted reload."""

        def __init__(self) -> None:
            """Initialize observed direct-read paths."""
            self.read_paths: list[str] = []

        def commit(self) -> None:
            """Accept unused durable commits for store construction."""

        def read_file(self, volume_path: str) -> Iterator[bytes]:
            """Yield the committed payload in multiple Modal-style chunks."""
            self.read_paths.append(volume_path)
            yield payload[:8]
            yield payload[8:]

        def reload(self) -> None:
            """Fail if the obsolete mounted-volume fallback is attempted."""
            raise AssertionError("Volume.reload() must not serve durable object reads.")

    volume = FakeVolume()
    if module_fixture_name == "modal_cloud_module":
        durable_invocation_module = _cloud_durable_invocation_owner()
        monkeypatch.setattr(durable_invocation_module, "volume_store", lambda: volume)
        monkeypatch.setattr(durable_invocation_module, "_DURABLE_OBJECT_STORE", None)
        monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    else:
        monkeypatch.setattr(runtime_module, "vol", volume, raising=False)
        monkeypatch.setattr(runtime_module, "_DURABLE_OBJECT_STORE", None)
    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(tmp_path / "storage"))
    if module_fixture_name != "modal_cloud_module":
        monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    runtime_module.get_settings.cache_clear()

    try:
        restored_payload = runtime_module._durable_object_store().get(object_ref)
    finally:
        runtime_module.get_settings.cache_clear()

    assert restored_payload == payload
    assert volume.read_paths == [f"durable_objects/{object_ref.object_path}"]


def test_cloud_invocation_replays_completed_result_without_reexecution(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client retry should receive the first completed result without rerunning work."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
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
    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
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
    caplog: pytest.LogCaptureFixture,
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

    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setattr(modal_cloud_module, "execute_node_locally", execute_node_locally)
    monkeypatch.setenv("COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES", "1")
    modal_cloud_module.get_settings.cache_clear()
    consumer_thread_id = threading.current_thread().ident
    caplog.set_level(logging.INFO)

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
    assert "Starting durable object batch commit" in caplog.text
    assert "Finished durable object batch commit" in caplog.text


def test_cloud_stream_logs_result_persistence_and_transport_boundaries(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Terminal-result diagnostics should distinguish buffering, persistence, and yield time."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    object_store = modal_cloud_module.FileDurableObjectStore(tmp_path)
    serialized_outputs = modal_cloud_module.serialize_node_outputs(("inline-result",))
    payload = {
        "invocation_id": "RIV_timing",
        "component_id": "component-timing",
    }

    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setattr(
        modal_cloud_module,
        "execute_node_locally",
        lambda payload, kwargs_payload: serialized_outputs,
    )
    caplog.set_level(logging.INFO, logger=modal_cloud_module.__name__)
    caplog.set_level(
        logging.INFO,
        logger=_cloud_durable_invocation_owner().__name__,
    )

    events = list(
        modal_cloud_module._stream_remote_payload_events(payload, b"inputs")
    )

    assert events == [{"kind": "result", "outputs": serialized_outputs}]
    assert "Remote stream worker produced result component=component-timing" in caplog.text
    assert "Finished publishing remote stream result to event buffer" in caplog.text
    assert "Remote stream consumer received buffered result" in caplog.text
    assert "result_storage=inline" in caplog.text
    assert "state=completed" in caplog.text
    assert f"inline_result_bytes={len(serialized_outputs)}" in caplog.text
    assert "Yielding remote stream result to Modal transport" in caplog.text
    assert "Remote stream result yield released after" in caplog.text


def test_cloud_invocation_waits_for_active_attempt_and_replays_result(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overlapping delivery should replay a result that finishes during its grace wait."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    invocation_id = "RIV_wait_for_result"
    created_at = time.time()
    invocation_store.put_record(
        modal_cloud_module.RemoteInvocationRecord(
            invocation_id=invocation_id,
            state="running",
            attempt=1,
            created_at=created_at,
            updated_at=created_at,
        )
    )
    serialized_outputs = modal_cloud_module.serialize_node_outputs(("completed",))
    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_RETRY_WAIT_SECONDS", 1.0)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_RETRY_POLL_SECONDS", 0.005)
    execution_count = 0

    def complete_original_attempt() -> None:
        """Publish the original attempt's result after the retry begins waiting."""
        time.sleep(0.02)
        invocation_store.put_record(
            modal_cloud_module.RemoteInvocationRecord(
                invocation_id=invocation_id,
                state="completed",
                attempt=1,
                created_at=created_at,
                updated_at=time.time(),
                result_inline=serialized_outputs,
            )
        )

    def execute_once() -> bytes:
        """Track any erroneous duplicate execution."""
        nonlocal execution_count
        execution_count += 1
        return serialized_outputs

    completion_thread = threading.Thread(target=complete_original_attempt)
    completion_thread.start()
    try:
        result = modal_cloud_module._execute_with_durable_invocation(
            {"invocation_id": invocation_id},
            execute_once,
        )
    finally:
        completion_thread.join(timeout=1.0)

    assert result == serialized_outputs
    assert execution_count == 0


def test_cloud_stream_abandonment_cancels_and_retries_failed_attempt(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Losing a stream should cancel its worker and leave the invocation retryable."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    commit_thread_ids: list[int | None] = []
    object_store = modal_cloud_module.FileDurableObjectStore(
        tmp_path,
        commit_callback=lambda: commit_thread_ids.append(threading.current_thread().ident),
    )
    cancellation_event = threading.Event()
    worker_stopped = threading.Event()
    invocation_id = "RIV_abandoned_stream"

    def execute_subgraph_locally(
        payload: Any,
        kwargs_payload: Any,
        *,
        status_callback: Any,
        cancellation_event: threading.Event | None = None,
        interrupt_store: Any = None,
        interrupt_flag_key: str | None = None,
    ) -> bytes:
        """Emit progress, then block until stream abandonment requests cancellation."""
        del payload, kwargs_payload, interrupt_store, interrupt_flag_key
        object_store.put("bridge_outputs", b"uncommitted-abandoned-output")
        status_callback({"event_type": "status", "phase": "executing"})
        try:
            assert cancellation_event is not None
            assert cancellation_event.wait(timeout=2.0)
            raise modal_cloud_module.RemoteSubgraphExecutionError(
                "abandoned stream cancelled"
            )
        finally:
            worker_stopped.set()

    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setattr(
        modal_cloud_module,
        "execute_subgraph_locally",
        execute_subgraph_locally,
    )
    monkeypatch.setattr(modal_cloud_module, "_REMOTE_INVOCATION_ABANDON_JOIN_SECONDS", 0.5)
    payload = {
        "invocation_id": invocation_id,
        "component_id": "component-1",
        "payload_kind": "subgraph",
    }
    stream = modal_cloud_module._stream_remote_payload_events(
        payload,
        b"inputs",
        cancellation_event=cancellation_event,
    )

    assert next(stream)["kind"] == "progress"
    stream.close()

    failed_record = invocation_store.get_record(invocation_id)
    assert cancellation_event.is_set()
    assert worker_stopped.wait(timeout=1.0)
    assert failed_record.state == "failed"
    assert failed_record.error_type == "RemoteInvocationAbandonedError"
    assert commit_thread_ids == []

    retry_outputs = modal_cloud_module.serialize_node_outputs(("retried",))
    retry_result = modal_cloud_module._execute_with_durable_invocation(
        payload,
        lambda: retry_outputs,
    )
    completed_record = invocation_store.get_record(invocation_id)

    assert retry_result == retry_outputs
    assert completed_record.state == "completed"
    assert completed_record.attempt == 2
    assert commit_thread_ids == []


def test_cloud_stream_close_after_result_preserves_completed_record(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Closing after the terminal result must not downgrade completed metadata."""
    invocation_store = modal_cloud_module.InMemoryRemoteInvocationStore()
    object_store = modal_cloud_module.FileDurableObjectStore(tmp_path)
    serialized_outputs = modal_cloud_module.serialize_node_outputs(("completed",))
    invocation_id = "RIV_close_after_result"

    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setattr(
        modal_cloud_module,
        "execute_node_locally",
        lambda payload, kwargs_payload: serialized_outputs,
    )
    stream = modal_cloud_module._stream_remote_payload_events(
        {"invocation_id": invocation_id, "component_id": "component-1"},
        b"inputs",
    )

    assert next(stream) == {"kind": "result", "outputs": serialized_outputs}
    stream.close()

    completed_record = invocation_store.get_record(invocation_id)
    assert completed_record.state == "completed"
    assert completed_record.attempt == 1


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
    _patch_cloud_durable_state(monkeypatch, "invocation_record_store", lambda: None)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_STORE", invocation_store)
    _patch_cloud_durable_state(monkeypatch, "_REMOTE_INVOCATION_RETRY_WAIT_SECONDS", 0.0)

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
    _patch_cloud_durable_state(
        monkeypatch,
        "invocation_record_store",
        lambda: barrier_store,
    )

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


def test_cloud_direct_bridge_output_omits_large_producer_inputs(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A directly restorable output should not persist producer inputs."""
    import torch

    object_store = modal_cloud_module.FileDurableObjectStore(tmp_path)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
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
        with pytest.raises(
            modal_cloud_module.RemoteSessionStateError,
            match="intentionally omitted",
        ):
            modal_cloud_module._deserialize_remote_session_bridge_producer_inputs(
                record
            )
    finally:
        _cloud_session_bridge_owner()._REMOTE_SESSION_STORE.clear_session(
            target_handle
        )
        modal_cloud_module.get_settings.cache_clear()

    assert record.producer_inputs == {}
    assert record.producer_inputs_object is None
    assert record.producer_inputs_retained is False
    assert record.recovery_kind is (
        modal_cloud_module.RemoteSessionBridgeRecoveryKind.SERIALIZED_OUTPUT
    )
    assert record.serialized_output is None
    assert record.serialized_output_object is not None
    assert torch.equal(restored_value["samples"], latent["samples"])


def test_cloud_large_image_bridge_output_uses_object_store(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote IMAGE boundaries should restore from Volume bytes without replay."""
    import torch

    object_store = modal_cloud_module.FileDurableObjectStore(tmp_path)
    _patch_cloud_durable_state(monkeypatch, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setenv("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", "1")
    modal_cloud_module.get_settings.cache_clear()
    image = torch.arange(48, dtype=torch.float32).reshape(1, 4, 4, 3)
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-image-object-backed",
        prompt_id="prompt-1",
        owner_component_id="component-2",
    )

    try:
        record = modal_cloud_module._build_remote_session_bridge_record(
            payload={"component_id": "component-1"},
            hydrated_inputs={},
            node_id="vae-decode-1",
            output_index=0,
            io_type="IMAGE",
            output_value=image,
        )
        monkeypatch.setattr(
            _cloud_session_bridge_owner(),
            "_execute_subgraph_prompt",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("durable IMAGE restore should skip producer replay")
            ),
        )
        restored_value = (
            modal_cloud_module._restore_serialized_remote_session_bridge_value(
                record,
                target_session_handle=target_handle,
            )
        )
    finally:
        _cloud_session_bridge_owner()._REMOTE_SESSION_STORE.clear_session(
            target_handle
        )
        modal_cloud_module.get_settings.cache_clear()

    assert record.serialized_output is None
    assert record.serialized_output_object is not None
    assert record.serialized_output_io_type == "IMAGE"
    assert torch.equal(restored_value, image)


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
    assert observed_payloads[0]["capture_remote_outputs"] is True
    assert "invocation_id" not in payload
    assert "capture_remote_outputs" not in payload


def test_local_client_retries_exhausted_llm_memory_on_fresh_throughput_worker(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marked recovery timeout should rotate affinity and retry exactly once."""
    observed_calls: list[tuple[Any, dict[str, Any]]] = []
    affinity_overrides: list[str | None] = []

    def invoke_payload(
        remote_engine: Any,
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: Any,
    ) -> bytes:
        """Fail the dirty worker and complete on the rotated worker."""
        del kwargs_payload, cancellation_event
        observed_calls.append((remote_engine, dict(payload)))
        if remote_engine == "dirty-worker":
            raise remote_modal_app_module.RemoteSubgraphExecutionError(
                "[comfy-modal-llm-memory-recovery-exhausted] "
                "vllm_mode=throughput recovery timed out"
            )
        return b"recovered"

    def lookup_engine(
        payload: dict[str, Any],
        *,
        affinity_key_override: str | None = None,
        protocol_probe: bool = False,
    ) -> str:
        """Capture the rotated worker identity selected by recovery."""
        del payload, protocol_probe
        affinity_overrides.append(affinity_key_override)
        return "fresh-worker"

    monkeypatch.setattr(
        remote_modal_app_module,
        "_invoke_remote_engine_payload",
        invoke_payload,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_lookup_deployed_remote_engine",
        lookup_engine,
    )
    monkeypatch.setattr(remote_modal_app_module, "_modal_lookup_error_types", lambda: ())
    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_remote_startup_status",
        lambda *args, **kwargs: None,
    )

    result = remote_modal_app_module._invoke_remote_engine_payload_with_recovery(
        "dirty-worker",
        {
            "component_id": "llm-component",
            "payload_kind": "subgraph",
            "remote_worker_affinity_group": "llm",
        },
        b"serialized-inputs",
        None,
    )

    assert result == b"recovered"
    assert len(observed_calls) == 2
    assert observed_calls[0][1]["invocation_id"] == observed_calls[1][1]["invocation_id"]
    assert observed_calls[1][1]["force_vllm_throughput_after_memory_recovery"] is True
    assert affinity_overrides[0] is not None
    assert affinity_overrides[0].startswith(
        "worker-pool:llm:slot:0:llm-memory-recovery:"
    )


def test_local_fallback_restores_object_backed_bridge_output(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The local fallback should use the same large bridge object format as Modal."""
    import torch

    object_store = remote_modal_app_module.FileDurableObjectStore(tmp_path)
    monkeypatch.setattr(host_session_bridge_module, "_DURABLE_OBJECT_STORE", object_store)
    monkeypatch.setenv("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", "1")
    remote_modal_app_module.get_settings.cache_clear()
    latent = {"samples": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)}
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="local-object-backed",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )

    try:
        record = host_session_bridge_module._build_remote_session_bridge_record(
            payload={"component_id": "component-1"},
            hydrated_inputs={"latent": latent},
            node_id="node-1",
            output_index=0,
            io_type="LATENT",
            output_value=latent,
        )
        restored_value = (
            host_session_bridge_module._restore_serialized_remote_session_bridge_value(
                record,
                target_session_handle=target_handle,
            )
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)
        remote_modal_app_module.get_settings.cache_clear()

    assert record.producer_inputs_object is None
    assert record.producer_inputs_retained is False
    assert record.recovery_kind is (
        remote_modal_app_module.RemoteSessionBridgeRecoveryKind.SERIALIZED_OUTPUT
    )
    assert record.serialized_output_object is not None
    assert torch.equal(restored_value["samples"], latent["samples"])


@pytest.mark.parametrize(
    "module_fixture_name",
    ["modal_cloud_module", "host_session_bridge_module"],
)
def test_direct_bridge_identity_includes_omitted_producer_inputs(
    module_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Omitted producer inputs should still distinguish durable bridge identities."""
    runtime_module = request.getfixturevalue(module_fixture_name)

    first_record = runtime_module._build_remote_session_bridge_record(
        payload={"component_id": "component-1"},
        hydrated_inputs={"seed": 1},
        node_id="node-1",
        output_index=0,
        io_type="INT",
        output_value=7,
    )
    second_record = runtime_module._build_remote_session_bridge_record(
        payload={"component_id": "component-1"},
        hydrated_inputs={"seed": 2},
        node_id="node-1",
        output_index=0,
        io_type="INT",
        output_value=7,
    )

    assert first_record.bridge_key != second_record.bridge_key
    assert first_record.producer_inputs == {}
    assert second_record.producer_inputs == {}
    assert first_record.producer_inputs_retained is False
    assert second_record.producer_inputs_retained is False


@pytest.mark.parametrize(
    "module_fixture_name",
    ["modal_cloud_module", "host_session_bridge_module"],
)
def test_literal_loader_plan_omits_producer_inputs(
    module_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """A self-contained loader plan should not persist producer component inputs."""
    runtime_module = request.getfixturevalue(module_fixture_name)

    record = runtime_module._build_remote_session_bridge_record(
        payload={
            "component_id": "loader-component",
            "subgraph_prompt": {
                "vae-loader": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "video-vae.safetensors"},
                }
            },
        },
        hydrated_inputs={"unrelated": "large-producer-value"},
        node_id="vae-loader",
        output_index=0,
        io_type="VAE",
        output_value=object(),
    )

    assert record.recovery_kind is (
        runtime_module.RemoteSessionBridgeRecoveryKind.SINGLE_NODE_PLAN
    )
    assert record.rehydration_plan == {
        "kind": "single_node_output",
        "node_data": {"class_type": "VAELoader"},
        "node_inputs": {"vae_name": "video-vae.safetensors"},
    }
    assert record.producer_inputs_retained is False
    assert record.producer_inputs == {}
    assert record.producer_inputs_object is None


@pytest.mark.parametrize(
    "module_fixture_name",
    ["modal_cloud_module", "host_session_bridge_module"],
)
def test_linked_loader_plan_retains_only_required_boundary_inputs(
    module_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """A linked loader plan should persist only its reduced dependency inputs."""
    runtime_module = request.getfixturevalue(module_fixture_name)
    payload = {
        "component_id": "loader-component",
        "subgraph_prompt": {
            "source": {"class_type": "BoundarySource", "inputs": {"value": 0}},
            "loader": {
                "class_type": "LinkedModelLoader",
                "inputs": {"source": ["source", 0]},
            },
            "unrelated": {
                "class_type": "BoundarySource",
                "inputs": {"value": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "needed",
                "io_type": "STRING",
                "targets": [{"node_id": "source", "input_name": "value"}],
            },
            {
                "proxy_input_name": "unneeded",
                "io_type": "STRING",
                "targets": [{"node_id": "unrelated", "input_name": "value"}],
            },
        ],
    }

    record = runtime_module._build_remote_session_bridge_record(
        payload=payload,
        hydrated_inputs={"needed": "keep", "unneeded": "omit"},
        node_id="loader",
        output_index=0,
        io_type="MODEL",
        output_value=object(),
    )

    assert record.recovery_kind is (
        runtime_module.RemoteSessionBridgeRecoveryKind.SUBGRAPH_PLAN
    )
    assert record.producer_inputs_retained is True
    assert runtime_module._deserialize_remote_session_bridge_producer_inputs(record) == {
        "needed": "keep"
    }
    assert record.rehydration_plan is not None
    assert [
        boundary_input["proxy_input_name"]
        for boundary_input in record.rehydration_plan["payload"]["boundary_inputs"]
    ] == ["needed"]
