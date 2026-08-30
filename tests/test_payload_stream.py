"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

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

    monkeypatch.setattr(
        _cloud_streaming_owner(),
        "execute_subgraph_locally",
        fake_execute_subgraph_locally,
    )

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

def test_modal_cloud_streams_remote_log_task_id_before_progress(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote streamed payloads should surface the current Modal task id for local log mirroring."""

    def fake_execute_subgraph_locally(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        status_callback: Any = None,
    ) -> bytes:
        if status_callback is not None:
            status_callback({"phase": "executing", "active_node_id": "7"})
        return b"serialized-outputs"

    monkeypatch.setenv("MODAL_TASK_ID", "ta-remote-123")
    monkeypatch.setattr(
        _cloud_streaming_owner(),
        "execute_subgraph_locally",
        fake_execute_subgraph_locally,
    )

    events = list(
        modal_cloud_module._stream_remote_payload_events(
            {"payload_kind": "subgraph", "component_id": "component-1"},
            b"{}",
        )
    )

    assert events[0] == {"kind": "remote_logs", "task_id": "ta-remote-123"}
    assert events[1:] == [
        {
            "kind": "progress",
            "phase": "executing",
            "active_node_id": "7",
        },
        {
            "kind": "result",
            "outputs": b"serialized-outputs",
        },
    ]


def test_modal_cloud_returns_large_durable_result_by_reference(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal must not place a durable multi-gigabyte result back on its result channel."""
    cloud_streaming = _cloud_streaming_owner()
    result_ref = cloud_streaming.DurableObjectRef(
        object_path="invocation_results/ab/result.bin",
        sha256="a" * 64,
        size_bytes=8 * 1024 * 1024,
    )
    begin_calls: list[bool] = []
    monkeypatch.setattr(
        cloud_streaming,
        "_begin_remote_invocation",
        lambda _payload, *, preserve_result_ref=False: (
            begin_calls.append(preserve_result_ref) or object(),
            None,
        ),
    )
    monkeypatch.setattr(
        cloud_streaming,
        "_complete_remote_invocation",
        lambda *_args, **_kwargs: result_ref,
    )
    monkeypatch.setattr(
        cloud_streaming,
        "execute_node_locally",
        lambda *_args, **_kwargs: b"serialized-large-result",
    )

    events = list(
        modal_cloud_module._stream_remote_payload_events(
            {
                "payload_kind": "node",
                "component_id": "component-large",
                "invocation_id": "RIV_large",
                "execution_provider": "modal",
            },
            b"{}",
        )
    )

    assert begin_calls == [True]
    assert events == [{"kind": "result", "output_ref": result_ref.to_payload()}]

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

    monkeypatch.setattr(
        _cloud_streaming_owner(),
        "execute_subgraph_locally",
        fake_execute_subgraph_locally,
    )

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

def test_modal_cloud_stream_buffer_coalesces_progress_and_preserves_terminal_events(
    modal_cloud_module: Any,
) -> None:
    """A slow stream consumer should have bounded progress memory without losing results."""
    event_buffer = modal_cloud_module._BoundedStreamEventBuffer(maxsize=4)
    for value in range(100):
        event_buffer.publish_progress({"value": value})

    terminal_thread = threading.Thread(
        target=lambda: (
            event_buffer.publish_terminal("result", b"outputs"),
            event_buffer.publish_terminal("done", None),
        )
    )
    terminal_thread.start()
    observed_events: list[tuple[str, Any]] = []
    while True:
        event = event_buffer.get()
        observed_events.append(event)
        if event[0] == "done":
            break
    terminal_thread.join(timeout=1.0)
    event_buffer.close()

    progress_values = [payload["value"] for kind, payload in observed_events if kind == "progress"]
    assert progress_values == [96, 97, 98, 99]
    assert observed_events[-2:] == [("result", b"outputs"), ("done", None)]
    assert event_buffer.dropped_progress_events == 96
    assert not terminal_thread.is_alive()

def test_remote_modal_consumes_streamed_progress_and_result(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    local_ui_events_module: Any,
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
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
    monkeypatch.setattr(local_ui_events_module, "_lookup_local_prompt_server", lambda: prompt_server)

    payload = {
        "prompt_id": "prompt-1",
        "component_id": "component-1",
        "invocation_id": "RIV_stream_timing",
        "component_node_ids": ["7", "8"],
        "modal_gpu": "B300",
        "extra_data": {"client_id": "client-1"},
    }
    caplog.set_level(logging.INFO, logger=payload_stream_module.__name__)
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
    assert "Starting local Modal stream consumption component=component-1" in caplog.text
    assert "Received streamed Modal result component=component-1" in caplog.text
    assert "result_bytes=18" in caplog.text
    assert "event_count=4 progress_event_count=3" in caplog.text
    assert "Finished local Modal result stream close" in caplog.text
    assert "Finished local Modal stream consumption" in caplog.text
    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["7", "8"],
                "modal_gpu": "B300",
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
                "modal_gpu": "B300",
                    "status_message": "Receiving remote outputs from Modal",
            },
            "client-1",
        ),
    ]

def test_remote_stream_first_event_triggers_speculative_prewarm_once(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    monkeypatch: Any,
) -> None:
    """The current worker must emit an event before its future affinity is prepared."""
    scheduled_payloads: list[tuple[dict[str, Any], str]] = []
    monkeypatch.setattr(
        payload_stream_module,
        "_schedule_speculative_affinity_prewarm",
        lambda payload, *, reason: scheduled_payloads.append((dict(payload), reason))
        or True,
    )

    payload = {
        "prompt_id": "prompt-spec",
        "component_id": "component-current",
        "speculative_remote_prewarm_target": {
            "component_id": "component-next",
        },
    }
    result = remote_modal_app_module._consume_remote_payload_stream(
        payload,
        iter(
            [
                {"kind": "remote_logs", "task_id": "ta-spec"},
                {"kind": "progress", "phase": "executing"},
                {"kind": "result", "outputs": b"serialized-outputs"},
            ]
        ),
    )

    assert result == b"serialized-outputs"
    assert scheduled_payloads == [(payload, "current_remote_stream_started")]


def test_remote_modal_downloads_referenced_result_with_progress(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    monkeypatch: Any,
) -> None:
    """A durable Modal result should report download bytes while materializing locally."""
    expected_result = b"result"
    reporter_calls: list[tuple[str, int]] = []

    class Reporter:
        """Capture lifecycle calls made for the referenced result download."""

        def __init__(
            self,
            _payload: dict[str, Any],
            *,
            direction: str,
            total_bytes: int,
            indeterminate: bool = False,
        ) -> None:
            """Record the transfer metadata."""
            del indeterminate
            reporter_calls.append((direction, total_bytes))

        def start(self) -> None:
            """Record transfer start."""
            reporter_calls.append(("start", 0))

        def update(self, transferred_bytes: int, *, force: bool = False) -> None:
            """Record one cumulative-byte update."""
            del force
            reporter_calls.append(("update", transferred_bytes))

        def complete(self) -> None:
            """Record transfer completion."""
            reporter_calls.append(("complete", 0))

    def materialize(_ref: Any, *, progress_callback: Any = None) -> bytes:
        """Simulate a two-chunk Modal Volume download."""
        progress_callback(3)
        progress_callback(6)
        return expected_result

    monkeypatch.setattr(payload_stream_module, "RemoteTransferProgressReporter", Reporter)
    monkeypatch.setattr(
        payload_stream_module,
        "materialize_modal_durable_object",
        materialize,
    )
    output_ref = payload_stream_module.DurableObjectRef(
        object_path="invocation_results/ab/result.bin",
        sha256="b" * 64,
        size_bytes=6,
    )

    result = remote_modal_app_module._consume_remote_payload_stream(
        {"component_id": "component-large"},
        iter([{"kind": "result", "output_ref": output_ref.to_payload()}]),
    )

    assert result == expected_result
    assert reporter_calls == [
        ("upload", 0),
        ("start", 0),
        ("complete", 0),
        ("download", 6),
        ("start", 0),
        ("update", 3),
        ("update", 6),
        ("complete", 0),
    ]


def test_large_transfer_reporter_emits_byte_ui_metadata(
    local_ui_events_module: Any,
    monkeypatch: Any,
) -> None:
    """Large transfers should use the existing byte-aware node progress UI."""
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        local_ui_events_module,
        "_emit_local_modal_progress",
        lambda **kwargs: emitted.append(kwargs),
    )
    payload = {
        "prompt_id": "prompt-transfer",
        "component_id": "170",
        "component_node_ids": ["170", "169"],
        "execution_provider": "vast",
        "execution_environment_id": "vast:video-worker",
        "extra_data": {"client_id": "client-transfer"},
    }
    reporter = local_ui_events_module.RemoteTransferProgressReporter(
        payload,
        direction="download",
        total_bytes=8 * 1024 * 1024,
    )

    reporter.start()
    reporter.complete()

    assert emitted == [
        {
            "prompt_id": "prompt-transfer",
            "client_id": "client-transfer",
            "node_id": "170",
            "value": 0.0,
            "max_value": float(8 * 1024 * 1024),
            "display_node_id": "170",
            "stage": "download",
            "message": "Receiving outputs from vast:video-worker",
            "unit": "bytes",
            "indeterminate": False,
            "pre_gpu": False,
            "execution_provider": "vast",
            "execution_environment_id": "vast:video-worker",
        },
        {
            "prompt_id": "prompt-transfer",
            "client_id": "client-transfer",
            "node_id": "170",
            "value": float(8 * 1024 * 1024),
            "max_value": float(8 * 1024 * 1024),
            "display_node_id": "170",
            "stage": "download",
            "message": "Receiving outputs from vast:video-worker",
            "unit": "bytes",
            "indeterminate": False,
            "pre_gpu": False,
            "execution_provider": "vast",
            "execution_environment_id": "vast:video-worker",
        },
    ]

def test_emit_local_mapped_lane_progress_start_marks_lane_as_setup_only(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    monkeypatch: Any,
) -> None:
    """Provisioning a mapped lane should emit a dedicated setup-only lane progress event."""
    emitted_progress: list[dict[str, Any]] = []

    monkeypatch.setattr(
        mapped_execution_module,
        "_emit_local_modal_progress",
        lambda **kwargs: emitted_progress.append(kwargs),
    )

    remote_modal_app_module._emit_local_mapped_lane_progress_start(
        {
            "prompt_id": "prompt-1",
            "component_id": "component-1",
            "extra_data": {"client_id": "client-1"},
        },
        lane_index=3,
    )

    assert emitted_progress == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_id": "component-1",
            "value": 0.0,
            "max_value": 1.0,
            "display_node_id": "component-1",
            "lane_id": "3",
            "item_index": None,
            "setup_only": True,
        }
    ]

def test_remote_modal_consumes_remote_log_stream_events_with_retain_release(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    modal_container_logs_module: Any,
    monkeypatch: Any,
) -> None:
    """Stream consumers should retain and release one container-log watcher per remote payload."""
    log_stream_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        modal_container_logs_module,
        "get_settings",
        lambda: types.SimpleNamespace(stream_remote_container_logs=True),
    )
    monkeypatch.setattr(
        payload_stream_module,
        "_retain_remote_container_log_stream",
        lambda task_id: log_stream_calls.append(("retain", task_id)) or task_id,
    )
    monkeypatch.setattr(
        payload_stream_module,
        "_release_remote_container_log_stream",
        lambda task_id: log_stream_calls.append(("release", task_id)),
    )

    result = remote_modal_app_module._consume_remote_payload_stream(
        {
            "prompt_id": "prompt-1",
            "component_id": "component-1",
            "component_node_ids": ["7"],
            "extra_data": {"client_id": "client-1"},
        },
        iter(
            [
                {"kind": "remote_logs", "task_id": "ta-123"},
                {"kind": "result", "outputs": b"serialized-outputs"},
            ]
        ),
    )

    assert result == b"serialized-outputs"
    assert log_stream_calls == [("retain", "ta-123"), ("release", "ta-123")]

def test_remote_modal_stops_consuming_stream_after_terminal_result(
    remote_modal_app_module: Any,
) -> None:
    """The local stream consumer should not wait for extra events after the final result arrives."""

    class FakeStreamEvents:
        """Iterator that fails if the consumer requests post-result events."""

        def __init__(self) -> None:
            """Initialize the deterministic event sequence."""
            self._events = [
                {"kind": "remote_logs", "task_id": "ta-123"},
                {"kind": "result", "outputs": b"serialized-outputs"},
            ]
            self._index = 0
            self.close_calls = 0

        def __iter__(self) -> "FakeStreamEvents":
            """Return the iterator itself."""
            return self

        def __next__(self) -> dict[str, Any]:
            """Return the next event and reject any post-result polling."""
            if self._index >= len(self._events):
                raise AssertionError("The stream consumer requested events after the terminal result.")
            next_event = self._events[self._index]
            self._index += 1
            return next_event

        def close(self) -> None:
            """Record one best-effort close from the local consumer."""
            self.close_calls += 1

    stream_events = FakeStreamEvents()

    result = remote_modal_app_module._consume_remote_payload_stream(
        {
            "prompt_id": "prompt-1",
            "component_id": "component-1",
            "component_node_ids": ["7"],
            "extra_data": {"client_id": "client-1"},
        },
        stream_events,
    )

    assert result == b"serialized-outputs"
    assert stream_events.close_calls == 1

def test_remote_modal_cli_log_stream_mirrors_lines_to_stderr(
    modal_container_logs_module: Any,
    monkeypatch: Any,
) -> None:
    """CLI-backed container log streaming should mirror complete prefixed lines into local stderr."""

    class FakeStdout:
        """Expose a deterministic binary read interface for the fake Modal CLI process."""

        def __init__(self, chunks: list[bytes]) -> None:
            """Store the binary chunks that should be returned to the caller."""
            self._chunks = chunks
            self._index = 0

        def read(self, size: int = -1) -> bytes:
            """Return one queued binary chunk or EOF when exhausted."""
            del size
            if self._index >= len(self._chunks):
                return b""
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

    class FakeProcess:
        """Minimal subprocess stub used to emulate `modal container logs -f`."""

        def __init__(self) -> None:
            """Expose a stdout pipe and process lifecycle methods."""
            self.stdout = FakeStdout([b"session reuse\n", b"session miss\n"])
            self.returncode: int | None = None

        def poll(self) -> int | None:
            """Report process completion after all fake lines have been read."""
            if self.stdout._index >= len(self.stdout._chunks):
                self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            """Mark the fake process as stopped."""
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            """Return the final process exit code."""
            self.returncode = 0
            return 0

        def kill(self) -> None:
            """Mark the fake process as killed."""
            self.returncode = 0

    stderr_buffer = StringIO()
    monkeypatch.setattr(modal_container_logs_module.shutil, "which", lambda name: "/usr/bin/modal")
    monkeypatch.setattr(modal_container_logs_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        modal_container_logs_module.select,
        "select",
        lambda streams, _write, _error, _timeout: (
            streams if streams[0]._index < len(streams[0]._chunks) else [],
            [],
            [],
        ),
    )
    monkeypatch.setattr(modal_container_logs_module.sys, "stderr", stderr_buffer)

    result = modal_container_logs_module._stream_remote_container_logs_via_modal_cli(
        "ta-123",
        threading.Event(),
    )

    assert result is True
    assert stderr_buffer.getvalue() == (
        "[modal:ta-123] session reuse\n"
        "[modal:ta-123] session miss\n"
    )

def test_remote_modal_log_stream_prefers_cli_before_sdk(
    modal_container_logs_module: Any,
    monkeypatch: Any,
) -> None:
    """The local log watcher should avoid the SDK path when the Modal CLI is available."""
    backend_calls: list[str] = []

    monkeypatch.setattr(
        modal_container_logs_module,
        "_stream_remote_container_logs_via_modal_cli",
        lambda task_id, stop_event: backend_calls.append(f"cli:{task_id}") or True,
    )
    monkeypatch.setattr(
        modal_container_logs_module,
        "_stream_remote_container_logs_via_modal_sdk",
        lambda task_id, stop_event: backend_calls.append(f"sdk:{task_id}") or True,
    )

    modal_container_logs_module._run_remote_container_log_stream("ta-123", threading.Event())

    assert backend_calls == ["cli:ta-123"]

def test_remote_modal_log_stream_survives_short_container_reuse_gap(
    modal_container_logs_module: Any,
    monkeypatch: Any,
) -> None:
    """A reused task id should keep one watcher and avoid replaying container history."""
    created_threads: list[Any] = []
    created_timers: list[Any] = []

    class FakeThread:
        """Record watcher lifecycle without starting a real thread."""

        def __init__(self, **kwargs: Any) -> None:
            """Store thread construction arguments."""
            self.kwargs = kwargs
            self.started = False
            self.join_calls: list[float | None] = []
            created_threads.append(self)

        def start(self) -> None:
            """Mark the watcher as alive."""
            self.started = True

        def is_alive(self) -> bool:
            """Return whether the fake watcher has started."""
            return self.started

        def join(self, timeout: float | None = None) -> None:
            """Record one bounded watcher join."""
            self.join_calls.append(timeout)

    class FakeTimer:
        """Expose a manually fired daemon idle timer."""

        def __init__(
            self,
            interval: float,
            function: Any,
            args: tuple[Any, ...],
        ) -> None:
            """Store the delayed callback without scheduling it."""
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.started = False
            self.cancelled = False
            created_timers.append(self)

        def start(self) -> None:
            """Mark the timer as scheduled."""
            self.started = True

        def cancel(self) -> None:
            """Mark the scheduled callback as cancelled."""
            self.cancelled = True

        def fire(self) -> None:
            """Run the callback when it has not been cancelled."""
            if not self.cancelled:
                self.function(*self.args)

    monkeypatch.setattr(modal_container_logs_module, "_REMOTE_CONTAINER_LOG_STREAMS", {})
    monkeypatch.setattr(modal_container_logs_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(modal_container_logs_module.threading, "Timer", FakeTimer)

    assert modal_container_logs_module._retain_remote_container_log_stream("ta-reused") == "ta-reused"
    stream_state = modal_container_logs_module._REMOTE_CONTAINER_LOG_STREAMS["ta-reused"]
    modal_container_logs_module._release_remote_container_log_stream("ta-reused")

    assert len(created_threads) == 1
    assert len(created_timers) == 1
    assert created_timers[0].daemon is True
    assert stream_state.stop_event.is_set() is False

    modal_container_logs_module._retain_remote_container_log_stream("ta-reused")

    assert len(created_threads) == 1
    assert created_timers[0].cancelled is True
    assert stream_state.refcount == 1

    modal_container_logs_module._release_remote_container_log_stream("ta-reused")
    assert len(created_timers) == 2
    created_timers[1].fire()

    assert stream_state.stop_event.is_set() is True
    assert stream_state.thread.join_calls == [0.2]
    assert "ta-reused" not in modal_container_logs_module._REMOTE_CONTAINER_LOG_STREAMS

def test_remote_modal_consumes_streamed_executed_outputs_and_previews(
    remote_modal_app_module: Any,
    local_ui_events_module: Any,
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
    monkeypatch.setattr(local_ui_events_module, "_lookup_local_prompt_server", lambda: prompt_server)

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
    local_ui_events_module: Any,
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
    monkeypatch.setattr(local_ui_events_module, "_lookup_local_prompt_server", lambda: prompt_server)
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

def test_modal_cloud_tracing_prompt_server_preserves_llm_progress_metadata(
    modal_cloud_module: Any,
) -> None:
    """Structured LLM stages and token telemetry should survive remote tracing."""
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {"9": {"class_type": "ModalLLM", "inputs": {}}},
        status_callback=observed_updates.append,
    )

    server.send_sync(
        "modal_llm_progress",
        {
            "node_id": "9",
            "stage": "generating",
            "message": "Generating",
            "value": 23,
            "max": 128,
            "unit": "tokens",
            "time_to_first_token_seconds": 1.25,
            "tokens_per_second": 17.5,
        },
        None,
    )

    assert observed_updates == [
        {
            "event_type": "node_progress",
            "node_id": "9",
            "display_node_id": "9",
            "real_node_id": "9",
            "value": 23.0,
            "max": 128.0,
            "stage": "generating",
            "message": "Generating",
            "indeterminate": False,
            "unit": "tokens",
            "time_to_first_token_seconds": 1.25,
            "tokens_per_second": 17.5,
        }
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

def test_invoke_remote_engine_payload_stream_detects_local_interrupt_without_outer_sync(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
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
        modal_interrupts_module,
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

def test_implicit_mapping_propagates_scheduler_list_through_downstream_component(
    api_intercept_module: Any,
    modal_executor_module: Any,
    remote_modal_app_module: Any,
    serialization_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Heterogeneous implicit LATENT outputs should map into an ordinary local consumer."""
    torch = pytest.importorskip("torch")

    class PromptSeedList:
        """Expose integer seeds as one scheduler list output."""

        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("seeds",)
        OUTPUT_IS_LIST = (True,)

    class ModalMapInput:
        """Represent the explicit queue-time mapped input marker."""

        RETURN_TYPES = ("*",)
        RETURN_NAMES = ("value",)
        OUTPUT_IS_LIST = (False,)

    class RemoteSeed:
        """Publish an inexpensive mapped integer across a remote component boundary."""

        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("seed",)
        OUTPUT_IS_LIST = (False,)

    class RemoteLatent:
        """Publish one latent whose shape depends on its mapped seed."""

        RETURN_TYPES = ("LATENT",)
        RETURN_NAMES = ("samples",)
        OUTPUT_IS_LIST = (False,)

    class LocalVAEDecode:
        """Model the ordinary mapping contract used by ComfyUI's VAE decoder."""

        RETURN_TYPES = ("IMAGE",)
        RETURN_NAMES = ("image",)
        OUTPUT_IS_LIST = (False,)

        @staticmethod
        def decode(samples: dict[str, Any]) -> int:
            """Require a LATENT mapping and return its width."""
            return int(samples["samples"].shape[-1])

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "PromptSeedList": PromptSeedList,
                "ModalMapInput": ModalMapInput,
                "RemoteSeed": RemoteSeed,
                "RemoteLatent": RemoteLatent,
                "VAEDecode": LocalVAEDecode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 49, "properties": {"is_modal_remote": False}},
            {"id": 50, "properties": {"is_modal_remote": False}},
            {"id": 51, "properties": {"is_modal_remote": True}},
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 11, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "49": {"class_type": "PromptSeedList", "inputs": {}},
        "50": {"class_type": "ModalMapInput", "inputs": {"value": ["49", 0]}},
        "51": {"class_type": "RemoteSeed", "inputs": {"seed": ["50", 0]}},
        "1": {"class_type": "RemoteLatent", "inputs": {"seed": ["51", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
        extra_data={"prompt_id": "implicit-list-regression"},
    )

    downstream_proxy_id = rewritten_prompt["1"]["class_type"]
    downstream_proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[downstream_proxy_id]
    downstream_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    downstream_execution_payload = modal_executor_module._rehydrate_proxy_payload(
        downstream_payload,
        unique_id="1",
    )

    assert summary.remote_component_ids == ["51", "1"]
    assert summary.mapped_component_ids == ["1", "51"]
    assert (
        downstream_execution_payload["boundary_outputs"][0]["scheduler_is_list"]
        is True
    )
    assert downstream_proxy_class.OUTPUT_IS_LIST == [True, False]
    assert rewritten_prompt["11"]["inputs"]["samples"] == ["1", 0]

    async def fake_invoke_remote_engine_async(
        payload: dict[str, Any],
        kwargs_payload: bytes,
    ) -> bytes:
        """Return distinct latent shapes for the downstream mapped items."""
        if not payload.get("execute_node_ids"):
            return serialization_module.serialize_node_outputs(())
        hydrated_inputs = serialization_module.deserialize_node_inputs(
            kwargs_payload
        )
        seed = int(hydrated_inputs["remote_input_0"])
        latent_size = 32 if seed == 10 else 35
        latent = {
            "samples": torch.zeros(
                (1, 4, latent_size, latent_size),
                dtype=torch.float32,
            )
        }
        return serialization_module.serialize_node_outputs((latent,))

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    class ImplicitMappingClient:
        """Run the downstream payload through implicit mapped aggregation."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[Any, ...]:
            """Return deserialized outputs from the implicit mapped subgraph runner."""
            invoke_implicitly_mapped = (
                remote_modal_app_module._invoke_implicitly_mapped_subgraph_async
            )
            response = await invoke_implicitly_mapped(
                payload,
                serialization_module.serialize_node_inputs(kwargs),
            )
            return serialization_module.deserialize_node_outputs(response)

    modal_executor_module.set_remote_executor_client_factory(
        lambda: ImplicitMappingClient()
    )
    try:
        proxy_result = asyncio.run(
            downstream_proxy_class.execute(
                original_node_data=downstream_payload,
                remote_input_0=[10, 11],
                unique_id="1",
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    latent_items = proxy_result.result[0]
    assert isinstance(latent_items, list)
    assert [LocalVAEDecode.decode(latent) for latent in latent_items] == [32, 35]

def test_split_hybrid_proxies_allow_local_downstream_work_before_mapped_completion(
    api_intercept_module: Any,
    modal_executor_module: Any,
    session_state_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A local consumer of the static proxy should be able to run while the mapped proxy is still in flight."""
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteModel": _FakeRewriteRemoteModelNode,
                "RemoteSampler": _FakeRewriteRemoteSamplerNode,
                "LatentSource": _FakeRewriteLatentSourceNode,
                "ModalMapInput": _FakeRewriteModalMapInputNode,
                "LocalSink": _FakeRewriteLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
            {"id": 5, "properties": {"is_modal_remote": False}},
            {"id": 6, "properties": {"is_modal_remote": True}},
            {"id": 7, "properties": {"is_modal_remote": True}},
            {"id": 8, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {},
            "_meta": {"title": "Shared Model"},
        },
        "2": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Single Latent"},
        },
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["2", 0]},
            "_meta": {"title": "Unmapped Sampler"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"image": ["3", 0]},
            "_meta": {"title": "Local Sink 1"},
        },
        "5": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Batch Latent Source"},
        },
        "6": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["5", 0]},
            "_meta": {"title": "Map Input"},
        },
        "7": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["6", 0]},
            "_meta": {"title": "Mapped Sampler"},
        },
        "8": {
            "class_type": "LocalSink",
            "inputs": {"image": ["7", 0]},
            "_meta": {"title": "Local Sink 2"},
        },
    }

    rewritten_prompt, _summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )
    static_proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[rewritten_prompt["1"]["class_type"]]
    mapped_proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[rewritten_prompt["1__mapped"]["class_type"]]
    static_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    mapped_payload = rewritten_prompt["1__mapped"]["inputs"]["original_node_data"]

    assert rewritten_prompt["4"]["inputs"]["image"] == ["1", 0]
    assert rewritten_prompt["8"]["inputs"]["image"] == ["1__mapped", 0]
    assert mapped_payload["static_phase"]["execute_node_ids"] == ["1"]
    assert mapped_payload["static_to_mapped_boundaries"] == [
        {
            "proxy_name": "static_input_0",
            "node_id": "1",
            "output_index": 0,
            "io_type": "MODEL",
            "is_list": False,
            "targets": [{"node_id": "7", "input_name": "model"}],
        }
    ]

    observed_order: list[str] = []
    mapped_started = asyncio.Event()
    release_mapped = asyncio.Event()

    class FakeClient:
        """Fake async remote client that blocks the mapped proxy until released."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[Any, ...]:
            """Return deterministic outputs for the split static and mapped proxies."""
            if str(payload.get("component_id")) == "1":
                observed_order.append("static_proxy_finish")
                return (
                    "static-latent",
                    session_state_module.RemoteSessionBridgeRef(
                        bridge_key="RSB_static_model",
                        node_id="1",
                        output_index=0,
                        session_id=str(payload["remote_session"]["session_id"]),
                    ).to_payload(),
                )

            if str(payload.get("component_id")) == "1__mapped":
                observed_order.append("mapped_proxy_start")
                mapped_started.set()
                await release_mapped.wait()
                observed_order.append("mapped_proxy_finish")
                return ("mapped-latent",)

            raise AssertionError(f"Unexpected proxy payload: {payload!r}")

    async def run_scenario() -> tuple[Any, ...]:
        """Run the split static and mapped proxies with a local consumer in between."""
        static_result = await static_proxy_class.execute(
            original_node_data=static_payload,
            unique_id="1",
            remote_input_0="single-latent",
        )
        mapped_task = asyncio.create_task(
            mapped_proxy_class.execute(
                original_node_data=mapped_payload,
                unique_id="1__mapped",
                remote_input_1="batched-latent",
                static_input_0=static_result.result[1],
            )
        )
        await mapped_started.wait()
        observed_order.append(f"local_sink:{static_result.result[0]}")
        assert not mapped_task.done()
        release_mapped.set()
        mapped_result = await mapped_task
        return static_result.result, mapped_result.result

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        static_outputs, mapped_outputs = asyncio.run(run_scenario())
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert static_outputs[0] == "static-latent"
    assert session_state_module.is_remote_session_bridge_ref_payload(static_outputs[1])
    assert static_outputs[2] is True
    assert mapped_outputs == ("mapped-latent", True)
    assert observed_order == [
        "static_proxy_finish",
        "mapped_proxy_start",
        "local_sink:static-latent",
        "mapped_proxy_finish",
    ]

def test_mixed_remote_and_preview_fanout_uses_bridge_and_local_materializer(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A remote continuation should use a bridge while its preview materializes locally."""
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteImage": _FakeRewriteRemoteImageNode,
            "RemoteText": _FakeRewriteRemoteTextNode,
            "PreviewImage": _FakeRewritePreviewImageNode,
            "LocalSink": _FakeRewriteLocalSinkNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": False}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteImage",
            "inputs": {},
            "_meta": {"title": "Generated Image"},
        },
        "2": {
            "class_type": "RemoteText",
            "inputs": {"image": ["1", 0]},
            "_meta": {"title": "Remote LLM"},
        },
        "3": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["1", 0]},
            "_meta": {"title": "Local Preview"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"text": ["2", 0]},
            "_meta": {"title": "Response"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
        extra_data={"prompt_id": "prompt-mixed-fanout"},
    )

    assert summary.remote_component_ids == ["1", "2"]
    assert summary.component_execution_stages == [["1"], ["2"]]
    producer_payload = modal_executor_module._rehydrate_proxy_payload(
        rewritten_prompt["1"]["inputs"]["original_node_data"],
        unique_id="1",
    )
    consumer_payload = modal_executor_module._rehydrate_proxy_payload(
        rewritten_prompt["2"]["inputs"]["original_node_data"],
        unique_id="2",
    )
    assert producer_payload["component_node_ids"] == ["1"]
    assert producer_payload["boundary_outputs"][0]["session_output"] is True
    assert consumer_payload["component_node_ids"] == ["2"]
    assert rewritten_prompt["2"]["inputs"]["remote_input_0"] == ["1", 0]

    materializer_node_ids = [
        node_id
        for node_id, prompt_node in rewritten_prompt.items()
        if prompt_node["class_type"]
        == api_intercept_module.MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID
    ]
    assert len(materializer_node_ids) == 1
    materializer_node_id = materializer_node_ids[0]
    assert len(summary.parallel_local_branch_node_ids) == 1
    dispatch_gate_node_id = summary.parallel_local_branch_node_ids[0]
    assert rewritten_prompt[materializer_node_id]["inputs"]["bridge_ref"] == [
        dispatch_gate_node_id,
        0,
    ]
    assert rewritten_prompt[dispatch_gate_node_id]["inputs"]["value"] == ["1", 0]
    assert rewritten_prompt["3"]["inputs"]["images"] == [materializer_node_id, 0]

def test_consume_remote_payload_stream_suppresses_status_but_keeps_boundary_previews(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped per-item remote calls should suppress status chatter but still forward previews and lane progress."""
    progress_calls: list[dict[str, Any]] = []
    status_calls: list[dict[str, Any]] = []
    preview_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_calls.append(kwargs),
    )
    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_modal_status",
        lambda **kwargs: status_calls.append(kwargs),
    )
    monkeypatch.setattr(
        payload_stream_module,
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
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Static sub-runs should still forward real execute-node progress under suppressed status streams."""
    progress_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        payload_stream_module,
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

def test_consume_remote_payload_stream_marks_progress_node_ancestors_complete(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Streamed progress should tell the UI which upstream remote nodes can be completed."""
    progress_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_calls.append(kwargs),
    )

    payload = {
        "component_id": "3",
        "prompt_id": "prompt-1",
        "component_node_ids": ["1", "2", "3"],
        "extra_data": {"client_id": "client-1"},
        "subgraph_prompt": {
            "1": {"class_type": "Loader", "inputs": {}},
            "2": {"class_type": "Condition", "inputs": {"model": [[["1", 0]]]}},
            "3": {"class_type": "Sampler", "inputs": {"conditioning": ["2", 0]}},
        },
    }
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "node_progress",
                "node_id": "3",
                "value": 1.0,
                "max": 4.0,
            },
            {
                "kind": "result",
                "outputs": serialization_module.serialize_node_outputs(("done",)),
            },
        ]
    )

    response = remote_modal_app_module._consume_remote_payload_stream(payload, stream_events)

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert progress_calls[0]["node_id"] == "3"
    assert progress_calls[0]["completed_ancestor_node_ids"] == ["1", "2"]

def test_consume_remote_payload_stream_clears_static_execute_node_progress_on_suppressed_completion(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Suppressed static sub-runs should emit an explicit clear for lane-less node progress on completion."""
    progress_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        payload_stream_module,
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
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped item streams should not forward executed or preview events for static sibling nodes."""
    executed_calls: list[dict[str, Any]] = []
    preview_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_executed_output",
        lambda **kwargs: executed_calls.append(kwargs),
    )
    monkeypatch.setattr(
        payload_stream_module,
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

def test_consume_remote_payload_stream_forwards_cached_node_markers(
    remote_modal_app_module: Any,
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Streamed node-cache hits should mark the matching UI node as cached without fake progress."""
    progress_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_calls.append(kwargs),
    )

    payload = {
        "component_id": "1::static",
        "prompt_id": "prompt-1",
        "component_node_ids": ["1", "2", "12", "4"],
        "execute_node_ids": ["12", "4"],
        "extra_data": {"client_id": "client-1"},
        "suppress_status_stream": True,
    }
    stream_events = iter(
        [
            {
                "kind": "progress",
                "event_type": "node_cached",
                "node_id": "1",
                "display_node_id": "1",
                "real_node_id": "12",
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
            "value": 0.0,
            "max_value": 1.0,
            "display_node_id": "1",
            "real_node_id": "12",
            "cached_hit": True,
        }
    ]

def test_modal_cloud_skips_downstream_cache_hit_when_boundary_ancestor_is_missing(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """A downstream distributed hit must not bypass a missing required boundary producer."""
    restored_values: dict[str, Any] = {}
    observed_logs: list[tuple[Any, ...]] = []

    class FakeOutputsCache:
        """Minimal outputs cache stub for boundary-aware restore tests."""

        def __init__(self) -> None:
            """Expose the cache-key-set marker read by restore."""
            self.cache_key_set = object()

        def get(self, node_id: str) -> Any:
            """Return any previously restored entry."""
            return restored_values.get(node_id)

        def set(self, node_id: str, cache_entry: Any) -> None:
            """Record one restored cache entry."""
            restored_values[node_id] = cache_entry

    async def fake_key_from_key_set(cache_key_set: Any, node_id: str) -> str:
        """Return stable fake distributed cache keys."""
        del cache_key_set
        return f"NC_{node_id}"

    async def fake_store_get(cache_store: Any, cache_key: str) -> Any:
        """Only the downstream execute target has a distributed cache hit."""
        del cache_store
        if cache_key == "NC_11":
            return {"cache_key": cache_key}
        return None

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_key_from_key_set_async",
        fake_key_from_key_set,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_store_get",
        fake_store_get,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_deserialize_node_output_cache_entry",
        lambda execution, record: record,
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_node_output_cache_ancestor_ids",
        lambda cache_key_set, node_id: {"14"} if node_id == "11" else set(),
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_emit_cloud_info",
        lambda message, *args: observed_logs.append((message, *args)),
    )

    restored_node_ids = asyncio.run(
        modal_cloud_module._restore_persisted_node_output_cache_entries_into_prepared_cache(
            object(),
            FakeOutputsCache(),
            prompt={
                "11": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0]}},
                "14": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            },
            cache_store={},
            required_materialized_node_ids={"14"},
        )
    )

    assert restored_node_ids == []
    assert restored_values == {}
    assert (
        "Node output cache lookup node=%s key_prefix=%s result=skip reason=missing-required-boundary-ancestors ancestors=%s",
        "11",
        "NC_11",
        ["14"],
    ) in observed_logs
