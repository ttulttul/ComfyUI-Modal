"""Remote Modal runtime and local execution fallback."""

import asyncio
import copy
from dataclasses import dataclass, field
import importlib
import importlib.util
from io import BytesIO
import logging
import os
import queue
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from ..serialization import (
    join_mapped_values,
    coerce_serialized_node_outputs,
    deserialize_value,
    deserialize_node_inputs,
    deserialize_node_outputs,
    serialize_mapping,
    serialize_value,
    split_mapped_value,
    serialize_node_outputs,
    serialize_node_inputs,
)
from ..session_state import (
    InMemoryRemoteSessionBridgeStore,
    InMemoryRemoteSessionStore,
    RemoteSessionBridgeRecord,
    RemoteSessionBridgeRef,
    RemoteSessionHandle,
    RemoteSessionStateError,
    RemoteSessionValueRef,
    is_remote_session_bridge_ref_payload,
    is_remote_session_handle_payload,
    is_remote_session_value_ref_payload,
    stable_session_bridge_key,
)
from ..settings import get_settings

logger = logging.getLogger(__name__)
_MODAL_CLOUD_MODULE_NAME = "comfyui_modal_sync_cloud"
_MODAL_AUTO_DEPLOY_LOCK = threading.Lock()
_MODAL_AUTO_DEPLOYED_APPS: set[tuple[str, str | None]] = set()
_MODAL_INTERRUPT_DICTS_LOCK = threading.Lock()
_MODAL_INTERRUPT_DICTS: dict[tuple[str, str | None], Any] = {}
_MAPPED_PROGRESS_NODE_IDS_LOCK = threading.Lock()
_MAPPED_PROGRESS_NODE_IDS: dict[tuple[str, str, str], str] = {}
_PROMPT_WARMUP_STATES_LOCK = threading.Lock()
_PROMPT_WARMUP_STATES: dict[str, "_PromptWarmupState"] = {}
_PROMPT_WARMUP_STATE_ORDER: queue.SimpleQueue[str] | None = None
_PROMPT_WARMUP_STATE_CACHE_LIMIT = 256
_PRIMITIVE_WIDGET_INPUT_TYPES = frozenset({"INT", "FLOAT", "BOOLEAN", "STRING"})
_BOUNDARY_INPUT_SIGNATURES_KEY = "__comfy_modal_boundary_input_signatures__"
_REMOTE_CONTAINER_LOG_STREAMS_LOCK = threading.Lock()
_REMOTE_CONTAINER_LOG_STREAMS: dict[str, "_RemoteContainerLogStreamState"] = {}
_REMOTE_CONTAINER_LOG_STDERR_LOCK = threading.Lock()
_REMOTE_SESSION_STORE = InMemoryRemoteSessionStore()
_REMOTE_SESSION_BRIDGE_STORE = InMemoryRemoteSessionBridgeStore()
_REMOTE_SESSION_BRIDGE_REPLAY_STATE = threading.local()
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK = threading.Lock()
_REMOTE_SESSION_BRIDGE_VALUE_CACHE: dict[str, Any] = {}
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER: list[str] = []
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_LIMIT = 32
_DURABLE_BRIDGE_SERIALIZATION_IO_TYPES = frozenset({"CONDITIONING"})


@dataclass
class _PromptWarmupState:
    """Track proactive warmup state for one local prompt."""

    scheduled_slots: set[int] = field(default_factory=set)
    exact_component_parallelism: dict[str, int] = field(default_factory=dict)


@dataclass
class _RemoteContainerLogLineBuffer:
    """Buffer partial remote log lines so stderr mirroring stays line-oriented."""

    task_id: str
    buffered_text: str = ""


@dataclass
class _RemoteContainerLogStreamState:
    """Track one active local watcher mirroring logs for a Modal container."""

    task_id: str
    stop_event: threading.Event
    thread: threading.Thread
    refcount: int = 0


@dataclass
class _RemoteSessionBridgeResolutionStats:
    """Track how one local fallback payload resolved session-backed inputs."""

    input_ref_count: int = 0
    live_session_hits: int = 0
    bridge_cache_hits: int = 0
    durable_bridge_hits: int = 0
    bridge_record_lookups: int = 0
    bridge_record_lookup_seconds: float = 0.0
    replay_count: int = 0
    replay_seconds: float = 0.0
    direct_restore_seconds: float = 0.0
    session_restore_writes: int = 0


def _remote_modal_call_worker_count() -> int:
    """Return the number of local worker threads reserved for blocking Modal calls."""
    configured_parallelism = get_settings().max_containers or 0
    return max(4, os.cpu_count() or 1, configured_parallelism)


_REMOTE_MODAL_CALL_EXECUTOR = ThreadPoolExecutor(max_workers=_remote_modal_call_worker_count())
_REMOTE_MODAL_WARMUP_EXECUTOR = ThreadPoolExecutor(max_workers=_remote_modal_call_worker_count())

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local fallback tests.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


class ModalRemoteInvocationError(RuntimeError):
    """Raised when the Modal client cannot invoke the remote runtime."""


def _is_remote_container_log_stream_enabled() -> bool:
    """Return whether remote Modal container logs should be mirrored locally."""
    return bool(get_settings().stream_remote_container_logs)


def _coerce_modal_task_id(value: Any) -> str | None:
    """Normalize one streamed Modal task id into a non-empty string."""
    if value is None:
        return None
    task_id = str(value).strip()
    return task_id or None


def _write_remote_container_log_line(
    task_id: str,
    line: str,
    *,
    stream: Any = None,
) -> None:
    """Write one complete remote container log line to the local stderr stream."""
    target_stream = stream if stream is not None else sys.stderr
    with _REMOTE_CONTAINER_LOG_STDERR_LOCK:
        target_stream.write(f"[modal:{task_id}] {line}")
        target_stream.flush()


def _write_remote_container_log_chunk(
    line_buffer: _RemoteContainerLogLineBuffer,
    chunk: str,
    *,
    stream: Any = None,
) -> None:
    """Buffer one remote log chunk and mirror complete lines to local stderr."""
    line_buffer.buffered_text += chunk
    while True:
        newline_index = line_buffer.buffered_text.find("\n")
        if newline_index < 0:
            return
        next_line = line_buffer.buffered_text[: newline_index + 1]
        line_buffer.buffered_text = line_buffer.buffered_text[newline_index + 1 :]
        _write_remote_container_log_line(line_buffer.task_id, next_line, stream=stream)


def _flush_remote_container_log_chunk(
    line_buffer: _RemoteContainerLogLineBuffer,
    *,
    stream: Any = None,
) -> None:
    """Flush any partial remote log text remaining in the local line buffer."""
    if not line_buffer.buffered_text:
        return
    _write_remote_container_log_line(
        line_buffer.task_id,
        f"{line_buffer.buffered_text}\n",
        stream=stream,
    )
    line_buffer.buffered_text = ""


async def _close_modal_client(client: Any) -> None:
    """Close one Modal SDK client if the concrete client exposes a close hook."""
    close_callable = getattr(client, "aclose", None)
    if callable(close_callable):
        close_result = close_callable()
        if asyncio.iscoroutine(close_result):
            await close_result
        return

    close_callable = getattr(client, "close", None)
    if callable(close_callable):
        close_result = close_callable()
        if asyncio.iscoroutine(close_result):
            await close_result


async def _stream_remote_container_logs_via_modal_sdk_async(
    task_id: str,
    stop_event: threading.Event,
) -> None:
    """Follow one Modal container log stream through the Python SDK internals."""
    if modal is None:
        raise ModuleNotFoundError("Modal SDK is unavailable.")

    client_module = importlib.import_module("modal.client")
    exception_module = importlib.import_module("modal.exception")
    api_pb2 = importlib.import_module("modal_proto.api_pb2")
    grpclib_exceptions = importlib.import_module("grpclib.exceptions")
    client = await client_module._Client.from_env()
    line_buffer = _RemoteContainerLogLineBuffer(task_id=task_id)
    last_entry_id = ""

    try:
        while not stop_event.is_set():
            request = api_pb2.AppGetLogsRequest(
                task_id=task_id,
                timeout=5,
                last_entry_id=last_entry_id,
            )
            try:
                async for log_batch in client.stub.AppGetLogs.unary_stream(request):
                    if stop_event.is_set():
                        break
                    if log_batch.entry_id:
                        last_entry_id = str(log_batch.entry_id)
                    if bool(log_batch.app_done):
                        logger.info("Modal SDK log stream finished for task_id=%s.", task_id)
                        return
                    for log_item in log_batch.items:
                        log_data = getattr(log_item, "data", "")
                        if not log_data:
                            continue
                        _write_remote_container_log_chunk(line_buffer, str(log_data))
            except (
                exception_module.ServiceError,
                exception_module.InternalError,
                grpclib_exceptions.StreamTerminatedError,
                socket.gaierror,
            ) as exc:
                if stop_event.is_set():
                    break
                logger.warning(
                    "Retrying Modal SDK log stream for task_id=%s after transient failure: %s",
                    task_id,
                    exc,
                )
                continue
            except AttributeError as exc:
                if stop_event.is_set():
                    break
                if "_write_appdata" in str(exc):
                    logger.warning(
                        "Retrying Modal SDK log stream for task_id=%s after connection loss: %s",
                        task_id,
                        exc,
                    )
                    continue
                raise
    finally:
        _flush_remote_container_log_chunk(line_buffer)
        await _close_modal_client(client)


def _stream_remote_container_logs_via_modal_sdk(
    task_id: str,
    stop_event: threading.Event,
) -> bool:
    """Try to mirror one Modal container log stream through the installed SDK."""
    if modal is None:
        return False
    asyncio.run(_stream_remote_container_logs_via_modal_sdk_async(task_id, stop_event))
    return True


def _stream_remote_container_logs_via_modal_cli(
    task_id: str,
    stop_event: threading.Event,
) -> bool:
    """Try to mirror one Modal container log stream through the Modal CLI."""
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        return False

    command = [modal_cli, "container", "logs", task_id, "-f"]
    line_buffer = _RemoteContainerLogLineBuffer(task_id=task_id)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    line_buffer = _RemoteContainerLogLineBuffer(task_id=task_id)
    try:
        stdout_stream = process.stdout
        if stdout_stream is None:
            raise RuntimeError("Modal CLI log process did not expose a stdout stream.")
        while not stop_event.is_set():
            ready_streams, _, _ = select.select([stdout_stream], [], [], 0.25)
            if ready_streams:
                next_chunk = stdout_stream.read(4096)
                if next_chunk:
                    _write_remote_container_log_chunk(
                        line_buffer,
                        next_chunk.decode("utf-8", errors="replace"),
                    )
                    continue
            if process.poll() is not None:
                trailing_chunk = stdout_stream.read()
                if trailing_chunk:
                    _write_remote_container_log_chunk(
                        line_buffer,
                        trailing_chunk.decode("utf-8", errors="replace"),
                    )
                break
        if stop_event.is_set() and process.poll() is None:
            process.terminate()
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        _flush_remote_container_log_chunk(line_buffer)

    if process.returncode not in {0, None} and not stop_event.is_set():
        raise RuntimeError(
            f"Modal CLI exited with status {process.returncode} while streaming logs for task_id={task_id}."
        )
    return True


def _run_remote_container_log_stream(task_id: str, stop_event: threading.Event) -> None:
    """Run the best available remote log streaming backend for one Modal container."""
    logger.info("Starting remote Modal container log stream for task_id=%s.", task_id)
    try:
        if _stream_remote_container_logs_via_modal_cli(task_id, stop_event):
            logger.info("Stopped remote Modal container log stream for task_id=%s.", task_id)
            return
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning(
            "Falling back from Modal CLI log streaming for task_id=%s after failure: %s",
            task_id,
            exc,
        )

    try:
        if _stream_remote_container_logs_via_modal_sdk(task_id, stop_event):
            logger.info("Stopped remote Modal container log stream for task_id=%s.", task_id)
            return
    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Modal SDK log streaming failed for task_id=%s: %s",
            task_id,
            exc,
        )

    logger.warning(
        "Unable to mirror remote Modal container logs for task_id=%s because neither the Modal SDK nor CLI is available locally.",
        task_id,
    )


def _retain_remote_container_log_stream(task_id: str) -> str:
    """Increment one shared Modal container log watcher and start it if needed."""
    with _REMOTE_CONTAINER_LOG_STREAMS_LOCK:
        stream_state = _REMOTE_CONTAINER_LOG_STREAMS.get(task_id)
        if stream_state is None:
            stop_event = threading.Event()
            stream_state = _RemoteContainerLogStreamState(
                task_id=task_id,
                stop_event=stop_event,
                thread=threading.Thread(
                    target=_run_remote_container_log_stream,
                    args=(task_id, stop_event),
                    name=f"modal-log-stream-{task_id}",
                    daemon=True,
                ),
            )
            _REMOTE_CONTAINER_LOG_STREAMS[task_id] = stream_state
            logger.info("Creating remote Modal container log stream for task_id=%s.", task_id)
            stream_state.thread.start()
        elif not stream_state.thread.is_alive() and stream_state.refcount == 0:
            stop_event = threading.Event()
            stream_state = _RemoteContainerLogStreamState(
                task_id=task_id,
                stop_event=stop_event,
                thread=threading.Thread(
                    target=_run_remote_container_log_stream,
                    args=(task_id, stop_event),
                    name=f"modal-log-stream-{task_id}",
                    daemon=True,
                ),
            )
            _REMOTE_CONTAINER_LOG_STREAMS[task_id] = stream_state
            logger.info("Restarting remote Modal container log stream for task_id=%s.", task_id)
            stream_state.thread.start()
        else:
            logger.info("Reusing remote Modal container log stream for task_id=%s.", task_id)
        stream_state.refcount += 1
        logger.info(
            "Remote Modal container log stream retain task_id=%s refcount=%d.",
            task_id,
            stream_state.refcount,
        )

    return task_id


def _release_remote_container_log_stream(task_id: str) -> None:
    """Release one retained Modal container log watcher when a payload finishes."""
    stream_state: _RemoteContainerLogStreamState | None = None
    should_stop = False
    with _REMOTE_CONTAINER_LOG_STREAMS_LOCK:
        stream_state = _REMOTE_CONTAINER_LOG_STREAMS.get(task_id)
        if stream_state is None:
            return
        stream_state.refcount = max(0, stream_state.refcount - 1)
        logger.info(
            "Remote Modal container log stream release task_id=%s refcount=%d.",
            task_id,
            stream_state.refcount,
        )
        if stream_state.refcount == 0:
            should_stop = True
            _REMOTE_CONTAINER_LOG_STREAMS.pop(task_id, None)

    if should_stop and stream_state is not None:
        logger.info("Stopping remote Modal container log stream for task_id=%s.", task_id)
        stream_state.stop_event.set()
        stream_state.thread.join(timeout=0.2)


def _close_remote_payload_stream(stream_events: Iterator[dict[str, Any]]) -> None:
    """Best-effort close one streamed Modal iterator after the terminal result arrives."""
    close_callable = getattr(stream_events, "close", None)
    if not callable(close_callable):
        return
    try:
        close_result = close_callable()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Ignoring remote payload stream close failure: %s", exc)
        return
    if not asyncio.iscoroutine(close_result):
        return
    try:
        asyncio.run(close_result)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Ignoring async remote payload stream close failure: %s", exc)


def _payload_remote_session_handle(payload: dict[str, Any]) -> RemoteSessionHandle | None:
    """Return the decoded prompt-scoped remote session handle for one payload."""
    remote_session = payload.get("remote_session")
    if not is_remote_session_handle_payload(remote_session):
        return None
    return RemoteSessionHandle.from_payload(remote_session)


def _sanitize_payload_for_session_bridge_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip run-scoped fields from one producer payload before persisting replay metadata."""
    sanitized_payload = copy.deepcopy(payload)
    sanitized_payload.pop("prompt_id", None)
    sanitized_payload.pop("remote_session", None)
    sanitized_payload.pop("clear_remote_session", None)
    sanitized_payload["extra_data"] = {}
    return sanitized_payload


def _build_remote_session_bridge_record(
    *,
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_id: str,
    output_index: int,
    io_type: str,
    output_value: Any,
) -> RemoteSessionBridgeRecord:
    """Build one durable bridge record for a session-backed boundary output."""
    producer_payload = _sanitize_payload_for_session_bridge_record(payload)
    producer_inputs = serialize_mapping(hydrated_inputs)
    serialized_output = _serialize_durable_bridge_output(output_value, io_type)
    return RemoteSessionBridgeRecord(
        bridge_key=stable_session_bridge_key(
            producer_payload=producer_payload,
            producer_inputs=producer_inputs,
            node_id=node_id,
            output_index=output_index,
        ),
        node_id=node_id,
        output_index=output_index,
        producer_payload=producer_payload,
        producer_inputs=producer_inputs,
        serialized_output=serialized_output,
        serialized_output_io_type=(str(io_type) if serialized_output is not None else None),
    )


def _record_remote_session_resolution_event(
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None",
    event_name: str,
    event_payload: Mapping[str, Any],
) -> None:
    """Accumulate one local fallback remote-session resolution event."""
    del event_payload
    if resolution_stats is None:
        return
    if event_name in {"session-value-hit", "bridge-target-hit", "bridge-source-hit"}:
        resolution_stats.live_session_hits += 1


def _clone_cached_bridge_value(value: Any) -> Any:
    """Clone one cached bridge value when the runtime object exposes a safe clone path."""
    clone_method = getattr(value, "clone", None)
    if callable(clone_method):
        return clone_method()
    return value


def _store_remote_session_bridge_value(
    bridge_key: str,
    value: Any,
) -> None:
    """Retain one live bridge value in-process so later mapped phases can skip replay."""
    with _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK:
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE[bridge_key] = _clone_cached_bridge_value(value)
        if bridge_key in _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER:
            _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.remove(bridge_key)
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.append(bridge_key)
        while len(_REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER) > _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LIMIT:
            evicted_key = _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.pop(0)
            _REMOTE_SESSION_BRIDGE_VALUE_CACHE.pop(evicted_key, None)


def _get_remote_session_bridge_value(bridge_key: str) -> Any | None:
    """Return one retained bridge value when the current process still has it."""
    with _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK:
        cached_value = _REMOTE_SESSION_BRIDGE_VALUE_CACHE.get(bridge_key)
        if cached_value is None:
            return None
        if bridge_key in _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER:
            _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.remove(bridge_key)
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.append(bridge_key)
        return _clone_cached_bridge_value(cached_value)


def _serialize_durable_bridge_output(output_value: Any, io_type: str) -> Any | None:
    """Serialize one bridge output when its io_type supports durable direct restore."""
    normalized_io_type = str(io_type or "")
    if normalized_io_type not in _DURABLE_BRIDGE_SERIALIZATION_IO_TYPES:
        return None
    try:
        return serialize_value(output_value)
    except TypeError:
        logger.warning(
            "Skipping durable bridge serialization for io_type=%s value_type=%s.",
            normalized_io_type,
            type(output_value).__name__,
        )
        return None


def _restore_serialized_remote_session_bridge_value(
    record: RemoteSessionBridgeRecord,
    *,
    target_session_handle: RemoteSessionHandle,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> Any | None:
    """Restore one bridge value directly from a durable serialized payload."""
    if record.serialized_output is None:
        return None

    restore_started_at = time.perf_counter()
    restored_value = deserialize_value(record.serialized_output)
    _REMOTE_SESSION_STORE.put_output(
        target_session_handle,
        node_id=record.node_id,
        output_index=record.output_index,
        value=restored_value,
    )
    _store_remote_session_bridge_value(record.bridge_key, restored_value)
    if resolution_stats is not None:
        resolution_stats.durable_bridge_hits += 1
        resolution_stats.session_restore_writes += 1
        resolution_stats.direct_restore_seconds += time.perf_counter() - restore_started_at
    logger.info(
        "Resolved remote session bridge bridge_key=%s from durable serialized %s payload into session_id=%s.",
        record.bridge_key,
        record.serialized_output_io_type or "bridge",
        target_session_handle.session_id,
    )
    return restored_value


def _remote_session_bridge_replay_stack() -> set[str]:
    """Return the thread-local guard set for bridge replay recursion detection."""
    replay_stack = getattr(_REMOTE_SESSION_BRIDGE_REPLAY_STATE, "bridge_keys", None)
    if replay_stack is None:
        replay_stack = set()
        _REMOTE_SESSION_BRIDGE_REPLAY_STATE.bridge_keys = replay_stack
    return replay_stack


def _rehydrate_remote_session_bridge_value(
    ref: RemoteSessionBridgeRef,
    *,
    target_session_handle: RemoteSessionHandle | None,
    node_mapping: dict[str, type[Any]] | None = None,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> Any:
    """Replay one producer phase into the current session when the live value is gone."""
    if target_session_handle is None:
        raise RemoteSessionStateError(
            "Remote session bridge replay requires a target remote_session handle."
        )

    cached_value = _get_remote_session_bridge_value(ref.bridge_key)
    if cached_value is not None:
        restore_started_at = time.perf_counter()
        _REMOTE_SESSION_STORE.put_output(
            target_session_handle,
            node_id=ref.node_id,
            output_index=ref.output_index,
            value=cached_value,
        )
        if resolution_stats is not None:
            resolution_stats.bridge_cache_hits += 1
            resolution_stats.session_restore_writes += 1
            resolution_stats.direct_restore_seconds += time.perf_counter() - restore_started_at
        logger.info(
            "Resolved remote session bridge bridge_key=%s directly from warm cache into session_id=%s.",
            ref.bridge_key,
            target_session_handle.session_id,
        )
        return cached_value

    replay_stack = _remote_session_bridge_replay_stack()
    if ref.bridge_key in replay_stack:
        raise RemoteSessionStateError(
            f"Detected recursive remote session bridge replay for {ref.bridge_key!r}."
        )

    record_lookup_started_at = time.perf_counter()
    record = _REMOTE_SESSION_BRIDGE_STORE.get_record(ref.bridge_key)
    if resolution_stats is not None:
        resolution_stats.bridge_record_lookups += 1
        resolution_stats.bridge_record_lookup_seconds += (
            time.perf_counter() - record_lookup_started_at
        )
    restored_value = _restore_serialized_remote_session_bridge_value(
        record,
        target_session_handle=target_session_handle,
        resolution_stats=resolution_stats,
    )
    if restored_value is not None:
        return restored_value
    replay_payload = copy.deepcopy(record.producer_payload)
    replay_payload["remote_session"] = target_session_handle.to_payload()
    replay_payload.pop("clear_remote_session", None)
    if target_session_handle.prompt_id is not None:
        replay_payload["prompt_id"] = target_session_handle.prompt_id
    replay_inputs = deserialize_node_inputs(record.producer_inputs)

    logger.info(
        "Replaying remote session bridge bridge_key=%s into session_id=%s via component=%s.",
        ref.bridge_key,
        target_session_handle.session_id,
        replay_payload.get("component_id"),
    )
    replay_stack.add(ref.bridge_key)
    replay_started_at = time.perf_counter()
    try:
        _execute_subgraph_prompt(replay_payload, replay_inputs, node_mapping)
    finally:
        replay_stack.remove(ref.bridge_key)
    if resolution_stats is not None:
        resolution_stats.replay_count += 1
        resolution_stats.replay_seconds += time.perf_counter() - replay_started_at

    return _REMOTE_SESSION_STORE.get_output(
        RemoteSessionValueRef(
            session_id=target_session_handle.session_id,
            node_id=ref.node_id,
            output_index=ref.output_index,
        )
    )


def _resolve_remote_session_inputs(
    hydrated_inputs: dict[str, Any],
    *,
    component_id: str | None = None,
    target_session_handle: RemoteSessionHandle | None = None,
    node_mapping: dict[str, type[Any]] | None = None,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> dict[str, Any]:
    """Resolve any remote-session value refs embedded in boundary inputs."""
    ref_input_names = [
        input_name
        for input_name, input_value in hydrated_inputs.items()
        if is_remote_session_value_ref_payload(input_value)
        or is_remote_session_bridge_ref_payload(input_value)
    ]
    if ref_input_names:
        logger.info(
            "Resolving %d remote session input refs for component=%s inputs=%s.",
            len(ref_input_names),
            component_id or "<unknown>",
            sorted(ref_input_names),
        )
    if resolution_stats is not None:
        resolution_stats.input_ref_count += len(ref_input_names)
    return {
        input_name: _REMOTE_SESSION_STORE.resolve_value_with_bridges(
            input_value,
            target_session_handle=target_session_handle,
            resolution_callback=(
                lambda event_name, event_payload: _record_remote_session_resolution_event(
                    resolution_stats,
                    event_name,
                    event_payload,
                )
            )
            if resolution_stats is not None
            else None,
            bridge_resolver=lambda ref: _rehydrate_remote_session_bridge_value(
                ref,
                target_session_handle=target_session_handle,
                node_mapping=node_mapping,
                resolution_stats=resolution_stats,
            ),
        )
        for input_name, input_value in hydrated_inputs.items()
    }


def _log_remote_session_resolution_summary(
    *,
    component_id: str,
    resolution_stats: "_RemoteSessionBridgeResolutionStats",
) -> None:
    """Emit one high-signal summary for local fallback bridge resolution work."""
    if resolution_stats.input_ref_count <= 0:
        return
    logger.info(
        "Remote session resolution summary component=%s refs=%d live_hits=%d warm_bridge_hits=%d durable_bridge_hits=%d bridge_record_lookups=%d bridge_record_lookup_seconds=%.3f replay_count=%d replay_seconds=%.3f direct_restore_seconds=%.3f session_restore_writes=%d.",
        component_id,
        resolution_stats.input_ref_count,
        resolution_stats.live_session_hits,
        resolution_stats.bridge_cache_hits,
        resolution_stats.durable_bridge_hits,
        resolution_stats.bridge_record_lookups,
        resolution_stats.bridge_record_lookup_seconds,
        resolution_stats.replay_count,
        resolution_stats.replay_seconds,
        resolution_stats.direct_restore_seconds,
        resolution_stats.session_restore_writes,
    )


def _remote_session_affinity_key(payload: dict[str, Any]) -> str | None:
    """Return the affinity key that should keep split proxy calls on one remote worker."""
    session_handle = _payload_remote_session_handle(payload)
    if session_handle is None:
        return None
    return session_handle.session_id


def _mapped_lane_affinity_key(payload: dict[str, Any], lane_index: int) -> str | None:
    """Return the stable per-lane affinity key used to pin one mapped worker lane."""
    session_affinity_key = _remote_session_affinity_key(payload)
    if session_affinity_key is None:
        return None
    return f"{session_affinity_key}::lane:{int(lane_index)}"


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug("Suppressed remote prompt event %s for client %s.", event, client_id)


def _extract_custom_nodes_bundle(bundle_path: str | None) -> None:
    """Extract a mirrored custom_nodes archive into a temporary import path."""
    if not bundle_path:
        return

    settings = get_settings()
    if settings.execution_mode == "local":
        logger.debug("Skipping custom_nodes bundle extraction in local execution mode.")
        return

    local_bundle = settings.local_storage_root / bundle_path.lstrip("/")
    if not local_bundle.exists():
        logger.warning("Custom nodes bundle %s was not found in local storage.", local_bundle)
        return

    extraction_root = Path(tempfile.gettempdir()) / "comfy-modal-sync-custom-nodes"
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_bundle, "r") as archive:
        archive.extractall(extraction_root)

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _load_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    import execution

    return execution


@contextmanager
def _temporary_node_mapping(node_mapping: dict[str, type[Any]] | None) -> Iterator[None]:
    """Temporarily overlay node mappings for tests or custom runtimes."""
    if node_mapping is None:
        yield
        return

    nodes_module = _load_nodes_module()
    original_mappings = dict(nodes_module.NODE_CLASS_MAPPINGS)
    original_display_mappings = dict(getattr(nodes_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    try:
        nodes_module.NODE_CLASS_MAPPINGS.update(node_mapping)
        for class_type in node_mapping:
            nodes_module.NODE_DISPLAY_NAME_MAPPINGS.setdefault(class_type, class_type)
        yield
    finally:
        nodes_module.NODE_CLASS_MAPPINGS.clear()
        nodes_module.NODE_CLASS_MAPPINGS.update(original_mappings)
        if hasattr(nodes_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            nodes_module.NODE_DISPLAY_NAME_MAPPINGS.clear()
            nodes_module.NODE_DISPLAY_NAME_MAPPINGS.update(original_display_mappings)


def _invoke_original_node(
    node_class: type[Any],
    node_data: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Execute an original V1 or V3 node class and normalize its outputs."""
    class_type = node_data["class_type"]
    logger.info("Executing remote node %s", class_type)

    if hasattr(node_class, "GET_SCHEMA"):
        node_output = node_class.execute(**kwargs)
        if hasattr(node_output, "result"):
            result = node_output.result
            return tuple(result) if result is not None else tuple()
        return tuple(node_output)

    instance = node_class()
    function_name = getattr(node_class, "FUNCTION", "execute")
    function = getattr(instance, function_name)
    result = function(**kwargs)
    if result is None:
        return tuple()
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def execute_node_locally(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> bytes:
    """Execute a single target node in-process and return serialized outputs."""
    _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    kwargs = deserialize_node_inputs(kwargs_payload)
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        outputs = _invoke_original_node(node_mapping[class_type], node_data, kwargs)
        return serialize_node_outputs(outputs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        outputs = _invoke_original_node(resolved_node_mapping[class_type], node_data, kwargs)
    return serialize_node_outputs(outputs)


def _apply_boundary_inputs(
    prompt: dict[str, Any],
    boundary_input_specs: list[dict[str, Any]],
    hydrated_inputs: dict[str, Any],
) -> None:
    """Inject hydrated local boundary inputs into a remote subgraph prompt."""
    logger.info("Applying %d hydrated boundary inputs to remote subgraph prompt.", len(boundary_input_specs))
    for boundary_input in boundary_input_specs:
        proxy_input_name = str(boundary_input["proxy_input_name"])
        if proxy_input_name not in hydrated_inputs:
            raise KeyError(f"Missing hydrated boundary input {proxy_input_name!r}.")
        value = hydrated_inputs[proxy_input_name]
        io_type = (
            str(boundary_input["io_type"])
            if boundary_input.get("io_type") is not None
            else None
        )
        logger.info(
            "Applying boundary input %s to %d targets.",
            proxy_input_name,
            len(boundary_input.get("targets", [])),
        )
        for target in boundary_input.get("targets", []):
            node_id = str(target["node_id"])
            input_name = str(target["input_name"])
            prompt_node = prompt[node_id]
            prompt_node["inputs"][input_name] = _normalize_prompt_input_value(
                value,
                io_type=io_type,
            )
            source_signature = boundary_input.get("source_signature")
            if isinstance(source_signature, str) and source_signature:
                boundary_signatures = prompt_node.setdefault(_BOUNDARY_INPUT_SIGNATURES_KEY, {})
                if isinstance(boundary_signatures, dict):
                    boundary_signatures[input_name] = source_signature


def _collapse_cache_slot(slot_values: Any, is_list: bool) -> Any:
    """Convert a PromptExecutor cache slot back into a node-style output value."""
    if is_list:
        return slot_values
    if not isinstance(slot_values, list):
        return slot_values
    if len(slot_values) == 1:
        return slot_values[0]
    return slot_values


def _extract_prompt_executor_error(executor: Any) -> str:
    """Extract a useful failure message from a PromptExecutor run."""
    for event, data in reversed(executor.status_messages):
        if event == "execution_error":
            return _format_prompt_executor_error_payload(data)
        if event == "execution_interrupted":
            return "Remote subgraph execution was interrupted."
    return "Remote subgraph execution failed."


def _format_prompt_executor_error_payload(data: Any) -> str:
    """Return a richer human-readable PromptExecutor failure message when available."""
    if not isinstance(data, dict):
        return "Remote subgraph execution failed."

    message = str(data.get("exception_message") or "Remote subgraph execution failed.")
    node_id = data.get("node_id")
    node_type = data.get("node_type")
    current_inputs = data.get("current_inputs")
    if node_id is None and node_type is None and not current_inputs:
        return message

    details: list[str] = [message]
    if node_id is not None or node_type is not None:
        details.append(f"node_id={node_id!r} node_type={node_type!r}")
    if current_inputs:
        details.append(f"current_inputs={current_inputs!r}")
    return " | ".join(details)


def _node_input_type_map(node_class: type[Any]) -> dict[str, str]:
    """Return one node class's declared V1 input types keyed by input name."""
    input_types_callable = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types_callable):
        return {}

    raw_input_types = input_types_callable()
    if not isinstance(raw_input_types, dict):
        return {}

    input_type_map: dict[str, str] = {}
    for section_name in ("required", "optional", "hidden"):
        section = raw_input_types.get(section_name)
        if not isinstance(section, dict):
            continue
        for input_name, input_config in section.items():
            if not isinstance(input_config, tuple) or not input_config:
                continue
            declared_type = input_config[0]
            if isinstance(declared_type, str):
                input_type_map[str(input_name)] = declared_type
    return input_type_map


def _coerce_primitive_prompt_input_value(
    *,
    node_id: str,
    class_type: str,
    input_name: str,
    declared_type: str,
    input_value: Any,
) -> Any:
    """Coerce one primitive prompt literal using ComfyUI's `validate_inputs` semantics."""
    literal_value = (
        input_value.get("__value__")
        if isinstance(input_value, dict) and "__value__" in input_value
        else input_value
    )
    if isinstance(literal_value, list):
        return input_value

    try:
        if declared_type == "INT":
            return int(literal_value)
        if declared_type == "FLOAT":
            return float(literal_value)
        if declared_type == "STRING":
            return str(literal_value)
        if declared_type == "BOOLEAN":
            return bool(literal_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RemoteSubgraphExecutionError(
            "Remote subgraph input could not be coerced to the declared primitive socket type."
            f" node_id={node_id!r} node_type={class_type!r}"
            f" input_name={input_name!r} declared_type={declared_type!r}"
            f" received_value={literal_value!r}"
        ) from exc

    return input_value


def _coerce_prompt_primitive_input_values(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> None:
    """Mutate prompt literals in-place to match ComfyUI's primitive widget coercion."""
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        input_type_map = _node_input_type_map(node_class)
        if not input_type_map:
            continue
        inputs = prompt_node.get("inputs") or {}
        for input_name, input_value in list(inputs.items()):
            declared_type = input_type_map.get(str(input_name))
            if declared_type not in _PRIMITIVE_WIDGET_INPUT_TYPES:
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
            ):
                continue
            coerced_value = _coerce_primitive_prompt_input_value(
                node_id=str(node_id),
                class_type=class_type,
                input_name=str(input_name),
                declared_type=declared_type,
                input_value=input_value,
            )
            if coerced_value is not input_value:
                logger.debug(
                    "Coerced remote primitive input %s.%s from %r to %r for type %s.",
                    node_id,
                    input_name,
                    input_value,
                    coerced_value,
                    declared_type,
                )
                inputs[input_name] = coerced_value


def _validate_prompt_input_shapes(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
    boundary_input_specs: list[dict[str, Any]] | None = None,
) -> None:
    """Reject prompt inputs that still look invalid for primitive widget sockets."""
    boundary_targets = {
        (str(target.get("node_id")), str(target.get("input_name")))
        for boundary_input in (boundary_input_specs or [])
        for target in boundary_input.get("targets", [])
        if target.get("node_id") is not None and target.get("input_name") is not None
    }
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        input_type_map = _node_input_type_map(node_class)
        if not input_type_map:
            continue
        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            declared_type = input_type_map.get(str(input_name))
            if declared_type not in _PRIMITIVE_WIDGET_INPUT_TYPES:
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
            ):
                continue
            if (str(node_id), str(input_name)) in boundary_targets:
                continue
            literal_value = (
                input_value.get("__value__")
                if isinstance(input_value, dict) and "__value__" in input_value
                else input_value
            )
            if isinstance(literal_value, list):
                raise RemoteSubgraphExecutionError(
                    "Remote subgraph input has an invalid list value for a primitive socket."
                    f" node_id={node_id!r} node_type={class_type!r}"
                    f" input_name={input_name!r} declared_type={declared_type!r}"
                    f" received_value={literal_value!r}"
                )


def _resolve_required_subgraph_nodes(
    prompt: dict[str, Any],
    execute_node_ids: list[str],
) -> list[str]:
    """Return the dependency closure needed to execute the requested subgraph nodes."""
    required: set[str] = set()
    pending = list(execute_node_ids)
    logger.info("Resolving dependency closure for remote execute targets: %s", execute_node_ids)
    while pending:
        node_id = str(pending.pop())
        if node_id not in prompt:
            logger.warning(
                "Skipping missing remote execute target %s while resolving dependency closure.",
                node_id,
            )
            continue
        if node_id in required:
            continue
        required.add(node_id)
        for input_value in (prompt[node_id].get("inputs") or {}).values():
            if _is_link(input_value):
                pending.append(str(input_value[0]))
    resolved = sorted(required)
    logger.info("Resolved remote dependency closure: %s", resolved)
    return resolved


def _trim_subgraph_payload_to_required_nodes(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim a subgraph payload down to the dependency closure of its execute targets."""
    trimmed_payload = copy.deepcopy(payload)
    prompt = trimmed_payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return trimmed_payload

    prompt_node_ids = {str(node_id) for node_id in prompt}
    requested_execute_node_ids = [
        str(node_id) for node_id in trimmed_payload.get("execute_node_ids", [])
    ]
    available_execute_node_ids = [
        node_id for node_id in requested_execute_node_ids if node_id in prompt_node_ids
    ]
    dropped_execute_node_ids = [
        node_id for node_id in requested_execute_node_ids if node_id not in prompt_node_ids
    ]
    if dropped_execute_node_ids:
        logger.warning(
            "Dropping remote execute targets absent from subgraph prompt for component=%s: %s",
            payload.get("component_id"),
            dropped_execute_node_ids,
        )

    required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=available_execute_node_ids,
        )
    )
    if not required_node_ids:
        return trimmed_payload

    original_node_ids = list(prompt.keys())
    trimmed_payload["subgraph_prompt"] = {
        str(node_id): prompt[node_id]
        for node_id in original_node_ids
        if str(node_id) in required_node_ids
    }
    trimmed_payload["boundary_inputs"] = [
        {
            **copy.deepcopy(boundary_input),
            "targets": [
                copy.deepcopy(target)
                for target in boundary_input.get("targets", [])
                if str(target.get("node_id")) in required_node_ids
            ],
        }
        for boundary_input in trimmed_payload.get("boundary_inputs", [])
        if any(str(target.get("node_id")) in required_node_ids for target in boundary_input.get("targets", []))
    ]
    trimmed_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in trimmed_payload.get("boundary_outputs", [])
        if str(boundary_output.get("node_id")) in required_node_ids
    ]
    trimmed_payload["component_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("component_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["mapped_execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("mapped_execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["static_execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("static_execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    logger.info(
        "Trimmed remote subgraph payload %s from %d prompt nodes to %d required nodes.",
        payload.get("component_id"),
        len(original_node_ids),
        len(trimmed_payload["subgraph_prompt"]),
    )
    return trimmed_payload


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _normalize_link_output_index(value: Any) -> Any:
    """Unwrap a singleton list around a prompt-link output index when present."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _unwrap_wrapped_prompt_link(value: Any) -> Any:
    """Collapse nested singleton wrappers around one serialized prompt link when present."""
    candidate = value
    while isinstance(candidate, list) and len(candidate) == 1:
        candidate = candidate[0]
    if _is_link(candidate):
        return [candidate[0], _normalize_link_output_index(candidate[1])]
    return value


def _normalize_prompt_input_value(value: Any, io_type: str | None = None) -> Any:
    """Unwrap transport-added singleton wrappers only for scalar-like prompt input values."""
    wrapped_link = _unwrap_wrapped_prompt_link(value)
    if wrapped_link is not value:
        return wrapped_link
    while (
        isinstance(value, list)
        and len(value) == 1
        and (
            io_type in _PRIMITIVE_WIDGET_INPUT_TYPES
            or value[0] is None
            or isinstance(value[0], bool | int | float | str)
        )
    ):
        value = value[0]
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return [value[0], _normalize_link_output_index(value[1])]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return value


def _normalize_subgraph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a subgraph payload with canonical prompt-link and output-index shapes."""
    normalized_payload = copy.deepcopy(payload)

    for node_info in normalized_payload.get("subgraph_prompt", {}).values():
        inputs = node_info.get("inputs") or {}
        for input_name, input_value in list(inputs.items()):
            inputs[input_name] = _normalize_prompt_input_value(input_value)

    for boundary_output in normalized_payload.get("boundary_outputs", []):
        if "node_id" in boundary_output and isinstance(boundary_output["node_id"], list):
            boundary_output["node_id"] = _normalize_prompt_input_value(boundary_output["node_id"])
        if "output_index" in boundary_output:
            boundary_output["output_index"] = _normalize_link_output_index(
                boundary_output["output_index"]
            )

    normalized_payload["execute_node_ids"] = [
        _normalize_prompt_input_value(node_id)
        for node_id in normalized_payload.get("execute_node_ids", [])
    ]

    return normalized_payload


def _execute_subgraph_with_mapping(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> tuple[Any, ...]:
    """Execute a rewritten remote component using an explicit node mapping."""
    normalized_payload = _trim_subgraph_payload_to_required_nodes(
        _normalize_subgraph_payload(payload)
    )
    session_handle = _payload_remote_session_handle(normalized_payload)
    resolution_stats = _RemoteSessionBridgeResolutionStats()
    resolved_inputs = _resolve_remote_session_inputs(
        dict(hydrated_inputs),
        component_id=str(payload.get("component_id") or ""),
        target_session_handle=session_handle,
        node_mapping=node_mapping,
        resolution_stats=resolution_stats,
    )
    _log_remote_session_resolution_summary(
        component_id=str(payload.get("component_id") or "modal-subgraph"),
        resolution_stats=resolution_stats,
    )
    if session_handle is not None:
        logger.info(
            "Executing mapped remote subgraph %s with remote_session session_id=%s prompt_id=%s owner_component_id=%s.",
            payload.get("component_id"),
            session_handle.session_id,
            session_handle.prompt_id,
            session_handle.owner_component_id,
        )
    prompt = copy.deepcopy(normalized_payload["subgraph_prompt"])
    logger.info(
        "Executing remote subgraph %s via test mapping with %d prompt nodes.",
        payload.get("component_id"),
        len(prompt),
    )
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
        hydrated_inputs=resolved_inputs,
    )
    _coerce_prompt_primitive_input_values(prompt, node_mapping)
    _validate_prompt_input_shapes(
        prompt,
        node_mapping,
        list(normalized_payload.get("boundary_inputs", [])),
    )
    required_node_ids = _resolve_required_subgraph_nodes(
        prompt=prompt,
        execute_node_ids=list(normalized_payload.get("execute_node_ids", [])),
    )
    executed_outputs: dict[str, tuple[Any, ...]] = {}
    pending = set(required_node_ids)

    while pending:
        progressed = False
        logger.info("Mapped remote subgraph pending nodes: %s", sorted(pending))
        for node_id in list(sorted(pending)):
            prompt_node = prompt[node_id]
            kwargs: dict[str, Any] = {}
            unresolved_dependency = False
            for input_name, input_value in (prompt_node.get("inputs") or {}).items():
                if _is_link(input_value):
                    upstream_node_id = str(input_value[0])
                    if upstream_node_id not in executed_outputs:
                        unresolved_dependency = True
                        break
                    kwargs[str(input_name)] = executed_outputs[upstream_node_id][int(input_value[1])]
                else:
                    kwargs[str(input_name)] = input_value
            if unresolved_dependency:
                continue

            class_type = str(prompt_node["class_type"])
            if class_type not in node_mapping:
                raise KeyError(f"Remote node class {class_type!r} is not registered.")
            logger.info(
                "Executing mapped remote node %s (%s) with %d inputs.",
                node_id,
                class_type,
                len(kwargs),
            )
            executed_outputs[node_id] = _invoke_original_node(
                node_mapping[class_type],
                prompt_node,
                kwargs,
            )
            pending.remove(node_id)
            progressed = True
        if not progressed:
            raise RemoteSubgraphExecutionError(
                "Unable to resolve execution order for remote subgraph payload."
            )

    outputs: list[Any] = []
    for boundary_output in normalized_payload.get("boundary_outputs", []):
        node_id = str(boundary_output["node_id"])
        output_index = int(boundary_output["output_index"])
        node_outputs = executed_outputs.get(node_id)
        if node_outputs is None:
            raise RemoteSubgraphExecutionError(
                f"Remote subgraph did not execute boundary output node {node_id}."
            )
        output_value = node_outputs[output_index]
        if bool(boundary_output.get("session_output")):
            if session_handle is None:
                raise RemoteSessionStateError(
                    "Session-backed boundary outputs require payload.remote_session."
                )
            live_ref = _REMOTE_SESSION_STORE.put_output(
                session_handle,
                node_id=node_id,
                output_index=output_index,
                value=output_value,
            )
            bridge_record = _build_remote_session_bridge_record(
                payload=normalized_payload,
                hydrated_inputs=hydrated_inputs,
                node_id=node_id,
                output_index=output_index,
                io_type=str(boundary_output.get("io_type") or "*"),
                output_value=output_value,
            )
            _REMOTE_SESSION_BRIDGE_STORE.put_record(bridge_record)
            _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
            output_value = RemoteSessionBridgeRef(
                bridge_key=bridge_record.bridge_key,
                node_id=node_id,
                output_index=output_index,
                session_id=live_ref.session_id,
            ).to_payload()
        outputs.append(output_value)
    logger.info(
        "Mapped remote subgraph %s produced %d exported outputs.",
        payload.get("component_id"),
        len(outputs),
    )
    return tuple(outputs)


def _execute_subgraph_prompt(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    if node_mapping is not None:
        return _execute_subgraph_with_mapping(payload, hydrated_inputs, node_mapping)

    normalized_payload = _trim_subgraph_payload_to_required_nodes(
        _normalize_subgraph_payload(payload)
    )
    session_handle = _payload_remote_session_handle(normalized_payload)
    resolution_stats = _RemoteSessionBridgeResolutionStats()
    resolved_inputs = _resolve_remote_session_inputs(
        dict(hydrated_inputs),
        component_id=str(payload.get("component_id") or ""),
        target_session_handle=session_handle,
        node_mapping=node_mapping,
        resolution_stats=resolution_stats,
    )
    _log_remote_session_resolution_summary(
        component_id=str(payload.get("component_id") or "modal-subgraph"),
        resolution_stats=resolution_stats,
    )
    if session_handle is not None:
        logger.info(
            "Executing PromptExecutor remote subgraph %s with remote_session session_id=%s prompt_id=%s owner_component_id=%s.",
            payload.get("component_id"),
            session_handle.session_id,
            session_handle.prompt_id,
            session_handle.owner_component_id,
        )
    prompt = copy.deepcopy(normalized_payload["subgraph_prompt"])
    logger.info(
        "Executing remote subgraph %s through PromptExecutor with %d prompt nodes, %d boundary inputs, and %d exported outputs.",
        payload.get("component_id"),
        len(prompt),
        len(normalized_payload.get("boundary_inputs", [])),
        len(normalized_payload.get("boundary_outputs", [])),
    )
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
        hydrated_inputs=resolved_inputs,
    )
    execution = _load_execution_module()
    resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    _coerce_prompt_primitive_input_values(prompt, resolved_node_mapping)
    _validate_prompt_input_shapes(
        prompt,
        resolved_node_mapping,
        list(normalized_payload.get("boundary_inputs", [])),
    )

    with _temporary_node_mapping(node_mapping):
        executor = execution.PromptExecutor(_NullPromptServer())
        execution_started_at = time.perf_counter()
        logger.info(
            "Starting PromptExecutor for remote subgraph %s with execute targets %s.",
            payload.get("component_id"),
            normalized_payload.get("execute_node_ids", []),
        )
        executor.execute(
            prompt=prompt,
            prompt_id=str(
                payload.get("prompt_id")
                or payload.get("component_id", "modal-subgraph")
            ),
            extra_data=copy.deepcopy(normalized_payload.get("extra_data") or {}),
            execute_outputs=list(normalized_payload.get("execute_node_ids", [])),
        )
        logger.info(
            "PromptExecutor finished for remote subgraph %s in %.3fs with success=%s and %d status messages.",
            payload.get("component_id"),
            time.perf_counter() - execution_started_at,
            executor.success,
            len(executor.status_messages),
        )
        if executor.status_messages:
            logger.info(
                "Remote subgraph %s status events: %s",
                payload.get("component_id"),
                [event for event, _data in executor.status_messages],
            )
        if not executor.success:
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))

        outputs: list[Any] = []
        for boundary_output in normalized_payload.get("boundary_outputs", []):
            node_id = str(boundary_output["node_id"])
            output_index = int(boundary_output["output_index"])
            cache_entry = executor.caches.outputs.get(node_id)
            if cache_entry is None:
                raise RemoteSubgraphExecutionError(
                    f"Remote subgraph did not produce cache entry for node {node_id}."
                )
            if output_index >= len(cache_entry.outputs):
                raise RemoteSubgraphExecutionError(
                    f"Remote subgraph output index {output_index} is missing for node {node_id}."
                )
            output_value = _collapse_cache_slot(
                slot_values=cache_entry.outputs[output_index],
                is_list=bool(boundary_output.get("is_list", False)),
            )
            if bool(boundary_output.get("session_output")):
                if session_handle is None:
                    raise RemoteSessionStateError(
                        "Session-backed boundary outputs require payload.remote_session."
                    )
                live_ref = _REMOTE_SESSION_STORE.put_output(
                    session_handle,
                    node_id=node_id,
                    output_index=output_index,
                    value=output_value,
                )
                bridge_record = _build_remote_session_bridge_record(
                    payload=normalized_payload,
                    hydrated_inputs=hydrated_inputs,
                    node_id=node_id,
                    output_index=output_index,
                    io_type=str(boundary_output.get("io_type") or "*"),
                    output_value=output_value,
                )
                _REMOTE_SESSION_BRIDGE_STORE.put_record(bridge_record)
                _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
                output_value = RemoteSessionBridgeRef(
                    bridge_key=bridge_record.bridge_key,
                    node_id=node_id,
                    output_index=output_index,
                    session_id=live_ref.session_id,
                ).to_payload()
            outputs.append(output_value)
            logger.info(
                "Collected exported output %s from node %s output %d.",
                boundary_output.get("proxy_output_name"),
                node_id,
                output_index,
            )
        return tuple(outputs)


def execute_subgraph_locally(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    session_handle = _payload_remote_session_handle(payload)
    logger.info(
        "Executing local fallback subgraph %s with %d hydrated inputs session_id=%s clear_remote_session=%s.",
        payload.get("component_id"),
        len(hydrated_inputs),
        session_handle.session_id if session_handle is not None else None,
        bool(payload.get("clear_remote_session")),
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_subgraph_prompt, payload, hydrated_inputs, node_mapping)
            try:
                outputs = future.result()
            except Exception:
                logger.exception(
                    "Local fallback subgraph %s raised while running in worker thread.",
                    payload.get("component_id"),
                )
                raise
    finally:
        if bool(payload.get("clear_remote_session")) and session_handle is not None:
            logger.info(
                "Clearing remote session after component=%s session_id=%s.",
                payload.get("component_id"),
                session_handle.session_id,
            )
            _REMOTE_SESSION_STORE.clear_session(session_handle)
    logger.info(
        "Local fallback subgraph %s completed with %d outputs.",
        payload.get("component_id"),
        len(outputs),
    )
    return serialize_node_outputs(outputs)


def _modal_lookup_error_types() -> tuple[type[BaseException], ...]:
    """Return Modal exception types that indicate lookup or hydration failure."""
    if modal is None:
        return tuple()
    exception_module = getattr(modal, "exception", None)
    if exception_module is None:
        return tuple()

    error_types: list[type[BaseException]] = []
    for error_name in ("NotFoundError", "ExecutionError", "InvalidError"):
        error_type = getattr(exception_module, error_name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            error_types.append(error_type)
    return tuple(error_types)


def _is_missing_modal_deployment_error(exc: BaseException) -> bool:
    """Return whether one Modal lookup failure indicates missing deployed app state."""
    message = str(exc).strip().lower()
    if "not found" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "lookup failed for cls",
            "app '",
            'app "',
            "class '",
            'class "',
            "not found in environment",
        )
    )


def _load_modal_cloud_module() -> Any:
    """Load the stable Modal cloud entry module under a valid Python name."""
    existing_module = sys.modules.get(_MODAL_CLOUD_MODULE_NAME)
    if existing_module is not None and getattr(existing_module, "app", None) is not None:
        return existing_module
    if existing_module is not None:
        logger.warning(
            "Discarding partially initialized Modal cloud module %s before reload.",
            _MODAL_CLOUD_MODULE_NAME,
        )
        sys.modules.pop(_MODAL_CLOUD_MODULE_NAME, None)

    cloud_module_path = Path(__file__).resolve().parents[1] / f"{_MODAL_CLOUD_MODULE_NAME}.py"
    module_spec = importlib.util.spec_from_file_location(
        _MODAL_CLOUD_MODULE_NAME,
        cloud_module_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ModalRemoteInvocationError(
            f"Unable to create module spec for Modal cloud entrypoint at {cloud_module_path}."
        )

    cloud_module = importlib.util.module_from_spec(module_spec)
    sys.modules[_MODAL_CLOUD_MODULE_NAME] = cloud_module
    try:
        module_spec.loader.exec_module(cloud_module)
    except BaseException:
        sys.modules.pop(_MODAL_CLOUD_MODULE_NAME, None)
        raise
    return cloud_module


def _lookup_local_prompt_server() -> Any | None:
    """Return the live local ComfyUI PromptServer instance when available."""
    try:
        import server
    except ModuleNotFoundError:
        return None

    return getattr(server.PromptServer, "instance", None)


def _emit_local_modal_status(
    *,
    prompt_id: str | None,
    client_id: str | None,
    phase: str,
    node_ids: list[str],
    active_node_id: str | None = None,
    active_node_class_type: str | None = None,
    active_node_role: str | None = None,
    error_message: str | None = None,
    status_message: str | None = None,
    status_current: int | None = None,
    status_total: int | None = None,
) -> None:
    """Forward remote execution progress into the local ComfyUI websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload: dict[str, Any] = {
        "phase": phase,
        "prompt_id": prompt_id,
        "node_ids": list(node_ids),
    }
    if active_node_id is not None:
        payload["active_node_id"] = active_node_id
    if active_node_class_type is not None:
        payload["active_node_class_type"] = active_node_class_type
    if active_node_role is not None:
        payload["active_node_role"] = active_node_role
    if error_message is not None:
        payload["error_message"] = error_message
    if status_message is not None:
        payload["status_message"] = status_message
    if status_current is not None:
        payload["status_current"] = int(status_current)
    if status_total is not None:
        payload["status_total"] = int(status_total)
    prompt_server.send_sync("modal_status", payload, client_id)


def _emit_local_modal_progress(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    value: float,
    max_value: float,
    display_node_id: str | None = None,
    real_node_id: str | None = None,
    lane_id: str | None = None,
    clear: bool = False,
    item_index: int | None = None,
    aggregate_only: bool = False,
) -> None:
    """Forward remote numeric node progress into the local ComfyUI websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "node_id": node_id,
        "value": float(value),
        "max": float(max_value),
    }
    if display_node_id is not None:
        payload["display_node_id"] = display_node_id
    if real_node_id is not None:
        payload["real_node_id"] = real_node_id
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if clear:
        payload["clear"] = True
    if item_index is not None:
        payload["item_index"] = int(item_index)
    if aggregate_only:
        payload["aggregate_only"] = True
    prompt_server.send_sync("modal_progress", payload, client_id)


def _emit_local_executed_output(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    display_node_id: str | None,
    output_payload: Any,
) -> None:
    """Forward one remote node's executed UI payload into the local websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload = {
        "prompt_id": prompt_id,
        "node": node_id,
        "display_node": display_node_id or node_id,
        "output": output_payload,
    }
    prompt_server.send_sync("executed", payload, client_id)


def _emit_local_preview_image(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    display_node_id: str | None,
    parent_node_id: str | None,
    real_node_id: str | None,
    image_type: str,
    image_bytes: bytes,
    max_size: int | None,
) -> None:
    """Forward one remote preview image into the local ComfyUI preview websocket path."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    try:
        from PIL import Image
        from protocol import BinaryEventTypes
    except ModuleNotFoundError:
        logger.warning("Preview forwarding is unavailable because Pillow or ComfyUI protocol imports failed.")
        return

    with BytesIO(image_bytes) as image_buffer:
        image = Image.open(image_buffer)
        image.load()

    metadata: dict[str, Any] = {
        "node_id": node_id,
        "prompt_id": prompt_id,
        "display_node_id": display_node_id or node_id,
        "real_node_id": real_node_id or node_id,
    }
    if parent_node_id is not None:
        metadata["parent_node_id"] = parent_node_id

    prompt_server.send_sync(
        BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA,
        ((image_type, image, max_size), metadata),
        client_id,
    )


def _emit_local_preview_boundary_output(
    *,
    prompt_id: str | None,
    client_id: str | None,
    preview_target_node_ids: list[str],
    image_value: Any,
) -> None:
    """Render one streamed remote boundary IMAGE value into local PreviewImage UI events."""
    if client_id is None or not preview_target_node_ids:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    try:
        import nodes
    except ModuleNotFoundError:
        logger.warning("Preview boundary streaming is unavailable because ComfyUI nodes could not be imported.")
        return

    preview_factory = getattr(nodes, "PreviewImage", None)
    if preview_factory is None:
        logger.warning("Preview boundary streaming is unavailable because PreviewImage is not registered.")
        return

    preview_result = preview_factory().save_images(images=image_value)
    if not isinstance(preview_result, dict):
        return
    output_payload = preview_result.get("ui")
    if not isinstance(output_payload, dict):
        return

    for preview_target_node_id in preview_target_node_ids:
        prompt_server.send_sync(
            "executed",
            {
                "prompt_id": prompt_id,
                "node": preview_target_node_id,
                "display_node": preview_target_node_id,
                "output": output_payload,
            },
            client_id,
        )


def _allowed_suppressed_stream_node_ids(payload: dict[str, Any]) -> set[str]:
    """Return the node ids that may surface UI events for a suppressed mapped/static stream."""
    allowed_node_ids = {
        str(node_id)
        for node_id in payload.get("execute_node_ids", [])
        if str(node_id)
    }
    allowed_node_ids.update(
        str(boundary_output["node_id"])
        for boundary_output in payload.get("boundary_outputs", [])
        if boundary_output.get("node_id") is not None and str(boundary_output["node_id"])
    )
    return allowed_node_ids


def _should_forward_suppressed_stream_event(
    payload: dict[str, Any],
    reported_node_id: Any,
) -> bool:
    """Return whether a suppressed mapped/static stream event belongs to this payload."""
    if not bool(payload.get("suppress_status_stream")):
        return True
    if reported_node_id is None:
        return False
    allowed_node_ids = _allowed_suppressed_stream_node_ids(payload)
    if not allowed_node_ids:
        return True
    return str(reported_node_id) in allowed_node_ids


def _progress_stream_event_node_id(stream_event: dict[str, Any]) -> str | None:
    """Return the best node id to use for progress filtering and forwarding."""
    for candidate in (
        stream_event.get("real_node_id"),
        stream_event.get("node_id"),
        stream_event.get("display_node_id"),
    ):
        if candidate is None:
            continue
        candidate_text = str(candidate)
        if candidate_text:
            return candidate_text
    return None


def _progress_stream_event_metadata(stream_event: dict[str, Any]) -> dict[str, str | None] | None:
    """Return normalized metadata for one streamed progress event."""
    reported_node_id = (
        str(stream_event["node_id"])
        if stream_event.get("node_id") is not None
        else None
    )
    display_node_id = (
        str(stream_event["display_node_id"])
        if stream_event.get("display_node_id") is not None
        else reported_node_id
    )
    real_node_id = (
        str(stream_event["real_node_id"])
        if stream_event.get("real_node_id") is not None
        else None
    )
    filter_node_id = real_node_id or reported_node_id or display_node_id
    if filter_node_id is None:
        return None
    return {
        "node_id": reported_node_id or filter_node_id,
        "display_node_id": display_node_id,
        "real_node_id": real_node_id,
        "filter_node_id": filter_node_id,
    }


def _should_stream_remote_progress(payload: dict[str, Any]) -> bool:
    """Return whether the local client has enough context to mirror remote node progress."""
    extra_data = payload.get("extra_data") or {}
    return (
        payload.get("payload_kind") in {"subgraph", "mapped_subgraph"}
        and isinstance(payload.get("prompt_id"), str)
        and bool(payload.get("prompt_id"))
        and isinstance(extra_data.get("client_id"), str)
        and bool(extra_data.get("client_id"))
        and isinstance(payload.get("component_node_ids"), list)
        and len(payload.get("component_node_ids")) > 0
    )


def _local_processing_interrupted() -> bool:
    """Return whether the current local ComfyUI execution was interrupted."""
    try:
        import comfy.model_management
    except ModuleNotFoundError:
        return False

    return bool(comfy.model_management.processing_interrupted())


def _raise_local_interrupt() -> None:
    """Raise ComfyUI's native interruption exception for the current execution."""
    import comfy.model_management

    raise comfy.model_management.InterruptProcessingException()


def _remote_interrupt_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the prompt/component pair used to interrupt one remote execution."""
    prompt_id = str(payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph")
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


def _lookup_modal_interrupt_store() -> Any | None:
    """Return the shared Modal Dict used to signal remote cancellation requests."""
    if modal is None or not hasattr(modal, "Dict"):
        return None

    settings = get_settings()
    cache_key = (settings.interrupt_dict_name, _modal_environment_name())
    with _MODAL_INTERRUPT_DICTS_LOCK:
        cached_store = _MODAL_INTERRUPT_DICTS.get(cache_key)
        if cached_store is not None:
            return cached_store

    interrupt_store = modal.Dict.from_name(
        settings.interrupt_dict_name,
        environment_name=cache_key[1],
        create_if_missing=True,
    )
    with _MODAL_INTERRUPT_DICTS_LOCK:
        _MODAL_INTERRUPT_DICTS[cache_key] = interrupt_store
    return interrupt_store


def _request_remote_interrupt(payload: dict[str, Any]) -> bool:
    """Write one remote cancellation request into the shared Modal interrupt store."""
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return False

    prompt_id, component_id = _remote_interrupt_key(payload)
    interrupt_store.put(
        _remote_interrupt_flag_key(prompt_id, component_id),
        {"requested_at": time.time()},
    )
    logger.info(
        "Propagated local interrupt to Modal prompt=%s component=%s through shared control state.",
        prompt_id,
        component_id,
    )
    return True


def _sync_local_interrupt_to_cancellation_event(
    payload: dict[str, Any],
    cancellation_event: threading.Event | None,
) -> bool:
    """Mirror ComfyUI's interrupt flag into the current Modal cancellation event."""
    if cancellation_event is not None and cancellation_event.is_set():
        return True
    if not _local_processing_interrupted():
        return False
    if cancellation_event is not None and not cancellation_event.is_set():
        logger.info(
            "Observed local interrupt while Modal component=%s was running; requesting remote cancellation.",
            payload.get("component_id"),
        )
        cancellation_event.set()
    return True


def _propagate_remote_interrupt_request(
    payload: dict[str, Any],
    interrupt_remote_call: Callable[[], Any] | None,
) -> None:
    """Send one best-effort remote cancellation request for an active Modal payload."""
    prompt_id, component_id = _remote_interrupt_key(payload)
    if interrupt_remote_call is None:
        logger.warning(
            "Local interrupt requested for component=%s, but no remote interrupt method is available.",
            component_id,
        )
        return
    try:
        interrupt_remote_call()
        logger.info(
            "Propagated local interrupt to Modal prompt=%s component=%s.",
            prompt_id,
            component_id,
        )
    except Exception:
        logger.exception(
            "Failed to propagate local interrupt to Modal prompt=%s component=%s.",
            prompt_id,
            component_id,
        )


def _invoke_remote_call_with_interrupts(
    *,
    payload: dict[str, Any],
    invoke_remote_call: Callable[[], bytes],
    interrupt_remote_call: Callable[[], Any] | None,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Run one blocking remote call while optionally propagating cancellation to Modal."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def execute_remote_call() -> None:
        """Run the blocking Modal request in a worker thread."""
        try:
            result_queue.put(("result", invoke_remote_call()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    request_thread = threading.Thread(
        target=execute_remote_call,
        name=f"modal-request-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    request_thread.start()
    interrupt_sent = False
    try:
        while True:
            try:
                result_kind, result_payload = result_queue.get(timeout=0.1)
            except queue.Empty:
                if (
                    _sync_local_interrupt_to_cancellation_event(payload, cancellation_event)
                    and not interrupt_sent
                ):
                    _propagate_remote_interrupt_request(payload, interrupt_remote_call)
                    interrupt_sent = True
                continue

            if result_kind == "result":
                return bytes(result_payload)
            raise result_payload
    finally:
        request_thread.join(timeout=1.0)


def _consume_remote_payload_stream(
    payload: dict[str, Any],
    stream_events: Iterator[dict[str, Any]],
) -> bytes:
    """Forward remote progress events into the local UI and return the final payload bytes."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    node_ids = [str(node_id) for node_id in payload.get("component_node_ids", [])]
    suppress_status_stream = bool(payload.get("suppress_status_stream"))
    result_payload: bytes | bytearray | None = None
    suppressed_progress_node_metadata: dict[str, dict[str, str | None]] = {}
    active_remote_log_task_id: str | None = None
    should_close_stream = False

    try:
        for stream_event in stream_events:
            event_kind = str(stream_event.get("kind", ""))
            if event_kind == "remote_logs":
                task_id = _coerce_modal_task_id(stream_event.get("task_id"))
                if (
                    task_id is not None
                    and active_remote_log_task_id is None
                    and _is_remote_container_log_stream_enabled()
                ):
                    active_remote_log_task_id = _retain_remote_container_log_stream(task_id)
                continue
            if event_kind == "progress":
                event_type = str(stream_event.get("event_type", ""))
                if event_type == "node_progress":
                    progress_metadata = _progress_stream_event_metadata(stream_event)
                    filter_node_id = (
                        progress_metadata["filter_node_id"] if progress_metadata is not None else None
                    )
                    lane_id = (
                        str(stream_event["lane_id"])
                        if stream_event.get("lane_id") is not None
                        else (
                            str(payload["mapped_progress_lane_id"])
                            if payload.get("mapped_progress_lane_id") is not None
                            else None
                        )
                    )
                    aggregate_only = bool(stream_event.get("aggregate_only", False))
                    if (
                        suppress_status_stream
                        and lane_id is None
                        and not aggregate_only
                        and not _should_forward_suppressed_stream_event(payload, filter_node_id)
                    ):
                        logger.debug(
                            "Suppressing streamed Modal node progress for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            stream_event.get("real_node_id"),
                        )
                        continue
                    reported_node_id = (
                        progress_metadata["node_id"] if progress_metadata is not None else None
                    )
                    if reported_node_id is not None:
                        display_node_id = (
                            progress_metadata["display_node_id"]
                            if progress_metadata is not None
                            else str(reported_node_id)
                        )
                        real_node_id = (
                            progress_metadata["real_node_id"] if progress_metadata is not None else None
                        )
                        progress_node_id = real_node_id or display_node_id
                        if lane_id is not None:
                            _remember_mapped_lane_node_id(payload, lane_id, progress_node_id)
                        elif suppress_status_stream and not aggregate_only and progress_metadata is not None:
                            suppressed_progress_node_metadata[str(progress_metadata["filter_node_id"])] = {
                                "node_id": str(reported_node_id),
                                "display_node_id": display_node_id,
                                "real_node_id": real_node_id,
                            }
                        logger.debug(
                            "Forwarding streamed Modal node progress for component=%s node_id=%s real_node_id=%s value=%s max=%s lane_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                            real_node_id,
                            stream_event.get("value"),
                            stream_event.get("max"),
                            lane_id,
                        )
                        _emit_local_modal_progress(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            value=float(stream_event.get("value", 0.0)),
                            max_value=float(stream_event.get("max", 1.0)),
                            display_node_id=display_node_id,
                            real_node_id=real_node_id,
                            lane_id=lane_id,
                            clear=bool(stream_event.get("clear", False)),
                            item_index=(
                                int(stream_event["item_index"])
                                if stream_event.get("item_index") is not None
                                else (
                                    int(payload["map_item_index"])
                                    if payload.get("map_item_index") is not None
                                    else None
                                )
                            ),
                            aggregate_only=aggregate_only,
                        )
                    continue
                if event_type == "executed":
                    reported_node_id = stream_event.get("node_id")
                    if reported_node_id is not None:
                        if not _should_forward_suppressed_stream_event(payload, reported_node_id):
                            logger.debug(
                                "Suppressing streamed Modal executed output for component=%s node_id=%s because it does not belong to this mapped/static payload.",
                                payload.get("component_id"),
                                reported_node_id,
                            )
                            continue
                        logger.debug(
                            "Forwarding streamed Modal executed output for component=%s node_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                        )
                        _emit_local_executed_output(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            display_node_id=(
                                str(stream_event["display_node_id"])
                                if stream_event.get("display_node_id") is not None
                                else None
                            ),
                            output_payload=deserialize_value(stream_event.get("output")),
                        )
                    continue
                if event_type == "preview":
                    reported_node_id = stream_event.get("node_id")
                    image_bytes = deserialize_value(stream_event.get("image_bytes"))
                    if reported_node_id is not None and isinstance(image_bytes, bytes):
                        if not _should_forward_suppressed_stream_event(payload, reported_node_id):
                            logger.debug(
                                "Suppressing streamed Modal preview image for component=%s node_id=%s because it does not belong to this mapped/static payload.",
                                payload.get("component_id"),
                                reported_node_id,
                            )
                            continue
                        logger.debug(
                            "Forwarding streamed Modal preview image for component=%s node_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                        )
                        _emit_local_preview_image(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            display_node_id=(
                                str(stream_event["display_node_id"])
                                if stream_event.get("display_node_id") is not None
                                else None
                            ),
                            parent_node_id=(
                                str(stream_event["parent_node_id"])
                                if stream_event.get("parent_node_id") is not None
                                else None
                            ),
                            real_node_id=(
                                str(stream_event["real_node_id"])
                                if stream_event.get("real_node_id") is not None
                                else None
                            ),
                            image_type=str(stream_event.get("image_type", "PNG")),
                            image_bytes=image_bytes,
                            max_size=(
                                int(stream_event["max_size"])
                                if stream_event.get("max_size") is not None
                                else None
                            ),
                        )
                    continue
                if event_type == "boundary_output":
                    preview_target_node_ids = [
                        str(node_id)
                        for node_id in stream_event.get("preview_target_node_ids", [])
                        if str(node_id)
                    ]
                    if preview_target_node_ids:
                        logger.debug(
                            "Forwarding streamed Modal boundary output previews for component=%s source_node=%s targets=%s.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            preview_target_node_ids,
                        )
                        _emit_local_preview_boundary_output(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            preview_target_node_ids=preview_target_node_ids,
                            image_value=deserialize_value(stream_event.get("value")),
                        )
                    continue
                logger.info(
                    "Forwarding streamed Modal progress for component=%s phase=%s active_node_id=%s.",
                    payload.get("component_id"),
                    stream_event.get("phase"),
                    stream_event.get("active_node_id"),
                )
                if suppress_status_stream:
                    remote_phase = str(stream_event.get("phase", "executing"))
                    if remote_phase in {"execution_success", "execution_error", "execution_interrupted"}:
                        for progress_metadata in suppressed_progress_node_metadata.values():
                            _emit_local_modal_progress(
                                prompt_id=prompt_id,
                                client_id=client_id,
                                node_id=str(progress_metadata["node_id"]),
                                value=0.0,
                                max_value=1.0,
                                display_node_id=progress_metadata["display_node_id"],
                                real_node_id=progress_metadata["real_node_id"],
                                clear=True,
                            )
                        suppressed_progress_node_metadata.clear()
                    continue
                remote_phase = str(stream_event.get("phase", "executing"))
                if remote_phase == "execution_success":
                    _emit_local_modal_status(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        phase="finalizing",
                        node_ids=node_ids,
                        status_message="Receiving Modal outputs",
                    )
                    continue
                _emit_local_modal_status(
                    prompt_id=prompt_id,
                    client_id=client_id,
                    phase=remote_phase,
                    node_ids=node_ids,
                    active_node_id=(
                        str(stream_event["active_node_id"])
                        if stream_event.get("active_node_id") is not None
                        else None
                    ),
                    active_node_class_type=(
                        str(stream_event["active_node_class_type"])
                        if stream_event.get("active_node_class_type") is not None
                        else None
                    ),
                    active_node_role=(
                        str(stream_event["active_node_role"])
                        if stream_event.get("active_node_role") is not None
                        else None
                    ),
                )
                continue
            if event_kind == "result":
                candidate_outputs = stream_event.get("outputs")
                try:
                    result_payload = coerce_serialized_node_outputs(candidate_outputs)
                except TypeError as exc:
                    raise ModalRemoteInvocationError(
                        "Modal streamed payload result did not include transport-safe outputs."
                    ) from exc
                should_close_stream = True
                break
            logger.debug(
                "Ignoring unexpected streamed Modal event kind=%s for component=%s.",
                event_kind,
                payload.get("component_id"),
            )
    finally:
        if should_close_stream:
            _close_remote_payload_stream(stream_events)
        if active_remote_log_task_id is not None:
            _release_remote_container_log_stream(active_remote_log_task_id)

    if result_payload is None:
        raise ModalRemoteInvocationError(
            f"Modal streamed payload for component={payload.get('component_id')!r} did not yield a final result."
        )
    return bytes(result_payload)


def _mapped_execution_parallelism(item_count: int) -> int:
    """Return the local worker width used to schedule mapped Modal item executions."""
    settings = get_settings()
    configured_limit = settings.max_containers or _remote_modal_call_worker_count()
    return max(1, min(item_count, configured_limit))


def _mapped_phase_definition(payload: dict[str, Any], phase_key: str) -> dict[str, Any] | None:
    """Return one explicit mapped phase definition when queue-time planning provided it."""
    phase_payload = payload.get(phase_key)
    if isinstance(phase_payload, dict):
        return phase_payload
    return None


def _shared_subgraph_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload fields shared by every explicit mapped phase."""
    shared_fields = {
        "prompt_id": payload.get("prompt_id"),
        "extra_data": copy.deepcopy(payload.get("extra_data") or {}),
        "requires_volume_reload": bool(payload.get("requires_volume_reload", True)),
        "volume_reload_marker": payload.get("volume_reload_marker"),
        "uploaded_volume_paths": list(payload.get("uploaded_volume_paths", [])),
        "terminate_container_on_error": bool(payload.get("terminate_container_on_error", True)),
        "custom_nodes_bundle": payload.get("custom_nodes_bundle"),
    }
    remote_session = payload.get("remote_session")
    if remote_session is not None:
        shared_fields["remote_session"] = copy.deepcopy(remote_session)
    if bool(payload.get("clear_remote_session")):
        shared_fields["clear_remote_session"] = True
    return shared_fields


def _build_phase_subgraph_payload(
    payload: dict[str, Any],
    phase_key: str,
    component_id: str,
    *,
    suppress_status_stream: bool = False,
    lane_id: str | None = None,
    item_index: int | None = None,
) -> dict[str, Any]:
    """Return one explicit static or mapped subgraph payload."""
    phase_definition = _mapped_phase_definition(payload, phase_key)
    if phase_definition is None:
        raise KeyError(f"Mapped payload is missing phase definition {phase_key!r}.")

    phase_payload = {
        "payload_kind": "subgraph",
        "component_id": component_id,
        **_shared_subgraph_payload_fields(payload),
        "component_node_ids": [
            str(node_id)
            for node_id in phase_definition.get("component_node_ids", [])
            if str(node_id)
        ],
        "subgraph_prompt": copy.deepcopy(phase_definition.get("subgraph_prompt", {})),
        "boundary_inputs": copy.deepcopy(phase_definition.get("boundary_inputs", [])),
        "boundary_outputs": copy.deepcopy(phase_definition.get("boundary_outputs", [])),
        "execute_node_ids": [
            str(node_id)
            for node_id in phase_definition.get("execute_node_ids", [])
            if str(node_id)
        ],
    }
    if suppress_status_stream:
        phase_payload["suppress_status_stream"] = True
    if lane_id is not None:
        phase_payload["mapped_progress_lane_id"] = str(lane_id)
        phase_payload["mapped_progress_display_node_id"] = str(
            payload.get("component_id", "modal-subgraph")
        )
    if item_index is not None:
        phase_payload["map_item_index"] = int(item_index)
    return phase_payload


def _split_phase_outputs(
    phase_outputs: tuple[Any, ...],
    boundary_outputs: list[dict[str, Any]],
    internal_output_names: set[str],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Split one phase result tuple into bridge values and external outputs."""
    internal_outputs: dict[str, Any] = {}
    external_outputs: list[Any] = []
    for boundary_output, output_value in zip(boundary_outputs, phase_outputs, strict=True):
        output_name = str(boundary_output.get("proxy_output_name") or "")
        if output_name in internal_output_names:
            internal_outputs[output_name] = output_value
            continue
        external_outputs.append(output_value)
    return internal_outputs, tuple(external_outputs)


def _execute_mapped_subgraph_payload(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> tuple[Any, ...]:
    """Execute one mapped payload locally using explicit static and mapped phases."""
    mapped_input = payload.get("mapped_input") or {}
    mapped_input_name = str(mapped_input.get("proxy_input_name") or "")
    if not mapped_input_name:
        raise ModalRemoteInvocationError("Mapped remote payloads must define mapped_input.proxy_input_name.")
    if mapped_input_name not in hydrated_inputs:
        raise KeyError(f"Mapped remote payload input {mapped_input_name!r} was not provided.")

    mapped_items = split_mapped_value(
        hydrated_inputs[mapped_input_name],
        str(mapped_input.get("io_type", "*")),
    )
    if not mapped_items:
        raise ValueError("Mapped remote execution requires at least one input item.")

    broadcast_inputs = dict(hydrated_inputs)
    broadcast_inputs.pop(mapped_input_name, None)
    static_to_mapped_boundaries = list(payload.get("static_to_mapped_boundaries", []))
    bridge_output_names = {
        str(boundary_spec.get("proxy_name") or "")
        for boundary_spec in static_to_mapped_boundaries
        if str(boundary_spec.get("proxy_name") or "")
    }

    static_outputs: tuple[Any, ...] = ()
    static_phase_payload: dict[str, Any] | None = None
    if payload.get("static_phase") is not None:
        static_phase_payload = _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
            suppress_status_stream=True,
        )
        static_phase_payload.pop("clear_remote_session", None)
    elif payload.get("static_execute_node_ids"):
        static_phase_payload = _build_static_mapped_payload(payload)

    if static_phase_payload is not None:
        if static_phase_payload.get("execute_node_ids"):
            logger.info(
                "Executing static mapped phase for component=%s with execute nodes=%s.",
                payload.get("component_id"),
                static_phase_payload.get("execute_node_ids", []),
            )
            static_phase_outputs = _execute_subgraph_prompt(
                static_phase_payload,
                dict(broadcast_inputs),
                node_mapping,
            )
            bridge_inputs, static_outputs = _split_phase_outputs(
                static_phase_outputs,
                list(static_phase_payload.get("boundary_outputs", [])),
                bridge_output_names,
            )
            broadcast_inputs.update(bridge_inputs)

    total_items = len(mapped_items)
    _emit_local_mapped_progress(payload, 0, total_items)
    per_item_outputs: list[tuple[Any, ...]] = []
    for item_index, item_value in enumerate(mapped_items):
        if _local_processing_interrupted():
            _raise_local_interrupt()
        if payload.get("mapped_phase") is not None:
            item_payload = _build_phase_subgraph_payload(
                payload,
                "mapped_phase",
                f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
                suppress_status_stream=True,
                lane_id="0",
                item_index=item_index,
            )
        else:
            item_payload = _build_mapped_item_payload(payload, item_index, 0)
        item_inputs = dict(broadcast_inputs)
        item_inputs[mapped_input_name] = item_value
        logger.info(
            "Executing mapped item %d/%d for component=%s with execute nodes=%s.",
            item_index + 1,
            total_items,
            payload.get("component_id"),
            item_payload.get("execute_node_ids", []),
        )
        per_item_outputs.append(
            _execute_subgraph_prompt(
                item_payload,
                item_inputs,
                node_mapping,
            )
        )
        _emit_local_mapped_progress(payload, item_index + 1, total_items)

    if payload.get("mapped_phase") is not None:
        mapped_phase_payload = _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::mapped",
        )
    else:
        mapped_phase_payload = {
            **payload,
            "boundary_outputs": [
                boundary_output
                for boundary_output in payload.get("boundary_outputs", [])
                if _is_mapped_boundary_output(boundary_output, payload)
            ],
        }
    mapped_outputs = _aggregate_mapped_outputs(
        per_item_outputs,
        {
            **payload,
            "boundary_outputs": list(mapped_phase_payload.get("boundary_outputs", [])),
        },
    )
    return _merge_static_and_mapped_outputs(
        static_outputs=static_outputs,
        mapped_outputs=mapped_outputs,
        payload=payload,
    )


def _build_mapped_item_payload(
    payload: dict[str, Any],
    item_index: int,
    lane_index: int,
) -> dict[str, Any]:
    """Return one per-item subgraph payload derived from a mapped remote component payload."""
    if _mapped_phase_definition(payload, "mapped_phase") is not None:
        return _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
            suppress_status_stream=True,
            lane_id=str(lane_index),
            item_index=item_index,
        )
    item_payload = copy.deepcopy(payload)
    item_payload["payload_kind"] = "subgraph"
    item_payload["component_id"] = f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}"
    item_payload["mapped_input"] = None
    item_payload["suppress_status_stream"] = True
    item_payload["map_item_index"] = item_index
    item_payload["mapped_progress_lane_id"] = str(lane_index)
    item_payload["mapped_progress_display_node_id"] = str(
        payload.get("component_id", "modal-subgraph")
    )
    item_payload.pop("clear_remote_session", None)
    item_payload["execute_node_ids"] = list(
        payload.get("mapped_execute_node_ids") or payload.get("execute_node_ids", [])
    )
    item_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in payload.get("boundary_outputs", [])
        if _is_mapped_boundary_output(boundary_output, payload)
    ]
    return item_payload


def _aggregate_mapped_outputs(
    per_item_outputs: list[tuple[Any, ...]],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble ordered per-item outputs from mapped execution into one proxy result tuple."""
    if not per_item_outputs:
        raise ValueError("Mapped execution produced no per-item outputs to aggregate.")

    output_count = len(per_item_outputs[0])
    if any(len(item_outputs) != output_count for item_outputs in per_item_outputs):
        raise RemoteSubgraphExecutionError("Mapped remote execution produced inconsistent output arity.")

    boundary_outputs = list(payload.get("boundary_outputs", []))
    aggregated_outputs: list[Any] = []
    for output_index in range(output_count):
        boundary_output = boundary_outputs[output_index] if output_index < len(boundary_outputs) else {}
        aggregated_outputs.append(
            join_mapped_values(
                [item_outputs[output_index] for item_outputs in per_item_outputs],
                io_type=str(boundary_output.get("io_type", "*")),
                is_list=bool(boundary_output.get("is_list", False)),
            )
        )
    return tuple(aggregated_outputs)


def _build_remote_session_cleanup_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a dedicated cleanup payload that clears one shared remote session once."""
    remote_session = payload.get("remote_session")
    if not bool(payload.get("clear_remote_session")) or not is_remote_session_handle_payload(remote_session):
        return None
    return {
        "payload_kind": "subgraph",
        "component_id": f"{payload.get('component_id', 'modal-subgraph')}::cleanup",
        **_shared_subgraph_payload_fields(payload),
        "component_node_ids": [],
        "subgraph_prompt": {},
        "boundary_inputs": [],
        "boundary_outputs": [],
        "execute_node_ids": [],
        "remote_session": copy.deepcopy(remote_session),
        "clear_remote_session": True,
        "suppress_status_stream": True,
        "terminate_container_on_error": False,
    }


def _is_mapped_boundary_output(boundary_output: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Return whether one boundary output belongs to the mapped per-item branch."""
    mapped_output = boundary_output.get("mapped_output")
    if mapped_output is not None:
        return bool(mapped_output)
    return bool(payload.get("mapped_input")) and not bool(payload.get("static_execute_node_ids"))


def _build_static_mapped_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the one-time static subgraph payload for a hybrid mapped component."""
    if _mapped_phase_definition(payload, "static_phase") is not None:
        return _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
            suppress_status_stream=True,
        )
    static_payload = copy.deepcopy(payload)
    static_payload["payload_kind"] = "subgraph"
    static_payload["component_id"] = f"{payload.get('component_id', 'modal-subgraph')}::static"
    static_payload["mapped_input"] = None
    static_payload["suppress_status_stream"] = True
    static_payload.pop("clear_remote_session", None)
    static_payload["execute_node_ids"] = list(payload.get("static_execute_node_ids") or [])
    static_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in payload.get("boundary_outputs", [])
        if not _is_mapped_boundary_output(boundary_output, payload)
    ]
    return static_payload


def _merge_static_and_mapped_outputs(
    *,
    static_outputs: tuple[Any, ...],
    mapped_outputs: tuple[Any, ...],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble one hybrid mapped component's static and mapped outputs in original order."""
    combined_outputs: list[Any] = []
    static_output_index = 0
    mapped_output_index = 0

    for boundary_output in payload.get("boundary_outputs", []):
        if _is_mapped_boundary_output(boundary_output, payload):
            if mapped_output_index >= len(mapped_outputs):
                raise RemoteSubgraphExecutionError(
                    "Mapped remote execution returned fewer mapped outputs than expected."
                )
            combined_outputs.append(mapped_outputs[mapped_output_index])
            mapped_output_index += 1
            continue
        if static_output_index >= len(static_outputs):
            raise RemoteSubgraphExecutionError(
                "Mapped remote execution returned fewer static outputs than expected."
            )
        combined_outputs.append(static_outputs[static_output_index])
        static_output_index += 1

    if static_output_index != len(static_outputs) or mapped_output_index != len(mapped_outputs):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced extra outputs that did not match the declared boundary outputs."
        )
    return tuple(combined_outputs)


def _emit_local_mapped_progress(
    payload: dict[str, Any],
    completed_items: int,
    total_items: int,
) -> None:
    """Emit one aggregate mapped-execution progress update for the component representative node."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    display_node_id = str(
        payload.get("mapped_progress_display_node_id")
        or payload.get("component_id")
        or ""
    )
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=float(completed_items),
        max_value=float(total_items),
        display_node_id=display_node_id,
        aggregate_only=True,
    )


def _mapped_progress_owner_component_id(payload: dict[str, Any]) -> str | None:
    """Return the stable component id used to track one mapped worker lane locally."""
    owner_component_id = payload.get("mapped_progress_display_node_id", payload.get("component_id"))
    if owner_component_id is None:
        return None
    owner_component = str(owner_component_id)
    return owner_component or None


def _remember_mapped_lane_node_id(payload: dict[str, Any], lane_id: str, node_id: str) -> None:
    """Remember the last real node id that emitted progress for one mapped worker lane."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    owner_component_id = _mapped_progress_owner_component_id(payload)
    if not prompt_id or not owner_component_id or not node_id:
        return
    with _MAPPED_PROGRESS_NODE_IDS_LOCK:
        _MAPPED_PROGRESS_NODE_IDS[(prompt_id, owner_component_id, lane_id)] = node_id


def _pop_mapped_lane_node_id(payload: dict[str, Any], lane_id: str) -> str | None:
    """Forget and return the last real node id that emitted progress for one mapped worker lane."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    owner_component_id = _mapped_progress_owner_component_id(payload)
    if not prompt_id or not owner_component_id:
        return None
    with _MAPPED_PROGRESS_NODE_IDS_LOCK:
        return _MAPPED_PROGRESS_NODE_IDS.pop((prompt_id, owner_component_id, lane_id), None)


def _clear_local_mapped_lane_progress(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int,
) -> None:
    """Remove one mapped worker lane from the local node overlay."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    lane_id = str(lane_index)
    display_node_id = _pop_mapped_lane_node_id(payload, lane_id) or str(payload.get("component_id") or "")
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=lane_id,
        clear=True,
        item_index=item_index,
    )


def _emit_local_mapped_lane_progress_start(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int,
) -> None:
    """Create or reset one mapped worker lane before remote progress begins arriving."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    display_node_id = str(payload.get("component_id") or "")
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=str(lane_index),
        item_index=item_index,
    )


def _split_batch_boundary_inputs(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
) -> tuple[dict[str, list[Any]], int] | None:
    """Return zipped per-item boundary inputs when an ordinary subgraph receives batched values."""
    implicitly_batchable_scalar_io_types = frozenset({"BOOLEAN", "FLOAT", "INT", "STRING"})
    implicitly_batchable_transport_io_types = frozenset({"IMAGE", "LATENT", "MASK", "NOISE", "SIGMAS"})
    split_inputs: dict[str, list[Any]] = {}
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if not proxy_input_name or proxy_input_name not in hydrated_inputs:
            continue
        input_value = hydrated_inputs[proxy_input_name]
        io_type = _implicit_batch_boundary_effective_io_type(
            payload=payload,
            boundary_input=boundary_input,
            input_value=input_value,
        )
        is_session_ref_list = (
            isinstance(input_value, list)
            and len(input_value) > 0
            and all(
                is_remote_session_value_ref_payload(item)
                or is_remote_session_bridge_ref_payload(item)
                for item in input_value
            )
        )
        if (
            isinstance(input_value, list)
            and not is_session_ref_list
            and io_type not in (
            implicitly_batchable_scalar_io_types | implicitly_batchable_transport_io_types
            )
        ):
            logger.info(
                "Skipping implicit batch split for boundary input %s io_type=%s because list-backed non-scalar values stay broadcast.",
                proxy_input_name,
                io_type,
            )
            continue
        try:
            items = split_mapped_value(
                input_value,
                io_type,
            )
        except (TypeError, ValueError):
            continue
        if len(items) <= 1:
            continue
        split_inputs[proxy_input_name] = items

    if not split_inputs:
        return None

    item_counts = {input_name: len(items) for input_name, items in split_inputs.items()}
    unique_counts = set(item_counts.values())
    if len(unique_counts) != 1:
        raise ModalRemoteInvocationError(
            "Implicit Modal batch boundary inputs must all have the same item count. "
            f"Received counts: {item_counts!r}"
        )
    return split_inputs, next(iter(unique_counts))


def _is_latent_like_mapping(value: Any) -> bool:
    """Return whether one runtime value looks like a ComfyUI LATENT mapping."""
    return isinstance(value, Mapping) and "samples" in value


def _list_is_latent_like_batch(value: Any) -> bool:
    """Return whether one runtime value is a list of LATENT-like mappings."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_latent_like_mapping(item) for item in value)
    )


def _implicit_batch_boundary_target_input_types(
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
) -> set[str]:
    """Return the declared input types of one boundary input's target sockets."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return set()

    try:
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    except ModuleNotFoundError:
        logger.debug(
            "Skipping target input type discovery for implicit batching because ComfyUI nodes are unavailable."
        )
        return set()

    target_input_types: set[str] = set()
    for target in boundary_input.get("targets", []):
        node_id = str(target.get("node_id") or "")
        input_name = str(target.get("input_name") or "")
        if not node_id or not input_name:
            continue
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, dict):
            continue
        node_class = resolved_node_mapping.get(str(prompt_node.get("class_type")))
        if node_class is None:
            continue
        declared_type = _node_input_type_map(node_class).get(input_name)
        if isinstance(declared_type, str) and declared_type:
            target_input_types.add(declared_type)
    return target_input_types


def _implicit_batch_boundary_effective_io_type(
    *,
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
    input_value: Any,
) -> str:
    """Return the best effective io_type for implicit batching of one boundary input."""
    declared_io_type = str(boundary_input.get("io_type", "*"))
    if declared_io_type != "*":
        return declared_io_type

    if _list_is_latent_like_batch(input_value):
        return "LATENT"

    target_input_types = _implicit_batch_boundary_target_input_types(payload, boundary_input)
    if len(target_input_types) == 1:
        return next(iter(target_input_types))
    return declared_io_type


def _implicit_batch_input_is_list_target_node_ids(
    payload: dict[str, Any],
    split_inputs: dict[str, list[Any]],
) -> list[str]:
    """Return split-boundary target nodes that must consume the full list in one execution."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return []

    try:
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    except ModuleNotFoundError:
        logger.debug(
            "Skipping INPUT_IS_LIST detection for implicit batching because ComfyUI nodes are unavailable."
        )
        return []
    target_node_ids: set[str] = set()
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if proxy_input_name not in split_inputs:
            continue
        for target in boundary_input.get("targets", []):
            target_node_id = str(target.get("node_id") or "")
            if not target_node_id:
                continue
            prompt_node = prompt.get(target_node_id)
            if prompt_node is None:
                continue
            class_type = str(prompt_node.get("class_type"))
            node_class = resolved_node_mapping.get(class_type)
            if node_class is None:
                continue
            if bool(getattr(node_class, "INPUT_IS_LIST", False)):
                target_node_ids.add(target_node_id)
    return sorted(target_node_ids)


def _partition_implicit_batched_execute_nodes(
    payload: dict[str, Any],
    split_inputs: dict[str, list[Any]],
) -> tuple[list[str], list[str]]:
    """Split one implicitly batched subgraph into static and per-item execute targets."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        execute_node_ids = [str(node_id) for node_id in payload.get("execute_node_ids", [])]
        return [], execute_node_ids

    batched_target_node_ids: set[str] = set()
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if proxy_input_name not in split_inputs:
            continue
        for target in boundary_input.get("targets", []):
            target_node_id = target.get("node_id")
            if target_node_id is not None:
                batched_target_node_ids.add(str(target_node_id))

    execute_node_ids = [str(node_id) for node_id in payload.get("execute_node_ids", [])]
    static_execute_node_ids: list[str] = []
    mapped_execute_node_ids: list[str] = []
    for execute_node_id in execute_node_ids:
        required_node_ids = set(
            _resolve_required_subgraph_nodes(
                prompt=prompt,
                execute_node_ids=[execute_node_id],
            )
        )
        if required_node_ids & batched_target_node_ids:
            mapped_execute_node_ids.append(execute_node_id)
            continue
        static_execute_node_ids.append(execute_node_id)

    if not mapped_execute_node_ids and execute_node_ids:
        logger.warning(
            "Implicitly batched Modal component=%s had batched inputs %s but no execute target depended on them; "
            "falling back to per-item execution for all execute nodes.",
            payload.get("component_id"),
            sorted(split_inputs),
        )
        return [], execute_node_ids

    logger.info(
        "Partitioned implicitly batched Modal component=%s into static execute nodes=%s and mapped execute nodes=%s.",
        payload.get("component_id"),
        static_execute_node_ids,
        mapped_execute_node_ids,
    )
    return static_execute_node_ids, mapped_execute_node_ids


def _annotate_implicit_batched_boundary_outputs(
    payload: dict[str, Any],
    mapped_execute_node_ids: list[str],
) -> list[dict[str, Any]]:
    """Mark which boundary outputs belong to the per-item branch of an implicitly batched subgraph."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict) or not mapped_execute_node_ids:
        return [copy.deepcopy(boundary_output) for boundary_output in payload.get("boundary_outputs", [])]

    mapped_required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=[str(node_id) for node_id in mapped_execute_node_ids],
        )
    )
    annotated_outputs: list[dict[str, Any]] = []
    for boundary_output in payload.get("boundary_outputs", []):
        annotated_output = copy.deepcopy(boundary_output)
        annotated_output["mapped_output"] = (
            str(boundary_output.get("node_id")) in mapped_required_node_ids
        )
        annotated_outputs.append(annotated_output)
    return annotated_outputs


async def _invoke_implicitly_mapped_subgraph_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Fan out one ordinary subgraph payload when batchable boundary inputs arrive zipped."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    split_batch_inputs = _split_batch_boundary_inputs(payload, hydrated_inputs)
    if split_batch_inputs is None:
        raise ModalRemoteInvocationError(
            "Implicit mapped subgraph execution requires at least one batched boundary input."
        )

    split_inputs, total_items = split_batch_inputs
    input_is_list_target_node_ids = _implicit_batch_input_is_list_target_node_ids(
        payload,
        split_inputs,
    )
    if input_is_list_target_node_ids:
        logger.info(
            "Executing implicitly batched Modal component=%s as one ordinary subgraph because split boundary inputs target INPUT_IS_LIST nodes=%s.",
            payload.get("component_id"),
            input_is_list_target_node_ids,
        )
        return await invoke_remote_engine_async(payload, kwargs_payload)

    parallelism = _mapped_execution_parallelism(total_items)
    refined_prompt_warmup_target = _register_exact_component_parallelism(payload, parallelism)
    ensure_remote_warm_capacity(
        _build_prompt_warmup_request(payload),
        warmup_target=refined_prompt_warmup_target,
        reason="implicit_mapped_component_exact_parallelism",
    )
    logger.info(
        "Scheduling implicitly mapped Modal component=%s for %d item(s) with local parallelism=%d prompt_warmup_target=%d across inputs=%s.",
        payload.get("component_id"),
        total_items,
        parallelism,
        refined_prompt_warmup_target,
        sorted(split_inputs),
    )
    _emit_local_mapped_progress(payload, 0, total_items)

    broadcast_inputs = {
        input_name: value
        for input_name, value in hydrated_inputs.items()
        if input_name not in split_inputs
    }
    static_execute_node_ids, mapped_execute_node_ids = _partition_implicit_batched_execute_nodes(
        payload,
        split_inputs,
    )
    hybrid_payload = copy.deepcopy(payload)
    hybrid_payload["static_execute_node_ids"] = static_execute_node_ids
    hybrid_payload["mapped_execute_node_ids"] = mapped_execute_node_ids
    hybrid_payload["boundary_outputs"] = _annotate_implicit_batched_boundary_outputs(
        payload,
        mapped_execute_node_ids,
    )
    cleanup_payload = _build_remote_session_cleanup_payload(hybrid_payload)
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    use_seeded_remote_lanes = (
        execution_mode != "local"
        and modal is not None
        and _payload_remote_session_handle(hybrid_payload) is not None
        and _mapped_phase_definition(hybrid_payload, "static_phase") is not None
    )

    static_outputs: tuple[Any, ...] = ()
    lane_remote_engines: list[Any] = []
    try:
        if use_seeded_remote_lanes:
            lane_remote_engines = [
                _lookup_deployed_remote_engine(
                    hybrid_payload,
                    affinity_key_override=_mapped_lane_affinity_key(hybrid_payload, lane_index),
                )
                for lane_index in range(parallelism)
            ]
            logger.info(
                "Seeding %d mapped worker lane(s) for component=%s before per-item dispatch.",
                parallelism,
                payload.get("component_id"),
            )

            async def seed_lane(lane_index: int) -> None:
                """Run the static bridge-producing phase once on one mapped worker lane."""
                lane_seed_payload = _build_phase_subgraph_payload(
                    hybrid_payload,
                    "static_phase",
                    f"{payload.get('component_id', 'modal-subgraph')}::seed:{lane_index}",
                    suppress_status_stream=True,
                )
                lane_seed_payload.pop("clear_remote_session", None)
                await _invoke_bound_remote_engine_async(
                    lane_remote_engines[lane_index],
                    lane_seed_payload,
                    serialize_node_inputs(broadcast_inputs),
                )

            await asyncio.gather(
                *(seed_lane(lane_index) for lane_index in range(parallelism))
            )
        elif static_execute_node_ids:
            static_response = await invoke_remote_engine_async(
                _build_static_mapped_payload(hybrid_payload),
                serialize_node_inputs(broadcast_inputs),
            )
            static_outputs = deserialize_node_outputs(static_response)
        else:
            lane_remote_engines = []

        per_item_outputs: list[tuple[Any, ...] | None] = [None] * total_items
        completed_items = 0
        item_queue: asyncio.Queue[int | None] = asyncio.Queue()
        for item_index in range(total_items):
            item_queue.put_nowait(item_index)
        for _ in range(parallelism):
            item_queue.put_nowait(None)

        async def run_worker(lane_index: int) -> None:
            """Execute queued implicit mapped items through one stable local worker lane."""
            nonlocal completed_items
            while True:
                item_index = await item_queue.get()
                if item_index is None:
                    return
                if _local_processing_interrupted():
                    _raise_local_interrupt()
                try:
                    item_payload = _build_mapped_item_payload(hybrid_payload, item_index, lane_index)
                    item_inputs = dict(broadcast_inputs)
                    for input_name, items in split_inputs.items():
                        item_inputs[input_name] = items[item_index]
                    if use_seeded_remote_lanes:
                        item_response = await _invoke_bound_remote_engine_async(
                            lane_remote_engines[lane_index],
                            item_payload,
                            serialize_node_inputs(item_inputs),
                        )
                    else:
                        item_response = await invoke_remote_engine_async(
                            item_payload,
                            serialize_node_inputs(item_inputs),
                        )
                    per_item_outputs[item_index] = deserialize_node_outputs(item_response)
                    completed_items += 1
                    _emit_local_mapped_progress(payload, completed_items, total_items)
                finally:
                    _clear_local_mapped_lane_progress(payload, lane_index, item_index)

        tasks = [asyncio.create_task(run_worker(lane_index)) for lane_index in range(parallelism)]
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return serialize_node_outputs(
            _merge_static_and_mapped_outputs(
                static_outputs=static_outputs,
                mapped_outputs=_aggregate_mapped_outputs(
                    [item_outputs for item_outputs in per_item_outputs if item_outputs is not None],
                    {
                        **hybrid_payload,
                        "boundary_outputs": [
                            boundary_output
                            for boundary_output in hybrid_payload.get("boundary_outputs", [])
                            if _is_mapped_boundary_output(boundary_output, hybrid_payload)
                        ],
                    },
                ),
                payload=hybrid_payload,
            )
        )
    finally:
        if cleanup_payload is not None:
            if use_seeded_remote_lanes and lane_remote_engines:
                await asyncio.gather(
                    *(
                        _invoke_bound_remote_engine_async(
                            lane_remote_engines[lane_index],
                            {
                                **cleanup_payload,
                                "component_id": (
                                    f"{payload.get('component_id', 'modal-subgraph')}::cleanup:{lane_index}"
                                ),
                            },
                            serialize_node_inputs({}),
                        )
                        for lane_index in range(parallelism)
                    )
                )
            else:
                await invoke_remote_engine_async(
                    cleanup_payload,
                    serialize_node_inputs({}),
                )


async def _invoke_mapped_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Execute one mapped payload locally using explicit static and mapped phases."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    return await asyncio.to_thread(
        lambda: serialize_node_outputs(
            _execute_mapped_subgraph_payload(
                payload,
                hydrated_inputs,
            )
        )
    )


def _lookup_deployed_remote_engine(
    payload: dict[str, Any],
    *,
    affinity_key_override: str | None = None,
) -> Any:
    """Look up the deployed Modal runtime class instance."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    settings = get_settings()
    affinity_key = affinity_key_override
    if affinity_key is None:
        affinity_key = _remote_session_affinity_key(payload)
    logger.info(
        "Attempting deployed Modal invocation for app=%s class=%s component=%s session_affinity=%s.",
        settings.app_name,
        "RemoteEngine",
        payload.get("component_id"),
        affinity_key,
    )
    remote_cls = modal.Cls.from_name(settings.app_name, "RemoteEngine")
    if affinity_key is not None:
        return remote_cls(session_affinity_key=affinity_key)
    return remote_cls()


async def _invoke_bound_remote_engine_async(
    remote_engine: Any,
    payload: dict[str, Any],
    kwargs_payload: bytes,
) -> bytes:
    """Invoke one pre-bound remote engine handle asynchronously with local interrupt mirroring."""
    logger.info(
        "Dispatching async Modal remote invocation via bound engine for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    cancellation_event = threading.Event()
    future = _REMOTE_MODAL_CALL_EXECUTOR.submit(
        _invoke_remote_engine_payload,
        remote_engine,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    wrapped_future = asyncio.wrap_future(future)
    try:
        while True:
            try:
                response = await asyncio.wait_for(asyncio.shield(wrapped_future), timeout=0.1)
                break
            except asyncio.TimeoutError:
                _sync_local_interrupt_to_cancellation_event(payload, cancellation_event)
                continue
    except asyncio.CancelledError:
        cancellation_event.set()
        raise
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising async Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Async Modal remote invocation via bound engine failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Async bound-engine invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Async Modal remote invocation via bound engine completed for component=%s.",
        payload.get("component_id"),
    )
    return response


def _modal_environment_name() -> str | None:
    """Return the active Modal environment name when explicitly configured."""
    environment_name = os.getenv("MODAL_ENVIRONMENT")
    if environment_name is None:
        return None
    normalized = environment_name.strip()
    return normalized or None


def _modal_deploy_cache_key() -> tuple[str, str | None]:
    """Return the cache key for auto-deployed Modal apps."""
    settings = get_settings()
    return (settings.app_name, _modal_environment_name())


def _warmup_prompt_id(warmup_request: dict[str, Any]) -> str | None:
    """Return the prompt id that scopes proactive warmup state."""
    prompt_id = warmup_request.get("prompt_id")
    if prompt_id is None:
        return None
    normalized_prompt_id = str(prompt_id).strip()
    return normalized_prompt_id or None


def _ensure_prompt_warmup_state(prompt_id: str) -> _PromptWarmupState:
    """Return the cached warmup state bucket for one prompt."""
    global _PROMPT_WARMUP_STATE_ORDER

    state = _PROMPT_WARMUP_STATES.get(prompt_id)
    if state is not None:
        return state

    state = _PromptWarmupState()
    _PROMPT_WARMUP_STATES[prompt_id] = state
    if _PROMPT_WARMUP_STATE_ORDER is None:
        _PROMPT_WARMUP_STATE_ORDER = queue.SimpleQueue()
    _PROMPT_WARMUP_STATE_ORDER.put(prompt_id)
    while len(_PROMPT_WARMUP_STATES) > _PROMPT_WARMUP_STATE_CACHE_LIMIT:
        expired_prompt_id = _PROMPT_WARMUP_STATE_ORDER.get()
        _PROMPT_WARMUP_STATES.pop(expired_prompt_id, None)
    return state


def _clamp_prompt_warmup_target(warmup_target: int) -> int:
    """Clamp one proactive warmup target to the configured Modal container cap."""
    normalized_target = max(0, int(warmup_target))
    max_containers = get_settings().max_containers
    if max_containers is not None:
        return min(normalized_target, max_containers)
    return normalized_target


def _build_prompt_warmup_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the prompt-scoped warmup-relevant fields from one payload."""
    return {
        "prompt_id": (
            str(payload.get("prompt_id"))
            if payload.get("prompt_id") is not None
            else None
        ),
        "component_id": str(
            payload.get("mapped_progress_display_node_id", payload.get("component_id", "modal-warmup"))
        ),
        "requires_volume_reload": bool(payload.get("requires_volume_reload", True)),
        "volume_reload_marker": payload.get("volume_reload_marker"),
        "uploaded_volume_paths": list(payload.get("uploaded_volume_paths", [])),
        "custom_nodes_bundle": payload.get("custom_nodes_bundle"),
    }


def _component_parallelism_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the prompt-level parallelism metadata attached by the queue rewrite."""
    extra_data = payload.get("extra_data") or {}
    modal_metadata = extra_data.get("modal") or {}
    if not isinstance(modal_metadata, dict):
        return {}
    return modal_metadata


def _prompt_parallelism_target(
    payload: dict[str, Any],
    *,
    exact_component_parallelism: dict[str, int] | None = None,
) -> int:
    """Return the current best-effort prompt-wide warmup target for one payload."""
    metadata = _component_parallelism_metadata(payload)
    execution_stages = metadata.get("component_execution_stages")
    if not isinstance(execution_stages, list):
        return 1

    mapped_component_ids = {
        str(component_id)
        for component_id in metadata.get("mapped_component_ids", [])
        if str(component_id)
    }
    exact_parallelism = exact_component_parallelism or {}
    stage_parallelism = 0
    for stage in execution_stages:
        if not isinstance(stage, list):
            continue
        current_stage_parallelism = 0
        for component_id_value in stage:
            component_id = str(component_id_value)
            if component_id in exact_parallelism:
                current_stage_parallelism += max(1, int(exact_parallelism[component_id]))
            elif component_id in mapped_component_ids:
                current_stage_parallelism += 1
            else:
                current_stage_parallelism += 1
        stage_parallelism = max(stage_parallelism, current_stage_parallelism)

    fallback_parallelism = metadata.get("estimated_max_parallel_requests")
    if stage_parallelism <= 0 and isinstance(fallback_parallelism, int):
        stage_parallelism = max(stage_parallelism, fallback_parallelism)
    return _clamp_prompt_warmup_target(max(1, stage_parallelism))


def _register_exact_component_parallelism(payload: dict[str, Any], component_parallelism: int) -> int:
    """Record exact mapped-component parallelism and return the refined prompt-wide warmup target."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    component_id = str(payload.get("component_id")) if payload.get("component_id") is not None else None
    if not prompt_id or not component_id:
        return _clamp_prompt_warmup_target(component_parallelism)

    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _ensure_prompt_warmup_state(prompt_id)
        warmup_state.exact_component_parallelism[component_id] = max(1, int(component_parallelism))
        exact_parallelism = dict(warmup_state.exact_component_parallelism)
    return _prompt_parallelism_target(payload, exact_component_parallelism=exact_parallelism)


def _warmup_slot_payload(warmup_request: dict[str, Any], slot_index: int) -> dict[str, Any]:
    """Return the remote warmup payload for one desired container slot."""
    slot_payload = copy.deepcopy(warmup_request)
    component_id = str(slot_payload.get("component_id") or slot_payload.get("prompt_id") or "modal-warmup")
    slot_payload["component_id"] = f"{component_id}::warmup:{slot_index}"
    slot_payload["warmup_slot_index"] = int(slot_index)
    slot_payload["warmup_only"] = True
    return slot_payload


def _invoke_remote_engine_warmup(remote_engine: Any, warmup_request: dict[str, Any]) -> Any:
    """Ask one prepared remote engine instance to warm a container for a prompt."""
    warmup_method = getattr(remote_engine, "warmup_for_request", None)
    if warmup_method is None:
        logger.warning(
            "Remote warmup method is unavailable for prompt=%s component=%s; skipping proactive warmup.",
            warmup_request.get("prompt_id"),
            warmup_request.get("component_id"),
        )
        return None
    if hasattr(warmup_method, "remote"):
        return warmup_method.remote(warmup_request)
    return warmup_method(warmup_request)


def _invoke_modal_warmup_blocking(warmup_request: dict[str, Any]) -> Any:
    """Warm one Modal container slot using deployed or ephemeral app state."""
    if modal is None:
        return None

    lookup_error_types = _modal_lookup_error_types()
    settings = get_settings()
    if lookup_error_types:
        try:
            remote_engine = _lookup_deployed_remote_engine(warmup_request)
            return _invoke_remote_engine_warmup(remote_engine, warmup_request)
        except lookup_error_types as exc:
            if settings.auto_deploy:
                _auto_deploy_modal_app(warmup_request, exc)
                try:
                    remote_engine = _lookup_deployed_remote_engine(warmup_request)
                    return _invoke_remote_engine_warmup(remote_engine, warmup_request)
                except lookup_error_types as retry_exc:
                    exc = retry_exc
            if not settings.allow_ephemeral_fallback:
                raise ModalRemoteInvocationError(
                    "Proactive Modal warmup requires a deployed Modal app or a successful first-run auto-deploy. "
                    f"Lookup failed for app={settings.app_name!r}: {exc}."
                ) from exc
    else:
        remote_engine = _lookup_deployed_remote_engine(warmup_request)
        return _invoke_remote_engine_warmup(remote_engine, warmup_request)

    cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    cloud_remote_engine = getattr(cloud_module, "RemoteEngine", None)
    if cloud_app is None or cloud_remote_engine is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose app and RemoteEngine."
        )
    run_context = cloud_app.run() if hasattr(cloud_app, "run") else nullcontext()
    with run_context:
        remote_engine = cloud_remote_engine()
        return _invoke_remote_engine_warmup(remote_engine, warmup_request)


def _run_prompt_warmup_slot(
    prompt_id: str,
    slot_index: int,
    warmup_request: dict[str, Any],
    reason: str,
) -> None:
    """Execute one proactive warmup slot and release it for retry on failure."""
    try:
        logger.info(
            "Starting proactive Modal warmup for prompt=%s slot=%d component=%s reason=%s.",
            prompt_id,
            slot_index,
            warmup_request.get("component_id"),
            reason,
        )
        _invoke_modal_warmup_blocking(_warmup_slot_payload(warmup_request, slot_index))
    except Exception:
        with _PROMPT_WARMUP_STATES_LOCK:
            warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
            if warmup_state is not None:
                warmup_state.scheduled_slots.discard(slot_index)
        logger.exception(
            "Proactive Modal warmup failed for prompt=%s slot=%d component=%s.",
            prompt_id,
            slot_index,
            warmup_request.get("component_id"),
        )


def ensure_remote_warm_capacity(
    warmup_request: dict[str, Any],
    *,
    warmup_target: int,
    reason: str,
) -> int:
    """Best-effort background warmup so enough Modal containers are ready for one prompt."""
    settings = get_settings()
    if not settings.enable_proactive_warmup:
        return 0
    if settings.execution_mode == "local" or modal is None:
        return 0

    prompt_id = _warmup_prompt_id(warmup_request)
    if prompt_id is None:
        return 0

    clamped_target = _clamp_prompt_warmup_target(warmup_target)
    if clamped_target <= 0:
        return 0

    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _ensure_prompt_warmup_state(prompt_id)
        missing_slots = [
            slot_index
            for slot_index in range(clamped_target)
            if slot_index not in warmup_state.scheduled_slots
        ]
        for slot_index in missing_slots:
            warmup_state.scheduled_slots.add(slot_index)

    if not missing_slots:
        return clamped_target

    logger.info(
        "Scheduling proactive Modal warmup for prompt=%s target=%d missing_slots=%s component=%s reason=%s.",
        prompt_id,
        clamped_target,
        missing_slots,
        warmup_request.get("component_id"),
        reason,
    )
    for slot_index in missing_slots:
        _REMOTE_MODAL_WARMUP_EXECUTOR.submit(
            _run_prompt_warmup_slot,
            prompt_id,
            slot_index,
            copy.deepcopy(warmup_request),
            reason,
        )
    return clamped_target


def _auto_deploy_modal_app(payload: dict[str, Any], lookup_error: BaseException) -> None:
    """Deploy the stable Modal cloud app once when deployed lookup fails."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    settings = get_settings()
    deploy_key = _modal_deploy_cache_key()
    cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    if cloud_app is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose a deployable app."
        )

    with _MODAL_AUTO_DEPLOY_LOCK:
        if deploy_key in _MODAL_AUTO_DEPLOYED_APPS:
            if _is_missing_modal_deployment_error(lookup_error):
                logger.warning(
                    "Discarding stale auto-deploy cache entry for app=%s env=%s after missing deployment lookup failure: %s",
                    settings.app_name,
                    deploy_key[1] or "<default>",
                    lookup_error,
                )
                _MODAL_AUTO_DEPLOYED_APPS.discard(deploy_key)
            else:
                logger.info(
                    "Auto-deploy already completed for app=%s env=%s; reusing cached deployment state.",
                    settings.app_name,
                    deploy_key[1] or "<default>",
                )
                return

        logger.warning(
            "Deployed Modal app lookup failed for app=%s component=%s: %s. "
            "Attempting first-run auto-deploy from the custom node.",
            settings.app_name,
            payload.get("component_id"),
            lookup_error,
        )
        output_context = modal.enable_output() if hasattr(modal, "enable_output") else nullcontext()
        deploy_started_at = time.perf_counter()
        with output_context:
            cloud_app.deploy(
                name=settings.app_name,
                environment_name=deploy_key[1],
            )
        _MODAL_AUTO_DEPLOYED_APPS.add(deploy_key)
        logger.info(
            "Auto-deployed Modal app %s for env=%s in %.3fs.",
            settings.app_name,
            deploy_key[1] or "<default>",
            time.perf_counter() - deploy_started_at,
        )


def _build_remote_interrupt_callback(_remote_engine: Any, payload: dict[str, Any]) -> Callable[[], Any] | None:
    """Return a callable that requests interruption for one active Modal payload."""
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return None
    return lambda: _request_remote_interrupt(payload)


def _invoke_remote_engine_payload(
    remote_engine: Any,
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Invoke one prepared remote engine instance with optional progress streaming."""
    stream_method = getattr(remote_engine, "execute_payload_stream", None)
    interrupt_remote_call = _build_remote_interrupt_callback(remote_engine, payload)
    if _should_stream_remote_progress(payload) and hasattr(stream_method, "remote_gen"):
        logger.info(
            "Using streamed Modal progress path for component=%s via execute_payload_stream.remote_gen(...).",
            payload.get("component_id"),
        )
        return _invoke_remote_call_with_interrupts(
            payload=payload,
            invoke_remote_call=lambda: _consume_remote_payload_stream(
                payload,
                stream_method.remote_gen(payload, kwargs_payload),
            ),
            interrupt_remote_call=interrupt_remote_call,
            cancellation_event=cancellation_event,
        )

    if _should_stream_remote_progress(payload):
        logger.warning(
            "Streamed Modal progress is unavailable for component=%s; falling back to execute_payload.remote(...).",
            payload.get("component_id"),
        )
    return _invoke_remote_call_with_interrupts(
        payload=payload,
        invoke_remote_call=lambda: remote_engine.execute_payload.remote(payload, kwargs_payload),
        interrupt_remote_call=interrupt_remote_call,
        cancellation_event=cancellation_event,
    )


def _invoke_modal_payload_blocking(
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None = None,
) -> bytes:
    """Invoke the Modal runtime from a worker thread using deployed or ephemeral app state."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    lookup_error_types = _modal_lookup_error_types()
    settings = get_settings()
    if lookup_error_types:
        try:
            remote_engine = _lookup_deployed_remote_engine(payload)
            logger.info(
                "Using deployed Modal app %s for component %s.",
                settings.app_name,
                payload.get("component_id"),
            )
            return _invoke_remote_engine_payload(
                remote_engine,
                payload,
                kwargs_payload,
                cancellation_event,
            )
        except lookup_error_types as exc:
            if settings.auto_deploy:
                _auto_deploy_modal_app(payload, exc)
                try:
                    remote_engine = _lookup_deployed_remote_engine(payload)
                    logger.info(
                        "Using auto-deployed Modal app %s for component %s.",
                        settings.app_name,
                        payload.get("component_id"),
                    )
                    return _invoke_remote_engine_payload(
                        remote_engine,
                        payload,
                        kwargs_payload,
                        cancellation_event,
                    )
                except lookup_error_types as retry_exc:
                    exc = retry_exc
            if not settings.allow_ephemeral_fallback:
                raise ModalRemoteInvocationError(
                    "Remote execution requires a deployed Modal app or a successful first-run auto-deploy. "
                    f"Lookup failed for app={settings.app_name!r} component={payload.get('component_id')!r}: {exc}. "
                    "Ensure Modal credentials are configured so the custom node can auto-deploy, "
                    "or set COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true to allow slow ephemeral app.run() fallback behavior."
                ) from exc
            logger.warning(
                "Deployed Modal app lookup failed for app=%s component=%s: %s. Falling back to ephemeral app.run(); this creates a temporary Modal app session, not a persistent deployment or endpoint.",
                settings.app_name,
                payload.get("component_id"),
                exc,
            )
    else:
        remote_engine = _lookup_deployed_remote_engine(payload)
        logger.info(
            "Using deployed Modal app %s for component %s.",
            settings.app_name,
            payload.get("component_id"),
        )
        return _invoke_remote_engine_payload(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )

    if "app" not in globals() or "RemoteEngine" not in globals():
        logger.debug("Local module Modal runtime objects are unavailable; loading stable cloud entry module.")

    cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    cloud_remote_engine = getattr(cloud_module, "RemoteEngine", None)
    if cloud_app is None or cloud_remote_engine is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose app and RemoteEngine."
        )
    logger.info(
        "Starting ephemeral Modal app.run() for component %s. This does not create a persistent deployed app or web endpoint.",
        payload.get("component_id"),
    )
    run_context = cloud_app.run() if hasattr(cloud_app, "run") else nullcontext()
    with run_context:
        remote_engine = cloud_remote_engine()
        result = _invoke_remote_engine_payload(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )
    logger.info(
        "Ephemeral Modal app.run() invocation completed for component %s.",
        payload.get("component_id"),
    )
    return result


def invoke_remote_engine(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Invoke Modal when configured, or fall back to local in-process execution."""
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    if payload.get("payload_kind") == "mapped_subgraph":
        if execution_mode == "local" or modal is None:
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
            return serialize_node_outputs(
                _execute_mapped_subgraph_payload(payload, hydrated_inputs)
            )
    if payload.get("payload_kind") == "subgraph":
        hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        if _split_batch_boundary_inputs(payload, hydrated_inputs) is not None:
            return asyncio.run(_invoke_implicitly_mapped_subgraph_async(payload, kwargs_payload))
    if execution_mode == "local" or modal is None:
        if payload.get("payload_kind") == "subgraph":
            return execute_subgraph_locally(payload, kwargs_payload)
        return execute_node_locally(payload, kwargs_payload)

    logger.info(
        "Dispatching Modal remote invocation for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    cancellation_event = threading.Event()
    future = _REMOTE_MODAL_CALL_EXECUTOR.submit(
        _invoke_modal_payload_blocking,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    try:
        while True:
            try:
                response = future.result(timeout=0.1)
                break
            except FutureTimeoutError:
                _sync_local_interrupt_to_cancellation_event(payload, cancellation_event)
                continue
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Modal remote invocation failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Remote invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Modal remote invocation completed for component=%s.",
        payload.get("component_id"),
    )
    return response


async def invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Invoke Modal asynchronously so multiple proxy nodes can wait on remote work in parallel."""
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    if payload.get("payload_kind") == "mapped_subgraph":
        if execution_mode == "local" or modal is None:
            return await _invoke_mapped_remote_engine_async(payload, kwargs_payload)
    if payload.get("payload_kind") == "subgraph":
        hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        if _split_batch_boundary_inputs(payload, hydrated_inputs) is not None:
            return await _invoke_implicitly_mapped_subgraph_async(payload, kwargs_payload)
    if execution_mode == "local" or modal is None:
        return await asyncio.to_thread(invoke_remote_engine, payload, kwargs_payload)

    logger.info(
        "Dispatching async Modal remote invocation for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    cancellation_event = threading.Event()
    future = _REMOTE_MODAL_CALL_EXECUTOR.submit(
        _invoke_modal_payload_blocking,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    wrapped_future = asyncio.wrap_future(future)
    try:
        while True:
            try:
                response = await asyncio.wait_for(asyncio.shield(wrapped_future), timeout=0.1)
                break
            except asyncio.TimeoutError:
                _sync_local_interrupt_to_cancellation_event(payload, cancellation_event)
                continue
    except asyncio.CancelledError:
        cancellation_event.set()
        raise
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising async Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Async Modal remote invocation failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Async remote invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Async Modal remote invocation completed for component=%s.",
        payload.get("component_id"),
    )
    return response


if modal is not None:  # pragma: no branch - simple import-time configuration.
    settings = get_settings()
    app = modal.App(settings.app_name)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    image = modal.Image.debian_slim().pip_install("torch", "safetensors", "pillow", "numpy")

    @app.cls(
        gpu="A100",
        volumes={settings.remote_storage_root: vol},
        scaledown_window=60,
        image=image,
    )
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""
        session_affinity_key: str = modal.parameter(default="")

        @modal.enter()
        def setup(self) -> None:
            """Prepare the container process for headless node execution."""
            logger.info(
                "RemoteEngine setup complete for session_affinity_key=%s.",
                self.session_affinity_key or None,
            )

        @modal.method()
        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            if payload.get("payload_kind") == "mapped_subgraph":
                hydrated_inputs = deserialize_node_inputs(kwargs_payload)
                return serialize_node_outputs(
                    _execute_mapped_subgraph_payload(payload, hydrated_inputs)
                )
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)

        @modal.method()
        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """No-op local warmup entrypoint for the simplified Modal runtime."""
            return {"component_id": str(payload.get("component_id") or "modal-warmup")}

else:

    class RemoteEngine:
        """Local fallback runtime used when the Modal SDK is unavailable."""

        def __init__(self, session_affinity_key: str | None = None) -> None:
            """Record the optional session-affinity key for split-proxy reuse."""
            self.session_affinity_key = session_affinity_key

        def setup(self) -> None:
            """No-op setup for local fallback execution."""

        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute the proxied node or subgraph locally."""
            if payload.get("payload_kind") == "mapped_subgraph":
                hydrated_inputs = deserialize_node_inputs(kwargs_payload)
                return serialize_node_outputs(
                    _execute_mapped_subgraph_payload(payload, hydrated_inputs)
                )
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)

        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Return a local no-op warmup result when Modal is unavailable."""
            return {"component_id": str(payload.get("component_id") or "modal-warmup")}
