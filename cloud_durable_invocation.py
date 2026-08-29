"""Durable Modal invocation lifecycle, output capture, and live canaries."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

try:
    from .cloud_runtime_context import invocation_record_store, volume_store
    from .durable_state import (
        DurableObjectCommitBatch,
        DurableStateError,
        FileDurableObjectStore,
        InMemoryRemoteInvocationStore,
        RemoteInvocationRecord,
        new_running_invocation_record,
        read_modal_volume_file,
    )
    from .output_artifacts import (
        RemoteOutputSnapshot,
        capture_execution_result,
        snapshot_output_directory,
    )
    from .serialization import (
        coerce_serialized_node_outputs,
        deserialize_node_inputs,
        serialize_node_outputs,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_runtime_context import invocation_record_store, volume_store
    from durable_state import (
        DurableObjectCommitBatch,
        DurableStateError,
        FileDurableObjectStore,
        InMemoryRemoteInvocationStore,
        RemoteInvocationRecord,
        new_running_invocation_record,
        read_modal_volume_file,
    )
    from output_artifacts import (
        RemoteOutputSnapshot,
        capture_execution_result,
        snapshot_output_directory,
    )
    from serialization import (
        coerce_serialized_node_outputs,
        deserialize_node_inputs,
        serialize_node_outputs,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)

_REMOTE_COMFYUI_ROOT = Path("/root/comfyui_src")
_REMOTE_INVOCATION_STORE = InMemoryRemoteInvocationStore()
_REMOTE_INVOCATION_RETRY_WAIT_SECONDS = 5.0
_REMOTE_INVOCATION_RETRY_POLL_SECONDS = 0.1
_DURABLE_OBJECT_STORE_LOCK = threading.Lock()
_DURABLE_OBJECT_STORE: FileDurableObjectStore | None = None


@dataclass(frozen=True)
class DurableInvocationErrors:
    """Stable entrypoint exception factories used by extracted invocation code."""

    invocation_in_progress: type[RuntimeError]
    canary_interrupted: type[RuntimeError]
    canary_barrier_timeout: type[TimeoutError]


_DURABLE_INVOCATION_ERRORS = DurableInvocationErrors(
    invocation_in_progress=RuntimeError,
    canary_interrupted=RuntimeError,
    canary_barrier_timeout=TimeoutError,
)


def configure_durable_invocation_errors(errors: DurableInvocationErrors) -> None:
    """Install stable cloud-entrypoint exception types for invocation failures."""
    global _DURABLE_INVOCATION_ERRORS
    _DURABLE_INVOCATION_ERRORS = errors


def _durable_invocation_errors() -> DurableInvocationErrors:
    """Return the configured stable exception factories."""
    return _DURABLE_INVOCATION_ERRORS


def _is_modal_container_runtime() -> bool:
    """Return whether this process is executing inside a Modal container."""
    return os.getenv("MODAL_IS_REMOTE") == "1" or bool(os.getenv("MODAL_TASK_ID"))


def _durable_object_store() -> FileDurableObjectStore:
    """Return the process-local handle for volume-backed durable binary objects."""
    global _DURABLE_OBJECT_STORE

    with _DURABLE_OBJECT_STORE_LOCK:
        if _DURABLE_OBJECT_STORE is not None:
            return _DURABLE_OBJECT_STORE
        settings = get_settings()
        commit_callback: Callable[[], Any] | None = None
        committed_read_callback: Callable[[str], bytes] | None = None
        if _is_modal_container_runtime():
            object_root = Path(settings.remote_storage_root) / "durable_objects"
            volume = volume_store()
            volume_commit = getattr(volume, "commit", None)
            if callable(volume_commit):
                commit_callback = volume_commit
            if callable(getattr(volume, "read_file", None)):

                def read_committed_object(object_path: str) -> bytes:
                    """Read one durable object without reloading the mounted volume."""
                    volume_path = (Path("durable_objects") / object_path).as_posix()
                    return read_modal_volume_file(volume, volume_path)

                committed_read_callback = read_committed_object
        else:
            object_root = settings.local_storage_root / "durable_objects"
        _DURABLE_OBJECT_STORE = FileDurableObjectStore(
            object_root,
            commit_callback=commit_callback,
            committed_read_callback=committed_read_callback,
        )
        return _DURABLE_OBJECT_STORE


def _invocation_record_store() -> Any:
    """Return the shared lifecycle store for idempotent remote invocations."""
    shared_store = invocation_record_store()
    return shared_store if shared_store is not None else _REMOTE_INVOCATION_STORE


def _load_remote_invocation_record(
    invocation_id: str,
) -> RemoteInvocationRecord | None:
    """Load one invocation record from the configured shared store."""
    store = _invocation_record_store()
    get_record = getattr(store, "get_record", None)
    if callable(get_record):
        return get_record(invocation_id)
    payload = store.get(invocation_id)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DurableStateError(
            f"Remote invocation record {invocation_id!r} is not a mapping."
        )
    return RemoteInvocationRecord.from_payload(payload)


def _store_remote_invocation_record(record: RemoteInvocationRecord) -> None:
    """Persist one invocation lifecycle update to the configured shared store."""
    store = _invocation_record_store()
    inline_result_bytes = len(record.result_inline or b"")
    result_object_bytes = (
        record.result_object.size_bytes if record.result_object is not None else 0
    )
    store_type = f"{type(store).__module__}.{type(store).__qualname__}"
    started_at = time.monotonic()
    logger.info(
        "Starting remote invocation record write invocation_id=%s state=%s attempt=%d "
        "store_type=%s inline_result_bytes=%d result_object_bytes=%d.",
        record.invocation_id,
        record.state,
        record.attempt,
        store_type,
        inline_result_bytes,
        result_object_bytes,
    )
    put_record = getattr(store, "put_record", None)
    if callable(put_record):
        put_record(record)
    else:
        store[record.invocation_id] = record.to_payload()
    logger.info(
        "Finished remote invocation record write in %.3fs invocation_id=%s state=%s "
        "attempt=%d store_type=%s inline_result_bytes=%d result_object_bytes=%d.",
        time.monotonic() - started_at,
        record.invocation_id,
        record.state,
        record.attempt,
        store_type,
        inline_result_bytes,
        result_object_bytes,
    )


def _load_completed_remote_invocation_result(record: RemoteInvocationRecord) -> bytes:
    """Load the validated result for one completed invocation record."""
    if record.result_inline is not None:
        return record.result_inline
    if record.result_object is not None:
        return _durable_object_store().get(record.result_object)
    raise DurableStateError(
        f"Completed remote invocation {record.invocation_id!r} has no result."
    )


def _wait_for_running_remote_invocation(
    running_record: RemoteInvocationRecord,
) -> RemoteInvocationRecord | None:
    """Wait briefly for an overlapping attempt to publish a terminal state."""
    wait_seconds = max(0.0, _REMOTE_INVOCATION_RETRY_WAIT_SECONDS)
    if wait_seconds <= 0.0:
        return running_record

    logger.warning(
        "Waiting up to %.3fs for overlapping remote invocation invocation_id=%s attempt=%d.",
        wait_seconds,
        running_record.invocation_id,
        running_record.attempt,
    )
    deadline = time.monotonic() + wait_seconds
    poll_seconds = max(0.001, _REMOTE_INVOCATION_RETRY_POLL_SECONDS)
    current_record: RemoteInvocationRecord | None = running_record
    while time.monotonic() < deadline:
        current_record = _load_remote_invocation_record(running_record.invocation_id)
        if (
            current_record is None
            or current_record.state != "running"
            or current_record.attempt != running_record.attempt
        ):
            return current_record
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return current_record


def _begin_remote_invocation(
    payload: Mapping[str, Any],
) -> tuple[RemoteInvocationRecord | None, bytes | None]:
    """Start an invocation attempt or return its already-completed result."""
    invocation_id = str(payload.get("invocation_id") or "").strip()
    if not invocation_id:
        return None, None
    previous_record = _load_remote_invocation_record(invocation_id)
    if previous_record is not None and previous_record.state == "completed":
        logger.info(
            "Replaying completed remote invocation invocation_id=%s attempt=%d.",
            invocation_id,
            previous_record.attempt,
        )
        return None, _load_completed_remote_invocation_result(previous_record)
    if previous_record is not None and previous_record.state == "running":
        settings = get_settings()
        stale_after_seconds = (
            settings.execution_timeout_seconds + settings.startup_timeout_seconds
        )
        if time.time() - previous_record.updated_at <= stale_after_seconds:
            previous_record = _wait_for_running_remote_invocation(previous_record)
            if previous_record is not None and previous_record.state == "completed":
                logger.info(
                    "Replaying completed remote invocation invocation_id=%s "
                    "attempt=%d after overlap wait.",
                    invocation_id,
                    previous_record.attempt,
                )
                return None, _load_completed_remote_invocation_result(previous_record)
            if previous_record is not None and previous_record.state == "running":
                raise _durable_invocation_errors().invocation_in_progress(
                    f"Remote invocation {invocation_id!r} is already running "
                    f"(attempt {previous_record.attempt})."
                )
        else:
            logger.warning(
                "Recovering stale remote invocation invocation_id=%s attempt=%d.",
                invocation_id,
                previous_record.attempt,
            )
    running_record = new_running_invocation_record(invocation_id, previous_record)
    _store_remote_invocation_record(running_record)
    return running_record, None


def _complete_remote_invocation(
    running_record: RemoteInvocationRecord,
    serialized_outputs: bytes,
    *,
    pending_batch: DurableObjectCommitBatch | None = None,
) -> None:
    """Commit successful object writes before publishing completed metadata."""
    settings = get_settings()
    completion_started_at = time.monotonic()
    result_bytes = len(serialized_outputs)
    inline_threshold_bytes = settings.invocation_result_inline_max_bytes
    result_storage = (
        "durable_object" if result_bytes > inline_threshold_bytes else "inline"
    )
    pending_object_write = bool(pending_batch and pending_batch.wrote_object)
    logger.info(
        "Starting remote invocation completion invocation_id=%s attempt=%d result_bytes=%d "
        "inline_threshold_bytes=%d result_storage=%s pending_object_write=%s.",
        running_record.invocation_id,
        running_record.attempt,
        result_bytes,
        inline_threshold_bytes,
        result_storage,
        pending_object_write,
    )
    result_inline: bytes | None = serialized_outputs
    result_object = None
    object_store = _durable_object_store()
    durable_started_at = time.monotonic()
    with object_store.batch_commits() as completion_batch:
        if pending_batch is not None:
            completion_batch.absorb(pending_batch)
        if result_storage == "durable_object":
            result_object = object_store.put(
                "invocation_results",
                serialized_outputs,
            )
            result_inline = None
    logger.info(
        "Finished remote invocation durable result preparation in %.3fs invocation_id=%s "
        "attempt=%d result_bytes=%d result_storage=%s pending_object_write=%s.",
        time.monotonic() - durable_started_at,
        running_record.invocation_id,
        running_record.attempt,
        result_bytes,
        result_storage,
        pending_object_write,
    )
    _store_remote_invocation_record(
        RemoteInvocationRecord(
            invocation_id=running_record.invocation_id,
            state="completed",
            attempt=running_record.attempt,
            created_at=running_record.created_at,
            updated_at=time.time(),
            result_inline=result_inline,
            result_object=result_object,
        )
    )
    logger.info(
        "Finished remote invocation completion in %.3fs invocation_id=%s attempt=%d "
        "result_bytes=%d result_storage=%s.",
        time.monotonic() - completion_started_at,
        running_record.invocation_id,
        running_record.attempt,
        result_bytes,
        result_storage,
    )


def _fail_remote_invocation(
    running_record: RemoteInvocationRecord,
    error: Exception,
) -> None:
    """Persist a failed invocation attempt while allowing a later retry."""
    _store_remote_invocation_record(
        RemoteInvocationRecord(
            invocation_id=running_record.invocation_id,
            state="failed",
            attempt=running_record.attempt,
            created_at=running_record.created_at,
            updated_at=time.time(),
            error_type=type(error).__name__,
            error_message=str(error)[:4096],
        )
    )


def _execute_with_durable_invocation(
    payload: Mapping[str, Any],
    execute_once: Callable[[], bytes | bytearray | str | Sequence[Any] | Any],
) -> bytes:
    """Execute once and durably replay the result for duplicate retries."""
    running_record, completed_result = _begin_remote_invocation(payload)
    if completed_result is not None:
        return completed_result
    object_store = _durable_object_store()
    pending_batch: DurableObjectCommitBatch | None = None
    try:
        with object_store.batch_commits(commit_on_exit=False) as pending_batch:
            serialized_outputs = _execute_payload_with_output_capture(
                payload, execute_once
            )
    except Exception as exc:
        if pending_batch is not None:
            object_store.commit_batch(pending_batch)
        if running_record is not None:
            _fail_remote_invocation(running_record, exc)
        raise
    if running_record is not None:
        _complete_remote_invocation(
            running_record,
            serialized_outputs,
            pending_batch=pending_batch,
        )
    elif pending_batch is not None:
        object_store.commit_batch(pending_batch)
    return serialized_outputs


def _remote_comfy_output_directory() -> Path:
    """Return the effective output directory inside the remote ComfyUI runtime."""
    try:
        import folder_paths
    except ModuleNotFoundError as exc:
        if exc.name != "folder_paths":
            raise
        return _REMOTE_COMFYUI_ROOT / "output"
    get_output_directory = getattr(folder_paths, "get_output_directory", None)
    if callable(get_output_directory):
        return Path(get_output_directory()).expanduser().resolve()
    return _REMOTE_COMFYUI_ROOT / "output"


def _remote_output_snapshot(payload: Mapping[str, Any]) -> RemoteOutputSnapshot | None:
    """Snapshot remote outputs when the local client requested artifact collection."""
    if not bool(payload.get("capture_remote_outputs")):
        return None
    output_directory = _remote_comfy_output_directory()
    snapshot = snapshot_output_directory(output_directory)
    logger.info(
        "Snapshot remote ComfyUI output directory %s before component=%s with %d existing file(s).",
        output_directory,
        payload.get("component_id"),
        len(snapshot.files),
    )
    return snapshot


def _execute_payload_with_output_capture(
    payload: Mapping[str, Any],
    execute_once: Callable[[], bytes | bytearray | str | Sequence[Any] | Any],
) -> bytes:
    """Execute one payload and attach files it created beneath remote output."""
    snapshot = _remote_output_snapshot(payload)
    serialized_outputs = coerce_serialized_node_outputs(execute_once())
    if snapshot is None:
        return serialized_outputs
    result = capture_execution_result(serialized_outputs, snapshot)
    if result is serialized_outputs or result == serialized_outputs:
        logger.info(
            "Remote component=%s created no new ComfyUI output files.",
            payload.get("component_id"),
        )
        return serialized_outputs
    logger.info(
        "Bundled remote ComfyUI output files for component=%s result_bytes=%d.",
        payload.get("component_id"),
        len(result),
    )
    return result


def _canary_interrupt_requested(
    *,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
) -> bool:
    """Consume and report one local or shared live-canary interrupt request."""
    if cancellation_event is not None and cancellation_event.is_set():
        return True
    if interrupt_store is None or interrupt_flag_key is None:
        return False
    if not bool(interrupt_store.contains(interrupt_flag_key)):
        return False
    interrupt_store.pop(interrupt_flag_key, None)
    if cancellation_event is not None:
        cancellation_event.set()
    return True


def _raise_if_canary_interrupted(
    *,
    component_id: str,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
) -> None:
    """Raise the canary interruption error when either cancellation source trips."""
    if _canary_interrupt_requested(
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    ):
        raise _durable_invocation_errors().canary_interrupted(
            f"Live Modal canary for component {component_id!r} was interrupted."
        )


def _put_canary_barrier_marker(
    marker_key: str, marker_payload: Mapping[str, Any]
) -> None:
    """Publish one live-canary barrier marker through shared invocation storage."""
    store = _invocation_record_store()
    put_method = getattr(store, "put", None)
    if callable(put_method):
        put_method(marker_key, dict(marker_payload))
        return
    store[marker_key] = dict(marker_payload)


def _canary_barrier_marker_exists(marker_key: str) -> bool:
    """Return whether one live-canary barrier member has reached its worker."""
    store = _invocation_record_store()
    contains_method = getattr(store, "contains", None)
    if callable(contains_method):
        return bool(contains_method(marker_key))
    return store.get(marker_key) is not None


def _canary_barrier_marker_key(barrier_id: str, member_id: str) -> str:
    """Return the shared invocation-store key for one canary barrier member."""
    return f"CANARY_BARRIER:{barrier_id}:{member_id}"


def _wait_for_canary_barrier(
    barrier: Mapping[str, Any],
    *,
    component_id: str,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
) -> float | None:
    """Coordinate multiple live Modal calls and return their release timestamp."""
    barrier_id = str(barrier.get("barrier_id") or "").strip()
    member_id = str(barrier.get("member_id") or "").strip()
    raw_members = barrier.get("members")
    if not barrier_id and not member_id and raw_members is None:
        return None
    if not barrier_id or not member_id or not isinstance(raw_members, list):
        raise ValueError(
            "Live Modal canary barriers require id, member, and member list."
        )
    members = [str(member).strip() for member in raw_members if str(member).strip()]
    if member_id not in members or len(set(members)) != len(members):
        raise ValueError(
            "Live Modal canary barrier members must be unique and include this call."
        )
    timeout_seconds = float(barrier.get("timeout_seconds", 60.0))
    if timeout_seconds <= 0.0 or timeout_seconds > 300.0:
        raise ValueError(
            "Live Modal canary barrier timeout must be within (0, 300] seconds."
        )

    _put_canary_barrier_marker(
        _canary_barrier_marker_key(barrier_id, member_id),
        {
            "component_id": component_id,
            "ready_at": time.time(),
            "task_id": os.getenv("MODAL_TASK_ID"),
        },
    )
    deadline = time.monotonic() + timeout_seconds
    member_keys = [
        _canary_barrier_marker_key(barrier_id, current_member)
        for current_member in members
    ]
    while not all(_canary_barrier_marker_exists(key) for key in member_keys):
        _raise_if_canary_interrupted(
            component_id=component_id,
            cancellation_event=cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key=interrupt_flag_key,
        )
        if time.monotonic() >= deadline:
            raise _durable_invocation_errors().canary_barrier_timeout(
                f"Live Modal canary barrier {barrier_id!r} did not reach "
                f"all {len(member_keys)} members within {timeout_seconds:.1f}s."
            )
        time.sleep(0.05)
    return time.time()


def _execute_canary_payload(
    payload: Mapping[str, Any],
    kwargs_payload: bytes | bytearray | str | Mapping[str, Any],
    *,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
) -> bytes:
    """Execute a dependency-light live canary inside the deployed Modal worker."""
    component_id = str(payload.get("component_id") or "live-canary")
    started_at = time.time()
    barrier = payload.get("canary_barrier")
    barrier_released_at = _wait_for_canary_barrier(
        barrier if isinstance(barrier, Mapping) else {},
        component_id=component_id,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )
    delay_seconds = float(payload.get("canary_delay_seconds", 0.0))
    if delay_seconds < 0.0 or delay_seconds > 300.0:
        raise ValueError("Live Modal canary delay must be within [0, 300] seconds.")
    delay_deadline = time.monotonic() + delay_seconds
    while time.monotonic() < delay_deadline:
        _raise_if_canary_interrupted(
            component_id=component_id,
            cancellation_event=cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key=interrupt_flag_key,
        )
        time.sleep(min(0.05, max(0.0, delay_deadline - time.monotonic())))
    _raise_if_canary_interrupted(
        component_id=component_id,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    normalized_transport = (
        bytes(kwargs_payload) if isinstance(kwargs_payload, bytes | bytearray) else None
    )
    metadata = {
        "barrier_released_at": barrier_released_at,
        "component_id": component_id,
        "finished_at": time.time(),
        "modal_task_id": os.getenv("MODAL_TASK_ID"),
        "started_at": started_at,
        "transport_kind": (
            "binary"
            if normalized_transport is not None
            and normalized_transport.startswith(b"CMODALB1")
            else "legacy"
        ),
    }
    return serialize_node_outputs((hydrated_inputs.get("value"), metadata))
