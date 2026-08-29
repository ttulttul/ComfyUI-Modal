"""Bounded progress streaming for cloud payload execution."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import os
import queue
import threading
import time
from typing import Any, Iterator, Mapping

try:
    from .cloud_comfy_bootstrap import (
        _ensure_comfy_runtime_initialized,
        _extract_custom_nodes_bundle,
    )
    from .cloud_durable_invocation import (
        _begin_remote_invocation,
        _complete_remote_invocation,
        _durable_object_store,
        _execute_canary_payload,
        _execute_payload_with_output_capture,
        _fail_remote_invocation,
    )
    from .cloud_prompt_execution import (
        _execute_mapped_subgraph_payload,
        execute_node_locally,
        execute_subgraph_locally,
    )
    from .durable_state import DurableObjectCommitBatch, RemoteInvocationRecord
    from .serialization import (
        coerce_serialized_node_outputs,
        deserialize_node_inputs,
        serialize_mapping,
        serialize_node_outputs,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_comfy_bootstrap import (
        _ensure_comfy_runtime_initialized,
        _extract_custom_nodes_bundle,
    )
    from cloud_durable_invocation import (
        _begin_remote_invocation,
        _complete_remote_invocation,
        _durable_object_store,
        _execute_canary_payload,
        _execute_payload_with_output_capture,
        _fail_remote_invocation,
    )
    from cloud_prompt_execution import (
        _execute_mapped_subgraph_payload,
        execute_node_locally,
        execute_subgraph_locally,
    )
    from durable_state import DurableObjectCommitBatch, RemoteInvocationRecord
    from serialization import (
        coerce_serialized_node_outputs,
        deserialize_node_inputs,
        serialize_mapping,
        serialize_node_outputs,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)

_REMOTE_INVOCATION_ABANDON_JOIN_SECONDS = 0.5


@dataclass(frozen=True)
class CloudStreamingErrors:
    """Stable exception identities supplied by the cloud entrypoint."""

    invocation_abandoned: type[RuntimeError]


_STREAMING_ERRORS: CloudStreamingErrors | None = None
RemoteInvocationAbandonedError: type[RuntimeError] = RuntimeError


def configure_cloud_streaming_errors(errors: CloudStreamingErrors) -> None:
    """Install stable exception identities without importing upward."""
    global _STREAMING_ERRORS
    global RemoteInvocationAbandonedError
    _STREAMING_ERRORS = errors
    RemoteInvocationAbandonedError = errors.invocation_abandoned


class _BoundedStreamEventBuffer:
    """Bound progress memory while preserving terminal stream events."""

    def __init__(self, maxsize: int) -> None:
        """Initialize a bounded event queue and close signal."""
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max(4, maxsize))
        self._closed = threading.Event()
        self._dropped_progress_events = 0
        self._dropped_lock = threading.Lock()

    @property
    def dropped_progress_events(self) -> int:
        """Return how many stale progress events were coalesced away."""
        with self._dropped_lock:
            return self._dropped_progress_events

    @property
    def queue_size(self) -> int:
        """Return the approximate number of currently buffered events."""
        return self._queue.qsize()

    def publish_progress(self, payload: Any) -> None:
        """Publish the newest progress event without exceeding the queue bound."""
        event = ("progress", payload)
        while not self._closed.is_set():
            try:
                self._queue.put_nowait(event)
                return
            except queue.Full:
                try:
                    discarded_event = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if discarded_event[0] != "progress":
                    self._queue.put_nowait(discarded_event)
                    return
                with self._dropped_lock:
                    self._dropped_progress_events += 1

    def publish_terminal(self, event_kind: str, payload: Any) -> bool:
        """Publish a result, error, or completion unless the consumer closed."""
        while not self._closed.is_set():
            try:
                self._queue.put((event_kind, payload), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def get(self) -> tuple[str, Any]:
        """Wait for and return the next buffered stream event."""
        return self._queue.get()

    def close(self) -> None:
        """Release any producer waiting after the consumer stops."""
        self._closed.set()


def _abandon_streamed_remote_invocation(
    *,
    payload: Mapping[str, Any],
    running_record: RemoteInvocationRecord | None,
    cancellation_event: threading.Event | None,
    event_buffer: _BoundedStreamEventBuffer,
    worker_thread: threading.Thread,
) -> None:
    """Cancel unfinished compute and make its invocation record retryable."""
    component_id = str(payload.get("component_id") or "payload")
    if cancellation_event is not None:
        cancellation_event.set()
    event_buffer.close()
    worker_thread.join(timeout=max(0.0, _REMOTE_INVOCATION_ABANDON_JOIN_SECONDS))
    abandoned_error = RemoteInvocationAbandonedError(
        f"Remote invocation stream for component {component_id!r} closed before completion."
    )
    if running_record is not None:
        _fail_remote_invocation(running_record, abandoned_error)
    logger.warning(
        "Abandoned remote invocation stream component=%s invocation_id=%s worker_stopped=%s; the invocation is retryable.",
        component_id,
        running_record.invocation_id if running_record is not None else "none",
        not worker_thread.is_alive(),
    )


def _stream_remote_payload_events(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield progress and result events for one remote payload execution."""
    event_buffer = _BoundedStreamEventBuffer(get_settings().stream_event_queue_maxsize)
    task_id = os.getenv("MODAL_TASK_ID")
    component_id = str(payload.get("component_id") or "payload")
    invocation_id = str(payload.get("invocation_id") or "none")

    def publish_status(progress_state: dict[str, Any]) -> None:
        """Queue a progress envelope for the remote caller."""
        event_buffer.publish_progress(serialize_mapping(progress_state))

    def execute_once() -> bytes:
        """Run the underlying payload once and return serialized outputs."""
        if payload.get("payload_kind") == "canary":
            return _execute_canary_payload(
                payload,
                kwargs_payload,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            )
        if payload.get("payload_kind") == "mapped_subgraph":
            custom_nodes_root = _extract_custom_nodes_bundle(
                payload.get("custom_nodes_bundle")
            )
            _ensure_comfy_runtime_initialized(custom_nodes_root)
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
            return serialize_node_outputs(
                _execute_mapped_subgraph_payload(
                    payload,
                    hydrated_inputs,
                    custom_nodes_root,
                    status_callback=publish_status,
                    cancellation_event=cancellation_event,
                    interrupt_store=interrupt_store,
                    interrupt_flag_key=interrupt_flag_key,
                )
            )
        if payload.get("payload_kind") == "subgraph":
            execute_subgraph_kwargs: dict[str, Any] = {
                "status_callback": publish_status
            }
            if (
                "cancellation_event"
                in inspect.signature(execute_subgraph_locally).parameters
            ):
                execute_subgraph_kwargs["cancellation_event"] = cancellation_event
            if (
                "interrupt_store"
                in inspect.signature(execute_subgraph_locally).parameters
            ):
                execute_subgraph_kwargs["interrupt_store"] = interrupt_store
            if (
                "interrupt_flag_key"
                in inspect.signature(execute_subgraph_locally).parameters
            ):
                execute_subgraph_kwargs["interrupt_flag_key"] = interrupt_flag_key
            return execute_subgraph_locally(
                payload,
                kwargs_payload,
                **execute_subgraph_kwargs,
            )
        execute_node_kwargs: dict[str, Any] = {}
        if "cancellation_event" in inspect.signature(execute_node_locally).parameters:
            execute_node_kwargs["cancellation_event"] = cancellation_event
        if "interrupt_store" in inspect.signature(execute_node_locally).parameters:
            execute_node_kwargs["interrupt_store"] = interrupt_store
        if "interrupt_flag_key" in inspect.signature(execute_node_locally).parameters:
            execute_node_kwargs["interrupt_flag_key"] = interrupt_flag_key
        return execute_node_locally(payload, kwargs_payload, **execute_node_kwargs)

    object_store = _durable_object_store()

    def execute_payload() -> None:
        """Run compute in a worker thread while deferring Modal volume commits."""
        pending_batch: DurableObjectCommitBatch | None = None
        try:
            with object_store.batch_commits(commit_on_exit=False) as pending_batch:
                outputs = _execute_payload_with_output_capture(payload, execute_once)
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised through generator consumer tests.
            event_buffer.publish_terminal("error", (exc, pending_batch))
        else:
            result_bytes = len(outputs)
            pending_object_write = bool(pending_batch and pending_batch.wrote_object)
            logger.info(
                "Remote stream worker produced result component=%s invocation_id=%s task_id=%s "
                "result_bytes=%d pending_object_write=%s buffer_queue_size=%d.",
                component_id,
                invocation_id,
                task_id or "none",
                result_bytes,
                pending_object_write,
                event_buffer.queue_size,
            )
            publish_started_at = time.monotonic()
            published = event_buffer.publish_terminal("result", (outputs, pending_batch))
            logger.info(
                "Finished publishing remote stream result to event buffer in %.3fs component=%s "
                "invocation_id=%s task_id=%s published=%s result_bytes=%d buffer_queue_size=%d.",
                time.monotonic() - publish_started_at,
                component_id,
                invocation_id,
                task_id or "none",
                published,
                result_bytes,
                event_buffer.queue_size,
            )
        finally:
            event_buffer.publish_terminal("done", None)

    worker_thread = threading.Thread(
        target=execute_payload,
        name=f"modal-stream-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    if task_id:
        yield {"kind": "remote_logs", "task_id": task_id}
    running_record, completed_result = _begin_remote_invocation(payload)
    if completed_result is not None:
        yield {"kind": "result", "outputs": completed_result}
        return
    worker_thread.start()
    invocation_finalized = False
    try:
        while True:
            event_kind, event_payload = event_buffer.get()
            if event_kind == "progress":
                yield {"kind": "progress", **event_payload}
                continue
            if event_kind == "result":
                outputs, pending_batch = event_payload
                serialized_outputs = coerce_serialized_node_outputs(outputs)
                logger.info(
                    "Remote stream consumer received buffered result component=%s invocation_id=%s "
                    "task_id=%s result_bytes=%d pending_object_write=%s buffer_queue_size=%d.",
                    component_id,
                    invocation_id,
                    task_id or "none",
                    len(serialized_outputs),
                    bool(pending_batch and pending_batch.wrote_object),
                    event_buffer.queue_size,
                )
                if running_record is not None:
                    _complete_remote_invocation(
                        running_record,
                        serialized_outputs,
                        pending_batch=pending_batch,
                    )
                elif pending_batch is not None:
                    object_store.commit_batch(pending_batch)
                invocation_finalized = True
                yield_started_at = time.monotonic()
                logger.info(
                    "Yielding remote stream result to Modal transport component=%s invocation_id=%s "
                    "task_id=%s result_bytes=%d buffer_queue_size=%d.",
                    component_id,
                    invocation_id,
                    task_id or "none",
                    len(serialized_outputs),
                    event_buffer.queue_size,
                )
                try:
                    yield {"kind": "result", "outputs": serialized_outputs}
                finally:
                    logger.info(
                        "Remote stream result yield released after %.3fs component=%s invocation_id=%s "
                        "task_id=%s result_bytes=%d buffer_queue_size=%d.",
                        time.monotonic() - yield_started_at,
                        component_id,
                        invocation_id,
                        task_id or "none",
                        len(serialized_outputs),
                        event_buffer.queue_size,
                    )
                continue
            if event_kind == "error":
                error, pending_batch = event_payload
                if pending_batch is not None:
                    object_store.commit_batch(pending_batch)
                if running_record is not None:
                    _fail_remote_invocation(running_record, error)
                invocation_finalized = True
                raise error
            if event_kind == "done":
                return
    finally:
        if invocation_finalized:
            event_buffer.close()
            worker_thread.join(timeout=1.0)
        else:
            _abandon_streamed_remote_invocation(
                payload=payload,
                running_record=running_record,
                cancellation_event=cancellation_event,
                event_buffer=event_buffer,
                worker_thread=worker_thread,
            )
        if event_buffer.dropped_progress_events:
            logger.info(
                "Coalesced %d stale remote progress event(s) for component=%s to keep the stream buffer bounded.",
                event_buffer.dropped_progress_events,
                payload.get("component_id"),
            )

