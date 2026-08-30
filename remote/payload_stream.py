"""Local UI forwarding for streamed Modal payload events."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..durable_state import DurableObjectRef, is_durable_object_ref_payload
from ..serialization import (
    coerce_serialized_node_outputs,
    deserialize_value,
)
from .local_ui_events import (
    RemoteTransferProgressReporter,
    _emit_local_executed_output,
    _emit_local_modal_progress,
    _emit_local_modal_status,
    _emit_local_preview_boundary_output,
    _emit_local_preview_image,
    _progress_stream_event_metadata,
    _remote_execution_destination,
    _remote_execution_identity,
    _remote_prompt_ancestor_node_ids,
    _should_forward_suppressed_stream_event,
)
from .host_session_bridge import materialize_modal_durable_object
from .mapped_execution import _remember_mapped_lane_node_id
from .modal_container_logs import (
    _coerce_modal_task_id,
    _is_remote_container_log_stream_enabled,
    _release_remote_container_log_stream,
    _retain_remote_container_log_stream,
)
from .modal_deployment import ModalRemoteInvocationError
from .modal_warmup import _schedule_speculative_affinity_prewarm

logger = logging.getLogger(__name__)


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





@dataclass
class _PayloadStreamState:
    """Track local forwarding state for one streamed remote invocation."""

    payload: dict[str, Any]
    prompt_id: str | None
    client_id: str | None
    node_ids: list[str]
    modal_gpu: str | None
    suppress_status_stream: bool
    component_id: str
    invocation_id: str
    stream_started_at: float
    previous_event_at: float
    result_payload: bytes | bytearray | None = None
    suppressed_progress_metadata: dict[str, dict[str, str | None]] = field(
        default_factory=dict
    )
    active_remote_task_id: str | None = None
    active_remote_log_task_id: str | None = None
    should_close_stream: bool = False
    event_count: int = 0
    progress_event_count: int = 0
    speculative_prewarm_checked: bool = False


def _new_payload_stream_state(payload: dict[str, Any]) -> _PayloadStreamState:
    """Build initial stream-forwarding state from one remote payload."""
    prompt_id = (
        str(payload["prompt_id"]) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    started_at = time.monotonic()
    return _PayloadStreamState(
        payload=payload,
        prompt_id=prompt_id,
        client_id=(
            str(extra_data["client_id"])
            if extra_data.get("client_id") is not None
            else None
        ),
        node_ids=[str(node_id) for node_id in payload.get("component_node_ids", [])],
        modal_gpu=(
            str(payload["modal_gpu"]) if payload.get("modal_gpu") is not None else None
        ),
        suppress_status_stream=bool(payload.get("suppress_status_stream")),
        component_id=str(payload.get("component_id") or "payload"),
        invocation_id=str(payload.get("invocation_id") or "none"),
        stream_started_at=started_at,
        previous_event_at=started_at,
    )


def _handle_remote_log_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Retain remote task identity and optionally start its container log stream."""
    task_id = _coerce_modal_task_id(stream_event.get("task_id"))
    if task_id is not None and state.active_remote_task_id is None:
        state.active_remote_task_id = task_id
    if (
        task_id is not None
        and state.active_remote_log_task_id is None
        and _is_remote_container_log_stream_enabled()
    ):
        state.active_remote_log_task_id = _retain_remote_container_log_stream(task_id)


def _mapped_progress_lane_id(
    payload: dict[str, Any],
    stream_event: dict[str, Any],
) -> str | None:
    """Return the explicit event lane or the payload's mapped lane fallback."""
    if stream_event.get("lane_id") is not None:
        return str(stream_event["lane_id"])
    if payload.get("mapped_progress_lane_id") is not None:
        return str(payload["mapped_progress_lane_id"])
    return None


def _node_progress_kwargs(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
    *,
    reported_node_id: str,
    display_node_id: str,
    real_node_id: str | None,
    lane_id: str | None,
    aggregate_only: bool,
) -> dict[str, Any]:
    """Build local progress emitter arguments from one remote node event."""
    kwargs: dict[str, Any] = {
        "prompt_id": state.prompt_id,
        "client_id": state.client_id,
        "node_id": reported_node_id,
        "value": float(stream_event.get("value", 0.0)),
        "max_value": float(stream_event.get("max", 1.0)),
        "display_node_id": display_node_id,
        "real_node_id": real_node_id,
        "lane_id": lane_id,
        "clear": bool(stream_event.get("clear", False)),
        "item_index": (
            int(stream_event["item_index"])
            if stream_event.get("item_index") is not None
            else (
                int(state.payload["map_item_index"])
                if state.payload.get("map_item_index") is not None
                else None
            )
        ),
        "aggregate_only": aggregate_only,
    }
    kwargs.update(
        _remote_execution_identity(state.payload, state.active_remote_task_id)
    )
    for field_name in ("stage", "message", "unit"):
        if stream_event.get(field_name) is not None:
            kwargs[field_name] = str(stream_event[field_name])
    if stream_event.get("indeterminate") is not None:
        kwargs["indeterminate"] = bool(stream_event["indeterminate"])
    for field_name in (
        "elapsed_seconds",
        "time_to_first_token_seconds",
        "tokens_per_second",
    ):
        if stream_event.get(field_name) is not None:
            kwargs[field_name] = float(stream_event[field_name])
    completed_ancestors = _remote_prompt_ancestor_node_ids(
        state.payload, real_node_id or display_node_id
    )
    if completed_ancestors:
        kwargs["completed_ancestor_node_ids"] = completed_ancestors
    return kwargs


def _handle_node_progress_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Filter and forward one remote per-node progress event."""
    metadata = _progress_stream_event_metadata(stream_event)
    filter_node_id = metadata["filter_node_id"] if metadata is not None else None
    lane_id = _mapped_progress_lane_id(state.payload, stream_event)
    aggregate_only = bool(stream_event.get("aggregate_only", False))
    if (
        state.suppress_status_stream
        and lane_id is None
        and not aggregate_only
        and not _should_forward_suppressed_stream_event(
            state.payload, filter_node_id
        )
    ):
        logger.debug(
            "Suppressing streamed Modal node progress for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
            state.payload.get("component_id"),
            stream_event.get("node_id"),
            stream_event.get("real_node_id"),
        )
        return
    reported_node_id = metadata["node_id"] if metadata is not None else None
    if reported_node_id is None:
        return
    display_node_id = (
        metadata["display_node_id"] if metadata is not None else str(reported_node_id)
    )
    real_node_id = metadata["real_node_id"] if metadata is not None else None
    progress_node_id = real_node_id or display_node_id
    if lane_id is not None:
        _remember_mapped_lane_node_id(state.payload, lane_id, progress_node_id)
    elif state.suppress_status_stream and not aggregate_only and metadata is not None:
        state.suppressed_progress_metadata[str(metadata["filter_node_id"])] = {
            "node_id": str(reported_node_id),
            "display_node_id": display_node_id,
            "real_node_id": real_node_id,
        }
    logger.debug(
        "Forwarding streamed Modal node progress for component=%s node_id=%s real_node_id=%s value=%s max=%s lane_id=%s.",
        state.payload.get("component_id"),
        reported_node_id,
        real_node_id,
        stream_event.get("value"),
        stream_event.get("max"),
        lane_id,
    )
    _emit_local_modal_progress(
        **_node_progress_kwargs(
            state,
            stream_event,
            reported_node_id=str(reported_node_id),
            display_node_id=display_node_id,
            real_node_id=real_node_id,
            lane_id=lane_id,
            aggregate_only=aggregate_only,
        )
    )


def _handle_executed_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Forward one remote executed-output event when it belongs to this payload."""
    reported_node_id = stream_event.get("node_id")
    if reported_node_id is None:
        return
    if not _should_forward_suppressed_stream_event(
        state.payload, reported_node_id
    ):
        logger.debug(
            "Suppressing streamed Modal executed output for component=%s node_id=%s because it does not belong to this mapped/static payload.",
            state.payload.get("component_id"),
            reported_node_id,
        )
        return
    logger.debug(
        "Forwarding streamed Modal executed output for component=%s node_id=%s.",
        state.payload.get("component_id"),
        reported_node_id,
    )
    _emit_local_executed_output(
        prompt_id=state.prompt_id,
        client_id=state.client_id,
        node_id=str(reported_node_id),
        display_node_id=(
            str(stream_event["display_node_id"])
            if stream_event.get("display_node_id") is not None
            else None
        ),
        output_payload=deserialize_value(stream_event.get("output")),
    )


def _handle_preview_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Forward one streamed preview image when it belongs to this payload."""
    reported_node_id = stream_event.get("node_id")
    image_bytes = deserialize_value(stream_event.get("image_bytes"))
    if reported_node_id is None or not isinstance(image_bytes, bytes):
        return
    if not _should_forward_suppressed_stream_event(
        state.payload, reported_node_id
    ):
        logger.debug(
            "Suppressing streamed Modal preview image for component=%s node_id=%s because it does not belong to this mapped/static payload.",
            state.payload.get("component_id"),
            reported_node_id,
        )
        return
    logger.debug(
        "Forwarding streamed Modal preview image for component=%s node_id=%s.",
        state.payload.get("component_id"),
        reported_node_id,
    )
    _emit_local_preview_image(
        prompt_id=state.prompt_id,
        client_id=state.client_id,
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


def _handle_boundary_output_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Forward one streamed boundary value to its local preview targets."""
    target_node_ids = [
        str(node_id)
        for node_id in stream_event.get("preview_target_node_ids", [])
        if str(node_id)
    ]
    if not target_node_ids:
        return
    logger.debug(
        "Forwarding streamed Modal boundary output previews for component=%s source_node=%s targets=%s.",
        state.payload.get("component_id"),
        stream_event.get("node_id"),
        target_node_ids,
    )
    _emit_local_preview_boundary_output(
        prompt_id=state.prompt_id,
        client_id=state.client_id,
        preview_target_node_ids=target_node_ids,
        image_value=deserialize_value(stream_event.get("value")),
    )


def _handle_cached_node_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Forward one cached-node marker through the local progress channel."""
    metadata = _progress_stream_event_metadata(stream_event)
    filter_node_id = metadata["filter_node_id"] if metadata is not None else None
    if state.suppress_status_stream and not _should_forward_suppressed_stream_event(
        state.payload, filter_node_id
    ):
        logger.debug(
            "Suppressing streamed Modal cached-node marker for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
            state.payload.get("component_id"),
            stream_event.get("node_id"),
            stream_event.get("real_node_id"),
        )
        return
    reported_node_id = metadata["node_id"] if metadata is not None else None
    if reported_node_id is None:
        return
    display_node_id = (
        metadata["display_node_id"] if metadata is not None else str(reported_node_id)
    )
    real_node_id = metadata["real_node_id"] if metadata is not None else None
    logger.debug(
        "Forwarding streamed Modal cached-node marker for component=%s node_id=%s real_node_id=%s.",
        state.payload.get("component_id"),
        reported_node_id,
        real_node_id,
    )
    _emit_local_modal_progress(
        prompt_id=state.prompt_id,
        client_id=state.client_id,
        node_id=str(reported_node_id),
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        real_node_id=real_node_id,
        cached_hit=True,
        **_remote_execution_identity(state.payload, state.active_remote_task_id),
    )


def _clear_suppressed_progress(state: _PayloadStreamState) -> None:
    """Clear retained node progress when a suppressed stream reaches a terminal phase."""
    for metadata in state.suppressed_progress_metadata.values():
        _emit_local_modal_progress(
            prompt_id=state.prompt_id,
            client_id=state.client_id,
            node_id=str(metadata["node_id"]),
            value=0.0,
            max_value=1.0,
            display_node_id=metadata["display_node_id"],
            real_node_id=metadata["real_node_id"],
            clear=True,
        )
    state.suppressed_progress_metadata.clear()


def _handle_status_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Forward one component-level remote execution status event."""
    remote_phase = str(stream_event.get("phase", "executing"))
    logger.info(
        "Forwarding streamed Modal progress for component=%s phase=%s active_node_id=%s.",
        state.payload.get("component_id"),
        remote_phase,
        stream_event.get("active_node_id"),
    )
    if state.suppress_status_stream:
        if remote_phase in {
            "execution_success",
            "execution_error",
            "execution_interrupted",
        }:
            _clear_suppressed_progress(state)
        return
    identity = _remote_execution_identity(
        state.payload, state.active_remote_task_id
    )
    if remote_phase == "execution_success":
        _emit_local_modal_status(
            prompt_id=state.prompt_id,
            client_id=state.client_id,
            phase="finalizing",
            node_ids=state.node_ids,
            modal_gpu=state.modal_gpu,
            status_message=(
                "Receiving remote outputs from "
                f"{_remote_execution_destination(state.payload)}"
            ),
            **identity,
        )
        return
    active_node_id = (
        str(stream_event["active_node_id"])
        if stream_event.get("active_node_id") is not None
        else None
    )
    _emit_local_modal_status(
        prompt_id=state.prompt_id,
        client_id=state.client_id,
        phase=remote_phase,
        node_ids=state.node_ids,
        modal_gpu=state.modal_gpu,
        active_node_id=active_node_id,
        completed_ancestor_node_ids=(
            _remote_prompt_ancestor_node_ids(state.payload, active_node_id) or None
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
        **identity,
    )


def _handle_progress_stream_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
) -> None:
    """Dispatch one progress envelope to its event-specific handler."""
    event_type = str(stream_event.get("event_type", ""))
    handlers = {
        "node_progress": _handle_node_progress_event,
        "executed": _handle_executed_event,
        "preview": _handle_preview_event,
        "boundary_output": _handle_boundary_output_event,
        "node_cached": _handle_cached_node_event,
    }
    handler = handlers.get(event_type)
    if handler is not None:
        handler(state, stream_event)
        return
    _handle_status_event(state, stream_event)


def _handle_result_event(
    state: _PayloadStreamState,
    stream_event: dict[str, Any],
    *,
    event_received_at: float,
    seconds_since_previous_event: float,
) -> None:
    """Validate and retain the final transport-safe result payload."""
    output_ref_payload = stream_event.get("output_ref")
    if is_durable_object_ref_payload(output_ref_payload):
        output_ref = DurableObjectRef.from_payload(output_ref_payload)
        reporter = RemoteTransferProgressReporter(
            state.payload,
            direction="download",
            total_bytes=output_ref.size_bytes,
        )
        reporter.start()
        download_started_at = time.monotonic()
        logger.info(
            "Downloading streamed Modal result object component=%s prompt_id=%s "
            "invocation_id=%s result_bytes=%d object_path=%s.",
            state.component_id,
            state.prompt_id or "none",
            state.invocation_id,
            output_ref.size_bytes,
            output_ref.object_path,
        )
        state.result_payload = materialize_modal_durable_object(
            output_ref,
            progress_callback=reporter.update,
        )
        reporter.complete()
        logger.info(
            "Downloaded streamed Modal result object in %.3fs component=%s "
            "prompt_id=%s invocation_id=%s result_bytes=%d.",
            time.monotonic() - download_started_at,
            state.component_id,
            state.prompt_id or "none",
            state.invocation_id,
            len(state.result_payload),
        )
        state.should_close_stream = True
        return
    candidate_outputs = stream_event.get("outputs")
    candidate_bytes = (
        len(candidate_outputs)
        if isinstance(candidate_outputs, bytes | bytearray)
        else -1
    )
    logger.info(
        "Received streamed Modal result component=%s prompt_id=%s invocation_id=%s "
        "result_bytes=%d stream_elapsed_seconds=%.3f seconds_since_previous_event=%.3f "
        "event_count=%d progress_event_count=%d.",
        state.component_id,
        state.prompt_id or "none",
        state.invocation_id,
        candidate_bytes,
        event_received_at - state.stream_started_at,
        seconds_since_previous_event,
        state.event_count,
        state.progress_event_count,
    )
    try:
        state.result_payload = coerce_serialized_node_outputs(candidate_outputs)
    except TypeError as exc:
        raise ModalRemoteInvocationError(
            "Modal streamed payload result did not include transport-safe outputs."
        ) from exc
    state.should_close_stream = True


def _close_consumed_payload_stream(
    state: _PayloadStreamState,
    stream_events: Iterator[dict[str, Any]],
) -> None:
    """Close a completed result stream and release any retained log stream."""
    if state.should_close_stream:
        close_started_at = time.monotonic()
        logger.info(
            "Starting local Modal result stream close component=%s prompt_id=%s invocation_id=%s.",
            state.component_id,
            state.prompt_id or "none",
            state.invocation_id,
        )
        _close_remote_payload_stream(stream_events)
        logger.info(
            "Finished local Modal result stream close in %.3fs component=%s prompt_id=%s invocation_id=%s.",
            time.monotonic() - close_started_at,
            state.component_id,
            state.prompt_id or "none",
            state.invocation_id,
        )
    if state.active_remote_log_task_id is not None:
        _release_remote_container_log_stream(state.active_remote_log_task_id)


def _consume_remote_payload_stream(
    payload: dict[str, Any],
    stream_events: Iterator[dict[str, Any]],
    *,
    input_transfer_bytes: int = 0,
) -> bytes:
    """Forward remote progress events into the local UI and return final bytes."""
    state = _new_payload_stream_state(payload)
    upload_reporter = RemoteTransferProgressReporter(
        payload,
        direction="upload",
        total_bytes=input_transfer_bytes,
        indeterminate=True,
    )
    upload_reporter.start()
    logger.info(
        "Starting local Modal stream consumption component=%s prompt_id=%s invocation_id=%s.",
        state.component_id,
        state.prompt_id or "none",
        state.invocation_id,
    )
    upload_completed = False
    try:
        for stream_event in stream_events:
            if not upload_completed:
                upload_reporter.complete()
                upload_completed = True
            event_received_at = time.monotonic()
            seconds_since_previous_event = (
                event_received_at - state.previous_event_at
            )
            state.previous_event_at = event_received_at
            state.event_count += 1
            if not state.speculative_prewarm_checked:
                state.speculative_prewarm_checked = True
                _schedule_speculative_affinity_prewarm(
                    payload, reason="current_remote_stream_started"
                )
            event_kind = str(stream_event.get("kind", ""))
            if event_kind == "remote_logs":
                _handle_remote_log_event(state, stream_event)
                continue
            if event_kind == "progress":
                state.progress_event_count += 1
                _handle_progress_stream_event(state, stream_event)
                continue
            if event_kind == "result":
                _handle_result_event(
                    state,
                    stream_event,
                    event_received_at=event_received_at,
                    seconds_since_previous_event=seconds_since_previous_event,
                )
                break
            logger.debug(
                "Ignoring unexpected streamed Modal event kind=%s for component=%s.",
                event_kind,
                payload.get("component_id"),
            )
    finally:
        _close_consumed_payload_stream(state, stream_events)

    if state.result_payload is None:
        raise ModalRemoteInvocationError(
            f"Modal streamed payload for component={payload.get('component_id')!r} did not yield a final result."
        )
    logger.info(
        "Finished local Modal stream consumption in %.3fs component=%s prompt_id=%s invocation_id=%s "
        "result_bytes=%d event_count=%d progress_event_count=%d.",
        time.monotonic() - state.stream_started_at,
        state.component_id,
        state.prompt_id or "none",
        state.invocation_id,
        len(state.result_payload),
        state.event_count,
        state.progress_event_count,
    )
    return bytes(state.result_payload)
