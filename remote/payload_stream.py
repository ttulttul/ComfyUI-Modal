"""Local UI forwarding for streamed Modal payload events."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterator

from ..serialization import (
    coerce_serialized_node_outputs,
    deserialize_value,
)
from .local_ui_events import (
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





def _consume_remote_payload_stream(
    payload: dict[str, Any],
    stream_events: Iterator[dict[str, Any]],
) -> bytes:
    """Forward remote progress events into the local UI and return the final payload bytes."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    node_ids = [str(node_id) for node_id in payload.get("component_node_ids", [])]
    modal_gpu = (
        str(payload["modal_gpu"]) if payload.get("modal_gpu") is not None else None
    )
    suppress_status_stream = bool(payload.get("suppress_status_stream"))
    result_payload: bytes | bytearray | None = None
    suppressed_progress_node_metadata: dict[str, dict[str, str | None]] = {}
    active_remote_task_id: str | None = None
    active_remote_log_task_id: str | None = None
    should_close_stream = False
    component_id = str(payload.get("component_id") or "payload")
    invocation_id = str(payload.get("invocation_id") or "none")
    stream_started_at = time.monotonic()
    previous_event_at = stream_started_at
    event_count = 0
    progress_event_count = 0
    speculative_prewarm_checked = False
    logger.info(
        "Starting local Modal stream consumption component=%s prompt_id=%s invocation_id=%s.",
        component_id,
        prompt_id or "none",
        invocation_id,
    )

    try:
        for stream_event in stream_events:
            event_received_at = time.monotonic()
            seconds_since_previous_event = event_received_at - previous_event_at
            previous_event_at = event_received_at
            event_count += 1
            if not speculative_prewarm_checked:
                speculative_prewarm_checked = True
                _schedule_speculative_affinity_prewarm(
                    payload,
                    reason="current_remote_stream_started",
                )
            event_kind = str(stream_event.get("kind", ""))
            if event_kind == "remote_logs":
                task_id = _coerce_modal_task_id(stream_event.get("task_id"))
                if task_id is not None and active_remote_task_id is None:
                    active_remote_task_id = task_id
                if (
                    task_id is not None
                    and active_remote_log_task_id is None
                    and _is_remote_container_log_stream_enabled()
                ):
                    active_remote_log_task_id = _retain_remote_container_log_stream(
                        task_id
                    )
                continue
            if event_kind == "progress":
                progress_event_count += 1
                event_type = str(stream_event.get("event_type", ""))
                if event_type == "node_progress":
                    progress_metadata = _progress_stream_event_metadata(stream_event)
                    filter_node_id = (
                        progress_metadata["filter_node_id"]
                        if progress_metadata is not None
                        else None
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
                        and not _should_forward_suppressed_stream_event(
                            payload, filter_node_id
                        )
                    ):
                        logger.debug(
                            "Suppressing streamed Modal node progress for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            stream_event.get("real_node_id"),
                        )
                        continue
                    reported_node_id = (
                        progress_metadata["node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    if reported_node_id is not None:
                        display_node_id = (
                            progress_metadata["display_node_id"]
                            if progress_metadata is not None
                            else str(reported_node_id)
                        )
                        real_node_id = (
                            progress_metadata["real_node_id"]
                            if progress_metadata is not None
                            else None
                        )
                        progress_node_id = real_node_id or display_node_id
                        completed_ancestor_node_ids = _remote_prompt_ancestor_node_ids(
                            payload,
                            progress_node_id,
                        )
                        if lane_id is not None:
                            _remember_mapped_lane_node_id(
                                payload, lane_id, progress_node_id
                            )
                        elif (
                            suppress_status_stream
                            and not aggregate_only
                            and progress_metadata is not None
                        ):
                            suppressed_progress_node_metadata[
                                str(progress_metadata["filter_node_id"])
                            ] = {
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
                        progress_kwargs = {
                            "prompt_id": prompt_id,
                            "client_id": client_id,
                            "node_id": str(reported_node_id),
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
                                    int(payload["map_item_index"])
                                    if payload.get("map_item_index") is not None
                                    else None
                                )
                            ),
                            "aggregate_only": aggregate_only,
                        }
                        progress_kwargs.update(
                            _remote_execution_identity(payload, active_remote_task_id)
                        )
                        for string_field in ("stage", "message", "unit"):
                            if stream_event.get(string_field) is not None:
                                progress_kwargs[string_field] = str(
                                    stream_event[string_field]
                                )
                        if stream_event.get("indeterminate") is not None:
                            progress_kwargs["indeterminate"] = bool(
                                stream_event["indeterminate"]
                            )
                        for numeric_field in (
                            "elapsed_seconds",
                            "time_to_first_token_seconds",
                            "tokens_per_second",
                        ):
                            if stream_event.get(numeric_field) is not None:
                                progress_kwargs[numeric_field] = float(
                                    stream_event[numeric_field]
                                )
                        if completed_ancestor_node_ids:
                            progress_kwargs[
                                "completed_ancestor_node_ids"
                            ] = completed_ancestor_node_ids
                        _emit_local_modal_progress(**progress_kwargs)
                    continue
                if event_type == "executed":
                    reported_node_id = stream_event.get("node_id")
                    if reported_node_id is not None:
                        if not _should_forward_suppressed_stream_event(
                            payload, reported_node_id
                        ):
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
                            output_payload=deserialize_value(
                                stream_event.get("output")
                            ),
                        )
                    continue
                if event_type == "preview":
                    reported_node_id = stream_event.get("node_id")
                    image_bytes = deserialize_value(stream_event.get("image_bytes"))
                    if reported_node_id is not None and isinstance(image_bytes, bytes):
                        if not _should_forward_suppressed_stream_event(
                            payload, reported_node_id
                        ):
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
                if event_type == "node_cached":
                    progress_metadata = _progress_stream_event_metadata(stream_event)
                    filter_node_id = (
                        progress_metadata["filter_node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    if (
                        suppress_status_stream
                        and not _should_forward_suppressed_stream_event(
                            payload, filter_node_id
                        )
                    ):
                        logger.debug(
                            "Suppressing streamed Modal cached-node marker for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            stream_event.get("real_node_id"),
                        )
                        continue
                    reported_node_id = (
                        progress_metadata["node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    if reported_node_id is not None:
                        display_node_id = (
                            progress_metadata["display_node_id"]
                            if progress_metadata is not None
                            else str(reported_node_id)
                        )
                        real_node_id = (
                            progress_metadata["real_node_id"]
                            if progress_metadata is not None
                            else None
                        )
                        logger.debug(
                            "Forwarding streamed Modal cached-node marker for component=%s node_id=%s real_node_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                            real_node_id,
                        )
                        _emit_local_modal_progress(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            value=0.0,
                            max_value=1.0,
                            display_node_id=display_node_id,
                            real_node_id=real_node_id,
                            cached_hit=True,
                            **_remote_execution_identity(
                                payload, active_remote_task_id
                            ),
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
                    if remote_phase in {
                        "execution_success",
                        "execution_error",
                        "execution_interrupted",
                    }:
                        for (
                            progress_metadata
                        ) in suppressed_progress_node_metadata.values():
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
                        modal_gpu=modal_gpu,
                        status_message=(
                            "Receiving remote outputs from "
                            f"{_remote_execution_destination(payload)}"
                        ),
                        **_remote_execution_identity(payload, active_remote_task_id),
                    )
                    continue
                _emit_local_modal_status(
                    prompt_id=prompt_id,
                    client_id=client_id,
                    phase=remote_phase,
                    node_ids=node_ids,
                    modal_gpu=modal_gpu,
                    active_node_id=(
                        str(stream_event["active_node_id"])
                        if stream_event.get("active_node_id") is not None
                        else None
                    ),
                    completed_ancestor_node_ids=_remote_prompt_ancestor_node_ids(
                        payload,
                        str(stream_event["active_node_id"])
                        if stream_event.get("active_node_id") is not None
                        else None,
                    )
                    or None,
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
                    **_remote_execution_identity(payload, active_remote_task_id),
                )
                continue
            if event_kind == "result":
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
                    component_id,
                    prompt_id or "none",
                    invocation_id,
                    candidate_bytes,
                    event_received_at - stream_started_at,
                    seconds_since_previous_event,
                    event_count,
                    progress_event_count,
                )
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
            close_started_at = time.monotonic()
            logger.info(
                "Starting local Modal result stream close component=%s prompt_id=%s invocation_id=%s.",
                component_id,
                prompt_id or "none",
                invocation_id,
            )
            _close_remote_payload_stream(stream_events)
            logger.info(
                "Finished local Modal result stream close in %.3fs component=%s prompt_id=%s "
                "invocation_id=%s.",
                time.monotonic() - close_started_at,
                component_id,
                prompt_id or "none",
                invocation_id,
            )
        if active_remote_log_task_id is not None:
            _release_remote_container_log_stream(active_remote_log_task_id)

    if result_payload is None:
        raise ModalRemoteInvocationError(
            f"Modal streamed payload for component={payload.get('component_id')!r} did not yield a final result."
        )
    logger.info(
        "Finished local Modal stream consumption in %.3fs component=%s prompt_id=%s invocation_id=%s "
        "result_bytes=%d event_count=%d progress_event_count=%d.",
        time.monotonic() - stream_started_at,
        component_id,
        prompt_id or "none",
        invocation_id,
        len(result_payload),
        event_count,
        progress_event_count,
    )
    return bytes(result_payload)


