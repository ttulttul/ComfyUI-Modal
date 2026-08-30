"""Client-scoped status emission and replay for remote workflow UI state."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_MODAL_UI_EVENT_RETENTION_SECONDS = 2 * 60 * 60
_MODAL_UI_EVENT_LIMIT_PER_CLIENT = 512
_MODAL_UI_EVENTS_LOCK = threading.Lock()
_MODAL_UI_EVENTS_BY_CLIENT: dict[str, deque[dict[str, Any]]] = {}


def _emit_modal_status(
    prompt_server: Any,
    phase: str,
    *,
    client_id: str | None,
    prompt_id: str | None,
    node_ids: list[str],
    configurator_node_id: str | None = None,
    modal_gpu: str | None = None,
    component_node_ids_by_representative: dict[str, list[str]] | None = None,
    active_node_id: str | None = None,
    active_node_class_type: str | None = None,
    active_node_role: str | None = None,
    error_message: str | None = None,
    status_message: str | None = None,
    status_current: int | None = None,
    status_total: int | None = None,
    execution_environment_id: str | None = None,
    remote_execution_assignments: Mapping[str, Mapping[str, Any]] | None = None,
    remote_execution_configurations: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Send a Modal execution status event to the active websocket client."""
    if client_id is None:
        logger.debug(
            "Skipping Modal status event %s because no client id is available.", phase
        )
        return

    payload: dict[str, Any] = {
        "phase": phase,
        "prompt_id": prompt_id,
        "node_ids": list(node_ids),
    }
    if modal_gpu is not None:
        payload["modal_gpu"] = modal_gpu
    if configurator_node_id is not None:
        payload["configurator_node_id"] = configurator_node_id
    if component_node_ids_by_representative:
        payload["components"] = [
            {
                "representative_node_id": representative_node_id,
                "node_ids": list(component_node_ids),
            }
            for representative_node_id, component_node_ids in sorted(
                component_node_ids_by_representative.items()
            )
        ]
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
    if execution_environment_id is not None:
        payload["execution_environment_id"] = execution_environment_id
    if remote_execution_assignments is not None:
        payload["remote_execution_assignments"] = copy.deepcopy(
            dict(remote_execution_assignments)
        )
    if remote_execution_configurations is not None:
        payload["remote_execution_configurations"] = copy.deepcopy(
            list(remote_execution_configurations)
        )

    record_modal_ui_event("modal_status", payload, client_id)
    prompt_server.send_sync("modal_status", payload, client_id)


def record_modal_ui_event(
    event: str, payload: Mapping[str, Any], client_id: str | None
) -> None:
    """Store one client-scoped Modal UI event for replay after browser refocus."""
    if client_id is None:
        return

    event_record = {
        "event": event,
        "payload": copy.deepcopy(dict(payload)),
        "updated_at": time.time(),
    }
    with _MODAL_UI_EVENTS_LOCK:
        client_events = _MODAL_UI_EVENTS_BY_CLIENT.setdefault(
            client_id,
            deque(maxlen=_MODAL_UI_EVENT_LIMIT_PER_CLIENT),
        )
        _discard_replaced_telemetry_event_locked(client_events, event_record)
        client_events.append(event_record)
        _prune_modal_ui_events_locked(client_events)


def _discard_replaced_telemetry_event_locked(
    client_events: deque[dict[str, Any]],
    event_record: Mapping[str, Any],
) -> None:
    """Retain only the newest replay sample for one prompt execution source."""
    if event_record.get("event") != "modal_telemetry":
        return
    payload = event_record.get("payload")
    if not isinstance(payload, Mapping):
        return
    replacement_key = (
        str(payload.get("prompt_id") or ""),
        str(payload.get("execution_environment_id") or ""),
        str(payload.get("execution_location") or ""),
        str(payload.get("component_id") or ""),
    )
    for existing_record in tuple(client_events):
        existing_payload = existing_record.get("payload")
        if existing_record.get("event") != "modal_telemetry" or not isinstance(
            existing_payload, Mapping
        ):
            continue
        existing_key = (
            str(existing_payload.get("prompt_id") or ""),
            str(existing_payload.get("execution_environment_id") or ""),
            str(existing_payload.get("execution_location") or ""),
            str(existing_payload.get("component_id") or ""),
        )
        if existing_key == replacement_key:
            client_events.remove(existing_record)
            return


def modal_ui_events_for_client(client_id: str | None) -> list[dict[str, Any]]:
    """Return recent Modal UI events for one websocket client."""
    if not client_id:
        return []

    with _MODAL_UI_EVENTS_LOCK:
        client_events = _MODAL_UI_EVENTS_BY_CLIENT.get(client_id)
        if client_events is None:
            return []
        _prune_modal_ui_events_locked(client_events)
        return [copy.deepcopy(event_record) for event_record in client_events]


def _prune_modal_ui_events_locked(client_events: deque[dict[str, Any]]) -> None:
    """Discard stale Modal UI events while the caller holds the event lock."""
    cutoff = time.time() - _MODAL_UI_EVENT_RETENTION_SECONDS
    while client_events and float(client_events[0].get("updated_at", 0.0)) < cutoff:
        client_events.popleft()
