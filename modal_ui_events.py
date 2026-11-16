"""Client-scoped status emission and replay for remote workflow UI state."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_MODAL_UI_EVENT_RETENTION_SECONDS = 2 * 60 * 60
_MODAL_UI_EVENT_LIMIT_PER_CLIENT = 512
_MODAL_UI_EVENTS_LOCK = threading.Lock()
_MODAL_UI_EVENTS_BY_CLIENT: dict[str, deque[dict[str, Any]]] = {}

PROGRESSIVE_PREVIEW_EVENT = "modal_progressive_preview"
_PROGRESSIVE_PREVIEW_SNAPSHOT_LIMIT_PER_CLIENT = 16
_PreviewSnapshotKey = tuple[str, str]


@dataclass
class ProgressivePreviewSnapshot:
    """Accumulated slot contents of one progressive preview node for one prompt."""

    prompt_id: str
    node: str
    slot_count: int = 0
    slots: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    updated_at: float = 0.0

    def apply(self, payload: Mapping[str, Any]) -> None:
        """Fold one ``start`` or ``slot`` websocket payload into the snapshot."""
        declared_count = max(0, int(payload.get("slot_count") or 0))
        if payload.get("phase") == "start":
            self.slot_count = declared_count
            self.slots = {}
        elif payload.get("phase") == "slot":
            slot_index = max(0, int(payload.get("slot_index") or 0))
            self.slots[slot_index] = copy.deepcopy(list(payload.get("items") or []))
            self.slot_count = max(self.slot_count, declared_count, slot_index + 1)
        self.updated_at = time.time()

    def as_event_record(self) -> dict[str, Any]:
        """Return the snapshot as one replayable ``snapshot`` phase event record."""
        return {
            "event": PROGRESSIVE_PREVIEW_EVENT,
            "payload": {
                "phase": "snapshot",
                "prompt_id": self.prompt_id,
                "node": self.node,
                "slot_count": self.slot_count,
                "slots": [
                    {"slot_index": index, "items": copy.deepcopy(items)}
                    for index, items in sorted(self.slots.items())
                ],
            },
            "updated_at": self.updated_at,
        }


_PROGRESSIVE_PREVIEW_SNAPSHOTS_BY_CLIENT: dict[
    str, OrderedDict[_PreviewSnapshotKey, ProgressivePreviewSnapshot]
] = {}


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
    if event == PROGRESSIVE_PREVIEW_EVENT:
        _record_progressive_preview_snapshot(payload, client_id)
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
        client_events.append(event_record)
        _prune_modal_ui_events_locked(client_events)


def modal_ui_events_for_client(client_id: str | None) -> list[dict[str, Any]]:
    """Return recent Modal UI events for one websocket client."""
    if not client_id:
        return []

    with _MODAL_UI_EVENTS_LOCK:
        client_events = _MODAL_UI_EVENTS_BY_CLIENT.get(client_id) or deque()
        _prune_modal_ui_events_locked(client_events)
        events = [copy.deepcopy(event_record) for event_record in client_events]
    return events + _progressive_preview_snapshot_records(client_id)


def _prune_modal_ui_events_locked(client_events: deque[dict[str, Any]]) -> None:
    """Discard stale Modal UI events while the caller holds the event lock."""
    cutoff = time.time() - _MODAL_UI_EVENT_RETENTION_SECONDS
    while client_events and float(client_events[0].get("updated_at", 0.0)) < cutoff:
        client_events.popleft()


def _record_progressive_preview_snapshot(
    payload: Mapping[str, Any], client_id: str
) -> None:
    """Fold one progressive preview event into its client's per-node snapshot.

    Preview slots are kept apart from the bounded event deque so that a long run's
    progress events cannot evict them before a backgrounded browser tab catches up.
    """
    prompt_id = payload.get("prompt_id")
    node = payload.get("node")
    if prompt_id is None or node is None:
        logger.debug("Skipping progressive preview snapshot without prompt or node.")
        return
    key = (str(prompt_id), str(node))
    with _MODAL_UI_EVENTS_LOCK:
        snapshots = _PROGRESSIVE_PREVIEW_SNAPSHOTS_BY_CLIENT.setdefault(
            client_id, OrderedDict()
        )
        snapshot = snapshots.pop(key, None) or ProgressivePreviewSnapshot(*key)
        snapshot.apply(payload)
        snapshots[key] = snapshot
        while len(snapshots) > _PROGRESSIVE_PREVIEW_SNAPSHOT_LIMIT_PER_CLIENT:
            snapshots.popitem(last=False)


def _progressive_preview_snapshot_records(client_id: str) -> list[dict[str, Any]]:
    """Return a client's unexpired preview snapshots, oldest update first."""
    cutoff = time.time() - _MODAL_UI_EVENT_RETENTION_SECONDS
    with _MODAL_UI_EVENTS_LOCK:
        snapshots = _PROGRESSIVE_PREVIEW_SNAPSHOTS_BY_CLIENT.get(client_id)
        if not snapshots:
            return []
        for key in [key for key, value in snapshots.items() if value.updated_at < cutoff]:
            del snapshots[key]
        return [snapshot.as_event_record() for snapshot in snapshots.values()]
