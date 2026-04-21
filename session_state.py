"""Prompt-scoped remote session helpers for split Modal proxy execution."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

_REMOTE_SESSION_HANDLE_MARKER = "__comfy_modal_remote_session_handle__"
_REMOTE_SESSION_VALUE_REF_MARKER = "__comfy_modal_remote_session_value_ref__"
_REMOTE_SESSION_BRIDGE_REF_MARKER = "__comfy_modal_remote_session_bridge_ref__"
_REMOTE_SESSION_BRIDGE_RECORD_MARKER = "__comfy_modal_remote_session_bridge_record__"


class RemoteSessionStateError(RuntimeError):
    """Raised when remote session metadata is invalid or missing."""


@dataclass(frozen=True)
class RemoteSessionHandle:
    """Opaque prompt-scoped handle used to group remote-only runtime state."""

    session_id: str
    prompt_id: str | None = None
    owner_component_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize this handle into a JSON-safe payload mapping."""
        return {
            _REMOTE_SESSION_HANDLE_MARKER: True,
            "session_id": self.session_id,
            "prompt_id": self.prompt_id,
            "owner_component_id": self.owner_component_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoteSessionHandle":
        """Deserialize a previously serialized handle payload."""
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise RemoteSessionStateError("Remote session handles must define session_id.")
        prompt_id = payload.get("prompt_id")
        owner_component_id = payload.get("owner_component_id")
        return cls(
            session_id=session_id,
            prompt_id=(str(prompt_id) if prompt_id is not None else None),
            owner_component_id=(
                str(owner_component_id) if owner_component_id is not None else None
            ),
        )


@dataclass(frozen=True)
class RemoteSessionValueRef:
    """Reference one remote-only value stored inside a prompt-scoped session."""

    session_id: str
    node_id: str
    output_index: int

    def to_payload(self) -> dict[str, Any]:
        """Serialize this value reference into a JSON-safe payload mapping."""
        return {
            _REMOTE_SESSION_VALUE_REF_MARKER: True,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "output_index": self.output_index,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoteSessionValueRef":
        """Deserialize a previously serialized value-reference payload."""
        session_id = str(payload.get("session_id") or "").strip()
        node_id = str(payload.get("node_id") or "").strip()
        if not session_id or not node_id:
            raise RemoteSessionStateError(
                "Remote session value refs must define session_id and node_id."
            )
        output_index = payload.get("output_index")
        if output_index is None:
            raise RemoteSessionStateError("Remote session value refs must define output_index.")
        return cls(
            session_id=session_id,
            node_id=node_id,
            output_index=int(output_index),
        )


@dataclass(frozen=True)
class RemoteSessionBridgeRef:
    """Durable reference to one replayable remote-only value."""

    bridge_key: str
    node_id: str
    output_index: int
    session_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize this durable bridge ref into a JSON-safe payload mapping."""
        return {
            _REMOTE_SESSION_BRIDGE_REF_MARKER: True,
            "bridge_key": self.bridge_key,
            "node_id": self.node_id,
            "output_index": self.output_index,
            "session_id": self.session_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoteSessionBridgeRef":
        """Deserialize a previously serialized durable bridge ref payload."""
        bridge_key = str(payload.get("bridge_key") or "").strip()
        node_id = str(payload.get("node_id") or "").strip()
        if not bridge_key or not node_id:
            raise RemoteSessionStateError(
                "Remote session bridge refs must define bridge_key and node_id."
            )
        output_index = payload.get("output_index")
        if output_index is None:
            raise RemoteSessionStateError("Remote session bridge refs must define output_index.")
        session_id = payload.get("session_id")
        return cls(
            bridge_key=bridge_key,
            node_id=node_id,
            output_index=int(output_index),
            session_id=(str(session_id) if session_id is not None and str(session_id).strip() else None),
        )


@dataclass(frozen=True)
class RemoteSessionBridgeRecord:
    """Replay metadata for one durable remote-only output reference."""

    bridge_key: str
    node_id: str
    output_index: int
    producer_payload: dict[str, Any]
    producer_inputs: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Serialize this bridge record into a JSON-safe payload mapping."""
        return {
            _REMOTE_SESSION_BRIDGE_RECORD_MARKER: True,
            "bridge_key": self.bridge_key,
            "node_id": self.node_id,
            "output_index": self.output_index,
            "producer_payload": self.producer_payload,
            "producer_inputs": self.producer_inputs,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoteSessionBridgeRecord":
        """Deserialize a previously serialized bridge record payload."""
        bridge_key = str(payload.get("bridge_key") or "").strip()
        node_id = str(payload.get("node_id") or "").strip()
        if not bridge_key or not node_id:
            raise RemoteSessionStateError(
                "Remote session bridge records must define bridge_key and node_id."
            )
        output_index = payload.get("output_index")
        if output_index is None:
            raise RemoteSessionStateError(
                "Remote session bridge records must define output_index."
            )
        producer_payload = payload.get("producer_payload")
        producer_inputs = payload.get("producer_inputs")
        if not isinstance(producer_payload, Mapping) or not isinstance(producer_inputs, Mapping):
            raise RemoteSessionStateError(
                "Remote session bridge records must define mapping producer_payload and producer_inputs."
            )
        return cls(
            bridge_key=bridge_key,
            node_id=node_id,
            output_index=int(output_index),
            producer_payload=dict(producer_payload),
            producer_inputs=dict(producer_inputs),
        )


def is_remote_session_handle_payload(payload: Any) -> bool:
    """Return whether one arbitrary value looks like a session-handle payload."""
    return isinstance(payload, Mapping) and bool(payload.get(_REMOTE_SESSION_HANDLE_MARKER))


def is_remote_session_value_ref_payload(payload: Any) -> bool:
    """Return whether one arbitrary value looks like a session-value-ref payload."""
    return isinstance(payload, Mapping) and bool(payload.get(_REMOTE_SESSION_VALUE_REF_MARKER))


def is_remote_session_bridge_ref_payload(payload: Any) -> bool:
    """Return whether one arbitrary value looks like a durable bridge-ref payload."""
    return isinstance(payload, Mapping) and bool(payload.get(_REMOTE_SESSION_BRIDGE_REF_MARKER))


def _canonicalize_bridge_key_value(value: Any) -> Any:
    """Normalize transient session metadata out of one bridge-key value."""
    if isinstance(value, list):
        return [_canonicalize_bridge_key_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    if is_remote_session_handle_payload(value):
        return {
            _REMOTE_SESSION_HANDLE_MARKER: True,
            "owner_component_id": value.get("owner_component_id"),
        }
    if is_remote_session_value_ref_payload(value):
        return {
            _REMOTE_SESSION_VALUE_REF_MARKER: True,
            "node_id": value.get("node_id"),
            "output_index": value.get("output_index"),
        }
    if is_remote_session_bridge_ref_payload(value):
        return {
            _REMOTE_SESSION_BRIDGE_REF_MARKER: True,
            "bridge_key": value.get("bridge_key"),
            "node_id": value.get("node_id"),
            "output_index": value.get("output_index"),
        }
    return {
        str(key): _canonicalize_bridge_key_value(item)
        for key, item in value.items()
    }


def stable_session_bridge_key(
    *,
    producer_payload: Mapping[str, Any],
    producer_inputs: Mapping[str, Any],
    node_id: str,
    output_index: int,
) -> str:
    """Return a stable digest for one replayable remote bridge output."""
    canonical_payload = {
        "producer_payload": _canonicalize_bridge_key_value(dict(producer_payload)),
        "producer_inputs": _canonicalize_bridge_key_value(dict(producer_inputs)),
        "node_id": str(node_id),
        "output_index": int(output_index),
    }
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"RSB_{hashlib.sha256(encoded).hexdigest()}"


@dataclass
class _RemoteSessionBucket:
    """Hold the live values cached for one prompt-scoped remote session."""

    handle: RemoteSessionHandle
    values: dict[tuple[str, int], Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class _RemoteSessionBridgeBucket:
    """Hold one durable bridge record stored for later replay."""

    record: RemoteSessionBridgeRecord
    created_at: float = field(default_factory=time.time)


class InMemoryRemoteSessionStore:
    """Thread-safe in-memory store for remote-only values shared across proxy calls."""

    def __init__(self) -> None:
        """Initialize the empty session store."""
        self._lock = threading.Lock()
        self._sessions: dict[str, _RemoteSessionBucket] = {}

    def ensure_session(self, handle: RemoteSessionHandle) -> RemoteSessionHandle:
        """Create one session bucket when it does not yet exist."""
        with self._lock:
            bucket = self._sessions.get(handle.session_id)
            if bucket is None:
                self._sessions[handle.session_id] = _RemoteSessionBucket(handle=handle)
                logger.info(
                    "Created remote session session_id=%s prompt_id=%s owner_component_id=%s.",
                    handle.session_id,
                    handle.prompt_id,
                    handle.owner_component_id,
                )
                return handle
            logger.info(
                "Reusing remote session session_id=%s prompt_id=%s owner_component_id=%s age_seconds=%.3f value_count=%d.",
                bucket.handle.session_id,
                bucket.handle.prompt_id,
                bucket.handle.owner_component_id,
                max(0.0, time.time() - bucket.created_at),
                len(bucket.values),
            )
            return bucket.handle

    def put_output(
        self,
        handle: RemoteSessionHandle,
        *,
        node_id: str,
        output_index: int,
        value: Any,
    ) -> RemoteSessionValueRef:
        """Store one output value under a prompt-scoped session and return its ref."""
        self.ensure_session(handle)
        ref = RemoteSessionValueRef(
            session_id=handle.session_id,
            node_id=str(node_id),
            output_index=int(output_index),
        )
        with self._lock:
            bucket = self._sessions[handle.session_id]
            value_key = (ref.node_id, ref.output_index)
            replacing_existing = value_key in bucket.values
            bucket.values[value_key] = value
        logger.info(
            "Stored remote session value session_id=%s node_id=%s output_index=%d result=%s value_type=%s total_value_count=%d.",
            ref.session_id,
            ref.node_id,
            ref.output_index,
            "replace" if replacing_existing else "create",
            type(value).__name__,
            len(bucket.values),
        )
        return ref

    def get_output(self, ref: RemoteSessionValueRef) -> Any:
        """Return one stored output value or raise when it no longer exists."""
        with self._lock:
            bucket = self._sessions.get(ref.session_id)
            if bucket is None:
                logger.warning(
                    "Remote session lookup missed session_id=%s node_id=%s output_index=%d reason=session-missing.",
                    ref.session_id,
                    ref.node_id,
                    ref.output_index,
                )
                raise RemoteSessionStateError(
                    f"Remote session {ref.session_id!r} was not found."
                )
            value_key = (ref.node_id, ref.output_index)
            if value_key not in bucket.values:
                logger.warning(
                    "Remote session lookup missed session_id=%s node_id=%s output_index=%d reason=value-missing age_seconds=%.3f value_count=%d.",
                    ref.session_id,
                    ref.node_id,
                    ref.output_index,
                    max(0.0, time.time() - bucket.created_at),
                    len(bucket.values),
                )
                raise RemoteSessionStateError(
                    "Remote session value was not found for "
                    f"session_id={ref.session_id!r} node_id={ref.node_id!r} "
                    f"output_index={ref.output_index}."
                )
            logger.info(
                "Resolved remote session value session_id=%s node_id=%s output_index=%d age_seconds=%.3f value_count=%d.",
                ref.session_id,
                ref.node_id,
                ref.output_index,
                max(0.0, time.time() - bucket.created_at),
                len(bucket.values),
            )
            return bucket.values[value_key]

    def try_get_output(self, ref: RemoteSessionValueRef) -> tuple[bool, Any | None]:
        """Return whether a live output exists without logging benign cache misses."""
        with self._lock:
            bucket = self._sessions.get(ref.session_id)
            if bucket is None:
                return False, None
            value_key = (ref.node_id, ref.output_index)
            if value_key not in bucket.values:
                return False, None
            return True, bucket.values[value_key]

    def clear_session(self, handle: RemoteSessionHandle) -> None:
        """Drop one prompt-scoped session and every value stored inside it."""
        with self._lock:
            removed = self._sessions.pop(handle.session_id, None)
        if removed is not None:
            logger.info(
                "Cleared remote session session_id=%s prompt_id=%s owner_component_id=%s age_seconds=%.3f value_count=%d.",
                handle.session_id,
                removed.handle.prompt_id,
                removed.handle.owner_component_id,
                max(0.0, time.time() - removed.created_at),
                len(removed.values),
            )
            return
        logger.warning(
            "Remote session clear skipped session_id=%s prompt_id=%s owner_component_id=%s reason=session-missing.",
            handle.session_id,
            handle.prompt_id,
            handle.owner_component_id,
        )

    def resolve_value(self, value: Any) -> Any:
        """Resolve one possible session reference back into the underlying live value."""
        return self.resolve_value_with_bridges(value)

    def resolve_value_with_bridges(
        self,
        value: Any,
        *,
        target_session_handle: RemoteSessionHandle | None = None,
        bridge_resolver: Callable[[RemoteSessionBridgeRef], Any] | None = None,
        resolution_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> Any:
        """Resolve one possible live or durable session reference into the underlying value."""
        if is_remote_session_value_ref_payload(value):
            ref = RemoteSessionValueRef.from_payload(value)
            logger.info(
                "Resolving remote session ref session_id=%s node_id=%s output_index=%d.",
                ref.session_id,
                ref.node_id,
                ref.output_index,
            )
            if resolution_callback is not None:
                resolution_callback(
                    "session-value-hit",
                    {
                        "session_id": ref.session_id,
                        "node_id": ref.node_id,
                        "output_index": ref.output_index,
                    },
                )
            return self.get_output(ref)

        if not is_remote_session_bridge_ref_payload(value):
            return value

        ref = RemoteSessionBridgeRef.from_payload(value)
        logger.info(
            "Resolving remote session bridge ref bridge_key=%s node_id=%s output_index=%d target_session_id=%s source_session_id=%s.",
            ref.bridge_key,
            ref.node_id,
            ref.output_index,
            target_session_handle.session_id if target_session_handle is not None else None,
            ref.session_id,
        )
        if target_session_handle is not None:
            found, live_value = self.try_get_output(
                RemoteSessionValueRef(
                    session_id=target_session_handle.session_id,
                    node_id=ref.node_id,
                    output_index=ref.output_index,
                )
            )
            if found:
                logger.info(
                    "Resolved remote session bridge ref bridge_key=%s via target session_id=%s.",
                    ref.bridge_key,
                    target_session_handle.session_id,
                )
                if resolution_callback is not None:
                    resolution_callback(
                        "bridge-target-hit",
                        {
                            "bridge_key": ref.bridge_key,
                            "session_id": target_session_handle.session_id,
                            "node_id": ref.node_id,
                            "output_index": ref.output_index,
                        },
                    )
                return live_value
        if ref.session_id:
            found, live_value = self.try_get_output(
                RemoteSessionValueRef(
                    session_id=ref.session_id,
                    node_id=ref.node_id,
                    output_index=ref.output_index,
                )
            )
            if found:
                logger.info(
                    "Resolved remote session bridge ref bridge_key=%s via source session_id=%s.",
                    ref.bridge_key,
                    ref.session_id,
                )
                if resolution_callback is not None:
                    resolution_callback(
                        "bridge-source-hit",
                        {
                            "bridge_key": ref.bridge_key,
                            "session_id": ref.session_id,
                            "node_id": ref.node_id,
                            "output_index": ref.output_index,
                        },
                    )
                return live_value
        if bridge_resolver is None:
            raise RemoteSessionStateError(
                "Remote session bridge value was not found and no bridge_resolver was provided "
                f"for bridge_key={ref.bridge_key!r}."
            )
        logger.info(
            "Rehydrating remote session bridge ref bridge_key=%s into target_session_id=%s.",
            ref.bridge_key,
            target_session_handle.session_id if target_session_handle is not None else None,
        )
        if resolution_callback is not None:
            resolution_callback(
                "bridge-rehydrate",
                {
                    "bridge_key": ref.bridge_key,
                    "target_session_id": (
                        target_session_handle.session_id if target_session_handle is not None else None
                    ),
                    "node_id": ref.node_id,
                    "output_index": ref.output_index,
                },
            )
        return bridge_resolver(ref)


class InMemoryRemoteSessionBridgeStore:
    """Thread-safe in-memory store for durable replay metadata."""

    def __init__(self) -> None:
        """Initialize the empty bridge-record store."""
        self._lock = threading.Lock()
        self._records: dict[str, _RemoteSessionBridgeBucket] = {}

    def put_record(self, record: RemoteSessionBridgeRecord) -> None:
        """Store or replace one durable bridge record."""
        with self._lock:
            replaced = record.bridge_key in self._records
            self._records[record.bridge_key] = _RemoteSessionBridgeBucket(record=record)
        logger.info(
            "Stored remote session bridge record bridge_key=%s node_id=%s output_index=%d result=%s.",
            record.bridge_key,
            record.node_id,
            record.output_index,
            "replace" if replaced else "create",
        )

    def get_record(self, bridge_key: str) -> RemoteSessionBridgeRecord:
        """Return one stored durable bridge record or raise when it does not exist."""
        with self._lock:
            bucket = self._records.get(bridge_key)
        if bucket is None:
            logger.warning(
                "Remote session bridge lookup missed bridge_key=%s reason=record-missing.",
                bridge_key,
            )
            raise RemoteSessionStateError(
                f"Remote session bridge record {bridge_key!r} was not found."
            )
        logger.info(
            "Resolved remote session bridge record bridge_key=%s node_id=%s output_index=%d age_seconds=%.3f.",
            bridge_key,
            bucket.record.node_id,
            bucket.record.output_index,
            max(0.0, time.time() - bucket.created_at),
        )
        return bucket.record
