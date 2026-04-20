"""Prompt-scoped remote session helpers for split Modal proxy execution."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_REMOTE_SESSION_HANDLE_MARKER = "__comfy_modal_remote_session_handle__"
_REMOTE_SESSION_VALUE_REF_MARKER = "__comfy_modal_remote_session_value_ref__"


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


def is_remote_session_handle_payload(payload: Any) -> bool:
    """Return whether one arbitrary value looks like a session-handle payload."""
    return isinstance(payload, Mapping) and bool(payload.get(_REMOTE_SESSION_HANDLE_MARKER))


def is_remote_session_value_ref_payload(payload: Any) -> bool:
    """Return whether one arbitrary value looks like a session-value-ref payload."""
    return isinstance(payload, Mapping) and bool(payload.get(_REMOTE_SESSION_VALUE_REF_MARKER))


@dataclass
class _RemoteSessionBucket:
    """Hold the live values cached for one prompt-scoped remote session."""

    handle: RemoteSessionHandle
    values: dict[tuple[str, int], Any] = field(default_factory=dict)
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
            self._sessions[handle.session_id].values[(ref.node_id, ref.output_index)] = value
        logger.info(
            "Stored remote session value session_id=%s node_id=%s output_index=%d.",
            ref.session_id,
            ref.node_id,
            ref.output_index,
        )
        return ref

    def get_output(self, ref: RemoteSessionValueRef) -> Any:
        """Return one stored output value or raise when it no longer exists."""
        with self._lock:
            bucket = self._sessions.get(ref.session_id)
            if bucket is None:
                raise RemoteSessionStateError(
                    f"Remote session {ref.session_id!r} was not found."
                )
            value_key = (ref.node_id, ref.output_index)
            if value_key not in bucket.values:
                raise RemoteSessionStateError(
                    "Remote session value was not found for "
                    f"session_id={ref.session_id!r} node_id={ref.node_id!r} "
                    f"output_index={ref.output_index}."
                )
            return bucket.values[value_key]

    def clear_session(self, handle: RemoteSessionHandle) -> None:
        """Drop one prompt-scoped session and every value stored inside it."""
        with self._lock:
            removed = self._sessions.pop(handle.session_id, None)
        if removed is not None:
            logger.info(
                "Cleared remote session session_id=%s prompt_id=%s owner_component_id=%s value_count=%d.",
                handle.session_id,
                removed.handle.prompt_id,
                removed.handle.owner_component_id,
                len(removed.values),
            )

    def resolve_value(self, value: Any) -> Any:
        """Resolve one possible session reference back into the underlying live value."""
        if not is_remote_session_value_ref_payload(value):
            return value
        return self.get_output(RemoteSessionValueRef.from_payload(value))
