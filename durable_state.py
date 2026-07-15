"""Durable invocation metadata and content-addressed object storage."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

_DURABLE_OBJECT_REF_MARKER = "__comfy_modal_durable_object_ref__"
_REMOTE_INVOCATION_RECORD_MARKER = "__comfy_modal_remote_invocation_record__"
_REMOTE_INVOCATION_STATES = frozenset({"running", "completed", "failed"})


class DurableStateError(RuntimeError):
    """Raised when durable execution state is missing, invalid, or corrupted."""


@dataclass(frozen=True)
class DurableObjectRef:
    """Reference one content-addressed binary object below a configured root."""

    object_path: str
    sha256: str
    size_bytes: int

    def to_payload(self) -> dict[str, Any]:
        """Serialize this reference into a persistence-safe mapping."""
        return {
            _DURABLE_OBJECT_REF_MARKER: True,
            "object_path": self.object_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DurableObjectRef":
        """Deserialize and validate one persisted object reference."""
        object_path = str(payload.get("object_path") or "").strip()
        sha256 = str(payload.get("sha256") or "").strip()
        size_bytes = payload.get("size_bytes")
        if not object_path or len(sha256) != 64:
            raise DurableStateError("Durable object refs require object_path and SHA256.")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise DurableStateError("Durable object refs require a non-negative size_bytes.")
        return cls(
            object_path=object_path,
            sha256=sha256,
            size_bytes=size_bytes,
        )


def is_durable_object_ref_payload(payload: Any) -> bool:
    """Return whether one value looks like a durable object reference."""
    return isinstance(payload, Mapping) and bool(payload.get(_DURABLE_OBJECT_REF_MARKER))


class FileDurableObjectStore:
    """Filesystem-backed content-addressed object store with integrity checks."""

    def __init__(
        self,
        root: Path,
        *,
        commit_callback: Callable[[], Any] | None = None,
        reload_callback: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize the object root and optional mounted-volume callbacks."""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._commit_callback = commit_callback
        self._reload_callback = reload_callback
        self._write_lock = threading.Lock()

    def put(self, namespace: str, payload: bytes) -> DurableObjectRef:
        """Persist bytes under a content-addressed namespace and return their ref."""
        normalized_namespace = self._normalize_namespace(namespace)
        sha256 = hashlib.sha256(payload).hexdigest()
        relative_path = Path(normalized_namespace) / sha256[:2] / f"{sha256}.bin"
        target_path = self.root / relative_path
        wrote_object = False
        with self._write_lock:
            if target_path.exists():
                self._validate_existing_object(target_path, payload, sha256)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = target_path.with_name(
                    f".{target_path.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary_path.open("xb") as output_file:
                        output_file.write(payload)
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    os.replace(temporary_path, target_path)
                    wrote_object = True
                finally:
                    temporary_path.unlink(missing_ok=True)
        if wrote_object and self._commit_callback is not None:
            self._commit_callback()
        return DurableObjectRef(
            object_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=len(payload),
        )

    def get(self, object_ref: DurableObjectRef) -> bytes:
        """Load and validate one referenced object."""
        target_path = self._resolve_object_path(object_ref.object_path)
        try:
            payload = target_path.read_bytes()
        except FileNotFoundError as exc:
            if self._reload_callback is not None:
                self._reload_callback()
                try:
                    payload = target_path.read_bytes()
                except FileNotFoundError:
                    pass
                else:
                    return self._validate_loaded_object(object_ref, payload)
            raise DurableStateError(
                f"Durable object {object_ref.object_path!r} was not found."
            ) from exc
        return self._validate_loaded_object(object_ref, payload)

    def _validate_loaded_object(
        self,
        object_ref: DurableObjectRef,
        payload: bytes,
    ) -> bytes:
        """Validate and return bytes loaded for one durable reference."""
        if len(payload) != object_ref.size_bytes:
            raise DurableStateError(
                f"Durable object {object_ref.object_path!r} size mismatch: "
                f"expected {object_ref.size_bytes}, got {len(payload)}."
            )
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != object_ref.sha256:
            raise DurableStateError(
                f"Durable object {object_ref.object_path!r} SHA256 mismatch."
            )
        return payload

    def _resolve_object_path(self, object_path: str) -> Path:
        """Resolve one safe relative object path beneath the configured root."""
        relative_path = Path(object_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise DurableStateError(f"Unsafe durable object path {object_path!r}.")
        resolved_path = (self.root / relative_path).resolve()
        if self.root not in resolved_path.parents:
            raise DurableStateError(f"Durable object path escapes its root: {object_path!r}.")
        return resolved_path

    def _normalize_namespace(self, namespace: str) -> str:
        """Return a safe single-directory object namespace."""
        normalized = str(namespace or "").strip()
        if not normalized or Path(normalized).name != normalized or normalized in {".", ".."}:
            raise DurableStateError(f"Invalid durable object namespace {namespace!r}.")
        return normalized

    def _validate_existing_object(
        self,
        target_path: Path,
        payload: bytes,
        expected_sha256: str,
    ) -> None:
        """Ensure a pre-existing content-addressed object is not corrupted."""
        existing_payload = target_path.read_bytes()
        if len(existing_payload) != len(payload):
            raise DurableStateError(
                f"Existing durable object {target_path} has an unexpected size."
            )
        if hashlib.sha256(existing_payload).hexdigest() != expected_sha256:
            raise DurableStateError(
                f"Existing durable object {target_path} failed its content-address check."
            )


@dataclass(frozen=True)
class RemoteInvocationRecord:
    """Durable lifecycle and optional result for one logical remote invocation."""

    invocation_id: str
    state: str
    attempt: int
    created_at: float
    updated_at: float
    result_inline: bytes | None = None
    result_object: DurableObjectRef | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize this record into a Modal Dict-safe mapping."""
        return {
            _REMOTE_INVOCATION_RECORD_MARKER: True,
            "invocation_id": self.invocation_id,
            "state": self.state,
            "attempt": self.attempt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_inline": self.result_inline,
            "result_object": (
                self.result_object.to_payload()
                if self.result_object is not None
                else None
            ),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoteInvocationRecord":
        """Deserialize and validate one durable invocation record."""
        invocation_id = str(payload.get("invocation_id") or "").strip()
        state = str(payload.get("state") or "").strip()
        attempt = payload.get("attempt")
        if not invocation_id or state not in _REMOTE_INVOCATION_STATES:
            raise DurableStateError("Remote invocation records require a valid id and state.")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise DurableStateError("Remote invocation records require a positive attempt.")
        result_inline = payload.get("result_inline")
        if result_inline is not None and not isinstance(result_inline, bytes | bytearray):
            raise DurableStateError("Inline invocation results must be bytes.")
        result_object_payload = payload.get("result_object")
        result_object = (
            DurableObjectRef.from_payload(result_object_payload)
            if isinstance(result_object_payload, Mapping)
            else None
        )
        return cls(
            invocation_id=invocation_id,
            state=state,
            attempt=attempt,
            created_at=float(payload.get("created_at") or 0.0),
            updated_at=float(payload.get("updated_at") or 0.0),
            result_inline=(bytes(result_inline) if result_inline is not None else None),
            result_object=result_object,
            error_type=(
                str(payload["error_type"])
                if payload.get("error_type") is not None
                else None
            ),
            error_message=(
                str(payload["error_message"])
                if payload.get("error_message") is not None
                else None
            ),
        )


class InMemoryRemoteInvocationStore:
    """Thread-safe in-memory invocation store used by local mode and tests."""

    def __init__(self) -> None:
        """Initialize an empty record mapping."""
        self._lock = threading.Lock()
        self._records: dict[str, RemoteInvocationRecord] = {}

    def get_record(self, invocation_id: str) -> RemoteInvocationRecord | None:
        """Return one invocation record when present."""
        with self._lock:
            return self._records.get(invocation_id)

    def put_record(self, record: RemoteInvocationRecord) -> None:
        """Store or replace one invocation record."""
        with self._lock:
            self._records[record.invocation_id] = record


def stable_remote_invocation_id(
    payload: Mapping[str, Any],
    kwargs_payload: bytes,
) -> str:
    """Return a stable idempotency key for one logical payload and input blob."""
    canonical_payload = {
        str(key): value
        for key, value in payload.items()
        if str(key) != "invocation_id"
    }
    encoded_payload = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(encoded_payload)
    digest.update(b"\0")
    digest.update(kwargs_payload)
    return f"RIV_{digest.hexdigest()}"


def new_running_invocation_record(
    invocation_id: str,
    previous_record: RemoteInvocationRecord | None,
) -> RemoteInvocationRecord:
    """Create the next running attempt while preserving original creation time."""
    now = time.time()
    return RemoteInvocationRecord(
        invocation_id=invocation_id,
        state="running",
        attempt=(previous_record.attempt + 1 if previous_record is not None else 1),
        created_at=(previous_record.created_at if previous_record is not None else now),
        updated_at=now,
    )
