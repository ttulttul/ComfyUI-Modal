"""Durable invocation metadata and content-addressed object storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

_DURABLE_OBJECT_REF_MARKER = "__comfy_modal_durable_object_ref__"
_REMOTE_INVOCATION_RECORD_MARKER = "__comfy_modal_remote_invocation_record__"
_REMOTE_INVOCATION_STATES = frozenset({"running", "completed", "failed"})

logger = logging.getLogger(__name__)


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


@dataclass
class DurableObjectCommitBatch:
    """Track whether one logical operation created durable filesystem objects."""

    wrote_object: bool = False

    def absorb(self, other: "DurableObjectCommitBatch") -> None:
        """Merge pending object writes from another execution context."""
        self.wrote_object = self.wrote_object or other.wrote_object
        other.wrote_object = False


class FileDurableObjectStore:
    """Filesystem-backed content-addressed object store with integrity checks."""

    def __init__(
        self,
        root: Path,
        *,
        commit_callback: Callable[[], Any] | None = None,
        committed_read_callback: Callable[[str], bytes] | None = None,
    ) -> None:
        """Initialize the object root and optional mounted-volume callbacks."""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._commit_callback = commit_callback
        self._committed_read_callback = committed_read_callback
        self._write_lock = threading.Lock()
        self._commit_batch_state = threading.local()

    @contextmanager
    def batch_commits(
        self,
        *,
        commit_on_exit: bool = True,
    ) -> Iterator[DurableObjectCommitBatch]:
        """Coalesce object writes into one optional mounted-volume commit."""
        batch = DurableObjectCommitBatch()
        batch_stack = self._commit_batch_stack()
        batch_stack.append(batch)
        try:
            yield batch
        finally:
            popped_batch = batch_stack.pop()
            if popped_batch is not batch:
                raise DurableStateError("Durable object commit batches exited out of order.")
            if batch_stack:
                batch_stack[-1].absorb(batch)
            elif commit_on_exit:
                self.commit_batch(batch)

    def commit_batch(self, batch: DurableObjectCommitBatch) -> None:
        """Commit pending object writes from a completed logical operation once."""
        if not batch.wrote_object:
            return
        if self._commit_callback is not None:
            self._commit_callback()
        batch.wrote_object = False

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
                    with temporary_path.open("xb", buffering=0) as output_file:
                        remaining = memoryview(payload)
                        while remaining:
                            written = output_file.write(remaining)
                            if written is None or written <= 0:
                                raise DurableStateError(
                                    f"Durable object write stalled for {temporary_path}."
                                )
                            remaining = remaining[written:]
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    self._validate_object_file(
                        temporary_path,
                        expected_size=len(payload),
                        expected_sha256=sha256,
                    )
                    os.replace(temporary_path, target_path)
                    wrote_object = True
                finally:
                    temporary_path.unlink(missing_ok=True)
        if wrote_object:
            active_batch = self._active_commit_batch()
            if active_batch is None:
                if self._commit_callback is not None:
                    self._commit_callback()
            else:
                active_batch.wrote_object = True
        logger.info(
            "Resolved durable object namespace=%s path=%s size_bytes=%d created=%s.",
            normalized_namespace,
            relative_path.as_posix(),
            len(payload),
            wrote_object,
        )
        return DurableObjectRef(
            object_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=len(payload),
        )

    def _commit_batch_stack(self) -> list[DurableObjectCommitBatch]:
        """Return this thread's active durable-object commit batch stack."""
        batch_stack = getattr(self._commit_batch_state, "stack", None)
        if batch_stack is None:
            batch_stack = []
            self._commit_batch_state.stack = batch_stack
        return batch_stack

    def _active_commit_batch(self) -> DurableObjectCommitBatch | None:
        """Return this thread's innermost commit batch when one is active."""
        batch_stack = self._commit_batch_stack()
        return batch_stack[-1] if batch_stack else None

    def get(self, object_ref: DurableObjectRef) -> bytes:
        """Load and validate one referenced object."""
        target_path = self._resolve_object_path(object_ref.object_path)
        try:
            payload = target_path.read_bytes()
        except FileNotFoundError as mounted_read_error:
            return self._read_committed_object(
                object_ref,
                reason="absent from the mounted snapshot",
                mounted_error=mounted_read_error,
            )
        try:
            return self._validate_loaded_object(object_ref, payload)
        except DurableStateError as mounted_validation_error:
            mounted_error_message = str(mounted_validation_error)
        del payload
        return self._read_committed_object(
            object_ref,
            reason=f"failed mounted-snapshot validation: {mounted_error_message}",
            mounted_error=DurableStateError(mounted_error_message),
        )

    def _read_committed_object(
        self,
        object_ref: DurableObjectRef,
        *,
        reason: str,
        mounted_error: OSError | DurableStateError,
    ) -> bytes:
        """Read and validate an object through the authoritative committed-store API."""
        if self._committed_read_callback is None:
            if isinstance(mounted_error, DurableStateError):
                raise mounted_error
            raise DurableStateError(
                f"Durable object {object_ref.object_path!r} was not found."
            ) from mounted_error
        logger.warning(
            "Durable object %s %s; reading committed bytes directly.",
            object_ref.object_path,
            reason,
        )
        try:
            payload = self._committed_read_callback(object_ref.object_path)
        except FileNotFoundError as committed_read_error:
            raise DurableStateError(
                f"Durable object {object_ref.object_path!r} was not found."
            ) from committed_read_error
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
        self._validate_object_file(
            target_path,
            expected_size=len(payload),
            expected_sha256=expected_sha256,
        )

    def _validate_object_file(
        self,
        target_path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> None:
        """Validate one object file without loading a second full copy into memory."""
        if target_path.stat().st_size != expected_size:
            raise DurableStateError(
                f"Existing durable object {target_path} has an unexpected size."
            )
        digest = hashlib.sha256()
        with target_path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise DurableStateError(
                f"Existing durable object {target_path} failed its content-address check."
            )


def read_modal_volume_file(volume: Any, volume_path: str) -> bytes:
    """Read one committed file through Modal's direct Volume API."""
    read_file = getattr(volume, "read_file", None)
    if not callable(read_file):
        raise DurableStateError("The configured Modal volume does not support read_file().")
    return b"".join(read_file(volume_path))


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
