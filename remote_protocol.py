"""Length-prefixed binary protocol shared by SSH controller and workers."""

from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import Any, BinaryIO, Callable, Mapping

REMOTE_PROTOCOL_MAGIC = b"CRMTRPC1"
REMOTE_PROTOCOL_VERSION = 1
BOUNDARY_INPUT_SIGNATURES_KEY = "__comfy_modal_boundary_input_signatures__"
PRIMITIVE_WIDGET_INPUT_TYPES = frozenset({"INT", "FLOAT", "BOOLEAN", "STRING"})
_FRAME_HEADER = struct.Struct(">8sBQ")
_MAX_FRAME_BYTES = 16 * 1024**3
_FRAME_IO_CHUNK_BYTES = 1024 * 1024


class RemoteFrameKind(IntEnum):
    """Identify one frame carried between the controller and a worker."""

    REQUEST = 1
    INPUTS = 2
    PROGRESS = 3
    RESULT = 4
    ERROR = 5
    CANCEL = 6
    ACKNOWLEDGEMENT = 7
    RUNTIME_INFO = 8


FrameProgressCallback = Callable[[RemoteFrameKind, int, int], None]


class RemoteProtocolError(RuntimeError):
    """Raised when a remote protocol stream is malformed or truncated."""


def encode_frame(kind: RemoteFrameKind, payload: bytes) -> bytes:
    """Encode one complete binary protocol frame."""
    if len(payload) > _MAX_FRAME_BYTES:
        raise RemoteProtocolError(
            f"Remote protocol frame exceeds {_MAX_FRAME_BYTES} bytes."
        )
    return _FRAME_HEADER.pack(REMOTE_PROTOCOL_MAGIC, int(kind), len(payload)) + payload


def write_frame(
    stream: BinaryIO,
    kind: RemoteFrameKind,
    payload: bytes,
    *,
    progress_callback: FrameProgressCallback | None = None,
) -> None:
    """Write and flush one frame in bounded chunks with optional byte progress."""
    if len(payload) > _MAX_FRAME_BYTES:
        raise RemoteProtocolError(
            f"Remote protocol frame exceeds {_MAX_FRAME_BYTES} bytes."
        )
    stream.write(_FRAME_HEADER.pack(REMOTE_PROTOCOL_MAGIC, int(kind), len(payload)))
    transferred_bytes = 0
    payload_view = memoryview(payload)
    while transferred_bytes < len(payload):
        next_offset = min(
            transferred_bytes + _FRAME_IO_CHUNK_BYTES,
            len(payload),
        )
        stream.write(payload_view[transferred_bytes:next_offset])
        transferred_bytes = next_offset
        if progress_callback is not None:
            progress_callback(kind, transferred_bytes, len(payload))
    stream.flush()


def read_frame(
    stream: BinaryIO,
    *,
    progress_callback: FrameProgressCallback | None = None,
) -> tuple[RemoteFrameKind, bytes] | None:
    """Read one complete frame, returning ``None`` at a clean end of stream."""
    header = _read_exact(stream, _FRAME_HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    kind, payload_length = _decode_frame_header(header)
    payload = _read_exact(
        stream,
        payload_length,
        allow_clean_eof=False,
        progress_callback=(
            lambda current, total: progress_callback(kind, current, total)
            if progress_callback is not None
            else None
        ),
    )
    if payload is None:
        raise RemoteProtocolError("Remote protocol payload was truncated.")
    return kind, payload


def relay_frame(
    source: BinaryIO,
    write_chunk: Callable[[bytes | memoryview], Any],
) -> RemoteFrameKind | None:
    """Relay one frame without buffering its potentially large payload."""
    header = _read_exact(source, _FRAME_HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    kind, payload_length = _decode_frame_header(header)
    _write_relay_chunk(write_chunk, header)
    remaining = payload_length
    while remaining:
        chunk = source.read(min(remaining, _FRAME_IO_CHUNK_BYTES))
        if not chunk:
            raise RemoteProtocolError(
                f"Remote protocol stream ended with {remaining} bytes outstanding."
            )
        _write_relay_chunk(write_chunk, chunk)
        remaining -= len(chunk)
    return kind


def _write_relay_chunk(
    write_chunk: Callable[[bytes | memoryview], Any],
    chunk: bytes,
) -> None:
    """Write one relay chunk completely for both stream writes and socket sendall."""
    chunk_view = memoryview(chunk)
    offset = 0
    while offset < len(chunk_view):
        written = write_chunk(chunk_view[offset:])
        if written is None:
            return
        if not isinstance(written, int) or written <= 0:
            raise RemoteProtocolError("Remote protocol relay could not write its frame.")
        offset += written


def encode_json_frame(kind: RemoteFrameKind, payload: Mapping[str, Any]) -> bytes:
    """Encode one canonical JSON protocol frame."""
    serialized = json.dumps(
        dict(payload),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encode_frame(kind, serialized)


def decode_json_payload(payload: bytes) -> dict[str, Any]:
    """Decode one protocol JSON payload into a mapping."""
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteProtocolError("Remote protocol JSON payload is invalid.") from exc
    if not isinstance(decoded, dict):
        raise RemoteProtocolError("Remote protocol JSON payload must be an object.")
    return decoded


def _read_exact(
    stream: BinaryIO,
    length: int,
    *,
    allow_clean_eof: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> bytes | None:
    """Read exactly ``length`` bytes from a binary stream."""
    if length == 0:
        return b""
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(min(remaining, _FRAME_IO_CHUNK_BYTES))
        if not chunk:
            if allow_clean_eof and remaining == length:
                return None
            raise RemoteProtocolError(
                f"Remote protocol stream ended with {remaining} bytes outstanding."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
        if progress_callback is not None:
            progress_callback(length - remaining, length)
    return b"".join(chunks)


def _decode_frame_header(header: bytes) -> tuple[RemoteFrameKind, int]:
    """Validate one frame header and return its kind and payload length."""
    magic, raw_kind, payload_length = _FRAME_HEADER.unpack(header)
    if magic != REMOTE_PROTOCOL_MAGIC:
        raise RemoteProtocolError("Remote protocol frame has an invalid magic value.")
    try:
        kind = RemoteFrameKind(raw_kind)
    except ValueError as exc:
        raise RemoteProtocolError(f"Unknown remote frame kind {raw_kind}.") from exc
    if payload_length > _MAX_FRAME_BYTES:
        raise RemoteProtocolError(
            f"Remote protocol payload length {payload_length} exceeds the safety limit."
        )
    return kind, payload_length
