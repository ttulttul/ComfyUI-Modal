"""Tests for the generic remote worker framing protocol."""

from __future__ import annotations

import io
from typing import Any

import pytest


def test_binary_frame_round_trip_preserves_arbitrary_bytes(remote_protocol_module: Any) -> None:
    """Binary tensor/result payloads must not pass through base64 or text."""
    payload = bytes(range(256)) * 1024
    encoded = remote_protocol_module.encode_frame(
        remote_protocol_module.RemoteFrameKind.RESULT,
        payload,
    )

    decoded = remote_protocol_module.read_frame(io.BytesIO(encoded))

    assert decoded == (remote_protocol_module.RemoteFrameKind.RESULT, payload)
    assert len(encoded) == len(payload) + 17


def test_json_frames_use_canonical_object_payloads(remote_protocol_module: Any) -> None:
    """Control frames should round-trip deterministic JSON objects."""
    encoded = remote_protocol_module.encode_json_frame(
        remote_protocol_module.RemoteFrameKind.CANCEL,
        {"invocation_id": "RIV_test", "reason": "user"},
    )
    kind, payload = remote_protocol_module.read_frame(io.BytesIO(encoded))

    assert kind is remote_protocol_module.RemoteFrameKind.CANCEL
    assert remote_protocol_module.decode_json_payload(payload) == {
        "invocation_id": "RIV_test",
        "reason": "user",
    }


def test_truncated_frame_is_rejected(remote_protocol_module: Any) -> None:
    """A transport disconnect must not be mistaken for a valid empty result."""
    encoded = remote_protocol_module.encode_frame(
        remote_protocol_module.RemoteFrameKind.INPUTS,
        b"abcdef",
    )

    with pytest.raises(remote_protocol_module.RemoteProtocolError, match="outstanding"):
        remote_protocol_module.read_frame(io.BytesIO(encoded[:-2]))


def test_frame_io_reports_incremental_payload_bytes(remote_protocol_module: Any) -> None:
    """Large framed transfers should expose byte progress in bounded chunks."""
    payload = b"x" * (2 * 1024 * 1024 + 17)
    encoded_stream = io.BytesIO()
    written: list[tuple[Any, int, int]] = []
    remote_protocol_module.write_frame(
        encoded_stream,
        remote_protocol_module.RemoteFrameKind.RESULT,
        payload,
        progress_callback=lambda kind, current, total: written.append(
            (kind, current, total)
        ),
    )
    read: list[tuple[Any, int, int]] = []

    decoded = remote_protocol_module.read_frame(
        io.BytesIO(encoded_stream.getvalue()),
        progress_callback=lambda kind, current, total: read.append(
            (kind, current, total)
        ),
    )

    assert decoded == (remote_protocol_module.RemoteFrameKind.RESULT, payload)
    assert [current for _kind, current, _total in written] == [
        1024 * 1024,
        2 * 1024 * 1024,
        len(payload),
    ]
    assert read == written


def test_relay_frame_streams_without_reassembling_payload(
    remote_protocol_module: Any,
) -> None:
    """The SSH relay should forward large frames a chunk at a time."""
    payload = b"z" * (2 * 1024 * 1024 + 7)
    encoded = remote_protocol_module.encode_frame(
        remote_protocol_module.RemoteFrameKind.INPUTS,
        payload,
    )
    forwarded_chunks: list[bytes] = []

    kind = remote_protocol_module.relay_frame(
        io.BytesIO(encoded),
        lambda chunk: forwarded_chunks.append(bytes(chunk)),
    )

    assert kind is remote_protocol_module.RemoteFrameKind.INPUTS
    assert b"".join(forwarded_chunks) == encoded
    assert len(forwarded_chunks) == 4


def test_protocol_exports_shared_prompt_boundary_constants(
    remote_protocol_module: Any,
) -> None:
    """Local and cloud runtimes must share exact prompt payload wire keys."""
    assert (
        remote_protocol_module.BOUNDARY_INPUT_SIGNATURES_KEY
        == "__comfy_modal_boundary_input_signatures__"
    )
    assert remote_protocol_module.PRIMITIVE_WIDGET_INPUT_TYPES == frozenset(
        {"INT", "FLOAT", "BOOLEAN", "STRING"}
    )
