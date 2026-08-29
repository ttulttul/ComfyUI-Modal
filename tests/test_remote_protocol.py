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
