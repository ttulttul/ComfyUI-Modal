"""Tests for persistent SSH worker state and relay behavior."""

from __future__ import annotations

import io
import socket
from typing import Any


def test_worker_execution_state_registers_and_cancels(ssh_worker_module: Any) -> None:
    """Cancellation should target a stable invocation identity."""
    state = ssh_worker_module.WorkerExecutionState()

    cancellation = state.register("RIV_test")

    assert state.cancel("RIV_test") is True
    assert cancellation.is_set()
    state.unregister("RIV_test")
    assert state.cancel("RIV_test") is False


def test_worker_request_relay_preserves_framed_binary_payload(
    ssh_worker_module: Any,
    remote_protocol_module: Any,
) -> None:
    """The docker-exec relay must preserve request and input frames byte-for-byte."""
    request = remote_protocol_module.encode_json_frame(
        remote_protocol_module.RemoteFrameKind.REQUEST,
        {"invocation_id": "RIV_test", "payload": {"payload_kind": "subgraph"}},
    )
    inputs = remote_protocol_module.encode_frame(
        remote_protocol_module.RemoteFrameKind.INPUTS,
        b"\x00\xfftensor-bytes",
    )
    left, right = socket.socketpair()
    try:
        ssh_worker_module._copy_request_frames(io.BytesIO(request + inputs), left)
        received = right.makefile("rb")
        try:
            assert remote_protocol_module.read_frame(received)[0] is remote_protocol_module.RemoteFrameKind.REQUEST
            assert remote_protocol_module.read_frame(received) == (
                remote_protocol_module.RemoteFrameKind.INPUTS,
                b"\x00\xfftensor-bytes",
            )
        finally:
            received.close()
    finally:
        left.close()
        right.close()
