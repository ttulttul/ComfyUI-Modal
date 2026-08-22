"""Tests for persistent SSH worker state and relay behavior."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


def test_top_level_worker_entrypoint_reports_runtime_info() -> None:
    """The OCI entrypoint must import when the repository is a top-level path."""
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    environment["COMFY_MODAL_RUNTIME_FINGERPRINT"] = "test-fingerprint"

    completed = subprocess.run(
        [sys.executable, "-m", "remote.ssh_worker", "runtime-info"],
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
        timeout=15,
    )

    runtime_info = json.loads(completed.stdout)
    assert runtime_info["runtime_fingerprint"] == "test-fingerprint"
    assert runtime_info["protocol_version"] > 0


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
