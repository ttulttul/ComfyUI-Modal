"""Tests for persistent SSH worker state and relay behavior."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
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


def test_worker_stage_profiles_streams_progress_and_result(
    ssh_worker_module: Any,
    llm_staging_module: Any,
    monkeypatch: Any,
    capsys: Any,
    tmp_path: Path,
) -> None:
    """The SSH staging command should expose machine-readable progress and metadata."""

    def stage(model_references: list[str], storage_root: Path, **kwargs: Any) -> Any:
        """Emit one progress update and return one immutable profile result."""
        assert model_references == ["owner/model"]
        assert storage_root == tmp_path
        kwargs["progress_callback"](
            SimpleNamespace(
                stage="download",
                message="Fetching files",
                value=1,
                maximum=2,
                unit="files",
                indeterminate=False,
                model_reference="owner/model",
            )
        )
        return [
            SimpleNamespace(
                to_dict=lambda: {
                    "requested_reference": "owner/model",
                    "profile_id": "hf-" + "b" * 64,
                    "revision": "8" * 40,
                }
            )
        ]

    monkeypatch.setattr(llm_staging_module, "resolve_and_stage_model_references", stage)

    results = ssh_worker_module.stage_profiles(["owner/model"], tmp_path)

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["kind"] == "progress"
    assert events[0]["max"] == 2
    assert events[0]["model_reference"] == "owner/model"
    assert events[1] == {"kind": "result", "results": results}


def test_worker_cancels_only_matching_owned_stager(
    ssh_worker_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The cancellation command should validate identity before killing a PID."""
    owner_id = "vast:RIV_test:owner"
    pid = 12345
    owner_path = ssh_worker_module._staging_owner_path(tmp_path, owner_id)
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text(
        json.dumps(
            {
                "owner_id": owner_id,
                "pid": pid,
                "process_start": "start-tick",
            }
        ),
        encoding="utf-8",
    )
    lease_path = tmp_path / "llm_models" / "repo" / ".revision.download.lock"
    lease_path.parent.mkdir(parents=True)
    lease_path.write_text(
        json.dumps(
            {
                "owner_id": owner_id,
                "pid": pid,
                "process_start": "start-tick",
                "token": "lease-token",
            }
        ),
        encoding="utf-8",
    )
    starts = iter(("start-tick", None))
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        ssh_worker_module,
        "_linux_process_start",
        lambda _pid: next(starts),
    )
    monkeypatch.setattr(
        ssh_worker_module.Path,
        "read_bytes",
        lambda _path: (
            b"python\0-m\0remote.ssh_worker\0stage-profiles\0--owner-id\0"
            + owner_id.encode("utf-8")
            + b"\0"
        ),
    )
    monkeypatch.setattr(
        ssh_worker_module.os,
        "kill",
        lambda target_pid, sent_signal: killed.append((target_pid, sent_signal)),
    )

    assert ssh_worker_module.cancel_staging_process(tmp_path, owner_id) is True

    assert not owner_path.exists()
    assert not lease_path.exists()
    assert killed == [(pid, ssh_worker_module.signal.SIGTERM)]


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


def test_cgroup_memory_snapshot_reads_v2_limits_and_events(
    ssh_worker_module: Any,
    tmp_path: Path,
) -> None:
    """The relay should retain the cgroup evidence needed to identify an OOM."""
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 42\noom 3\noom_kill 2\n",
        encoding="ascii",
    )
    (tmp_path / "memory.current").write_text("1073741824\n", encoding="ascii")
    (tmp_path / "memory.max").write_text("94623498240\n", encoding="ascii")
    (tmp_path / "memory.swap.max").write_text("0\n", encoding="ascii")

    snapshot = ssh_worker_module.read_cgroup_memory_snapshot(tmp_path)

    assert snapshot.oom == 3
    assert snapshot.oom_kill == 2
    assert snapshot.memory_current_bytes == 1024**3
    assert snapshot.memory_limit_bytes == 94623498240
    assert snapshot.swap_limit_bytes == 0


def test_worker_failure_payload_identifies_oom_and_includes_log_tail(
    ssh_worker_module: Any,
    tmp_path: Path,
) -> None:
    """A cgroup OOM delta should become the primary user-facing failure."""
    worker_log = tmp_path / "vast-worker.log"
    worker_log.write_text(
        "loaded text encoder\nRequested to load MiniMaxH3\n",
        encoding="utf-8",
    )
    before = ssh_worker_module.CgroupMemorySnapshot(oom=0, oom_kill=0)
    after = ssh_worker_module.CgroupMemorySnapshot(
        oom=1,
        oom_kill=1,
        memory_limit_bytes=94623498240,
        swap_limit_bytes=0,
    )

    payload = ssh_worker_module.worker_failure_payload(
        before,
        after,
        worker_log_path=worker_log,
    )

    assert payload["error_type"] == "WorkerOutOfMemoryError"
    assert payload["failure_kind"] == "out_of_memory"
    assert payload["oom_kill_delta"] == 1
    assert "container cgroup OOM" in payload["message"]
    assert "88.1 GiB" in payload["message"]
    assert "swap disabled" in payload["message"]
    assert "Requested to load MiniMaxH3" in payload["message"]


def test_worker_failure_payload_distinguishes_non_oom_process_loss(
    ssh_worker_module: Any,
    tmp_path: Path,
) -> None:
    """A vanished worker without an OOM delta should be reported as a crash."""
    snapshot = ssh_worker_module.CgroupMemorySnapshot(oom=4, oom_kill=2)

    payload = ssh_worker_module.worker_failure_payload(
        snapshot,
        snapshot,
        relay_error="ConnectionResetError: reset by peer",
        worker_log_path=tmp_path / "missing.log",
    )

    assert payload["error_type"] == "WorkerProcessLostError"
    assert payload["failure_kind"] == "worker_process_lost"
    assert "OOM counters did not increase" in payload["message"]
    assert "reset by peer" in payload["message"]
