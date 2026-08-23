"""Tests for the Vast direct-worker process supervisor."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any


def test_managed_processes_use_isolated_pid_and_log_files(
    vast_supervisor_module: Any,
    tmp_path: Path,
) -> None:
    """Worker and watchdog specifications must not share durable diagnostics."""
    processes = vast_supervisor_module.managed_processes(
        run_root=tmp_path / "run",
        log_root=tmp_path / "logs",
        watchdog_state_path=tmp_path / "state.json",
    )

    assert [process.name for process in processes] == ["worker", "watchdog"]
    assert processes[0].pid_path != processes[1].pid_path
    assert processes[0].log_path != processes[1].log_path
    assert processes[1].argv[-1] == str(tmp_path / "state.json")


def test_start_process_is_idempotent(
    vast_supervisor_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An already-live PID must be returned without launching another process."""
    process = vast_supervisor_module.ManagedProcess(
        name="worker",
        argv=("python", "worker.py"),
        pid_path=tmp_path / "worker.pid",
        log_path=tmp_path / "worker.log",
    )
    process.pid_path.write_text("4321\n", encoding="ascii")
    monkeypatch.setattr(vast_supervisor_module, "process_is_running", lambda _path: True)
    monkeypatch.setattr(
        vast_supervisor_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("launched")),
    )

    assert vast_supervisor_module.start_process(process) == 4321


def test_stop_process_signals_exact_process_group(
    vast_supervisor_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Stopping uses the recorded process group and removes its PID file."""
    process = vast_supervisor_module.ManagedProcess(
        name="worker",
        argv=("python", "worker.py"),
        pid_path=tmp_path / "worker.pid",
        log_path=tmp_path / "worker.log",
    )
    process.pid_path.write_text("9876\n", encoding="ascii")
    running = iter((True, False))
    monkeypatch.setattr(
        vast_supervisor_module,
        "process_is_running",
        lambda _path: next(running),
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        vast_supervisor_module.os,
        "killpg",
        lambda pid, requested_signal: signals.append((pid, requested_signal)),
    )
    monkeypatch.setattr(vast_supervisor_module.time, "sleep", lambda _seconds: None)

    assert vast_supervisor_module.stop_process(process) is True
    assert signals == [(9876, signal.SIGTERM)]
    assert not process.pid_path.exists()
