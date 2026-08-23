"""Small process supervisor for a direct Vast ComfyUI worker and watchdog."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_RUN_ROOT = Path("/run/comfy-remote")
DEFAULT_LOG_ROOT = Path("/storage/logs")


@dataclass(frozen=True)
class ManagedProcess:
    """Describe one supervised process and its durable diagnostics."""

    name: str
    argv: tuple[str, ...]
    pid_path: Path
    log_path: Path


def process_is_running(pid_path: Path) -> bool:
    """Return whether a validated PID file identifies a live process."""
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        return False
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_process(process: ManagedProcess) -> int:
    """Start one process when absent and atomically publish its PID."""
    if process_is_running(process.pid_path):
        return int(process.pid_path.read_text(encoding="ascii").strip())
    process.pid_path.parent.mkdir(parents=True, exist_ok=True)
    process.log_path.parent.mkdir(parents=True, exist_ok=True)
    with process.log_path.open("ab", buffering=0) as log_file:
        child = subprocess.Popen(
            process.argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    temporary_path = process.pid_path.with_suffix(".tmp")
    temporary_path.write_text(f"{child.pid}\n", encoding="ascii")
    os.replace(temporary_path, process.pid_path)
    logger.info("Started Vast %s pid=%d.", process.name, child.pid)
    return child.pid


def stop_process(process: ManagedProcess, *, timeout_seconds: float = 15.0) -> bool:
    """Stop one exact supervised process group when running."""
    try:
        pid = int(process.pid_path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        process.pid_path.unlink(missing_ok=True)
        return False
    if pid <= 1:
        process.pid_path.unlink(missing_ok=True)
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        process.pid_path.unlink(missing_ok=True)
        return False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_is_running(process.pid_path):
            process.pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.pid_path.unlink(missing_ok=True)
    logger.warning("Killed unresponsive Vast %s pid=%d.", process.name, pid)
    return True


def managed_processes(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    log_root: Path = DEFAULT_LOG_ROOT,
    watchdog_state_path: Path,
) -> tuple[ManagedProcess, ManagedProcess]:
    """Return the fixed worker and watchdog process specifications."""
    return (
        ManagedProcess(
            name="worker",
            argv=(sys.executable, "-m", "remote.ssh_worker", "serve"),
            pid_path=run_root / "vast-worker.pid",
            log_path=log_root / "vast-worker.log",
        ),
        ManagedProcess(
            name="watchdog",
            argv=(
                sys.executable,
                "-m",
                "remote.vast_watchdog",
                "--state",
                str(watchdog_state_path),
            ),
            pid_path=run_root / "vast-watchdog.pid",
            log_path=log_root / "vast-watchdog.log",
        ),
    )


def status_payload(processes: Sequence[ManagedProcess]) -> dict[str, object]:
    """Return a JSON-compatible liveness summary."""
    return {
        process.name: {
            "running": process_is_running(process.pid_path),
            "pid_path": str(process.pid_path),
            "log_path": str(process.log_path),
        }
        for process in processes
    }


def _parser() -> argparse.ArgumentParser:
    """Return the supervisor command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "restart-worker", "stop"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--watchdog-state",
        type=Path,
        default=Path("/storage/comfy-vast-watchdog.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Start, inspect, restart, or stop the direct Vast runtime processes."""
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    processes = managed_processes(
        run_root=arguments.run_root,
        log_root=arguments.log_root,
        watchdog_state_path=arguments.watchdog_state,
    )
    worker, watchdog = processes
    if arguments.command == "start":
        start_process(worker)
        start_process(watchdog)
    elif arguments.command == "restart-worker":
        stop_process(worker)
        start_process(worker)
    elif arguments.command == "stop":
        stop_process(worker)
        stop_process(watchdog)
    print(json.dumps(status_payload(processes), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - remote process entrypoint.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOG_ROOT",
    "DEFAULT_RUN_ROOT",
    "ManagedProcess",
    "managed_processes",
    "process_is_running",
    "start_process",
    "status_payload",
    "stop_process",
]
