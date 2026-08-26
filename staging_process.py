"""Bounded controller-side consumption for remote model-staging processes."""

from __future__ import annotations

import logging
import math
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

if __package__:
    from .remote_protocol import RemoteProtocolError, decode_json_payload
else:  # pragma: no cover - direct debugging imports.
    from remote_protocol import RemoteProtocolError, decode_json_payload

_DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS = 600.0
_STREAM_END = object()
logger = logging.getLogger(__name__)


class RemoteStagingProcessError(RuntimeError):
    """Raised when a remote staging subprocess fails or stops reporting progress."""


def staging_no_progress_timeout_seconds() -> float:
    """Return the bounded interval allowed without a staging JSON event."""
    raw_value = os.getenv("COMFY_MODAL_LLM_STAGE_NO_PROGRESS_TIMEOUT_SECONDS")
    timeout = (
        _DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
        if raw_value is None
        else float(raw_value)
    )
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "COMFY_MODAL_LLM_STAGE_NO_PROGRESS_TIMEOUT_SECONDS must be positive."
        )
    return timeout


def _collect_bounded_stream(stream: Any, chunks: list[bytes]) -> None:
    """Drain diagnostics while retaining at most the most recent MiB."""
    while chunk := stream.read(65536):
        chunks.append(chunk)
        if sum(len(value) for value in chunks) > 1024 * 1024:
            del chunks[:-8]


def _read_progress_lines(stream: Any, output: queue.Queue[Any]) -> None:
    """Read blocking process output without blocking timeout enforcement."""
    try:
        for raw_line in stream:
            output.put(raw_line)
    finally:
        output.put(_STREAM_END)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Boundedly terminate one local transport process."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def terminate_staging_transport(process: subprocess.Popen[bytes]) -> None:
    """Terminate a controller-side SSH transport for one active stager."""
    _stop_process(process)


def _abort_remote_safely(abort_remote: Callable[[], None]) -> None:
    """Attempt remote cleanup without hiding the primary staging failure."""
    try:
        abort_remote()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Unable to clean up the remote model stager: %s", exc)


def _next_progress_line(
    progress_lines: queue.Queue[Any],
    deadline: float,
    timeout_seconds: float,
    provider_label: str,
) -> Any:
    """Return the next line or raise the provider-specific silence failure."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RemoteStagingProcessError(
            f"{provider_label} model staging produced no progress for "
            f"{timeout_seconds:.0f} seconds."
        )
    try:
        return progress_lines.get(timeout=remaining)
    except queue.Empty as exc:
        raise RemoteStagingProcessError(
            f"{provider_label} model staging produced no progress for "
            f"{timeout_seconds:.0f} seconds."
        ) from exc


def _consume_progress_lines(
    progress_lines: queue.Queue[Any],
    payload: dict[str, Any],
    progress_callback: Callable[[dict[str, Any], Mapping[str, Any]], None],
    *,
    provider_label: str,
    timeout_seconds: float,
) -> list[dict[str, Any]] | None:
    """Consume staging JSON until stdout closes, resetting timeout per event."""
    results: list[dict[str, Any]] | None = None
    deadline = time.monotonic() + timeout_seconds
    while True:
        raw_line = _next_progress_line(
            progress_lines,
            deadline,
            timeout_seconds,
            provider_label,
        )
        if raw_line is _STREAM_END:
            return results
        deadline = time.monotonic() + timeout_seconds
        try:
            event = decode_json_payload(raw_line)
        except (RemoteProtocolError, UnicodeDecodeError) as exc:
            raise RemoteStagingProcessError(
                f"{provider_label} model stager returned invalid progress JSON."
            ) from exc
        if event.get("kind") == "progress":
            progress_callback(payload, event)
        elif event.get("kind") == "result" and isinstance(
            event.get("results"), list
        ):
            results = [
                dict(result)
                for result in event["results"]
                if isinstance(result, Mapping)
            ]


def _validated_staging_result(
    process: subprocess.Popen[bytes],
    results: list[dict[str, Any]] | None,
    stderr_chunks: list[bytes],
    provider_label: str,
) -> list[dict[str, Any]]:
    """Wait for a clean process exit and require one terminal result event."""
    try:
        returncode = process.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:
        raise RemoteStagingProcessError(
            f"{provider_label} model stager kept running after closing output."
        ) from exc
    if returncode == 0 and results is not None:
        return results
    diagnostics = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
    raise RemoteStagingProcessError(
        f"{provider_label} model staging failed: "
        f"{diagnostics or f'exit status {returncode}'}."
    )


def consume_staging_process(
    process: subprocess.Popen[bytes],
    payload: dict[str, Any],
    progress_callback: Callable[[dict[str, Any], Mapping[str, Any]], None],
    *,
    provider_label: str,
    abort_remote: Callable[[], None],
    timeout_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Consume JSON events and abort local plus remote work after inactivity."""
    if process.stdout is None or process.stderr is None:
        _stop_process(process)
        _abort_remote_safely(abort_remote)
        raise RemoteStagingProcessError(
            f"{provider_label} model stager did not expose output streams."
        )
    try:
        resolved_timeout = (
            staging_no_progress_timeout_seconds()
            if timeout_seconds is None
            else timeout_seconds
        )
        if not math.isfinite(resolved_timeout) or resolved_timeout <= 0:
            raise ValueError("Staging no-progress timeout must be positive.")
    except ValueError:
        _stop_process(process)
        _abort_remote_safely(abort_remote)
        raise
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(
        target=_collect_bounded_stream,
        args=(process.stderr, stderr_chunks),
        name=f"{provider_label.lower()}-stager-stderr",
        daemon=True,
    )
    progress_lines: queue.Queue[Any] = queue.Queue()
    stdout_thread = threading.Thread(
        target=_read_progress_lines,
        args=(process.stdout, progress_lines),
        name=f"{provider_label.lower()}-stager-stdout",
        daemon=True,
    )
    stderr_thread.start()
    stdout_thread.start()
    completed = False
    try:
        results = _consume_progress_lines(
            progress_lines,
            payload,
            progress_callback,
            provider_label=provider_label,
            timeout_seconds=resolved_timeout,
        )
        stderr_thread.join(timeout=1.0)
        validated_results = _validated_staging_result(
            process,
            results,
            stderr_chunks,
            provider_label,
        )
        completed = True
        return validated_results
    finally:
        if not completed:
            _stop_process(process)
            _abort_remote_safely(abort_remote)
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)


__all__ = [
    "RemoteStagingProcessError",
    "consume_staging_process",
    "staging_no_progress_timeout_seconds",
    "terminate_staging_transport",
]
