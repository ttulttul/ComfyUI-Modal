"""Managed Modal container inventory, termination, and log streaming."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
import importlib
import logging
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
from typing import Any, Callable

from ..settings import (
    MODAL_GPU_TYPES,
    ModalSyncSettings,
    get_settings,
    modal_deployment_app_name,
    settings_for_modal_gpu,
)

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback environments.
    modal = None

_REMOTE_CONTAINER_LOG_STREAMS_LOCK = threading.Lock()
_REMOTE_CONTAINER_LOG_STREAMS: dict[str, "_RemoteContainerLogStreamState"] = {}
_REMOTE_CONTAINER_LOG_STDERR_LOCK = threading.Lock()
_REMOTE_CONTAINER_LOG_STREAM_IDLE_GRACE_SECONDS = 30.0

MODAL_GPU_PRICING_EFFECTIVE_DATE = "2026-08-13"
MODAL_GPU_ESTIMATED_USD_PER_SECOND: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "H100!": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "B200+": 0.001736,
    "B300": 0.001972,
}


def _modal_environment_name() -> str | None:
    """Return the active Modal environment name when explicitly configured."""
    environment_name = os.getenv("MODAL_ENVIRONMENT")
    if environment_name is None:
        return None
    normalized = environment_name.strip()
    return normalized or None


@dataclass
class _RemoteContainerLogLineBuffer:
    """Buffer partial remote log lines so stderr mirroring stays line-oriented."""

    task_id: str
    buffered_text: str = ""


@dataclass
class _RemoteContainerLogStreamState:
    """Track one active local watcher mirroring logs for a Modal container."""

    task_id: str
    stop_event: threading.Event
    thread: threading.Thread
    refcount: int = 0
    idle_stop_timer: threading.Timer | None = None


class ModalContainerStatusError(RuntimeError):
    """Raised when active Modal container status cannot be queried."""



@dataclass(frozen=True)
class ModalContainerStatus:
    """Describe one active Modal container owned by this ComfyUI instance."""

    container_id: str
    app_id: str
    app_name: str
    modal_gpu: str
    estimated_gpu_cost_per_second: float
    state: str
    enqueued_at: float | None
    started_at: float | None

    def as_dict(self) -> dict[str, str | float | None]:
        """Return a JSON-serializable representation for the frontend."""
        return {
            "container_id": self.container_id,
            "app_id": self.app_id,
            "app_name": self.app_name,
            "modal_gpu": self.modal_gpu,
            "estimated_gpu_cost_per_second": self.estimated_gpu_cost_per_second,
            "state": self.state,
            "enqueued_at": self.enqueued_at,
            "started_at": self.started_at,
        }

def _is_remote_container_log_stream_enabled() -> bool:
    """Return whether remote Modal container logs should be mirrored locally."""
    return bool(get_settings().stream_remote_container_logs)


def _coerce_modal_task_id(value: Any) -> str | None:
    """Normalize one streamed Modal task id into a non-empty string."""
    if value is None:
        return None
    task_id = str(value).strip()
    return task_id or None


def _write_remote_container_log_line(
    task_id: str,
    line: str,
    *,
    stream: Any = None,
) -> None:
    """Write one complete remote container log line to the local stderr stream."""
    target_stream = stream if stream is not None else sys.stderr
    with _REMOTE_CONTAINER_LOG_STDERR_LOCK:
        target_stream.write(f"[modal:{task_id}] {line}")
        target_stream.flush()


def _write_remote_container_log_chunk(
    line_buffer: _RemoteContainerLogLineBuffer,
    chunk: str,
    *,
    stream: Any = None,
) -> None:
    """Buffer one remote log chunk and mirror complete lines to local stderr."""
    line_buffer.buffered_text += chunk
    while True:
        newline_index = line_buffer.buffered_text.find("\n")
        if newline_index < 0:
            return
        next_line = line_buffer.buffered_text[: newline_index + 1]
        line_buffer.buffered_text = line_buffer.buffered_text[newline_index + 1 :]
        _write_remote_container_log_line(line_buffer.task_id, next_line, stream=stream)


def _flush_remote_container_log_chunk(
    line_buffer: _RemoteContainerLogLineBuffer,
    *,
    stream: Any = None,
) -> None:
    """Flush any partial remote log text remaining in the local line buffer."""
    if not line_buffer.buffered_text:
        return
    _write_remote_container_log_line(
        line_buffer.task_id,
        f"{line_buffer.buffered_text}\n",
        stream=stream,
    )
    line_buffer.buffered_text = ""


def _managed_modal_app_gpus(settings: ModalSyncSettings) -> dict[str, str]:
    """Map every GPU-specific app name owned by this ComfyUI instance to its GPU."""
    return {
        modal_deployment_app_name(
            settings_for_modal_gpu(settings, modal_gpu)
        ): modal_gpu
        for modal_gpu in MODAL_GPU_TYPES
    }


def _optional_modal_timestamp(value: Any) -> float | None:
    """Normalize an unset Modal protobuf timestamp to ``None``."""
    timestamp = float(value or 0.0)
    return timestamp if timestamp > 0 else None



def _resolved_modal_environment_name() -> str | None:
    """Return the explicitly selected Modal environment, if one is configured."""
    object_module = importlib.import_module("modal._object")
    environments_module = importlib.import_module("modal.environments")
    environment = environments_module.ensure_env(_modal_environment_name())
    environment_name = object_module._get_environment_name(environment)
    if environment_name is None:
        return None
    normalized_environment_name = str(environment_name).strip()
    return normalized_environment_name or None


@lru_cache(maxsize=None)
def _synchronized_modal_callable(
    async_callable: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap one async Modal SDK operation on Modal's managed event loop."""
    async_utils_module = importlib.import_module("modal._utils.async_utils")
    return async_utils_module.synchronize_api(async_callable)


async def _list_modal_tasks_on_sdk_loop(
    client_module: Any,
    api_pb2: Any,
    environment_name: str | None,
) -> Any:
    """List Modal tasks while running on Modal's managed event loop."""
    client = await client_module._Client.from_env()
    return await client.stub.TaskList(
        api_pb2.TaskListRequest(environment_name=environment_name)
    )


def _list_modal_tasks_synchronously(
    client_module: Any,
    api_pb2: Any,
    environment_name: str | None,
) -> Any:
    """Run one Modal task-list query through the synchronized SDK bridge."""
    blocking_callable = _synchronized_modal_callable(_list_modal_tasks_on_sdk_loop)
    return blocking_callable(client_module, api_pb2, environment_name)


async def _stop_modal_task_on_sdk_loop(
    client_module: Any,
    api_pb2: Any,
    container_id: str,
) -> Any:
    """Stop one exact Modal task while running on Modal's managed event loop."""
    client = await client_module._Client.from_env()
    return await client.stub.ContainerStop(
        api_pb2.ContainerStopRequest(task_id=container_id)
    )


def _stop_modal_task_synchronously(
    client_module: Any,
    api_pb2: Any,
    container_id: str,
) -> Any:
    """Run one Modal task-stop request through the synchronized SDK bridge."""
    blocking_callable = _synchronized_modal_callable(_stop_modal_task_on_sdk_loop)
    return blocking_callable(client_module, api_pb2, container_id)



async def list_active_modal_containers(
    settings: ModalSyncSettings | None = None,
) -> list[ModalContainerStatus]:
    """List active Modal containers belonging to this ComfyUI instance."""
    if modal is None:
        raise ModalContainerStatusError("The Modal SDK is unavailable.")

    resolved_settings = settings or get_settings()
    managed_app_gpus = _managed_modal_app_gpus(resolved_settings)
    try:
        client_module = importlib.import_module("modal.client")
        exception_module = importlib.import_module("modal.exception")
        api_pb2 = importlib.import_module("modal_proto.api_pb2")
    except ModuleNotFoundError as exc:
        raise ModalContainerStatusError(
            "The installed Modal SDK does not expose the container list API."
        ) from exc

    modal_error_type = getattr(exception_module, "Error", RuntimeError)
    try:
        environment_name = _resolved_modal_environment_name()
        response = await asyncio.to_thread(
            _list_modal_tasks_synchronously,
            client_module,
            api_pb2,
            environment_name,
        )
    except (modal_error_type, OSError, AttributeError, RuntimeError) as exc:
        raise ModalContainerStatusError(
            f"Unable to list Modal containers: {exc}"
        ) from exc

    containers: list[ModalContainerStatus] = []
    for task in response.tasks:
        app_name = str(task.app_description)
        modal_gpu = managed_app_gpus.get(app_name)
        if modal_gpu is None:
            continue
        started_at = _optional_modal_timestamp(task.started_at)
        containers.append(
            ModalContainerStatus(
                container_id=str(task.task_id),
                app_id=str(task.app_id),
                app_name=app_name,
                modal_gpu=modal_gpu,
                estimated_gpu_cost_per_second=MODAL_GPU_ESTIMATED_USD_PER_SECOND[
                    modal_gpu
                ],
                state="running" if started_at is not None else "starting",
                enqueued_at=_optional_modal_timestamp(task.enqueued_at),
                started_at=started_at,
            )
        )

    containers.sort(
        key=lambda container: (
            container.enqueued_at or container.started_at or 0.0,
            container.container_id,
        )
    )
    logger.debug(
        "Listed %d active Modal container(s) for the status UI.", len(containers)
    )
    return containers


async def stop_managed_modal_container(
    container_id: str,
    settings: ModalSyncSettings | None = None,
) -> bool:
    """Stop one active container after verifying it belongs to this installation."""
    normalized_container_id = container_id.strip()
    if not normalized_container_id:
        raise ValueError("Modal container_id must not be empty.")

    active_containers = await list_active_modal_containers(settings)
    if not any(
        container.container_id == normalized_container_id
        for container in active_containers
    ):
        return False

    try:
        client_module = importlib.import_module("modal.client")
        exception_module = importlib.import_module("modal.exception")
        api_pb2 = importlib.import_module("modal_proto.api_pb2")
    except ModuleNotFoundError as exc:
        raise ModalContainerStatusError(
            "The installed Modal SDK does not expose the container stop API."
        ) from exc

    modal_error_type = getattr(exception_module, "Error", RuntimeError)
    try:
        await asyncio.to_thread(
            _stop_modal_task_synchronously,
            client_module,
            api_pb2,
            normalized_container_id,
        )
    except (modal_error_type, OSError, AttributeError, RuntimeError) as exc:
        raise ModalContainerStatusError(
            f"Unable to stop Modal container {normalized_container_id}: {exc}"
        ) from exc
    logger.warning("Stopped managed Modal container %s.", normalized_container_id)
    return True


async def _stream_remote_container_logs_via_modal_sdk_async(
    task_id: str,
    stop_event: threading.Event,
) -> None:
    """Follow one Modal container log stream through the Python SDK internals."""
    if modal is None:
        raise ModuleNotFoundError("Modal SDK is unavailable.")

    client_module = importlib.import_module("modal.client")
    exception_module = importlib.import_module("modal.exception")
    api_pb2 = importlib.import_module("modal_proto.api_pb2")
    grpclib_exceptions = importlib.import_module("grpclib.exceptions")
    client = await client_module._Client.from_env()
    line_buffer = _RemoteContainerLogLineBuffer(task_id=task_id)
    last_entry_id = ""

    try:
        while not stop_event.is_set():
            request = api_pb2.AppGetLogsRequest(
                task_id=task_id,
                timeout=5,
                last_entry_id=last_entry_id,
            )
            try:
                async for log_batch in client.stub.AppGetLogs.unary_stream(request):
                    if stop_event.is_set():
                        break
                    if log_batch.entry_id:
                        last_entry_id = str(log_batch.entry_id)
                    if bool(log_batch.app_done):
                        logger.info(
                            "Modal SDK log stream finished for task_id=%s.", task_id
                        )
                        return
                    for log_item in log_batch.items:
                        log_data = getattr(log_item, "data", "")
                        if not log_data:
                            continue
                        _write_remote_container_log_chunk(line_buffer, str(log_data))
            except (
                exception_module.ServiceError,
                exception_module.InternalError,
                grpclib_exceptions.StreamTerminatedError,
                socket.gaierror,
            ) as exc:
                if stop_event.is_set():
                    break
                logger.warning(
                    "Retrying Modal SDK log stream for task_id=%s after transient failure: %s",
                    task_id,
                    exc,
                )
                continue
            except AttributeError as exc:
                if stop_event.is_set():
                    break
                if "_write_appdata" in str(exc):
                    logger.warning(
                        "Retrying Modal SDK log stream for task_id=%s after connection loss: %s",
                        task_id,
                        exc,
                    )
                    continue
                raise
    finally:
        _flush_remote_container_log_chunk(line_buffer)


def _stream_remote_container_logs_via_modal_sdk(
    task_id: str,
    stop_event: threading.Event,
) -> bool:
    """Try to mirror one Modal container log stream through the installed SDK."""
    if modal is None:
        return False
    blocking_callable = _synchronized_modal_callable(
        _stream_remote_container_logs_via_modal_sdk_async
    )
    blocking_callable(task_id, stop_event)
    return True


def _stream_remote_container_logs_via_modal_cli(
    task_id: str,
    stop_event: threading.Event,
) -> bool:
    """Try to mirror one Modal container log stream through the Modal CLI."""
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        return False

    command = [modal_cli, "container", "logs", task_id, "-f"]
    line_buffer = _RemoteContainerLogLineBuffer(task_id=task_id)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    line_buffer = _RemoteContainerLogLineBuffer(task_id=task_id)
    try:
        stdout_stream = process.stdout
        if stdout_stream is None:
            raise RuntimeError("Modal CLI log process did not expose a stdout stream.")
        while not stop_event.is_set():
            ready_streams, _, _ = select.select([stdout_stream], [], [], 0.25)
            if ready_streams:
                next_chunk = stdout_stream.read(4096)
                if next_chunk:
                    _write_remote_container_log_chunk(
                        line_buffer,
                        next_chunk.decode("utf-8", errors="replace"),
                    )
                    continue
            if process.poll() is not None:
                trailing_chunk = stdout_stream.read()
                if trailing_chunk:
                    _write_remote_container_log_chunk(
                        line_buffer,
                        trailing_chunk.decode("utf-8", errors="replace"),
                    )
                break
        if stop_event.is_set() and process.poll() is None:
            process.terminate()
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        _flush_remote_container_log_chunk(line_buffer)

    if process.returncode not in {0, None} and not stop_event.is_set():
        raise RuntimeError(
            f"Modal CLI exited with status {process.returncode} while streaming logs for task_id={task_id}."
        )
    return True


def _run_remote_container_log_stream(task_id: str, stop_event: threading.Event) -> None:
    """Run the best available remote log streaming backend for one Modal container."""
    logger.info("Starting remote Modal container log stream for task_id=%s.", task_id)
    try:
        if _stream_remote_container_logs_via_modal_cli(task_id, stop_event):
            logger.info(
                "Stopped remote Modal container log stream for task_id=%s.", task_id
            )
            return
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning(
            "Falling back from Modal CLI log streaming for task_id=%s after failure: %s",
            task_id,
            exc,
        )

    try:
        if _stream_remote_container_logs_via_modal_sdk(task_id, stop_event):
            logger.info(
                "Stopped remote Modal container log stream for task_id=%s.", task_id
            )
            return
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        logger.warning(
            "Modal SDK log streaming failed for task_id=%s: %s",
            task_id,
            exc,
        )

    logger.warning(
        "Unable to mirror remote Modal container logs for task_id=%s because neither the Modal SDK nor CLI is available locally.",
        task_id,
    )


def _new_remote_container_log_stream_state(
    task_id: str,
) -> _RemoteContainerLogStreamState:
    """Create an unstarted container-log watcher state."""
    stop_event = threading.Event()
    return _RemoteContainerLogStreamState(
        task_id=task_id,
        stop_event=stop_event,
        thread=threading.Thread(
            target=_run_remote_container_log_stream,
            args=(task_id, stop_event),
            name=f"modal-log-stream-{task_id}",
            daemon=True,
        ),
    )


def _cancel_remote_container_log_idle_stop(
    stream_state: _RemoteContainerLogStreamState,
) -> None:
    """Cancel one pending idle-stop timer before a container is reused."""
    idle_stop_timer = stream_state.idle_stop_timer
    if idle_stop_timer is None:
        return
    idle_stop_timer.cancel()
    stream_state.idle_stop_timer = None


def _stop_remote_container_log_stream_if_idle(
    task_id: str,
    expected_state: _RemoteContainerLogStreamState,
) -> None:
    """Stop one watcher after its grace period if no payload reused it."""
    with _REMOTE_CONTAINER_LOG_STREAMS_LOCK:
        stream_state = _REMOTE_CONTAINER_LOG_STREAMS.get(task_id)
        if stream_state is not expected_state or stream_state.refcount != 0:
            return
        stream_state.idle_stop_timer = None
        _REMOTE_CONTAINER_LOG_STREAMS.pop(task_id, None)

    logger.info(
        "Stopping idle remote Modal container log stream for task_id=%s.",
        task_id,
    )
    stream_state.stop_event.set()
    stream_state.thread.join(timeout=0.2)


def _schedule_remote_container_log_idle_stop(
    stream_state: _RemoteContainerLogStreamState,
) -> None:
    """Schedule one daemon timer to stop an unreferenced log watcher."""
    idle_stop_timer = threading.Timer(
        _REMOTE_CONTAINER_LOG_STREAM_IDLE_GRACE_SECONDS,
        _stop_remote_container_log_stream_if_idle,
        args=(stream_state.task_id, stream_state),
    )
    idle_stop_timer.daemon = True
    stream_state.idle_stop_timer = idle_stop_timer
    idle_stop_timer.start()


def _retain_remote_container_log_stream(task_id: str) -> str:
    """Increment one shared Modal container log watcher and start it if needed."""
    with _REMOTE_CONTAINER_LOG_STREAMS_LOCK:
        stream_state = _REMOTE_CONTAINER_LOG_STREAMS.get(task_id)
        if stream_state is not None and not stream_state.thread.is_alive():
            _cancel_remote_container_log_idle_stop(stream_state)
            _REMOTE_CONTAINER_LOG_STREAMS.pop(task_id, None)
            stream_state = None

        if stream_state is None:
            stream_state = _new_remote_container_log_stream_state(task_id)
            _REMOTE_CONTAINER_LOG_STREAMS[task_id] = stream_state
            logger.info(
                "Creating remote Modal container log stream for task_id=%s.", task_id
            )
            stream_state.thread.start()
        else:
            _cancel_remote_container_log_idle_stop(stream_state)
            logger.info(
                "Reusing remote Modal container log stream for task_id=%s.", task_id
            )
        stream_state.refcount += 1
        logger.info(
            "Remote Modal container log stream retain task_id=%s refcount=%d.",
            task_id,
            stream_state.refcount,
        )

    return task_id


def _release_remote_container_log_stream(task_id: str) -> None:
    """Release one watcher while allowing prompt-local container reuse."""
    with _REMOTE_CONTAINER_LOG_STREAMS_LOCK:
        stream_state = _REMOTE_CONTAINER_LOG_STREAMS.get(task_id)
        if stream_state is None:
            return
        stream_state.refcount = max(0, stream_state.refcount - 1)
        logger.info(
            "Remote Modal container log stream release task_id=%s refcount=%d.",
            task_id,
            stream_state.refcount,
        )
        if stream_state.refcount != 0:
            return
        if not stream_state.thread.is_alive():
            _REMOTE_CONTAINER_LOG_STREAMS.pop(task_id, None)
            return
        if stream_state.idle_stop_timer is None:
            logger.info(
                "Keeping remote Modal container log stream alive for %.1fs "
                "to allow task reuse without replaying history task_id=%s.",
                _REMOTE_CONTAINER_LOG_STREAM_IDLE_GRACE_SECONDS,
                task_id,
            )
            _schedule_remote_container_log_idle_stop(stream_state)
