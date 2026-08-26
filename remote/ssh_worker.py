"""Persistent ComfyUI worker server used by SSH/Docker environments."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import signal
import socket
import socketserver
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
from uuid import uuid4

try:
    from ..remote_protocol import (
        REMOTE_PROTOCOL_VERSION,
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_frame,
        encode_json_frame,
        read_frame,
        write_frame,
    )
except ImportError:  # pragma: no cover - remote image imports ``remote`` top-level.
    from remote_protocol import (
        REMOTE_PROTOCOL_VERSION,
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_frame,
        encode_json_frame,
        read_frame,
        write_frame,
    )

logger = logging.getLogger(__name__)

DEFAULT_WORKER_SOCKET_PATH = Path("/run/comfy-remote/worker.sock")
DEFAULT_STORAGE_ROOT = Path("/storage")
_STAGING_OWNER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_STAGING_TERMINATE_GRACE_SECONDS = 5.0


class SshWorkerError(RuntimeError):
    """Raised when an SSH worker request cannot be executed."""


class WorkerExecutionState:
    """Track active invocations and serialize GPU execution in one worker."""

    def __init__(self) -> None:
        """Initialize empty active-invocation state."""
        self.execution_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._cancellations: dict[str, threading.Event] = {}

    def register(self, invocation_id: str) -> threading.Event:
        """Register a new active invocation and return its cancellation event."""
        cancellation_event = threading.Event()
        with self._active_lock:
            if invocation_id in self._cancellations:
                raise SshWorkerError(
                    f"Invocation {invocation_id!r} is already active on this worker."
                )
            self._cancellations[invocation_id] = cancellation_event
        return cancellation_event

    def unregister(self, invocation_id: str) -> None:
        """Remove one completed invocation from active state."""
        with self._active_lock:
            self._cancellations.pop(invocation_id, None)

    def cancel(self, invocation_id: str) -> bool:
        """Signal one active invocation and report whether it was found."""
        with self._active_lock:
            cancellation_event = self._cancellations.get(invocation_id)
        if cancellation_event is None:
            return False
        cancellation_event.set()
        return True


_WORKER_STATE = WorkerExecutionState()


class _ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Serve independent control connections while one GPU request executes."""

    daemon_threads = True
    allow_reuse_address = True


class _WorkerRequestHandler(socketserver.StreamRequestHandler):
    """Execute one framed request or cancellation command."""

    def handle(self) -> None:
        """Dispatch the first frame on this Unix-socket connection."""
        first_frame = read_frame(self.rfile)
        if first_frame is None:
            return
        kind, payload = first_frame
        if kind is RemoteFrameKind.CANCEL:
            self._handle_cancel(payload)
            return
        if kind is not RemoteFrameKind.REQUEST:
            raise RemoteProtocolError(
                f"Worker expected REQUEST or CANCEL, received {kind.name}."
            )
        self._handle_execution(payload)

    def _handle_cancel(self, payload: bytes) -> None:
        """Signal one active invocation through the worker's shared state."""
        request = decode_json_payload(payload)
        invocation_id = _required_string(request, "invocation_id")
        cancelled = _WORKER_STATE.cancel(invocation_id)
        self.wfile.write(
            encode_json_frame(
                RemoteFrameKind.ACKNOWLEDGEMENT,
                {"invocation_id": invocation_id, "cancelled": cancelled},
            )
        )
        self.wfile.flush()

    def _handle_execution(self, payload: bytes) -> None:
        """Run one payload through the shared ComfyUI execution kernel."""
        request = decode_json_payload(payload)
        raw_payload = request.get("payload")
        if not isinstance(raw_payload, dict):
            raise RemoteProtocolError("Worker request requires a payload object.")
        invocation_id = _required_string(request, "invocation_id")
        inputs_frame = read_frame(self.rfile)
        if inputs_frame is None or inputs_frame[0] is not RemoteFrameKind.INPUTS:
            raise RemoteProtocolError("Worker request must include one INPUTS frame.")
        cancellation_event = _WORKER_STATE.register(invocation_id)
        try:
            with _WORKER_STATE.execution_lock:
                self._stream_execution(
                    raw_payload,
                    inputs_frame[1],
                    cancellation_event,
                )
        except (BrokenPipeError, ConnectionResetError):
            cancellation_event.set()
            logger.warning(
                "SSH worker client disconnected during invocation=%s.", invocation_id
            )
        except Exception as exc:
            logger.exception("SSH worker invocation=%s failed.", invocation_id)
            self._write_error(invocation_id, exc)
        finally:
            _WORKER_STATE.unregister(invocation_id)

    def _stream_execution(
        self,
        payload: dict[str, Any],
        inputs_payload: bytes,
        cancellation_event: threading.Event,
    ) -> None:
        """Forward shared execution-kernel events onto the framed socket."""
        for event in _execution_events(payload, inputs_payload, cancellation_event):
            event_kind = str(event.get("kind") or "")
            if event_kind == "result":
                outputs = event.get("outputs")
                if not isinstance(outputs, bytes | bytearray):
                    raise SshWorkerError("Execution result did not contain binary outputs.")
                write_frame(self.wfile, RemoteFrameKind.RESULT, bytes(outputs))
                continue
            if event_kind == "remote_logs":
                continue
            write_frame(
                self.wfile,
                RemoteFrameKind.PROGRESS,
                json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            )

    def _write_error(self, invocation_id: str, error: Exception) -> None:
        """Return a bounded error envelope when the client remains connected."""
        try:
            self.wfile.write(
                encode_json_frame(
                    RemoteFrameKind.ERROR,
                    {
                        "invocation_id": invocation_id,
                        "error_type": type(error).__name__,
                        "message": str(error)[:4096],
                    },
                )
            )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def _execution_events(
    payload: dict[str, Any],
    inputs_payload: bytes,
    cancellation_event: threading.Event,
) -> Iterator[dict[str, Any]]:
    """Invoke the provider-neutral surface of the existing remote kernel."""
    import comfyui_modal_sync_cloud as execution_kernel

    yield from execution_kernel._stream_remote_payload_events(
        payload,
        inputs_payload,
        cancellation_event=cancellation_event,
    )


def serve(socket_path: Path = DEFAULT_WORKER_SOCKET_PATH) -> None:
    """Run the persistent Unix-socket worker until the container stops."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = _ThreadingUnixStreamServer(str(socket_path), _WorkerRequestHandler)
    os.chmod(socket_path, 0o600)
    logger.info(
        "SSH ComfyUI worker ready socket=%s protocol=%d fingerprint=%s.",
        socket_path,
        REMOTE_PROTOCOL_VERSION,
        os.getenv("COMFY_MODAL_RUNTIME_FINGERPRINT", "unknown"),
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        socket_path.unlink(missing_ok=True)


def relay_client(socket_path: Path = DEFAULT_WORKER_SOCKET_PATH) -> int:
    """Relay framed stdin/stdout between ``docker exec`` and the worker socket."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(str(socket_path))
        _copy_request_frames(sys.stdin.buffer, client_socket)
        _copy_response_frames(client_socket, sys.stdout.buffer)
    return 0


def runtime_info() -> dict[str, Any]:
    """Return worker compatibility information without allocating model state."""
    return {
        "protocol_version": REMOTE_PROTOCOL_VERSION,
        "runtime_fingerprint": os.getenv("COMFY_MODAL_RUNTIME_FINGERPRINT", ""),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "worker_socket": str(DEFAULT_WORKER_SOCKET_PATH),
        "worker_socket_ready": DEFAULT_WORKER_SOCKET_PATH.is_socket(),
    }


def stage_profiles(
    model_references: list[str],
    storage_root: Path = DEFAULT_STORAGE_ROOT,
    *,
    resolved_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve and stage immutable LLM profiles in the worker's storage volume."""
    try:
        from ..llm_staging import resolve_and_stage_model_references
    except ImportError:  # pragma: no cover - remote image imports top-level modules.
        from llm_staging import resolve_and_stage_model_references

    def emit_progress(progress: Any) -> None:
        """Write one JSON-line progress event for the SSH controller."""
        print(
            json.dumps(
                {
                    "kind": "progress",
                    "stage": progress.stage,
                    "message": progress.message,
                    "value": progress.value,
                    "max": progress.maximum,
                    "unit": progress.unit,
                    "indeterminate": progress.indeterminate,
                    "model_reference": progress.model_reference,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )

    results = resolve_and_stage_model_references(
        model_references,
        storage_root,
        progress_callback=emit_progress,
        resolved_profiles=resolved_profiles,
        owner_id=owner_id,
    )
    result_payload = [result.to_dict() for result in results]
    print(
        json.dumps(
            {"kind": "result", "results": result_payload},
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return result_payload


def _validated_staging_owner_id(owner_id: str) -> str:
    """Return a filesystem-safe controller-issued staging owner identity."""
    normalized = owner_id.strip()
    if not _STAGING_OWNER_ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid staging owner id {owner_id!r}.")
    return normalized


def _staging_owner_path(storage_root: Path, owner_id: str) -> Path:
    """Return the isolated process record for one staging invocation."""
    return (
        storage_root.resolve()
        / "llm_staging"
        / "owners"
        / f"{_validated_staging_owner_id(owner_id)}.json"
    )


def _linux_process_start(pid: int) -> str | None:
    """Return one Linux process start tick for PID-reuse validation."""
    try:
        stat_record = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    fields_after_command = stat_record.rpartition(") ")[2].split()
    return fields_after_command[19] if len(fields_after_command) > 19 else None


def _owner_record_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    """Return whether a process record still belongs to the expected owner."""
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(
        isinstance(current, Mapping)
        and current.get("pid") == expected.get("pid")
        and current.get("process_start") == expected.get("process_start")
    )


def _remove_owned_snapshot_leases(
    storage_root: Path,
    owner: Mapping[str, Any],
) -> None:
    """Remove snapshot leases that still name one terminated staging owner."""
    model_root = storage_root.resolve() / "llm_models"
    if not model_root.is_dir():
        return
    for lease_path in model_root.rglob(".*.download.lock"):
        try:
            lease_owner = json.loads(lease_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if not isinstance(lease_owner, Mapping):
            continue
        if any(
            lease_owner.get(field) != owner.get(field)
            for field in ("owner_id", "pid", "process_start")
        ):
            continue
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def staging_process_owner(storage_root: Path, owner_id: str) -> Iterator[None]:
    """Publish an exact process record that a controller can safely terminate."""
    owner_path = _staging_owner_path(storage_root, owner_id)
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    owner_record = {
        "version": 1,
        "owner_id": _validated_staging_owner_id(owner_id),
        "pid": pid,
        "process_start": _linux_process_start(pid),
        "created_at": time.time(),
    }
    temporary_path = owner_path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(owner_record, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, owner_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    try:
        yield
    finally:
        if _owner_record_matches(owner_path, owner_record):
            try:
                owner_path.unlink()
            except FileNotFoundError:
                pass


def cancel_staging_process(storage_root: Path, owner_id: str) -> bool:
    """Terminate only the live process matching one controller owner record."""
    owner_path = _staging_owner_path(storage_root, owner_id)
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (json.JSONDecodeError, OSError) as exc:
        raise SshWorkerError(f"Staging owner record is unreadable: {exc}") from exc
    if not isinstance(owner, Mapping):
        raise SshWorkerError("Staging owner record must be an object.")
    try:
        pid = int(owner.get("pid"))
    except (TypeError, ValueError) as exc:
        raise SshWorkerError("Staging owner record has an invalid PID.") from exc
    if pid <= 0:
        raise SshWorkerError("Staging owner record has an invalid PID.")
    expected_start = str(owner.get("process_start") or "")
    if _linux_process_start(pid) != expected_start:
        _remove_owned_snapshot_leases(storage_root, owner)
        owner_path.unlink(missing_ok=True)
        return False
    try:
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, OSError):
        owner_path.unlink(missing_ok=True)
        return False
    encoded_owner = owner_id.encode("utf-8")
    if (
        b"stage-profiles" not in command_line
        or b"--owner-id" not in command_line
        or encoded_owner not in command_line
    ):
        raise SshWorkerError(
            f"Refusing to terminate PID {pid}; its command does not match owner "
            f"{owner_id!r}."
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + _STAGING_TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        if _linux_process_start(pid) != expected_start:
            _remove_owned_snapshot_leases(storage_root, owner)
            owner_path.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    _remove_owned_snapshot_leases(storage_root, owner)
    owner_path.unlink(missing_ok=True)
    return True


def _decode_resolved_profiles(encoded_payload: str | None) -> dict[str, dict[str, Any]]:
    """Decode planner-resolved profile metadata from a credential-free CLI value."""
    if not encoded_payload:
        return {}
    try:
        decoded = base64.urlsafe_b64decode(encoded_payload.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Resolved LLM profile payload is invalid.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Resolved LLM profile payload must be an object.")
    return {
        str(reference): dict(profile)
        for reference, profile in payload.items()
        if isinstance(reference, str) and isinstance(profile, Mapping)
    }


def _copy_request_frames(source: BinaryIO, destination: socket.socket) -> None:
    """Copy exactly one request command and its optional input frame."""
    first_frame = read_frame(source)
    if first_frame is None:
        raise RemoteProtocolError("Worker relay received an empty request stream.")
    destination.sendall(encode_frame(*first_frame))
    if first_frame[0] is RemoteFrameKind.REQUEST:
        inputs_frame = read_frame(source)
        if inputs_frame is None or inputs_frame[0] is not RemoteFrameKind.INPUTS:
            raise RemoteProtocolError("Worker relay request omitted its INPUTS frame.")
        destination.sendall(encode_frame(*inputs_frame))
    destination.shutdown(socket.SHUT_WR)


def _copy_response_frames(source: socket.socket, destination: BinaryIO) -> None:
    """Copy response frames from the worker socket until terminal state."""
    source_file = source.makefile("rb")
    try:
        while True:
            frame = read_frame(source_file)
            if frame is None:
                return
            destination.write(encode_frame(*frame))
            destination.flush()
            if frame[0] in {
                RemoteFrameKind.RESULT,
                RemoteFrameKind.ERROR,
                RemoteFrameKind.ACKNOWLEDGEMENT,
                RemoteFrameKind.RUNTIME_INFO,
            }:
                return
    finally:
        source_file.close()


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    """Return one required non-empty string from a protocol object."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RemoteProtocolError(f"Worker request requires non-empty {key!r}.")
    return value.strip()


def _argument_parser() -> argparse.ArgumentParser:
    """Return the worker command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "serve",
            "client",
            "runtime-info",
            "stage-profiles",
            "cancel-staging",
        ),
    )
    parser.add_argument(
        "--socket",
        type=Path,
        default=DEFAULT_WORKER_SOCKET_PATH,
        help="Unix socket shared by the worker server and docker-exec relay.",
    )
    parser.add_argument(
        "--model-reference",
        action="append",
        default=[],
        help="Curated profile or Hugging Face model reference to stage.",
    )
    parser.add_argument(
        "--owner-id",
        default="",
        help="Controller-issued identity used for targeted staging cancellation.",
    )
    parser.add_argument(
        "--resolved-profiles",
        default="",
        help="URL-safe base64 encoded planner-resolved profile metadata.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
        help="Persistent worker storage root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one worker server, relay, or metadata command."""
    arguments = _argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if arguments.command == "serve":
        serve(arguments.socket)
        return 0
    if arguments.command == "client":
        return relay_client(arguments.socket)
    if arguments.command == "stage-profiles":
        if not arguments.model_reference:
            raise ValueError("stage-profiles requires at least one --model-reference.")
        resolved_profiles = _decode_resolved_profiles(arguments.resolved_profiles)
        if arguments.owner_id:
            with staging_process_owner(arguments.storage_root, arguments.owner_id):
                stage_profiles(
                    arguments.model_reference,
                    arguments.storage_root,
                    resolved_profiles=resolved_profiles,
                    owner_id=arguments.owner_id,
                )
        else:
            stage_profiles(
                arguments.model_reference,
                arguments.storage_root,
                resolved_profiles=resolved_profiles,
            )
        return 0
    if arguments.command == "cancel-staging":
        if not arguments.owner_id:
            raise ValueError("cancel-staging requires --owner-id.")
        print(
            json.dumps(
                {"cancelled": cancel_staging_process(
                    arguments.storage_root,
                    arguments.owner_id,
                )},
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(runtime_info(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised inside remote containers.
    raise SystemExit(main())
