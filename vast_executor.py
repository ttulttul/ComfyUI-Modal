"""Remote executor client for direct workers inside managed Vast.ai leases."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

if __package__:
    from .durable_state import stable_remote_invocation_id
    from .llm_profiles import (
        llm_model_references_from_payload,
        rewrite_llm_model_references,
    )
    from .remote_protocol import (
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_frame,
        encode_json_frame,
        read_frame,
    )
    from .serialization import deserialize_node_outputs, serialize_node_inputs
    from .settings import ModalSyncSettings, get_settings
    from .vast_leases import VastLeaseRecord, VastLeaseRegistry
    from .vast_runtime import VastRuntimeConfiguration, VastRuntimeManager
    from .vast_ssh import (
        VastSshError,
        VastSshRunner,
        vast_connection_from_lease,
    )
else:  # pragma: no cover - direct debugging imports.
    from durable_state import stable_remote_invocation_id
    from llm_profiles import (
        llm_model_references_from_payload,
        rewrite_llm_model_references,
    )
    from remote_protocol import (
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_frame,
        encode_json_frame,
        read_frame,
    )
    from serialization import deserialize_node_outputs, serialize_node_inputs
    from settings import ModalSyncSettings, get_settings
    from vast_leases import VastLeaseRecord, VastLeaseRegistry
    from vast_runtime import VastRuntimeConfiguration, VastRuntimeManager
    from vast_ssh import VastSshError, VastSshRunner, vast_connection_from_lease

logger = logging.getLogger(__name__)
_STAGED_VAST_PROFILES_LOCK = threading.Lock()
_STAGED_VAST_PROFILE_RESULTS: dict[tuple[int, str], dict[str, Any]] = {}


class VastRemoteInvocationError(RuntimeError):
    """Raised when a Vast worker rejects one application invocation."""


class VastRemoteTransportError(VastRemoteInvocationError):
    """Raised when the direct SSH relay ends without an application result."""


class VastLeaseActivityManager(Protocol):
    """Track activity and retention for one managed Vast lease."""

    def begin_activity(self, instance_id: int) -> VastLeaseRecord:
        """Mark a lease busy."""

    def finish_activity(
        self,
        instance_id: int,
        *,
        idle_retention_seconds: float,
        error: str | None = None,
    ) -> VastLeaseRecord:
        """Release activity and reset the idle deadline."""


@dataclass
class VastExecutorClient:
    """Execute serialized ComfyUI components on one direct Vast worker."""

    registry: VastLeaseRegistry
    activity_manager: VastLeaseActivityManager
    runtime_configuration: VastRuntimeConfiguration
    user_directory: Path
    settings: ModalSyncSettings | None = None
    identity_file: Path | None = None
    runner_factory: Callable[[VastLeaseRecord], VastSshRunner] | None = None

    def execute_payload(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one payload synchronously and deserialize its outputs."""
        inputs_payload = serialize_node_inputs(kwargs)
        prepared_payload = self._prepare_payload(payload, inputs_payload)
        response = self._execute_with_activity(prepared_payload, inputs_payload)
        return deserialize_node_outputs(response)

    async def execute_payload_async(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute without blocking ComfyUI's event loop."""
        inputs_payload = serialize_node_inputs(kwargs)
        prepared_payload = self._prepare_payload(payload, inputs_payload)
        invocation_id = str(prepared_payload["invocation_id"])
        try:
            response = await asyncio.to_thread(
                self._execute_with_activity,
                prepared_payload,
                inputs_payload,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(self.cancel, prepared_payload, invocation_id)
            raise
        return deserialize_node_outputs(response)

    def cancel(self, payload: Mapping[str, Any], invocation_id: str) -> bool:
        """Signal one active direct worker invocation."""
        _lease, runner, runtime = self._runtime(payload)
        runtime.ensure_worker()
        request = encode_json_frame(
            RemoteFrameKind.CANCEL,
            {"invocation_id": invocation_id},
        )
        process = runner.popen(self._relay_arguments())
        try:
            stdout, stderr = process.communicate(request, timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            logger.warning("Vast cancellation timed out invocation=%s.", invocation_id)
            return False
        if process.returncode != 0:
            logger.warning(
                "Vast cancellation failed invocation=%s error=%s.",
                invocation_id,
                stderr.decode("utf-8", errors="replace").strip(),
            )
            return False
        frame = read_frame(_BytesReader(stdout))
        if frame is None or frame[0] is not RemoteFrameKind.ACKNOWLEDGEMENT:
            return False
        return bool(decode_json_payload(frame[1]).get("cancelled"))

    def _prepare_payload(
        self,
        payload: Mapping[str, Any],
        inputs_payload: bytes,
    ) -> dict[str, Any]:
        """Attach provider and stable invocation identity."""
        prepared = dict(payload)
        prepared["execution_provider"] = "vast"
        prepared.setdefault(
            "invocation_id",
            stable_remote_invocation_id(prepared, inputs_payload),
        )
        prepared.setdefault("capture_remote_outputs", True)
        return prepared

    def _execute_with_activity(
        self,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> bytes:
        """Keep registry and in-instance watchdog state around one invocation."""
        lease, _runner, runtime = self._runtime(payload)
        active_lease = self.activity_manager.begin_activity(lease.instance_id)
        runtime.update_watchdog(active_lease)
        terminal_error: str | None = None
        try:
            return self._consume_stream(payload, inputs_payload)
        except (RuntimeError, OSError, ValueError) as exc:
            terminal_error = str(exc)
            raise
        finally:
            retention_seconds = _payload_retention_seconds(payload)
            try:
                finished = self.activity_manager.finish_activity(
                    lease.instance_id,
                    idle_retention_seconds=retention_seconds,
                    error=terminal_error,
                )
                runtime.update_watchdog(finished)
            except (KeyError, OSError, RuntimeError, ValueError) as state_error:
                logger.error(
                    "Unable to publish terminal Vast lease activity instance=%d: %s",
                    lease.instance_id,
                    state_error,
                )

    def _consume_stream(
        self,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> bytes:
        """Consume direct worker frames through the shared Modal-compatible UI relay."""
        from .remote.modal_app import (
            _consume_remote_payload_stream,
            _materialize_remote_execution_result,
        )

        response: bytes | None = None
        for attempt in range(1, 3):
            lease, runner, runtime = self._runtime(payload)
            try:
                runtime.ensure_worker()
                self._ensure_llm_profiles_staged(lease, runner, payload)
                response = _consume_remote_payload_stream(
                    payload,
                    self._invoke_stream(runner, payload, inputs_payload),
                )
                break
            except VastRemoteTransportError as exc:
                transport_error: BaseException = exc
            except VastRemoteInvocationError:
                raise
            except (
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                RemoteProtocolError,
                VastSshError,
                subprocess.SubprocessError,
            ) as exc:
                transport_error = exc
            if attempt >= 2:
                raise VastRemoteTransportError(
                    "Vast worker transport failed after one recovery attempt: "
                    f"{transport_error}"
                ) from transport_error
            logger.warning(
                "Recovering Vast worker transport instance=%d error=%s.",
                lease.instance_id,
                transport_error,
            )
            runtime.restart_worker()
        if response is None:
            raise VastRemoteTransportError("Vast worker returned no execution result.")
        return _materialize_remote_execution_result(
            response,
            settings=self.settings or get_settings(),
        )

    def _ensure_llm_profiles_staged(
        self,
        lease: VastLeaseRecord,
        runner: VastSshRunner,
        payload: dict[str, Any],
    ) -> None:
        """Resolve and stage LLM profiles on the selected instance storage."""
        model_references = llm_model_references_from_payload(payload)
        if not model_references:
            return
        with _STAGED_VAST_PROFILES_LOCK:
            missing_references = [
                reference
                for reference in model_references
                if (lease.instance_id, reference) not in _STAGED_VAST_PROFILE_RESULTS
            ]
            if missing_references:
                results = self._run_profile_stager(runner, payload, missing_references)
                self._cache_staged_profiles(
                    lease.instance_id,
                    missing_references,
                    results,
                )
            resolved_results = {
                reference: _STAGED_VAST_PROFILE_RESULTS[(lease.instance_id, reference)]
                for reference in model_references
            }
        rewrite_llm_model_references(
            payload,
            {
                reference: str(result["profile_id"])
                for reference, result in resolved_results.items()
            },
        )

    def _run_profile_stager(
        self,
        runner: VastSshRunner,
        payload: dict[str, Any],
        model_references: list[str],
    ) -> list[dict[str, Any]]:
        """Run the shared CPU stager directly in the Vast container."""
        from .remote.modal_app import (
            _emit_local_llm_staging_progress,
            _emit_local_remote_startup_status,
        )

        _emit_local_remote_startup_status(
            payload,
            phase="llm_staging",
            status_message="Inspecting and staging LLM on the Vast.ai instance",
        )
        arguments = ["python", "-m", "remote.ssh_worker", "stage-profiles"]
        for reference in model_references:
            arguments.extend(("--model-reference", reference))
        process = runner.popen(tuple(arguments))
        results = _consume_stager_output(
            process,
            payload,
            _emit_local_llm_staging_progress,
        )
        _emit_local_remote_startup_status(
            payload,
            phase="llm_staged",
            status_message="Vast.ai LLM staging complete",
        )
        return results

    def _cache_staged_profiles(
        self,
        instance_id: int,
        requested_references: list[str],
        results: list[dict[str, Any]],
    ) -> None:
        """Cache stager aliases and require every requested model result."""
        confirmed: set[str] = set()
        by_requested = {
            str(result.get("requested_reference") or ""): result for result in results
        }
        for requested_reference in requested_references:
            result = by_requested.get(requested_reference)
            if result is None:
                continue
            profile_id = str(result.get("profile_id") or "").strip()
            if not profile_id:
                continue
            _STAGED_VAST_PROFILE_RESULTS[(instance_id, requested_reference)] = result
            _STAGED_VAST_PROFILE_RESULTS[(instance_id, profile_id)] = result
            confirmed.add(requested_reference)
        missing = set(requested_references) - confirmed
        if missing:
            raise VastRemoteInvocationError(
                f"Vast model stager did not confirm models {sorted(missing)}."
            )

    def _invoke_stream(
        self,
        runner: VastSshRunner,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> Iterator[dict[str, Any]]:
        """Start one direct worker relay and yield shared event mappings."""
        request = b"".join(
            (
                encode_json_frame(
                    RemoteFrameKind.REQUEST,
                    {
                        "invocation_id": str(payload["invocation_id"]),
                        "payload": payload,
                    },
                ),
                encode_frame(RemoteFrameKind.INPUTS, inputs_payload),
            )
        )
        process = runner.popen(self._relay_arguments())
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise VastRemoteTransportError(
                "Vast worker relay did not expose binary streams."
            )
        stderr_chunks: list[bytes] = []
        stderr_thread = threading.Thread(
            target=_collect_bounded_stream,
            args=(process.stderr, stderr_chunks),
            daemon=True,
        )
        stderr_thread.start()
        process.stdin.write(request)
        process.stdin.close()
        terminal_received = False
        try:
            while True:
                frame = read_frame(process.stdout)
                if frame is None:
                    break
                kind, frame_payload = frame
                if kind is RemoteFrameKind.PROGRESS:
                    yield decode_json_payload(frame_payload)
                    continue
                if kind is RemoteFrameKind.RESULT:
                    terminal_received = True
                    yield {"kind": "result", "outputs": frame_payload}
                    break
                if kind is RemoteFrameKind.ERROR:
                    terminal_received = True
                    error = decode_json_payload(frame_payload)
                    raise VastRemoteInvocationError(
                        f"Vast worker {error.get('error_type', 'Error')}: "
                        f"{error.get('message', 'remote execution failed')}"
                    )
                raise RemoteProtocolError(
                    f"Unexpected Vast worker response frame {kind.name}."
                )
        finally:
            if not terminal_received and process.poll() is None:
                process.terminate()
            try:
                returncode = process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=5.0)
            stderr_thread.join(timeout=1.0)
            if returncode != 0 and terminal_received:
                logger.warning(
                    "Vast worker relay exited after terminal frame status=%d stderr=%s.",
                    returncode,
                    b"".join(stderr_chunks)
                    .decode("utf-8", errors="replace")
                    .strip(),
                )
        if not terminal_received:
            diagnostics = (
                b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
            )
            raise VastRemoteTransportError(
                "Vast worker stream ended without a result: "
                f"{diagnostics or 'no diagnostics'}."
            )

    def _runtime(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[VastLeaseRecord, VastSshRunner, VastRuntimeManager]:
        """Resolve one assigned lease and direct runtime manager."""
        instance_id = _payload_instance_id(payload)
        lease = next(
            (
                candidate
                for candidate in self.registry.load().leases
                if candidate.instance_id == instance_id
            ),
            None,
        )
        if lease is None:
            raise VastRemoteInvocationError(
                f"Assigned Vast lease {instance_id} is not present in the local registry."
            )
        if lease.draining:
            raise VastRemoteInvocationError(
                f"Assigned Vast lease {instance_id} is draining."
            )
        runner = (
            self.runner_factory(lease)
            if self.runner_factory is not None
            else VastSshRunner(
                vast_connection_from_lease(
                    ssh_host=lease.ssh_host,
                    ssh_port=lease.ssh_port,
                    user_directory=self.user_directory,
                    identity_file=self.identity_file,
                )
            )
        )
        runtime = VastRuntimeManager(
            runner=runner,
            configuration=self.runtime_configuration,
        )
        return lease, runner, runtime

    @staticmethod
    def _relay_arguments() -> tuple[str, ...]:
        """Return the direct worker binary relay command."""
        return ("python", "-m", "remote.ssh_worker", "client")


def _payload_instance_id(payload: Mapping[str, Any]) -> int:
    """Return the concrete Vast instance assigned by queue-time planning."""
    raw_instance_id = payload.get("vast_instance_id")
    if raw_instance_id is None:
        environment_id = str(payload.get("execution_environment_id") or "")
        raw_instance_id = environment_id.rsplit(":", maxsplit=1)[-1]
    try:
        instance_id = int(raw_instance_id)
    except (TypeError, ValueError) as exc:
        raise VastRemoteInvocationError(
            "Vast execution payload is missing a concrete instance identity."
        ) from exc
    if instance_id <= 0:
        raise VastRemoteInvocationError(
            "Vast execution payload contains an invalid instance identity."
        )
    return instance_id


def _payload_retention_seconds(payload: Mapping[str, Any]) -> float:
    """Return the validated per-profile idle retention stamped at queue time."""
    try:
        retention = float(payload.get("vast_idle_retention_seconds", 24 * 3600))
    except (TypeError, ValueError) as exc:
        raise VastRemoteInvocationError(
            "Vast execution payload contains invalid idle retention."
        ) from exc
    if retention < 0:
        raise VastRemoteInvocationError(
            "Vast execution payload idle retention must not be negative."
        )
    return retention


def _consume_stager_output(
    process: subprocess.Popen[bytes],
    payload: dict[str, Any],
    progress_callback: Callable[[dict[str, Any], Mapping[str, Any]], None],
) -> list[dict[str, Any]]:
    """Consume JSON-line progress and terminal direct staging results."""
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise VastRemoteInvocationError(
            "Vast model stager did not expose output streams."
        )
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(
        target=_collect_bounded_stream,
        args=(process.stderr, stderr_chunks),
        daemon=True,
    )
    stderr_thread.start()
    results: list[dict[str, Any]] | None = None
    for raw_line in process.stdout:
        try:
            event = decode_json_payload(raw_line)
        except (RemoteProtocolError, UnicodeDecodeError) as exc:
            process.terminate()
            raise VastRemoteInvocationError(
                "Vast model stager returned invalid progress JSON."
            ) from exc
        if event.get("kind") == "progress":
            progress_callback(payload, event)
        elif event.get("kind") == "result" and isinstance(event.get("results"), list):
            results = [
                dict(result)
                for result in event["results"]
                if isinstance(result, Mapping)
            ]
    returncode = process.wait()
    stderr_thread.join(timeout=1.0)
    if returncode != 0 or results is None:
        diagnostics = (
            b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        )
        raise VastRemoteInvocationError(
            "Vast model staging failed: "
            f"{diagnostics or f'exit status {returncode}'}."
        )
    return results


def _collect_bounded_stream(stream: Any, chunks: list[bytes]) -> None:
    """Drain diagnostics while retaining at most the recent MiB."""
    while chunk := stream.read(65536):
        chunks.append(chunk)
        if sum(len(value) for value in chunks) > 1024 * 1024:
            del chunks[:-8]


class _BytesReader:
    """Minimal immutable byte cursor for one-shot protocol responses."""

    def __init__(self, payload: bytes) -> None:
        """Initialize the cursor."""
        self._payload = payload
        self._offset = 0

    def read(self, length: int = -1) -> bytes:
        """Read up to ``length`` bytes."""
        if length < 0:
            length = len(self._payload) - self._offset
        end = min(len(self._payload), self._offset + length)
        result = self._payload[self._offset : end]
        self._offset = end
        return result


__all__ = [
    "VastExecutorClient",
    "VastLeaseActivityManager",
    "VastRemoteInvocationError",
    "VastRemoteTransportError",
]
