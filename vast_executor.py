"""Remote executor client for direct workers inside managed Vast.ai leases."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

if __package__:
    from .durable_state import stable_remote_invocation_id
    from .llm_profiles import (
        encoded_resolved_llm_profile_payloads,
        llm_model_references_from_payload,
        rewrite_llm_model_references,
    )
    from .remote_protocol import (
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_json_frame,
        read_frame,
        write_frame,
    )
    from .serialization import deserialize_node_outputs, serialize_node_inputs
    from .settings import ModalSyncSettings, get_settings
    from .staging_process import (
        RemoteStagingProcessError,
        consume_staging_process,
        terminate_staging_transport,
    )
    from .vast_leases import VastLeaseRecord, VastLeaseRegistry
    from .vast_image_reference import vast_worker_images_compatible
    from .vast_runtime import VastRuntimeConfiguration, VastRuntimeManager
    from .vast_ssh import (
        VastSshError,
        VastSshRunner,
        vast_connection_from_lease,
    )
else:  # pragma: no cover - direct debugging imports.
    from durable_state import stable_remote_invocation_id
    from llm_profiles import (
        encoded_resolved_llm_profile_payloads,
        llm_model_references_from_payload,
        rewrite_llm_model_references,
    )
    from remote_protocol import (
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_json_frame,
        read_frame,
        write_frame,
    )
    from serialization import deserialize_node_outputs, serialize_node_inputs
    from settings import ModalSyncSettings, get_settings
    from staging_process import (
        RemoteStagingProcessError,
        consume_staging_process,
        terminate_staging_transport,
    )
    from vast_leases import VastLeaseRecord, VastLeaseRegistry
    from vast_image_reference import vast_worker_images_compatible
    from vast_runtime import VastRuntimeConfiguration, VastRuntimeManager
    from vast_ssh import VastSshError, VastSshRunner, vast_connection_from_lease

logger = logging.getLogger(__name__)
_STAGED_VAST_PROFILES_LOCK = threading.Lock()
_STAGED_VAST_PROFILE_RESULTS: dict[tuple[int, str], dict[str, Any]] = {}
_ACTIVE_VAST_STAGERS_LOCK = threading.Lock()
_ACTIVE_VAST_STAGERS: dict[
    str,
    tuple[VastSshRunner, str, subprocess.Popen[bytes]],
] = {}


class VastRemoteInvocationError(RuntimeError):
    """Raised when a Vast worker rejects one application invocation."""


class VastRemoteTransportError(VastRemoteInvocationError):
    """Raised when the direct SSH relay ends without an application result."""


class VastRemoteResourceError(VastRemoteInvocationError):
    """Raised when remote resource evidence explains a terminated invocation."""


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
        with _ACTIVE_VAST_STAGERS_LOCK:
            active_stager = _ACTIVE_VAST_STAGERS.get(invocation_id)
        if active_stager is not None:
            runner, owner_id, process = active_stager
            terminate_staging_transport(process)
            remote_cancelled = self._cancel_remote_stager(
                runner,
                owner_id,
                wait_for_owner_seconds=5.0,
            )
            logger.info(
                "Cancelled Vast model staging invocation=%s "
                "remote_process_found=%s.",
                invocation_id,
                remote_cancelled,
            )
            return True
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
                    "Vast worker failed again after one automatic restart. "
                    f"Latest failure: {transport_error}"
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
            status_message="Preparing LLM model snapshots on the Vast.ai instance",
        )
        invocation_id = str(payload["invocation_id"])
        owner_id = f"vast:{invocation_id}:{uuid4().hex}"
        arguments = [
            "python",
            "-m",
            "remote.ssh_worker",
            "stage-profiles",
            "--owner-id",
            owner_id,
        ]
        for reference in model_references:
            arguments.extend(("--model-reference", reference))
        resolved_profiles = encoded_resolved_llm_profile_payloads(
            payload,
            model_references,
        )
        if resolved_profiles is not None:
            arguments.extend(("--resolved-profiles", resolved_profiles))
        process = runner.popen(tuple(arguments))
        with _ACTIVE_VAST_STAGERS_LOCK:
            _ACTIVE_VAST_STAGERS[invocation_id] = (runner, owner_id, process)
        try:
            results = consume_staging_process(
                process,
                payload,
                _emit_local_llm_staging_progress,
                provider_label="Vast",
                abort_remote=lambda: self._cancel_remote_stager(
                    runner,
                    owner_id,
                ),
            )
        except RemoteStagingProcessError as exc:
            raise VastRemoteInvocationError(str(exc)) from exc
        finally:
            with _ACTIVE_VAST_STAGERS_LOCK:
                _ACTIVE_VAST_STAGERS.pop(invocation_id, None)
        _emit_local_remote_startup_status(
            payload,
            phase="llm_staged",
            status_message="Vast.ai LLM staging complete",
        )
        return results

    @staticmethod
    def _cancel_remote_stager(
        runner: VastSshRunner,
        owner_id: str,
        *,
        wait_for_owner_seconds: float = 0.0,
    ) -> bool:
        """Terminate one exact owner-tagged stager on a Vast instance."""
        deadline = time.monotonic() + max(0.0, wait_for_owner_seconds)
        while True:
            result = runner.run(
                (
                    "python",
                    "-m",
                    "remote.ssh_worker",
                    "cancel-staging",
                    "--owner-id",
                    owner_id,
                ),
                timeout_seconds=15.0,
                check=False,
                transport_attempts=1,
            )
            if result.returncode != 0:
                logger.warning(
                    "Vast remote stager cancellation failed owner=%s error=%s.",
                    owner_id,
                    result.stderr_text.strip() or result.stdout_text.strip(),
                )
                return False
            try:
                response = decode_json_payload(result.stdout)
            except (RemoteProtocolError, UnicodeDecodeError):
                return False
            if response.get("cancelled"):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

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
        from .remote.local_ui_events import RemoteTransferProgressReporter

        request = encode_json_frame(
            RemoteFrameKind.REQUEST,
            {
                "invocation_id": str(payload["invocation_id"]),
                "payload": payload,
            },
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
        upload_reporter = RemoteTransferProgressReporter(
            payload,
            direction="upload",
            total_bytes=len(inputs_payload),
        )
        download_reporter: RemoteTransferProgressReporter | None = None
        upload_reporter.start()
        process.stdin.write(request)
        write_frame(
            process.stdin,
            RemoteFrameKind.INPUTS,
            inputs_payload,
            progress_callback=(
                lambda _kind, current, _total: upload_reporter.update(current)
            ),
        )
        upload_reporter.complete()
        process.stdin.close()
        terminal_received = False
        try:
            while True:
                def report_download(
                    kind: RemoteFrameKind,
                    current: int,
                    total: int,
                ) -> None:
                    """Forward RESULT frame byte progress to the local UI."""
                    nonlocal download_reporter
                    if kind is not RemoteFrameKind.RESULT:
                        return
                    if download_reporter is None:
                        download_reporter = RemoteTransferProgressReporter(
                            payload,
                            direction="download",
                            total_bytes=total,
                        )
                        download_reporter.start()
                    download_reporter.update(current)

                frame = read_frame(
                    process.stdout,
                    progress_callback=report_download,
                )
                if frame is None:
                    break
                kind, frame_payload = frame
                if kind is RemoteFrameKind.PROGRESS:
                    yield decode_json_payload(frame_payload)
                    continue
                if kind is RemoteFrameKind.RESULT:
                    if download_reporter is not None:
                        download_reporter.complete()
                    terminal_received = True
                    yield {"kind": "result", "outputs": frame_payload}
                    break
                if kind is RemoteFrameKind.ERROR:
                    terminal_received = True
                    error = decode_json_payload(frame_payload)
                    raise _vast_invocation_error(error)
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
            diagnostics = _useful_ssh_stderr(
                b"".join(stderr_chunks).decode("utf-8", errors="replace")
            )
            detail = f" SSH diagnostic: {diagnostics}" if diagnostics else ""
            raise VastRemoteTransportError(
                "Vast worker process exited without returning a result "
                f"(SSH relay status {returncode}). No structured worker postmortem "
                "was available. Rebuild the configured Vast worker image to enable "
                f"resource and crash diagnostics.{detail}"
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
        if not vast_worker_images_compatible(
            self.runtime_configuration.image,
            lease.worker_image,
        ):
            raise VastRemoteInvocationError(
                f"Assigned Vast lease {instance_id} does not use the expected "
                "immutable worker image digest."
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


def _vast_invocation_error(error: Mapping[str, Any]) -> VastRemoteInvocationError:
    """Map a structured worker error to retryable or terminal controller state."""
    error_type = str(error.get("error_type") or "Error")
    message = str(error.get("message") or "remote execution failed")
    if error_type == "WorkerOutOfMemoryError":
        return VastRemoteResourceError(message)
    if error_type == "WorkerProcessLostError":
        return VastRemoteTransportError(message)
    return VastRemoteInvocationError(f"Vast worker {error_type}: {message}")


def _useful_ssh_stderr(stderr: str) -> str:
    """Remove Vast's generic login greeting from actionable relay stderr."""
    diagnostic = stderr
    for greeting in (
        "Welcome to vast.ai.",
        "If authentication fails, try again after a few seconds, and double check "
        "your ssh key.",
        "Have fun!.",
        "Have fun!",
    ):
        diagnostic = diagnostic.replace(greeting, "")
    normalized = " ".join(diagnostic.split())
    return normalized if normalized.strip(".! ") else ""


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
    "VastRemoteResourceError",
    "VastRemoteTransportError",
]
