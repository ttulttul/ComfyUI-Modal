"""Remote executor client backed by warm Docker workers reached over SSH."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

if __package__:
    from .durable_state import stable_remote_invocation_id
    from .llm_profiles import (
        encoded_resolved_llm_profile_payloads,
        llm_model_references_from_payload,
        rewrite_llm_model_references,
    )
    from .remote_hosts import RemoteHostRegistry, SshHostConfig
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
    from .ssh_docker import SshDockerController, SshDockerError
    from .ssh_runtime import SshRuntimeManager, SshRuntimeSpec
else:  # pragma: no cover - top-level remote imports.
    from durable_state import stable_remote_invocation_id
    from llm_profiles import (
        encoded_resolved_llm_profile_payloads,
        llm_model_references_from_payload,
        rewrite_llm_model_references,
    )
    from remote_hosts import RemoteHostRegistry, SshHostConfig
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
    from ssh_docker import SshDockerController, SshDockerError
    from ssh_runtime import SshRuntimeManager, SshRuntimeSpec

logger = logging.getLogger(__name__)
_STAGED_SSH_PROFILES_LOCK = threading.Lock()
_STAGED_SSH_PROFILE_RESULTS: dict[tuple[str, str], dict[str, Any]] = {}
_ACTIVE_SSH_STAGERS_LOCK = threading.Lock()
_ACTIVE_SSH_STAGERS: dict[
    str,
    tuple[SshRuntimeManager, SshRuntimeSpec, str, subprocess.Popen[bytes]],
] = {}


class SshRemoteInvocationError(RuntimeError):
    """Raised when an SSH worker rejects or loses an invocation."""


class SshRemoteTransportError(SshRemoteInvocationError):
    """Raised when the SSH relay ends without a remote application error."""


@dataclass
class SshDockerExecutorClient:
    """Execute serialized ComfyUI components on one configured SSH host."""

    registry: RemoteHostRegistry | None
    repo_root: Path
    settings: ModalSyncSettings | None = None

    def execute_payload(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one payload synchronously and deserialize its outputs."""
        inputs_payload = serialize_node_inputs(kwargs)
        prepared_payload = self._prepare_payload(payload, inputs_payload)
        result = self._consume_stream(
            prepared_payload,
            inputs_payload,
        )
        return deserialize_node_outputs(result)

    async def execute_payload_async(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one payload without blocking ComfyUI's event loop."""
        inputs_payload = serialize_node_inputs(kwargs)
        prepared_payload = self._prepare_payload(payload, inputs_payload)
        invocation_id = str(prepared_payload["invocation_id"])
        try:
            result = await asyncio.to_thread(
                self._consume_stream,
                prepared_payload,
                inputs_payload,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(self.cancel, prepared_payload, invocation_id)
            raise
        return deserialize_node_outputs(result)

    def cancel(
        self,
        payload: Mapping[str, Any],
        invocation_id: str,
    ) -> bool:
        """Request cancellation of one active invocation on its assigned worker."""
        with _ACTIVE_SSH_STAGERS_LOCK:
            active_stager = _ACTIVE_SSH_STAGERS.get(invocation_id)
        if active_stager is not None:
            active_manager, active_spec, owner_id, process = active_stager
            terminate_staging_transport(process)
            remote_cancelled = self._cancel_remote_stager(
                active_manager,
                active_spec,
                owner_id,
                wait_for_owner_seconds=5.0,
            )
            logger.info(
                "Cancelled SSH model staging invocation=%s "
                "environment=%s worker_index=%d remote_process_found=%s.",
                invocation_id,
                active_manager.controller.host.environment_id,
                active_spec.worker_index,
                remote_cancelled,
            )
            return True
        manager, spec = self._runtime(payload)
        request = encode_json_frame(
            RemoteFrameKind.CANCEL,
            {"invocation_id": invocation_id},
        )
        process = manager.controller.docker_popen(self._relay_arguments(spec))
        try:
            stdout, stderr = process.communicate(request, timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            logger.warning(
                "SSH cancellation timed out environment=%s invocation=%s.",
                manager.controller.host.environment_id,
                invocation_id,
            )
            return False
        if process.returncode != 0:
            logger.warning(
                "SSH cancellation failed environment=%s invocation=%s error=%s.",
                manager.controller.host.environment_id,
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
        """Attach provider and stable invocation identity to one payload."""
        prepared = dict(payload)
        prepared["execution_provider"] = "ssh_docker"
        prepared.setdefault(
            "invocation_id",
            stable_remote_invocation_id(prepared, inputs_payload),
        )
        prepared.setdefault("capture_remote_outputs", True)
        return prepared

    def _consume_stream(self, payload: dict[str, Any], inputs_payload: bytes) -> bytes:
        """Consume SSH frames through the existing Modal-compatible UI relay."""
        from .remote.modal_app import (
            _consume_remote_payload_stream,
            _materialize_remote_execution_result,
        )

        response: bytes | None = None
        for attempt in range(1, 3):
            manager, spec = self._runtime(payload)
            try:
                self._ensure_llm_profiles_staged(manager, spec, payload)
                stream = self._invoke_stream(manager, spec, payload, inputs_payload)
                response = _consume_remote_payload_stream(payload, stream)
                break
            except (
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                RemoteProtocolError,
                SshDockerError,
                SshRemoteTransportError,
                subprocess.SubprocessError,
            ) as exc:
                if attempt >= 2:
                    raise SshRemoteTransportError(
                        f"SSH worker transport failed after one recovery attempt: {exc}"
                    ) from exc
                logger.warning(
                    "Recovering SSH worker transport environment=%s worker_index=%d error=%s.",
                    manager.controller.host.environment_id,
                    spec.worker_index,
                    exc,
                )
                try:
                    manager.stop_worker(spec.worker_index)
                except SshDockerError as stop_error:
                    logger.warning(
                        "Unable to stop failed SSH worker before retry: %s",
                        stop_error,
                    )
        if response is None:
            raise SshRemoteTransportError("SSH worker returned no execution result.")
        return _materialize_remote_execution_result(
            response,
            settings=self.settings or get_settings(),
        )

    def _ensure_llm_profiles_staged(
        self,
        manager: SshRuntimeManager,
        spec: SshRuntimeSpec,
        payload: dict[str, Any],
    ) -> None:
        """Resolve, stage, and rewrite LLM model references on one SSH host."""
        model_references = llm_model_references_from_payload(payload)
        if not model_references:
            return
        environment_id = manager.controller.host.environment_id
        with _STAGED_SSH_PROFILES_LOCK:
            missing_references = [
                reference
                for reference in model_references
                if (environment_id, reference) not in _STAGED_SSH_PROFILE_RESULTS
            ]
            if missing_references:
                stage_results = self._run_profile_stager(
                    manager,
                    spec,
                    payload,
                    missing_references,
                )
                self._cache_staged_profiles(
                    environment_id,
                    missing_references,
                    stage_results,
                )
            resolved_results = {
                reference: _STAGED_SSH_PROFILE_RESULTS[(environment_id, reference)]
                for reference in model_references
            }
        rewrite_llm_model_references(
            payload,
            {
                reference: str(result["profile_id"])
                for reference, result in resolved_results.items()
            },
        )
        logger.info(
            "SSH LLM models are resolved and staged environment=%s component=%s profiles=%s.",
            environment_id,
            payload.get("component_id"),
            sorted(str(result["profile_id"]) for result in resolved_results.values()),
        )

    def _run_profile_stager(
        self,
        manager: SshRuntimeManager,
        spec: SshRuntimeSpec,
        payload: dict[str, Any],
        model_references: list[str],
    ) -> list[dict[str, Any]]:
        """Run the CPU model stager inside one persistent SSH worker container."""
        from .remote.modal_app import (
            _emit_local_llm_staging_progress,
            _emit_local_remote_startup_status,
        )

        _emit_local_remote_startup_status(
            payload,
            phase="llm_staging",
            status_message="Preparing LLM model snapshots on the SSH host",
        )
        arguments = [
            "exec",
            spec.container_name,
            "python",
            "-m",
            "remote.ssh_worker",
            "stage-profiles",
        ]
        invocation_id = str(payload["invocation_id"])
        owner_id = f"ssh:{invocation_id}:{uuid4().hex}"
        arguments.extend(("--owner-id", owner_id))
        for model_reference in model_references:
            arguments.extend(("--model-reference", model_reference))
        resolved_profiles = encoded_resolved_llm_profile_payloads(
            payload,
            model_references,
        )
        if resolved_profiles is not None:
            arguments.extend(("--resolved-profiles", resolved_profiles))
        process = manager.controller.docker_popen(tuple(arguments))
        with _ACTIVE_SSH_STAGERS_LOCK:
            _ACTIVE_SSH_STAGERS[invocation_id] = (
                manager,
                spec,
                owner_id,
                process,
            )
        try:
            results = consume_staging_process(
                process,
                payload,
                _emit_local_llm_staging_progress,
                provider_label="SSH",
                abort_remote=lambda: self._cancel_remote_stager(
                    manager,
                    spec,
                    owner_id,
                ),
            )
        except RemoteStagingProcessError as exc:
            raise SshRemoteInvocationError(str(exc)) from exc
        finally:
            with _ACTIVE_SSH_STAGERS_LOCK:
                _ACTIVE_SSH_STAGERS.pop(invocation_id, None)
        downloaded_gib = sum(
            float(result.get("artifact_bytes") or 0) / 1024**3
            for result in results
            if result.get("downloaded")
        )
        _emit_local_remote_startup_status(
            payload,
            phase="llm_staged",
            status_message=(
                f"SSH LLM staging complete ({downloaded_gib:.1f} GiB downloaded)"
            ),
        )
        return results

    @staticmethod
    def _cancel_remote_stager(
        manager: SshRuntimeManager,
        spec: SshRuntimeSpec,
        owner_id: str,
        *,
        wait_for_owner_seconds: float = 0.0,
    ) -> bool:
        """Terminate one exact owner-tagged stager without recycling its worker."""
        deadline = time.monotonic() + max(0.0, wait_for_owner_seconds)
        while True:
            result = manager.controller.docker(
                (
                    "exec",
                    spec.container_name,
                    "python",
                    "-m",
                    "remote.ssh_worker",
                    "cancel-staging",
                    "--owner-id",
                    owner_id,
                ),
                timeout_seconds=15.0,
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    "SSH remote stager cancellation failed owner=%s error=%s.",
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
        environment_id: str,
        requested_references: list[str],
        stage_results: list[dict[str, Any]],
    ) -> None:
        """Validate and cache SSH staging results by request and immutable ID."""
        confirmed_references: set[str] = set()
        for result in stage_results:
            requested_reference = str(result.get("requested_reference") or "")
            profile_id = str(result.get("profile_id") or "")
            revision = str(result.get("revision") or "")
            if not requested_reference or not profile_id or not revision:
                continue
            _STAGED_SSH_PROFILE_RESULTS[(environment_id, requested_reference)] = result
            _STAGED_SSH_PROFILE_RESULTS[(environment_id, profile_id)] = result
            confirmed_references.add(requested_reference)
        missing_results = set(requested_references) - confirmed_references
        if missing_results:
            raise SshRemoteInvocationError(
                f"SSH model stager did not confirm models {sorted(missing_results)}."
            )

    def _invoke_stream(
        self,
        manager: SshRuntimeManager,
        spec: SshRuntimeSpec,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> Iterator[dict[str, Any]]:
        """Start one worker relay and yield Modal-compatible event mappings."""
        from .remote.local_ui_events import RemoteTransferProgressReporter

        request = encode_json_frame(
            RemoteFrameKind.REQUEST,
            {
                "invocation_id": str(payload["invocation_id"]),
                "payload": payload,
            },
        )
        process = manager.controller.docker_popen(self._relay_arguments(spec))
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise SshRemoteInvocationError("SSH worker relay did not expose binary streams.")
        stderr_chunks: list[bytes] = []

        def collect_stderr() -> None:
            """Drain bounded diagnostic stderr so the SSH process cannot deadlock."""
            while chunk := process.stderr.read(65536):
                stderr_chunks.append(chunk)
                if sum(len(value) for value in stderr_chunks) > 1024 * 1024:
                    del stderr_chunks[:-8]

        stderr_thread = threading.Thread(target=collect_stderr, daemon=True)
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
                    raise SshRemoteInvocationError(
                        f"SSH worker {error.get('error_type', 'Error')}: "
                        f"{error.get('message', 'remote execution failed')}"
                    )
                raise RemoteProtocolError(
                    f"Unexpected SSH worker response frame {kind.name}."
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
                    "SSH worker relay exited after terminal frame status=%d stderr=%s.",
                    returncode,
                    b"".join(stderr_chunks).decode("utf-8", errors="replace").strip(),
                )
        if not terminal_received:
            diagnostics = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
            raise SshRemoteTransportError(
                f"SSH worker stream ended without a result: {diagnostics or 'no diagnostics'}."
            )

    def _runtime(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SshRuntimeManager, SshRuntimeSpec]:
        """Resolve and prepare the worker assigned by the planner."""
        environment_id = str(payload.get("execution_environment_id") or "").strip()
        if not environment_id:
            raise SshRemoteInvocationError(
                "SSH execution payload is missing execution_environment_id."
            )
        raw_host = payload.get("ssh_host_config")
        if isinstance(raw_host, Mapping):
            try:
                host = SshHostConfig.from_dict(raw_host)
            except (TypeError, ValueError) as exc:
                raise SshRemoteInvocationError(
                    "SSH execution payload contains an invalid workflow host "
                    f"configuration: {exc}"
                ) from exc
            if host.environment_id != environment_id:
                raise SshRemoteInvocationError(
                    "SSH execution payload environment identity does not match its "
                    "workflow host configuration."
                )
        else:
            if self.registry is None:
                raise SshRemoteInvocationError(
                    "Legacy SSH execution requires a persistent host registry."
                )
            host = self.registry.get_host(environment_id)
        if not host.enabled or host.draining:
            raise SshRemoteInvocationError(
                f"SSH execution environment {environment_id!r} is not accepting work."
            )
        controller = SshDockerController(host)
        if host.capabilities is None:
            logger.info(
                "Reprobing SSH execution capabilities missing from the queued host "
                "snapshot environment=%s.",
                environment_id,
            )
            try:
                host = replace(host, capabilities=controller.probe_capabilities())
            except (OSError, SshDockerError, ValueError) as exc:
                raise SshRemoteInvocationError(
                    f"Unable to refresh SSH execution capabilities for "
                    f"{environment_id!r}: {exc}"
                ) from exc
            controller = SshDockerController(host)
        resolved_settings = self.settings or get_settings()
        manager = SshRuntimeManager(
            controller=controller,
            repo_root=self.repo_root,
            settings=resolved_settings,
        )
        worker_index = int(payload.get("execution_worker_index", 0))

        def emit_runtime_status(message: str) -> None:
            """Forward worker lifecycle progress to its Configurator row."""
            from .remote.local_ui_events import _emit_local_remote_startup_status

            _emit_local_remote_startup_status(
                payload,
                phase="starting",
                status_message=message,
            )

        return manager, manager.ensure_worker(
            worker_index,
            status_callback=emit_runtime_status,
        )

    def _relay_arguments(self, spec: SshRuntimeSpec) -> tuple[str, ...]:
        """Return the Docker exec argv for the worker's binary relay client."""
        return (
            "exec",
            "-i",
            spec.container_name,
            "python",
            "-m",
            "remote.ssh_worker",
            "client",
        )


def _collect_bounded_stream(stream: Any, chunks: list[bytes]) -> None:
    """Drain a diagnostic stream while retaining at most its recent output."""
    while chunk := stream.read(65536):
        chunks.append(chunk)
        if sum(len(value) for value in chunks) > 1024 * 1024:
            del chunks[:-8]


class _BytesReader:
    """Minimal binary reader wrapper used for one-shot protocol responses."""

    def __init__(self, payload: bytes) -> None:
        """Initialize a read cursor over immutable bytes."""
        self._payload = payload
        self._offset = 0

    def read(self, length: int = -1) -> bytes:
        """Read up to ``length`` bytes from the cursor."""
        if length < 0:
            length = len(self._payload) - self._offset
        end = min(len(self._payload), self._offset + length)
        result = self._payload[self._offset:end]
        self._offset = end
        return result
