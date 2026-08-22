"""Remote executor client backed by warm Docker workers reached over SSH."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .durable_state import stable_remote_invocation_id
    from .remote_hosts import RemoteHostRegistry
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
    from .ssh_docker import SshDockerController, SshDockerError
    from .ssh_runtime import SshRuntimeManager, SshRuntimeSpec
else:  # pragma: no cover - top-level remote imports.
    from durable_state import stable_remote_invocation_id
    from remote_hosts import RemoteHostRegistry
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
    from ssh_docker import SshDockerController, SshDockerError
    from ssh_runtime import SshRuntimeManager, SshRuntimeSpec

logger = logging.getLogger(__name__)


class SshRemoteInvocationError(RuntimeError):
    """Raised when an SSH worker rejects or loses an invocation."""


@dataclass
class SshDockerExecutorClient:
    """Execute serialized ComfyUI components on one configured SSH host."""

    registry: RemoteHostRegistry
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

        manager, spec = self._runtime(payload)
        stream = self._invoke_stream(manager, spec, payload, inputs_payload)
        response = _consume_remote_payload_stream(payload, stream)
        return _materialize_remote_execution_result(
            response,
            settings=self.settings or get_settings(),
        )

    def _invoke_stream(
        self,
        manager: SshRuntimeManager,
        spec: SshRuntimeSpec,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> Iterator[dict[str, Any]]:
        """Start one worker relay and yield Modal-compatible event mappings."""
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
            raise SshRemoteInvocationError(
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
        host = self.registry.get_host(environment_id)
        if not host.enabled or host.draining:
            raise SshRemoteInvocationError(
                f"SSH execution environment {environment_id!r} is not accepting work."
            )
        resolved_settings = self.settings or get_settings()
        manager = SshRuntimeManager(
            controller=SshDockerController(host),
            repo_root=self.repo_root,
            settings=resolved_settings,
        )
        worker_index = int(payload.get("execution_worker_index", 0))
        return manager, manager.ensure_worker(worker_index)

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
