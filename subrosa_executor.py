"""Remote executor client backed by the Subrosa lane-framed WebSocket relay."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import struct
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import aiohttp

if __package__:
    from .durable_state import stable_remote_invocation_id
    from .remote_protocol import (
        REMOTE_PROTOCOL_MAGIC,
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_frame,
        encode_json_frame,
    )
    from .serialization import deserialize_node_outputs, serialize_node_inputs
    from .settings import ModalSyncSettings, get_settings
    from .subrosa_credentials import SubrosaCredentialStore
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from durable_state import stable_remote_invocation_id
    from remote_protocol import (
        REMOTE_PROTOCOL_MAGIC,
        RemoteFrameKind,
        RemoteProtocolError,
        decode_json_payload,
        encode_frame,
        encode_json_frame,
    )
    from serialization import deserialize_node_outputs, serialize_node_inputs
    from settings import ModalSyncSettings, get_settings
    from subrosa_credentials import SubrosaCredentialStore

logger = logging.getLogger(__name__)

_LANE_STREAM = 0
_LANE_CONTROL = 1
_STREAM_CHUNK_BYTES = 512 * 1024
_FRAME_HEADER = struct.Struct(">8sBQ")
_MAX_FRAME_BYTES = 16 * 1024**3
_READY_TIMEOUT_SECONDS = 130.0
_SETTLEMENT_TIMEOUT_SECONDS = 15.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) ComfyUI-Modal/0.4.2"
)


class SubrosaRemoteInvocationError(RuntimeError):
    """Raised when a Subrosa worker rejects an invocation."""


class SubrosaRemoteResourceError(SubrosaRemoteInvocationError):
    """Raised for terminal worker resource exhaustion such as GPU OOM."""


class SubrosaRemoteTransportError(SubrosaRemoteInvocationError):
    """Raised for retryable relay or worker-process transport loss."""


class SubrosaRelayRejectedError(SubrosaRemoteInvocationError):
    """Raised when the relay rejects authentication, pool, or account state."""


@dataclass(frozen=True)
class SubrosaSettlement:
    """Describe relay-authoritative metering for one completed invocation."""

    status: str
    gpu_seconds: float | None = None
    centicredits: float | None = None
    worker_reported_seconds: float | None = None

    @classmethod
    def from_control(cls, payload: Mapping[str, Any]) -> SubrosaSettlement:
        """Parse one lane-control settlement object."""
        return cls(
            status=str(payload.get("status") or "unknown"),
            gpu_seconds=_optional_float(payload.get("gpu_seconds")),
            centicredits=_optional_float(payload.get("centicredits")),
            worker_reported_seconds=_optional_float(
                payload.get("worker_reported_seconds")
            ),
        )


@dataclass(frozen=True)
class _ActiveRelay:
    """Retain the event-loop-owned socket needed for same-connection cancellation."""

    loop: asyncio.AbstractEventLoop
    websocket: aiohttp.ClientWebSocketResponse


@dataclass(frozen=True)
class _StreamFailure:
    """Carry a producer exception through the synchronous stream bridge."""

    error: BaseException


_STREAM_END = object()


class _LaneZeroFrameDecoder:
    """Reassemble CRMTRPC1 frames from arbitrary lane-0 WebSocket chunks."""

    def __init__(self) -> None:
        """Initialize an empty byte-stream buffer."""
        self._buffer = bytearray()

    def push(self, chunk: bytes) -> list[tuple[RemoteFrameKind, bytes]]:
        """Append one lane chunk and return every newly complete protocol frame."""
        self._buffer.extend(chunk)
        frames: list[tuple[RemoteFrameKind, bytes]] = []
        while len(self._buffer) >= _FRAME_HEADER.size:
            magic, raw_kind, payload_length = _FRAME_HEADER.unpack_from(self._buffer)
            if magic != REMOTE_PROTOCOL_MAGIC:
                raise RemoteProtocolError(
                    "Subrosa lane-0 stream has an invalid CRMTRPC1 magic value."
                )
            if payload_length > _MAX_FRAME_BYTES:
                raise RemoteProtocolError(
                    f"Subrosa lane-0 frame exceeds {_MAX_FRAME_BYTES} bytes."
                )
            frame_length = _FRAME_HEADER.size + payload_length
            if len(self._buffer) < frame_length:
                break
            try:
                kind = RemoteFrameKind(raw_kind)
            except ValueError as exc:
                raise RemoteProtocolError(
                    f"Subrosa lane-0 stream contains unknown frame kind {raw_kind}."
                ) from exc
            payload = bytes(self._buffer[_FRAME_HEADER.size:frame_length])
            del self._buffer[:frame_length]
            frames.append((kind, payload))
        return frames

    def require_empty(self) -> None:
        """Reject a relay close that truncates a protocol frame."""
        if self._buffer:
            raise RemoteProtocolError(
                f"Subrosa relay closed with {len(self._buffer)} partial lane-0 bytes."
            )


@dataclass
class SubrosaExecutorClient:
    """Execute serialized ComfyUI components through one Subrosa relay pool."""

    credential_store: SubrosaCredentialStore = field(
        default_factory=SubrosaCredentialStore,
        repr=False,
    )
    settings: ModalSyncSettings | None = None
    ready_timeout_seconds: float = _READY_TIMEOUT_SECONDS
    settlement_timeout_seconds: float = _SETTLEMENT_TIMEOUT_SECONDS
    last_settlement: SubrosaSettlement | None = field(default=None, init=False)
    _active_relays: dict[str, _ActiveRelay] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _active_relays_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def execute_payload(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one payload synchronously and deserialize its outputs."""
        inputs_payload = serialize_node_inputs(kwargs)
        prepared_payload = self._prepare_payload(payload, inputs_payload)
        result = self._consume_stream(prepared_payload, inputs_payload)
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
            await asyncio.to_thread(self.cancel, invocation_id)
            raise
        return deserialize_node_outputs(result)

    def cancel(self, invocation_id: str) -> bool:
        """Send cancellation on the active invocation's existing relay socket."""
        with self._active_relays_lock:
            active = self._active_relays.get(invocation_id)
        if active is None:
            return False
        payload = bytes((_LANE_CONTROL,)) + _control_json({"type": "cancel"})
        try:
            future = asyncio.run_coroutine_threadsafe(
                active.websocket.send_bytes(payload),
                active.loop,
            )
            future.result(timeout=5.0)
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.warning(
                "Unable to send Subrosa cancellation invocation=%s error=%s.",
                invocation_id,
                exc,
            )
            return False
        logger.info("Requested Subrosa cancellation invocation=%s.", invocation_id)
        return True

    async def whoami(
        self,
        relay_url: str,
        credential_id: str,
    ) -> dict[str, Any]:
        """Validate a keyring-backed configuration against the relay REST API."""
        token = self.credential_store.require(credential_id)
        url = _http_base_url(relay_url) + "/api/v1/extension/whoami"
        timeout = aiohttp.ClientTimeout(total=30.0)
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": _USER_AGENT,
        }
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(url, headers=headers) as response,
            ):
                body = await response.json(content_type=None)
                if response.status != 200:
                    detail = _safe_error_message(body)
                    raise SubrosaRelayRejectedError(
                        f"Subrosa configuration validation failed with HTTP "
                        f"{response.status}: {detail}"
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise SubrosaRemoteTransportError(
                f"Subrosa configuration validation could not reach the relay: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise SubrosaRemoteTransportError(
                "Subrosa whoami returned an invalid response object."
            )
        return dict(body)

    def _prepare_payload(
        self,
        payload: Mapping[str, Any],
        inputs_payload: bytes,
    ) -> dict[str, Any]:
        """Attach stable invocation identity while preserving queue-time provider."""
        prepared = dict(payload)
        prepared["execution_provider"] = str(
            prepared.get("execution_provider") or "subrosa"
        ).strip().lower()
        prepared.setdefault(
            "invocation_id",
            stable_remote_invocation_id(prepared, inputs_payload),
        )
        prepared.setdefault("capture_remote_outputs", True)
        return prepared

    def _consume_stream(self, payload: dict[str, Any], inputs_payload: bytes) -> bytes:
        """Consume relay events through the provider-neutral local UI stream."""
        if __package__:
            from .remote.modal_app import (
                _consume_remote_payload_stream,
                _materialize_remote_execution_result,
            )
        else:  # pragma: no cover - direct ComfyUI loading fallback.
            from remote.modal_app import (
                _consume_remote_payload_stream,
                _materialize_remote_execution_result,
            )

        response: bytes | None = None
        for attempt in range(1, 3):
            try:
                response = _consume_remote_payload_stream(
                    payload,
                    self._invoke_stream(payload, inputs_payload),
                )
                break
            except (aiohttp.ClientError, OSError, RemoteProtocolError, SubrosaRemoteTransportError) as exc:
                if attempt >= 2:
                    raise SubrosaRemoteTransportError(
                        "Subrosa relay transport failed after one recovery attempt: "
                        f"{exc}"
                    ) from exc
                logger.warning(
                    "Retrying Subrosa relay invocation=%s after transport failure: %s.",
                    payload.get("invocation_id"),
                    exc,
                )
        if response is None:
            raise SubrosaRemoteTransportError(
                "Subrosa relay returned no execution result."
            )
        return _materialize_remote_execution_result(
            response,
            settings=self.settings or get_settings(),
        )

    def _invoke_stream(
        self,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> Iterator[dict[str, Any]]:
        """Bridge async WebSocket events into the existing synchronous consumer."""
        items: queue.Queue[dict[str, Any] | _StreamFailure | object] = queue.Queue()

        def produce() -> None:
            """Run the relay socket in its own event loop and publish stream items."""
            try:
                asyncio.run(
                    self._run_relay(
                        payload,
                        inputs_payload,
                        lambda event: items.put(event),
                    )
                )
            except (
                aiohttp.ClientError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                items.put(_StreamFailure(exc))
            finally:
                items.put(_STREAM_END)

        producer = threading.Thread(
            target=produce,
            name=f"subrosa-relay-{payload['invocation_id']}",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = items.get()
                if item is _STREAM_END:
                    break
                if isinstance(item, _StreamFailure):
                    raise item.error
                if not isinstance(item, dict):
                    raise SubrosaRemoteTransportError(
                        "Subrosa stream bridge received an invalid event."
                    )
                yield item
        finally:
            producer.join(timeout=1.0)

    async def _run_relay(
        self,
        payload: dict[str, Any],
        inputs_payload: bytes,
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        """Open one relay job, stream progress, and await terminal settlement."""
        relay_url = _required_payload_string(payload, "relay_url").rstrip("/")
        pool = _required_payload_string(payload, "pool")
        credential_id = str(
            payload.get("credential_id") or payload.get("configuration_id") or ""
        ).strip()
        token = self.credential_store.require(credential_id)
        invocation_id = _required_payload_string(payload, "invocation_id")
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Subrosa-Pool": pool,
            "User-Agent": _USER_AGENT,
        }
        timeout = aiohttp.ClientTimeout(total=None, connect=30.0, sock_read=None)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.ws_connect(
                    relay_url + "/api/v1/relay/client",
                    headers=headers,
                    heartbeat=30.0,
                    max_msg_size=0,
                ) as websocket,
            ):
                loop = asyncio.get_running_loop()
                with self._active_relays_lock:
                    self._active_relays[invocation_id] = _ActiveRelay(
                        loop=loop,
                        websocket=websocket,
                    )
                try:
                    await self._wait_until_ready(websocket)
                    await self._send_request_frames(
                        websocket,
                        payload,
                        inputs_payload,
                    )
                    terminal_event, settlement = await self._receive_response(
                        websocket,
                        payload,
                        emit,
                    )
                finally:
                    with self._active_relays_lock:
                        self._active_relays.pop(invocation_id, None)
        except aiohttp.WSServerHandshakeError as exc:
            if exc.status in {401, 402, 404}:
                raise SubrosaRelayRejectedError(
                    _handshake_rejection_message(exc.status)
                ) from exc
            raise SubrosaRemoteTransportError(
                f"Subrosa WebSocket upgrade failed with HTTP {exc.status}."
            ) from exc
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise SubrosaRemoteTransportError(
                f"Subrosa relay connection failed: {exc}"
            ) from exc

        self.last_settlement = settlement
        if settlement is not None:
            emit(_settlement_status_event(settlement))
        emit(terminal_event)

    async def _wait_until_ready(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        """Wait until the relay has claimed and attached a worker."""
        async with asyncio.timeout(self.ready_timeout_seconds):
            while True:
                message = await websocket.receive()
                data = _binary_message_data(message, "before ready")
                lane, lane_payload = data[0], data[1:]
                if lane != _LANE_CONTROL:
                    raise RemoteProtocolError(
                        "Subrosa relay sent lane-0 bytes before the ready signal."
                    )
                control = _decode_control(lane_payload)
                control_type = str(control.get("type") or "")
                if control_type == "ready":
                    return
                if control_type == "error":
                    raise SubrosaRemoteInvocationError(
                        f"Subrosa relay: {_safe_error_message(control)}"
                    )

    async def _send_request_frames(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        payload: dict[str, Any],
        inputs_payload: bytes,
    ) -> None:
        """Send REQUEST and INPUTS as one chunked lane-0 byte stream."""
        request = encode_json_frame(
            RemoteFrameKind.REQUEST,
            {
                "invocation_id": str(payload["invocation_id"]),
                "payload": payload,
            },
        )
        stream = request + encode_frame(RemoteFrameKind.INPUTS, inputs_payload)
        for offset in range(0, len(stream), _STREAM_CHUNK_BYTES):
            chunk = stream[offset : offset + _STREAM_CHUNK_BYTES]
            await websocket.send_bytes(bytes((_LANE_STREAM,)) + chunk)
        logger.info(
            "Sent Subrosa request invocation=%s pool=%s stream_bytes=%d chunks=%d.",
            payload.get("invocation_id"),
            payload.get("pool"),
            len(stream),
            max(1, (len(stream) + _STREAM_CHUNK_BYTES - 1) // _STREAM_CHUNK_BYTES),
        )

    async def _receive_response(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        payload: Mapping[str, Any],
        emit: Callable[[dict[str, Any]], None],
    ) -> tuple[dict[str, Any], SubrosaSettlement | None]:
        """Stream response frames and retain settlement before exposing terminal data."""
        decoder = _LaneZeroFrameDecoder()
        terminal_event: dict[str, Any] | None = None
        terminal_error: SubrosaRemoteInvocationError | None = None
        settlement: SubrosaSettlement | None = None
        execution_timeout = float(
            (self.settings or get_settings()).execution_timeout_seconds
        )
        async with asyncio.timeout(execution_timeout + self.ready_timeout_seconds):
            while True:
                receive_timeout = (
                    self.settlement_timeout_seconds
                    if terminal_event is not None or terminal_error is not None
                    else None
                )
                try:
                    message = await websocket.receive(timeout=receive_timeout)
                except TimeoutError:
                    if terminal_event is not None or terminal_error is not None:
                        logger.warning(
                            "Timed out waiting for Subrosa settlement invocation=%s.",
                            payload.get("invocation_id"),
                        )
                        break
                    raise
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                }:
                    break
                data = _binary_message_data(message, "while receiving response")
                lane, lane_payload = data[0], data[1:]
                if lane == _LANE_CONTROL:
                    control = _decode_control(lane_payload)
                    control_type = str(control.get("type") or "")
                    if control_type == "settled":
                        settlement = SubrosaSettlement.from_control(control)
                        if terminal_event is not None or terminal_error is not None:
                            break
                    elif control_type == "error":
                        raise SubrosaRemoteInvocationError(
                            f"Subrosa relay: {_safe_error_message(control)}"
                        )
                    continue
                if lane != _LANE_STREAM:
                    raise RemoteProtocolError(
                        f"Subrosa relay sent unsupported lane {lane}."
                    )
                if terminal_event is not None or terminal_error is not None:
                    raise RemoteProtocolError(
                        "Subrosa relay sent lane-0 bytes after a terminal frame."
                    )
                for kind, frame_payload in decoder.push(lane_payload):
                    if terminal_event is not None or terminal_error is not None:
                        raise RemoteProtocolError(
                            "Subrosa relay sent a protocol frame after a terminal frame."
                        )
                    if kind is RemoteFrameKind.PROGRESS:
                        emit(decode_json_payload(frame_payload))
                        continue
                    if kind is RemoteFrameKind.RESULT:
                        terminal_event = {"kind": "result", "outputs": frame_payload}
                        continue
                    if kind is RemoteFrameKind.ERROR:
                        terminal_error = _subrosa_invocation_error(
                            decode_json_payload(frame_payload)
                        )
                        continue
                    raise RemoteProtocolError(
                        f"Unexpected Subrosa worker response frame {kind.name}."
                    )
        decoder.require_empty()
        if terminal_error is not None:
            raise terminal_error
        if terminal_event is None:
            settlement_status = settlement.status if settlement is not None else "none"
            if settlement_status == "lost":
                raise SubrosaRemoteTransportError(
                    "Subrosa worker was lost before returning a terminal frame."
                )
            if settlement_status == "cancelled":
                raise SubrosaRemoteInvocationError(
                    "Subrosa invocation was cancelled before returning a terminal frame."
                )
            raise SubrosaRemoteTransportError(
                "Subrosa relay closed without a terminal worker frame "
                f"(settlement={settlement_status})."
            )
        return terminal_event, settlement


def _subrosa_invocation_error(
    error: Mapping[str, Any],
) -> SubrosaRemoteInvocationError:
    """Classify both application and v0.4.2 worker-postmortem error shapes."""
    error_type = str(error.get("error_type") or "Error")
    failure_kind = str(error.get("failure_kind") or "")
    message = str(error.get("message") or "remote execution failed")
    if error_type == "WorkerOutOfMemoryError" or failure_kind == "out_of_memory":
        return SubrosaRemoteResourceError(message)
    if (
        error_type == "WorkerProcessLostError"
        or failure_kind == "worker_process_lost"
    ):
        return SubrosaRemoteTransportError(message)
    return SubrosaRemoteInvocationError(
        f"Subrosa worker {error_type}: {message}"
    )


def _binary_message_data(message: aiohttp.WSMessage, phase: str) -> bytes:
    """Return non-empty binary lane data or raise a transport/protocol error."""
    if message.type is aiohttp.WSMsgType.ERROR:
        error = message.data
        raise SubrosaRemoteTransportError(
            f"Subrosa WebSocket failed {phase}: {error}"
        )
    if message.type is not aiohttp.WSMsgType.BINARY:
        raise RemoteProtocolError(
            f"Subrosa relay sent a non-binary WebSocket message {phase}."
        )
    data = bytes(message.data)
    if not data:
        raise RemoteProtocolError(f"Subrosa relay sent an empty message {phase}.")
    return data


def _decode_control(payload: bytes) -> dict[str, Any]:
    """Decode one lane-1 JSON object."""
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteProtocolError("Subrosa control JSON is invalid.") from exc
    if not isinstance(value, dict):
        raise RemoteProtocolError("Subrosa control payload must be an object.")
    return value


def _control_json(payload: Mapping[str, Any]) -> bytes:
    """Encode one compact lane-1 control object."""
    return json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _required_payload_string(payload: Mapping[str, Any], key: str) -> str:
    """Return one required non-empty queue-time provider metadata value."""
    value = str(payload.get(key) or "").strip()
    if not value:
        raise SubrosaRemoteInvocationError(
            f"Subrosa execution payload is missing {key!r}."
        )
    return value


def _http_base_url(relay_url: str) -> str:
    """Convert one WebSocket relay base URL to its HTTP API base."""
    normalized = relay_url.strip().rstrip("/")
    if normalized.startswith("wss://"):
        return "https://" + normalized.removeprefix("wss://")
    if normalized.startswith("ws://"):
        return "http://" + normalized.removeprefix("ws://")
    raise ValueError("Subrosa relay_url must use ws:// or wss://.")


def _safe_error_message(payload: Any) -> str:
    """Return a bounded error string without echoing credential-bearing fields."""
    if isinstance(payload, Mapping):
        return str(payload.get("error") or payload.get("message") or "request rejected")[
            :500
        ]
    return "request rejected"


def _handshake_rejection_message(status: int) -> str:
    """Return a credential-safe explanation for a relay upgrade rejection."""
    messages = {
        401: "Subrosa rejected the extension token (HTTP 401).",
        402: "Subrosa account balance is insufficient for this pool (HTTP 402).",
        404: "Subrosa does not recognize the configured pool (HTTP 404).",
    }
    return messages.get(status, f"Subrosa rejected the relay connection (HTTP {status}).")


def _optional_float(value: Any) -> float | None:
    """Parse one optional finite relay measurement."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _settlement_status_event(settlement: SubrosaSettlement) -> dict[str, Any]:
    """Build a provider-neutral UI status event for relay metering."""
    details: list[str] = [f"Subrosa settled {settlement.status}"]
    if settlement.gpu_seconds is not None:
        details.append(f"{settlement.gpu_seconds:.2f} GPU-s")
    if settlement.centicredits is not None:
        details.append(f"{settlement.centicredits:g} centicredits")
    return {
        "kind": "progress",
        "event_type": "status",
        "phase": "settled",
        "message": " · ".join(details),
    }


__all__ = [
    "SubrosaExecutorClient",
    "SubrosaRelayRejectedError",
    "SubrosaRemoteInvocationError",
    "SubrosaRemoteResourceError",
    "SubrosaRemoteTransportError",
    "SubrosaSettlement",
]
