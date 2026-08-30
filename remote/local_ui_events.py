"""Local ComfyUI websocket event emission for remote execution."""

from __future__ import annotations

from io import BytesIO
import logging
import time
from typing import Any, Literal, Mapping

from .local_execution import _iter_prompt_links

logger = logging.getLogger(__name__)

_REMOTE_TRANSFER_PROGRESS_MIN_BYTES = 4 * 1024 * 1024
_REMOTE_TRANSFER_PROGRESS_MIN_INTERVAL_SECONDS = 0.1
_REMOTE_TRANSFER_PROGRESS_MAX_UPDATES = 1000


def _lookup_local_prompt_server() -> Any | None:
    """Return the live local ComfyUI PromptServer instance when available."""
    try:
        import server
    except ModuleNotFoundError:
        return None

    return getattr(server.PromptServer, "instance", None)


def _record_local_modal_ui_event(
    event: str, payload: Mapping[str, Any], client_id: str | None
) -> None:
    """Record one local Modal UI event for browser refocus replay when available."""
    if client_id is None:
        return
    try:
        from ..api_intercept import record_modal_ui_event
    except (ImportError, AttributeError):
        logger.debug(
            "Modal UI event replay recorder is unavailable for event %s.", event
        )
        return

    record_modal_ui_event(event, payload, client_id)


def _emit_local_modal_status(
    *,
    prompt_id: str | None,
    client_id: str | None,
    phase: str,
    node_ids: list[str],
    modal_gpu: str | None = None,
    active_node_id: str | None = None,
    active_node_class_type: str | None = None,
    active_node_role: str | None = None,
    error_message: str | None = None,
    status_message: str | None = None,
    status_current: int | None = None,
    status_total: int | None = None,
    completed_ancestor_node_ids: list[str] | None = None,
    execution_provider: str | None = None,
    execution_environment_id: str | None = None,
    execution_location: str | None = None,
) -> None:
    """Forward remote execution progress into the local ComfyUI websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload: dict[str, Any] = {
        "phase": phase,
        "prompt_id": prompt_id,
        "node_ids": list(node_ids),
    }
    if modal_gpu is not None:
        payload["modal_gpu"] = modal_gpu
    if active_node_id is not None:
        payload["active_node_id"] = active_node_id
    if active_node_class_type is not None:
        payload["active_node_class_type"] = active_node_class_type
    if active_node_role is not None:
        payload["active_node_role"] = active_node_role
    if error_message is not None:
        payload["error_message"] = error_message
    if status_message is not None:
        payload["status_message"] = status_message
    if status_current is not None:
        payload["status_current"] = int(status_current)
    if status_total is not None:
        payload["status_total"] = int(status_total)
    if completed_ancestor_node_ids:
        payload["completed_ancestor_node_ids"] = list(completed_ancestor_node_ids)
    if execution_provider:
        payload["execution_provider"] = execution_provider
    if execution_environment_id:
        payload["execution_environment_id"] = execution_environment_id
    if execution_location:
        payload["execution_location"] = execution_location
    _record_local_modal_ui_event("modal_status", payload, client_id)
    prompt_server.send_sync("modal_status", payload, client_id)


def _remote_execution_destination(payload: Mapping[str, Any]) -> str:
    """Return the user-facing destination for a provider-stamped payload."""
    environment_id = str(payload.get("execution_environment_id") or "").strip()
    provider = str(payload.get("execution_provider") or "modal").strip()
    return (
        environment_id
        if provider != "modal" and environment_id
        else "Modal"
    )


def _remote_execution_identity(
    payload: Mapping[str, Any],
    modal_task_id: str | None = None,
) -> dict[str, str]:
    """Return non-empty provider, environment, and runtime location fields."""
    provider = str(
        payload.get("execution_provider") or ("modal" if modal_task_id else "")
    ).strip()
    environment_id = str(payload.get("execution_environment_id") or "").strip()
    location = (
        modal_task_id
        if provider == "modal" and modal_task_id
        else str(payload.get("execution_location") or "").strip()
    )
    return {
        key: value
        for key, value in {
            "execution_provider": provider,
            "execution_environment_id": environment_id,
            "execution_location": location,
        }.items()
        if value
    }


def _emit_local_remote_dispatch_status(payload: dict[str, Any]) -> None:
    """Tell the local UI a remote component was dispatched before progress streams."""
    _emit_local_remote_startup_status(
        payload,
        phase="starting",
        status_message=(
            f"Starting remote component on {_remote_execution_destination(payload)}"
        ),
    )


def _emit_local_remote_startup_status(
    payload: Mapping[str, Any],
    *,
    phase: str,
    status_message: str,
) -> None:
    """Show one prompt-scoped Modal startup phase in the local global status pill."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    node_ids = [
        str(node_id)
        for node_id in payload.get("component_node_ids", [])
        if str(node_id)
    ]
    if not node_ids and payload.get("component_id") is not None:
        node_ids = [str(payload["component_id"])]
    if not prompt_id or not client_id or not node_ids:
        return
    _emit_local_modal_status(
        prompt_id=prompt_id,
        client_id=client_id,
        phase=phase,
        node_ids=node_ids,
        modal_gpu=(
            str(payload["modal_gpu"]) if payload.get("modal_gpu") is not None else None
        ),
        status_message=status_message,
        **_remote_execution_identity(payload),
    )


def _emit_local_modal_progress(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    value: float,
    max_value: float,
    display_node_id: str | None = None,
    real_node_id: str | None = None,
    lane_id: str | None = None,
    clear: bool = False,
    item_index: int | None = None,
    aggregate_only: bool = False,
    setup_only: bool = False,
    cached_hit: bool = False,
    completed_ancestor_node_ids: list[str] | None = None,
    stage: str | None = None,
    message: str | None = None,
    unit: str | None = None,
    indeterminate: bool = False,
    elapsed_seconds: float | None = None,
    time_to_first_token_seconds: float | None = None,
    tokens_per_second: float | None = None,
    pre_gpu: bool = False,
    execution_provider: str | None = None,
    execution_environment_id: str | None = None,
    execution_location: str | None = None,
) -> None:
    """Forward remote numeric and stage progress into the local websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "node_id": node_id,
        "value": float(value),
        "max": float(max_value),
    }
    if display_node_id is not None:
        payload["display_node_id"] = display_node_id
    if real_node_id is not None:
        payload["real_node_id"] = real_node_id
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if clear:
        payload["clear"] = True
    if item_index is not None:
        payload["item_index"] = int(item_index)
    if aggregate_only:
        payload["aggregate_only"] = True
    if setup_only:
        payload["setup_only"] = True
    if cached_hit:
        payload["cached_hit"] = True
    if completed_ancestor_node_ids:
        payload["completed_ancestor_node_ids"] = list(completed_ancestor_node_ids)
    if stage:
        payload["stage"] = stage
    if message:
        payload["message"] = message
    if unit:
        payload["unit"] = unit
    if indeterminate:
        payload["indeterminate"] = True
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = float(elapsed_seconds)
    if time_to_first_token_seconds is not None:
        payload["time_to_first_token_seconds"] = float(
            time_to_first_token_seconds
        )
    if tokens_per_second is not None:
        payload["tokens_per_second"] = float(tokens_per_second)
    if pre_gpu:
        payload["pre_gpu"] = True
    if execution_provider:
        payload["execution_provider"] = execution_provider
    if execution_environment_id:
        payload["execution_environment_id"] = execution_environment_id
    if execution_location:
        payload["execution_location"] = execution_location
    _record_local_modal_ui_event("modal_progress", payload, client_id)
    prompt_server.send_sync("modal_progress", payload, client_id)


class RemoteTransferProgressReporter:
    """Throttle byte-transfer progress while preserving start and completion events."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        direction: Literal["upload", "download"],
        total_bytes: int,
        indeterminate: bool = False,
    ) -> None:
        """Initialize a reporter for one remote payload transfer."""
        self._payload = payload
        self._direction = direction
        self._total_bytes = max(0, int(total_bytes))
        self._indeterminate = indeterminate
        self._last_emitted_bytes = -1
        self._last_emitted_at = 0.0

    @property
    def enabled(self) -> bool:
        """Return whether this transfer is large enough to surface in the UI."""
        return self._total_bytes >= _REMOTE_TRANSFER_PROGRESS_MIN_BYTES

    def start(self) -> None:
        """Emit the initial transfer state when the payload is large."""
        self.update(0, force=True)

    def update(self, transferred_bytes: int, *, force: bool = False) -> None:
        """Emit a throttled transfer update."""
        if not self.enabled:
            return
        current_bytes = min(max(0, int(transferred_bytes)), self._total_bytes)
        now = time.monotonic()
        minimum_delta = max(
            1024 * 1024,
            self._total_bytes // _REMOTE_TRANSFER_PROGRESS_MAX_UPDATES,
        )
        if not force and current_bytes < self._total_bytes:
            if current_bytes - self._last_emitted_bytes < minimum_delta:
                return
            if (
                now - self._last_emitted_at
                < _REMOTE_TRANSFER_PROGRESS_MIN_INTERVAL_SECONDS
            ):
                return
        self._last_emitted_bytes = current_bytes
        self._last_emitted_at = now
        _emit_local_remote_transfer_progress(
            self._payload,
            direction=self._direction,
            current_bytes=current_bytes,
            total_bytes=self._total_bytes,
            indeterminate=self._indeterminate and current_bytes < self._total_bytes,
        )

    def complete(self) -> None:
        """Emit a determinate completion update."""
        self.update(self._total_bytes, force=True)


def _emit_local_remote_transfer_progress(
    payload: Mapping[str, Any],
    *,
    direction: Literal["upload", "download"],
    current_bytes: int,
    total_bytes: int,
    indeterminate: bool = False,
) -> None:
    """Show one large remote transfer on the component's representative node."""
    if total_bytes < _REMOTE_TRANSFER_PROGRESS_MIN_BYTES:
        return
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data["client_id"])
        if isinstance(extra_data, Mapping) and extra_data.get("client_id") is not None
        else None
    )
    node_ids = [
        str(node_id)
        for node_id in payload.get("component_node_ids", [])
        if str(node_id)
    ]
    node_id = str(payload.get("component_id") or (node_ids[0] if node_ids else ""))
    if not node_id:
        return
    destination = _remote_execution_destination(payload)
    action = "Sending inputs to" if direction == "upload" else "Receiving outputs from"
    _emit_local_modal_progress(
        prompt_id=(
            str(payload["prompt_id"])
            if payload.get("prompt_id") is not None
            else None
        ),
        client_id=client_id,
        node_id=node_id,
        value=float(current_bytes),
        max_value=float(total_bytes),
        display_node_id=node_id,
        stage=direction,
        message=f"{action} {destination}",
        unit="bytes",
        indeterminate=indeterminate,
        pre_gpu=direction == "upload",
        **_remote_execution_identity(payload),
    )


def _emit_local_executed_output(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    display_node_id: str | None,
    output_payload: Any,
) -> None:
    """Forward one remote node's executed UI payload into the local websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload = {
        "prompt_id": prompt_id,
        "node": node_id,
        "display_node": display_node_id or node_id,
        "output": output_payload,
    }
    prompt_server.send_sync("executed", payload, client_id)


def _emit_local_preview_image(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    display_node_id: str | None,
    parent_node_id: str | None,
    real_node_id: str | None,
    image_type: str,
    image_bytes: bytes,
    max_size: int | None,
) -> None:
    """Forward one remote preview image into the local ComfyUI preview websocket path."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    try:
        from PIL import Image
        from protocol import BinaryEventTypes
    except ModuleNotFoundError:
        logger.warning(
            "Preview forwarding is unavailable because Pillow or ComfyUI protocol imports failed."
        )
        return

    with BytesIO(image_bytes) as image_buffer:
        image = Image.open(image_buffer)
        image.load()

    metadata: dict[str, Any] = {
        "node_id": node_id,
        "prompt_id": prompt_id,
        "display_node_id": display_node_id or node_id,
        "real_node_id": real_node_id or node_id,
    }
    if parent_node_id is not None:
        metadata["parent_node_id"] = parent_node_id

    prompt_server.send_sync(
        BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA,
        ((image_type, image, max_size), metadata),
        client_id,
    )


def _emit_local_preview_boundary_output(
    *,
    prompt_id: str | None,
    client_id: str | None,
    preview_target_node_ids: list[str],
    image_value: Any,
) -> None:
    """Render one streamed remote boundary IMAGE value into local PreviewImage UI events."""
    if client_id is None or not preview_target_node_ids:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    try:
        import nodes
    except ModuleNotFoundError:
        logger.warning(
            "Preview boundary streaming is unavailable because ComfyUI nodes could not be imported."
        )
        return

    preview_factory = getattr(nodes, "PreviewImage", None)
    if preview_factory is None:
        logger.warning(
            "Preview boundary streaming is unavailable because PreviewImage is not registered."
        )
        return

    preview_result = preview_factory().save_images(images=image_value)
    if not isinstance(preview_result, dict):
        return
    output_payload = preview_result.get("ui")
    if not isinstance(output_payload, dict):
        return

    for preview_target_node_id in preview_target_node_ids:
        prompt_server.send_sync(
            "executed",
            {
                "prompt_id": prompt_id,
                "node": preview_target_node_id,
                "display_node": preview_target_node_id,
                "output": output_payload,
            },
            client_id,
        )


def _allowed_suppressed_stream_node_ids(payload: dict[str, Any]) -> set[str]:
    """Return the node ids that may surface UI events for a suppressed mapped/static stream."""
    allowed_node_ids = {
        str(node_id) for node_id in payload.get("execute_node_ids", []) if str(node_id)
    }
    allowed_node_ids.update(
        str(boundary_output["node_id"])
        for boundary_output in payload.get("boundary_outputs", [])
        if boundary_output.get("node_id") is not None
        and str(boundary_output["node_id"])
    )
    return allowed_node_ids


def _should_forward_suppressed_stream_event(
    payload: dict[str, Any],
    reported_node_id: Any,
) -> bool:
    """Return whether a suppressed mapped/static stream event belongs to this payload."""
    if not bool(payload.get("suppress_status_stream")):
        return True
    if reported_node_id is None:
        return False
    allowed_node_ids = _allowed_suppressed_stream_node_ids(payload)
    if not allowed_node_ids:
        return True
    return str(reported_node_id) in allowed_node_ids


def _progress_stream_event_node_id(stream_event: dict[str, Any]) -> str | None:
    """Return the best node id to use for progress filtering and forwarding."""
    for candidate in (
        stream_event.get("real_node_id"),
        stream_event.get("node_id"),
        stream_event.get("display_node_id"),
    ):
        if candidate is None:
            continue
        candidate_text = str(candidate)
        if candidate_text:
            return candidate_text
    return None


def _progress_stream_event_metadata(
    stream_event: dict[str, Any]
) -> dict[str, str | None] | None:
    """Return normalized metadata for one streamed progress event."""
    reported_node_id = (
        str(stream_event["node_id"])
        if stream_event.get("node_id") is not None
        else None
    )
    display_node_id = (
        str(stream_event["display_node_id"])
        if stream_event.get("display_node_id") is not None
        else reported_node_id
    )
    real_node_id = (
        str(stream_event["real_node_id"])
        if stream_event.get("real_node_id") is not None
        else None
    )
    filter_node_id = real_node_id or reported_node_id or display_node_id
    if filter_node_id is None:
        return None
    return {
        "node_id": reported_node_id or filter_node_id,
        "display_node_id": display_node_id,
        "real_node_id": real_node_id,
        "filter_node_id": filter_node_id,
    }


def _remote_prompt_ancestor_node_ids(
    payload: dict[str, Any], node_id: str | None
) -> list[str]:
    """Return subgraph dependency ancestors for one currently executing remote node."""
    if node_id is None:
        return []
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict) or node_id not in prompt:
        return []

    ancestors: set[str] = set()
    pending = [node_id]
    while pending:
        current_node_id = str(pending.pop())
        prompt_node = prompt.get(current_node_id)
        if not isinstance(prompt_node, dict):
            continue
        inputs = prompt_node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        for input_value in inputs.values():
            for input_link in _iter_prompt_links(input_value):
                upstream_node_id = str(input_link[0])
                if upstream_node_id == node_id or upstream_node_id in ancestors:
                    continue
                if upstream_node_id not in prompt:
                    continue
                ancestors.add(upstream_node_id)
                pending.append(upstream_node_id)
    return sorted(ancestors)


def _should_stream_remote_progress(payload: dict[str, Any]) -> bool:
    """Return whether the local client has enough context to mirror remote node progress."""
    extra_data = payload.get("extra_data") or {}
    return (
        payload.get("payload_kind") in {"subgraph", "mapped_subgraph"}
        and isinstance(payload.get("prompt_id"), str)
        and bool(payload.get("prompt_id"))
        and isinstance(extra_data.get("client_id"), str)
        and bool(extra_data.get("client_id"))
        and isinstance(payload.get("component_node_ids"), list)
        and len(payload.get("component_node_ids")) > 0
    )
