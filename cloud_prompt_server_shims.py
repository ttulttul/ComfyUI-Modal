"""Headless ComfyUI prompt-server shims and execution event tracing."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudPromptServerHooks:
    """Callbacks supplied by the cloud orchestration and execution owners."""

    collapse_cache_slot: Callable[[Any, bool], Any]
    emit_cloud_info: Callable[..., None]
    meaningful_progress_values: Callable[[dict[str, Any]], tuple[float, float] | None]


_PROMPT_SERVER_HOOKS: CloudPromptServerHooks | None = None


def configure_cloud_prompt_server_hooks(hooks: CloudPromptServerHooks) -> None:
    """Install prompt-server callbacks without importing upward into the entrypoint."""
    global _PROMPT_SERVER_HOOKS
    _PROMPT_SERVER_HOOKS = hooks


def _prompt_server_hooks() -> CloudPromptServerHooks:
    """Return configured callbacks or fail with a clear import-order error."""
    if _PROMPT_SERVER_HOOKS is None:
        raise RuntimeError("Cloud prompt-server hooks have not been configured.")
    return _PROMPT_SERVER_HOOKS


def _collapse_cache_slot(slot_values: Any, is_list: bool) -> Any:
    """Delegate cache-slot normalization to prompt execution."""
    return _prompt_server_hooks().collapse_cache_slot(slot_values, is_list)


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _prompt_server_hooks().emit_cloud_info(message, *args)


def _meaningful_progress_values(
    node_state: dict[str, Any],
) -> tuple[float, float] | None:
    """Delegate progress-state filtering to the stable entrypoint."""
    return _prompt_server_hooks().meaningful_progress_values(node_state)


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.last_prompt_id: str | None = None

    def send_sync(
        self, event: str, data: dict[str, Any], client_id: str | None
    ) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug(
            "Suppressed remote prompt event %s for client %s.", event, client_id
        )


class _HeadlessPromptQueue:
    """In-memory PromptQueue compatibility surface for remote custom-node hooks."""

    def __init__(self) -> None:
        """Initialize the queue collections exposed by ComfyUI's PromptQueue."""
        self._lock = threading.RLock()
        self.queue: list[Any] = []
        self.currently_running: dict[int, Any] = {}
        self.history: dict[str, Any] = {}
        self.flags: dict[str, Any] = {}

    def put(self, item: Any) -> None:
        """Retain a custom-node requeue request for compatibility diagnostics."""
        with self._lock:
            self.queue.append(item)
        logger.warning(
            "A remote custom node queued headless follow-up work; the request is retained "
            "for compatibility but has no background queue consumer."
        )

    def get_current_queue(self) -> tuple[list[Any], list[Any]]:
        """Return snapshots of running and pending compatibility queue items."""
        with self._lock:
            return list(self.currently_running.values()), list(self.queue)

    def get_current_queue_volatile(self) -> tuple[list[Any], list[Any]]:
        """Return shallow running and pending queue snapshots."""
        return self.get_current_queue()

    def get_tasks_remaining(self) -> int:
        """Return the number of running and pending compatibility tasks."""
        with self._lock:
            return len(self.currently_running) + len(self.queue)

    def set_flag(self, name: str, data: Any) -> None:
        """Store a queue flag for custom nodes that inspect PromptQueue state."""
        with self._lock:
            self.flags[name] = data

    def get_flags(self, reset: bool = True) -> dict[str, Any]:
        """Return queue flags and optionally clear them."""
        with self._lock:
            flags = dict(self.flags)
            if reset:
                self.flags.clear()
            return flags


class _HeadlessPromptServerInstance:
    """Minimal PromptServer.instance replacement for custom-node import side effects."""

    def __init__(self, node_replace_manager: Any | None) -> None:
        """Initialize route registration and no-op websocket state."""
        from aiohttp import web

        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.node_replace_manager = node_replace_manager
        self.supports = ["custom_nodes_from_web"]
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.number = 0
        self.prompt_queue = _HeadlessPromptQueue()
        self.on_prompt_handlers: list[Any] = []

    def queue_updated(self) -> None:
        """Accept PromptQueue update notifications without a websocket client."""
        logger.debug("Suppressed headless remote prompt queue update.")

    async def send(
        self, event: str, data: dict[str, Any], sid: str | None = None
    ) -> None:
        """Discard async websocket sends from import-time custom-node helpers."""
        logger.debug(
            "Suppressed headless remote prompt event %s for client %s.", event, sid
        )

    def send_sync(
        self, event: str, data: dict[str, Any], sid: str | None = None
    ) -> None:
        """Discard sync websocket sends from import-time custom-node helpers."""
        logger.debug(
            "Suppressed headless remote prompt event %s for client %s.", event, sid
        )

    def send_progress_text(
        self,
        text: bytes | bytearray | str,
        node_id: str,
        sid: str | None = None,
    ) -> None:
        """Discard node progress text emitted by headless remote node execution."""
        del text
        logger.debug(
            "Suppressed headless remote prompt text for node %s client %s.",
            node_id,
            sid,
        )

    def add_on_prompt_handler(self, handler: Any) -> None:
        """Record prompt handlers registered by custom nodes during import."""
        self.on_prompt_handlers.append(handler)


class _TracingPromptServer(_NullPromptServer):
    """PromptExecutor server stub that records coarse per-node execution timings."""

    def __init__(
        self,
        prompt_id: str,
        prompt: dict[str, Any],
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize timing state for a specific prompt execution."""
        super().__init__()
        self.prompt_id = prompt_id
        self.prompt = prompt
        self._status_callback = status_callback
        self._active_node_id: str | None = None
        self._active_node_started_at: float | None = None
        self.last_prompt_id = prompt_id
        self._boundary_outputs_by_node_id: dict[str, list[dict[str, Any]]] = {}
        self._lookup_cache_entry: Callable[[str], Any | None] | None = None
        self._published_boundary_outputs: set[tuple[str, int]] = set()

    def _classify_node_role(self, class_type: str) -> str:
        """Return a coarse role name for a node class."""
        normalized = class_type.lower()
        if "loader" in normalized or normalized in {"clipvisionencode"}:
            return "model_load"
        if "ksampler" in normalized or "sampler" in normalized:
            return "sampling"
        if "encode" in normalized:
            return "conditioning"
        return "node"

    def _log_node_finish(self, reason: str) -> None:
        """Emit a timing line for the currently active node when one is running."""
        if self._active_node_id is None or self._active_node_started_at is None:
            return

        node_id = self._active_node_id
        node_info = self.prompt.get(node_id, {})
        class_type = str(node_info.get("class_type", "<unknown>"))
        role = self._classify_node_role(class_type)
        elapsed_seconds = time.perf_counter() - self._active_node_started_at
        _emit_cloud_info(
            "Remote node %s class_type=%s role=%s finished in %.3fs reason=%s",
            node_id,
            class_type,
            role,
            elapsed_seconds,
            reason,
        )
        self._active_node_id = None
        self._active_node_started_at = None

    def emit_preview_update(
        self,
        *,
        node_id: str,
        preview_image: Any,
    ) -> None:
        """Publish one preview image update through the status callback."""
        if self._status_callback is None:
            return

        try:
            image_type, image, max_size = preview_image
            image_buffer = BytesIO()
            save_kwargs: dict[str, Any] = {"format": image_type}
            if image_type == "JPEG":
                save_kwargs["quality"] = 95
            elif image_type == "PNG":
                save_kwargs["compress_level"] = 1
            image.save(image_buffer, **save_kwargs)
        except Exception:
            logger.exception(
                "Failed to serialize remote preview image for node %s.", node_id
            )
            return

        try:
            from comfy_execution.progress import get_progress_state

            registry = get_progress_state()
            display_node_id = registry.dynprompt.get_display_node_id(node_id)
            parent_node_id = registry.dynprompt.get_parent_node_id(node_id)
            real_node_id = registry.dynprompt.get_real_node_id(node_id)
        except Exception:
            logger.exception(
                "Failed to resolve preview metadata for remote node %s.", node_id
            )
            display_node_id = node_id
            parent_node_id = None
            real_node_id = node_id

        self._status_callback(
            {
                "event_type": "preview",
                "node_id": str(node_id),
                "display_node_id": (
                    str(display_node_id) if display_node_id is not None else None
                ),
                "parent_node_id": (
                    str(parent_node_id) if parent_node_id is not None else None
                ),
                "real_node_id": str(real_node_id) if real_node_id is not None else None,
                "image_type": str(image_type),
                "image_bytes": image_buffer.getvalue(),
                "max_size": int(max_size) if max_size is not None else None,
            }
        )

    def configure_boundary_output_stream(
        self,
        *,
        boundary_outputs: list[dict[str, Any]],
        lookup_cache_entry: Callable[[str], Any | None],
    ) -> None:
        """Configure streamed remote boundary-output publication for this execution."""
        outputs_by_node_id: dict[str, list[dict[str, Any]]] = {}
        for boundary_output in boundary_outputs:
            preview_target_node_ids = [
                str(node_id)
                for node_id in boundary_output.get("preview_target_node_ids", [])
                if str(node_id)
            ]
            if not preview_target_node_ids:
                continue
            if str(boundary_output.get("io_type")) != "IMAGE":
                continue
            node_id = str(boundary_output["node_id"])
            outputs_by_node_id.setdefault(node_id, []).append(boundary_output)

        self._boundary_outputs_by_node_id = outputs_by_node_id
        self._lookup_cache_entry = lookup_cache_entry
        self._published_boundary_outputs.clear()

    def _emit_boundary_outputs_for_node(self, node_id: str | None) -> None:
        """Publish configured boundary image outputs for one completed node once."""
        if (
            node_id is None
            or self._status_callback is None
            or self._lookup_cache_entry is None
        ):
            return

        boundary_outputs = self._boundary_outputs_by_node_id.get(str(node_id), [])
        if not boundary_outputs:
            return

        cache_entry = self._lookup_cache_entry(str(node_id))
        if cache_entry is None:
            return

        cache_outputs = getattr(cache_entry, "outputs", None)
        if not isinstance(cache_outputs, (list, tuple)):
            return

        for boundary_output in boundary_outputs:
            output_index = int(boundary_output["output_index"])
            publication_key = (str(node_id), output_index)
            if publication_key in self._published_boundary_outputs:
                continue
            if output_index >= len(cache_outputs):
                continue

            preview_target_node_ids = [
                str(target_node_id)
                for target_node_id in boundary_output.get("preview_target_node_ids", [])
                if str(target_node_id)
            ]
            if not preview_target_node_ids:
                continue

            self._status_callback(
                {
                    "event_type": "boundary_output",
                    "node_id": str(node_id),
                    "output_index": output_index,
                    "io_type": str(boundary_output.get("io_type", "")),
                    "is_list": bool(boundary_output.get("is_list", False)),
                    "preview_target_node_ids": preview_target_node_ids,
                    "value": _collapse_cache_slot(
                        slot_values=cache_outputs[output_index],
                        is_list=bool(boundary_output.get("is_list", False)),
                    ),
                }
            )
            self._published_boundary_outputs.add(publication_key)

    def send_sync(
        self, event: str, data: dict[str, Any], client_id: str | None
    ) -> None:
        """Track per-node timing transitions from PromptExecutor progress events."""
        if event == "modal_llm_progress":
            if self._status_callback is None:
                return
            node_id = str(data.get("node_id") or self._active_node_id or "")
            if not node_id:
                return
            progress_event = {
                "event_type": "node_progress",
                "node_id": node_id,
                "display_node_id": node_id,
                "real_node_id": node_id,
                "value": float(data.get("value", 0.0)),
                "max": float(data.get("max", 1.0)),
                "stage": str(data.get("stage") or ""),
                "message": str(data.get("message") or ""),
                "indeterminate": bool(data.get("indeterminate", False)),
            }
            for field_name in (
                "unit",
                "elapsed_seconds",
                "time_to_first_token_seconds",
                "tokens_per_second",
            ):
                if data.get(field_name) is not None:
                    progress_event[field_name] = data[field_name]
            self._status_callback(progress_event)
            return

        if event == "executing":
            next_node_id = data.get("node")
            if next_node_id != self._active_node_id:
                self._emit_boundary_outputs_for_node(self._active_node_id)
                self._log_node_finish(reason="next_node")
            if next_node_id is not None and next_node_id != self._active_node_id:
                node_info = self.prompt.get(str(next_node_id), {})
                class_type = str(node_info.get("class_type", "<unknown>"))
                role = self._classify_node_role(class_type)
                if self._status_callback is not None:
                    self._status_callback(
                        {
                            "phase": "executing",
                            "active_node_id": str(next_node_id),
                            "active_node_class_type": class_type,
                            "active_node_role": role,
                        }
                    )
                self._active_node_id = str(next_node_id)
                self._active_node_started_at = time.perf_counter()
                self.last_node_id = self._active_node_id
                _emit_cloud_info(
                    "Remote node %s class_type=%s role=%s started",
                    self._active_node_id,
                    class_type,
                    role,
                )
            return

        if event == "progress_state":
            if self._status_callback is None:
                return

            nodes_payload = data.get("nodes")
            if not isinstance(nodes_payload, dict):
                return

            tracked_node_id = self._active_node_id
            tracked_node_state: dict[str, Any] | None = None
            if tracked_node_id is not None:
                candidate_state = nodes_payload.get(tracked_node_id)
                if isinstance(candidate_state, dict):
                    tracked_node_state = candidate_state

            if tracked_node_state is None:
                for node_state in nodes_payload.values():
                    if (
                        isinstance(node_state, dict)
                        and node_state.get("state") == "running"
                    ):
                        tracked_node_state = node_state
                        break

            if tracked_node_state is None:
                return

            display_node_id = tracked_node_state.get("display_node_id")
            real_node_id = tracked_node_state.get("real_node_id")
            reported_node_id = (
                display_node_id or real_node_id or tracked_node_state.get("node_id")
            )
            if reported_node_id is None:
                return
            progress_values = _meaningful_progress_values(tracked_node_state)
            if progress_values is None:
                logger.debug(
                    "Ignoring non-meaningful remote progress_state for node_id=%s state=%s value=%r max=%r.",
                    reported_node_id,
                    tracked_node_state.get("state"),
                    tracked_node_state.get("value"),
                    tracked_node_state.get("max"),
                )
                return
            progress_value, max_value = progress_values

            self._status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": str(reported_node_id),
                    "display_node_id": (
                        str(display_node_id) if display_node_id is not None else None
                    ),
                    "real_node_id": str(real_node_id)
                    if real_node_id is not None
                    else None,
                    "value": progress_value,
                    "max": max_value,
                }
            )
            return

        if event == "executed":
            executed_node_id = data.get("node")
            if (
                executed_node_id is not None
                and str(executed_node_id) == self._active_node_id
            ):
                self._log_node_finish(reason="executed")
            if self._status_callback is not None and data.get("output") is not None:
                self._status_callback(
                    {
                        "event_type": "executed",
                        "node_id": str(data.get("node")),
                        "display_node_id": (
                            str(data["display_node"])
                            if data.get("display_node") is not None
                            else None
                        ),
                        "output": data.get("output"),
                    }
                )
            return

        if event in {"execution_error", "execution_interrupted", "execution_success"}:
            self._emit_boundary_outputs_for_node(self._active_node_id)
            self._log_node_finish(reason=event)
            if self._status_callback is not None:
                self._status_callback({"phase": event})
            return

        super().send_sync(event, data, client_id)

