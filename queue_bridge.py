"""ComfyUI queue insertion, interrupt bridging, and remote preparation state."""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Mapping

from aiohttp import web

if __package__:
    from .execution_environments import ExecutionProvider
    from .prompt_diagnostics import _log_modal_rewritten_prompt_diagnostics
    from .sync_engine import finish_r2_writeback_prompt
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import ExecutionProvider
    from prompt_diagnostics import _log_modal_rewritten_prompt_diagnostics
    from sync_engine import finish_r2_writeback_prompt

logger = logging.getLogger(__name__)

_MODAL_INTERRUPT_QUEUE_BRIDGE_ATTR = "__comfy_modal_interrupt_queue_bridge_installed"
_REMOTE_PREPARATION_PROMPTS_ATTR = "__comfy_modal_remote_preparation_prompts"
_REMOTE_PREPARATION_CANCELLATIONS_ATTR = (
    "__comfy_modal_remote_preparation_cancellations"
)
_REMOTE_PREPARATION_LOCK_ATTR = "__comfy_modal_remote_preparation_lock"


def _get_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    import execution

    return execution


async def _queue_prompt_json(
    prompt_server: Any,
    json_data: dict[str, Any],
    modal_response_payload: dict[str, Any] | None = None,
) -> web.Response:
    """Queue a possibly rewritten prompt using ComfyUI's native semantics."""
    execution = _get_execution_module()
    json_data = prompt_server.trigger_on_prompt(json_data)

    if "number" in json_data:
        number = float(json_data["number"])
    else:
        number = prompt_server.number
        if json_data.get("front"):
            number = -number
        prompt_server.number += 1

    if "prompt" not in json_data:
        return web.json_response(
            {
                "error": {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {},
                }
            },
            status=400,
        )

    prompt = json_data["prompt"]
    prompt_id = str(json_data.get("prompt_id", uuid.uuid4()))
    partial_execution_targets = json_data.get("partial_execution_targets")
    extra_data = dict(json_data.get("extra_data", {}))
    if "client_id" in json_data:
        extra_data["client_id"] = json_data["client_id"]
    valid = await execution.validate_prompt(
        prompt_id, prompt, partial_execution_targets
    )

    if not valid[0]:
        modal_extra = extra_data.get("modal")
        if isinstance(modal_extra, Mapping) and modal_extra.get("remote_component_ids"):
            logger.warning(
                "ComfyUI rejected rewritten Modal prompt prompt_id=%s error=%s node_errors=%s",
                prompt_id,
                valid[1],
                valid[3],
            )
            _log_modal_rewritten_prompt_diagnostics(
                prompt_id=prompt_id,
                prompt=prompt,
                reason="comfy_validation_failure",
                level=logging.WARNING,
            )
        else:
            logger.warning("invalid prompt: %s", valid[1])
        return web.json_response(
            {"error": valid[1], "node_errors": valid[3]}, status=400
        )

    outputs_to_execute = valid[2]
    sensitive: dict[str, Any] = {}
    for sensitive_key in execution.SENSITIVE_EXTRA_DATA_KEYS:
        if sensitive_key in extra_data:
            sensitive[sensitive_key] = extra_data.pop(sensitive_key)

    extra_data["create_time"] = int(time.time() * 1000)
    prompt_server.prompt_queue.put(
        (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
    )
    response_payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "number": number,
        "node_errors": valid[3],
    }
    if modal_response_payload:
        response_payload.update(modal_response_payload)
    return web.json_response(response_payload)


def _install_modal_interrupt_queue_bridge(prompt_server: Any) -> None:
    """Expose active remote work through every ComfyUI queue-state view."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    if prompt_queue is None or getattr(
        prompt_queue, _MODAL_INTERRUPT_QUEUE_BRIDGE_ATTR, False
    ):
        return

    original_get_current_queue = getattr(prompt_queue, "get_current_queue", None)
    original_get_current_queue_volatile = getattr(
        prompt_queue, "get_current_queue_volatile", None
    )
    original_get_tasks_remaining = getattr(prompt_queue, "get_tasks_remaining", None)
    original_interrupt_if_running = getattr(
        prompt_queue, "interrupt_if_running", None
    )
    original_task_done = getattr(prompt_queue, "task_done", None)
    original_wipe_queue = getattr(prompt_queue, "wipe_queue", None)
    original_delete_queue_item = getattr(prompt_queue, "delete_queue_item", None)
    if not any(
        callable(method)
        for method in (
            original_get_current_queue,
            original_get_current_queue_volatile,
            original_get_tasks_remaining,
        )
    ):
        logger.debug(
            "Prompt queue does not expose queue-state methods; skipping remote queue bridge."
        )
        return

    preparation_prompts: dict[str, tuple[Any, ...]] = {}
    preparation_cancellations: dict[str, threading.Event] = {}
    preparation_lock = threading.RLock()
    setattr(prompt_queue, _REMOTE_PREPARATION_PROMPTS_ATTR, preparation_prompts)
    setattr(
        prompt_queue,
        _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
        preparation_cancellations,
    )
    setattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, preparation_lock)

    def preparation_items() -> list[tuple[Any, ...]]:
        """Return a stable snapshot of prompts still preparing remote capacity."""
        with preparation_lock:
            return list(preparation_prompts.values())

    def append_missing_preparations(
        running: Iterable[Any],
        queued: Iterable[Any],
    ) -> tuple[list[Any], Any]:
        """Add preparation entries that are not already in ComfyUI's native queue."""
        running_items = list(running)
        queued_items = list(queued)
        native_prompt_ids = _queue_item_prompt_ids((*running_items, *queued_items))
        running_items.extend(
            item
            for item in preparation_items()
            if str(item[1]) not in native_prompt_ids
        )
        return running_items, queued

    if callable(original_get_current_queue):

        def remote_get_current_queue() -> tuple[list[Any], Any]:
            """Return native work plus preparation and active remote prompt entries."""
            running, queued = original_get_current_queue()
            running_items, queued = append_missing_preparations(running, queued)
            running_prompt_ids = _queue_item_prompt_ids(running_items)
            try:
                if __package__:
                    from .remote.modal_app import active_remote_modal_prompt_ids
                else:  # pragma: no cover - flat import inside the Modal container.
                    from remote.modal_app import active_remote_modal_prompt_ids
            except ImportError:
                return running_items, queued

            for prompt_id in sorted(
                active_remote_modal_prompt_ids() - running_prompt_ids
            ):
                running_items.append((0, prompt_id, {}, {}, [], {}))
            return running_items, queued

        setattr(prompt_queue, "get_current_queue", remote_get_current_queue)

    if callable(original_get_current_queue_volatile):

        def remote_get_current_queue_volatile() -> tuple[list[Any], Any]:
            """Include remote preparation in ComfyUI's public `/queue` response."""
            running, queued = original_get_current_queue_volatile()
            return append_missing_preparations(running, queued)

        setattr(
            prompt_queue,
            "get_current_queue_volatile",
            remote_get_current_queue_volatile,
        )

    if callable(original_get_tasks_remaining):

        def remote_get_tasks_remaining() -> int:
            """Count remote preparation as work in websocket queue status."""
            remaining = int(original_get_tasks_remaining())
            native_prompt_ids: set[str] = set()
            native_queue_method = (
                original_get_current_queue_volatile
                if callable(original_get_current_queue_volatile)
                else original_get_current_queue
            )
            if callable(native_queue_method):
                running, queued = native_queue_method()
                native_prompt_ids = _queue_item_prompt_ids((*running, *queued))
            return remaining + sum(
                str(item[1]) not in native_prompt_ids
                for item in preparation_items()
            )

        setattr(prompt_queue, "get_tasks_remaining", remote_get_tasks_remaining)

    if callable(original_interrupt_if_running):

        def remote_interrupt_if_running(prompt_id: str) -> bool:
            """Cancel remote preparation or interrupt matching native execution."""
            if _cancel_remote_preparation(prompt_server, prompt_id):
                return True
            return bool(original_interrupt_if_running(prompt_id))

        setattr(prompt_queue, "interrupt_if_running", remote_interrupt_if_running)

    if callable(original_task_done):

        def remote_task_done(item_id: Any, *args: Any, **kwargs: Any) -> Any:
            """Release background cache work after the whole prompt terminates."""
            currently_running = getattr(prompt_queue, "currently_running", {})
            running_item = (
                currently_running.get(item_id)
                if isinstance(currently_running, Mapping)
                else None
            )
            prompt_id = (
                str(running_item[1])
                if isinstance(running_item, (list, tuple)) and len(running_item) > 1
                else None
            )
            try:
                return original_task_done(item_id, *args, **kwargs)
            finally:
                if prompt_id is not None:
                    finish_r2_writeback_prompt(prompt_id)

        setattr(prompt_queue, "task_done", remote_task_done)

    if callable(original_wipe_queue):

        def remote_wipe_queue() -> Any:
            """Release reservations belonging to every discarded queued prompt."""
            queued_items: list[Any] = []
            if callable(original_get_current_queue):
                _running, queued = original_get_current_queue()
                queued_items = list(queued)
            try:
                return original_wipe_queue()
            finally:
                for queued_prompt_id in _queue_item_prompt_ids(queued_items):
                    finish_r2_writeback_prompt(queued_prompt_id)

        setattr(prompt_queue, "wipe_queue", remote_wipe_queue)

    if callable(original_delete_queue_item):

        def remote_delete_queue_item(predicate: Callable[[Any], bool]) -> Any:
            """Release the exact reservation removed through ComfyUI's queue API."""
            before_prompt_ids: set[str] = set()
            if callable(original_get_current_queue):
                _running, queued = original_get_current_queue()
                before_prompt_ids = _queue_item_prompt_ids(queued)
            result = original_delete_queue_item(predicate)
            if result and callable(original_get_current_queue):
                _running, queued = original_get_current_queue()
                after_prompt_ids = _queue_item_prompt_ids(queued)
                for removed_prompt_id in before_prompt_ids - after_prompt_ids:
                    finish_r2_writeback_prompt(removed_prompt_id)
            return result

        setattr(prompt_queue, "delete_queue_item", remote_delete_queue_item)

    setattr(prompt_queue, _MODAL_INTERRUPT_QUEUE_BRIDGE_ATTR, True)
    logger.info("Installed remote preparation bridge on ComfyUI prompt queue.")


def _queue_item_prompt_ids(items: Iterable[Any]) -> set[str]:
    """Return prompt IDs from well-formed native or synthetic queue items."""
    return {
        str(item[1])
        for item in items
        if isinstance(item, (list, tuple)) and len(item) > 1
    }


def _cancel_remote_preparation(prompt_server: Any, prompt_id: str) -> bool:
    """Signal cancellation when the prompt is preparing remote capacity."""
    normalized_prompt_id = str(prompt_id).strip()
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    cancellations = getattr(
        prompt_queue,
        _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
        None,
    )
    preparation_lock = getattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, None)
    if (
        not normalized_prompt_id
        or not isinstance(cancellations, dict)
        or preparation_lock is None
    ):
        return False
    with preparation_lock:
        cancellation_event = cancellations.get(normalized_prompt_id)
        if cancellation_event is None:
            return False
        cancellation_event.set()
    logger.info(
        "Cancelled remote preparation for prompt %s.", normalized_prompt_id
    )
    return True


def _queued_ssh_environment_ids(
    prompt_server: Any,
    *,
    excluding_prompt_id: str | None = None,
) -> frozenset[str]:
    """Return SSH environments reserved by prompts already in ComfyUI's queue."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    get_current_queue = getattr(prompt_queue, "get_current_queue", None)
    if not callable(get_current_queue):
        return frozenset()
    queue_state = get_current_queue()
    if not isinstance(queue_state, (list, tuple)) or len(queue_state) != 2:
        return frozenset()
    running, queued = queue_state
    queue_items = [
        item
        for collection in (running, queued)
        if isinstance(collection, (list, tuple))
        for item in collection
    ]
    environment_ids: set[str] = set()
    for item in queue_items:
        if not isinstance(item, (list, tuple)) or len(item) <= 3:
            continue
        prompt_id = str(item[1]) if len(item) > 1 else ""
        if excluding_prompt_id is not None and prompt_id == excluding_prompt_id:
            continue
        extra_data = item[3]
        if not isinstance(extra_data, Mapping):
            continue
        remote_execution = extra_data.get("remote_execution")
        if not isinstance(remote_execution, Mapping):
            continue
        assignments = remote_execution.get("assignments")
        if not isinstance(assignments, Mapping):
            continue
        for assignment in assignments.values():
            if not isinstance(assignment, Mapping):
                continue
            provider = str(assignment.get("provider") or "").strip().lower()
            environment_id = str(assignment.get("environment_id") or "").strip()
            if provider == ExecutionProvider.SSH_DOCKER.value and environment_id:
                environment_ids.add(environment_id)
    return frozenset(environment_ids)


def _set_remote_preparation(
    prompt_server: Any,
    *,
    prompt_id: str,
    prompt: Mapping[str, Any],
    extra_data: Mapping[str, Any],
    cancellation_event: threading.Event | None = None,
) -> bool:
    """Register one pre-queue remote workflow and publish its active queue state."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    preparations = getattr(prompt_queue, _REMOTE_PREPARATION_PROMPTS_ATTR, None)
    preparation_lock = getattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, None)
    if not isinstance(preparations, dict) or preparation_lock is None:
        return False
    preparation_extra_data = copy.deepcopy(dict(extra_data))
    preparation_extra_data.setdefault("create_time", int(time.time() * 1000))
    with preparation_lock:
        preparations[prompt_id] = (
            0,
            prompt_id,
            copy.deepcopy(dict(prompt)),
            preparation_extra_data,
            [],
            {},
        )
        cancellations = getattr(
            prompt_queue,
            _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
            None,
        )
        if isinstance(cancellations, dict) and cancellation_event is not None:
            cancellations[prompt_id] = cancellation_event
    queue_updated = getattr(prompt_server, "queue_updated", None)
    if callable(queue_updated):
        queue_updated()
    return True


def _clear_remote_preparation(prompt_server: Any, prompt_id: str) -> None:
    """Remove one pre-queue remote workflow and publish the resulting queue state."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    preparations = getattr(prompt_queue, _REMOTE_PREPARATION_PROMPTS_ATTR, None)
    preparation_lock = getattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, None)
    if not isinstance(preparations, dict) or preparation_lock is None:
        return
    with preparation_lock:
        removed = preparations.pop(prompt_id, None)
        cancellations = getattr(
            prompt_queue,
            _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
            None,
        )
        if isinstance(cancellations, dict):
            cancellations.pop(prompt_id, None)
    if removed is not None:
        queue_updated = getattr(prompt_server, "queue_updated", None)
        if callable(queue_updated):
            queue_updated()
