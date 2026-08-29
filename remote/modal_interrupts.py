"""Modal prompt cancellation state and interrupt propagation."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import logging
import queue
import threading
import time
from typing import Any, Callable, Iterator

from ..settings import get_settings
from .modal_deployment import ModalRemoteInvocationError, _modal_environment_name

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback tests.
    modal = None

_MODAL_INTERRUPT_DICTS_LOCK = threading.Lock()
_MODAL_INTERRUPT_DICTS: dict[tuple[str, str | None], Any] = {}
_ACTIVE_REMOTE_INVOCATIONS_LOCK = threading.Lock()
_ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT: dict[
    str, dict[str, "_ActiveRemoteInvocation"]
] = {}


@dataclass
class _ActiveRemoteInvocation:
    """Track one local proxy call that is currently waiting on remote Modal work."""

    prompt_id: str
    component_id: str
    cancellation_event: threading.Event | None
    interrupt_remote_call: Callable[[], Any] | None


def _local_processing_interrupted() -> bool:
    """Return whether the current local ComfyUI execution was interrupted."""
    try:
        import comfy.model_management
    except ModuleNotFoundError:
        return False

    return bool(comfy.model_management.processing_interrupted())


def _raise_local_interrupt() -> None:
    """Raise ComfyUI's native interruption exception for the current execution."""
    import comfy.model_management

    raise comfy.model_management.InterruptProcessingException()


def _exception_indicates_interruption(exc: BaseException) -> bool:
    """Return whether an exception represents cancellation or interrupted execution."""
    if isinstance(exc, asyncio.CancelledError):
        return True
    message = str(exc).lower()
    return "interrupt" in message or "cancel" in message


def _remote_interrupt_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the prompt/component pair used to interrupt one remote execution."""
    prompt_id = str(
        payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph"
    )
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


def _lookup_modal_interrupt_store() -> Any | None:
    """Return the shared Modal Dict used to signal remote cancellation requests."""
    if modal is None or not hasattr(modal, "Dict"):
        return None

    settings = get_settings()
    cache_key = (settings.interrupt_dict_name, _modal_environment_name())
    with _MODAL_INTERRUPT_DICTS_LOCK:
        cached_store = _MODAL_INTERRUPT_DICTS.get(cache_key)
        if cached_store is not None:
            return cached_store

    interrupt_store = modal.Dict.from_name(
        settings.interrupt_dict_name,
        environment_name=cache_key[1],
        create_if_missing=True,
    )
    with _MODAL_INTERRUPT_DICTS_LOCK:
        _MODAL_INTERRUPT_DICTS[cache_key] = interrupt_store
    return interrupt_store


def _remote_interrupt_flag_value() -> dict[str, float]:
    """Return the shared Modal interrupt-store value for one cancellation request."""
    return {"requested_at": time.time()}


def _write_remote_interrupt_flag(
    interrupt_store: Any, prompt_id: str, component_id: str
) -> None:
    """Write one remote cancellation request with the blocking Modal Dict API."""
    interrupt_store.put(
        _remote_interrupt_flag_key(prompt_id, component_id),
        _remote_interrupt_flag_value(),
    )


async def _write_remote_interrupt_flag_async(
    interrupt_store: Any,
    prompt_id: str,
    component_id: str,
) -> None:
    """Write one remote cancellation request without blocking the async caller."""
    put_method = getattr(interrupt_store, "put", None)
    put_async = getattr(put_method, "aio", None)
    if callable(put_async):
        result = put_async(
            _remote_interrupt_flag_key(prompt_id, component_id),
            _remote_interrupt_flag_value(),
        )
        if inspect.isawaitable(result):
            await result
        return

    await asyncio.to_thread(
        _write_remote_interrupt_flag, interrupt_store, prompt_id, component_id
    )


def _request_remote_interrupt(payload: dict[str, Any]) -> bool:
    """Write one remote cancellation request into the shared Modal interrupt store."""
    _abandon_local_modal_workflow_gate(payload, "local interrupt requested")
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return False

    prompt_id, component_id = _remote_interrupt_key(payload)
    _write_remote_interrupt_flag(interrupt_store, prompt_id, component_id)
    logger.info(
        "Propagated local interrupt to Modal prompt=%s component=%s through shared control state.",
        prompt_id,
        component_id,
    )
    return True


async def _request_remote_interrupt_async(payload: dict[str, Any]) -> bool:
    """Write one remote cancellation request into the shared Modal interrupt store asynchronously."""
    _abandon_local_modal_workflow_gate(payload, "local interrupt requested")
    interrupt_store = await asyncio.to_thread(_lookup_modal_interrupt_store)
    if interrupt_store is None:
        return False

    prompt_id, component_id = _remote_interrupt_key(payload)
    await _write_remote_interrupt_flag_async(interrupt_store, prompt_id, component_id)
    logger.info(
        "Propagated local interrupt to Modal prompt=%s component=%s through shared control state.",
        prompt_id,
        component_id,
    )
    return True


def active_remote_modal_prompt_ids() -> set[str]:
    """Return prompt ids that currently have local proxies waiting on Modal work."""
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        return set(_ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT)


@contextmanager
def _registered_active_remote_invocation(
    payload: dict[str, Any],
    cancellation_event: threading.Event | None,
    interrupt_remote_call: Callable[[], Any] | None,
) -> Iterator[None]:
    """Register one active Modal call so targeted ComfyUI interrupts can find it."""
    prompt_id, component_id = _remote_interrupt_key(payload)
    invocation = _ActiveRemoteInvocation(
        prompt_id=prompt_id,
        component_id=component_id,
        cancellation_event=cancellation_event,
        interrupt_remote_call=interrupt_remote_call,
    )
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        prompt_invocations = _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.setdefault(
            prompt_id, {}
        )
        prompt_invocations[component_id] = invocation
    logger.info(
        "Registered active Modal invocation prompt=%s component=%s for targeted cancellation.",
        prompt_id,
        component_id,
    )
    try:
        yield
    finally:
        with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
            prompt_invocations = _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.get(prompt_id)
            if prompt_invocations is not None:
                prompt_invocations.pop(component_id, None)
                if not prompt_invocations:
                    _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.pop(prompt_id, None)
        logger.info(
            "Unregistered active Modal invocation prompt=%s component=%s.",
            prompt_id,
            component_id,
        )


def request_remote_modal_prompt_interrupt(prompt_id: str) -> bool:
    """Request cancellation for every active Modal invocation belonging to one prompt."""
    normalized_prompt_id = str(prompt_id)
    _abandon_local_modal_workflow_gate(
        {"prompt_id": normalized_prompt_id},
        "prompt-level interrupt requested",
    )
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        invocations = list(
            _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.get(normalized_prompt_id, {}).values()
        )
    if not invocations:
        return False

    logger.info(
        "Requesting remote Modal cancellation for prompt=%s across %d active component(s).",
        normalized_prompt_id,
        len(invocations),
    )
    for invocation in invocations:
        if invocation.cancellation_event is not None:
            invocation.cancellation_event.set()
        _propagate_remote_interrupt_request(
            {
                "prompt_id": invocation.prompt_id,
                "component_id": invocation.component_id,
            },
            invocation.interrupt_remote_call,
        )
    return True


async def request_remote_modal_prompt_interrupt_async(prompt_id: str) -> bool:
    """Request cancellation for every active Modal invocation belonging to one prompt asynchronously."""
    normalized_prompt_id = str(prompt_id)
    _abandon_local_modal_workflow_gate(
        {"prompt_id": normalized_prompt_id},
        "prompt-level interrupt requested",
    )
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        invocations = list(
            _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.get(normalized_prompt_id, {}).values()
        )
    if not invocations:
        return False

    logger.info(
        "Requesting async remote Modal cancellation for prompt=%s across %d active component(s).",
        normalized_prompt_id,
        len(invocations),
    )
    for invocation in invocations:
        if invocation.cancellation_event is not None:
            invocation.cancellation_event.set()
        await _request_remote_interrupt_async(
            {"prompt_id": invocation.prompt_id, "component_id": invocation.component_id}
        )
    return True


def _sync_local_interrupt_to_cancellation_event(
    payload: dict[str, Any],
    cancellation_event: threading.Event | None,
) -> bool:
    """Mirror ComfyUI's interrupt flag into the current Modal cancellation event."""
    if cancellation_event is not None and cancellation_event.is_set():
        return True
    if not _local_processing_interrupted():
        return False
    if cancellation_event is not None and not cancellation_event.is_set():
        logger.info(
            "Observed local interrupt while Modal component=%s was running; requesting remote cancellation.",
            payload.get("component_id"),
        )
        cancellation_event.set()
    _abandon_local_modal_workflow_gate(payload, "observed local interrupt")
    return True


def _abandon_local_modal_workflow_gate(payload: dict[str, Any], reason: str) -> None:
    """Release the local prompt gate for a Modal prompt that ComfyUI has cancelled."""
    prompt_id = payload.get("prompt_id")
    if prompt_id is None:
        return

    try:
        from ..modal_executor_node import abandon_modal_workflow_execution_prompt
    except ImportError:
        logger.debug(
            "Unable to import Modal workflow gate helper while abandoning prompt."
        )
        return

    abandon_modal_workflow_execution_prompt(str(prompt_id), reason)


def _propagate_remote_interrupt_request(
    payload: dict[str, Any],
    interrupt_remote_call: Callable[[], Any] | None,
) -> None:
    """Send one best-effort remote cancellation request for an active Modal payload."""
    prompt_id, component_id = _remote_interrupt_key(payload)
    if interrupt_remote_call is None:
        logger.warning(
            "Local interrupt requested for component=%s, but no remote interrupt method is available.",
            component_id,
        )
        return
    try:
        interrupt_remote_call()
        logger.info(
            "Propagated local interrupt to Modal prompt=%s component=%s.",
            prompt_id,
            component_id,
        )
    except Exception:
        logger.exception(
            "Failed to propagate local interrupt to Modal prompt=%s component=%s.",
            prompt_id,
            component_id,
        )


def _handle_modal_wait_cancellation(
    payload: dict[str, Any],
    cancellation_event: threading.Event,
    *,
    interrupt_sent: bool,
    cancellation_started_at: float | None,
) -> tuple[bool, float | None]:
    """Propagate and bound local waiting after cancellation during any Modal wait phase."""
    if not _sync_local_interrupt_to_cancellation_event(payload, cancellation_event):
        return interrupt_sent, cancellation_started_at

    if not interrupt_sent:
        _request_remote_interrupt(payload)
        return True, time.monotonic()

    if cancellation_started_at is None:
        return interrupt_sent, time.monotonic()

    grace_seconds = max(0.0, get_settings().remote_cancel_grace_seconds)
    if time.monotonic() - cancellation_started_at >= grace_seconds:
        logger.info(
            "Modal component=%s did not reach a cancellable remote call within %.3fs of local interrupt; releasing the local prompt while remote cancellation continues.",
            payload.get("component_id"),
            grace_seconds,
        )
        raise ModalRemoteInvocationError(
            "Remote Modal call did not reach a cancellable remote phase after local interrupt propagation."
        )

    return interrupt_sent, cancellation_started_at


async def _handle_modal_wait_cancellation_async(
    payload: dict[str, Any],
    cancellation_event: threading.Event,
    *,
    interrupt_sent: bool,
    cancellation_started_at: float | None,
) -> tuple[bool, float | None]:
    """Propagate and bound local waiting after cancellation during an async Modal wait phase."""
    if not _sync_local_interrupt_to_cancellation_event(payload, cancellation_event):
        return interrupt_sent, cancellation_started_at

    if not interrupt_sent:
        await _request_remote_interrupt_async(payload)
        return True, time.monotonic()

    if cancellation_started_at is None:
        return interrupt_sent, time.monotonic()

    grace_seconds = max(0.0, get_settings().remote_cancel_grace_seconds)
    if time.monotonic() - cancellation_started_at >= grace_seconds:
        logger.info(
            "Modal component=%s did not reach a cancellable remote call within %.3fs of local interrupt; releasing the local prompt while remote cancellation continues.",
            payload.get("component_id"),
            grace_seconds,
        )
        raise ModalRemoteInvocationError(
            "Remote Modal call did not reach a cancellable remote phase after local interrupt propagation."
        )

    return interrupt_sent, cancellation_started_at


def _invoke_remote_call_with_interrupts(
    *,
    payload: dict[str, Any],
    invoke_remote_call: Callable[[], bytes],
    interrupt_remote_call: Callable[[], Any] | None,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Run one blocking remote call while optionally propagating cancellation to Modal."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    cancellation_started_at: float | None = None

    def execute_remote_call() -> None:
        """Run the blocking Modal request in a worker thread."""
        try:
            result_queue.put(("result", invoke_remote_call()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    request_thread = threading.Thread(
        target=execute_remote_call,
        name=f"modal-request-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    request_thread.start()
    interrupt_sent = False
    try:
        with _registered_active_remote_invocation(
            payload, cancellation_event, interrupt_remote_call
        ):
            while True:
                try:
                    result_kind, result_payload = result_queue.get(timeout=0.1)
                except queue.Empty:
                    if _sync_local_interrupt_to_cancellation_event(
                        payload, cancellation_event
                    ):
                        if not interrupt_sent:
                            _propagate_remote_interrupt_request(
                                payload, interrupt_remote_call
                            )
                            interrupt_sent = True
                            cancellation_started_at = time.monotonic()
                        elif cancellation_started_at is not None:
                            grace_seconds = max(
                                0.0, get_settings().remote_cancel_grace_seconds
                            )
                            if (
                                time.monotonic() - cancellation_started_at
                                >= grace_seconds
                            ):
                                logger.info(
                                    "Modal component=%s did not return within %.3fs of local interrupt propagation; releasing the local prompt while remote cancellation continues.",
                                    payload.get("component_id"),
                                    grace_seconds,
                                )
                                raise ModalRemoteInvocationError(
                                    "Remote Modal call did not finish after local interrupt propagation."
                                )
                    continue

                if result_kind == "result":
                    return bytes(result_payload)
                raise result_payload
    finally:
        request_thread.join(
            timeout=0.1
            if cancellation_event is not None and cancellation_event.is_set()
            else 1.0
        )


