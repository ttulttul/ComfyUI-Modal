"""Active cloud execution registration and interrupt cleanup."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
from typing import Any, Iterator

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - remote entrypoint only.
    modal = None

try:
    from .cloud_runtime_context import interrupt_flag_store
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_runtime_context import interrupt_flag_store

logger = logging.getLogger(__name__)


@dataclass
class _RemoteExecutionControl:
    """Track cancellation state for one active remote execution."""

    cancellation_event: threading.Event
    interrupt_flag_key: str


def _remote_execution_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the registry key for one active remote execution."""
    prompt_id = str(
        payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph"
    )
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


class RemoteExecutionCancelledBeforeStart(RuntimeError):
    """Signal that a queued remote request was interrupted before this container ran it.

    The message deliberately contains "interrupt" so the cloud failure classifier
    treats it as an expected interruption and keeps the container alive.
    """


def _cancel_flag_is_set(interrupt_store: Any | None, interrupt_flag_key: str) -> bool:
    """Return whether the shared interrupt store holds a cancel flag for one execution."""
    if interrupt_store is None:
        return False
    return bool(interrupt_store.contains(interrupt_flag_key))


def _raise_if_cancelled_before_start(
    interrupt_store: Any | None,
    prompt_id: str,
    component_id: str,
    interrupt_flag_key: str,
) -> None:
    """Refuse a queued request whose prompt was interrupted while it waited for a container.

    Modal cannot withdraw a queued generator call, so a request the user already
    cancelled can still be delivered to the next free container. Skipping it here,
    before any volume reload or model loading, keeps that container available for
    the requests that are still wanted.
    """
    if not _cancel_flag_is_set(interrupt_store, interrupt_flag_key):
        return
    logger.warning(
        "Skipping Modal request for prompt=%s component=%s because it was interrupted "
        "while queued; keeping this container available for the next request.",
        prompt_id,
        component_id,
    )
    raise RemoteExecutionCancelledBeforeStart(
        f"Remote execution for prompt={prompt_id} component={component_id} was "
        "interrupted before this container started it."
    )


@contextmanager
def _registered_remote_execution(
    payload: dict[str, Any],
) -> Iterator[_RemoteExecutionControl]:
    """Prepare interruption state for one active remote execution.

    Raises RemoteExecutionCancelledBeforeStart when the request was interrupted
    while it waited in Modal's queue.
    """
    prompt_id, component_id = _remote_execution_key(payload)
    control = _RemoteExecutionControl(
        cancellation_event=threading.Event(),
        interrupt_flag_key=_remote_interrupt_flag_key(prompt_id, component_id),
    )
    try:
        _raise_if_cancelled_before_start(
            interrupt_flag_store(), prompt_id, component_id, control.interrupt_flag_key
        )
        yield control
    finally:
        interrupt_store = interrupt_flag_store()
        if modal is not None and interrupt_store is not None:
            interrupt_store.pop(control.interrupt_flag_key, None)
