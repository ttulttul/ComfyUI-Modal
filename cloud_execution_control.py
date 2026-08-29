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


@contextmanager
def _registered_remote_execution(
    payload: dict[str, Any],
) -> Iterator[_RemoteExecutionControl]:
    """Prepare interruption state for one active remote execution."""
    prompt_id, component_id = _remote_execution_key(payload)
    control = _RemoteExecutionControl(
        cancellation_event=threading.Event(),
        interrupt_flag_key=_remote_interrupt_flag_key(prompt_id, component_id),
    )
    try:
        yield control
    finally:
        interrupt_store = interrupt_flag_store()
        if modal is not None and interrupt_store is not None:
            interrupt_store.pop(control.interrupt_flag_key, None)
