"""Cap concurrent Modal calls per deployed app at the configured instance count."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_WAIT_POLL_SECONDS = 0.1


class ModalContainerSlotWaitCancelled(RuntimeError):
    """Raised when a call is cancelled while waiting for a free container slot."""


@dataclass
class ModalContainerSlotLimiter:
    """Bound in-flight Modal calls per deployment app across every container pool.

    Modal enforces ``max_containers`` separately for each ``RemoteEngine``
    parameter set, so calls spread across several pools could otherwise run on
    more containers than the user configured. Holding one slot per in-flight
    call keeps the number of busy containers within the configured limit.
    """

    _condition: threading.Condition = field(default_factory=threading.Condition)
    _in_flight: dict[str, int] = field(default_factory=dict)

    def in_flight(self, app_name: str) -> int:
        """Return the number of slots currently held for one deployment app."""
        with self._condition:
            return self._in_flight.get(app_name, 0)

    def _acquire(
        self,
        app_name: str,
        capacity: int,
        cancellation_event: threading.Event | None,
        label: str,
    ) -> None:
        """Block until a slot is free, or raise when the wait is cancelled."""
        with self._condition:
            logged_wait = False
            while self._in_flight.get(app_name, 0) >= capacity:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise ModalContainerSlotWaitCancelled(
                        f"Modal call {label} was cancelled while waiting for one of "
                        f"{capacity} container slots on app={app_name}."
                    )
                if not logged_wait:
                    logger.info(
                        "Waiting for a Modal container slot app=%s call=%s in_flight=%d capacity=%d.",
                        app_name,
                        label,
                        self._in_flight.get(app_name, 0),
                        capacity,
                    )
                    logged_wait = True
                self._condition.wait(timeout=_WAIT_POLL_SECONDS)
            self._in_flight[app_name] = self._in_flight.get(app_name, 0) + 1
            logger.debug(
                "Acquired Modal container slot app=%s call=%s in_flight=%d capacity=%d.",
                app_name,
                label,
                self._in_flight[app_name],
                capacity,
            )

    def _release(self, app_name: str) -> None:
        """Return one slot and wake waiting calls."""
        with self._condition:
            remaining = self._in_flight.get(app_name, 0) - 1
            if remaining < 0:
                raise RuntimeError(
                    f"Released more Modal container slots than acquired for app={app_name}."
                )
            if remaining:
                self._in_flight[app_name] = remaining
            else:
                self._in_flight.pop(app_name, None)
            self._condition.notify_all()

    @contextmanager
    def slot(
        self,
        app_name: str,
        capacity: int | None,
        *,
        cancellation_event: threading.Event | None = None,
        label: str = "",
    ) -> Iterator[None]:
        """Hold one container slot for the duration of a call; no limit when capacity is None."""
        if capacity is None:
            yield
            return
        if capacity <= 0:
            raise ValueError("Modal container slot capacity must be positive.")
        self._acquire(app_name, capacity, cancellation_event, label)
        try:
            yield
        finally:
            self._release(app_name)


MODAL_CONTAINER_SLOTS = ModalContainerSlotLimiter()
"""Process-wide limiter shared by every Modal call from this ComfyUI host."""
