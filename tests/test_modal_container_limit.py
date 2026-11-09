"""Tests for the per-app Modal container slot limit."""

from __future__ import annotations

import importlib
import threading
import time
import types
from typing import Any

import pytest

from conftest import PACKAGE_NAME


@pytest.fixture()
def limit_module(extension_package: object) -> Any:
    """Return the Modal container limit module."""
    del extension_package
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_container_limit")


def _run_concurrently(
    limiter: Any, app_name: str, capacity: int | None, call_count: int
) -> int:
    """Run overlapping slot holders and return the peak number held at once."""
    peak = 0
    active = 0
    lock = threading.Lock()

    def hold_slot() -> None:
        """Hold one slot briefly while recording concurrency."""
        nonlocal peak, active
        with limiter.slot(app_name, capacity, label="test"):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=hold_slot) for _ in range(call_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return peak


def test_slot_limit_caps_concurrent_calls_per_app(limit_module: Any) -> None:
    """No more calls than the capacity may hold slots for one app at once."""
    limiter = limit_module.ModalContainerSlotLimiter()

    peak = _run_concurrently(limiter, "app-a", 4, 8)

    assert peak == 4
    assert limiter.in_flight("app-a") == 0


def test_slot_limit_without_capacity_is_unbounded(limit_module: Any) -> None:
    """A missing instance count must not throttle calls."""
    limiter = limit_module.ModalContainerSlotLimiter()

    assert _run_concurrently(limiter, "app-a", None, 6) == 6


def test_slot_limit_tracks_apps_independently(limit_module: Any) -> None:
    """A full app must not block calls to another deployment app."""
    limiter = limit_module.ModalContainerSlotLimiter()

    with limiter.slot("app-a", 1):
        with limiter.slot("app-b", 1):
            assert limiter.in_flight("app-a") == 1
            assert limiter.in_flight("app-b") == 1


def test_slot_wait_raises_when_cancelled(limit_module: Any) -> None:
    """A call waiting for a slot should stop promptly when cancelled."""
    limiter = limit_module.ModalContainerSlotLimiter()
    cancellation_event = threading.Event()
    cancellation_event.set()

    with limiter.slot("app-a", 1):
        with pytest.raises(limit_module.ModalContainerSlotWaitCancelled):
            with limiter.slot("app-a", 1, cancellation_event=cancellation_event):
                pass

    assert limiter.in_flight("app-a") == 0


def test_slot_is_released_when_the_call_fails(limit_module: Any) -> None:
    """A failing call must return its slot."""
    limiter = limit_module.ModalContainerSlotLimiter()

    with pytest.raises(KeyError):
        with limiter.slot("app-a", 1):
            raise KeyError("boom")

    assert limiter.in_flight("app-a") == 0


def test_remote_engine_calls_share_one_slot_budget(
    remote_modal_app_module: Any,
    limit_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal calls for one app should never exceed the configured max containers."""
    limiter = limit_module.ModalContainerSlotLimiter()
    settings = types.SimpleNamespace(max_containers=2)
    monkeypatch.setattr(remote_modal_app_module, "MODAL_CONTAINER_SLOTS", limiter)
    monkeypatch.setattr(remote_modal_app_module, "_settings_for_payload", lambda payload: settings)
    monkeypatch.setattr(remote_modal_app_module, "modal_deployment_app_name", lambda s: "app-a")
    peak = 0
    active = 0
    lock = threading.Lock()

    def fake_call(*args: Any) -> bytes:
        """Record how many calls overlap inside the slot."""
        nonlocal peak, active
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return b"ok"

    monkeypatch.setattr(remote_modal_app_module, "_invoke_remote_engine_payload_in_slot", fake_call)
    threads = [
        threading.Thread(
            target=remote_modal_app_module._invoke_remote_engine_payload,
            args=(object(), {"component_id": f"c{index}"}, b"{}", None),
        )
        for index in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak == 2
