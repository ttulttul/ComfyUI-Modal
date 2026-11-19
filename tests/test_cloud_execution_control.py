"""Tests for container-side handling of requests interrupted while queued on Modal."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Iterator

import pytest


class _FakeInterruptStore:
    """Minimal stand-in for the Modal Dict that holds remote interrupt flags."""

    def __init__(self, keys: set[str] | None = None) -> None:
        """Create a store pre-populated with the given flag keys."""
        self.flags: dict[str, Any] = {key: {"requested_at": 1.0} for key in keys or set()}

    def contains(self, key: str) -> bool:
        """Return whether a flag exists for the key."""
        return key in self.flags

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return a flag, like modal.Dict.pop."""
        return self.flags.pop(key, default)


@pytest.fixture
def execution_control_module(modal_cloud_module: object) -> Iterator[ModuleType]:
    """Return the flat cloud execution-control module with clean runtime stores."""
    del modal_cloud_module
    runtime_context = importlib.import_module("cloud_runtime_context")
    runtime_context.clear_cloud_runtime_stores()
    yield importlib.import_module("cloud_execution_control")
    runtime_context.clear_cloud_runtime_stores()


def _register_interrupt_store(store: _FakeInterruptStore) -> None:
    """Publish a fake interrupt store the way the cloud entrypoint does."""
    importlib.import_module("cloud_runtime_context").register_cloud_runtime_stores(
        session_bridge_cache=None,
        invocation_records=None,
        volume=None,
        snapshot_profiles=None,
        node_output_cache=None,
        interrupt_flags=store,
    )


_PAYLOAD = {"prompt_id": "prompt-1", "component_id": "269:251"}


def test_interrupted_queued_request_is_refused_before_it_runs(
    execution_control_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request whose prompt was interrupted while queued must not start executing."""
    store = _FakeInterruptStore({"prompt-1:269:251"})
    _register_interrupt_store(store)
    monkeypatch.setattr(execution_control_module, "modal", object())
    body_ran = False

    with pytest.raises(execution_control_module.RemoteExecutionCancelledBeforeStart):
        with execution_control_module._registered_remote_execution(dict(_PAYLOAD)):
            body_ran = True

    assert body_ran is False
    assert store.flags == {}


def test_request_without_cancel_flag_runs_normally(
    execution_control_module: ModuleType,
) -> None:
    """Requests for prompts that were not interrupted should execute as before."""
    _register_interrupt_store(_FakeInterruptStore({"other-prompt:269:251"}))

    with execution_control_module._registered_remote_execution(dict(_PAYLOAD)) as control:
        assert control.interrupt_flag_key == "prompt-1:269:251"
        assert not control.cancellation_event.is_set()


def test_request_runs_when_no_interrupt_store_is_registered(
    execution_control_module: ModuleType,
) -> None:
    """Without a shared interrupt store there is nothing to check, so execution proceeds."""
    with execution_control_module._registered_remote_execution(dict(_PAYLOAD)) as control:
        assert control.interrupt_flag_key == "prompt-1:269:251"


def test_cancelled_before_start_keeps_the_container_alive(
    modal_cloud_module: Any,
    execution_control_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping a cancelled queued request is expected and must not retire the worker."""
    scheduled_exits: list[tuple[float, int]] = []
    monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", False)
    monkeypatch.setattr(modal_cloud_module, "_is_modal_container_runtime", lambda: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_schedule_process_exit",
        lambda delay_seconds, exit_code: scheduled_exits.append((delay_seconds, exit_code)),
    )
    error = execution_control_module.RemoteExecutionCancelledBeforeStart(
        "Remote execution for prompt=prompt-1 component=269:251 was interrupted "
        "before this container started it."
    )

    scheduled = modal_cloud_module._maybe_schedule_container_termination_on_error(
        dict(_PAYLOAD), error
    )

    assert scheduled is False
    assert scheduled_exits == []
    assert (
        modal_cloud_module._remote_failure_disposition(error)
        is modal_cloud_module.RemoteFailureDisposition.EXPECTED
    )


def test_entrypoints_check_cancellation_before_observing_the_workflow() -> None:
    """Cancelled requests must not count as a container's first workflow for vLLM mode."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "comfyui_modal_sync_cloud.py").read_text()
    entry = (
        "with _registered_remote_execution(payload) as execution_control:\n"
        "                    _observe_remote_workflow_for_llm_mode(payload)"
    )
    assert source.count(entry) == 2
