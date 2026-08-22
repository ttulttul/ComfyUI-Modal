"""Tests for provider routing at the generated proxy boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def test_router_defaults_legacy_payloads_to_modal(
    modal_executor_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing workflows without provider metadata must retain Modal behavior."""
    calls: list[str] = []

    class FakeModalClient:
        """Record one routed Modal execution."""

        def execute_payload(self, payload: Any, kwargs: Any) -> list[str]:
            """Return a deterministic provider marker."""
            del payload, kwargs
            calls.append("modal")
            return ["modal-output"]

    monkeypatch.setattr(modal_executor_module, "ModalRemoteExecutorClient", FakeModalClient)

    result = modal_executor_module.RemoteExecutorRouterClient().execute_payload({}, {})

    assert result == ["modal-output"]
    assert calls == ["modal"]


def test_router_rejects_unknown_provider(modal_executor_module: Any) -> None:
    """Planner/provider drift should fail before any remote call begins."""
    with pytest.raises(ValueError, match="Unsupported remote execution provider"):
        modal_executor_module.RemoteExecutorRouterClient().execute_payload(
            {"execution_provider": "mystery"},
            {},
        )


def test_router_records_successful_component_runtime(
    modal_executor_module: Any,
    settings_module: Any,
    execution_history_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Completed proxy calls should feed future environment cost estimates."""

    class FakeModalClient:
        """Return immediately from one deterministic execution."""

        def execute_payload(self, payload: Any, kwargs: Any) -> list[str]:
            """Return one output."""
            del payload, kwargs
            return ["output"]

    monkeypatch.setattr(modal_executor_module, "ModalRemoteExecutorClient", FakeModalClient)
    monkeypatch.setattr(settings_module, "get_settings", lambda: SimpleNamespace(modal_gpu="L40S"))
    monkeypatch.setattr(
        settings_module,
        "discover_comfyui_user_directory",
        lambda _settings: tmp_path,
    )

    result = modal_executor_module.RemoteExecutorRouterClient().execute_payload(
        {
            "execution_provider": "modal",
            "execution_environment_id": "modal:L40S",
            "execution_history_signature": "signature",
        },
        {},
    )

    estimates = execution_history_module.ExecutionHistory.for_user_directory(
        tmp_path
    ).estimates("signature", ("modal:L40S",))
    assert result == ["output"]
    assert estimates["modal:L40S"].sample_count == 1
