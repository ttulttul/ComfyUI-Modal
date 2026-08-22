"""Tests for provider routing at the generated proxy boundary."""

from __future__ import annotations

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
