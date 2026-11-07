"""Tests for provider routing at the generated proxy boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest


def test_router_defaults_legacy_payloads_to_modal(
    remote_executor_router_module: Any,
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

    monkeypatch.setattr(
        remote_executor_router_module,
        "ModalRemoteExecutorClient",
        FakeModalClient,
    )

    result = remote_executor_router_module.RemoteExecutorRouterClient().execute_payload({}, {})

    assert result == ["modal-output"]
    assert calls == ["modal"]


def test_router_rejects_unknown_provider(remote_executor_router_module: Any) -> None:
    """Planner/provider drift should fail before any remote call begins."""
    with pytest.raises(ValueError, match="Unsupported remote execution provider"):
        remote_executor_router_module.RemoteExecutorRouterClient().execute_payload(
            {"execution_provider": "mystery"},
            {},
        )


def test_router_threads_queue_time_vast_runtime_expectation(
    remote_executor_router_module: Any,
    settings_module: Any,
    vast_service_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-invocation construction should use the fingerprint captured at planning."""
    fingerprint = "f" * 64
    image = "ghcr.io/example/worker@sha256:" + "a" * 64
    observed: dict[str, Any] = {}
    executor = SimpleNamespace()

    def fake_from_environment(_settings: Any, **kwargs: Any) -> Any:
        """Capture the explicit runtime expectation and return a fake service."""
        observed.update(kwargs)
        return SimpleNamespace(executor=lambda: executor)

    monkeypatch.setattr(settings_module, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        vast_service_module.VastService,
        "from_environment",
        fake_from_environment,
    )

    client = remote_executor_router_module.RemoteExecutorRouterClient()._client_for_payload(
        {
            "execution_provider": "vast",
            "vast_runtime_fingerprint": fingerprint,
            "vast_worker_image": image,
        }
    )

    assert client is executor
    assert observed["runtime_fingerprint"] == fingerprint
    assert observed["worker_image"] == image


def test_router_records_successful_component_runtime(
    remote_executor_router_module: Any,
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

    monkeypatch.setattr(
        remote_executor_router_module,
        "ModalRemoteExecutorClient",
        FakeModalClient,
    )
    monkeypatch.setattr(settings_module, "get_settings", lambda: SimpleNamespace(modal_gpu="L40S"))
    monkeypatch.setattr(
        settings_module,
        "discover_comfyui_user_directory",
        lambda _settings: tmp_path,
    )

    result = remote_executor_router_module.RemoteExecutorRouterClient().execute_payload(
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


def _ssh_mapped_text_payload() -> dict[str, Any]:
    """Return the SSH ``1__mapped`` shape from a ModalMapInput prompt workflow."""
    return {
        "payload_kind": "subgraph",
        "component_id": "1__mapped",
        "prompt_id": "prompt-ssh",
        "execution_provider": "ssh_docker",
        "execution_environment_id": "lambda",
        "execute_node_ids": ["5"],
        "subgraph_prompt": {
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": ["remote_input_0", 0],
                    "clip": ["static_input_0", 0],
                },
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "STRING",
                "targets": [{"node_id": "5", "input_name": "text"}],
            },
            {
                "proxy_input_name": "static_input_0",
                "io_type": "CLIP",
                "targets": [{"node_id": "5", "input_name": "clip"}],
            },
        ],
        "boundary_outputs": [{"node_id": "5", "io_type": "STRING", "is_list": False}],
        "extra_data": {"client_id": "client-1"},
    }


class _RecordingProviderClient:
    """Fake SSH client that records each payload and echoes its text input."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def execute_payload_async(
        self,
        payload: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> list[str]:
        """Reject list-valued text, which is the original SSH failure mode."""
        self.calls.append((dict(payload), dict(kwargs)))
        text = kwargs["remote_input_0"]
        assert isinstance(text, str), f"mapped item received {type(text).__name__}"
        clip = kwargs["static_input_0"]
        if isinstance(clip, list):
            assert len(clip) == 1, "broadcast inputs must stay singleton lists"
            clip = clip[0]
        return [f"{text}|{clip}"]


def test_router_fans_out_mapped_inputs_for_ssh_provider(
    remote_executor_router_module: Any,
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSH workers must receive one ModalMapInput item per invocation, not the whole list."""
    provider_client = _RecordingProviderClient()
    router = remote_executor_router_module.RemoteExecutorRouterClient()
    monkeypatch.setattr(router, "_client_for_payload", lambda _payload: provider_client)
    monkeypatch.setattr(router, "_record_success", lambda *_args: None)

    async def forbid_modal_invocation(*_args: Any, **_kwargs: Any) -> bytes:
        """Fail if the SSH fan-out leaks into the Modal invocation path."""
        raise AssertionError("SSH mapped items must not dispatch through Modal.")

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        forbid_modal_invocation,
    )

    result = asyncio.run(
        router.execute_payload_async(
            _ssh_mapped_text_payload(),
            {
                "remote_input_0": ["prompt-a", "prompt-b", "prompt-c"],
                "static_input_0": ["clip-ref"],
            },
        )
    )

    item_calls = [
        (payload, kwargs)
        for payload, kwargs in provider_client.calls
        if payload.get("map_item_index") is not None
    ]
    assert [payload["map_item_index"] for payload, _ in item_calls] == [0, 1, 2]
    assert all(payload["map_item_count"] == 3 for payload, _ in item_calls)
    assert all(payload["execution_provider"] == "ssh_docker" for payload, _ in item_calls)
    assert [kwargs["remote_input_0"] for _, kwargs in item_calls] == [
        "prompt-a",
        "prompt-b",
        "prompt-c",
    ]
    assert list(result) == [
        ["prompt-a|clip-ref", "prompt-b|clip-ref", "prompt-c|clip-ref"]
    ]


def test_router_leaves_unbatched_ssh_payloads_unchanged(
    remote_executor_router_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-value SSH subgraph should still execute as exactly one invocation."""
    provider_client = _RecordingProviderClient()
    router = remote_executor_router_module.RemoteExecutorRouterClient()
    monkeypatch.setattr(router, "_client_for_payload", lambda _payload: provider_client)
    monkeypatch.setattr(router, "_record_success", lambda *_args: None)

    result = asyncio.run(
        router.execute_payload_async(
            _ssh_mapped_text_payload(),
            {"remote_input_0": "only-prompt", "static_input_0": "clip-ref"},
        )
    )

    assert len(provider_client.calls) == 1
    assert "map_item_index" not in provider_client.calls[0][0]
    assert result == ["only-prompt|clip-ref"]
