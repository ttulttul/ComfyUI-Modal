"""Opt-in live canaries for the deployed Modal execution path."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import os
import time
from typing import Any, Iterator
import uuid

import pytest


def _environment_flag_enabled(name: str) -> bool:
    """Return whether one opt-in environment flag is truthy."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.live_modal,
    pytest.mark.skipif(
        not _environment_flag_enabled("COMFY_MODAL_RUN_LIVE_CANARIES"),
        reason="set COMFY_MODAL_RUN_LIVE_CANARIES=1 to spend live Modal resources",
    ),
]


@dataclass
class _LiveModalCanaryContext:
    """Track the configured client and shared-state keys created by live canaries."""

    remote_module: Any
    settings: Any
    shared_store_keys: set[str] = field(default_factory=set)

    def payload(
        self,
        name: str,
        *,
        delay_seconds: float = 0.0,
        canary_barrier: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one isolated dependency-light canary payload."""
        unique_suffix = uuid.uuid4().hex
        invocation_id = f"RIV_CANARY_{unique_suffix}"
        self.shared_store_keys.add(invocation_id)
        payload: dict[str, Any] = {
            "payload_kind": "canary",
            "prompt_id": f"live-canary-{name}-{unique_suffix}",
            "component_id": f"live-canary-{name}",
            "invocation_id": invocation_id,
            "canary_delay_seconds": delay_seconds,
            "requires_volume_reload": False,
            "terminate_container_on_error": False,
        }
        if canary_barrier is not None:
            payload["canary_barrier"] = canary_barrier
            barrier_id = str(canary_barrier["barrier_id"])
            for member_id in canary_barrier["members"]:
                self.shared_store_keys.add(
                    f"CANARY_BARRIER:{barrier_id}:{member_id}"
                )
        return payload

    def invoke(self, payload: dict[str, Any], value: Any) -> tuple[Any, dict[str, Any]]:
        """Invoke one live canary and deserialize its echoed value and metadata."""
        serialized_inputs = self.remote_module.serialize_node_inputs({"value": value})
        response = self.remote_module.invoke_remote_engine(
            payload,
            serialized_inputs,
            allow_implicit_mapping=False,
        )
        outputs = self.remote_module.deserialize_node_outputs(response)
        assert len(outputs) == 2
        assert isinstance(outputs[1], dict)
        return outputs[0], outputs[1]

    def cleanup_shared_state(self) -> None:
        """Remove invocation and barrier metadata created by the live canaries."""
        if not self.shared_store_keys:
            return
        modal_module = self.remote_module.modal
        invocation_store = modal_module.Dict.from_name(
            self.settings.invocation_dict_name,
            environment_name=self.remote_module._modal_environment_name(),
            create_if_missing=True,
        )
        for shared_store_key in self.shared_store_keys:
            invocation_store.pop(shared_store_key, None)


@pytest.fixture
def live_modal_canary(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_LiveModalCanaryContext]:
    """Configure a real Modal client while keeping the normal suite local-only."""
    if remote_modal_app_module.modal is None:
        pytest.fail(
            "Live Modal canaries require the remote extra: "
            "uv run --extra remote pytest tests/test_live_modal_canary.py"
        )
    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK", "false")
    remote_modal_app_module.get_settings.cache_clear()
    settings = remote_modal_app_module.get_settings()
    context = _LiveModalCanaryContext(
        remote_module=remote_modal_app_module,
        settings=settings,
    )
    try:
        yield context
    finally:
        context.cleanup_shared_state()
        remote_modal_app_module.get_settings.cache_clear()


def test_live_modal_runtime_handshake(live_modal_canary: _LiveModalCanaryContext) -> None:
    """The deployed worker should echo data and match the local runtime fingerprint."""
    payload = live_modal_canary.payload("handshake")

    echoed_value, metadata = live_modal_canary.invoke(payload, "handshake-ok")
    remote_engine = live_modal_canary.remote_module._lookup_deployed_remote_engine(
        dict(payload)
    )
    version_payload = live_modal_canary.remote_module._remote_engine_runtime_version(
        remote_engine
    )

    assert echoed_value == "handshake-ok"
    assert metadata["component_id"] == "live-canary-handshake"
    assert live_modal_canary.remote_module._is_runtime_version_payload_current(
        version_payload
    )


def test_live_modal_binary_transport_and_durable_replay(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """Tensor RPC should stay binary and a duplicate call should replay exact metadata."""
    import torch

    payload = live_modal_canary.payload("binary-replay")
    value = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1, 1024, 1024)

    first_value, first_metadata = live_modal_canary.invoke(payload, value)
    replayed_value, replayed_metadata = live_modal_canary.invoke(payload, value)

    assert torch.equal(first_value, value)
    assert torch.equal(replayed_value, value)
    assert first_metadata["transport_kind"] == "binary"
    assert replayed_metadata == first_metadata


def test_live_modal_parallel_dispatch_reaches_barrier(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """Two remote calls should be active together instead of serializing on one worker."""
    if live_modal_canary.settings.max_inflight_calls < 2:
        pytest.skip("parallel canary requires COMFY_MODAL_MAX_INFLIGHT_CALLS >= 2")
    if (
        live_modal_canary.settings.max_containers is not None
        and live_modal_canary.settings.max_containers < 2
    ):
        pytest.skip("parallel canary requires COMFY_MODAL_MAX_CONTAINERS >= 2")

    barrier_id = f"parallel-{uuid.uuid4().hex}"
    members = ["member-a", "member-b"]
    payloads = [
        live_modal_canary.payload(
            member_id,
            delay_seconds=0.25,
            canary_barrier={
                "barrier_id": barrier_id,
                "member_id": member_id,
                "members": members,
                "timeout_seconds": 90.0,
            },
        )
        for member_id in members
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(live_modal_canary.invoke, payload, member_id)
            for payload, member_id in zip(payloads, members, strict=True)
        ]
        results = [future.result(timeout=180.0) for future in futures]

    metadata = [result[1] for result in results]
    assert [result[0] for result in results] == members
    assert all(item["barrier_released_at"] is not None for item in metadata)
    assert len({item["modal_task_id"] for item in metadata}) == 2


def _wait_for_active_prompt(
    context: _LiveModalCanaryContext,
    prompt_id: str,
    *,
    timeout_seconds: float,
) -> None:
    """Wait until the local client has registered a cancellable Modal invocation."""
    deadline = time.monotonic() + timeout_seconds
    while prompt_id not in context.remote_module.active_remote_modal_prompt_ids():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Live Modal invocation {prompt_id!r} never became cancellable."
            )
        time.sleep(0.05)


def test_live_modal_cancellation_propagates(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """Prompt cancellation should reach and stop a deliberately delayed remote call."""
    import comfy.model_management

    payload = live_modal_canary.payload("cancellation", delay_seconds=30.0)
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future[tuple[Any, dict[str, Any]]] = executor.submit(
        live_modal_canary.invoke,
        payload,
        "must-not-complete",
    )
    try:
        _wait_for_active_prompt(
            live_modal_canary,
            str(payload["prompt_id"]),
            timeout_seconds=120.0,
        )
        assert live_modal_canary.remote_module.request_remote_modal_prompt_interrupt(
            str(payload["prompt_id"])
        )
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            future.result(timeout=15.0)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
