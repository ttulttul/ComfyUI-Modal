"""Tests for the routes modal containers boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_emit_modal_status_targets_prompt_client(
    api_intercept_module: Any,
    modal_ui_events_module: Any,
) -> None:
    """Modal status events should preserve prompt and component metadata for the UI."""
    modal_ui_events_module._MODAL_UI_EVENTS_BY_CLIENT.clear()

    class FakePromptServer:
        """Capture websocket events emitted by the queue route."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[str, dict[str, Any], str | None]] = []

        def send_sync(self, event: str, data: dict[str, Any], sid: str | None) -> None:
            """Record an emitted websocket message."""
            self.messages.append((event, data, sid))

    prompt_server = FakePromptServer()
    modal_ui_events_module._emit_modal_status(
        prompt_server=prompt_server,
        phase="executing",
        client_id="client-1",
        prompt_id="prompt-1",
        node_ids=["4", "5"],
        configurator_node_id="99",
        modal_gpu="B300",
        component_node_ids_by_representative={"4": ["4", "5"]},
        active_node_id="5",
        active_node_class_type="KSampler",
        active_node_role="sampling",
        execution_environment_id="vast:48602895",
        remote_execution_assignments={
            "4": {"provider": "vast", "node_ids": ["4", "5"]}
        },
        remote_execution_configurations=[
            {"configuration_id": "vast-big", "display_name": "Vast Big"}
        ],
    )

    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["4", "5"],
                "configurator_node_id": "99",
                "modal_gpu": "B300",
                "active_node_id": "5",
                "active_node_class_type": "KSampler",
                "active_node_role": "sampling",
                "execution_environment_id": "vast:48602895",
                "components": [
                    {
                        "representative_node_id": "4",
                        "node_ids": ["4", "5"],
                    }
                ],
                "remote_execution_assignments": {
                    "4": {"provider": "vast", "node_ids": ["4", "5"]}
                },
                "remote_execution_configurations": [
                    {
                        "configuration_id": "vast-big",
                        "display_name": "Vast Big",
                    }
                ],
            },
            "client-1",
        )
    ]
    replay_events = modal_ui_events_module.modal_ui_events_for_client("client-1")
    assert replay_events == [
        {
            "event": "modal_status",
            "payload": {
                "phase": "executing",
                "prompt_id": "prompt-1",
                "node_ids": ["4", "5"],
                "configurator_node_id": "99",
                "modal_gpu": "B300",
                "active_node_id": "5",
                "active_node_class_type": "KSampler",
                "active_node_role": "sampling",
                "execution_environment_id": "vast:48602895",
                "components": [
                    {
                        "representative_node_id": "4",
                        "node_ids": ["4", "5"],
                    }
                ],
                "remote_execution_assignments": {
                    "4": {"provider": "vast", "node_ids": ["4", "5"]}
                },
                "remote_execution_configurations": [
                    {
                        "configuration_id": "vast-big",
                        "display_name": "Vast Big",
                    }
                ],
            },
            "updated_at": replay_events[0]["updated_at"],
        }
    ]

def test_modal_ui_event_replay_is_client_scoped(modal_ui_events_module: Any) -> None:
    """Refocus replay should only return events for the requesting ComfyUI client."""
    modal_ui_events_module._MODAL_UI_EVENTS_BY_CLIENT.clear()

    modal_ui_events_module.record_modal_ui_event(
        "modal_progress",
        {"prompt_id": "prompt-1", "node_id": "4", "value": 2.0, "max": 10.0},
        "client-1",
    )
    modal_ui_events_module.record_modal_ui_event(
        "modal_status",
        {"prompt_id": "prompt-2", "phase": "executing", "node_ids": ["9"]},
        "client-2",
    )

    replay_events = modal_ui_events_module.modal_ui_events_for_client("client-1")

    assert len(replay_events) == 1
    assert replay_events[0]["event"] == "modal_progress"
    assert replay_events[0]["payload"] == {
        "prompt_id": "prompt-1",
        "node_id": "4",
        "value": 2.0,
        "max": 10.0,
    }
    assert modal_ui_events_module.modal_ui_events_for_client(None) == []


def test_modal_status_preserves_attributed_failure_node(
    modal_ui_events_module: Any,
) -> None:
    """Queue failures should tell the frontend which configuration to highlight."""

    class FakePromptServer:
        """Capture one status event."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[str, dict[str, Any], str | None]] = []

        def send_sync(self, event: str, data: dict[str, Any], sid: str | None) -> None:
            """Record an emitted websocket message."""
            self.messages.append((event, data, sid))

    prompt_server = FakePromptServer()
    modal_ui_events_module._emit_modal_status(
        prompt_server=prompt_server,
        phase="error",
        client_id="client-1",
        prompt_id="prompt-1",
        node_ids=["7"],
        failed_node_id="42",
        error_code="subrosa_login_required",
        error_message="Click Login again.",
    )

    assert prompt_server.messages[0][1]["failed_node_id"] == "42"
    assert prompt_server.messages[0][1]["error_code"] == "subrosa_login_required"


def test_modal_telemetry_replay_coalesces_each_execution_source(
    modal_ui_events_module: Any,
) -> None:
    """Periodic samples must not evict durable status events from replay history."""
    modal_ui_events_module._MODAL_UI_EVENTS_BY_CLIENT.clear()
    base_payload = {
        "prompt_id": "prompt-memory",
        "execution_environment_id": "modal:gpu-a",
        "execution_location": "ta-one",
        "component_id": "170",
    }
    modal_ui_events_module.record_modal_ui_event(
        "modal_telemetry",
        {**base_payload, "cpu_memory_peak_bytes": 100},
        "client-memory",
    )
    modal_ui_events_module.record_modal_ui_event(
        "modal_status",
        {"prompt_id": "prompt-memory", "phase": "executing"},
        "client-memory",
    )
    modal_ui_events_module.record_modal_ui_event(
        "modal_telemetry",
        {**base_payload, "cpu_memory_peak_bytes": 300},
        "client-memory",
    )
    modal_ui_events_module.record_modal_ui_event(
        "modal_telemetry",
        {**base_payload, "component_id": "171", "cpu_memory_peak_bytes": 200},
        "client-memory",
    )

    replay_events = modal_ui_events_module.modal_ui_events_for_client("client-memory")

    assert [event["event"] for event in replay_events] == [
        "modal_status",
        "modal_telemetry",
        "modal_telemetry",
    ]
    assert [
        event["payload"]["cpu_memory_peak_bytes"]
        for event in replay_events
        if event["event"] == "modal_telemetry"
    ] == [300, 200]

def test_progress_state_route_is_queue_route_sibling(api_intercept_module: Any) -> None:
    """The frontend should have a stable sibling route for Modal UI replay."""
    assert api_intercept_module._progress_state_route_path("/modal/queue_prompt") == (
        "/modal/progress_state"
    )
    assert api_intercept_module._progress_state_route_path("/custom/modal") == (
        "/custom/modal/progress_state"
    )

def test_container_status_route_is_queue_route_sibling(api_intercept_module: Any) -> None:
    """The frontend should have a stable sibling route for active Modal containers."""
    assert api_intercept_module._container_status_route_path("/modal/queue_prompt") == (
        "/modal/container_status"
    )

def test_modal_reset_route_paths_are_queue_route_siblings(api_intercept_module: Any) -> None:
    """The frontend should have stable sibling routes for Modal maintenance actions."""
    assert api_intercept_module._delete_modal_caches_route_path("/modal/queue_prompt") == (
        "/modal/delete_caches"
    )
    assert api_intercept_module._delete_modal_volume_route_path("/modal/queue_prompt") == (
        "/modal/delete_volume"
    )
    assert api_intercept_module._delete_modal_caches_route_path("/custom/modal") == (
        "/custom/modal/delete_caches"
    )
    assert api_intercept_module._delete_modal_volume_route_path("/custom/modal") == (
        "/custom/modal/delete_volume"
    )
