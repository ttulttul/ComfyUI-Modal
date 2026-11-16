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

def _clear_modal_ui_replay_state(modal_ui_events_module: Any) -> None:
    """Reset the module-level replay buffers between tests."""
    modal_ui_events_module._MODAL_UI_EVENTS_BY_CLIENT.clear()
    modal_ui_events_module._PROGRESSIVE_PREVIEW_SNAPSHOTS_BY_CLIENT.clear()


def _preview_slot_payload(prompt_id: str, slot_index: int, slot_count: int) -> dict[str, Any]:
    """Return one progressive preview slot websocket payload."""
    return {
        "phase": "slot",
        "prompt_id": prompt_id,
        "node": "7",
        "slot_index": slot_index,
        "slot_count": slot_count,
        "items": [{"kind": "image", "filename": f"{slot_index}.png", "type": "temp"}],
    }


def test_progressive_preview_replay_survives_progress_event_flood(
    modal_ui_events_module: Any,
) -> None:
    """Preview slots must still replay after a run's progress events overflow the deque."""
    _clear_modal_ui_replay_state(modal_ui_events_module)
    record = modal_ui_events_module.record_modal_ui_event
    record(
        "modal_progressive_preview",
        {"phase": "start", "prompt_id": "p1", "node": "7", "slot_count": 3},
        "client-1",
    )
    record("modal_progressive_preview", _preview_slot_payload("p1", 2, 3), "client-1")
    record("modal_progressive_preview", _preview_slot_payload("p1", 0, 3), "client-1")
    for step in range(modal_ui_events_module._MODAL_UI_EVENT_LIMIT_PER_CLIENT + 10):
        record("modal_progress", {"prompt_id": "p1", "value": step}, "client-1")

    replay_events = modal_ui_events_module.modal_ui_events_for_client("client-1")

    preview_events = [e for e in replay_events if e["event"] == "modal_progressive_preview"]
    assert len(preview_events) == 1
    assert preview_events[0]["payload"] == {
        "phase": "snapshot",
        "prompt_id": "p1",
        "node": "7",
        "slot_count": 3,
        "slots": [
            {"slot_index": 0, "items": [{"kind": "image", "filename": "0.png", "type": "temp"}]},
            {"slot_index": 2, "items": [{"kind": "image", "filename": "2.png", "type": "temp"}]},
        ],
    }
    assert modal_ui_events_module.modal_ui_events_for_client("client-2") == []


def test_progressive_preview_start_resets_snapshot_and_old_prompts_are_bounded(
    modal_ui_events_module: Any,
) -> None:
    """A repeated start clears slots, and each client keeps only recent prompt snapshots."""
    _clear_modal_ui_replay_state(modal_ui_events_module)
    record = modal_ui_events_module.record_modal_ui_event
    record("modal_progressive_preview", _preview_slot_payload("p1", 0, 2), "client-1")
    record(
        "modal_progressive_preview",
        {"phase": "start", "prompt_id": "p1", "node": "7", "slot_count": 2},
        "client-1",
    )
    payload = modal_ui_events_module.modal_ui_events_for_client("client-1")[0]["payload"]
    assert payload["slots"] == []
    assert payload["slot_count"] == 2

    limit = modal_ui_events_module._PROGRESSIVE_PREVIEW_SNAPSHOT_LIMIT_PER_CLIENT
    for index in range(limit + 3):
        record("modal_progressive_preview", _preview_slot_payload(f"q{index}", 0, 1), "client-1")
    replayed = modal_ui_events_module.modal_ui_events_for_client("client-1")
    assert len(replayed) == limit
    assert replayed[-1]["payload"]["prompt_id"] == f"q{limit + 2}"


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

