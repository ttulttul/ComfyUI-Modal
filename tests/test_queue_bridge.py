"""Tests for the queue bridge boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_modal_interrupt_queue_bridge_exposes_active_remote_prompts(
    api_intercept_module: Any,
    queue_bridge_module: Any,
    remote_modal_app_module: Any,
) -> None:
    """Targeted ComfyUI interrupts should see prompts currently blocked on Modal work."""

    class FakePromptQueue:
        """Minimal ComfyUI prompt queue with no native running prompts."""

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return an empty running queue and one pending item."""
            return [], ["queued"]

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(prompt_queue=prompt_queue)
    cancellation_event = remote_modal_app_module.threading.Event()

    queue_bridge_module._install_modal_interrupt_queue_bridge(prompt_server)
    with remote_modal_app_module._registered_active_remote_invocation(
        {"prompt_id": "prompt-1", "component_id": "component-1"},
        cancellation_event,
        None,
    ):
        running, queued = prompt_queue.get_current_queue()

    assert queued == ["queued"]
    assert [item[1] for item in running] == ["prompt-1"]

def test_remote_preparation_bridge_exposes_work_to_all_queue_views(
    api_intercept_module: Any,
    queue_bridge_module: Any,
) -> None:
    """Capacity acquisition should look active before the rewritten prompt is queued."""

    class FakePromptQueue:
        """Minimal native queue with every state method used by ComfyUI."""

        def __init__(self) -> None:
            """Initialize an empty native queue."""
            self.running: list[tuple[Any, ...]] = []
            self.queued: list[tuple[Any, ...]] = []

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return stable native queue state."""
            return list(self.running), list(self.queued)

        def get_current_queue_volatile(self) -> tuple[list[Any], list[Any]]:
            """Return volatile native queue state."""
            return list(self.running), list(self.queued)

        def get_tasks_remaining(self) -> int:
            """Count native running and pending prompts."""
            return len(self.running) + len(self.queued)

    prompt_queue = FakePromptQueue()
    queue_update_counts: list[int] = []
    prompt_server = SimpleNamespace(
        prompt_queue=prompt_queue,
        queue_updated=lambda: queue_update_counts.append(
            prompt_queue.get_tasks_remaining()
        ),
    )
    queue_bridge_module._install_modal_interrupt_queue_bridge(prompt_server)

    registered = queue_bridge_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-preparing",
        prompt={"1": {"class_type": "RemoteImage", "inputs": {}}},
        extra_data={"client_id": "client-1"},
    )

    assert registered is True
    assert prompt_queue.get_tasks_remaining() == 1
    assert [item[1] for item in prompt_queue.get_current_queue()[0]] == [
        "prompt-preparing"
    ]
    assert [item[1] for item in prompt_queue.get_current_queue_volatile()[0]] == [
        "prompt-preparing"
    ]
    preparation_item = prompt_queue.get_current_queue_volatile()[0][0]
    assert preparation_item[3]["client_id"] == "client-1"
    assert isinstance(preparation_item[3]["create_time"], int)
    assert preparation_item[3]["create_time"] > 0

    prompt_queue.queued.append((1, "prompt-preparing", {}, {}, []))
    assert prompt_queue.get_tasks_remaining() == 1
    assert [item[1] for item in prompt_queue.get_current_queue_volatile()[0]] == []

    queue_bridge_module._clear_remote_preparation(
        prompt_server,
        "prompt-preparing",
    )

    assert prompt_queue.get_tasks_remaining() == 1
    assert queue_update_counts == [1, 1]

def test_queued_ssh_environment_ids_reads_earlier_prompt_assignments(
    api_intercept_module: Any,
    queue_bridge_module: Any,
) -> None:
    """Queue-time planning should recognize SSH hosts owned by earlier prompts."""

    def assignment(provider: str, environment_id: str) -> dict[str, str]:
        """Return minimal serialized placement metadata."""
        return {
            "provider": provider,
            "environment_id": environment_id,
        }
    prompt_server = SimpleNamespace(
        prompt_queue=SimpleNamespace(
            get_current_queue=lambda: (
                [
                    (
                        1,
                        "prompt-running",
                        {},
                        {
                            "remote_execution": {
                                "assignments": {
                                    "257": assignment("ssh_docker", "lambda")
                                }
                            }
                        },
                    )
                ],
                [
                    (
                        2,
                        "prompt-current",
                        {},
                        {
                            "remote_execution": {
                                "assignments": {
                                    "300": assignment("ssh_docker", "ignored")
                                }
                            }
                        },
                    ),
                    (
                        3,
                        "prompt-modal",
                        {},
                        {
                            "remote_execution": {
                                "assignments": {
                                    "400": assignment("modal", "modal:H100")
                                }
                            }
                        },
                    ),
                ],
            )
        )
    )

    environment_ids = queue_bridge_module._queued_ssh_environment_ids(
        prompt_server,
        excluding_prompt_id="prompt-current",
    )

    assert environment_ids == frozenset({"lambda"})

def test_remote_preparation_bridge_clears_failed_submission(
    api_intercept_module: Any,
    queue_bridge_module: Any,
) -> None:
    """A rejected pre-queue prompt must not leave phantom active queue work."""

    class FakePromptQueue:
        """Minimal empty queue used to exercise preparation cleanup."""

        def get_current_queue_volatile(self) -> tuple[list[Any], list[Any]]:
            """Return empty native queue state."""
            return [], []

        def get_tasks_remaining(self) -> int:
            """Return the empty native queue count."""
            return 0

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(
        prompt_queue=prompt_queue,
        queue_updated=lambda: None,
    )
    queue_bridge_module._install_modal_interrupt_queue_bridge(prompt_server)
    assert queue_bridge_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-failed",
        prompt={},
        extra_data={},
    )
    assert prompt_queue.get_tasks_remaining() == 1

    queue_bridge_module._clear_remote_preparation(prompt_server, "prompt-failed")

    assert prompt_queue.get_tasks_remaining() == 0
    assert prompt_queue.get_current_queue_volatile() == ([], [])

def test_remote_preparation_bridge_tracks_prompt_cancellation(
    api_intercept_module: Any,
    queue_bridge_module: Any,
) -> None:
    """Queue-time work should expose a prompt-scoped cancellation event."""

    class FakePromptQueue:
        """Provide the queue method required by the bridge."""

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return no native work."""
            return [], []

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(prompt_queue=prompt_queue)
    cancellation_event = api_intercept_module.threading.Event()
    queue_bridge_module._install_modal_interrupt_queue_bridge(prompt_server)

    assert queue_bridge_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-cancel",
        prompt={},
        extra_data={},
        cancellation_event=cancellation_event,
    )
    cancellations = getattr(
        prompt_queue,
        queue_bridge_module._REMOTE_PREPARATION_CANCELLATIONS_ATTR,
    )
    assert cancellations["prompt-cancel"] is cancellation_event

    queue_bridge_module._clear_remote_preparation(prompt_server, "prompt-cancel")

    assert "prompt-cancel" not in cancellations

def test_jobs_api_interrupt_cancels_remote_preparation(
    api_intercept_module: Any,
    queue_bridge_module: Any,
) -> None:
    """ComfyUI's normal Jobs API cancellation should stop remote setup."""

    class FakePromptQueue:
        """Provide queue and interruption methods used by the bridge."""

        def __init__(self) -> None:
            """Track whether native interruption was used."""
            self.native_interrupts: list[str] = []

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return no native work."""
            return [], []

        def interrupt_if_running(self, prompt_id: str) -> bool:
            """Record native fallback interruptions."""
            self.native_interrupts.append(prompt_id)
            return False

    prompt_queue = FakePromptQueue()
    prompt_server = SimpleNamespace(prompt_queue=prompt_queue)
    cancellation_event = api_intercept_module.threading.Event()
    queue_bridge_module._install_modal_interrupt_queue_bridge(prompt_server)
    queue_bridge_module._set_remote_preparation(
        prompt_server,
        prompt_id="prompt-cancel",
        prompt={},
        extra_data={},
        cancellation_event=cancellation_event,
    )

    running, queued = prompt_queue.get_current_queue()
    assert [item[1] for item in running] == ["prompt-cancel"]
    assert queued == []
    assert prompt_queue.interrupt_if_running("prompt-cancel") is True
    assert cancellation_event.is_set()
    assert prompt_queue.native_interrupts == []
    assert prompt_queue.interrupt_if_running("native-prompt") is False
    assert prompt_queue.native_interrupts == ["native-prompt"]

def test_queue_bridge_releases_r2_writeback_reservations(
    api_intercept_module: Any,
    queue_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """Completed, deleted, and wiped prompts should release idle cache work."""

    class FakePromptQueue:
        """Model the native lifecycle methods wrapped by the remote queue bridge."""

        def __init__(self) -> None:
            """Initialize one running prompt and two queued prompts."""
            self.currently_running = {
                7: (0, "prompt-running", {}, {}, [], {}),
            }
            self.queue = [
                (1, "prompt-delete", {}, {}, [], {}),
                (2, "prompt-wipe", {}, {}, [], {}),
            ]

        def get_current_queue(self) -> tuple[list[Any], list[Any]]:
            """Return native running and queued snapshots."""
            return list(self.currently_running.values()), list(self.queue)

        def task_done(self, item_id: int, *_args: Any, **_kwargs: Any) -> None:
            """Remove one completed running prompt."""
            self.currently_running.pop(item_id)

        def delete_queue_item(self, predicate: Any) -> bool:
            """Delete the first queued item matching a predicate."""
            for index, item in enumerate(self.queue):
                if predicate(item):
                    self.queue.pop(index)
                    return True
            return False

        def wipe_queue(self) -> None:
            """Delete every queued prompt."""
            self.queue.clear()

    released: list[str] = []
    monkeypatch.setattr(
        queue_bridge_module,
        "finish_r2_writeback_prompt",
        released.append,
    )
    prompt_queue = FakePromptQueue()
    queue_bridge_module._install_modal_interrupt_queue_bridge(
        SimpleNamespace(prompt_queue=prompt_queue)
    )

    prompt_queue.task_done(7, {})
    assert prompt_queue.delete_queue_item(
        lambda item: item[1] == "prompt-delete"
    ) is True
    prompt_queue.wipe_queue()

    assert released == ["prompt-running", "prompt-delete", "prompt-wipe"]

