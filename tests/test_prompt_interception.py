"""Tests for the prompt interception boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_queue_prompt_json_includes_resolved_modal_metadata(
    api_intercept_module: Any,
    queue_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """Successful queue responses should include resolved remote nodes and component membership."""

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def __init__(self) -> None:
            """Initialize captured queue items."""
            self.items: list[tuple[Any, ...]] = []

        def put(self, item: tuple[Any, ...]) -> None:
            """Record one queued prompt item."""
            self.items.append(item)

    class FakePromptServer:
        """Minimal PromptServer double for queue-response tests."""

        def __init__(self) -> None:
            """Initialize queue state."""
            self.number = 0
            self.prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeExecutionModule:
        """Minimal execution module exposing prompt validation."""

        SENSITIVE_EXTRA_DATA_KEYS: tuple[str, ...] = ()

        @staticmethod
        async def validate_prompt(
            prompt_id: str,
            prompt: dict[str, Any],
            partial_execution_targets: Any,
        ) -> tuple[bool, None, list[str], list[Any]]:
            """Accept the supplied prompt with one fake execution target."""
            return True, None, ["1"], []

    monkeypatch.setattr(queue_bridge_module, "_get_execution_module", lambda: FakeExecutionModule)
    prompt_server = FakePromptServer()

    response = asyncio.run(
        queue_bridge_module._queue_prompt_json(
            prompt_server,
            {
                "prompt_id": "prompt-1",
                "prompt": {"1": {"class_type": "Anything", "inputs": {}}},
                "extra_data": {},
            },
            modal_response_payload={
                "modal_remote_node_ids": ["1", "2"],
                "modal_components": [
                    {
                        "representative_node_id": "1",
                        "node_ids": ["1", "2"],
                    }
                ],
            },
        )
    )

    response_payload = json.loads(response.text)
    assert response_payload["prompt_id"] == "prompt-1"
    assert response_payload["modal_remote_node_ids"] == ["1", "2"]
    assert response_payload["modal_components"] == [
        {
            "representative_node_id": "1",
            "node_ids": ["1", "2"],
        }
    ]

def test_queue_prompt_json_logs_rewritten_modal_diagnostics_on_validation_failure(
    api_intercept_module: Any,
    queue_bridge_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation failures for Modal prompts should log the rewritten dependency graph."""

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def put(self, _item: tuple[Any, ...]) -> None:
            """Fail if an invalid prompt reaches the queue."""
            raise AssertionError("invalid prompt must not be queued")

    class FakePromptServer:
        """Minimal PromptServer double for validation-failure tests."""

        number = 0
        prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeExecutionModule:
        """Minimal execution module that rejects the prompt."""

        SENSITIVE_EXTRA_DATA_KEYS: tuple[str, ...] = ()

        @staticmethod
        async def validate_prompt(
            prompt_id: str,
            prompt: dict[str, Any],
            partial_execution_targets: Any,
        ) -> tuple[bool, dict[str, Any], list[str], dict[str, Any]]:
            """Reject the supplied prompt with a dependency-cycle shaped error."""
            del prompt_id, prompt, partial_execution_targets
            return (
                False,
                {
                    "type": "execution_error",
                    "message": "Dependency cycle detected",
                    "details": "",
                    "extra_info": {},
                },
                [],
                {},
            )

    prompt = {
        "1": {
            "class_type": "ModalUniversalExecutor_a",
            "inputs": {"remote_input_0": ["2", 0]},
        },
        "2": {
            "class_type": "ModalUniversalExecutor_b",
            "inputs": {"remote_input_0": ["1", 0]},
        },
    }

    diagnostic_calls: list[dict[str, Any]] = []

    def record_diagnostics(**kwargs: Any) -> None:
        """Record one rewritten-prompt diagnostics request."""
        diagnostic_calls.append(dict(kwargs))

    monkeypatch.setattr(queue_bridge_module, "_get_execution_module", lambda: FakeExecutionModule)
    monkeypatch.setattr(
        queue_bridge_module,
        "_log_modal_rewritten_prompt_diagnostics",
        record_diagnostics,
    )

    response = asyncio.run(
        queue_bridge_module._queue_prompt_json(
            FakePromptServer(),
            {
                "prompt_id": "prompt-cycle",
                "prompt": prompt,
                "extra_data": {
                    "modal": {
                        "remote_component_ids": ["1", "2"],
                    }
                },
            },
        )
    )

    assert response.status == 400
    assert diagnostic_calls == [
        {
            "prompt_id": "prompt-cycle",
            "prompt": prompt,
            "reason": "comfy_validation_failure",
            "level": api_intercept_module.logging.WARNING,
        }
    ]

def test_remote_execution_configurator_identity_is_preserved(
    api_intercept_module: Any,
) -> None:
    """Queue-time UI events should address the exact serialized configurator node."""
    prompt = {
        "12": {"class_type": "KSampler", "inputs": {}},
        "99": {
            "class_type": "RemoteExecutionConfigurator",
            "inputs": {"configurations.configuration_0": ["20", 0]},
        },
    }

    assert (
        api_intercept_module._remote_execution_configurator_node_id(prompt)
        == "99"
    )
    assert api_intercept_module._remote_execution_configurator_node_id(
        {
            **prompt,
            "100": {
                "class_type": "RemoteExecutionConfigurator",
                "inputs": {},
            },
        }
    ) is None

def test_modal_prompt_rewrite_keeps_event_loop_responsive(
    api_intercept_module: Any,
    prompt_interception_module: Any,
    monkeypatch: Any,
) -> None:
    """Hashing and upload preparation should execute outside the ComfyUI event loop."""
    rewrite_started = threading.Event()
    release_rewrite = threading.Event()

    def blocking_rewrite(**kwargs: Any) -> tuple[dict[str, Any], Any]:
        """Hold one fake rewrite until the async test proves the loop is responsive."""
        rewrite_started.set()
        assert release_rewrite.wait(timeout=1.0)
        return kwargs["prompt"], api_intercept_module.RewriteSummary()

    monkeypatch.setattr(
        prompt_interception_module,
        "rewrite_prompt_for_modal",
        blocking_rewrite,
    )

    async def run_test() -> None:
        """Run the blocking rewrite and an independent event-loop callback together."""
        rewrite_task = asyncio.create_task(
            api_intercept_module.rewrite_prompt_for_modal_async(
                prompt={"1": {"class_type": "RemoteImage", "inputs": {}}},
                workflow=None,
            )
        )
        deadline = asyncio.get_running_loop().time() + 1.0
        while not rewrite_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("background rewrite did not start")
            await asyncio.sleep(0)

        loop_progress: list[str] = []
        asyncio.get_running_loop().call_soon(loop_progress.append, "responsive")
        await asyncio.sleep(0)
        assert loop_progress == ["responsive"]
        assert not rewrite_task.done()

        release_rewrite.set()
        await asyncio.wait_for(rewrite_task, timeout=1.0)

    asyncio.run(run_test())

def test_environment_setup_status_callback_preserves_environment_identity(
    api_intercept_module: Any,
) -> None:
    """Environment setup updates should also retain prompt-wide progress."""
    prompt_updates: list[tuple[str, int | None, int | None]] = []
    environment_updates: list[tuple[str, str, int | None, int | None]] = []
    callback = api_intercept_module._environment_setup_status_callback(
        "vast:48602895",
        lambda message, current, total: prompt_updates.append(
            (message, current, total)
        ),
        lambda environment_id, message, current, total: environment_updates.append(
            (environment_id, message, current, total)
        ),
    )

    assert callback is not None
    callback("Uploading asset", 3, 10)

    assert prompt_updates == [("Uploading asset", 3, 10)]
    assert environment_updates == [
        ("vast:48602895", "Uploading asset", 3, 10)
    ]

def test_remote_environment_assets_are_prepared_in_parallel(
    api_intercept_module: Any,
) -> None:
    """Distinct environments should enter asset preparation concurrently."""
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    active_environment_count = 0
    maximum_active_environment_count = 0
    environment_events: list[tuple[str, str]] = []

    class FakeSyncEngine:
        """Block custom-node setup until both environment workers are active."""

        def __init__(self) -> None:
            """Record the environment-local preparation order."""
            self.calls: list[str] = []

        def preflight_r2_access(self, *, status_callback: Any) -> None:
            """Represent an environment without configured R2 backing."""
            del status_callback
            self.calls.append("r2_preflight")

        def sync_custom_nodes_directory(self, *, status_callback: Any) -> None:
            """Prove the second environment starts before the first can finish."""
            nonlocal active_environment_count, maximum_active_environment_count
            self.calls.append("custom_nodes")
            with state_lock:
                active_environment_count += 1
                maximum_active_environment_count = max(
                    maximum_active_environment_count,
                    active_environment_count,
                )
            status_callback("Uploading custom nodes", None, None)
            try:
                barrier.wait(timeout=2.0)
            finally:
                with state_lock:
                    active_environment_count -= 1

        def create_request_asset_cache(self, values: Any) -> object:
            """Consume the environment-local input plan and return a sentinel cache."""
            tuple(values)
            self.calls.append("asset_plan")
            return object()

        def sync_prompt_inputs(
            self,
            inputs: dict[str, Any],
            *,
            status_callback: Any,
            request_cache: object,
        ) -> tuple[dict[str, Any], list[Any]]:
            """Return one prepared prompt after publishing environment-local status."""
            assert request_cache is not None
            self.calls.append("prompt_assets")
            status_callback("Downloading prompt asset", 1, 1)
            return inputs, []

    components = [
        SimpleNamespace(representative_node_id="a", node_ids=["a"]),
        SimpleNamespace(representative_node_id="b", node_ids=["b"]),
    ]
    assignments = {
        "a": SimpleNamespace(
            environment_id="vast:big:1",
            provider=api_intercept_module.ExecutionProvider.VAST,
        ),
        "b": SimpleNamespace(
            environment_id="lambda",
            provider=api_intercept_module.ExecutionProvider.SSH_DOCKER,
        ),
    }
    vast_engine = FakeSyncEngine()
    ssh_engine = FakeSyncEngine()
    results = api_intercept_module._prepare_remote_environment_assets(
        components=components,
        assignments_by_component_id=assignments,
        sync_engines_by_environment={
            "vast:big:1": vast_engine,
            "lambda": ssh_engine,
        },
        rewritten_prompt={
            "a": {"class_type": "VAELoader", "inputs": {"vae": "a.safetensors"}},
            "b": {"class_type": "VAELoader", "inputs": {"vae": "b.safetensors"}},
        },
        sync_custom_nodes=True,
        status_callback=None,
        environment_status_callback=(
            lambda environment_id, message, _current, _total: environment_events.append(
                (environment_id, message)
            )
        ),
    )

    assert maximum_active_environment_count == 2
    assert list(results) == ["vast:big:1", "lambda"]
    assert results["vast:big:1"].component_prompts["a"]["a"]["inputs"] == {
        "vae": "a.safetensors"
    }
    assert results["lambda"].component_prompts["b"]["b"]["inputs"] == {
        "vae": "b.safetensors"
    }
    assert vast_engine.calls == [
        "r2_preflight",
        "custom_nodes",
        "asset_plan",
        "prompt_assets",
    ]
    assert ssh_engine.calls == vast_engine.calls
    for environment_id in results:
        messages = [
            message
            for event_environment_id, message in environment_events
            if event_environment_id == environment_id
        ]
        expected_completion = (
            "Ready for remote execution"
            if environment_id == "vast:big:1"
            else "Remote assets prepared; SSH runtime starts on dispatch"
        )
        assert messages == [
            "Preparing remote assets",
            "Uploading custom nodes",
            "Downloading prompt asset",
            expected_completion,
        ]

def test_remote_environment_asset_worker_failures_bubble_up(
    api_intercept_module: Any,
) -> None:
    """A failed environment worker must fail queue preparation with its cause."""

    class FailingSyncEngine:
        """Fail before prompt assets are considered prepared."""

        def preflight_r2_access(self, *, status_callback: Any) -> None:
            """Represent an environment without configured R2 backing."""
            del status_callback

        def sync_custom_nodes_directory(self, *, status_callback: Any) -> None:
            """Raise the representative environment-specific setup failure."""
            status_callback("Uploading custom nodes", None, None)
            raise OSError("remote storage is unavailable")

    with pytest.raises(OSError, match="remote storage is unavailable"):
        api_intercept_module._prepare_remote_environment_assets(
            components=[
                SimpleNamespace(representative_node_id="a", node_ids=["a"])
            ],
            assignments_by_component_id={
                "a": SimpleNamespace(
                    environment_id="vast:broken:1",
                    provider=api_intercept_module.ExecutionProvider.VAST,
                )
            },
            sync_engines_by_environment={
                "vast:broken:1": FailingSyncEngine()
            },
            rewritten_prompt={
                "a": {"class_type": "VAELoader", "inputs": {}}
            },
            sync_custom_nodes=True,
            status_callback=None,
            environment_status_callback=lambda *_args: None,
        )

