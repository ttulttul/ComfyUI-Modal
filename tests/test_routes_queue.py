"""Tests for the routes queue boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403


def test_attributed_queue_error_targets_subrosa_configuration_node(
    routes_queue_module: Any,
    subrosa_login_module: Any,
) -> None:
    """Queue credential failures should use ComfyUI's node-error shape."""
    emitted: list[dict[str, Any]] = []
    context = SimpleNamespace(emit_status=lambda **kwargs: emitted.append(kwargs))
    state = routes_queue_module._QueueRequestState(
        json_data={
            "prompt": {
                "42": {
                    "class_type": "SubrosaRemoteConfiguration",
                    "inputs": {},
                }
            }
        },
        remote_node_ids=["7"],
        client_id="client-1",
        prompt_id="prompt-1",
    )
    error = subrosa_login_module.SubrosaConfigurationValidationError(
        "42",
        subrosa_login_module.SubrosaLoginRequiredError(),
    )

    response = routes_queue_module._queue_error_response(
        object(),
        context,
        state,
        error,
        phase="error",
        status=400,
    )
    payload = json.loads(response.text)

    assert payload["error"]["type"] == "subrosa_login_required"
    assert payload["error"]["extra_info"]["node_id"] == "42"
    assert payload["node_errors"] == {
        "42": {
            "errors": [
                {
                    "type": "subrosa_login_required",
                    "message": "Subrosa Configuration failed validation",
                    "details": subrosa_login_module.SUBROSA_LOGIN_REQUIRED_MESSAGE,
                    "extra_info": {},
                }
            ],
            "dependent_outputs": [],
            "class_type": "SubrosaRemoteConfiguration",
        }
    }
    assert emitted[0]["failed_node_id"] == "42"


def test_invalid_subrosa_token_stops_before_rewrite_and_comfyui_execution(
    routes_queue_module: Any,
    subrosa_login_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue boundary must reject credentials before handing off the prompt."""

    class FakeRequest:
        """Return one remote prompt with a connected Subrosa configuration."""

        async def json(self) -> dict[str, Any]:
            """Return the serialized queue body."""
            return {
                "prompt_id": "prompt-1",
                "client_id": "client-1",
                "prompt": {
                    "7": {"class_type": "NextSeeds", "inputs": {}},
                    "42": {
                        "class_type": "SubrosaRemoteConfiguration",
                        "inputs": {
                            "configuration_name": "Subrosa staging",
                            "relay_url": "wss://staging.subrosa.red",
                            "pool": "RTX-PRO-6000",
                            "credential_id": "subrosa-default",
                        },
                    },
                    "99": {
                        "class_type": "RemoteExecutionConfigurator",
                        "inputs": {"configuration_0": ["42", 0]},
                    },
                },
                "extra_data": {},
            }

    async def reject_preflight(_configuration_set: Any) -> None:
        """Model one invalid saved extension token."""
        raise subrosa_login_module.SubrosaConfigurationValidationError(
            "42",
            subrosa_login_module.SubrosaLoginRequiredError(),
        )

    async def fail_queue(*_args: Any, **_kwargs: Any) -> None:
        """Fail if the prompt reaches ComfyUI's execution queue."""
        raise AssertionError("invalid credentials must stop before ComfyUI execution")

    async def fail_rewrite(*_args: Any, **_kwargs: Any) -> None:
        """Fail if credential validation occurs after prompt rewriting."""
        raise AssertionError("invalid credentials must stop before prompt rewriting")

    emitted: list[dict[str, Any]] = []
    context = SimpleNamespace(
        settings=object(),
        configurator_node_id=lambda _prompt: "99",
        emit_status=lambda **kwargs: emitted.append(kwargs),
        rewrite_prompt=fail_rewrite,
    )
    monkeypatch.setattr(
        routes_queue_module,
        "requested_remote_node_ids",
        lambda **_kwargs: {"7"},
    )
    monkeypatch.setattr(
        routes_queue_module,
        "preflight_subrosa_configurations",
        reject_preflight,
    )
    monkeypatch.setattr(routes_queue_module, "_queue_prompt_json", fail_queue)

    response = asyncio.run(
        routes_queue_module._handle_modal_queue_prompt(
            FakeRequest(),
            object(),
            context,
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert list(payload["node_errors"]) == ["42"]
    assert emitted[0]["failed_node_id"] == "42"

def test_queue_prompt_route_does_not_warm_modal_at_queue_time(
    api_intercept_module: Any,
    prompt_interception_module: Any,
    queue_bridge_module: Any,
    remote_modal_app_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Accepting a queued prompt should not launch Modal warmup containers."""

    class FakeRoutes:
        """Capture aiohttp route registrations."""

        def __init__(self) -> None:
            """Initialize the route handler map."""
            self.handlers: dict[str, Any] = {}

        def post(self, path: str) -> Any:
            """Return a decorator that records one POST handler."""

            def register(handler: Any) -> Any:
                """Store the decorated handler unchanged."""
                self.handlers[path] = handler
                return handler

            return register

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def __init__(self) -> None:
            """Initialize captured queue items."""
            self.items: list[tuple[Any, ...]] = []

        def put(self, item: tuple[Any, ...]) -> None:
            """Record one queued prompt item."""
            self.items.append(item)

    class FakePromptServer:
        """Minimal PromptServer double with route registration."""

        def __init__(self) -> None:
            """Initialize routing and queue state."""
            self.number = 0
            self.routes = FakeRoutes()
            self.prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeRequest:
        """Minimal aiohttp request double."""

        async def json(self) -> dict[str, Any]:
            """Return one Modal-marked prompt request."""
            return {
                "prompt_id": "prompt-queue-warmup",
                "prompt": {"1": {"class_type": "RemoteImage", "inputs": {}}},
                "extra_data": {
                    "extra_pnginfo": {
                        "workflow": {
                            "extra": {"comfy_modal": {"gpu": "B300"}},
                            "nodes": [
                                {
                                    "id": 1,
                                    "type": "RemoteImage",
                                    "properties": {"is_modal_remote": True},
                                }
                            ]
                        }
                    }
                },
            }

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

    def fail_queue_time_warmup(*_args: Any, **_kwargs: Any) -> int:
        """Fail the test if queue handling tries to launch proactive warmup."""
        raise AssertionError("queue route must not schedule Modal warmup")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="remote",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )
    prompt_server = FakePromptServer()
    summary = api_intercept_module.RewriteSummary(
        remote_node_ids=["1"],
        remote_component_ids=["1"],
        component_node_ids_by_representative={"1": ["1"]},
        sandwiched_local_node_ids=["4"],
        component_execution_stages=[["1"]],
        estimated_max_parallel_requests=1,
        max_parallel_requests_upper_bound=1,
        execution_assignments_by_representative={
            "1": api_intercept_module.ExecutionAssignment(
                environment_id="modal:B300",
                provider=api_intercept_module.ExecutionProvider.MODAL,
                predicted_cost_usd=0.01,
                predicted_completion_seconds=10.0,
            )
        },
    )
    observed_rewrite_settings: list[Any] = []

    def capture_rewrite_settings(**kwargs: Any) -> tuple[dict[str, Any], Any]:
        """Capture the workflow-derived settings passed into prompt rewriting."""
        observed_rewrite_settings.append(kwargs["settings"])
        return kwargs["prompt"], summary

    monkeypatch.setattr(api_intercept_module, "_ROUTE_REGISTERED", False)
    monkeypatch.setattr(
        api_intercept_module,
        "_get_server_module",
        lambda: SimpleNamespace(PromptServer=SimpleNamespace(instance=prompt_server)),
    )
    monkeypatch.setattr(queue_bridge_module, "_get_execution_module", lambda: FakeExecutionModule)
    monkeypatch.setattr(api_intercept_module, "_emit_modal_status", lambda **_kwargs: None)
    monkeypatch.setattr(
        prompt_interception_module,
        "rewrite_prompt_for_modal",
        capture_rewrite_settings,
    )
    monkeypatch.setattr(remote_modal_app_module, "ensure_remote_warm_capacity", fail_queue_time_warmup)

    api_intercept_module.setup_modal_queue_route(
        prompt_server=prompt_server,
        sync_engine=object(),
        settings=settings,
    )
    response = asyncio.run(prompt_server.routes.handlers["/modal/queue_prompt"](FakeRequest()))

    response_payload = json.loads(response.text)
    assert response_payload["prompt_id"] == "prompt-queue-warmup"
    assert response_payload["modal_gpu"] == "B300"
    assert response_payload["modal_remote_node_ids"] == ["1"]
    assert response_payload["modal_sandwiched_local_node_ids"] == ["4"]
    assert observed_rewrite_settings[0].modal_gpu == "B300"
    queued_extra_data = prompt_server.prompt_queue.items[0][3]
    assert queued_extra_data["modal"]["gpu"] == "B300"
    assert queued_extra_data["extra_pnginfo"][
        api_intercept_module.MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY
    ] == "prompt-queue-warmup"
    assert len(prompt_server.prompt_queue.items) == 1

def test_queue_prompt_route_without_remote_nodes_skips_modal_status_and_rewrite(
    api_intercept_module: Any,
    prompt_interception_module: Any,
    queue_bridge_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Prompts with no Modal-enabled workflow nodes should queue without Modal UI setup."""

    class FakeRoutes:
        """Capture aiohttp route registrations."""

        def __init__(self) -> None:
            """Initialize the route handler map."""
            self.handlers: dict[str, Any] = {}

        def post(self, path: str) -> Any:
            """Return a decorator that records one POST handler."""

            def register(handler: Any) -> Any:
                """Store the decorated handler unchanged."""
                self.handlers[path] = handler
                return handler

            return register

    class FakePromptQueue:
        """Minimal prompt queue sink."""

        def __init__(self) -> None:
            """Initialize captured queue items."""
            self.items: list[tuple[Any, ...]] = []

        def put(self, item: tuple[Any, ...]) -> None:
            """Record one queued prompt item."""
            self.items.append(item)

    class FakePromptServer:
        """Minimal PromptServer double with route registration."""

        def __init__(self) -> None:
            """Initialize routing and queue state."""
            self.number = 0
            self.routes = FakeRoutes()
            self.prompt_queue = FakePromptQueue()

        def trigger_on_prompt(self, json_data: dict[str, Any]) -> dict[str, Any]:
            """Return the prompt unchanged."""
            return json_data

    class FakeRequest:
        """Minimal aiohttp request double."""

        async def json(self) -> dict[str, Any]:
            """Return one ordinary prompt request."""
            return {
                "prompt_id": "prompt-no-modal",
                "prompt": {"1": {"class_type": "LocalImage", "inputs": {}}},
                "extra_data": {"extra_pnginfo": {"workflow": {"nodes": []}}},
            }

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

    def fail_modal_status(*_args: Any, **_kwargs: Any) -> None:
        """Fail if the no-remote fast path emits Modal UI state."""
        raise AssertionError("no-remote prompts must not emit Modal status")

    def fail_rewrite(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
        """Fail if the no-remote fast path enters Modal prompt rewriting."""
        raise AssertionError("no-remote prompts must not be rewritten for Modal")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="remote",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )
    prompt_server = FakePromptServer()
    monkeypatch.setattr(api_intercept_module, "_ROUTE_REGISTERED", False)
    monkeypatch.setattr(
        api_intercept_module,
        "_get_server_module",
        lambda: SimpleNamespace(PromptServer=SimpleNamespace(instance=prompt_server)),
    )
    monkeypatch.setattr(queue_bridge_module, "_get_execution_module", lambda: FakeExecutionModule)
    monkeypatch.setattr(api_intercept_module, "_emit_modal_status", fail_modal_status)
    monkeypatch.setattr(
        prompt_interception_module,
        "rewrite_prompt_for_modal",
        fail_rewrite,
    )

    api_intercept_module.setup_modal_queue_route(
        prompt_server=prompt_server,
        sync_engine=object(),
        settings=settings,
    )
    response = asyncio.run(prompt_server.routes.handlers["/modal/queue_prompt"](FakeRequest()))

    response_payload = json.loads(response.text)
    assert response_payload["prompt_id"] == "prompt-no-modal"
    assert "modal_remote_node_ids" not in response_payload
    assert len(prompt_server.prompt_queue.items) == 1

def test_cancel_preparation_route_is_queue_route_sibling(
    api_intercept_module: Any,
) -> None:
    """Queue-time cancellation should use a stable sibling route."""
    assert api_intercept_module._cancel_preparation_route_path(
        "/modal/queue_prompt"
    ) == "/modal/cancel_preparation"
    assert api_intercept_module._cancel_preparation_route_path(
        "/custom/modal"
    ) == "/custom/modal/cancel_preparation"
    assert api_intercept_module._container_status_route_path("/custom/modal") == (
        "/custom/modal/container_status"
    )
