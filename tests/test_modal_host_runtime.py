"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_remote_modal_call_worker_count_is_bounded_independently_of_local_cpus(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Local CPU count should not determine expensive remote GPU concurrency."""
    monkeypatch.setattr(remote_modal_app_module.os, "cpu_count", lambda: 8)
    remote_modal_app_module.get_settings.cache_clear()

    assert remote_modal_app_module._remote_modal_call_worker_count() == 4

def test_remote_modal_call_worker_count_honors_explicit_inflight_limit(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Operators should be able to choose the local remote-call budget explicitly."""
    monkeypatch.setattr(remote_modal_app_module.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("COMFY_MODAL_MAX_INFLIGHT_CALLS", "12")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        assert remote_modal_app_module._remote_modal_call_worker_count() == 12
    finally:
        remote_modal_app_module.get_settings.cache_clear()

def test_remote_modal_does_not_classify_remote_execution_error_as_missing_app(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
) -> None:
    """Remote execution tracebacks should surface directly instead of triggering redeploy."""
    remote_error = RuntimeError(
        "Could not deserialize remote exception due to local error:\n"
        "Here is the remote traceback:\n"
        "comfyui_modal_sync_cloud.RemoteSubgraphExecutionError: "
        "Object of type CLIP is not JSON serializable\n"
        "Lookup failed for Cls 'RemoteEngine' from the 'comfy-modal-sync' app: "
        "App 'comfy-modal-sync' not found in environment 'main'."
    )
    missing_lookup_error = RuntimeError(
        "Lookup failed for Cls 'RemoteEngine' from the 'comfy-modal-sync' app: "
        "App 'comfy-modal-sync' not found in environment 'main'."
    )

    assert not modal_deployment_module._is_missing_modal_deployment_error(remote_error)
    assert modal_deployment_module._is_missing_modal_deployment_error(missing_lookup_error)

def test_remote_modal_stops_app_through_experimental_sdk(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal 1.4 should use its non-interactive experimental stop API."""
    stop_calls: list[tuple[str, str | None]] = []

    def fake_stop_app(app_name: str, *, environment_name: str | None = None) -> None:
        """Record one SDK stop request."""
        stop_calls.append((app_name, environment_name))

    fake_experimental_namespace = types.SimpleNamespace(stop_app=fake_stop_app)
    original_import_module = modal_deployment_module.importlib.import_module

    def fake_import_module(module_name: str) -> Any:
        """Return the Modal experimental namespace and delegate other imports."""
        if module_name == "modal.experimental":
            return fake_experimental_namespace
        return original_import_module(module_name)

    monkeypatch.setattr(
        modal_deployment_module,
        "modal",
        types.SimpleNamespace(exception=types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        modal_deployment_module.importlib,
        "import_module",
        fake_import_module,
    )
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")

    stopped = modal_deployment_module._stop_modal_app_via_sdk("comfy-modal-sync-instance")

    assert stopped is True
    assert stop_calls == [("comfy-modal-sync-instance", "main")]

def test_remote_modal_cli_stop_is_noninteractive_and_environment_scoped(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """The CLI fallback should never block waiting for confirmation."""
    observed_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        """Record the CLI command and return success."""
        observed_calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(modal_deployment_module.shutil, "which", lambda _: "/venv/bin/modal")
    monkeypatch.setattr(modal_deployment_module.subprocess, "run", fake_run)
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")

    stopped = modal_deployment_module._stop_modal_app_via_cli("comfy-modal-sync-instance")

    assert stopped is True
    assert observed_calls == [
        (
            [
                "/venv/bin/modal",
                "app",
                "stop",
                "comfy-modal-sync-instance",
                "--yes",
                "--env",
                "main",
            ],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": remote_modal_app_module._MODAL_APP_STOP_TIMEOUT_SECONDS,
            },
        )
    ]

def test_remote_modal_cli_stop_timeout_returns_controlled_failure(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """A genuine CLI timeout should not leak TimeoutExpired into prompt execution."""
    command = ["/venv/bin/modal", "app", "stop", "comfy-modal-sync-instance", "--yes"]

    def fake_run(*_: Any, **__: Any) -> Any:
        """Simulate an unresponsive Modal CLI."""
        raise subprocess.TimeoutExpired(
            command,
            remote_modal_app_module._MODAL_APP_STOP_TIMEOUT_SECONDS,
        )

    monkeypatch.setattr(modal_deployment_module.shutil, "which", lambda _: "/venv/bin/modal")
    monkeypatch.setattr(modal_deployment_module.subprocess, "run", fake_run)
    monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)

    with caplog.at_level(logging.WARNING):
        stopped = modal_deployment_module._stop_modal_app_via_cli(
            "comfy-modal-sync-instance"
        )

    assert stopped is False
    assert "Modal CLI app stop timed out" in caplog.text

def test_completed_modal_billing_interval_uses_hourly_resolution_and_buffer(
    modal_billing_module: Any,
) -> None:
    """Billing should use the latest full UTC hour after a collection buffer."""
    interval_start, interval_end, next_refresh_at = (
        modal_billing_module._completed_modal_billing_interval(
            datetime(2026, 8, 19, 8, 5, tzinfo=timezone.utc)
        )
    )

    assert interval_start == datetime(2026, 8, 19, 6, 0, tzinfo=timezone.utc)
    assert interval_end == datetime(2026, 8, 19, 7, 0, tzinfo=timezone.utc)
    assert next_refresh_at == datetime(2026, 8, 19, 8, 10, tzinfo=timezone.utc)

def test_get_hourly_modal_app_billing_sums_historical_ids_in_default_environment(
    modal_billing_module: Any,
    monkeypatch: Any,
) -> None:
    """Redeployments within one implicit environment should sum every app ID."""
    settings = modal_billing_module.get_settings()
    selected_settings = modal_billing_module.settings_for_modal_gpu(settings, "L4")
    app_name = modal_billing_module.modal_deployment_app_name(selected_settings)
    interval_start = datetime(2026, 8, 19, 7, 0, tzinfo=timezone.utc)
    original_import_module = modal_billing_module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        """Supply an implicit environment with two historical app identities."""
        if name == "modal._object":
            return types.SimpleNamespace(_get_environment_name=lambda _environment: None)
        if name == "modal.environments":
            return types.SimpleNamespace(ensure_env=lambda _environment: None)
        if name == "modal.billing":
            return types.SimpleNamespace(
                workspace_billing_report=lambda **_kwargs: [
                    {
                        "object_id": "ap-workspace-default-old",
                        "description": app_name,
                        "environment_name": "main",
                        "interval_start": interval_start,
                        "cost": Decimal("0.17"),
                    },
                    {
                        "object_id": "ap-workspace-default-new",
                        "description": app_name,
                        "environment_name": "main",
                        "interval_start": interval_start,
                        "cost": Decimal("0.25"),
                    },
                ]
            )
        if name == "modal.exception":
            return types.SimpleNamespace(Error=RuntimeError)
        return original_import_module(name)

    monkeypatch.setattr(modal_billing_module, "modal", object())
    monkeypatch.setattr(
        modal_billing_module.importlib,
        "import_module",
        fake_import_module,
    )
    modal_billing_module._MODAL_HOURLY_BILLING_CACHE.clear()
    modal_billing_module._MODAL_HOURLY_BILLING_ERROR_CACHE.clear()

    status = asyncio.run(
        modal_billing_module.get_hourly_modal_app_billing(
            "L4",
            settings,
            now=datetime(2026, 8, 19, 8, 15, tzinfo=timezone.utc),
        )
    )

    assert status.app_id is None
    assert status.environment_name == "main"
    assert status.app_cost_usd_before_credits == Decimal("0.42")

def test_modal_hourly_billing_rejects_multiple_implicit_environments(
    modal_billing_module: Any,
) -> None:
    """An implicit environment should reject only genuinely distinct environments."""
    interval_start = datetime(2026, 8, 19, 7, 0, tzinfo=timezone.utc)
    rows = [
        {
            "object_id": "ap-main",
            "description": "shared-app-name",
            "environment_name": "main",
            "interval_start": interval_start,
            "cost": Decimal("0.17"),
        },
        {
            "object_id": "ap-dev",
            "description": "shared-app-name",
            "environment_name": "dev",
            "interval_start": interval_start,
            "cost": Decimal("0.25"),
        },
    ]

    with pytest.raises(
        modal_billing_module.ModalBillingStatusError,
        match="multiple environments",
    ):
        modal_billing_module._matching_modal_hourly_billing_rows(
            rows,
            app_name="shared-app-name",
            environment_name=None,
            interval_start=interval_start,
        )

def test_get_hourly_modal_app_billing_reports_zero_for_no_usage(
    modal_billing_module: Any,
    monkeypatch: Any,
) -> None:
    """An absent app row should render as zero cost for that completed hour."""
    original_import_module = modal_billing_module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        """Supply an empty public Modal billing report."""
        if name == "modal._object":
            return types.SimpleNamespace(_get_environment_name=lambda _environment: "main")
        if name == "modal.environments":
            return types.SimpleNamespace(ensure_env=lambda environment: environment or "main")
        if name == "modal.billing":
            return types.SimpleNamespace(workspace_billing_report=lambda **_kwargs: [])
        if name == "modal.exception":
            return types.SimpleNamespace(Error=RuntimeError)
        return original_import_module(name)

    monkeypatch.setattr(modal_billing_module, "modal", object())
    monkeypatch.setattr(
        modal_billing_module.importlib,
        "import_module",
        fake_import_module,
    )
    modal_billing_module._MODAL_HOURLY_BILLING_CACHE.clear()
    modal_billing_module._MODAL_HOURLY_BILLING_ERROR_CACHE.clear()

    status = asyncio.run(
        modal_billing_module.get_hourly_modal_app_billing(
            "L4",
            now=datetime(2026, 8, 19, 8, 15, tzinfo=timezone.utc),
        )
    )

    assert status.app_cost_usd_before_credits == Decimal("0")
    assert status.has_usage is False
    assert status.as_dict()["resolution"] == "hour"
    assert status.as_dict()["collection_buffer_seconds"] == 600

def test_invoke_remote_engine_propagates_local_interrupt_to_modal(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
    monkeypatch: Any,
) -> None:
    """The local proxy should propagate ComfyUI interrupts to the remote Modal call."""

    class FakeInterrupt(Exception):
        """Stand-in for ComfyUI's InterruptProcessingException."""

    observed_cancellation_events: list[threading.Event] = []
    interrupt_checks = iter([False, True, True])

    def fake_blocking_invoke(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: threading.Event | None = None,
    ) -> bytes:
        assert cancellation_event is not None
        observed_cancellation_events.append(cancellation_event)
        while not cancellation_event.is_set():
            time.sleep(0.01)
        raise RuntimeError("remote interrupted")

    def fake_local_processing_interrupted() -> bool:
        return next(interrupt_checks, True)

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(modal_interrupts_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_invoke_modal_payload_blocking", fake_blocking_invoke)
    monkeypatch.setattr(
        modal_interrupts_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_raise_local_interrupt",
        lambda: (_ for _ in ()).throw(FakeInterrupt()),
    )

    with pytest.raises(FakeInterrupt):
        remote_modal_app_module.invoke_remote_engine(
            {"component_id": "component-1", "payload_kind": "subgraph"},
            b"{}",
        )

    assert len(observed_cancellation_events) == 1
    assert observed_cancellation_events[0].is_set()

def test_invoke_remote_engine_async_propagates_local_interrupt_to_modal(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
    monkeypatch: Any,
) -> None:
    """The async local proxy should also propagate ComfyUI interrupts to the remote Modal call."""

    class FakeInterrupt(Exception):
        """Stand-in for ComfyUI's InterruptProcessingException."""

    observed_cancellation_events: list[threading.Event] = []
    interrupt_checks = iter([False, True, True])

    def fake_blocking_invoke(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: threading.Event | None = None,
    ) -> bytes:
        assert cancellation_event is not None
        observed_cancellation_events.append(cancellation_event)
        while not cancellation_event.is_set():
            time.sleep(0.01)
        raise RuntimeError("remote interrupted")

    def fake_local_processing_interrupted() -> bool:
        return next(interrupt_checks, True)

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(modal_interrupts_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_invoke_modal_payload_blocking", fake_blocking_invoke)
    monkeypatch.setattr(
        modal_interrupts_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_raise_local_interrupt",
        lambda: (_ for _ in ()).throw(FakeInterrupt()),
    )

    with pytest.raises(FakeInterrupt):
        asyncio.run(
            remote_modal_app_module.invoke_remote_engine_async(
                {"component_id": "component-1", "payload_kind": "subgraph"},
                b"{}",
            )
        )

    assert len(observed_cancellation_events) == 1
    assert observed_cancellation_events[0].is_set()

def test_invoke_remote_engine_payload_refuses_dispatch_after_prestart_cancel(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A cancellation observed during deploy should prevent the later remote payload call."""
    cancellation_event = threading.Event()
    cancellation_event.set()
    remote_interrupt_payloads: list[dict[str, Any]] = []
    module_name = modal_deployment_module._MODAL_CLOUD_MODULE_NAME
    original_cloud_module = sys.modules.pop(module_name, None)

    class FakeExecuteMethod:
        """Remote method double that must not be called after cancellation."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Fail if the cancelled payload reaches Modal execution."""
            del payload, kwargs_payload
            raise AssertionError("cancelled payload should not dispatch")

    monkeypatch.setattr(
        remote_modal_app_module,
        "_request_remote_interrupt",
        lambda payload: remote_interrupt_payloads.append(dict(payload)) or True,
    )

    try:
        with pytest.raises(remote_modal_app_module.ModalRemoteInvocationError, match="cancelled before"):
            remote_modal_app_module._invoke_remote_engine_payload(
                types.SimpleNamespace(execute_payload=FakeExecuteMethod()),
                {"prompt_id": "prompt-1", "component_id": "component-1"},
                b"{}",
                cancellation_event,
            )
    finally:
        sys.modules.pop(module_name, None)
        if original_cloud_module is not None:
            sys.modules[module_name] = original_cloud_module

    assert [payload["component_id"] for payload in remote_interrupt_payloads] == ["component-1"]

def test_remote_modal_interrupt_callback_writes_shared_control_flag(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
    monkeypatch: Any,
) -> None:
    """The local interrupt callback should write to the shared Modal Dict control store."""

    class FakeInterruptStore:
        """Simple Modal Dict double that records written interruption flags."""

        def __init__(self) -> None:
            """Initialize captured writes."""
            self.put_calls: list[tuple[str, Any]] = []

        def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
            """Record one interrupt flag write."""
            self.put_calls.append((key, value))
            return True

    interrupt_store = FakeInterruptStore()

    class FakeModalDict:
        """Minimal modal.Dict shim that returns the fake interrupt store."""

        @staticmethod
        def from_name(
            name: str,
            *,
            environment_name: str | None = None,
            create_if_missing: bool = False,
            client: Any | None = None,
        ) -> FakeInterruptStore:
            return interrupt_store

    monkeypatch.setattr(
        modal_interrupts_module,
        "modal",
        types.SimpleNamespace(Dict=FakeModalDict),
    )
    monkeypatch.setenv("COMFY_MODAL_INTERRUPT_DICT_NAME", "shared-interrupts")
    remote_modal_app_module.get_settings.cache_clear()
    modal_interrupts_module._MODAL_INTERRUPT_DICTS.clear()
    try:
        callback = remote_modal_app_module._build_remote_interrupt_callback(
            object(),
            {"prompt_id": "prompt-1", "component_id": "component-2"},
        )
        assert callback is not None
        callback()
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_interrupts_module._MODAL_INTERRUPT_DICTS.clear()

    assert len(interrupt_store.put_calls) == 1
    interrupt_key, interrupt_value = interrupt_store.put_calls[0]
    assert interrupt_key == "prompt-1:component-2"
    assert isinstance(interrupt_value["requested_at"], float)

def test_request_remote_modal_prompt_interrupt_cancels_active_components(
    remote_modal_app_module: Any,
) -> None:
    """Prompt-level cancellation should interrupt every active remote component."""
    cancellation_event = threading.Event()
    interrupt_calls: list[str] = []

    with remote_modal_app_module._registered_active_remote_invocation(
        {"prompt_id": "prompt-1", "component_id": "component-1"},
        cancellation_event,
        lambda: interrupt_calls.append("component-1"),
    ):
        assert remote_modal_app_module.active_remote_modal_prompt_ids() == {"prompt-1"}
        assert remote_modal_app_module.request_remote_modal_prompt_interrupt("prompt-1") is True

    assert cancellation_event.is_set()
    assert interrupt_calls == ["component-1"]
    assert remote_modal_app_module.active_remote_modal_prompt_ids() == set()

def test_invoke_remote_engine_payload_releases_local_prompt_after_cancel_grace(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
    monkeypatch: Any,
) -> None:
    """The local proxy should not wait forever for a cancelled Modal call to unwind."""
    cancellation_event = threading.Event()
    remote_release_event = threading.Event()
    remote_finished_event = threading.Event()
    interrupt_calls: list[str] = []
    interrupt_checks = iter([False, True, True, True])

    def fake_local_processing_interrupted() -> bool:
        """Report a local interrupt after the first poll interval."""
        return next(interrupt_checks, True)

    def fake_interrupt_remote_call() -> None:
        """Record the propagated remote interrupt."""
        interrupt_calls.append("interrupt")

    def never_returns() -> bytes:
        """Simulate a Modal call stuck in remote work after cancellation."""
        remote_release_event.wait()
        remote_finished_event.set()
        return b"late-response"

    monkeypatch.setenv("COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS", "0.05")
    remote_modal_app_module.get_settings.cache_clear()
    monkeypatch.setattr(
        modal_interrupts_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )

    try:
        with pytest.raises(remote_modal_app_module.ModalRemoteInvocationError):
            remote_modal_app_module._invoke_remote_call_with_interrupts(
                payload={"prompt_id": "prompt-1", "component_id": "component-1"},
                invoke_remote_call=never_returns,
                interrupt_remote_call=fake_interrupt_remote_call,
                cancellation_event=cancellation_event,
            )
    finally:
        remote_release_event.set()
        assert remote_finished_event.wait(1.0)
        remote_modal_app_module.get_settings.cache_clear()

    assert cancellation_event.is_set()
    assert interrupt_calls == ["interrupt"]
