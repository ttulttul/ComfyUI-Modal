"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_post_deploy_seed_registers_a_joinable_profile_future(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """Automatic deployment should seed the current important profile before dispatch."""
    submitted_tasks: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        """Capture the automatic seed without invoking Modal."""

        def submit(self, function: Any, *args: Any) -> Future[Any]:
            """Record one seed job and return its pending future."""
            submitted_tasks.append((function, args))
            return Future()

    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(modal_warmup_module, "modal", object())
    monkeypatch.setattr(
        modal_warmup_module,
        "_REMOTE_MODAL_WARMUP_EXECUTOR",
        FakeExecutor(),
    )
    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "false")
    remote_modal_app_module.get_settings.cache_clear()
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()
    try:
        scheduled = modal_warmup_module._schedule_post_deploy_runtime_seed(
            {
                "prompt_id": "prompt-deploy",
                "component_id": "llm-component",
                "remote_worker_affinity_group": "llm",
                "subgraph_prompt": {
                    "263": {
                        "class_type": "ModalLLM",
                        "inputs": {"model_profile": "qwen-test"},
                    }
                },
            }
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert scheduled is True
    assert len(submitted_tasks) == 1
    scheduled_function, scheduled_args = submitted_tasks[0]
    assert scheduled_function is modal_warmup_module._run_speculative_affinity_prewarm
    assert scheduled_args[-1] == "post_deploy_runtime_seed"
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        state = modal_warmup_module._PROMPT_WARMUP_STATES["prompt-deploy"]
        assert len(state.speculative_affinity_futures) == 1
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()

def test_model_stager_image_environment_preserves_deployment_identity(
    modal_cloud_module: Any,
) -> None:
    """The CPU protocol helper must report the exact deployed GPU runtime identity."""
    settings = types.SimpleNamespace(
        app_name="comfy-modal-sync-instance",
        modal_gpu="RTX-PRO-6000",
        remote_storage_root="/storage",
    )

    image_environment = modal_cloud_module._model_stager_image_environment(
        settings,
        "fingerprint-2",
    )

    assert image_environment == {
        "COMFY_MODAL_APP_NAME": "comfy-modal-sync-instance",
        "COMFY_MODAL_GPU": "RTX-PRO-6000",
        "COMFY_MODAL_REMOTE_STORAGE_ROOT": "/storage",
        "COMFY_MODAL_RUNTIME_FINGERPRINT": "fingerprint-2",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }

def test_remote_modal_auto_deploys_missing_app_by_default(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_deployment_module: Any,
    local_ui_events_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote mode should auto-deploy the stable Modal app on first lookup failure."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []
    stage_calls: list[list[str]] = []
    status_events: list[dict[str, Any]] = []

    class FakeExecuteMethod:
        """Minimal Modal method handle that records remote calls."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Return deterministic remote bytes."""
            return b"remote-response"

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance."""

        execute_payload = FakeExecuteMethod()
        runtime_version = types.SimpleNamespace(
            remote=lambda: _current_remote_runtime_payload(remote_modal_app_module)
        )

    class FakeModelStager:
        """Minimal CPU model stager used by the first deployed LLM request."""

        stage_profiles = types.SimpleNamespace(
            remote=lambda profile_ids: stage_calls.append(list(profile_ids))
            or [{"profile_id": profile_id} for profile_id in profile_ids]
        )

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(self, *, name: str | None = None, environment_name: str | None = None, **_: Any) -> "FakeApp":
            """Record the deploy request and mark the deployment available."""
            deploy_calls.append((name, environment_name))
            FakeModal.deployed = True
            return self

    class FakeModal:
        """Minimal modal SDK double with deployed lookup failure types."""

        deployed = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a deployed class after the first auto-deploy."""
                if not FakeModal.deployed:
                    raise FakeLookupError("not deployed")
                if class_name == "ModelStager":
                    return lambda: FakeModelStager()
                return lambda **kwargs: FakeRemoteEngine()

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(
        local_ui_events_module,
        "_emit_local_modal_status",
        lambda **event: status_events.append(event),
    )
    monkeypatch.setattr(
        modal_deployment_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    with modal_llm_profile_staging_module._STAGED_LLM_PROFILES_LOCK:
        modal_llm_profile_staging_module._STAGED_LLM_PROFILES.clear()
    try:
        response = remote_modal_app_module._invoke_modal_payload_blocking(
            {
                "component_id": "component-1",
                "component_node_ids": ["node-1"],
                "prompt_id": "prompt-1",
                "extra_data": {"client_id": "client-1"},
                "subgraph_prompt": {
                    "node-1": {
                        "class_type": "ModalLLM",
                        "inputs": {"model_profile": "smolvlm2-2.2b-instruct"},
                    }
                },
            },
            b"{}",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
        with modal_llm_profile_staging_module._STAGED_LLM_PROFILES_LOCK:
            modal_llm_profile_staging_module._STAGED_LLM_PROFILES.clear()

    assert response == b"remote-response"
    assert deploy_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, None)]
    assert stage_calls == [["smolvlm2-2.2b-instruct"]]
    assert [
        (event["phase"], event["status_message"])
        for event in status_events
        ] == [
            ("setup", "Rebuilding Modal app"),
            (
                "llm_staging",
                "Preparing LLM model snapshots on CPU; no GPU is allocated yet",
            ),
            (
                "llm_staged",
                "LLM staging complete (0.0 GiB downloaded); starting GPU worker",
            ),
            ("starting", "Starting remote component on Modal"),
        ]

def test_remote_modal_does_not_redeploy_after_remote_execution_error(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """The deployed invocation path should not auto-deploy after a remote runtime failure."""

    class FakeLookupError(Exception):
        """Stand-in for Modal errors that share the lookup error type."""

    class FakeExecuteMethod:
        """Minimal Modal method handle that raises a remote execution failure."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Raise the wrapped remote traceback from the deployed call."""
            del payload, kwargs_payload
            raise FakeLookupError(
                "Could not deserialize remote exception due to local error:\n"
                "Here is the remote traceback:\n"
                "comfyui_modal_sync_cloud.RemoteSubgraphExecutionError: "
                "Object of type CLIP is not JSON serializable\n"
                "App 'comfy-modal-sync' not found in environment 'main'."
            )

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance."""

        execute_payload = FakeExecuteMethod()
        runtime_version = types.SimpleNamespace(
            remote=lambda: _current_remote_runtime_payload(remote_modal_app_module)
        )

    class FakeModal:
        """Minimal Modal SDK double with shared lookup/runtime error types."""

        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a deployed class successfully."""
                del app_name, class_name
                return lambda **kwargs: FakeRemoteEngine()

    def fail_load_cloud_module() -> Any:
        """Fail if runtime errors attempt auto-deploy."""
        raise AssertionError("remote execution errors must not trigger auto-deploy")

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "_load_modal_cloud_module", fail_load_cloud_module)
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        with pytest.raises(FakeLookupError, match="Object of type CLIP"):
            remote_modal_app_module._invoke_modal_payload_blocking(
                {"component_id": "component-1"},
                b"{}",
            )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

def test_remote_modal_redeploys_when_cached_app_was_deleted(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A stale in-process auto-deploy cache should not block redeploy after app deletion."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []

    class FakeExecuteMethod:
        """Minimal Modal method handle that records remote calls."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Return deterministic remote bytes."""
            del payload, kwargs_payload
            return b"remote-response"

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance."""

        execute_payload = FakeExecuteMethod()
        runtime_version = types.SimpleNamespace(
            remote=lambda: _current_remote_runtime_payload(remote_modal_app_module)
        )

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(self, *, name: str | None = None, environment_name: str | None = None, **_: Any) -> "FakeApp":
            """Record the redeploy request and mark the deployment available again."""
            deploy_calls.append((name, environment_name))
            FakeModal.deployed = True
            return self

    class FakeModal:
        """Minimal modal SDK double with deployed lookup failure types."""

        deployed = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Raise the same missing-app error until auto-deploy recreates the app."""
                del app_name, class_name
                if not FakeModal.deployed:
                    raise FakeLookupError(
                        "Lookup failed for Cls 'RemoteEngine' from the 'comfy-modal-sync' app: "
                        "App 'comfy-modal-sync' not found in environment 'main'."
                    )
                return lambda **kwargs: FakeRemoteEngine()

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(
        modal_deployment_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES[
        (DEFAULT_TEST_DEPLOYMENT_APP_NAME, "main")
    ] = (
        modal_deployment_module._ModalAutoDeployState(ready=True)
    )
    try:
        response = remote_modal_app_module._invoke_modal_payload_blocking(
            {"component_id": "component-1"},
            b"{}",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert response == b"remote-response"
    assert deploy_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, "main")]

def test_remote_modal_redeploys_when_deployed_handle_disappears_during_payload_invoke(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A stale deployed engine handle should auto-redeploy when invocation hydrate fails."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance with stale-handle behavior."""

        def __init__(self, *, stale: bool) -> None:
            """Record whether this engine should fail once as a stale handle."""
            self._stale = stale
            self.execute_payload = types.SimpleNamespace(remote=self._remote)
            self.runtime_version = types.SimpleNamespace(
                remote=lambda: _current_remote_runtime_payload(remote_modal_app_module)
            )

        def _remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Raise the missing-app error for stale handles and otherwise succeed."""
            del payload, kwargs_payload
            if self._stale:
                raise FakeLookupError(
                    "Lookup failed for Cls 'RemoteEngine' from the 'comfy-modal-sync' app: "
                    "App 'comfy-modal-sync' not found in environment 'main'."
                )
            return b"remote-response"

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(
            self,
            *,
            name: str | None = None,
            environment_name: str | None = None,
            **_: Any,
        ) -> "FakeApp":
            """Record the redeploy request and mark future handles as fresh."""
            deploy_calls.append((name, environment_name))
            FakeModal.redeployed = True
            return self

    class FakeModal:
        """Minimal modal SDK double with stale-handle invocation failures."""

        redeployed = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return stale handles before redeploy and fresh handles after."""
                del app_name, class_name
                return lambda **kwargs: FakeRemoteEngine(stale=not FakeModal.redeployed)

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(
        modal_deployment_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    try:
        response = remote_modal_app_module._invoke_modal_payload_blocking(
            {"component_id": "component-1"},
            b"{}",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert response == b"remote-response"
    assert deploy_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, "main")]

def test_remote_modal_redeploys_when_deployed_handle_disappears_during_warmup(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_warmup_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A stale deployed warmup handle should auto-redeploy instead of surfacing a warmup error."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance with stale-handle warmup behavior."""

        def __init__(self, *, stale: bool) -> None:
            """Expose a warmup method that fails for stale handles."""
            self._stale = stale
            self.warmup_for_request = types.SimpleNamespace(remote=self._remote)
            self.runtime_version = types.SimpleNamespace(
                remote=lambda: _current_remote_runtime_payload(remote_modal_app_module)
            )

        def _remote(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Raise the missing-app error for stale handles and otherwise succeed."""
            if self._stale:
                raise FakeLookupError(
                    "Lookup failed for Cls 'RemoteEngine' from the 'comfy-modal-sync' app: "
                    "App 'comfy-modal-sync' not found in environment 'main'."
                )
            return {"component_id": str(payload.get("component_id"))}

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(
            self,
            *,
            name: str | None = None,
            environment_name: str | None = None,
            **_: Any,
        ) -> "FakeApp":
            """Record the redeploy request and mark future handles as fresh."""
            deploy_calls.append((name, environment_name))
            FakeModal.redeployed = True
            return self

    class FakeModal:
        """Minimal modal SDK double with stale-handle warmup failures."""

        redeployed = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return stale handles before redeploy and fresh handles after."""
                del app_name, class_name
                return lambda **kwargs: FakeRemoteEngine(stale=not FakeModal.redeployed)

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(
        modal_deployment_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    try:
        response = modal_warmup_module._invoke_modal_warmup_blocking(
            {"component_id": "component-1::warmup:0"},
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert response == {"component_id": "component-1::warmup:0"}
    assert deploy_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, "main")]

def test_remote_modal_replaces_out_of_date_deployed_app(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A deployed app with an old protocol should be stopped and auto-deployed again."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []
    stop_calls: list[str] = []

    class FakeExecuteMethod:
        """Minimal Modal method handle that records remote calls."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Return deterministic remote bytes."""
            del payload, kwargs_payload
            return b"remote-response"

    class FakeRemoteEngine:
        """Minimal deployed remote engine with protocol metadata."""

        def __init__(self, *, current: bool) -> None:
            """Record whether this engine should report the current protocol."""
            version_payload = _current_remote_runtime_payload(remote_modal_app_module)
            if not current:
                version_payload["protocol_version"] = (
                    remote_modal_app_module._REMOTE_APP_PROTOCOL_VERSION - 1
                )
            self.execute_payload = FakeExecuteMethod()
            self.runtime_version = types.SimpleNamespace(
                remote=lambda: version_payload
            )

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(
            self,
            *,
            name: str | None = None,
            environment_name: str | None = None,
            **_: Any,
        ) -> "FakeApp":
            """Record the deploy request and make subsequent lookups current."""
            deploy_calls.append((name, environment_name))
            FakeModal.deployed = True
            FakeModal.current = True
            return self

    class FakeModal:
        """Minimal Modal SDK double with an initially stale deployed app."""

        deployed = True
        current = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return the stale app first and the current replacement after deploy."""
                del app_name, class_name
                if not FakeModal.deployed:
                    raise FakeLookupError("Lookup failed for Cls 'RemoteEngine': not deployed")
                return lambda **kwargs: FakeRemoteEngine(current=FakeModal.current)

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    def fake_stop_app(app_name: str) -> None:
        """Record that stale app replacement stopped the old deployment."""
        stop_calls.append(app_name)
        FakeModal.deployed = False

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "_stop_modal_app_for_replacement", fake_stop_app)
    monkeypatch.setattr(
        modal_deployment_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    try:
        response = remote_modal_app_module._invoke_modal_payload_blocking(
            {"component_id": "component-1"},
            b"{}",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert response == b"remote-response"
    assert stop_calls == [DEFAULT_TEST_DEPLOYMENT_APP_NAME]
    assert deploy_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, "main")]

def test_remote_modal_rejects_fingerprint_mismatch_when_auto_deploy_is_disabled(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A same-protocol worker with different source or settings must not execute."""
    stale_version_payload = _current_remote_runtime_payload(remote_modal_app_module)
    stale_version_payload["runtime_fingerprint"] = "stale-runtime"
    remote_engine = types.SimpleNamespace(
        runtime_version=types.SimpleNamespace(remote=lambda: stale_version_payload)
    )
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "false")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    try:
        with pytest.raises(
            modal_deployment_module.ModalRemoteInvocationError,
            match="runtime fingerprint is out of date",
        ):
            modal_deployment_module._ensure_remote_engine_protocol_current(
                remote_engine,
                {"component_id": "component-1"},
            )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

def test_remote_modal_rebinds_affinity_after_compatible_protocol_probe(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A legacy-safe version probe should be replaced before real execution."""
    version_payload = _current_remote_runtime_payload(remote_modal_app_module)
    probe_engine = types.SimpleNamespace(
        runtime_version=types.SimpleNamespace(remote=lambda: version_payload)
    )
    affinity_engine = object()
    observed_payloads: list[dict[str, Any]] = []

    def fake_lookup(payload: dict[str, Any]) -> object:
        """Return the affinity-aware handle created after protocol validation."""
        observed_payloads.append(payload)
        return affinity_engine

    monkeypatch.setattr(
        modal_deployment_module,
        "_lookup_deployed_remote_engine",
        fake_lookup,
    )
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    payload = {"component_id": "component-1"}
    try:
        result = modal_deployment_module._ensure_remote_engine_protocol_current(
            probe_engine,
            payload,
        )
    finally:
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert result is affinity_engine
    assert observed_payloads == [payload]

def test_remote_modal_rebinds_affinity_after_cached_protocol_validation(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A cached protocol result must not turn its parameterless probe into execution."""
    probe_engine = object()
    affinity_engine = object()
    observed_payloads: list[dict[str, Any]] = []

    def fake_lookup(payload: dict[str, Any]) -> object:
        """Return the affinity-aware handle created for actual execution."""
        observed_payloads.append(payload)
        return affinity_engine

    monkeypatch.setattr(
        modal_deployment_module,
        "_lookup_deployed_remote_engine",
        fake_lookup,
    )
    payload = {
        "component_id": "component-llm",
        "remote_worker_affinity_group": "llm",
    }
    runtime_cache_key = modal_deployment_module._modal_runtime_cache_key(payload)
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.add(runtime_cache_key)
    try:
        result = modal_deployment_module._ensure_remote_engine_protocol_current(
            probe_engine,
            payload,
        )
    finally:
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert result is affinity_engine
    assert observed_payloads == [payload]

def test_remote_modal_cached_protocol_skips_parameterless_probe_construction(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A cached runtime must construct only the affinity-bound engine handle."""
    affinity_engine = object()
    observed_lookups: list[tuple[dict[str, Any], bool]] = []

    def fake_lookup(
        payload: dict[str, Any],
        *,
        affinity_key_override: str | None = None,
        protocol_probe: bool = False,
    ) -> object:
        """Record whether lookup attempted to allocate a parameterless probe."""
        del affinity_key_override
        observed_lookups.append((payload, protocol_probe))
        return affinity_engine

    monkeypatch.setattr(
        modal_deployment_module,
        "_lookup_deployed_remote_engine",
        fake_lookup,
    )
    payload = {
        "component_id": "component-comfy::warmup:0",
        "remote_worker_affinity_group": "comfy",
    }
    runtime_cache_key = modal_deployment_module._modal_runtime_cache_key(payload)
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.add(runtime_cache_key)
    try:
        result = modal_deployment_module._lookup_protocol_current_remote_engine(payload)
    finally:
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert result is affinity_engine
    assert observed_lookups == [(payload, False)]

def test_remote_modal_uncached_protocol_uses_cpu_stager_before_gpu_lookup(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """First validation should use CPU metadata and construct one bound GPU handle."""
    affinity_engine = object()
    observed_lookups: list[bool] = []
    version_payload = _current_remote_runtime_payload(remote_modal_app_module)

    def fake_lookup(
        payload: dict[str, Any],
        *,
        affinity_key_override: str | None = None,
        protocol_probe: bool = False,
    ) -> object:
        """Record whether lookup attempted to allocate a GPU protocol probe."""
        del payload, affinity_key_override
        observed_lookups.append(protocol_probe)
        return affinity_engine

    monkeypatch.setattr(
        modal_deployment_module,
        "_remote_runtime_version_from_cpu_stager",
        lambda payload: version_payload,
    )
    monkeypatch.setattr(
        modal_deployment_module,
        "_lookup_deployed_remote_engine",
        fake_lookup,
    )
    payload = {
        "component_id": "component-llm",
        "remote_worker_affinity_group": "llm",
    }
    runtime_cache_key = modal_deployment_module._modal_runtime_cache_key(payload)
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()
    try:
        result = modal_deployment_module._lookup_protocol_current_remote_engine(payload)
        runtime_was_cached = (
            runtime_cache_key
            in modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK
        )
    finally:
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert result is affinity_engine
    assert runtime_was_cached is True
    assert observed_lookups == [False]

def test_remote_modal_auto_deploy_is_shared_across_concurrent_first_run_callers(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    modal_warmup_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Concurrent first-run callers should share one deploy and wait for lookup readiness."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    deploy_calls: list[tuple[str | None, str | None]] = []
    deployed_ready_event = threading.Event()
    lookup_lock = threading.Lock()
    ready_after_deploy_misses = 0

    class FakeExecuteMethod:
        """Minimal Modal method handle that returns deterministic bytes."""

        def remote(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Return deterministic remote bytes."""
            del payload, kwargs_payload
            return b"remote-response"

    class FakeWarmupMethod:
        """Minimal Modal warmup handle."""

        def remote(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Return deterministic warmup metadata."""
            return {"component_id": str(payload.get("component_id"))}

    class FakeRemoteEngine:
        """Minimal deployed remote engine instance."""

        execute_payload = FakeExecuteMethod()
        warmup_for_request = FakeWarmupMethod()
        runtime_version = types.SimpleNamespace(
            remote=lambda: _current_remote_runtime_payload(remote_modal_app_module)
        )

    class FakeApp:
        """Minimal deployable cloud app double."""

        def deploy(self, *, name: str | None = None, environment_name: str | None = None, **_: Any) -> "FakeApp":
            """Record one deploy request and make the app discoverable shortly after."""
            nonlocal ready_after_deploy_misses
            deploy_calls.append((name, environment_name))
            with lookup_lock:
                FakeModal.deployed = True
                ready_after_deploy_misses = 2
            deployed_ready_event.set()
            return self

    class FakeModal:
        """Minimal modal SDK double with eventual-consistency lookup behavior."""

        deployed = False
        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a deployed class after one deploy and a short readiness delay."""
                nonlocal ready_after_deploy_misses
                del app_name, class_name
                with lookup_lock:
                    if not FakeModal.deployed:
                        raise FakeLookupError("Lookup failed for Cls 'RemoteEngine': not deployed")
                    if ready_after_deploy_misses > 0:
                        ready_after_deploy_misses -= 1
                        raise FakeLookupError(
                            "Lookup failed for Cls 'RemoteEngine' from the 'comfy-modal-sync' app: "
                            "App 'comfy-modal-sync' not found in environment 'main'."
                        )
                return lambda **kwargs: FakeRemoteEngine()

        @staticmethod
        def enable_output() -> Any:
            """Provide a no-op output context manager."""
            return nullcontext()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_llm_profile_staging_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setattr(
        modal_deployment_module,
        "_load_modal_cloud_module",
        lambda: types.SimpleNamespace(app=FakeApp()),
    )
    monkeypatch.setattr(modal_deployment_module.time, "sleep", lambda _: None)
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    remote_modal_app_module.get_settings.cache_clear()
    modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
    modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    try:
        payload_response: list[bytes] = []
        warmup_response: list[dict[str, Any]] = []
        thread_errors: list[BaseException] = []

        def run_payload() -> None:
            """Invoke the payload path from one thread."""
            try:
                payload_response.append(
                    remote_modal_app_module._invoke_modal_payload_blocking(
                        {"component_id": "component-1"},
                        b"{}",
                    )
                )
            except Exception as exc:  # pragma: no cover - assertion surface
                thread_errors.append(exc)

        def run_warmup() -> None:
            """Invoke the warmup path from one thread."""
            try:
                warmup_response.append(
                    modal_warmup_module._invoke_modal_warmup_blocking(
                        {"component_id": "component-1", "prompt_id": "prompt-1"},
                    )
                )
            except Exception as exc:  # pragma: no cover - assertion surface
                thread_errors.append(exc)

        payload_thread = threading.Thread(target=run_payload)
        warmup_thread = threading.Thread(target=run_warmup)
        payload_thread.start()
        assert deployed_ready_event.wait(1.0), thread_errors
        warmup_thread.start()
        payload_thread.join(timeout=1.0)
        warmup_thread.join(timeout=1.0)
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_deployment_module._MODAL_AUTO_DEPLOY_STATES.clear()
        modal_deployment_module._MODAL_REMOTE_APP_VERSION_OK.clear()

    assert thread_errors == []
    assert payload_response == [b"remote-response"]
    assert warmup_response == [{"component_id": "component-1"}]
    assert deploy_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, "main")]

def test_lookup_deployed_remote_engine_uses_affinity_as_modal_class_parameter(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Deployed Modal lookups should isolate reusable worker slots by class identity."""
    observed_kwargs: list[dict[str, Any]] = []

    class FakeModal:
        """Minimal modal SDK double that captures class construction kwargs."""

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a factory that records the provided class parameters."""
                assert app_name == DEFAULT_TEST_DEPLOYMENT_APP_NAME
                assert class_name == "RemoteEngine"

                def build_remote_engine(**kwargs: Any) -> dict[str, Any]:
                    """Record one synthesized Modal class constructor call."""
                    observed_kwargs.append(kwargs)
                    return kwargs

                return build_remote_engine

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)

    result = modal_deployment_module._lookup_deployed_remote_engine(
        {
            "component_id": "component-1",
            "remote_session": remote_modal_app_module.RemoteSessionHandle(
                session_id="session-123",
                prompt_id="prompt-1",
                owner_component_id="component-1",
            ).to_payload(),
        }
    )
    probe_result = modal_deployment_module._lookup_deployed_remote_engine(
        {"component_id": "component-1"},
        protocol_probe=True,
    )

    assert result == {
        "gpu_snapshot_enabled": False,
        "worker_affinity_key": "worker-pool:slot:0",
    }
    assert probe_result == {"gpu_snapshot_enabled": False}
    assert observed_kwargs == [
        {
            "gpu_snapshot_enabled": False,
            "worker_affinity_key": "worker-pool:slot:0",
        },
        {"gpu_snapshot_enabled": False},
    ]

def test_lookup_deployed_remote_engine_uses_workflow_gpu_app_identity(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """A B300 workflow must miss or reuse its own app without contacting the A100 class."""
    observed_lookups: list[tuple[str, str]] = []

    class FakeModal:
        """Minimal Modal SDK double that captures deployed class identity."""

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a no-op class factory for the requested deployment."""
                observed_lookups.append((app_name, class_name))
                return lambda **kwargs: kwargs

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)

    result = modal_deployment_module._lookup_deployed_remote_engine(
        {"component_id": "component-1", "modal_gpu": "B300"}
    )

    assert result == {
        "gpu_snapshot_enabled": False,
        "worker_affinity_key": "worker-pool:slot:0",
    }
    assert observed_lookups == [("comfy-modal-sync-gpu-b300", "RemoteEngine")]

def test_lookup_deployed_remote_engine_extends_local_gap_pool_scaledown(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Sandwiched local work should use a longer independently autoscaled Modal pool."""
    observed_options: list[dict[str, Any]] = []
    observed_kwargs: list[dict[str, Any]] = []

    class FakeRemoteCls:
        """Minimal deployed class supporting Modal runtime option overrides."""

        def with_options(self, **kwargs: Any) -> "FakeRemoteCls":
            """Record the independent autoscaler configuration."""
            observed_options.append(kwargs)
            return self

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            """Record the parametrized class instance identity."""
            observed_kwargs.append(kwargs)
            return kwargs

    class FakeModal:
        """Minimal Modal SDK double for deployed class lookup."""

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> FakeRemoteCls:
                """Return the shared fake deployed class."""
                assert app_name == DEFAULT_TEST_DEPLOYMENT_APP_NAME
                assert class_name == "RemoteEngine"
                return FakeRemoteCls()

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)

    result = modal_deployment_module._lookup_deployed_remote_engine(
        {
            "component_id": "component-1",
            "prompt_id": "prompt-1",
            "remote_local_gap_pool": True,
        }
    )

    assert observed_options == [{"scaledown_window": 900}]
    assert observed_kwargs == [result]
    assert result["worker_affinity_key"] == "worker-pool:slot:0"

def test_lookup_deployed_remote_engine_passes_snapshot_profile_parameter_for_gpu_snapshots(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Deployed Modal lookups should parameterize workers by snapshot profile when enabled."""
    observed_kwargs: list[dict[str, Any]] = []
    snapshot_profiles: dict[str, Any] = {}

    class FakeModal:
        """Minimal modal SDK double that captures class construction kwargs."""

        class Dict:
            """Namespace for fake dict lookups."""

            @staticmethod
            def from_name(dict_name: str, create_if_missing: bool = False) -> Any:
                """Return the shared fake snapshot profile store."""
                assert dict_name == "comfy-modal-sync-snapshot-profiles"
                assert create_if_missing is True
                return snapshot_profiles

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a factory that records the provided class parameters."""
                assert app_name == DEFAULT_TEST_DEPLOYMENT_APP_NAME
                assert class_name == "RemoteEngine"

                def build_remote_engine(**kwargs: Any) -> dict[str, Any]:
                    """Record one synthesized Modal class constructor call."""
                    observed_kwargs.append(kwargs)
                    return kwargs

                return build_remote_engine

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()
    payload = {
        "component_id": "component-1",
        "subgraph_prompt": {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model-a.safetensors", "weight_dtype": "default"},
            }
        },
    }
    try:
        result = modal_deployment_module._lookup_deployed_remote_engine(payload)
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()

    assert result["gpu_snapshot_enabled"] is True
    assert result["snapshot_profile_key"].startswith("loader-profile:")
    assert observed_kwargs == [result]
    assert result["snapshot_profile_key"] in snapshot_profiles
    assert payload["snapshot_profile_key"] == result["snapshot_profile_key"]

def test_lookup_deployed_remote_engine_stores_existing_snapshot_profile_record_when_possible(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Deployed lookups should backfill snapshot records even when the payload already carries the key."""
    observed_kwargs: list[dict[str, Any]] = []
    snapshot_profiles: dict[str, Any] = {}

    class FakeModal:
        """Minimal modal SDK double that captures class construction kwargs."""

        class Dict:
            """Namespace for fake dict lookups."""

            @staticmethod
            def from_name(dict_name: str, create_if_missing: bool = False) -> Any:
                """Return the shared fake snapshot profile store."""
                assert dict_name == "comfy-modal-sync-snapshot-profiles"
                assert create_if_missing is True
                return snapshot_profiles

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a factory that records the provided class parameters."""
                assert app_name == DEFAULT_TEST_DEPLOYMENT_APP_NAME
                assert class_name == "RemoteEngine"

                def build_remote_engine(**kwargs: Any) -> dict[str, Any]:
                    """Record one synthesized Modal class constructor call."""
                    observed_kwargs.append(kwargs)
                    return kwargs

                return build_remote_engine

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()
    expected_snapshot_profile_key = modal_warmup_module._loader_snapshot_profile_key(
        [
            {
                "signature": modal_warmup_module._loader_prewarm_plan_signature(
                    "UNETLoader",
                    {"unet_name": "model-a.safetensors", "weight_dtype": "default"},
                )
            }
        ]
    )
    payload = {
        "component_id": "component-1",
        "snapshot_profile_key": expected_snapshot_profile_key,
        "subgraph_prompt": {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model-a.safetensors", "weight_dtype": "default"},
            }
        },
    }
    try:
        result = modal_deployment_module._lookup_deployed_remote_engine(payload)
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()

    assert result["snapshot_profile_key"] == expected_snapshot_profile_key
    assert observed_kwargs == [result]
    assert expected_snapshot_profile_key in snapshot_profiles

def test_lookup_deployed_remote_engine_reuses_worker_pool_slots_across_prompt_sessions(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Different prompt sessions should reuse one shared deployed RemoteEngine identity."""
    observed_kwargs: list[dict[str, Any]] = []

    class FakeModal:
        """Minimal modal SDK double that captures class construction kwargs."""

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a factory that records the provided class parameters."""
                assert app_name == DEFAULT_TEST_DEPLOYMENT_APP_NAME
                assert class_name == "RemoteEngine"

                def build_remote_engine(**kwargs: Any) -> dict[str, Any]:
                    """Record one synthesized Modal class constructor call."""
                    observed_kwargs.append(kwargs)
                    return kwargs

                return build_remote_engine

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)

    first_result = modal_deployment_module._lookup_deployed_remote_engine(
        {
            "component_id": "component-1",
            "prompt_id": "prompt-1",
            "remote_session": remote_modal_app_module.RemoteSessionHandle(
                session_id="session-123",
                prompt_id="prompt-1",
                owner_component_id="component-1",
            ).to_payload(),
        }
    )
    second_result = modal_deployment_module._lookup_deployed_remote_engine(
        {
            "component_id": "component-1",
            "prompt_id": "prompt-2",
            "remote_session": remote_modal_app_module.RemoteSessionHandle(
                session_id="session-456",
                prompt_id="prompt-2",
                owner_component_id="component-1",
            ).to_payload(),
        }
    )

    assert first_result == {
        "gpu_snapshot_enabled": False,
        "worker_affinity_key": "worker-pool:slot:0",
    }
    assert second_result == {
        "gpu_snapshot_enabled": False,
        "worker_affinity_key": "worker-pool:slot:0",
    }
    assert observed_kwargs == [
        {
            "gpu_snapshot_enabled": False,
            "worker_affinity_key": "worker-pool:slot:0",
        },
        {
            "gpu_snapshot_enabled": False,
            "worker_affinity_key": "worker-pool:slot:0",
        },
    ]

def test_lookup_deployed_remote_engine_isolates_profiled_lane_overrides(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Lane-specific affinity overrides should create stable separate Modal identities."""
    observed_kwargs: list[dict[str, Any]] = []
    snapshot_profiles: dict[str, Any] = {}

    class FakeModal:
        """Minimal modal SDK double that captures class construction kwargs."""

        class Dict:
            """Namespace for fake dict lookups."""

            @staticmethod
            def from_name(dict_name: str, create_if_missing: bool = False) -> Any:
                """Return the shared fake snapshot profile store."""
                assert dict_name == "comfy-modal-sync-snapshot-profiles"
                assert create_if_missing is True
                return snapshot_profiles

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Return a factory that records the provided class parameters."""
                assert app_name == DEFAULT_TEST_DEPLOYMENT_APP_NAME
                assert class_name == "RemoteEngine"

                def build_remote_engine(**kwargs: Any) -> dict[str, Any]:
                    """Record one synthesized Modal class constructor call."""
                    observed_kwargs.append(kwargs)
                    return kwargs

                return build_remote_engine

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()
    payload = {
        "component_id": "component-1__mapped",
        "subgraph_prompt": {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model-a.safetensors", "weight_dtype": "default"},
            }
        },
    }
    try:
        first_result = modal_deployment_module._lookup_deployed_remote_engine(
            payload,
            affinity_key_override="worker-pool:slot:0",
        )
        second_result = modal_deployment_module._lookup_deployed_remote_engine(
            payload,
            affinity_key_override="worker-pool:slot:3",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()

    assert first_result != second_result
    assert first_result["worker_affinity_key"] == "worker-pool:slot:0"
    assert second_result["worker_affinity_key"] == "worker-pool:slot:3"
    assert first_result["snapshot_profile_key"].startswith("loader-profile:")
    assert observed_kwargs == [first_result, second_result]

def test_remote_modal_requires_manual_deploy_when_auto_deploy_disabled(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote mode should fail clearly when auto-deploy and ephemeral fallback are both disabled."""

    class FakeLookupError(Exception):
        """Stand-in for Modal deployed lookup failures."""

    class FakeModal:
        """Minimal modal SDK double with deployed lookup failure types."""

        exception = types.SimpleNamespace(
            NotFoundError=FakeLookupError,
            ExecutionError=FakeLookupError,
            InvalidError=FakeLookupError,
        )

        class Cls:
            """Namespace for deployed class lookups."""

            @staticmethod
            def from_name(app_name: str, class_name: str) -> Any:
                """Simulate a missing deployed app."""
                raise FakeLookupError("not deployed")

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_deployment_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "false")
    monkeypatch.setenv("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK", "false")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        try:
            remote_modal_app_module._invoke_modal_payload_blocking(
                {"component_id": "component-1"},
                b"{}",
            )
        except remote_modal_app_module.ModalRemoteInvocationError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ModalRemoteInvocationError to be raised.")
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert "requires a deployed Modal app or a successful first-run auto-deploy" in message
    assert "COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true" in message

def test_invoke_remote_engine_releases_prompt_when_cancelled_during_deploy(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
    monkeypatch: Any,
) -> None:
    """Cancellation during Modal deploy/provisioning should release the local prompt promptly."""

    class FakeInterrupt(Exception):
        """Stand-in for ComfyUI's InterruptProcessingException."""

    release_blocking_call = threading.Event()
    observed_cancellation_events: list[threading.Event] = []
    interrupt_checks = iter([False, True, True, True])
    remote_interrupt_payloads: list[dict[str, Any]] = []

    def fake_blocking_invoke(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        cancellation_event: threading.Event | None = None,
    ) -> bytes:
        """Simulate a worker stuck in Modal deployment until the test releases it."""
        del payload, kwargs_payload
        assert cancellation_event is not None
        observed_cancellation_events.append(cancellation_event)
        release_blocking_call.wait(timeout=5.0)
        return b"late-response"

    def fake_local_processing_interrupted() -> bool:
        """Report a local interrupt after the first outer wait poll."""
        return next(interrupt_checks, True)

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setenv("COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS", "0.01")
    remote_modal_app_module.get_settings.cache_clear()
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_invoke_modal_payload_blocking", fake_blocking_invoke)
    monkeypatch.setattr(
        modal_interrupts_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        modal_interrupts_module,
        "_request_remote_interrupt",
        lambda payload: remote_interrupt_payloads.append(dict(payload)) or True,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_raise_local_interrupt",
        lambda: (_ for _ in ()).throw(FakeInterrupt()),
    )

    try:
        with pytest.raises(FakeInterrupt):
            remote_modal_app_module.invoke_remote_engine(
                {"prompt_id": "prompt-1", "component_id": "component-1", "payload_kind": "subgraph"},
                b"{}",
            )
    finally:
        release_blocking_call.set()
        remote_modal_app_module.get_settings.cache_clear()

    assert len(observed_cancellation_events) == 1
    assert observed_cancellation_events[0].is_set()
    assert [payload["component_id"] for payload in remote_interrupt_payloads] == ["component-1"]
