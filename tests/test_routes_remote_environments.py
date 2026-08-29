"""Tests for the routes remote environments boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_ssh_hostname_extracts_safe_runtime_badge_label(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
) -> None:
    """Planner UI metadata should show the host rather than an SSH user target."""
    assert execution_scheduling_module._ssh_hostname("worker@example.internal") == "example.internal"
    assert execution_scheduling_module._ssh_hostname("[2001:db8::17]") == "2001:db8::17"

def test_remote_environment_routes_save_and_probe_hosts(
    api_intercept_module: Any,
    routes_r2_module: Any,
    routes_remote_environments_module: Any,
    remote_hosts_module: Any,
    execution_environments_module: Any,
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The ComfyUI API should persist and refresh credential-free SSH hosts."""

    class FakeRoutes:
        """Capture handlers by HTTP method and route."""

        def __init__(self) -> None:
            """Initialize an empty handler map."""
            self.handlers: dict[tuple[str, str], Any] = {}

        def _decorator(self, method: str, path: str) -> Any:
            """Return one route registration decorator."""

            def register(handler: Any) -> Any:
                """Store and return one handler."""
                self.handlers[(method, path)] = handler
                return handler

            return register

        def get(self, path: str) -> Any:
            """Register one GET route."""
            return self._decorator("GET", path)

        def put(self, path: str) -> Any:
            """Register one PUT route."""
            return self._decorator("PUT", path)

        def post(self, path: str) -> Any:
            """Register one POST route."""
            return self._decorator("POST", path)

    class FakeRequest:
        """Return one predefined JSON body."""

        def __init__(self, payload: dict[str, Any]) -> None:
            """Store the request body."""
            self.payload = payload
            self.query: dict[str, str] = {}

        async def json(self) -> dict[str, Any]:
            """Return the request body."""
            return self.payload

    registry = remote_hosts_module.RemoteHostRegistry.for_user_directory(tmp_path)
    routes = FakeRoutes()
    prompt_server = SimpleNamespace(routes=routes, prompt_queue=None)
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
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
    capabilities = execution_environments_module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=60 * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
    )

    class FakeController:
        """Return fixed capabilities without opening SSH."""

        def __init__(self, host: Any) -> None:
            """Retain the selected host."""
            self.host = host

        def probe_capabilities(self) -> Any:
            """Return fixed ready capabilities."""
            return capabilities

    monkeypatch.setattr(api_intercept_module, "_ROUTE_REGISTERED", False)
    monkeypatch.setattr(
        api_intercept_module,
        "_ssh_host_registry",
        lambda _settings: registry,
    )
    monkeypatch.setattr(
        routes_remote_environments_module,
        "SshDockerController",
        FakeController,
    )
    monkeypatch.setattr(
        routes_r2_module,
        "_refresh_r2_storage_usage",
        lambda _storage: api_intercept_module.R2StorageUsage(
            size_bytes=7 * 1024**3,
            object_count=17,
        ),
    )
    unlock_requests: list[bool] = []
    monkeypatch.setattr(
        routes_r2_module,
        "request_macos_keychain_unlock",
        lambda: unlock_requests.append(True),
    )
    monkeypatch.setattr(
        api_intercept_module,
        "_get_server_module",
        lambda: SimpleNamespace(PromptServer=SimpleNamespace(instance=prompt_server)),
    )

    api_intercept_module.setup_modal_queue_route(
        prompt_server=prompt_server,
        sync_engine=object(),
        settings=settings,
    )
    update = routes.handlers[("PUT", "/remote/environments")]
    update_response = asyncio.run(
        update(
            FakeRequest(
                {
                    "version": 1,
                    "hosts": [
                        {
                            "environment_id": "gpu-one",
                            "display_name": "GPU one",
                            "ssh_target": "gpu-one",
                        }
                    ],
                }
            )
        )
    )
    probe = routes.handlers[("POST", "/remote/environments/probe")]
    probe_response = asyncio.run(
        probe(FakeRequest({"environment_id": "gpu-one"}))
    )
    vast_verify = routes.handlers[("POST", "/remote/vast/verify")]
    vast_verify_response = asyncio.run(vast_verify(FakeRequest({})))
    r2_usage = routes.handlers[("POST", "/remote/storage/r2/usage")]
    r2_usage_response = asyncio.run(
        r2_usage(
            FakeRequest(
                {
                    "configuration_id": "385",
                    "display_name": "Shared R2",
                    "account_id": "a" * 32,
                    "bucket": "models",
                    "credential_id": "opaque-reference",
                    "jurisdiction": "eu",
                }
            )
        )
    )
    r2_unlock = routes.handlers[("POST", "/remote/storage/r2/keychain/unlock")]
    r2_unlock_response = asyncio.run(r2_unlock(FakeRequest({})))
    monkeypatch.setattr(
        routes_r2_module,
        "_refresh_r2_storage_usage",
        lambda _storage: (_ for _ in ()).throw(
            api_intercept_module.R2CredentialError(
                "The macOS login keychain must be unlocked.",
                code=api_intercept_module.R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
            )
        ),
    )
    r2_locked_response = asyncio.run(
        r2_usage(
            FakeRequest(
                {
                    "configuration_id": "385",
                    "display_name": "Shared R2",
                    "account_id": "a" * 32,
                    "bucket": "models",
                    "credential_id": "opaque-reference",
                    "jurisdiction": "eu",
                }
            )
        )
    )

    assert update_response.status == 200
    assert probe_response.status == 200
    assert vast_verify_response.status == 400
    assert r2_usage_response.status == 200
    assert r2_unlock_response.status == 200
    assert unlock_requests == [True]
    assert r2_locked_response.status == 423
    assert json.loads(r2_locked_response.text)["code"] == "keychain_unlock_required"
    assert json.loads(r2_usage_response.text)["storage_usage_bytes"] == 7 * 1024**3
    assert json.loads(r2_usage_response.text)["storage_object_count"] == 17
    assert ("GET", "/remote/vast/status") in routes.handlers
    assert ("POST", "/remote/vast/reap") in routes.handlers
    assert ("POST", "/remote/vast/destroy") in routes.handlers
    assert ("POST", "/modal/container_stop") in routes.handlers
    assert ("POST", "/remote/storage/r2/oauth/start") in routes.handlers
    assert ("GET", "/remote/storage/r2/oauth/callback") in routes.handlers
    assert ("POST", "/remote/storage/r2/credentials") in routes.handlers
    assert ("GET", "/remote/storage/r2/status") in routes.handlers
    assert ("POST", "/remote/storage/r2/usage") in routes.handlers
    assert ("POST", "/remote/storage/r2/keychain/unlock") in routes.handlers
    assert registry.get_host("gpu-one").health.value == "ready"
    assert registry.get_host("gpu-one").capabilities == capabilities

