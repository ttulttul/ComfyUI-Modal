"""Tests for Subrosa configuration, scheduling, credentials, and relay framing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Self

import aiohttp
import pytest
from keyring.errors import KeyringError


@dataclass
class _MemoryPasswordStore:
    """Provide a deterministic keyring-compatible store for tests."""

    values: dict[tuple[str, str], str] = field(default_factory=dict)

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return one stored value."""
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Store one value."""
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        """Delete one value."""
        self.values.pop((service_name, username), None)


class _FakeWebSocket:
    """Expose the aiohttp WebSocket methods used by the executor."""

    def __init__(self, messages: list[aiohttp.WSMessage] | None = None) -> None:
        """Initialize queued receive messages and captured sends."""
        self.messages = list(messages or [])
        self.sent: list[bytes] = []

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        """Return the next queued server message."""
        del timeout
        return self.messages.pop(0)

    async def __aenter__(self) -> Self:
        """Enter the fake socket context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the fake socket context."""

    async def send_bytes(self, payload: bytes) -> None:
        """Capture one client binary message."""
        self.sent.append(payload)


def _binary_message(payload: bytes) -> aiohttp.WSMessage:
    """Build one aiohttp binary message for a fake socket."""
    return aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, payload, None)


def _control_message(payload: Mapping[str, Any]) -> aiohttp.WSMessage:
    """Build one lane-1 control WebSocket message."""
    return _binary_message(
        b"\x01"
        + json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode()
    )


def test_subrosa_configuration_is_credential_free_and_compiles(
    remote_configuration_nodes_module: Any,
    execution_environments_module: Any,
) -> None:
    """A queued node should retain only an opaque credential reference."""
    nodes = remote_configuration_nodes_module
    prompt = {
        "42": {
            "class_type": nodes.SUBROSA_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "Subrosa staging",
                "relay_url": "wss://staging.subrosa.red/",
                "pool": "mock-4090",
                "maximum_workers": 2,
                "credential_id": "subrosa-default",
            },
        },
        "99": {
            "class_type": nodes.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {"configuration_0": ["42", 0]},
        },
    }

    configuration_set = nodes.compile_remote_configuration_set(prompt)

    assert configuration_set is not None
    configuration = configuration_set.capacity_configurations[0]
    assert configuration.provider is execution_environments_module.ExecutionProvider.SUBROSA
    assert configuration.capacity_limit == 2
    assert configuration.relay_url == "wss://staging.subrosa.red"
    safe = configuration.to_safe_dict()
    assert safe["configuration_id"] == "42"
    assert safe["credential_id"] == "subrosa-default"
    assert "token" not in json.dumps(safe).casefold()


@pytest.mark.parametrize("credential_inputs", [{}, {"credential_id": ""}])
def test_subrosa_omitted_or_blank_credential_id_falls_back_to_node_id(
    remote_configuration_nodes_module: Any,
    credential_inputs: dict[str, str],
) -> None:
    """Legacy queued prompts should retain their graph ID as a lookup fallback."""
    configuration = remote_configuration_nodes_module.subrosa_configuration_from_inputs(
        "42",
        {
            "relay_url": "wss://staging.subrosa.red",
            "pool": "mock-4090",
            **credential_inputs,
        },
    )

    assert configuration.credential_id == "42"


def test_subrosa_configuration_rejects_empty_credential_id(
    remote_configurations_module: Any,
) -> None:
    """Direct configuration construction must reject unusable keyring references."""
    with pytest.raises(ValueError, match="credential_id must not be empty"):
        remote_configurations_module.SubrosaRemoteConfiguration(
            configuration_id="42",
            display_name="Subrosa staging",
            relay_url="wss://staging.subrosa.red",
            pool="mock-4090",
            credential_id="  ",
        )


def test_subrosa_credential_store_never_serializes_token(
    subrosa_credentials_module: Any,
) -> None:
    """Extension tokens should round-trip only through the password store."""
    password_store = _MemoryPasswordStore()
    store = subrosa_credentials_module.SubrosaCredentialStore(
        password_store=password_store
    )
    token = "srk_test-secret-value"

    store.save("staging", token)

    assert store.require("staging") == token
    assert token not in repr(store)
    store.delete("staging")
    assert store.load("staging") is None


def test_subrosa_credential_store_reports_locked_macos_keychain(
    subrosa_credentials_module: Any,
) -> None:
    """A headless keychain denial should carry the existing UI recovery code."""

    class LockedPasswordStore(_MemoryPasswordStore):
        """Reject writes with macOS errSecInteractionNotAllowed."""

        def set_password(
            self,
            service_name: str,
            username: str,
            password: str,
        ) -> None:
            """Raise the status returned by a locked login keychain."""
            del service_name, username, password
            raise KeyringError(-25308, "interaction not allowed")

    store = subrosa_credentials_module.SubrosaCredentialStore(
        password_store=LockedPasswordStore()
    )

    with pytest.raises(
        subrosa_credentials_module.SubrosaCredentialError
    ) as caught:
        store.save("staging", "srk_test-secret-value")

    assert caught.value.code == "keychain_unlock_required"


def test_subrosa_candidate_has_schedulable_mock_capabilities(
    execution_scheduling_module: Any,
    execution_environments_module: Any,
    remote_configurations_module: Any,
    settings_module: Any,
) -> None:
    """The configured pool should pass GPU and NVIDIA-runtime admission gates."""
    configuration = remote_configurations_module.SubrosaRemoteConfiguration(
        configuration_id="42",
        display_name="Subrosa staging",
        relay_url="wss://staging.subrosa.red",
        pool="mock-4090",
        credential_id="subrosa-default",
        maximum_workers=3,
    )

    state, quote = execution_scheduling_module._configured_candidate_environment(
        configuration=configuration,
        requirements=execution_environments_module.ComponentResourceRequirements(),
        settings=settings_module.get_settings(),
        ssh_hosts_by_id={},
        vast_service=None,
        vast_unavailable_reason=None,
    )

    assert quote is None
    assert state is not None
    assert state.provider is execution_environments_module.ExecutionProvider.SUBROSA
    assert state.environment_id == "subrosa:42"
    assert state.maximum_workers == 3
    assert state.capabilities is not None
    assert state.capabilities.nvidia_container_runtime is True
    assert state.capabilities.maximum_vram_bytes == 24 * 1024**3


def test_connected_subrosa_configuration_plans_without_provider_side_effects(
    execution_scheduling_module: Any,
    execution_environments_module: Any,
    remote_configuration_nodes_module: Any,
    settings_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Subrosa-only configurator should produce a normal capacity assignment."""
    api = execution_scheduling_module
    nodes = remote_configuration_nodes_module
    monkeypatch.setattr(api, "_execution_history", lambda _settings: None)
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "42": {
            "class_type": nodes.SUBROSA_REMOTE_CONFIGURATION_NODE_ID,
            "inputs": {
                "configuration_name": "Subrosa staging",
                "relay_url": "wss://staging.subrosa.red",
                "pool": "mock-4090",
                "maximum_workers": 1,
            },
        },
        "99": {
            "class_type": nodes.REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            "inputs": {"configuration_0": ["42", 0]},
        },
    }
    component = api.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["1"],
        contains_output_node=False,
    )
    plan_events: list[dict[str, Any]] = []

    plan = api._plan_component_execution(
        components=[component],
        prompt=prompt,
        workflow=None,
        settings=settings_module.get_settings(),
        plan_callback=lambda assignments, _configurations: plan_events.append(
            assignments
        ),
    )

    assignment = plan.assignments["1"]
    assert assignment.provider is execution_environments_module.ExecutionProvider.SUBROSA
    assert assignment.environment_id == "subrosa:42"
    assert plan_events[-1]["1"]["hardware"]["gpu_memory_bytes_total"] == 24 * 1024**3


def test_subrosa_provider_metadata_contains_relay_reference_only(
    execution_scheduling_module: Any,
    execution_environments_module: Any,
    remote_configurations_module: Any,
) -> None:
    """Queue-time payload metadata should contain routing data but no token."""
    configuration = remote_configurations_module.SubrosaRemoteConfiguration(
        configuration_id="42",
        display_name="Subrosa staging",
        relay_url="wss://staging.subrosa.red",
        pool="mock-4090",
        credential_id="subrosa-default",
    )
    assignment = execution_environments_module.ExecutionAssignment(
        environment_id="subrosa:42",
        provider=execution_environments_module.ExecutionProvider.SUBROSA,
        predicted_cost_usd=None,
        predicted_completion_seconds=60.0,
        configuration_id="42",
    )
    plan = SimpleNamespace(configurations_by_id={"42": configuration})

    metadata = execution_scheduling_module._configured_provider_metadata(
        execution_plan=plan,
        assignment=assignment,
        vast_leases_by_environment={},
    )

    assert metadata == {
        "relay_url": "wss://staging.subrosa.red",
        "pool": "mock-4090",
        "configuration_id": "42",
        "credential_id": "subrosa-default",
    }
    assert "srk_" not in json.dumps(metadata)


def test_subrosa_noop_sync_preserves_inputs(
    subrosa_sync_module: Any,
    settings_module: Any,
) -> None:
    """The mock milestone must skip uploads without rewriting model paths."""
    engine = subrosa_sync_module.subrosa_noop_sync_engine(
        settings_module.get_settings(),
        None,
    )
    inputs = {"ckpt_name": "models/checkpoint.safetensors", "seed": 7}

    rewritten, assets = engine.sync_prompt_inputs(inputs)

    assert rewritten == inputs
    assert assets == []
    assert engine.volume.exists("/anything") is True


def test_lane_zero_decoder_reassembles_arbitrary_chunks(
    subrosa_executor_module: Any,
    remote_protocol_module: Any,
) -> None:
    """WebSocket message boundaries must not affect CRMTRPC1 frame boundaries."""
    module = subrosa_executor_module
    protocol = remote_protocol_module
    stream = protocol.encode_json_frame(
        protocol.RemoteFrameKind.PROGRESS,
        {"kind": "progress", "value": 1},
    ) + protocol.encode_frame(protocol.RemoteFrameKind.RESULT, b"outputs")
    decoder = module._LaneZeroFrameDecoder()
    frames: list[tuple[Any, bytes]] = []

    for offset in range(0, len(stream), 3):
        frames.extend(decoder.push(stream[offset : offset + 3]))

    decoder.require_empty()
    assert [kind for kind, _payload in frames] == [
        protocol.RemoteFrameKind.PROGRESS,
        protocol.RemoteFrameKind.RESULT,
    ]
    assert frames[-1][1] == b"outputs"


def test_request_stream_chunks_to_cloudflare_safe_messages(
    subrosa_executor_module: Any,
) -> None:
    """Every outbound lane-0 message should stay at or below 512 KiB plus lane."""
    module = subrosa_executor_module
    websocket = _FakeWebSocket()
    client = module.SubrosaExecutorClient(
        credential_store=SimpleNamespace(),
        settings=SimpleNamespace(execution_timeout_seconds=60),
    )
    payload = {
        "invocation_id": "RIV_chunked",
        "execution_provider": "subrosa",
        "pool": "mock-4090",
    }

    asyncio.run(
        client._send_request_frames(
            websocket,
            payload,
            b"x" * (module._STREAM_CHUNK_BYTES * 2),
        )
    )

    assert len(websocket.sent) >= 3
    assert all(message[0] == 0 for message in websocket.sent)
    assert all(len(message) <= module._STREAM_CHUNK_BYTES + 1 for message in websocket.sent)


def test_response_streams_progress_then_surfaces_settlement(
    subrosa_executor_module: Any,
    remote_protocol_module: Any,
) -> None:
    """Progress should stream before a settled terminal RESULT is emitted."""
    module = subrosa_executor_module
    protocol = remote_protocol_module
    progress_frame = protocol.encode_json_frame(
        protocol.RemoteFrameKind.PROGRESS,
        {"kind": "progress", "phase": "executing", "value": 1},
    )
    result_frame = protocol.encode_frame(protocol.RemoteFrameKind.RESULT, b"outputs")
    websocket = _FakeWebSocket(
        [
            _binary_message(b"\x00" + progress_frame[:9]),
            _binary_message(b"\x00" + progress_frame[9:] + result_frame),
            _control_message(
                {
                    "type": "settled",
                    "status": "ok",
                    "gpu_seconds": 1.25,
                    "centicredits": 3,
                }
            ),
        ]
    )
    client = module.SubrosaExecutorClient(
        credential_store=SimpleNamespace(),
        settings=SimpleNamespace(execution_timeout_seconds=60),
    )
    events: list[dict[str, Any]] = []

    terminal, settlement = asyncio.run(
        client._receive_response(websocket, {}, events.append)
    )

    assert events == [{"kind": "progress", "phase": "executing", "value": 1}]
    assert terminal == {"kind": "result", "outputs": b"outputs"}
    assert settlement.status == "ok"
    assert settlement.centicredits == 3


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (
            {"failure_kind": "out_of_memory", "memory_current": 10},
            "SubrosaRemoteResourceError",
        ),
        (
            {"failure_kind": "worker_process_lost", "memory_current": 10},
            "SubrosaRemoteTransportError",
        ),
        (
            {"error_type": "ValueError", "message": "bad workflow"},
            "SubrosaRemoteInvocationError",
        ),
    ],
)
def test_error_classifier_accepts_both_worker_shapes(
    subrosa_executor_module: Any,
    error: dict[str, Any],
    expected_type: str,
) -> None:
    """Postmortem failures should preserve Vast's retry/resource semantics."""
    classified = subrosa_executor_module._subrosa_invocation_error(error)

    assert type(classified).__name__ == expected_type


def test_prepare_payload_preserves_queue_time_provider(
    subrosa_executor_module: Any,
) -> None:
    """The Subrosa client must not copy SSH's hard-coded provider-stamp bug."""
    client = subrosa_executor_module.SubrosaExecutorClient(
        credential_store=SimpleNamespace(),
    )

    prepared = client._prepare_payload(
        {"execution_provider": "subrosa", "component_id": "42"},
        b"inputs",
    )

    assert prepared["execution_provider"] == "subrosa"
    assert str(prepared["invocation_id"]).startswith("RIV_")


def test_run_relay_uses_stable_credential_id_instead_of_configuration_id(
    subrosa_executor_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch must query the human-stable reference, never the graph node ID."""
    module = subrosa_executor_module
    websocket = _FakeWebSocket()
    queried_ids: list[str] = []

    class CredentialStore:
        """Return a token only for the stable keyring account name."""

        def require(self, credential_id: str) -> str:
            """Record the exact keyring reference used by dispatch."""
            queried_ids.append(credential_id)
            if credential_id != "subrosa-default":
                raise AssertionError(f"Unexpected credential lookup {credential_id!r}")
            return "srk_test"

    class FakeSession:
        """Provide a context-managed fake relay connection."""

        def __init__(self, *, timeout: Any) -> None:
            """Accept the executor's client timeout."""
            del timeout

        async def __aenter__(self) -> Self:
            """Enter the fake client session."""
            return self

        async def __aexit__(self, *_args: object) -> None:
            """Exit the fake client session."""

        def ws_connect(self, *_args: Any, **_kwargs: Any) -> _FakeWebSocket:
            """Return the fake active relay socket."""
            return websocket

    client = module.SubrosaExecutorClient(
        credential_store=CredentialStore(),
        settings=SimpleNamespace(execution_timeout_seconds=60),
    )

    async def wait_until_ready(_websocket: Any) -> None:
        """Skip the already-covered ready handshake."""

    async def send_request_frames(
        _websocket: Any,
        _payload: Any,
        _inputs_payload: bytes,
    ) -> None:
        """Skip the already-covered request framing."""

    async def receive_response(
        _websocket: Any,
        _payload: Any,
        _emit: Any,
    ) -> tuple[dict[str, Any], None]:
        """Return one deterministic terminal response."""
        return {"kind": "result", "outputs": b"outputs"}, None

    monkeypatch.setattr(module.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(client, "_wait_until_ready", wait_until_ready)
    monkeypatch.setattr(client, "_send_request_frames", send_request_frames)
    monkeypatch.setattr(client, "_receive_response", receive_response)
    events: list[dict[str, Any]] = []

    asyncio.run(
        client._run_relay(
            {
                "invocation_id": "RIV_stable-credential",
                "relay_url": "wss://staging.subrosa.red",
                "pool": "mock-4090",
                "configuration_id": "42",
                "credential_id": "subrosa-default",
            },
            b"inputs",
            events.append,
        )
    )

    assert queried_ids == ["subrosa-default"]
    assert events == [{"kind": "result", "outputs": b"outputs"}]


def test_cancel_uses_the_active_relay_socket(
    subrosa_executor_module: Any,
) -> None:
    """Cancellation must be lane 1 on the invocation's existing WebSocket."""
    module = subrosa_executor_module
    websocket = _FakeWebSocket()
    client = module.SubrosaExecutorClient(credential_store=SimpleNamespace())

    async def exercise() -> bool:
        """Register the loop-owned socket and cancel from a worker thread."""
        client._active_relays["RIV_cancel"] = module._ActiveRelay(
            loop=asyncio.get_running_loop(),
            websocket=websocket,
        )
        return await asyncio.to_thread(client.cancel, "RIV_cancel")

    assert asyncio.run(exercise()) is True
    assert len(websocket.sent) == 1
    assert websocket.sent[0][0] == 1
    assert json.loads(websocket.sent[0][1:]) == {"type": "cancel"}


def test_whoami_uses_browser_user_agent_and_keyring_token(
    subrosa_executor_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REST validation should satisfy browser-integrity without exposing the token."""
    module = subrosa_executor_module
    observed: dict[str, Any] = {}

    class FakeResponse:
        """Return one successful whoami response."""

        status = 200

        async def __aenter__(self) -> Self:
            """Enter the response context."""
            return self

        async def __aexit__(self, *_args: object) -> None:
            """Exit the response context."""

        async def json(self, *, content_type: Any = None) -> dict[str, Any]:
            """Return safe account metadata."""
            del content_type
            return {"account_number": "acct_test", "balance": 100, "status": "active"}

    class FakeSession:
        """Capture the REST URL and headers."""

        def __init__(self, *, timeout: Any) -> None:
            """Record the configured timeout."""
            observed["timeout"] = timeout

        async def __aenter__(self) -> Self:
            """Enter the session context."""
            return self

        async def __aexit__(self, *_args: object) -> None:
            """Exit the session context."""

        def get(self, url: str, *, headers: Mapping[str, str]) -> FakeResponse:
            """Capture one request and return a successful response context."""
            observed["url"] = url
            observed["headers"] = dict(headers)
            return FakeResponse()

    monkeypatch.setattr(module.aiohttp, "ClientSession", FakeSession)
    client = module.SubrosaExecutorClient(
        credential_store=SimpleNamespace(require=lambda _credential_id: "srk_test")
    )

    account = asyncio.run(
        client.whoami("wss://staging.subrosa.red/", "staging-ref")
    )

    assert account["status"] == "active"
    assert observed["url"] == "https://staging.subrosa.red/api/v1/extension/whoami"
    assert observed["headers"]["Authorization"] == "Bearer srk_test"
    assert observed["headers"]["User-Agent"].startswith("Mozilla/5.0")


def test_router_constructs_subrosa_client(
    remote_executor_router_module: Any,
    settings_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider router should branch to Subrosa before SSH fallback."""
    sentinel = object()
    monkeypatch.setattr(settings_module, "get_settings", lambda: sentinel)

    client = remote_executor_router_module.RemoteExecutorRouterClient()._client_for_payload(
        {"execution_provider": "subrosa"}
    )

    assert type(client).__name__ == "SubrosaExecutorClient"
    assert client.settings is sentinel


def test_extension_registers_subrosa_v3_node(
    extension_package: Any,
    remote_configuration_nodes_module: Any,
) -> None:
    """The ComfyUI extension entrypoint should export the Subrosa node class."""
    assert (
        extension_package.SubrosaConfiguration
        is remote_configuration_nodes_module.SubrosaConfiguration
    )
