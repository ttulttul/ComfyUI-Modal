"""Tests for the Modal hosted-model endpoint node."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Self

import pytest
import torch


class FakeKeyringError(Exception):
    """Error raised by the fake keyring backend."""


class FakeKeyring:
    """Small in-memory implementation of the keyring API used by the node."""

    errors = type("Errors", (), {"KeyringError": FakeKeyringError})

    def __init__(self) -> None:
        """Create an empty fake credential vault."""
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        """Return one fake secret value."""
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        """Store one fake secret value."""
        self.values[(service, username)] = value

    def delete_password(self, service: str, username: str) -> None:
        """Delete one fake secret value."""
        self.values.pop((service, username), None)


@dataclass
class FakeInputFile:
    """Pydantic-like file input used by the built-in OpenAI file node."""

    filename: str
    file_data: str

    def model_dump(self, exclude_none: bool = True) -> dict[str, str]:
        """Return the fields exposed by ComfyUI's InputFileContent model."""
        del exclude_none
        return {
            "type": "input_file",
            "filename": self.filename,
            "file_data": self.file_data,
        }


class FakeResponseContent:
    """Async byte stream for a fake aiohttp response."""

    def __init__(self, payload: dict[str, Any] | bytes) -> None:
        """Serialize one response object."""
        self._body = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload).encode("utf-8")
        )

    async def iter_chunked(self, chunk_size: int) -> Any:
        """Yield the serialized body in bounded pieces."""
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]


class FakeResponse:
    """Async context manager matching the aiohttp response surface used by the client."""

    def __init__(
        self, payload: dict[str, Any] | bytes, status: int = 200
    ) -> None:
        """Configure the response JSON and status code."""
        self.status = status
        self.content = FakeResponseContent(payload)

    async def __aenter__(self) -> Self:
        """Enter the fake response context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Leave the fake response context."""
        del exc_info


class FakeSession:
    """Capture one aiohttp-style request call."""

    def __init__(self, response: FakeResponse | list[FakeResponse]) -> None:
        """Configure one or more responses returned by request calls."""
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        """Record and return one request context manager."""
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def _credentials(modal_endpoint_module: Any) -> Any:
    """Return a syntactically valid test proxy-token pair."""
    return modal_endpoint_module.ModalProxyCredentials(
        key="wk-test-key",
        secret="ws-test-secret",
    )


def test_endpoint_url_normalization_accepts_modal_direct_suffixes(
    modal_endpoint_module: Any,
) -> None:
    """Accept the endpoint forms users are likely to paste from Modal."""
    endpoint = "https://example--model.us-west.modal.direct"

    assert modal_endpoint_module._normalize_endpoint_url(endpoint) == endpoint
    assert modal_endpoint_module._normalize_endpoint_url(f"{endpoint}/v1") == endpoint
    assert (
        modal_endpoint_module._normalize_endpoint_url(f"{endpoint}/v1/chat/completions")
        == endpoint
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example--model.us-west.modal.direct",
        "https://example.com",
        "https://example--model.us-west.modal.direct@attacker.example",
        "https://example--model.us-west.modal.direct/other",
        "https://example--model.us-west.modal.direct?next=https://attacker.example",
    ],
)
def test_endpoint_url_normalization_rejects_credential_exfiltration_targets(
    modal_endpoint_module: Any,
    endpoint: str,
) -> None:
    """Never attach Modal credentials to a noncanonical destination."""
    with pytest.raises(ValueError):
        modal_endpoint_module._normalize_endpoint_url(endpoint)


def test_secret_manager_round_trips_credentials(modal_endpoint_module: Any) -> None:
    """Persist both proxy-token values under the ComfyUI-specific vault service."""
    keyring = FakeKeyring()
    manager = modal_endpoint_module.ComfyUISecretManager(keyring)
    credentials = _credentials(modal_endpoint_module)

    assert manager.load() is None
    manager.ensure_writable()
    assert keyring.values == {}
    manager.save(credentials)

    assert manager.load() == credentials
    assert set(keyring.values.values()) == {credentials.key, credentials.secret}


def test_secret_manager_rejects_partial_pair(modal_endpoint_module: Any) -> None:
    """Do not create a replacement token over a damaged stored pair."""
    keyring = FakeKeyring()
    manager = modal_endpoint_module.ComfyUISecretManager(keyring)
    keyring.set_password("ComfyUI Modal-Sync", "MODAL_KEY", "wk-only")

    with pytest.raises(RuntimeError, match="only one half"):
        manager.load()


def test_credential_resolver_prefers_environment(
    modal_endpoint_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use explicit environment credentials without reading or writing the vault."""
    credentials = _credentials(modal_endpoint_module)
    monkeypatch.setenv("MODAL_KEY", credentials.key)
    monkeypatch.setenv("MODAL_SECRET", credentials.secret)

    class Unused:
        """Fail when an environment-only resolution touches a fallback."""

        def __getattr__(self, name: str) -> Any:
            """Reject every fallback operation."""
            raise AssertionError(f"unexpected fallback access: {name}")

    resolver = modal_endpoint_module.ModalCredentialResolver(Unused(), Unused())

    assert resolver.resolve() == credentials


def test_credential_resolver_creates_and_stores_missing_pair(
    modal_endpoint_module: Any,
) -> None:
    """Create exactly one pair when neither environment nor vault has credentials."""
    credentials = _credentials(modal_endpoint_module)

    class Store:
        """Record credential-store interactions."""

        saved: Any = None

        def load(self) -> None:
            """Report an empty vault."""

        def save(self, value: Any) -> None:
            """Record the newly created credential pair."""
            self.saved = value

        def ensure_writable(self) -> None:
            """Confirm the fake store can accept the new token."""

    class Creator:
        """Return one deterministic proxy token."""

        calls = 0

        def create(self) -> Any:
            """Create one test pair."""
            self.calls += 1
            return credentials

    store = Store()
    creator = Creator()

    assert (
        modal_endpoint_module.ModalCredentialResolver(store, creator).resolve()
        == credentials
    )
    assert creator.calls == 1
    assert store.saved == credentials


def test_credential_resolver_authorizes_vault_credentials(
    modal_endpoint_module: Any,
) -> None:
    """Authorize node-managed scoped tokens for the endpoint environment."""
    credentials = _credentials(modal_endpoint_module)

    class Store:
        """Return a credential pair previously created by the node."""

        def load(self) -> Any:
            """Return stored credentials."""
            return credentials

    class Authorizer:
        """Record the environment association request."""

        def __init__(self) -> None:
            """Create an empty authorization call log."""
            self.calls: list[tuple[str, str]] = []

        def allow(self, token_key: str, environment: str) -> None:
            """Record one authorization operation."""
            self.calls.append((token_key, environment))

    authorizer = Authorizer()
    resolver = modal_endpoint_module.ModalCredentialResolver(
        Store(),
        object(),
        authorizer=authorizer,
        environment="ComfyUI",
    )

    assert resolver.resolve() == credentials
    assert authorizer.calls == [(credentials.key, "ComfyUI")]


def test_credential_resolver_saves_new_token_before_authorization(
    modal_endpoint_module: Any,
) -> None:
    """Keep the one-time secret recoverable when environment authorization fails."""
    credentials = _credentials(modal_endpoint_module)
    events: list[str] = []

    class Store:
        """Record persistence ordering for a new token."""

        def load(self) -> None:
            """Report no stored credentials."""

        def ensure_writable(self) -> None:
            """Accept the credential-vault probe."""

        def save(self, value: Any) -> None:
            """Record persistence of the new token."""
            assert value == credentials
            events.append("saved")

    class Creator:
        """Return one deterministic one-time token pair."""

        def create(self) -> Any:
            """Return generated credentials."""
            return credentials

    class Authorizer:
        """Fail after verifying the secret was persisted."""

        def allow(self, token_key: str, environment: str) -> None:
            """Simulate an RBAC association failure."""
            assert token_key == credentials.key
            assert environment == "main"
            assert events == ["saved"]
            raise RuntimeError("authorization failed")

    resolver = modal_endpoint_module.ModalCredentialResolver(
        Store(), Creator(), authorizer=Authorizer()
    )

    with pytest.raises(RuntimeError, match="authorization failed"):
        resolver.resolve()
    assert events == ["saved"]


def test_credential_resolver_checks_vault_before_creating_token(
    modal_endpoint_module: Any,
) -> None:
    """Do not mint a one-time proxy token when secure persistence is unavailable."""

    class Store:
        """Represent a readable but non-writable credential vault."""

        def load(self) -> None:
            """Report no stored pair."""

        def ensure_writable(self) -> None:
            """Reject writes before token creation begins."""
            raise RuntimeError("vault unavailable")

    class Creator:
        """Fail if token creation is attempted."""

        def create(self) -> Any:
            """Reject an unsafe token-creation attempt."""
            raise AssertionError("token creation should not run")

    with pytest.raises(RuntimeError, match="vault unavailable"):
        modal_endpoint_module.ModalCredentialResolver(Store(), Creator()).resolve()


def test_cli_creator_parses_json_response(modal_endpoint_module: Any) -> None:
    """Parse Modal's documented --json response without exposing values elsewhere."""
    output = json.dumps({"Modal-Key": "wk-created", "Modal-Secret": "ws-created"})

    assert modal_endpoint_module.ModalCliProxyTokenCreator._parse_credentials(
        output
    ) == (modal_endpoint_module.ModalProxyCredentials("wk-created", "ws-created"))


def test_cli_creator_falls_back_from_outdated_modal(
    modal_endpoint_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use uvx when an installed Modal CLI predates workspace proxy-token commands."""
    creator = modal_endpoint_module.ModalCliProxyTokenCreator()
    commands = [["old-modal"], ["uvx-modal"]]
    results = iter(
        [
            subprocess.CompletedProcess(
                commands[0],
                2,
                stdout="",
                stderr="No such command 'workspace'.",
            ),
            subprocess.CompletedProcess(
                commands[1],
                0,
                stdout=json.dumps({"Modal-Key": "wk-new", "Modal-Secret": "ws-new"}),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(creator, "_candidate_commands", lambda: commands)
    monkeypatch.setattr(creator, "_run_command", lambda command: next(results))

    assert creator.create() == modal_endpoint_module.ModalProxyCredentials(
        "wk-new", "ws-new"
    )


def test_cli_authorizer_falls_back_and_caches_success(
    modal_endpoint_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a current CLI once for a scoped token and avoid per-request CLI latency."""
    authorizer = modal_endpoint_module.ModalCliProxyTokenAuthorizer()
    commands = [["old-modal"], ["uvx-modal"]]
    results = iter(
        [
            subprocess.CompletedProcess(
                commands[0],
                2,
                stdout="",
                stderr="No such command 'workspace'.",
            ),
            subprocess.CompletedProcess(commands[1], 0, stdout="allowed", stderr=""),
        ]
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        authorizer,
        "_candidate_commands",
        lambda token_key, environment: commands,
    )

    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        """Record one fake CLI invocation."""
        calls.append(command)
        return next(results)

    monkeypatch.setattr(
        modal_endpoint_module.ModalCliProxyTokenCreator,
        "_run_command",
        run_command,
    )

    authorizer.allow("wk-authorizer-test", "main")
    authorizer.allow("wk-authorizer-test", "main")

    assert calls == commands


def test_image_batch_becomes_inline_png_content_parts(
    modal_endpoint_module: Any,
) -> None:
    """Encode every BHWC tensor item as an OpenAI-compatible image_url block."""
    images = torch.zeros((2, 3, 4, 3), dtype=torch.float32)

    content_parts = modal_endpoint_module._image_content_parts(images)

    assert len(content_parts) == 2
    data_uri = content_parts[0]["image_url"]["url"]
    assert data_uri.startswith("data:image/png;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")


def test_builtin_openai_file_input_becomes_chat_file_part(
    modal_endpoint_module: Any,
) -> None:
    """Accept the existing OpenAI ChatGPT Input Files node output directly."""
    file_input = FakeInputFile(
        filename="notes.txt",
        file_data="data:text/plain;base64,SGVsbG8=",
    )

    assert modal_endpoint_module._file_content_part(file_input) == {
        "type": "file",
        "file": {
            "filename": "notes.txt",
            "file_data": "data:text/plain;base64,SGVsbG8=",
        },
    }


def test_request_payload_matches_chat_completions_contract(
    modal_endpoint_module: Any,
) -> None:
    """Build system and multimodal user messages with explicit generation options."""
    payload = modal_endpoint_module.ModalEndpointClient._request_payload(
        prompt="Describe this",
        model="org/model",
        images=torch.zeros((1, 2, 2, 3)),
        files=[FakeInputFile("notes.txt", "data:text/plain;base64,SGVsbG8=")],
        system_prompt="Be concise",
        max_tokens=123,
        temperature=0.25,
    )

    assert payload["model"] == "org/model"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 123
    assert payload["temperature"] == 0.25
    assert payload["messages"][0] == {"role": "system", "content": "Be concise"}
    assert [part["type"] for part in payload["messages"][1]["content"]] == [
        "text",
        "image_url",
        "file",
    ]


def test_http_client_authenticates_without_following_redirects(
    modal_endpoint_module: Any,
) -> None:
    """Attach both proxy-token headers only to the validated canonical endpoint URL."""
    credentials = _credentials(modal_endpoint_module)
    client = modal_endpoint_module.ModalEndpointClient(
        "https://example--model.us-west.modal.direct/v1",
        credentials,
        timeout_seconds=30,
    )
    session = FakeSession(FakeResponse({"data": []}))

    response = asyncio.run(client._request_json(session, "GET", "/v1/models"))

    assert response == {"data": []}
    assert session.calls == [
        {
            "method": "GET",
            "url": "https://example--model.us-west.modal.direct/v1/models",
            "headers": {
                "Content-Type": "application/json",
                "Modal-Key": credentials.key,
                "Modal-Secret": credentials.secret,
            },
            "json": None,
            "allow_redirects": False,
        }
    ]


def test_endpoint_error_message_redacts_modal_proxy_tokens(
    modal_endpoint_module: Any,
) -> None:
    """Do not repeat credential identifiers from provider errors into ComfyUI logs."""
    detail = modal_endpoint_module._response_error_message(
        {"error": "Webhook token not found: wk-sensitive-token"}
    )

    assert detail == '{"error": "Webhook token not found: [redacted]"}'


def test_http_client_retries_empty_503_until_modal_replica_is_ready(
    modal_endpoint_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat Modal Server's empty 503 as a cold-start signal, not malformed JSON."""
    client = modal_endpoint_module.ModalEndpointClient(
        "https://example--model.us-west.modal.direct",
        _credentials(modal_endpoint_module),
        timeout_seconds=30,
    )
    session = FakeSession(
        [
            FakeResponse(b"", status=503),
            FakeResponse({"data": [{"id": "org/model"}]}),
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        """Record retry backoff without delaying the test."""
        delays.append(delay)

    monkeypatch.setattr(modal_endpoint_module.asyncio, "sleep", record_sleep)

    response = asyncio.run(client._request_json(session, "GET", "/v1/models"))

    assert response == {"data": [{"id": "org/model"}]}
    assert delays == [1.0]
    assert len(session.calls) == 2


def test_http_client_reports_status_for_empty_nonretryable_response(
    modal_endpoint_module: Any,
) -> None:
    """Preserve the HTTP status when an endpoint error has no JSON body."""
    client = modal_endpoint_module.ModalEndpointClient(
        "https://example--model.us-west.modal.direct",
        _credentials(modal_endpoint_module),
        timeout_seconds=30,
    )
    session = FakeSession(FakeResponse(b"", status=502))

    with pytest.raises(RuntimeError, match="HTTP 502 with an empty response body"):
        asyncio.run(client._request_json(session, "GET", "/v1/models"))


def test_http_client_does_not_retry_application_json_503(
    modal_endpoint_module: Any,
) -> None:
    """Surface an application-level 503 body instead of treating it as scale-up."""
    client = modal_endpoint_module.ModalEndpointClient(
        "https://example--model.us-west.modal.direct",
        _credentials(modal_endpoint_module),
        timeout_seconds=30,
    )
    session = FakeSession(
        FakeResponse({"error": {"message": "model failed"}}, status=503)
    )

    with pytest.raises(RuntimeError, match="HTTP 503: model failed"):
        asyncio.run(client._request_json(session, "GET", "/v1/models"))
    assert len(session.calls) == 1


def test_http_client_bounds_503_retries_by_total_timeout(
    modal_endpoint_module: Any,
) -> None:
    """Stop cold-start polling at the node's total request deadline."""
    client = modal_endpoint_module.ModalEndpointClient(
        "https://example--model.us-west.modal.direct",
        _credentials(modal_endpoint_module),
        timeout_seconds=0.01,
    )
    session = FakeSession(FakeResponse(b"", status=503))

    with pytest.raises(TimeoutError, match="stayed unavailable with HTTP 503"):
        asyncio.run(client._request_json(session, "GET", "/v1/models"))


def test_http_client_reports_bounded_non_json_error_body(
    modal_endpoint_module: Any,
) -> None:
    """Include safe upstream text when an endpoint does not return JSON."""
    client = modal_endpoint_module.ModalEndpointClient(
        "https://example--model.us-west.modal.direct",
        _credentials(modal_endpoint_module),
        timeout_seconds=30,
    )
    session = FakeSession(FakeResponse(b"upstream unavailable", status=502))

    with pytest.raises(
        RuntimeError,
        match="HTTP 502 with a non-JSON response: upstream unavailable",
    ):
        asyncio.run(client._request_json(session, "GET", "/v1/models"))


def test_chat_response_text_supports_string_and_part_lists(
    modal_endpoint_module: Any,
) -> None:
    """Extract common OpenAI-compatible assistant content representations."""
    assert (
        modal_endpoint_module._chat_response_text(
            {"choices": [{"message": {"content": "hello"}}]}
        )
        == "hello"
    )
    assert (
        modal_endpoint_module._chat_response_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ]
                        }
                    }
                ]
            }
        )
        == "hello world"
    )


def test_node_schema_matches_builtin_chat_shape(modal_endpoint_module: Any) -> None:
    """Expose prompt, image, file, endpoint, and model controls as a V3 node."""
    schema = modal_endpoint_module.ModalEndpointChat.define_schema()
    input_ids = [input_spec.id for input_spec in schema.inputs]

    assert schema.node_id == "ModalEndpointChat"
    assert schema.display_name == "Modal Endpoint Chat"
    assert schema.essentials_category == "Text Generation"
    assert input_ids[:5] == ["prompt", "endpoint_url", "model", "images", "files"]
    assert schema.outputs[0].io_type == "STRING"


def test_extension_exports_endpoint_node(
    extension_package: Any, modal_endpoint_module: Any
) -> None:
    """Register the endpoint node through the package's ComfyUI V3 entrypoint module."""
    assert (
        extension_package.ModalEndpointChat is modal_endpoint_module.ModalEndpointChat
    )


def test_node_execute_returns_endpoint_text(
    modal_endpoint_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire credential resolution and the async client into a standard node output."""
    credentials = _credentials(modal_endpoint_module)

    class Resolver:
        """Return deterministic credentials."""

        def __init__(
            self,
            store: Any,
            creator: Any,
            authorizer: Any,
            environment: str,
        ) -> None:
            """Accept production constructor dependencies."""
            del store, creator
            assert authorizer is not None
            assert environment == "main"

        def resolve(self) -> Any:
            """Return the test credential pair."""
            return credentials

    class Client:
        """Return a deterministic endpoint completion."""

        def __init__(
            self, endpoint_url: str, supplied_credentials: Any, timeout: int
        ) -> None:
            """Record the validated constructor arguments."""
            assert endpoint_url.endswith("modal.direct")
            assert supplied_credentials == credentials
            assert timeout == 30
            self._origin = endpoint_url

        @property
        def endpoint_hostname(self) -> str:
            """Return the fake endpoint hostname."""
            return "example--model.us-west.modal.direct"

        async def complete(self, **kwargs: Any) -> str:
            """Return one fake assistant response."""
            assert kwargs["prompt"] == "hello"
            return "modal reply"

    monkeypatch.setattr(modal_endpoint_module, "ComfyUISecretManager", lambda: object())
    monkeypatch.setattr(
        modal_endpoint_module, "ModalCliProxyTokenCreator", lambda: object()
    )
    monkeypatch.setattr(
        modal_endpoint_module, "ModalCliProxyTokenAuthorizer", lambda: object()
    )
    monkeypatch.setattr(modal_endpoint_module, "ModalCredentialResolver", Resolver)
    monkeypatch.setattr(modal_endpoint_module, "ModalEndpointClient", Client)

    output = asyncio.run(
        modal_endpoint_module.ModalEndpointChat.execute(
            prompt="hello",
            endpoint_url="https://example--model.us-west.modal.direct",
            timeout_seconds=30,
        )
    )

    assert output.result == ("modal reply",)
