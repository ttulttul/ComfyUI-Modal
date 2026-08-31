"""Tests for Subrosa Login validation, keyring import, and status routes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import pytest


@dataclass
class _CredentialStore:
    """Keep synthetic extension tokens in memory."""

    values: dict[str, str] = field(default_factory=dict)

    def save(self, credential_id: str, token: str) -> None:
        """Store one token by reference."""
        self.values[credential_id] = token

    def load(self, credential_id: str) -> str | None:
        """Load one token by reference."""
        return self.values.get(credential_id)


class _Response:
    """Return one configured whoami response."""

    def __init__(self, status: int, body: Any) -> None:
        """Retain HTTP status and JSON payload."""
        self.status = status
        self.body = body

    async def __aenter__(self) -> Self:
        """Enter the response context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the response context."""

    async def json(self, *, content_type: Any = None) -> Any:
        """Return the configured JSON body."""
        del content_type
        return self.body


class _Session:
    """Capture validation requests and return one fake response."""

    response: ClassVar[_Response] = _Response(
        200,
        {"account_number": "0000-TEST", "balance": 500, "status": "active"},
    )
    requests: ClassVar[list[tuple[str, dict[str, str]]]] = []

    def __init__(self, *, timeout: Any) -> None:
        """Accept the production timeout."""
        self.timeout = timeout

    async def __aenter__(self) -> Self:
        """Enter the session context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the session context."""

    def get(self, url: str, *, headers: dict[str, str]) -> _Response:
        """Capture the credential-safe request metadata."""
        self.requests.append((url, dict(headers)))
        return self.response


@dataclass
class _JsonRequest:
    """Provide an aiohttp-like JSON request body."""

    payload: dict[str, Any]

    async def json(self) -> dict[str, Any]:
        """Return the configured body."""
        return dict(self.payload)


def test_token_validation_uses_whoami_and_browser_user_agent(
    subrosa_login_module: Any,
) -> None:
    """Login validation should authenticate with the exact portal-minted token."""
    _Session.requests = []
    _Session.response = _Response(
        200,
        {"account_number": "0000-TEST", "balance": 500, "status": "active"},
    )

    account = asyncio.run(
        subrosa_login_module.validate_subrosa_token(
            "wss://staging.subrosa.red/path",
            "srk_test-token",
            session_factory=_Session,
        )
    )

    assert account["account_number"] == "0000-TEST"
    url, headers = _Session.requests[0]
    assert url == "https://staging.subrosa.red/api/v1/extension/whoami"
    assert headers["Authorization"] == "Bearer srk_test-token"
    assert headers["User-Agent"].startswith("Mozilla/5.0")


def test_rejected_token_uses_concise_login_again_error(
    subrosa_login_module: Any,
) -> None:
    """HTTP 401 must point users back to the node's Login button."""
    _Session.response = _Response(401, {"error": "invalid token"})

    with pytest.raises(
        subrosa_login_module.SubrosaLoginRequiredError,
        match="Click.*Login to Subrosa",
    ):
        asyncio.run(
            subrosa_login_module.validate_subrosa_token(
                "wss://staging.subrosa.red",
                "srk_rejected",
                session_factory=_Session,
            )
        )


def test_token_validation_rejects_an_untrusted_relay_before_sending(
    subrosa_login_module: Any,
) -> None:
    """A workflow-supplied relay must not receive a saved Subrosa token."""
    _Session.requests = []

    with pytest.raises(
        subrosa_login_module.SubrosaLoginError,
        match="only sends extension tokens",
    ):
        asyncio.run(
            subrosa_login_module.validate_subrosa_token(
                "wss://attacker.example",
                "srk_saved-secret",
                session_factory=_Session,
            )
        )

    assert _Session.requests == []


def test_import_validates_before_saving_and_never_returns_token(
    subrosa_login_module: Any,
) -> None:
    """The local route should persist only a remotely accepted token."""
    _Session.response = _Response(
        200,
        {"account_number": "0000-TEST", "balance": 500, "status": "active"},
    )
    store = _CredentialStore()
    service = subrosa_login_module.SubrosaLoginService(
        credential_store=store,
        session_factory=_Session,
    )
    request = _JsonRequest(
        {
            "credential_id": "subrosa-default",
            "relay_url": "wss://staging.subrosa.red",
            "token": "srk_one-time-secret",
        }
    )

    response = asyncio.run(
        subrosa_login_module._import_subrosa_token_response(request, service)
    )

    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert store.values["subrosa-default"] == "srk_one-time-secret"
    assert "srk_one-time-secret" not in response.text
    assert json.loads(response.text)["connected"] is True


def test_status_reports_invalid_saved_token_as_disconnected(
    subrosa_login_module: Any,
) -> None:
    """The node should offer Login again when its saved token is rejected."""
    _Session.response = _Response(401, {"error": "invalid token"})
    service = subrosa_login_module.SubrosaLoginService(
        credential_store=_CredentialStore(
            {"subrosa-default": "srk_expired"}
        ),
        session_factory=_Session,
    )

    status = asyncio.run(
        service.status(
            credential_id="subrosa-default",
            relay_url="wss://staging.subrosa.red",
        )
    )

    assert status == {
        "connected": False,
        "reason": "subrosa_login_required",
    }
