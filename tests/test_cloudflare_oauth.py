"""Tests for Cloudflare OAuth PKCE and bucket-scoped R2 provisioning."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

class CapturingCredentialStore:
    """Capture credential records persisted after manual import."""

    def __init__(self) -> None:
        """Initialize captured writes."""
        self.saved: list[tuple[str, Any]] = []

    def save(self, credential_id: str, record: Any) -> None:
        """Capture one credential write."""
        self.saved.append((credential_id, record))

    def load(self, credential_id: str) -> None:
        """Report that no credential is currently stored."""
        del credential_id
        return None


class JsonRequest:
    """Provide the JSON request surface used by route helpers."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store one request payload."""
        self.payload = payload

    async def json(self) -> dict[str, Any]:
        """Return the stored request payload."""
        return self.payload


def test_authorization_url_uses_public_client_pkce(
    cloudflare_oauth_module: Any,
) -> None:
    """Desktop Login must use S256 PKCE and never require a client secret."""
    service = cloudflare_oauth_module.CloudflareR2OAuthService(
        cloudflare_oauth_module.CloudflareOAuthConfiguration(client_id="public-client")
    )

    url = service.authorization_url(
        node_id="20",
        credential_id="opaque-id",
        bucket="models",
        jurisdiction="default",
        requested_account_id=None,
        request_origin="http://127.0.0.1:8188",
    )

    query = parse_qs(urlparse(url).query)
    assert query["client_id"] == ["public-client"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["redirect_uri"] == [
        "http://127.0.0.1:8188/remote/storage/r2/oauth/callback"
    ]
    assert "client_secret" not in query
    state = service.state_store.consume(query["state"][0])
    assert state.credential_id == "opaque-id"
    assert state.bucket == "models"
    assert len(query["code_challenge"][0]) == 43
    assert query["scope"] == ["account-settings.read workers-r2.write"]


def test_complete_provisions_bucket_without_attempting_token_creation(
    cloudflare_oauth_module: Any,
) -> None:
    """OAuth completion should stop after bucket provisioning and request import."""
    credential_store = CapturingCredentialStore()

    class Service(cloudflare_oauth_module.CloudflareR2OAuthService):
        """Provide deterministic completion steps without external requests."""

        async def _exchange_code(self, state: Any, code: str) -> str:
            """Return a synthetic OAuth bearer token."""
            assert state.code_verifier
            assert code == "authorization-code"
            return "oauth-access-token"

        async def _select_account(self, access_token: str, state: Any) -> str:
            """Select one deterministic account."""
            del state
            assert access_token == "oauth-access-token"
            return "a" * 32

        async def _ensure_bucket(
            self,
            access_token: str,
            account_id: str,
            state: Any,
        ) -> None:
            """Validate the bucket provisioning request."""
            assert access_token == "oauth-access-token"
            assert account_id == "a" * 32
            assert state.bucket == "models"

    service = Service(
        cloudflare_oauth_module.CloudflareOAuthConfiguration(client_id="public-client"),
        credential_store=credential_store,
    )
    authorization_url = service.authorization_url(
        node_id="20",
        credential_id="opaque-id",
        bucket="models",
        jurisdiction="default",
        requested_account_id=None,
        request_origin="http://127.0.0.1:8188",
    )
    state_token = parse_qs(urlparse(authorization_url).query)["state"][0]

    result = asyncio.run(
        service.complete(state_token=state_token, code="authorization-code")
    )

    assert result["account_id"] == "a" * 32
    assert result["credentials_required"] is True
    assert credential_store.saved == []


def test_import_validates_and_saves_user_created_r2_credentials(
    cloudflare_oauth_module: Any,
    monkeypatch: Any,
) -> None:
    """Credential import should verify bucket access before touching the keyring."""
    store = CapturingCredentialStore()
    validated: list[Any] = []
    monkeypatch.setattr(
        cloudflare_oauth_module,
        "validate_r2_credentials",
        validated.append,
    )
    request = JsonRequest(
        {
            "credential_id": "opaque-id",
            "account_id": "a" * 32,
            "bucket": "models",
            "jurisdiction": "eu",
            "access_key_id": "manual-access-key",
            "secret_access_key": "manual-secret-key",
        }
    )

    response = asyncio.run(
        cloudflare_oauth_module._import_r2_credentials_response(request, store)
    )

    assert response.status == 200
    assert json.loads(response.text)["connected"] is True
    assert validated[0].bucket == "models"
    assert store.saved[0][0] == "opaque-id"
    assert store.saved[0][1].access_key_id == "manual-access-key"
    assert "manual-secret-key" not in response.text


def test_import_does_not_save_rejected_credentials(
    cloudflare_oauth_module: Any,
    monkeypatch: Any,
) -> None:
    """Rejected S3 credentials must not replace a previously stored credential."""
    store = CapturingCredentialStore()

    def reject(record: Any) -> None:
        """Reject the synthetic credential without exposing either key."""
        del record
        raise cloudflare_oauth_module.R2CredentialError("Cloudflare rejected access.")

    monkeypatch.setattr(cloudflare_oauth_module, "validate_r2_credentials", reject)
    request = JsonRequest(
        {
            "credential_id": "opaque-id",
            "account_id": "a" * 32,
            "bucket": "models",
            "jurisdiction": "default",
            "access_key_id": "bad-access-key",
            "secret_access_key": "bad-secret-key",
        }
    )

    response = asyncio.run(
        cloudflare_oauth_module._import_r2_credentials_response(request, store)
    )

    assert response.status == 400
    assert store.saved == []
    assert "bad-access-key" not in response.text
    assert "bad-secret-key" not in response.text
