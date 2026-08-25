"""Tests for Cloudflare OAuth PKCE and bucket-scoped R2 provisioning."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest


class CapturingCredentialStore:
    """Capture the credential record persisted after OAuth completes."""

    def __init__(self) -> None:
        """Initialize captured writes."""
        self.saved: list[tuple[str, Any]] = []

    def save(self, credential_id: str, record: Any) -> None:
        """Capture one credential write."""
        self.saved.append((credential_id, record))

    def load(self, credential_id: str) -> None:
        """Report that no previous credential needs revocation."""
        del credential_id
        return None


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


def test_r2_token_response_becomes_s3_signing_credentials(
    cloudflare_oauth_module: Any,
) -> None:
    """Cloudflare token ID/value should map to R2's S3 access/secret pair."""
    calls: list[tuple[str, str, Any]] = []

    class Service(cloudflare_oauth_module.CloudflareR2OAuthService):
        """Return deterministic Cloudflare API responses."""

        async def _cloudflare_api(
            self,
            method: str,
            path: str,
            *,
            access_token: str,
            headers: Any = None,
            json_payload: Any = None,
        ) -> Any:
            """Capture API calls and return permission/token fixtures."""
            del access_token, headers
            calls.append((method, path, json_payload))
            if path == "/user/tokens/permission_groups":
                return [{"id": "permission-id", "name": "Workers R2 Storage Bucket Item Write"}]
            return {"id": "access-key-id", "value": "one-time-token-value"}

    service = Service(
        cloudflare_oauth_module.CloudflareOAuthConfiguration(client_id="public-client")
    )
    state = cloudflare_oauth_module.R2OAuthState(
        node_id="20",
        credential_id="opaque-id",
        bucket="models",
        jurisdiction="eu",
        requested_account_id=None,
        redirect_uri="http://127.0.0.1/callback",
        code_verifier="verifier",
    )

    record = asyncio.run(
        service._create_bucket_credential("oauth-access", "a" * 32, state)
    )

    assert record.access_key_id == "access-key-id"
    assert record.secret_access_key == hashlib.sha256(
        b"one-time-token-value"
    ).hexdigest()
    create_payload = calls[-1][2]
    resources = create_payload["policies"][0]["resources"]
    assert resources == {
        f"com.cloudflare.edge.r2.bucket.{'a' * 32}_eu_models": "*"
    }
    assert "one-time-token-value" not in str(calls)


def test_complete_persists_only_the_r2_credential(
    cloudflare_oauth_module: Any,
) -> None:
    """OAuth completion should finish provisioning before writing the keyring record."""
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

        async def _create_bucket_credential(
            self,
            access_token: str,
            account_id: str,
            state: Any,
        ) -> Any:
            """Return the final bucket-scoped credential."""
            del access_token, state
            return cloudflare_oauth_module.R2CredentialRecord(
                account_id=account_id,
                bucket="models",
                access_key_id="access-id",
                secret_access_key="secret-key",
            )

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
    assert credential_store.saved[0][0] == "opaque-id"
    record = credential_store.saved[0][1]
    assert record.access_key_id == "access-id"
    assert record.secret_access_key == "secret-key"


def test_failed_keyring_write_revokes_new_cloudflare_token(
    cloudflare_oauth_module: Any,
) -> None:
    """A local vault failure must not leave a newly created durable token behind."""
    revoked_paths: list[str] = []

    class Store:
        """Reject every secure persistence attempt."""

        def save(self, credential_id: str, record: Any) -> None:
            """Simulate an unavailable OS credential vault."""
            del credential_id, record
            raise cloudflare_oauth_module.R2CredentialError("keyring unavailable")

    class Service(cloudflare_oauth_module.CloudflareR2OAuthService):
        """Capture cleanup requests without external traffic."""

        async def _cloudflare_api(
            self,
            method: str,
            path: str,
            *,
            access_token: str,
            headers: Any = None,
            json_payload: Any = None,
        ) -> dict[str, Any]:
            """Capture the exact token deletion request."""
            del headers, json_payload
            assert method == "DELETE"
            assert access_token == "oauth-token"
            revoked_paths.append(path)
            return {}

    service = Service(
        cloudflare_oauth_module.CloudflareOAuthConfiguration(client_id="public-client"),
        credential_store=Store(),
    )
    record = cloudflare_oauth_module.R2CredentialRecord(
        account_id="a" * 32,
        bucket="models",
        access_key_id="new-access-id",
        secret_access_key="new-secret",
    )

    with pytest.raises(cloudflare_oauth_module.R2CredentialError):
        asyncio.run(
            service._save_credential_or_revoke(
                "oauth-token",
                "opaque-id",
                record,
            )
        )

    assert revoked_paths == ["/user/tokens/new-access-id"]
