"""Cloudflare OAuth bootstrap and R2 bucket provisioning routes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

import aiohttp
from aiohttp import web

if __package__:
    from .r2_credentials import (
        R2CredentialError,
        R2CredentialRecord,
        R2CredentialStore,
    )
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from r2_credentials import (
        R2CredentialError,
        R2CredentialRecord,
        R2CredentialStore,
    )

logger = logging.getLogger(__name__)

CLOUDFLARE_OAUTH_CLIENT_ID_ENV = "COMFY_MODAL_CLOUDFLARE_OAUTH_CLIENT_ID"
CLOUDFLARE_OAUTH_REDIRECT_URI_ENV = "COMFY_MODAL_CLOUDFLARE_OAUTH_REDIRECT_URI"
CLOUDFLARE_OAUTH_SCOPES_ENV = "COMFY_MODAL_CLOUDFLARE_OAUTH_SCOPES"
CLOUDFLARE_OAUTH_AUTHORIZE_URL = "https://dash.cloudflare.com/oauth2/auth"
CLOUDFLARE_OAUTH_TOKEN_URL = "https://dash.cloudflare.com/oauth2/token"
CLOUDFLARE_API_BASE_URL = "https://api.cloudflare.com/client/v4"
R2_OAUTH_START_ROUTE = "/remote/storage/r2/oauth/start"
R2_OAUTH_CALLBACK_ROUTE = "/remote/storage/r2/oauth/callback"
R2_CREDENTIAL_STATUS_ROUTE = "/remote/storage/r2/status"

DEFAULT_CLOUDFLARE_OAUTH_SCOPES = (
    "account.read",
    "api-tokens.write",
    "workers-r2-storage.write",
)
_OAUTH_STATE_TTL_SECONDS = 10 * 60
_R2_BUCKET_PERMISSION_NAME = "Workers R2 Storage Bucket Item Write"
_ALLOWED_JURISDICTIONS = frozenset({"default", "eu", "fedramp", "us"})
_ACCOUNT_ID_PATTERN = re.compile(r"^[a-fA-F0-9]{32}$")
_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")


class CloudflareOAuthError(RuntimeError):
    """Raised when Cloudflare authorization or R2 provisioning fails safely."""


@dataclass(frozen=True)
class CloudflareOAuthConfiguration:
    """Hold the public OAuth client settings needed by the local controller."""

    client_id: str
    scopes: tuple[str, ...] = DEFAULT_CLOUDFLARE_OAUTH_SCOPES
    redirect_uri: str | None = None
    authorize_url: str = CLOUDFLARE_OAUTH_AUTHORIZE_URL
    token_url: str = CLOUDFLARE_OAUTH_TOKEN_URL
    api_base_url: str = CLOUDFLARE_API_BASE_URL

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "CloudflareOAuthConfiguration":
        """Load public client metadata without requiring a client secret."""
        source = os.environ if environment is None else environment
        client_id = str(source.get(CLOUDFLARE_OAUTH_CLIENT_ID_ENV) or "").strip()
        if not client_id:
            raise CloudflareOAuthError(
                f"Set {CLOUDFLARE_OAUTH_CLIENT_ID_ENV} to the public Cloudflare "
                "OAuth client ID registered for ComfyUI-Modal."
            )
        raw_scopes = str(source.get(CLOUDFLARE_OAUTH_SCOPES_ENV) or "").strip()
        scopes = tuple(
            scope
            for scope in raw_scopes.replace(",", " ").split()
            if scope
        ) or DEFAULT_CLOUDFLARE_OAUTH_SCOPES
        redirect_uri = str(
            source.get(CLOUDFLARE_OAUTH_REDIRECT_URI_ENV) or ""
        ).strip()
        return cls(
            client_id=client_id,
            scopes=scopes,
            redirect_uri=redirect_uri or None,
        )


@dataclass(frozen=True)
class R2OAuthState:
    """Bind one PKCE authorization to a specific local node and bucket request."""

    node_id: str
    credential_id: str
    bucket: str
    jurisdiction: str
    requested_account_id: str | None
    redirect_uri: str
    code_verifier: str = field(repr=False)
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class R2OAuthStateStore:
    """Keep short-lived one-time OAuth state and PKCE verifiers in controller memory."""

    ttl_seconds: float = _OAUTH_STATE_TTL_SECONDS
    _states: dict[str, R2OAuthState] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def create(self, state: R2OAuthState) -> str:
        """Store a state record under a cryptographically random one-time key."""
        state_token = secrets.token_urlsafe(32)
        with self._lock:
            self._remove_expired_locked()
            self._states[state_token] = state
        return state_token

    def consume(self, state_token: str) -> R2OAuthState:
        """Return and remove one valid state record."""
        with self._lock:
            self._remove_expired_locked()
            state = self._states.pop(state_token, None)
        if state is None:
            raise CloudflareOAuthError(
                "Cloudflare Login expired or returned an invalid OAuth state."
            )
        return state

    def _remove_expired_locked(self) -> None:
        """Remove state records older than the configured lifetime while locked."""
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [
            state_token
            for state_token, state in self._states.items()
            if state.created_at < cutoff
        ]
        for state_token in expired:
            self._states.pop(state_token, None)


@dataclass
class CloudflareR2OAuthService:
    """Exchange OAuth authorization for one persistent bucket-scoped R2 key."""

    configuration: CloudflareOAuthConfiguration
    credential_store: R2CredentialStore = field(default_factory=R2CredentialStore)
    state_store: R2OAuthStateStore = field(default_factory=R2OAuthStateStore)

    def authorization_url(
        self,
        *,
        node_id: str,
        credential_id: str,
        bucket: str,
        jurisdiction: str,
        requested_account_id: str | None,
        request_origin: str,
    ) -> str:
        """Create a one-time Cloudflare Authorization Code + PKCE URL."""
        redirect_uri = self._redirect_uri(request_origin)
        code_verifier = secrets.token_urlsafe(64)
        challenge = _base64url_sha256(code_verifier)
        state_token = self.state_store.create(
            R2OAuthState(
                node_id=_single_line(node_id, "node ID"),
                credential_id=_single_line(credential_id, "credential ID"),
                bucket=_bucket(bucket),
                jurisdiction=_jurisdiction(jurisdiction),
                requested_account_id=(
                    _account_id(requested_account_id)
                    if requested_account_id
                    else None
                ),
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
            )
        )
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.configuration.client_id,
                "redirect_uri": redirect_uri,
                "scope": " ".join(self.configuration.scopes),
                "state": state_token,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{self.configuration.authorize_url}?{query}"

    async def complete(self, *, state_token: str, code: str) -> dict[str, str]:
        """Exchange a code, provision R2, and save only the resulting S3 credential."""
        state = self.state_store.consume(_single_line(state_token, "OAuth state"))
        previous_record = await asyncio.to_thread(
            self.credential_store.load, state.credential_id
        )
        access_token = await self._exchange_code(state, code)
        account_id = await self._select_account(access_token, state)
        await self._ensure_bucket(access_token, account_id, state)
        record = await self._create_bucket_credential(access_token, account_id, state)
        await self._save_credential_or_revoke(access_token, state.credential_id, record)
        await self._revoke_replaced_credential(
            access_token,
            previous_record,
            replacement=record,
        )
        return {
            "node_id": state.node_id,
            "credential_id": state.credential_id,
            "account_id": account_id,
            "bucket": state.bucket,
            "jurisdiction": state.jurisdiction,
        }

    async def _exchange_code(self, state: R2OAuthState, code: str) -> str:
        """Exchange one authorization code with its exact PKCE verifier."""
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.configuration.client_id,
            "code": _single_line(code, "authorization code"),
            "redirect_uri": state.redirect_uri,
            "code_verifier": state.code_verifier,
        }
        response = await self._request_json(
            "POST",
            self.configuration.token_url,
            data=payload,
            include_cloudflare_envelope=False,
        )
        access_token = str(response.get("access_token") or "").strip()
        if not access_token:
            raise CloudflareOAuthError(
                "Cloudflare did not return an OAuth access token."
            )
        return access_token

    async def _select_account(self, access_token: str, state: R2OAuthState) -> str:
        """Select the requested account or require an unambiguous OAuth grant."""
        result = await self._cloudflare_api(
            "GET", "/accounts", access_token=access_token
        )
        if not isinstance(result, list):
            raise CloudflareOAuthError("Cloudflare returned an invalid account list.")
        account_ids = [
            _account_id(str(account.get("id") or "").strip())
            for account in result
            if isinstance(account, Mapping) and account.get("id")
        ]
        if state.requested_account_id:
            if state.requested_account_id not in account_ids:
                raise CloudflareOAuthError(
                    "The requested Cloudflare account was not included in the OAuth grant."
                )
            return state.requested_account_id
        if len(account_ids) != 1:
            raise CloudflareOAuthError(
                "Cloudflare authorized multiple accounts. Enter the desired account ID "
                "on the R2 node and select Login again."
            )
        return account_ids[0]

    async def _ensure_bucket(
        self,
        access_token: str,
        account_id: str,
        state: R2OAuthState,
    ) -> None:
        """Reuse an exact R2 bucket or create it in the requested jurisdiction."""
        headers = _jurisdiction_headers(state.jurisdiction)
        result = await self._cloudflare_api(
            "GET",
            f"/accounts/{account_id}/r2/buckets?name_contains={state.bucket}",
            access_token=access_token,
            headers=headers,
        )
        buckets = result.get("buckets", []) if isinstance(result, Mapping) else []
        if any(
            isinstance(bucket, Mapping) and bucket.get("name") == state.bucket
            for bucket in buckets
        ):
            return
        await self._cloudflare_api(
            "POST",
            f"/accounts/{account_id}/r2/buckets",
            access_token=access_token,
            headers=headers,
            json_payload={"name": state.bucket},
        )

    async def _create_bucket_credential(
        self,
        access_token: str,
        account_id: str,
        state: R2OAuthState,
    ) -> R2CredentialRecord:
        """Create a user token restricted to object access in one exact R2 bucket."""
        permission_groups = await self._cloudflare_api(
            "GET", "/user/tokens/permission_groups", access_token=access_token
        )
        permission_id = _permission_group_id(
            permission_groups, _R2_BUCKET_PERMISSION_NAME
        )
        resource = (
            "com.cloudflare.edge.r2.bucket."
            f"{account_id}_{state.jurisdiction}_{state.bucket}"
        )
        result = await self._cloudflare_api(
            "POST",
            "/user/tokens",
            access_token=access_token,
            json_payload={
                "name": f"ComfyUI-Modal R2 {state.bucket}",
                "policies": [
                    {
                        "effect": "allow",
                        "resources": {resource: "*"},
                        "permission_groups": [{"id": permission_id}],
                    }
                ],
            },
        )
        if not isinstance(result, Mapping):
            raise CloudflareOAuthError("Cloudflare returned an invalid R2 token.")
        token_id = str(result.get("id") or "").strip()
        token_value = str(result.get("value") or "").strip()
        if not token_id or not token_value:
            raise CloudflareOAuthError(
                "Cloudflare created an API token without returning its one-time secret."
            )
        return R2CredentialRecord(
            account_id=account_id,
            bucket=state.bucket,
            access_key_id=token_id,
            secret_access_key=hashlib.sha256(token_value.encode("utf-8")).hexdigest(),
            jurisdiction=state.jurisdiction,
        )

    async def _save_credential_or_revoke(
        self,
        access_token: str,
        credential_id: str,
        record: R2CredentialRecord,
    ) -> None:
        """Persist a new token or revoke it when the local secure write fails."""
        try:
            await asyncio.to_thread(self.credential_store.save, credential_id, record)
        except R2CredentialError:
            try:
                await self._cloudflare_api(
                    "DELETE",
                    f"/user/tokens/{record.access_key_id}",
                    access_token=access_token,
                )
            except CloudflareOAuthError as cleanup_error:
                logger.warning(
                    "Could not revoke an R2 token after keyring persistence failed: %s",
                    cleanup_error,
                )
            raise

    async def _revoke_replaced_credential(
        self,
        access_token: str,
        previous: R2CredentialRecord | None,
        *,
        replacement: R2CredentialRecord,
    ) -> None:
        """Best-effort revoke the old user token after a replacement is secured."""
        if previous is None or previous.access_key_id == replacement.access_key_id:
            return
        try:
            await self._cloudflare_api(
                "DELETE",
                f"/user/tokens/{previous.access_key_id}",
                access_token=access_token,
            )
        except CloudflareOAuthError as exc:
            logger.warning(
                "Stored the replacement R2 credential but could not revoke the "
                "previous Cloudflare token ID: %s",
                exc,
            )

    async def _cloudflare_api(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        headers: Mapping[str, str] | None = None,
        json_payload: Mapping[str, Any] | None = None,
    ) -> Any:
        """Call one Cloudflare v4 endpoint and unwrap its result envelope."""
        request_headers = {
            "Authorization": f"Bearer {access_token}",
            **dict(headers or {}),
        }
        return await self._request_json(
            method,
            f"{self.configuration.api_base_url}{path}",
            headers=request_headers,
            json_payload=json_payload,
            include_cloudflare_envelope=True,
        )

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
        json_payload: Mapping[str, Any] | None = None,
        include_cloudflare_envelope: bool,
    ) -> Any:
        """Execute one bounded HTTPS request and return credential-safe JSON."""
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    data=data,
                    json=json_payload,
                ) as response:
                    try:
                        payload = await response.json(content_type=None)
                    except json.JSONDecodeError as exc:
                        raise CloudflareOAuthError(
                            "Cloudflare returned an invalid JSON response."
                        ) from exc
                    if response.status < 200 or response.status >= 300:
                        raise CloudflareOAuthError(_cloudflare_error(payload, response.status))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise CloudflareOAuthError(
                "Cloudflare could not be reached while completing Login."
            ) from exc
        if not isinstance(payload, Mapping):
            raise CloudflareOAuthError("Cloudflare returned an invalid JSON response.")
        if not include_cloudflare_envelope:
            return payload
        if payload.get("success") is not True:
            raise CloudflareOAuthError(_cloudflare_error(payload, 400))
        return payload.get("result")

    def _redirect_uri(self, request_origin: str) -> str:
        """Return the registered callback, allowing automatic loopback callbacks."""
        if self.configuration.redirect_uri:
            return self.configuration.redirect_uri
        parsed = urlparse(request_origin)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise CloudflareOAuthError(
                f"Set {CLOUDFLARE_OAUTH_REDIRECT_URI_ENV} when ComfyUI is not "
                "opened through an HTTP loopback address."
            )
        return f"{request_origin.rstrip('/')}{R2_OAUTH_CALLBACK_ROUTE}"


def setup_r2_oauth_routes(
    prompt_server: Any,
    service: CloudflareR2OAuthService | None = None,
) -> None:
    """Register credential-safe R2 Login and status routes on ComfyUI's server."""
    if not all(
        hasattr(prompt_server.routes, method_name)
        for method_name in ("get", "post")
    ):
        logger.debug("ComfyUI route table cannot register R2 OAuth GET and POST routes.")
        return
    oauth_service = service
    credential_store = (
        service.credential_store if service is not None else R2CredentialStore()
    )

    def resolved_service() -> CloudflareR2OAuthService:
        """Load environment-backed OAuth settings only when Login is used."""
        nonlocal oauth_service
        if oauth_service is None:
            oauth_service = CloudflareR2OAuthService(
                CloudflareOAuthConfiguration.from_environment(),
                credential_store=credential_store,
            )
        return oauth_service

    @prompt_server.routes.post(R2_OAUTH_START_ROUTE)
    async def start_r2_oauth(request: web.Request) -> web.Response:
        """Return a PKCE authorization URL without executing a workflow."""
        return await _start_r2_oauth_response(request, resolved_service)

    @prompt_server.routes.get(R2_OAUTH_CALLBACK_ROUTE)
    async def complete_r2_oauth(request: web.Request) -> web.Response:
        """Complete Cloudflare Login and notify the opener without exposing secrets."""
        return await _complete_r2_oauth_response(request, resolved_service)

    @prompt_server.routes.get(R2_CREDENTIAL_STATUS_ROUTE)
    async def r2_credential_status(request: web.Request) -> web.Response:
        """Report whether one opaque workflow reference exists in the OS keyring."""
        return await _r2_credential_status_response(request, credential_store)


async def _start_r2_oauth_response(
    request: web.Request,
    service_factory: Callable[[], CloudflareR2OAuthService],
) -> web.Response:
    """Build one authorization response from a credential-free browser request."""
    try:
        payload = await request.json()
        authorization_url = service_factory().authorization_url(
            node_id=str(payload.get("node_id") or ""),
            credential_id=str(payload.get("credential_id") or ""),
            bucket=str(payload.get("bucket") or ""),
            jurisdiction=str(payload.get("jurisdiction") or "default"),
            requested_account_id=(
                str(payload.get("account_id") or "").strip() or None
            ),
            request_origin=str(payload.get("origin") or ""),
        )
    except (CloudflareOAuthError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"authorization_url": authorization_url})


async def _complete_r2_oauth_response(
    request: web.Request,
    service_factory: Callable[[], CloudflareR2OAuthService],
) -> web.Response:
    """Complete OAuth and return the credential-free popup notification page."""
    oauth_error = str(
        request.query.get("error_description")
        or request.query.get("error")
        or ""
    ).strip()
    try:
        if oauth_error:
            raise CloudflareOAuthError(
                f"Cloudflare Login was not completed: {oauth_error}"
            )
        result = await service_factory().complete(
            state_token=str(request.query.get("state") or ""),
            code=str(request.query.get("code") or ""),
        )
        payload: dict[str, Any] = {
            "type": "comfy-modal-r2-oauth",
            "ok": True,
            **result,
        }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Cloudflare R2 Login failed: %s", exc)
        payload = {
            "type": "comfy-modal-r2-oauth",
            "ok": False,
            "error": str(exc),
        }
    return _oauth_popup_response(payload)


async def _r2_credential_status_response(
    request: web.Request,
    credential_store: R2CredentialStore,
) -> web.Response:
    """Return one credential-free OS-keyring status response."""
    credential_id = str(request.query.get("credential_id") or "").strip()
    try:
        status = await asyncio.to_thread(credential_store.status, credential_id)
    except (RuntimeError, ValueError) as exc:
        return web.json_response({"connected": False, "error": str(exc)}, status=400)
    return web.json_response(status)


def _permission_group_id(permission_groups: Any, expected_name: str) -> str:
    """Return the current permission group ID for one exact Cloudflare name."""
    if not isinstance(permission_groups, list):
        raise CloudflareOAuthError("Cloudflare returned invalid permission groups.")
    for permission_group in permission_groups:
        if not isinstance(permission_group, Mapping):
            continue
        if permission_group.get("name") == expected_name:
            permission_id = str(permission_group.get("id") or "").strip()
            if permission_id:
                return permission_id
    raise CloudflareOAuthError(
        f"Cloudflare did not expose the required {expected_name!r} permission."
    )


def _cloudflare_error(payload: Any, status: int) -> str:
    """Return a bounded provider error without reflecting credentials or request data."""
    if isinstance(payload, Mapping):
        errors = payload.get("errors")
        if isinstance(errors, list):
            messages = [
                str(error.get("message") or "").strip()
                for error in errors
                if isinstance(error, Mapping) and error.get("message")
            ]
            if messages:
                return f"Cloudflare API error ({status}): {'; '.join(messages)[:1000]}"
        description = str(payload.get("error_description") or "").strip()
        if description:
            return f"Cloudflare OAuth error ({status}): {description[:1000]}"
    return f"Cloudflare request failed with HTTP {status}."


def _oauth_popup_response(payload: Mapping[str, Any]) -> web.Response:
    """Return a same-origin popup page that posts a credential-free result."""
    encoded_payload = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    message = (
        "Cloudflare Login complete. This window can be closed."
        if payload.get("ok")
        else html.escape(str(payload.get("error") or "Cloudflare Login failed."))
    )
    body = (
        "<!doctype html><meta charset='utf-8'><title>Cloudflare R2 Login</title>"
        f"<p>{message}</p><script>"
        f"const payload={encoded_payload};"
        "if(window.opener){window.opener.postMessage(payload,window.location.origin);"
        "window.close();}"
        "</script>"
    )
    return web.Response(text=body, content_type="text/html")


def _base64url_sha256(value: str) -> str:
    """Return an unpadded RFC 7636 S256 code challenge."""
    digest = hashlib.sha256(value.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _single_line(value: str, label: str) -> str:
    """Validate one non-empty value that may be stored in OAuth state."""
    normalized = str(value).strip()
    if not normalized or any(character in normalized for character in ("\x00", "\n", "\r")):
        raise ValueError(f"Cloudflare R2 {label} must be a non-empty single-line value.")
    return normalized


def _jurisdiction(value: str) -> str:
    """Validate one supported R2 jurisdiction identifier."""
    normalized = str(value or "default").strip().casefold()
    if normalized not in _ALLOWED_JURISDICTIONS:
        raise ValueError("Cloudflare R2 jurisdiction is not supported.")
    return normalized


def _account_id(value: str) -> str:
    """Validate one Cloudflare account identifier."""
    normalized = str(value).strip()
    if not _ACCOUNT_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Cloudflare account ID must be 32 hexadecimal characters.")
    return normalized


def _bucket(value: str) -> str:
    """Validate one R2 bucket name before beginning authorization."""
    normalized = _single_line(value, "bucket")
    if not _BUCKET_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Cloudflare R2 bucket must contain 3-63 lowercase letters, digits, "
            "or hyphens and begin and end with a letter or digit."
        )
    return normalized


def _jurisdiction_headers(jurisdiction: str) -> dict[str, str]:
    """Return the optional Cloudflare REST jurisdiction header."""
    return {} if jurisdiction == "default" else {"cf-r2-jurisdiction": jurisdiction}


__all__ = [
    "CLOUDFLARE_OAUTH_CLIENT_ID_ENV",
    "CLOUDFLARE_OAUTH_REDIRECT_URI_ENV",
    "CLOUDFLARE_OAUTH_SCOPES_ENV",
    "CloudflareOAuthConfiguration",
    "CloudflareOAuthError",
    "CloudflareR2OAuthService",
    "R2_CREDENTIAL_STATUS_ROUTE",
    "R2_OAUTH_CALLBACK_ROUTE",
    "R2_OAUTH_START_ROUTE",
    "R2OAuthState",
    "R2OAuthStateStore",
    "setup_r2_oauth_routes",
]
