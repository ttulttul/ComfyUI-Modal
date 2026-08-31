"""Credential-safe Subrosa Login routes and token validation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

if __package__:
    from .remote_configurations import (
        RemoteConfigurationSet,
        SubrosaRemoteConfiguration,
    )
    from .subrosa_credentials import (
        SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE,
        SubrosaCredentialError,
        SubrosaCredentialStore,
    )
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from remote_configurations import (
        RemoteConfigurationSet,
        SubrosaRemoteConfiguration,
    )
    from subrosa_credentials import (
        SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE,
        SubrosaCredentialError,
        SubrosaCredentialStore,
    )

logger = logging.getLogger(__name__)

SUBROSA_CREDENTIAL_IMPORT_ROUTE = "/remote/subrosa/credentials"
SUBROSA_CREDENTIAL_STATUS_ROUTE = "/remote/subrosa/status"
SUBROSA_LOGIN_REQUIRED_CODE = "subrosa_login_required"
SUBROSA_LOGIN_REQUIRED_MESSAGE = (
    'Subrosa authentication required. Click the Subrosa Configuration node\'s '
    '"Click to Authenticate" button, then queue the workflow again.'
)
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) ComfyUI-Modal/0.4.2"
)


class SubrosaLoginError(RuntimeError):
    """Raised when Subrosa Login cannot validate or persist a token."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Retain an optional credential-safe recovery code for the UI."""
        super().__init__(message)
        self.code = code


class SubrosaLoginRequiredError(SubrosaLoginError):
    """Raised when a missing or rejected token requires another Login."""

    def __init__(self) -> None:
        """Use the single concise workflow-facing recovery instruction."""
        super().__init__(
            SUBROSA_LOGIN_REQUIRED_MESSAGE,
            code=SUBROSA_LOGIN_REQUIRED_CODE,
        )


class SubrosaConfigurationValidationError(SubrosaLoginError):
    """Attribute one queue-time credential failure to its configuration node."""

    def __init__(
        self,
        configuration_id: str,
        cause: SubrosaCredentialError | SubrosaLoginError,
    ) -> None:
        """Preserve the safe error and originating serialized node ID."""
        super().__init__(str(cause), code=getattr(cause, "code", None))
        self.configuration_id = configuration_id


SessionFactory = Callable[..., Any]


@dataclass
class SubrosaLoginService:
    """Validate portal-minted tokens and persist them in the local keyring."""

    credential_store: SubrosaCredentialStore = field(
        default_factory=SubrosaCredentialStore,
        repr=False,
    )
    session_factory: SessionFactory = field(default=aiohttp.ClientSession, repr=False)

    async def save(
        self,
        *,
        credential_id: str,
        relay_url: str,
        token: str,
    ) -> dict[str, Any]:
        """Validate one token remotely before replacing the saved credential."""
        account = await validate_subrosa_token(
            relay_url,
            token,
            session_factory=self.session_factory,
        )
        self.credential_store.save(credential_id, token)
        return account

    async def status(self, *, credential_id: str, relay_url: str) -> dict[str, Any]:
        """Return safe connection state without exposing keyring material."""
        try:
            token = self.credential_store.load(credential_id)
        except SubrosaCredentialError as exc:
            if exc.code == SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE:
                raise
            return {"connected": False, "reason": SUBROSA_LOGIN_REQUIRED_CODE}
        if token is None:
            return {"connected": False, "reason": SUBROSA_LOGIN_REQUIRED_CODE}
        try:
            account = await validate_subrosa_token(
                relay_url,
                token,
                session_factory=self.session_factory,
            )
        except SubrosaLoginRequiredError:
            return {"connected": False, "reason": SUBROSA_LOGIN_REQUIRED_CODE}
        return {"connected": True, **_safe_account(account)}


async def validate_subrosa_token(
    relay_url: str,
    token: str,
    *,
    session_factory: SessionFactory = aiohttp.ClientSession,
) -> dict[str, Any]:
    """Validate one `srk_` token with Subrosa and return safe account metadata."""
    normalized_token = str(token).strip()
    if not normalized_token.startswith("srk_") or len(normalized_token) <= 4:
        raise SubrosaLoginRequiredError()
    url = _http_base_url(relay_url) + "/api/v1/extension/whoami"
    headers = {
        "Authorization": f"Bearer {normalized_token}",
        "User-Agent": _USER_AGENT,
    }
    timeout = aiohttp.ClientTimeout(total=30.0)
    try:
        async with (
            session_factory(timeout=timeout) as session,
            session.get(url, headers=headers) as response,
        ):
            body = await response.json(content_type=None)
            if response.status == 401:
                raise SubrosaLoginRequiredError()
            if response.status != 200:
                raise SubrosaLoginError(
                    f"Subrosa token validation failed with HTTP {response.status}: "
                    f"{_safe_error_message(body)}"
                )
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise SubrosaLoginError(
            f"Subrosa token validation could not reach the service: {exc}"
        ) from exc
    if not isinstance(body, dict):
        raise SubrosaLoginError("Subrosa whoami returned an invalid response object.")
    return dict(body)


def require_saved_subrosa_token(
    credential_store: SubrosaCredentialStore,
    credential_id: str,
) -> str:
    """Load a token while converting missing credentials to the Login instruction."""
    try:
        return credential_store.require(credential_id)
    except SubrosaCredentialError as exc:
        if exc.code == SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE:
            raise
        raise SubrosaLoginRequiredError() from exc


async def preflight_subrosa_configurations(
    configuration_set: RemoteConfigurationSet,
    *,
    credential_store: SubrosaCredentialStore | None = None,
    session_factory: SessionFactory = aiohttp.ClientSession,
) -> None:
    """Validate connected Subrosa credentials before ComfyUI starts execution."""
    store = credential_store or SubrosaCredentialStore()
    for configuration in configuration_set.capacity_configurations:
        if not isinstance(configuration, SubrosaRemoteConfiguration):
            continue
        try:
            token = require_saved_subrosa_token(
                store,
                configuration.credential_id,
            )
            await validate_subrosa_token(
                configuration.relay_url,
                token,
                session_factory=session_factory,
            )
        except (SubrosaCredentialError, SubrosaLoginError) as exc:
            raise SubrosaConfigurationValidationError(
                configuration.configuration_id,
                exc,
            ) from exc


def setup_subrosa_login_routes(
    prompt_server: Any,
    service: SubrosaLoginService | None = None,
) -> None:
    """Register local status and validated keyring-import routes."""
    if not all(
        hasattr(prompt_server.routes, method_name)
        for method_name in ("get", "post")
    ):
        logger.debug("ComfyUI route table cannot register Subrosa Login routes.")
        return
    login_service = service or SubrosaLoginService()

    @prompt_server.routes.post(SUBROSA_CREDENTIAL_IMPORT_ROUTE)
    async def import_subrosa_token(request: web.Request) -> web.Response:
        """Validate and save a one-time portal token without returning it."""
        return await _import_subrosa_token_response(request, login_service)

    @prompt_server.routes.get(SUBROSA_CREDENTIAL_STATUS_ROUTE)
    async def subrosa_login_status(request: web.Request) -> web.Response:
        """Return remote validity for one local keyring reference."""
        return await _subrosa_login_status_response(request, login_service)


async def _import_subrosa_token_response(
    request: web.Request,
    service: SubrosaLoginService,
) -> web.Response:
    """Handle one credential import with no secret-bearing response fields."""
    try:
        payload = await request.json()
        account = await service.save(
            credential_id=str(payload.get("credential_id") or ""),
            relay_url=str(payload.get("relay_url") or ""),
            token=str(payload.get("token") or ""),
        )
    except SubrosaLoginRequiredError as exc:
        return _json_response(
            {"error": str(exc), "code": exc.code},
            status=401,
        )
    except (SubrosaCredentialError, SubrosaLoginError, TypeError, json.JSONDecodeError) as exc:
        return _json_response(
            {"error": str(exc), "code": getattr(exc, "code", None)},
            status=400,
        )
    return _json_response({"connected": True, **_safe_account(account)})


async def _subrosa_login_status_response(
    request: web.Request,
    service: SubrosaLoginService,
) -> web.Response:
    """Handle one status lookup without retrieving a token into the browser."""
    try:
        status = await service.status(
            credential_id=str(request.query.get("credential_id") or ""),
            relay_url=str(request.query.get("relay_url") or ""),
        )
    except (SubrosaCredentialError, SubrosaLoginError, ValueError) as exc:
        return _json_response(
            {"error": str(exc), "code": getattr(exc, "code", None)},
            status=400,
        )
    return _json_response(status)


def _safe_account(account: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only non-secret account status returned by `whoami`."""
    return {
        "account_number": account.get("account_number"),
        "balance": account.get("balance"),
        "status": account.get("status"),
    }


def _http_base_url(relay_url: str) -> str:
    """Convert one trusted relay WebSocket URL to its HTTP origin."""
    parsed = urlparse(str(relay_url).strip())
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise SubrosaLoginError(
            "Subrosa relay_url must be an absolute ws:// or wss:// URL."
        )
    hostname = (parsed.hostname or "").lower()
    is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
    is_subrosa = hostname == "subrosa.red" or hostname.endswith(".subrosa.red")
    if not is_loopback and (parsed.scheme != "wss" or not is_subrosa):
        raise SubrosaLoginError(
            "Subrosa Login only sends extension tokens to an HTTPS Subrosa service."
        )
    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}"


def _safe_error_message(body: Any) -> str:
    """Extract a bounded server diagnostic without reflecting credentials."""
    if isinstance(body, Mapping):
        detail = str(body.get("error") or body.get("message") or "").strip()
        if detail:
            return detail[:500]
    return "request rejected"


def _json_response(payload: Mapping[str, Any], *, status: int = 200) -> web.Response:
    """Return a no-store JSON response for all credential-adjacent routes."""
    return web.json_response(
        dict(payload),
        status=status,
        headers={"Cache-Control": "no-store"},
    )


__all__ = [
    "SUBROSA_CREDENTIAL_IMPORT_ROUTE",
    "SUBROSA_CREDENTIAL_STATUS_ROUTE",
    "SUBROSA_LOGIN_REQUIRED_CODE",
    "SUBROSA_LOGIN_REQUIRED_MESSAGE",
    "SubrosaConfigurationValidationError",
    "SubrosaLoginError",
    "SubrosaLoginRequiredError",
    "SubrosaLoginService",
    "preflight_subrosa_configurations",
    "require_saved_subrosa_token",
    "setup_subrosa_login_routes",
    "validate_subrosa_token",
]
