"""R2 storage and keychain route registration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Mapping

from aiohttp import web

if __package__:
    from .execution_scheduling import (
        _r2_storage_from_usage_payload,
        _refresh_r2_storage_usage,
    )
    from .r2_cache import R2CacheError
    from .r2_credentials import (
        R2CredentialError,
        R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
        request_macos_keychain_unlock,
    )
    from .route_context import RouteContext
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_scheduling import (
        _r2_storage_from_usage_payload,
        _refresh_r2_storage_usage,
    )
    from r2_cache import R2CacheError
    from r2_credentials import (
        R2CredentialError,
        R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
        request_macos_keychain_unlock,
    )
    from route_context import RouteContext

logger = logging.getLogger(__name__)

R2_STORAGE_USAGE_ROUTE = "/remote/storage/r2/usage"
R2_KEYCHAIN_UNLOCK_ROUTE = "/remote/storage/r2/keychain/unlock"


def register_r2_routes(prompt_server: Any, ctx: RouteContext) -> None:
    """Register R2 storage-usage and macOS keychain routes."""
    del ctx

    @prompt_server.routes.post(R2_STORAGE_USAGE_ROUTE)
    async def refresh_r2_storage_usage(request: web.Request) -> web.Response:
        """Refresh one configured R2 bucket's safe aggregate storage state."""
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping):
                raise TypeError("R2 storage usage request must be a JSON object.")
            storage = _r2_storage_from_usage_payload(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        try:
            usage = await asyncio.to_thread(_refresh_r2_storage_usage, storage)
        except R2CredentialError as exc:
            logger.warning(
                "Unable to refresh R2 storage usage configuration=%s bucket=%s: %s",
                storage.configuration_id,
                storage.bucket,
                exc,
            )
            status = 423 if exc.code == R2_KEYCHAIN_UNLOCK_REQUIRED_CODE else 502
            return web.json_response(
                {"error": str(exc), "code": exc.code},
                status=status,
            )
        except (R2CacheError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Unable to refresh R2 storage usage configuration=%s bucket=%s: %s",
                storage.configuration_id,
                storage.bucket,
                exc,
            )
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {
                "configuration_id": storage.configuration_id,
                "storage_usage_bytes": usage.size_bytes,
                "storage_object_count": usage.object_count,
                "refreshed_at": time.time(),
            }
        )

    @prompt_server.routes.post(R2_KEYCHAIN_UNLOCK_ROUTE)
    async def unlock_r2_keychain(request: web.Request) -> web.Response:
        """Display macOS's system-owned login-keychain unlock prompt."""
        del request
        try:
            await asyncio.to_thread(request_macos_keychain_unlock)
        except R2CredentialError as exc:
            logger.warning("Unable to unlock the macOS login keychain: %s", exc)
            return web.json_response({"error": str(exc)}, status=409)
        return web.json_response({"unlocked": True})
