"""Vast.ai lease inventory and administration route registration."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from aiohttp import web

if __package__:
    from .route_context import RouteContext
    from .vast_api import VastApiClient
    from .vast_leases import VastLeaseManager
    from .vast_service import VastService
else:  # pragma: no cover - flat import inside the Modal container.
    from route_context import RouteContext
    from vast_api import VastApiClient
    from vast_leases import VastLeaseManager
    from vast_service import VastService


def _vast_api_client() -> VastApiClient:
    """Build a Vast API client from the current environment."""
    api_key = str(os.getenv("VAST_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set VAST_API_KEY first.")
    base_url = str(os.getenv("COMFY_MODAL_VAST_API_BASE_URL") or "").strip()
    return VastApiClient(api_key, **({"base_url": base_url} if base_url else {}))


def register_vast_routes(prompt_server: Any, ctx: RouteContext) -> None:
    """Register Vast status, credential verification, reap, and destroy routes."""
    registry = ctx.vast_registry

    if hasattr(prompt_server.routes, "get"):

        @prompt_server.routes.get("/remote/vast/status")
        async def vast_status(request: web.Request) -> web.Response:
            """Return refreshed credential-free managed Vast lease inventory."""
            del request
            if registry is None:
                return web.json_response(
                    {
                        "configured": False,
                        "leases": [],
                        "error": "ComfyUI user directory unavailable.",
                    },
                    status=503,
                )
            try:
                if not os.getenv("VAST_API_KEY"):
                    state = await asyncio.to_thread(registry.load)
                    leases = state.leases
                else:
                    manager = VastLeaseManager.for_inventory(
                        api_client=_vast_api_client(),
                        registry=registry,
                        owner_id=ctx.settings.app_name,
                    )
                    leases = await manager.refresh_owned_leases()
            except (OSError, RuntimeError, ValueError) as exc:
                return web.json_response(
                    {
                        "configured": bool(os.getenv("VAST_API_KEY")),
                        "leases": [],
                        "error": str(exc),
                    },
                    status=502,
                )
            return web.json_response(
                {
                    "configured": bool(os.getenv("VAST_API_KEY")),
                    "image_configured": bool(os.getenv("COMFY_MODAL_VAST_IMAGE")),
                    "leases": [lease.to_dict() for lease in leases],
                }
            )

    @prompt_server.routes.post("/remote/vast/verify")
    async def vast_verify(request: web.Request) -> web.Response:
        """Verify the configured Vast credential without returning it."""
        del request
        try:
            account = await _vast_api_client().verify_credentials()
        except RuntimeError as exc:
            status = 400 if not os.getenv("VAST_API_KEY") else 502
            return web.json_response(
                {"verified": False, "error": str(exc)}, status=status
            )
        except (OSError, ValueError) as exc:
            return web.json_response(
                {"verified": False, "error": str(exc)}, status=502
            )
        return web.json_response({"verified": True, "account": account})

    @prompt_server.routes.post("/remote/vast/reap")
    async def vast_reap(request: web.Request) -> web.Response:
        """Destroy only owned idle leases whose configured deadline has expired."""
        del request
        try:
            service = VastService.from_environment(
                ctx.settings,
                repo_root=Path(__file__).resolve().parent,
            )
            destroyed = await service.lease_manager.destroy_expired()
        except (OSError, RuntimeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response({"destroyed_instance_ids": list(destroyed)})

    @prompt_server.routes.post("/remote/vast/destroy")
    async def vast_destroy(request: web.Request) -> web.Response:
        """Destroy one exact idle registry-owned lease after ownership checks."""
        try:
            payload = await request.json()
            instance_id = int(payload.get("instance_id"))
            if registry is None:
                raise RuntimeError("ComfyUI user directory is unavailable.")
            manager = VastLeaseManager.for_inventory(
                api_client=_vast_api_client(),
                registry=registry,
                owner_id=ctx.settings.app_name,
            )
            destroyed = await manager.destroy_owned_lease(
                instance_id,
                allow_active_work=payload.get("force") is True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response({"instance_id": instance_id, "destroyed": destroyed})
