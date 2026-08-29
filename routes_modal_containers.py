"""Modal progress, container, cache, and volume route registration."""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from typing import Any

from aiohttp import web

if __package__:
    from .modal_admin_ops import delete_modal_cache_dicts, delete_modal_volume
    from .modal_ui_events import modal_ui_events_for_client
    from .route_context import RouteContext
    from .settings import settings_for_modal_gpu
else:  # pragma: no cover - flat import inside the Modal container.
    from modal_admin_ops import delete_modal_cache_dicts, delete_modal_volume
    from modal_ui_events import modal_ui_events_for_client
    from route_context import RouteContext
    from settings import settings_for_modal_gpu

logger = logging.getLogger(__name__)


def _modal_app_module() -> Any:
    """Load the host-side Modal application module for live administration."""
    module_name = f"{__package__}.remote.modal_app" if __package__ else "remote.modal_app"
    return importlib.import_module(module_name)


def register_modal_container_routes(prompt_server: Any, ctx: RouteContext) -> None:
    """Register Modal progress, container, cache, and volume routes."""
    if hasattr(prompt_server.routes, "get"):

        @prompt_server.routes.get(ctx.progress_state_route_path)
        async def modal_progress_state(request: web.Request) -> web.Response:
            """Return recent Modal UI events for the requesting ComfyUI client."""
            client_id = request.query.get("client_id")
            return web.json_response(
                {"events": modal_ui_events_for_client(client_id)}
            )

        @prompt_server.routes.get(ctx.container_status_route_path)
        async def modal_container_status(request: web.Request) -> web.Response:
            """Return active containers and hourly billing for one selected GPU app."""
            modal_app = _modal_app_module()
            requested_modal_gpu = request.query.get(
                "modal_gpu",
                ctx.settings.modal_gpu,
            )
            try:
                selected_settings = settings_for_modal_gpu(
                    ctx.settings,
                    requested_modal_gpu,
                )
            except ValueError as exc:
                return web.json_response(
                    {"containers": [], "error": str(exc), "polled_at": time.time()},
                    status=400,
                )

            containers_task = asyncio.create_task(
                modal_app.list_active_modal_containers(ctx.settings)
            )
            include_billing = request.query.get(
                "include_billing", "true"
            ).casefold() not in {"0", "false", "no"}
            billing_task = (
                asyncio.create_task(
                    modal_app.get_hourly_modal_app_billing(
                        selected_settings.modal_gpu,
                        ctx.settings,
                    )
                )
                if include_billing
                else None
            )
            try:
                containers = await containers_task
            except modal_app.ModalContainerStatusError as exc:
                if billing_task is not None:
                    billing_task.cancel()
                    await asyncio.gather(billing_task, return_exceptions=True)
                logger.warning("Unable to refresh Modal container status: %s", exc)
                return web.json_response(
                    {"containers": [], "error": str(exc), "polled_at": time.time()},
                    status=502,
                )
            billing = None
            billing_error = None
            if billing_task is not None:
                try:
                    billing = await billing_task
                except modal_app.ModalBillingStatusError as exc:
                    billing_error = str(exc)
                    logger.warning("Unable to refresh Modal hourly billing: %s", exc)
            return web.json_response(
                {
                    "containers": [container.as_dict() for container in containers],
                    "billing": billing.as_dict() if billing is not None else None,
                    "billing_error": billing_error,
                    "polled_at": time.time(),
                }
            )

        @prompt_server.routes.post(ctx.container_stop_route_path)
        async def modal_container_stop(request: web.Request) -> web.Response:
            """Stop one exact active Modal container owned by this installation."""
            modal_app = _modal_app_module()
            try:
                payload = await request.json()
                container_id = str(payload.get("container_id") or "").strip()
                stopped = await modal_app.stop_managed_modal_container(
                    container_id,
                    ctx.settings,
                )
            except (modal_app.ModalContainerStatusError, TypeError, ValueError) as exc:
                return web.json_response({"error": str(exc)}, status=502)
            return web.json_response(
                {"container_id": container_id, "stopped": stopped}
            )

    @prompt_server.routes.post(ctx.delete_caches_route_path)
    async def modal_delete_caches(request: web.Request) -> web.Response:
        """Delete persistent Modal cache Dicts for the active configuration."""
        del request
        logger.info("Received Modal cache deletion request.")
        try:
            return web.json_response(await delete_modal_cache_dicts(ctx.settings))
        except RuntimeError as exc:
            logger.warning("Modal cache deletion request failed: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)

    @prompt_server.routes.post(ctx.delete_volume_route_path)
    async def modal_delete_volume_route(request: web.Request) -> web.Response:
        """Delete the configured Modal Volume for the active configuration."""
        del request
        logger.info("Received Modal volume deletion request.")
        try:
            return web.json_response(await delete_modal_volume(ctx.settings))
        except RuntimeError as exc:
            logger.warning("Modal volume deletion request failed: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
