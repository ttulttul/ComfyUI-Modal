"""SSH remote-environment CRUD, probe, bootstrap, and worker routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

from aiohttp import web

if __package__:
    from .execution_environments import EnvironmentHealth
    from .remote_hosts import RemoteExecutionConfig
    from .route_context import RouteContext
    from .ssh_docker import SshDockerController
    from .ssh_runtime import SshRuntimeManager
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import EnvironmentHealth
    from remote_hosts import RemoteExecutionConfig
    from route_context import RouteContext
    from ssh_docker import SshDockerController
    from ssh_runtime import SshRuntimeManager


def _unavailable_registry_response() -> web.Response:
    """Return the common response for an unavailable ComfyUI user directory."""
    return web.json_response(
        {"error": "The ComfyUI user directory could not be resolved."},
        status=503,
    )


def register_remote_environment_routes(prompt_server: Any, ctx: RouteContext) -> None:
    """Register SSH host configuration and managed-worker routes."""
    registry = ctx.remote_host_registry

    if hasattr(prompt_server.routes, "get"):

        @prompt_server.routes.get(ctx.remote_environments_route_path)
        async def remote_environments(request: web.Request) -> web.Response:
            """Return credential-free SSH host configuration and discovered state."""
            del request
            if registry is None:
                return web.json_response(
                    {
                        "error": "The ComfyUI user directory could not be resolved.",
                        "hosts": [],
                    },
                    status=503,
                )
            try:
                config = await asyncio.to_thread(registry.load)
            except ValueError as exc:
                return web.json_response({"error": str(exc), "hosts": []}, status=500)
            return web.json_response(config.to_dict())

    if hasattr(prompt_server.routes, "put"):

        @prompt_server.routes.put(ctx.remote_environments_route_path)
        async def remote_environments_update(request: web.Request) -> web.Response:
            """Validate and atomically replace SSH host configuration."""
            if registry is None:
                return _unavailable_registry_response()
            try:
                payload = await request.json()
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "Remote environment configuration must be a JSON object."
                    )
                config = RemoteExecutionConfig.from_dict(payload)
                await asyncio.to_thread(registry.save, config)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
            return web.json_response(config.to_dict())

    @prompt_server.routes.post(ctx.remote_environment_probe_route_path)
    async def remote_environment_probe(request: web.Request) -> web.Response:
        """Probe one configured SSH host and persist its discovered capabilities."""
        if registry is None:
            return _unavailable_registry_response()
        environment_id = ""
        try:
            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            host = await asyncio.to_thread(registry.get_host, environment_id)
            capabilities = await asyncio.to_thread(
                SshDockerController(host).probe_capabilities
            )
            updated_host = await asyncio.to_thread(
                registry.update_probe_result,
                environment_id,
                capabilities=capabilities,
                health=EnvironmentHealth.READY,
                last_error=None,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            if environment_id:
                try:
                    await asyncio.to_thread(
                        registry.update_probe_result,
                        environment_id,
                        capabilities=None,
                        health=EnvironmentHealth.UNAVAILABLE,
                        last_error=str(exc),
                    )
                except (KeyError, ValueError):
                    pass
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(updated_host.to_dict())

    @prompt_server.routes.post(ctx.remote_environment_bootstrap_route_path)
    async def remote_environment_bootstrap(request: web.Request) -> web.Response:
        """Build the current runtime and start one compatible warm SSH worker."""
        if registry is None:
            return _unavailable_registry_response()
        environment_id = ""
        try:
            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            worker_index = int(payload.get("worker_index", 0))
            host = await asyncio.to_thread(registry.get_host, environment_id)
            manager = SshRuntimeManager(
                controller=SshDockerController(host),
                repo_root=Path(__file__).resolve().parent,
                settings=ctx.settings,
            )
            spec = await asyncio.to_thread(manager.ensure_worker, worker_index)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            if environment_id:
                try:
                    await asyncio.to_thread(
                        registry.update_probe_result,
                        environment_id,
                        capabilities=host.capabilities if "host" in locals() else None,
                        health=EnvironmentHealth.UNAVAILABLE,
                        last_error=str(exc),
                    )
                except (KeyError, ValueError):
                    pass
            return web.json_response({"error": str(exc)}, status=502)
        await asyncio.to_thread(
            registry.update_probe_result,
            environment_id,
            capabilities=host.capabilities,
            health=EnvironmentHealth.READY,
            last_error=None,
        )
        return web.json_response(
            {
                "environment_id": environment_id,
                "worker_index": worker_index,
                "container_name": spec.container_name,
                "image_tag": spec.image_tag,
                "runtime_fingerprint": spec.identity.fingerprint,
            }
        )

    @prompt_server.routes.post(ctx.remote_environment_status_route_path)
    async def remote_environment_status(request: web.Request) -> web.Response:
        """Return managed worker state for one SSH execution environment."""
        if registry is None:
            return _unavailable_registry_response()
        try:
            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            host = await asyncio.to_thread(registry.get_host, environment_id)
            workers = await asyncio.to_thread(
                SshDockerController(host).list_managed_workers
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {
                "environment_id": environment_id,
                "workers": [worker.to_dict() for worker in workers],
            }
        )

    @prompt_server.routes.post(ctx.remote_environment_stop_route_path)
    async def remote_environment_stop(request: web.Request) -> web.Response:
        """Stop all node-pack-managed workers on one configured SSH host."""
        if registry is None:
            return _unavailable_registry_response()
        try:
            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            host = await asyncio.to_thread(registry.get_host, environment_id)
            manager = SshRuntimeManager(
                controller=SshDockerController(host),
                repo_root=Path(__file__).resolve().parent,
                settings=ctx.settings,
            )
            removed = await asyncio.to_thread(manager.stop_all_workers)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {"environment_id": environment_id, "removed_containers": list(removed)}
        )
