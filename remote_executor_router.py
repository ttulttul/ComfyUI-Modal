"""Provider router and client protocol for remote proxy execution."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

if __package__:
    from .serialization import deserialize_node_outputs, serialize_node_inputs
else:  # pragma: no cover - flat import inside the Modal container.
    from serialization import deserialize_node_outputs, serialize_node_inputs

class RemoteExecutorClient(Protocol):
    """Execution client interface used by Modal proxy nodes."""

    def execute_payload(self, payload: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Sequence[Any]:
        """Execute a serialized Modal payload and return its outputs."""

    async def execute_payload_async(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute a serialized Modal payload asynchronously and return its outputs."""


class ModalRemoteExecutorClient:
    """Default execution client backed by the remote Modal app module."""

    def execute_payload(self, payload: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Sequence[Any]:
        """Serialize inputs, invoke the remote engine, and deserialize outputs."""
        from .remote.modal_app import invoke_remote_engine

        response = invoke_remote_engine(dict(payload), serialize_node_inputs(kwargs))
        return deserialize_node_outputs(response)

    async def execute_payload_async(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Serialize inputs, invoke the remote engine asynchronously, and deserialize outputs."""
        from .remote.modal_app import invoke_remote_engine_async

        response = await invoke_remote_engine_async(dict(payload), serialize_node_inputs(kwargs))
        return deserialize_node_outputs(response)


class RemoteExecutorRouterClient:
    """Route one proxy payload to its selected execution provider."""

    def execute_payload(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one payload through Modal or an assigned SSH Docker host."""
        client = self._client_for_payload(payload)
        started_at = time.monotonic()
        result = client.execute_payload(payload, kwargs)
        self._record_success(payload, time.monotonic() - started_at)
        return result

    async def execute_payload_async(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one payload asynchronously through its selected provider."""
        client = self._client_for_payload(payload)
        started_at = time.monotonic()
        execute_async = getattr(client, "execute_payload_async", None)
        if callable(execute_async):
            result = execute_async(payload, kwargs)
            if inspect.isawaitable(result):
                result = await result
            self._record_success(payload, time.monotonic() - started_at)
            return result
        result = await asyncio.to_thread(client.execute_payload, payload, kwargs)
        self._record_success(payload, time.monotonic() - started_at)
        return result

    def _record_success(
        self,
        payload: Mapping[str, Any],
        elapsed_seconds: float,
    ) -> None:
        """Best-effort persist timing feedback for future cost-aware placement."""
        from .execution_history import ExecutionHistory, record_completed_execution
        from .settings import discover_comfyui_user_directory, get_settings

        settings = get_settings()
        user_directory = discover_comfyui_user_directory(settings)
        history = (
            ExecutionHistory.for_user_directory(user_directory)
            if user_directory is not None
            else None
        )
        provider = str(payload.get("execution_provider") or "modal").strip().lower()
        environment_id = str(
            payload.get("execution_environment_id")
            or f"modal:{settings.modal_gpu}"
        ).strip()
        signature = payload.get("execution_history_signature")
        record_completed_execution(
            history=history,
            component_signature=(str(signature) if signature is not None else None),
            environment_id=environment_id,
            provider=provider,
            elapsed_seconds=elapsed_seconds,
        )

    def _client_for_payload(self, payload: Mapping[str, Any]) -> RemoteExecutorClient:
        """Instantiate the provider client selected by one planned payload."""
        provider = str(payload.get("execution_provider") or "modal").strip().lower()
        if provider == "modal":
            return ModalRemoteExecutorClient()
        if provider == "vast":
            from pathlib import Path

            from .settings import get_settings
            from .vast_service import VastService

            settings = get_settings()
            return VastService.from_environment(
                settings,
                repo_root=Path(__file__).resolve().parent,
            ).executor()
        if provider != "ssh_docker":
            raise ValueError(f"Unsupported remote execution provider {provider!r}.")

        from pathlib import Path

        from .remote_hosts import RemoteHostRegistry
        from .settings import discover_comfyui_user_directory, get_settings
        from .ssh_executor import SshDockerExecutorClient

        settings = get_settings()
        user_directory = discover_comfyui_user_directory(settings)
        return SshDockerExecutorClient(
            registry=(
                RemoteHostRegistry.for_user_directory(user_directory)
                if user_directory is not None
                else None
            ),
            repo_root=Path(__file__).resolve().parent,
            settings=settings,
        )


_REMOTE_EXECUTOR_CLIENT_FACTORY: Callable[[], RemoteExecutorClient] = RemoteExecutorRouterClient

def set_remote_executor_client_factory(
    factory: Callable[[], RemoteExecutorClient] | None,
) -> None:
    """Install a custom client factory, primarily for tests."""
    global _REMOTE_EXECUTOR_CLIENT_FACTORY
    _REMOTE_EXECUTOR_CLIENT_FACTORY = factory or RemoteExecutorRouterClient


def get_remote_executor_client() -> RemoteExecutorClient:
    """Instantiate the configured execution client."""
    return _REMOTE_EXECUTOR_CLIENT_FACTORY()


