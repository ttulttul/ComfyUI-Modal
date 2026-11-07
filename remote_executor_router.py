"""Provider router and client protocol for remote proxy execution."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

if __package__:
    from .serialization import (
        deserialize_node_inputs,
        deserialize_node_outputs,
        serialize_node_inputs,
        serialize_node_outputs,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from serialization import (
        deserialize_node_inputs,
        deserialize_node_outputs,
        serialize_node_inputs,
        serialize_node_outputs,
    )

logger = logging.getLogger(__name__)

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
        if _requires_provider_mapped_fan_out(client, payload, kwargs):
            result = asyncio.run(_execute_provider_mapped_fan_out(client, payload, kwargs))
        else:
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
        if _requires_provider_mapped_fan_out(client, payload, kwargs):
            result = await _execute_provider_mapped_fan_out(client, payload, kwargs)
        else:
            result = await _execute_client_payload_async(client, payload, kwargs)
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
                runtime_fingerprint=_vast_payload_value(
                    payload,
                    "vast_runtime_fingerprint",
                ),
                worker_image=_vast_payload_value(payload, "vast_worker_image"),
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


def _requires_provider_mapped_fan_out(
    client: RemoteExecutorClient,
    payload: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> bool:
    """Return whether a non-Modal provider must fan out zipped mapped inputs on the host.

    The Modal client performs this fan-out inside ``invoke_remote_engine_async``.
    SSH and Vast workers execute exactly the payload they receive, so without a
    host-side split a ``ModalMapInput`` list would reach one node as a single value.
    """
    if isinstance(client, ModalRemoteExecutorClient):
        return False
    if payload.get("payload_kind") != "subgraph":
        return False
    from .remote.modal_app import _split_batch_boundary_inputs

    return _split_batch_boundary_inputs(dict(payload), dict(kwargs)) is not None


async def _execute_provider_mapped_fan_out(
    client: RemoteExecutorClient,
    payload: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Sequence[Any]:
    """Run the shared mapped orchestration with every phase executed by ``client``."""
    from .remote.modal_app import _invoke_implicitly_mapped_subgraph_async

    async def invoke_with_client(phase_payload: dict[str, Any], phase_inputs: bytes) -> bytes:
        """Execute one phase, item, or cleanup payload on the provider client."""
        outputs = await _execute_client_payload_async(
            client,
            phase_payload,
            deserialize_node_inputs(phase_inputs),
        )
        return serialize_node_outputs(tuple(outputs))

    logger.info(
        "Fanning out mapped component=%s on provider=%s through the host orchestrator.",
        payload.get("component_id"),
        payload.get("execution_provider"),
    )
    response = await _invoke_implicitly_mapped_subgraph_async(
        dict(payload),
        serialize_node_inputs(kwargs),
        invoker=invoke_with_client,
    )
    return deserialize_node_outputs(response)


async def _execute_client_payload_async(
    client: RemoteExecutorClient,
    payload: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Sequence[Any]:
    """Execute one payload on a client, adapting synchronous clients to a thread."""
    execute_async = getattr(client, "execute_payload_async", None)
    if callable(execute_async):
        result = execute_async(payload, kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result
    return await asyncio.to_thread(client.execute_payload, payload, kwargs)


_REMOTE_EXECUTOR_CLIENT_FACTORY: Callable[[], RemoteExecutorClient] = RemoteExecutorRouterClient


def _vast_payload_value(payload: Mapping[str, Any], key: str) -> str | None:
    """Return one non-empty queue-time Vast runtime expectation value."""
    value = str(payload.get(key) or "").strip()
    return value or None


def set_remote_executor_client_factory(
    factory: Callable[[], RemoteExecutorClient] | None,
) -> None:
    """Install a custom client factory, primarily for tests."""
    global _REMOTE_EXECUTOR_CLIENT_FACTORY
    _REMOTE_EXECUTOR_CLIENT_FACTORY = factory or RemoteExecutorRouterClient


def get_remote_executor_client() -> RemoteExecutorClient:
    """Instantiate the configured execution client."""
    return _REMOTE_EXECUTOR_CLIENT_FACTORY()
