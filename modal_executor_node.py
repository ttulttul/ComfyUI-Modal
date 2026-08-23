"""Dynamic Modal proxy nodes for ComfyUI execution."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from comfy_api.latest import _io as io

from .serialization import deserialize_node_outputs, serialize_node_inputs, split_mapped_value

logger = logging.getLogger(__name__)
MODAL_MAP_INPUT_NODE_ID = "ModalMapInput"
MODAL_ARTIFACT_FINALIZER_NODE_ID = "ModalArtifactFinalizer"
MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID = "ModalParallelLocalPassthrough"
MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID = "ModalLocalBridgeMaterializer"
MODAL_COMPONENT_COMPLETION_OUTPUT_NAME = "modal_component_complete"
MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS = 100
MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY = "comfy_modal_prompt_id"
_PROXY_CACHE_CONTEXT_ID_KEY = "__comfy_modal_proxy_cache_context_id__"
_VOLATILE_PROXY_CACHE_KEYS = frozenset(
    {
        "prompt_id",
        "remote_session",
        "clear_remote_session",
        "extra_data",
        "requires_volume_reload",
        "volume_reload_marker",
        "uploaded_volume_paths",
        "speculative_remote_prewarm_target",
    }
)


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
        if user_directory is None:
            raise RuntimeError(
                "SSH execution requires a persistent ComfyUI user directory."
            )
        return SshDockerExecutorClient(
            registry=RemoteHostRegistry.for_user_directory(user_directory),
            repo_root=Path(__file__).resolve().parent,
            settings=settings,
        )


_REMOTE_EXECUTOR_CLIENT_FACTORY: Callable[[], RemoteExecutorClient] = RemoteExecutorRouterClient
_PROXY_NODE_CACHE: dict[str, type[io.ComfyNode]] = {}
_PROXY_EXECUTION_CONTEXTS_LOCK = threading.Lock()
_PROXY_EXECUTION_CONTEXTS: dict[str, "_ProxyExecutionContext"] = {}
_PROXY_EXECUTION_CONTEXTS_BY_PROMPT: OrderedDict[
    tuple[str, str], "_ProxyExecutionContext"
] = OrderedDict()
_PROXY_EXECUTION_CONTEXT_LIMIT = 2048
_MODAL_MAP_WARMUP_CONTEXTS_LOCK = threading.Lock()
_MODAL_MAP_WARMUP_CONTEXTS: dict[str, "_ModalMapWarmupContext"] = {}
_MODAL_WORKFLOW_EXECUTION_GATE = threading.Condition()
_MODAL_WORKFLOW_ACTIVE_PROMPT_ID: str | None = None
_MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS = 0
_MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID: dict[str, int] = {}
_MODAL_PARALLEL_DISPATCH_EVENTS_LOCK = threading.Lock()
_MODAL_PARALLEL_DISPATCH_EVENTS: OrderedDict[
    tuple[str, str], asyncio.Event
] = OrderedDict()
_MODAL_PARALLEL_DISPATCH_EVENT_LIMIT = 512


@dataclass(frozen=True)
class _ProxyExecutionContext:
    """Run-scoped execution context used to rehydrate cache-friendly proxy payloads."""

    execution_payload: dict[str, Any]


@dataclass(frozen=True)
class _ModalMapWarmupContext:
    """Run-scoped warmup context used by one local Modal Map Input node."""

    execution_payload: dict[str, Any]
    mapped_io_type: str


def set_remote_executor_client_factory(
    factory: Callable[[], RemoteExecutorClient] | None,
) -> None:
    """Install a custom client factory, primarily for tests."""
    global _REMOTE_EXECUTOR_CLIENT_FACTORY
    _REMOTE_EXECUTOR_CLIENT_FACTORY = factory or RemoteExecutorRouterClient


def get_remote_executor_client() -> RemoteExecutorClient:
    """Instantiate the configured execution client."""
    return _REMOTE_EXECUTOR_CLIENT_FACTORY()


def _acquire_modal_workflow_execution_slot(
    prompt_id: str | None,
    component_id: Any,
) -> bool:
    """Try to reserve a remote execution slot for one prompt."""
    if prompt_id is None:
        return False

    global _MODAL_WORKFLOW_ACTIVE_PROMPT_ID
    global _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS

    with _MODAL_WORKFLOW_EXECUTION_GATE:
        if (
            _MODAL_WORKFLOW_ACTIVE_PROMPT_ID is not None
            and _MODAL_WORKFLOW_ACTIVE_PROMPT_ID != prompt_id
        ):
            return False

        _MODAL_WORKFLOW_ACTIVE_PROMPT_ID = prompt_id
        _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS += 1
        logger.debug(
            "Acquired Modal workflow execution slot for prompt=%s component=%s active_remote_calls=%d.",
            prompt_id,
            component_id,
            _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS,
        )
    return True


def _release_modal_workflow_execution_slot(prompt_id: str | None, component_id: Any) -> None:
    """Release a prompt-scoped remote execution slot and unblock the next prompt if idle."""
    if prompt_id is None:
        return

    global _MODAL_WORKFLOW_ACTIVE_PROMPT_ID
    global _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS
    global _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID

    with _MODAL_WORKFLOW_EXECUTION_GATE:
        if _MODAL_WORKFLOW_ACTIVE_PROMPT_ID != prompt_id:
            abandoned_release_count = _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID.get(
                prompt_id,
                0,
            )
            if abandoned_release_count > 0:
                logger.debug(
                    "Ignoring Modal workflow gate release for abandoned prompt=%s component=%s.",
                    prompt_id,
                    component_id,
                )
                if abandoned_release_count == 1:
                    _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID.pop(prompt_id, None)
                else:
                    _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID[prompt_id] = (
                        abandoned_release_count - 1
                    )
            else:
                logger.warning(
                    "Ignoring Modal workflow gate release for prompt=%s component=%s because active prompt is %s.",
                    prompt_id,
                    component_id,
                    _MODAL_WORKFLOW_ACTIVE_PROMPT_ID,
                )
            return

        _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS = max(0, _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS - 1)
        logger.debug(
            "Released Modal workflow execution slot for prompt=%s component=%s active_remote_calls=%d.",
            prompt_id,
            component_id,
            _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS,
        )
        if _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS == 0:
            _MODAL_WORKFLOW_ACTIVE_PROMPT_ID = None
            _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID.pop(prompt_id, None)
            _MODAL_WORKFLOW_EXECUTION_GATE.notify_all()


def abandon_modal_workflow_execution_prompt(prompt_id: str | None, reason: str) -> None:
    """Release a prompt-wide Modal gate when ComfyUI has cancelled that prompt locally."""
    normalized_prompt_id = _normalize_prompt_id(prompt_id)
    if normalized_prompt_id is None:
        return

    _clear_parallel_dispatch_events(normalized_prompt_id)

    global _MODAL_WORKFLOW_ACTIVE_PROMPT_ID
    global _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS
    global _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID

    with _MODAL_WORKFLOW_EXECUTION_GATE:
        if _MODAL_WORKFLOW_ACTIVE_PROMPT_ID != normalized_prompt_id:
            return

        logger.info(
            "Abandoning Modal workflow execution slot for prompt=%s with %d active remote call(s): %s.",
            normalized_prompt_id,
            _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS,
            reason,
        )
        _MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID[normalized_prompt_id] = (
            _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS
        )
        _MODAL_WORKFLOW_ACTIVE_PROMPT_ID = None
        _MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS = 0
        _MODAL_WORKFLOW_EXECUTION_GATE.notify_all()


@asynccontextmanager
async def _modal_workflow_execution_slot(payload: Mapping[str, Any]) -> AsyncIterator[None]:
    """Async context that serializes remote work across different prompt ids."""
    prompt_id = _normalize_prompt_id(payload.get("prompt_id"))
    component_id = payload.get("component_id")
    if prompt_id is None:
        yield
        return

    acquired = False
    waiting_logged = False
    while not acquired:
        acquired = await asyncio.to_thread(
            _acquire_modal_workflow_execution_slot,
            prompt_id,
            component_id,
        )
        if not acquired:
            if not waiting_logged:
                logger.info(
                    "Waiting to start Modal component=%s for prompt=%s until active prompt=%s finishes remote execution.",
                    component_id,
                    prompt_id,
                    _MODAL_WORKFLOW_ACTIVE_PROMPT_ID,
                )
                waiting_logged = True
            await asyncio.sleep(0.1)
    try:
        yield
    finally:
        if acquired:
            await asyncio.to_thread(
                _release_modal_workflow_execution_slot,
                prompt_id,
                component_id,
            )


def _parallel_dispatch_key(
    dispatch_group_id: Any,
    component_id: Any,
) -> tuple[str, str] | None:
    """Return one normalized parallel-preview dispatch key."""
    normalized_group_id = _normalize_prompt_id(dispatch_group_id)
    normalized_component_id = _normalize_prompt_id(component_id)
    if normalized_group_id is None or normalized_component_id is None:
        return None
    return (normalized_group_id, normalized_component_id)


def _parallel_dispatch_event(
    dispatch_group_id: Any,
    component_id: Any,
) -> asyncio.Event | None:
    """Return the bounded dispatch event for one remote continuation."""
    dispatch_key = _parallel_dispatch_key(dispatch_group_id, component_id)
    if dispatch_key is None:
        return None
    with _MODAL_PARALLEL_DISPATCH_EVENTS_LOCK:
        dispatch_event = _MODAL_PARALLEL_DISPATCH_EVENTS.get(dispatch_key)
        if dispatch_event is None:
            dispatch_event = asyncio.Event()
            _MODAL_PARALLEL_DISPATCH_EVENTS[dispatch_key] = dispatch_event
            while len(_MODAL_PARALLEL_DISPATCH_EVENTS) > (
                _MODAL_PARALLEL_DISPATCH_EVENT_LIMIT
            ):
                completed_dispatch_key = next(
                    (
                        candidate_key
                        for candidate_key, candidate_event in (
                            _MODAL_PARALLEL_DISPATCH_EVENTS.items()
                        )
                        if candidate_event.is_set()
                    ),
                    None,
                )
                if completed_dispatch_key is None:
                    break
                _MODAL_PARALLEL_DISPATCH_EVENTS.pop(completed_dispatch_key, None)
        else:
            _MODAL_PARALLEL_DISPATCH_EVENTS.move_to_end(dispatch_key)
    return dispatch_event


def _signal_parallel_local_dispatch(payload: Mapping[str, Any]) -> bool:
    """Signal that one remote continuation acquired its local dispatch slot."""
    if not bool(payload.get("signal_parallel_local_dispatch")):
        return False
    dispatch_event = _parallel_dispatch_event(
        payload.get("parallel_local_dispatch_group_id"),
        payload.get("component_id"),
    )
    if dispatch_event is None:
        return False
    dispatch_event.set()
    logger.info(
        "Released parallel local branches after remote dispatch started group=%s component=%s.",
        payload.get("parallel_local_dispatch_group_id"),
        payload.get("component_id"),
    )
    return True


async def _wait_for_parallel_local_dispatches(
    dispatch_context: Mapping[str, Any],
) -> None:
    """Wait until every nearest downstream remote continuation has started."""
    dispatch_group_id = dispatch_context.get("dispatch_group_id")
    component_ids = dispatch_context.get("component_ids")
    if not isinstance(component_ids, Sequence) or isinstance(
        component_ids,
        str | bytes | bytearray,
    ):
        raise TypeError("Parallel local dispatch context must define component_ids.")
    dispatch_events = [
        dispatch_event
        for component_id in component_ids
        if (
            dispatch_event := _parallel_dispatch_event(
                dispatch_group_id,
                component_id,
            )
        )
        is not None
    ]
    if not dispatch_events:
        raise RuntimeError(
            "Parallel local dispatch context did not identify a remote continuation."
        )
    await asyncio.gather(*(dispatch_event.wait() for dispatch_event in dispatch_events))


def _clear_parallel_dispatch_events(dispatch_group_id: str) -> None:
    """Forget dispatch events for one completed or abandoned prompt group."""
    with _MODAL_PARALLEL_DISPATCH_EVENTS_LOCK:
        dispatch_keys = [
            dispatch_key
            for dispatch_key in _MODAL_PARALLEL_DISPATCH_EVENTS
            if dispatch_key[0] == dispatch_group_id
        ]
        for dispatch_key in dispatch_keys:
            _MODAL_PARALLEL_DISPATCH_EVENTS.pop(dispatch_key, None)


async def _execute_payload_async(
    client: RemoteExecutorClient,
    payload: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Sequence[Any]:
    """Execute one Modal payload through the client, adapting sync clients when needed."""
    execute_payload_async = getattr(client, "execute_payload_async", None)
    if callable(execute_payload_async):
        result = execute_payload_async(payload, kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    execute_payload = getattr(client, "execute_payload", None)
    if not callable(execute_payload):
        raise TypeError("Remote executor client must define execute_payload or execute_payload_async.")
    return await asyncio.to_thread(execute_payload, payload, kwargs)


def _output_spec(io_type: str, name: str, is_list: bool) -> io.Output:
    """Create a v3 output specification from a legacy ComfyUI return type."""
    comfy_type = io.AnyType if io_type == "*" else io.Custom(io_type)
    return comfy_type.Output(display_name=name, is_output_list=is_list)


def _unwrap_proxy_singleton(value: Any) -> Any:
    """Unwrap one value wrapped by ComfyUI for an INPUT_IS_LIST proxy."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _pop_proxy_hidden_value(
    proxy_class: type[Any],
    kwargs: dict[str, Any],
    hidden_input: io.Hidden,
) -> Any:
    """Read a V3 class-clone hidden value with a legacy kwargs fallback."""
    legacy_value = kwargs.pop(hidden_input.name, None)
    hidden_holder = getattr(proxy_class, "hidden", None)
    hidden_value = getattr(hidden_holder, hidden_input.name, None)
    return legacy_value if hidden_value is None else hidden_value


def _normalize_proxy_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ComfyUI INPUT_IS_LIST proxy kwargs back into ordinary runtime values."""
    return {
        str(input_name): _unwrap_proxy_singleton(input_value)
        for input_name, input_value in kwargs.items()
    }


def _normalize_proxy_payload(payload: Any) -> Mapping[str, Any]:
    """Convert ComfyUI INPUT_IS_LIST payload wrappers back into one payload mapping."""
    payload = _unwrap_proxy_singleton(payload)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError("original_node_data must be a mapping or JSON object.")
    return payload


def _normalize_scheduler_list_outputs(
    payload: Mapping[str, Any],
    outputs: Sequence[Any],
) -> tuple[Any, ...]:
    """Make remote output containers match the proxy's scheduler-list declarations."""
    normalized_outputs = list(outputs)
    boundary_outputs = payload.get("boundary_outputs")
    if not isinstance(boundary_outputs, Sequence) or isinstance(
        boundary_outputs,
        str | bytes | bytearray,
    ):
        return tuple(normalized_outputs)

    for output_index, boundary_output in enumerate(boundary_outputs):
        if output_index >= len(normalized_outputs):
            break
        if not isinstance(boundary_output, Mapping):
            continue
        if not bool(boundary_output.get("scheduler_is_list", False)):
            continue
        if isinstance(normalized_outputs[output_index], list):
            continue
        normalized_outputs[output_index] = [normalized_outputs[output_index]]
        logger.debug(
            "Wrapped singleton remote output %d for component=%s to satisfy its scheduler-list contract.",
            output_index,
            payload.get("component_id"),
        )

    return tuple(normalized_outputs)


def _normalize_prompt_id(value: Any) -> str | None:
    """Return one non-empty prompt id string when available."""
    if value is None:
        return None
    prompt_id = str(value).strip()
    return prompt_id or None


def _payload_is_local_cache_safe(payload: Mapping[str, Any]) -> bool:
    """Return whether one proxy payload can safely reuse local ComfyUI outputs across prompt runs."""
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, Mapping):
        return all(
            isinstance(nested_payload, Mapping) and _payload_is_local_cache_safe(nested_payload)
            for nested_payload in split_proxy_payloads.values()
        )
    if isinstance(split_proxy_payloads, Sequence) and not isinstance(split_proxy_payloads, (str, bytes, bytearray)):
        return all(
            isinstance(nested_payload, Mapping) and _payload_is_local_cache_safe(nested_payload)
            for nested_payload in split_proxy_payloads
        )

    for phase_name in ("static_phase", "mapped_phase"):
        phase_payload = payload.get(phase_name)
        if isinstance(phase_payload, Mapping) and not _payload_is_local_cache_safe(phase_payload):
            return False
    return True


def _sanitize_cache_surface_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip run-scoped fields from one proxy payload before exposing it to ComfyUI caching."""
    sanitized_payload = dict(payload)
    for field_name in _VOLATILE_PROXY_CACHE_KEYS:
        sanitized_payload.pop(field_name, None)

    split_proxy_payloads = sanitized_payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, Mapping):
        sanitized_payload["split_proxy_payloads"] = {
            str(phase_name): _sanitize_cache_surface_payload(dict(phase_payload))
            for phase_name, phase_payload in split_proxy_payloads.items()
            if isinstance(phase_payload, Mapping)
        }
    elif isinstance(split_proxy_payloads, Sequence) and not isinstance(
        split_proxy_payloads,
        (str, bytes, bytearray),
    ):
        sanitized_payload["split_proxy_payloads"] = [
            _sanitize_cache_surface_payload(dict(phase_payload))
            for phase_payload in split_proxy_payloads
            if isinstance(phase_payload, Mapping)
        ]

    for phase_name in ("static_phase", "mapped_phase"):
        phase_payload = sanitized_payload.get(phase_name)
        if isinstance(phase_payload, Mapping):
            sanitized_payload[phase_name] = _sanitize_cache_surface_payload(dict(phase_payload))
    return sanitized_payload


def register_cache_friendly_proxy_payload(
    node_id: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the payload that should be embedded in the proxy node input for local cache reuse."""
    if not _payload_is_local_cache_safe(payload):
        return dict(payload)

    sanitized_payload = _sanitize_cache_surface_payload(payload)
    sanitized_payload[_PROXY_CACHE_CONTEXT_ID_KEY] = str(node_id)
    prompt_id = _normalize_prompt_id(payload.get("prompt_id"))
    context = _ProxyExecutionContext(execution_payload=dict(payload))
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        _PROXY_EXECUTION_CONTEXTS[str(node_id)] = context
        if prompt_id is not None:
            context_key = (prompt_id, str(node_id))
            _PROXY_EXECUTION_CONTEXTS_BY_PROMPT[context_key] = context
            _PROXY_EXECUTION_CONTEXTS_BY_PROMPT.move_to_end(context_key)
            while len(_PROXY_EXECUTION_CONTEXTS_BY_PROMPT) > _PROXY_EXECUTION_CONTEXT_LIMIT:
                _PROXY_EXECUTION_CONTEXTS_BY_PROMPT.popitem(last=False)
    logger.debug(
        "Registered cache-friendly Modal proxy payload for node_id=%s prompt_id=%s session_backed=%s.",
        node_id,
        prompt_id,
        payload.get("remote_session") is not None,
    )
    return sanitized_payload


def update_registered_proxy_payload_fields(
    node_id: str,
    embedded_payload: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Update both the embedded cache surface and run-scoped execution payload."""
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        context = _PROXY_EXECUTION_CONTEXTS.get(str(node_id))
        execution_payload = dict(
            context.execution_payload if context is not None else embedded_payload
        )
    execution_payload.update(fields)
    return register_cache_friendly_proxy_payload(node_id, execution_payload)


def registered_proxy_execution_payload(
    node_id: str,
    embedded_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the full run-scoped payload registered for one proxy node."""
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        context = _PROXY_EXECUTION_CONTEXTS.get(str(node_id))
        return dict(
            context.execution_payload if context is not None else embedded_payload
        )


def register_modal_map_input_warmup_context(
    node_id: str,
    payload: Mapping[str, Any],
    mapped_io_type: str,
) -> None:
    """Register prompt-scoped warmup metadata for one local Modal Map Input node."""
    with _MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        _MODAL_MAP_WARMUP_CONTEXTS[str(node_id)] = _ModalMapWarmupContext(
            execution_payload=dict(payload),
            mapped_io_type=str(mapped_io_type or "*"),
        )
    logger.debug(
        "Registered Modal Map Input warmup context for node_id=%s component_id=%s prompt_id=%s io_type=%s.",
        node_id,
        payload.get("component_id"),
        _normalize_prompt_id(payload.get("prompt_id")),
        mapped_io_type,
    )


def _boost_modal_map_input_warmup(
    unique_id: str | None,
    value: Any,
) -> None:
    """Best-effort exact warmup boost for one local Modal Map Input execution."""
    if unique_id is None:
        return

    with _MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        context = _MODAL_MAP_WARMUP_CONTEXTS.get(str(unique_id))
    if context is None:
        return

    try:
        total_items = len(split_mapped_value(value, context.mapped_io_type))
    except (TypeError, ValueError) as exc:
        logger.debug(
            "Skipping Modal Map Input warmup boost for node_id=%s because the runtime value was not splittable as io_type=%s: %s",
            unique_id,
            context.mapped_io_type,
            exc,
        )
        return

    from .remote.modal_app import boost_mapped_component_warmup

    boost_mapped_component_warmup(
        payload=context.execution_payload,
        total_items=total_items,
        reason="modal_map_input_execute",
    )


def _rehydrate_proxy_payload(
    payload: Mapping[str, Any],
    *,
    unique_id: str | None,
    prompt_id: str | None = None,
) -> Mapping[str, Any]:
    """Restore any execution-scoped fields stripped from a cache-friendly proxy payload."""
    candidate_context_id = payload.get(_PROXY_CACHE_CONTEXT_ID_KEY)
    if candidate_context_id is None:
        return payload

    context_id = unique_id
    if context_id is None:
        normalized_context_id = str(candidate_context_id).strip()
        context_id = normalized_context_id or None
    if context_id is None:
        return payload

    normalized_prompt_id = _normalize_prompt_id(prompt_id)
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        if normalized_prompt_id is not None:
            context_key = (normalized_prompt_id, str(context_id))
            context = _PROXY_EXECUTION_CONTEXTS_BY_PROMPT.pop(context_key, None)
        else:
            context = _PROXY_EXECUTION_CONTEXTS.get(str(context_id))
    if context is None:
        if normalized_prompt_id is not None:
            logger.warning(
                "No prompt-scoped Modal proxy payload found for prompt_id=%s node_id=%s; using embedded cache surface.",
                normalized_prompt_id,
                context_id,
            )
        return payload

    return dict(context.execution_payload)


def _prompt_id_from_extra_pnginfo(extra_pnginfo: Any) -> str | None:
    """Read the queue prompt id carried through ComfyUI's hidden PNG metadata input."""
    extra_pnginfo = _unwrap_proxy_singleton(extra_pnginfo)
    if not isinstance(extra_pnginfo, Mapping):
        return None
    return _normalize_prompt_id(extra_pnginfo.get(MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY))


def _normalized_output_metadata(
    original_class: type[Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    """Normalize output metadata from a source node class."""
    if hasattr(original_class, "GET_SCHEMA"):
        original_class.GET_SCHEMA()

    output_types = tuple(getattr(original_class, "RETURN_TYPES", ("*",))) or ("*",)
    default_names = tuple(f"output_{index}" for index, _ in enumerate(output_types))
    output_names = tuple(getattr(original_class, "RETURN_NAMES", default_names))
    output_is_list = tuple(getattr(original_class, "OUTPUT_IS_LIST", (False,) * len(output_types)))

    if len(output_names) < len(output_types):
        output_names = output_names + default_names[len(output_names) :]
    if len(output_is_list) < len(output_types):
        output_is_list = output_is_list + (False,) * (len(output_types) - len(output_is_list))

    return output_types, output_names[: len(output_types)], output_is_list[: len(output_types)]


def _proxy_node_id(original_class_type: str, output_types: Sequence[str]) -> str:
    """Build a stable proxy node identifier for an original node signature."""
    digest = hashlib.sha256(
        json.dumps({"class_type": original_class_type, "outputs": list(output_types)}).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"ModalUniversalExecutor_{digest}"


def _build_proxy_node_class(
    node_id: str,
    proxy_display_name: str,
    payload_input_name: str,
    output_types: tuple[str, ...],
    output_names: tuple[str, ...],
    output_is_list: tuple[bool, ...],
    *,
    is_output_node: bool,
    include_completion_output: bool = False,
) -> type[io.ComfyNode]:
    """Create a v3 proxy node that mirrors an original node output signature."""

    class _DynamicModalExecutor(io.ComfyNode):
        """Internal proxy node that forwards execution to Modal."""

        OUTPUT_NODE = is_output_node

        @classmethod
        def define_schema(cls) -> io.Schema:
            """Return a schema that accepts any original node inputs."""
            outputs = [
                _output_spec(io_type, name, is_list)
                for io_type, name, is_list in zip(output_types, output_names, output_is_list, strict=False)
            ]
            if include_completion_output:
                outputs.append(
                    io.Boolean.Output(
                        display_name=MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
                    )
                )
            return io.Schema(
                node_id=node_id,
                display_name=proxy_display_name,
                category="Modal",
                description=(
                    "Internal proxy node that forwards a rewritten Modal execution "
                    "payload to a Modal-backed runtime."
                ),
                inputs=[
                    io.AnyType.Input(payload_input_name),
                ],
                outputs=outputs,
                is_input_list=True,
                accept_all_inputs=True,
                hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
                is_dev_only=True,
                is_experimental=True,
            )

        @classmethod
        async def execute(cls, **kwargs: Any) -> io.NodeOutput:
            """Forward the execution payload to the configured remote executor."""
            unique_id = _normalize_prompt_id(
                _unwrap_proxy_singleton(
                    _pop_proxy_hidden_value(cls, kwargs, io.Hidden.unique_id)
                )
            )
            prompt_id = _prompt_id_from_extra_pnginfo(
                _pop_proxy_hidden_value(cls, kwargs, io.Hidden.extra_pnginfo)
            )
            payload = _rehydrate_proxy_payload(
                _normalize_proxy_payload(kwargs.pop(payload_input_name, None)),
                unique_id=unique_id,
                prompt_id=prompt_id,
            )

            async with _modal_workflow_execution_slot(payload):
                _signal_parallel_local_dispatch(payload)
                outputs = _normalize_scheduler_list_outputs(
                    payload,
                    await _execute_payload_async(
                        get_remote_executor_client(),
                        payload,
                        _normalize_proxy_kwargs(kwargs),
                    ),
                )
            logger.debug(
                "Remote execution completed for payload kind=%s with %d outputs.",
                payload.get("payload_kind"),
                len(outputs),
            )
            if include_completion_output:
                return io.NodeOutput(*outputs, True)
            return io.NodeOutput(*outputs)

    _DynamicModalExecutor.__name__ = f"DynamicModalExecutor_{node_id}"
    return _DynamicModalExecutor


def ensure_modal_proxy_node_registered(
    original_class_type: str,
    original_class: type[Any],
    nodes_module: Any,
) -> str:
    """Register and return a proxy node id for the supplied original node class."""
    output_types, output_names, output_is_list = _normalized_output_metadata(original_class)
    proxy_node_id = _proxy_node_id(original_class_type, output_types)

    if proxy_node_id in _PROXY_NODE_CACHE:
        _register_modal_node(
            nodes_module,
            proxy_node_id,
            _PROXY_NODE_CACHE[proxy_node_id],
            "Modal Universal Executor",
        )
        return proxy_node_id

    proxy_class = _build_proxy_node_class(
        node_id=proxy_node_id,
        proxy_display_name="Modal Universal Executor",
        payload_input_name="original_node_data",
        output_types=output_types,
        output_names=output_names,
        output_is_list=output_is_list,
        is_output_node=False,
    )
    _register_modal_node(
        nodes_module,
        proxy_node_id,
        proxy_class,
        "Modal Universal Executor",
    )
    _PROXY_NODE_CACHE[proxy_node_id] = proxy_class
    logger.info("Registered Modal proxy node %s for %s", proxy_node_id, original_class_type)
    return proxy_node_id


def ensure_modal_component_proxy_node_registered(
    output_types: Sequence[str],
    output_names: Sequence[str],
    output_is_list: Sequence[bool],
    nodes_module: Any,
    *,
    is_output_node: bool,
    include_completion_output: bool = False,
) -> str:
    """Register and return a proxy node id for a remote component signature."""
    normalized_output_types = tuple(output_types)
    normalized_output_names = tuple(output_names)
    normalized_output_is_list = tuple(output_is_list)
    proxy_node_id = _proxy_node_id(
        "ModalRemoteComponent",
        normalized_output_types
        + tuple(f"name:{name}" for name in normalized_output_names)
        + tuple(f"list:{is_list}" for is_list in normalized_output_is_list)
        + (str(is_output_node), str(include_completion_output)),
    )

    if proxy_node_id in _PROXY_NODE_CACHE:
        _register_modal_node(
            nodes_module,
            proxy_node_id,
            _PROXY_NODE_CACHE[proxy_node_id],
            "Modal Remote Component",
        )
        return proxy_node_id

    proxy_class = _build_proxy_node_class(
        node_id=proxy_node_id,
        proxy_display_name="Modal Remote Component",
        payload_input_name="original_node_data",
        output_types=normalized_output_types,
        output_names=normalized_output_names,
        output_is_list=normalized_output_is_list,
        is_output_node=is_output_node,
        include_completion_output=include_completion_output,
    )
    _register_modal_node(
        nodes_module,
        proxy_node_id,
        proxy_class,
        "Modal Remote Component",
    )
    _PROXY_NODE_CACHE[proxy_node_id] = proxy_class
    logger.info("Registered Modal component proxy node %s", proxy_node_id)
    return proxy_node_id


def _register_modal_node(
    nodes_module: Any,
    node_id: str,
    node_class: type[io.ComfyNode],
    display_name: str,
) -> None:
    """Register a runtime Modal node with the plugin's ComfyUI module identity."""
    node_class.RELATIVE_PYTHON_MODULE = ModalUniversalExecutor.RELATIVE_PYTHON_MODULE
    nodes_module.NODE_CLASS_MAPPINGS[node_id] = node_class
    nodes_module.NODE_DISPLAY_NAME_MAPPINGS[node_id] = display_name


def ensure_modal_artifact_finalizer_registered(nodes_module: Any) -> None:
    """Register the internal artifact-finalization sink in a ComfyUI node mapping."""
    _register_modal_node(
        nodes_module,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        ModalArtifactFinalizer,
        "Modal Artifact Finalizer",
    )


def ensure_modal_parallel_local_passthrough_registered(nodes_module: Any) -> None:
    """Register the internal parallel local-branch dispatch gate."""
    _register_modal_node(
        nodes_module,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ModalParallelLocalPassthrough,
        "Modal Parallel Local Passthrough",
    )


def ensure_modal_local_bridge_materializer_registered(nodes_module: Any) -> None:
    """Register the internal durable-bridge local materializer."""
    _register_modal_node(
        nodes_module,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        ModalLocalBridgeMaterializer,
        "Modal Local Bridge Materializer",
    )


class ModalUniversalExecutor(io.ComfyNode):
    """Base debug node for Modal execution routing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define a minimal dev-only schema for the base proxy node."""
        return io.Schema(
            node_id="ModalUniversalExecutor",
            display_name="Modal Universal Executor",
            category="Modal",
            description=(
                "Debug entrypoint for Modal-backed execution. Production rewrites use "
                "signature-preserving dynamic proxy variants."
            ),
            inputs=[io.AnyType.Input("original_node_data")],
            outputs=[io.AnyType.Output(display_name="output")],
            accept_all_inputs=True,
            is_dev_only=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, original_node_data: Any, **kwargs: Any) -> io.NodeOutput:
        """Execute the base debug proxy node through the configured client."""
        if isinstance(original_node_data, str):
            original_node_data = json.loads(original_node_data)
        outputs = tuple(get_remote_executor_client().execute_node(original_node_data, kwargs))
        return io.NodeOutput(*outputs)


class ModalMapInput(io.ComfyNode):
    """Queue-time marker node that turns one remote boundary input into mapped parallel work."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose a simple any-to-any adapter node for mapped remote execution."""
        return io.Schema(
            node_id=MODAL_MAP_INPUT_NODE_ID,
            display_name="Modal Map Input",
            category="Modal",
            description=(
                "Pass-through marker for data-parallel Modal execution. "
                "When used inside a remote-marked component, list inputs and batched tensors "
                "can fan out across multiple Modal executions and reassemble automatically."
            ),
            inputs=[io.AnyType.Input("value")],
            outputs=[io.AnyType.Output(display_name="value")],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, value: Any, unique_id: str | None = None) -> io.NodeOutput:
        """Pass the input value through unchanged at runtime."""
        _boost_modal_map_input_warmup(
            _normalize_prompt_id(unique_id),
            value,
        )
        return io.NodeOutput(value)


class ModalArtifactFinalizer(io.ComfyNode):
    """Internal output sink that anchors remote execution and artifact materialization."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Accept one completion token from every rewritten terminal Modal component."""
        completion_template = io.Autogrow.TemplatePrefix(
            input=io.Boolean.Input("completion", force_input=True),
            prefix="component_",
            min=1,
            max=MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        )
        return io.Schema(
            node_id=MODAL_ARTIFACT_FINALIZER_NODE_ID,
            display_name="Modal Artifact Finalizer",
            category="Modal",
            description=(
                "Internal output sink inserted into rewritten prompts so Modal components "
                "run and materialize their remote output artifacts locally."
            ),
            inputs=[
                io.Autogrow.Input(
                    "components",
                    template=completion_template,
                )
            ],
            outputs=[],
            is_output_node=True,
            is_dev_only=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, components: io.Autogrow.Type) -> io.NodeOutput:
        """Confirm every upstream Modal component completed before the prompt finishes."""
        incomplete_component_names = [
            str(component_name)
            for component_name, completed in components.items()
            if completed is not True
        ]
        if incomplete_component_names:
            raise RuntimeError(
                "Modal artifact finalization received incomplete component tokens: "
                f"{incomplete_component_names}."
            )
        logger.info(
            "Modal artifact finalization completed after %d remote component(s).",
            len(components),
        )
        return io.NodeOutput()


class ModalParallelLocalPassthrough(io.ComfyNode):
    """Release a local-only branch once continuing remote work is in flight."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Accept one arbitrary value plus remote dispatch metadata."""
        return io.Schema(
            node_id=MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
            display_name="Modal Parallel Local Passthrough",
            category="Modal",
            description=(
                "Internal async gate that releases an independent local tap once "
                "downstream Modal execution has started."
            ),
            inputs=[
                io.AnyType.Input("value"),
                io.AnyType.Input("dispatch_context"),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
            is_dev_only=True,
            is_experimental=True,
        )

    @classmethod
    async def execute(
        cls,
        value: Any,
        dispatch_context: Mapping[str, Any],
    ) -> io.NodeOutput:
        """Return the original value once the remote continuation is in flight."""
        await _wait_for_parallel_local_dispatches(dispatch_context)
        return io.NodeOutput(value)


class ModalLocalBridgeMaterializer(io.ComfyNode):
    """Download a durable remote bridge value for an independent local branch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Accept one durable bridge reference and return its local value."""
        return io.Schema(
            node_id=MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
            display_name="Modal Local Bridge Materializer",
            category="Modal",
            description=(
                "Internal async node that materializes a durable Modal bridge for "
                "local-only work while remote execution continues."
            ),
            inputs=[io.AnyType.Input("bridge_ref")],
            outputs=[io.AnyType.Output(display_name="value")],
            is_dev_only=True,
            is_experimental=True,
        )

    @classmethod
    async def execute(cls, bridge_ref: Mapping[str, Any]) -> io.NodeOutput:
        """Materialize the bridge without blocking ComfyUI's async scheduler."""
        from .remote.modal_app import materialize_remote_session_bridge_ref_locally

        value = await asyncio.to_thread(
            materialize_remote_session_bridge_ref_locally,
            bridge_ref,
        )
        return io.NodeOutput(value)
