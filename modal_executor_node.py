"""Dynamic Modal proxy nodes for ComfyUI execution."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from comfy_api.latest import _io as io

from .serialization import deserialize_node_outputs, serialize_node_inputs

logger = logging.getLogger(__name__)
MODAL_MAP_INPUT_NODE_ID = "ModalMapInput"
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


_REMOTE_EXECUTOR_CLIENT_FACTORY: Callable[[], RemoteExecutorClient] = ModalRemoteExecutorClient
_PROXY_NODE_CACHE: dict[str, type[io.ComfyNode]] = {}
_PROXY_EXECUTION_CONTEXTS_LOCK = threading.Lock()
_PROXY_EXECUTION_CONTEXTS: dict[str, "_ProxyExecutionContext"] = {}


@dataclass(frozen=True)
class _ProxyExecutionContext:
    """Run-scoped execution context used to rehydrate cache-friendly proxy payloads."""

    execution_payload: dict[str, Any]


def set_remote_executor_client_factory(
    factory: Callable[[], RemoteExecutorClient] | None,
) -> None:
    """Install a custom client factory, primarily for tests."""
    global _REMOTE_EXECUTOR_CLIENT_FACTORY
    _REMOTE_EXECUTOR_CLIENT_FACTORY = factory or ModalRemoteExecutorClient


def get_remote_executor_client() -> RemoteExecutorClient:
    """Instantiate the configured execution client."""
    return _REMOTE_EXECUTOR_CLIENT_FACTORY()


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


def _normalize_proxy_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ComfyUI INPUT_IS_LIST proxy kwargs back into ordinary runtime values."""
    normalized_kwargs: dict[str, Any] = {}
    for input_name, input_value in kwargs.items():
        if isinstance(input_value, list) and len(input_value) == 1:
            normalized_kwargs[str(input_name)] = input_value[0]
            continue
        normalized_kwargs[str(input_name)] = input_value
    return normalized_kwargs


def _normalize_proxy_payload(payload: Any) -> Mapping[str, Any]:
    """Convert ComfyUI INPUT_IS_LIST payload wrappers back into one payload mapping."""
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError("original_node_data must be a mapping or JSON object.")
    return payload


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
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        _PROXY_EXECUTION_CONTEXTS[str(node_id)] = _ProxyExecutionContext(
            execution_payload=dict(payload),
        )
    logger.debug(
        "Registered cache-friendly Modal proxy payload for node_id=%s prompt_id=%s session_backed=%s.",
        node_id,
        _normalize_prompt_id(payload.get("prompt_id")),
        payload.get("remote_session") is not None,
    )
    return sanitized_payload


def _rehydrate_proxy_payload(
    payload: Mapping[str, Any],
    *,
    unique_id: str | None,
) -> Mapping[str, Any]:
    """Restore any execution-scoped fields stripped from a cache-friendly proxy payload."""
    context_id = unique_id
    if context_id is None:
        candidate_context_id = payload.get(_PROXY_CACHE_CONTEXT_ID_KEY)
        if candidate_context_id is not None:
            normalized_context_id = str(candidate_context_id).strip()
            context_id = normalized_context_id or None
    if context_id is None:
        return payload

    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        context = _PROXY_EXECUTION_CONTEXTS.get(str(context_id))
    if context is None:
        return payload

    return dict(context.execution_payload)


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
                hidden=[io.Hidden.unique_id],
                is_dev_only=True,
                is_experimental=True,
            )

        @classmethod
        async def execute(cls, **kwargs: Any) -> io.NodeOutput:
            """Forward the execution payload to the configured remote executor."""
            unique_id = _normalize_prompt_id(kwargs.pop(io.Hidden.unique_id.name, None))
            payload = _rehydrate_proxy_payload(
                _normalize_proxy_payload(kwargs.pop(payload_input_name, None)),
                unique_id=unique_id,
            )

            outputs = tuple(
                await _execute_payload_async(
                    get_remote_executor_client(),
                    payload,
                    _normalize_proxy_kwargs(kwargs),
                )
            )
            logger.debug(
                "Remote execution completed for payload kind=%s with %d outputs.",
                payload.get("payload_kind"),
                len(outputs),
            )
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
        nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id] = _PROXY_NODE_CACHE[proxy_node_id]
        nodes_module.NODE_DISPLAY_NAME_MAPPINGS[proxy_node_id] = "Modal Universal Executor"
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
    nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id] = proxy_class
    nodes_module.NODE_DISPLAY_NAME_MAPPINGS[proxy_node_id] = "Modal Universal Executor"
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
) -> str:
    """Register and return a proxy node id for a remote component signature."""
    normalized_output_types = tuple(output_types)
    normalized_output_names = tuple(output_names)
    normalized_output_is_list = tuple(output_is_list)
    proxy_node_id = _proxy_node_id(
        "ModalRemoteComponent",
        normalized_output_types + (str(is_output_node),),
    )

    if proxy_node_id in _PROXY_NODE_CACHE:
        nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id] = _PROXY_NODE_CACHE[proxy_node_id]
        nodes_module.NODE_DISPLAY_NAME_MAPPINGS[proxy_node_id] = "Modal Remote Component"
        return proxy_node_id

    proxy_class = _build_proxy_node_class(
        node_id=proxy_node_id,
        proxy_display_name="Modal Remote Component",
        payload_input_name="original_node_data",
        output_types=normalized_output_types,
        output_names=normalized_output_names,
        output_is_list=normalized_output_is_list,
        is_output_node=is_output_node,
    )
    nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id] = proxy_class
    nodes_module.NODE_DISPLAY_NAME_MAPPINGS[proxy_node_id] = "Modal Remote Component"
    _PROXY_NODE_CACHE[proxy_node_id] = proxy_class
    logger.info("Registered Modal component proxy node %s", proxy_node_id)
    return proxy_node_id


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
            is_experimental=True,
        )

    @classmethod
    def execute(cls, value: Any) -> io.NodeOutput:
        """Pass the input value through unchanged at runtime."""
        return io.NodeOutput(value)
