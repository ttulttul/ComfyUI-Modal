"""Dynamic ComfyUI proxy-node construction and mapping registration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

from comfy_api.latest import _io as io

if __package__:
    from .proxy_payloads import (
        _normalize_prompt_id,
        _normalize_proxy_kwargs,
        _normalize_proxy_payload,
        _normalize_scheduler_list_outputs,
        _pop_proxy_hidden_value,
        _prompt_id_from_extra_pnginfo,
        _rehydrate_proxy_payload,
        _unwrap_proxy_singleton,
    )
    from .remote_executor_router import get_remote_executor_client
else:  # pragma: no cover - flat import inside the Modal container.
    from proxy_payloads import (
        _normalize_prompt_id,
        _normalize_proxy_kwargs,
        _normalize_proxy_payload,
        _normalize_scheduler_list_outputs,
        _pop_proxy_hidden_value,
        _prompt_id_from_extra_pnginfo,
        _rehydrate_proxy_payload,
        _unwrap_proxy_singleton,
    )
    from remote_executor_router import get_remote_executor_client

logger = logging.getLogger(__name__)

MODAL_ARTIFACT_FINALIZER_NODE_ID = "ModalArtifactFinalizer"
MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID = "ModalParallelLocalPassthrough"
MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID = "ModalLocalBridgeMaterializer"
MODAL_COMPONENT_COMPLETION_OUTPUT_NAME = "modal_component_complete"
_PROXY_NODE_CACHE: dict[str, type[io.ComfyNode]] = {}


@dataclass(frozen=True)
class ProxyNodeFactoryHooks:
    """Entrypoint-owned execution and static-node dependencies."""

    modal_workflow_execution_slot: Callable[..., Any]
    signal_parallel_local_dispatch: Callable[[Mapping[str, Any]], bool]
    execute_payload_async: Callable[..., Any]
    modal_universal_executor: type[io.ComfyNode]
    modal_artifact_finalizer: type[io.ComfyNode]
    modal_parallel_local_passthrough: type[io.ComfyNode]
    modal_local_bridge_materializer: type[io.ComfyNode]


_PROXY_NODE_FACTORY_HOOKS: ProxyNodeFactoryHooks | None = None


def configure_proxy_node_factory_hooks(hooks: ProxyNodeFactoryHooks) -> None:
    """Install residual executor and static-node dependencies."""
    global _PROXY_NODE_FACTORY_HOOKS
    _PROXY_NODE_FACTORY_HOOKS = hooks


def _factory_hooks() -> ProxyNodeFactoryHooks:
    """Return configured factory hooks or fail on invalid import order."""
    if _PROXY_NODE_FACTORY_HOOKS is None:
        raise RuntimeError("Proxy node factory hooks have not been configured.")
    return _PROXY_NODE_FACTORY_HOOKS

def _output_spec(io_type: str, name: str, is_list: bool) -> io.Output:
    """Create a v3 output specification from a legacy ComfyUI return type."""
    comfy_type = io.AnyType if io_type == "*" else io.Custom(io_type)
    return comfy_type.Output(display_name=name, is_output_list=is_list)



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

            async with _factory_hooks().modal_workflow_execution_slot(payload):
                _factory_hooks().signal_parallel_local_dispatch(payload)
                outputs = _normalize_scheduler_list_outputs(
                    payload,
                    await _factory_hooks().execute_payload_async(
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
    node_class.RELATIVE_PYTHON_MODULE = _factory_hooks().modal_universal_executor.RELATIVE_PYTHON_MODULE
    nodes_module.NODE_CLASS_MAPPINGS[node_id] = node_class
    nodes_module.NODE_DISPLAY_NAME_MAPPINGS[node_id] = display_name


def ensure_modal_artifact_finalizer_registered(nodes_module: Any) -> None:
    """Register the internal artifact-finalization sink in a ComfyUI node mapping."""
    _register_modal_node(
        nodes_module,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        _factory_hooks().modal_artifact_finalizer,
        "Modal Artifact Finalizer",
    )


def ensure_modal_parallel_local_passthrough_registered(nodes_module: Any) -> None:
    """Register the internal parallel local-branch dispatch gate."""
    _register_modal_node(
        nodes_module,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        _factory_hooks().modal_parallel_local_passthrough,
        "Modal Parallel Local Passthrough",
    )


def ensure_modal_local_bridge_materializer_registered(nodes_module: Any) -> None:
    """Register the internal durable-bridge local materializer."""
    _register_modal_node(
        nodes_module,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        _factory_hooks().modal_local_bridge_materializer,
        "Modal Local Bridge Materializer",
    )



