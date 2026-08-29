"""Dynamic Modal proxy nodes for ComfyUI execution."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from comfy_api.latest import _io as io

if __package__:
    from .serialization import (
        deserialize_node_outputs,
        serialize_node_inputs,
        split_mapped_value,
    )
    from .proxy_node_factory import (
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ProxyNodeFactoryHooks,
        _build_proxy_node_class,
        _normalized_output_metadata,
        _output_spec,
        _proxy_node_id,
        _register_modal_node,
        configure_proxy_node_factory_hooks,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        ensure_modal_parallel_local_passthrough_registered,
        ensure_modal_proxy_node_registered,
    )
    from .proxy_payloads import (
        MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY,
        _boost_modal_map_input_warmup,
        _normalize_prompt_id,
        _normalize_proxy_kwargs,
        _normalize_proxy_payload,
        _normalize_scheduler_list_outputs,
        _payload_is_local_cache_safe,
        _pop_proxy_hidden_value,
        _prompt_id_from_extra_pnginfo,
        _rehydrate_proxy_payload,
        _sanitize_cache_surface_payload,
        _unwrap_proxy_singleton,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from .remote_executor_router import (
        ModalRemoteExecutorClient,
        RemoteExecutorClient,
        RemoteExecutorRouterClient,
        get_remote_executor_client,
        set_remote_executor_client_factory,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from serialization import (
        deserialize_node_outputs,
        serialize_node_inputs,
        split_mapped_value,
    )
    from proxy_node_factory import (
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ProxyNodeFactoryHooks,
        _build_proxy_node_class,
        _normalized_output_metadata,
        _output_spec,
        _proxy_node_id,
        _register_modal_node,
        configure_proxy_node_factory_hooks,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        ensure_modal_parallel_local_passthrough_registered,
        ensure_modal_proxy_node_registered,
    )
    from proxy_payloads import (
        MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY,
        _boost_modal_map_input_warmup,
        _normalize_prompt_id,
        _normalize_proxy_kwargs,
        _normalize_proxy_payload,
        _normalize_scheduler_list_outputs,
        _payload_is_local_cache_safe,
        _pop_proxy_hidden_value,
        _prompt_id_from_extra_pnginfo,
        _rehydrate_proxy_payload,
        _sanitize_cache_surface_payload,
        _unwrap_proxy_singleton,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from remote_executor_router import (
        ModalRemoteExecutorClient,
        RemoteExecutorClient,
        RemoteExecutorRouterClient,
        get_remote_executor_client,
        set_remote_executor_client_factory,
    )

logger = logging.getLogger(__name__)
MODAL_MAP_INPUT_NODE_ID = "ModalMapInput"
MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS = 100








_MODAL_WORKFLOW_EXECUTION_GATE = threading.Condition()
_MODAL_WORKFLOW_ACTIVE_PROMPT_ID: str | None = None
_MODAL_WORKFLOW_ACTIVE_REMOTE_CALLS = 0
_MODAL_WORKFLOW_ABANDONED_RELEASES_BY_PROMPT_ID: dict[str, int] = {}
_MODAL_PARALLEL_DISPATCH_EVENTS_LOCK = threading.Lock()
_MODAL_PARALLEL_DISPATCH_EVENTS: OrderedDict[
    tuple[str, str], asyncio.Event
] = OrderedDict()
_MODAL_PARALLEL_DISPATCH_EVENT_LIMIT = 512










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


configure_proxy_node_factory_hooks(
    ProxyNodeFactoryHooks(
        modal_workflow_execution_slot=_modal_workflow_execution_slot,
        signal_parallel_local_dispatch=_signal_parallel_local_dispatch,
        execute_payload_async=_execute_payload_async,
        modal_universal_executor=ModalUniversalExecutor,
        modal_artifact_finalizer=ModalArtifactFinalizer,
        modal_parallel_local_passthrough=ModalParallelLocalPassthrough,
        modal_local_bridge_materializer=ModalLocalBridgeMaterializer,
    )
)
