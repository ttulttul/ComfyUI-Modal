"""ComfyUI prompt validation, executor management, and subgraph execution."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
import copy
from dataclasses import dataclass
import importlib
import inspect
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Mapping

try:
    from .cloud_comfy_bootstrap import (
        _ensure_comfy_runtime_initialized,
        _ensure_prompt_node_classes_registered,
        _extract_custom_nodes_bundle,
        _load_execution_module,
        _load_nodes_module,
        _loader_cache_metric_snapshot,
        _patched_folder_paths_absolute_lookup,
        _rewrite_modal_asset_references,
    )
    from .cloud_node_output_cache import (
        _PersistedNodeCacheRestoreState,
        _boundary_output_node_ids,
        _emit_restored_node_cache_events,
        _install_prompt_executor_persisted_cache_restore,
        _node_output_cache_store,
        _persist_node_output_cache_entries,
        _prompt_executor_cache_get_sync,
    )
    from .cloud_prompt_server_shims import _NullPromptServer, _TracingPromptServer
    from .cloud_session_bridge import (
        _RemoteSessionBridgeResolutionStats,
        _build_remote_session_bridge_record,
        _log_remote_session_resolution_summary,
        _payload_remote_session_handle,
        _resolve_remote_session_inputs,
        _store_remote_session_bridge_record,
        _store_remote_session_bridge_value,
        remote_session_store,
    )
    from .remote_protocol import (
        BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
        PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
    )
    from .serialization import (
        coerce_serialized_node_outputs,
        deserialize_node_inputs,
        serialize_mapping,
        serialize_node_inputs,
        serialize_node_outputs,
    )
    from .session_state import (
        RemoteSessionBridgeRef,
        RemoteSessionHandle,
        RemoteSessionStateError,
        RemoteSessionValueRef,
        is_remote_session_bridge_ref_payload,
        is_remote_session_value_ref_payload,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_comfy_bootstrap import (
        _ensure_comfy_runtime_initialized,
        _ensure_prompt_node_classes_registered,
        _extract_custom_nodes_bundle,
        _load_execution_module,
        _load_nodes_module,
        _loader_cache_metric_snapshot,
        _patched_folder_paths_absolute_lookup,
        _rewrite_modal_asset_references,
    )
    from cloud_node_output_cache import (
        _PersistedNodeCacheRestoreState,
        _boundary_output_node_ids,
        _emit_restored_node_cache_events,
        _install_prompt_executor_persisted_cache_restore,
        _node_output_cache_store,
        _persist_node_output_cache_entries,
        _prompt_executor_cache_get_sync,
    )
    from cloud_prompt_server_shims import _NullPromptServer, _TracingPromptServer
    from cloud_session_bridge import (
        _RemoteSessionBridgeResolutionStats,
        _build_remote_session_bridge_record,
        _log_remote_session_resolution_summary,
        _payload_remote_session_handle,
        _resolve_remote_session_inputs,
        _store_remote_session_bridge_record,
        _store_remote_session_bridge_value,
        remote_session_store,
    )
    from remote_protocol import (
        BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
        PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
    )
    from serialization import (
        coerce_serialized_node_outputs,
        deserialize_node_inputs,
        serialize_mapping,
        serialize_node_inputs,
        serialize_node_outputs,
    )
    from session_state import (
        RemoteSessionBridgeRef,
        RemoteSessionHandle,
        RemoteSessionStateError,
        RemoteSessionValueRef,
        is_remote_session_bridge_ref_payload,
        is_remote_session_value_ref_payload,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudPromptExecutionHooks:
    """Callbacks and stable errors supplied by the cloud entrypoint."""

    emit_cloud_info: Callable[..., None]
    timed_phase: Callable[..., AbstractContextManager[None]]
    schedule_remote_cancel_restart: Callable[..., bool]
    remote_subgraph_error: type[RuntimeError]


_PROMPT_EXECUTION_HOOKS: CloudPromptExecutionHooks | None = None
RemoteSubgraphExecutionError: type[RuntimeError] = RuntimeError


def configure_cloud_prompt_execution_hooks(
    hooks: CloudPromptExecutionHooks,
) -> None:
    """Install prompt-execution callbacks without importing upward."""
    global _PROMPT_EXECUTION_HOOKS
    global RemoteSubgraphExecutionError
    _PROMPT_EXECUTION_HOOKS = hooks
    RemoteSubgraphExecutionError = hooks.remote_subgraph_error


def _prompt_execution_hooks() -> CloudPromptExecutionHooks:
    """Return configured callbacks or fail with a clear import-order error."""
    if _PROMPT_EXECUTION_HOOKS is None:
        raise RuntimeError("Cloud prompt-execution hooks have not been configured.")
    return _PROMPT_EXECUTION_HOOKS


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _prompt_execution_hooks().emit_cloud_info(message, *args)


def _timed_phase(phase: str, **fields: Any) -> AbstractContextManager[None]:
    """Delegate phase timing to the stable entrypoint."""
    return _prompt_execution_hooks().timed_phase(phase, **fields)


def _schedule_remote_cancel_restart(**kwargs: Any) -> bool:
    """Delegate poisoned-container retirement scheduling to the entrypoint."""
    return _prompt_execution_hooks().schedule_remote_cancel_restart(**kwargs)


@dataclass
class _ReusablePromptExecutorState:
    """Hold one warm PromptExecutor and serialize access to its mutable caches."""

    executor: Any
    lock: threading.Lock


_PROMPT_EXECUTOR_STATES_LOCK = threading.Lock()
_PROMPT_EXECUTOR_STATES: dict[str, _ReusablePromptExecutorState] = {}
_DYNAMIC_PROMPT_WRAPPER_LOCK = threading.Lock()
_PROMPT_METADATA_STATE = threading.local()


def clear_warm_caches() -> None:
    """Clear reusable PromptExecutor state owned by this module."""
    with _PROMPT_EXECUTOR_STATES_LOCK:
        _PROMPT_EXECUTOR_STATES.clear()


@contextmanager
def _temporary_node_mapping(
    node_mapping: dict[str, type[Any]] | None
) -> Iterator[None]:
    """Temporarily overlay node mappings for tests or custom runtimes."""
    if node_mapping is None:
        yield
        return

    import nodes

    original_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    original_display_mappings = dict(getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    try:
        nodes.NODE_CLASS_MAPPINGS.update(node_mapping)
        for class_type in node_mapping:
            nodes.NODE_DISPLAY_NAME_MAPPINGS.setdefault(class_type, class_type)
        yield
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(original_mappings)
        if hasattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS"):
            nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
            nodes.NODE_DISPLAY_NAME_MAPPINGS.update(original_display_mappings)




@contextmanager
def _temporary_progress_hook(prompt_server: _NullPromptServer) -> Iterator[None]:
    """Install a ComfyUI progress hook so remote samplers emit numeric progress updates."""
    import comfy.utils
    import comfy.model_management
    from comfy_execution.progress import get_progress_state
    from comfy_execution.utils import get_executing_context

    previous_hook = comfy.utils.PROGRESS_BAR_HOOK

    def hook(
        value: float,
        total: float,
        preview_image: Any,
        prompt_id: str | None = None,
        node_id: str | None = None,
    ) -> None:
        """Mirror ComfyUI progress-bar updates into the headless progress registry."""
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = prompt_server.last_prompt_id
        if node_id is None:
            node_id = prompt_server.last_node_id
        if node_id is None:
            return

        resolved_node_id = str(node_id)
        get_progress_state().update_progress(
            resolved_node_id, value, total, preview_image
        )
        preview_emitter = getattr(prompt_server, "emit_preview_update", None)
        if preview_image is not None and callable(preview_emitter):
            preview_emitter(node_id=resolved_node_id, preview_image=preview_image)

    comfy.utils.set_progress_bar_global_hook(hook)
    try:
        yield
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_hook)


@contextmanager
def _temporary_remote_interrupt_monitor(
    component_id: str,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[None]:
    """Mirror shared cancellation requests into ComfyUI's interrupt flag inside Modal."""
    if cancellation_event is None and (
        interrupt_store is None or interrupt_flag_key is None
    ):
        yield
        return

    import nodes

    stop_event = threading.Event()
    execution_completed_event = threading.Event()
    restart_scheduled = False
    try:
        modal_exception_module = importlib.import_module("modal.exception")
    except ModuleNotFoundError:
        modal_exception_module = None
    modal_client_closed_error = getattr(modal_exception_module, "ClientClosed", None)

    def shared_cancel_flag_exists() -> bool:
        """Return whether the shared interrupt flag is present, tolerating Modal shutdown races."""
        try:
            return bool(interrupt_store.contains(interrupt_flag_key))
        except Exception as exc:
            if modal_client_closed_error is not None and isinstance(
                exc, modal_client_closed_error
            ):
                logger.info(
                    "Remote interrupt monitor stopped after Modal client shutdown for component=%s.",
                    component_id,
                )
                stop_event.set()
                return False
            raise

    def consume_shared_cancel_flag() -> None:
        """Remove the shared interrupt flag if the Modal client is still alive."""
        try:
            interrupt_store.pop(interrupt_flag_key, None)
        except Exception as exc:
            if modal_client_closed_error is not None and isinstance(
                exc, modal_client_closed_error
            ):
                logger.info(
                    "Remote interrupt monitor skipped flag cleanup after Modal client shutdown for component=%s.",
                    component_id,
                )
                stop_event.set()
                return
            raise

    def monitor_interrupts() -> None:
        """Set ComfyUI's interrupt flag once the caller requests cancellation."""
        nonlocal restart_scheduled
        while not stop_event.is_set():
            if cancellation_event is not None and cancellation_event.wait(timeout=0.1):
                logger.info(
                    "Remote interrupt monitor tripped local event for component=%s.",
                    component_id,
                )
                if not restart_scheduled:
                    restart_scheduled = _schedule_remote_cancel_restart(
                        component_id=component_id,
                        completion_event=execution_completed_event,
                    )
                nodes.interrupt_processing()
                return
            if interrupt_store is None or interrupt_flag_key is None:
                continue
            if not shared_cancel_flag_exists():
                continue
            logger.info(
                "Remote interrupt monitor observed shared cancel flag for component=%s.",
                component_id,
            )
            consume_shared_cancel_flag()
            if stop_event.is_set():
                return
            if cancellation_event is not None:
                cancellation_event.set()
            if not restart_scheduled:
                restart_scheduled = _schedule_remote_cancel_restart(
                    component_id=component_id,
                    completion_event=execution_completed_event,
                )
            nodes.interrupt_processing()
            return

    interrupt_thread = threading.Thread(
        target=monitor_interrupts,
        name=f"modal-interrupt-{component_id}",
        daemon=True,
    )
    interrupt_thread.start()
    try:
        yield
    finally:
        execution_completed_event.set()
        stop_event.set()
        interrupt_thread.join(timeout=1.0)




def _prompt_executor_ram_thresholds(
    cache_ram_values: list[float],
) -> tuple[float, float]:
    """Return current ComfyUI active and inactive RAM-cache thresholds in GiB."""
    model_management = importlib.import_module("comfy.model_management")
    active_threshold = min(10.0, max(2.0, model_management.total_ram * 0.10 / 1024.0))
    inactive_threshold = min(128.0, model_management.total_ram / 1024.0)
    if cache_ram_values:
        active_threshold = cache_ram_values[0]
    if len(cache_ram_values) > 1:
        inactive_threshold = cache_ram_values[1]
    return active_threshold, inactive_threshold


def _prompt_executor_cache_config(execution: Any) -> tuple[Any, dict[str, float]]:
    """Return cache settings compatible with legacy and current ComfyUI CLI shapes."""
    from comfy.cli_args import args

    cache_ram = args.cache_ram
    if isinstance(cache_ram, list):
        active_threshold = 0.0
        inactive_threshold = 0.0
        if not args.cache_classic and not args.cache_none and args.cache_lru <= 0:
            active_threshold, inactive_threshold = _prompt_executor_ram_thresholds(
                cache_ram
            )

        cache_type = execution.CacheType.RAM_PRESSURE
        if args.cache_classic:
            cache_type = execution.CacheType.CLASSIC
        elif args.cache_lru > 0:
            cache_type = execution.CacheType.LRU
        elif args.cache_none:
            cache_type = execution.CacheType.NONE
        return cache_type, {
            "lru": args.cache_lru,
            "ram": active_threshold,
            "ram_inactive": inactive_threshold,
        }

    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE
    return cache_type, {"lru": args.cache_lru, "ram": cache_ram}


def _serialize_prompt_executor_cache_scope(
    cache_type: Any,
    cache_args: dict[str, Any],
    custom_nodes_root: Path | None,
) -> str:
    """Return a stable cache scope key for reusable PromptExecutor instances."""
    return json.dumps(
        {
            "cache_type": str(cache_type),
            "cache_args": cache_args,
            "custom_nodes_root": str(custom_nodes_root.resolve())
            if custom_nodes_root is not None
            else None,
        },
        sort_keys=True,
        default=str,
    )


def _reset_prompt_executor_request_state(executor: Any, prompt_server: Any) -> None:
    """Prepare a reusable PromptExecutor for a fresh request without discarding its caches."""
    executor.server = prompt_server
    executor.status_messages = []
    executor.success = True
    executor.history_result = {}
    prompt_server.client_id = None
    prompt_server.last_node_id = None


def _get_or_create_prompt_executor_state(
    execution: Any,
    prompt_server: Any,
    cache_type: Any,
    cache_args: dict[str, Any],
    custom_nodes_root: Path | None,
) -> _ReusablePromptExecutorState:
    """Return the warm-container PromptExecutor state for a cache scope, creating it once."""
    state_key = _serialize_prompt_executor_cache_scope(
        cache_type, cache_args, custom_nodes_root
    )
    with _PROMPT_EXECUTOR_STATES_LOCK:
        existing_state = _PROMPT_EXECUTOR_STATES.get(state_key)
        if existing_state is not None:
            _emit_cloud_info("Prompt executor cache hit scope=%s", state_key)
            return existing_state

        _emit_cloud_info("Prompt executor cache miss scope=%s", state_key)
        executor = execution.PromptExecutor(
            prompt_server,
            cache_type=cache_type,
            cache_args=cache_args,
        )
        state = _ReusablePromptExecutorState(executor=executor, lock=threading.Lock())
        _PROMPT_EXECUTOR_STATES[state_key] = state
        return state


def _execute_prompt_executor_compat(
    executor: Any,
    *,
    prompt: dict[str, Any],
    prompt_id: str,
    extra_data: dict[str, Any],
    execute_outputs: list[str],
) -> None:
    """Execute a ComfyUI prompt across synchronous and asynchronous executor APIs."""
    execution_result = executor.execute(
        prompt=prompt,
        prompt_id=prompt_id,
        extra_data=extra_data,
        execute_outputs=execute_outputs,
    )
    if inspect.isawaitable(execution_result):
        asyncio.run(execution_result)


def _install_metadata_safe_dynamic_prompt_wrapper(execution: Any) -> None:
    """Let hidden PROMPT inputs see metadata-safe graphs instead of hydrated tensors."""
    with _DYNAMIC_PROMPT_WRAPPER_LOCK:
        dynamic_prompt_class = getattr(execution, "DynamicPrompt", None)
        if dynamic_prompt_class is None:
            raise RemoteSubgraphExecutionError(
                "The ComfyUI execution module does not expose DynamicPrompt."
            )
        if bool(getattr(dynamic_prompt_class, "_comfy_modal_metadata_safe", False)):
            return

        class MetadataSafeDynamicPrompt(dynamic_prompt_class):
            """DynamicPrompt variant with a separate hidden-input metadata graph."""

            _comfy_modal_metadata_safe = True

            def __init__(self, original_prompt: dict[str, Any]) -> None:
                """Capture the calling thread's metadata prompt before execution starts."""
                super().__init__(original_prompt)
                self._comfy_modal_metadata_prompt = getattr(
                    _PROMPT_METADATA_STATE,
                    "prompt",
                    None,
                )

            def get_original_prompt(self) -> dict[str, Any]:
                """Return the JSON-safe graph exposed through ComfyUI's hidden PROMPT input."""
                if self._comfy_modal_metadata_prompt is not None:
                    return self._comfy_modal_metadata_prompt
                return super().get_original_prompt()

        execution.DynamicPrompt = MetadataSafeDynamicPrompt
        logger.info(
            "Installed metadata-safe DynamicPrompt wrapper for hydrated remote boundary inputs."
        )


@contextmanager
def _temporary_prompt_metadata(prompt: dict[str, Any]) -> Iterator[None]:
    """Expose one JSON-safe hidden PROMPT graph to this execution thread."""
    had_previous_prompt = hasattr(_PROMPT_METADATA_STATE, "prompt")
    previous_prompt = getattr(_PROMPT_METADATA_STATE, "prompt", None)
    _PROMPT_METADATA_STATE.prompt = prompt
    try:
        yield
    finally:
        if had_previous_prompt:
            _PROMPT_METADATA_STATE.prompt = previous_prompt
        else:
            delattr(_PROMPT_METADATA_STATE, "prompt")


def _copy_json_safe_prompt_metadata(prompt: dict[str, Any]) -> dict[str, Any]:
    """Copy and validate the graph exposed to metadata-writing custom nodes."""
    metadata_prompt = copy.deepcopy(prompt)
    try:
        json.dumps(metadata_prompt)
    except (TypeError, ValueError) as exc:
        raise RemoteSubgraphExecutionError(
            "Remote subgraph prompt metadata was not JSON-compatible before boundary hydration."
        ) from exc
    return metadata_prompt


def _invoke_original_node(
    node_class: type[Any],
    node_data: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Execute an original V1 or V3 node class and normalize its outputs."""
    class_type = node_data["class_type"]
    logger.info("Executing remote node %s", class_type)

    if hasattr(node_class, "GET_SCHEMA"):
        node_output = node_class.execute(**kwargs)
        if hasattr(node_output, "result"):
            result = node_output.result
            return tuple(result) if result is not None else tuple()
        return tuple(node_output)

    instance = node_class()
    function_name = getattr(node_class, "FUNCTION", "execute")
    function = getattr(instance, function_name)
    result = function(**kwargs)
    if result is None:
        return tuple()
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def execute_node_locally(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> bytes:
    """Execute a single target node in-process and return serialized outputs."""
    outputs = _execute_node_locally_raw(
        node_data,
        kwargs_payload,
        node_mapping=node_mapping,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )
    return serialize_node_outputs(outputs)


def _execute_node_locally_raw(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute a single target node in-process and return raw node outputs."""
    custom_nodes_root = _extract_custom_nodes_bundle(
        node_data.get("custom_nodes_bundle")
    )
    _ensure_comfy_runtime_initialized(custom_nodes_root)
    kwargs = _rewrite_modal_asset_references(deserialize_node_inputs(kwargs_payload))
    component_id = str(
        node_data.get("component_id") or node_data.get("class_type") or "single-node"
    )
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        with (
            _patched_folder_paths_absolute_lookup(),
            _temporary_remote_interrupt_monitor(
                component_id,
                cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            ),
        ):
            return _invoke_original_node(node_mapping[class_type], node_data, kwargs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        with (
            _patched_folder_paths_absolute_lookup(),
            _temporary_remote_interrupt_monitor(
                component_id,
                cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            ),
        ):
            return _invoke_original_node(
                resolved_node_mapping[class_type], node_data, kwargs
            )


def _remote_session_ref_cache_signature(value: Any) -> Any | None:
    """Return cache-key metadata for any remote-session refs nested in `value`."""
    if is_remote_session_bridge_ref_payload(value):
        return {
            "kind": "remote_session_bridge_ref",
            "bridge_key": value.get("bridge_key"),
            "node_id": value.get("node_id"),
            "output_index": value.get("output_index"),
        }
    if is_remote_session_value_ref_payload(value):
        return {
            "kind": "remote_session_value_ref",
            "session_id": value.get("session_id"),
            "node_id": value.get("node_id"),
            "output_index": value.get("output_index"),
        }
    if isinstance(value, list):
        items = [_remote_session_ref_cache_signature(item) for item in value]
        if any(item is not None for item in items):
            return {"kind": "list", "items": items}
        return None
    if isinstance(value, tuple):
        items = [_remote_session_ref_cache_signature(item) for item in value]
        if any(item is not None for item in items):
            return {"kind": "tuple", "items": items}
        return None
    if isinstance(value, Mapping):
        items = {
            str(key): _remote_session_ref_cache_signature(item)
            for key, item in value.items()
        }
        filtered_items = {key: item for key, item in items.items() if item is not None}
        if filtered_items:
            return {"kind": "mapping", "items": filtered_items}
    return None


def _boundary_input_cache_signature(
    *,
    source_signature: Any,
    cache_value: Any,
) -> Any | None:
    """Return a cache signature for one boundary input and its unresolved source value."""
    ref_signature = _remote_session_ref_cache_signature(cache_value)
    if ref_signature is None:
        if isinstance(source_signature, str) and source_signature:
            return source_signature
        return None
    return {
        "source_signature": source_signature
        if isinstance(source_signature, str)
        else None,
        "remote_session_refs": ref_signature,
    }


def _apply_boundary_inputs(
    prompt: dict[str, Any],
    boundary_input_specs: list[dict[str, Any]],
    hydrated_inputs: dict[str, Any],
    cache_signature_inputs: dict[str, Any] | None = None,
) -> None:
    """Inject hydrated local boundary inputs into a remote subgraph prompt."""
    for boundary_input in boundary_input_specs:
        proxy_input_name = str(boundary_input["proxy_input_name"])
        if proxy_input_name not in hydrated_inputs:
            raise KeyError(f"Missing hydrated boundary input {proxy_input_name!r}.")
        value = hydrated_inputs[proxy_input_name]
        io_type = (
            str(boundary_input["io_type"])
            if boundary_input.get("io_type") is not None
            else None
        )
        for target in boundary_input.get("targets", []):
            node_id = str(target["node_id"])
            input_name = str(target["input_name"])
            prompt_node = prompt[node_id]
            prompt_node["inputs"][input_name] = _normalize_prompt_input_value(
                value,
                io_type=io_type,
            )
            cache_signature = _boundary_input_cache_signature(
                source_signature=boundary_input.get("source_signature"),
                cache_value=(
                    cache_signature_inputs.get(proxy_input_name)
                    if cache_signature_inputs is not None
                    and proxy_input_name in cache_signature_inputs
                    else value
                ),
            )
            if cache_signature is not None:
                boundary_signatures = prompt_node.setdefault(
                    _BOUNDARY_INPUT_SIGNATURES_KEY, {}
                )
                if isinstance(boundary_signatures, dict):
                    boundary_signatures[input_name] = cache_signature


def _collapse_cache_slot(slot_values: Any, is_list: bool) -> Any:
    """Convert a PromptExecutor cache slot back into a node-style output value."""
    if is_list:
        return slot_values
    if not isinstance(slot_values, list):
        return slot_values
    if len(slot_values) == 1:
        return slot_values[0]
    return slot_values


def _extract_prompt_executor_error(executor: Any) -> str:
    """Extract a useful failure message from a PromptExecutor run."""
    for event, data in reversed(executor.status_messages):
        if event == "execution_error":
            return _format_prompt_executor_error_payload(data)
        if event == "execution_interrupted":
            return "Remote subgraph execution was interrupted."
    return "Remote subgraph execution failed."


def _format_prompt_executor_error_payload(data: Any) -> str:
    """Return a richer human-readable PromptExecutor failure message when available."""
    if not isinstance(data, dict):
        return "Remote subgraph execution failed."

    message = str(data.get("exception_message") or "Remote subgraph execution failed.")
    node_id = data.get("node_id")
    node_type = data.get("node_type")
    current_inputs = data.get("current_inputs")
    if node_id is None and node_type is None and not current_inputs:
        return message

    details: list[str] = [message]
    if node_id is not None or node_type is not None:
        details.append(f"node_id={node_id!r} node_type={node_type!r}")
    if current_inputs:
        details.append(f"current_inputs={current_inputs!r}")
    return " | ".join(details)


def _extract_prompt_executor_error_payload(executor: Any) -> dict[str, Any] | None:
    """Return the most recent PromptExecutor execution_error payload when present."""
    for event, data in reversed(executor.status_messages):
        if event == "execution_error" and isinstance(data, dict):
            return data
    return None


def _summarize_suspicious_prompt_inputs(prompt: dict[str, Any]) -> list[str]:
    """Return compact descriptions of prompt inputs that still look list-wrapped."""
    findings: list[str] = []
    for node_id, node_info in sorted(prompt.items()):
        inputs = node_info.get("inputs") or {}
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 1:
                findings.append(f"{node_id}.{input_name}={input_value!r}")
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
                and isinstance(input_value[1], list)
            ):
                findings.append(f"{node_id}.{input_name}={input_value!r}")
    return findings


def _node_input_types(
    node_class: type[Any],
    live_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one node class's input schema finalized for the live prompt inputs."""
    input_types_callable = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types_callable):
        return {}

    raw_input_types = input_types_callable()
    if not isinstance(raw_input_types, dict):
        return {}

    has_v3_dynamic_inputs = any(
        isinstance(input_config, tuple)
        and bool(input_config)
        and isinstance(input_config[0], str)
        and input_config[0].startswith("COMFY_")
        and input_config[0].endswith("_V3")
        for section_name in ("required", "optional")
        for input_config in (
            raw_input_types.get(section_name, {}).values()
            if isinstance(raw_input_types.get(section_name), dict)
            else ()
        )
    )
    if not has_v3_dynamic_inputs:
        return raw_input_types

    from comfy_api.latest import _io

    finalized_input_types, _, _ = _io.get_finalized_class_inputs(
        raw_input_types,
        live_inputs or {},
    )
    logger.debug(
        "Finalized V3 input schema for %s against prompt inputs %s.",
        node_class.__name__,
        sorted(str(input_name) for input_name in (live_inputs or {})),
    )
    return finalized_input_types


def _node_input_type_map(
    node_class: type[Any],
    live_inputs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return one node class's finalized input types keyed by prompt input name."""
    input_types = _node_input_types(node_class, live_inputs)

    input_type_map: dict[str, str] = {}
    for section_name in ("required", "optional", "hidden"):
        section = input_types.get(section_name)
        if not isinstance(section, dict):
            continue
        for input_name, input_config in section.items():
            if not isinstance(input_config, tuple) or not input_config:
                continue
            declared_type = input_config[0]
            if isinstance(declared_type, str):
                input_type_map[str(input_name)] = declared_type
    return input_type_map


def _node_required_input_names(
    node_class: type[Any],
    live_inputs: dict[str, Any] | None = None,
) -> set[str]:
    """Return required input names after expanding V3 dynamic prompt inputs."""
    required_inputs = _node_input_types(node_class, live_inputs).get("required")
    if not isinstance(required_inputs, dict):
        return set()
    return {str(input_name) for input_name in required_inputs}


def _coerce_primitive_prompt_input_value(
    *,
    node_id: str,
    class_type: str,
    input_name: str,
    declared_type: str,
    input_value: Any,
) -> Any:
    """Coerce one primitive prompt literal using ComfyUI's `validate_inputs` semantics."""
    literal_value = (
        input_value.get("__value__")
        if isinstance(input_value, dict) and "__value__" in input_value
        else input_value
    )
    if isinstance(literal_value, list):
        return input_value

    try:
        if declared_type == "INT":
            return int(literal_value)
        if declared_type == "FLOAT":
            return float(literal_value)
        if declared_type == "STRING":
            return str(literal_value)
        if declared_type == "BOOLEAN":
            return bool(literal_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RemoteSubgraphExecutionError(
            "Remote subgraph input could not be coerced to the declared primitive socket type."
            f" node_id={node_id!r} node_type={class_type!r}"
            f" input_name={input_name!r} declared_type={declared_type!r}"
            f" received_value={literal_value!r}"
        ) from exc

    return input_value


def _coerce_prompt_primitive_input_values(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> None:
    """Mutate prompt literals in-place to match ComfyUI's primitive widget coercion."""
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        inputs = prompt_node.get("inputs") or {}
        input_type_map = _node_input_type_map(node_class, inputs)
        if not input_type_map:
            continue
        for input_name, input_value in list(inputs.items()):
            declared_type = input_type_map.get(str(input_name))
            if declared_type not in _PRIMITIVE_WIDGET_INPUT_TYPES:
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
            ):
                continue
            coerced_value = _coerce_primitive_prompt_input_value(
                node_id=str(node_id),
                class_type=class_type,
                input_name=str(input_name),
                declared_type=declared_type,
                input_value=input_value,
            )
            if coerced_value is not input_value:
                logger.debug(
                    "Coerced remote primitive input %s.%s from %r to %r for type %s.",
                    node_id,
                    input_name,
                    input_value,
                    coerced_value,
                    declared_type,
                )
                inputs[input_name] = coerced_value


def _validate_prompt_input_shapes(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
    boundary_input_specs: list[dict[str, Any]] | None = None,
) -> None:
    """Reject prompt inputs that still look invalid for primitive widget sockets."""
    boundary_targets = {
        (str(target.get("node_id")), str(target.get("input_name")))
        for boundary_input in (boundary_input_specs or [])
        for target in boundary_input.get("targets", [])
        if target.get("node_id") is not None and target.get("input_name") is not None
    }
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        inputs = prompt_node.get("inputs") or {}
        input_type_map = _node_input_type_map(node_class, inputs)
        if not input_type_map:
            continue
        for input_name, input_value in inputs.items():
            declared_type = input_type_map.get(str(input_name))
            if declared_type not in _PRIMITIVE_WIDGET_INPUT_TYPES:
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
            ):
                continue
            if (str(node_id), str(input_name)) in boundary_targets:
                continue
            literal_value = (
                input_value.get("__value__")
                if isinstance(input_value, dict) and "__value__" in input_value
                else input_value
            )
            if isinstance(literal_value, list):
                raise RemoteSubgraphExecutionError(
                    "Remote subgraph input has an invalid list value for a primitive socket."
                    f" node_id={node_id!r} node_type={class_type!r}"
                    f" input_name={input_name!r} declared_type={declared_type!r}"
                    f" received_value={literal_value!r}"
                )


def _validate_required_prompt_inputs(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> None:
    """Reject remote prompt nodes that are missing declared required inputs."""
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        inputs = prompt_node.get("inputs") or {}
        missing_inputs = sorted(
            _node_required_input_names(node_class, inputs) - set(inputs.keys())
        )
        if not missing_inputs:
            continue
        raise RemoteSubgraphExecutionError(
            "Remote subgraph prompt is missing required node inputs before execution."
            f" node_id={node_id!r} node_type={class_type!r}"
            f" missing_inputs={missing_inputs!r} available_inputs={sorted(str(key) for key in inputs)!r}"
        )


def _log_prompt_executor_failure_details(
    *,
    component_id: str,
    prompt: dict[str, Any],
    normalized_payload: dict[str, Any],
    executor: Any,
) -> None:
    """Emit high-signal diagnostics for one remote PromptExecutor failure."""
    error_payload = _extract_prompt_executor_error_payload(executor)
    suspicious_inputs = _summarize_suspicious_prompt_inputs(prompt)
    logger.error(
        "Remote PromptExecutor failed for component=%s execute_node_ids=%s boundary_outputs=%s suspicious_inputs=%s error_payload=%s",
        component_id,
        normalized_payload.get("execute_node_ids", []),
        [
            {
                "node_id": boundary_output.get("node_id"),
                "output_index": boundary_output.get("output_index"),
            }
            for boundary_output in normalized_payload.get("boundary_outputs", [])
        ],
        suspicious_inputs,
        error_payload,
    )


def _resolve_required_subgraph_nodes(
    prompt: dict[str, Any],
    execute_node_ids: list[str],
) -> list[str]:
    """Return the dependency closure needed to execute the requested subgraph nodes."""
    required: set[str] = set()
    pending = list(execute_node_ids)
    logger.info(
        "Resolving dependency closure for remote execute targets: %s", execute_node_ids
    )
    while pending:
        node_id = str(pending.pop())
        if node_id not in prompt:
            logger.warning(
                "Skipping missing remote execute target %s while resolving dependency closure.",
                node_id,
            )
            continue
        if node_id in required:
            continue
        required.add(node_id)
        for input_value in (prompt[node_id].get("inputs") or {}).values():
            if _is_link(input_value):
                pending.append(str(input_value[0]))
    resolved = sorted(required)
    logger.info("Resolved remote dependency closure: %s", resolved)
    return resolved


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _normalize_link_output_index(value: Any) -> Any:
    """Unwrap a singleton list around a prompt-link output index when present."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _unwrap_wrapped_prompt_link(value: Any) -> Any:
    """Collapse nested singleton wrappers around one serialized prompt link when present."""
    candidate = value
    while isinstance(candidate, list) and len(candidate) == 1:
        candidate = candidate[0]
    if _is_link(candidate):
        return [candidate[0], _normalize_link_output_index(candidate[1])]
    return value


def _normalize_prompt_input_value(value: Any, io_type: str | None = None) -> Any:
    """Unwrap transport-added singleton wrappers only for scalar-like prompt input values."""
    wrapped_link = _unwrap_wrapped_prompt_link(value)
    if wrapped_link is not value:
        return wrapped_link
    while (
        isinstance(value, list)
        and len(value) == 1
        and (
            io_type in _PRIMITIVE_WIDGET_INPUT_TYPES
            or value[0] is None
            or isinstance(value[0], bool | int | float | str)
        )
    ):
        value = value[0]
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return [value[0], _normalize_link_output_index(value[1])]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return value


def _normalize_subgraph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a subgraph payload with canonical prompt-link and output-index shapes."""
    normalized_payload = copy.deepcopy(payload)

    for node_info in normalized_payload.get("subgraph_prompt", {}).values():
        inputs = node_info.get("inputs") or {}
        for input_name, input_value in list(inputs.items()):
            inputs[input_name] = _normalize_prompt_input_value(input_value)

    for boundary_output in normalized_payload.get("boundary_outputs", []):
        if "node_id" in boundary_output and isinstance(
            boundary_output["node_id"], list
        ):
            boundary_output["node_id"] = _normalize_prompt_input_value(
                boundary_output["node_id"]
            )
        if "output_index" in boundary_output:
            boundary_output["output_index"] = _normalize_link_output_index(
                boundary_output["output_index"]
            )

    normalized_payload["execute_node_ids"] = [
        _normalize_prompt_input_value(node_id)
        for node_id in normalized_payload.get("execute_node_ids", [])
    ]

    return normalized_payload


def _trim_subgraph_payload_to_required_nodes(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim a subgraph payload down to the dependency closure of its execute targets."""
    trimmed_payload = copy.deepcopy(payload)
    prompt = trimmed_payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return trimmed_payload

    prompt_node_ids = {str(node_id) for node_id in prompt}
    requested_execute_node_ids = [
        str(node_id) for node_id in trimmed_payload.get("execute_node_ids", [])
    ]
    available_execute_node_ids = [
        node_id for node_id in requested_execute_node_ids if node_id in prompt_node_ids
    ]
    dropped_execute_node_ids = [
        node_id
        for node_id in requested_execute_node_ids
        if node_id not in prompt_node_ids
    ]
    if dropped_execute_node_ids:
        logger.warning(
            "Dropping remote execute targets absent from subgraph prompt for component=%s: %s",
            payload.get("component_id"),
            dropped_execute_node_ids,
        )

    required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=available_execute_node_ids,
        )
    )
    if not required_node_ids:
        return trimmed_payload

    original_node_ids = list(prompt.keys())
    trimmed_payload["subgraph_prompt"] = {
        str(node_id): prompt[node_id]
        for node_id in original_node_ids
        if str(node_id) in required_node_ids
    }
    trimmed_payload["boundary_inputs"] = [
        {
            **copy.deepcopy(boundary_input),
            "targets": [
                copy.deepcopy(target)
                for target in boundary_input.get("targets", [])
                if str(target.get("node_id")) in required_node_ids
            ],
        }
        for boundary_input in trimmed_payload.get("boundary_inputs", [])
        if any(
            str(target.get("node_id")) in required_node_ids
            for target in boundary_input.get("targets", [])
        )
    ]
    trimmed_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in trimmed_payload.get("boundary_outputs", [])
        if str(boundary_output.get("node_id")) in required_node_ids
    ]
    trimmed_payload["component_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("component_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["mapped_execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("mapped_execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["static_execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("static_execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    logger.info(
        "Trimmed remote subgraph payload %s from %d prompt nodes to %d required nodes.",
        payload.get("component_id"),
        len(original_node_ids),
        len(trimmed_payload["subgraph_prompt"]),
    )
    return trimmed_payload


@dataclass(frozen=True)
class _PreparedSubgraphExecution:
    """Hold validated prompt state required by one PromptExecutor invocation."""

    component_id: str
    payload: dict[str, Any]
    prompt: dict[str, Any]
    metadata_prompt: dict[str, Any]
    execution: Any
    cache_type: Any
    cache_args: dict[str, float]
    session_handle: RemoteSessionHandle | None


def _resolve_subgraph_session_inputs(
    *,
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
) -> tuple[
    dict[str, Any],
    RemoteSessionHandle | None,
    dict[str, Any],
    tuple[Any, ...] | None,
]:
    """Normalize a payload and resolve its session-backed boundary inputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    normalized_payload = _trim_subgraph_payload_to_required_nodes(
        _normalize_subgraph_payload(payload)
    )
    session_handle = _payload_remote_session_handle(normalized_payload)
    resolution_stats = _RemoteSessionBridgeResolutionStats()
    loader_cache_before = _loader_cache_metric_snapshot()
    resolved_inputs = _resolve_remote_session_inputs(
        dict(hydrated_inputs),
        component_id=component_id,
        target_session_handle=session_handle,
        custom_nodes_root=custom_nodes_root,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
        resolution_stats=resolution_stats,
    )
    _log_remote_session_resolution_summary(
        component_id=component_id,
        resolution_stats=resolution_stats,
        loader_cache_before=loader_cache_before,
        loader_cache_after=_loader_cache_metric_snapshot(),
    )
    short_circuit_outputs = _short_circuit_restored_session_output_subgraph(
        payload=normalized_payload,
        hydrated_inputs=hydrated_inputs,
        session_handle=session_handle,
        resolution_stats=resolution_stats,
    )
    return normalized_payload, session_handle, resolved_inputs, short_circuit_outputs


def _load_subgraph_execution_runtime(
    *,
    component_id: str,
    normalized_payload: dict[str, Any],
    prompt: dict[str, Any],
    custom_nodes_root: Path | None,
) -> tuple[Any, Any, dict[str, float], Mapping[str, type[Any]]]:
    """Load ComfyUI execution APIs and resolve every prompt node class."""
    with _timed_phase("load_execution_module", component=component_id):
        execution = _load_execution_module()
        _install_metadata_safe_dynamic_prompt_wrapper(execution)
        cache_type, cache_args = _prompt_executor_cache_config(execution)
        custom_nodes_bundle_path = normalized_payload.get("custom_nodes_bundle")
        resolved_node_mapping = _ensure_prompt_node_classes_registered(
            component_id=component_id,
            prompt=prompt,
            custom_nodes_root=custom_nodes_root,
            custom_nodes_bundle_path=(
                custom_nodes_bundle_path
                if isinstance(custom_nodes_bundle_path, str)
                else None
            ),
        )
    return execution, cache_type, cache_args, resolved_node_mapping


def _prepare_subgraph_execution(
    *,
    source_payload: dict[str, Any],
    normalized_payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    resolved_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    session_handle: RemoteSessionHandle | None,
) -> _PreparedSubgraphExecution:
    """Build, hydrate, load, coerce, and validate one remote ComfyUI prompt."""
    component_id = str(source_payload.get("component_id", "modal-subgraph"))
    if session_handle is not None:
        logger.info(
            "Executing cloud subgraph component=%s with remote_session session_id=%s prompt_id=%s owner_component_id=%s.",
            component_id,
            session_handle.session_id,
            session_handle.prompt_id,
            session_handle.owner_component_id,
        )
    with _timed_phase("prepare_subgraph_prompt", component=component_id):
        prompt = _rewrite_modal_asset_references(
            copy.deepcopy(normalized_payload["subgraph_prompt"])
        )
        metadata_prompt = _copy_json_safe_prompt_metadata(prompt)
        _apply_boundary_inputs(
            prompt=prompt,
            boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
            hydrated_inputs=resolved_inputs,
            cache_signature_inputs=hydrated_inputs,
        )
    execution, cache_type, cache_args, resolved_node_mapping = (
        _load_subgraph_execution_runtime(
            component_id=component_id,
            normalized_payload=normalized_payload,
            prompt=prompt,
            custom_nodes_root=custom_nodes_root,
        )
    )
    _coerce_prompt_primitive_input_values(prompt, resolved_node_mapping)
    _validate_prompt_input_shapes(
        prompt,
        resolved_node_mapping,
        list(normalized_payload.get("boundary_inputs", [])),
    )
    _validate_required_prompt_inputs(prompt, resolved_node_mapping)
    return _PreparedSubgraphExecution(
        component_id=component_id,
        payload=normalized_payload,
        prompt=prompt,
        metadata_prompt=metadata_prompt,
        execution=execution,
        cache_type=cache_type,
        cache_args=cache_args,
        session_handle=session_handle,
    )


def _install_subgraph_node_cache_restore(
    *,
    prepared: _PreparedSubgraphExecution,
    executor: Any,
    cache_store: Any | None,
) -> _PersistedNodeCacheRestoreState | None:
    """Install persisted-cache hydration when distributed caching is enabled."""
    if cache_store is None or get_settings().node_output_cache_max_bytes <= 0:
        return None
    return _install_prompt_executor_persisted_cache_restore(
        prepared.execution,
        executor,
        component_id=prepared.component_id,
        prompt=prepared.prompt,
        required_materialized_node_ids=_boundary_output_node_ids(
            prepared.payload.get("boundary_outputs", [])
        ),
        cache_store=cache_store,
    )


def _emit_restored_subgraph_cache_events(
    component_id: str,
    restore_state: _PersistedNodeCacheRestoreState | None,
    status_callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Log and publish persisted-cache hits from one prompt execution."""
    restored_node_ids = (
        restore_state.restored_node_ids if restore_state is not None else []
    )
    if not restored_node_ids:
        return
    _emit_cloud_info(
        "Restored %d persisted node cache entries for component=%s: %s",
        len(restored_node_ids),
        component_id,
        restored_node_ids,
    )
    _emit_restored_node_cache_events(status_callback, restored_node_ids)


def _execute_prepared_prompt(
    *,
    source_payload: dict[str, Any],
    prepared: _PreparedSubgraphExecution,
    executor: Any,
    prompt_server: _TracingPromptServer,
    cache_store: Any | None,
    status_callback: Callable[[dict[str, Any]], None] | None,
) -> _PersistedNodeCacheRestoreState | None:
    """Execute one prepared prompt while protecting its reusable executor state."""
    _reset_prompt_executor_request_state(executor, prompt_server)
    restore_state = _install_subgraph_node_cache_restore(
        prepared=prepared,
        executor=executor,
        cache_store=cache_store,
    )
    prompt_server.configure_boundary_output_stream(
        boundary_outputs=list(prepared.payload.get("boundary_outputs", [])),
        lookup_cache_entry=lambda node_id: _prompt_executor_cache_get_sync(
            executor.caches.outputs,
            node_id,
        ),
    )
    try:
        with _timed_phase(
            "prompt_executor_execute",
            component=prepared.component_id,
            execute_nodes=list(prepared.payload.get("execute_node_ids", [])),
        ):
            with _temporary_prompt_metadata(prepared.metadata_prompt):
                _execute_prompt_executor_compat(
                    executor,
                    prompt=prepared.prompt,
                    prompt_id=str(
                        source_payload.get("prompt_id") or prepared.component_id
                    ),
                    extra_data=copy.deepcopy(prepared.payload.get("extra_data") or {}),
                    execute_outputs=list(prepared.payload.get("execute_node_ids", [])),
                )
    finally:
        if restore_state is not None:
            restore_state.restore_original_method()
    _emit_restored_subgraph_cache_events(
        prepared.component_id,
        restore_state,
        status_callback,
    )
    return restore_state


def _persist_prepared_prompt_cache(
    *,
    prepared: _PreparedSubgraphExecution,
    executor: Any,
    cache_store: Any | None,
    restore_state: _PersistedNodeCacheRestoreState | None,
) -> None:
    """Write eligible prompt outputs into the distributed node cache."""
    if cache_store is None or get_settings().node_output_cache_max_bytes <= 0:
        return
    with _timed_phase("persist_node_cache", component=prepared.component_id):
        persisted_node_ids = asyncio.run(
            _persist_node_output_cache_entries(
                executor,
                prompt=prepared.prompt,
                cache_store=cache_store,
                restored_cache_keys_by_node_id=(
                    restore_state.restored_cache_keys_by_node_id
                    if restore_state is not None
                    else None
                ),
            ),
        )
    if persisted_node_ids:
        _emit_cloud_info(
            "Persisted %d node cache entries for component=%s: %s",
            len(persisted_node_ids),
            prepared.component_id,
            persisted_node_ids,
        )


def _store_session_boundary_output(
    *,
    prepared: _PreparedSubgraphExecution,
    hydrated_inputs: dict[str, Any],
    boundary_output: Mapping[str, Any],
    output_value: Any,
) -> Any:
    """Store one session-backed boundary value and return its durable reference."""
    session_handle = prepared.session_handle
    if session_handle is None:
        raise RemoteSessionStateError(
            "Session-backed boundary outputs require payload.remote_session."
        )
    node_id = str(boundary_output["node_id"])
    output_index = int(boundary_output["output_index"])
    bridge_record = _build_remote_session_bridge_record(
        payload=prepared.payload,
        hydrated_inputs=hydrated_inputs,
        node_id=node_id,
        output_index=output_index,
        io_type=str(boundary_output.get("io_type") or "*"),
        output_value=output_value,
    )
    _store_remote_session_bridge_record(bridge_record)
    _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
    live_ref = remote_session_store().put_bridge_output(
        session_handle,
        bridge_key=bridge_record.bridge_key,
        node_id=node_id,
        output_index=output_index,
        value=output_value,
    )
    return RemoteSessionBridgeRef(
        bridge_key=bridge_record.bridge_key,
        node_id=node_id,
        output_index=output_index,
        session_id=live_ref.session_id,
    ).to_payload()


def _collect_subgraph_boundary_outputs(
    *,
    prepared: _PreparedSubgraphExecution,
    executor: Any,
    hydrated_inputs: dict[str, Any],
) -> tuple[Any, ...]:
    """Collect and optionally persist every declared subgraph boundary output."""
    boundary_outputs = list(prepared.payload.get("boundary_outputs", []))
    outputs: list[Any] = []
    with _timed_phase(
        "collect_boundary_outputs",
        component=prepared.component_id,
        output_count=len(boundary_outputs),
    ):
        for boundary_output in boundary_outputs:
            node_id = str(boundary_output["node_id"])
            output_index = int(boundary_output["output_index"])
            cache_entry = _prompt_executor_cache_get_sync(
                executor.caches.outputs,
                node_id,
            )
            if cache_entry is None:
                raise RemoteSubgraphExecutionError(
                    f"Remote subgraph did not produce cache entry for node {node_id}."
                )
            if output_index >= len(cache_entry.outputs):
                raise RemoteSubgraphExecutionError(
                    f"Remote subgraph output index {output_index} is missing for node {node_id}."
                )
            output_value = _collapse_cache_slot(
                slot_values=cache_entry.outputs[output_index],
                is_list=bool(boundary_output.get("is_list", False)),
            )
            if bool(boundary_output.get("session_output")):
                output_value = _store_session_boundary_output(
                    prepared=prepared,
                    hydrated_inputs=hydrated_inputs,
                    boundary_output=boundary_output,
                    output_value=output_value,
                )
            outputs.append(output_value)
    return tuple(outputs)


def _execute_prepared_subgraph(
    *,
    source_payload: dict[str, Any],
    prepared: _PreparedSubgraphExecution,
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
) -> tuple[Any, ...]:
    """Run a prepared subgraph under ComfyUI's temporary execution adapters."""
    prompt_server = _TracingPromptServer(
        prepared.component_id,
        prepared.prompt,
        status_callback=status_callback,
    )
    with (
        _temporary_node_mapping(None),
        _patched_folder_paths_absolute_lookup(),
        _temporary_remote_interrupt_monitor(
            prepared.component_id,
            cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key=interrupt_flag_key,
        ),
        _temporary_progress_hook(prompt_server),
    ):
        with _timed_phase("create_prompt_executor", component=prepared.component_id):
            executor_state = _get_or_create_prompt_executor_state(
                execution=prepared.execution,
                prompt_server=prompt_server,
                cache_type=prepared.cache_type,
                cache_args=prepared.cache_args,
                custom_nodes_root=custom_nodes_root,
            )
        cache_store = _node_output_cache_store()
        with executor_state.lock:
            restore_state = _execute_prepared_prompt(
                source_payload=source_payload,
                prepared=prepared,
                executor=executor_state.executor,
                prompt_server=prompt_server,
                cache_store=cache_store,
                status_callback=status_callback,
            )
        executor = executor_state.executor
        if not executor.success:
            _log_prompt_executor_failure_details(
                component_id=prepared.component_id,
                prompt=prepared.prompt,
                normalized_payload=prepared.payload,
                executor=executor,
            )
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))
        _persist_prepared_prompt_cache(
            prepared=prepared,
            executor=executor,
            cache_store=cache_store,
            restore_state=restore_state,
        )
        return _collect_subgraph_boundary_outputs(
            prepared=prepared,
            executor=executor,
            hydrated_inputs=hydrated_inputs,
        )


def _execute_subgraph_prompt(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    normalized_payload, session_handle, resolved_inputs, short_circuit_outputs = (
        _resolve_subgraph_session_inputs(
            payload=payload,
            hydrated_inputs=hydrated_inputs,
            custom_nodes_root=custom_nodes_root,
            cancellation_event=cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key=interrupt_flag_key,
        )
    )
    if short_circuit_outputs is not None:
        logger.info(
            "Skipping prompt_executor_execute for component=%s because all %d session-backed outputs were restored into session_id=%s.",
            payload.get("component_id", "modal-subgraph"),
            len(short_circuit_outputs),
            session_handle.session_id if session_handle is not None else None,
        )
        return short_circuit_outputs
    prepared = _prepare_subgraph_execution(
        source_payload=payload,
        normalized_payload=normalized_payload,
        hydrated_inputs=hydrated_inputs,
        resolved_inputs=resolved_inputs,
        custom_nodes_root=custom_nodes_root,
        session_handle=session_handle,
    )
    return _execute_prepared_subgraph(
        source_payload=payload,
        prepared=prepared,
        hydrated_inputs=hydrated_inputs,
        custom_nodes_root=custom_nodes_root,
        status_callback=status_callback,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )


def _short_circuit_restored_session_output_subgraph(
    *,
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    session_handle: RemoteSessionHandle | None,
    resolution_stats: _RemoteSessionBridgeResolutionStats,
) -> tuple[Any, ...] | None:
    """Return session-backed outputs directly when bridge restoration already satisfied them all."""
    boundary_outputs = list(payload.get("boundary_outputs", []))
    if session_handle is None or not boundary_outputs:
        return None
    if resolution_stats.input_ref_count <= 0 or resolution_stats.replay_count > 0:
        return None
    if any(
        not bool(boundary_output.get("session_output"))
        for boundary_output in boundary_outputs
    ):
        return None

    restored_outputs: list[Any] = []
    for boundary_output in boundary_outputs:
        node_id = str(boundary_output["node_id"])
        output_index = int(boundary_output["output_index"])
        try:
            output_value = remote_session_store().get_output(
                RemoteSessionValueRef(
                    session_id=session_handle.session_id,
                    node_id=node_id,
                    output_index=output_index,
                )
            )
        except RemoteSessionStateError:
            return None

        bridge_record = _build_remote_session_bridge_record(
            payload=payload,
            hydrated_inputs=hydrated_inputs,
            node_id=node_id,
            output_index=output_index,
            io_type=str(boundary_output.get("io_type") or "*"),
            output_value=output_value,
        )
        _store_remote_session_bridge_record(bridge_record)
        _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
        live_ref = remote_session_store().put_bridge_output(
            session_handle,
            bridge_key=bridge_record.bridge_key,
            node_id=node_id,
            output_index=output_index,
            value=output_value,
        )
        restored_outputs.append(
            RemoteSessionBridgeRef(
                bridge_key=bridge_record.bridge_key,
                node_id=node_id,
                output_index=output_index,
                session_id=live_ref.session_id,
            ).to_payload()
        )
    return tuple(restored_outputs)


def execute_subgraph_locally(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    session_handle = _payload_remote_session_handle(payload)
    with _timed_phase("execute_subgraph_locally", component=component_id):
        custom_nodes_root = _extract_custom_nodes_bundle(
            payload.get("custom_nodes_bundle")
        )
        _ensure_comfy_runtime_initialized(custom_nodes_root)
        with _timed_phase("deserialize_boundary_inputs", component=component_id):
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        logger.info(
            "Executing cloud-local subgraph component=%s hydrated_inputs=%d session_id=%s clear_remote_session=%s.",
            component_id,
            len(hydrated_inputs),
            session_handle.session_id if session_handle is not None else None,
            bool(payload.get("clear_remote_session")),
        )
        try:
            with _timed_phase("subgraph_worker_roundtrip", component=component_id):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _execute_subgraph_prompt,
                        payload,
                        hydrated_inputs,
                        custom_nodes_root,
                        status_callback,
                        cancellation_event,
                        interrupt_store,
                        interrupt_flag_key,
                    )
                    outputs = future.result()
        finally:
            if bool(payload.get("clear_remote_session")) and session_handle is not None:
                logger.info(
                    "Clearing remote session after cloud component=%s session_id=%s.",
                    component_id,
                    session_handle.session_id,
                )
                remote_session_store().clear_session(session_handle)
        with _timed_phase("serialize_boundary_outputs", component=component_id):
            return serialize_node_outputs(outputs)


def _mapped_phase_definition(
    payload: dict[str, Any], phase_key: str
) -> dict[str, Any] | None:
    """Return one explicit mapped phase definition when queue-time planning provided it."""
    phase_payload = payload.get(phase_key)
    if isinstance(phase_payload, dict):
        return phase_payload
    return None


def _shared_subgraph_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload fields shared by every explicit mapped phase."""
    shared_fields = {
        "prompt_id": payload.get("prompt_id"),
        "extra_data": copy.deepcopy(payload.get("extra_data") or {}),
        "requires_volume_reload": bool(payload.get("requires_volume_reload", True)),
        "volume_reload_marker": payload.get("volume_reload_marker"),
        "uploaded_volume_paths": list(payload.get("uploaded_volume_paths", [])),
        "terminate_container_on_error": bool(
            payload.get("terminate_container_on_error", True)
        ),
        "custom_nodes_bundle": payload.get("custom_nodes_bundle"),
    }
    snapshot_profile_key = payload.get("snapshot_profile_key")
    if isinstance(snapshot_profile_key, str) and snapshot_profile_key.strip():
        shared_fields["snapshot_profile_key"] = snapshot_profile_key.strip()
    remote_session = payload.get("remote_session")
    if remote_session is not None:
        shared_fields["remote_session"] = copy.deepcopy(remote_session)
    if bool(payload.get("clear_remote_session")):
        shared_fields["clear_remote_session"] = True
    return shared_fields


def _build_phase_subgraph_payload(
    payload: dict[str, Any],
    phase_key: str,
    component_id: str,
) -> dict[str, Any]:
    """Return one explicit static or mapped subgraph payload."""
    phase_definition = _mapped_phase_definition(payload, phase_key)
    if phase_definition is None:
        raise KeyError(f"Mapped payload is missing phase definition {phase_key!r}.")

    return {
        "payload_kind": "subgraph",
        "component_id": component_id,
        **_shared_subgraph_payload_fields(payload),
        "component_node_ids": [
            str(node_id)
            for node_id in phase_definition.get("component_node_ids", [])
            if str(node_id)
        ],
        "subgraph_prompt": copy.deepcopy(phase_definition.get("subgraph_prompt", {})),
        "boundary_inputs": copy.deepcopy(phase_definition.get("boundary_inputs", [])),
        "boundary_outputs": copy.deepcopy(phase_definition.get("boundary_outputs", [])),
        "execute_node_ids": [
            str(node_id)
            for node_id in phase_definition.get("execute_node_ids", [])
            if str(node_id)
        ],
    }


def _split_phase_outputs(
    phase_outputs: tuple[Any, ...],
    boundary_outputs: list[dict[str, Any]],
    internal_output_names: set[str],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Split one phase result tuple into bridge values and external outputs."""
    internal_outputs: dict[str, Any] = {}
    external_outputs: list[Any] = []
    for boundary_output, output_value in zip(
        boundary_outputs, phase_outputs, strict=True
    ):
        output_name = str(boundary_output.get("proxy_output_name") or "")
        if output_name in internal_output_names:
            internal_outputs[output_name] = output_value
            continue
        external_outputs.append(output_value)
    return internal_outputs, tuple(external_outputs)


def _aggregate_mapped_phase_outputs(
    per_item_outputs: list[tuple[Any, ...]],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Join ordered mapped-phase outputs back into one proxy result tuple."""
    if not per_item_outputs:
        raise ValueError("Mapped execution produced no per-item outputs to aggregate.")

    output_count = len(per_item_outputs[0])
    if any(len(item_outputs) != output_count for item_outputs in per_item_outputs):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced inconsistent output arity."
        )

    aggregated_outputs: list[Any] = []
    boundary_outputs = list(payload.get("boundary_outputs", []))
    for output_index in range(output_count):
        boundary_output = (
            boundary_outputs[output_index]
            if output_index < len(boundary_outputs)
            else {}
        )
        aggregated_outputs.append(
            _merge_static_or_mapped_values(
                [item_outputs[output_index] for item_outputs in per_item_outputs],
                io_type=str(boundary_output.get("io_type", "*")),
                is_list=bool(boundary_output.get("is_list", False)),
                scheduler_is_list=bool(boundary_output.get("scheduler_is_list", False)),
            )
        )
    return tuple(aggregated_outputs)


def _merge_static_or_mapped_values(
    values: list[Any],
    *,
    io_type: str,
    is_list: bool,
    scheduler_is_list: bool,
) -> Any:
    """Join mapped per-item outputs using the shared transport serializer rules."""
    from serialization import join_mapped_values_for_scheduler

    return join_mapped_values_for_scheduler(
        values,
        io_type=io_type,
        is_list=is_list,
        scheduler_is_list=scheduler_is_list,
    )


def _merge_static_and_mapped_outputs(
    *,
    static_outputs: tuple[Any, ...],
    mapped_outputs: tuple[Any, ...],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble one mapped component's static and mapped outputs in proxy order."""
    combined_outputs: list[Any] = []
    static_output_index = 0
    mapped_output_index = 0
    for boundary_output in payload.get("boundary_outputs", []):
        if bool(boundary_output.get("mapped_output")):
            if mapped_output_index >= len(mapped_outputs):
                raise RemoteSubgraphExecutionError(
                    "Mapped remote execution returned fewer mapped outputs than expected."
                )
            combined_outputs.append(mapped_outputs[mapped_output_index])
            mapped_output_index += 1
            continue
        if static_output_index >= len(static_outputs):
            raise RemoteSubgraphExecutionError(
                "Mapped remote execution returned fewer static outputs than expected."
            )
        combined_outputs.append(static_outputs[static_output_index])
        static_output_index += 1

    if static_output_index != len(static_outputs) or mapped_output_index != len(
        mapped_outputs
    ):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced extra outputs that did not match the declared boundary outputs."
        )
    return tuple(combined_outputs)


def _execute_mapped_subgraph_payload(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute one mapped payload inside a single remote runtime process."""
    mapped_input = payload.get("mapped_input") or {}
    mapped_input_name = str(mapped_input.get("proxy_input_name") or "")
    if not mapped_input_name:
        raise RemoteSubgraphExecutionError(
            "Mapped remote payloads must define mapped_input.proxy_input_name."
        )
    if mapped_input_name not in hydrated_inputs:
        raise KeyError(
            f"Mapped remote payload input {mapped_input_name!r} was not provided."
        )

    from serialization import split_mapped_value

    mapped_items = split_mapped_value(
        hydrated_inputs[mapped_input_name],
        str(mapped_input.get("io_type", "*")),
    )
    if not mapped_items:
        raise ValueError("Mapped remote execution requires at least one input item.")

    broadcast_inputs = dict(hydrated_inputs)
    broadcast_inputs.pop(mapped_input_name, None)
    static_to_mapped_boundaries = list(payload.get("static_to_mapped_boundaries", []))
    bridge_output_names = {
        str(boundary_spec.get("proxy_name") or "")
        for boundary_spec in static_to_mapped_boundaries
        if str(boundary_spec.get("proxy_name") or "")
    }

    static_outputs: tuple[Any, ...] = ()
    if payload.get("static_phase") is not None:
        static_phase_payload = _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
        )
        if static_phase_payload.get("execute_node_ids"):
            static_phase_outputs = _execute_subgraph_prompt(
                static_phase_payload,
                dict(broadcast_inputs),
                custom_nodes_root,
                status_callback=status_callback,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            )
            bridge_inputs, static_outputs = _split_phase_outputs(
                static_phase_outputs,
                list(static_phase_payload.get("boundary_outputs", [])),
                bridge_output_names,
            )
            broadcast_inputs.update(bridge_inputs)

    if status_callback is not None:
        status_callback(
            {
                "event_type": "node_progress",
                "node_id": str(payload.get("component_id") or "modal-subgraph"),
                "display_node_id": str(payload.get("component_id") or "modal-subgraph"),
                "value": 0.0,
                "max": float(len(mapped_items)),
                "aggregate_only": True,
            }
        )

    per_item_outputs: list[tuple[Any, ...]] = []
    for item_index, item_value in enumerate(mapped_items):
        last_lane_node_id: str | None = None
        lane_id = str(payload.get("mapped_progress_lane_id") or item_index)

        def publish_item_status(progress_state: dict[str, Any]) -> None:
            """Attach mapped-lane metadata to one per-item progress event."""
            nonlocal last_lane_node_id
            if status_callback is None:
                return
            event_type = str(progress_state.get("event_type", ""))
            if event_type == "node_progress":
                reported_node_id = progress_state.get(
                    "real_node_id"
                ) or progress_state.get("node_id")
                if reported_node_id is not None:
                    last_lane_node_id = str(reported_node_id)
                status_callback(
                    {
                        **progress_state,
                        "lane_id": lane_id,
                        "item_index": item_index,
                    }
                )
                return
            if event_type in {"executed", "preview", "boundary_output"}:
                status_callback({**progress_state, "item_index": item_index})

        item_payload = _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
        )
        item_inputs = dict(broadcast_inputs)
        item_inputs[mapped_input_name] = item_value
        per_item_outputs.append(
            _execute_subgraph_prompt(
                item_payload,
                item_inputs,
                custom_nodes_root,
                status_callback=publish_item_status,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            )
        )
        if status_callback is not None:
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": last_lane_node_id
                    or str(payload.get("component_id") or "modal-subgraph"),
                    "display_node_id": last_lane_node_id
                    or str(payload.get("component_id") or "modal-subgraph"),
                    "value": 0.0,
                    "max": 1.0,
                    "lane_id": lane_id,
                    "item_index": item_index,
                    "clear": True,
                }
            )
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": str(payload.get("component_id") or "modal-subgraph"),
                    "display_node_id": str(
                        payload.get("component_id") or "modal-subgraph"
                    ),
                    "value": float(item_index + 1),
                    "max": float(len(mapped_items)),
                    "aggregate_only": True,
                }
            )

    mapped_phase_payload = _build_phase_subgraph_payload(
        payload,
        "mapped_phase",
        f"{payload.get('component_id', 'modal-subgraph')}::mapped",
    )
    mapped_outputs = _aggregate_mapped_phase_outputs(
        per_item_outputs,
        {"boundary_outputs": list(mapped_phase_payload.get("boundary_outputs", []))},
    )
    return _merge_static_and_mapped_outputs(
        static_outputs=static_outputs,
        mapped_outputs=mapped_outputs,
        payload=payload,
    )
