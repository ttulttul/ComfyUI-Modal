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
    from .cloud_prompt_validation import (
        _coerce_primitive_prompt_input_value,
        _coerce_prompt_primitive_input_values,
        _extract_prompt_executor_error,
        _extract_prompt_executor_error_payload,
        _format_prompt_executor_error_payload,
        _is_link,
        _log_prompt_executor_failure_details,
        _node_input_type_map,
        _node_input_types,
        _node_required_input_names,
        _normalize_link_output_index,
        _normalize_prompt_input_value,
        _normalize_subgraph_payload,
        _resolve_required_subgraph_nodes,
        _summarize_suspicious_prompt_inputs,
        _trim_subgraph_payload_to_required_nodes,
        _unwrap_wrapped_prompt_link,
        _validate_prompt_input_shapes,
        _validate_required_prompt_inputs,
    )
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
    from cloud_prompt_validation import (
        _coerce_primitive_prompt_input_value,
        _coerce_prompt_primitive_input_values,
        _extract_prompt_executor_error,
        _extract_prompt_executor_error_payload,
        _format_prompt_executor_error_payload,
        _is_link,
        _log_prompt_executor_failure_details,
        _node_input_type_map,
        _node_input_types,
        _node_required_input_names,
        _normalize_link_output_index,
        _normalize_prompt_input_value,
        _normalize_subgraph_payload,
        _resolve_required_subgraph_nodes,
        _summarize_suspicious_prompt_inputs,
        _trim_subgraph_payload_to_required_nodes,
        _unwrap_wrapped_prompt_link,
        _validate_prompt_input_shapes,
        _validate_required_prompt_inputs,
    )
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
