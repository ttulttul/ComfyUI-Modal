"""Remote Modal runtime and local execution fallback."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
from io import BytesIO
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator

from ..serialization import (
    join_mapped_values,
    coerce_serialized_node_outputs,
    deserialize_value,
    deserialize_node_inputs,
    deserialize_node_outputs,
    split_mapped_value,
    serialize_node_outputs,
    serialize_node_inputs,
)
from ..settings import get_settings

logger = logging.getLogger(__name__)
_MODAL_CLOUD_MODULE_NAME = "comfyui_modal_sync_cloud"
_MODAL_AUTO_DEPLOY_LOCK = threading.Lock()
_MODAL_AUTO_DEPLOYED_APPS: set[tuple[str, str | None]] = set()
_MODAL_INTERRUPT_DICTS_LOCK = threading.Lock()
_MODAL_INTERRUPT_DICTS: dict[tuple[str, str | None], Any] = {}


def _remote_modal_call_worker_count() -> int:
    """Return the number of local worker threads reserved for blocking Modal calls."""
    configured_parallelism = get_settings().max_containers or 0
    return max(4, os.cpu_count() or 1, configured_parallelism)


_REMOTE_MODAL_CALL_EXECUTOR = ThreadPoolExecutor(max_workers=_remote_modal_call_worker_count())

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local fallback tests.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


class ModalRemoteInvocationError(RuntimeError):
    """Raised when the Modal client cannot invoke the remote runtime."""


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug("Suppressed remote prompt event %s for client %s.", event, client_id)


def _extract_custom_nodes_bundle(bundle_path: str | None) -> None:
    """Extract a mirrored custom_nodes archive into a temporary import path."""
    if not bundle_path:
        return

    settings = get_settings()
    if settings.execution_mode == "local":
        logger.debug("Skipping custom_nodes bundle extraction in local execution mode.")
        return

    local_bundle = settings.local_storage_root / bundle_path.lstrip("/")
    if not local_bundle.exists():
        logger.warning("Custom nodes bundle %s was not found in local storage.", local_bundle)
        return

    extraction_root = Path(tempfile.gettempdir()) / "comfy-modal-sync-custom-nodes"
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_bundle, "r") as archive:
        archive.extractall(extraction_root)

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _load_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    import execution

    return execution


@contextmanager
def _temporary_node_mapping(node_mapping: dict[str, type[Any]] | None) -> Iterator[None]:
    """Temporarily overlay node mappings for tests or custom runtimes."""
    if node_mapping is None:
        yield
        return

    nodes_module = _load_nodes_module()
    original_mappings = dict(nodes_module.NODE_CLASS_MAPPINGS)
    original_display_mappings = dict(getattr(nodes_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    try:
        nodes_module.NODE_CLASS_MAPPINGS.update(node_mapping)
        for class_type in node_mapping:
            nodes_module.NODE_DISPLAY_NAME_MAPPINGS.setdefault(class_type, class_type)
        yield
    finally:
        nodes_module.NODE_CLASS_MAPPINGS.clear()
        nodes_module.NODE_CLASS_MAPPINGS.update(original_mappings)
        if hasattr(nodes_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            nodes_module.NODE_DISPLAY_NAME_MAPPINGS.clear()
            nodes_module.NODE_DISPLAY_NAME_MAPPINGS.update(original_display_mappings)


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
) -> bytes:
    """Execute a single target node in-process and return serialized outputs."""
    _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    kwargs = deserialize_node_inputs(kwargs_payload)
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        outputs = _invoke_original_node(node_mapping[class_type], node_data, kwargs)
        return serialize_node_outputs(outputs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        outputs = _invoke_original_node(resolved_node_mapping[class_type], node_data, kwargs)
    return serialize_node_outputs(outputs)


def _apply_boundary_inputs(
    prompt: dict[str, Any],
    boundary_input_specs: list[dict[str, Any]],
    hydrated_inputs: dict[str, Any],
) -> None:
    """Inject hydrated local boundary inputs into a remote subgraph prompt."""
    logger.info("Applying %d hydrated boundary inputs to remote subgraph prompt.", len(boundary_input_specs))
    for boundary_input in boundary_input_specs:
        proxy_input_name = str(boundary_input["proxy_input_name"])
        if proxy_input_name not in hydrated_inputs:
            raise KeyError(f"Missing hydrated boundary input {proxy_input_name!r}.")
        value = hydrated_inputs[proxy_input_name]
        logger.info(
            "Applying boundary input %s to %d targets.",
            proxy_input_name,
            len(boundary_input.get("targets", [])),
        )
        for target in boundary_input.get("targets", []):
            node_id = str(target["node_id"])
            input_name = str(target["input_name"])
            prompt[node_id]["inputs"][input_name] = value


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
            return str(data.get("exception_message") or "Remote subgraph execution failed.")
        if event == "execution_interrupted":
            return "Remote subgraph execution was interrupted."
    return "Remote subgraph execution failed."


def _resolve_required_subgraph_nodes(
    prompt: dict[str, Any],
    execute_node_ids: list[str],
) -> list[str]:
    """Return the dependency closure needed to execute the requested subgraph nodes."""
    required: set[str] = set()
    pending = list(execute_node_ids)
    logger.info("Resolving dependency closure for remote execute targets: %s", execute_node_ids)
    while pending:
        node_id = str(pending.pop())
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


def _execute_subgraph_with_mapping(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> tuple[Any, ...]:
    """Execute a rewritten remote component using an explicit node mapping."""
    prompt = copy.deepcopy(payload["subgraph_prompt"])
    logger.info(
        "Executing remote subgraph %s via test mapping with %d prompt nodes.",
        payload.get("component_id"),
        len(prompt),
    )
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(payload.get("boundary_inputs", [])),
        hydrated_inputs=hydrated_inputs,
    )
    required_node_ids = _resolve_required_subgraph_nodes(
        prompt=prompt,
        execute_node_ids=list(payload.get("execute_node_ids", [])),
    )
    executed_outputs: dict[str, tuple[Any, ...]] = {}
    pending = set(required_node_ids)

    while pending:
        progressed = False
        logger.info("Mapped remote subgraph pending nodes: %s", sorted(pending))
        for node_id in list(sorted(pending)):
            prompt_node = prompt[node_id]
            kwargs: dict[str, Any] = {}
            unresolved_dependency = False
            for input_name, input_value in (prompt_node.get("inputs") or {}).items():
                if _is_link(input_value):
                    upstream_node_id = str(input_value[0])
                    if upstream_node_id not in executed_outputs:
                        unresolved_dependency = True
                        break
                    kwargs[str(input_name)] = executed_outputs[upstream_node_id][int(input_value[1])]
                else:
                    kwargs[str(input_name)] = input_value
            if unresolved_dependency:
                continue

            class_type = str(prompt_node["class_type"])
            if class_type not in node_mapping:
                raise KeyError(f"Remote node class {class_type!r} is not registered.")
            logger.info(
                "Executing mapped remote node %s (%s) with %d inputs.",
                node_id,
                class_type,
                len(kwargs),
            )
            executed_outputs[node_id] = _invoke_original_node(
                node_mapping[class_type],
                prompt_node,
                kwargs,
            )
            pending.remove(node_id)
            progressed = True
        if not progressed:
            raise RemoteSubgraphExecutionError(
                "Unable to resolve execution order for remote subgraph payload."
            )

    outputs: list[Any] = []
    for boundary_output in payload.get("boundary_outputs", []):
        node_id = str(boundary_output["node_id"])
        output_index = int(boundary_output["output_index"])
        node_outputs = executed_outputs.get(node_id)
        if node_outputs is None:
            raise RemoteSubgraphExecutionError(
                f"Remote subgraph did not execute boundary output node {node_id}."
            )
        outputs.append(node_outputs[output_index])
    logger.info(
        "Mapped remote subgraph %s produced %d exported outputs.",
        payload.get("component_id"),
        len(outputs),
    )
    return tuple(outputs)


def _execute_subgraph_prompt(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    if node_mapping is not None:
        return _execute_subgraph_with_mapping(payload, hydrated_inputs, node_mapping)

    prompt = copy.deepcopy(payload["subgraph_prompt"])
    logger.info(
        "Executing remote subgraph %s through PromptExecutor with %d prompt nodes, %d boundary inputs, and %d exported outputs.",
        payload.get("component_id"),
        len(prompt),
        len(payload.get("boundary_inputs", [])),
        len(payload.get("boundary_outputs", [])),
    )
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(payload.get("boundary_inputs", [])),
        hydrated_inputs=hydrated_inputs,
    )
    execution = _load_execution_module()

    with _temporary_node_mapping(node_mapping):
        executor = execution.PromptExecutor(_NullPromptServer())
        execution_started_at = time.perf_counter()
        logger.info(
            "Starting PromptExecutor for remote subgraph %s with execute targets %s.",
            payload.get("component_id"),
            payload.get("execute_node_ids", []),
        )
        executor.execute(
            prompt=prompt,
            prompt_id=str(payload.get("component_id", "modal-subgraph")),
            extra_data=copy.deepcopy(payload.get("extra_data") or {}),
            execute_outputs=list(payload.get("execute_node_ids", [])),
        )
        logger.info(
            "PromptExecutor finished for remote subgraph %s in %.3fs with success=%s and %d status messages.",
            payload.get("component_id"),
            time.perf_counter() - execution_started_at,
            executor.success,
            len(executor.status_messages),
        )
        if executor.status_messages:
            logger.info(
                "Remote subgraph %s status events: %s",
                payload.get("component_id"),
                [event for event, _data in executor.status_messages],
            )
        if not executor.success:
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))

        outputs: list[Any] = []
        for boundary_output in payload.get("boundary_outputs", []):
            node_id = str(boundary_output["node_id"])
            output_index = int(boundary_output["output_index"])
            cache_entry = executor.caches.outputs.get(node_id)
            if cache_entry is None:
                raise RemoteSubgraphExecutionError(
                    f"Remote subgraph did not produce cache entry for node {node_id}."
                )
            if output_index >= len(cache_entry.outputs):
                raise RemoteSubgraphExecutionError(
                    f"Remote subgraph output index {output_index} is missing for node {node_id}."
                )
            outputs.append(
                _collapse_cache_slot(
                    slot_values=cache_entry.outputs[output_index],
                    is_list=bool(boundary_output.get("is_list", False)),
                )
            )
            logger.info(
                "Collected exported output %s from node %s output %d.",
                boundary_output.get("proxy_output_name"),
                node_id,
                output_index,
            )
        return tuple(outputs)


def execute_subgraph_locally(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    logger.info(
        "Executing local fallback subgraph %s with %d hydrated inputs.",
        payload.get("component_id"),
        len(hydrated_inputs),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_subgraph_prompt, payload, hydrated_inputs, node_mapping)
        try:
            outputs = future.result()
        except Exception:
            logger.exception(
                "Local fallback subgraph %s raised while running in worker thread.",
                payload.get("component_id"),
            )
            raise
    logger.info(
        "Local fallback subgraph %s completed with %d outputs.",
        payload.get("component_id"),
        len(outputs),
    )
    return serialize_node_outputs(outputs)


def _modal_lookup_error_types() -> tuple[type[BaseException], ...]:
    """Return Modal exception types that indicate lookup or hydration failure."""
    if modal is None:
        return tuple()
    exception_module = getattr(modal, "exception", None)
    if exception_module is None:
        return tuple()

    error_types: list[type[BaseException]] = []
    for error_name in ("NotFoundError", "ExecutionError", "InvalidError"):
        error_type = getattr(exception_module, error_name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            error_types.append(error_type)
    return tuple(error_types)


def _load_modal_cloud_module() -> Any:
    """Load the stable Modal cloud entry module under a valid Python name."""
    existing_module = sys.modules.get(_MODAL_CLOUD_MODULE_NAME)
    if existing_module is not None and getattr(existing_module, "app", None) is not None:
        return existing_module
    if existing_module is not None:
        logger.warning(
            "Discarding partially initialized Modal cloud module %s before reload.",
            _MODAL_CLOUD_MODULE_NAME,
        )
        sys.modules.pop(_MODAL_CLOUD_MODULE_NAME, None)

    cloud_module_path = Path(__file__).resolve().parents[1] / f"{_MODAL_CLOUD_MODULE_NAME}.py"
    module_spec = importlib.util.spec_from_file_location(
        _MODAL_CLOUD_MODULE_NAME,
        cloud_module_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ModalRemoteInvocationError(
            f"Unable to create module spec for Modal cloud entrypoint at {cloud_module_path}."
        )

    cloud_module = importlib.util.module_from_spec(module_spec)
    sys.modules[_MODAL_CLOUD_MODULE_NAME] = cloud_module
    try:
        module_spec.loader.exec_module(cloud_module)
    except BaseException:
        sys.modules.pop(_MODAL_CLOUD_MODULE_NAME, None)
        raise
    return cloud_module


def _lookup_local_prompt_server() -> Any | None:
    """Return the live local ComfyUI PromptServer instance when available."""
    try:
        import server
    except ModuleNotFoundError:
        return None

    return getattr(server.PromptServer, "instance", None)


def _emit_local_modal_status(
    *,
    prompt_id: str | None,
    client_id: str | None,
    phase: str,
    node_ids: list[str],
    active_node_id: str | None = None,
    active_node_class_type: str | None = None,
    active_node_role: str | None = None,
    error_message: str | None = None,
) -> None:
    """Forward remote execution progress into the local ComfyUI websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload: dict[str, Any] = {
        "phase": phase,
        "prompt_id": prompt_id,
        "node_ids": list(node_ids),
    }
    if active_node_id is not None:
        payload["active_node_id"] = active_node_id
    if active_node_class_type is not None:
        payload["active_node_class_type"] = active_node_class_type
    if active_node_role is not None:
        payload["active_node_role"] = active_node_role
    if error_message is not None:
        payload["error_message"] = error_message
    prompt_server.send_sync("modal_status", payload, client_id)


def _emit_local_modal_progress(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    value: float,
    max_value: float,
    display_node_id: str | None = None,
    lane_id: str | None = None,
    clear: bool = False,
    item_index: int | None = None,
) -> None:
    """Forward remote numeric node progress into the local ComfyUI websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "node_id": node_id,
        "value": float(value),
        "max": float(max_value),
    }
    if display_node_id is not None:
        payload["display_node_id"] = display_node_id
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if clear:
        payload["clear"] = True
    if item_index is not None:
        payload["item_index"] = int(item_index)
    prompt_server.send_sync("modal_progress", payload, client_id)


def _emit_local_executed_output(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    display_node_id: str | None,
    output_payload: Any,
) -> None:
    """Forward one remote node's executed UI payload into the local websocket stream."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    payload = {
        "prompt_id": prompt_id,
        "node": node_id,
        "display_node": display_node_id or node_id,
        "output": output_payload,
    }
    prompt_server.send_sync("executed", payload, client_id)


def _emit_local_preview_image(
    *,
    prompt_id: str | None,
    client_id: str | None,
    node_id: str,
    display_node_id: str | None,
    parent_node_id: str | None,
    real_node_id: str | None,
    image_type: str,
    image_bytes: bytes,
    max_size: int | None,
) -> None:
    """Forward one remote preview image into the local ComfyUI preview websocket path."""
    if client_id is None:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    try:
        from PIL import Image
        from protocol import BinaryEventTypes
    except ModuleNotFoundError:
        logger.warning("Preview forwarding is unavailable because Pillow or ComfyUI protocol imports failed.")
        return

    with BytesIO(image_bytes) as image_buffer:
        image = Image.open(image_buffer)
        image.load()

    metadata: dict[str, Any] = {
        "node_id": node_id,
        "prompt_id": prompt_id,
        "display_node_id": display_node_id or node_id,
        "real_node_id": real_node_id or node_id,
    }
    if parent_node_id is not None:
        metadata["parent_node_id"] = parent_node_id

    prompt_server.send_sync(
        BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA,
        ((image_type, image, max_size), metadata),
        client_id,
    )


def _emit_local_preview_boundary_output(
    *,
    prompt_id: str | None,
    client_id: str | None,
    preview_target_node_ids: list[str],
    image_value: Any,
) -> None:
    """Render one streamed remote boundary IMAGE value into local PreviewImage UI events."""
    if client_id is None or not preview_target_node_ids:
        return

    prompt_server = _lookup_local_prompt_server()
    if prompt_server is None:
        return

    try:
        import nodes
    except ModuleNotFoundError:
        logger.warning("Preview boundary streaming is unavailable because ComfyUI nodes could not be imported.")
        return

    preview_factory = getattr(nodes, "PreviewImage", None)
    if preview_factory is None:
        logger.warning("Preview boundary streaming is unavailable because PreviewImage is not registered.")
        return

    preview_result = preview_factory().save_images(images=image_value)
    if not isinstance(preview_result, dict):
        return
    output_payload = preview_result.get("ui")
    if not isinstance(output_payload, dict):
        return

    for preview_target_node_id in preview_target_node_ids:
        prompt_server.send_sync(
            "executed",
            {
                "prompt_id": prompt_id,
                "node": preview_target_node_id,
                "display_node": preview_target_node_id,
                "output": output_payload,
            },
            client_id,
        )


def _allowed_suppressed_stream_node_ids(payload: dict[str, Any]) -> set[str]:
    """Return the node ids that may surface UI events for a suppressed mapped/static stream."""
    allowed_node_ids = {
        str(node_id)
        for node_id in payload.get("execute_node_ids", [])
        if str(node_id)
    }
    allowed_node_ids.update(
        str(boundary_output["node_id"])
        for boundary_output in payload.get("boundary_outputs", [])
        if boundary_output.get("node_id") is not None and str(boundary_output["node_id"])
    )
    return allowed_node_ids


def _should_forward_suppressed_stream_event(
    payload: dict[str, Any],
    reported_node_id: Any,
) -> bool:
    """Return whether a suppressed mapped/static stream event belongs to this payload."""
    if not bool(payload.get("suppress_status_stream")):
        return True
    if reported_node_id is None:
        return False
    allowed_node_ids = _allowed_suppressed_stream_node_ids(payload)
    if not allowed_node_ids:
        return True
    return str(reported_node_id) in allowed_node_ids


def _should_stream_remote_progress(payload: dict[str, Any]) -> bool:
    """Return whether the local client has enough context to mirror remote node progress."""
    extra_data = payload.get("extra_data") or {}
    return (
        payload.get("payload_kind") == "subgraph"
        and isinstance(payload.get("prompt_id"), str)
        and bool(payload.get("prompt_id"))
        and isinstance(extra_data.get("client_id"), str)
        and bool(extra_data.get("client_id"))
        and isinstance(payload.get("component_node_ids"), list)
        and len(payload.get("component_node_ids")) > 0
    )


def _local_processing_interrupted() -> bool:
    """Return whether the current local ComfyUI execution was interrupted."""
    try:
        import comfy.model_management
    except ModuleNotFoundError:
        return False

    return bool(comfy.model_management.processing_interrupted())


def _raise_local_interrupt() -> None:
    """Raise ComfyUI's native interruption exception for the current execution."""
    import comfy.model_management

    raise comfy.model_management.InterruptProcessingException()


def _remote_interrupt_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the prompt/component pair used to interrupt one remote execution."""
    prompt_id = str(payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph")
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


def _lookup_modal_interrupt_store() -> Any | None:
    """Return the shared Modal Dict used to signal remote cancellation requests."""
    if modal is None or not hasattr(modal, "Dict"):
        return None

    settings = get_settings()
    cache_key = (settings.interrupt_dict_name, _modal_environment_name())
    with _MODAL_INTERRUPT_DICTS_LOCK:
        cached_store = _MODAL_INTERRUPT_DICTS.get(cache_key)
        if cached_store is not None:
            return cached_store

    interrupt_store = modal.Dict.from_name(
        settings.interrupt_dict_name,
        environment_name=cache_key[1],
        create_if_missing=True,
    )
    with _MODAL_INTERRUPT_DICTS_LOCK:
        _MODAL_INTERRUPT_DICTS[cache_key] = interrupt_store
    return interrupt_store


def _request_remote_interrupt(payload: dict[str, Any]) -> bool:
    """Write one remote cancellation request into the shared Modal interrupt store."""
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return False

    prompt_id, component_id = _remote_interrupt_key(payload)
    interrupt_store.put(
        _remote_interrupt_flag_key(prompt_id, component_id),
        {"requested_at": time.time()},
    )
    logger.info(
        "Propagated local interrupt to Modal prompt=%s component=%s through shared control state.",
        prompt_id,
        component_id,
    )
    return True


def _invoke_remote_call_with_interrupts(
    *,
    payload: dict[str, Any],
    invoke_remote_call: Callable[[], bytes],
    interrupt_remote_call: Callable[[], Any] | None,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Run one blocking remote call while optionally propagating cancellation to Modal."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def execute_remote_call() -> None:
        """Run the blocking Modal request in a worker thread."""
        try:
            result_queue.put(("result", invoke_remote_call()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    request_thread = threading.Thread(
        target=execute_remote_call,
        name=f"modal-request-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    request_thread.start()
    interrupt_sent = False
    prompt_id, component_id = _remote_interrupt_key(payload)
    try:
        while True:
            try:
                result_kind, result_payload = result_queue.get(timeout=0.1)
            except queue.Empty:
                if (
                    cancellation_event is not None
                    and cancellation_event.is_set()
                    and not interrupt_sent
                ):
                    if interrupt_remote_call is None:
                        logger.warning(
                            "Local interrupt requested for component=%s, but no remote interrupt method is available.",
                            component_id,
                        )
                    else:
                        try:
                            interrupt_remote_call()
                            logger.info(
                                "Propagated local interrupt to Modal prompt=%s component=%s.",
                                prompt_id,
                                component_id,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to propagate local interrupt to Modal prompt=%s component=%s.",
                                prompt_id,
                                component_id,
                            )
                    interrupt_sent = True
                continue

            if result_kind == "result":
                return bytes(result_payload)
            raise result_payload
    finally:
        request_thread.join(timeout=1.0)


def _consume_remote_payload_stream(
    payload: dict[str, Any],
    stream_events: Iterator[dict[str, Any]],
) -> bytes:
    """Forward remote progress events into the local UI and return the final payload bytes."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    node_ids = [str(node_id) for node_id in payload.get("component_node_ids", [])]
    suppress_status_stream = bool(payload.get("suppress_status_stream"))
    result_payload: bytes | bytearray | None = None

    for stream_event in stream_events:
        event_kind = str(stream_event.get("kind", ""))
        if event_kind == "progress":
            event_type = str(stream_event.get("event_type", ""))
            if event_type == "node_progress":
                lane_id = (
                    str(payload["mapped_progress_lane_id"])
                    if payload.get("mapped_progress_lane_id") is not None
                    else None
                )
                if suppress_status_stream and lane_id is None:
                    continue
                reported_node_id = stream_event.get("node_id")
                if reported_node_id is not None:
                    logger.debug(
                        "Forwarding streamed Modal node progress for component=%s node_id=%s value=%s max=%s lane_id=%s.",
                        payload.get("component_id"),
                        reported_node_id,
                        stream_event.get("value"),
                        stream_event.get("max"),
                        lane_id,
                    )
                    _emit_local_modal_progress(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        node_id=str(reported_node_id),
                        value=float(stream_event.get("value", 0.0)),
                        max_value=float(stream_event.get("max", 1.0)),
                        display_node_id=(
                            str(payload["mapped_progress_display_node_id"])
                            if payload.get("mapped_progress_display_node_id") is not None
                            else (
                                str(stream_event["display_node_id"])
                                if stream_event.get("display_node_id") is not None
                                else None
                            )
                        ),
                        lane_id=lane_id,
                        item_index=(
                            int(payload["map_item_index"])
                            if payload.get("map_item_index") is not None
                            else None
                        ),
                    )
                continue
            if event_type == "executed":
                reported_node_id = stream_event.get("node_id")
                if reported_node_id is not None:
                    if not _should_forward_suppressed_stream_event(payload, reported_node_id):
                        logger.debug(
                            "Suppressing streamed Modal executed output for component=%s node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            reported_node_id,
                        )
                        continue
                    logger.debug(
                        "Forwarding streamed Modal executed output for component=%s node_id=%s.",
                        payload.get("component_id"),
                        reported_node_id,
                    )
                    _emit_local_executed_output(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        node_id=str(reported_node_id),
                        display_node_id=(
                            str(stream_event["display_node_id"])
                            if stream_event.get("display_node_id") is not None
                            else None
                        ),
                        output_payload=deserialize_value(stream_event.get("output")),
                    )
                continue
            if event_type == "preview":
                reported_node_id = stream_event.get("node_id")
                image_bytes = deserialize_value(stream_event.get("image_bytes"))
                if reported_node_id is not None and isinstance(image_bytes, bytes):
                    if not _should_forward_suppressed_stream_event(payload, reported_node_id):
                        logger.debug(
                            "Suppressing streamed Modal preview image for component=%s node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            reported_node_id,
                        )
                        continue
                    logger.debug(
                        "Forwarding streamed Modal preview image for component=%s node_id=%s.",
                        payload.get("component_id"),
                        reported_node_id,
                    )
                    _emit_local_preview_image(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        node_id=str(reported_node_id),
                        display_node_id=(
                            str(stream_event["display_node_id"])
                            if stream_event.get("display_node_id") is not None
                            else None
                        ),
                        parent_node_id=(
                            str(stream_event["parent_node_id"])
                            if stream_event.get("parent_node_id") is not None
                            else None
                        ),
                        real_node_id=(
                            str(stream_event["real_node_id"])
                            if stream_event.get("real_node_id") is not None
                            else None
                        ),
                        image_type=str(stream_event.get("image_type", "PNG")),
                        image_bytes=image_bytes,
                        max_size=(
                            int(stream_event["max_size"])
                            if stream_event.get("max_size") is not None
                            else None
                        ),
                    )
                continue
            if event_type == "boundary_output":
                preview_target_node_ids = [
                    str(node_id)
                    for node_id in stream_event.get("preview_target_node_ids", [])
                    if str(node_id)
                ]
                if preview_target_node_ids:
                    logger.debug(
                        "Forwarding streamed Modal boundary output previews for component=%s source_node=%s targets=%s.",
                        payload.get("component_id"),
                        stream_event.get("node_id"),
                        preview_target_node_ids,
                    )
                    _emit_local_preview_boundary_output(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        preview_target_node_ids=preview_target_node_ids,
                        image_value=deserialize_value(stream_event.get("value")),
                    )
                continue
            logger.info(
                "Forwarding streamed Modal progress for component=%s phase=%s active_node_id=%s.",
                payload.get("component_id"),
                stream_event.get("phase"),
                stream_event.get("active_node_id"),
            )
            if suppress_status_stream:
                continue
            _emit_local_modal_status(
                prompt_id=prompt_id,
                client_id=client_id,
                phase=str(stream_event.get("phase", "executing")),
                node_ids=node_ids,
                active_node_id=(
                    str(stream_event["active_node_id"])
                    if stream_event.get("active_node_id") is not None
                    else None
                ),
                active_node_class_type=(
                    str(stream_event["active_node_class_type"])
                    if stream_event.get("active_node_class_type") is not None
                    else None
                ),
                active_node_role=(
                    str(stream_event["active_node_role"])
                    if stream_event.get("active_node_role") is not None
                    else None
                ),
            )
            continue
        if event_kind == "result":
            candidate_outputs = stream_event.get("outputs")
            try:
                result_payload = coerce_serialized_node_outputs(candidate_outputs)
            except TypeError as exc:
                raise ModalRemoteInvocationError(
                    "Modal streamed payload result did not include transport-safe outputs."
                ) from exc
            continue
        logger.debug(
            "Ignoring unexpected streamed Modal event kind=%s for component=%s.",
            event_kind,
            payload.get("component_id"),
        )

    if result_payload is None:
        raise ModalRemoteInvocationError(
            f"Modal streamed payload for component={payload.get('component_id')!r} did not yield a final result."
        )
    return bytes(result_payload)


def _mapped_execution_parallelism(item_count: int) -> int:
    """Return the local worker width used to schedule mapped Modal item executions."""
    settings = get_settings()
    configured_limit = settings.max_containers or _remote_modal_call_worker_count()
    return max(1, min(item_count, configured_limit))


def _build_mapped_item_payload(
    payload: dict[str, Any],
    item_index: int,
    lane_index: int,
) -> dict[str, Any]:
    """Return one per-item subgraph payload derived from a mapped remote component payload."""
    item_payload = copy.deepcopy(payload)
    item_payload["payload_kind"] = "subgraph"
    item_payload["component_id"] = f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}"
    item_payload["mapped_input"] = None
    item_payload["suppress_status_stream"] = True
    item_payload["map_item_index"] = item_index
    item_payload["mapped_progress_lane_id"] = str(lane_index)
    item_payload["mapped_progress_display_node_id"] = str(
        payload.get("component_id", "modal-subgraph")
    )
    item_payload["execute_node_ids"] = list(
        payload.get("mapped_execute_node_ids") or payload.get("execute_node_ids", [])
    )
    item_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in payload.get("boundary_outputs", [])
        if _is_mapped_boundary_output(boundary_output, payload)
    ]
    return item_payload


def _aggregate_mapped_outputs(
    per_item_outputs: list[tuple[Any, ...]],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble ordered per-item outputs from mapped execution into one proxy result tuple."""
    if not per_item_outputs:
        raise ValueError("Mapped execution produced no per-item outputs to aggregate.")

    output_count = len(per_item_outputs[0])
    if any(len(item_outputs) != output_count for item_outputs in per_item_outputs):
        raise RemoteSubgraphExecutionError("Mapped remote execution produced inconsistent output arity.")

    boundary_outputs = list(payload.get("boundary_outputs", []))
    aggregated_outputs: list[Any] = []
    for output_index in range(output_count):
        boundary_output = boundary_outputs[output_index] if output_index < len(boundary_outputs) else {}
        aggregated_outputs.append(
            join_mapped_values(
                [item_outputs[output_index] for item_outputs in per_item_outputs],
                io_type=str(boundary_output.get("io_type", "*")),
                is_list=bool(boundary_output.get("is_list", False)),
            )
        )
    return tuple(aggregated_outputs)


def _is_mapped_boundary_output(boundary_output: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Return whether one boundary output belongs to the mapped per-item branch."""
    mapped_output = boundary_output.get("mapped_output")
    if mapped_output is not None:
        return bool(mapped_output)
    return bool(payload.get("mapped_input")) and not bool(payload.get("static_execute_node_ids"))


def _build_static_mapped_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the one-time static subgraph payload for a hybrid mapped component."""
    static_payload = copy.deepcopy(payload)
    static_payload["payload_kind"] = "subgraph"
    static_payload["component_id"] = f"{payload.get('component_id', 'modal-subgraph')}::static"
    static_payload["mapped_input"] = None
    static_payload["suppress_status_stream"] = True
    static_payload["execute_node_ids"] = list(payload.get("static_execute_node_ids") or [])
    static_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in payload.get("boundary_outputs", [])
        if not _is_mapped_boundary_output(boundary_output, payload)
    ]
    return static_payload


def _merge_static_and_mapped_outputs(
    *,
    static_outputs: tuple[Any, ...],
    mapped_outputs: tuple[Any, ...],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble one hybrid mapped component's static and mapped outputs in original order."""
    combined_outputs: list[Any] = []
    static_output_index = 0
    mapped_output_index = 0

    for boundary_output in payload.get("boundary_outputs", []):
        if _is_mapped_boundary_output(boundary_output, payload):
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

    if static_output_index != len(static_outputs) or mapped_output_index != len(mapped_outputs):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced extra outputs that did not match the declared boundary outputs."
        )
    return tuple(combined_outputs)


def _emit_local_mapped_progress(
    payload: dict[str, Any],
    completed_items: int,
    total_items: int,
) -> None:
    """Emit one aggregate mapped-execution progress update for the component representative node."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    display_node_id = str(payload.get("component_id") or "")
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=float(completed_items),
        max_value=float(total_items),
        display_node_id=display_node_id,
    )


def _clear_local_mapped_lane_progress(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int,
) -> None:
    """Remove one mapped worker lane from the local node overlay."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    display_node_id = str(payload.get("component_id") or "")
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=str(lane_index),
        clear=True,
        item_index=item_index,
    )


def _emit_local_mapped_lane_progress_start(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int,
) -> None:
    """Create or reset one mapped worker lane before remote progress begins arriving."""
    prompt_id = str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    extra_data = payload.get("extra_data") or {}
    client_id = str(extra_data.get("client_id")) if extra_data.get("client_id") is not None else None
    display_node_id = str(payload.get("component_id") or "")
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=str(lane_index),
        item_index=item_index,
    )


async def _invoke_mapped_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Split one mapped boundary input, run per-item remote executions, and reassemble outputs."""
    mapped_input = payload.get("mapped_input") or {}
    mapped_input_name = str(mapped_input.get("proxy_input_name") or "")
    if not mapped_input_name:
        raise ModalRemoteInvocationError("Mapped remote payloads must define mapped_input.proxy_input_name.")

    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    if mapped_input_name not in hydrated_inputs:
        raise KeyError(f"Mapped remote payload input {mapped_input_name!r} was not provided.")

    mapped_items = split_mapped_value(
        hydrated_inputs[mapped_input_name],
        str(mapped_input.get("io_type", "*")),
    )
    if not mapped_items:
        raise ValueError("Mapped remote execution requires at least one input item.")

    broadcast_inputs = dict(hydrated_inputs)
    broadcast_inputs.pop(mapped_input_name, None)
    total_items = len(mapped_items)
    parallelism = _mapped_execution_parallelism(total_items)
    logger.info(
        "Scheduling mapped Modal component=%s for %d item(s) with local parallelism=%d.",
        payload.get("component_id"),
        total_items,
        parallelism,
    )
    _emit_local_mapped_progress(payload, 0, total_items)

    static_outputs: tuple[Any, ...] = ()
    if payload.get("static_execute_node_ids"):
        static_response = await invoke_remote_engine_async(
            _build_static_mapped_payload(payload),
            kwargs_payload,
        )
        static_outputs = deserialize_node_outputs(static_response)

    per_item_outputs: list[tuple[Any, ...] | None] = [None] * total_items
    completed_items = 0
    item_queue: asyncio.Queue[tuple[int, Any] | None] = asyncio.Queue()
    for item_index, item_value in enumerate(mapped_items):
        item_queue.put_nowait((item_index, item_value))
    for _ in range(parallelism):
        item_queue.put_nowait(None)

    async def run_worker(lane_index: int) -> None:
        """Execute queued mapped items through one stable local worker lane."""
        nonlocal completed_items
        while True:
            queued_item = await item_queue.get()
            if queued_item is None:
                return
            item_index, item_value = queued_item
            if _local_processing_interrupted():
                _raise_local_interrupt()
            _emit_local_mapped_lane_progress_start(payload, lane_index, item_index)
            try:
                item_payload = _build_mapped_item_payload(payload, item_index, lane_index)
                item_inputs = dict(broadcast_inputs)
                item_inputs[mapped_input_name] = item_value
                item_response = await invoke_remote_engine_async(
                    item_payload,
                    serialize_node_inputs(item_inputs),
                )
                per_item_outputs[item_index] = deserialize_node_outputs(item_response)
                completed_items += 1
                _emit_local_mapped_progress(payload, completed_items, total_items)
            finally:
                _clear_local_mapped_lane_progress(payload, lane_index, item_index)

    tasks = [asyncio.create_task(run_worker(lane_index)) for lane_index in range(parallelism)]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return serialize_node_outputs(
        _merge_static_and_mapped_outputs(
            static_outputs=static_outputs,
            mapped_outputs=_aggregate_mapped_outputs(
                [item_outputs for item_outputs in per_item_outputs if item_outputs is not None],
                {
                    **payload,
                    "boundary_outputs": [
                        boundary_output
                        for boundary_output in payload.get("boundary_outputs", [])
                        if _is_mapped_boundary_output(boundary_output, payload)
                    ],
                },
            ),
            payload=payload,
        )
    )


def _lookup_deployed_remote_engine(payload: dict[str, Any]) -> Any:
    """Look up the deployed Modal runtime class instance."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    settings = get_settings()
    logger.info(
        "Attempting deployed Modal invocation for app=%s class=%s component=%s.",
        settings.app_name,
        "RemoteEngine",
        payload.get("component_id"),
    )
    remote_cls = modal.Cls.from_name(settings.app_name, "RemoteEngine")
    return remote_cls()


def _modal_environment_name() -> str | None:
    """Return the active Modal environment name when explicitly configured."""
    environment_name = os.getenv("MODAL_ENVIRONMENT")
    if environment_name is None:
        return None
    normalized = environment_name.strip()
    return normalized or None


def _modal_deploy_cache_key() -> tuple[str, str | None]:
    """Return the cache key for auto-deployed Modal apps."""
    settings = get_settings()
    return (settings.app_name, _modal_environment_name())


def _auto_deploy_modal_app(payload: dict[str, Any], lookup_error: BaseException) -> None:
    """Deploy the stable Modal cloud app once when deployed lookup fails."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    settings = get_settings()
    deploy_key = _modal_deploy_cache_key()
    cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    if cloud_app is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose a deployable app."
        )

    with _MODAL_AUTO_DEPLOY_LOCK:
        if deploy_key in _MODAL_AUTO_DEPLOYED_APPS:
            logger.info(
                "Auto-deploy already completed for app=%s env=%s; reusing cached deployment state.",
                settings.app_name,
                deploy_key[1] or "<default>",
            )
            return

        logger.warning(
            "Deployed Modal app lookup failed for app=%s component=%s: %s. "
            "Attempting first-run auto-deploy from the custom node.",
            settings.app_name,
            payload.get("component_id"),
            lookup_error,
        )
        output_context = modal.enable_output() if hasattr(modal, "enable_output") else nullcontext()
        deploy_started_at = time.perf_counter()
        with output_context:
            cloud_app.deploy(
                name=settings.app_name,
                environment_name=deploy_key[1],
            )
        _MODAL_AUTO_DEPLOYED_APPS.add(deploy_key)
        logger.info(
            "Auto-deployed Modal app %s for env=%s in %.3fs.",
            settings.app_name,
            deploy_key[1] or "<default>",
            time.perf_counter() - deploy_started_at,
        )


def _build_remote_interrupt_callback(_remote_engine: Any, payload: dict[str, Any]) -> Callable[[], Any] | None:
    """Return a callable that requests interruption for one active Modal payload."""
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return None
    return lambda: _request_remote_interrupt(payload)


def _invoke_remote_engine_payload(
    remote_engine: Any,
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Invoke one prepared remote engine instance with optional progress streaming."""
    stream_method = getattr(remote_engine, "execute_payload_stream", None)
    interrupt_remote_call = _build_remote_interrupt_callback(remote_engine, payload)
    if _should_stream_remote_progress(payload) and hasattr(stream_method, "remote_gen"):
        logger.info(
            "Using streamed Modal progress path for component=%s via execute_payload_stream.remote_gen(...).",
            payload.get("component_id"),
        )
        return _invoke_remote_call_with_interrupts(
            payload=payload,
            invoke_remote_call=lambda: _consume_remote_payload_stream(
                payload,
                stream_method.remote_gen(payload, kwargs_payload),
            ),
            interrupt_remote_call=interrupt_remote_call,
            cancellation_event=cancellation_event,
        )

    if _should_stream_remote_progress(payload):
        logger.warning(
            "Streamed Modal progress is unavailable for component=%s; falling back to execute_payload.remote(...).",
            payload.get("component_id"),
        )
    return _invoke_remote_call_with_interrupts(
        payload=payload,
        invoke_remote_call=lambda: remote_engine.execute_payload.remote(payload, kwargs_payload),
        interrupt_remote_call=interrupt_remote_call,
        cancellation_event=cancellation_event,
    )


def _invoke_modal_payload_blocking(
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None = None,
) -> bytes:
    """Invoke the Modal runtime from a worker thread using deployed or ephemeral app state."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    lookup_error_types = _modal_lookup_error_types()
    settings = get_settings()
    if lookup_error_types:
        try:
            remote_engine = _lookup_deployed_remote_engine(payload)
            logger.info(
                "Using deployed Modal app %s for component %s.",
                settings.app_name,
                payload.get("component_id"),
            )
            return _invoke_remote_engine_payload(
                remote_engine,
                payload,
                kwargs_payload,
                cancellation_event,
            )
        except lookup_error_types as exc:
            if settings.auto_deploy:
                _auto_deploy_modal_app(payload, exc)
                try:
                    remote_engine = _lookup_deployed_remote_engine(payload)
                    logger.info(
                        "Using auto-deployed Modal app %s for component %s.",
                        settings.app_name,
                        payload.get("component_id"),
                    )
                    return _invoke_remote_engine_payload(
                        remote_engine,
                        payload,
                        kwargs_payload,
                        cancellation_event,
                    )
                except lookup_error_types as retry_exc:
                    exc = retry_exc
            if not settings.allow_ephemeral_fallback:
                raise ModalRemoteInvocationError(
                    "Remote execution requires a deployed Modal app or a successful first-run auto-deploy. "
                    f"Lookup failed for app={settings.app_name!r} component={payload.get('component_id')!r}: {exc}. "
                    "Ensure Modal credentials are configured so the custom node can auto-deploy, "
                    "or set COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true to allow slow ephemeral app.run() fallback behavior."
                ) from exc
            logger.warning(
                "Deployed Modal app lookup failed for app=%s component=%s: %s. Falling back to ephemeral app.run(); this creates a temporary Modal app session, not a persistent deployment or endpoint.",
                settings.app_name,
                payload.get("component_id"),
                exc,
            )
    else:
        remote_engine = _lookup_deployed_remote_engine(payload)
        logger.info(
            "Using deployed Modal app %s for component %s.",
            settings.app_name,
            payload.get("component_id"),
        )
        return _invoke_remote_engine_payload(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )

    if "app" not in globals() or "RemoteEngine" not in globals():
        logger.debug("Local module Modal runtime objects are unavailable; loading stable cloud entry module.")

    cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    cloud_remote_engine = getattr(cloud_module, "RemoteEngine", None)
    if cloud_app is None or cloud_remote_engine is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose app and RemoteEngine."
        )
    logger.info(
        "Starting ephemeral Modal app.run() for component %s. This does not create a persistent deployed app or web endpoint.",
        payload.get("component_id"),
    )
    run_context = cloud_app.run() if hasattr(cloud_app, "run") else nullcontext()
    with run_context:
        remote_engine = cloud_remote_engine()
        result = _invoke_remote_engine_payload(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )
    logger.info(
        "Ephemeral Modal app.run() invocation completed for component %s.",
        payload.get("component_id"),
    )
    return result


def invoke_remote_engine(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Invoke Modal when configured, or fall back to local in-process execution."""
    if payload.get("payload_kind") == "mapped_subgraph":
        return asyncio.run(_invoke_mapped_remote_engine_async(payload, kwargs_payload))

    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    if execution_mode == "local" or modal is None:
        if payload.get("payload_kind") == "subgraph":
            return execute_subgraph_locally(payload, kwargs_payload)
        return execute_node_locally(payload, kwargs_payload)

    logger.info(
        "Dispatching Modal remote invocation for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    cancellation_event = threading.Event()
    future = _REMOTE_MODAL_CALL_EXECUTOR.submit(
        _invoke_modal_payload_blocking,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    try:
        while True:
            try:
                response = future.result(timeout=0.1)
                break
            except FutureTimeoutError:
                if _local_processing_interrupted() and not cancellation_event.is_set():
                    logger.info(
                        "Observed local interrupt while Modal component=%s was running; requesting remote cancellation.",
                        payload.get("component_id"),
                    )
                    cancellation_event.set()
                continue
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Modal remote invocation failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Remote invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Modal remote invocation completed for component=%s.",
        payload.get("component_id"),
    )
    return response


async def invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Invoke Modal asynchronously so multiple proxy nodes can wait on remote work in parallel."""
    if payload.get("payload_kind") == "mapped_subgraph":
        return await _invoke_mapped_remote_engine_async(payload, kwargs_payload)

    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    if execution_mode == "local" or modal is None:
        return await asyncio.to_thread(invoke_remote_engine, payload, kwargs_payload)

    logger.info(
        "Dispatching async Modal remote invocation for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    cancellation_event = threading.Event()
    future = _REMOTE_MODAL_CALL_EXECUTOR.submit(
        _invoke_modal_payload_blocking,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    wrapped_future = asyncio.wrap_future(future)
    try:
        while True:
            try:
                response = await asyncio.wait_for(asyncio.shield(wrapped_future), timeout=0.1)
                break
            except asyncio.TimeoutError:
                if _local_processing_interrupted() and not cancellation_event.is_set():
                    logger.info(
                        "Observed local interrupt while async Modal component=%s was running; requesting remote cancellation.",
                        payload.get("component_id"),
                    )
                    cancellation_event.set()
                continue
    except asyncio.CancelledError:
        cancellation_event.set()
        raise
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising async Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Async Modal remote invocation failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Async remote invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Async Modal remote invocation completed for component=%s.",
        payload.get("component_id"),
    )
    return response


if modal is not None:  # pragma: no branch - simple import-time configuration.
    settings = get_settings()
    app = modal.App(settings.app_name)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    image = modal.Image.debian_slim().pip_install("torch", "safetensors", "pillow", "numpy")

    @app.cls(
        gpu="A100",
        volumes={settings.remote_storage_root: vol},
        scaledown_window=60,
        image=image,
    )
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""

        @modal.enter()
        def setup(self) -> None:
            """Prepare the container process for headless node execution."""
            logger.info("RemoteEngine setup complete.")

        @modal.method()
        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)

else:

    class RemoteEngine:
        """Local fallback runtime used when the Modal SDK is unavailable."""

        def setup(self) -> None:
            """No-op setup for local fallback execution."""

        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute the proxied node or subgraph locally."""
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)
