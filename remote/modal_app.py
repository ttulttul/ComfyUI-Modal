"""Remote Modal runtime and local execution fallback."""

from __future__ import annotations

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
    coerce_serialized_node_outputs,
    deserialize_value,
    deserialize_node_inputs,
    serialize_node_outputs,
)
from ..settings import get_settings

logger = logging.getLogger(__name__)
_REMOTE_MODAL_CALL_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_MODAL_CLOUD_MODULE_NAME = "comfyui_modal_sync_cloud"
_MODAL_AUTO_DEPLOY_LOCK = threading.Lock()
_MODAL_AUTO_DEPLOYED_APPS: set[tuple[str, str | None]] = set()

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
    if _MODAL_CLOUD_MODULE_NAME in sys.modules:
        return sys.modules[_MODAL_CLOUD_MODULE_NAME]

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
    module_spec.loader.exec_module(cloud_module)
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
    result_payload: bytes | bytearray | None = None

    for stream_event in stream_events:
        event_kind = str(stream_event.get("kind", ""))
        if event_kind == "progress":
            event_type = str(stream_event.get("event_type", ""))
            if event_type == "node_progress":
                reported_node_id = stream_event.get("node_id")
                if reported_node_id is not None:
                    logger.debug(
                        "Forwarding streamed Modal node progress for component=%s node_id=%s value=%s max=%s.",
                        payload.get("component_id"),
                        reported_node_id,
                        stream_event.get("value"),
                        stream_event.get("max"),
                    )
                    _emit_local_modal_progress(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        node_id=str(reported_node_id),
                        value=float(stream_event.get("value", 0.0)),
                        max_value=float(stream_event.get("max", 1.0)),
                        display_node_id=(
                            str(stream_event["display_node_id"])
                            if stream_event.get("display_node_id") is not None
                            else None
                        ),
                    )
                continue
            if event_type == "executed":
                reported_node_id = stream_event.get("node_id")
                if reported_node_id is not None:
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


def _build_remote_interrupt_callback(remote_engine: Any, payload: dict[str, Any]) -> Callable[[], Any] | None:
    """Return a callable that interrupts one active Modal payload when supported."""
    interrupt_method = getattr(remote_engine, "interrupt_payload", None)
    if interrupt_method is None or not hasattr(interrupt_method, "remote"):
        return None

    prompt_id, component_id = _remote_interrupt_key(payload)
    return lambda: interrupt_method.remote(prompt_id, component_id)


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
