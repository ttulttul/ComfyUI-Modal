"""Headless local node and subgraph execution for remote payload fallback."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Mapping
import zipfile

from ..remote_protocol import (
    BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
    PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
)
from ..serialization import (
    deserialize_node_inputs,
    serialize_node_outputs,
    unwrap_mapped_output_value,
)
from ..session_state import (
    RemoteSessionBridgeRef,
    RemoteSessionHandle,
    RemoteSessionStateError,
    RemoteSessionValueRef,
)
from ..settings import get_settings
from . import host_session_bridge as _host_session_bridge
from .host_session_bridge import (
    _RemoteSessionBridgeResolutionStats,
    _build_remote_session_bridge_record,
    _log_remote_session_resolution_summary,
    _payload_remote_session_handle,
    _resolve_remote_session_inputs,
    _store_remote_session_bridge_value,
)

logger = logging.getLogger(__name__)



class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""

class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None

    def send_sync(
        self, event: str, data: dict[str, Any], client_id: str | None
    ) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug(
            "Suppressed remote prompt event %s for client %s.", event, client_id
        )


def _extract_custom_nodes_bundle(bundle_path: str | None) -> None:
    """Extract a mirrored custom_nodes bundle ZIP or manifest into a temporary import path."""
    if not bundle_path:
        return

    settings = get_settings()
    if settings.execution_mode == "local":
        logger.debug("Skipping custom_nodes bundle extraction in local execution mode.")
        return

    local_bundle = settings.local_storage_root / bundle_path.lstrip("/")
    if not local_bundle.exists():
        logger.warning(
            "Custom nodes bundle %s was not found in local storage.", local_bundle
        )
        return

    extraction_root = (
        Path(tempfile.gettempdir())
        / "comfy-modal-sync-custom-nodes"
        / local_bundle.stem
    )
    extraction_root.mkdir(parents=True, exist_ok=True)
    for archive_path in _resolve_local_custom_nodes_archives(
        local_bundle, settings.local_storage_root
    ):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extraction_root)
    _materialize_local_custom_node_assets(
        local_bundle,
        settings.local_storage_root,
        extraction_root,
    )

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)


def _resolve_local_custom_nodes_archives(
    local_bundle: Path, storage_root: Path
) -> list[Path]:
    """Return the archive paths described by one local custom_nodes bundle ZIP or manifest."""
    if local_bundle.suffix.lower() == ".zip":
        return [local_bundle]
    if local_bundle.suffix.lower() != ".json":
        raise RuntimeError(
            f"Unsupported custom_nodes bundle format {local_bundle.suffix!r} for {local_bundle}."
        )

    manifest_payload = _load_local_custom_nodes_manifest(local_bundle)
    entry_payloads = manifest_payload.get("entries")
    if not isinstance(entry_payloads, list):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} did not contain a valid entries list."
        )

    archive_paths: list[Path] = []
    for entry_payload in entry_payloads:
        if not isinstance(entry_payload, dict):
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contained a non-object entry."
            )
        remote_path = entry_payload.get("remote_path")
        if not isinstance(remote_path, str) or not remote_path.strip():
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contained an entry without remote_path."
            )
        archive_path = storage_root / remote_path.lstrip("/")
        if not archive_path.exists():
            raise RuntimeError(
                f"Custom nodes archive {remote_path} referenced by {local_bundle} was not found in local storage."
            )
        archive_paths.append(archive_path)
    return archive_paths


def _load_local_custom_nodes_manifest(local_bundle: Path) -> dict[str, Any]:
    """Load and validate one local custom-node bundle manifest."""
    try:
        manifest_payload = json.loads(local_bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} is unreadable."
        ) from exc
    if not isinstance(manifest_payload, dict):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} must be a JSON object."
        )
    if manifest_payload.get("version", 1) not in {1, 2}:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} has an unsupported version."
        )
    return manifest_payload


def _materialize_local_custom_node_assets(
    local_bundle: Path,
    storage_root: Path,
    extraction_root: Path,
) -> None:
    """Link version-two package assets into the local fallback extraction tree."""
    if local_bundle.suffix.lower() != ".json":
        return
    manifest_payload = _load_local_custom_nodes_manifest(local_bundle)
    if manifest_payload.get("version", 1) < 2:
        return
    for asset_payload in _iter_local_custom_node_assets(local_bundle, manifest_payload):
        relative_path = _validated_local_custom_node_asset_path(
            local_bundle, asset_payload
        )
        source_path = storage_root / str(asset_payload["remote_path"]).lstrip("/")
        if not source_path.is_file():
            raise RuntimeError(
                f"Custom-node asset {source_path} was not found in local storage."
            )
        if source_path.stat().st_size != int(asset_payload["size_bytes"]):
            raise RuntimeError(
                f"Custom-node asset {source_path} size did not match its manifest."
            )
        destination = extraction_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if (
                destination.is_symlink()
                and destination.resolve() == source_path.resolve()
            ):
                continue
            raise RuntimeError(
                f"Custom-node asset destination {destination} already exists."
            )
        destination.symlink_to(source_path)


def _iter_local_custom_node_assets(
    local_bundle: Path,
    manifest_payload: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield asset objects from one validated version-two local manifest."""
    entries = manifest_payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} has no valid entries list."
        )
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("assets", []), list):
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contains invalid assets."
            )
        for asset_payload in entry.get("assets", []):
            if not isinstance(asset_payload, dict):
                raise RuntimeError(
                    f"Custom nodes manifest {local_bundle} contains an invalid asset."
                )
            yield asset_payload


def _validated_local_custom_node_asset_path(
    local_bundle: Path,
    asset_payload: Mapping[str, Any],
) -> Path:
    """Return one safe local extraction-relative package asset path."""
    relative_path = asset_payload.get("relative_path")
    remote_path = asset_payload.get("remote_path")
    sha256 = asset_payload.get("sha256")
    size_bytes = asset_payload.get("size_bytes")
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an asset without a path."
        )
    if not isinstance(remote_path, str) or not remote_path:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an asset without storage."
        )
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or not Path(remote_path).name.startswith(f"{sha256}_")
    ):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an invalid asset digest."
        )
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an invalid asset size."
        )
    normalized_path = Path(relative_path)
    if normalized_path.is_absolute() or ".." in normalized_path.parts:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an unsafe asset path."
        )
    return normalized_path


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _load_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    import execution

    return execution


@contextmanager
def _temporary_node_mapping(
    node_mapping: dict[str, type[Any]] | None
) -> Iterator[None]:
    """Temporarily overlay node mappings for tests or custom runtimes."""
    if node_mapping is None:
        yield
        return

    nodes_module = _load_nodes_module()
    original_mappings = dict(nodes_module.NODE_CLASS_MAPPINGS)
    original_display_mappings = dict(
        getattr(nodes_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
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
    outputs = _execute_node_locally_raw(
        node_data,
        kwargs_payload,
        node_mapping=node_mapping,
    )
    return serialize_node_outputs(outputs)


def _execute_node_locally_raw(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> tuple[Any, ...]:
    """Execute a single target node in-process and return raw node outputs."""
    _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    kwargs = deserialize_node_inputs(kwargs_payload)
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        return _invoke_original_node(node_mapping[class_type], node_data, kwargs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        return _invoke_original_node(
            resolved_node_mapping[class_type], node_data, kwargs
        )


def _apply_boundary_inputs(
    prompt: dict[str, Any],
    boundary_input_specs: list[dict[str, Any]],
    hydrated_inputs: dict[str, Any],
) -> None:
    """Inject hydrated local boundary inputs into a remote subgraph prompt."""
    logger.info(
        "Applying %d hydrated boundary inputs to remote subgraph prompt.",
        len(boundary_input_specs),
    )
    for boundary_input in boundary_input_specs:
        proxy_input_name = str(boundary_input["proxy_input_name"])
        if proxy_input_name not in hydrated_inputs:
            raise KeyError(f"Missing hydrated boundary input {proxy_input_name!r}.")
        value = unwrap_mapped_output_value(hydrated_inputs[proxy_input_name])
        io_type = (
            str(boundary_input["io_type"])
            if boundary_input.get("io_type") is not None
            else None
        )
        logger.info(
            "Applying boundary input %s to %d targets.",
            proxy_input_name,
            len(boundary_input.get("targets", [])),
        )
        for target in boundary_input.get("targets", []):
            node_id = str(target["node_id"])
            input_name = str(target["input_name"])
            prompt_node = prompt[node_id]
            prompt_node["inputs"][input_name] = _normalize_prompt_input_value(
                value,
                io_type=io_type,
            )
            source_signature = boundary_input.get("source_signature")
            if isinstance(source_signature, str) and source_signature:
                boundary_signatures = prompt_node.setdefault(
                    _BOUNDARY_INPUT_SIGNATURES_KEY, {}
                )
                if isinstance(boundary_signatures, dict):
                    boundary_signatures[input_name] = source_signature


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


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _iter_prompt_links(value: Any) -> Iterator[list[Any]]:
    """Yield ComfyUI prompt links found inside a prompt input value."""
    if _is_link(value):
        yield value
        return
    if isinstance(value, Mapping):
        for nested_value in value.values():
            yield from _iter_prompt_links(nested_value)
        return
    if isinstance(value, list):
        for nested_value in value:
            yield from _iter_prompt_links(nested_value)


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


def _execute_subgraph_with_mapping(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> tuple[Any, ...]:
    """Execute a rewritten remote component using an explicit node mapping."""
    normalized_payload = _trim_subgraph_payload_to_required_nodes(
        _normalize_subgraph_payload(payload)
    )
    session_handle = _payload_remote_session_handle(normalized_payload)
    resolution_stats = _RemoteSessionBridgeResolutionStats()
    resolved_inputs = _resolve_remote_session_inputs(
        dict(hydrated_inputs),
        component_id=str(payload.get("component_id") or ""),
        target_session_handle=session_handle,
        node_mapping=node_mapping,
        resolution_stats=resolution_stats,
    )
    _log_remote_session_resolution_summary(
        component_id=str(payload.get("component_id") or "modal-subgraph"),
        resolution_stats=resolution_stats,
    )
    short_circuit_outputs = _short_circuit_restored_session_output_subgraph(
        payload=normalized_payload,
        hydrated_inputs=hydrated_inputs,
        session_handle=session_handle,
        resolution_stats=resolution_stats,
    )
    if short_circuit_outputs is not None:
        logger.info(
            "Skipping mapped-node execution for component=%s because all %d session-backed outputs were restored into session_id=%s.",
            payload.get("component_id"),
            len(short_circuit_outputs),
            session_handle.session_id if session_handle is not None else None,
        )
        return short_circuit_outputs
    if session_handle is not None:
        logger.info(
            "Executing mapped remote subgraph %s with remote_session session_id=%s prompt_id=%s owner_component_id=%s.",
            payload.get("component_id"),
            session_handle.session_id,
            session_handle.prompt_id,
            session_handle.owner_component_id,
        )
    prompt = copy.deepcopy(normalized_payload["subgraph_prompt"])
    logger.info(
        "Executing remote subgraph %s via test mapping with %d prompt nodes.",
        payload.get("component_id"),
        len(prompt),
    )
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
        hydrated_inputs=resolved_inputs,
    )
    _coerce_prompt_primitive_input_values(prompt, node_mapping)
    _validate_prompt_input_shapes(
        prompt,
        node_mapping,
        list(normalized_payload.get("boundary_inputs", [])),
    )
    _validate_required_prompt_inputs(prompt, node_mapping)
    required_node_ids = _resolve_required_subgraph_nodes(
        prompt=prompt,
        execute_node_ids=list(normalized_payload.get("execute_node_ids", [])),
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
                    kwargs[str(input_name)] = executed_outputs[upstream_node_id][
                        int(input_value[1])
                    ]
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
    for boundary_output in normalized_payload.get("boundary_outputs", []):
        node_id = str(boundary_output["node_id"])
        output_index = int(boundary_output["output_index"])
        node_outputs = executed_outputs.get(node_id)
        if node_outputs is None:
            raise RemoteSubgraphExecutionError(
                f"Remote subgraph did not execute boundary output node {node_id}."
            )
        output_value = node_outputs[output_index]
        if bool(boundary_output.get("session_output")):
            if session_handle is None:
                raise RemoteSessionStateError(
                    "Session-backed boundary outputs require payload.remote_session."
                )
            bridge_record = _build_remote_session_bridge_record(
                payload=normalized_payload,
                hydrated_inputs=hydrated_inputs,
                node_id=node_id,
                output_index=output_index,
                io_type=str(boundary_output.get("io_type") or "*"),
                output_value=output_value,
            )
            _host_session_bridge._REMOTE_SESSION_BRIDGE_STORE.put_record(bridge_record)
            _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
            live_ref = _host_session_bridge._REMOTE_SESSION_STORE.put_bridge_output(
                session_handle,
                bridge_key=bridge_record.bridge_key,
                node_id=node_id,
                output_index=output_index,
                value=output_value,
            )
            output_value = RemoteSessionBridgeRef(
                bridge_key=bridge_record.bridge_key,
                node_id=node_id,
                output_index=output_index,
                session_id=live_ref.session_id,
            ).to_payload()
        outputs.append(output_value)
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

    normalized_payload = _trim_subgraph_payload_to_required_nodes(
        _normalize_subgraph_payload(payload)
    )
    session_handle = _payload_remote_session_handle(normalized_payload)
    resolution_stats = _RemoteSessionBridgeResolutionStats()
    resolved_inputs = _resolve_remote_session_inputs(
        dict(hydrated_inputs),
        component_id=str(payload.get("component_id") or ""),
        target_session_handle=session_handle,
        node_mapping=node_mapping,
        resolution_stats=resolution_stats,
    )
    _log_remote_session_resolution_summary(
        component_id=str(payload.get("component_id") or "modal-subgraph"),
        resolution_stats=resolution_stats,
    )
    short_circuit_outputs = _short_circuit_restored_session_output_subgraph(
        payload=normalized_payload,
        hydrated_inputs=hydrated_inputs,
        session_handle=session_handle,
        resolution_stats=resolution_stats,
    )
    if short_circuit_outputs is not None:
        logger.info(
            "Skipping PromptExecutor execution for component=%s because all %d session-backed outputs were restored into session_id=%s.",
            payload.get("component_id"),
            len(short_circuit_outputs),
            session_handle.session_id if session_handle is not None else None,
        )
        return short_circuit_outputs
    if session_handle is not None:
        logger.info(
            "Executing PromptExecutor remote subgraph %s with remote_session session_id=%s prompt_id=%s owner_component_id=%s.",
            payload.get("component_id"),
            session_handle.session_id,
            session_handle.prompt_id,
            session_handle.owner_component_id,
        )
    prompt = copy.deepcopy(normalized_payload["subgraph_prompt"])
    logger.info(
        "Executing remote subgraph %s through PromptExecutor with %d prompt nodes, %d boundary inputs, and %d exported outputs.",
        payload.get("component_id"),
        len(prompt),
        len(normalized_payload.get("boundary_inputs", [])),
        len(normalized_payload.get("boundary_outputs", [])),
    )
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
        hydrated_inputs=resolved_inputs,
    )
    execution = _load_execution_module()
    resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    _coerce_prompt_primitive_input_values(prompt, resolved_node_mapping)
    _validate_prompt_input_shapes(
        prompt,
        resolved_node_mapping,
        list(normalized_payload.get("boundary_inputs", [])),
    )
    _validate_required_prompt_inputs(prompt, resolved_node_mapping)

    with _temporary_node_mapping(node_mapping):
        executor = execution.PromptExecutor(_NullPromptServer())
        execution_started_at = time.perf_counter()
        logger.info(
            "Starting PromptExecutor for remote subgraph %s with execute targets %s.",
            payload.get("component_id"),
            normalized_payload.get("execute_node_ids", []),
        )
        executor.execute(
            prompt=prompt,
            prompt_id=str(
                payload.get("prompt_id")
                or payload.get("component_id", "modal-subgraph")
            ),
            extra_data=copy.deepcopy(normalized_payload.get("extra_data") or {}),
            execute_outputs=list(normalized_payload.get("execute_node_ids", [])),
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
        for boundary_output in normalized_payload.get("boundary_outputs", []):
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
            output_value = _collapse_cache_slot(
                slot_values=cache_entry.outputs[output_index],
                is_list=bool(boundary_output.get("is_list", False)),
            )
            if bool(boundary_output.get("session_output")):
                if session_handle is None:
                    raise RemoteSessionStateError(
                        "Session-backed boundary outputs require payload.remote_session."
                    )
                bridge_record = _build_remote_session_bridge_record(
                    payload=normalized_payload,
                    hydrated_inputs=hydrated_inputs,
                    node_id=node_id,
                    output_index=output_index,
                    io_type=str(boundary_output.get("io_type") or "*"),
                    output_value=output_value,
                )
                _host_session_bridge._REMOTE_SESSION_BRIDGE_STORE.put_record(bridge_record)
                _store_remote_session_bridge_value(
                    bridge_record.bridge_key, output_value
                )
                live_ref = _host_session_bridge._REMOTE_SESSION_STORE.put_bridge_output(
                    session_handle,
                    bridge_key=bridge_record.bridge_key,
                    node_id=node_id,
                    output_index=output_index,
                    value=output_value,
                )
                output_value = RemoteSessionBridgeRef(
                    bridge_key=bridge_record.bridge_key,
                    node_id=node_id,
                    output_index=output_index,
                    session_id=live_ref.session_id,
                ).to_payload()
            outputs.append(output_value)
            logger.info(
                "Collected exported output %s from node %s output %d.",
                boundary_output.get("proxy_output_name"),
                node_id,
                output_index,
            )
        return tuple(outputs)


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
            output_value = _host_session_bridge._REMOTE_SESSION_STORE.get_output(
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
        _host_session_bridge._REMOTE_SESSION_BRIDGE_STORE.put_record(bridge_record)
        _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
        live_ref = _host_session_bridge._REMOTE_SESSION_STORE.put_bridge_output(
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
    node_mapping: dict[str, type[Any]] | None = None,
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    session_handle = _payload_remote_session_handle(payload)
    logger.info(
        "Executing local fallback subgraph %s with %d hydrated inputs session_id=%s clear_remote_session=%s.",
        payload.get("component_id"),
        len(hydrated_inputs),
        session_handle.session_id if session_handle is not None else None,
        bool(payload.get("clear_remote_session")),
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _execute_subgraph_prompt, payload, hydrated_inputs, node_mapping
            )
            try:
                outputs = future.result()
            except Exception:
                logger.exception(
                    "Local fallback subgraph %s raised while running in worker thread.",
                    payload.get("component_id"),
                )
                raise
    finally:
        if bool(payload.get("clear_remote_session")) and session_handle is not None:
            logger.info(
                "Clearing remote session after component=%s session_id=%s.",
                payload.get("component_id"),
                session_handle.session_id,
            )
            _host_session_bridge._REMOTE_SESSION_STORE.clear_session(session_handle)
    logger.info(
        "Local fallback subgraph %s completed with %d outputs.",
        payload.get("component_id"),
        len(outputs),
    )
    return serialize_node_outputs(outputs)

