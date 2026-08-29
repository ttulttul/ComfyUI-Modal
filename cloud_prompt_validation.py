"""Prompt normalization, schema validation, and failure diagnostics."""

from __future__ import annotations

import copy
import logging
from typing import Any

try:
    from .remote_protocol import (
        PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
    )
except ImportError:  # pragma: no cover - flat Modal-container import.
    from remote_protocol import (
        PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
    )

logger = logging.getLogger(__name__)

RemoteSubgraphExecutionError: type[RuntimeError] = RuntimeError


def configure_cloud_prompt_validation_error(
    error_type: type[RuntimeError],
) -> None:
    """Install the stable cloud entrypoint's subgraph execution error type."""
    global RemoteSubgraphExecutionError
    RemoteSubgraphExecutionError = error_type


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


