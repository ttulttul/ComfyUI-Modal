"""Mapped and phased execution for the cloud worker runtime."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import threading
from typing import Any, Callable

try:
    from . import cloud_prompt_execution as _prompt_execution
    from .serialization import (
        join_mapped_values_for_scheduler,
        split_mapped_value,
    )
except ImportError:  # pragma: no cover - flat Modal-container import.
    import cloud_prompt_execution as _prompt_execution
    from serialization import (
        join_mapped_values_for_scheduler,
        split_mapped_value,
    )

logger = logging.getLogger(__name__)


def _execute_prompt_subgraph(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute one phase through the cloud prompt-execution owner."""
    return _prompt_execution._execute_subgraph_prompt(
        payload,
        hydrated_inputs,
        custom_nodes_root,
        status_callback=status_callback,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )


def _remote_subgraph_execution_error(message: str) -> RuntimeError:
    """Construct the stable cloud entrypoint's configured subgraph error."""
    return _prompt_execution.RemoteSubgraphExecutionError(message)


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
        raise _remote_subgraph_execution_error(
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
                raise _remote_subgraph_execution_error(
                    "Mapped remote execution returned fewer mapped outputs than expected."
                )
            combined_outputs.append(mapped_outputs[mapped_output_index])
            mapped_output_index += 1
            continue
        if static_output_index >= len(static_outputs):
            raise _remote_subgraph_execution_error(
                "Mapped remote execution returned fewer static outputs than expected."
            )
        combined_outputs.append(static_outputs[static_output_index])
        static_output_index += 1

    if static_output_index != len(static_outputs) or mapped_output_index != len(
        mapped_outputs
    ):
        raise _remote_subgraph_execution_error(
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
        raise _remote_subgraph_execution_error(
            "Mapped remote payloads must define mapped_input.proxy_input_name."
        )
    if mapped_input_name not in hydrated_inputs:
        raise KeyError(
            f"Mapped remote payload input {mapped_input_name!r} was not provided."
        )

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
            static_phase_outputs = _execute_prompt_subgraph(
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
            _execute_prompt_subgraph(
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

