"""Mapped and implicit-batch remote execution orchestration."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
import logging
import os
import threading
from typing import Any, Awaitable, Callable, Mapping

from ..serialization import (
    deserialize_node_inputs,
    deserialize_node_outputs,
    is_mapped_output_value,
    join_mapped_values_for_scheduler,
    serialize_node_inputs,
    serialize_node_outputs,
    split_mapped_value,
)
from ..session_state import (
    is_remote_session_bridge_ref_payload,
    is_remote_session_handle_payload,
    is_remote_session_value_ref_payload,
)
from .host_session_bridge import _payload_remote_session_handle
from .local_execution import (
    RemoteSubgraphExecutionError,
    _execute_subgraph_prompt,
    _load_nodes_module,
    _node_input_type_map,
    _resolve_required_subgraph_nodes,
)
from .local_ui_events import _emit_local_modal_progress
from .modal_deployment import (
    ModalRemoteInvocationError,
    _lookup_deployed_remote_engine,
    _mapped_lane_affinity_key,
)
from .modal_warmup import (
    _await_prompt_warmup_slots,
    _prompt_warmup_head_start_seconds,
    boost_mapped_component_warmup,
)
from ..settings import get_settings

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback environments.
    modal = None

_IMPLICIT_BATCH_PRESERVING_TARGETS = frozenset({("CreateVideo", "images")})
_MAPPED_PROGRESS_NODE_IDS_LOCK = threading.Lock()
_MAPPED_PROGRESS_NODE_IDS: dict[tuple[str, str, str], str] = {}


@dataclass(frozen=True)
class MappedExecutionHooks:
    """Callbacks supplied by the host orchestrator for invocation and interrupts."""

    invoke_remote_engine_async: Callable[..., Awaitable[bytes]]
    invoke_bound_remote_engine_async: Callable[[Any, dict[str, Any], bytes], Awaitable[bytes]]
    local_processing_interrupted: Callable[[], bool]
    raise_local_interrupt: Callable[[], None]
    exception_indicates_interruption: Callable[[BaseException], bool]


_MAPPED_EXECUTION_HOOKS: MappedExecutionHooks | None = None


def configure_mapped_execution_hooks(hooks: MappedExecutionHooks) -> None:
    """Install host callbacks without importing upward into the orchestrator."""
    global _MAPPED_EXECUTION_HOOKS
    _MAPPED_EXECUTION_HOOKS = hooks


def _mapped_execution_hooks() -> MappedExecutionHooks:
    """Return configured host callbacks or fail with a clear import-order error."""
    if _MAPPED_EXECUTION_HOOKS is None:
        raise RuntimeError("Mapped execution hooks have not been configured.")
    return _MAPPED_EXECUTION_HOOKS


async def invoke_remote_engine_async(*args: Any, **kwargs: Any) -> bytes:
    """Delegate remote invocation through the injected host callback."""
    return await _mapped_execution_hooks().invoke_remote_engine_async(*args, **kwargs)


async def _invoke_bound_remote_engine_async(
    remote_engine: Any, payload: dict[str, Any], kwargs_payload: bytes
) -> bytes:
    """Delegate affinity-bound invocation through the injected host callback."""
    return await _mapped_execution_hooks().invoke_bound_remote_engine_async(
        remote_engine, payload, kwargs_payload
    )


def _local_processing_interrupted() -> bool:
    """Return local interrupt state through the injected host callback."""
    return _mapped_execution_hooks().local_processing_interrupted()


def _raise_local_interrupt() -> None:
    """Raise the local interrupt through the injected host callback."""
    _mapped_execution_hooks().raise_local_interrupt()


def _exception_indicates_interruption(exc: BaseException) -> bool:
    """Classify interruption through the injected host callback."""
    return _mapped_execution_hooks().exception_indicates_interruption(exc)


def _mapped_execution_parallelism(item_count: int) -> int:
    """Return the local worker width used to schedule mapped Modal item executions."""
    settings = get_settings()
    configured_limit = settings.max_inflight_calls
    if settings.max_containers is not None:
        configured_limit = min(configured_limit, settings.max_containers)
    return max(1, min(item_count, configured_limit))


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
    *,
    suppress_status_stream: bool = False,
    lane_id: str | None = None,
    item_index: int | None = None,
) -> dict[str, Any]:
    """Return one explicit static or mapped subgraph payload."""
    phase_definition = _mapped_phase_definition(payload, phase_key)
    if phase_definition is None:
        raise KeyError(f"Mapped payload is missing phase definition {phase_key!r}.")

    phase_payload = {
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
    if suppress_status_stream:
        phase_payload["suppress_status_stream"] = True
    if lane_id is not None:
        phase_payload["mapped_progress_lane_id"] = str(lane_id)
        phase_payload["mapped_progress_display_node_id"] = str(
            payload.get("component_id", "modal-subgraph")
        )
    if item_index is not None:
        phase_payload["map_item_index"] = int(item_index)
    return phase_payload


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


def _execute_mapped_subgraph_payload(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> tuple[Any, ...]:
    """Execute one mapped payload locally using explicit static and mapped phases."""
    mapped_input = payload.get("mapped_input") or {}
    mapped_input_name = str(mapped_input.get("proxy_input_name") or "")
    if not mapped_input_name:
        raise ModalRemoteInvocationError(
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
    static_phase_payload: dict[str, Any] | None = None
    if payload.get("static_phase") is not None:
        static_phase_payload = _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
            suppress_status_stream=True,
        )
        static_phase_payload.pop("clear_remote_session", None)
    elif payload.get("static_execute_node_ids"):
        static_phase_payload = _build_static_mapped_payload(payload)

    if static_phase_payload is not None:
        if static_phase_payload.get("execute_node_ids"):
            logger.info(
                "Executing static mapped phase for component=%s with execute nodes=%s.",
                payload.get("component_id"),
                static_phase_payload.get("execute_node_ids", []),
            )
            static_phase_outputs = _execute_subgraph_prompt(
                static_phase_payload,
                dict(broadcast_inputs),
                node_mapping,
            )
            bridge_inputs, static_outputs = _split_phase_outputs(
                static_phase_outputs,
                list(static_phase_payload.get("boundary_outputs", [])),
                bridge_output_names,
            )
            broadcast_inputs.update(bridge_inputs)

    total_items = len(mapped_items)
    _emit_local_mapped_progress(payload, 0, total_items)
    per_item_outputs: list[tuple[Any, ...]] = []
    for item_index, item_value in enumerate(mapped_items):
        if _local_processing_interrupted():
            _raise_local_interrupt()
        if payload.get("mapped_phase") is not None:
            item_payload = _build_phase_subgraph_payload(
                payload,
                "mapped_phase",
                f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
                suppress_status_stream=True,
                lane_id="0",
                item_index=item_index,
            )
        else:
            item_payload = _build_mapped_item_payload(payload, item_index, 0)
        item_inputs = dict(broadcast_inputs)
        item_inputs[mapped_input_name] = item_value
        logger.info(
            "Executing mapped item %d/%d for component=%s with execute nodes=%s.",
            item_index + 1,
            total_items,
            payload.get("component_id"),
            item_payload.get("execute_node_ids", []),
        )
        per_item_outputs.append(
            _execute_subgraph_prompt(
                item_payload,
                item_inputs,
                node_mapping,
            )
        )
        _emit_local_mapped_progress(payload, item_index + 1, total_items)

    if payload.get("mapped_phase") is not None:
        mapped_phase_payload = _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::mapped",
        )
    else:
        mapped_phase_payload = {
            **payload,
            "boundary_outputs": [
                boundary_output
                for boundary_output in payload.get("boundary_outputs", [])
                if _is_mapped_boundary_output(boundary_output, payload)
            ],
        }
    mapped_outputs = _aggregate_mapped_outputs(
        per_item_outputs,
        {
            **payload,
            "boundary_outputs": list(mapped_phase_payload.get("boundary_outputs", [])),
        },
    )
    return _merge_static_and_mapped_outputs(
        static_outputs=static_outputs,
        mapped_outputs=mapped_outputs,
        payload=payload,
    )


def _build_mapped_item_payload(
    payload: dict[str, Any],
    item_index: int,
    lane_index: int,
) -> dict[str, Any]:
    """Return one per-item subgraph payload derived from a mapped remote component payload."""
    if _mapped_phase_definition(payload, "mapped_phase") is not None:
        return _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
            suppress_status_stream=True,
            lane_id=str(lane_index),
            item_index=item_index,
        )
    item_payload = copy.deepcopy(payload)
    item_payload["payload_kind"] = "subgraph"
    item_payload[
        "component_id"
    ] = f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}"
    item_payload["mapped_input"] = None
    item_payload["suppress_status_stream"] = True
    item_payload["map_item_index"] = item_index
    item_payload["mapped_progress_lane_id"] = str(lane_index)
    item_payload["mapped_progress_display_node_id"] = str(
        payload.get("component_id", "modal-subgraph")
    )
    item_payload.pop("clear_remote_session", None)
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
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced inconsistent output arity."
        )

    boundary_outputs = list(payload.get("boundary_outputs", []))
    aggregated_outputs: list[Any] = []
    for output_index in range(output_count):
        boundary_output = (
            boundary_outputs[output_index]
            if output_index < len(boundary_outputs)
            else {}
        )
        aggregated_outputs.append(
            join_mapped_values_for_scheduler(
                [item_outputs[output_index] for item_outputs in per_item_outputs],
                io_type=str(boundary_output.get("io_type", "*")),
                is_list=bool(boundary_output.get("is_list", False)),
                scheduler_is_list=bool(boundary_output.get("scheduler_is_list", False)),
            )
        )
    return tuple(aggregated_outputs)


def _build_remote_session_cleanup_payload(
    payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a dedicated cleanup payload that clears one shared remote session once."""
    remote_session = payload.get("remote_session")
    if not bool(
        payload.get("clear_remote_session")
    ) or not is_remote_session_handle_payload(remote_session):
        return None
    return {
        "payload_kind": "subgraph",
        "component_id": f"{payload.get('component_id', 'modal-subgraph')}::cleanup",
        **_shared_subgraph_payload_fields(payload),
        "component_node_ids": [],
        "subgraph_prompt": {},
        "boundary_inputs": [],
        "boundary_outputs": [],
        "execute_node_ids": [],
        "remote_session": copy.deepcopy(remote_session),
        "clear_remote_session": True,
        "suppress_status_stream": True,
        "terminate_container_on_error": False,
    }


def _is_mapped_boundary_output(
    boundary_output: dict[str, Any], payload: dict[str, Any]
) -> bool:
    """Return whether one boundary output belongs to the mapped per-item branch."""
    mapped_output = boundary_output.get("mapped_output")
    if mapped_output is not None:
        return bool(mapped_output)
    return bool(payload.get("mapped_input")) and not bool(
        payload.get("static_execute_node_ids")
    )


def _build_static_mapped_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the one-time static subgraph payload for a hybrid mapped component."""
    if _mapped_phase_definition(payload, "static_phase") is not None:
        return _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
            suppress_status_stream=True,
        )
    static_payload = copy.deepcopy(payload)
    static_payload["payload_kind"] = "subgraph"
    static_payload[
        "component_id"
    ] = f"{payload.get('component_id', 'modal-subgraph')}::static"
    static_payload["mapped_input"] = None
    static_payload["suppress_status_stream"] = True
    static_payload.pop("clear_remote_session", None)
    static_payload["execute_node_ids"] = list(
        payload.get("static_execute_node_ids") or []
    )
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

    if static_output_index != len(static_outputs) or mapped_output_index != len(
        mapped_outputs
    ):
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
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    display_node_id = str(
        payload.get("mapped_progress_display_node_id")
        or payload.get("component_id")
        or ""
    )
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=float(completed_items),
        max_value=float(total_items),
        display_node_id=display_node_id,
        aggregate_only=True,
    )


def _mapped_progress_owner_component_id(payload: dict[str, Any]) -> str | None:
    """Return the stable component id used to track one mapped worker lane locally."""
    owner_component_id = payload.get(
        "mapped_progress_display_node_id", payload.get("component_id")
    )
    if owner_component_id is None:
        return None
    owner_component = str(owner_component_id)
    return owner_component or None


def _remember_mapped_lane_node_id(
    payload: dict[str, Any], lane_id: str, node_id: str
) -> None:
    """Remember the last real node id that emitted progress for one mapped worker lane."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    owner_component_id = _mapped_progress_owner_component_id(payload)
    if not prompt_id or not owner_component_id or not node_id:
        return
    with _MAPPED_PROGRESS_NODE_IDS_LOCK:
        _MAPPED_PROGRESS_NODE_IDS[(prompt_id, owner_component_id, lane_id)] = node_id


def _pop_mapped_lane_node_id(payload: dict[str, Any], lane_id: str) -> str | None:
    """Forget and return the last real node id that emitted progress for one mapped worker lane."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    owner_component_id = _mapped_progress_owner_component_id(payload)
    if not prompt_id or not owner_component_id:
        return None
    with _MAPPED_PROGRESS_NODE_IDS_LOCK:
        return _MAPPED_PROGRESS_NODE_IDS.pop(
            (prompt_id, owner_component_id, lane_id), None
        )


def _clear_local_mapped_lane_progress(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int,
) -> None:
    """Remove one mapped worker lane from the local node overlay."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    lane_id = str(lane_index)
    display_node_id = _pop_mapped_lane_node_id(payload, lane_id) or str(
        payload.get("component_id") or ""
    )
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=lane_id,
        clear=True,
        item_index=item_index,
    )


def _emit_local_mapped_lane_progress_start(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int | None = None,
) -> None:
    """Create or reset one mapped worker lane before remote progress begins arriving."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
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
        setup_only=True,
    )


def _implicit_batch_preserving_targets(
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
) -> list[str]:
    """Return target sockets that must receive one complete tensor batch."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return []

    preserving_targets: list[str] = []
    for target in boundary_input.get("targets", []):
        node_id = str(target.get("node_id") or "")
        input_name = str(target.get("input_name") or "")
        prompt_node = prompt.get(node_id)
        if not node_id or not input_name or not isinstance(prompt_node, dict):
            continue
        class_type = str(prompt_node.get("class_type") or "")
        if (class_type, input_name) in _IMPLICIT_BATCH_PRESERVING_TARGETS:
            preserving_targets.append(f"{node_id}.{input_name}")
    return sorted(preserving_targets)


def _split_batch_boundary_inputs(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
) -> tuple[dict[str, list[Any]], int] | None:
    """Return zipped per-item boundary inputs when an ordinary subgraph receives batched values."""
    implicitly_batchable_scalar_io_types = frozenset(
        {"BOOLEAN", "FLOAT", "INT", "STRING"}
    )
    implicitly_batchable_transport_io_types = frozenset(
        {"IMAGE", "LATENT", "MASK", "NOISE", "SIGMAS"}
    )
    split_inputs: dict[str, list[Any]] = {}

    def is_session_ref_list(value: Any) -> bool:
        """Return whether `value` is a non-empty list of remote session refs."""
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(
                is_remote_session_value_ref_payload(item)
                or is_remote_session_bridge_ref_payload(item)
                for item in value
            )
        )

    def unwrap_singleton_session_ref_list(value: Any) -> Any:
        """Unwrap Comfy's output-list wrapper around mapped remote session refs."""
        if (
            isinstance(value, list)
            and len(value) == 1
            and is_session_ref_list(value[0])
        ):
            return value[0]
        return value

    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if not proxy_input_name or proxy_input_name not in hydrated_inputs:
            continue
        input_value = unwrap_singleton_session_ref_list(
            hydrated_inputs[proxy_input_name]
        )
        io_type = _implicit_batch_boundary_effective_io_type(
            payload=payload,
            boundary_input=boundary_input,
            input_value=input_value,
        )
        preserving_targets = _implicit_batch_preserving_targets(
            payload,
            boundary_input,
        )
        if preserving_targets:
            logger.info(
                "Skipping implicit batch split for boundary input %s io_type=%s "
                "because target sockets consume the complete tensor batch: %s.",
                proxy_input_name,
                io_type,
                preserving_targets,
            )
            continue
        input_is_session_ref_list = is_session_ref_list(input_value)
        if (
            isinstance(input_value, list)
            and not is_mapped_output_value(input_value)
            and not input_is_session_ref_list
            and io_type
            not in (
                implicitly_batchable_scalar_io_types
                | implicitly_batchable_transport_io_types
            )
        ):
            logger.info(
                "Skipping implicit batch split for boundary input %s io_type=%s because list-backed non-scalar values stay broadcast.",
                proxy_input_name,
                io_type,
            )
            continue
        try:
            items = split_mapped_value(
                input_value,
                io_type,
            )
        except (TypeError, ValueError):
            continue
        if len(items) <= 1:
            continue
        split_inputs[proxy_input_name] = items

    if not split_inputs:
        return None

    item_counts = {input_name: len(items) for input_name, items in split_inputs.items()}
    unique_counts = set(item_counts.values())
    if len(unique_counts) != 1:
        raise ModalRemoteInvocationError(
            "Implicit Modal batch boundary inputs must all have the same item count. "
            f"Received counts: {item_counts!r}"
        )
    return split_inputs, next(iter(unique_counts))


def _is_latent_like_mapping(value: Any) -> bool:
    """Return whether one runtime value looks like a ComfyUI LATENT mapping."""
    return isinstance(value, Mapping) and "samples" in value


def _list_is_latent_like_batch(value: Any) -> bool:
    """Return whether one runtime value is a list of LATENT-like mappings."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_latent_like_mapping(item) for item in value)
    )


def _implicit_batch_boundary_target_input_types(
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
) -> set[str]:
    """Return the declared input types of one boundary input's target sockets."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return set()

    try:
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    except ModuleNotFoundError:
        logger.debug(
            "Skipping target input type discovery for implicit batching because ComfyUI nodes are unavailable."
        )
        return set()

    target_input_types: set[str] = set()
    for target in boundary_input.get("targets", []):
        node_id = str(target.get("node_id") or "")
        input_name = str(target.get("input_name") or "")
        if not node_id or not input_name:
            continue
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, dict):
            continue
        node_class = resolved_node_mapping.get(str(prompt_node.get("class_type")))
        if node_class is None:
            continue
        prompt_inputs = prompt_node.get("inputs") or {}
        declared_type = _node_input_type_map(node_class, prompt_inputs).get(input_name)
        if isinstance(declared_type, str) and declared_type:
            target_input_types.add(declared_type)
    return target_input_types


def _implicit_batch_boundary_effective_io_type(
    *,
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
    input_value: Any,
) -> str:
    """Return the best effective io_type for implicit batching of one boundary input."""
    declared_io_type = str(boundary_input.get("io_type", "*"))
    if declared_io_type != "*":
        return declared_io_type

    if _list_is_latent_like_batch(input_value):
        return "LATENT"

    target_input_types = _implicit_batch_boundary_target_input_types(
        payload, boundary_input
    )
    if len(target_input_types) == 1:
        return next(iter(target_input_types))
    return declared_io_type


def _implicit_batch_input_is_list_target_node_ids(
    payload: dict[str, Any],
    split_inputs: dict[str, list[Any]],
) -> list[str]:
    """Return split-boundary target nodes that must consume the full list in one execution."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return []

    try:
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    except ModuleNotFoundError:
        logger.debug(
            "Skipping INPUT_IS_LIST detection for implicit batching because ComfyUI nodes are unavailable."
        )
        return []
    target_node_ids: set[str] = set()
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if proxy_input_name not in split_inputs:
            continue
        for target in boundary_input.get("targets", []):
            target_node_id = str(target.get("node_id") or "")
            if not target_node_id:
                continue
            prompt_node = prompt.get(target_node_id)
            if prompt_node is None:
                continue
            class_type = str(prompt_node.get("class_type"))
            node_class = resolved_node_mapping.get(class_type)
            if node_class is None:
                continue
            if bool(getattr(node_class, "INPUT_IS_LIST", False)):
                target_node_ids.add(target_node_id)
    return sorted(target_node_ids)


def _partition_implicit_batched_execute_nodes(
    payload: dict[str, Any],
    split_inputs: dict[str, list[Any]],
) -> tuple[list[str], list[str]]:
    """Split one implicitly batched subgraph into static and per-item execute targets."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        execute_node_ids = [
            str(node_id) for node_id in payload.get("execute_node_ids", [])
        ]
        return [], execute_node_ids

    batched_target_node_ids: set[str] = set()
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if proxy_input_name not in split_inputs:
            continue
        for target in boundary_input.get("targets", []):
            target_node_id = target.get("node_id")
            if target_node_id is not None:
                batched_target_node_ids.add(str(target_node_id))

    execute_node_ids = [str(node_id) for node_id in payload.get("execute_node_ids", [])]
    static_execute_node_ids: list[str] = []
    mapped_execute_node_ids: list[str] = []
    for execute_node_id in execute_node_ids:
        required_node_ids = set(
            _resolve_required_subgraph_nodes(
                prompt=prompt,
                execute_node_ids=[execute_node_id],
            )
        )
        if required_node_ids & batched_target_node_ids:
            mapped_execute_node_ids.append(execute_node_id)
            continue
        static_execute_node_ids.append(execute_node_id)

    if not mapped_execute_node_ids and execute_node_ids:
        logger.warning(
            "Implicitly batched Modal component=%s had batched inputs %s but no execute target depended on them; "
            "falling back to per-item execution for all execute nodes.",
            payload.get("component_id"),
            sorted(split_inputs),
        )
        return [], execute_node_ids

    logger.info(
        "Partitioned implicitly batched Modal component=%s into static execute nodes=%s and mapped execute nodes=%s.",
        payload.get("component_id"),
        static_execute_node_ids,
        mapped_execute_node_ids,
    )
    return static_execute_node_ids, mapped_execute_node_ids


def _annotate_implicit_batched_boundary_outputs(
    payload: dict[str, Any],
    mapped_execute_node_ids: list[str],
) -> list[dict[str, Any]]:
    """Mark which boundary outputs belong to the per-item branch of an implicitly batched subgraph."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict) or not mapped_execute_node_ids:
        return [
            copy.deepcopy(boundary_output)
            for boundary_output in payload.get("boundary_outputs", [])
        ]

    mapped_required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=[str(node_id) for node_id in mapped_execute_node_ids],
        )
    )
    annotated_outputs: list[dict[str, Any]] = []
    for boundary_output in payload.get("boundary_outputs", []):
        annotated_output = copy.deepcopy(boundary_output)
        annotated_output["mapped_output"] = (
            str(boundary_output.get("node_id")) in mapped_required_node_ids
        )
        annotated_outputs.append(annotated_output)
    return annotated_outputs


def _log_detached_mapped_lane_result(
    task: asyncio.Task[None],
    *,
    component_id: str,
    lane_index: int,
) -> None:
    """Consume and log any late failure from one detached mapped worker lane."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        logger.warning(
            "Detached mapped worker lane=%d for component=%s finished with an unawaited failure: %s",
            lane_index,
            component_id,
            exc,
        )


async def _invoke_implicitly_mapped_subgraph_async(
    payload: dict[str, Any], kwargs_payload: bytes
) -> bytes:
    """Fan out one ordinary subgraph payload when batchable boundary inputs arrive zipped."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    split_batch_inputs = _split_batch_boundary_inputs(payload, hydrated_inputs)
    if split_batch_inputs is None:
        raise ModalRemoteInvocationError(
            "Implicit mapped subgraph execution requires at least one batched boundary input."
        )

    split_inputs, total_items = split_batch_inputs
    input_is_list_target_node_ids = _implicit_batch_input_is_list_target_node_ids(
        payload,
        split_inputs,
    )
    if input_is_list_target_node_ids:
        logger.info(
            "Executing implicitly batched Modal component=%s as one ordinary subgraph because split boundary inputs target INPUT_IS_LIST nodes=%s.",
            payload.get("component_id"),
            input_is_list_target_node_ids,
        )
        return await invoke_remote_engine_async(
            payload,
            kwargs_payload,
            allow_implicit_mapping=False,
        )

    parallelism, refined_prompt_warmup_target = boost_mapped_component_warmup(
        payload,
        total_items=total_items,
        reason="implicit_mapped_component_exact_parallelism",
    )
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    if prompt_id is not None:
        await _await_prompt_warmup_slots(
            prompt_id,
            list(range(refined_prompt_warmup_target)),
            _prompt_warmup_head_start_seconds(),
        )
    logger.info(
        "Scheduling implicitly mapped Modal component=%s for %d item(s) with local parallelism=%d prompt_warmup_target=%d across inputs=%s.",
        payload.get("component_id"),
        total_items,
        parallelism,
        refined_prompt_warmup_target,
        sorted(split_inputs),
    )
    _emit_local_mapped_progress(payload, 0, total_items)

    broadcast_inputs = {
        input_name: value
        for input_name, value in hydrated_inputs.items()
        if input_name not in split_inputs
    }
    (
        static_execute_node_ids,
        mapped_execute_node_ids,
    ) = _partition_implicit_batched_execute_nodes(
        payload,
        split_inputs,
    )
    hybrid_payload = copy.deepcopy(payload)
    hybrid_payload["static_execute_node_ids"] = static_execute_node_ids
    hybrid_payload["mapped_execute_node_ids"] = mapped_execute_node_ids
    hybrid_payload["boundary_outputs"] = _annotate_implicit_batched_boundary_outputs(
        payload,
        mapped_execute_node_ids,
    )
    cleanup_payload = _build_remote_session_cleanup_payload(hybrid_payload)
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    use_seeded_remote_lanes = (
        execution_mode != "local"
        and modal is not None
        and _payload_remote_session_handle(hybrid_payload) is not None
        and _mapped_phase_definition(hybrid_payload, "static_phase") is not None
    )

    static_outputs: tuple[Any, ...] = ()
    lane_remote_engines: list[Any] = []
    seeded_lane_indices: set[int] = set()
    skip_cleanup_after_interrupt = False
    try:
        if use_seeded_remote_lanes:
            lane_remote_engines = [
                _lookup_deployed_remote_engine(
                    hybrid_payload,
                    affinity_key_override=_mapped_lane_affinity_key(
                        hybrid_payload, lane_index
                    ),
                )
                for lane_index in range(parallelism)
            ]
            logger.info(
                "Seeding %d mapped worker lane(s) for component=%s and allowing ready lanes to start item dispatch independently.",
                parallelism,
                payload.get("component_id"),
            )

            async def seed_lane(lane_index: int) -> None:
                """Run the static bridge-producing phase once on one mapped worker lane."""
                lane_seed_payload = _build_phase_subgraph_payload(
                    hybrid_payload,
                    "static_phase",
                    f"{payload.get('component_id', 'modal-subgraph')}::seed:{lane_index}",
                    suppress_status_stream=True,
                )
                lane_seed_payload.pop("clear_remote_session", None)
                await _invoke_bound_remote_engine_async(
                    lane_remote_engines[lane_index],
                    lane_seed_payload,
                    serialize_node_inputs(broadcast_inputs),
                )
                seeded_lane_indices.add(lane_index)

        elif static_execute_node_ids:
            static_response = await invoke_remote_engine_async(
                _build_static_mapped_payload(hybrid_payload),
                serialize_node_inputs(broadcast_inputs),
            )
            static_outputs = deserialize_node_outputs(static_response)
        else:
            lane_remote_engines = []

        per_item_outputs: list[tuple[Any, ...] | None] = [None] * total_items
        completed_items = 0
        all_items_completed = asyncio.Event()
        stop_dispatch_requested = asyncio.Event()
        worker_failure: asyncio.Future[
            BaseException
        ] = asyncio.get_running_loop().create_future()
        item_queue: asyncio.Queue[int | None] = asyncio.Queue()
        for item_index in range(total_items):
            item_queue.put_nowait(item_index)
        for _ in range(parallelism):
            item_queue.put_nowait(None)

        def request_interrupt_stop() -> None:
            """Stop queued mapped item dispatch and suppress cleanup after interruption."""
            nonlocal skip_cleanup_after_interrupt
            skip_cleanup_after_interrupt = True
            stop_dispatch_requested.set()

        def raise_if_local_interrupted() -> None:
            """Raise the native ComfyUI interrupt after marking mapped dispatch as stopped."""
            if _local_processing_interrupted():
                request_interrupt_stop()
                _raise_local_interrupt()

        async def run_worker(lane_index: int) -> None:
            """Execute queued implicit mapped items through one stable local worker lane."""
            nonlocal completed_items
            try:
                raise_if_local_interrupted()
                if use_seeded_remote_lanes:
                    _emit_local_mapped_lane_progress_start(payload, lane_index)
                    await seed_lane(lane_index)
                while True:
                    if stop_dispatch_requested.is_set():
                        return
                    item_index = await item_queue.get()
                    if item_index is None:
                        return
                    if stop_dispatch_requested.is_set():
                        return
                    raise_if_local_interrupted()
                    try:
                        item_payload = _build_mapped_item_payload(
                            hybrid_payload, item_index, lane_index
                        )
                        item_inputs = dict(broadcast_inputs)
                        for input_name, items in split_inputs.items():
                            item_inputs[input_name] = items[item_index]
                        if use_seeded_remote_lanes:
                            item_response = await _invoke_bound_remote_engine_async(
                                lane_remote_engines[lane_index],
                                item_payload,
                                serialize_node_inputs(item_inputs),
                            )
                        else:
                            item_response = await invoke_remote_engine_async(
                                item_payload,
                                serialize_node_inputs(item_inputs),
                            )
                        raise_if_local_interrupted()
                        per_item_outputs[item_index] = deserialize_node_outputs(
                            item_response
                        )
                        completed_items += 1
                        _emit_local_mapped_progress(
                            payload, completed_items, total_items
                        )
                        if completed_items >= total_items:
                            all_items_completed.set()
                    finally:
                        _clear_local_mapped_lane_progress(
                            payload, lane_index, item_index
                        )
            except BaseException as exc:
                if all_items_completed.is_set():
                    logger.info(
                        "Ignoring late mapped worker failure after component=%s already completed all items lane=%d: %s",
                        payload.get("component_id"),
                        lane_index,
                        exc,
                    )
                    return
                stop_dispatch_requested.set()
                if _exception_indicates_interruption(exc):
                    request_interrupt_stop()
                if not worker_failure.done():
                    worker_failure.set_result(exc)
                raise

        tasks = [
            asyncio.create_task(run_worker(lane_index))
            for lane_index in range(parallelism)
        ]
        for lane_index, task in enumerate(tasks):
            task.add_done_callback(
                lambda completed_task, lane_index=lane_index: _log_detached_mapped_lane_result(
                    completed_task,
                    component_id=str(payload.get("component_id", "modal-subgraph")),
                    lane_index=lane_index,
                )
            )
        completion_task = asyncio.create_task(all_items_completed.wait())
        try:
            while not all_items_completed.is_set():
                wait_set: set[asyncio.Task[Any] | asyncio.Future[BaseException]] = {
                    completion_task,
                    worker_failure,
                    *tasks,
                }
                done, _ = await asyncio.wait(
                    wait_set, return_when=asyncio.FIRST_COMPLETED
                )
                if completion_task in done or all_items_completed.is_set():
                    break
                if worker_failure in done:
                    stop_dispatch_requested.set()
                    raise worker_failure.result()
                for task in done:
                    if task in tasks:
                        task_exc = task.exception()
                        if task_exc is not None:
                            stop_dispatch_requested.set()
                            raise task_exc
            if not all(item_outputs is not None for item_outputs in per_item_outputs):
                await asyncio.gather(*tasks)
        except BaseException as exc:
            stop_dispatch_requested.set()
            if _exception_indicates_interruption(exc):
                skip_cleanup_after_interrupt = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            completion_task.cancel()
            await asyncio.gather(completion_task, return_exceptions=True)

        return serialize_node_outputs(
            _merge_static_and_mapped_outputs(
                static_outputs=static_outputs,
                mapped_outputs=_aggregate_mapped_outputs(
                    [
                        item_outputs
                        for item_outputs in per_item_outputs
                        if item_outputs is not None
                    ],
                    {
                        **hybrid_payload,
                        "boundary_outputs": [
                            boundary_output
                            for boundary_output in hybrid_payload.get(
                                "boundary_outputs", []
                            )
                            if _is_mapped_boundary_output(
                                boundary_output, hybrid_payload
                            )
                        ],
                    },
                ),
                payload=hybrid_payload,
            )
        )
    finally:
        if cleanup_payload is not None and not skip_cleanup_after_interrupt:
            if use_seeded_remote_lanes and lane_remote_engines:
                await asyncio.gather(
                    *(
                        _invoke_bound_remote_engine_async(
                            lane_remote_engines[lane_index],
                            {
                                **cleanup_payload,
                                "component_id": (
                                    f"{payload.get('component_id', 'modal-subgraph')}::cleanup:{lane_index}"
                                ),
                            },
                            serialize_node_inputs({}),
                        )
                        for lane_index in sorted(seeded_lane_indices)
                    )
                )
            else:
                await invoke_remote_engine_async(
                    cleanup_payload,
                    serialize_node_inputs({}),
                )
        elif cleanup_payload is not None:
            logger.info(
                "Skipping remote-session cleanup for implicitly mapped Modal component=%s because execution was interrupted.",
                payload.get("component_id"),
            )


async def _invoke_mapped_remote_engine_async(
    payload: dict[str, Any], kwargs_payload: bytes
) -> bytes:
    """Execute one mapped payload locally using explicit static and mapped phases."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    return await asyncio.to_thread(
        lambda: serialize_node_outputs(
            _execute_mapped_subgraph_payload(
                payload,
                hydrated_inputs,
            )
        )
    )



