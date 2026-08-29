"""Remote component payload construction for prompt rewriting."""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass
from typing import Any

if __package__:
    from .component_planning import (
        _boundary_output_payload,
        _component_has_local_reentry_dependency,
        _component_has_parallel_local_remote_fanout,
        _component_upstream_closure,
        _filter_boundary_inputs_for_node_ids,
        _filter_boundary_outputs_for_node_ids,
        _order_execute_node_ids_for_transportable_splits,
        _subgraph_topological_node_order,
        _subset_component_prompt,
    )
    from .prompt_payload_metadata import (
        _attach_snapshot_profile_key,
        _serialize_boundary_input_specs,
    )
    from .remote_graph_analysis import (
        _is_link,
        _remote_output_io_type,
        _remote_output_is_list,
    )
    from .remote_plan_types import (
        BoundaryInputSpec,
        BoundaryOutputSpec,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        ProducedPhaseOutputSpec,
        RemoteComponentPlan,
        StaticToMappedBoundarySpec,
    )
    from .session_state import RemoteSessionHandle
    from .settings import ModalSyncSettings
    from .sync_engine import SyncedAsset
else:  # pragma: no cover - flat import inside the Modal container.
    from component_planning import (
        _boundary_output_payload,
        _component_has_local_reentry_dependency,
        _component_has_parallel_local_remote_fanout,
        _component_upstream_closure,
        _filter_boundary_inputs_for_node_ids,
        _filter_boundary_outputs_for_node_ids,
        _order_execute_node_ids_for_transportable_splits,
        _subgraph_topological_node_order,
        _subset_component_prompt,
    )
    from prompt_payload_metadata import (
        _attach_snapshot_profile_key,
        _serialize_boundary_input_specs,
    )
    from remote_graph_analysis import (
        _is_link,
        _remote_output_io_type,
        _remote_output_is_list,
    )
    from remote_plan_types import (
        BoundaryInputSpec,
        BoundaryOutputSpec,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        ProducedPhaseOutputSpec,
        RemoteComponentPlan,
        StaticToMappedBoundarySpec,
    )
    from session_state import RemoteSessionHandle
    from settings import ModalSyncSettings
    from sync_engine import SyncedAsset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PayloadBuildContext:
    """Hold immutable inputs shared by component payload builders."""

    component_prompt: dict[str, Any]
    signature_prompt: dict[str, Any]
    prompt_id: Any
    extra_data: dict[str, Any] | None
    settings: ModalSyncSettings
    requires_volume_reload: bool
    volume_reload_marker: str | None
    custom_nodes_bundle_path: str | None
    uploaded_volume_paths: list[str]
    terminate_container_on_error: bool
    nodes_module: Any


def _remote_worker_affinity_group(
    context: _PayloadBuildContext,
    component_node_ids: list[str],
) -> str:
    """Return the worker-pool group required by one remote component phase."""
    class_types = {
        str((context.component_prompt.get(node_id) or {}).get("class_type") or "")
        for node_id in component_node_ids
    }
    return "llm" if "ModalLLM" in class_types else "comfy"


def _build_subgraph_payload(
    context: _PayloadBuildContext,
    *,
    component_id: str,
    component_node_ids: list[str],
    boundary_inputs: list[BoundaryInputSpec],
    boundary_outputs: list[dict[str, Any]],
    execute_node_ids: list[str],
    remote_session: dict[str, Any] | None = None,
    clear_remote_session: bool = False,
    mapped_progress_display_node_id: str | None = None,
) -> dict[str, Any]:
    """Build one ordinary subgraph payload for a proxy node."""
    payload = {
        "payload_kind": "subgraph",
        "component_id": component_id,
        "prompt_id": context.prompt_id,
        "modal_gpu": context.settings.modal_gpu,
        "remote_worker_affinity_group": _remote_worker_affinity_group(
            context,
            component_node_ids,
        ),
        "component_node_ids": list(component_node_ids),
        "subgraph_prompt": _subset_component_prompt(
            context.component_prompt,
            component_node_ids,
        ),
        "boundary_inputs": _serialize_boundary_input_specs(
            boundary_inputs,
            signature_prompt=context.signature_prompt,
        ),
        "boundary_outputs": copy.deepcopy(boundary_outputs),
        "execute_node_ids": list(execute_node_ids),
        "extra_data": copy.deepcopy(context.extra_data or {}),
        "requires_volume_reload": context.requires_volume_reload,
        "volume_reload_marker": context.volume_reload_marker,
        "uploaded_volume_paths": list(context.uploaded_volume_paths),
        "terminate_container_on_error": context.terminate_container_on_error,
        "custom_nodes_bundle": context.custom_nodes_bundle_path,
    }
    if remote_session is not None:
        payload["remote_session"] = copy.deepcopy(remote_session)
    if clear_remote_session:
        payload["clear_remote_session"] = True
    if mapped_progress_display_node_id is not None:
        payload["mapped_progress_display_node_id"] = mapped_progress_display_node_id
    return payload


@dataclass
class _PhasePayloadBuildState:
    """Track mutable state while constructing ordered transportable phases."""

    remaining_node_ids: set[str]
    remaining_execute_node_ids: list[str]
    payloads: list[dict[str, Any]]
    produced_outputs: dict[LinkedOutputRef, ProducedPhaseOutputSpec]
    local_boundary_outputs: dict[LinkedOutputRef, BoundaryOutputSpec]
    bridge_output_counter: int = 0


def _component_requires_ordered_phases(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
) -> bool:
    """Return whether local feedback or fanout requires ordered proxies."""
    if len(component.execute_node_ids) <= 1:
        return False
    has_local_reentry = _component_has_local_reentry_dependency(
        prompt=context.signature_prompt, component=component
    )
    has_parallel_fanout = _component_has_parallel_local_remote_fanout(component)
    if not has_local_reentry and not has_parallel_fanout:
        logger.info(
            "Keeping remote component %s as one proxy because execute targets %s "
            "have neither a local re-entry dependency nor parallel local/remote fanout.",
            component.representative_node_id,
            component.execute_node_ids,
        )
        return False
    if component.local_tap_node_ids:
        logger.info(
            "Allowing remote component %s with local tap nodes %s to split because it has a local re-entry dependency.",
            component.representative_node_id,
            component.local_tap_node_ids,
        )
    if has_parallel_fanout:
        logger.info(
            "Forcing ordered phases for remote component %s because a remote output "
            "feeds both a non-returning local branch and later remote execution.",
            component.representative_node_id,
        )
    return True


def _ordered_phase_execute_node_ids(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
) -> list[str]:
    """Return execute targets in the safe order for transportable splits."""
    component_node_ids = set(component.node_ids)
    execute_node_ids = set(component.execute_node_ids)
    topological_node_ids = _subgraph_topological_node_order(
        context.component_prompt, component_node_ids
    )
    return _order_execute_node_ids_for_transportable_splits(
        prompt=context.signature_prompt,
        component_prompt=context.component_prompt,
        component_node_ids=component_node_ids,
        execute_node_ids=[
            node_id for node_id in topological_node_ids if node_id in execute_node_ids
        ],
    )


def _phase_boundary_inputs(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    state: _PhasePayloadBuildState,
    phase_node_ids: list[str],
) -> list[BoundaryInputSpec]:
    """Build external and earlier-phase inputs for one ordered phase."""
    phase_node_id_set = set(phase_node_ids)
    boundary_inputs = _filter_boundary_inputs_for_node_ids(
        component.boundary_inputs, phase_node_id_set
    )
    boundary_inputs_by_name = {
        boundary.proxy_input_name: boundary for boundary in boundary_inputs
    }
    for phase_node_id in phase_node_ids:
        prompt_node = context.component_prompt.get(phase_node_id) or {}
        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(str(input_value[0]), int(input_value[1]))
            produced_output = state.produced_outputs.get(source)
            if source.node_id in phase_node_id_set or produced_output is None:
                continue
            boundary_input = boundary_inputs_by_name.get(
                produced_output.proxy_output_name
            )
            if boundary_input is None:
                boundary_input = BoundaryInputSpec(
                    proxy_input_name=produced_output.proxy_output_name,
                    source=source,
                    io_type=produced_output.io_type,
                )
                boundary_inputs.append(boundary_input)
                boundary_inputs_by_name[boundary_input.proxy_input_name] = boundary_input
            boundary_input.targets.append(
                InputTarget(node_id=phase_node_id, input_name=str(input_name))
            )
    return boundary_inputs


def _phase_bridge_output_metadata(
    source: LinkedOutputRef,
    *,
    context: _PayloadBuildContext,
    state: _PhasePayloadBuildState,
) -> tuple[str, bool]:
    """Return the type metadata for one output bridged to a later phase."""
    local_output = state.local_boundary_outputs.get(source)
    if local_output is not None:
        return local_output.io_type, local_output.is_list
    io_type = str(
        _remote_output_io_type(
            prompt=context.component_prompt,
            node_id=source.node_id,
            output_index=source.output_index,
            nodes_module=context.nodes_module,
        )
        or "*"
    )
    is_list = _remote_output_is_list(
        prompt=context.component_prompt,
        node_id=source.node_id,
        output_index=source.output_index,
        nodes_module=context.nodes_module,
    )
    return io_type, is_list


def _record_phase_bridge_output(
    source: LinkedOutputRef,
    *,
    context: _PayloadBuildContext,
    state: _PhasePayloadBuildState,
    output_names_by_source: dict[LinkedOutputRef, str],
    boundary_outputs: list[dict[str, Any]],
) -> None:
    """Publish one phase output needed by a later ordered phase."""
    io_type, is_list = _phase_bridge_output_metadata(
        source, context=context, state=state
    )
    proxy_output_name = output_names_by_source.get(source)
    if proxy_output_name is None:
        proxy_output_name = f"phase_bridge_{state.bridge_output_counter}"
        state.bridge_output_counter += 1
        boundary_outputs.append(
            {
                "proxy_output_name": proxy_output_name,
                "node_id": source.node_id,
                "output_index": source.output_index,
                "io_type": io_type,
                "is_list": is_list,
                "preview_target_node_ids": [],
                "session_output": True,
            }
        )
        output_names_by_source[source] = proxy_output_name
    state.produced_outputs[source] = ProducedPhaseOutputSpec(
        proxy_output_name=proxy_output_name,
        source=source,
        io_type=io_type,
        is_list=is_list,
        session_output=True,
    )


def _phase_boundary_outputs(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    state: _PhasePayloadBuildState,
    phase_node_ids: set[str],
) -> list[dict[str, Any]]:
    """Build local and later-phase outputs for one ordered phase."""
    local_outputs = _filter_boundary_outputs_for_node_ids(
        component.boundary_outputs, phase_node_ids
    )
    boundary_outputs = [_boundary_output_payload(output) for output in local_outputs]
    output_names_by_source = {
        output.source: output.proxy_output_name for output in local_outputs
    }
    for pending_node_id in sorted(state.remaining_node_ids - phase_node_ids):
        prompt_node = context.component_prompt.get(pending_node_id) or {}
        for input_value in (prompt_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(str(input_value[0]), int(input_value[1]))
            if source.node_id not in phase_node_ids or source in state.produced_outputs:
                continue
            _record_phase_bridge_output(
                source,
                context=context,
                state=state,
                output_names_by_source=output_names_by_source,
                boundary_outputs=boundary_outputs,
            )
    return boundary_outputs


def _append_next_phase_payload(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    state: _PhasePayloadBuildState,
) -> None:
    """Build and consume the next ordered phase from mutable split state."""
    target_node_id = state.remaining_execute_node_ids[0]
    phase_node_ids = sorted(
        _component_upstream_closure(
            prompt=context.component_prompt,
            seed_node_ids={target_node_id},
            candidate_node_ids=state.remaining_node_ids,
        )
    )
    if not phase_node_ids:
        raise ModalPromptValidationError(
            f"Unable to derive split phase nodes for remote component {component.representative_node_id}."
        )
    phase_node_id_set = set(phase_node_ids)
    state.payloads.append(
        _build_subgraph_payload(
            context,
            component_id=str(target_node_id),
            component_node_ids=phase_node_ids,
            boundary_inputs=_phase_boundary_inputs(
                component, context, state, phase_node_ids
            ),
            boundary_outputs=_phase_boundary_outputs(
                component, context, state, phase_node_id_set
            ),
            execute_node_ids=[
                node_id
                for node_id in component.execute_node_ids
                if node_id in phase_node_id_set
            ],
        )
    )
    state.remaining_node_ids -= phase_node_id_set
    state.remaining_execute_node_ids = [
        node_id
        for node_id in state.remaining_execute_node_ids
        if node_id not in phase_node_id_set
    ]


def _attach_ordered_phase_session(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    payloads: list[dict[str, Any]],
    remote_session: dict[str, Any] | None,
) -> None:
    """Attach a shared session when ordered phases exchange remote values."""
    has_session_bridges = any(
        bool(output.get("session_output"))
        for payload in payloads
        for output in payload.get("boundary_outputs", [])
    )
    active_session = remote_session
    if active_session is None and has_session_bridges:
        active_session = RemoteSessionHandle(
            session_id=uuid.uuid4().hex,
            prompt_id=(
                str(context.prompt_id) if context.prompt_id is not None else None
            ),
            owner_component_id=component.representative_node_id,
        ).to_payload()
    if active_session is None:
        return
    for index, payload in enumerate(payloads):
        payload["remote_session"] = copy.deepcopy(active_session)
        if index == len(payloads) - 1:
            payload["clear_remote_session"] = True


def _build_phase_payloads_for_transportable_splits(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    remote_session: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Return ordered phases for local feedback or parallel local fanout."""
    if not _component_requires_ordered_phases(component, context):
        return None
    execute_node_ids = _ordered_phase_execute_node_ids(component, context)
    if len(execute_node_ids) <= 1:
        return None
    state = _PhasePayloadBuildState(
        remaining_node_ids=set(component.node_ids),
        remaining_execute_node_ids=execute_node_ids,
        payloads=[],
        produced_outputs={},
        local_boundary_outputs={
            output.source: output for output in component.boundary_outputs
        },
    )
    while state.remaining_execute_node_ids:
        _append_next_phase_payload(component, context, state)
    if len(state.payloads) <= 1:
        return None
    _attach_ordered_phase_session(
        component, context, state.payloads, remote_session
    )
    logger.info(
        "Split ordinary remote component %s into ordered phases: %s",
        component.representative_node_id,
        [
            {
                "component_id": payload["component_id"],
                "component_node_ids": payload["component_node_ids"],
                "execute_node_ids": payload["execute_node_ids"],
            }
            for payload in state.payloads
        ],
    )
    return state.payloads



@dataclass(frozen=True)
class _HybridPayloadParts:
    """Hold the static and mapped boundary partitions for one component."""

    static_boundary_inputs: list[BoundaryInputSpec]
    mapped_boundary_inputs: list[BoundaryInputSpec]
    static_boundary_outputs: list[BoundaryOutputSpec]
    mapped_boundary_outputs: list[BoundaryOutputSpec]
    static_bridge_outputs: list[dict[str, Any]]
    static_to_mapped_inputs: list[BoundaryInputSpec]
    static_to_mapped_payloads: list[dict[str, Any]]


def _static_bridge_output_payloads(
    boundaries: list[StaticToMappedBoundarySpec],
) -> list[dict[str, Any]]:
    """Serialize static outputs that are retained in the remote session."""
    return [
        {
            "proxy_output_name": boundary.proxy_name,
            "node_id": boundary.source.node_id,
            "output_index": boundary.source.output_index,
            "io_type": boundary.io_type,
            "is_list": boundary.is_list,
            "preview_target_node_ids": [],
            "session_output": True,
        }
        for boundary in boundaries
    ]


def _static_to_mapped_input_specs(
    boundaries: list[StaticToMappedBoundarySpec],
) -> list[BoundaryInputSpec]:
    """Convert static-to-mapped links into mapped-phase boundary inputs."""
    return [
        BoundaryInputSpec(
            proxy_input_name=boundary.proxy_name,
            source=boundary.source,
            io_type=boundary.io_type,
            targets=list(boundary.targets),
        )
        for boundary in boundaries
    ]


def _static_to_mapped_boundary_payloads(
    boundaries: list[StaticToMappedBoundarySpec],
) -> list[dict[str, Any]]:
    """Serialize links from the static phase into mapped item execution."""
    return [
        {
            "proxy_name": boundary.proxy_name,
            "node_id": boundary.source.node_id,
            "output_index": boundary.source.output_index,
            "io_type": boundary.io_type,
            "is_list": boundary.is_list,
            "targets": [
                {"node_id": target.node_id, "input_name": target.input_name}
                for target in boundary.targets
            ],
        }
        for boundary in boundaries
    ]


def _hybrid_payload_parts(component: RemoteComponentPlan) -> _HybridPayloadParts:
    """Partition one hybrid component into its static and mapped boundaries."""
    static_node_ids = set(component.static_node_ids)
    mapped_node_ids = set(component.mapped_node_ids)
    boundaries = component.static_to_mapped_boundaries
    return _HybridPayloadParts(
        static_boundary_inputs=_filter_boundary_inputs_for_node_ids(
            component.boundary_inputs, static_node_ids
        ),
        mapped_boundary_inputs=_filter_boundary_inputs_for_node_ids(
            component.boundary_inputs, mapped_node_ids
        ),
        static_boundary_outputs=_filter_boundary_outputs_for_node_ids(
            component.boundary_outputs, static_node_ids
        ),
        mapped_boundary_outputs=_filter_boundary_outputs_for_node_ids(
            component.boundary_outputs, mapped_node_ids
        ),
        static_bridge_outputs=_static_bridge_output_payloads(boundaries),
        static_to_mapped_inputs=_static_to_mapped_input_specs(boundaries),
        static_to_mapped_payloads=_static_to_mapped_boundary_payloads(boundaries),
    )


def _build_base_component_payload(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    remote_session: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the common payload fields shared by mapped and ordinary work."""
    mapped_node_ids = set(component.mapped_node_ids)
    payload = {
        "payload_kind": (
            "mapped_subgraph" if component.mapped_boundary_input_name else "subgraph"
        ),
        "component_id": component.representative_node_id,
        "prompt_id": context.prompt_id,
        "modal_gpu": context.settings.modal_gpu,
        "remote_worker_affinity_group": _remote_worker_affinity_group(
            context, list(component.node_ids)
        ),
        "component_node_ids": list(component.node_ids),
        "subgraph_prompt": context.component_prompt,
        "boundary_inputs": _serialize_boundary_input_specs(
            component.boundary_inputs,
            signature_prompt=context.signature_prompt,
        ),
        "boundary_outputs": [
            _boundary_output_payload(
                boundary_output,
                mapped_output=(
                    boundary_output.source.node_id in mapped_node_ids
                    if component.mapped_boundary_input_name
                    else None
                ),
            )
            for boundary_output in component.boundary_outputs
        ],
        "execute_node_ids": list(component.execute_node_ids),
        "mapped_execute_node_ids": list(component.mapped_execute_node_ids),
        "static_execute_node_ids": list(component.static_execute_node_ids),
        "extra_data": copy.deepcopy(context.extra_data or {}),
        "requires_volume_reload": context.requires_volume_reload,
        "volume_reload_marker": context.volume_reload_marker,
        "uploaded_volume_paths": list(context.uploaded_volume_paths),
        "terminate_container_on_error": context.terminate_container_on_error,
        "custom_nodes_bundle": context.custom_nodes_bundle_path,
        "mapped_input": (
            {
                "proxy_input_name": component.mapped_boundary_input_name,
                "io_type": str(component.mapped_boundary_input_io_type or "*"),
            }
            if component.mapped_boundary_input_name
            else None
        ),
    }
    if remote_session is not None:
        payload["remote_session"] = copy.deepcopy(remote_session)
        payload["clear_remote_session"] = True
    return payload


def _attach_hybrid_phase_metadata(
    payload: dict[str, Any],
    *,
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    parts: _HybridPayloadParts,
) -> None:
    """Attach static and mapped phase descriptions to a mapped payload."""
    payload["static_to_mapped_boundaries"] = parts.static_to_mapped_payloads
    payload["static_phase"] = {
        "component_node_ids": list(component.static_node_ids),
        "subgraph_prompt": _subset_component_prompt(
            context.component_prompt, component.static_node_ids
        ),
        "boundary_inputs": _serialize_boundary_input_specs(
            parts.static_boundary_inputs,
            signature_prompt=context.signature_prompt,
        ),
        "boundary_outputs": [
            _boundary_output_payload(output)
            for output in parts.static_boundary_outputs
        ]
        + parts.static_bridge_outputs,
        "execute_node_ids": list(component.static_execute_node_ids),
    }
    payload["mapped_phase"] = {
        "component_node_ids": list(component.mapped_node_ids),
        "subgraph_prompt": _subset_component_prompt(
            context.component_prompt, component.mapped_node_ids
        ),
        "boundary_inputs": _serialize_boundary_input_specs(
            parts.mapped_boundary_inputs + parts.static_to_mapped_inputs,
            signature_prompt=context.signature_prompt,
        ),
        "boundary_outputs": [
            _boundary_output_payload(output, mapped_output=True)
            for output in parts.mapped_boundary_outputs
        ],
        "execute_node_ids": list(component.mapped_execute_node_ids),
    }


def _hybrid_remote_session(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    remote_session: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the caller session or create one for a split hybrid component."""
    if remote_session is not None:
        return remote_session
    return RemoteSessionHandle(
        session_id=uuid.uuid4().hex,
        prompt_id=(str(context.prompt_id) if context.prompt_id is not None else None),
        owner_component_id=component.representative_node_id,
    ).to_payload()


def _build_split_hybrid_payload(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    parts: _HybridPayloadParts,
    remote_session: dict[str, Any],
) -> dict[str, Any]:
    """Build the paired static and mapped proxy payloads for a hybrid component."""
    static_payload = _build_subgraph_payload(
        context,
        component_id=component.static_node_ids[0],
        component_node_ids=list(component.static_node_ids),
        boundary_inputs=parts.static_boundary_inputs,
        boundary_outputs=[
            _boundary_output_payload(output) for output in parts.static_boundary_outputs
        ]
        + parts.static_bridge_outputs,
        execute_node_ids=list(component.static_execute_node_ids),
        remote_session=remote_session,
    )
    mapped_payload = _build_subgraph_payload(
        context,
        component_id=f"{component.representative_node_id}__mapped",
        component_node_ids=list(component.mapped_node_ids),
        boundary_inputs=parts.mapped_boundary_inputs + parts.static_to_mapped_inputs,
        boundary_outputs=[
            _boundary_output_payload(output) for output in parts.mapped_boundary_outputs
        ],
        execute_node_ids=list(component.mapped_execute_node_ids),
        remote_session=remote_session,
        clear_remote_session=True,
        mapped_progress_display_node_id=component.static_node_ids[0],
    )
    mapped_payload["static_to_mapped_boundaries"] = parts.static_to_mapped_payloads
    mapped_payload["static_phase"] = {
        "component_node_ids": list(component.static_node_ids),
        "subgraph_prompt": _subset_component_prompt(
            context.component_prompt, component.static_node_ids
        ),
        "boundary_inputs": _serialize_boundary_input_specs(
            parts.static_boundary_inputs,
            signature_prompt=context.signature_prompt,
        ),
        "boundary_outputs": copy.deepcopy(parts.static_bridge_outputs),
        "execute_node_ids": list(
            dict.fromkeys(
                boundary.source.node_id
                for boundary in component.static_to_mapped_boundaries
            )
        ),
    }
    return {"split_proxy_payloads": {"static": static_payload, "mapped": mapped_payload}}


def _log_component_payload(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    payload: dict[str, Any],
) -> None:
    """Log the stable payload summary used during queue-time diagnostics."""
    logger.info(
        "Built remote payload for component %s: boundary_inputs=%d boundary_outputs=%d execute_nodes=%s custom_nodes_bundle=%s",
        component.representative_node_id,
        len(payload["boundary_inputs"]),
        len(payload["boundary_outputs"]),
        payload["execute_node_ids"],
        payload["custom_nodes_bundle"],
    )
    logger.info(
        "Remote payload for component %s requires_volume_reload=%s volume_reload_marker=%s",
        component.representative_node_id,
        context.requires_volume_reload,
        context.volume_reload_marker,
    )


def _build_component_payload(
    component: RemoteComponentPlan,
    component_prompt: dict[str, Any],
    signature_prompt: dict[str, Any],
    extra_data: dict[str, Any] | None,
    settings: ModalSyncSettings,
    requires_volume_reload: bool,
    volume_reload_marker: str | None,
    custom_nodes_bundle: SyncedAsset | None,
    uploaded_volume_paths: list[str],
    terminate_container_on_error: bool,
    nodes_module: Any,
    remote_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the serialized execution payload for one remote component."""
    context = _PayloadBuildContext(
        component_prompt=component_prompt,
        signature_prompt=signature_prompt,
        prompt_id=(extra_data or {}).get("prompt_id"),
        extra_data=extra_data,
        settings=settings,
        requires_volume_reload=requires_volume_reload,
        volume_reload_marker=volume_reload_marker,
        custom_nodes_bundle_path=(
            custom_nodes_bundle.remote_path
            if custom_nodes_bundle is not None
            else None
        ),
        uploaded_volume_paths=uploaded_volume_paths,
        terminate_container_on_error=terminate_container_on_error,
        nodes_module=nodes_module,
    )
    split_phases = _build_phase_payloads_for_transportable_splits(
        component, context, remote_session
    )
    if split_phases is not None:
        return _attach_snapshot_profile_key(
            {"split_proxy_payloads": split_phases}, settings
        )

    payload = _build_base_component_payload(component, context, remote_session)
    _log_component_payload(component, context, payload)
    if not component.mapped_boundary_input_name:
        return _attach_snapshot_profile_key(payload, settings)

    parts = _hybrid_payload_parts(component)
    _attach_hybrid_phase_metadata(
        payload, component=component, context=context, parts=parts
    )
    if not component.static_node_ids:
        return _attach_snapshot_profile_key(payload, settings)

    active_session = _hybrid_remote_session(component, context, remote_session)
    logger.info(
        "Split hybrid component %s into static nodes=%s and mapped nodes=%s using remote_session session_id=%s with %d static bridge outputs.",
        component.representative_node_id,
        component.static_node_ids,
        component.mapped_node_ids,
        active_session["session_id"],
        len(parts.static_bridge_outputs),
    )
    split_payload = _build_split_hybrid_payload(
        component, context, parts, active_session
    )
    return _attach_snapshot_profile_key(split_payload, settings)
