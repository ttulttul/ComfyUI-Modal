"""Remote component payload construction and proxy prompt rewriting."""

from __future__ import annotations

import copy
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

if __package__:
    from .component_planning import (
        _boundary_output_payload,
        _component_ancestors_of_local_source,
        _component_has_local_reentry_dependency,
        _component_has_parallel_local_remote_fanout,
        _component_upstream_closure,
        _filter_boundary_inputs_for_node_ids,
        _filter_boundary_outputs_for_node_ids,
        _order_execute_node_ids_for_transportable_splits,
        _proxy_boundary_output_is_list,
        _subgraph_topological_node_order,
        _subset_component_prompt,
    )
    from .execution_environments import ExecutionProvider
    from .modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        ensure_modal_parallel_local_passthrough_registered,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from .prompt_payload_metadata import (
        _attach_snapshot_profile_key,
        _serialize_boundary_input_specs,
    )
    from .remote_graph_analysis import (
        _build_consumer_map,
        _downstream_node_ids_from_targets,
        _is_link,
        _iter_payload_input_strings,
        _node_output_refs,
        _normalize_output_metadata,
        _remote_output_io_type,
        _remote_output_is_list,
    )
    from .remote_plan_types import (
        BoundaryInputSpec,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        ProducedPhaseOutputSpec,
        RemoteComponentPlan,
    )
    from .session_state import RemoteSessionHandle
    from .settings import ModalSyncSettings
    from .sync_engine import AssetSyncRequestCache, ModalAssetSyncEngine, SyncedAsset
else:  # pragma: no cover - flat import inside the Modal container.
    from component_planning import (
        _boundary_output_payload,
        _component_ancestors_of_local_source,
        _component_has_local_reentry_dependency,
        _component_has_parallel_local_remote_fanout,
        _component_upstream_closure,
        _filter_boundary_inputs_for_node_ids,
        _filter_boundary_outputs_for_node_ids,
        _order_execute_node_ids_for_transportable_splits,
        _proxy_boundary_output_is_list,
        _subgraph_topological_node_order,
        _subset_component_prompt,
    )
    from execution_environments import ExecutionProvider
    from modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        ensure_modal_parallel_local_passthrough_registered,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from prompt_payload_metadata import (
        _attach_snapshot_profile_key,
        _serialize_boundary_input_specs,
    )
    from remote_graph_analysis import (
        _build_consumer_map,
        _downstream_node_ids_from_targets,
        _is_link,
        _iter_payload_input_strings,
        _node_output_refs,
        _normalize_output_metadata,
        _remote_output_io_type,
        _remote_output_is_list,
    )
    from remote_plan_types import (
        BoundaryInputSpec,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        ProducedPhaseOutputSpec,
        RemoteComponentPlan,
    )
    from session_state import RemoteSessionHandle
    from settings import ModalSyncSettings
    from sync_engine import AssetSyncRequestCache, ModalAssetSyncEngine, SyncedAsset

logger = logging.getLogger(__name__)

_SPECULATIVE_PREWARM_TARGET_KEY = "speculative_remote_prewarm_target"
_SPECULATIVE_PREWARM_PAYLOAD_FIELDS = frozenset(
    {
        "component_id",
        "prompt_id",
        "modal_gpu",
        "execution_provider",
        "execution_environment_id",
        "remote_worker_affinity_group",
        "remote_local_gap_pool",
        "subgraph_prompt",
        "requires_volume_reload",
        "volume_reload_marker",
        "uploaded_volume_paths",
        "custom_nodes_bundle",
        "snapshot_profile_key",
    }
)


def _sync_component_prompt_inputs(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    sync_engine: ModalAssetSyncEngine,
    request_cache: AssetSyncRequestCache,
    status_callback: Any | None = None,
) -> tuple[dict[str, Any], list[SyncedAsset]]:
    """Build a synced prompt payload for one remote component."""
    component_prompt: dict[str, Any] = {}
    synced_assets: list[SyncedAsset] = []
    logger.info(
        "Syncing prompt inputs for remote component %s with %d nodes.",
        component.representative_node_id,
        len(component.node_ids),
    )
    for node_id in component.node_ids:
        prompt_node = rewritten_prompt[node_id]
        synced_inputs, node_assets = sync_engine.sync_prompt_inputs(
            copy.deepcopy(prompt_node.get("inputs", {})),
            status_callback=status_callback,
            request_cache=request_cache,
        )
        synced_assets.extend(node_assets)
        component_prompt[node_id] = {
            "class_type": str(prompt_node["class_type"]),
            "inputs": synced_inputs,
            "_meta": copy.deepcopy(prompt_node.get("_meta", {})),
        }
        logger.info(
            "Prepared remote node %s (%s) with %d synced assets.",
            node_id,
            component_prompt[node_id]["class_type"],
            len(node_assets),
        )
    logger.info(
        "Finished syncing remote component %s with %d total synced assets.",
        component.representative_node_id,
        len(synced_assets),
    )
    return component_prompt, synced_assets


def _deduplicate_synced_assets(synced_assets: list[SyncedAsset]) -> list[SyncedAsset]:
    """Return one request summary record per content-addressed remote path."""
    unique_assets_by_remote_path: dict[str, SyncedAsset] = {}
    for synced_asset in synced_assets:
        unique_assets_by_remote_path.setdefault(synced_asset.remote_path, synced_asset)
    return list(unique_assets_by_remote_path.values())


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


def _build_phase_payloads_for_transportable_splits(
    component: RemoteComponentPlan,
    context: _PayloadBuildContext,
    remote_session: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Return ordered phases for local feedback or parallel local fanout."""
    if len(component.execute_node_ids) <= 1:
        return None
    has_local_reentry_dependency = _component_has_local_reentry_dependency(
        prompt=context.signature_prompt,
        component=component,
    )
    has_parallel_local_remote_fanout = _component_has_parallel_local_remote_fanout(
        component
    )
    if not has_local_reentry_dependency and not has_parallel_local_remote_fanout:
        logger.info(
            "Keeping remote component %s as one proxy because execute targets %s "
            "have neither a local re-entry dependency nor parallel local/remote fanout.",
            component.representative_node_id,
            component.execute_node_ids,
        )
        return None
    if component.local_tap_node_ids:
        logger.info(
            "Allowing remote component %s with local tap nodes %s to split because it has a local re-entry dependency.",
            component.representative_node_id,
            component.local_tap_node_ids,
        )
    if has_parallel_local_remote_fanout:
        logger.info(
            "Forcing ordered phases for remote component %s because a remote output "
            "feeds both a non-returning local branch and later remote execution.",
            component.representative_node_id,
        )

    component_node_id_set = set(component.node_ids)
    topological_node_ids = _subgraph_topological_node_order(
        context.component_prompt, component_node_id_set
    )
    remaining_node_ids = set(component.node_ids)
    remaining_execute_node_ids = [
        node_id
        for node_id in topological_node_ids
        if node_id in set(component.execute_node_ids)
    ]
    remaining_execute_node_ids = _order_execute_node_ids_for_transportable_splits(
        prompt=context.signature_prompt,
        component_prompt=context.component_prompt,
        component_node_ids=component_node_id_set,
        execute_node_ids=remaining_execute_node_ids,
    )
    if len(remaining_execute_node_ids) <= 1:
        return None

    phase_payloads: list[dict[str, Any]] = []
    produced_outputs_by_source: dict[LinkedOutputRef, ProducedPhaseOutputSpec] = {}
    local_boundary_outputs_by_source = {
        boundary_output.source: boundary_output
        for boundary_output in component.boundary_outputs
    }
    bridge_output_counter = 0

    while remaining_execute_node_ids:
        target_node_id = remaining_execute_node_ids[0]
        phase_node_ids = sorted(
            _component_upstream_closure(
                prompt=context.component_prompt,
                seed_node_ids={target_node_id},
                candidate_node_ids=remaining_node_ids,
            )
        )
        phase_node_id_set = set(phase_node_ids)
        if not phase_node_ids:
            raise ModalPromptValidationError(
                f"Unable to derive split phase nodes for remote component {component.representative_node_id}."
            )

        phase_boundary_inputs = _filter_boundary_inputs_for_node_ids(
            component.boundary_inputs,
            phase_node_id_set,
        )
        phase_boundary_inputs_by_name = {
            boundary_input.proxy_input_name: boundary_input
            for boundary_input in phase_boundary_inputs
        }
        for phase_node_id in phase_node_ids:
            prompt_node = context.component_prompt.get(phase_node_id)
            if prompt_node is None:
                continue
            for input_name, input_value in (
                prompt_node.get("inputs") or {}
            ).items():
                if not _is_link(input_value):
                    continue
                source = LinkedOutputRef(
                    node_id=str(input_value[0]),
                    output_index=int(input_value[1]),
                )
                if source.node_id in phase_node_id_set:
                    continue
                produced_output = produced_outputs_by_source.get(source)
                if produced_output is None:
                    continue
                boundary_input = phase_boundary_inputs_by_name.get(
                    produced_output.proxy_output_name
                )
                if boundary_input is None:
                    boundary_input = BoundaryInputSpec(
                        proxy_input_name=produced_output.proxy_output_name,
                        source=source,
                        io_type=produced_output.io_type,
                    )
                    phase_boundary_inputs.append(boundary_input)
                    phase_boundary_inputs_by_name[
                        boundary_input.proxy_input_name
                    ] = boundary_input
                boundary_input.targets.append(
                    InputTarget(node_id=phase_node_id, input_name=str(input_name))
                )

        phase_boundary_outputs: list[dict[str, Any]] = []
        phase_output_names_by_source: dict[LinkedOutputRef, str] = {}
        for boundary_output in _filter_boundary_outputs_for_node_ids(
            component.boundary_outputs,
            phase_node_id_set,
        ):
            phase_boundary_outputs.append(_boundary_output_payload(boundary_output))
            phase_output_names_by_source[
                boundary_output.source
            ] = boundary_output.proxy_output_name

        pending_node_ids = remaining_node_ids - phase_node_id_set
        for pending_node_id in sorted(pending_node_ids):
            prompt_node = context.component_prompt.get(pending_node_id)
            if prompt_node is None:
                continue
            for input_value in (prompt_node.get("inputs") or {}).values():
                if not _is_link(input_value):
                    continue
                source = LinkedOutputRef(
                    node_id=str(input_value[0]),
                    output_index=int(input_value[1]),
                )
                if (
                    source.node_id not in phase_node_id_set
                    or source in produced_outputs_by_source
                ):
                    continue
                local_boundary_output = local_boundary_outputs_by_source.get(source)
                io_type = (
                    local_boundary_output.io_type
                    if local_boundary_output is not None
                    else str(
                        _remote_output_io_type(
                            prompt=context.component_prompt,
                            node_id=source.node_id,
                            output_index=source.output_index,
                            nodes_module=context.nodes_module,
                        )
                        or "*"
                    )
                )
                is_list = (
                    local_boundary_output.is_list
                    if local_boundary_output is not None
                    else _remote_output_is_list(
                        prompt=context.component_prompt,
                        node_id=source.node_id,
                        output_index=source.output_index,
                        nodes_module=context.nodes_module,
                    )
                )
                proxy_output_name = phase_output_names_by_source.get(source)
                if proxy_output_name is None:
                    proxy_output_name = f"phase_bridge_{bridge_output_counter}"
                    bridge_output_counter += 1
                    phase_boundary_outputs.append(
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
                    phase_output_names_by_source[source] = proxy_output_name
                produced_outputs_by_source[source] = ProducedPhaseOutputSpec(
                    proxy_output_name=proxy_output_name,
                    source=source,
                    io_type=io_type,
                    is_list=is_list,
                    session_output=True,
                )

        phase_execute_node_ids = [
            node_id
            for node_id in component.execute_node_ids
            if node_id in phase_node_id_set
        ]
        phase_payloads.append(
            _build_subgraph_payload(
                context,
                component_id=str(target_node_id),
                component_node_ids=phase_node_ids,
                boundary_inputs=phase_boundary_inputs,
                boundary_outputs=phase_boundary_outputs,
                execute_node_ids=phase_execute_node_ids,
            )
        )
        remaining_node_ids -= phase_node_id_set
        remaining_execute_node_ids = [
            node_id
            for node_id in remaining_execute_node_ids
            if node_id not in phase_node_id_set
        ]

    has_session_bridges = any(
        bool(boundary_output.get("session_output"))
        for phase_payload in phase_payloads
        for boundary_output in phase_payload.get("boundary_outputs", [])
    )
    if not phase_payloads or len(phase_payloads) <= 1:
        return None
    active_remote_session = remote_session
    if active_remote_session is None and has_session_bridges:
        active_remote_session = RemoteSessionHandle(
            session_id=uuid.uuid4().hex,
            prompt_id=(
                str(context.prompt_id) if context.prompt_id is not None else None
            ),
            owner_component_id=component.representative_node_id,
        ).to_payload()
    if active_remote_session is not None:
        for phase_index, phase_payload in enumerate(phase_payloads):
            phase_payload["remote_session"] = copy.deepcopy(active_remote_session)
            if phase_index == len(phase_payloads) - 1:
                phase_payload["clear_remote_session"] = True
    logger.info(
        "Split ordinary remote component %s into ordered phases: %s",
        component.representative_node_id,
        [
            {
                "component_id": phase_payload["component_id"],
                "component_node_ids": phase_payload["component_node_ids"],
                "execute_node_ids": phase_payload["execute_node_ids"],
            }
            for phase_payload in phase_payloads
        ],
    )
    return phase_payloads



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
    prompt_id = (extra_data or {}).get("prompt_id")
    custom_nodes_bundle_path = (
        custom_nodes_bundle.remote_path if custom_nodes_bundle is not None else None
    )
    context = _PayloadBuildContext(
        component_prompt=component_prompt,
        signature_prompt=signature_prompt,
        prompt_id=prompt_id,
        extra_data=extra_data,
        settings=settings,
        requires_volume_reload=requires_volume_reload,
        volume_reload_marker=volume_reload_marker,
        custom_nodes_bundle_path=custom_nodes_bundle_path,
        uploaded_volume_paths=uploaded_volume_paths,
        terminate_container_on_error=terminate_container_on_error,
        nodes_module=nodes_module,
    )

    split_phase_payloads = _build_phase_payloads_for_transportable_splits(
        component,
        context,
        remote_session,
    )
    if split_phase_payloads is not None:
        return _attach_snapshot_profile_key(
            {"split_proxy_payloads": split_phase_payloads},
            settings,
        )

    payload = {
        "payload_kind": "mapped_subgraph"
        if component.mapped_boundary_input_name
        else "subgraph",
        "component_id": component.representative_node_id,
        "prompt_id": prompt_id,
        "modal_gpu": settings.modal_gpu,
        "remote_worker_affinity_group": _remote_worker_affinity_group(
            context,
            list(component.node_ids)
        ),
        "component_node_ids": list(component.node_ids),
        "subgraph_prompt": component_prompt,
        "boundary_inputs": _serialize_boundary_input_specs(
            component.boundary_inputs,
            signature_prompt=signature_prompt,
        ),
        "boundary_outputs": [
            _boundary_output_payload(
                boundary_output,
                mapped_output=(
                    boundary_output.source.node_id in set(component.mapped_node_ids)
                    if component.mapped_boundary_input_name
                    else None
                ),
            )
            for boundary_output in component.boundary_outputs
        ],
        "execute_node_ids": list(component.execute_node_ids),
        "mapped_execute_node_ids": list(component.mapped_execute_node_ids),
        "static_execute_node_ids": list(component.static_execute_node_ids),
        "extra_data": copy.deepcopy(extra_data or {}),
        "requires_volume_reload": requires_volume_reload,
        "volume_reload_marker": volume_reload_marker,
        "uploaded_volume_paths": list(uploaded_volume_paths),
        "terminate_container_on_error": terminate_container_on_error,
        "custom_nodes_bundle": custom_nodes_bundle_path,
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
        requires_volume_reload,
        volume_reload_marker,
    )
    if component.mapped_boundary_input_name:
        static_node_id_set = set(component.static_node_ids)
        mapped_node_id_set = set(component.mapped_node_ids)
        static_boundary_inputs = _filter_boundary_inputs_for_node_ids(
            component.boundary_inputs,
            static_node_id_set,
        )
        mapped_boundary_inputs = _filter_boundary_inputs_for_node_ids(
            component.boundary_inputs,
            mapped_node_id_set,
        )
        static_boundary_outputs = _filter_boundary_outputs_for_node_ids(
            component.boundary_outputs,
            static_node_id_set,
        )
        mapped_boundary_outputs = _filter_boundary_outputs_for_node_ids(
            component.boundary_outputs,
            mapped_node_id_set,
        )
        static_bridge_outputs = [
            {
                "proxy_output_name": boundary_spec.proxy_name,
                "node_id": boundary_spec.source.node_id,
                "output_index": boundary_spec.source.output_index,
                "io_type": boundary_spec.io_type,
                "is_list": boundary_spec.is_list,
                "preview_target_node_ids": [],
                "session_output": True,
            }
            for boundary_spec in component.static_to_mapped_boundaries
        ]
        payload["static_to_mapped_boundaries"] = [
            {
                "proxy_name": boundary_spec.proxy_name,
                "node_id": boundary_spec.source.node_id,
                "output_index": boundary_spec.source.output_index,
                "io_type": boundary_spec.io_type,
                "is_list": boundary_spec.is_list,
                "targets": [
                    {"node_id": target.node_id, "input_name": target.input_name}
                    for target in boundary_spec.targets
                ],
            }
            for boundary_spec in component.static_to_mapped_boundaries
        ]
        payload["static_phase"] = {
            "component_node_ids": list(component.static_node_ids),
            "subgraph_prompt": _subset_component_prompt(
                component_prompt, component.static_node_ids
            ),
            "boundary_inputs": _serialize_boundary_input_specs(
                static_boundary_inputs,
                signature_prompt=signature_prompt,
            ),
            "boundary_outputs": [
                _boundary_output_payload(boundary_output)
                for boundary_output in static_boundary_outputs
            ]
            + static_bridge_outputs,
            "execute_node_ids": list(component.static_execute_node_ids),
        }
        payload["mapped_phase"] = {
            "component_node_ids": list(component.mapped_node_ids),
            "subgraph_prompt": _subset_component_prompt(
                component_prompt, component.mapped_node_ids
            ),
            "boundary_inputs": _serialize_boundary_input_specs(
                mapped_boundary_inputs
                + [
                    BoundaryInputSpec(
                        proxy_input_name=boundary_spec.proxy_name,
                        source=boundary_spec.source,
                        io_type=boundary_spec.io_type,
                        targets=list(boundary_spec.targets),
                    )
                    for boundary_spec in component.static_to_mapped_boundaries
                ],
                signature_prompt=signature_prompt,
            ),
            "boundary_outputs": [
                _boundary_output_payload(boundary_output, mapped_output=True)
                for boundary_output in mapped_boundary_outputs
            ],
            "execute_node_ids": list(component.mapped_execute_node_ids),
        }
        if not component.static_node_ids:
            return _attach_snapshot_profile_key(payload, settings)
        if remote_session is None:
            remote_session = RemoteSessionHandle(
                session_id=uuid.uuid4().hex,
                prompt_id=(str(prompt_id) if prompt_id is not None else None),
                owner_component_id=component.representative_node_id,
            ).to_payload()
        logger.info(
            "Split hybrid component %s into static nodes=%s and mapped nodes=%s using remote_session session_id=%s with %d static bridge outputs.",
            component.representative_node_id,
            component.static_node_ids,
            component.mapped_node_ids,
            remote_session["session_id"],
            len(static_bridge_outputs),
        )
        payload = {
            "split_proxy_payloads": {
                "static": _build_subgraph_payload(
                    context,
                    component_id=component.static_node_ids[0],
                    component_node_ids=list(component.static_node_ids),
                    boundary_inputs=static_boundary_inputs,
                    boundary_outputs=[
                        _boundary_output_payload(boundary_output)
                        for boundary_output in static_boundary_outputs
                    ]
                    + static_bridge_outputs,
                    execute_node_ids=list(component.static_execute_node_ids),
                    remote_session=remote_session,
                ),
                "mapped": _build_subgraph_payload(
                    context,
                    component_id=f"{component.representative_node_id}__mapped",
                    component_node_ids=list(component.mapped_node_ids),
                    boundary_inputs=mapped_boundary_inputs
                    + [
                        BoundaryInputSpec(
                            proxy_input_name=boundary_spec.proxy_name,
                            source=boundary_spec.source,
                            io_type=boundary_spec.io_type,
                            targets=list(boundary_spec.targets),
                        )
                        for boundary_spec in component.static_to_mapped_boundaries
                    ],
                    boundary_outputs=[
                        _boundary_output_payload(boundary_output)
                        for boundary_output in mapped_boundary_outputs
                    ],
                    execute_node_ids=list(component.mapped_execute_node_ids),
                    remote_session=remote_session,
                    clear_remote_session=True,
                    mapped_progress_display_node_id=component.static_node_ids[0],
                ),
            }
        }
        payload["split_proxy_payloads"]["mapped"]["static_to_mapped_boundaries"] = [
            {
                "proxy_name": boundary_spec.proxy_name,
                "node_id": boundary_spec.source.node_id,
                "output_index": boundary_spec.source.output_index,
                "io_type": boundary_spec.io_type,
                "is_list": boundary_spec.is_list,
                "targets": [
                    {"node_id": target.node_id, "input_name": target.input_name}
                    for target in boundary_spec.targets
                ],
            }
            for boundary_spec in component.static_to_mapped_boundaries
        ]
        payload["split_proxy_payloads"]["mapped"]["static_phase"] = {
            "component_node_ids": list(component.static_node_ids),
            "subgraph_prompt": _subset_component_prompt(
                component_prompt, component.static_node_ids
            ),
            "boundary_inputs": _serialize_boundary_input_specs(
                static_boundary_inputs,
                signature_prompt=signature_prompt,
            ),
            "boundary_outputs": copy.deepcopy(static_bridge_outputs),
            "execute_node_ids": list(
                dict.fromkeys(
                    boundary_spec.source.node_id
                    for boundary_spec in component.static_to_mapped_boundaries
                )
            ),
        }
    return _attach_snapshot_profile_key(payload, settings)


def _component_uploaded_volume_paths(
    *,
    component_prompt: dict[str, Any],
    synced_assets: list[SyncedAsset],
    custom_nodes_bundle: SyncedAsset | None,
) -> list[str]:
    """Return newly uploaded mounted-volume paths that this component can actually reference."""
    referenced_paths: set[str] = set()

    for prompt_node in component_prompt.values():
        if not isinstance(prompt_node, dict):
            continue
        inputs = prompt_node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for input_value in inputs.values():
            for candidate_path in _iter_payload_input_strings(input_value):
                if isinstance(candidate_path, str) and candidate_path.startswith("/"):
                    referenced_paths.add(candidate_path)

    uploaded_paths = {
        asset.remote_path
        for asset in synced_assets
        if asset.uploaded and asset.remote_path in referenced_paths
    }
    if custom_nodes_bundle is not None and custom_nodes_bundle.uploaded:
        uploaded_paths.add(custom_nodes_bundle.remote_path)
    return sorted(uploaded_paths)


def _contains_output_node(
    node_ids: list[str],
    *,
    rewritten_prompt: dict[str, Any],
    nodes_module: Any,
) -> bool:
    """Return whether one node subset contains an output node."""
    for node_id in node_ids:
        prompt_node = rewritten_prompt.get(node_id)
        if prompt_node is None:
            continue
        node_class = nodes_module.NODE_CLASS_MAPPINGS.get(
            str(prompt_node["class_type"])
        )
        if node_class is not None and getattr(node_class, "OUTPUT_NODE", False):
            return True
    return False


def _proxy_inputs_for_boundary_inputs(
    boundary_inputs: list[dict[str, Any]],
    *,
    rewritten_prompt: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one proxy input mapping from the current prompt graph."""
    proxy_inputs: dict[str, Any] = {}
    for boundary_input in boundary_inputs:
        current_input_value: Any = None
        for target in boundary_input.get("targets", []):
            target_prompt_node = rewritten_prompt.get(str(target["node_id"]))
            if target_prompt_node is None:
                continue
            target_input_value = (target_prompt_node.get("inputs") or {}).get(
                str(target["input_name"])
            )
            if _is_link(target_input_value):
                current_input_value = list(target_input_value)
                break
        if current_input_value is None:
            raise ModalPromptValidationError(
                "Unable to resolve proxy boundary input wiring while rewriting split Modal proxies."
            )
        proxy_inputs[str(boundary_input["proxy_input_name"])] = current_input_value
    return proxy_inputs


def _register_proxy_node(
    *,
    prompt_node_id: str,
    payload_mapping: dict[str, Any],
    proxy_inputs: dict[str, Any],
    meta: dict[str, Any],
    is_output_node: bool,
    rewritten_prompt: dict[str, Any],
    nodes_module: Any,
) -> None:
    """Insert one dynamic proxy node into the rewritten prompt."""
    boundary_outputs = list(payload_mapping.get("boundary_outputs", []))
    proxy_node_id = ensure_modal_component_proxy_node_registered(
        output_types=tuple(str(output["io_type"]) for output in boundary_outputs),
        output_names=tuple(
            str(output["proxy_output_name"]) for output in boundary_outputs
        ),
        output_is_list=tuple(
            _proxy_boundary_output_is_list(output) for output in boundary_outputs
        ),
        nodes_module=nodes_module,
        is_output_node=is_output_node,
        include_completion_output=True,
    )
    proxy_inputs["original_node_data"] = register_cache_friendly_proxy_payload(
        prompt_node_id,
        payload_mapping,
    )
    rewritten_prompt[prompt_node_id] = {
        "class_type": proxy_node_id,
        "inputs": proxy_inputs,
        "_meta": copy.deepcopy(meta),
    }


def _rewrite_component_into_proxy(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    payload: dict[str, Any],
    nodes_module: Any,
) -> list[str]:
    """Replace a remote component with a single proxy node in the prompt."""

    split_proxy_payloads = payload.get("split_proxy_payloads")
    component_node_id_set = set(component.node_ids)
    if isinstance(split_proxy_payloads, list):
        phase_payloads = [dict(phase_payload) for phase_payload in split_proxy_payloads]
        phase_proxy_node_ids: list[str] = []
        produced_output_indices_by_name: dict[str, list[Any]] = {}
        replacement_output_indices: dict[LinkedOutputRef, list[Any]] = {}
        materializer_output_by_consumer_and_source: dict[
            tuple[str, LinkedOutputRef], list[Any]
        ] = {}
        proxy_output_by_consumer_and_source: dict[
            tuple[str, LinkedOutputRef], list[Any]
        ] = {}
        materializer_proxy_output_by_node_id: dict[str, list[Any]] = {}
        boundary_output_specs_by_source = {
            boundary_output.source: boundary_output
            for boundary_output in component.boundary_outputs
        }
        component_proxy_node_ids: set[str] = set()
        phase_proxy_inputs_by_node_id: dict[str, dict[str, Any]] = {}
        phase_proxy_meta_by_node_id: dict[str, dict[str, Any]] = {}

        if any(
            spec.local_materializer_node_id is not None
            for spec in component.boundary_outputs
        ):
            ensure_modal_local_bridge_materializer_registered(nodes_module)

        for phase_payload in phase_payloads:
            phase_proxy_node_id = str(phase_payload["component_id"])
            while (
                phase_proxy_node_id in rewritten_prompt
                and phase_proxy_node_id not in component_node_id_set
            ):
                phase_proxy_node_id = f"{phase_proxy_node_id}_proxy"
            phase_payload["component_id"] = phase_proxy_node_id
            phase_proxy_node_ids.append(phase_proxy_node_id)
            component_proxy_node_ids.add(phase_proxy_node_id)

        for phase_payload in phase_payloads:
            phase_proxy_node_id = str(phase_payload["component_id"])
            phase_proxy_inputs = _proxy_inputs_for_boundary_inputs(
                list(phase_payload["boundary_inputs"]),
                rewritten_prompt=rewritten_prompt,
            )
            for boundary_input in phase_payload.get("boundary_inputs", []):
                proxy_input_name = str(boundary_input["proxy_input_name"])
                produced_output_index = produced_output_indices_by_name.get(
                    proxy_input_name
                )
                if produced_output_index is None:
                    continue
                phase_proxy_inputs[proxy_input_name] = list(produced_output_index)

            first_phase_node_id = str(phase_payload["component_node_ids"][0])
            phase_proxy_meta = copy.deepcopy(
                rewritten_prompt[first_phase_node_id].get("_meta", {})
            )
            phase_proxy_inputs_by_node_id[phase_proxy_node_id] = phase_proxy_inputs
            phase_proxy_meta_by_node_id[phase_proxy_node_id] = phase_proxy_meta

            for output_index, boundary_output in enumerate(
                phase_payload.get("boundary_outputs", [])
            ):
                proxy_output_name = str(boundary_output["proxy_output_name"])
                proxy_output = [phase_proxy_node_id, output_index]
                produced_output_indices_by_name[proxy_output_name] = proxy_output
                source = LinkedOutputRef(
                    node_id=str(boundary_output["node_id"]),
                    output_index=int(boundary_output["output_index"]),
                )
                boundary_output_spec = boundary_output_specs_by_source.get(source)
                if boundary_output_spec is not None:
                    for (
                        consumer_node_id
                    ) in boundary_output_spec.session_consumer_node_ids:
                        proxy_output_by_consumer_and_source[
                            (consumer_node_id, source)
                        ] = proxy_output
                    materializer_node_id = (
                        boundary_output_spec.local_materializer_node_id
                    )
                    if materializer_node_id is not None:
                        materializer_proxy_output_by_node_id[
                            materializer_node_id
                        ] = proxy_output
                        for (
                            consumer_node_id
                        ) in boundary_output_spec.local_materializer_consumer_node_ids:
                            materializer_output_by_consumer_and_source[
                                (consumer_node_id, source)
                            ] = [materializer_node_id, 0]
                if bool(boundary_output.get("session_output")):
                    continue
                replacement_output_indices[source] = proxy_output

        for node_id in component.node_ids:
            rewritten_prompt.pop(node_id, None)

        for phase_payload in phase_payloads:
            phase_proxy_node_id = str(phase_payload["component_id"])
            _register_proxy_node(
                prompt_node_id=phase_proxy_node_id,
                payload_mapping=phase_payload,
                proxy_inputs=phase_proxy_inputs_by_node_id[phase_proxy_node_id],
                meta=phase_proxy_meta_by_node_id[phase_proxy_node_id],
                is_output_node=_contains_output_node(
                    list(phase_payload["component_node_ids"]),
                    rewritten_prompt=rewritten_prompt,
                    nodes_module=nodes_module,
                ),
                rewritten_prompt=rewritten_prompt,
                nodes_module=nodes_module,
            )

        for (
            materializer_node_id,
            proxy_output,
        ) in materializer_proxy_output_by_node_id.items():
            rewritten_prompt[materializer_node_id] = {
                "class_type": MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
                "inputs": {"bridge_ref": list(proxy_output)},
                "_meta": {"title": "Modal Local Bridge Materializer"},
            }

        if component.mapped_boundary_source_node_id is not None:
            mapped_node_id_set = set(component.mapped_node_ids)
            for phase_payload in phase_payloads:
                phase_node_ids = {
                    str(node_id)
                    for node_id in phase_payload.get("component_node_ids", [])
                }
                if not (phase_node_ids & mapped_node_id_set):
                    continue
                register_modal_map_input_warmup_context(
                    component.mapped_boundary_source_node_id,
                    phase_payload,
                    str(component.mapped_boundary_input_io_type or "*"),
                )
                break

        for node_id, prompt_node in list(rewritten_prompt.items()):
            if node_id in component_proxy_node_ids:
                continue
            for input_name, input_value in list(
                (prompt_node.get("inputs") or {}).items()
            ):
                if not _is_link(input_value):
                    continue
                source = LinkedOutputRef(
                    node_id=str(input_value[0]), output_index=int(input_value[1])
                )
                replacement_output = materializer_output_by_consumer_and_source.get(
                    (node_id, source)
                )
                if replacement_output is None:
                    replacement_output = proxy_output_by_consumer_and_source.get(
                        (node_id, source)
                    )
                if replacement_output is None:
                    replacement_output = replacement_output_indices.get(source)
                if replacement_output is not None:
                    prompt_node["inputs"][input_name] = list(replacement_output)

        logger.info(
            "Rewrote remote component %s into ordered proxies %s.",
            component.representative_node_id,
            phase_proxy_node_ids,
        )
        return phase_proxy_node_ids

    if isinstance(split_proxy_payloads, dict):
        static_payload = dict(split_proxy_payloads["static"])
        mapped_payload = dict(split_proxy_payloads["mapped"])
        static_proxy_node_id = str(static_payload["component_id"])
        mapped_proxy_node_id = str(mapped_payload["component_id"])
        while (
            mapped_proxy_node_id in rewritten_prompt
            and mapped_proxy_node_id not in component_node_id_set
        ):
            mapped_proxy_node_id = f"{mapped_proxy_node_id}_proxy"
        mapped_payload["component_id"] = mapped_proxy_node_id

        static_proxy_inputs = _proxy_inputs_for_boundary_inputs(
            list(static_payload["boundary_inputs"]),
            rewritten_prompt=rewritten_prompt,
        )
        static_boundary_outputs = list(static_payload["boundary_outputs"])
        static_proxy_meta = copy.deepcopy(
            rewritten_prompt[static_proxy_node_id].get("_meta", {})
        )
        mapped_proxy_meta = copy.deepcopy(
            rewritten_prompt[component.mapped_node_ids[0]].get("_meta", {})
        )
        bridge_output_indices = {
            str(boundary_output["proxy_output_name"]): output_index
            for output_index, boundary_output in enumerate(static_boundary_outputs)
            if bool(boundary_output.get("session_output"))
        }
        mapped_proxy_inputs = _proxy_inputs_for_boundary_inputs(
            list(mapped_payload["boundary_inputs"]),
            rewritten_prompt=rewritten_prompt,
        )
        mapped_static_phase = mapped_payload.get("static_phase")
        if isinstance(mapped_static_phase, dict):
            for input_name, input_value in _proxy_inputs_for_boundary_inputs(
                list(mapped_static_phase.get("boundary_inputs", [])),
                rewritten_prompt=rewritten_prompt,
            ).items():
                mapped_proxy_inputs.setdefault(input_name, input_value)
        for boundary_input in mapped_payload.get("boundary_inputs", []):
            proxy_input_name = str(boundary_input["proxy_input_name"])
            if proxy_input_name not in bridge_output_indices:
                continue
            mapped_proxy_inputs[proxy_input_name] = [
                static_proxy_node_id,
                bridge_output_indices[proxy_input_name],
            ]

        replacement_output_indices = {
            LinkedOutputRef(
                node_id=str(boundary_output["node_id"]),
                output_index=int(boundary_output["output_index"]),
            ): [static_proxy_node_id, output_index]
            for output_index, boundary_output in enumerate(static_boundary_outputs)
            if not bool(boundary_output.get("session_output"))
        }
        replacement_output_indices.update(
            {
                LinkedOutputRef(
                    node_id=str(boundary_output["node_id"]),
                    output_index=int(boundary_output["output_index"]),
                ): [mapped_proxy_node_id, output_index]
                for output_index, boundary_output in enumerate(
                    mapped_payload.get("boundary_outputs", [])
                )
            }
        )

        for node_id in component.node_ids:
            rewritten_prompt.pop(node_id, None)
        _register_proxy_node(
            prompt_node_id=static_proxy_node_id,
            payload_mapping=static_payload,
            proxy_inputs=static_proxy_inputs,
            meta=static_proxy_meta,
            is_output_node=_contains_output_node(
                component.static_node_ids,
                rewritten_prompt=rewritten_prompt,
                nodes_module=nodes_module,
            ),
            rewritten_prompt=rewritten_prompt,
            nodes_module=nodes_module,
        )
        _register_proxy_node(
            prompt_node_id=mapped_proxy_node_id,
            payload_mapping=mapped_payload,
            proxy_inputs=mapped_proxy_inputs,
            meta=mapped_proxy_meta,
            is_output_node=_contains_output_node(
                component.mapped_node_ids,
                rewritten_prompt=rewritten_prompt,
                nodes_module=nodes_module,
            ),
            rewritten_prompt=rewritten_prompt,
            nodes_module=nodes_module,
        )
        if component.mapped_boundary_source_node_id is not None:
            register_modal_map_input_warmup_context(
                component.mapped_boundary_source_node_id,
                mapped_payload,
                str(component.mapped_boundary_input_io_type or "*"),
            )

        for node_id, prompt_node in list(rewritten_prompt.items()):
            if node_id in {static_proxy_node_id, mapped_proxy_node_id}:
                continue
            for input_name, input_value in list(
                (prompt_node.get("inputs") or {}).items()
            ):
                if not _is_link(input_value):
                    continue
                source = LinkedOutputRef(
                    node_id=str(input_value[0]), output_index=int(input_value[1])
                )
                if source in replacement_output_indices:
                    prompt_node["inputs"][input_name] = list(
                        replacement_output_indices[source]
                    )

        logger.info(
            "Rewrote hybrid remote component %s into static proxy %s and mapped proxy %s.",
            component.representative_node_id,
            static_proxy_node_id,
            mapped_proxy_node_id,
        )
        return [static_proxy_node_id, mapped_proxy_node_id]

    boundary_outputs = list(payload.get("boundary_outputs", []))
    proxy_node_id = ensure_modal_component_proxy_node_registered(
        output_types=tuple(str(output["io_type"]) for output in boundary_outputs),
        output_names=tuple(
            str(output["proxy_output_name"]) for output in boundary_outputs
        ),
        output_is_list=tuple(
            _proxy_boundary_output_is_list(output) for output in boundary_outputs
        ),
        nodes_module=nodes_module,
        is_output_node=component.contains_output_node,
        include_completion_output=True,
    )
    representative_node_id = component.representative_node_id
    proxy_inputs = _proxy_inputs_for_boundary_inputs(
        list(payload.get("boundary_inputs", [])),
        rewritten_prompt=rewritten_prompt,
    )
    proxy_inputs["original_node_data"] = register_cache_friendly_proxy_payload(
        representative_node_id,
        payload,
    )
    representative_meta = copy.deepcopy(
        rewritten_prompt[representative_node_id].get("_meta", {})
    )
    rewritten_prompt[representative_node_id] = {
        "class_type": proxy_node_id,
        "inputs": proxy_inputs,
        "_meta": representative_meta,
    }
    if component.mapped_boundary_source_node_id is not None:
        register_modal_map_input_warmup_context(
            component.mapped_boundary_source_node_id,
            payload,
            str(component.mapped_boundary_input_io_type or "*"),
        )
    materializer_output_by_consumer_and_source: dict[
        tuple[str, LinkedOutputRef], list[Any]
    ] = {}
    proxy_output_by_consumer_and_source: dict[
        tuple[str, LinkedOutputRef], list[Any]
    ] = {}
    default_proxy_output_by_source: dict[LinkedOutputRef, list[Any]] = {}
    if any(
        spec.local_materializer_node_id is not None
        for spec in component.boundary_outputs
    ):
        ensure_modal_local_bridge_materializer_registered(nodes_module)
    for output_index, spec in enumerate(component.boundary_outputs):
        proxy_output = [representative_node_id, output_index]
        default_proxy_output_by_source.setdefault(spec.source, proxy_output)
        for consumer_node_id in spec.session_consumer_node_ids:
            proxy_output_by_consumer_and_source[
                (consumer_node_id, spec.source)
            ] = proxy_output
        materializer_node_id = spec.local_materializer_node_id
        if materializer_node_id is None:
            continue
        rewritten_prompt[materializer_node_id] = {
            "class_type": MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
            "inputs": {"bridge_ref": proxy_output},
            "_meta": {"title": "Modal Local Bridge Materializer"},
        }
        for consumer_node_id in spec.local_materializer_consumer_node_ids:
            materializer_output_by_consumer_and_source[
                (consumer_node_id, spec.source)
            ] = [materializer_node_id, 0]
    for node_id, prompt_node in list(rewritten_prompt.items()):
        if node_id in component_node_id_set and node_id != representative_node_id:
            del rewritten_prompt[node_id]
            continue
        if node_id == representative_node_id:
            continue
        for input_name, input_value in list((prompt_node.get("inputs") or {}).items()):
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(
                node_id=str(input_value[0]), output_index=int(input_value[1])
            )
            replacement_output = materializer_output_by_consumer_and_source.get(
                (node_id, source)
            )
            if replacement_output is None:
                replacement_output = proxy_output_by_consumer_and_source.get(
                    (node_id, source)
                )
            if replacement_output is None:
                replacement_output = default_proxy_output_by_source.get(source)
            if replacement_output is not None:
                prompt_node["inputs"][input_name] = list(replacement_output)
    logger.info(
        "Rewrote remote component %s with %d nodes to Modal proxy %s.",
        representative_node_id,
        len(component.node_ids),
        proxy_node_id,
    )
    return [representative_node_id]


def _modal_component_completion_output_index(
    *,
    proxy_node_id: str,
    rewritten_prompt: dict[str, Any],
    nodes_module: Any,
) -> int:
    """Return the synthetic completion-token output index for one Modal proxy."""
    prompt_node = rewritten_prompt.get(proxy_node_id)
    if prompt_node is None:
        raise ModalPromptValidationError(
            f"Modal proxy node {proxy_node_id!r} is missing before artifact finalization."
        )
    class_type = str(prompt_node.get("class_type"))
    node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        raise ModalPromptValidationError(
            f"Modal proxy class {class_type!r} is not registered before artifact finalization."
        )
    _output_types, output_names, _output_is_list = _normalize_output_metadata(
        node_class
    )
    try:
        return output_names.index(MODAL_COMPONENT_COMPLETION_OUTPUT_NAME)
    except ValueError as exc:
        raise ModalPromptValidationError(
            f"Modal proxy {proxy_node_id!r} does not expose its completion token."
        ) from exc


def _attach_modal_artifact_finalizer(
    *,
    rewritten_prompt: dict[str, Any],
    remote_component_ids: list[str],
    nodes_module: Any,
) -> str:
    """Attach one internal output sink to every rewritten Modal component."""
    if not remote_component_ids:
        raise ModalPromptValidationError(
            "Modal artifact finalization requires at least one remote component."
        )
    if len(remote_component_ids) > MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS:
        raise ModalPromptValidationError(
            "Modal artifact finalization supports at most "
            f"{MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS} remote components per prompt."
        )

    ensure_modal_artifact_finalizer_registered(nodes_module)
    finalizer_node_id = f"__{MODAL_ARTIFACT_FINALIZER_NODE_ID}__"
    while finalizer_node_id in rewritten_prompt:
        finalizer_node_id = f"{finalizer_node_id}_proxy"

    finalizer_inputs = {
        f"components.component_{component_index}": [
            proxy_node_id,
            _modal_component_completion_output_index(
                proxy_node_id=proxy_node_id,
                rewritten_prompt=rewritten_prompt,
                nodes_module=nodes_module,
            ),
        ]
        for component_index, proxy_node_id in enumerate(remote_component_ids)
    }
    rewritten_prompt[finalizer_node_id] = {
        "class_type": MODAL_ARTIFACT_FINALIZER_NODE_ID,
        "inputs": finalizer_inputs,
        "_meta": {"title": "Modal Artifact Finalizer"},
    }
    logger.info(
        "Attached Modal artifact finalizer %s to remote components %s.",
        finalizer_node_id,
        remote_component_ids,
    )
    return finalizer_node_id


def _nearest_downstream_remote_component_ids(
    *,
    rewritten_prompt: dict[str, Any],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    seed_targets: list[InputTarget],
    remote_component_id_set: set[str],
    nodes_module: Any,
) -> list[str]:
    """Return the first remote proxies reached along every downstream branch."""
    nearest_remote_component_ids: set[str] = set()
    visited_node_ids: set[str] = set()
    pending_node_ids = deque(str(target.node_id) for target in seed_targets)
    while pending_node_ids:
        node_id = pending_node_ids.popleft()
        if node_id in visited_node_ids or node_id not in rewritten_prompt:
            continue
        visited_node_ids.add(node_id)
        if node_id in remote_component_id_set:
            nearest_remote_component_ids.add(node_id)
            continue
        for output_ref in _node_output_refs(
            rewritten_prompt,
            node_id,
            nodes_module,
        ):
            for downstream_target in consumers.get(output_ref, []):
                pending_node_ids.append(str(downstream_target.node_id))
    return sorted(nearest_remote_component_ids)


def _parallelize_non_returning_local_branches(
    *,
    rewritten_prompt: dict[str, Any],
    remote_component_ids: list[str],
    nodes_module: Any,
) -> list[str]:
    """Release local-only taps once the nearest remote continuations are in flight."""
    remote_component_id_set = set(remote_component_ids)
    if len(remote_component_id_set) < 2:
        return []

    consumers = _build_consumer_map(rewritten_prompt)
    ensure_modal_parallel_local_passthrough_registered(nodes_module)
    passthrough_node_ids: list[str] = []
    dispatch_group_id = uuid.uuid4().hex
    for component_id in remote_component_ids:
        embedded_payload = (
            rewritten_prompt.get(component_id, {})
            .get("inputs", {})
            .get("original_node_data")
        )
        if not isinstance(embedded_payload, Mapping):
            continue
        payload_prompt_id = embedded_payload.get("prompt_id")
        if payload_prompt_id is not None and str(payload_prompt_id).strip():
            dispatch_group_id = str(payload_prompt_id)
            break

    for source_component_id in remote_component_ids:
        for source in _node_output_refs(
            rewritten_prompt,
            source_component_id,
            nodes_module,
        ):
            output_consumers = list(consumers.get(source, []))
            if not output_consumers:
                continue
            downstream_remote_component_ids = _nearest_downstream_remote_component_ids(
                rewritten_prompt=rewritten_prompt,
                consumers=consumers,
                seed_targets=output_consumers,
                remote_component_id_set=remote_component_id_set,
                nodes_module=nodes_module,
            )
            downstream_remote_component_ids = [
                component_id
                for component_id in downstream_remote_component_ids
                if component_id != source_component_id
            ]
            if not downstream_remote_component_ids:
                continue

            local_only_consumers: list[InputTarget] = []
            for target in output_consumers:
                if target.node_id in remote_component_id_set:
                    continue
                branch_node_ids = _downstream_node_ids_from_targets(
                    prompt=rewritten_prompt,
                    consumers=consumers,
                    seed_targets=[target],
                    nodes_module=nodes_module,
                )
                if branch_node_ids & remote_component_id_set:
                    continue
                local_only_consumers.append(target)
            if not local_only_consumers:
                continue

            passthrough_node_id = (
                f"__{MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID}__"
                f"{len(passthrough_node_ids)}"
            )
            while passthrough_node_id in rewritten_prompt:
                passthrough_node_id = f"{passthrough_node_id}_proxy"
            passthrough_inputs: dict[str, Any] = {
                "value": [source.node_id, source.output_index],
                "dispatch_context": {
                    "dispatch_group_id": dispatch_group_id,
                    "component_ids": downstream_remote_component_ids,
                },
            }
            rewritten_prompt[passthrough_node_id] = {
                "class_type": MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
                "inputs": passthrough_inputs,
                "_meta": {"title": "Modal Parallel Local Branch"},
            }
            for target in local_only_consumers:
                rewritten_prompt[target.node_id]["inputs"][target.input_name] = [
                    passthrough_node_id,
                    0,
                ]
            for component_id in downstream_remote_component_ids:
                prompt_inputs = rewritten_prompt[component_id].get("inputs", {})
                embedded_payload = prompt_inputs.get("original_node_data")
                if not isinstance(embedded_payload, Mapping):
                    continue
                prompt_inputs[
                    "original_node_data"
                ] = update_registered_proxy_payload_fields(
                    component_id,
                    embedded_payload,
                    {
                        "parallel_local_dispatch_group_id": dispatch_group_id,
                        "signal_parallel_local_dispatch": True,
                    },
                )
            passthrough_node_ids.append(passthrough_node_id)
            logger.info(
                "Parallelized local-only consumers %s of remote output %s:%d after downstream Modal dispatches %s via passthrough=%s.",
                [target.node_id for target in local_only_consumers],
                source.node_id,
                source.output_index,
                downstream_remote_component_ids,
                passthrough_node_id,
            )
    return passthrough_node_ids


def _configure_local_gap_keepalive_payloads(
    *,
    rewritten_prompt: dict[str, Any],
    remote_component_ids: list[str],
    sandwiched_local_node_ids: set[str],
) -> None:
    """Mark only remote proxies that precede a local-to-remote execution gap."""
    if not sandwiched_local_node_ids:
        return

    payloads_by_component_id: dict[str, Mapping[str, Any]] = {}
    for component_id in remote_component_ids:
        payload = (
            rewritten_prompt.get(component_id, {})
            .get("inputs", {})
            .get("original_node_data")
        )
        if not isinstance(payload, dict):
            continue
        payloads_by_component_id[component_id] = payload

    keepalive_component_ids: set[str] = set()
    continuation_component_ids: set[str] = set()
    for local_node_id in sandwiched_local_node_ids:
        upstream_component_ids: set[str] = set()
        downstream_component_ids: set[str] = set()
        for component_id in remote_component_ids:
            if _component_ancestors_of_local_source(
                prompt=rewritten_prompt,
                source_node_id=local_node_id,
                component_node_id_set={component_id},
            ):
                upstream_component_ids.add(component_id)
            if _component_ancestors_of_local_source(
                prompt=rewritten_prompt,
                source_node_id=component_id,
                component_node_id_set={local_node_id},
            ):
                downstream_component_ids.add(component_id)

        for upstream_component_id in upstream_component_ids:
            upstream_payload = payloads_by_component_id.get(upstream_component_id)
            if upstream_payload is None:
                continue
            upstream_environment_id = str(
                upstream_payload.get("execution_environment_id") or ""
            )
            if upstream_payload.get("execution_provider") != ExecutionProvider.MODAL.value:
                continue
            for downstream_component_id in downstream_component_ids:
                downstream_payload = payloads_by_component_id.get(
                    downstream_component_id
                )
                if (
                    downstream_payload is None
                    or downstream_payload.get("execution_provider")
                    != ExecutionProvider.MODAL.value
                    or str(downstream_payload.get("execution_environment_id") or "")
                    != upstream_environment_id
                ):
                    continue
                keepalive_component_ids.add(upstream_component_id)
                continuation_component_ids.add(downstream_component_id)

    for component_id, embedded_payload in payloads_by_component_id.items():
        if component_id not in keepalive_component_ids | continuation_component_ids:
            continue
        payload_fields: dict[str, Any] = {"remote_local_gap_pool": True}
        if component_id in keepalive_component_ids:
            payload_fields["keepalive_after_remote_component"] = True
        if component_id in continuation_component_ids:
            payload_fields["stop_local_gap_keepalive_before_remote_component"] = True
        rewritten_prompt[component_id]["inputs"][
            "original_node_data"
        ] = update_registered_proxy_payload_fields(
            component_id,
            embedded_payload,
            payload_fields,
        )
    logger.info(
        "Configured local-gap Modal pool for components=%s keepalive_producers=%s continuations=%s.",
        sorted(keepalive_component_ids | continuation_component_ids),
        sorted(keepalive_component_ids),
        sorted(continuation_component_ids),
    )


def _remote_proxy_payload(
    rewritten_prompt: Mapping[str, Any], component_id: str
) -> Mapping[str, Any] | None:
    """Return the registered execution payload embedded in one remote proxy."""
    prompt_node = rewritten_prompt.get(component_id)
    if not isinstance(prompt_node, Mapping):
        return None
    inputs = prompt_node.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    payload = inputs.get("original_node_data")
    if not isinstance(payload, Mapping):
        return None
    return registered_proxy_execution_payload(component_id, payload)


def _remote_proxy_dependency_edges(
    rewritten_prompt: Mapping[str, Any],
    component_ids: set[str],
) -> dict[str, set[str]]:
    """Return nearest remote dependencies while traversing intervening local nodes."""
    dependency_edges = {component_id: set() for component_id in component_ids}
    for downstream_component_id in component_ids:
        downstream_node = rewritten_prompt.get(downstream_component_id)
        if not isinstance(downstream_node, Mapping):
            continue
        pending_node_ids = [
            str(input_value[0])
            for input_value in (downstream_node.get("inputs") or {}).values()
            if _is_link(input_value)
        ]
        visited_node_ids: set[str] = set()
        while pending_node_ids:
            upstream_node_id = pending_node_ids.pop()
            if upstream_node_id in visited_node_ids:
                continue
            visited_node_ids.add(upstream_node_id)
            if upstream_node_id in component_ids:
                dependency_edges[upstream_node_id].add(downstream_component_id)
                continue
            upstream_node = rewritten_prompt.get(upstream_node_id)
            if not isinstance(upstream_node, Mapping):
                continue
            pending_node_ids.extend(
                str(input_value[0])
                for input_value in (upstream_node.get("inputs") or {}).values()
                if _is_link(input_value)
            )
    return dependency_edges


def _component_descendant_distances(
    component_id: str,
    dependency_edges: Mapping[str, set[str]],
) -> dict[str, int]:
    """Return shortest downstream distances from one remote component."""
    distances: dict[str, int] = {}
    pending = deque(
        (descendant_id, 1)
        for descendant_id in dependency_edges.get(component_id, set())
    )
    while pending:
        descendant_id, distance = pending.popleft()
        previous_distance = distances.get(descendant_id)
        if previous_distance is not None and previous_distance <= distance:
            continue
        distances[descendant_id] = distance
        pending.extend(
            (nested_descendant_id, distance + 1)
            for nested_descendant_id in dependency_edges.get(descendant_id, set())
        )
    return distances


def _speculative_prewarm_target_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the minimal side-effect-free payload needed to prepare one worker."""
    return {
        field_name: copy.deepcopy(payload[field_name])
        for field_name in _SPECULATIVE_PREWARM_PAYLOAD_FIELDS
        if field_name in payload
    }


def _configure_speculative_affinity_prewarm_payloads(
    *,
    rewritten_prompt: dict[str, Any],
    execution_stages: list[list[str]],
) -> None:
    """Attach one next-affinity worker preparation target to each eligible proxy."""
    stage_index_by_component_id = {
        component_id: stage_index
        for stage_index, stage in enumerate(execution_stages)
        for component_id in stage
    }
    payloads_by_component_id = {
        component_id: payload
        for component_id in stage_index_by_component_id
        if (payload := _remote_proxy_payload(rewritten_prompt, component_id))
        is not None
    }
    dependency_edges = _remote_proxy_dependency_edges(
        rewritten_prompt,
        set(payloads_by_component_id),
    )

    configured_targets: dict[str, str] = {}
    for component_id, payload in payloads_by_component_id.items():
        if payload.get("execution_provider") != ExecutionProvider.MODAL.value:
            continue
        execution_environment_id = str(
            payload.get("execution_environment_id") or ""
        )
        affinity_group = (
            str(payload.get("remote_worker_affinity_group") or "comfy").strip().lower()
        )
        descendant_distances = _component_descendant_distances(
            component_id, dependency_edges
        )
        candidate_ids = sorted(
            (
                descendant_id
                for descendant_id in descendant_distances
                if descendant_id in payloads_by_component_id
                and payloads_by_component_id[descendant_id].get(
                    "execution_provider"
                )
                == ExecutionProvider.MODAL.value
                and str(
                    payloads_by_component_id[descendant_id].get(
                        "execution_environment_id"
                    )
                    or ""
                )
                == execution_environment_id
                and str(
                    payloads_by_component_id[descendant_id].get(
                        "remote_worker_affinity_group"
                    )
                    or "comfy"
                )
                .strip()
                .lower()
                != affinity_group
            ),
            key=lambda descendant_id: (
                descendant_distances[descendant_id],
                stage_index_by_component_id[descendant_id],
                descendant_id,
            ),
        )
        if not candidate_ids:
            continue

        target_component_id = candidate_ids[0]
        target_payload = _speculative_prewarm_target_payload(
            payloads_by_component_id[target_component_id]
        )
        rewritten_prompt[component_id]["inputs"][
            "original_node_data"
        ] = update_registered_proxy_payload_fields(
            component_id,
            payload,
            {_SPECULATIVE_PREWARM_TARGET_KEY: target_payload},
        )
        configured_targets[component_id] = target_component_id

    if configured_targets:
        logger.info(
            "Configured one-step speculative Modal worker prewarm targets=%s.",
            configured_targets,
        )
