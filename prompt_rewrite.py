"""Remote component payload construction and proxy prompt rewriting."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Mapping

if __package__:
    from .component_planning import _proxy_boundary_output_is_list
    from .modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
    )
    from .prompt_payload_building import _build_component_payload
    from .prompt_affinity_planning import (
        _SPECULATIVE_PREWARM_PAYLOAD_FIELDS,
        _SPECULATIVE_PREWARM_TARGET_KEY,
        _component_descendant_distances,
        _configure_local_gap_keepalive_payloads,
        _configure_speculative_affinity_prewarm_payloads,
        _nearest_downstream_remote_component_ids,
        _parallelize_non_returning_local_branches,
        _remote_proxy_dependency_edges,
        _remote_proxy_payload,
        _speculative_prewarm_target_payload,
    )
    from .remote_graph_analysis import (
        _is_link,
        _iter_payload_input_strings,
        _normalize_output_metadata,
    )
    from .remote_plan_types import (
        BoundaryOutputSpec,
        LinkedOutputRef,
        ModalPromptValidationError,
        RemoteComponentPlan,
    )
    from .sync_engine import AssetSyncRequestCache, ModalAssetSyncEngine, SyncedAsset
else:  # pragma: no cover - flat import inside the Modal container.
    from component_planning import _proxy_boundary_output_is_list
    from modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
    )
    from prompt_payload_building import _build_component_payload
    from prompt_affinity_planning import (
        _SPECULATIVE_PREWARM_PAYLOAD_FIELDS,
        _SPECULATIVE_PREWARM_TARGET_KEY,
        _component_descendant_distances,
        _configure_local_gap_keepalive_payloads,
        _configure_speculative_affinity_prewarm_payloads,
        _nearest_downstream_remote_component_ids,
        _parallelize_non_returning_local_branches,
        _remote_proxy_dependency_edges,
        _remote_proxy_payload,
        _speculative_prewarm_target_payload,
    )
    from remote_graph_analysis import (
        _is_link,
        _iter_payload_input_strings,
        _normalize_output_metadata,
    )
    from remote_plan_types import (
        BoundaryOutputSpec,
        LinkedOutputRef,
        ModalPromptValidationError,
        RemoteComponentPlan,
    )
    from sync_engine import AssetSyncRequestCache, ModalAssetSyncEngine, SyncedAsset

logger = logging.getLogger(__name__)


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


@dataclass
class _ProxyReplacementMaps:
    """Collect consumer-specific and default proxy output rewrites."""

    materializer_by_consumer: dict[tuple[str, LinkedOutputRef], list[Any]]
    session_by_consumer: dict[tuple[str, LinkedOutputRef], list[Any]]
    default_by_source: dict[LinkedOutputRef, list[Any]]


@dataclass
class _OrderedProxyRewriteState:
    """Collect intermediate state while preparing ordered phase proxies."""

    phase_payloads: list[dict[str, Any]]
    proxy_node_ids: list[str]
    proxy_inputs_by_node_id: dict[str, dict[str, Any]]
    proxy_meta_by_node_id: dict[str, dict[str, Any]]
    replacements: _ProxyReplacementMaps
    materializer_proxy_outputs: dict[str, list[Any]]


def _empty_proxy_replacement_maps() -> _ProxyReplacementMaps:
    """Return empty output replacement maps for one component rewrite."""
    return _ProxyReplacementMaps(
        materializer_by_consumer={},
        session_by_consumer={},
        default_by_source={},
    )


def _linked_output_ref(payload: Mapping[str, Any]) -> LinkedOutputRef:
    """Return the typed source reference encoded in one boundary payload."""
    return LinkedOutputRef(
        node_id=str(payload["node_id"]),
        output_index=int(payload["output_index"]),
    )


def _unique_proxy_node_id(
    candidate: str,
    *,
    rewritten_prompt: dict[str, Any],
    component_node_ids: set[str],
) -> str:
    """Return a proxy id that does not overwrite an unrelated prompt node."""
    while candidate in rewritten_prompt and candidate not in component_node_ids:
        candidate = f"{candidate}_proxy"
    return candidate


def _install_local_bridge_materializer(
    rewritten_prompt: dict[str, Any],
    materializer_node_id: str,
    proxy_output: list[Any],
) -> None:
    """Insert one local bridge materializer wired to a proxy output."""
    rewritten_prompt[materializer_node_id] = {
        "class_type": MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        "inputs": {"bridge_ref": list(proxy_output)},
        "_meta": {"title": "Modal Local Bridge Materializer"},
    }


def _replacement_output(
    replacements: _ProxyReplacementMaps,
    *,
    consumer_node_id: str,
    source: LinkedOutputRef,
) -> list[Any] | None:
    """Resolve the most specific proxy output for one surviving consumer."""
    replacement = replacements.materializer_by_consumer.get(
        (consumer_node_id, source)
    )
    if replacement is None:
        replacement = replacements.session_by_consumer.get(
            (consumer_node_id, source)
        )
    if replacement is None:
        replacement = replacements.default_by_source.get(source)
    return replacement


def _rewire_proxy_consumers(
    rewritten_prompt: dict[str, Any],
    *,
    skip_node_ids: set[str],
    replacements: _ProxyReplacementMaps,
) -> None:
    """Rewrite surviving prompt links to the selected proxy outputs."""
    for node_id, prompt_node in list(rewritten_prompt.items()):
        if node_id in skip_node_ids:
            continue
        for input_name, input_value in list((prompt_node.get("inputs") or {}).items()):
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(
                node_id=str(input_value[0]), output_index=int(input_value[1])
            )
            replacement = _replacement_output(
                replacements,
                consumer_node_id=node_id,
                source=source,
            )
            if replacement is not None:
                prompt_node["inputs"][input_name] = list(replacement)


def _record_boundary_output_routes(
    spec: BoundaryOutputSpec,
    proxy_output: list[Any],
    replacements: _ProxyReplacementMaps,
    materializer_proxy_outputs: dict[str, list[Any]],
) -> None:
    """Record session and local-materializer routes for one boundary output."""
    for consumer_node_id in spec.session_consumer_node_ids:
        replacements.session_by_consumer[(consumer_node_id, spec.source)] = (
            proxy_output
        )
    materializer_node_id = spec.local_materializer_node_id
    if materializer_node_id is None:
        return
    materializer_proxy_outputs[materializer_node_id] = proxy_output
    for consumer_node_id in spec.local_materializer_consumer_node_ids:
        replacements.materializer_by_consumer[(consumer_node_id, spec.source)] = [
            materializer_node_id,
            0,
        ]


def _assign_ordered_phase_proxy_ids(
    state: _OrderedProxyRewriteState,
    *,
    rewritten_prompt: dict[str, Any],
    component_node_ids: set[str],
) -> None:
    """Assign collision-free proxy ids to every ordered phase payload."""
    for phase_payload in state.phase_payloads:
        proxy_node_id = _unique_proxy_node_id(
            str(phase_payload["component_id"]),
            rewritten_prompt=rewritten_prompt,
            component_node_ids=component_node_ids,
        )
        phase_payload["component_id"] = proxy_node_id
        state.proxy_node_ids.append(proxy_node_id)


def _prepare_ordered_phase_proxy(
    state: _OrderedProxyRewriteState,
    phase_payload: dict[str, Any],
    *,
    rewritten_prompt: dict[str, Any],
    boundary_specs_by_source: dict[LinkedOutputRef, BoundaryOutputSpec],
    produced_outputs_by_name: dict[str, list[Any]],
) -> None:
    """Prepare inputs, metadata, and output routes for one ordered phase."""
    proxy_node_id = str(phase_payload["component_id"])
    proxy_inputs = _proxy_inputs_for_boundary_inputs(
        list(phase_payload["boundary_inputs"]),
        rewritten_prompt=rewritten_prompt,
    )
    for boundary_input in phase_payload.get("boundary_inputs", []):
        input_name = str(boundary_input["proxy_input_name"])
        produced_output = produced_outputs_by_name.get(input_name)
        if produced_output is not None:
            proxy_inputs[input_name] = list(produced_output)
    first_node_id = str(phase_payload["component_node_ids"][0])
    state.proxy_inputs_by_node_id[proxy_node_id] = proxy_inputs
    state.proxy_meta_by_node_id[proxy_node_id] = copy.deepcopy(
        rewritten_prompt[first_node_id].get("_meta", {})
    )
    for output_index, boundary_output in enumerate(
        phase_payload.get("boundary_outputs", [])
    ):
        proxy_output = [proxy_node_id, output_index]
        produced_outputs_by_name[
            str(boundary_output["proxy_output_name"])
        ] = proxy_output
        source = _linked_output_ref(boundary_output)
        spec = boundary_specs_by_source.get(source)
        if spec is not None:
            _record_boundary_output_routes(
                spec,
                proxy_output,
                state.replacements,
                state.materializer_proxy_outputs,
            )
        if not bool(boundary_output.get("session_output")):
            state.replacements.default_by_source[source] = proxy_output


def _register_ordered_phase_proxies(
    state: _OrderedProxyRewriteState,
    *,
    rewritten_prompt: dict[str, Any],
    nodes_module: Any,
) -> None:
    """Register every prepared phase proxy and local bridge materializer."""
    for phase_payload in state.phase_payloads:
        proxy_node_id = str(phase_payload["component_id"])
        _register_proxy_node(
            prompt_node_id=proxy_node_id,
            payload_mapping=phase_payload,
            proxy_inputs=state.proxy_inputs_by_node_id[proxy_node_id],
            meta=state.proxy_meta_by_node_id[proxy_node_id],
            is_output_node=_contains_output_node(
                list(phase_payload["component_node_ids"]),
                rewritten_prompt=rewritten_prompt,
                nodes_module=nodes_module,
            ),
            rewritten_prompt=rewritten_prompt,
            nodes_module=nodes_module,
        )
    for materializer_node_id, proxy_output in state.materializer_proxy_outputs.items():
        _install_local_bridge_materializer(
            rewritten_prompt, materializer_node_id, proxy_output
        )


def _register_ordered_mapped_warmup(
    component: RemoteComponentPlan,
    phase_payloads: list[dict[str, Any]],
) -> None:
    """Register mapped-input warmup against the phase that owns mapped nodes."""
    if component.mapped_boundary_source_node_id is None:
        return
    mapped_node_ids = set(component.mapped_node_ids)
    for phase_payload in phase_payloads:
        phase_node_ids = {
            str(node_id) for node_id in phase_payload.get("component_node_ids", [])
        }
        if not (phase_node_ids & mapped_node_ids):
            continue
        register_modal_map_input_warmup_context(
            component.mapped_boundary_source_node_id,
            phase_payload,
            str(component.mapped_boundary_input_io_type or "*"),
        )
        return


def _rewrite_ordered_phase_proxies(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    split_payloads: list[Any],
    nodes_module: Any,
) -> list[str]:
    """Replace one component with its ordered sequence of phase proxies."""
    state = _OrderedProxyRewriteState(
        phase_payloads=[dict(phase_payload) for phase_payload in split_payloads],
        proxy_node_ids=[],
        proxy_inputs_by_node_id={},
        proxy_meta_by_node_id={},
        replacements=_empty_proxy_replacement_maps(),
        materializer_proxy_outputs={},
    )
    if any(spec.local_materializer_node_id for spec in component.boundary_outputs):
        ensure_modal_local_bridge_materializer_registered(nodes_module)
    _assign_ordered_phase_proxy_ids(
        state,
        rewritten_prompt=rewritten_prompt,
        component_node_ids=set(component.node_ids),
    )
    boundary_specs = {spec.source: spec for spec in component.boundary_outputs}
    produced_outputs: dict[str, list[Any]] = {}
    for phase_payload in state.phase_payloads:
        _prepare_ordered_phase_proxy(
            state,
            phase_payload,
            rewritten_prompt=rewritten_prompt,
            boundary_specs_by_source=boundary_specs,
            produced_outputs_by_name=produced_outputs,
        )
    for node_id in component.node_ids:
        rewritten_prompt.pop(node_id, None)
    _register_ordered_phase_proxies(
        state, rewritten_prompt=rewritten_prompt, nodes_module=nodes_module
    )
    _register_ordered_mapped_warmup(component, state.phase_payloads)
    _rewire_proxy_consumers(
        rewritten_prompt,
        skip_node_ids=set(state.proxy_node_ids),
        replacements=state.replacements,
    )
    logger.info(
        "Rewrote remote component %s into ordered proxies %s.",
        component.representative_node_id,
        state.proxy_node_ids,
    )
    return state.proxy_node_ids


def _hybrid_proxy_inputs(
    *,
    static_payload: dict[str, Any],
    mapped_payload: dict[str, Any],
    static_proxy_node_id: str,
    rewritten_prompt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve static and mapped proxy inputs, including session bridge links."""
    static_inputs = _proxy_inputs_for_boundary_inputs(
        list(static_payload["boundary_inputs"]), rewritten_prompt=rewritten_prompt
    )
    mapped_inputs = _proxy_inputs_for_boundary_inputs(
        list(mapped_payload["boundary_inputs"]), rewritten_prompt=rewritten_prompt
    )
    mapped_static_phase = mapped_payload.get("static_phase")
    if isinstance(mapped_static_phase, dict):
        static_phase_inputs = _proxy_inputs_for_boundary_inputs(
            list(mapped_static_phase.get("boundary_inputs", [])),
            rewritten_prompt=rewritten_prompt,
        )
        for input_name, input_value in static_phase_inputs.items():
            mapped_inputs.setdefault(input_name, input_value)
    bridge_indices = {
        str(output["proxy_output_name"]): index
        for index, output in enumerate(static_payload["boundary_outputs"])
        if bool(output.get("session_output"))
    }
    for boundary_input in mapped_payload.get("boundary_inputs", []):
        input_name = str(boundary_input["proxy_input_name"])
        if input_name in bridge_indices:
            mapped_inputs[input_name] = [
                static_proxy_node_id,
                bridge_indices[input_name],
            ]
    return static_inputs, mapped_inputs


def _hybrid_output_replacements(
    static_payload: dict[str, Any],
    mapped_payload: dict[str, Any],
    *,
    static_proxy_node_id: str,
    mapped_proxy_node_id: str,
) -> _ProxyReplacementMaps:
    """Build default output rewrites for paired static and mapped proxies."""
    replacements = _empty_proxy_replacement_maps()
    for output_index, boundary_output in enumerate(static_payload["boundary_outputs"]):
        if not bool(boundary_output.get("session_output")):
            replacements.default_by_source[_linked_output_ref(boundary_output)] = [
                static_proxy_node_id,
                output_index,
            ]
    for output_index, boundary_output in enumerate(
        mapped_payload.get("boundary_outputs", [])
    ):
        replacements.default_by_source[_linked_output_ref(boundary_output)] = [
            mapped_proxy_node_id,
            output_index,
        ]
    return replacements


def _rewrite_hybrid_split_proxies(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    split_payloads: dict[str, Any],
    nodes_module: Any,
) -> list[str]:
    """Replace one hybrid component with paired static and mapped proxies."""
    static_payload = dict(split_payloads["static"])
    mapped_payload = dict(split_payloads["mapped"])
    static_proxy_node_id = str(static_payload["component_id"])
    mapped_proxy_node_id = _unique_proxy_node_id(
        str(mapped_payload["component_id"]),
        rewritten_prompt=rewritten_prompt,
        component_node_ids=set(component.node_ids),
    )
    mapped_payload["component_id"] = mapped_proxy_node_id
    static_inputs, mapped_inputs = _hybrid_proxy_inputs(
        static_payload=static_payload,
        mapped_payload=mapped_payload,
        static_proxy_node_id=static_proxy_node_id,
        rewritten_prompt=rewritten_prompt,
    )
    static_meta = copy.deepcopy(rewritten_prompt[static_proxy_node_id].get("_meta", {}))
    mapped_meta = copy.deepcopy(
        rewritten_prompt[component.mapped_node_ids[0]].get("_meta", {})
    )
    replacements = _hybrid_output_replacements(
        static_payload,
        mapped_payload,
        static_proxy_node_id=static_proxy_node_id,
        mapped_proxy_node_id=mapped_proxy_node_id,
    )
    for node_id in component.node_ids:
        rewritten_prompt.pop(node_id, None)
    for proxy_node_id, proxy_payload, proxy_inputs, meta, node_ids in (
        (static_proxy_node_id, static_payload, static_inputs, static_meta, component.static_node_ids),
        (mapped_proxy_node_id, mapped_payload, mapped_inputs, mapped_meta, component.mapped_node_ids),
    ):
        _register_proxy_node(
            prompt_node_id=proxy_node_id,
            payload_mapping=proxy_payload,
            proxy_inputs=proxy_inputs,
            meta=meta,
            is_output_node=_contains_output_node(
                node_ids, rewritten_prompt=rewritten_prompt, nodes_module=nodes_module
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
    proxy_node_ids = [static_proxy_node_id, mapped_proxy_node_id]
    _rewire_proxy_consumers(
        rewritten_prompt,
        skip_node_ids=set(proxy_node_ids),
        replacements=replacements,
    )
    logger.info(
        "Rewrote hybrid remote component %s into static proxy %s and mapped proxy %s.",
        component.representative_node_id,
        static_proxy_node_id,
        mapped_proxy_node_id,
    )
    return proxy_node_ids


def _single_proxy_replacement_maps(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    *,
    representative_node_id: str,
) -> _ProxyReplacementMaps:
    """Build output rewrites and materializer nodes for one ordinary proxy."""
    replacements = _empty_proxy_replacement_maps()
    materializer_proxy_outputs: dict[str, list[Any]] = {}
    for output_index, spec in enumerate(component.boundary_outputs):
        proxy_output = [representative_node_id, output_index]
        replacements.default_by_source.setdefault(spec.source, proxy_output)
        _record_boundary_output_routes(
            spec, proxy_output, replacements, materializer_proxy_outputs
        )
    for materializer_node_id, proxy_output in materializer_proxy_outputs.items():
        _install_local_bridge_materializer(
            rewritten_prompt, materializer_node_id, proxy_output
        )
    return replacements


def _rewrite_single_component_proxy(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    payload: dict[str, Any],
    nodes_module: Any,
) -> list[str]:
    """Replace one ordinary remote component with its representative proxy."""
    representative_node_id = component.representative_node_id
    proxy_inputs = _proxy_inputs_for_boundary_inputs(
        list(payload.get("boundary_inputs", [])), rewritten_prompt=rewritten_prompt
    )
    representative_meta = copy.deepcopy(
        rewritten_prompt[representative_node_id].get("_meta", {})
    )
    _register_proxy_node(
        prompt_node_id=representative_node_id,
        payload_mapping=payload,
        proxy_inputs=proxy_inputs,
        meta=representative_meta,
        is_output_node=component.contains_output_node,
        rewritten_prompt=rewritten_prompt,
        nodes_module=nodes_module,
    )
    if component.mapped_boundary_source_node_id is not None:
        register_modal_map_input_warmup_context(
            component.mapped_boundary_source_node_id,
            payload,
            str(component.mapped_boundary_input_io_type or "*"),
        )
    if any(spec.local_materializer_node_id for spec in component.boundary_outputs):
        ensure_modal_local_bridge_materializer_registered(nodes_module)
    replacements = _single_proxy_replacement_maps(
        component,
        rewritten_prompt,
        representative_node_id=representative_node_id,
    )
    for node_id in component.node_ids:
        if node_id != representative_node_id:
            rewritten_prompt.pop(node_id, None)
    _rewire_proxy_consumers(
        rewritten_prompt,
        skip_node_ids={representative_node_id},
        replacements=replacements,
    )
    logger.info(
        "Rewrote remote component %s with %d nodes to its representative proxy.",
        representative_node_id,
        len(component.node_ids),
    )
    return [representative_node_id]


def _rewrite_component_into_proxy(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    payload: dict[str, Any],
    nodes_module: Any,
) -> list[str]:
    """Replace one remote component with the payload's selected proxy shape."""
    split_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_payloads, list):
        return _rewrite_ordered_phase_proxies(
            component, rewritten_prompt, split_payloads, nodes_module
        )
    if isinstance(split_payloads, dict):
        return _rewrite_hybrid_split_proxies(
            component, rewritten_prompt, split_payloads, nodes_module
        )
    return _rewrite_single_component_proxy(
        component, rewritten_prompt, payload, nodes_module
    )


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
