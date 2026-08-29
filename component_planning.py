"""Remote component construction, replication, and boundary validation."""

from __future__ import annotations

import copy
import hashlib
import logging
from collections import defaultdict, deque
from typing import Any, Iterable, Mapping

if __package__:
    from .execution_environments import ExecutionAssignment, ExecutionProvider
    from .modal_executor_node import MODAL_MAP_INPUT_NODE_ID
    from .remote_graph_analysis import (
        _build_consumer_map,
        _build_remote_components,
        _expand_component_for_non_transportable_local_outputs,
        _is_link,
        _is_transportable_output_type,
        _non_returning_local_output_consumers,
        _non_returning_local_tap_node_ids,
        _normalize_output_metadata,
        _remote_component_partition_groups,
        _remote_output_io_type,
        _remote_output_is_list,
    )
    from .remote_plan_types import (
        BoundaryInputSpec,
        BoundaryOutputSpec,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        RemoteComponentPlan,
        StaticToMappedBoundarySpec,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import ExecutionAssignment, ExecutionProvider
    from modal_executor_node import MODAL_MAP_INPUT_NODE_ID
    from remote_graph_analysis import (
        _build_consumer_map,
        _build_remote_components,
        _expand_component_for_non_transportable_local_outputs,
        _is_link,
        _is_transportable_output_type,
        _non_returning_local_output_consumers,
        _non_returning_local_tap_node_ids,
        _normalize_output_metadata,
        _remote_component_partition_groups,
        _remote_output_io_type,
        _remote_output_is_list,
    )
    from remote_plan_types import (
        BoundaryInputSpec,
        BoundaryOutputSpec,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        RemoteComponentPlan,
        StaticToMappedBoundarySpec,
    )

logger = logging.getLogger(__name__)

_REMOTE_REPLICA_NODE_PREFIX = "__ComfyModalReplica__"
_REPLICABLE_REMOTE_OBJECT_TYPES = frozenset(
    {"CLIP", "MODEL", "NOISE", "SAMPLER", "VAE"}
)


def _build_component_plan(
    component_node_ids: list[str],
    prompt: dict[str, Any],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> RemoteComponentPlan:
    """Build rewrite metadata for a connected remote component."""
    original_component_node_id_set = set(component_node_ids)
    representative_node_id = component_node_ids[0]
    local_tap_node_ids, local_tap_terminal_node_ids = _non_returning_local_tap_node_ids(
        prompt=prompt,
        component_node_ids=original_component_node_id_set,
        remote_node_ids=remote_node_ids,
        consumers=consumers,
        nodes_module=nodes_module,
    )
    if local_tap_node_ids:
        logger.info(
            "Absorbing non-returning local tap nodes into remote component %s: %s",
            representative_node_id,
            sorted(local_tap_node_ids),
        )
    if local_tap_terminal_node_ids:
        logger.info(
            "Keeping non-returning local tap terminal nodes outside remote component %s: %s",
            representative_node_id,
            sorted(local_tap_terminal_node_ids),
        )
    component_node_id_set = original_component_node_id_set | local_tap_node_ids
    if local_tap_node_ids:
        (
            expanded_component_node_id_set,
            tap_dependency_node_ids,
        ) = _expand_component_for_non_transportable_local_outputs(
            prompt=prompt,
            component_node_ids=component_node_id_set,
            remote_node_ids=remote_node_ids,
            consumers=consumers,
            nodes_module=nodes_module,
        )
        if tap_dependency_node_ids:
            logger.info(
                "Absorbing non-transportable boundary dependencies for remote component %s preview taps: %s",
                representative_node_id,
                sorted(tap_dependency_node_ids),
            )
            local_tap_node_ids.update(tap_dependency_node_ids)
            component_node_id_set = expanded_component_node_id_set
    component_node_ids = sorted(component_node_id_set)
    boundary_inputs_by_source: dict[LinkedOutputRef, BoundaryInputSpec] = {}
    boundary_outputs_by_source: dict[LinkedOutputRef, BoundaryOutputSpec] = {}
    output_execution_targets: set[str] = set()
    contains_output_node = False

    for node_id in component_node_ids:
        prompt_node = prompt[node_id]
        class_type = str(prompt_node["class_type"])
        node_class = nodes_module.NODE_CLASS_MAPPINGS[class_type]
        output_types, output_names, output_is_list = _normalize_output_metadata(
            node_class
        )

        if getattr(node_class, "OUTPUT_NODE", False):
            contains_output_node = True
            output_execution_targets.add(node_id)

        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            if upstream_node_id in component_node_id_set:
                continue
            source = LinkedOutputRef(
                node_id=upstream_node_id, output_index=int(input_value[1])
            )
            spec = boundary_inputs_by_source.get(source)
            if spec is None:
                source_io_type = _remote_output_io_type(
                    prompt=prompt,
                    node_id=source.node_id,
                    output_index=source.output_index,
                    nodes_module=nodes_module,
                )
                spec = BoundaryInputSpec(
                    proxy_input_name=f"remote_input_{len(boundary_inputs_by_source)}",
                    source=source,
                    io_type=source_io_type,
                )
                boundary_inputs_by_source[source] = spec
            spec.targets.append(
                InputTarget(node_id=node_id, input_name=str(input_name))
            )

        has_downstream_consumer = False
        for output_index, io_type in enumerate(output_types):
            source = LinkedOutputRef(node_id=node_id, output_index=output_index)
            output_consumers = consumers.get(source, [])
            if output_consumers:
                has_downstream_consumer = True
            local_consumers = [
                consumer
                for consumer in output_consumers
                if consumer.node_id not in component_node_id_set
            ]
            if not local_consumers:
                continue
            output_execution_targets.add(node_id)
            if source in boundary_outputs_by_source:
                continue
            output_name = output_names[output_index]
            boundary_outputs_by_source[source] = BoundaryOutputSpec(
                proxy_output_name=f"{node_id}_{output_name}",
                source=source,
                io_type=str(io_type),
                is_list=bool(output_is_list[output_index]),
                preview_target_node_ids=_preview_target_node_ids(
                    prompt=prompt,
                    local_consumers=local_consumers,
                ),
            )

        if not has_downstream_consumer:
            output_execution_targets.add(node_id)

    mapped_boundary_spec: BoundaryInputSpec | None = None
    mapped_boundary_input_io_type: str | None = None
    for boundary_input in boundary_inputs_by_source.values():
        source_prompt_node = prompt.get(boundary_input.source.node_id)
        source_class_type = (
            str(source_prompt_node.get("class_type"))
            if source_prompt_node is not None
            else None
        )
        mapped_targets = [
            target
            for target in boundary_input.targets
            if str(prompt[target.node_id]["class_type"]) == MODAL_MAP_INPUT_NODE_ID
        ]
        source_is_modal_map_input = source_class_type == MODAL_MAP_INPUT_NODE_ID
        if not mapped_targets and not source_is_modal_map_input:
            continue
        if mapped_targets and len(mapped_targets) != len(boundary_input.targets):
            raise ModalPromptValidationError(
                "Mapped remote execution requires the mapped boundary input to feed only ModalMapInput nodes."
            )
        if mapped_boundary_spec is not None:
            raise ModalPromptValidationError(
                "Remote components currently support only one mapped ModalMapInput boundary."
            )
        mapped_boundary_spec = boundary_input
        mapped_boundary_input_io_type = _mapped_boundary_origin_io_type(
            prompt,
            boundary_input,
            nodes_module,
        )

    mapped_node_ids: list[str] = []
    mapped_execute_node_ids: list[str] = []
    static_execute_node_ids: list[str] = []
    static_node_ids: list[str] = []
    static_to_mapped_boundaries: list[StaticToMappedBoundarySpec] = []
    if mapped_boundary_spec is not None:
        mapped_reachable_node_ids = _component_downstream_closure(
            seed_node_ids={target.node_id for target in mapped_boundary_spec.targets},
            component_node_id_set=component_node_id_set,
            consumers=consumers,
        )
        mapped_node_ids = sorted(mapped_reachable_node_ids)
        mapped_node_id_set = set(mapped_node_ids)
        static_node_ids = sorted(component_node_id_set - mapped_node_id_set)
        static_to_mapped_boundaries_by_source: dict[
            LinkedOutputRef, StaticToMappedBoundarySpec
        ] = {}
        for mapped_node_id in mapped_node_ids:
            prompt_node = prompt[mapped_node_id]
            for input_name, input_value in (prompt_node.get("inputs") or {}).items():
                if not _is_link(input_value):
                    continue
                upstream_node_id = str(input_value[0])
                if (
                    upstream_node_id not in component_node_id_set
                    or upstream_node_id in mapped_node_id_set
                ):
                    continue
                source = LinkedOutputRef(
                    node_id=upstream_node_id,
                    output_index=int(input_value[1]),
                )
                boundary_spec = static_to_mapped_boundaries_by_source.get(source)
                if boundary_spec is None:
                    boundary_spec = StaticToMappedBoundarySpec(
                        proxy_name=f"static_input_{len(static_to_mapped_boundaries_by_source)}",
                        source=source,
                        io_type=str(
                            _remote_output_io_type(
                                prompt=prompt,
                                node_id=source.node_id,
                                output_index=source.output_index,
                                nodes_module=nodes_module,
                            )
                            or "*"
                        ),
                        is_list=_remote_output_is_list(
                            prompt=prompt,
                            node_id=source.node_id,
                            output_index=source.output_index,
                            nodes_module=nodes_module,
                        ),
                    )
                    static_to_mapped_boundaries_by_source[source] = boundary_spec
                boundary_spec.targets.append(
                    InputTarget(node_id=mapped_node_id, input_name=str(input_name))
                )
        static_to_mapped_boundaries = sorted(
            static_to_mapped_boundaries_by_source.values(),
            key=lambda spec: (spec.source.node_id, spec.source.output_index),
        )
        mapped_execute_node_ids = sorted(output_execution_targets & mapped_node_id_set)
        static_execute_node_ids = sorted(
            (output_execution_targets - mapped_node_id_set)
            | {
                boundary_spec.source.node_id
                for boundary_spec in static_to_mapped_boundaries
            }
        )

    component = RemoteComponentPlan(
        node_ids=component_node_ids,
        representative_node_id=representative_node_id,
        boundary_inputs=sorted(
            boundary_inputs_by_source.values(),
            key=lambda spec: (spec.source.node_id, spec.source.output_index),
        ),
        boundary_outputs=sorted(
            boundary_outputs_by_source.values(),
            key=lambda spec: (spec.source.node_id, spec.source.output_index),
        ),
        execute_node_ids=sorted(output_execution_targets),
        contains_output_node=contains_output_node,
        mapped_boundary_input_name=(
            mapped_boundary_spec.proxy_input_name
            if mapped_boundary_spec is not None
            else None
        ),
        mapped_boundary_input_io_type=mapped_boundary_input_io_type,
        mapped_boundary_source_node_id=(
            mapped_boundary_spec.source.node_id
            if mapped_boundary_spec is not None
            and prompt.get(mapped_boundary_spec.source.node_id, {}).get("class_type")
            == MODAL_MAP_INPUT_NODE_ID
            else None
        ),
        static_node_ids=static_node_ids,
        mapped_node_ids=mapped_node_ids,
        mapped_execute_node_ids=mapped_execute_node_ids,
        static_execute_node_ids=static_execute_node_ids,
        static_to_mapped_boundaries=static_to_mapped_boundaries,
        local_tap_node_ids=sorted(local_tap_node_ids),
        local_tap_terminal_node_ids=sorted(local_tap_terminal_node_ids),
    )
    logger.info(
        "Planned remote component %s: nodes=%s boundary_inputs=%d boundary_outputs=%d execute_nodes=%s output_node=%s mapped_input=%s static_nodes=%s mapped_nodes=%s mapped_execute_nodes=%s static_execute_nodes=%s local_tap_nodes=%s local_tap_terminal_nodes=%s static_to_mapped_boundaries=%s",
        component.representative_node_id,
        component.node_ids,
        len(component.boundary_inputs),
        len(component.boundary_outputs),
        component.execute_node_ids,
        component.contains_output_node,
        component.mapped_boundary_input_name,
        component.static_node_ids,
        component.mapped_node_ids,
        component.mapped_execute_node_ids,
        component.static_execute_node_ids,
        component.local_tap_node_ids,
        component.local_tap_terminal_node_ids,
        [
            {
                "proxy_name": boundary_spec.proxy_name,
                "source": (
                    boundary_spec.source.node_id,
                    boundary_spec.source.output_index,
                ),
                "targets": [
                    (target.node_id, target.input_name)
                    for target in boundary_spec.targets
                ],
            }
            for boundary_spec in component.static_to_mapped_boundaries
        ],
    )
    return component


def _filter_boundary_inputs_for_node_ids(
    boundary_inputs: list[BoundaryInputSpec],
    allowed_node_ids: set[str],
) -> list[BoundaryInputSpec]:
    """Return boundary inputs whose targets belong to one node-id subset."""
    filtered_boundary_inputs: list[BoundaryInputSpec] = []
    for boundary_input in boundary_inputs:
        filtered_targets = [
            target
            for target in boundary_input.targets
            if target.node_id in allowed_node_ids
        ]
        if not filtered_targets:
            continue
        filtered_boundary_inputs.append(
            BoundaryInputSpec(
                proxy_input_name=boundary_input.proxy_input_name,
                source=boundary_input.source,
                io_type=boundary_input.io_type,
                targets=filtered_targets,
            )
        )
    return filtered_boundary_inputs


def _filter_boundary_outputs_for_node_ids(
    boundary_outputs: list[BoundaryOutputSpec],
    allowed_node_ids: set[str],
) -> list[BoundaryOutputSpec]:
    """Return boundary outputs exported by one node-id subset."""
    return [
        boundary_output
        for boundary_output in boundary_outputs
        if boundary_output.source.node_id in allowed_node_ids
    ]


def _subset_component_prompt(
    component_prompt: dict[str, Any],
    node_ids: list[str],
) -> dict[str, Any]:
    """Return one deep-copied prompt subset for a phase-local node set."""
    return {
        node_id: copy.deepcopy(component_prompt[node_id])
        for node_id in node_ids
        if node_id in component_prompt
    }


def _preview_target_node_ids(
    *,
    prompt: dict[str, Any],
    local_consumers: list[InputTarget],
) -> list[str]:
    """Return direct local PreviewImage consumers for one remote boundary output."""
    preview_target_node_ids: set[str] = set()
    for local_consumer in local_consumers:
        consumer_prompt_node = prompt.get(local_consumer.node_id)
        if consumer_prompt_node is None:
            continue
        if str(consumer_prompt_node.get("class_type")) != "PreviewImage":
            continue
        preview_target_node_ids.add(str(local_consumer.node_id))
    return sorted(preview_target_node_ids)


def _component_downstream_closure(
    *,
    seed_node_ids: set[str],
    component_node_id_set: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
) -> set[str]:
    """Return component-local nodes reachable downstream from one seed set."""
    reachable_node_ids: set[str] = set()
    pending_node_ids = list(sorted(seed_node_ids))
    while pending_node_ids:
        current_node_id = pending_node_ids.pop()
        if (
            current_node_id in reachable_node_ids
            or current_node_id not in component_node_id_set
        ):
            continue
        reachable_node_ids.add(current_node_id)
        for consumer_source, consumer_targets in consumers.items():
            if consumer_source.node_id != current_node_id:
                continue
            for consumer_target in consumer_targets:
                if consumer_target.node_id in component_node_id_set:
                    pending_node_ids.append(consumer_target.node_id)
    return reachable_node_ids


def _component_upstream_closure(
    *,
    prompt: dict[str, Any],
    seed_node_ids: set[str],
    candidate_node_ids: set[str],
) -> set[str]:
    """Return one candidate-local upstream closure for the supplied seed nodes."""
    reachable_node_ids: set[str] = set()
    pending_node_ids = list(sorted(seed_node_ids))

    while pending_node_ids:
        current_node_id = pending_node_ids.pop()
        if (
            current_node_id in reachable_node_ids
            or current_node_id not in candidate_node_ids
        ):
            continue
        reachable_node_ids.add(current_node_id)
        prompt_node = prompt.get(current_node_id)
        if prompt_node is None:
            continue
        for input_value in (prompt_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            if upstream_node_id in candidate_node_ids:
                pending_node_ids.append(upstream_node_id)
    return reachable_node_ids


def _component_ancestors_of_local_source(
    *,
    prompt: dict[str, Any],
    source_node_id: str,
    component_node_id_set: set[str],
) -> set[str]:
    """Return component nodes that feed one local boundary-input source."""
    ancestor_node_ids: set[str] = set()
    visited_node_ids: set[str] = set()
    pending_node_ids = [source_node_id]

    while pending_node_ids:
        current_node_id = str(pending_node_ids.pop())
        if current_node_id in visited_node_ids:
            continue
        visited_node_ids.add(current_node_id)
        if current_node_id in component_node_id_set:
            ancestor_node_ids.add(current_node_id)
            continue
        prompt_node = prompt.get(current_node_id)
        if prompt_node is None:
            continue
        for input_value in (prompt_node.get("inputs") or {}).values():
            if _is_link(input_value):
                pending_node_ids.append(str(input_value[0]))
    return ancestor_node_ids


def _component_has_local_reentry_dependency(
    *,
    prompt: dict[str, Any],
    component: RemoteComponentPlan,
) -> bool:
    """Return whether a component boundary input depends on that same component's output."""
    component_node_id_set = set(component.node_ids)
    for boundary_input in component.boundary_inputs:
        if boundary_input.source.node_id in component_node_id_set:
            continue
        if _component_ancestors_of_local_source(
            prompt=prompt,
            source_node_id=boundary_input.source.node_id,
            component_node_id_set=component_node_id_set,
        ):
            return True
    return False


def _component_has_parallel_local_remote_fanout(
    component: RemoteComponentPlan,
) -> bool:
    """Return whether one remote output feeds local work and a remote continuation."""
    return any(
        boundary_output.local_materializer_node_id is not None
        and bool(boundary_output.session_consumer_node_ids)
        for boundary_output in component.boundary_outputs
    )


def _order_execute_node_ids_for_transportable_splits(
    *,
    prompt: dict[str, Any],
    component_prompt: dict[str, Any],
    component_node_ids: set[str],
    execute_node_ids: list[str],
) -> list[str]:
    """Return split execute targets ordered by component and local feedback dependencies."""
    base_order = list(execute_node_ids)
    if len(base_order) <= 1:
        return base_order

    closure_by_execute_node_id = {
        execute_node_id: _component_upstream_closure(
            prompt=component_prompt,
            seed_node_ids={execute_node_id},
            candidate_node_ids=component_node_ids,
        )
        for execute_node_id in base_order
    }
    producer_execute_node_ids_by_component_node_id: dict[str, list[str]] = defaultdict(
        list
    )
    for execute_node_id in base_order:
        for component_node_id in closure_by_execute_node_id[execute_node_id]:
            producer_execute_node_ids_by_component_node_id[component_node_id].append(
                execute_node_id
            )

    base_index_by_execute_node_id = {
        execute_node_id: index for index, execute_node_id in enumerate(base_order)
    }
    dependency_edges: dict[str, set[str]] = {
        execute_node_id: set() for execute_node_id in base_order
    }
    indegree_by_execute_node_id: dict[str, int] = {
        execute_node_id: 0 for execute_node_id in base_order
    }

    def producer_for_component_node(component_node_id: str) -> str | None:
        """Return the earliest execute target that produces a component node."""
        producer_execute_node_ids = producer_execute_node_ids_by_component_node_id.get(
            component_node_id,
            [],
        )
        if not producer_execute_node_ids:
            return None
        exact_producers = [
            execute_node_id
            for execute_node_id in producer_execute_node_ids
            if execute_node_id == component_node_id
        ]
        if exact_producers:
            return exact_producers[0]
        return min(
            producer_execute_node_ids,
            key=lambda execute_node_id: base_index_by_execute_node_id[execute_node_id],
        )

    for target_execute_node_id in base_order:
        target_closure = closure_by_execute_node_id[target_execute_node_id]
        for phase_node_id in sorted(target_closure):
            prompt_node = component_prompt.get(phase_node_id)
            if prompt_node is None:
                continue
            for input_value in (prompt_node.get("inputs") or {}).values():
                if not _is_link(input_value):
                    continue
                source_node_id = str(input_value[0])
                if source_node_id in component_node_ids:
                    continue
                source_ancestor_node_ids = _component_ancestors_of_local_source(
                    prompt=prompt,
                    source_node_id=source_node_id,
                    component_node_id_set=component_node_ids,
                )
                for source_ancestor_node_id in sorted(source_ancestor_node_ids):
                    if source_ancestor_node_id in target_closure:
                        continue
                    producer_execute_node_id = producer_for_component_node(
                        source_ancestor_node_id
                    )
                    if (
                        producer_execute_node_id is None
                        or producer_execute_node_id == target_execute_node_id
                        or target_execute_node_id
                        in dependency_edges[producer_execute_node_id]
                    ):
                        continue
                    dependency_edges[producer_execute_node_id].add(
                        target_execute_node_id
                    )
                    indegree_by_execute_node_id[target_execute_node_id] += 1

    ready_execute_node_ids = [
        execute_node_id
        for execute_node_id in base_order
        if indegree_by_execute_node_id[execute_node_id] == 0
    ]
    ordered_execute_node_ids: list[str] = []
    while ready_execute_node_ids:
        ready_execute_node_ids.sort(
            key=lambda node_id: base_index_by_execute_node_id[node_id]
        )
        execute_node_id = ready_execute_node_ids.pop(0)
        ordered_execute_node_ids.append(execute_node_id)
        for downstream_execute_node_id in sorted(
            dependency_edges[execute_node_id],
            key=lambda node_id: base_index_by_execute_node_id[node_id],
        ):
            indegree_by_execute_node_id[downstream_execute_node_id] -= 1
            if indegree_by_execute_node_id[downstream_execute_node_id] == 0:
                ready_execute_node_ids.append(downstream_execute_node_id)

    if len(ordered_execute_node_ids) != len(base_order):
        logger.warning(
            "Split phase local-feedback dependency ordering encountered a cycle; keeping base execute order %s.",
            base_order,
        )
        return base_order

    if ordered_execute_node_ids != base_order:
        logger.info(
            "Reordered split execute targets from %s to %s using local-feedback dependencies %s.",
            base_order,
            ordered_execute_node_ids,
            {
                execute_node_id: sorted(downstream_execute_node_ids)
                for execute_node_id, downstream_execute_node_ids in dependency_edges.items()
                if downstream_execute_node_ids
            },
        )
    return ordered_execute_node_ids


def _subgraph_topological_node_order(
    prompt: dict[str, Any],
    node_ids: set[str],
) -> list[str]:
    """Return one best-effort topological order for a prompt node subset."""
    indegree_by_node_id: dict[str, int] = {node_id: 0 for node_id in node_ids}
    downstream_node_ids_by_node_id: dict[str, set[str]] = {
        node_id: set() for node_id in node_ids
    }

    for node_id in sorted(node_ids):
        prompt_node = prompt.get(node_id)
        if prompt_node is None:
            continue
        for input_value in (prompt_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            if upstream_node_id not in node_ids:
                continue
            if node_id in downstream_node_ids_by_node_id[upstream_node_id]:
                continue
            downstream_node_ids_by_node_id[upstream_node_id].add(node_id)
            indegree_by_node_id[node_id] += 1

    ready_node_ids = deque(
        sorted(
            node_id
            for node_id, indegree in indegree_by_node_id.items()
            if indegree == 0
        )
    )
    ordered_node_ids: list[str] = []

    while ready_node_ids:
        node_id = ready_node_ids.popleft()
        ordered_node_ids.append(node_id)
        for downstream_node_id in sorted(downstream_node_ids_by_node_id[node_id]):
            indegree_by_node_id[downstream_node_id] -= 1
            if indegree_by_node_id[downstream_node_id] == 0:
                ready_node_ids.append(downstream_node_id)

    if len(ordered_node_ids) == len(node_ids):
        return ordered_node_ids

    logger.warning(
        "Component phase planning encountered a node-level cycle; falling back to stable node order for %s.",
        sorted(node_ids),
    )
    return sorted(node_ids)


def _remote_dependency_closure(
    prompt: Mapping[str, Any],
    source_node_id: str,
) -> set[str]:
    """Return the complete upstream dependency closure for one remote producer."""
    closure: set[str] = set()
    pending = [source_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in closure:
            continue
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        closure.add(node_id)
        pending.extend(
            str(input_value[0])
            for input_value in (prompt_node.get("inputs") or {}).values()
            if _is_link(input_value)
        )
    return closure


def _can_replicate_remote_dependency_closure(
    *,
    prompt: Mapping[str, Any],
    source_node_id: str,
    dependency_node_ids: set[str],
    remote_node_ids: set[str],
) -> bool:
    """Return whether a producer closure can be safely rebuilt on another worker."""
    if not dependency_node_ids or not dependency_node_ids.issubset(remote_node_ids):
        return False
    source_node = prompt.get(source_node_id)
    if not isinstance(source_node, Mapping):
        return False
    source_inputs = source_node.get("inputs")
    if not isinstance(source_inputs, Mapping):
        return False
    if not any(_is_link(value) for value in source_inputs.values()):
        return True
    return all(
        "sampler"
        not in str((prompt.get(node_id) or {}).get("class_type") or "").casefold()
        for node_id in dependency_node_ids
    )


def _replica_node_id_mapping(
    *,
    prompt: Mapping[str, Any],
    dependency_node_ids: set[str],
    source_node_id: str,
    target_group_id: str,
) -> dict[str, str]:
    """Return collision-free deterministic ids for one replicated dependency closure."""
    replica_digest = hashlib.sha256(
        f"{source_node_id}\0{target_group_id}".encode("utf-8")
    ).hexdigest()[:12]
    reserved_node_ids = set(prompt)
    mapping: dict[str, str] = {}
    for node_id in sorted(dependency_node_ids):
        base_id = f"{_REMOTE_REPLICA_NODE_PREFIX}{replica_digest}__{node_id}"
        replica_node_id = base_id
        suffix = 1
        while replica_node_id in reserved_node_ids:
            replica_node_id = f"{base_id}_{suffix}"
            suffix += 1
        mapping[node_id] = replica_node_id
        reserved_node_ids.add(replica_node_id)
    return mapping


def _install_remote_dependency_replica(
    *,
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    dependency_node_ids: set[str],
    source_node_id: str,
    target_group_id: str,
) -> str:
    """Clone one safe producer closure and return its replicated source node id."""
    replica_ids = _replica_node_id_mapping(
        prompt=prompt,
        dependency_node_ids=dependency_node_ids,
        source_node_id=source_node_id,
        target_group_id=target_group_id,
    )
    for node_id in sorted(dependency_node_ids):
        replica_node = copy.deepcopy(prompt[node_id])
        for input_name, input_value in list(
            (replica_node.get("inputs") or {}).items()
        ):
            if _is_link(input_value) and str(input_value[0]) in replica_ids:
                replica_node["inputs"][input_name] = [
                    replica_ids[str(input_value[0])],
                    int(input_value[1]),
                ]
        prompt[replica_ids[node_id]] = replica_node
    remote_node_ids.update(replica_ids.values())
    return replica_ids[source_node_id]


def _cross_group_replicable_boundaries(
    *,
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    component_groups: Mapping[str, set[str]],
    nodes_module: Any,
) -> list[tuple[str, int, str, str, str]]:
    """Return replicable non-transportable edges crossing tentative components."""
    group_by_node_id = {
        node_id: group_id
        for group_id, node_ids in component_groups.items()
        for node_id in node_ids
    }
    boundaries: list[tuple[str, int, str, str, str]] = []
    for target_node_id in sorted(remote_node_ids):
        target_node = prompt.get(target_node_id)
        if not isinstance(target_node, Mapping):
            continue
        for input_name, input_value in (target_node.get("inputs") or {}).items():
            if not _is_link(input_value):
                continue
            source_node_id = str(input_value[0])
            if (
                source_node_id not in remote_node_ids
                or group_by_node_id.get(source_node_id)
                == group_by_node_id.get(target_node_id)
            ):
                continue
            output_index = int(input_value[1])
            io_type = _remote_output_io_type(
                prompt=prompt,
                node_id=source_node_id,
                output_index=output_index,
                nodes_module=nodes_module,
            )
            if io_type not in _REPLICABLE_REMOTE_OBJECT_TYPES:
                continue
            boundaries.append(
                (
                    source_node_id,
                    output_index,
                    target_node_id,
                    str(input_name),
                    str(group_by_node_id[target_node_id]),
                )
            )
    return boundaries


def _replicate_safe_nontransportable_provider_boundaries(
    *,
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> None:
    """Replicate safe object producers instead of forcing provider cycles together."""
    for _iteration in range(len(prompt) + 1):
        consumers = _build_consumer_map(prompt)
        tentative_groups = _remote_component_partition_groups(
            prompt,
            remote_node_ids,
            consumers,
            nodes_module,
            preserve_nontransportable_affinity=False,
        )
        boundaries = _cross_group_replicable_boundaries(
            prompt=prompt,
            remote_node_ids=remote_node_ids,
            component_groups=tentative_groups,
            nodes_module=nodes_module,
        )
        replica_source_by_group: dict[tuple[str, str], str] = {}
        changed = False
        for (
            source_node_id,
            output_index,
            target_node_id,
            input_name,
            group_id,
        ) in boundaries:
            dependency_node_ids = _remote_dependency_closure(prompt, source_node_id)
            if not _can_replicate_remote_dependency_closure(
                prompt=prompt,
                source_node_id=source_node_id,
                dependency_node_ids=dependency_node_ids,
                remote_node_ids=remote_node_ids,
            ):
                continue
            replica_key = (source_node_id, group_id)
            replica_source_node_id = replica_source_by_group.get(replica_key)
            if replica_source_node_id is None:
                replica_source_node_id = _install_remote_dependency_replica(
                    prompt=prompt,
                    remote_node_ids=remote_node_ids,
                    dependency_node_ids=dependency_node_ids,
                    source_node_id=source_node_id,
                    target_group_id=group_id,
                )
                replica_source_by_group[replica_key] = replica_source_node_id
                logger.info(
                    "Replicated non-transportable producer %s and dependencies %s "
                    "for remote component group %s.",
                    source_node_id,
                    sorted(dependency_node_ids),
                    group_id,
                )
            prompt[target_node_id]["inputs"][input_name] = [
                replica_source_node_id,
                output_index,
            ]
            changed = True
        if not changed:
            return
    raise ModalPromptValidationError(
        "Unable to stabilize provider-aware non-transportable dependency replicas."
    )


def _workflow_visible_remote_node_ids(node_ids: Iterable[str]) -> list[str]:
    """Exclude internal dependency replicas from workflow-facing rewrite metadata."""
    return [
        str(node_id)
        for node_id in node_ids
        if not str(node_id).startswith(_REMOTE_REPLICA_NODE_PREFIX)
    ]


def _build_component_plans(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> list[RemoteComponentPlan]:
    """Build plans for every connected remote component."""
    _replicate_safe_nontransportable_provider_boundaries(
        prompt=prompt,
        remote_node_ids=remote_node_ids,
        nodes_module=nodes_module,
    )
    consumers = _build_consumer_map(prompt)
    components = _build_remote_components(
        prompt, remote_node_ids, consumers, nodes_module
    )
    return [
        _build_component_plan(
            component, prompt, consumers, remote_node_ids, nodes_module
        )
        for component in components
    ]


def _mark_remote_to_remote_session_boundaries(
    prompt: dict[str, Any],
    components: list[RemoteComponentPlan],
    nodes_module: Any,
    assignments_by_component_id: Mapping[str, ExecutionAssignment],
) -> set[str]:
    """Keep only same-environment remote edges in provider-local session storage."""
    consumers = _build_consumer_map(prompt)
    component_id_by_node_id = {
        node_id: component.representative_node_id
        for component in components
        for node_id in component.node_ids
    }
    session_sources: set[LinkedOutputRef] = set()

    for component in components:
        for boundary_output in component.boundary_outputs:
            output_consumers = consumers.get(boundary_output.source, [])
            if not output_consumers:
                continue
            remote_consumers = [
                consumer
                for consumer in output_consumers
                if consumer.node_id in component_id_by_node_id
            ]
            if not remote_consumers:
                continue
            producer_assignment = assignments_by_component_id[
                component.representative_node_id
            ]
            consumer_component_ids = {
                component_id_by_node_id[consumer.node_id]
                for consumer in remote_consumers
            }
            cross_environment_consumers = sorted(
                consumer_component_id
                for consumer_component_id in consumer_component_ids
                if assignments_by_component_id[consumer_component_id].environment_id
                != producer_assignment.environment_id
            )
            if cross_environment_consumers:
                logger.info(
                    "Materializing remote boundary output through ComfyUI across "
                    "execution environments source=%s:%d producer=%s environment=%s "
                    "consumer_components=%s.",
                    boundary_output.source.node_id,
                    boundary_output.source.output_index,
                    component.representative_node_id,
                    producer_assignment.environment_id,
                    cross_environment_consumers,
                )
                continue
            (
                non_returning_local_consumers,
                returning_local_consumers,
            ) = _non_returning_local_output_consumers(
                prompt=prompt,
                source=boundary_output.source,
                remote_node_ids=set(component_id_by_node_id),
                consumers=consumers,
                nodes_module=nodes_module,
            )
            if returning_local_consumers:
                logger.info(
                    "Materializing remote boundary output normally because local consumers %s later return to remote execution source=%s:%d.",
                    [consumer.node_id for consumer in returning_local_consumers],
                    boundary_output.source.node_id,
                    boundary_output.source.output_index,
                )
                continue
            if (
                non_returning_local_consumers
                and producer_assignment.provider is not ExecutionProvider.MODAL
            ):
                logger.info(
                    "Materializing remote boundary output through ComfyUI because "
                    "provider-local bridge storage is not accessible to local consumers "
                    "source=%s:%d provider=%s producer_component=%s local_consumers=%s.",
                    boundary_output.source.node_id,
                    boundary_output.source.output_index,
                    producer_assignment.provider.value,
                    component.representative_node_id,
                    [
                        consumer.node_id
                        for consumer in non_returning_local_consumers
                    ],
                )
                continue
            if boundary_output.is_list and _is_transportable_output_type(
                boundary_output.io_type
            ):
                logger.info(
                    "Materializing transportable list boundary output through ComfyUI "
                    "to preserve scheduler item semantics source=%s:%d io_type=%s "
                    "producer_component=%s consumer_components=%s.",
                    boundary_output.source.node_id,
                    boundary_output.source.output_index,
                    boundary_output.io_type,
                    component.representative_node_id,
                    sorted(consumer_component_ids),
                )
                continue
            boundary_output.session_output = True
            boundary_output.session_consumer_node_ids = sorted(
                {consumer.node_id for consumer in remote_consumers}
            )
            if non_returning_local_consumers:
                materializer_node_id = (
                    f"__ModalLocalBridgeMaterializer__"
                    f"{boundary_output.source.node_id}_"
                    f"{boundary_output.source.output_index}"
                )
                reserved_node_ids = {
                    output.local_materializer_node_id
                    for planned_component in components
                    for output in planned_component.boundary_outputs
                    if output.local_materializer_node_id is not None
                }
                while (
                    materializer_node_id in prompt
                    or materializer_node_id in reserved_node_ids
                ):
                    materializer_node_id = f"{materializer_node_id}_proxy"
                boundary_output.local_materializer_node_id = materializer_node_id
                boundary_output.local_materializer_consumer_node_ids = sorted(
                    {consumer.node_id for consumer in non_returning_local_consumers}
                )
                boundary_output.preview_target_node_ids = []
            session_sources.add(boundary_output.source)
            logger.info(
                "Keeping remote boundary output in provider-local storage source=%s:%d io_type=%s provider=%s producer_component=%s consumer_components=%s local_materializer=%s local_consumers=%s.",
                boundary_output.source.node_id,
                boundary_output.source.output_index,
                boundary_output.io_type,
                producer_assignment.provider.value,
                component.representative_node_id,
                sorted(
                    {
                        component_id_by_node_id[consumer.node_id]
                        for consumer in output_consumers
                        if consumer.node_id in component_id_by_node_id
                    }
                ),
                boundary_output.local_materializer_node_id,
                boundary_output.local_materializer_consumer_node_ids,
            )

    return _remote_session_component_ids(
        components=components,
        session_sources=session_sources,
    )


def _remote_session_component_ids(
    *,
    components: list[RemoteComponentPlan],
    session_sources: set[LinkedOutputRef],
) -> set[str]:
    """Return components that produce or consume provider-local boundary refs."""
    return {
        component.representative_node_id
        for component in components
        if any(output.session_output for output in component.boundary_outputs)
        or any(
            boundary_input.source in session_sources
            for boundary_input in component.boundary_inputs
        )
    }


def _boundary_output_payload(
    boundary_output: BoundaryOutputSpec,
    *,
    mapped_output: bool | None = None,
) -> dict[str, Any]:
    """Serialize one component boundary output for a remote payload."""
    payload = {
        "proxy_output_name": boundary_output.proxy_output_name,
        "node_id": boundary_output.source.node_id,
        "output_index": boundary_output.source.output_index,
        "io_type": boundary_output.io_type,
        "is_list": boundary_output.is_list,
        "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
    }
    if boundary_output.session_output:
        payload["session_output"] = True
    if mapped_output is not None:
        payload["mapped_output"] = mapped_output
        if mapped_output:
            payload["scheduler_is_list"] = True
    return payload


def _proxy_boundary_output_is_list(boundary_output: Mapping[str, Any]) -> bool:
    """Return whether ComfyUI should expose one proxy boundary as a list output."""
    return bool(
        boundary_output.get("is_list", False)
        or boundary_output.get("scheduler_is_list", False)
    )


def _implicitly_mapped_boundary_output_sources(
    *,
    component: RemoteComponentPlan,
    original_prompt: dict[str, Any],
    rewritten_prompt: dict[str, Any],
    nodes_module: Any,
) -> set[LinkedOutputRef]:
    """Return component outputs transitively driven by scheduler-list proxy inputs."""
    mapped_target_node_ids: set[str] = set()
    list_boundary_input_names: set[str] = set()

    for boundary_input in component.boundary_inputs:
        input_is_scheduler_list = False
        for target in boundary_input.targets:
            target_prompt_node = rewritten_prompt.get(target.node_id)
            if target_prompt_node is None:
                continue
            current_input_value = (target_prompt_node.get("inputs") or {}).get(
                target.input_name
            )
            if not _is_link(current_input_value):
                continue
            if not _remote_output_is_list(
                prompt=rewritten_prompt,
                node_id=str(current_input_value[0]),
                output_index=int(current_input_value[1]),
                nodes_module=nodes_module,
            ):
                continue
            input_is_scheduler_list = True
            mapped_target_node_ids.add(target.node_id)

        if input_is_scheduler_list:
            list_boundary_input_names.add(boundary_input.proxy_input_name)

    if not mapped_target_node_ids:
        return set()

    consumers = _build_consumer_map(original_prompt)
    mapped_node_ids = _component_downstream_closure(
        seed_node_ids=mapped_target_node_ids,
        component_node_id_set=set(component.node_ids),
        consumers=consumers,
    )
    mapped_output_sources = {
        boundary_output.source
        for boundary_output in component.boundary_outputs
        if boundary_output.source.node_id in mapped_node_ids
    }
    logger.info(
        "Propagating scheduler-list metadata through remote component %s from "
        "boundary inputs=%s to outputs=%s.",
        component.representative_node_id,
        sorted(list_boundary_input_names),
        sorted(
            f"{source.node_id}:{source.output_index}"
            for source in mapped_output_sources
        ),
    )
    return mapped_output_sources


def _mark_payload_scheduler_list_outputs(
    payload: dict[str, Any],
    scheduler_list_output_sources: set[LinkedOutputRef],
) -> None:
    """Mark matching outputs across a component payload and any split phase payloads."""
    if not scheduler_list_output_sources:
        return

    boundary_outputs = payload.get("boundary_outputs")
    if isinstance(boundary_outputs, list):
        for boundary_output in boundary_outputs:
            if not isinstance(boundary_output, dict):
                continue
            source = LinkedOutputRef(
                node_id=str(boundary_output.get("node_id")),
                output_index=int(boundary_output.get("output_index", -1)),
            )
            if source in scheduler_list_output_sources:
                boundary_output["scheduler_is_list"] = True

    for phase_name in ("static_phase", "mapped_phase"):
        phase_payload = payload.get(phase_name)
        if isinstance(phase_payload, dict):
            _mark_payload_scheduler_list_outputs(
                phase_payload,
                scheduler_list_output_sources,
            )

    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        for split_payload in split_proxy_payloads.values():
            if isinstance(split_payload, dict):
                _mark_payload_scheduler_list_outputs(
                    split_payload,
                    scheduler_list_output_sources,
                )
    elif isinstance(split_proxy_payloads, list):
        for split_payload in split_proxy_payloads:
            if isinstance(split_payload, dict):
                _mark_payload_scheduler_list_outputs(
                    split_payload,
                    scheduler_list_output_sources,
                )


def _mapped_boundary_origin_io_type(
    prompt: dict[str, Any],
    boundary_input: BoundaryInputSpec,
    nodes_module: Any,
) -> str | None:
    """Return the effective io_type for one mapped boundary, unwrapping local ModalMapInput markers."""
    source_prompt_node = prompt.get(boundary_input.source.node_id)
    if source_prompt_node is None:
        return boundary_input.io_type

    if str(source_prompt_node.get("class_type")) != MODAL_MAP_INPUT_NODE_ID:
        return _remote_output_io_type(
            prompt=prompt,
            node_id=boundary_input.source.node_id,
            output_index=boundary_input.source.output_index,
            nodes_module=nodes_module,
        )

    mapped_value = (source_prompt_node.get("inputs") or {}).get("value")
    if not _is_link(mapped_value):
        return boundary_input.io_type

    return _remote_output_io_type(
        prompt=prompt,
        node_id=str(mapped_value[0]),
        output_index=int(mapped_value[1]),
        nodes_module=nodes_module,
    )


def _describe_output_boundary_error(
    component: RemoteComponentPlan,
    source: LinkedOutputRef,
    source_class_type: str,
    io_type: str,
    local_consumer: InputTarget,
    local_consumer_class_type: str,
) -> str:
    """Format a human-readable component-output transport validation error."""
    return (
        "Remote component rooted at node "
        f"{component.representative_node_id} exports node {source.node_id} "
        f"({source_class_type}) output index {source.output_index} of type '{io_type}' "
        f"to node {local_consumer.node_id} ({local_consumer_class_type}) outside that "
        f"component at input '{local_consumer.input_name}', which cannot cross the "
        "current component boundary. "
        "Current ComfyUI-Modal transport only supports JSON-compatible values, bytes, "
        "and media or tensor-like outputs such as VIDEO, AUDIO, IMAGE, MASK, LATENT, "
        "SIGMAS, INT, FLOAT, BOOLEAN, and STRING."
    )


def _describe_input_boundary_error(
    component: RemoteComponentPlan,
    target: InputTarget,
    target_class_type: str,
    source: LinkedOutputRef,
    source_class_type: str,
    io_type: str,
) -> str:
    """Format a human-readable component-input transport validation error."""
    return (
        "Remote node "
        f"{target.node_id} ({target_class_type}) input '{target.input_name}' "
        f"depends on node {source.node_id} ({source_class_type}) outside its component, "
        f"using output index {source.output_index} of type '{io_type}', which cannot "
        "cross the current component boundary. Current ComfyUI-Modal transport only "
        "supports JSON-compatible values, bytes, and media or tensor-like outputs such "
        "as VIDEO, AUDIO, IMAGE, MASK, LATENT, SIGMAS, INT, FLOAT, BOOLEAN, and STRING."
    )


def validate_remote_component_transport_compatibility(
    prompt: dict[str, Any],
    components: list[RemoteComponentPlan],
    nodes_module: Any,
) -> None:
    """Reject remote components whose true graph boundaries require unsupported transport."""
    validation_errors: list[str] = []
    consumers = _build_consumer_map(prompt)
    logger.info(
        "Validating %d remote components for transport compatibility.", len(components)
    )

    for component in components:
        for boundary_input in component.boundary_inputs:
            source_prompt_node = prompt.get(boundary_input.source.node_id)
            if source_prompt_node is None:
                continue
            source_class_type = str(source_prompt_node["class_type"])
            source_class = nodes_module.NODE_CLASS_MAPPINGS.get(source_class_type)
            if source_class is None:
                continue

            source_output_types, _, _ = _normalize_output_metadata(source_class)
            if boundary_input.source.output_index >= len(source_output_types):
                continue
            io_type = str(source_output_types[boundary_input.source.output_index])
            if _is_transportable_output_type(io_type):
                continue

            for target in boundary_input.targets:
                target_class_type = str(prompt[target.node_id]["class_type"])
                validation_errors.append(
                    _describe_input_boundary_error(
                        component=component,
                        target=target,
                        target_class_type=target_class_type,
                        source=boundary_input.source,
                        source_class_type=source_class_type,
                        io_type=io_type,
                    )
                )

        for boundary_output in component.boundary_outputs:
            if _is_transportable_output_type(boundary_output.io_type):
                continue

            source_class_type = str(
                prompt[boundary_output.source.node_id]["class_type"]
            )
            for local_consumer in consumers.get(boundary_output.source, []):
                if local_consumer.node_id in component.node_ids:
                    continue
                local_consumer_class_type = str(
                    prompt[local_consumer.node_id]["class_type"]
                )
                validation_errors.append(
                    _describe_output_boundary_error(
                        component=component,
                        source=boundary_output.source,
                        source_class_type=source_class_type,
                        io_type=boundary_output.io_type,
                        local_consumer=local_consumer,
                        local_consumer_class_type=local_consumer_class_type,
                    )
                )

    if validation_errors:
        raise ModalPromptValidationError("\n".join(validation_errors))
    logger.info("Remote component transport validation passed.")

