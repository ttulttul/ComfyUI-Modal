"""Workflow mapping, transportability, and remote graph partitioning."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Callable, Iterator, Mapping

if __package__:
    from .execution_environments import ExecutionProvider, WorkflowExecutionPreferences
    from .llm_profiles import get_llm_profile
    from .modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_MAP_INPUT_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
    )
    from .remote_configuration_nodes import REMOTE_CONFIGURATION_NODE_IDS
    from .remote_plan_types import (
        InputTarget,
        LinkedOutputRef,
        RemoteExpansionReason,
        RemoteNodeAnalysis,
    )
    from .settings import ModalSyncSettings, get_settings
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import ExecutionProvider, WorkflowExecutionPreferences
    from llm_profiles import get_llm_profile
    from modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_MAP_INPUT_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
    )
    from remote_configuration_nodes import REMOTE_CONFIGURATION_NODE_IDS
    from remote_plan_types import (
        InputTarget,
        LinkedOutputRef,
        RemoteExpansionReason,
        RemoteNodeAnalysis,
    )
    from settings import ModalSyncSettings, get_settings

logger = logging.getLogger(__name__)

_REMOTE_TOGGLE_WIDGET_NAME = "Run Remotely"
_LEGACY_REMOTE_TOGGLE_WIDGET_NAME = "Run on Modal"
_LOCAL_ONLY_REMOTE_CLASS_TYPES = frozenset(
    {
        "ModalEndpointChat",
        "VastAILeaseConfiguration",
        *REMOTE_CONFIGURATION_NODE_IDS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_MAP_INPUT_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
    }
)
_TRANSPORTABLE_OUTPUT_TYPES = frozenset(
    {
        "*",
        "AUDIO",
        "BOOLEAN",
        "FLOAT",
        "IMAGE",
        "INT",
        "LATENT",
        "MASK",
        "SIGMAS",
        "STRING",
        "VIDEO",
    }
)
_INEXPENSIVE_REMOTE_BOUNDARY_TYPES = frozenset(
    {
        "BOOLEAN",
        "FLOAT",
        "INT",
        "STRING",
    }
)


def _get_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _prompt_node_required_provider(
    prompt_node: Mapping[str, Any],
) -> ExecutionProvider | None:
    """Return the provider required by one curated node backend, when known."""
    if str(prompt_node.get("class_type") or "") != "ModalLLM":
        return None
    inputs = prompt_node.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    model_reference = inputs.get("model_profile")
    if not isinstance(model_reference, str) or not model_reference.strip():
        return None

    try:
        profile = get_llm_profile(model_reference.strip())
    except ValueError:
        return None
    if profile.backend == "llama_cpp_server":
        return ExecutionProvider.SSH_DOCKER
    return None


def _iter_payload_input_strings(value: Any) -> Iterator[str]:
    """Yield string literals nested inside one serialized prompt input value."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], str):
            return
        for item in value:
            yield from _iter_payload_input_strings(item)
        return
    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_payload_input_strings(nested_value)


def _looks_like_workflow_node(fragment: dict[str, Any]) -> bool:
    """Return whether a JSON fragment resembles a saved ComfyUI workflow node."""
    return "id" in fragment and "properties" in fragment


def _iter_workflow_nodes(
    workflow_fragment: Any,
    visited_object_ids: set[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield workflow node dictionaries from a nested saved workflow fragment."""
    if visited_object_ids is None:
        visited_object_ids = set()

    if isinstance(workflow_fragment, dict):
        object_id = id(workflow_fragment)
        if object_id in visited_object_ids:
            return
        visited_object_ids.add(object_id)

        if _looks_like_workflow_node(workflow_fragment):
            yield workflow_fragment

        for value in workflow_fragment.values():
            yield from _iter_workflow_nodes(value, visited_object_ids)
        return

    if isinstance(workflow_fragment, list):
        object_id = id(workflow_fragment)
        if object_id in visited_object_ids:
            return
        visited_object_ids.add(object_id)

        for item in workflow_fragment:
            yield from _iter_workflow_nodes(item, visited_object_ids)


def _workflow_subgraph_definitions(
    workflow_fragment: Any,
) -> dict[str, dict[str, Any]]:
    """Return reusable ComfyUI subgraph definitions keyed by their node type id."""
    if not isinstance(workflow_fragment, dict):
        return {}
    definitions = workflow_fragment.get("definitions")
    if not isinstance(definitions, dict):
        return {}
    subgraphs = definitions.get("subgraphs")
    if not isinstance(subgraphs, list):
        return {}

    return {
        str(subgraph["id"]): subgraph
        for subgraph in subgraphs
        if isinstance(subgraph, dict) and subgraph.get("id") is not None
    }


def _iter_workflow_nodes_with_ancestors(
    workflow_fragment: Any,
    ancestor_node_ids: tuple[str, ...] = (),
    visited_fragments: set[tuple[int, tuple[str, ...], tuple[str, ...]]] | None = None,
    subgraph_definitions: Mapping[str, dict[str, Any]] | None = None,
    active_subgraph_definition_ids: tuple[str, ...] = (),
) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
    """Yield workflow nodes with instance paths across embedded and defined subgraphs."""
    if visited_fragments is None:
        visited_fragments = set()
    if subgraph_definitions is None:
        subgraph_definitions = _workflow_subgraph_definitions(workflow_fragment)

    if isinstance(workflow_fragment, dict):
        fragment_identity = (
            id(workflow_fragment),
            ancestor_node_ids,
            active_subgraph_definition_ids,
        )
        if fragment_identity in visited_fragments:
            return
        visited_fragments.add(fragment_identity)

        next_ancestor_node_ids = ancestor_node_ids
        if _looks_like_workflow_node(workflow_fragment):
            node_id = str(workflow_fragment.get("id"))
            yield workflow_fragment, ancestor_node_ids
            next_ancestor_node_ids = ancestor_node_ids + (node_id,)

            subgraph_definition_id = str(workflow_fragment.get("type") or "")
            subgraph_definition = subgraph_definitions.get(subgraph_definition_id)
            if (
                subgraph_definition is not None
                and subgraph_definition_id not in active_subgraph_definition_ids
            ):
                yield from _iter_workflow_nodes_with_ancestors(
                    subgraph_definition.get("nodes", []),
                    next_ancestor_node_ids,
                    visited_fragments,
                    subgraph_definitions,
                    active_subgraph_definition_ids + (subgraph_definition_id,),
                )

        for key, value in workflow_fragment.items():
            if key == "definitions":
                continue
            yield from _iter_workflow_nodes_with_ancestors(
                value,
                next_ancestor_node_ids,
                visited_fragments,
                subgraph_definitions,
                active_subgraph_definition_ids,
            )
        return

    if isinstance(workflow_fragment, list):
        fragment_identity = (
            id(workflow_fragment),
            ancestor_node_ids,
            active_subgraph_definition_ids,
        )
        if fragment_identity in visited_fragments:
            return
        visited_fragments.add(fragment_identity)

        for item in workflow_fragment:
            yield from _iter_workflow_nodes_with_ancestors(
                item,
                ancestor_node_ids,
                visited_fragments,
                subgraph_definitions,
                active_subgraph_definition_ids,
            )


def _resolve_prompt_node_ids_for_workflow_node(
    workflow_node_id: str,
    ancestor_node_ids: tuple[str, ...],
    prompt_node_ids: set[str],
) -> set[str]:
    """Resolve one saved workflow node id to matching queued prompt node ids."""
    path_segments = ancestor_node_ids + (workflow_node_id,)

    for index in range(len(path_segments)):
        candidate = ":".join(path_segments[index:])
        resolved_prompt_node_ids: set[str] = set()
        if candidate in prompt_node_ids:
            resolved_prompt_node_ids.add(candidate)
        descendant_prefix = f"{candidate}:"
        resolved_prompt_node_ids.update(
            prompt_node_id
            for prompt_node_id in prompt_node_ids
            if prompt_node_id.startswith(descendant_prefix)
        )
        if resolved_prompt_node_ids:
            return resolved_prompt_node_ids

    resolved_prompt_node_ids: set[str] = set()

    for ancestor_node_id in reversed(ancestor_node_ids):
        if ancestor_node_id in prompt_node_ids:
            resolved_prompt_node_ids.add(ancestor_node_id)
            descendant_prefix = f"{ancestor_node_id}:"
            resolved_prompt_node_ids.update(
                prompt_node_id
                for prompt_node_id in prompt_node_ids
                if prompt_node_id.startswith(descendant_prefix)
            )
            break

    return resolved_prompt_node_ids


def _workflow_node_path(
    workflow_node_id: str, ancestor_node_ids: tuple[str, ...]
) -> str:
    """Return one workflow node's composed path, including subgraph ancestors."""
    if not ancestor_node_ids:
        return workflow_node_id
    return ":".join((*ancestor_node_ids, workflow_node_id))


def _extract_marked_workflow_node_paths(
    workflow: dict[str, Any] | None,
    settings: ModalSyncSettings | None = None,
) -> set[str]:
    """Return composed workflow paths for nodes explicitly marked remote in metadata."""
    if workflow is None:
        return set()

    marker = (settings or get_settings()).marker_property
    marked_workflow_node_paths: set[str] = set()
    for node, ancestor_node_ids in _iter_workflow_nodes_with_ancestors(workflow):
        properties = node.get("properties") or {}
        if not properties.get(marker):
            continue
        marked_workflow_node_paths.add(
            _workflow_node_path(str(node.get("id")), ancestor_node_ids)
        )
    return marked_workflow_node_paths


def _build_workflow_prompt_resolution_maps(
    workflow: dict[str, Any] | None,
    prompt_node_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return bidirectional mappings between workflow paths and queued prompt ids."""
    workflow_path_to_prompt_node_ids: dict[str, set[str]] = {}
    prompt_node_id_to_workflow_paths: dict[str, set[str]] = defaultdict(set)

    if workflow is None:
        for prompt_node_id in prompt_node_ids:
            workflow_path_to_prompt_node_ids[prompt_node_id] = {prompt_node_id}
            prompt_node_id_to_workflow_paths[prompt_node_id].add(prompt_node_id)
        return workflow_path_to_prompt_node_ids, prompt_node_id_to_workflow_paths

    for node, ancestor_node_ids in _iter_workflow_nodes_with_ancestors(workflow):
        workflow_node_id = str(node.get("id"))
        workflow_node_path = _workflow_node_path(workflow_node_id, ancestor_node_ids)
        resolved_prompt_node_ids = _resolve_prompt_node_ids_for_workflow_node(
            workflow_node_id,
            ancestor_node_ids,
            prompt_node_ids,
        )
        if not ancestor_node_ids and workflow_node_id in prompt_node_ids:
            resolved_prompt_node_ids.add(workflow_node_id)
        if workflow_node_path in prompt_node_ids:
            resolved_prompt_node_ids.add(workflow_node_path)
        if not resolved_prompt_node_ids:
            continue

        workflow_path_to_prompt_node_ids[workflow_node_path] = resolved_prompt_node_ids
        for prompt_node_id in resolved_prompt_node_ids:
            prompt_node_id_to_workflow_paths[prompt_node_id].add(workflow_node_path)

    for prompt_node_id in prompt_node_ids:
        workflow_path_to_prompt_node_ids.setdefault(prompt_node_id, {prompt_node_id})
        prompt_node_id_to_workflow_paths[prompt_node_id].add(prompt_node_id)

    return workflow_path_to_prompt_node_ids, prompt_node_id_to_workflow_paths


def _resolve_requested_prompt_node_ids(
    requested_workflow_node_paths: set[str],
    prompt_node_ids: set[str],
    workflow_path_to_prompt_node_ids: dict[str, set[str]],
) -> set[str]:
    """Resolve requested workflow-node paths to the prompt ids that queue-time rewrite sees."""
    requested_prompt_node_ids: set[str] = set()
    for requested_workflow_node_path in requested_workflow_node_paths:
        if requested_workflow_node_path in workflow_path_to_prompt_node_ids:
            requested_prompt_node_ids.update(
                workflow_path_to_prompt_node_ids[requested_workflow_node_path]
            )
            continue
        if requested_workflow_node_path in prompt_node_ids:
            requested_prompt_node_ids.add(requested_workflow_node_path)
    return requested_prompt_node_ids


def _best_workflow_path_for_prompt_node(
    prompt_node_id: str,
    prompt_node_id_to_workflow_paths: dict[str, set[str]],
) -> str:
    """Choose the most specific workflow path for one queued prompt node id."""
    workflow_node_paths = prompt_node_id_to_workflow_paths.get(
        prompt_node_id, {prompt_node_id}
    )
    return max(
        workflow_node_paths,
        key=lambda workflow_node_path: (
            workflow_node_path.count(":"),
            len(workflow_node_path),
            workflow_node_path,
        ),
    )


def _resolve_workflow_node_paths_for_prompt_nodes(
    prompt_node_ids: set[str],
    prompt_node_id_to_workflow_paths: dict[str, set[str]],
) -> set[str]:
    """Map queued prompt ids back to the workflow node paths the UI can mark remote."""
    return {
        _best_workflow_path_for_prompt_node(
            prompt_node_id, prompt_node_id_to_workflow_paths
        )
        for prompt_node_id in prompt_node_ids
    }


def extract_remote_node_ids(
    workflow: dict[str, Any] | None,
    settings: ModalSyncSettings | None = None,
    prompt_node_ids: set[str] | None = None,
) -> set[str]:
    """Return the node ids marked for remote execution in the workflow metadata."""
    if workflow is None:
        return set()

    marker = (settings or get_settings()).marker_property
    remote_node_ids: set[str] = set()
    for node, ancestor_node_ids in _iter_workflow_nodes_with_ancestors(workflow):
        if _workflow_node_remote_enabled(node, marker):
            node_id = str(node.get("id"))
            if prompt_node_ids is None:
                remote_node_ids.add(node_id)
                continue

            resolved_prompt_node_ids = _resolve_prompt_node_ids_for_workflow_node(
                node_id,
                ancestor_node_ids,
                prompt_node_ids,
            )
            if resolved_prompt_node_ids:
                remote_node_ids.update(resolved_prompt_node_ids)
                logger.info(
                    "Mapped workflow Modal marker from node %s with ancestors %s to prompt nodes %s.",
                    node_id,
                    list(ancestor_node_ids),
                    sorted(resolved_prompt_node_ids),
                )
    return remote_node_ids


def requested_remote_node_ids(
    *,
    prompt: Mapping[str, Any],
    workflow: Mapping[str, Any] | None,
    settings: ModalSyncSettings,
) -> set[str]:
    """Return explicit remote markers or every eligible node for auto placement."""
    preferences = WorkflowExecutionPreferences.from_workflow(workflow)
    if not preferences.auto_place:
        return extract_remote_node_ids(
            dict(workflow) if workflow is not None else None,
            settings,
            set(prompt),
        )
    selected_node_ids = {
        str(node_id)
        for node_id, prompt_node in prompt.items()
        if isinstance(prompt_node, Mapping)
        and str(prompt_node.get("class_type") or "")
        not in _LOCAL_ONLY_REMOTE_CLASS_TYPES
        and not str(prompt_node.get("class_type") or "").startswith(
            "ModalUniversalExecutor"
        )
    }
    logger.info(
        "Automatic remote placement selected %d eligible prompt nodes.",
        len(selected_node_ids),
    )
    return selected_node_ids


def _workflow_node_remote_enabled(node: Mapping[str, Any], marker: str) -> bool:
    """Return the visible remote toggle, falling back to the saved marker property."""
    named_widget_values = node.get("widgets_values_named")
    if isinstance(named_widget_values, Mapping):
        visible_toggle_value = named_widget_values.get(_REMOTE_TOGGLE_WIDGET_NAME)
        if not isinstance(visible_toggle_value, bool):
            visible_toggle_value = named_widget_values.get(
                _LEGACY_REMOTE_TOGGLE_WIDGET_NAME
            )
        if isinstance(visible_toggle_value, bool):
            return visible_toggle_value

    properties = node.get("properties")
    return isinstance(properties, Mapping) and bool(properties.get(marker))


def _normalize_output_metadata(
    node_class: type[Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    """Return normalized output metadata for a node class."""
    if hasattr(node_class, "GET_SCHEMA"):
        node_class.GET_SCHEMA()

    output_types = tuple(getattr(node_class, "RETURN_TYPES", ("*",))) or ("*",)
    default_names = tuple(f"output_{index}" for index, _ in enumerate(output_types))
    output_names = tuple(getattr(node_class, "RETURN_NAMES", default_names))
    output_is_list = tuple(
        getattr(node_class, "OUTPUT_IS_LIST", (False,) * len(output_types))
    )

    if len(output_names) < len(output_types):
        output_names = output_names + default_names[len(output_names) :]
    if len(output_is_list) < len(output_types):
        output_is_list = output_is_list + (False,) * (
            len(output_types) - len(output_is_list)
        )

    return (
        output_types,
        output_names[: len(output_types)],
        output_is_list[: len(output_types)],
    )


def _is_transportable_output_type(io_type: str) -> bool:
    """Return whether a ComfyUI output type can cross the current transport."""
    normalized_parts = [part.strip() for part in io_type.split(",") if part.strip()]
    return bool(normalized_parts) and all(
        part in _TRANSPORTABLE_OUTPUT_TYPES for part in normalized_parts
    )


def _is_inexpensive_remote_boundary_type(io_type: str) -> bool:
    """Return whether a remote edge is cheap enough to remain a component boundary."""
    normalized_parts = [part.strip() for part in io_type.split(",") if part.strip()]
    return bool(normalized_parts) and all(
        part in _INEXPENSIVE_REMOTE_BOUNDARY_TYPES for part in normalized_parts
    )


def _build_consumer_map(
    prompt: dict[str, Any]
) -> dict[LinkedOutputRef, list[InputTarget]]:
    """Build a reverse map from node outputs to downstream prompt inputs."""
    consumers: dict[LinkedOutputRef, list[InputTarget]] = defaultdict(list)
    for node_id, prompt_node in prompt.items():
        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(
                node_id=str(input_value[0]), output_index=int(input_value[1])
            )
            consumers[source].append(
                InputTarget(node_id=str(node_id), input_name=str(input_name))
            )
    return consumers


def _sandwiched_local_node_ids(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
) -> set[str]:
    """Return local nodes on paths that leave and then re-enter remote execution."""
    downstream_node_ids: dict[str, set[str]] = defaultdict(set)
    upstream_node_ids: dict[str, set[str]] = defaultdict(set)
    for target_node_id, prompt_node in prompt.items():
        for input_value in (prompt_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            source_node_id = str(input_value[0])
            safe_target_node_id = str(target_node_id)
            downstream_node_ids[source_node_id].add(safe_target_node_id)
            upstream_node_ids[safe_target_node_id].add(source_node_id)

    def local_closure(adjacency: Mapping[str, set[str]]) -> set[str]:
        """Return local nodes reachable from the remote seed set in one direction."""
        reachable_local_node_ids: set[str] = set()
        pending_node_ids = list(sorted(remote_node_ids))
        visited_node_ids = set(remote_node_ids)
        while pending_node_ids:
            current_node_id = pending_node_ids.pop()
            for adjacent_node_id in adjacency.get(current_node_id, set()):
                if adjacent_node_id in visited_node_ids:
                    continue
                visited_node_ids.add(adjacent_node_id)
                if adjacent_node_id in remote_node_ids:
                    continue
                if adjacent_node_id not in prompt:
                    continue
                reachable_local_node_ids.add(adjacent_node_id)
                pending_node_ids.append(adjacent_node_id)
        return reachable_local_node_ids

    downstream_local_node_ids = local_closure(downstream_node_ids)
    upstream_local_node_ids = local_closure(upstream_node_ids)
    return downstream_local_node_ids & upstream_local_node_ids


def _node_output_refs(
    prompt: dict[str, Any], node_id: str, nodes_module: Any
) -> list[LinkedOutputRef]:
    """Return declared output refs for one prompt node."""
    prompt_node = prompt.get(node_id)
    if prompt_node is None:
        return []
    node_class = nodes_module.NODE_CLASS_MAPPINGS.get(
        str(prompt_node.get("class_type"))
    )
    if node_class is None:
        return []
    output_types, _, _ = _normalize_output_metadata(node_class)
    return [
        LinkedOutputRef(node_id=node_id, output_index=output_index)
        for output_index, _io_type in enumerate(output_types)
    ]


def _downstream_node_ids_from_targets(
    *,
    prompt: dict[str, Any],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    seed_targets: list[InputTarget],
    nodes_module: Any,
) -> set[str]:
    """Return every node reachable downstream from the supplied input targets."""
    visited_node_ids: set[str] = set()
    pending_node_ids = deque(str(target.node_id) for target in seed_targets)
    while pending_node_ids:
        node_id = pending_node_ids.popleft()
        if node_id in visited_node_ids or node_id not in prompt:
            continue
        visited_node_ids.add(node_id)
        for output_ref in _node_output_refs(prompt, node_id, nodes_module):
            for downstream_target in consumers.get(output_ref, []):
                downstream_node_id = str(downstream_target.node_id)
                if downstream_node_id not in visited_node_ids:
                    pending_node_ids.append(downstream_node_id)
    return visited_node_ids


def _is_non_returning_tap_terminal_node(
    *,
    prompt_node: dict[str, Any] | None,
    nodes_module: Any,
) -> bool:
    """Return whether a local tap node should stay local instead of being absorbed."""
    if prompt_node is None:
        return False
    class_type = str(prompt_node.get("class_type"))
    if class_type == "PreviewImage":
        return True
    node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
    return bool(getattr(node_class, "OUTPUT_NODE", False))


def _non_returning_local_tap_node_ids(
    *,
    prompt: dict[str, Any],
    component_node_ids: set[str],
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> tuple[set[str], set[str]]:
    """Return non-returning local tap branches without making local nodes remote."""
    local_tap_node_ids: set[str] = set()
    local_tap_terminal_node_ids: set[str] = set()
    for component_node_id in sorted(component_node_ids):
        for source in _node_output_refs(prompt, component_node_id, nodes_module):
            output_consumers = consumers.get(source, [])
            local_consumers = [
                target
                for target in output_consumers
                if target.node_id not in component_node_ids
                and target.node_id not in remote_node_ids
            ]
            remote_consumers = [
                target
                for target in output_consumers
                if target.node_id in remote_node_ids
            ]
            if not local_consumers or not remote_consumers:
                continue
            for local_consumer in local_consumers:
                branch_node_ids = _downstream_node_ids_from_targets(
                    prompt=prompt,
                    consumers=consumers,
                    seed_targets=[local_consumer],
                    nodes_module=nodes_module,
                )
                if branch_node_ids & remote_node_ids:
                    continue
                local_tap_terminal_node_ids.update(branch_node_ids)
    return local_tap_node_ids, local_tap_terminal_node_ids


def _non_returning_local_output_consumers(
    *,
    prompt: dict[str, Any],
    source: LinkedOutputRef,
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> tuple[list[InputTarget], list[InputTarget]]:
    """Partition local consumers by whether their branch later returns remote."""
    non_returning_consumers: list[InputTarget] = []
    returning_consumers: list[InputTarget] = []
    for target in consumers.get(source, []):
        if target.node_id in remote_node_ids:
            continue
        branch_node_ids = _downstream_node_ids_from_targets(
            prompt=prompt,
            consumers=consumers,
            seed_targets=[target],
            nodes_module=nodes_module,
        )
        if branch_node_ids and not (branch_node_ids & remote_node_ids):
            non_returning_consumers.append(target)
        else:
            returning_consumers.append(target)
    return non_returning_consumers, returning_consumers


def _output_supports_parallel_local_materialization(
    *,
    prompt: dict[str, Any],
    source: LinkedOutputRef,
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> bool:
    """Return whether every local consumer is independent of later remote work."""
    (
        non_returning_consumers,
        returning_consumers,
    ) = _non_returning_local_output_consumers(
        prompt=prompt,
        source=source,
        remote_node_ids=remote_node_ids,
        consumers=consumers,
        nodes_module=nodes_module,
    )
    return bool(non_returning_consumers) and not returning_consumers


def _expand_component_for_non_transportable_local_outputs(
    *,
    prompt: dict[str, Any],
    component_node_ids: set[str],
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> tuple[set[str], set[str]]:
    """Absorb local consumers needed to avoid non-transportable component outputs."""
    expanded_node_ids = set(component_node_ids)
    added_node_ids: set[str] = set()
    changed = True
    while changed:
        changed = False
        (
            upstream_expanded_node_ids,
            _expansion_reasons,
        ) = _expand_remote_node_ids_for_non_transportable_inputs(
            prompt=prompt,
            remote_node_ids=expanded_node_ids,
            nodes_module=nodes_module,
        )
        upstream_added_node_ids = upstream_expanded_node_ids - expanded_node_ids
        if upstream_added_node_ids:
            expanded_node_ids = upstream_expanded_node_ids
            added_node_ids.update(upstream_added_node_ids)
            changed = True

        for node_id in sorted(expanded_node_ids):
            prompt_node = prompt.get(node_id)
            if prompt_node is None:
                continue
            node_class = nodes_module.NODE_CLASS_MAPPINGS.get(
                str(prompt_node.get("class_type"))
            )
            if node_class is None:
                continue
            output_types, _, _ = _normalize_output_metadata(node_class)
            for output_index, io_type in enumerate(output_types):
                if _is_transportable_output_type(str(io_type)):
                    continue
                source = LinkedOutputRef(node_id=node_id, output_index=output_index)
                for target in consumers.get(source, []):
                    target_node_id = str(target.node_id)
                    if (
                        target_node_id in expanded_node_ids
                        or target_node_id in remote_node_ids
                    ):
                        continue
                    if target_node_id not in prompt:
                        continue
                    if _is_non_returning_tap_terminal_node(
                        prompt_node=prompt.get(target_node_id),
                        nodes_module=nodes_module,
                    ):
                        continue
                    expanded_node_ids.add(target_node_id)
                    added_node_ids.add(target_node_id)
                    changed = True
                    logger.info(
                        "Absorbing local node %s because it consumes non-transportable output %s:%d from a preview-tap remote component.",
                        target_node_id,
                        source.node_id,
                        source.output_index,
                    )
    return expanded_node_ids, added_node_ids


def _remote_output_io_type(
    *,
    prompt: dict[str, Any],
    node_id: str,
    output_index: int,
    nodes_module: Any,
) -> str | None:
    """Return the declared output type for one prompt node output when available."""
    prompt_node = prompt.get(node_id)
    if prompt_node is None:
        return None

    class_type = str(prompt_node["class_type"])
    node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        return None

    output_types, _, _ = _normalize_output_metadata(node_class)
    if output_index < 0 or output_index >= len(output_types):
        return None
    return str(output_types[output_index])


def _remote_output_is_list(
    *,
    prompt: dict[str, Any],
    node_id: str,
    output_index: int,
    nodes_module: Any,
) -> bool:
    """Return whether one prompt node output is declared as a list output."""
    prompt_node = prompt.get(node_id)
    if prompt_node is None:
        return False

    class_type = str(prompt_node["class_type"])
    node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        return False

    output_types, _output_names, output_is_list = _normalize_output_metadata(node_class)
    if output_index < 0 or output_index >= len(output_types):
        return False
    return bool(output_is_list[output_index])


def _remote_group_dependency_edges(
    prompt: Mapping[str, Any],
    remote_node_ids: set[str],
    group_for_node: Callable[[str], str],
) -> dict[str, set[str]]:
    """Return current coarse dependency edges for a remote-node grouping."""
    dependency_edges: dict[str, set[str]] = defaultdict(set)
    for downstream_node_id in remote_node_ids:
        downstream_node = prompt.get(downstream_node_id)
        if not isinstance(downstream_node, Mapping):
            continue
        downstream_group = group_for_node(downstream_node_id)
        for input_value in (downstream_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            if upstream_node_id not in remote_node_ids:
                continue
            upstream_group = group_for_node(upstream_node_id)
            if upstream_group != downstream_group:
                dependency_edges[upstream_group].add(downstream_group)
    return dependency_edges


def _has_alternate_group_path_through_protected_group(
    dependency_edges: Mapping[str, set[str]],
    start_group: str,
    target_group: str,
    protected_groups: set[str],
) -> bool:
    """Return whether a non-direct path crosses a provider-constrained group."""
    pending = [
        (downstream_group, downstream_group in protected_groups)
        for downstream_group in dependency_edges.get(start_group, set())
        if downstream_group != target_group
    ]
    visited: set[tuple[str, bool]] = set()
    while pending:
        current_group, crossed_protected_group = pending.pop()
        if current_group == target_group and crossed_protected_group:
            return True
        state = (current_group, crossed_protected_group)
        if state in visited:
            continue
        visited.add(state)
        pending.extend(
            (
                downstream_group,
                crossed_protected_group or downstream_group in protected_groups,
            )
            for downstream_group in dependency_edges.get(current_group, set())
        )
    return False


def _remote_component_partition_groups(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
    *,
    preserve_nontransportable_affinity: bool = True,
) -> dict[str, set[str]]:
    """Return component groups after merging remote nodes across costly boundaries."""
    parent: dict[str, str] = {node_id: node_id for node_id in remote_node_ids}
    downstream_remote_node_ids_by_node_id: dict[str, set[str]] = defaultdict(set)
    provider_constrained_node_ids = {
        node_id
        for node_id in remote_node_ids
        if isinstance((prompt_node := prompt.get(node_id)), Mapping)
        and _prompt_node_required_provider(prompt_node) is not None
    }

    def find(node_id: str) -> str:
        """Return the canonical union-find representative for one remote node."""
        root = parent[node_id]
        while root != parent[root]:
            root = parent[root]
        while node_id != root:
            next_node_id = parent[node_id]
            parent[node_id] = root
            node_id = next_node_id
        return root

    def union(
        left_node_id: str,
        right_node_id: str,
        *,
        preserve_coarse_dag: bool | None = None,
    ) -> bool:
        """Merge two remote nodes unless doing so creates a coarse dependency cycle."""
        left_root = find(left_node_id)
        right_root = find(right_node_id)
        if left_root == right_root:
            return True
        should_preserve_coarse_dag = (
            bool(provider_constrained_node_ids)
            if preserve_coarse_dag is None
            else preserve_coarse_dag
        )
        if should_preserve_coarse_dag:
            dependency_edges = _remote_group_dependency_edges(
                prompt,
                remote_node_ids,
                find,
            )
            protected_groups = {
                find(node_id) for node_id in provider_constrained_node_ids
            }
            creates_cycle = _has_alternate_group_path_through_protected_group(
                dependency_edges,
                left_root,
                right_root,
                protected_groups,
            ) or _has_alternate_group_path_through_protected_group(
                dependency_edges,
                right_root,
                left_root,
                protected_groups,
            )
            if creates_cycle:
                logger.info(
                    "Splitting remote nodes %s and %s to preserve an acyclic "
                    "component graph.",
                    left_node_id,
                    right_node_id,
                )
                return False
        canonical_root = min(left_root, right_root)
        merged_root = max(left_root, right_root)
        parent[merged_root] = canonical_root
        return True

    for node_id in sorted(remote_node_ids):
        prompt_node = prompt.get(node_id)
        if prompt_node is None:
            continue
        for input_value in (prompt_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            if upstream_node_id not in remote_node_ids:
                continue
            downstream_remote_node_ids_by_node_id[upstream_node_id].add(node_id)
            upstream_prompt_node = prompt.get(upstream_node_id)
            source = LinkedOutputRef(
                node_id=upstream_node_id, output_index=int(input_value[1])
            )
            if (
                upstream_prompt_node is not None
                and str(upstream_prompt_node.get("class_type"))
                == MODAL_MAP_INPUT_NODE_ID
            ):
                union(node_id, upstream_node_id, preserve_coarse_dag=False)
                continue
            io_type = _remote_output_io_type(
                prompt=prompt,
                node_id=upstream_node_id,
                output_index=source.output_index,
                nodes_module=nodes_module,
            )
            if (
                preserve_nontransportable_affinity
                and io_type is not None
                and not _is_transportable_output_type(io_type)
            ):
                logger.info(
                    "Co-locating remote nodes %s -> %s because %s output %d "
                    "cannot cross a component boundary.",
                    upstream_node_id,
                    node_id,
                    io_type,
                    source.output_index,
                )
                union(node_id, upstream_node_id, preserve_coarse_dag=False)
                continue
            downstream_provider = _prompt_node_required_provider(prompt_node)
            upstream_provider = (
                _prompt_node_required_provider(upstream_prompt_node)
                if isinstance(upstream_prompt_node, Mapping)
                else None
            )
            if downstream_provider != upstream_provider and (
                downstream_provider is not None or upstream_provider is not None
            ):
                logger.info(
                    "Splitting remote nodes %s -> %s across provider boundary %s -> %s.",
                    upstream_node_id,
                    node_id,
                    upstream_provider.value if upstream_provider is not None else "any",
                    downstream_provider.value
                    if downstream_provider is not None
                    else "any",
                )
                continue
            if _output_supports_parallel_local_materialization(
                prompt=prompt,
                source=source,
                remote_node_ids=remote_node_ids,
                consumers=consumers,
                nodes_module=nodes_module,
            ):
                logger.info(
                    "Splitting remote nodes %s -> %s because output %d also feeds only non-returning local work.",
                    upstream_node_id,
                    node_id,
                    source.output_index,
                )
                continue
            if io_type is not None and _is_inexpensive_remote_boundary_type(io_type):
                continue
            if io_type is not None and _is_transportable_output_type(io_type):
                logger.info(
                    "Co-locating remote nodes %s -> %s because %s output %d is "
                    "transportable but expensive.",
                    upstream_node_id,
                    node_id,
                    io_type,
                    source.output_index,
                )
            union(node_id, upstream_node_id)

    for remote_node_id in sorted(remote_node_ids):
        prompt_node = prompt.get(remote_node_id)
        if (
            prompt_node is None
            or str(prompt_node.get("class_type")) != MODAL_MAP_INPUT_NODE_ID
        ):
            continue
        pending_node_ids = [remote_node_id]
        visited_node_ids: set[str] = set()
        while pending_node_ids:
            current_node_id = pending_node_ids.pop()
            if current_node_id in visited_node_ids:
                continue
            visited_node_ids.add(current_node_id)
            union(
                remote_node_id,
                current_node_id,
                preserve_coarse_dag=False,
            )
            for downstream_node_id in sorted(
                downstream_remote_node_ids_by_node_id.get(current_node_id, set())
            ):
                pending_node_ids.append(downstream_node_id)

    groups: dict[str, set[str]] = defaultdict(set)
    for node_id in sorted(remote_node_ids):
        groups[find(node_id)].add(node_id)
    return groups


def _component_topological_order(
    prompt: dict[str, Any],
    component_groups: dict[str, set[str]],
) -> list[list[str]]:
    """Return component node ids ordered from upstream to downstream."""
    _, dependency_edges, indegree_by_component_id = _component_dependency_graph(
        prompt, component_groups
    )
    merged_component_groups = _merge_cyclic_component_groups(
        component_groups=component_groups,
        dependency_edges=dependency_edges,
    )
    if merged_component_groups != component_groups:
        return _component_topological_order(prompt, merged_component_groups)

    ready_component_ids = deque(
        sorted(
            [
                component_id
                for component_id, indegree in indegree_by_component_id.items()
                if indegree == 0
            ]
        )
    )
    ordered_components: list[list[str]] = []
    emitted_component_ids: set[str] = set()

    while ready_component_ids:
        component_id = ready_component_ids.popleft()
        if component_id in emitted_component_ids:
            continue
        emitted_component_ids.add(component_id)
        ordered_components.append(sorted(component_groups[component_id]))
        for downstream_component_id in sorted(dependency_edges[component_id]):
            indegree_by_component_id[downstream_component_id] -= 1
            if indegree_by_component_id[downstream_component_id] == 0:
                ready_component_ids.append(downstream_component_id)

    if len(emitted_component_ids) == len(component_groups):
        return ordered_components

    logger.warning(
        "Transport-aware component ordering encountered a cycle or unresolved dependency; falling back to stable component order."
    )
    for component_id in sorted(component_groups):
        if component_id in emitted_component_ids:
            continue
        ordered_components.append(sorted(component_groups[component_id]))
    return ordered_components


def _component_dependency_graph(
    prompt: dict[str, Any],
    component_groups: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, int]]:
    """Return component membership, downstream edges, and indegrees for the coarse component DAG."""
    component_id_by_node_id: dict[str, str] = {}
    for representative_node_id, component_node_ids in component_groups.items():
        for node_id in component_node_ids:
            component_id_by_node_id[node_id] = representative_node_id

    dependency_edges: dict[str, set[str]] = {
        representative_node_id: set() for representative_node_id in component_groups
    }
    indegree_by_component_id: dict[str, int] = {
        representative_node_id: 0 for representative_node_id in component_groups
    }

    for node_id, representative_node_id in component_id_by_node_id.items():
        prompt_node = prompt.get(node_id)
        if prompt_node is None:
            continue
        for input_value in (prompt_node.get("inputs") or {}).values():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            upstream_component_id = component_id_by_node_id.get(upstream_node_id)
            if (
                upstream_component_id is None
                or upstream_component_id == representative_node_id
            ):
                continue
            if representative_node_id in dependency_edges[upstream_component_id]:
                continue
            dependency_edges[upstream_component_id].add(representative_node_id)
            indegree_by_component_id[representative_node_id] += 1

    return component_id_by_node_id, dependency_edges, indegree_by_component_id


def _component_execution_stages(
    prompt: dict[str, Any],
    component_groups: dict[str, set[str]],
) -> list[list[str]]:
    """Return one best-effort stage decomposition for concurrent remote component execution."""
    _, dependency_edges, indegree_by_component_id = _component_dependency_graph(
        prompt, component_groups
    )
    merged_component_groups = _merge_cyclic_component_groups(
        component_groups=component_groups,
        dependency_edges=dependency_edges,
    )
    if merged_component_groups != component_groups:
        return _component_execution_stages(prompt, merged_component_groups)

    remaining_indegrees = dict(indegree_by_component_id)
    ready_component_ids = sorted(
        [
            component_id
            for component_id, indegree in remaining_indegrees.items()
            if indegree == 0
        ]
    )
    execution_stages: list[list[str]] = []
    emitted_component_ids: set[str] = set()

    while ready_component_ids:
        current_stage = list(ready_component_ids)
        execution_stages.append(current_stage)
        next_ready_component_ids: set[str] = set()
        for component_id in current_stage:
            emitted_component_ids.add(component_id)
            for downstream_component_id in sorted(
                dependency_edges.get(component_id, set())
            ):
                remaining_indegrees[downstream_component_id] -= 1
                if remaining_indegrees[downstream_component_id] == 0:
                    next_ready_component_ids.add(downstream_component_id)
        ready_component_ids = sorted(
            component_id
            for component_id in next_ready_component_ids
            if component_id not in emitted_component_ids
        )

    if len(emitted_component_ids) == len(component_groups):
        return execution_stages

    fallback_stage = [
        component_id
        for component_id in sorted(component_groups)
        if component_id not in emitted_component_ids
    ]
    if fallback_stage:
        logger.warning(
            "Transport-aware execution-stage planning encountered a cycle or unresolved dependency; appending fallback stage %s.",
            fallback_stage,
        )
        execution_stages.append(fallback_stage)
    return execution_stages


def _estimated_stage_parallelism(
    execution_stages: list[list[str]],
    mapped_component_ids: set[str],
    *,
    mapped_component_weight: int,
    max_parallelism_cap: int | None = None,
) -> int:
    """Return the weighted best-effort parallelism estimate over staged remote execution."""
    stage_parallelism = 0
    for stage in execution_stages:
        current_stage_parallelism = sum(
            mapped_component_weight if component_id in mapped_component_ids else 1
            for component_id in stage
        )
        stage_parallelism = max(stage_parallelism, current_stage_parallelism)
    if max_parallelism_cap is not None:
        return min(stage_parallelism, max_parallelism_cap)
    return stage_parallelism


def _merge_cyclic_component_groups(
    *,
    component_groups: dict[str, set[str]],
    dependency_edges: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Collapse cyclic coarse component groups into SCC-merged groups."""
    component_ids = sorted(component_groups)
    reverse_edges: dict[str, set[str]] = {
        component_id: set() for component_id in component_ids
    }
    for upstream_component_id, downstream_component_ids in dependency_edges.items():
        for downstream_component_id in downstream_component_ids:
            reverse_edges.setdefault(downstream_component_id, set()).add(
                upstream_component_id
            )

    visited_component_ids: set[str] = set()
    finish_order: list[str] = []

    def visit_forward(component_id: str) -> None:
        """Record reverse-topological finish order over the coarse graph."""
        if component_id in visited_component_ids:
            return
        visited_component_ids.add(component_id)
        for downstream_component_id in sorted(
            dependency_edges.get(component_id, set())
        ):
            visit_forward(downstream_component_id)
        finish_order.append(component_id)

    for component_id in component_ids:
        visit_forward(component_id)

    assigned_component_ids: set[str] = set()
    merged_groups: dict[str, set[str]] = {}
    merged_sccs = 0

    def visit_reverse(component_id: str, scc_component_ids: set[str]) -> None:
        """Collect one SCC by walking reverse edges from the finish-order seed."""
        if component_id in assigned_component_ids:
            return
        assigned_component_ids.add(component_id)
        scc_component_ids.add(component_id)
        for upstream_component_id in sorted(reverse_edges.get(component_id, set())):
            visit_reverse(upstream_component_id, scc_component_ids)

    for component_id in reversed(finish_order):
        if component_id in assigned_component_ids:
            continue
        scc_component_ids: set[str] = set()
        visit_reverse(component_id, scc_component_ids)
        merged_node_ids: set[str] = set()
        for scc_component_id in scc_component_ids:
            merged_node_ids.update(component_groups[scc_component_id])
        representative_node_id = min(merged_node_ids)
        merged_groups[representative_node_id] = merged_node_ids
        if len(scc_component_ids) > 1:
            merged_sccs += 1

    if merged_sccs:
        logger.warning(
            "Transport-aware coarse component graph contained %d cyclic SCC(s); merging them back into larger remote components.",
            merged_sccs,
        )
    return merged_groups


def _build_remote_components(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> list[list[str]]:
    """Partition remote-marked nodes into transport-aware DAG components."""
    component_groups = _remote_component_partition_groups(
        prompt,
        remote_node_ids,
        consumers,
        nodes_module,
    )
    components = _component_topological_order(prompt, component_groups)
    logger.info(
        "Partitioned %d remote nodes into %d transport-aware remote components: %s",
        len(remote_node_ids),
        len(components),
        components,
    )
    return components


def _expand_remote_node_ids_for_non_transportable_inputs(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> tuple[set[str], list[RemoteExpansionReason]]:
    """Grow the remote set upstream until non-transportable inputs stay inside the remote island."""
    expanded_remote_node_ids = set(remote_node_ids)
    added_node_ids: set[str] = set()
    reasons: list[RemoteExpansionReason] = []

    changed = True
    while changed:
        changed = False
        for node_id in sorted(expanded_remote_node_ids):
            prompt_node = prompt.get(node_id)
            if prompt_node is None:
                continue
            for input_value in (prompt_node.get("inputs") or {}).values():
                if not _is_link(input_value):
                    continue
                upstream_node_id = str(input_value[0])
                if upstream_node_id in expanded_remote_node_ids:
                    continue

                upstream_prompt_node = prompt.get(upstream_node_id)
                if upstream_prompt_node is None:
                    continue

                upstream_class_type = str(upstream_prompt_node["class_type"])
                upstream_class = nodes_module.NODE_CLASS_MAPPINGS.get(
                    upstream_class_type
                )
                if upstream_class is None:
                    continue

                output_types, _, _ = _normalize_output_metadata(upstream_class)
                output_index = int(input_value[1])
                if output_index >= len(output_types):
                    continue

                io_type = str(output_types[output_index])
                if _is_transportable_output_type(io_type):
                    continue

                expanded_remote_node_ids.add(upstream_node_id)
                reason = RemoteExpansionReason(
                    node_id=upstream_node_id,
                    class_type=upstream_class_type,
                    required_by_node_id=node_id,
                    required_by_class_type=str(prompt_node["class_type"]),
                    output_index=output_index,
                    io_type=io_type,
                )
                reasons.append(reason)
                added_node_ids.add(upstream_node_id)
                changed = True
                logger.info(
                    "Auto-expanded remote execution upstream: added node %s (%s) because node %s depends on non-transportable type '%s'.",
                    reason.node_id,
                    reason.class_type,
                    reason.required_by_node_id,
                    reason.io_type,
                )
                break
            if changed:
                break

    if added_node_ids:
        logger.info(
            "Expanded remote node set from %d to %d nodes by absorbing upstream non-transportable dependencies: %s",
            len(remote_node_ids),
            len(expanded_remote_node_ids),
            sorted(added_node_ids),
        )
    return expanded_remote_node_ids, reasons


def _terminal_remote_video_source(
    *,
    node_id: str,
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> LinkedOutputRef | None:
    """Return the remote VIDEO feeding one safe terminal SaveVideo artifact sink."""
    prompt_node = prompt.get(node_id)
    if prompt_node is None or str(prompt_node.get("class_type")) != "SaveVideo":
        return None
    inputs = prompt_node.get("inputs") or {}
    video_input = inputs.get("video")
    if not _is_link(video_input):
        return None
    video_source = LinkedOutputRef(str(video_input[0]), int(video_input[1]))
    linked_source_node_ids = {
        str(input_value[0]) for input_value in inputs.values() if _is_link(input_value)
    }
    if not linked_source_node_ids.issubset(remote_node_ids):
        return None
    if any(
        consumers.get(output_ref)
        for output_ref in _node_output_refs(prompt, node_id, nodes_module)
    ):
        return None
    io_type = _remote_output_io_type(
        prompt=prompt,
        node_id=video_source.node_id,
        output_index=video_source.output_index,
        nodes_module=nodes_module,
    )
    return video_source if io_type == "VIDEO" else None


def _expand_remote_node_ids_for_terminal_video_sinks(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> set[str]:
    """Run terminal SaveVideo consumers remotely to avoid exporting raw VIDEO tensors."""
    expanded_remote_node_ids = set(remote_node_ids)
    consumers = _build_consumer_map(prompt)
    added_node_ids: set[str] = set()
    for node_id in sorted(prompt):
        if node_id in expanded_remote_node_ids:
            continue
        video_source = _terminal_remote_video_source(
            node_id=node_id,
            prompt=prompt,
            remote_node_ids=expanded_remote_node_ids,
            consumers=consumers,
            nodes_module=nodes_module,
        )
        if video_source is None:
            continue
        expanded_remote_node_ids.add(node_id)
        added_node_ids.add(node_id)
        logger.info(
            "Auto-expanded terminal SaveVideo node %s into remote execution because its "
            "VIDEO input comes from remote node %s; the encoded output artifact will be "
            "materialized locally without exporting raw frames.",
            node_id,
            video_source.node_id,
        )

    if added_node_ids:
        logger.info(
            "Expanded remote node set with terminal video artifact sinks: %s.",
            sorted(added_node_ids),
        )
    return expanded_remote_node_ids


def analyze_remote_node_selection(
    prompt: dict[str, Any],
    workflow: dict[str, Any] | None,
    seed_workflow_node_paths: list[str],
    settings: ModalSyncSettings | None = None,
    nodes_module: Any | None = None,
) -> RemoteNodeAnalysis:
    """Return the nodes the UI should mark remote for one context-menu expansion request."""
    resolved_settings = settings or get_settings()
    resolved_nodes_module = nodes_module or _get_nodes_module()
    prompt_node_ids = {str(node_id) for node_id in prompt.keys()}
    requested_workflow_node_paths = {
        str(seed_workflow_node_path)
        for seed_workflow_node_path in seed_workflow_node_paths
        if str(seed_workflow_node_path)
    }
    current_remote_node_ids = extract_remote_node_ids(
        workflow,
        resolved_settings,
        prompt_node_ids,
    )
    current_remote_workflow_node_paths = _extract_marked_workflow_node_paths(
        workflow,
        resolved_settings,
    )
    (
        workflow_path_to_prompt_node_ids,
        prompt_node_id_to_workflow_paths,
    ) = _build_workflow_prompt_resolution_maps(workflow, prompt_node_ids)
    requested_node_ids = _resolve_requested_prompt_node_ids(
        requested_workflow_node_paths,
        prompt_node_ids,
        workflow_path_to_prompt_node_ids,
    )
    initial_remote_node_ids = current_remote_node_ids | requested_node_ids
    (
        resolved_remote_node_ids,
        reasons,
    ) = _expand_remote_node_ids_for_non_transportable_inputs(
        prompt=prompt,
        remote_node_ids=initial_remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    resolved_remote_node_ids = _expand_remote_node_ids_for_terminal_video_sinks(
        prompt=prompt,
        remote_node_ids=resolved_remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    sandwiched_local_node_ids = _sandwiched_local_node_ids(
        prompt,
        resolved_remote_node_ids,
    )
    resolved_workflow_node_paths = (
        _resolve_workflow_node_paths_for_prompt_nodes(
            resolved_remote_node_ids,
            prompt_node_id_to_workflow_paths,
        )
        | current_remote_workflow_node_paths
    )
    added_node_ids = resolved_remote_node_ids - current_remote_node_ids
    added_workflow_node_paths = (
        resolved_workflow_node_paths - current_remote_workflow_node_paths
    )

    return RemoteNodeAnalysis(
        requested_node_ids=sorted(requested_node_ids),
        requested_workflow_node_paths=sorted(requested_workflow_node_paths),
        current_remote_node_ids=sorted(current_remote_node_ids),
        current_remote_workflow_node_paths=sorted(current_remote_workflow_node_paths),
        resolved_remote_node_ids=sorted(resolved_remote_node_ids),
        resolved_workflow_node_paths=sorted(resolved_workflow_node_paths),
        added_node_ids=sorted(added_node_ids),
        added_workflow_node_paths=sorted(added_workflow_node_paths),
        sandwiched_local_node_ids=sorted(sandwiched_local_node_ids),
        reasons=reasons,
    )

