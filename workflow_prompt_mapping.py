"""Saved-workflow traversal and queued-prompt node identifier mapping."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Iterator, Mapping

if __package__:
    from .settings import ModalSyncSettings, get_settings
else:  # pragma: no cover - flat import inside the Modal container.
    from settings import ModalSyncSettings, get_settings

logger = logging.getLogger(__name__)


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


