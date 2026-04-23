"""Prompt interception and graph rewriting for Modal-backed execution."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from aiohttp import web

from .modal_executor_node import (
    MODAL_MAP_INPUT_NODE_ID,
    ensure_modal_component_proxy_node_registered,
    register_cache_friendly_proxy_payload,
    register_modal_map_input_warmup_context,
)
from .session_state import RemoteSessionHandle
from .settings import ModalSyncSettings, get_settings
from .sync_engine import ModalAssetSyncEngine, SyncedAsset

logger = logging.getLogger(__name__)

_ROUTE_REGISTERED = False
_TRANSPORTABLE_OUTPUT_TYPES = frozenset(
    {
        "*",
        "BOOLEAN",
        "FLOAT",
        "IMAGE",
        "INT",
        "LATENT",
        "MASK",
        "NOISE",
        "SIGMAS",
        "STRING",
    }
)
_ROOT_LOADER_PREWARM_CLASS_TYPES = frozenset(
    {
        "CheckpointLoaderSimple",
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
    }
)


@dataclass(frozen=True)
class LinkedOutputRef:
    """Reference a node output slot within a prompt graph."""

    node_id: str
    output_index: int


@dataclass(frozen=True)
class InputTarget:
    """Describe a target input inside a remote component."""

    node_id: str
    input_name: str


@dataclass
class BoundaryInputSpec:
    """Describe one local-to-remote boundary value for a component."""

    proxy_input_name: str
    source: LinkedOutputRef
    io_type: str
    targets: list[InputTarget] = field(default_factory=list)


@dataclass
class BoundaryOutputSpec:
    """Describe one remote-to-local boundary value for a component."""

    proxy_output_name: str
    source: LinkedOutputRef
    io_type: str
    is_list: bool
    preview_target_node_ids: list[str] = field(default_factory=list)


@dataclass
class StaticToMappedBoundarySpec:
    """Describe one static-phase output injected into each mapped item run."""

    proxy_name: str
    source: LinkedOutputRef
    io_type: str
    is_list: bool
    targets: list[InputTarget] = field(default_factory=list)


@dataclass(frozen=True)
class ProducedPhaseOutputSpec:
    """Describe one output published by an earlier split-proxy phase."""

    proxy_output_name: str
    source: LinkedOutputRef
    io_type: str
    is_list: bool
    session_output: bool = False


@dataclass
class RemoteComponentPlan:
    """Execution and rewrite plan for one connected remote component."""

    node_ids: list[str]
    representative_node_id: str
    boundary_inputs: list[BoundaryInputSpec]
    boundary_outputs: list[BoundaryOutputSpec]
    execute_node_ids: list[str]
    contains_output_node: bool
    mapped_boundary_input_name: str | None = None
    mapped_boundary_input_io_type: str | None = None
    mapped_boundary_source_node_id: str | None = None
    static_node_ids: list[str] = field(default_factory=list)
    mapped_node_ids: list[str] = field(default_factory=list)
    mapped_execute_node_ids: list[str] = field(default_factory=list)
    static_execute_node_ids: list[str] = field(default_factory=list)
    static_to_mapped_boundaries: list[StaticToMappedBoundarySpec] = field(default_factory=list)


@dataclass
class RewriteSummary:
    """Summary of the prompt rewrite performed for a queue request."""

    remote_node_ids: list[str] = field(default_factory=list)
    remote_component_ids: list[str] = field(default_factory=list)
    component_node_ids_by_representative: dict[str, list[str]] = field(default_factory=dict)
    component_dependency_ids_by_representative: dict[str, list[str]] = field(default_factory=dict)
    component_execution_stages: list[list[str]] = field(default_factory=list)
    mapped_component_ids: list[str] = field(default_factory=list)
    estimated_max_parallel_requests: int = 0
    max_parallel_requests_upper_bound: int | None = None
    requires_volume_reload: bool = False
    volume_reload_marker: str | None = None
    uploaded_volume_paths: list[str] = field(default_factory=list)
    rewritten_node_id_map: dict[str, str] = field(default_factory=dict)
    synced_assets: list[SyncedAsset] = field(default_factory=list)
    custom_nodes_bundle: SyncedAsset | None = None


@dataclass(frozen=True)
class RemoteExpansionReason:
    """Describe why one upstream node had to join a remote component."""

    node_id: str
    class_type: str
    required_by_node_id: str
    required_by_class_type: str
    output_index: int
    io_type: str


@dataclass
class RemoteNodeAnalysis:
    """Structured dry-run result for context-menu remote expansion."""

    requested_node_ids: list[str] = field(default_factory=list)
    requested_workflow_node_paths: list[str] = field(default_factory=list)
    current_remote_node_ids: list[str] = field(default_factory=list)
    current_remote_workflow_node_paths: list[str] = field(default_factory=list)
    resolved_remote_node_ids: list[str] = field(default_factory=list)
    resolved_workflow_node_paths: list[str] = field(default_factory=list)
    added_node_ids: list[str] = field(default_factory=list)
    added_workflow_node_paths: list[str] = field(default_factory=list)
    reasons: list[RemoteExpansionReason] = field(default_factory=list)


class ModalPromptValidationError(ValueError):
    """Raised when a prompt cannot be executed with the current Modal transport."""


def _emit_modal_status(
    prompt_server: Any,
    phase: str,
    *,
    client_id: str | None,
    prompt_id: str | None,
    node_ids: list[str],
    component_node_ids_by_representative: dict[str, list[str]] | None = None,
    active_node_id: str | None = None,
    active_node_class_type: str | None = None,
    active_node_role: str | None = None,
    error_message: str | None = None,
    status_message: str | None = None,
    status_current: int | None = None,
    status_total: int | None = None,
) -> None:
    """Send a Modal execution status event to the active websocket client."""
    if client_id is None:
        logger.debug("Skipping Modal status event %s because no client id is available.", phase)
        return

    payload: dict[str, Any] = {
        "phase": phase,
        "prompt_id": prompt_id,
        "node_ids": list(node_ids),
    }
    if component_node_ids_by_representative:
        payload["components"] = [
            {
                "representative_node_id": representative_node_id,
                "node_ids": list(component_node_ids),
            }
            for representative_node_id, component_node_ids in sorted(
                component_node_ids_by_representative.items()
            )
        ]
    if active_node_id is not None:
        payload["active_node_id"] = active_node_id
    if active_node_class_type is not None:
        payload["active_node_class_type"] = active_node_class_type
    if active_node_role is not None:
        payload["active_node_role"] = active_node_role
    if error_message is not None:
        payload["error_message"] = error_message
    if status_message is not None:
        payload["status_message"] = status_message
    if status_current is not None:
        payload["status_current"] = int(status_current)
    if status_total is not None:
        payload["status_total"] = int(status_total)

    prompt_server.send_sync("modal_status", payload, client_id)


def _get_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _get_server_module() -> Any:
    """Import the ComfyUI server module lazily."""
    import server

    return server


def _get_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    import execution

    return execution


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _prompt_value_signature_fragment(
    prompt: dict[str, Any],
    value: Any,
    *,
    memo: dict[str, str],
) -> Any:
    """Return a stable structural signature fragment for one prompt input value."""
    if _is_link(value):
        source_node_id = str(value[0])
        return {
            "kind": "link",
            "source_node_id": source_node_id,
            "output_index": int(value[1]),
            "source_digest": _prompt_node_signature_digest(
                prompt,
                source_node_id,
                memo=memo,
            ),
        }
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if value != value:
            return {"kind": "float", "value": "nan"}
        if value == float("inf"):
            return {"kind": "float", "value": "inf"}
        if value == float("-inf"):
            return {"kind": "float", "value": "-inf"}
        return value
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _prompt_value_signature_fragment(prompt, item, memo=memo)
                for item in value
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _prompt_value_signature_fragment(prompt, item, memo=memo)
                for item in value
            ],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                {
                    "key": str(key),
                    "value": _prompt_value_signature_fragment(prompt, value[key], memo=memo),
                }
                for key in sorted(value)
            ],
        }
    return {
        "kind": "repr",
        "type": type(value).__name__,
        "value": repr(value),
    }


def _prompt_node_signature_digest(
    prompt: dict[str, Any],
    node_id: str,
    *,
    memo: dict[str, str],
) -> str:
    """Return a stable digest for one prompt node and its upstream prompt inputs."""
    if node_id in memo:
        return memo[node_id]

    prompt_node = prompt.get(str(node_id))
    if prompt_node is None:
        digest = hashlib.sha256(
            json.dumps(
                {"kind": "missing-node", "node_id": str(node_id)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        memo[str(node_id)] = digest
        return digest

    payload = {
        "kind": "prompt-node",
        "node_id": str(node_id),
        "class_type": str(prompt_node.get("class_type", "")),
        "inputs": [
            {
                "name": str(input_name),
                "value": _prompt_value_signature_fragment(
                    prompt,
                    input_value,
                    memo=memo,
                ),
            }
            for input_name, input_value in sorted((prompt_node.get("inputs") or {}).items())
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    memo[str(node_id)] = digest
    return digest


def _iter_loader_snapshot_prompt_payloads(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield prompt-bearing payload fragments that may contain root loader nodes."""
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        for phase_payload in split_proxy_payloads.values():
            if isinstance(phase_payload, Mapping):
                yield phase_payload
        return
    if isinstance(split_proxy_payloads, list):
        for phase_payload in split_proxy_payloads:
            if isinstance(phase_payload, Mapping):
                yield phase_payload
        return
    if isinstance(payload.get("subgraph_prompt"), Mapping):
        yield payload


def _is_root_literal_loader_node(prompt_node: Mapping[str, Any]) -> bool:
    """Return whether one prompt node is a supported root loader with literal inputs."""
    class_type = str(prompt_node.get("class_type") or "")
    if class_type not in _ROOT_LOADER_PREWARM_CLASS_TYPES:
        return False
    inputs = prompt_node.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    return not any(_is_link(input_value) for input_value in inputs.values())


def _loader_prewarm_plan_signature(class_type: str, inputs: Mapping[str, Any]) -> str:
    """Return a stable signature for one synthetic loader-prewarm plan."""
    return json.dumps(
        {
            "class_type": class_type,
            "inputs": copy.deepcopy(dict(inputs)),
        },
        sort_keys=True,
        default=str,
    )


def _payload_loader_snapshot_profile_key(payload: Mapping[str, Any]) -> str:
    """Return the stable loader snapshot profile key derivable from one payload."""
    prompt_id = payload.get("prompt_id")
    normalized_prompt_id = str(prompt_id) if prompt_id is not None else None
    plan_signatures: set[str] = set()
    for prompt_payload in _iter_loader_snapshot_prompt_payloads(payload):
        subgraph_prompt = prompt_payload.get("subgraph_prompt")
        if not isinstance(subgraph_prompt, Mapping):
            continue
        for node_id, prompt_node in subgraph_prompt.items():
            if not isinstance(prompt_node, Mapping) or not _is_root_literal_loader_node(prompt_node):
                continue
            class_type = str(prompt_node.get("class_type") or "")
            inputs = prompt_node.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            plan_signatures.add(_loader_prewarm_plan_signature(class_type, inputs))
            logger.debug(
                "Derived rewrite-time loader prewarm plan for component=%s node=%s class_type=%s prompt_id=%s.",
                payload.get("component_id"),
                node_id,
                class_type,
                normalized_prompt_id,
            )
    if not plan_signatures:
        return ""
    profile_digest = hashlib.sha256(
        json.dumps({"plan_signatures": sorted(plan_signatures)}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"loader-profile:{profile_digest}"


def _stamp_snapshot_profile_key(payload: dict[str, Any], snapshot_profile_key: str) -> None:
    """Attach one loader snapshot profile key to a payload and any split descendants."""
    if not snapshot_profile_key:
        return
    payload["snapshot_profile_key"] = snapshot_profile_key
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        for phase_payload in split_proxy_payloads.values():
            if isinstance(phase_payload, dict):
                phase_payload["snapshot_profile_key"] = snapshot_profile_key
        return
    if isinstance(split_proxy_payloads, list):
        for phase_payload in split_proxy_payloads:
            if isinstance(phase_payload, dict):
                phase_payload["snapshot_profile_key"] = snapshot_profile_key


def _attach_snapshot_profile_key(payload: dict[str, Any], settings: ModalSyncSettings) -> dict[str, Any]:
    """Stamp a deterministic loader snapshot profile onto one payload when enabled."""
    if not settings.enable_gpu_memory_snapshot or not settings.enable_loader_prewarm:
        return payload
    snapshot_profile_key = _payload_loader_snapshot_profile_key(payload)
    if snapshot_profile_key:
        _stamp_snapshot_profile_key(payload, snapshot_profile_key)
        logger.info(
            "Attached rewrite-time loader snapshot profile %s to component=%s payload_kind=%s.",
            snapshot_profile_key,
            payload.get("component_id"),
            payload.get("payload_kind"),
        )
    return payload


def _boundary_source_signature(
    prompt: dict[str, Any],
    source: LinkedOutputRef,
) -> str:
    """Return a stable prompt-structural fingerprint for one boundary source output."""
    payload = {
        "kind": "boundary-source",
        "source_node_id": source.node_id,
        "output_index": int(source.output_index),
        "source_digest": _prompt_node_signature_digest(
            prompt,
            source.node_id,
            memo={},
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"SRC_{digest}"


def _serialize_boundary_input_specs(
    boundary_inputs: list[BoundaryInputSpec],
    *,
    signature_prompt: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return serialized boundary input payloads with stable provenance for non-transportable inputs."""
    serialized_boundary_inputs: list[dict[str, Any]] = []
    for boundary_input in boundary_inputs:
        serialized_boundary_input = {
            "proxy_input_name": boundary_input.proxy_input_name,
            "io_type": boundary_input.io_type,
            "targets": [
                {"node_id": target.node_id, "input_name": target.input_name}
                for target in boundary_input.targets
            ],
        }
        if not _is_transportable_output_type(boundary_input.io_type):
            serialized_boundary_input["source_signature"] = _boundary_source_signature(
                signature_prompt,
                boundary_input.source,
            )
        serialized_boundary_inputs.append(serialized_boundary_input)
    return serialized_boundary_inputs


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


def _iter_workflow_nodes_with_ancestors(
    workflow_fragment: Any,
    ancestor_node_ids: tuple[str, ...] = (),
    visited_object_ids: set[int] | None = None,
) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
    """Yield workflow nodes together with their ancestor workflow-node ids."""
    if visited_object_ids is None:
        visited_object_ids = set()

    if isinstance(workflow_fragment, dict):
        object_id = id(workflow_fragment)
        if object_id in visited_object_ids:
            return
        visited_object_ids.add(object_id)

        next_ancestor_node_ids = ancestor_node_ids
        if _looks_like_workflow_node(workflow_fragment):
            node_id = str(workflow_fragment.get("id"))
            yield workflow_fragment, ancestor_node_ids
            next_ancestor_node_ids = ancestor_node_ids + (node_id,)

        for value in workflow_fragment.values():
            yield from _iter_workflow_nodes_with_ancestors(
                value,
                next_ancestor_node_ids,
                visited_object_ids,
            )
        return

    if isinstance(workflow_fragment, list):
        object_id = id(workflow_fragment)
        if object_id in visited_object_ids:
            return
        visited_object_ids.add(object_id)

        for item in workflow_fragment:
            yield from _iter_workflow_nodes_with_ancestors(
                item,
                ancestor_node_ids,
                visited_object_ids,
            )


def _resolve_prompt_node_ids_for_workflow_node(
    workflow_node_id: str,
    ancestor_node_ids: tuple[str, ...],
    prompt_node_ids: set[str],
) -> set[str]:
    """Resolve one saved workflow node id to matching queued prompt node ids."""
    path_segments = ancestor_node_ids + (workflow_node_id,)
    resolved_prompt_node_ids: set[str] = set()

    for index in range(len(path_segments)):
        candidate = ":".join(path_segments[index:])
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


def _workflow_node_path(workflow_node_id: str, ancestor_node_ids: tuple[str, ...]) -> str:
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
        if workflow_node_id in prompt_node_ids:
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
    workflow_node_paths = prompt_node_id_to_workflow_paths.get(prompt_node_id, {prompt_node_id})
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
        _best_workflow_path_for_prompt_node(prompt_node_id, prompt_node_id_to_workflow_paths)
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
        properties = node.get("properties") or {}
        if properties.get(marker):
            node_id = str(node.get("id"))
            if prompt_node_ids is None or node_id in prompt_node_ids:
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


def _normalize_output_metadata(node_class: type[Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    """Return normalized output metadata for a node class."""
    if hasattr(node_class, "GET_SCHEMA"):
        node_class.GET_SCHEMA()

    output_types = tuple(getattr(node_class, "RETURN_TYPES", ("*",))) or ("*",)
    default_names = tuple(f"output_{index}" for index, _ in enumerate(output_types))
    output_names = tuple(getattr(node_class, "RETURN_NAMES", default_names))
    output_is_list = tuple(getattr(node_class, "OUTPUT_IS_LIST", (False,) * len(output_types)))

    if len(output_names) < len(output_types):
        output_names = output_names + default_names[len(output_names) :]
    if len(output_is_list) < len(output_types):
        output_is_list = output_is_list + (False,) * (len(output_types) - len(output_is_list))

    return output_types, output_names[: len(output_types)], output_is_list[: len(output_types)]


def _is_transportable_output_type(io_type: str) -> bool:
    """Return whether a ComfyUI output type can cross the current transport."""
    normalized_parts = [part.strip() for part in io_type.split(",") if part.strip()]
    return bool(normalized_parts) and all(part in _TRANSPORTABLE_OUTPUT_TYPES for part in normalized_parts)


def _build_consumer_map(prompt: dict[str, Any]) -> dict[LinkedOutputRef, list[InputTarget]]:
    """Build a reverse map from node outputs to downstream prompt inputs."""
    consumers: dict[LinkedOutputRef, list[InputTarget]] = defaultdict(list)
    for node_id, prompt_node in prompt.items():
        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(node_id=str(input_value[0]), output_index=int(input_value[1]))
            consumers[source].append(InputTarget(node_id=str(node_id), input_name=str(input_name)))
    return consumers


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


def _remote_component_partition_groups(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> dict[str, set[str]]:
    """Return component groups after merging remote nodes across non-transportable edges."""
    parent: dict[str, str] = {node_id: node_id for node_id in remote_node_ids}
    downstream_remote_node_ids_by_node_id: dict[str, set[str]] = defaultdict(set)

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

    def union(left_node_id: str, right_node_id: str) -> None:
        """Merge two remote nodes into the same mandatory component."""
        left_root = find(left_node_id)
        right_root = find(right_node_id)
        if left_root == right_root:
            return
        canonical_root = min(left_root, right_root)
        merged_root = max(left_root, right_root)
        parent[merged_root] = canonical_root

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
            if (
                upstream_prompt_node is not None
                and str(upstream_prompt_node.get("class_type")) == MODAL_MAP_INPUT_NODE_ID
            ):
                union(node_id, upstream_node_id)
                continue
            io_type = _remote_output_io_type(
                prompt=prompt,
                node_id=upstream_node_id,
                output_index=int(input_value[1]),
                nodes_module=nodes_module,
            )
            if io_type is None or _is_transportable_output_type(io_type):
                continue
            union(node_id, upstream_node_id)

    for remote_node_id in sorted(remote_node_ids):
        prompt_node = prompt.get(remote_node_id)
        if prompt_node is None or str(prompt_node.get("class_type")) != MODAL_MAP_INPUT_NODE_ID:
            continue
        pending_node_ids = [remote_node_id]
        visited_node_ids: set[str] = set()
        while pending_node_ids:
            current_node_id = pending_node_ids.pop()
            if current_node_id in visited_node_ids:
                continue
            visited_node_ids.add(current_node_id)
            union(remote_node_id, current_node_id)
            for downstream_node_id in sorted(downstream_remote_node_ids_by_node_id.get(current_node_id, set())):
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
    _, dependency_edges, indegree_by_component_id = _component_dependency_graph(prompt, component_groups)
    merged_component_groups = _merge_cyclic_component_groups(
        component_groups=component_groups,
        dependency_edges=dependency_edges,
    )
    if merged_component_groups != component_groups:
        return _component_topological_order(prompt, merged_component_groups)

    ready_component_ids = deque(sorted(
        [
            component_id
            for component_id, indegree in indegree_by_component_id.items()
            if indegree == 0
        ]
    ))
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
        representative_node_id: set()
        for representative_node_id in component_groups
    }
    indegree_by_component_id: dict[str, int] = {
        representative_node_id: 0
        for representative_node_id in component_groups
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
            if upstream_component_id is None or upstream_component_id == representative_node_id:
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
    _, dependency_edges, indegree_by_component_id = _component_dependency_graph(prompt, component_groups)
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
            for downstream_component_id in sorted(dependency_edges.get(component_id, set())):
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
    reverse_edges: dict[str, set[str]] = {component_id: set() for component_id in component_ids}
    for upstream_component_id, downstream_component_ids in dependency_edges.items():
        for downstream_component_id in downstream_component_ids:
            reverse_edges.setdefault(downstream_component_id, set()).add(upstream_component_id)

    visited_component_ids: set[str] = set()
    finish_order: list[str] = []

    def visit_forward(component_id: str) -> None:
        """Record reverse-topological finish order over the coarse graph."""
        if component_id in visited_component_ids:
            return
        visited_component_ids.add(component_id)
        for downstream_component_id in sorted(dependency_edges.get(component_id, set())):
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
    nodes_module: Any,
) -> list[list[str]]:
    """Partition remote-marked nodes into transport-aware DAG components."""
    component_groups = _remote_component_partition_groups(prompt, remote_node_ids, nodes_module)
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
                upstream_class = nodes_module.NODE_CLASS_MAPPINGS.get(upstream_class_type)
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
    workflow_path_to_prompt_node_ids, prompt_node_id_to_workflow_paths = (
        _build_workflow_prompt_resolution_maps(workflow, prompt_node_ids)
    )
    requested_node_ids = _resolve_requested_prompt_node_ids(
        requested_workflow_node_paths,
        prompt_node_ids,
        workflow_path_to_prompt_node_ids,
    )
    initial_remote_node_ids = current_remote_node_ids | requested_node_ids
    resolved_remote_node_ids, reasons = _expand_remote_node_ids_for_non_transportable_inputs(
        prompt=prompt,
        remote_node_ids=initial_remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    resolved_workflow_node_paths = (
        _resolve_workflow_node_paths_for_prompt_nodes(
            resolved_remote_node_ids,
            prompt_node_id_to_workflow_paths,
        )
        | current_remote_workflow_node_paths
    )
    added_node_ids = resolved_remote_node_ids - current_remote_node_ids
    added_workflow_node_paths = resolved_workflow_node_paths - current_remote_workflow_node_paths

    return RemoteNodeAnalysis(
        requested_node_ids=sorted(requested_node_ids),
        requested_workflow_node_paths=sorted(requested_workflow_node_paths),
        current_remote_node_ids=sorted(current_remote_node_ids),
        current_remote_workflow_node_paths=sorted(current_remote_workflow_node_paths),
        resolved_remote_node_ids=sorted(resolved_remote_node_ids),
        resolved_workflow_node_paths=sorted(resolved_workflow_node_paths),
        added_node_ids=sorted(added_node_ids),
        added_workflow_node_paths=sorted(added_workflow_node_paths),
        reasons=reasons,
    )


def _build_component_plan(
    component_node_ids: list[str],
    prompt: dict[str, Any],
    consumers: dict[LinkedOutputRef, list[InputTarget]],
    nodes_module: Any,
) -> RemoteComponentPlan:
    """Build rewrite metadata for a connected remote component."""
    component_node_id_set = set(component_node_ids)
    representative_node_id = component_node_ids[0]
    boundary_inputs_by_source: dict[LinkedOutputRef, BoundaryInputSpec] = {}
    boundary_outputs_by_source: dict[LinkedOutputRef, BoundaryOutputSpec] = {}
    output_execution_targets: set[str] = set()
    contains_output_node = False

    for node_id in component_node_ids:
        prompt_node = prompt[node_id]
        class_type = str(prompt_node["class_type"])
        node_class = nodes_module.NODE_CLASS_MAPPINGS[class_type]
        output_types, output_names, output_is_list = _normalize_output_metadata(node_class)

        if getattr(node_class, "OUTPUT_NODE", False):
            contains_output_node = True
            output_execution_targets.add(node_id)

        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            if not _is_link(input_value):
                continue
            upstream_node_id = str(input_value[0])
            if upstream_node_id in component_node_id_set:
                continue
            source = LinkedOutputRef(node_id=upstream_node_id, output_index=int(input_value[1]))
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
            spec.targets.append(InputTarget(node_id=node_id, input_name=str(input_name)))

        for output_index, io_type in enumerate(output_types):
            source = LinkedOutputRef(node_id=node_id, output_index=output_index)
            local_consumers = [
                consumer for consumer in consumers.get(source, []) if consumer.node_id not in component_node_id_set
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
        static_to_mapped_boundaries_by_source: dict[LinkedOutputRef, StaticToMappedBoundarySpec] = {}
        for mapped_node_id in mapped_node_ids:
            prompt_node = prompt[mapped_node_id]
            for input_name, input_value in (prompt_node.get("inputs") or {}).items():
                if not _is_link(input_value):
                    continue
                upstream_node_id = str(input_value[0])
                if upstream_node_id not in component_node_id_set or upstream_node_id in mapped_node_id_set:
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
            mapped_boundary_spec.proxy_input_name if mapped_boundary_spec is not None else None
        ),
        mapped_boundary_input_io_type=mapped_boundary_input_io_type,
        mapped_boundary_source_node_id=(
            mapped_boundary_spec.source.node_id
            if mapped_boundary_spec is not None
            and prompt.get(mapped_boundary_spec.source.node_id, {}).get("class_type") == MODAL_MAP_INPUT_NODE_ID
            else None
        ),
        static_node_ids=static_node_ids,
        mapped_node_ids=mapped_node_ids,
        mapped_execute_node_ids=mapped_execute_node_ids,
        static_execute_node_ids=static_execute_node_ids,
        static_to_mapped_boundaries=static_to_mapped_boundaries,
    )
    logger.info(
        "Planned remote component %s: nodes=%s boundary_inputs=%d boundary_outputs=%d execute_nodes=%s output_node=%s mapped_input=%s static_nodes=%s mapped_nodes=%s mapped_execute_nodes=%s static_execute_nodes=%s static_to_mapped_boundaries=%s",
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
        [
            {
                "proxy_name": boundary_spec.proxy_name,
                "source": (boundary_spec.source.node_id, boundary_spec.source.output_index),
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
        if current_node_id in reachable_node_ids or current_node_id not in component_node_id_set:
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
        if current_node_id in reachable_node_ids or current_node_id not in candidate_node_ids:
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

    ready_node_ids = deque(sorted(
        node_id
        for node_id, indegree in indegree_by_node_id.items()
        if indegree == 0
    ))
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


def _build_component_plans(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> list[RemoteComponentPlan]:
    """Build plans for every connected remote component."""
    consumers = _build_consumer_map(prompt)
    components = _build_remote_components(prompt, remote_node_ids, nodes_module)
    return [
        _build_component_plan(component, prompt, consumers, nodes_module)
        for component in components
    ]


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
    """Format a human-readable remote-to-local transport validation error."""
    return (
        "Remote component rooted at node "
        f"{component.representative_node_id} exports node {source.node_id} "
        f"({source_class_type}) output index {source.output_index} of type '{io_type}' "
        f"to local node {local_consumer.node_id} ({local_consumer_class_type}) input "
        f"'{local_consumer.input_name}', which cannot cross the current local/remote boundary. "
        "Current ComfyUI-Modal transport only supports JSON-compatible values, bytes, "
        "and tensor-like outputs such as IMAGE, MASK, LATENT, SIGMAS, NOISE, INT, "
        "FLOAT, BOOLEAN, and STRING."
    )


def _describe_input_boundary_error(
    component: RemoteComponentPlan,
    target: InputTarget,
    target_class_type: str,
    source: LinkedOutputRef,
    source_class_type: str,
    io_type: str,
) -> str:
    """Format a human-readable local-to-remote transport validation error."""
    return (
        "Remote node "
        f"{target.node_id} ({target_class_type}) input '{target.input_name}' "
        f"depends on upstream node {source.node_id} ({source_class_type}) output index "
        f"{source.output_index} of type '{io_type}', which cannot cross the current "
        "local/remote boundary. Current ComfyUI-Modal transport only supports "
        "JSON-compatible values, bytes, and tensor-like outputs such as IMAGE, MASK, "
        "LATENT, SIGMAS, NOISE, INT, FLOAT, BOOLEAN, and STRING."
    )


def validate_remote_component_transport_compatibility(
    prompt: dict[str, Any],
    components: list[RemoteComponentPlan],
    nodes_module: Any,
) -> None:
    """Reject remote components whose true graph boundaries require unsupported transport."""
    validation_errors: list[str] = []
    consumers = _build_consumer_map(prompt)
    logger.info("Validating %d remote components for transport compatibility.", len(components))

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

            source_class_type = str(prompt[boundary_output.source.node_id]["class_type"])
            for local_consumer in consumers.get(boundary_output.source, []):
                if local_consumer.node_id in component.node_ids:
                    continue
                local_consumer_class_type = str(prompt[local_consumer.node_id]["class_type"])
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


def _sync_component_prompt_inputs(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    sync_engine: ModalAssetSyncEngine,
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
) -> dict[str, Any]:
    """Build the serialized execution payload for one remote component."""
    prompt_id = (extra_data or {}).get("prompt_id")
    custom_nodes_bundle_path = (
        custom_nodes_bundle.remote_path if custom_nodes_bundle is not None else None
    )

    def build_subgraph_payload(
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
            "prompt_id": prompt_id,
            "component_node_ids": list(component_node_ids),
            "subgraph_prompt": _subset_component_prompt(component_prompt, component_node_ids),
            "boundary_inputs": _serialize_boundary_input_specs(
                boundary_inputs,
                signature_prompt=signature_prompt,
            ),
            "boundary_outputs": copy.deepcopy(boundary_outputs),
            "execute_node_ids": list(execute_node_ids),
            "extra_data": copy.deepcopy(extra_data or {}),
            "requires_volume_reload": requires_volume_reload,
            "volume_reload_marker": volume_reload_marker,
            "uploaded_volume_paths": list(uploaded_volume_paths),
            "terminate_container_on_error": terminate_container_on_error,
            "custom_nodes_bundle": custom_nodes_bundle_path,
        }
        if remote_session is not None:
            payload["remote_session"] = copy.deepcopy(remote_session)
        if clear_remote_session:
            payload["clear_remote_session"] = True
        if mapped_progress_display_node_id is not None:
            payload["mapped_progress_display_node_id"] = mapped_progress_display_node_id
        return payload

    def build_phase_payloads_for_transportable_splits() -> list[dict[str, Any]] | None:
        """Return ordered split-proxy phase payloads for a non-mapped coarse component."""
        if component.mapped_boundary_input_name is not None or len(component.execute_node_ids) <= 1:
            return None

        component_node_id_set = set(component.node_ids)
        topological_node_ids = _subgraph_topological_node_order(component_prompt, component_node_id_set)
        remaining_node_ids = set(component.node_ids)
        remaining_execute_node_ids = [
            node_id
            for node_id in topological_node_ids
            if node_id in set(component.execute_node_ids)
        ]
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
                    prompt=component_prompt,
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
                prompt_node = component_prompt.get(phase_node_id)
                if prompt_node is None:
                    continue
                for input_name, input_value in (prompt_node.get("inputs") or {}).items():
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
                    boundary_input = phase_boundary_inputs_by_name.get(produced_output.proxy_output_name)
                    if boundary_input is None:
                        boundary_input = BoundaryInputSpec(
                            proxy_input_name=produced_output.proxy_output_name,
                            source=source,
                            io_type=produced_output.io_type,
                        )
                        phase_boundary_inputs.append(boundary_input)
                        phase_boundary_inputs_by_name[boundary_input.proxy_input_name] = boundary_input
                    boundary_input.targets.append(
                        InputTarget(node_id=phase_node_id, input_name=str(input_name))
                    )

            phase_boundary_outputs: list[dict[str, Any]] = []
            phase_output_names_by_source: dict[LinkedOutputRef, str] = {}
            for boundary_output in _filter_boundary_outputs_for_node_ids(
                component.boundary_outputs,
                phase_node_id_set,
            ):
                phase_boundary_outputs.append(
                    {
                        "proxy_output_name": boundary_output.proxy_output_name,
                        "node_id": boundary_output.source.node_id,
                        "output_index": boundary_output.source.output_index,
                        "io_type": boundary_output.io_type,
                        "is_list": boundary_output.is_list,
                        "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                    }
                )
                phase_output_names_by_source[boundary_output.source] = boundary_output.proxy_output_name

            pending_node_ids = remaining_node_ids - phase_node_id_set
            for pending_node_id in sorted(pending_node_ids):
                prompt_node = component_prompt.get(pending_node_id)
                if prompt_node is None:
                    continue
                for input_value in (prompt_node.get("inputs") or {}).values():
                    if not _is_link(input_value):
                        continue
                    source = LinkedOutputRef(
                        node_id=str(input_value[0]),
                        output_index=int(input_value[1]),
                    )
                    if source.node_id not in phase_node_id_set or source in produced_outputs_by_source:
                        continue
                    local_boundary_output = local_boundary_outputs_by_source.get(source)
                    io_type = (
                        local_boundary_output.io_type
                        if local_boundary_output is not None
                        else str(
                            _remote_output_io_type(
                                prompt=component_prompt,
                                node_id=source.node_id,
                                output_index=source.output_index,
                                nodes_module=nodes_module,
                            )
                            or "*"
                        )
                    )
                    is_list = (
                        local_boundary_output.is_list
                        if local_boundary_output is not None
                        else _remote_output_is_list(
                            prompt=component_prompt,
                            node_id=source.node_id,
                            output_index=source.output_index,
                            nodes_module=nodes_module,
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
                                "session_output": not _is_transportable_output_type(io_type),
                            }
                        )
                        phase_output_names_by_source[source] = proxy_output_name
                    produced_outputs_by_source[source] = ProducedPhaseOutputSpec(
                        proxy_output_name=proxy_output_name,
                        source=source,
                        io_type=io_type,
                        is_list=is_list,
                        session_output=not _is_transportable_output_type(io_type),
                    )

            phase_execute_node_ids = [
                node_id
                for node_id in component.execute_node_ids
                if node_id in phase_node_id_set
            ]
            phase_payloads.append(
                build_subgraph_payload(
                    component_id=str(target_node_id),
                    component_node_ids=phase_node_ids,
                    boundary_inputs=phase_boundary_inputs,
                    boundary_outputs=phase_boundary_outputs,
                    execute_node_ids=phase_execute_node_ids,
                )
            )
            remaining_node_ids -= phase_node_id_set
            remaining_execute_node_ids = [
                node_id for node_id in remaining_execute_node_ids if node_id not in phase_node_id_set
            ]

        has_session_bridges = any(
            bool(boundary_output.get("session_output"))
            for phase_payload in phase_payloads
            for boundary_output in phase_payload.get("boundary_outputs", [])
        )
        if not phase_payloads or len(phase_payloads) <= 1:
            return None
        if has_session_bridges:
            remote_session = RemoteSessionHandle(
                session_id=uuid.uuid4().hex,
                prompt_id=(str(prompt_id) if prompt_id is not None else None),
                owner_component_id=component.representative_node_id,
            ).to_payload()
            for phase_index, phase_payload in enumerate(phase_payloads):
                phase_payload["remote_session"] = copy.deepcopy(remote_session)
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

    split_phase_payloads = build_phase_payloads_for_transportable_splits()
    if split_phase_payloads is not None:
        return _attach_snapshot_profile_key(
            {"split_proxy_payloads": split_phase_payloads},
            settings,
        )

    payload = {
        "payload_kind": "mapped_subgraph" if component.mapped_boundary_input_name else "subgraph",
        "component_id": component.representative_node_id,
        "prompt_id": prompt_id,
        "component_node_ids": list(component.node_ids),
        "subgraph_prompt": component_prompt,
        "boundary_inputs": _serialize_boundary_input_specs(
            component.boundary_inputs,
            signature_prompt=signature_prompt,
        ),
        "boundary_outputs": [
            (
                {
                    "proxy_output_name": boundary_output.proxy_output_name,
                    "node_id": boundary_output.source.node_id,
                    "output_index": boundary_output.source.output_index,
                    "io_type": boundary_output.io_type,
                    "is_list": boundary_output.is_list,
                    "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                    "mapped_output": bool(boundary_output.source.node_id in set(component.mapped_node_ids)),
                }
                if component.mapped_boundary_input_name
                else {
                    "proxy_output_name": boundary_output.proxy_output_name,
                    "node_id": boundary_output.source.node_id,
                    "output_index": boundary_output.source.output_index,
                    "io_type": boundary_output.io_type,
                    "is_list": boundary_output.is_list,
                    "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                }
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
    logger.info(
        "Built remote payload for component %s: boundary_inputs=%d boundary_outputs=%d execute_nodes=%s",
        component.representative_node_id,
        len(payload["boundary_inputs"]),
        len(payload["boundary_outputs"]),
        payload["execute_node_ids"],
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
            "subgraph_prompt": _subset_component_prompt(component_prompt, component.static_node_ids),
            "boundary_inputs": _serialize_boundary_input_specs(
                static_boundary_inputs,
                signature_prompt=signature_prompt,
            ),
            "boundary_outputs": [
                {
                    "proxy_output_name": boundary_output.proxy_output_name,
                    "node_id": boundary_output.source.node_id,
                    "output_index": boundary_output.source.output_index,
                    "io_type": boundary_output.io_type,
                    "is_list": boundary_output.is_list,
                    "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                }
                for boundary_output in static_boundary_outputs
            ]
            + static_bridge_outputs,
            "execute_node_ids": list(component.static_execute_node_ids),
        }
        payload["mapped_phase"] = {
            "component_node_ids": list(component.mapped_node_ids),
            "subgraph_prompt": _subset_component_prompt(component_prompt, component.mapped_node_ids),
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
                {
                    "proxy_output_name": boundary_output.proxy_output_name,
                    "node_id": boundary_output.source.node_id,
                    "output_index": boundary_output.source.output_index,
                    "io_type": boundary_output.io_type,
                    "is_list": boundary_output.is_list,
                    "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                    "mapped_output": True,
                }
                for boundary_output in mapped_boundary_outputs
            ],
            "execute_node_ids": list(component.mapped_execute_node_ids),
        }
        if not component.static_node_ids:
            return _attach_snapshot_profile_key(payload, settings)
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
                "static": build_subgraph_payload(
                    component_id=component.static_node_ids[0],
                    component_node_ids=list(component.static_node_ids),
                    boundary_inputs=static_boundary_inputs,
                    boundary_outputs=[
                        {
                            "proxy_output_name": boundary_output.proxy_output_name,
                            "node_id": boundary_output.source.node_id,
                            "output_index": boundary_output.source.output_index,
                            "io_type": boundary_output.io_type,
                            "is_list": boundary_output.is_list,
                            "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                        }
                        for boundary_output in static_boundary_outputs
                    ]
                    + static_bridge_outputs,
                    execute_node_ids=list(component.static_execute_node_ids),
                    remote_session=remote_session,
                ),
                "mapped": build_subgraph_payload(
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
                        {
                            "proxy_output_name": boundary_output.proxy_output_name,
                            "node_id": boundary_output.source.node_id,
                            "output_index": boundary_output.source.output_index,
                            "io_type": boundary_output.io_type,
                            "is_list": boundary_output.is_list,
                            "preview_target_node_ids": list(boundary_output.preview_target_node_ids),
                        }
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
            "subgraph_prompt": _subset_component_prompt(component_prompt, component.static_node_ids),
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


def _rewrite_component_into_proxy(
    component: RemoteComponentPlan,
    rewritten_prompt: dict[str, Any],
    payload: dict[str, Any],
    nodes_module: Any,
) -> list[str]:
    """Replace a remote component with a single proxy node in the prompt."""
    def contains_output_node(node_ids: list[str]) -> bool:
        """Return whether one node subset contains an output node."""
        for node_id in node_ids:
            prompt_node = rewritten_prompt.get(node_id)
            if prompt_node is None:
                continue
            node_class = nodes_module.NODE_CLASS_MAPPINGS.get(str(prompt_node["class_type"]))
            if node_class is not None and getattr(node_class, "OUTPUT_NODE", False):
                return True
        return False

    def proxy_inputs_for_boundary_inputs(boundary_inputs: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve one proxy input mapping from the current prompt graph."""
        proxy_inputs: dict[str, Any] = {}
        for boundary_input in boundary_inputs:
            current_input_value: Any = None
            for target in boundary_input.get("targets", []):
                target_prompt_node = rewritten_prompt.get(str(target["node_id"]))
                if target_prompt_node is None:
                    continue
                target_input_value = (target_prompt_node.get("inputs") or {}).get(str(target["input_name"]))
                if _is_link(target_input_value):
                    current_input_value = list(target_input_value)
                    break
            if current_input_value is None:
                raise ModalPromptValidationError(
                    "Unable to resolve proxy boundary input wiring while rewriting split Modal proxies."
                )
            proxy_inputs[str(boundary_input["proxy_input_name"])] = current_input_value
        return proxy_inputs

    def register_proxy_node(
        *,
        prompt_node_id: str,
        payload_mapping: dict[str, Any],
        proxy_inputs: dict[str, Any],
        meta: dict[str, Any],
        is_output_node: bool,
    ) -> None:
        """Insert one dynamic proxy node into the rewritten prompt."""
        boundary_outputs = list(payload_mapping.get("boundary_outputs", []))
        proxy_node_id = ensure_modal_component_proxy_node_registered(
            output_types=tuple(str(output["io_type"]) for output in boundary_outputs),
            output_names=tuple(str(output["proxy_output_name"]) for output in boundary_outputs),
            output_is_list=tuple(bool(output.get("is_list", False)) for output in boundary_outputs),
            nodes_module=nodes_module,
            is_output_node=is_output_node,
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

    split_proxy_payloads = payload.get("split_proxy_payloads")
    component_node_id_set = set(component.node_ids)
    if isinstance(split_proxy_payloads, list):
        phase_payloads = [dict(phase_payload) for phase_payload in split_proxy_payloads]
        phase_proxy_node_ids: list[str] = []
        produced_output_indices_by_name: dict[str, list[Any]] = {}
        replacement_output_indices: dict[LinkedOutputRef, list[Any]] = {}
        component_proxy_node_ids: set[str] = set()
        phase_proxy_inputs_by_node_id: dict[str, dict[str, Any]] = {}
        phase_proxy_meta_by_node_id: dict[str, dict[str, Any]] = {}

        for phase_payload in phase_payloads:
            phase_proxy_node_id = str(phase_payload["component_id"])
            while phase_proxy_node_id in rewritten_prompt and phase_proxy_node_id not in component_node_id_set:
                phase_proxy_node_id = f"{phase_proxy_node_id}_proxy"
            phase_payload["component_id"] = phase_proxy_node_id
            phase_proxy_node_ids.append(phase_proxy_node_id)
            component_proxy_node_ids.add(phase_proxy_node_id)

        for phase_payload in phase_payloads:
            phase_proxy_node_id = str(phase_payload["component_id"])
            phase_proxy_inputs = proxy_inputs_for_boundary_inputs(list(phase_payload["boundary_inputs"]))
            for boundary_input in phase_payload.get("boundary_inputs", []):
                proxy_input_name = str(boundary_input["proxy_input_name"])
                produced_output_index = produced_output_indices_by_name.get(proxy_input_name)
                if produced_output_index is None:
                    continue
                phase_proxy_inputs[proxy_input_name] = list(produced_output_index)

            first_phase_node_id = str(phase_payload["component_node_ids"][0])
            phase_proxy_meta = copy.deepcopy(rewritten_prompt[first_phase_node_id].get("_meta", {}))
            phase_proxy_inputs_by_node_id[phase_proxy_node_id] = phase_proxy_inputs
            phase_proxy_meta_by_node_id[phase_proxy_node_id] = phase_proxy_meta

            for output_index, boundary_output in enumerate(phase_payload.get("boundary_outputs", [])):
                proxy_output_name = str(boundary_output["proxy_output_name"])
                produced_output_indices_by_name[proxy_output_name] = [phase_proxy_node_id, output_index]
                if bool(boundary_output.get("session_output")):
                    continue
                replacement_output_indices[
                    LinkedOutputRef(
                        node_id=str(boundary_output["node_id"]),
                        output_index=int(boundary_output["output_index"]),
                    )
                ] = [phase_proxy_node_id, output_index]

        for node_id in component.node_ids:
            rewritten_prompt.pop(node_id, None)

        for phase_payload in phase_payloads:
            phase_proxy_node_id = str(phase_payload["component_id"])
            register_proxy_node(
                prompt_node_id=phase_proxy_node_id,
                payload_mapping=phase_payload,
                proxy_inputs=phase_proxy_inputs_by_node_id[phase_proxy_node_id],
                meta=phase_proxy_meta_by_node_id[phase_proxy_node_id],
                is_output_node=contains_output_node(list(phase_payload["component_node_ids"])),
            )

        for node_id, prompt_node in list(rewritten_prompt.items()):
            if node_id in component_proxy_node_ids:
                continue
            for input_name, input_value in list((prompt_node.get("inputs") or {}).items()):
                if not _is_link(input_value):
                    continue
                source = LinkedOutputRef(node_id=str(input_value[0]), output_index=int(input_value[1]))
                if source in replacement_output_indices:
                    prompt_node["inputs"][input_name] = list(replacement_output_indices[source])

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
        while mapped_proxy_node_id in rewritten_prompt and mapped_proxy_node_id not in component_node_id_set:
            mapped_proxy_node_id = f"{mapped_proxy_node_id}_proxy"
        mapped_payload["component_id"] = mapped_proxy_node_id

        static_proxy_inputs = proxy_inputs_for_boundary_inputs(list(static_payload["boundary_inputs"]))
        static_boundary_outputs = list(static_payload["boundary_outputs"])
        static_proxy_meta = copy.deepcopy(rewritten_prompt[static_proxy_node_id].get("_meta", {}))
        mapped_proxy_meta = copy.deepcopy(
            rewritten_prompt[component.mapped_node_ids[0]].get("_meta", {})
        )
        bridge_output_indices = {
            str(boundary_output["proxy_output_name"]): output_index
            for output_index, boundary_output in enumerate(static_boundary_outputs)
            if bool(boundary_output.get("session_output"))
        }
        mapped_proxy_inputs = proxy_inputs_for_boundary_inputs(list(mapped_payload["boundary_inputs"]))
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
                for output_index, boundary_output in enumerate(mapped_payload.get("boundary_outputs", []))
            }
        )

        for node_id in component.node_ids:
            rewritten_prompt.pop(node_id, None)
        register_proxy_node(
            prompt_node_id=static_proxy_node_id,
            payload_mapping=static_payload,
            proxy_inputs=static_proxy_inputs,
            meta=static_proxy_meta,
            is_output_node=contains_output_node(component.static_node_ids),
        )
        register_proxy_node(
            prompt_node_id=mapped_proxy_node_id,
            payload_mapping=mapped_payload,
            proxy_inputs=mapped_proxy_inputs,
            meta=mapped_proxy_meta,
            is_output_node=contains_output_node(component.mapped_node_ids),
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
            for input_name, input_value in list((prompt_node.get("inputs") or {}).items()):
                if not _is_link(input_value):
                    continue
                source = LinkedOutputRef(node_id=str(input_value[0]), output_index=int(input_value[1]))
                if source in replacement_output_indices:
                    prompt_node["inputs"][input_name] = list(replacement_output_indices[source])

        logger.info(
            "Rewrote hybrid remote component %s into static proxy %s and mapped proxy %s.",
            component.representative_node_id,
            static_proxy_node_id,
            mapped_proxy_node_id,
        )
        return [static_proxy_node_id, mapped_proxy_node_id]

    proxy_node_id = ensure_modal_component_proxy_node_registered(
        output_types=tuple(spec.io_type for spec in component.boundary_outputs),
        output_names=tuple(spec.proxy_output_name for spec in component.boundary_outputs),
        output_is_list=tuple(spec.is_list for spec in component.boundary_outputs),
        nodes_module=nodes_module,
        is_output_node=component.contains_output_node,
    )
    representative_node_id = component.representative_node_id
    proxy_inputs = proxy_inputs_for_boundary_inputs(list(payload.get("boundary_inputs", [])))
    proxy_inputs["original_node_data"] = register_cache_friendly_proxy_payload(
        representative_node_id,
        payload,
    )
    representative_meta = copy.deepcopy(rewritten_prompt[representative_node_id].get("_meta", {}))
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
    boundary_output_indices = {
        spec.source: index for index, spec in enumerate(component.boundary_outputs)
    }
    for node_id, prompt_node in list(rewritten_prompt.items()):
        if node_id in component_node_id_set and node_id != representative_node_id:
            del rewritten_prompt[node_id]
            continue
        if node_id == representative_node_id:
            continue
        for input_name, input_value in list((prompt_node.get("inputs") or {}).items()):
            if not _is_link(input_value):
                continue
            source = LinkedOutputRef(node_id=str(input_value[0]), output_index=int(input_value[1]))
            if source in boundary_output_indices:
                prompt_node["inputs"][input_name] = [representative_node_id, boundary_output_indices[source]]
    logger.info(
        "Rewrote remote component %s with %d nodes to Modal proxy %s.",
        representative_node_id,
        len(component.node_ids),
        proxy_node_id,
    )
    return [representative_node_id]


def rewrite_prompt_for_modal(
    prompt: dict[str, Any],
    workflow: dict[str, Any] | None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
    nodes_module: Any | None = None,
    extra_data: dict[str, Any] | None = None,
    status_callback: Any | None = None,
) -> tuple[dict[str, Any], RewriteSummary]:
    """Rewrite connected remote components into Modal proxy nodes."""
    resolved_settings = settings or get_settings()
    remote_node_ids = extract_remote_node_ids(workflow, resolved_settings, set(prompt.keys()))
    summary = RewriteSummary(remote_node_ids=sorted(remote_node_ids))
    logger.info("Found %d workflow nodes marked for Modal execution.", len(remote_node_ids))

    if not remote_node_ids:
        return copy.deepcopy(prompt), summary

    resolved_nodes_module = nodes_module or _get_nodes_module()
    resolved_sync_engine = sync_engine or ModalAssetSyncEngine.from_environment(resolved_settings)
    rewritten_prompt = copy.deepcopy(prompt)
    expanded_remote_node_ids, _ = _expand_remote_node_ids_for_non_transportable_inputs(
        prompt=rewritten_prompt,
        remote_node_ids=remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    summary.remote_node_ids = sorted(expanded_remote_node_ids)
    components = _build_component_plans(
        rewritten_prompt,
        expanded_remote_node_ids,
        resolved_nodes_module,
    )
    validate_remote_component_transport_compatibility(
        prompt=rewritten_prompt,
        components=components,
        nodes_module=resolved_nodes_module,
    )

    if status_callback is not None:
        status_callback("Preparing remote assets for Modal", None, None)

    if resolved_settings.sync_custom_nodes:
        summary.custom_nodes_bundle = resolved_sync_engine.sync_custom_nodes_directory(
            status_callback=status_callback,
        )
    else:
        logger.info(
            "Skipping custom_nodes bundle sync because sync is disabled for execution_mode=%s.",
            resolved_settings.execution_mode,
        )

    synced_component_prompts: dict[str, dict[str, Any]] = {}
    synced_assets_by_component_id: dict[str, list[SyncedAsset]] = {}
    for component in components:
        component_prompt, synced_assets = _sync_component_prompt_inputs(
            component=component,
            rewritten_prompt=rewritten_prompt,
            sync_engine=resolved_sync_engine,
            status_callback=status_callback,
        )
        synced_component_prompts[component.representative_node_id] = component_prompt
        synced_assets_by_component_id[component.representative_node_id] = list(synced_assets)
        summary.synced_assets.extend(synced_assets)

    requires_volume_reload = any(asset.uploaded for asset in summary.synced_assets) or (
        summary.custom_nodes_bundle is not None and summary.custom_nodes_bundle.uploaded
    )
    volume_reload_marker = uuid.uuid4().hex if requires_volume_reload else None
    logger.info(
        "Resolved request-wide Modal volume reload requirement: requires_volume_reload=%s volume_reload_marker=%s synced_assets=%d custom_nodes_uploaded=%s",
        requires_volume_reload,
        volume_reload_marker,
        len(summary.synced_assets),
        bool(summary.custom_nodes_bundle is not None and summary.custom_nodes_bundle.uploaded),
    )
    summary.requires_volume_reload = requires_volume_reload
    summary.volume_reload_marker = volume_reload_marker
    summary.uploaded_volume_paths = [
        asset.remote_path for asset in summary.synced_assets if asset.uploaded
    ]
    mapped_proxy_component_ids: set[str] = set()
    for component in components:
        logger.info(
            "Rewriting remote component %s covering nodes %s.",
            component.representative_node_id,
            component.node_ids,
        )
        uploaded_volume_paths = _component_uploaded_volume_paths(
            component_prompt=synced_component_prompts[component.representative_node_id],
            synced_assets=synced_assets_by_component_id[component.representative_node_id],
            custom_nodes_bundle=summary.custom_nodes_bundle,
        )
        payload = _build_component_payload(
            component=component,
            component_prompt=synced_component_prompts[component.representative_node_id],
            signature_prompt=prompt,
            extra_data=extra_data,
            settings=resolved_settings,
            requires_volume_reload=bool(uploaded_volume_paths),
            volume_reload_marker=volume_reload_marker,
            custom_nodes_bundle=summary.custom_nodes_bundle,
            uploaded_volume_paths=uploaded_volume_paths,
            terminate_container_on_error=resolved_settings.terminate_container_on_error,
            nodes_module=resolved_nodes_module,
        )
        proxy_node_ids = _rewrite_component_into_proxy(
            component=component,
            rewritten_prompt=rewritten_prompt,
            payload=payload,
            nodes_module=resolved_nodes_module,
        )
        split_proxy_payloads = payload.get("split_proxy_payloads")
        if isinstance(split_proxy_payloads, dict):
            static_proxy_node_id, mapped_proxy_node_id = proxy_node_ids
            summary.remote_component_ids.extend(proxy_node_ids)
            summary.component_node_ids_by_representative[static_proxy_node_id] = list(
                component.static_node_ids
            )
            summary.component_node_ids_by_representative[mapped_proxy_node_id] = list(
                component.mapped_node_ids
            )
            for node_id in component.static_node_ids:
                summary.rewritten_node_id_map[node_id] = static_proxy_node_id
            for node_id in component.mapped_node_ids:
                summary.rewritten_node_id_map[node_id] = mapped_proxy_node_id
            mapped_proxy_component_ids.add(mapped_proxy_node_id)
            continue
        if isinstance(split_proxy_payloads, list):
            summary.remote_component_ids.extend(proxy_node_ids)
            for phase_payload in split_proxy_payloads:
                phase_proxy_node_id = str(phase_payload["component_id"])
                phase_component_node_ids = [str(node_id) for node_id in phase_payload["component_node_ids"]]
                summary.component_node_ids_by_representative[phase_proxy_node_id] = phase_component_node_ids
                for node_id in phase_component_node_ids:
                    summary.rewritten_node_id_map[node_id] = phase_proxy_node_id
            continue

        summary.remote_component_ids.extend(proxy_node_ids)
        summary.component_node_ids_by_representative[proxy_node_ids[0]] = list(component.node_ids)
        for node_id in component.node_ids:
            summary.rewritten_node_id_map[node_id] = proxy_node_ids[0]
        if component.mapped_boundary_input_name:
            mapped_proxy_component_ids.add(proxy_node_ids[0])

    proxy_component_groups = {
        component_id: {component_id}
        for component_id in summary.remote_component_ids
    }
    _, dependency_edges, _ = _component_dependency_graph(rewritten_prompt, proxy_component_groups)
    execution_stages = _component_execution_stages(rewritten_prompt, proxy_component_groups)
    summary.component_dependency_ids_by_representative = {
        representative_node_id: sorted(
            upstream_component_id
            for upstream_component_id, downstream_component_ids in dependency_edges.items()
            if representative_node_id in downstream_component_ids
        )
        for representative_node_id in sorted(proxy_component_groups)
    }
    summary.component_execution_stages = [list(stage) for stage in execution_stages]
    summary.mapped_component_ids = sorted(mapped_proxy_component_ids)
    summary.estimated_max_parallel_requests = _estimated_stage_parallelism(
        execution_stages,
        mapped_proxy_component_ids,
        mapped_component_weight=1,
    )
    if resolved_settings.max_containers is not None:
        summary.max_parallel_requests_upper_bound = min(
            summary.estimated_max_parallel_requests,
            resolved_settings.max_containers,
        )
    else:
        summary.max_parallel_requests_upper_bound = summary.estimated_max_parallel_requests

    logger.info(
        "Estimated remote parallelism after proxy rewrite: known_max_parallel_requests=%d max_parallel_requests_upper_bound=%s mapped_components=%s execution_stages=%s",
        summary.estimated_max_parallel_requests,
        summary.max_parallel_requests_upper_bound,
        summary.mapped_component_ids,
        summary.component_execution_stages,
    )

    return rewritten_prompt, summary


async def _queue_prompt_json(
    prompt_server: Any,
    json_data: dict[str, Any],
    modal_response_payload: dict[str, Any] | None = None,
) -> web.Response:
    """Queue a possibly rewritten prompt using ComfyUI's native semantics."""
    execution = _get_execution_module()
    json_data = prompt_server.trigger_on_prompt(json_data)

    if "number" in json_data:
        number = float(json_data["number"])
    else:
        number = prompt_server.number
        if json_data.get("front"):
            number = -number
        prompt_server.number += 1

    if "prompt" not in json_data:
        return web.json_response(
            {
                "error": {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {},
                }
            },
            status=400,
        )

    prompt = json_data["prompt"]
    prompt_id = str(json_data.get("prompt_id", uuid.uuid4()))
    partial_execution_targets = json_data.get("partial_execution_targets")
    valid = await execution.validate_prompt(prompt_id, prompt, partial_execution_targets)

    extra_data = dict(json_data.get("extra_data", {}))
    if "client_id" in json_data:
        extra_data["client_id"] = json_data["client_id"]

    if not valid[0]:
        logger.warning("invalid prompt: %s", valid[1])
        return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)

    outputs_to_execute = valid[2]
    sensitive: dict[str, Any] = {}
    for sensitive_key in execution.SENSITIVE_EXTRA_DATA_KEYS:
        if sensitive_key in extra_data:
            sensitive[sensitive_key] = extra_data.pop(sensitive_key)

    extra_data["create_time"] = int(time.time() * 1000)
    prompt_server.prompt_queue.put(
        (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
    )
    response_payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "number": number,
        "node_errors": valid[3],
    }
    if modal_response_payload:
        response_payload.update(modal_response_payload)
    return web.json_response(response_payload)


def _analysis_route_path(route_path: str) -> str:
    """Return the sibling HTTP route used for dry-run remote-node expansion."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/analyze_remote_nodes"
    return f"{route_path.rstrip('/')}/analyze_remote_nodes"


def setup_modal_queue_route(
    prompt_server: Any | None = None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
) -> None:
    """Register the `/modal/queue_prompt` route once for the active PromptServer."""
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return

    try:
        resolved_server_module = _get_server_module()
    except ModuleNotFoundError:
        logger.debug("ComfyUI server module is not available; skipping route registration.")
        return

    resolved_settings = settings or get_settings()
    prompt_server = prompt_server or getattr(resolved_server_module.PromptServer, "instance", None)
    if prompt_server is None:
        logger.debug("PromptServer.instance is not available; skipping route registration.")
        return

    resolved_sync_engine = sync_engine or ModalAssetSyncEngine.from_environment(resolved_settings)
    analysis_route_path = _analysis_route_path(resolved_settings.route_path)

    @prompt_server.routes.post(analysis_route_path)
    async def modal_analyze_remote_nodes(request: web.Request) -> web.Response:
        """Analyze which workflow nodes should be marked remote for the current graph."""
        logger.info("Received Modal remote-node analysis request.")
        try:
            request_started_at = time.perf_counter()
            json_data = await request.json()
            prompt = json_data.get("prompt")
            if not isinstance(prompt, dict):
                raise ValueError("Modal remote-node analysis requires a 'prompt' object.")

            workflow = json_data.get("workflow")
            seed_node_ids = json_data.get("seed_node_ids") or []
            if not isinstance(seed_node_ids, list):
                raise ValueError("Modal remote-node analysis requires 'seed_node_ids' to be a list.")

            analysis = analyze_remote_node_selection(
                prompt=prompt,
                workflow=workflow if isinstance(workflow, dict) else None,
                seed_workflow_node_paths=[str(seed_node_id) for seed_node_id in seed_node_ids],
                settings=resolved_settings,
            )
            logger.info(
                "Modal remote-node analysis finished in %.3fs with %d requested nodes and %d additions.",
                time.perf_counter() - request_started_at,
                len(analysis.requested_workflow_node_paths),
                len(analysis.added_workflow_node_paths),
            )
            return web.json_response(
                {
                    "requested_node_ids": analysis.requested_node_ids,
                    "requested_workflow_node_paths": analysis.requested_workflow_node_paths,
                    "current_remote_node_ids": analysis.current_remote_node_ids,
                    "current_remote_workflow_node_paths": (
                        analysis.current_remote_workflow_node_paths
                    ),
                    "resolved_remote_node_ids": analysis.resolved_remote_node_ids,
                    "resolved_workflow_node_paths": analysis.resolved_workflow_node_paths,
                    "added_node_ids": analysis.added_node_ids,
                    "added_workflow_node_paths": analysis.added_workflow_node_paths,
                    "reasons": [
                        {
                            "node_id": reason.node_id,
                            "class_type": reason.class_type,
                            "required_by_node_id": reason.required_by_node_id,
                            "required_by_class_type": reason.required_by_class_type,
                            "output_index": reason.output_index,
                            "io_type": reason.io_type,
                        }
                        for reason in analysis.reasons
                    ],
                }
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Modal remote-node analysis request was invalid: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)

    @prompt_server.routes.post(resolved_settings.route_path)
    async def modal_queue_prompt(request: web.Request) -> web.Response:
        """Handle prompt queue requests that include Modal remote markers."""
        logger.info("Received Modal queue request.")
        json_data: dict[str, Any] | None = None
        workflow: dict[str, Any] | None = None
        remote_node_ids: list[str] = []
        summary = RewriteSummary()
        try:
            request_started_at = time.perf_counter()
            json_data = await request.json()
            json_data.setdefault("prompt_id", str(uuid.uuid4()))
            json_data.setdefault("extra_data", {})
            json_data["extra_data"]["prompt_id"] = json_data["prompt_id"]
            if json_data.get("client_id") is not None:
                json_data["extra_data"]["client_id"] = json_data["client_id"]
            client_id = str(json_data.get("client_id")) if json_data.get("client_id") else None
            prompt_id = str(json_data.get("prompt_id")) if json_data.get("prompt_id") else None
            extra_pnginfo = ((json_data.get("extra_data") or {}).get("extra_pnginfo") or {})
            workflow = extra_pnginfo.get("workflow")
            prompt_node_ids = (
                {str(node_id) for node_id in json_data.get("prompt", {}).keys()}
                if "prompt" in json_data
                else None
            )
            remote_node_ids = sorted(
                extract_remote_node_ids(workflow, resolved_settings, prompt_node_ids)
            )

            def emit_setup_status(
                message: str,
                current: int | None = None,
                total: int | None = None,
            ) -> None:
                """Forward one queue-time Modal setup update into the websocket stream."""
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="setup",
                    client_id=client_id,
                    prompt_id=prompt_id,
                    node_ids=remote_node_ids,
                    component_node_ids_by_representative=(
                        summary.component_node_ids_by_representative or None
                    ),
                    status_message=message,
                    status_current=current,
                    status_total=total,
                )

            if "prompt" in json_data:
                emit_setup_status("Preparing Modal workflow")
                rewrite_started_at = time.perf_counter()
                rewritten_prompt, summary = rewrite_prompt_for_modal(
                    prompt=json_data["prompt"],
                    workflow=workflow,
                    sync_engine=resolved_sync_engine,
                    settings=resolved_settings,
                    extra_data=json_data.get("extra_data"),
                    status_callback=emit_setup_status,
                )
                logger.info(
                    "Modal prompt rewrite finished in %.3fs for %d remote nodes across %d components.",
                    time.perf_counter() - rewrite_started_at,
                    len(summary.remote_node_ids),
                    len(summary.remote_component_ids),
                )
                remote_node_ids = list(summary.remote_node_ids)
                json_data["prompt"] = rewritten_prompt
                if json_data.get("partial_execution_targets"):
                    rewritten_targets = {
                        summary.rewritten_node_id_map.get(str(target), str(target))
                        for target in json_data["partial_execution_targets"]
                    }
                    json_data["partial_execution_targets"] = sorted(rewritten_targets)
                json_data.setdefault("extra_data", {}).setdefault("modal", {})
                json_data["extra_data"]["modal"]["remote_node_ids"] = summary.remote_node_ids
                json_data["extra_data"]["modal"]["remote_component_ids"] = summary.remote_component_ids
                json_data["extra_data"]["modal"]["component_dependency_ids_by_representative"] = (
                    summary.component_dependency_ids_by_representative
                )
                json_data["extra_data"]["modal"]["component_execution_stages"] = (
                    summary.component_execution_stages
                )
                json_data["extra_data"]["modal"]["mapped_component_ids"] = summary.mapped_component_ids
                json_data["extra_data"]["modal"]["estimated_max_parallel_requests"] = (
                    summary.estimated_max_parallel_requests
                )
                json_data["extra_data"]["modal"]["max_parallel_requests_upper_bound"] = (
                    summary.max_parallel_requests_upper_bound
                )
                json_data["extra_data"]["modal"]["synced_assets"] = [
                    asset.remote_path for asset in summary.synced_assets
                ]
                if summary.custom_nodes_bundle is not None:
                    json_data["extra_data"]["modal"]["custom_nodes_bundle"] = (
                        summary.custom_nodes_bundle.remote_path
                    )
                if summary.remote_component_ids:
                    from .remote.modal_app import ensure_remote_warm_capacity

                    warmup_target = (
                        summary.max_parallel_requests_upper_bound
                        or summary.estimated_max_parallel_requests
                    )
                    ensure_remote_warm_capacity(
                        {
                            "prompt_id": prompt_id,
                            "component_id": summary.remote_component_ids[0],
                            "requires_volume_reload": summary.requires_volume_reload,
                            "volume_reload_marker": summary.volume_reload_marker,
                            "uploaded_volume_paths": list(summary.uploaded_volume_paths),
                            "custom_nodes_bundle": (
                                summary.custom_nodes_bundle.remote_path
                                if summary.custom_nodes_bundle is not None
                                else None
                            ),
                        },
                        warmup_target=warmup_target,
                        reason="queue_time_structural_estimate",
                    )
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="setup",
                    client_id=client_id,
                    prompt_id=prompt_id,
                    node_ids=remote_node_ids,
                    component_node_ids_by_representative=summary.component_node_ids_by_representative,
                    status_message="Submitting Modal workflow",
                )
            response = await _queue_prompt_json(
                prompt_server,
                json_data,
                modal_response_payload=(
                    {
                        "modal_remote_node_ids": list(summary.remote_node_ids),
                        "modal_components": [
                            {
                                "representative_node_id": representative_node_id,
                                "node_ids": list(component_node_ids),
                            }
                            for representative_node_id, component_node_ids in sorted(
                                summary.component_node_ids_by_representative.items()
                            )
                        ],
                    }
                    if summary.remote_node_ids
                    else None
                ),
            )
            logger.info(
                "Modal queue request completed in %.3fs.",
                time.perf_counter() - request_started_at,
            )
            return response
        except FileNotFoundError as exc:
            logger.exception("Modal asset sync failed.")
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="error",
                    client_id=str(json_data.get("client_id")) if json_data.get("client_id") else None,
                    prompt_id=str(json_data.get("prompt_id")) if json_data.get("prompt_id") else None,
                    node_ids=remote_node_ids,
                    error_message=str(exc),
                )
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
        except ModalPromptValidationError as exc:
            logger.exception("Modal prompt validation failed.")
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="error",
                    client_id=str(json_data.get("client_id")) if json_data.get("client_id") else None,
                    prompt_id=str(json_data.get("prompt_id")) if json_data.get("prompt_id") else None,
                    node_ids=remote_node_ids,
                    error_message=str(exc),
                )
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
        except Exception as exc:
            logger.exception("Modal queue handler failed.")
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="error",
                    client_id=str(json_data.get("client_id")) if json_data.get("client_id") else None,
                    prompt_id=str(json_data.get("prompt_id")) if json_data.get("prompt_id") else None,
                    node_ids=remote_node_ids,
                    error_message=str(exc),
                )
            return web.json_response({"error": str(exc), "node_errors": []}, status=500)

    _ROUTE_REGISTERED = True
    logger.info(
        "Registered Modal queue route at %s and analysis route at %s",
        resolved_settings.route_path,
        analysis_route_path,
    )
