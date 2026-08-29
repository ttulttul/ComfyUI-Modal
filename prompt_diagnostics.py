"""Diagnostics for rewritten remote prompts and dependency cycles."""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from typing import Any, Mapping

if __package__:
    from .remote_graph_analysis import _is_link
    from .remote_plan_types import PromptGraphLink, RewriteSummary
else:  # pragma: no cover - flat import inside the Modal container.
    from remote_graph_analysis import _is_link
    from remote_plan_types import PromptGraphLink, RewriteSummary

logger = logging.getLogger(__name__)


def _prompt_node_class_type(prompt: Mapping[str, Any], node_id: str) -> str:
    """Return the class type for a prompt node, or a diagnostic placeholder."""
    prompt_node = prompt.get(str(node_id))
    if not isinstance(prompt_node, Mapping):
        return "<missing>"
    return str(prompt_node.get("class_type", "<missing>"))


def _prompt_graph_links(prompt: Mapping[str, Any]) -> list[PromptGraphLink]:
    """Return direct prompt dependency edges from linked inputs."""
    links: list[PromptGraphLink] = []
    for target_node_id in sorted(str(node_id) for node_id in prompt):
        prompt_node = prompt.get(target_node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs") or {}
        if not isinstance(inputs, Mapping):
            continue
        for input_name, input_value in sorted(
            inputs.items(), key=lambda item: str(item[0])
        ):
            if not _is_link(input_value):
                continue
            source_node_id = str(input_value[0])
            links.append(
                PromptGraphLink(
                    source_node_id=source_node_id,
                    source_output_index=int(input_value[1]),
                    target_node_id=target_node_id,
                    target_input_name=str(input_name),
                    source_class_type=_prompt_node_class_type(prompt, source_node_id),
                    target_class_type=_prompt_node_class_type(prompt, target_node_id),
                )
            )
    return links


def _prompt_link_dict(link: PromptGraphLink) -> dict[str, Any]:
    """Return a JSON-safe representation of one prompt dependency edge."""
    return {
        "source_node_id": link.source_node_id,
        "source_output_index": link.source_output_index,
        "source_class_type": link.source_class_type,
        "target_node_id": link.target_node_id,
        "target_input_name": link.target_input_name,
        "target_class_type": link.target_class_type,
    }


def _find_prompt_dependency_cycles(prompt: Mapping[str, Any]) -> list[list[str]]:
    """Return representative dependency cycles in the prompt graph."""
    links = _prompt_graph_links(prompt)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for link in links:
        if link.source_node_id not in prompt or link.target_node_id not in prompt:
            continue
        adjacency[link.source_node_id].append(link.target_node_id)

    visited_node_ids: set[str] = set()
    active_node_ids: set[str] = set()
    path: list[str] = []
    cycles: list[list[str]] = []
    seen_cycle_keys: set[tuple[str, ...]] = set()

    def normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
        """Return a rotation-stable key for a detected cycle."""
        if not cycle:
            return ()
        rotations = [
            tuple(cycle[index:] + cycle[:index]) for index in range(len(cycle))
        ]
        return min(rotations)

    def visit(node_id: str) -> None:
        """Depth-first search one prompt node for dependency back-edges."""
        if node_id in active_node_ids:
            cycle_start_index = path.index(node_id)
            cycle = path[cycle_start_index:] + [node_id]
            cycle_key = normalize_cycle(cycle[:-1])
            if cycle_key not in seen_cycle_keys:
                seen_cycle_keys.add(cycle_key)
                cycles.append(cycle)
            return
        if node_id in visited_node_ids:
            return
        visited_node_ids.add(node_id)
        active_node_ids.add(node_id)
        path.append(node_id)
        for downstream_node_id in sorted(adjacency.get(node_id, [])):
            visit(downstream_node_id)
        path.pop()
        active_node_ids.remove(node_id)

    for node_id in sorted(str(node_id) for node_id in prompt):
        visit(node_id)
    return cycles


def _modal_proxy_payload_summaries(prompt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact summaries for Modal proxy payloads embedded in a rewritten prompt."""
    summaries: list[dict[str, Any]] = []
    for node_id in sorted(str(node_id) for node_id in prompt):
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs") or {}
        if not isinstance(inputs, Mapping):
            continue
        payload = inputs.get("original_node_data")
        if not isinstance(payload, Mapping):
            continue
        boundary_inputs = [
            {
                "proxy_input_name": str(boundary_input.get("proxy_input_name")),
                "io_type": str(boundary_input.get("io_type")),
                "targets": copy.deepcopy(boundary_input.get("targets", [])),
            }
            for boundary_input in payload.get("boundary_inputs", [])
            if isinstance(boundary_input, Mapping)
        ]
        boundary_outputs = [
            {
                "proxy_output_name": str(boundary_output.get("proxy_output_name")),
                "node_id": str(boundary_output.get("node_id")),
                "output_index": int(boundary_output.get("output_index", 0)),
                "io_type": str(boundary_output.get("io_type")),
                "is_list": bool(boundary_output.get("is_list", False)),
                "session_output": bool(boundary_output.get("session_output", False)),
            }
            for boundary_output in payload.get("boundary_outputs", [])
            if isinstance(boundary_output, Mapping)
        ]
        summaries.append(
            {
                "proxy_node_id": node_id,
                "proxy_class_type": _prompt_node_class_type(prompt, node_id),
                "payload_kind": str(payload.get("payload_kind")),
                "component_id": str(payload.get("component_id")),
                "component_node_ids": [
                    str(value) for value in payload.get("component_node_ids", [])
                ],
                "execute_node_ids": [
                    str(value) for value in payload.get("execute_node_ids", [])
                ],
                "boundary_inputs": boundary_inputs,
                "boundary_outputs": boundary_outputs,
            }
        )
    return summaries


def _modal_rewritten_prompt_diagnostics(
    prompt: Mapping[str, Any],
    summary: RewriteSummary | None = None,
) -> dict[str, Any]:
    """Return compact diagnostics for a rewritten Modal prompt graph."""
    node_class_types = {
        str(node_id): _prompt_node_class_type(prompt, str(node_id))
        for node_id in sorted(str(node_id) for node_id in prompt)
    }
    diagnostics: dict[str, Any] = {
        "node_count": len(prompt),
        "node_class_types": node_class_types,
        "links": [_prompt_link_dict(link) for link in _prompt_graph_links(prompt)],
        "cycles": _find_prompt_dependency_cycles(prompt),
        "modal_proxy_payloads": _modal_proxy_payload_summaries(prompt),
    }
    if summary is not None:
        diagnostics["remote_node_ids"] = list(summary.remote_node_ids)
        diagnostics["remote_component_ids"] = list(summary.remote_component_ids)
        diagnostics["component_node_ids_by_representative"] = copy.deepcopy(
            summary.component_node_ids_by_representative
        )
        diagnostics["component_dependency_ids_by_representative"] = copy.deepcopy(
            summary.component_dependency_ids_by_representative
        )
        diagnostics["component_execution_stages"] = copy.deepcopy(
            summary.component_execution_stages
        )
        diagnostics["parallel_local_branch_node_ids"] = list(
            summary.parallel_local_branch_node_ids
        )
        diagnostics["rewritten_node_id_map"] = copy.deepcopy(
            summary.rewritten_node_id_map
        )
    return diagnostics


def _log_modal_rewritten_prompt_diagnostics(
    *,
    prompt_id: str | None,
    prompt: Mapping[str, Any],
    summary: RewriteSummary | None = None,
    reason: str,
    level: int = logging.INFO,
) -> None:
    """Log compact diagnostics for a rewritten Modal prompt graph."""
    diagnostics = _modal_rewritten_prompt_diagnostics(prompt, summary)
    cycles = diagnostics.get("cycles") or []
    if cycles:
        logger.warning(
            "Modal rewritten prompt contains dependency cycle(s) prompt_id=%s reason=%s cycles=%s",
            prompt_id,
            reason,
            cycles,
        )
    logger.log(
        level,
        "Modal rewritten prompt diagnostics prompt_id=%s reason=%s diagnostics=%s",
        prompt_id,
        reason,
        json.dumps(diagnostics, sort_keys=True),
    )
