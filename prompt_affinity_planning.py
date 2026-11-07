"""Affinity, local-gap keepalive, and speculative prewarm prompt rewriting."""

from __future__ import annotations

import copy
from collections import deque
import logging
from typing import Any, Mapping
import uuid

if __package__:
    from .component_planning import _component_ancestors_of_local_source
    from .execution_environments import ExecutionProvider
    from .modal_executor_node import (
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ensure_modal_parallel_local_passthrough_registered,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from .remote_graph_analysis import (
        _build_consumer_map,
        _downstream_node_ids_from_targets,
        _is_link,
        _node_output_refs,
    )
    from .remote_plan_types import InputTarget, LinkedOutputRef
    from .settings import (
        MODAL_GLOBAL_COOLDOWN_FIELD,
        MODAL_INHERITED_PAYLOAD_FIELDS,
        MODAL_RESERVE_COOLDOWN_POOL_FIELD,
        MODAL_WORKER_POOL_COMFY,
        MODAL_WORKER_POOL_LLM,
        MODAL_WORKER_POOL_MAPPED,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from component_planning import _component_ancestors_of_local_source
    from execution_environments import ExecutionProvider
    from modal_executor_node import (
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        ensure_modal_parallel_local_passthrough_registered,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from remote_graph_analysis import (
        _build_consumer_map,
        _downstream_node_ids_from_targets,
        _is_link,
        _node_output_refs,
    )
    from remote_plan_types import InputTarget, LinkedOutputRef
    from settings import (
        MODAL_GLOBAL_COOLDOWN_FIELD,
        MODAL_INHERITED_PAYLOAD_FIELDS,
        MODAL_RESERVE_COOLDOWN_POOL_FIELD,
        MODAL_WORKER_POOL_COMFY,
        MODAL_WORKER_POOL_LLM,
        MODAL_WORKER_POOL_MAPPED,
    )

logger = logging.getLogger(__name__)

_SPECULATIVE_PREWARM_TARGET_KEY = "speculative_remote_prewarm_target"
_SPECULATIVE_PREWARM_PAYLOAD_FIELDS = frozenset(
    {
        "component_id",
        "prompt_id",
        *MODAL_INHERITED_PAYLOAD_FIELDS,
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


def _reserve_cooldown_pool(
    component_ids: list[str],
    payloads_by_component_id: Mapping[str, Mapping[str, Any]],
    mapped_component_ids: set[str],
) -> str:
    """Pick the single worker pool whose leading slots keep the reserve cooldown.

    Mapped lanes do the bulk of the work when present; otherwise ordinary ComfyUI
    workers, and LLM workers only when nothing else runs on this Modal target.
    """
    if any(component_id in mapped_component_ids for component_id in component_ids):
        return MODAL_WORKER_POOL_MAPPED
    affinity_groups = {
        str(
            payloads_by_component_id[component_id].get("remote_worker_affinity_group")
            or MODAL_WORKER_POOL_COMFY
        )
        .strip()
        .lower()
        for component_id in component_ids
    }
    if MODAL_WORKER_POOL_COMFY in affinity_groups:
        return MODAL_WORKER_POOL_COMFY
    return MODAL_WORKER_POOL_LLM


def _cooldown_payloads_by_environment(
    rewritten_prompt: dict[str, Any],
    component_ids: list[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    """Return Modal payloads carrying a cooldown policy, grouped by environment."""
    payloads_by_component_id: dict[str, Mapping[str, Any]] = {}
    component_ids_by_environment: dict[str, list[str]] = {}
    for component_id in component_ids:
        payload = _remote_proxy_payload(rewritten_prompt, component_id)
        if (
            payload is None
            or payload.get("execution_provider") != ExecutionProvider.MODAL.value
            or payload.get(MODAL_GLOBAL_COOLDOWN_FIELD) is None
        ):
            continue
        payloads_by_component_id[component_id] = payload
        environment_id = str(payload.get("execution_environment_id") or "")
        component_ids_by_environment.setdefault(environment_id, []).append(
            component_id
        )
    return payloads_by_component_id, component_ids_by_environment


def _configure_reserve_cooldown_pools(
    *,
    rewritten_prompt: dict[str, Any],
    component_ids: list[str],
    mapped_component_ids: set[str],
) -> None:
    """Stamp each Modal payload with the one pool that holds reserved workers.

    Without this, slot 0 of the static ``comfy`` pool and slot 0 of the mapped
    lanes would both keep the reserve cooldown, doubling the idle GPUs paid for.
    """
    payloads_by_component_id, component_ids_by_environment = (
        _cooldown_payloads_by_environment(rewritten_prompt, component_ids)
    )
    for environment_id, environment_component_ids in (
        component_ids_by_environment.items()
    ):
        reserve_pool = _reserve_cooldown_pool(
            environment_component_ids,
            payloads_by_component_id,
            mapped_component_ids,
        )
        for component_id in environment_component_ids:
            rewritten_prompt[component_id]["inputs"][
                "original_node_data"
            ] = update_registered_proxy_payload_fields(
                component_id,
                payloads_by_component_id[component_id],
                {MODAL_RESERVE_COOLDOWN_POOL_FIELD: reserve_pool},
            )
        logger.info(
            "Assigned Modal reserve cooldown pool=%s environment=%s components=%s.",
            reserve_pool,
            environment_id or "<default>",
            environment_component_ids,
        )


def _comfy_pool_modal_payloads_by_environment(
    rewritten_prompt: dict[str, Any],
    component_ids: list[str],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Return Comfy-pool Modal payloads grouped by environment and component."""
    payloads_by_environment: dict[str, dict[str, Mapping[str, Any]]] = {}
    for component_id in component_ids:
        payload = _remote_proxy_payload(rewritten_prompt, component_id)
        if (
            payload is None
            or payload.get("execution_provider") != ExecutionProvider.MODAL.value
        ):
            continue
        affinity_group = (
            str(payload.get("remote_worker_affinity_group") or MODAL_WORKER_POOL_COMFY)
            .strip()
            .lower()
        )
        if affinity_group == MODAL_WORKER_POOL_LLM:
            continue
        environment_id = str(payload.get("execution_environment_id") or "")
        payloads_by_environment.setdefault(environment_id, {})[component_id] = payload
    return payloads_by_environment


def _payload_snapshot_profile_key(payload: Mapping[str, Any]) -> str:
    """Return the loader snapshot profile stamped on one payload, or an empty string."""
    return str(payload.get("snapshot_profile_key") or "").strip()


def _shared_snapshot_profile_key(payloads: list[Mapping[str, Any]]) -> str:
    """Return the most common loader profile among payloads, preferring the lowest key on ties."""
    counts: dict[str, int] = {}
    for payload in payloads:
        profile_key = _payload_snapshot_profile_key(payload)
        if profile_key:
            counts[profile_key] = counts.get(profile_key, 0) + 1
    if not counts:
        return ""
    return min(counts, key=lambda profile_key: (-counts[profile_key], profile_key))


def _with_snapshot_profile_on_unprofiled_payloads(
    payload: Mapping[str, Any], snapshot_profile_key: str
) -> dict[str, Any]:
    """Return a payload copy whose profile-less levels inherit the shared profile."""
    updated_payload = dict(payload)
    if not _payload_snapshot_profile_key(updated_payload):
        updated_payload["snapshot_profile_key"] = snapshot_profile_key
    split_proxy_payloads = updated_payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        updated_payload["split_proxy_payloads"] = {
            key: _with_snapshot_profile_on_unprofiled_payloads(phase, snapshot_profile_key)
            if isinstance(phase, Mapping)
            else phase
            for key, phase in split_proxy_payloads.items()
        }
    elif isinstance(split_proxy_payloads, list):
        updated_payload["split_proxy_payloads"] = [
            _with_snapshot_profile_on_unprofiled_payloads(phase, snapshot_profile_key)
            if isinstance(phase, Mapping)
            else phase
            for phase in split_proxy_payloads
        ]
    return updated_payload


def _configure_shared_snapshot_profiles(
    *,
    rewritten_prompt: dict[str, Any],
    component_ids: list[str],
) -> None:
    """Route profile-less Comfy components into the environment's loader-profile pool.

    Modal scales every distinct ``RemoteEngine`` parameter set independently, each
    with its own ``max_containers``. A component without loaders (for example a
    remote SaveImage) would otherwise open a second container pool beside the
    samplers and double the GPU containers beyond the configured instance count.
    """
    for environment_id, payloads_by_component_id in (
        _comfy_pool_modal_payloads_by_environment(rewritten_prompt, component_ids).items()
    ):
        shared_profile_key = _shared_snapshot_profile_key(
            list(payloads_by_component_id.values())
        )
        if not shared_profile_key:
            continue
        distinct_profile_keys = sorted(
            {
                _payload_snapshot_profile_key(payload)
                for payload in payloads_by_component_id.values()
            }
            - {""}
        )
        if len(distinct_profile_keys) > 1:
            logger.warning(
                "Modal environment=%s uses %d loader snapshot profiles %s; each opens its "
                "own container pool, so only the in-flight call limit caps them jointly.",
                environment_id or "<default>",
                len(distinct_profile_keys),
                distinct_profile_keys,
            )
        unprofiled_component_ids = [
            component_id
            for component_id, payload in payloads_by_component_id.items()
            if not _payload_snapshot_profile_key(payload)
        ]
        for component_id in unprofiled_component_ids:
            rewritten_prompt[component_id]["inputs"][
                "original_node_data"
            ] = update_registered_proxy_payload_fields(
                component_id,
                payloads_by_component_id[component_id],
                _with_snapshot_profile_on_unprofiled_payloads(
                    payloads_by_component_id[component_id], shared_profile_key
                ),
            )
        if unprofiled_component_ids:
            logger.info(
                "Sharing Modal loader snapshot profile %s with profile-less components=%s "
                "in environment=%s so they reuse the same container pool.",
                shared_profile_key,
                unprofiled_component_ids,
                environment_id or "<default>",
            )


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
