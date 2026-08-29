"""Durable remote-session bridge storage, recovery, and replay."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Mapping

try:
    from .cloud_durable_invocation import _durable_object_store
    from .cloud_runtime_context import session_bridge_store, snapshot_profile_store
    from .durable_state import DurableObjectRef, DurableStateError
    from .serialization import (
        deserialize_node_inputs,
        deserialize_node_outputs,
        deserialize_value,
        serialize_mapping,
        serialize_node_inputs,
        serialize_node_outputs,
        serialize_value,
    )
    from .session_state import (
        InMemoryRemoteSessionBridgeStore,
        InMemoryRemoteSessionStore,
        RemoteSessionBridgeRecord,
        RemoteSessionBridgeRecoveryKind,
        RemoteSessionBridgeRef,
        RemoteSessionHandle,
        RemoteSessionStateError,
        RemoteSessionValueRef,
        is_remote_session_bridge_ref_payload,
        is_remote_session_handle_payload,
        is_remote_session_value_ref_payload,
        stable_session_bridge_key,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_durable_invocation import _durable_object_store
    from cloud_runtime_context import session_bridge_store, snapshot_profile_store
    from durable_state import DurableObjectRef, DurableStateError
    from serialization import (
        deserialize_node_inputs,
        deserialize_node_outputs,
        deserialize_value,
        serialize_mapping,
        serialize_node_inputs,
        serialize_node_outputs,
        serialize_value,
    )
    from session_state import (
        InMemoryRemoteSessionBridgeStore,
        InMemoryRemoteSessionStore,
        RemoteSessionBridgeRecord,
        RemoteSessionBridgeRecoveryKind,
        RemoteSessionBridgeRef,
        RemoteSessionHandle,
        RemoteSessionStateError,
        RemoteSessionValueRef,
        is_remote_session_bridge_ref_payload,
        is_remote_session_handle_payload,
        is_remote_session_value_ref_payload,
        stable_session_bridge_key,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)

_SNAPSHOT_PROFILE_CACHE_LOCK = threading.Lock()
_SNAPSHOT_PROFILE_CACHE: dict[str, list[dict[str, Any]]] = {}
_REMOTE_SESSION_STORE = InMemoryRemoteSessionStore()
_REMOTE_SESSION_BRIDGE_STORE = InMemoryRemoteSessionBridgeStore()
_REMOTE_SESSION_BRIDGE_REPLAY_STATE = threading.local()
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK = threading.Lock()
_REMOTE_SESSION_BRIDGE_VALUE_CACHE: dict[str, Any] = {}
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER: list[str] = []
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_LIMIT = 32
_DURABLE_BRIDGE_SERIALIZATION_IO_TYPES = frozenset(
    {
        "AUDIO",
        "BOOLEAN",
        "CONDITIONING",
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
_DURABLE_BRIDGE_REHYDRATION_IO_TYPES = frozenset(
    {"CLIP", "MODEL", "NOISE", "SAMPLER", "VAE"}
)


@dataclass(frozen=True)
class CloudSessionBridgeHooks:
    """Callbacks supplied by cloud bootstrap and prompt-execution owners."""

    clone_loader_cache_value: Callable[[Any], Any]
    emit_cloud_info: Callable[..., None]
    execute_node_locally_raw: Callable[..., tuple[Any, ...]]
    execute_subgraph_prompt: Callable[..., tuple[Any, ...]]
    is_link: Callable[[Any], bool]
    normalize_prompt_input_value: Callable[[Any, str | None], Any]
    resolve_required_subgraph_nodes: Callable[[dict[str, Any], list[str]], list[str]]


_SESSION_BRIDGE_HOOKS: CloudSessionBridgeHooks | None = None


def configure_cloud_session_bridge_hooks(hooks: CloudSessionBridgeHooks) -> None:
    """Install callbacks without importing upward into cloud orchestration modules."""
    global _SESSION_BRIDGE_HOOKS
    _SESSION_BRIDGE_HOOKS = hooks


def _session_bridge_hooks() -> CloudSessionBridgeHooks:
    """Return configured callbacks or fail with a clear import-order error."""
    if _SESSION_BRIDGE_HOOKS is None:
        raise RuntimeError("Cloud session-bridge hooks have not been configured.")
    return _SESSION_BRIDGE_HOOKS


def remote_session_store() -> InMemoryRemoteSessionStore:
    """Return the process-local live remote-session value store."""
    return _REMOTE_SESSION_STORE


def clear_warm_caches() -> None:
    """Clear bridge and snapshot caches owned by this module."""
    with _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK:
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE.clear()
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.clear()
    with _SNAPSHOT_PROFILE_CACHE_LOCK:
        _SNAPSHOT_PROFILE_CACHE.clear()


def _clone_loader_cache_value(value: Any) -> Any:
    """Delegate request-safe loader-value cloning to the bootstrap owner."""
    return _session_bridge_hooks().clone_loader_cache_value(value)


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _session_bridge_hooks().emit_cloud_info(message, *args)


def _execute_node_locally_raw(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Delegate local node execution to the prompt-execution owner."""
    return _session_bridge_hooks().execute_node_locally_raw(*args, **kwargs)


def _execute_subgraph_prompt(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Delegate subgraph replay to the prompt-execution owner."""
    return _session_bridge_hooks().execute_subgraph_prompt(*args, **kwargs)


def _is_link(value: Any) -> bool:
    """Delegate ComfyUI prompt-link recognition to prompt execution."""
    return _session_bridge_hooks().is_link(value)


def _normalize_prompt_input_value(value: Any, io_type: str | None = None) -> Any:
    """Delegate serialized prompt-input normalization to prompt execution."""
    return _session_bridge_hooks().normalize_prompt_input_value(value, io_type)


def _resolve_required_subgraph_nodes(
    prompt: dict[str, Any],
    execute_node_ids: list[str],
) -> list[str]:
    """Delegate prompt dependency closure resolution to prompt execution."""
    return _session_bridge_hooks().resolve_required_subgraph_nodes(
        prompt,
        execute_node_ids,
    )


def _payload_remote_session_handle(
    payload: dict[str, Any]
) -> RemoteSessionHandle | None:
    """Return the decoded prompt-scoped remote session handle for one payload."""
    remote_session = payload.get("remote_session")
    if not is_remote_session_handle_payload(remote_session):
        return None
    return RemoteSessionHandle.from_payload(remote_session)


def _session_bridge_store() -> Any:
    """Return the durable store used to replay session-backed outputs across containers."""
    shared_store = session_bridge_store()
    return shared_store if shared_store is not None else _REMOTE_SESSION_BRIDGE_STORE




def _snapshot_profile_store() -> Any | None:
    """Return the shared store used to look up snapshot loader-prewarm profiles."""
    return snapshot_profile_store()


def _load_loader_snapshot_profile(snapshot_profile_key: str) -> list[dict[str, Any]]:
    """Return the loader prewarm plans associated with one snapshot profile key."""
    normalized_key = str(snapshot_profile_key).strip()
    if not normalized_key:
        return []

    with _SNAPSHOT_PROFILE_CACHE_LOCK:
        cached_plans = _SNAPSHOT_PROFILE_CACHE.get(normalized_key)
        if cached_plans is not None:
            return copy.deepcopy(cached_plans)

    store = _snapshot_profile_store()
    if store is None:
        return []

    payload = store.get(normalized_key)
    if not isinstance(payload, dict):
        logger.warning(
            "Snapshot profile %s was not found in the shared store.", normalized_key
        )
        return []

    loader_prewarm_plans = payload.get("loader_prewarm_plans")
    if not isinstance(loader_prewarm_plans, list):
        return []

    normalized_plans = [
        copy.deepcopy(plan) for plan in loader_prewarm_plans if isinstance(plan, dict)
    ]
    with _SNAPSHOT_PROFILE_CACHE_LOCK:
        _SNAPSHOT_PROFILE_CACHE[normalized_key] = copy.deepcopy(normalized_plans)
    return normalized_plans


def _sanitize_payload_for_session_bridge_record(
    payload: dict[str, Any]
) -> dict[str, Any]:
    """Strip run-scoped fields from one producer payload before persisting replay metadata."""
    sanitized_payload = copy.deepcopy(payload)
    sanitized_payload.pop("prompt_id", None)
    sanitized_payload.pop("remote_session", None)
    sanitized_payload.pop("clear_remote_session", None)
    sanitized_payload["extra_data"] = {}
    return sanitized_payload


def _json_payload_size_bytes(payload: Any) -> int:
    """Return the compact UTF-8 size of one JSON-safe payload."""
    return len(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _select_remote_session_bridge_recovery_kind(
    *,
    serialized_output: Any | None,
    rehydration_plan: Mapping[str, Any] | None,
) -> RemoteSessionBridgeRecoveryKind:
    """Choose the least expensive complete recovery mechanism for one bridge."""
    if serialized_output is not None:
        return RemoteSessionBridgeRecoveryKind.SERIALIZED_OUTPUT
    if isinstance(rehydration_plan, Mapping):
        plan_kind = str(rehydration_plan.get("kind") or "")
        if plan_kind == "single_node_output":
            return RemoteSessionBridgeRecoveryKind.SINGLE_NODE_PLAN
        if plan_kind == "subgraph_output":
            return RemoteSessionBridgeRecoveryKind.SUBGRAPH_PLAN
    return RemoteSessionBridgeRecoveryKind.PRODUCER_REPLAY


def _remote_session_bridge_recovery_input_names(
    *,
    recovery_kind: RemoteSessionBridgeRecoveryKind,
    rehydration_plan: Mapping[str, Any] | None,
    hydrated_inputs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return producer input names that the selected recovery mechanism needs."""
    if recovery_kind in {
        RemoteSessionBridgeRecoveryKind.SERIALIZED_OUTPUT,
        RemoteSessionBridgeRecoveryKind.SINGLE_NODE_PLAN,
    }:
        return ()
    if recovery_kind is RemoteSessionBridgeRecoveryKind.PRODUCER_REPLAY:
        return tuple(str(input_name) for input_name in hydrated_inputs)

    plan_payload = (
        rehydration_plan.get("payload")
        if isinstance(rehydration_plan, Mapping)
        else None
    )
    boundary_inputs = (
        plan_payload.get("boundary_inputs")
        if isinstance(plan_payload, Mapping)
        else None
    )
    if not isinstance(boundary_inputs, list):
        raise RemoteSessionStateError(
            "Durable bridge subgraph recovery plans must define boundary_inputs."
        )
    input_names = tuple(
        dict.fromkeys(
            str(boundary_input.get("proxy_input_name") or "").strip()
            for boundary_input in boundary_inputs
            if isinstance(boundary_input, Mapping)
            and str(boundary_input.get("proxy_input_name") or "").strip()
        )
    )
    missing_input_names = [
        input_name for input_name in input_names if input_name not in hydrated_inputs
    ]
    if missing_input_names:
        raise RemoteSessionStateError(
            "Durable bridge subgraph recovery inputs are missing hydrated values: "
            f"{missing_input_names}."
        )
    return input_names


def _offload_large_bridge_payloads(
    *,
    hydrated_inputs: Mapping[str, Any],
    producer_inputs: dict[str, Any],
    output_value: Any,
    serialized_output: Any | None,
) -> tuple[
    dict[str, Any],
    DurableObjectRef | None,
    Any | None,
    DurableObjectRef | None,
]:
    """Move oversized bridge inputs and outputs into durable binary objects."""
    max_inline_bytes = get_settings().bridge_inline_max_bytes
    producer_inputs_object: DurableObjectRef | None = None
    serialized_output_object: DurableObjectRef | None = None
    if producer_inputs and _json_payload_size_bytes(producer_inputs) > max_inline_bytes:
        producer_inputs_object = _durable_object_store().put(
            "bridge_objects",
            serialize_node_inputs(hydrated_inputs),
        )
        producer_inputs = {}
    if (
        serialized_output is not None
        and _json_payload_size_bytes(serialized_output) > max_inline_bytes
    ):
        serialized_output_object = _durable_object_store().put(
            "bridge_objects",
            serialize_node_outputs((output_value,)),
        )
        serialized_output = None
    return (
        producer_inputs,
        producer_inputs_object,
        serialized_output,
        serialized_output_object,
    )


def _deserialize_remote_session_bridge_producer_inputs(
    record: RemoteSessionBridgeRecord,
) -> dict[str, Any]:
    """Restore bridge producer inputs from inline metadata or durable storage."""
    if record.producer_inputs_object is not None:
        return deserialize_node_inputs(
            _durable_object_store().get(record.producer_inputs_object)
        )
    if record.producer_inputs_retained is False:
        recovery_kind = (
            record.recovery_kind.value
            if record.recovery_kind is not None
            else "unspecified"
        )
        raise RemoteSessionStateError(
            "Remote session bridge producer inputs were intentionally omitted for "
            f"bridge_key={record.bridge_key!r} recovery_kind={recovery_kind!r}."
        )
    return deserialize_node_inputs(record.producer_inputs)


def _build_remote_session_bridge_record(
    *,
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_id: str,
    output_index: int,
    io_type: str,
    output_value: Any,
) -> RemoteSessionBridgeRecord:
    """Build one durable bridge record for a session-backed boundary output."""
    producer_payload = _sanitize_payload_for_session_bridge_record(payload)
    producer_input_identity = serialize_mapping(hydrated_inputs)
    serialized_output = _serialize_durable_bridge_output(output_value, io_type)
    rehydration_plan = _build_durable_bridge_rehydration_plan(
        payload=producer_payload,
        node_id=node_id,
        output_index=output_index,
        io_type=io_type,
    )
    recovery_kind = _select_remote_session_bridge_recovery_kind(
        serialized_output=serialized_output,
        rehydration_plan=rehydration_plan,
    )
    recovery_input_names = _remote_session_bridge_recovery_input_names(
        recovery_kind=recovery_kind,
        rehydration_plan=rehydration_plan,
        hydrated_inputs=hydrated_inputs,
    )
    producer_inputs_retained = recovery_kind in {
        RemoteSessionBridgeRecoveryKind.SUBGRAPH_PLAN,
        RemoteSessionBridgeRecoveryKind.PRODUCER_REPLAY,
    }
    retained_hydrated_inputs = {
        input_name: hydrated_inputs[input_name] for input_name in recovery_input_names
    }
    producer_inputs = {
        input_name: producer_input_identity[input_name]
        for input_name in recovery_input_names
    }
    bridge_key = stable_session_bridge_key(
        producer_payload=producer_payload,
        producer_inputs=producer_input_identity,
        node_id=node_id,
        output_index=output_index,
    )
    (
        producer_inputs,
        producer_inputs_object,
        serialized_output,
        serialized_output_object,
    ) = _offload_large_bridge_payloads(
        hydrated_inputs=retained_hydrated_inputs,
        producer_inputs=producer_inputs,
        output_value=output_value,
        serialized_output=serialized_output,
    )
    logger.info(
        "Prepared remote session bridge bridge_key=%s recovery_kind=%s "
        "producer_inputs_retained=%s retained_input_count=%d omitted_input_count=%d.",
        bridge_key,
        recovery_kind.value,
        producer_inputs_retained,
        len(recovery_input_names),
        len(producer_input_identity) - len(recovery_input_names),
    )
    return RemoteSessionBridgeRecord(
        bridge_key=bridge_key,
        node_id=node_id,
        output_index=output_index,
        producer_payload=producer_payload,
        producer_inputs=producer_inputs,
        recovery_kind=recovery_kind,
        producer_inputs_retained=producer_inputs_retained,
        producer_inputs_object=producer_inputs_object,
        serialized_output=serialized_output,
        serialized_output_object=serialized_output_object,
        serialized_output_io_type=(
            str(io_type)
            if serialized_output is not None or serialized_output_object is not None
            else None
        ),
        rehydration_plan=rehydration_plan,
        rehydration_plan_io_type=(
            str(io_type) if rehydration_plan is not None else None
        ),
    )


def _store_remote_session_bridge_record(record: RemoteSessionBridgeRecord) -> None:
    """Persist one durable bridge record for replaying remote-only outputs."""
    store = _session_bridge_store()
    put_record = getattr(store, "put_record", None)
    if callable(put_record):
        put_record(record)
        return
    store[record.bridge_key] = record.to_payload()
    logger.info(
        "Stored remote session bridge record bridge_key=%s node_id=%s output_index=%d.",
        record.bridge_key,
        record.node_id,
        record.output_index,
    )


def _load_remote_session_bridge_record(bridge_key: str) -> RemoteSessionBridgeRecord:
    """Load one durable bridge record from the configured backing store."""
    store = _session_bridge_store()
    get_record = getattr(store, "get_record", None)
    if callable(get_record):
        return get_record(bridge_key)
    payload = store.get(bridge_key)
    if not isinstance(payload, dict):
        raise RemoteSessionStateError(
            f"Remote session bridge record {bridge_key!r} was not found."
        )
    logger.info(
        "Resolved remote session bridge record bridge_key=%s from shared store.",
        bridge_key,
    )
    return RemoteSessionBridgeRecord.from_payload(payload)


def _store_remote_session_bridge_value(
    bridge_key: str,
    value: Any,
) -> None:
    """Retain one live bridge value in-process so later mapped phases can skip replay."""
    cached_value = _clone_loader_cache_value(value)
    with _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK:
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE[bridge_key] = cached_value
        if bridge_key in _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER:
            _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.remove(bridge_key)
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.append(bridge_key)
        while (
            len(_REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER)
            > _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LIMIT
        ):
            evicted_key = _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.pop(0)
            _REMOTE_SESSION_BRIDGE_VALUE_CACHE.pop(evicted_key, None)


def _get_remote_session_bridge_value(bridge_key: str) -> Any | None:
    """Return one retained bridge value when the current worker still has it."""
    with _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK:
        cached_value = _REMOTE_SESSION_BRIDGE_VALUE_CACHE.get(bridge_key)
        if cached_value is None:
            return None
        if bridge_key in _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER:
            _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.remove(bridge_key)
        _REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.append(bridge_key)
    return _clone_loader_cache_value(cached_value)


def _serialize_durable_bridge_output(output_value: Any, io_type: str) -> Any | None:
    """Serialize one bridge output when its io_type supports durable direct restore."""
    normalized_io_type = str(io_type or "")
    if normalized_io_type not in _DURABLE_BRIDGE_SERIALIZATION_IO_TYPES:
        return None
    try:
        return serialize_value(output_value)
    except TypeError:
        logger.warning(
            "Skipping durable bridge serialization for io_type=%s value_type=%s.",
            normalized_io_type,
            type(output_value).__name__,
        )
        return None


def _restore_serialized_remote_session_bridge_value(
    record: RemoteSessionBridgeRecord,
    *,
    target_session_handle: RemoteSessionHandle,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> Any | None:
    """Restore one bridge value directly from a durable serialized payload."""
    if record.serialized_output is None and record.serialized_output_object is None:
        return None

    restore_started_at = time.perf_counter()
    if record.serialized_output_object is not None:
        restored_outputs = deserialize_node_outputs(
            _durable_object_store().get(record.serialized_output_object)
        )
        if len(restored_outputs) != 1:
            raise DurableStateError(
                f"Bridge object {record.serialized_output_object.object_path!r} "
                "must contain exactly one output."
            )
        restored_value = restored_outputs[0]
        storage_kind = "object-backed"
    else:
        restored_value = deserialize_value(record.serialized_output)
        storage_kind = "inline"
    _REMOTE_SESSION_STORE.put_bridge_output(
        target_session_handle,
        bridge_key=record.bridge_key,
        node_id=record.node_id,
        output_index=record.output_index,
        value=restored_value,
    )
    _store_remote_session_bridge_value(record.bridge_key, restored_value)
    if resolution_stats is not None:
        resolution_stats.durable_bridge_hits += 1
        resolution_stats.session_restore_writes += 1
        resolution_stats.direct_restore_seconds += (
            time.perf_counter() - restore_started_at
        )
    logger.info(
        "Restored remote session bridge bridge_key=%s from durable %s serialized %s payload into session_id=%s.",
        record.bridge_key,
        storage_kind,
        record.serialized_output_io_type or "bridge",
        target_session_handle.session_id,
    )
    return restored_value


def _build_durable_bridge_rehydration_plan(
    *,
    payload: dict[str, Any],
    node_id: str,
    output_index: int,
    io_type: str,
) -> dict[str, Any] | None:
    """Return a direct rehydration plan when one bridge output can be rebuilt without replay."""
    if str(io_type or "") not in _DURABLE_BRIDGE_REHYDRATION_IO_TYPES:
        return None
    prompt = payload.get("subgraph_prompt")
    if not isinstance(prompt, dict):
        return None
    prompt_node = prompt.get(str(node_id))
    if not isinstance(prompt_node, dict):
        return None
    class_type = prompt_node.get("class_type")
    inputs = prompt_node.get("inputs")
    if (
        not isinstance(class_type, str)
        or not class_type.strip()
        or not isinstance(inputs, dict)
    ):
        return None

    normalized_inputs: dict[str, Any] = {}
    has_linked_input = False
    for input_name, input_value in inputs.items():
        normalized_value = _normalize_prompt_input_value(copy.deepcopy(input_value))
        if _is_link(normalized_value):
            has_linked_input = True
            continue
        normalized_inputs[str(input_name)] = normalized_value

    if not has_linked_input:
        node_data: dict[str, Any] = {"class_type": class_type}
        custom_nodes_bundle = payload.get("custom_nodes_bundle")
        if isinstance(custom_nodes_bundle, str) and custom_nodes_bundle.strip():
            node_data["custom_nodes_bundle"] = custom_nodes_bundle
        return {
            "kind": "single_node_output",
            "node_data": node_data,
            "node_inputs": normalized_inputs,
        }

    required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=[str(node_id)],
        )
    )
    if str(node_id) not in required_node_ids:
        return None
    if _subgraph_contains_sampling_node(prompt, required_node_ids):
        logger.info(
            "Skipping durable %s bridge subgraph rehydration plan for node_id=%s class_type=%s because its dependency closure includes a sampler.",
            io_type,
            node_id,
            class_type,
        )
        return None

    rehydration_payload = copy.deepcopy(payload)
    rehydration_payload[
        "component_id"
    ] = f"{payload.get('component_id', 'component')}::rehydrate:{node_id}"
    rehydration_payload["subgraph_prompt"] = {
        str(current_node_id): copy.deepcopy(prompt[current_node_id])
        for current_node_id in prompt
        if str(current_node_id) in required_node_ids
    }
    rehydration_payload["component_node_ids"] = sorted(required_node_ids)
    rehydration_payload["execute_node_ids"] = [str(node_id)]
    rehydration_payload["mapped_execute_node_ids"] = []
    rehydration_payload["static_execute_node_ids"] = []
    rehydration_payload["boundary_inputs"] = [
        {
            **copy.deepcopy(boundary_input),
            "targets": [
                copy.deepcopy(target)
                for target in boundary_input.get("targets", [])
                if str(target.get("node_id")) in required_node_ids
            ],
        }
        for boundary_input in payload.get("boundary_inputs", [])
        if any(
            str(target.get("node_id")) in required_node_ids
            for target in boundary_input.get("targets", [])
        )
    ]
    rehydration_payload["boundary_outputs"] = [
        {
            "node_id": str(node_id),
            "output_index": int(output_index),
            "io_type": str(io_type),
            "is_list": False,
            "session_output": False,
            "proxy_output_name": f"{node_id}_rehydrated_{output_index}",
        }
    ]
    return {
        "kind": "subgraph_output",
        "payload": rehydration_payload,
    }


def _restore_planned_remote_session_bridge_value(
    record: RemoteSessionBridgeRecord,
    *,
    target_session_handle: RemoteSessionHandle,
    custom_nodes_root: Path | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> Any | None:
    """Restore one bridge value directly from a durable node rehydration plan."""
    if not isinstance(record.rehydration_plan, Mapping):
        return None

    restore_started_at = time.perf_counter()
    plan_kind = str(record.rehydration_plan.get("kind") or "")
    if plan_kind == "single_node_output":
        node_data = record.rehydration_plan.get("node_data")
        node_inputs = record.rehydration_plan.get("node_inputs")
        if not isinstance(node_data, Mapping) or not isinstance(node_inputs, Mapping):
            return None
        outputs = _execute_node_locally_raw(
            dict(node_data),
            dict(node_inputs),
            node_mapping=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
        )
        output_index = record.output_index
    elif plan_kind == "subgraph_output":
        plan_payload = record.rehydration_plan.get("payload")
        if not isinstance(plan_payload, Mapping):
            return None
        executable_payload = dict(plan_payload)
        executable_payload["remote_session"] = target_session_handle.to_payload()
        executable_payload.pop("clear_remote_session", None)
        outputs = _execute_subgraph_prompt(
            executable_payload,
            _deserialize_remote_session_bridge_producer_inputs(record),
            custom_nodes_root,
            None,
            cancellation_event,
            interrupt_store,
            interrupt_flag_key,
        )
        output_index = 0
    else:
        return None
    if output_index < 0 or output_index >= len(outputs):
        raise RemoteSessionStateError(
            f"Durable bridge rehydration plan for {record.bridge_key!r} did not produce output index {record.output_index}."
        )
    restored_value = outputs[output_index]
    _REMOTE_SESSION_STORE.put_bridge_output(
        target_session_handle,
        bridge_key=record.bridge_key,
        node_id=record.node_id,
        output_index=record.output_index,
        value=restored_value,
    )
    _store_remote_session_bridge_value(record.bridge_key, restored_value)
    if resolution_stats is not None:
        resolution_stats.durable_bridge_hits += 1
        resolution_stats.session_restore_writes += 1
        resolution_stats.direct_restore_seconds += (
            time.perf_counter() - restore_started_at
        )
    logger.info(
        "Restored remote session bridge bridge_key=%s from durable %s rehydration plan into session_id=%s.",
        record.bridge_key,
        record.rehydration_plan_io_type or "bridge",
        target_session_handle.session_id,
    )
    return restored_value


def _remote_session_bridge_replay_stack() -> set[str]:
    """Return the thread-local guard set for bridge replay recursion detection."""
    replay_stack = getattr(_REMOTE_SESSION_BRIDGE_REPLAY_STATE, "bridge_keys", None)
    if replay_stack is None:
        replay_stack = set()
        _REMOTE_SESSION_BRIDGE_REPLAY_STATE.bridge_keys = replay_stack
    return replay_stack


def _bridge_record_replays_sampling_node(record: RemoteSessionBridgeRecord) -> bool:
    """Return whether replaying one bridge record would rerun sampler work."""
    execute_node_ids = [
        str(node_id)
        for node_id in record.producer_payload.get("execute_node_ids", [])
        if str(node_id)
    ]
    if not execute_node_ids:
        return False
    subgraph_prompt = record.producer_payload.get("subgraph_prompt")
    if not isinstance(subgraph_prompt, dict):
        return False
    required_node_ids = _resolve_required_subgraph_nodes(
        prompt=subgraph_prompt,
        execute_node_ids=execute_node_ids,
    )
    return _subgraph_contains_sampling_node(subgraph_prompt, required_node_ids)


def _subgraph_contains_sampling_node(
    subgraph_prompt: Mapping[str, Any],
    node_ids: Iterable[str],
) -> bool:
    """Return whether any selected node looks like sampler work."""
    for node_id in node_ids:
        prompt_node = subgraph_prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        class_type = str(prompt_node.get("class_type") or "")
        if "sampler" in class_type.lower():
            return True
    return False


def _rehydrate_remote_session_bridge_value(
    ref: RemoteSessionBridgeRef,
    *,
    target_session_handle: RemoteSessionHandle | None,
    custom_nodes_root: Path | None,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None,
    interrupt_flag_key: str | None,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> Any:
    """Replay one producer phase into the current session when the live value is gone."""
    if target_session_handle is None:
        raise RemoteSessionStateError(
            "Remote session bridge replay requires a target remote_session handle."
        )

    cached_value = _get_remote_session_bridge_value(ref.bridge_key)
    if cached_value is not None:
        restore_started_at = time.perf_counter()
        _REMOTE_SESSION_STORE.put_bridge_output(
            target_session_handle,
            bridge_key=ref.bridge_key,
            node_id=ref.node_id,
            output_index=ref.output_index,
            value=cached_value,
        )
        if resolution_stats is not None:
            resolution_stats.bridge_cache_hits += 1
            resolution_stats.session_restore_writes += 1
            resolution_stats.direct_restore_seconds += (
                time.perf_counter() - restore_started_at
            )
        logger.info(
            "Restored remote session bridge bridge_key=%s directly from warm cache into session_id=%s.",
            ref.bridge_key,
            target_session_handle.session_id,
        )
        return cached_value

    replay_stack = _remote_session_bridge_replay_stack()
    if ref.bridge_key in replay_stack:
        raise RemoteSessionStateError(
            f"Detected recursive remote session bridge replay for {ref.bridge_key!r}."
        )

    record_lookup_started_at = time.perf_counter()
    record = _load_remote_session_bridge_record(ref.bridge_key)
    if resolution_stats is not None:
        resolution_stats.bridge_record_lookups += 1
        resolution_stats.bridge_record_lookup_seconds += (
            time.perf_counter() - record_lookup_started_at
        )
    restored_value = _restore_serialized_remote_session_bridge_value(
        record,
        target_session_handle=target_session_handle,
        resolution_stats=resolution_stats,
    )
    if restored_value is not None:
        return restored_value
    restored_value = _restore_planned_remote_session_bridge_value(
        record,
        target_session_handle=target_session_handle,
        custom_nodes_root=custom_nodes_root,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
        resolution_stats=resolution_stats,
    )
    if restored_value is not None:
        return restored_value
    if _bridge_record_replays_sampling_node(record):
        message = (
            "Remote session bridge replay would rerun a sampler component "
            f"for bridge_key={ref.bridge_key!r}; expected live or warm bridge reuse instead."
        )
        logger.error(message)
        raise RemoteSessionStateError(message)
    replay_payload = copy.deepcopy(record.producer_payload)
    replay_payload["remote_session"] = target_session_handle.to_payload()
    replay_payload.pop("clear_remote_session", None)
    if target_session_handle.prompt_id is not None:
        replay_payload["prompt_id"] = target_session_handle.prompt_id
    replay_inputs = _deserialize_remote_session_bridge_producer_inputs(record)

    logger.info(
        "Replaying remote session bridge bridge_key=%s into session_id=%s via component=%s.",
        ref.bridge_key,
        target_session_handle.session_id,
        replay_payload.get("component_id"),
    )
    replay_stack.add(ref.bridge_key)
    replay_started_at = time.perf_counter()
    try:
        _execute_subgraph_prompt(
            replay_payload,
            replay_inputs,
            custom_nodes_root,
            None,
            cancellation_event,
            interrupt_store,
            interrupt_flag_key,
        )
    finally:
        replay_stack.remove(ref.bridge_key)
    if resolution_stats is not None:
        resolution_stats.replay_count += 1
        resolution_stats.replay_seconds += time.perf_counter() - replay_started_at

    return _REMOTE_SESSION_STORE.get_output(
        RemoteSessionValueRef(
            session_id=target_session_handle.session_id,
            node_id=ref.node_id,
            output_index=ref.output_index,
        )
    )


def _resolve_remote_session_inputs(
    hydrated_inputs: dict[str, Any],
    *,
    component_id: str | None = None,
    target_session_handle: RemoteSessionHandle | None = None,
    custom_nodes_root: Path | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> dict[str, Any]:
    """Resolve any remote-session value refs embedded in boundary inputs."""
    ref_input_names = [
        input_name
        for input_name, input_value in hydrated_inputs.items()
        if is_remote_session_value_ref_payload(input_value)
        or is_remote_session_bridge_ref_payload(input_value)
    ]
    if ref_input_names:
        logger.info(
            "Resolving %d remote session input refs for component=%s inputs=%s.",
            len(ref_input_names),
            component_id or "<unknown>",
            sorted(ref_input_names),
        )
    if resolution_stats is not None:
        resolution_stats.input_ref_count += len(ref_input_names)
    return {
        input_name: _REMOTE_SESSION_STORE.resolve_value_with_bridges(
            input_value,
            target_session_handle=target_session_handle,
            resolution_callback=(
                lambda event_name, event_payload: _record_remote_session_resolution_event(
                    resolution_stats,
                    event_name,
                    event_payload,
                )
            )
            if resolution_stats is not None
            else None,
            bridge_resolver=lambda ref: _rehydrate_remote_session_bridge_value(
                ref,
                target_session_handle=target_session_handle,
                custom_nodes_root=custom_nodes_root,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
                resolution_stats=resolution_stats,
            ),
        )
        for input_name, input_value in hydrated_inputs.items()
    }


def _log_remote_session_resolution_summary(
    *,
    component_id: str,
    resolution_stats: "_RemoteSessionBridgeResolutionStats",
    loader_cache_before: dict[str, int],
    loader_cache_after: dict[str, int],
) -> None:
    """Emit one high-signal log line summarizing bridge resolution cost for a payload."""
    if resolution_stats.input_ref_count <= 0:
        return

    loader_hit_delta = loader_cache_after.get("hit", 0) - loader_cache_before.get(
        "hit", 0
    )
    loader_miss_delta = loader_cache_after.get("miss", 0) - loader_cache_before.get(
        "miss", 0
    )
    _emit_cloud_info(
        "Remote session resolution summary component=%s refs=%d live_hits=%d warm_bridge_hits=%d durable_bridge_hits=%d bridge_record_lookups=%d bridge_record_lookup_seconds=%.3f replay_count=%d replay_seconds=%.3f direct_restore_seconds=%.3f session_restore_writes=%d loader_cache_hits=%d loader_cache_misses=%d",
        component_id,
        resolution_stats.input_ref_count,
        resolution_stats.live_session_hits,
        resolution_stats.bridge_cache_hits,
        resolution_stats.durable_bridge_hits,
        resolution_stats.bridge_record_lookups,
        resolution_stats.bridge_record_lookup_seconds,
        resolution_stats.replay_count,
        resolution_stats.replay_seconds,
        resolution_stats.direct_restore_seconds,
        resolution_stats.session_restore_writes,
        loader_hit_delta,
        loader_miss_delta,
    )


@dataclass
class _RemoteSessionBridgeResolutionStats:
    """Track how one payload resolved session-backed boundary inputs."""

    input_ref_count: int = 0
    live_session_hits: int = 0
    bridge_cache_hits: int = 0
    durable_bridge_hits: int = 0
    bridge_record_lookups: int = 0
    bridge_record_lookup_seconds: float = 0.0
    replay_count: int = 0
    replay_seconds: float = 0.0
    direct_restore_seconds: float = 0.0
    session_restore_writes: int = 0


def _record_remote_session_resolution_event(
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None",
    event_name: str,
    event_payload: Mapping[str, Any],
) -> None:
    """Accumulate one remote-session resolution event when stats collection is active."""
    if resolution_stats is None:
        return
    if event_name in {"session-value-hit", "bridge-target-hit", "bridge-source-hit"}:
        resolution_stats.live_session_hits += 1
