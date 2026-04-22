"""Stable Modal cloud entrypoint for ComfyUI Modal-Sync."""

import asyncio
import base64
import copy
import gc
import hashlib
import importlib
import importlib.util
from io import BytesIO
import inspect
import json
import logging
import math
import os
import queue
import sys
import tempfile
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

_REPO_ROOT = Path(__file__).resolve().parent
_REMOTE_REPO_ROOT = Path("/root/comfyui_modal_sync_repo")
_LOCAL_COMFYUI_ROOT = (Path.home() / "git" / "ComfyUI").resolve()
_REMOTE_COMFYUI_ROOT = Path("/root/comfyui_src")
_PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
_COMFYUI_TORCH_VERSION = "2.10.0"
_COMFYUI_TORCHVISION_VERSION = "0.25.0"
_COMFYUI_TORCHAUDIO_VERSION = "2.10.0"
for candidate in (_REPO_ROOT, _REMOTE_REPO_ROOT, _LOCAL_COMFYUI_ROOT, _REMOTE_COMFYUI_ROOT):
    candidate_str = str(candidate)
    try:
        candidate_exists = candidate.exists()
    except PermissionError:
        candidate_exists = False
    if candidate_exists and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serialization import (
    coerce_serialized_node_outputs,
    deserialize_node_inputs,
    deserialize_node_outputs,
    deserialize_value,
    serialize_mapping,
    serialize_node_outputs,
    serialize_value,
)
from session_state import (
    InMemoryRemoteSessionBridgeStore,
    InMemoryRemoteSessionStore,
    RemoteSessionBridgeRecord,
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
_CLOUD_HANDLER_NAME = "comfyui-modal-sync-cloud-timestamped"
_COMFY_RUNTIME_INIT_LOCK = threading.Lock()
_COMFY_RUNTIME_BASE_INITIALIZED = False
_COMFY_RUNTIME_CUSTOM_NODE_ROOTS: set[str] = set()
_EXTRACTED_CUSTOM_NODE_BUNDLES: dict[str, Path] = {}
_LOADER_CACHE_LOCK = threading.Lock()
_LOADER_CACHE_WRAPPED_CLASSES: set[str] = set()
_LOADER_OUTPUT_CACHE: dict[tuple[str, str], tuple[Any, ...]] = {}
_LOADER_CACHE_METRICS_LOCK = threading.Lock()
_LOADER_CACHE_METRICS: dict[str, int] = {"hit": 0, "miss": 0}
_LOADER_PREWARM_PLAN_KEYS_LOCK = threading.Lock()
_LOADER_PREWARM_PLAN_KEYS: set[str] = set()
_SNAPSHOT_PROFILE_CACHE_LOCK = threading.Lock()
_SNAPSHOT_PROFILE_CACHE: dict[str, list[dict[str, Any]]] = {}
_NODE_OUTPUT_CACHE_KEY_PREFIX = "NC_"
_BOUNDARY_INPUT_SIGNATURES_KEY = "__comfy_modal_boundary_input_signatures__"
_NODE_OUTPUT_CACHE_RECORD_VERSION = 1
_PROMPT_EXECUTOR_STATES_LOCK = threading.Lock()
_MODAL_VOLUME_RELOAD_MARKERS_LOCK = threading.Lock()
_CONTAINER_TERMINATION_LOCK = threading.Lock()
_REMOTE_SESSION_STORE = InMemoryRemoteSessionStore()
_REMOTE_SESSION_BRIDGE_STORE = InMemoryRemoteSessionBridgeStore()
_REMOTE_SESSION_BRIDGE_REPLAY_STATE = threading.local()
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_LOCK = threading.Lock()
_REMOTE_SESSION_BRIDGE_VALUE_CACHE: dict[str, Any] = {}
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER: list[str] = []
_REMOTE_SESSION_BRIDGE_VALUE_CACHE_LIMIT = 32
_DURABLE_BRIDGE_SERIALIZATION_IO_TYPES = frozenset({"CONDITIONING"})

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - remote entrypoint only.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


def _payload_remote_session_handle(payload: dict[str, Any]) -> RemoteSessionHandle | None:
    """Return the decoded prompt-scoped remote session handle for one payload."""
    remote_session = payload.get("remote_session")
    if not is_remote_session_handle_payload(remote_session):
        return None
    return RemoteSessionHandle.from_payload(remote_session)


def _session_bridge_store() -> Any:
    """Return the durable store used to replay session-backed outputs across containers."""
    return globals().get("session_bridge_cache") or _REMOTE_SESSION_BRIDGE_STORE


def _snapshot_profile_store() -> Any | None:
    """Return the shared store used to look up snapshot loader-prewarm profiles."""
    return globals().get("snapshot_profiles")


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
        logger.warning("Snapshot profile %s was not found in the shared store.", normalized_key)
        return []

    loader_prewarm_plans = payload.get("loader_prewarm_plans")
    if not isinstance(loader_prewarm_plans, list):
        return []

    normalized_plans = [
        copy.deepcopy(plan)
        for plan in loader_prewarm_plans
        if isinstance(plan, dict)
    ]
    with _SNAPSHOT_PROFILE_CACHE_LOCK:
        _SNAPSHOT_PROFILE_CACHE[normalized_key] = copy.deepcopy(normalized_plans)
    return normalized_plans


def _sanitize_payload_for_session_bridge_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip run-scoped fields from one producer payload before persisting replay metadata."""
    sanitized_payload = copy.deepcopy(payload)
    sanitized_payload.pop("prompt_id", None)
    sanitized_payload.pop("remote_session", None)
    sanitized_payload.pop("clear_remote_session", None)
    sanitized_payload["extra_data"] = {}
    return sanitized_payload


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
    producer_inputs = serialize_mapping(hydrated_inputs)
    serialized_output = _serialize_durable_bridge_output(output_value, io_type)
    rehydration_plan = _build_durable_bridge_rehydration_plan(
        payload=producer_payload,
        node_id=node_id,
        io_type=io_type,
    )
    return RemoteSessionBridgeRecord(
        bridge_key=stable_session_bridge_key(
            producer_payload=producer_payload,
            producer_inputs=producer_inputs,
            node_id=node_id,
            output_index=output_index,
        ),
        node_id=node_id,
        output_index=output_index,
        producer_payload=producer_payload,
        producer_inputs=producer_inputs,
        serialized_output=serialized_output,
        serialized_output_io_type=(str(io_type) if serialized_output is not None else None),
        rehydration_plan=rehydration_plan,
        rehydration_plan_io_type=(str(io_type) if rehydration_plan is not None else None),
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
    logger.info("Resolved remote session bridge record bridge_key=%s from shared store.", bridge_key)
    return RemoteSessionBridgeRecord.from_payload(payload)


def _record_loader_cache_metric(result: str) -> None:
    """Increment one warm-worker loader-cache metric counter."""
    with _LOADER_CACHE_METRICS_LOCK:
        _LOADER_CACHE_METRICS[result] = _LOADER_CACHE_METRICS.get(result, 0) + 1


def _loader_cache_metric_snapshot() -> dict[str, int]:
    """Return the current cumulative loader-cache metrics."""
    with _LOADER_CACHE_METRICS_LOCK:
        return dict(_LOADER_CACHE_METRICS)


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
        while len(_REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER) > _REMOTE_SESSION_BRIDGE_VALUE_CACHE_LIMIT:
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
    if record.serialized_output is None:
        return None

    restore_started_at = time.perf_counter()
    restored_value = deserialize_value(record.serialized_output)
    _REMOTE_SESSION_STORE.put_output(
        target_session_handle,
        node_id=record.node_id,
        output_index=record.output_index,
        value=restored_value,
    )
    _store_remote_session_bridge_value(record.bridge_key, restored_value)
    if resolution_stats is not None:
        resolution_stats.durable_bridge_hits += 1
        resolution_stats.session_restore_writes += 1
        resolution_stats.direct_restore_seconds += time.perf_counter() - restore_started_at
    logger.info(
        "Restored remote session bridge bridge_key=%s from durable serialized %s payload into session_id=%s.",
        record.bridge_key,
        record.serialized_output_io_type or "bridge",
        target_session_handle.session_id,
    )
    return restored_value


def _build_durable_bridge_rehydration_plan(
    *,
    payload: dict[str, Any],
    node_id: str,
    io_type: str,
) -> dict[str, Any] | None:
    """Return a direct rehydration plan when one bridge output can be rebuilt without replay."""
    if str(io_type or "") != "MODEL":
        return None
    prompt = payload.get("subgraph_prompt")
    if not isinstance(prompt, dict):
        return None
    prompt_node = prompt.get(str(node_id))
    if not isinstance(prompt_node, dict):
        return None
    class_type = prompt_node.get("class_type")
    inputs = prompt_node.get("inputs")
    if not isinstance(class_type, str) or not class_type.strip() or not isinstance(inputs, dict):
        return None

    normalized_inputs: dict[str, Any] = {}
    for input_name, input_value in inputs.items():
        normalized_value = _normalize_prompt_input_value(copy.deepcopy(input_value))
        if _is_link(normalized_value):
            logger.info(
                "Skipping durable MODEL bridge rehydration plan for node_id=%s class_type=%s because input %s is still linked.",
                node_id,
                class_type,
                input_name,
            )
            return None
        normalized_inputs[str(input_name)] = normalized_value

    node_data: dict[str, Any] = {"class_type": class_type}
    custom_nodes_bundle = payload.get("custom_nodes_bundle")
    if isinstance(custom_nodes_bundle, str) and custom_nodes_bundle.strip():
        node_data["custom_nodes_bundle"] = custom_nodes_bundle
    return {
        "kind": "single_node_output",
        "node_data": node_data,
        "node_inputs": normalized_inputs,
    }


def _restore_planned_remote_session_bridge_value(
    record: RemoteSessionBridgeRecord,
    *,
    target_session_handle: RemoteSessionHandle,
    resolution_stats: "_RemoteSessionBridgeResolutionStats | None" = None,
) -> Any | None:
    """Restore one bridge value directly from a durable node rehydration plan."""
    if not isinstance(record.rehydration_plan, Mapping):
        return None
    if str(record.rehydration_plan.get("kind") or "") != "single_node_output":
        return None
    node_data = record.rehydration_plan.get("node_data")
    node_inputs = record.rehydration_plan.get("node_inputs")
    if not isinstance(node_data, Mapping) or not isinstance(node_inputs, Mapping):
        return None

    restore_started_at = time.perf_counter()
    outputs = _execute_node_locally_raw(
        dict(node_data),
        dict(node_inputs),
        node_mapping=None,
        cancellation_event=None,
        interrupt_store=None,
        interrupt_flag_key=None,
    )
    if record.output_index < 0 or record.output_index >= len(outputs):
        raise RemoteSessionStateError(
            f"Durable bridge rehydration plan for {record.bridge_key!r} did not produce output index {record.output_index}."
        )
    restored_value = outputs[record.output_index]
    _REMOTE_SESSION_STORE.put_output(
        target_session_handle,
        node_id=record.node_id,
        output_index=record.output_index,
        value=restored_value,
    )
    _store_remote_session_bridge_value(record.bridge_key, restored_value)
    if resolution_stats is not None:
        resolution_stats.durable_bridge_hits += 1
        resolution_stats.session_restore_writes += 1
        resolution_stats.direct_restore_seconds += time.perf_counter() - restore_started_at
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
        _REMOTE_SESSION_STORE.put_output(
            target_session_handle,
            node_id=ref.node_id,
            output_index=ref.output_index,
            value=cached_value,
        )
        if resolution_stats is not None:
            resolution_stats.bridge_cache_hits += 1
            resolution_stats.session_restore_writes += 1
            resolution_stats.direct_restore_seconds += time.perf_counter() - restore_started_at
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
        resolution_stats=resolution_stats,
    )
    if restored_value is not None:
        return restored_value
    replay_payload = copy.deepcopy(record.producer_payload)
    replay_payload["remote_session"] = target_session_handle.to_payload()
    replay_payload.pop("clear_remote_session", None)
    if target_session_handle.prompt_id is not None:
        replay_payload["prompt_id"] = target_session_handle.prompt_id
    replay_inputs = deserialize_node_inputs(record.producer_inputs)

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

    loader_hit_delta = loader_cache_after.get("hit", 0) - loader_cache_before.get("hit", 0)
    loader_miss_delta = loader_cache_after.get("miss", 0) - loader_cache_before.get("miss", 0)
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
class _ReusablePromptExecutorState:
    """Hold a warm-container PromptExecutor and the lock guarding its reuse."""

    executor: Any
    lock: threading.Lock


@dataclass
class _PersistedNodeCacheRestoreState:
    """Track which distributed cache entries were restored into one prompt execution."""

    restored_node_ids: list[str]
    restored_cache_keys_by_node_id: dict[str, str]
    restore_original_method: Callable[[], None]


@dataclass(frozen=True)
class _NodeOutputCacheLookupResult:
    """Hold one distributed cache lookup result before hydration into the live outputs cache."""

    node_id: str
    cache_key: str | None
    raw_record: Any | None
    cache_entry: Any | None


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


_PROMPT_EXECUTOR_STATES: dict[str, _ReusablePromptExecutorState] = {}
_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT = 256
_MODAL_VOLUME_RELOAD_MARKERS: queue.SimpleQueue[str] | None = None
_MODAL_VOLUME_RELOAD_MARKER_SET: set[str] = set()
_REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS = 1.0
_CONTAINER_TERMINATION_SCHEDULED = False
_PRIMITIVE_WIDGET_INPUT_TYPES = frozenset({"INT", "FLOAT", "BOOLEAN", "STRING"})


@dataclass
class _RemoteExecutionControl:
    """Track interruption state for one active remote payload execution."""

    cancellation_event: threading.Event
    interrupt_flag_key: str


def _meaningful_progress_values(node_state: dict[str, Any]) -> tuple[float, float] | None:
    """Return numeric progress values only for node states that represent real progress."""
    try:
        progress_value = float(node_state.get("value", 0.0))
        max_value = float(node_state.get("max", 1.0))
    except (TypeError, ValueError):
        return None

    if max_value <= 1.0:
        return None
    return progress_value, max_value


def _schedule_process_exit(delay_seconds: float, exit_code: int) -> None:
    """Exit the current process after a short delay to retire a bad Modal container."""

    def exit_later() -> None:
        """Sleep briefly so Modal can ship the error response before exiting the worker."""
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        logger.error("Exiting Modal container process with code=%s after remote failure.", exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    threading.Thread(
        target=exit_later,
        name="modal-container-exit",
        daemon=True,
    ).start()


def _is_interrupt_like_failure(exc: Exception) -> bool:
    """Return whether one remote failure represents an expected interruption rather than a crash."""
    return "interrupt" in str(exc).lower()


def _is_session_state_like_failure(exc: Exception) -> bool:
    """Return whether one remote failure came from prompt-scoped session routing/state issues."""
    if isinstance(exc, RemoteSessionStateError):
        return True
    return "remote session" in str(exc).lower()


def _maybe_schedule_container_termination_on_error(
    payload: dict[str, Any],
    exc: Exception,
) -> bool:
    """Retire the current Modal container after a remote execution crash when configured."""
    if not _is_modal_container_runtime():
        return False
    if not bool(payload.get("terminate_container_on_error", True)):
        return False
    if _is_interrupt_like_failure(exc):
        return False
    if _is_session_state_like_failure(exc):
        logger.warning(
            "Skipping Modal container termination for component=%s because the failure looks like a remote session routing/state miss.",
            payload.get("component_id"),
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False

    global _CONTAINER_TERMINATION_SCHEDULED
    with _CONTAINER_TERMINATION_LOCK:
        if _CONTAINER_TERMINATION_SCHEDULED:
            return False
        _CONTAINER_TERMINATION_SCHEDULED = True

    logger.error(
        "Scheduling Modal container termination after remote execution failure for component=%s.",
        payload.get("component_id"),
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    _schedule_process_exit(_REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS, 1)
    return True


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.last_prompt_id: str | None = None

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug("Suppressed remote prompt event %s for client %s.", event, client_id)


class _HeadlessPromptServerInstance:
    """Minimal PromptServer.instance replacement for custom-node import side effects."""

    def __init__(self) -> None:
        """Initialize route registration and no-op websocket state."""
        from aiohttp import web

        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.supports = ["custom_nodes_from_web"]
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.on_prompt_handlers: list[Any] = []

    async def send(self, event: str, data: dict[str, Any], sid: str | None = None) -> None:
        """Discard async websocket sends from import-time custom-node helpers."""
        logger.debug("Suppressed headless remote prompt event %s for client %s.", event, sid)

    def send_sync(self, event: str, data: dict[str, Any], sid: str | None = None) -> None:
        """Discard sync websocket sends from import-time custom-node helpers."""
        logger.debug("Suppressed headless remote prompt event %s for client %s.", event, sid)

    def add_on_prompt_handler(self, handler: Any) -> None:
        """Record prompt handlers registered by custom nodes during import."""
        self.on_prompt_handlers.append(handler)


class _TracingPromptServer(_NullPromptServer):
    """PromptExecutor server stub that records coarse per-node execution timings."""

    def __init__(
        self,
        prompt_id: str,
        prompt: dict[str, Any],
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize timing state for a specific prompt execution."""
        super().__init__()
        self.prompt_id = prompt_id
        self.prompt = prompt
        self._status_callback = status_callback
        self._active_node_id: str | None = None
        self._active_node_started_at: float | None = None
        self.last_prompt_id = prompt_id
        self._boundary_outputs_by_node_id: dict[str, list[dict[str, Any]]] = {}
        self._lookup_cache_entry: Callable[[str], Any | None] | None = None
        self._published_boundary_outputs: set[tuple[str, int]] = set()

    def _classify_node_role(self, class_type: str) -> str:
        """Return a coarse role name for a node class."""
        normalized = class_type.lower()
        if "loader" in normalized or normalized in {"clipvisionencode"}:
            return "model_load"
        if "ksampler" in normalized or "sampler" in normalized:
            return "sampling"
        if "encode" in normalized:
            return "conditioning"
        return "node"

    def _log_node_finish(self, reason: str) -> None:
        """Emit a timing line for the currently active node when one is running."""
        if self._active_node_id is None or self._active_node_started_at is None:
            return

        node_id = self._active_node_id
        node_info = self.prompt.get(node_id, {})
        class_type = str(node_info.get("class_type", "<unknown>"))
        role = self._classify_node_role(class_type)
        elapsed_seconds = time.perf_counter() - self._active_node_started_at
        _emit_cloud_info(
            "Remote node %s class_type=%s role=%s finished in %.3fs reason=%s",
            node_id,
            class_type,
            role,
            elapsed_seconds,
            reason,
        )
        self._active_node_id = None
        self._active_node_started_at = None

    def emit_preview_update(
        self,
        *,
        node_id: str,
        preview_image: Any,
    ) -> None:
        """Publish one preview image update through the status callback."""
        if self._status_callback is None:
            return

        try:
            image_type, image, max_size = preview_image
            image_buffer = BytesIO()
            save_kwargs: dict[str, Any] = {"format": image_type}
            if image_type == "JPEG":
                save_kwargs["quality"] = 95
            elif image_type == "PNG":
                save_kwargs["compress_level"] = 1
            image.save(image_buffer, **save_kwargs)
        except Exception:
            logger.exception("Failed to serialize remote preview image for node %s.", node_id)
            return

        try:
            from comfy_execution.progress import get_progress_state

            registry = get_progress_state()
            display_node_id = registry.dynprompt.get_display_node_id(node_id)
            parent_node_id = registry.dynprompt.get_parent_node_id(node_id)
            real_node_id = registry.dynprompt.get_real_node_id(node_id)
        except Exception:
            logger.exception("Failed to resolve preview metadata for remote node %s.", node_id)
            display_node_id = node_id
            parent_node_id = None
            real_node_id = node_id

        self._status_callback(
            {
                "event_type": "preview",
                "node_id": str(node_id),
                "display_node_id": (
                    str(display_node_id) if display_node_id is not None else None
                ),
                "parent_node_id": (
                    str(parent_node_id) if parent_node_id is not None else None
                ),
                "real_node_id": str(real_node_id) if real_node_id is not None else None,
                "image_type": str(image_type),
                "image_bytes": image_buffer.getvalue(),
                "max_size": int(max_size) if max_size is not None else None,
            }
        )

    def configure_boundary_output_stream(
        self,
        *,
        boundary_outputs: list[dict[str, Any]],
        lookup_cache_entry: Callable[[str], Any | None],
    ) -> None:
        """Configure streamed remote boundary-output publication for this execution."""
        outputs_by_node_id: dict[str, list[dict[str, Any]]] = {}
        for boundary_output in boundary_outputs:
            preview_target_node_ids = [
                str(node_id)
                for node_id in boundary_output.get("preview_target_node_ids", [])
                if str(node_id)
            ]
            if not preview_target_node_ids:
                continue
            if str(boundary_output.get("io_type")) != "IMAGE":
                continue
            node_id = str(boundary_output["node_id"])
            outputs_by_node_id.setdefault(node_id, []).append(boundary_output)

        self._boundary_outputs_by_node_id = outputs_by_node_id
        self._lookup_cache_entry = lookup_cache_entry
        self._published_boundary_outputs.clear()

    def _emit_boundary_outputs_for_node(self, node_id: str | None) -> None:
        """Publish configured boundary image outputs for one completed node once."""
        if (
            node_id is None
            or self._status_callback is None
            or self._lookup_cache_entry is None
        ):
            return

        boundary_outputs = self._boundary_outputs_by_node_id.get(str(node_id), [])
        if not boundary_outputs:
            return

        cache_entry = self._lookup_cache_entry(str(node_id))
        if cache_entry is None:
            return

        cache_outputs = getattr(cache_entry, "outputs", None)
        if not isinstance(cache_outputs, (list, tuple)):
            return

        for boundary_output in boundary_outputs:
            output_index = int(boundary_output["output_index"])
            publication_key = (str(node_id), output_index)
            if publication_key in self._published_boundary_outputs:
                continue
            if output_index >= len(cache_outputs):
                continue

            preview_target_node_ids = [
                str(target_node_id)
                for target_node_id in boundary_output.get("preview_target_node_ids", [])
                if str(target_node_id)
            ]
            if not preview_target_node_ids:
                continue

            self._status_callback(
                {
                    "event_type": "boundary_output",
                    "node_id": str(node_id),
                    "output_index": output_index,
                    "io_type": str(boundary_output.get("io_type", "")),
                    "is_list": bool(boundary_output.get("is_list", False)),
                    "preview_target_node_ids": preview_target_node_ids,
                    "value": _collapse_cache_slot(
                        slot_values=cache_outputs[output_index],
                        is_list=bool(boundary_output.get("is_list", False)),
                    ),
                }
            )
            self._published_boundary_outputs.add(publication_key)

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Track per-node timing transitions from PromptExecutor progress events."""
        if event == "executing":
            next_node_id = data.get("node")
            if next_node_id != self._active_node_id:
                self._emit_boundary_outputs_for_node(self._active_node_id)
                self._log_node_finish(reason="next_node")
            if next_node_id is not None and next_node_id != self._active_node_id:
                node_info = self.prompt.get(str(next_node_id), {})
                class_type = str(node_info.get("class_type", "<unknown>"))
                role = self._classify_node_role(class_type)
                if self._status_callback is not None:
                    self._status_callback(
                        {
                            "phase": "executing",
                            "active_node_id": str(next_node_id),
                            "active_node_class_type": class_type,
                            "active_node_role": role,
                        }
                    )
                self._active_node_id = str(next_node_id)
                self._active_node_started_at = time.perf_counter()
                self.last_node_id = self._active_node_id
                _emit_cloud_info(
                    "Remote node %s class_type=%s role=%s started",
                    self._active_node_id,
                    class_type,
                    role,
                )
            return

        if event == "progress_state":
            if self._status_callback is None:
                return

            nodes_payload = data.get("nodes")
            if not isinstance(nodes_payload, dict):
                return

            tracked_node_id = self._active_node_id
            tracked_node_state: dict[str, Any] | None = None
            if tracked_node_id is not None:
                candidate_state = nodes_payload.get(tracked_node_id)
                if isinstance(candidate_state, dict):
                    tracked_node_state = candidate_state

            if tracked_node_state is None:
                for node_state in nodes_payload.values():
                    if isinstance(node_state, dict) and node_state.get("state") == "running":
                        tracked_node_state = node_state
                        break

            if tracked_node_state is None:
                return

            display_node_id = tracked_node_state.get("display_node_id")
            real_node_id = tracked_node_state.get("real_node_id")
            reported_node_id = display_node_id or real_node_id or tracked_node_state.get("node_id")
            if reported_node_id is None:
                return
            progress_values = _meaningful_progress_values(tracked_node_state)
            if progress_values is None:
                logger.debug(
                    "Ignoring non-meaningful remote progress_state for node_id=%s state=%s value=%r max=%r.",
                    reported_node_id,
                    tracked_node_state.get("state"),
                    tracked_node_state.get("value"),
                    tracked_node_state.get("max"),
                )
                return
            progress_value, max_value = progress_values

            self._status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": str(reported_node_id),
                    "display_node_id": (
                        str(display_node_id) if display_node_id is not None else None
                    ),
                    "real_node_id": str(real_node_id) if real_node_id is not None else None,
                    "value": progress_value,
                    "max": max_value,
                }
            )
            return

        if event == "executed":
            executed_node_id = data.get("node")
            if executed_node_id is not None and str(executed_node_id) == self._active_node_id:
                self._log_node_finish(reason="executed")
            if self._status_callback is not None and data.get("output") is not None:
                self._status_callback(
                    {
                        "event_type": "executed",
                        "node_id": str(data.get("node")),
                        "display_node_id": (
                            str(data["display_node"])
                            if data.get("display_node") is not None
                            else None
                        ),
                        "output": data.get("output"),
                    }
                )
            return

        if event in {"execution_error", "execution_interrupted", "execution_success"}:
            self._emit_boundary_outputs_for_node(self._active_node_id)
            self._log_node_finish(reason=event)
            if self._status_callback is not None:
                self._status_callback({"phase": event})
            return

        super().send_sync(event, data, client_id)


def _build_cloud_log_formatter() -> logging.Formatter:
    """Return the default formatter for remote Modal-Sync logs with timestamps."""
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d +%(relativeCreated)07.0fms %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _configure_cloud_logging() -> logging.Logger:
    """Install a dedicated timestamped handler for the cloud runtime logger."""
    logger.setLevel(logging.INFO)
    for existing_handler in logger.handlers:
        if getattr(existing_handler, "name", "") == _CLOUD_HANDLER_NAME:
            return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_CLOUD_HANDLER_NAME)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_build_cloud_log_formatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _is_modal_container_runtime() -> bool:
    """Return whether the current process is executing inside a Modal container."""
    return os.getenv("MODAL_IS_REMOTE") == "1" or bool(os.getenv("MODAL_TASK_ID"))


def _cloud_formatter() -> logging.Formatter:
    """Return the configured formatter used for cloud phase trace lines."""
    for existing_handler in logger.handlers:
        if getattr(existing_handler, "name", "") == _CLOUD_HANDLER_NAME:
            formatter = existing_handler.formatter
            if formatter is not None:
                return formatter
    return _build_cloud_log_formatter()


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Emit an info line through logging and mirror it to stdout inside Modal containers."""
    if not _is_modal_container_runtime():
        logger.info(message, *args)
        return

    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args,
        exc_info=None,
    )
    print(_cloud_formatter().format(record), file=sys.stdout, flush=True)


def _remote_execution_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the registry key for one active remote execution."""
    prompt_id = str(payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph")
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


@contextmanager
def _registered_remote_execution(
    payload: dict[str, Any],
) -> Iterator[_RemoteExecutionControl]:
    """Prepare interruption state for one active remote execution."""
    prompt_id, component_id = _remote_execution_key(payload)
    control = _RemoteExecutionControl(
        cancellation_event=threading.Event(),
        interrupt_flag_key=_remote_interrupt_flag_key(prompt_id, component_id),
    )
    if modal is not None and "interrupt_flags" in globals():
        interrupt_flags.pop(control.interrupt_flag_key, None)
    try:
        yield control
    finally:
        if modal is not None and "interrupt_flags" in globals():
            interrupt_flags.pop(control.interrupt_flag_key, None)


@contextmanager
def _timed_phase(phase: str, **fields: Any) -> Iterator[None]:
    """Log a start/finish pair with elapsed time for a named execution phase."""
    field_suffix = ""
    if fields:
        rendered_fields = " ".join(f"{key}={value}" for key, value in fields.items())
        field_suffix = f" {rendered_fields}"
    phase_started_at = time.perf_counter()
    _emit_cloud_info("Starting %s%s", phase, field_suffix)
    try:
        yield
    finally:
        _emit_cloud_info(
            "Finished %s in %.3fs%s",
            phase,
            time.perf_counter() - phase_started_at,
            field_suffix,
        )


_configure_cloud_logging()


@contextmanager
def _temporary_node_mapping(node_mapping: dict[str, type[Any]] | None) -> Iterator[None]:
    """Temporarily overlay node mappings for tests or custom runtimes."""
    if node_mapping is None:
        yield
        return

    import nodes

    original_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    original_display_mappings = dict(getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    try:
        nodes.NODE_CLASS_MAPPINGS.update(node_mapping)
        for class_type in node_mapping:
            nodes.NODE_DISPLAY_NAME_MAPPINGS.setdefault(class_type, class_type)
        yield
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(original_mappings)
        if hasattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS"):
            nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
            nodes.NODE_DISPLAY_NAME_MAPPINGS.update(original_display_mappings)


def _extract_custom_nodes_bundle(bundle_path: str | None) -> Path | None:
    """Extract a mirrored custom_nodes bundle ZIP or manifest into a temporary import path."""
    if not bundle_path:
        return None

    settings = get_settings()
    storage_roots = [Path(settings.remote_storage_root)]
    if settings.local_storage_root is not None:
        storage_roots.append(settings.local_storage_root)

    local_bundle: Path | None = None
    for storage_root in storage_roots:
        candidate = storage_root / bundle_path.lstrip("/")
        if candidate.exists():
            local_bundle = candidate
            break

    if local_bundle is None:
        logger.warning("Custom nodes bundle %s was not found in any known storage root.", bundle_path)
        return None

    cached_extraction_root = _EXTRACTED_CUSTOM_NODE_BUNDLES.get(local_bundle.name)
    if cached_extraction_root is not None and cached_extraction_root.exists():
        if str(cached_extraction_root) not in sys.path:
            sys.path.insert(0, str(cached_extraction_root))
        _emit_cloud_info(
            "Reusing extracted remote custom_nodes bundle from %s for %s.",
            cached_extraction_root,
            local_bundle.name,
        )
        return cached_extraction_root

    extraction_root = Path(tempfile.gettempdir()) / "comfy-modal-sync-custom-nodes" / local_bundle.stem
    extraction_root.mkdir(parents=True, exist_ok=True)
    with _timed_phase("extract_custom_nodes_bundle", bundle=local_bundle.name):
        archives_to_extract = _resolve_custom_nodes_archives(local_bundle, storage_roots)
        for archive_path in archives_to_extract:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extraction_root)

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    _EXTRACTED_CUSTOM_NODE_BUNDLES[local_bundle.name] = extraction_root
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)
    return extraction_root


def _resolve_custom_nodes_archives(
    local_bundle: Path,
    storage_roots: list[Path],
) -> list[Path]:
    """Return the archive paths described by one custom_nodes bundle ZIP or manifest."""
    if local_bundle.suffix.lower() == ".zip":
        return [local_bundle]
    if local_bundle.suffix.lower() != ".json":
        raise RuntimeError(
            f"Unsupported custom_nodes bundle format {local_bundle.suffix!r} for {local_bundle}."
        )

    try:
        manifest_payload = json.loads(local_bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Custom nodes manifest {local_bundle} is unreadable.") from exc
    if not isinstance(manifest_payload, dict):
        raise RuntimeError(f"Custom nodes manifest {local_bundle} must be a JSON object.")
    entry_payloads = manifest_payload.get("entries")
    if not isinstance(entry_payloads, list):
        raise RuntimeError(f"Custom nodes manifest {local_bundle} did not contain a valid entries list.")

    archive_paths: list[Path] = []
    for entry_payload in entry_payloads:
        if not isinstance(entry_payload, dict):
            raise RuntimeError(f"Custom nodes manifest {local_bundle} contained a non-object entry.")
        remote_path = entry_payload.get("remote_path")
        if not isinstance(remote_path, str) or not remote_path.strip():
            raise RuntimeError(f"Custom nodes manifest {local_bundle} contained an entry without remote_path.")
        archive_path = _resolve_custom_nodes_bundle_path(remote_path, storage_roots)
        if archive_path is None:
            raise RuntimeError(
                f"Custom nodes archive {remote_path} referenced by {local_bundle} was not found in any storage root."
            )
        archive_paths.append(archive_path)
    return archive_paths


def _resolve_custom_nodes_bundle_path(bundle_path: str, storage_roots: list[Path]) -> Path | None:
    """Resolve one custom_nodes bundle or archive path against the known storage roots."""
    for storage_root in storage_roots:
        candidate = storage_root / bundle_path.lstrip("/")
        if candidate.exists():
            return candidate
    return None


def _register_custom_nodes_root(custom_nodes_root: Path) -> None:
    """Expose an extracted custom_nodes directory to ComfyUI's folder path registry."""
    import folder_paths

    folder_paths.add_model_folder_path("custom_nodes", str(custom_nodes_root), is_default=True)


def _active_comfyui_root() -> Path | None:
    """Return the ComfyUI source root visible to this runtime."""
    for candidate in (_REMOTE_COMFYUI_ROOT, _LOCAL_COMFYUI_ROOT):
        try:
            if candidate.exists():
                return candidate
        except PermissionError:
            continue
    return None


def _force_import_package_from_root(module_name: str, package_root: Path) -> None:
    """Load a top-level package from a specific root, replacing a non-package shadow if needed."""
    existing_module = sys.modules.get(module_name)
    if existing_module is not None and getattr(existing_module, "__path__", None):
        return

    package_dir = package_root / module_name
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        logger.debug("Package %s does not exist under %s.", module_name, package_root)
        return

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for package {module_name!r}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logger.info("Preloaded ComfyUI package %s from %s.", module_name, package_dir)


def _ensure_comfyui_support_packages() -> None:
    """Preload top-level ComfyUI support packages that are vulnerable to name shadowing."""
    comfyui_root = _active_comfyui_root()
    if comfyui_root is None:
        return

    _force_import_package_from_root("utils", comfyui_root)


def _ensure_headless_prompt_server_instance() -> None:
    """Install a minimal PromptServer.instance for custom-node import-time hooks."""
    try:
        import server
    except ModuleNotFoundError:
        return

    prompt_server_class = getattr(server, "PromptServer", None)
    if prompt_server_class is None:
        return
    if getattr(prompt_server_class, "instance", None) is not None:
        return

    prompt_server_class.instance = _HeadlessPromptServerInstance()
    logger.info("Installed headless PromptServer.instance for remote custom-node initialization.")


@contextmanager
def _temporary_progress_hook(prompt_server: _NullPromptServer) -> Iterator[None]:
    """Install a ComfyUI progress hook so remote samplers emit numeric progress updates."""
    import comfy.utils
    import comfy.model_management
    from comfy_execution.progress import get_progress_state
    from comfy_execution.utils import get_executing_context

    previous_hook = comfy.utils.PROGRESS_BAR_HOOK

    def hook(
        value: float,
        total: float,
        preview_image: Any,
        prompt_id: str | None = None,
        node_id: str | None = None,
    ) -> None:
        """Mirror ComfyUI progress-bar updates into the headless progress registry."""
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = prompt_server.last_prompt_id
        if node_id is None:
            node_id = prompt_server.last_node_id
        if node_id is None:
            return

        resolved_node_id = str(node_id)
        get_progress_state().update_progress(resolved_node_id, value, total, preview_image)
        preview_emitter = getattr(prompt_server, "emit_preview_update", None)
        if preview_image is not None and callable(preview_emitter):
            preview_emitter(node_id=resolved_node_id, preview_image=preview_image)

    comfy.utils.set_progress_bar_global_hook(hook)
    try:
        yield
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_hook)


@contextmanager
def _temporary_remote_interrupt_monitor(
    component_id: str,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[None]:
    """Mirror shared cancellation requests into ComfyUI's interrupt flag inside Modal."""
    if cancellation_event is None and (interrupt_store is None or interrupt_flag_key is None):
        yield
        return

    import nodes

    stop_event = threading.Event()
    try:
        modal_exception_module = importlib.import_module("modal.exception")
    except ModuleNotFoundError:
        modal_exception_module = None
    modal_client_closed_error = getattr(modal_exception_module, "ClientClosed", None)

    def shared_cancel_flag_exists() -> bool:
        """Return whether the shared interrupt flag is present, tolerating Modal shutdown races."""
        try:
            return bool(interrupt_store.contains(interrupt_flag_key))
        except Exception as exc:
            if modal_client_closed_error is not None and isinstance(exc, modal_client_closed_error):
                logger.info(
                    "Remote interrupt monitor stopped after Modal client shutdown for component=%s.",
                    component_id,
                )
                stop_event.set()
                return False
            raise

    def consume_shared_cancel_flag() -> None:
        """Remove the shared interrupt flag if the Modal client is still alive."""
        try:
            interrupt_store.pop(interrupt_flag_key, None)
        except Exception as exc:
            if modal_client_closed_error is not None and isinstance(exc, modal_client_closed_error):
                logger.info(
                    "Remote interrupt monitor skipped flag cleanup after Modal client shutdown for component=%s.",
                    component_id,
                )
                stop_event.set()
                return
            raise

    def monitor_interrupts() -> None:
        """Set ComfyUI's interrupt flag once the caller requests cancellation."""
        while not stop_event.is_set():
            if cancellation_event is not None and cancellation_event.wait(timeout=0.1):
                logger.info("Remote interrupt monitor tripped local event for component=%s.", component_id)
                nodes.interrupt_processing()
                return
            if interrupt_store is None or interrupt_flag_key is None:
                continue
            if not shared_cancel_flag_exists():
                continue
            logger.info(
                "Remote interrupt monitor observed shared cancel flag for component=%s.",
                component_id,
            )
            consume_shared_cancel_flag()
            if stop_event.is_set():
                return
            if cancellation_event is not None:
                cancellation_event.set()
            nodes.interrupt_processing()
            return

    interrupt_thread = threading.Thread(
        target=monitor_interrupts,
        name=f"modal-interrupt-{component_id}",
        daemon=True,
    )
    interrupt_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        interrupt_thread.join(timeout=1.0)


def _ensure_default_custom_nodes_dir() -> Path | None:
    """Create the default ComfyUI custom_nodes directory when the image omits its contents."""
    comfyui_root = _active_comfyui_root()
    if comfyui_root is None:
        return None

    custom_nodes_dir = comfyui_root / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    return custom_nodes_dir


def _materialize_remote_asset_path(value: str) -> str:
    """Resolve a mirrored Modal asset reference to the container-local absolute file path."""
    settings = get_settings()
    remote_storage_root = settings.remote_storage_root.rstrip("/")
    if value.startswith(f"{remote_storage_root}/"):
        return value
    if value.startswith("/"):
        volume_relative_roots = (
            "/assets/",
            "/custom_nodes/",
            "/hashes/",
            "/input/",
            "/models/",
            "/output/",
            "/temp/",
            "/user/",
        )
        if any(value.startswith(root) for root in volume_relative_roots):
            return f"{remote_storage_root}{value}"
    if value.startswith("/assets/"):
        return f"{remote_storage_root}{value}"
    return value


def _clone_loader_cache_value(value: Any) -> Any:
    """Clone a cached loader output when the runtime object supports safe cloning."""
    clone_method = getattr(value, "clone", None)
    if callable(clone_method):
        return clone_method()
    return value


def _clone_loader_cache_outputs(outputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return a request-safe copy of cached loader outputs."""
    return tuple(_clone_loader_cache_value(output) for output in outputs)


def _serialize_loader_cache_key(parts: dict[str, Any]) -> str:
    """Serialize a loader cache key into a stable string representation."""
    return json.dumps(parts, sort_keys=True, default=str)


def _build_unet_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI UNET loader."""
    import folder_paths

    return _serialize_loader_cache_key(
        {
            "unet_path": folder_paths.get_full_path_or_raise(
                "diffusion_models",
                str(kwargs["unet_name"]),
            ),
            "weight_dtype": kwargs.get("weight_dtype", "default"),
        }
    )


def _build_clip_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI CLIP loader."""
    import folder_paths

    return _serialize_loader_cache_key(
        {
            "clip_path": folder_paths.get_full_path_or_raise(
                "text_encoders",
                str(kwargs["clip_name"]),
            ),
            "type": kwargs.get("type", "stable_diffusion"),
            "device": kwargs.get("device", "default"),
        }
    )


def _build_dual_clip_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI dual CLIP loader."""
    import folder_paths

    return _serialize_loader_cache_key(
        {
            "clip_path_1": folder_paths.get_full_path_or_raise(
                "text_encoders",
                str(kwargs["clip_name1"]),
            ),
            "clip_path_2": folder_paths.get_full_path_or_raise(
                "text_encoders",
                str(kwargs["clip_name2"]),
            ),
            "type": kwargs.get("type"),
            "device": kwargs.get("device", "default"),
        }
    )


def _build_vae_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI VAE loader."""
    return _serialize_loader_cache_key({"vae_name": kwargs.get("vae_name")})


def _build_checkpoint_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for checkpoint-style model loaders."""
    import folder_paths

    key_parts: dict[str, Any] = {}
    if "config_name" in kwargs:
        key_parts["config_path"] = folder_paths.get_full_path("configs", str(kwargs["config_name"]))
    if "ckpt_name" in kwargs:
        key_parts["ckpt_path"] = folder_paths.get_full_path_or_raise(
            "checkpoints",
            str(kwargs["ckpt_name"]),
        )
    if "model_path" in kwargs:
        key_parts["model_path"] = str(kwargs["model_path"])
    return _serialize_loader_cache_key(key_parts)


def _wrap_loader_method_with_cache(
    class_type: str,
    node_class: type[Any],
    method_name: str,
    cache_key_builder: Any,
) -> None:
    """Install a warm-container cache wrapper around a heavy loader method."""
    if class_type in _LOADER_CACHE_WRAPPED_CLASSES:
        return

    original_method = getattr(node_class, method_name)
    method_signature = inspect.signature(original_method)

    def cached_method(self: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return cached loader outputs when an identical request was already loaded."""
        bound = method_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        normalized_kwargs = {key: value for key, value in bound.arguments.items() if key != "self"}
        cache_key = (class_type, cache_key_builder(normalized_kwargs))

        with _LOADER_CACHE_LOCK:
            cached_outputs = _LOADER_OUTPUT_CACHE.get(cache_key)
        if cached_outputs is not None:
            _record_loader_cache_metric("hit")
            _emit_cloud_info("Loader cache hit class_type=%s key=%s", class_type, cache_key[1])
            return _clone_loader_cache_outputs(cached_outputs)

        _record_loader_cache_metric("miss")
        _emit_cloud_info("Loader cache miss class_type=%s key=%s", class_type, cache_key[1])
        outputs = original_method(self, *args, **kwargs)
        normalized_outputs = tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
        with _LOADER_CACHE_LOCK:
            _LOADER_OUTPUT_CACHE[cache_key] = normalized_outputs
        return _clone_loader_cache_outputs(normalized_outputs)

    setattr(node_class, method_name, cached_method)
    _LOADER_CACHE_WRAPPED_CLASSES.add(class_type)


def _install_loader_cache_wrappers() -> None:
    """Patch the heavyweight built-in model loaders to reuse warm-container state."""
    nodes_module = _load_nodes_module()
    cacheable_loader_specs = {
        "CheckpointLoader": ("load_checkpoint", _build_checkpoint_loader_cache_key),
        "CheckpointLoaderSimple": ("load_checkpoint", _build_checkpoint_loader_cache_key),
        "UNETLoader": ("load_unet", _build_unet_loader_cache_key),
        "CLIPLoader": ("load_clip", _build_clip_loader_cache_key),
        "DualCLIPLoader": ("load_clip", _build_dual_clip_loader_cache_key),
        "VAELoader": ("load_vae", _build_vae_loader_cache_key),
        "unCLIPCheckpointLoader": ("load_checkpoint", _build_checkpoint_loader_cache_key),
        "ImageOnlyCheckpointLoader": ("load_checkpoint", _build_checkpoint_loader_cache_key),
    }

    for class_type, (method_name, cache_key_builder) in cacheable_loader_specs.items():
        node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
        if node_class is None:
            continue
        _wrap_loader_method_with_cache(class_type, node_class, method_name, cache_key_builder)


def _rewrite_modal_asset_references(value: Any) -> Any:
    """Recursively replace mirrored asset markers with container-local absolute file paths."""
    if isinstance(value, str):
        return _materialize_remote_asset_path(value)
    if isinstance(value, list):
        return [_rewrite_modal_asset_references(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _rewrite_modal_asset_references(item) for key, item in value.items()}
    return value


@contextmanager
def _patched_folder_paths_absolute_lookup() -> Iterator[None]:
    """Teach ComfyUI folder lookups to accept already-materialized absolute asset paths."""
    import folder_paths

    original_get_full_path = folder_paths.get_full_path
    original_get_full_path_or_raise = folder_paths.get_full_path_or_raise

    def patched_get_full_path(folder_name: str, filename: str) -> str | None:
        """Return the absolute file when the prompt already points at a materialized asset."""
        resolved_filename = _materialize_remote_asset_path(filename)
        if os.path.isabs(resolved_filename) and Path(resolved_filename).is_file():
            return resolved_filename
        return original_get_full_path(folder_name, resolved_filename)

    def patched_get_full_path_or_raise(folder_name: str, filename: str) -> str:
        """Raise with the original message when no absolute or folder-based match exists."""
        full_path = patched_get_full_path(folder_name, filename)
        if full_path is None:
            raise FileNotFoundError(
                f"Model in folder '{folder_name}' with filename '{filename}' not found."
            )
        return full_path

    folder_paths.get_full_path = patched_get_full_path
    folder_paths.get_full_path_or_raise = patched_get_full_path_or_raise
    try:
        yield
    finally:
        folder_paths.get_full_path = original_get_full_path
        folder_paths.get_full_path_or_raise = original_get_full_path_or_raise


def _ensure_comfy_runtime_initialized(custom_nodes_root: Path | None) -> None:
    """Initialize ComfyUI's built-in and external node registries for remote execution."""
    global _COMFY_RUNTIME_BASE_INITIALIZED

    custom_nodes_root_key = str(custom_nodes_root.resolve()) if custom_nodes_root is not None else None
    with _COMFY_RUNTIME_INIT_LOCK:
        with _timed_phase(
            "ensure_comfy_runtime_initialized",
            custom_nodes=custom_nodes_root_key or "none",
        ):
            _ensure_comfyui_support_packages()
            _ensure_default_custom_nodes_dir()
            _ensure_headless_prompt_server_instance()
            nodes_module = _load_nodes_module()

            if not _COMFY_RUNTIME_BASE_INITIALIZED:
                if custom_nodes_root is not None:
                    _register_custom_nodes_root(custom_nodes_root)
                logger.info(
                    "Initializing remote ComfyUI node registry with built-in extras%s.",
                    " and extracted custom nodes" if custom_nodes_root is not None else "",
                )
                with _timed_phase(
                    "init_extra_nodes",
                    custom_nodes=bool(custom_nodes_root is not None),
                    api_nodes=True,
                ):
                    asyncio.run(
                        nodes_module.init_extra_nodes(
                            init_custom_nodes=custom_nodes_root is not None,
                            init_api_nodes=True,
                        )
                    )
                _install_loader_cache_wrappers()
                _COMFY_RUNTIME_BASE_INITIALIZED = True
                if custom_nodes_root_key is not None:
                    _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)
                return

            if custom_nodes_root_key is None or custom_nodes_root_key in _COMFY_RUNTIME_CUSTOM_NODE_ROOTS:
                logger.info(
                    "Reusing initialized remote ComfyUI runtime for custom_nodes=%s without re-running custom node import.",
                    custom_nodes_root_key or "<default>",
                )
                _install_loader_cache_wrappers()
                return

            _register_custom_nodes_root(custom_nodes_root)
            logger.info("Loading extracted remote custom nodes from %s.", custom_nodes_root)
            with _timed_phase("init_external_custom_nodes", custom_nodes=custom_nodes_root_key):
                asyncio.run(nodes_module.init_external_custom_nodes())
            _install_loader_cache_wrappers()
            _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)


def _load_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    _ensure_comfyui_support_packages()
    import execution

    return execution


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _prompt_executor_cache_config(execution: Any) -> tuple[Any, dict[str, float]]:
    """Return the cache settings used by ComfyUI's normal prompt worker."""
    from comfy.cli_args import args

    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    return cache_type, {"lru": args.cache_lru, "ram": args.cache_ram}


def _serialize_prompt_executor_cache_scope(
    cache_type: Any,
    cache_args: dict[str, Any],
    custom_nodes_root: Path | None,
) -> str:
    """Return a stable cache scope key for reusable PromptExecutor instances."""
    return json.dumps(
        {
            "cache_type": str(cache_type),
            "cache_args": cache_args,
            "custom_nodes_root": str(custom_nodes_root.resolve()) if custom_nodes_root is not None else None,
        },
        sort_keys=True,
        default=str,
    )


def _reset_prompt_executor_request_state(executor: Any, prompt_server: Any) -> None:
    """Prepare a reusable PromptExecutor for a fresh request without discarding its caches."""
    executor.server = prompt_server
    executor.status_messages = []
    executor.success = True
    executor.history_result = {}
    prompt_server.client_id = None
    prompt_server.last_node_id = None


def _get_or_create_prompt_executor_state(
    execution: Any,
    prompt_server: Any,
    cache_type: Any,
    cache_args: dict[str, Any],
    custom_nodes_root: Path | None,
) -> _ReusablePromptExecutorState:
    """Return the warm-container PromptExecutor state for a cache scope, creating it once."""
    state_key = _serialize_prompt_executor_cache_scope(cache_type, cache_args, custom_nodes_root)
    with _PROMPT_EXECUTOR_STATES_LOCK:
        existing_state = _PROMPT_EXECUTOR_STATES.get(state_key)
        if existing_state is not None:
            _emit_cloud_info("Prompt executor cache hit scope=%s", state_key)
            return existing_state

        _emit_cloud_info("Prompt executor cache miss scope=%s", state_key)
        executor = execution.PromptExecutor(
            prompt_server,
            cache_type=cache_type,
            cache_args=cache_args,
        )
        state = _ReusablePromptExecutorState(executor=executor, lock=threading.Lock())
        _PROMPT_EXECUTOR_STATES[state_key] = state
        return state


def _node_output_cache_store() -> Any | None:
    """Return the shared Modal Dict used for persisted transport-safe node outputs."""
    if modal is None:
        return None
    return globals().get("node_output_cache")


def _node_output_cache_key_preview(cache_key: str | None, *, max_chars: int = 32) -> str:
    """Return a short human-readable prefix of one persisted node-cache key."""
    if cache_key is None:
        return "<none>"
    return cache_key[:max_chars]


def _node_output_cache_value_preview(value: Any, *, max_chars: int = 160) -> str:
    """Return a truncated repr for node-cache debug logging."""
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive logging path.
        rendered = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}..."


def _tensor_cache_key_digest(value: Any) -> dict[str, Any]:
    """Return a stable digest payload for one tensor used inside a cache key."""
    from safetensors.torch import save

    tensor = value.detach().contiguous().cpu()
    tensor_bytes = save({"value": tensor})
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(tensor_bytes).hexdigest(),
    }


async def _node_output_cache_store_get(cache_store: Any, cache_key: str) -> Any:
    """Return one persisted node-cache record, preferring Modal's async Dict interface."""
    aio_get = getattr(getattr(cache_store, "get", None), "aio", None)
    if callable(aio_get):
        return await aio_get(cache_key)
    return cache_store.get(cache_key)


def _is_input_signature_cache_key_set(cache_key_set: Any) -> bool:
    """Return whether one cache-key set uses ComfyUI input-signature semantics."""
    return all(
        hasattr(cache_key_set, attribute)
        for attribute in ("dynprompt", "is_changed_cache", "get_ordered_ancestry", "include_node_id_in_input")
    )


def _include_unique_id_in_input_signature(class_type: str) -> bool:
    """Return whether ComfyUI includes the unique node id in this input signature."""
    from comfy_execution.caching import include_unique_id_in_input

    return bool(include_unique_id_in_input(class_type))


def _build_node_output_cache_immediate_signature(
    cache_key_set: Any,
    *,
    dynprompt: Any,
    node_id: str,
    ancestor_order_mapping: dict[str, int],
    is_changed_value: Any,
) -> list[Any]:
    """Return one raw ComfyUI input-signature fragment before `to_hashable()` runs."""
    if not dynprompt.has_node(node_id):
        return [float("NaN")]

    node = dynprompt.get_node(node_id)
    class_type = node["class_type"]
    class_def = _load_nodes_module().NODE_CLASS_MAPPINGS[class_type]
    signature: list[Any] = [class_type, is_changed_value]
    if (
        cache_key_set.include_node_id_in_input()
        or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT)
        or _include_unique_id_in_input_signature(class_type)
    ):
        signature.append(node_id)

    inputs = node["inputs"]
    boundary_input_signatures = node.get(_BOUNDARY_INPUT_SIGNATURES_KEY)
    for key in sorted(inputs.keys()):
        input_value = inputs[key]
        if _is_link(input_value):
            ancestor_id = str(input_value[0])
            ancestor_socket = int(input_value[1])
            ancestor_index = int(ancestor_order_mapping[ancestor_id])
            signature.append((key, ("ANCESTOR", ancestor_index, ancestor_socket)))
        else:
            boundary_signature = None
            if isinstance(boundary_input_signatures, dict):
                candidate_signature = boundary_input_signatures.get(str(key))
                if isinstance(candidate_signature, str) and candidate_signature:
                    boundary_signature = candidate_signature
            if boundary_signature is not None:
                signature.append((key, ("BOUNDARY_SOURCE", boundary_signature)))
            else:
                signature.append((key, input_value))
    return signature


async def _build_node_output_cache_signature_from_key_set_async(
    cache_key_set: Any,
    node_id: str,
) -> Any:
    """Return one distributed cache signature derived from a live ComfyUI cache-key set."""
    if not _is_input_signature_cache_key_set(cache_key_set):
        return cache_key_set.get_data_key(node_id)

    dynprompt = cache_key_set.dynprompt
    ancestors, order_mapping = cache_key_set.get_ordered_ancestry(dynprompt, node_id)
    signature = [
        _build_node_output_cache_immediate_signature(
            cache_key_set,
            dynprompt=dynprompt,
            node_id=node_id,
            ancestor_order_mapping=order_mapping,
            is_changed_value=await cache_key_set.is_changed_cache.get(node_id),
        )
    ]
    for ancestor_id in ancestors:
        signature.append(
            _build_node_output_cache_immediate_signature(
                cache_key_set,
                dynprompt=dynprompt,
                node_id=str(ancestor_id),
                ancestor_order_mapping=order_mapping,
                is_changed_value=await cache_key_set.is_changed_cache.get(ancestor_id),
            )
        )
    return signature


def _build_node_output_cache_signature_from_key_set_sync(
    cache_key_set: Any,
    node_id: str,
) -> Any | None:
    """Return one distributed cache signature using cached `is_changed` values only."""
    if not _is_input_signature_cache_key_set(cache_key_set):
        return cache_key_set.get_data_key(node_id)

    cached_is_changed = getattr(cache_key_set.is_changed_cache, "is_changed", None)
    if not isinstance(cached_is_changed, dict):
        return None

    dynprompt = cache_key_set.dynprompt
    ancestors, order_mapping = cache_key_set.get_ordered_ancestry(dynprompt, node_id)
    all_node_ids = [str(node_id), *[str(ancestor_id) for ancestor_id in ancestors]]
    missing_node_ids = [candidate for candidate in all_node_ids if candidate not in cached_is_changed]
    if missing_node_ids:
        _emit_cloud_info(
            "Node output cache signature rebuild node=%s result=skip reason=missing-is-changed values=%s",
            node_id,
            missing_node_ids,
        )
        return None

    return [
        _build_node_output_cache_immediate_signature(
            cache_key_set,
            dynprompt=dynprompt,
            node_id=candidate,
            ancestor_order_mapping=order_mapping,
            is_changed_value=cached_is_changed[candidate],
        )
        for candidate in all_node_ids
    ]


def _canonicalize_node_output_cache_key_part(
    value: Any,
    *,
    path: str = "root",
) -> Any | None:
    """Return a JSON-stable representation of one CacheKeySetInputSignature fragment."""
    value_type_name = type(value).__name__
    if value_type_name == "Unhashable":
        _emit_cloud_info(
            "Node output cache canonicalization path=%s result=unhashable reason=comfy-unhashable-marker type=%s value=%s",
            path,
            value_type_name,
            _node_output_cache_value_preview(value),
        )
        return None
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "float", "value": "nan"}
        if math.isinf(value):
            return {"kind": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return _tensor_cache_key_digest(value)
    if isinstance(value, tuple):
        items = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            canonical_item = _canonicalize_node_output_cache_key_part(
                item,
                path=child_path,
            )
            if canonical_item is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=tuple-child child_path=%s parent_type=%s parent_value=%s",
                    path,
                    child_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            items.append(canonical_item)
        return {"kind": "tuple", "items": items}
    if isinstance(value, list):
        items = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            canonical_item = _canonicalize_node_output_cache_key_part(
                item,
                path=child_path,
            )
            if canonical_item is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=list-child child_path=%s parent_type=%s parent_value=%s",
                    path,
                    child_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            items.append(canonical_item)
        return {"kind": "list", "items": items}
    if isinstance(value, dict):
        items: list[dict[str, Any]] = []
        for key in sorted(value):
            rendered_key = _node_output_cache_value_preview(key, max_chars=48)
            key_path = f"{path}.key[{rendered_key}]"
            value_path = f"{path}[{rendered_key}]"
            canonical_key = _canonicalize_node_output_cache_key_part(
                key,
                path=key_path,
            )
            canonical_value = _canonicalize_node_output_cache_key_part(
                value[key],
                path=value_path,
            )
            if canonical_key is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=dict-key child_path=%s parent_type=%s parent_value=%s",
                    path,
                    key_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            if canonical_value is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=dict-value key=%s child_path=%s parent_type=%s parent_value=%s",
                    path,
                    rendered_key,
                    value_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            items.append({"key": canonical_key, "value": canonical_value})
        return {"kind": "dict", "items": items}
    if isinstance(value, frozenset):
        canonical_items: list[Any] = []
        for index, item in enumerate(
            sorted(
                value,
                key=lambda item: _node_output_cache_value_preview(item, max_chars=120),
            )
        ):
            child_path = f"{path}{{{index}}}"
            canonical_item = _canonicalize_node_output_cache_key_part(
                item,
                path=child_path,
            )
            if canonical_item is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=frozenset-child child_path=%s parent_type=%s parent_value=%s",
                    path,
                    child_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            canonical_items.append(canonical_item)
        canonical_items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {"kind": "frozenset", "items": canonical_items}
    _emit_cloud_info(
        "Node output cache canonicalization path=%s result=unhashable reason=unsupported-type type=%s value=%s",
        path,
        value_type_name,
        _node_output_cache_value_preview(value),
    )
    return None


def _node_output_cache_key(signature: Any) -> str | None:
    """Return the persisted Modal Dict key for one ComfyUI cache signature."""
    canonical_signature = _canonicalize_node_output_cache_key_part(signature)
    if canonical_signature is None:
        return None
    signature_payload = json.dumps(
        canonical_signature,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature_digest = hashlib.sha256(signature_payload).hexdigest()
    return f"{_NODE_OUTPUT_CACHE_KEY_PREFIX}{signature_digest}"


async def _node_output_cache_key_from_key_set_async(
    cache_key_set: Any,
    node_id: str,
) -> str | None:
    """Return the persisted Modal Dict key for one live ComfyUI cache-key-set node."""
    return _node_output_cache_key(
        await _build_node_output_cache_signature_from_key_set_async(
            cache_key_set,
            node_id,
        )
    )


def _node_output_cache_key_from_key_set_sync(
    cache_key_set: Any,
    node_id: str,
) -> str | None:
    """Return the persisted Modal Dict key for one executed ComfyUI cache-key-set node."""
    signature = _build_node_output_cache_signature_from_key_set_sync(cache_key_set, node_id)
    if signature is None:
        return None
    return _node_output_cache_key(signature)


def _estimate_node_output_cache_value_size_bytes(
    value: Any,
    *,
    byte_limit: int,
) -> int | None:
    """Return a best-effort raw-size estimate for one transport-safe value."""
    if byte_limit < 0:
        return None
    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 8
    if isinstance(value, float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())

    if isinstance(value, tuple | list):
        total_size = 0
        for item in value:
            item_size = _estimate_node_output_cache_value_size_bytes(item, byte_limit=byte_limit)
            if item_size is None:
                return None
            total_size += item_size
            if total_size > byte_limit:
                return total_size
        return total_size

    if isinstance(value, dict):
        total_size = 0
        for key, item in value.items():
            total_size += len(str(key).encode("utf-8"))
            if total_size > byte_limit:
                return total_size
            item_size = _estimate_node_output_cache_value_size_bytes(item, byte_limit=byte_limit)
            if item_size is None:
                return None
            total_size += item_size
            if total_size > byte_limit:
                return total_size
        return total_size

    return None


def _serialize_node_output_cache_entry(
    cache_entry: Any,
    *,
    max_bytes: int,
) -> dict[str, Any] | None:
    """Return a persisted node-cache record when the outputs are safe and small enough."""
    if max_bytes <= 0:
        return None

    outputs_size = _estimate_node_output_cache_value_size_bytes(
        list(getattr(cache_entry, "outputs", [])),
        byte_limit=max_bytes,
    )
    if outputs_size is None or outputs_size > max_bytes:
        return None

    try:
        serialized_outputs = serialize_node_outputs(tuple(getattr(cache_entry, "outputs", [])))
    except TypeError:
        return None

    ui_payload: Any | None = None
    ui_value = getattr(cache_entry, "ui", None)
    if ui_value is not None:
        ui_size = _estimate_node_output_cache_value_size_bytes(ui_value, byte_limit=max_bytes)
        if ui_size is not None and ui_size <= max_bytes:
            try:
                ui_payload = serialize_value(ui_value)
            except TypeError:
                ui_payload = None

    return {
        "version": _NODE_OUTPUT_CACHE_RECORD_VERSION,
        "outputs_zlib": zlib.compress(serialized_outputs),
        "outputs_size_bytes": outputs_size,
        "ui": ui_payload,
    }


def _deserialize_node_output_cache_entry(
    execution: Any,
    record: Any,
) -> Any | None:
    """Return a ComfyUI CacheEntry reconstructed from one persisted Modal Dict record."""
    if not isinstance(record, dict):
        return None
    if int(record.get("version", -1)) != _NODE_OUTPUT_CACHE_RECORD_VERSION:
        return None
    compressed_outputs = record.get("outputs_zlib")
    if not isinstance(compressed_outputs, (bytes, bytearray)):
        return None

    try:
        outputs = list(deserialize_node_outputs(zlib.decompress(bytes(compressed_outputs))))
    except (TypeError, ValueError, zlib.error):
        return None

    ui_payload = record.get("ui")
    try:
        ui_value = deserialize_value(ui_payload) if ui_payload is not None else None
    except TypeError:
        ui_value = None
    return execution.CacheEntry(ui=ui_value, outputs=outputs)


async def _restore_persisted_node_output_cache_entries(
    execution: Any,
    executor: Any,
    *,
    prompt_id: str,
    prompt: dict[str, Any],
    cache_store: Any,
    restored_cache_keys_by_node_id: dict[str, str] | None = None,
) -> list[str]:
    """Hydrate PromptExecutor output-cache misses from the shared Modal Dict."""
    outputs_cache = executor.caches.outputs
    dynamic_prompt = execution.DynamicPrompt(prompt)
    is_changed_cache = execution.IsChangedCache(prompt_id, dynamic_prompt, outputs_cache)
    await outputs_cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
    outputs_cache.clean_unused()

    return await _restore_persisted_node_output_cache_entries_into_prepared_cache(
        execution,
        outputs_cache,
        prompt=prompt,
        cache_store=cache_store,
        restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
    )


async def _restore_persisted_node_output_cache_entries_into_prepared_cache(
    execution: Any,
    outputs_cache: Any,
    *,
    prompt: dict[str, Any],
    cache_store: Any,
    restored_cache_keys_by_node_id: dict[str, str] | None = None,
) -> list[str]:
    """Hydrate one already-prepared PromptExecutor outputs cache from the shared Modal Dict."""
    restored_node_ids: list[str] = []
    pending_lookup_tasks: list[asyncio.Task[_NodeOutputCacheLookupResult]] = []

    async def lookup_node(node_id: str) -> _NodeOutputCacheLookupResult:
        """Resolve one distributed cache candidate without mutating the live outputs cache."""
        cache_key = await _node_output_cache_key_from_key_set_async(outputs_cache.cache_key_set, node_id)
        if cache_key is None:
            return _NodeOutputCacheLookupResult(
                node_id=node_id,
                cache_key=None,
                raw_record=None,
                cache_entry=None,
            )
        raw_record = await _node_output_cache_store_get(cache_store, cache_key)
        return _NodeOutputCacheLookupResult(
            node_id=node_id,
            cache_key=cache_key,
            raw_record=raw_record,
            cache_entry=_deserialize_node_output_cache_entry(execution, raw_record),
        )

    for node_id in prompt:
        if outputs_cache.get(node_id) is not None:
            _emit_cloud_info(
                "Node output cache lookup node=%s result=local-hit",
                node_id,
            )
            continue
        pending_lookup_tasks.append(asyncio.create_task(lookup_node(str(node_id))))

    if not pending_lookup_tasks:
        return restored_node_ids

    for lookup_result in await asyncio.gather(*pending_lookup_tasks):
        node_id = lookup_result.node_id
        cache_key = lookup_result.cache_key
        if cache_key is None:
            _emit_cloud_info(
                "Node output cache lookup node=%s key_prefix=%s result=skip reason=key-unhashable",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        raw_record = lookup_result.raw_record
        cache_entry = lookup_result.cache_entry
        if cache_entry is None:
            result = "miss"
            if raw_record is not None:
                result = "miss-invalid"
            _emit_cloud_info(
                "Node output cache lookup node=%s key_prefix=%s result=%s",
                node_id,
                _node_output_cache_key_preview(cache_key),
                result,
            )
            continue
        outputs_cache.set(node_id, cache_entry)
        _emit_cloud_info(
            "Node output cache lookup node=%s key_prefix=%s result=hit",
            node_id,
            _node_output_cache_key_preview(cache_key),
        )
        if restored_cache_keys_by_node_id is not None:
            restored_cache_keys_by_node_id[str(node_id)] = cache_key
        restored_node_ids.append(str(node_id))
    return restored_node_ids


def _install_prompt_executor_persisted_cache_restore(
    execution: Any,
    executor: Any,
    *,
    component_id: str,
    prompt: dict[str, Any],
    cache_store: Any,
) -> _PersistedNodeCacheRestoreState:
    """Patch one executor so persisted-cache restore runs after its live `set_prompt()` call."""
    restored_node_ids: list[str] = []
    restored_cache_keys_by_node_id: dict[str, str] = {}
    outputs_cache = executor.caches.outputs
    original_set_prompt = outputs_cache.set_prompt

    async def wrapped_set_prompt(dynprompt: Any, node_ids: Any, is_changed_cache: Any) -> None:
        await original_set_prompt(dynprompt, node_ids, is_changed_cache)
        with _timed_phase("restore_persisted_node_cache", component=component_id):
            restored_cache_keys_by_node_id.clear()
            restored_node_ids[:] = await _restore_persisted_node_output_cache_entries_into_prepared_cache(
                execution,
                outputs_cache,
                prompt=prompt,
                cache_store=cache_store,
                restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
            )

    outputs_cache.set_prompt = wrapped_set_prompt

    def restore_original_method() -> None:
        outputs_cache.set_prompt = original_set_prompt

    return _PersistedNodeCacheRestoreState(
        restored_node_ids=restored_node_ids,
        restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
        restore_original_method=restore_original_method,
    )


def _persist_node_output_cache_entries(
    executor: Any,
    *,
    prompt: dict[str, Any],
    cache_store: Any,
    restored_cache_keys_by_node_id: dict[str, str] | None = None,
) -> list[str]:
    """Persist eligible PromptExecutor cache entries into the shared Modal Dict."""
    max_bytes = get_settings().node_output_cache_max_bytes
    if max_bytes <= 0:
        return []

    outputs_cache = executor.caches.outputs
    cache_key_set = getattr(outputs_cache, "cache_key_set", None)
    if cache_key_set is None:
        return []

    persisted_node_ids: list[str] = []
    for node_id in prompt:
        cache_entry = outputs_cache.get(node_id)
        if cache_entry is None:
            _emit_cloud_info(
                "Node output cache write node=%s result=skip reason=no-local-cache-entry",
                node_id,
            )
            continue
        cache_key = _node_output_cache_key_from_key_set_sync(cache_key_set, str(node_id))
        if cache_key is None:
            _emit_cloud_info(
                "Node output cache write node=%s key_prefix=%s result=skip reason=key-unhashable",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        restored_cache_key = None
        if restored_cache_keys_by_node_id is not None:
            restored_cache_key = restored_cache_keys_by_node_id.get(str(node_id))
        if restored_cache_key == cache_key:
            _emit_cloud_info(
                "Node output cache write node=%s key_prefix=%s result=skip reason=restored-hit",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        record = _serialize_node_output_cache_entry(cache_entry, max_bytes=max_bytes)
        if record is None:
            _emit_cloud_info(
                "Node output cache write node=%s key_prefix=%s result=skip reason=ineligible-or-oversize",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        cache_store[cache_key] = record
        _emit_cloud_info(
            "Node output cache write node=%s key_prefix=%s result=write outputs_size_bytes=%s",
            node_id,
            _node_output_cache_key_preview(cache_key),
            record.get("outputs_size_bytes"),
        )
        persisted_node_ids.append(str(node_id))
    return persisted_node_ids


def _invoke_original_node(
    node_class: type[Any],
    node_data: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Execute an original V1 or V3 node class and normalize its outputs."""
    class_type = node_data["class_type"]
    logger.info("Executing remote node %s", class_type)

    if hasattr(node_class, "GET_SCHEMA"):
        node_output = node_class.execute(**kwargs)
        if hasattr(node_output, "result"):
            result = node_output.result
            return tuple(result) if result is not None else tuple()
        return tuple(node_output)

    instance = node_class()
    function_name = getattr(node_class, "FUNCTION", "execute")
    function = getattr(instance, function_name)
    result = function(**kwargs)
    if result is None:
        return tuple()
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def execute_node_locally(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> bytes:
    """Execute a single target node in-process and return serialized outputs."""
    outputs = _execute_node_locally_raw(
        node_data,
        kwargs_payload,
        node_mapping=node_mapping,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )
    return serialize_node_outputs(outputs)


def _execute_node_locally_raw(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute a single target node in-process and return raw node outputs."""
    custom_nodes_root = _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    _ensure_comfy_runtime_initialized(custom_nodes_root)
    kwargs = _rewrite_modal_asset_references(deserialize_node_inputs(kwargs_payload))
    component_id = str(node_data.get("component_id") or node_data.get("class_type") or "single-node")
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        with (
            _patched_folder_paths_absolute_lookup(),
            _temporary_remote_interrupt_monitor(
                component_id,
                cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            ),
        ):
            return _invoke_original_node(node_mapping[class_type], node_data, kwargs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        with (
            _patched_folder_paths_absolute_lookup(),
            _temporary_remote_interrupt_monitor(
                component_id,
                cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            ),
        ):
            return _invoke_original_node(resolved_node_mapping[class_type], node_data, kwargs)


def _apply_boundary_inputs(
    prompt: dict[str, Any],
    boundary_input_specs: list[dict[str, Any]],
    hydrated_inputs: dict[str, Any],
) -> None:
    """Inject hydrated local boundary inputs into a remote subgraph prompt."""
    for boundary_input in boundary_input_specs:
        proxy_input_name = str(boundary_input["proxy_input_name"])
        if proxy_input_name not in hydrated_inputs:
            raise KeyError(f"Missing hydrated boundary input {proxy_input_name!r}.")
        value = hydrated_inputs[proxy_input_name]
        io_type = (
            str(boundary_input["io_type"])
            if boundary_input.get("io_type") is not None
            else None
        )
        for target in boundary_input.get("targets", []):
            node_id = str(target["node_id"])
            input_name = str(target["input_name"])
            prompt_node = prompt[node_id]
            prompt_node["inputs"][input_name] = _normalize_prompt_input_value(
                value,
                io_type=io_type,
            )
            source_signature = boundary_input.get("source_signature")
            if isinstance(source_signature, str) and source_signature:
                boundary_signatures = prompt_node.setdefault(_BOUNDARY_INPUT_SIGNATURES_KEY, {})
                if isinstance(boundary_signatures, dict):
                    boundary_signatures[input_name] = source_signature


def _collapse_cache_slot(slot_values: Any, is_list: bool) -> Any:
    """Convert a PromptExecutor cache slot back into a node-style output value."""
    if is_list:
        return slot_values
    if not isinstance(slot_values, list):
        return slot_values
    if len(slot_values) == 1:
        return slot_values[0]
    return slot_values


def _extract_prompt_executor_error(executor: Any) -> str:
    """Extract a useful failure message from a PromptExecutor run."""
    for event, data in reversed(executor.status_messages):
        if event == "execution_error":
            return _format_prompt_executor_error_payload(data)
        if event == "execution_interrupted":
            return "Remote subgraph execution was interrupted."
    return "Remote subgraph execution failed."


def _format_prompt_executor_error_payload(data: Any) -> str:
    """Return a richer human-readable PromptExecutor failure message when available."""
    if not isinstance(data, dict):
        return "Remote subgraph execution failed."

    message = str(data.get("exception_message") or "Remote subgraph execution failed.")
    node_id = data.get("node_id")
    node_type = data.get("node_type")
    current_inputs = data.get("current_inputs")
    if node_id is None and node_type is None and not current_inputs:
        return message

    details: list[str] = [message]
    if node_id is not None or node_type is not None:
        details.append(f"node_id={node_id!r} node_type={node_type!r}")
    if current_inputs:
        details.append(f"current_inputs={current_inputs!r}")
    return " | ".join(details)


def _extract_prompt_executor_error_payload(executor: Any) -> dict[str, Any] | None:
    """Return the most recent PromptExecutor execution_error payload when present."""
    for event, data in reversed(executor.status_messages):
        if event == "execution_error" and isinstance(data, dict):
            return data
    return None


def _summarize_suspicious_prompt_inputs(prompt: dict[str, Any]) -> list[str]:
    """Return compact descriptions of prompt inputs that still look list-wrapped."""
    findings: list[str] = []
    for node_id, node_info in sorted(prompt.items()):
        inputs = node_info.get("inputs") or {}
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 1:
                findings.append(f"{node_id}.{input_name}={input_value!r}")
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
                and isinstance(input_value[1], list)
            ):
                findings.append(f"{node_id}.{input_name}={input_value!r}")
    return findings


def _node_input_type_map(node_class: type[Any]) -> dict[str, str]:
    """Return one node class's declared V1 input types keyed by input name."""
    input_types_callable = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types_callable):
        return {}

    raw_input_types = input_types_callable()
    if not isinstance(raw_input_types, dict):
        return {}

    input_type_map: dict[str, str] = {}
    for section_name in ("required", "optional", "hidden"):
        section = raw_input_types.get(section_name)
        if not isinstance(section, dict):
            continue
        for input_name, input_config in section.items():
            if not isinstance(input_config, tuple) or not input_config:
                continue
            declared_type = input_config[0]
            if isinstance(declared_type, str):
                input_type_map[str(input_name)] = declared_type
    return input_type_map


def _coerce_primitive_prompt_input_value(
    *,
    node_id: str,
    class_type: str,
    input_name: str,
    declared_type: str,
    input_value: Any,
) -> Any:
    """Coerce one primitive prompt literal using ComfyUI's `validate_inputs` semantics."""
    literal_value = (
        input_value.get("__value__")
        if isinstance(input_value, dict) and "__value__" in input_value
        else input_value
    )
    if isinstance(literal_value, list):
        return input_value

    try:
        if declared_type == "INT":
            return int(literal_value)
        if declared_type == "FLOAT":
            return float(literal_value)
        if declared_type == "STRING":
            return str(literal_value)
        if declared_type == "BOOLEAN":
            return bool(literal_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RemoteSubgraphExecutionError(
            "Remote subgraph input could not be coerced to the declared primitive socket type."
            f" node_id={node_id!r} node_type={class_type!r}"
            f" input_name={input_name!r} declared_type={declared_type!r}"
            f" received_value={literal_value!r}"
        ) from exc

    return input_value


def _coerce_prompt_primitive_input_values(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
) -> None:
    """Mutate prompt literals in-place to match ComfyUI's primitive widget coercion."""
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        input_type_map = _node_input_type_map(node_class)
        if not input_type_map:
            continue
        inputs = prompt_node.get("inputs") or {}
        for input_name, input_value in list(inputs.items()):
            declared_type = input_type_map.get(str(input_name))
            if declared_type not in _PRIMITIVE_WIDGET_INPUT_TYPES:
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
            ):
                continue
            coerced_value = _coerce_primitive_prompt_input_value(
                node_id=str(node_id),
                class_type=class_type,
                input_name=str(input_name),
                declared_type=declared_type,
                input_value=input_value,
            )
            if coerced_value is not input_value:
                logger.debug(
                    "Coerced remote primitive input %s.%s from %r to %r for type %s.",
                    node_id,
                    input_name,
                    input_value,
                    coerced_value,
                    declared_type,
                )
                inputs[input_name] = coerced_value


def _validate_prompt_input_shapes(
    prompt: dict[str, Any],
    node_mapping: dict[str, type[Any]],
    boundary_input_specs: list[dict[str, Any]] | None = None,
) -> None:
    """Reject prompt inputs that still look invalid for primitive widget sockets."""
    boundary_targets = {
        (str(target.get("node_id")), str(target.get("input_name")))
        for boundary_input in (boundary_input_specs or [])
        for target in boundary_input.get("targets", [])
        if target.get("node_id") is not None and target.get("input_name") is not None
    }
    for node_id, prompt_node in sorted(prompt.items()):
        class_type = str(prompt_node.get("class_type"))
        node_class = node_mapping.get(class_type)
        if node_class is None:
            continue
        input_type_map = _node_input_type_map(node_class)
        if not input_type_map:
            continue
        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            declared_type = input_type_map.get(str(input_name))
            if declared_type not in _PRIMITIVE_WIDGET_INPUT_TYPES:
                continue
            if (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[0], str)
            ):
                continue
            if (str(node_id), str(input_name)) in boundary_targets:
                continue
            literal_value = (
                input_value.get("__value__")
                if isinstance(input_value, dict) and "__value__" in input_value
                else input_value
            )
            if isinstance(literal_value, list):
                raise RemoteSubgraphExecutionError(
                    "Remote subgraph input has an invalid list value for a primitive socket."
                    f" node_id={node_id!r} node_type={class_type!r}"
                    f" input_name={input_name!r} declared_type={declared_type!r}"
                    f" received_value={literal_value!r}"
                )


def _log_prompt_executor_failure_details(
    *,
    component_id: str,
    prompt: dict[str, Any],
    normalized_payload: dict[str, Any],
    executor: Any,
) -> None:
    """Emit high-signal diagnostics for one remote PromptExecutor failure."""
    error_payload = _extract_prompt_executor_error_payload(executor)
    suspicious_inputs = _summarize_suspicious_prompt_inputs(prompt)
    logger.error(
        "Remote PromptExecutor failed for component=%s execute_node_ids=%s boundary_outputs=%s suspicious_inputs=%s error_payload=%s",
        component_id,
        normalized_payload.get("execute_node_ids", []),
        [
            {
                "node_id": boundary_output.get("node_id"),
                "output_index": boundary_output.get("output_index"),
            }
            for boundary_output in normalized_payload.get("boundary_outputs", [])
        ],
        suspicious_inputs,
        error_payload,
    )


def _resolve_required_subgraph_nodes(
    prompt: dict[str, Any],
    execute_node_ids: list[str],
) -> list[str]:
    """Return the dependency closure needed to execute the requested subgraph nodes."""
    required: set[str] = set()
    pending = list(execute_node_ids)
    logger.info("Resolving dependency closure for remote execute targets: %s", execute_node_ids)
    while pending:
        node_id = str(pending.pop())
        if node_id not in prompt:
            logger.warning(
                "Skipping missing remote execute target %s while resolving dependency closure.",
                node_id,
            )
            continue
        if node_id in required:
            continue
        required.add(node_id)
        for input_value in (prompt[node_id].get("inputs") or {}).values():
            if _is_link(input_value):
                pending.append(str(input_value[0]))
    resolved = sorted(required)
    logger.info("Resolved remote dependency closure: %s", resolved)
    return resolved


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _normalize_link_output_index(value: Any) -> Any:
    """Unwrap a singleton list around a prompt-link output index when present."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _unwrap_wrapped_prompt_link(value: Any) -> Any:
    """Collapse nested singleton wrappers around one serialized prompt link when present."""
    candidate = value
    while isinstance(candidate, list) and len(candidate) == 1:
        candidate = candidate[0]
    if _is_link(candidate):
        return [candidate[0], _normalize_link_output_index(candidate[1])]
    return value


def _normalize_prompt_input_value(value: Any, io_type: str | None = None) -> Any:
    """Unwrap transport-added singleton wrappers only for scalar-like prompt input values."""
    wrapped_link = _unwrap_wrapped_prompt_link(value)
    if wrapped_link is not value:
        return wrapped_link
    while (
        isinstance(value, list)
        and len(value) == 1
        and (
            io_type in _PRIMITIVE_WIDGET_INPUT_TYPES
            or value[0] is None
            or isinstance(value[0], bool | int | float | str)
        )
    ):
        value = value[0]
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        return [value[0], _normalize_link_output_index(value[1])]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return value


def _normalize_subgraph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a subgraph payload with canonical prompt-link and output-index shapes."""
    normalized_payload = copy.deepcopy(payload)

    for node_info in normalized_payload.get("subgraph_prompt", {}).values():
        inputs = node_info.get("inputs") or {}
        for input_name, input_value in list(inputs.items()):
            inputs[input_name] = _normalize_prompt_input_value(input_value)

    for boundary_output in normalized_payload.get("boundary_outputs", []):
        if "node_id" in boundary_output and isinstance(boundary_output["node_id"], list):
            boundary_output["node_id"] = _normalize_prompt_input_value(boundary_output["node_id"])
        if "output_index" in boundary_output:
            boundary_output["output_index"] = _normalize_link_output_index(
                boundary_output["output_index"]
            )

    normalized_payload["execute_node_ids"] = [
        _normalize_prompt_input_value(node_id)
        for node_id in normalized_payload.get("execute_node_ids", [])
    ]

    return normalized_payload


def _trim_subgraph_payload_to_required_nodes(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim a subgraph payload down to the dependency closure of its execute targets."""
    trimmed_payload = copy.deepcopy(payload)
    prompt = trimmed_payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return trimmed_payload

    prompt_node_ids = {str(node_id) for node_id in prompt}
    requested_execute_node_ids = [
        str(node_id) for node_id in trimmed_payload.get("execute_node_ids", [])
    ]
    available_execute_node_ids = [
        node_id for node_id in requested_execute_node_ids if node_id in prompt_node_ids
    ]
    dropped_execute_node_ids = [
        node_id for node_id in requested_execute_node_ids if node_id not in prompt_node_ids
    ]
    if dropped_execute_node_ids:
        logger.warning(
            "Dropping remote execute targets absent from subgraph prompt for component=%s: %s",
            payload.get("component_id"),
            dropped_execute_node_ids,
        )

    required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=available_execute_node_ids,
        )
    )
    if not required_node_ids:
        return trimmed_payload

    original_node_ids = list(prompt.keys())
    trimmed_payload["subgraph_prompt"] = {
        str(node_id): prompt[node_id]
        for node_id in original_node_ids
        if str(node_id) in required_node_ids
    }
    trimmed_payload["boundary_inputs"] = [
        {
            **copy.deepcopy(boundary_input),
            "targets": [
                copy.deepcopy(target)
                for target in boundary_input.get("targets", [])
                if str(target.get("node_id")) in required_node_ids
            ],
        }
        for boundary_input in trimmed_payload.get("boundary_inputs", [])
        if any(str(target.get("node_id")) in required_node_ids for target in boundary_input.get("targets", []))
    ]
    trimmed_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in trimmed_payload.get("boundary_outputs", [])
        if str(boundary_output.get("node_id")) in required_node_ids
    ]
    trimmed_payload["component_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("component_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["mapped_execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("mapped_execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    trimmed_payload["static_execute_node_ids"] = [
        str(node_id)
        for node_id in trimmed_payload.get("static_execute_node_ids", [])
        if str(node_id) in required_node_ids
    ]
    logger.info(
        "Trimmed remote subgraph payload %s from %d prompt nodes to %d required nodes.",
        payload.get("component_id"),
        len(original_node_ids),
        len(trimmed_payload["subgraph_prompt"]),
    )
    return trimmed_payload


def _execute_subgraph_prompt(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    normalized_payload = _trim_subgraph_payload_to_required_nodes(
        _normalize_subgraph_payload(payload)
    )
    session_handle = _payload_remote_session_handle(normalized_payload)
    resolution_stats = _RemoteSessionBridgeResolutionStats()
    loader_cache_before = _loader_cache_metric_snapshot()
    resolved_inputs = _resolve_remote_session_inputs(
        dict(hydrated_inputs),
        component_id=component_id,
        target_session_handle=session_handle,
        custom_nodes_root=custom_nodes_root,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
        resolution_stats=resolution_stats,
    )
    _log_remote_session_resolution_summary(
        component_id=component_id,
        resolution_stats=resolution_stats,
        loader_cache_before=loader_cache_before,
        loader_cache_after=_loader_cache_metric_snapshot(),
    )
    short_circuit_outputs = _short_circuit_restored_session_output_subgraph(
        payload=normalized_payload,
        hydrated_inputs=hydrated_inputs,
        session_handle=session_handle,
        resolution_stats=resolution_stats,
    )
    if short_circuit_outputs is not None:
        logger.info(
            "Skipping prompt_executor_execute for component=%s because all %d session-backed outputs were restored into session_id=%s.",
            component_id,
            len(short_circuit_outputs),
            session_handle.session_id if session_handle is not None else None,
        )
        return short_circuit_outputs
    if session_handle is not None:
        logger.info(
            "Executing cloud subgraph component=%s with remote_session session_id=%s prompt_id=%s owner_component_id=%s.",
            component_id,
            session_handle.session_id,
            session_handle.prompt_id,
            session_handle.owner_component_id,
        )
    with _timed_phase("prepare_subgraph_prompt", component=component_id):
        prompt = _rewrite_modal_asset_references(copy.deepcopy(normalized_payload["subgraph_prompt"]))
        _apply_boundary_inputs(
            prompt=prompt,
            boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
            hydrated_inputs=resolved_inputs,
        )
    with _timed_phase("load_execution_module", component=component_id):
        execution = _load_execution_module()
        cache_type, cache_args = _prompt_executor_cache_config(execution)
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    _coerce_prompt_primitive_input_values(prompt, resolved_node_mapping)
    _validate_prompt_input_shapes(
        prompt,
        resolved_node_mapping,
        list(normalized_payload.get("boundary_inputs", [])),
    )

    with (
        _temporary_node_mapping(None),
        _patched_folder_paths_absolute_lookup(),
        _temporary_remote_interrupt_monitor(
            component_id,
            cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key=interrupt_flag_key,
        ),
        _temporary_progress_hook(
            prompt_server := _TracingPromptServer(
                component_id,
                prompt,
                status_callback=status_callback,
            )
        ),
    ):
        with _timed_phase("create_prompt_executor", component=component_id):
            executor_state = _get_or_create_prompt_executor_state(
                execution=execution,
                prompt_server=prompt_server,
                cache_type=cache_type,
                cache_args=cache_args,
                custom_nodes_root=custom_nodes_root,
            )
        cache_store = _node_output_cache_store()
        with executor_state.lock:
            _reset_prompt_executor_request_state(executor_state.executor, prompt_server)
            restore_state: _PersistedNodeCacheRestoreState | None = None
            if cache_store is not None and get_settings().node_output_cache_max_bytes > 0:
                restore_state = (
                    _install_prompt_executor_persisted_cache_restore(
                        execution,
                        executor_state.executor,
                        component_id=component_id,
                        prompt=prompt,
                        cache_store=cache_store,
                    )
                )
            prompt_server.configure_boundary_output_stream(
                boundary_outputs=list(normalized_payload.get("boundary_outputs", [])),
                lookup_cache_entry=lambda node_id: executor_state.executor.caches.outputs.get(node_id),
            )
            try:
                with _timed_phase(
                    "prompt_executor_execute",
                    component=component_id,
                    execute_nodes=list(normalized_payload.get("execute_node_ids", [])),
                ):
                    executor_state.executor.execute(
                        prompt=prompt,
                        prompt_id=str(
                            payload.get("prompt_id")
                            or component_id
                        ),
                        extra_data=copy.deepcopy(normalized_payload.get("extra_data") or {}),
                        execute_outputs=list(normalized_payload.get("execute_node_ids", [])),
                    )
            finally:
                if restore_state is not None:
                    restore_state.restore_original_method()
            executor = executor_state.executor
            restored_node_ids = restore_state.restored_node_ids if restore_state is not None else []
            if restored_node_ids:
                _emit_cloud_info(
                    "Restored %d persisted node cache entries for component=%s: %s",
                    len(restored_node_ids),
                    component_id,
                    restored_node_ids,
                )
        if not executor.success:
            _log_prompt_executor_failure_details(
                component_id=component_id,
                prompt=prompt,
                normalized_payload=normalized_payload,
                executor=executor,
            )
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))

        if cache_store is not None and get_settings().node_output_cache_max_bytes > 0:
            with _timed_phase("persist_node_cache", component=component_id):
                persisted_node_ids = _persist_node_output_cache_entries(
                    executor,
                    prompt=prompt,
                    cache_store=cache_store,
                    restored_cache_keys_by_node_id=(
                        restore_state.restored_cache_keys_by_node_id
                        if restore_state is not None
                        else None
                    ),
                )
            if persisted_node_ids:
                _emit_cloud_info(
                    "Persisted %d node cache entries for component=%s: %s",
                    len(persisted_node_ids),
                    component_id,
                    persisted_node_ids,
                )

        outputs: list[Any] = []
        with _timed_phase(
            "collect_boundary_outputs",
            component=component_id,
            output_count=len(normalized_payload.get("boundary_outputs", [])),
        ):
            for boundary_output in normalized_payload.get("boundary_outputs", []):
                node_id = str(boundary_output["node_id"])
                output_index = int(boundary_output["output_index"])
                cache_entry = executor.caches.outputs.get(node_id)
                if cache_entry is None:
                    raise RemoteSubgraphExecutionError(
                        f"Remote subgraph did not produce cache entry for node {node_id}."
                    )
                if output_index >= len(cache_entry.outputs):
                    raise RemoteSubgraphExecutionError(
                        f"Remote subgraph output index {output_index} is missing for node {node_id}."
                    )
                output_value = _collapse_cache_slot(
                    slot_values=cache_entry.outputs[output_index],
                    is_list=bool(boundary_output.get("is_list", False)),
                )
                if bool(boundary_output.get("session_output")):
                    if session_handle is None:
                        raise RemoteSessionStateError(
                            "Session-backed boundary outputs require payload.remote_session."
                        )
                    live_ref = _REMOTE_SESSION_STORE.put_output(
                        session_handle,
                        node_id=node_id,
                        output_index=output_index,
                        value=output_value,
                    )
                    bridge_record = _build_remote_session_bridge_record(
                        payload=normalized_payload,
                        hydrated_inputs=hydrated_inputs,
                        node_id=node_id,
                        output_index=output_index,
                        io_type=str(boundary_output.get("io_type") or "*"),
                        output_value=output_value,
                    )
                    _store_remote_session_bridge_record(bridge_record)
                    _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
                    output_value = RemoteSessionBridgeRef(
                        bridge_key=bridge_record.bridge_key,
                        node_id=node_id,
                        output_index=output_index,
                        session_id=live_ref.session_id,
                    ).to_payload()
                outputs.append(output_value)
        return tuple(outputs)


def _short_circuit_restored_session_output_subgraph(
    *,
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    session_handle: RemoteSessionHandle | None,
    resolution_stats: _RemoteSessionBridgeResolutionStats,
) -> tuple[Any, ...] | None:
    """Return session-backed outputs directly when bridge restoration already satisfied them all."""
    boundary_outputs = list(payload.get("boundary_outputs", []))
    if session_handle is None or not boundary_outputs:
        return None
    if resolution_stats.input_ref_count <= 0 or resolution_stats.replay_count > 0:
        return None
    if any(not bool(boundary_output.get("session_output")) for boundary_output in boundary_outputs):
        return None

    restored_outputs: list[Any] = []
    for boundary_output in boundary_outputs:
        node_id = str(boundary_output["node_id"])
        output_index = int(boundary_output["output_index"])
        try:
            output_value = _REMOTE_SESSION_STORE.get_output(
                RemoteSessionValueRef(
                    session_id=session_handle.session_id,
                    node_id=node_id,
                    output_index=output_index,
                )
            )
        except RemoteSessionStateError:
            return None

        live_ref = _REMOTE_SESSION_STORE.put_output(
            session_handle,
            node_id=node_id,
            output_index=output_index,
            value=output_value,
        )
        bridge_record = _build_remote_session_bridge_record(
            payload=payload,
            hydrated_inputs=hydrated_inputs,
            node_id=node_id,
            output_index=output_index,
            io_type=str(boundary_output.get("io_type") or "*"),
            output_value=output_value,
        )
        _store_remote_session_bridge_record(bridge_record)
        _store_remote_session_bridge_value(bridge_record.bridge_key, output_value)
        restored_outputs.append(
            RemoteSessionBridgeRef(
                bridge_key=bridge_record.bridge_key,
                node_id=node_id,
                output_index=output_index,
                session_id=live_ref.session_id,
            ).to_payload()
        )
    return tuple(restored_outputs)


def execute_subgraph_locally(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    session_handle = _payload_remote_session_handle(payload)
    with _timed_phase("execute_subgraph_locally", component=component_id):
        custom_nodes_root = _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
        _ensure_comfy_runtime_initialized(custom_nodes_root)
        with _timed_phase("deserialize_boundary_inputs", component=component_id):
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        logger.info(
            "Executing cloud-local subgraph component=%s hydrated_inputs=%d session_id=%s clear_remote_session=%s.",
            component_id,
            len(hydrated_inputs),
            session_handle.session_id if session_handle is not None else None,
            bool(payload.get("clear_remote_session")),
        )
        try:
            with _timed_phase("subgraph_worker_roundtrip", component=component_id):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _execute_subgraph_prompt,
                        payload,
                        hydrated_inputs,
                        custom_nodes_root,
                        status_callback,
                        cancellation_event,
                        interrupt_store,
                        interrupt_flag_key,
                    )
                    outputs = future.result()
        finally:
            if bool(payload.get("clear_remote_session")) and session_handle is not None:
                logger.info(
                    "Clearing remote session after cloud component=%s session_id=%s.",
                    component_id,
                    session_handle.session_id,
                )
                _REMOTE_SESSION_STORE.clear_session(session_handle)
        with _timed_phase("serialize_boundary_outputs", component=component_id):
            return serialize_node_outputs(outputs)


def _mapped_phase_definition(payload: dict[str, Any], phase_key: str) -> dict[str, Any] | None:
    """Return one explicit mapped phase definition when queue-time planning provided it."""
    phase_payload = payload.get(phase_key)
    if isinstance(phase_payload, dict):
        return phase_payload
    return None


def _shared_subgraph_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload fields shared by every explicit mapped phase."""
    shared_fields = {
        "prompt_id": payload.get("prompt_id"),
        "extra_data": copy.deepcopy(payload.get("extra_data") or {}),
        "requires_volume_reload": bool(payload.get("requires_volume_reload", True)),
        "volume_reload_marker": payload.get("volume_reload_marker"),
        "uploaded_volume_paths": list(payload.get("uploaded_volume_paths", [])),
        "terminate_container_on_error": bool(payload.get("terminate_container_on_error", True)),
        "custom_nodes_bundle": payload.get("custom_nodes_bundle"),
    }
    remote_session = payload.get("remote_session")
    if remote_session is not None:
        shared_fields["remote_session"] = copy.deepcopy(remote_session)
    if bool(payload.get("clear_remote_session")):
        shared_fields["clear_remote_session"] = True
    return shared_fields


def _build_phase_subgraph_payload(
    payload: dict[str, Any],
    phase_key: str,
    component_id: str,
) -> dict[str, Any]:
    """Return one explicit static or mapped subgraph payload."""
    phase_definition = _mapped_phase_definition(payload, phase_key)
    if phase_definition is None:
        raise KeyError(f"Mapped payload is missing phase definition {phase_key!r}.")

    return {
        "payload_kind": "subgraph",
        "component_id": component_id,
        **_shared_subgraph_payload_fields(payload),
        "component_node_ids": [
            str(node_id)
            for node_id in phase_definition.get("component_node_ids", [])
            if str(node_id)
        ],
        "subgraph_prompt": copy.deepcopy(phase_definition.get("subgraph_prompt", {})),
        "boundary_inputs": copy.deepcopy(phase_definition.get("boundary_inputs", [])),
        "boundary_outputs": copy.deepcopy(phase_definition.get("boundary_outputs", [])),
        "execute_node_ids": [
            str(node_id)
            for node_id in phase_definition.get("execute_node_ids", [])
            if str(node_id)
        ],
    }


def _split_phase_outputs(
    phase_outputs: tuple[Any, ...],
    boundary_outputs: list[dict[str, Any]],
    internal_output_names: set[str],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Split one phase result tuple into bridge values and external outputs."""
    internal_outputs: dict[str, Any] = {}
    external_outputs: list[Any] = []
    for boundary_output, output_value in zip(boundary_outputs, phase_outputs, strict=True):
        output_name = str(boundary_output.get("proxy_output_name") or "")
        if output_name in internal_output_names:
            internal_outputs[output_name] = output_value
            continue
        external_outputs.append(output_value)
    return internal_outputs, tuple(external_outputs)


def _aggregate_mapped_phase_outputs(
    per_item_outputs: list[tuple[Any, ...]],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Join ordered mapped-phase outputs back into one proxy result tuple."""
    if not per_item_outputs:
        raise ValueError("Mapped execution produced no per-item outputs to aggregate.")

    output_count = len(per_item_outputs[0])
    if any(len(item_outputs) != output_count for item_outputs in per_item_outputs):
        raise RemoteSubgraphExecutionError("Mapped remote execution produced inconsistent output arity.")

    aggregated_outputs: list[Any] = []
    boundary_outputs = list(payload.get("boundary_outputs", []))
    for output_index in range(output_count):
        boundary_output = boundary_outputs[output_index] if output_index < len(boundary_outputs) else {}
        aggregated_outputs.append(
            _merge_static_or_mapped_values(
                [item_outputs[output_index] for item_outputs in per_item_outputs],
                io_type=str(boundary_output.get("io_type", "*")),
                is_list=bool(boundary_output.get("is_list", False)),
            )
        )
    return tuple(aggregated_outputs)


def _merge_static_or_mapped_values(
    values: list[Any],
    *,
    io_type: str,
    is_list: bool,
) -> Any:
    """Join mapped per-item outputs using the shared transport serializer rules."""
    from serialization import join_mapped_values

    return join_mapped_values(values, io_type=io_type, is_list=is_list)


def _merge_static_and_mapped_outputs(
    *,
    static_outputs: tuple[Any, ...],
    mapped_outputs: tuple[Any, ...],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble one mapped component's static and mapped outputs in proxy order."""
    combined_outputs: list[Any] = []
    static_output_index = 0
    mapped_output_index = 0
    for boundary_output in payload.get("boundary_outputs", []):
        if bool(boundary_output.get("mapped_output")):
            if mapped_output_index >= len(mapped_outputs):
                raise RemoteSubgraphExecutionError(
                    "Mapped remote execution returned fewer mapped outputs than expected."
                )
            combined_outputs.append(mapped_outputs[mapped_output_index])
            mapped_output_index += 1
            continue
        if static_output_index >= len(static_outputs):
            raise RemoteSubgraphExecutionError(
                "Mapped remote execution returned fewer static outputs than expected."
            )
        combined_outputs.append(static_outputs[static_output_index])
        static_output_index += 1

    if static_output_index != len(static_outputs) or mapped_output_index != len(mapped_outputs):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced extra outputs that did not match the declared boundary outputs."
        )
    return tuple(combined_outputs)


def _execute_mapped_subgraph_payload(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute one mapped payload inside a single remote runtime process."""
    mapped_input = payload.get("mapped_input") or {}
    mapped_input_name = str(mapped_input.get("proxy_input_name") or "")
    if not mapped_input_name:
        raise RemoteSubgraphExecutionError("Mapped remote payloads must define mapped_input.proxy_input_name.")
    if mapped_input_name not in hydrated_inputs:
        raise KeyError(f"Mapped remote payload input {mapped_input_name!r} was not provided.")

    from serialization import split_mapped_value

    mapped_items = split_mapped_value(
        hydrated_inputs[mapped_input_name],
        str(mapped_input.get("io_type", "*")),
    )
    if not mapped_items:
        raise ValueError("Mapped remote execution requires at least one input item.")

    broadcast_inputs = dict(hydrated_inputs)
    broadcast_inputs.pop(mapped_input_name, None)
    static_to_mapped_boundaries = list(payload.get("static_to_mapped_boundaries", []))
    bridge_output_names = {
        str(boundary_spec.get("proxy_name") or "")
        for boundary_spec in static_to_mapped_boundaries
        if str(boundary_spec.get("proxy_name") or "")
    }

    static_outputs: tuple[Any, ...] = ()
    if payload.get("static_phase") is not None:
        static_phase_payload = _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
        )
        if static_phase_payload.get("execute_node_ids"):
            static_phase_outputs = _execute_subgraph_prompt(
                static_phase_payload,
                dict(broadcast_inputs),
                custom_nodes_root,
                status_callback=status_callback,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            )
            bridge_inputs, static_outputs = _split_phase_outputs(
                static_phase_outputs,
                list(static_phase_payload.get("boundary_outputs", [])),
                bridge_output_names,
            )
            broadcast_inputs.update(bridge_inputs)

    if status_callback is not None:
        status_callback(
            {
                "event_type": "node_progress",
                "node_id": str(payload.get("component_id") or "modal-subgraph"),
                "display_node_id": str(payload.get("component_id") or "modal-subgraph"),
                "value": 0.0,
                "max": float(len(mapped_items)),
                "aggregate_only": True,
            }
        )

    per_item_outputs: list[tuple[Any, ...]] = []
    for item_index, item_value in enumerate(mapped_items):
        last_lane_node_id: str | None = None
        lane_id = str(payload.get("mapped_progress_lane_id") or item_index)

        def publish_item_status(progress_state: dict[str, Any]) -> None:
            """Attach mapped-lane metadata to one per-item progress event."""
            nonlocal last_lane_node_id
            if status_callback is None:
                return
            event_type = str(progress_state.get("event_type", ""))
            if event_type == "node_progress":
                reported_node_id = progress_state.get("real_node_id") or progress_state.get("node_id")
                if reported_node_id is not None:
                    last_lane_node_id = str(reported_node_id)
                status_callback(
                    {
                        **progress_state,
                        "lane_id": lane_id,
                        "item_index": item_index,
                    }
                )
                return
            if event_type in {"executed", "preview", "boundary_output"}:
                status_callback({**progress_state, "item_index": item_index})

        item_payload = _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
        )
        item_inputs = dict(broadcast_inputs)
        item_inputs[mapped_input_name] = item_value
        per_item_outputs.append(
            _execute_subgraph_prompt(
                item_payload,
                item_inputs,
                custom_nodes_root,
                status_callback=publish_item_status,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            )
        )
        if status_callback is not None:
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": last_lane_node_id or str(payload.get("component_id") or "modal-subgraph"),
                    "display_node_id": last_lane_node_id or str(payload.get("component_id") or "modal-subgraph"),
                    "value": 0.0,
                    "max": 1.0,
                    "lane_id": lane_id,
                    "item_index": item_index,
                    "clear": True,
                }
            )
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": str(payload.get("component_id") or "modal-subgraph"),
                    "display_node_id": str(payload.get("component_id") or "modal-subgraph"),
                    "value": float(item_index + 1),
                    "max": float(len(mapped_items)),
                    "aggregate_only": True,
                }
            )

    mapped_phase_payload = _build_phase_subgraph_payload(
        payload,
        "mapped_phase",
        f"{payload.get('component_id', 'modal-subgraph')}::mapped",
    )
    mapped_outputs = _aggregate_mapped_phase_outputs(
        per_item_outputs,
        {"boundary_outputs": list(mapped_phase_payload.get("boundary_outputs", []))},
    )
    return _merge_static_and_mapped_outputs(
        static_outputs=static_outputs,
        mapped_outputs=mapped_outputs,
        payload=payload,
    )


def _stream_remote_payload_events(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield progress and result events for one remote payload execution."""
    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    task_id = os.getenv("MODAL_TASK_ID")

    def publish_status(progress_state: dict[str, Any]) -> None:
        """Queue a progress envelope for the remote caller."""
        event_queue.put(("progress", serialize_mapping(progress_state)))

    def execute_payload() -> None:
        """Run the payload in a worker thread and enqueue the terminal outcome."""
        try:
            if payload.get("payload_kind") == "mapped_subgraph":
                custom_nodes_root = _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
                _ensure_comfy_runtime_initialized(custom_nodes_root)
                hydrated_inputs = deserialize_node_inputs(kwargs_payload)
                outputs = serialize_node_outputs(
                    _execute_mapped_subgraph_payload(
                        payload,
                        hydrated_inputs,
                        custom_nodes_root,
                        status_callback=publish_status,
                        cancellation_event=cancellation_event,
                        interrupt_store=interrupt_store,
                        interrupt_flag_key=interrupt_flag_key,
                    )
                )
            elif payload.get("payload_kind") == "subgraph":
                execute_subgraph_kwargs: dict[str, Any] = {"status_callback": publish_status}
                if "cancellation_event" in inspect.signature(execute_subgraph_locally).parameters:
                    execute_subgraph_kwargs["cancellation_event"] = cancellation_event
                if "interrupt_store" in inspect.signature(execute_subgraph_locally).parameters:
                    execute_subgraph_kwargs["interrupt_store"] = interrupt_store
                if "interrupt_flag_key" in inspect.signature(execute_subgraph_locally).parameters:
                    execute_subgraph_kwargs["interrupt_flag_key"] = interrupt_flag_key
                outputs = execute_subgraph_locally(
                    payload,
                    kwargs_payload,
                    **execute_subgraph_kwargs,
                )
            else:
                execute_node_kwargs: dict[str, Any] = {}
                if "cancellation_event" in inspect.signature(execute_node_locally).parameters:
                    execute_node_kwargs["cancellation_event"] = cancellation_event
                if "interrupt_store" in inspect.signature(execute_node_locally).parameters:
                    execute_node_kwargs["interrupt_store"] = interrupt_store
                if "interrupt_flag_key" in inspect.signature(execute_node_locally).parameters:
                    execute_node_kwargs["interrupt_flag_key"] = interrupt_flag_key
                outputs = execute_node_locally(payload, kwargs_payload, **execute_node_kwargs)
        except Exception as exc:  # pragma: no cover - exercised through generator consumer tests.
            event_queue.put(("error", exc))
        else:
            event_queue.put(("result", outputs))
        finally:
            event_queue.put(("done", None))

    worker_thread = threading.Thread(
        target=execute_payload,
        name=f"modal-stream-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    if task_id:
        yield {"kind": "remote_logs", "task_id": task_id}
    worker_thread.start()
    try:
        while True:
            event_kind, event_payload = event_queue.get()
            if event_kind == "progress":
                yield {"kind": "progress", **event_payload}
                continue
            if event_kind == "result":
                yield {"kind": "result", "outputs": coerce_serialized_node_outputs(event_payload)}
                continue
            if event_kind == "error":
                raise event_payload
            if event_kind == "done":
                return
    finally:
        worker_thread.join(timeout=1.0)


def _should_ignore_repo_path(path: Path) -> bool:
    """Return whether a local repo path should be omitted from the Modal image mount."""
    parts = set(path.parts)
    if {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} & parts:
        return True
    return path.suffix.lower() in {".log", ".pyc", ".pyo", ".swp", ".tmp"}


def _should_ignore_comfyui_path(path: Path) -> bool:
    """Return whether a local ComfyUI path should be omitted from the Modal image mount."""
    parts = path.parts
    if not parts:
        return False

    if {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} & set(parts):
        return True

    if parts[0] in {"custom_nodes", "input", "models", "output", "temp", "user"}:
        return True

    return path.suffix.lower() in {".bin", ".ckpt", ".log", ".pt", ".pyc", ".pyo", ".safetensors", ".swp", ".tmp"}


def _comfyui_runtime_packages() -> tuple[str, ...]:
    """Return the Python packages needed to import and execute ComfyUI core inside Modal."""
    return (
        "aiohttp",
        "alembic",
        "av",
        "comfy-kitchen>=0.2.7",
        "einops",
        "kornia",
        "numpy",
        "opencv-python-headless",
        "packaging",
        "pillow",
        "psutil",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "requests",
        "safetensors",
        "scipy",
        "sentencepiece",
        "spandrel",
        "sqlalchemy",
        "torchsde",
        "tqdm",
        "transformers",
    )


def _comfyui_torch_packages() -> tuple[str, ...]:
    """Return the pinned CUDA 12.8 PyTorch stack used by the remote Modal image."""
    return (
        f"torch=={_COMFYUI_TORCH_VERSION}",
        f"torchvision=={_COMFYUI_TORCHVISION_VERSION}",
        f"torchaudio=={_COMFYUI_TORCHAUDIO_VERSION}",
    )


def _prewarm_snapshot_state(settings: Any, snapshot_profile_key: str = "") -> None:
    """Run snapshot-safe initialization before Modal captures a memory snapshot."""
    with _timed_phase(
        "prewarm_snapshot_state",
        gpu_snapshot=settings.enable_gpu_memory_snapshot,
        snapshot_profile=snapshot_profile_key or None,
    ):
        _ensure_comfyui_support_packages()
        if settings.enable_gpu_memory_snapshot:
            _ensure_comfy_runtime_initialized(None)
            _load_execution_module()
            loader_prewarm_plans = _load_loader_snapshot_profile(snapshot_profile_key)
            if loader_prewarm_plans:
                _execute_loader_prewarm_plans(
                    component_id=f"snapshot-profile:{snapshot_profile_key}",
                    loader_prewarm_plans=loader_prewarm_plans,
                    custom_nodes_root=None,
                )
            _emit_cloud_info("Completed GPU-snapshot ComfyUI prewarm before snapshot capture.")
            return

        _emit_cloud_info(
            "Skipping full ComfyUI runtime prewarm during CPU-only snapshot to avoid accidental CUDA initialization."
        )


def _prewarm_restored_runtime() -> None:
    """Run post-restore initialization that should be ready before serving requests."""
    with _timed_phase("prewarm_restored_runtime"):
        _ensure_comfy_runtime_initialized(None)
        _load_execution_module()


def _remote_engine_cls_options(settings: Any, vol: Any, image: Any) -> dict[str, Any]:
    """Build the Modal class options for the deployed remote execution runtime."""
    options: dict[str, Any] = {
        "gpu": settings.modal_gpu,
        "volumes": {settings.remote_storage_root: vol},
        "scaledown_window": settings.scaledown_window_seconds,
        "min_containers": settings.min_containers,
        "image": image,
        "enable_memory_snapshot": settings.enable_memory_snapshot,
    }
    max_containers = getattr(settings, "max_containers", None)
    buffer_containers = getattr(settings, "buffer_containers", None)
    if max_containers is not None:
        options["max_containers"] = max_containers
    if buffer_containers is not None:
        options["buffer_containers"] = buffer_containers
    if settings.enable_gpu_memory_snapshot:
        options["experimental_options"] = {"enable_gpu_snapshot": True}
    return options


def _should_reload_modal_volume(payload: dict[str, Any]) -> bool:
    """Return whether this request needs the mounted Modal volume reloaded."""
    if not bool(payload.get("requires_volume_reload", True)):
        return False
    if _payload_uploaded_volume_paths_visible(payload):
        reload_marker = _modal_volume_reload_marker(payload)
        if reload_marker is not None:
            _record_modal_volume_reload_marker(reload_marker)
        return False
    reload_marker = _modal_volume_reload_marker(payload)
    if reload_marker is None:
        return True
    return not _has_seen_modal_volume_reload_marker(reload_marker)


def _modal_volume_reload_marker(payload: dict[str, Any]) -> str | None:
    """Return the per-request Modal volume reload marker attached to this payload."""
    marker = payload.get("volume_reload_marker")
    if marker is None:
        return None
    marker_text = str(marker).strip()
    return marker_text or None


def _has_seen_modal_volume_reload_marker(reload_marker: str) -> bool:
    """Return whether this container already reloaded the volume for this marker."""
    with _MODAL_VOLUME_RELOAD_MARKERS_LOCK:
        return reload_marker in _MODAL_VOLUME_RELOAD_MARKER_SET


def _record_modal_volume_reload_marker(reload_marker: str) -> None:
    """Remember that this container has already reloaded the volume for one marker."""
    global _MODAL_VOLUME_RELOAD_MARKERS

    with _MODAL_VOLUME_RELOAD_MARKERS_LOCK:
        if reload_marker in _MODAL_VOLUME_RELOAD_MARKER_SET:
            return
        if _MODAL_VOLUME_RELOAD_MARKERS is None:
            _MODAL_VOLUME_RELOAD_MARKERS = queue.SimpleQueue()
        _MODAL_VOLUME_RELOAD_MARKER_SET.add(reload_marker)
        _MODAL_VOLUME_RELOAD_MARKERS.put(reload_marker)
        while len(_MODAL_VOLUME_RELOAD_MARKER_SET) > _MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT:
            expired_marker = _MODAL_VOLUME_RELOAD_MARKERS.get()
            _MODAL_VOLUME_RELOAD_MARKER_SET.discard(expired_marker)


def _clear_warm_remote_caches() -> None:
    """Drop warm-container caches that may retain references to mounted volume files."""
    with _PROMPT_EXECUTOR_STATES_LOCK:
        _PROMPT_EXECUTOR_STATES.clear()
    with _LOADER_CACHE_LOCK:
        _LOADER_OUTPUT_CACHE.clear()


def _prepare_for_modal_volume_reload() -> None:
    """Release warm runtime state so a Modal volume reload can proceed safely."""
    _clear_warm_remote_caches()
    try:
        import comfy.model_management as model_management
    except ModuleNotFoundError:
        gc.collect()
        return

    model_management.unload_all_models()
    model_management.cleanup_models()
    model_management.soft_empty_cache(True)
    gc.collect()


def _is_modal_volume_open_files_error(exc: RuntimeError) -> bool:
    """Return whether a Modal volume reload failed because mounted files are still open."""
    return "open files" in str(exc)


def _sleep_before_modal_volume_reload_retry(delay_seconds: float) -> None:
    """Pause briefly so recently cancelled work can release mounted-volume file handles."""
    if delay_seconds <= 0:
        return
    time.sleep(delay_seconds)


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


def _payload_volume_paths(payload: dict[str, Any]) -> set[Path]:
    """Return mounted-volume paths referenced by this remote payload."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    referenced_paths: set[Path] = set()

    custom_nodes_bundle = payload.get("custom_nodes_bundle")
    if isinstance(custom_nodes_bundle, str):
        bundle_path = Path(_materialize_remote_asset_path(custom_nodes_bundle))
        if bundle_path.is_absolute() and bundle_path.resolve().is_relative_to(remote_storage_root):
            referenced_paths.add(bundle_path)

    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return referenced_paths

    for prompt_node in prompt.values():
        if not isinstance(prompt_node, dict):
            continue
        inputs = prompt_node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for input_value in inputs.values():
            for candidate_path in _iter_payload_input_strings(input_value):
                materialized_path = _materialize_remote_asset_path(candidate_path)
                materialized_path_obj = Path(materialized_path)
                if (
                    materialized_path_obj.is_absolute()
                    and materialized_path_obj.resolve().is_relative_to(remote_storage_root)
                ):
                    referenced_paths.add(materialized_path_obj)
    return referenced_paths


def _payload_uploaded_volume_paths(payload: dict[str, Any]) -> set[Path]:
    """Return newly uploaded mounted-volume paths relevant to this payload."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    uploaded_paths: set[Path] = set()
    for candidate_path in payload.get("uploaded_volume_paths", []):
        if isinstance(candidate_path, str) and candidate_path.strip():
            materialized_path = Path(_materialize_remote_asset_path(candidate_path))
            if (
                materialized_path.is_absolute()
                and materialized_path.resolve().is_relative_to(remote_storage_root)
            ):
                uploaded_paths.add(materialized_path)
    return uploaded_paths


def _payload_uploaded_volume_paths_visible(payload: dict[str, Any]) -> bool:
    """Return whether every newly uploaded mounted-volume path is already visible."""
    uploaded_paths = _payload_uploaded_volume_paths(payload)
    if not uploaded_paths:
        return False
    return all(path.exists() for path in uploaded_paths)


def _payload_volume_paths_visible(payload: dict[str, Any]) -> bool:
    """Return whether every mounted-volume path referenced by this payload is already visible."""
    referenced_paths = _payload_volume_paths(payload)
    if not referenced_paths:
        return False
    return all(path.exists() for path in referenced_paths)


def _log_payload_volume_reload_diagnostics(
    component_id: str,
    payload: dict[str, Any] | None,
    *,
    context: str,
) -> None:
    """Log the mounted-volume paths relevant to one reload decision or failure."""
    if payload is None:
        return

    uploaded_paths = sorted(str(path) for path in _payload_uploaded_volume_paths(payload))
    referenced_paths = sorted(str(path) for path in _payload_volume_paths(payload))
    logger.info(
        "Modal volume reload diagnostics for component=%s context=%s uploaded_paths=%s referenced_paths=%s visible_uploaded=%s visible_referenced=%s.",
        component_id,
        context,
        uploaded_paths,
        referenced_paths,
        _payload_uploaded_volume_paths_visible(payload),
        _payload_volume_paths_visible(payload),
    )


def _reload_modal_volume_for_request(
    volume: Any,
    component_id: str,
    reload_marker: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Reload the Modal volume, retrying briefly while warm state releases open files."""
    with _timed_phase("modal_volume_reload", component=component_id):
        diagnostics_logged = False
        for attempt_index, retry_delay_seconds in enumerate(
            _MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS,
            start=1,
        ):
            if attempt_index > 1:
                _sleep_before_modal_volume_reload_retry(retry_delay_seconds)
            try:
                volume.reload()
                if reload_marker is not None:
                    _record_modal_volume_reload_marker(reload_marker)
                if attempt_index > 1:
                    _emit_cloud_info(
                        "Modal volume reload succeeded for component=%s after %d attempt(s).",
                        component_id,
                        attempt_index,
                    )
                return
            except RuntimeError as exc:
                if not _is_modal_volume_open_files_error(exc):
                    raise
                if payload is not None and not diagnostics_logged:
                    _log_payload_volume_reload_diagnostics(
                        component_id,
                        payload,
                        context="open_files_retry",
                    )
                    diagnostics_logged = True
                if attempt_index == len(_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS):
                    if payload is not None and _payload_volume_paths_visible(payload):
                        _emit_cloud_info(
                            "Modal volume reload still reported open files for component=%s after %d attempt(s), "
                            "but all referenced mounted-volume paths are already visible. Proceeding without reload.",
                            component_id,
                            attempt_index,
                        )
                        if reload_marker is not None:
                            _record_modal_volume_reload_marker(reload_marker)
                        return
                    raise
                _emit_cloud_info(
                    "Modal volume reload hit open files for component=%s on attempt %d/%d; clearing warm caches and retrying after %.2fs.",
                    component_id,
                    attempt_index,
                    len(_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS),
                    _MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS[attempt_index],
                )
                _prepare_for_modal_volume_reload()


def _emit_modal_volume_reload_skip(component_id: Any, payload: dict[str, Any]) -> None:
    """Log why a request did not need a Modal volume reload."""
    if _payload_uploaded_volume_paths_visible(payload):
        _emit_cloud_info(
            "Skipping modal_volume_reload for component=%s because all uploaded mounted-volume paths are already visible in this container.",
            component_id,
        )
        _log_payload_volume_reload_diagnostics(
            str(component_id),
            payload,
            context="skip_visible_uploaded_paths",
        )
        return
    reload_marker = _modal_volume_reload_marker(payload)
    if reload_marker is not None and _has_seen_modal_volume_reload_marker(reload_marker):
        _emit_cloud_info(
            "Skipping modal_volume_reload for component=%s because this container already reloaded marker=%s.",
            component_id,
            reload_marker,
        )
        _log_payload_volume_reload_diagnostics(
            str(component_id),
            payload,
            context="skip_reload_marker_seen",
        )
        return
    _emit_cloud_info(
        "Skipping modal_volume_reload for component=%s because no new assets were uploaded for this request.",
        component_id,
    )
    _log_payload_volume_reload_diagnostics(
        str(component_id),
        payload,
        context="skip_no_new_assets",
    )


def _prepare_warm_container_for_request(volume: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Prime one RemoteEngine container for a request before the first real execution payload arrives."""
    component_id = str(payload.get("component_id") or "modal-warmup")
    reload_marker = _modal_volume_reload_marker(payload)
    needs_volume_reload = _should_reload_modal_volume(payload)
    with _timed_phase("remote_engine_warmup", component=component_id):
        if needs_volume_reload:
            _reload_modal_volume_for_request(
                volume,
                component_id,
                reload_marker=reload_marker,
                payload=payload,
            )
        else:
            _emit_modal_volume_reload_skip(component_id, payload)
        custom_nodes_bundle = payload.get("custom_nodes_bundle")
        custom_nodes_root: Path | None = None
        if isinstance(custom_nodes_bundle, str) and custom_nodes_bundle.strip():
            custom_nodes_root = _extract_custom_nodes_bundle(custom_nodes_bundle)
            if custom_nodes_root is not None:
                _register_custom_nodes_root(custom_nodes_root)
        loader_prewarm_plans = payload.get("loader_prewarm_plans")
        if isinstance(loader_prewarm_plans, list) and loader_prewarm_plans:
            _execute_loader_prewarm_plans(
                component_id=component_id,
                loader_prewarm_plans=loader_prewarm_plans,
                custom_nodes_root=custom_nodes_root,
            )
        return {
            "component_id": component_id,
            "task_id": os.getenv("MODAL_TASK_ID"),
            "warmup_slot_index": (
                int(payload["warmup_slot_index"])
                if payload.get("warmup_slot_index") is not None
                else None
            ),
            "reloaded_volume": needs_volume_reload,
        }


def _loader_prewarm_plan_key(plan: Mapping[str, Any]) -> str | None:
    """Return the stable worker-local dedupe key for one loader prewarm plan."""
    signature = plan.get("signature")
    if signature is None:
        return None
    normalized_signature = str(signature).strip()
    return normalized_signature or None


def _build_loader_prewarm_payload(
    *,
    component_id: str,
    plan_index: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one synthetic single-node subgraph payload for loader warmup."""
    plan_node_id = str(plan.get("node_id") or f"loader-{plan_index}")
    prompt_id = plan.get("prompt_id")
    return {
        "payload_kind": "subgraph",
        "component_id": f"{component_id}::loader-prewarm:{plan_node_id}",
        "prompt_id": (str(prompt_id) if prompt_id is not None else None),
        "component_node_ids": [plan_node_id],
        "subgraph_prompt": copy.deepcopy(dict(plan["subgraph_prompt"])),
        "boundary_inputs": [],
        "boundary_outputs": [],
        "execute_node_ids": list(plan.get("execute_node_ids") or [plan_node_id]),
        "extra_data": {},
    }


def _execute_loader_prewarm_plans(
    *,
    component_id: str,
    loader_prewarm_plans: list[dict[str, Any]],
    custom_nodes_root: Path | None,
) -> None:
    """Execute synthetic one-node loader workflows so fresh workers preload heavyweight models."""
    if not get_settings().enable_loader_prewarm:
        return

    _ensure_comfy_runtime_initialized(custom_nodes_root)
    executed_plan_count = 0
    skipped_plan_count = 0
    for plan_index, plan in enumerate(loader_prewarm_plans):
        if not isinstance(plan, Mapping):
            continue
        plan_key = _loader_prewarm_plan_key(plan)
        if plan_key is not None:
            with _LOADER_PREWARM_PLAN_KEYS_LOCK:
                if plan_key in _LOADER_PREWARM_PLAN_KEYS:
                    skipped_plan_count += 1
                    continue
                _LOADER_PREWARM_PLAN_KEYS.add(plan_key)
        try:
            _execute_subgraph_prompt(
                _build_loader_prewarm_payload(
                    component_id=component_id,
                    plan_index=plan_index,
                    plan=plan,
                ),
                hydrated_inputs={},
                custom_nodes_root=custom_nodes_root,
            )
            executed_plan_count += 1
        except Exception:
            if plan_key is not None:
                with _LOADER_PREWARM_PLAN_KEYS_LOCK:
                    _LOADER_PREWARM_PLAN_KEYS.discard(plan_key)
            raise
    if executed_plan_count or skipped_plan_count:
        logger.info(
            "Warm container loader prewarm finished for component=%s executed=%d skipped=%d.",
            component_id,
            executed_plan_count,
            skipped_plan_count,
        )


if modal is not None:  # pragma: no branch - remote entrypoint configuration.
    settings = get_settings()
    app = modal.App(settings.app_name)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    interrupt_flags = modal.Dict.from_name(
        settings.interrupt_dict_name,
        create_if_missing=True,
    )
    node_output_cache = modal.Dict.from_name(
        settings.node_output_cache_dict_name,
        create_if_missing=True,
    )
    session_bridge_cache = modal.Dict.from_name(
        settings.session_bridge_dict_name,
        create_if_missing=True,
    )
    snapshot_profiles = modal.Dict.from_name(
        settings.snapshot_profile_dict_name,
        create_if_missing=True,
    )
    image = (
        modal.Image.debian_slim()
        .pip_install(*_comfyui_runtime_packages())
        .pip_install(*_comfyui_torch_packages(), index_url=_PYTORCH_CUDA_INDEX_URL)
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/comfyui_modal_sync_repo",
            ignore=_should_ignore_repo_path,
        )
    )
    if settings.comfyui_root is not None and settings.comfyui_root.exists():
        image = image.add_local_dir(
            settings.comfyui_root,
            remote_path=str(_REMOTE_COMFYUI_ROOT),
            ignore=_should_ignore_comfyui_path,
        )
        logger.info(
            "Including local ComfyUI checkout %s in Modal image at %s.",
            settings.comfyui_root,
            _REMOTE_COMFYUI_ROOT,
        )
    else:
        logger.warning(
            "No local ComfyUI checkout was discovered; remote Modal execution may fail to import ComfyUI core modules."
        )

    @app.cls(**_remote_engine_cls_options(settings, vol, image))
    @modal.concurrent(max_inputs=1)
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""
        worker_affinity_key: str = modal.parameter(default="")
        snapshot_profile_key: str = modal.parameter(default="")

        @modal.enter(snap=True)
        def setup_snapshot_state(self) -> None:
            """Prepare snapshot-friendly runtime state before Modal captures memory."""
            with _timed_phase("remote_engine_setup_snapshot"):
                _prewarm_snapshot_state(settings, self.snapshot_profile_key)
                logger.info(
                    "RemoteEngine snapshot setup complete for worker_affinity_key=%s snapshot_profile_key=%s.",
                    self.worker_affinity_key or None,
                    self.snapshot_profile_key or None,
                )

        @modal.enter(snap=False)
        def setup_restored_runtime(self) -> None:
            """Prepare request-serving runtime state after a fresh boot or snapshot restore."""
            with _timed_phase("remote_engine_setup_restored"):
                _prewarm_restored_runtime()
                logger.info(
                    "RemoteEngine restored-runtime setup complete for worker_affinity_key=%s snapshot_profile_key=%s.",
                    self.worker_affinity_key or None,
                    self.snapshot_profile_key or None,
                )

        @modal.method()
        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            try:
                with _registered_remote_execution(payload) as execution_control:
                    with _timed_phase(
                        "remote_engine_execute_payload",
                        component=component_id,
                        payload_kind=payload.get("payload_kind"),
                    ):
                        if _should_reload_modal_volume(payload):
                            _reload_modal_volume_for_request(
                                vol,
                                str(component_id),
                                reload_marker=reload_marker,
                                payload=payload,
                            )
                        else:
                            _emit_modal_volume_reload_skip(component_id, payload)
                        if payload.get("payload_kind") == "mapped_subgraph":
                            custom_nodes_root = _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
                            _ensure_comfy_runtime_initialized(custom_nodes_root)
                            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
                            return serialize_node_outputs(
                                _execute_mapped_subgraph_payload(
                                    payload,
                                    hydrated_inputs,
                                    custom_nodes_root,
                                    cancellation_event=execution_control.cancellation_event,
                                    interrupt_store=interrupt_flags,
                                    interrupt_flag_key=execution_control.interrupt_flag_key,
                                )
                            )
                        if payload.get("payload_kind") == "subgraph":
                            return execute_subgraph_locally(
                                payload,
                                kwargs_payload,
                                cancellation_event=execution_control.cancellation_event,
                                interrupt_store=interrupt_flags,
                                interrupt_flag_key=execution_control.interrupt_flag_key,
                            )
                        return execute_node_locally(
                            payload,
                            kwargs_payload,
                            cancellation_event=execution_control.cancellation_event,
                            interrupt_store=interrupt_flags,
                            interrupt_flag_key=execution_control.interrupt_flag_key,
                        )
            except Exception as exc:
                _maybe_schedule_container_termination_on_error(payload, exc)
                raise

        @modal.method()
        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Prime the current or a newly started Modal container for one prompt."""
            return _prepare_warm_container_for_request(vol, payload)

        @modal.method()
        def execute_payload_stream(
            self,
            payload: dict[str, Any],
            kwargs_payload: bytes,
        ) -> Iterator[dict[str, Any]]:
            """Stream progress envelopes and a final serialized result for one payload."""
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            try:
                with _registered_remote_execution(payload) as execution_control:
                    with _timed_phase(
                        "remote_engine_execute_payload",
                        component=component_id,
                        payload_kind=payload.get("payload_kind"),
                    ):
                        if _should_reload_modal_volume(payload):
                            _reload_modal_volume_for_request(
                                vol,
                                str(component_id),
                                reload_marker=reload_marker,
                                payload=payload,
                            )
                        else:
                            _emit_modal_volume_reload_skip(component_id, payload)
                        yield from _stream_remote_payload_events(
                            payload,
                            kwargs_payload,
                            cancellation_event=execution_control.cancellation_event,
                            interrupt_store=interrupt_flags,
                            interrupt_flag_key=execution_control.interrupt_flag_key,
                        )
            except Exception as exc:
                _maybe_schedule_container_termination_on_error(payload, exc)
                raise

else:
    app = None
    RemoteEngine = None
