"""Remote Modal runtime and local execution fallback."""

import asyncio
import copy
from dataclasses import dataclass, field, replace
import hashlib
import importlib
import importlib.util
import inspect
from io import BytesIO
import json
import logging
import os
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
import uuid
import zipfile
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from ..serialization import (
    coerce_serialized_node_outputs,
    deserialize_value,
    deserialize_node_inputs,
    deserialize_node_outputs,
    is_mapped_output_value,
    join_mapped_values_for_scheduler,
    serialize_mapping,
    serialize_node_inputs,
    serialize_node_outputs,
    serialize_value,
    split_mapped_value,
    unwrap_mapped_output_value,
)
from ..durable_state import (
    DurableObjectRef,
    DurableStateError,
    FileDurableObjectStore,
    read_modal_volume_file,
    stable_remote_invocation_id,
)
from ..output_artifacts import (
    RemoteExecutionResult,
    materialize_remote_output_artifacts,
    unpack_remote_execution_result,
)
from ..remote_protocol import (
    BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
    PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
)
from ..session_state import (
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
from ..runtime_environment import (
    REMOTE_APP_PROTOCOL_VERSION,
    RemoteRuntimeIdentity,
    build_remote_runtime_identity,
)
from ..llm_profiles import (
    get_llm_profile,
    llm_model_reference_node_ids_from_payload,
    llm_model_references_from_payload,
    resolved_llm_profile_payloads,
    rewrite_llm_model_references,
)
from ..llm_recovery import (
    LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY,
    exhausted_recovery_used_vllm_throughput,
    is_llm_memory_recovery_exhausted,
)
from ..staging_process import staging_no_progress_timeout_seconds
from ..settings import (
    MODAL_GPU_TYPES,
    ModalSyncSettings,
    get_settings,
    modal_deployment_app_name,
    settings_for_modal_gpu,
)
from .modal_billing import (
    ModalBillingStatusError,
    ModalHourlyBillingStatus,
    _completed_modal_billing_interval,
    _fetch_modal_hourly_billing_synchronously,
    _matching_modal_hourly_billing_rows,
    _modal_billing_cost,
    _modal_billing_row_value,
    _modal_hourly_billing_status_from_rows,
    _prune_modal_hourly_billing_cache,
    get_hourly_modal_app_billing,
)
from .modal_container_logs import (
    MODAL_GPU_ESTIMATED_USD_PER_SECOND,
    MODAL_GPU_PRICING_EFFECTIVE_DATE,
    ModalContainerStatus,
    ModalContainerStatusError,
    _coerce_modal_task_id,
    _is_remote_container_log_stream_enabled,
    _release_remote_container_log_stream,
    _retain_remote_container_log_stream,
    _stop_modal_task_synchronously,
    list_active_modal_containers,
    stop_managed_modal_container,
)
from .host_session_bridge import (
    _RemoteSessionBridgeResolutionStats,
    _build_remote_session_bridge_record,
    _log_remote_session_resolution_summary,
    _payload_remote_session_handle,
    _rehydrate_remote_session_bridge_value,
    _resolve_remote_session_inputs,
    _restore_serialized_remote_session_bridge_value,
    _store_remote_session_bridge_value,
    materialize_remote_session_bridge_ref_locally,
)
from . import host_session_bridge as _host_session_bridge
from .local_execution import (
    RemoteSubgraphExecutionError,
    _execute_subgraph_prompt,
    _is_link,
    _iter_prompt_links,
    _load_nodes_module,
    _materialize_local_custom_node_assets,
    _node_input_type_map,
    _resolve_required_subgraph_nodes,
    execute_node_locally,
    execute_subgraph_locally,
)
from .local_ui_events import (
    _emit_local_executed_output,
    _emit_local_modal_progress,
    _emit_local_modal_status,
    _emit_local_preview_boundary_output,
    _emit_local_preview_image,
    _emit_local_remote_dispatch_status,
    _emit_local_remote_startup_status,
    _lookup_local_prompt_server,
    _progress_stream_event_metadata,
    _remote_execution_destination,
    _remote_execution_identity,
    _remote_prompt_ancestor_node_ids,
    _should_forward_suppressed_stream_event,
    _should_stream_remote_progress,
)
from .modal_deployment import (
    _MODAL_APP_STOP_TIMEOUT_SECONDS,
    _MODAL_CLOUD_MODULE_NAME,
    _REMOTE_APP_PROTOCOL_VERSION,
    ModalDeploymentHooks,
    ModalRemoteAppOutOfDateError,
    ModalRemoteInvocationError,
    _auto_deploy_modal_app,
    _call_modal_method,
    _component_pool_slot_index,
    _ensure_remote_engine_protocol_current,
    _expected_remote_runtime_fingerprint,
    _install_modal_cloud_exception_compatibility_module,
    _is_missing_modal_deployment_error,
    _load_modal_cloud_module,
    _lookup_deployed_remote_engine,
    _lookup_deployed_remote_engine_with_retry,
    _lookup_protocol_current_remote_engine,
    _mapped_lane_affinity_key,
    _modal_auto_deploy_state,
    _modal_cloud_settings_override,
    _modal_deploy_cache_key,
    _modal_environment_name,
    _modal_lookup_error_types,
    _modal_runtime_cache_key,
    _remote_runtime_identity_for_settings,
    _remote_worker_affinity_key,
    _remote_worker_pool_affinity_key,
    _replace_outdated_modal_app,
    _runtime_fingerprint_from_payload,
    _settings_for_payload,
    _stop_modal_app_for_replacement,
    _stop_modal_app_via_cli,
    _stop_modal_app_via_sdk,
    configure_modal_deployment_hooks,
)
from .modal_warmup import (
    ModalWarmupHooks,
    _await_matching_speculative_prewarm,
    _await_prompt_warmup_slots,
    _build_llm_prewarm_plans,
    _build_loader_prewarm_plans,
    _build_prompt_warmup_request,
    _component_parallelism_metadata,
    _ensure_prompt_warmup_state,
    _invoke_modal_warmup_blocking,
    _loader_prewarm_plan_signature,
    _loader_snapshot_profile_key,
    _prepare_snapshot_profile_fields,
    _prompt_parallelism_target,
    _prompt_warmup_head_start_seconds,
    _record_snapshot_warmup_measurement,
    _register_exact_component_parallelism,
    _run_speculative_affinity_prewarm,
    _schedule_post_deploy_runtime_seed,
    _schedule_speculative_affinity_prewarm,
    _select_gpu_snapshot_for_profile,
    _speculative_warmup_identity,
    _start_local_gap_keepalive,
    _stop_local_gap_keepalive,
    _track_prompt_warmup_future,
    _warmup_prompt_id,
    boost_mapped_component_warmup,
    configure_modal_warmup_hooks,
    ensure_remote_warm_capacity,
)

logger = logging.getLogger(__name__)

# LLM staging and remote durable-object handles.
_STAGED_LLM_PROFILES_LOCK = threading.Lock()
_STAGED_LLM_PROFILES: set[tuple[str, str, str]] = set()
_STAGED_LLM_PROFILE_RESULTS: dict[tuple[str, str], dict[str, Any]] = {}
_MODAL_INTERRUPT_DICTS_LOCK = threading.Lock()
_MODAL_INTERRUPT_DICTS: dict[tuple[str, str | None], Any] = {}
# Mapped execution state.
_MAPPED_PROGRESS_NODE_IDS_LOCK = threading.Lock()
_MAPPED_PROGRESS_NODE_IDS: dict[tuple[str, str, str], str] = {}
_MODAL_STAGE_STREAM_END = object()

# Active invocation and local durable-session state.
_IMPLICIT_BATCH_PRESERVING_TARGETS = frozenset({("CreateVideo", "images")})
_ACTIVE_REMOTE_INVOCATIONS_LOCK = threading.Lock()
_ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT: dict[
    str, dict[str, "_ActiveRemoteInvocation"]
] = {}

def _remote_modal_call_worker_count() -> int:
    """Return the number of local worker threads reserved for blocking Modal calls."""
    return max(1, int(get_settings().max_inflight_calls))


# Blocking Modal call executor state.
_REMOTE_MODAL_CALL_EXECUTOR_LOCK = threading.Lock()
_REMOTE_MODAL_CALL_EXECUTOR: ThreadPoolExecutor | None = None


def _remote_modal_call_executor() -> ThreadPoolExecutor:
    """Return the lazily constructed executor for blocking Modal calls."""
    global _REMOTE_MODAL_CALL_EXECUTOR
    with _REMOTE_MODAL_CALL_EXECUTOR_LOCK:
        if _REMOTE_MODAL_CALL_EXECUTOR is None:
            _REMOTE_MODAL_CALL_EXECUTOR = ThreadPoolExecutor(
                max_workers=_remote_modal_call_worker_count()
            )
        return _REMOTE_MODAL_CALL_EXECUTOR


@dataclass(frozen=True)
class _ModalStageStreamFailure:
    """Carry an arbitrary Modal stream exception across a reader thread."""

    error: Exception






@dataclass
class _ActiveRemoteInvocation:
    """Track one local proxy call that is currently waiting on remote Modal work."""

    prompt_id: str
    component_id: str
    cancellation_event: threading.Event | None
    interrupt_remote_call: Callable[[], Any] | None




try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local fallback tests.
    modal = None
































































def _close_remote_payload_stream(stream_events: Iterator[dict[str, Any]]) -> None:
    """Best-effort close one streamed Modal iterator after the terminal result arrives."""
    close_callable = getattr(stream_events, "close", None)
    if not callable(close_callable):
        return
    try:
        close_result = close_callable()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Ignoring remote payload stream close failure: %s", exc)
        return
    if not asyncio.iscoroutine(close_result):
        return
    try:
        asyncio.run(close_result)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Ignoring async remote payload stream close failure: %s", exc)
















































































































































































def _local_processing_interrupted() -> bool:
    """Return whether the current local ComfyUI execution was interrupted."""
    try:
        import comfy.model_management
    except ModuleNotFoundError:
        return False

    return bool(comfy.model_management.processing_interrupted())


def _raise_local_interrupt() -> None:
    """Raise ComfyUI's native interruption exception for the current execution."""
    import comfy.model_management

    raise comfy.model_management.InterruptProcessingException()


def _exception_indicates_interruption(exc: BaseException) -> bool:
    """Return whether an exception represents cancellation or interrupted execution."""
    if isinstance(exc, asyncio.CancelledError):
        return True
    message = str(exc).lower()
    return "interrupt" in message or "cancel" in message


def _remote_interrupt_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the prompt/component pair used to interrupt one remote execution."""
    prompt_id = str(
        payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph"
    )
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


def _lookup_modal_interrupt_store() -> Any | None:
    """Return the shared Modal Dict used to signal remote cancellation requests."""
    if modal is None or not hasattr(modal, "Dict"):
        return None

    settings = get_settings()
    cache_key = (settings.interrupt_dict_name, _modal_environment_name())
    with _MODAL_INTERRUPT_DICTS_LOCK:
        cached_store = _MODAL_INTERRUPT_DICTS.get(cache_key)
        if cached_store is not None:
            return cached_store

    interrupt_store = modal.Dict.from_name(
        settings.interrupt_dict_name,
        environment_name=cache_key[1],
        create_if_missing=True,
    )
    with _MODAL_INTERRUPT_DICTS_LOCK:
        _MODAL_INTERRUPT_DICTS[cache_key] = interrupt_store
    return interrupt_store


def _remote_interrupt_flag_value() -> dict[str, float]:
    """Return the shared Modal interrupt-store value for one cancellation request."""
    return {"requested_at": time.time()}


def _write_remote_interrupt_flag(
    interrupt_store: Any, prompt_id: str, component_id: str
) -> None:
    """Write one remote cancellation request with the blocking Modal Dict API."""
    interrupt_store.put(
        _remote_interrupt_flag_key(prompt_id, component_id),
        _remote_interrupt_flag_value(),
    )


async def _write_remote_interrupt_flag_async(
    interrupt_store: Any,
    prompt_id: str,
    component_id: str,
) -> None:
    """Write one remote cancellation request without blocking the async caller."""
    put_method = getattr(interrupt_store, "put", None)
    put_async = getattr(put_method, "aio", None)
    if callable(put_async):
        result = put_async(
            _remote_interrupt_flag_key(prompt_id, component_id),
            _remote_interrupt_flag_value(),
        )
        if inspect.isawaitable(result):
            await result
        return

    await asyncio.to_thread(
        _write_remote_interrupt_flag, interrupt_store, prompt_id, component_id
    )


def _request_remote_interrupt(payload: dict[str, Any]) -> bool:
    """Write one remote cancellation request into the shared Modal interrupt store."""
    _abandon_local_modal_workflow_gate(payload, "local interrupt requested")
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return False

    prompt_id, component_id = _remote_interrupt_key(payload)
    _write_remote_interrupt_flag(interrupt_store, prompt_id, component_id)
    logger.info(
        "Propagated local interrupt to Modal prompt=%s component=%s through shared control state.",
        prompt_id,
        component_id,
    )
    return True


async def _request_remote_interrupt_async(payload: dict[str, Any]) -> bool:
    """Write one remote cancellation request into the shared Modal interrupt store asynchronously."""
    _abandon_local_modal_workflow_gate(payload, "local interrupt requested")
    interrupt_store = await asyncio.to_thread(_lookup_modal_interrupt_store)
    if interrupt_store is None:
        return False

    prompt_id, component_id = _remote_interrupt_key(payload)
    await _write_remote_interrupt_flag_async(interrupt_store, prompt_id, component_id)
    logger.info(
        "Propagated local interrupt to Modal prompt=%s component=%s through shared control state.",
        prompt_id,
        component_id,
    )
    return True


def active_remote_modal_prompt_ids() -> set[str]:
    """Return prompt ids that currently have local proxies waiting on Modal work."""
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        return set(_ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT)


@contextmanager
def _registered_active_remote_invocation(
    payload: dict[str, Any],
    cancellation_event: threading.Event | None,
    interrupt_remote_call: Callable[[], Any] | None,
) -> Iterator[None]:
    """Register one active Modal call so targeted ComfyUI interrupts can find it."""
    prompt_id, component_id = _remote_interrupt_key(payload)
    invocation = _ActiveRemoteInvocation(
        prompt_id=prompt_id,
        component_id=component_id,
        cancellation_event=cancellation_event,
        interrupt_remote_call=interrupt_remote_call,
    )
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        prompt_invocations = _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.setdefault(
            prompt_id, {}
        )
        prompt_invocations[component_id] = invocation
    logger.info(
        "Registered active Modal invocation prompt=%s component=%s for targeted cancellation.",
        prompt_id,
        component_id,
    )
    try:
        yield
    finally:
        with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
            prompt_invocations = _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.get(prompt_id)
            if prompt_invocations is not None:
                prompt_invocations.pop(component_id, None)
                if not prompt_invocations:
                    _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.pop(prompt_id, None)
        logger.info(
            "Unregistered active Modal invocation prompt=%s component=%s.",
            prompt_id,
            component_id,
        )


def request_remote_modal_prompt_interrupt(prompt_id: str) -> bool:
    """Request cancellation for every active Modal invocation belonging to one prompt."""
    normalized_prompt_id = str(prompt_id)
    _abandon_local_modal_workflow_gate(
        {"prompt_id": normalized_prompt_id},
        "prompt-level interrupt requested",
    )
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        invocations = list(
            _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.get(normalized_prompt_id, {}).values()
        )
    if not invocations:
        return False

    logger.info(
        "Requesting remote Modal cancellation for prompt=%s across %d active component(s).",
        normalized_prompt_id,
        len(invocations),
    )
    for invocation in invocations:
        if invocation.cancellation_event is not None:
            invocation.cancellation_event.set()
        _propagate_remote_interrupt_request(
            {
                "prompt_id": invocation.prompt_id,
                "component_id": invocation.component_id,
            },
            invocation.interrupt_remote_call,
        )
    return True


async def request_remote_modal_prompt_interrupt_async(prompt_id: str) -> bool:
    """Request cancellation for every active Modal invocation belonging to one prompt asynchronously."""
    normalized_prompt_id = str(prompt_id)
    _abandon_local_modal_workflow_gate(
        {"prompt_id": normalized_prompt_id},
        "prompt-level interrupt requested",
    )
    with _ACTIVE_REMOTE_INVOCATIONS_LOCK:
        invocations = list(
            _ACTIVE_REMOTE_INVOCATIONS_BY_PROMPT.get(normalized_prompt_id, {}).values()
        )
    if not invocations:
        return False

    logger.info(
        "Requesting async remote Modal cancellation for prompt=%s across %d active component(s).",
        normalized_prompt_id,
        len(invocations),
    )
    for invocation in invocations:
        if invocation.cancellation_event is not None:
            invocation.cancellation_event.set()
        await _request_remote_interrupt_async(
            {"prompt_id": invocation.prompt_id, "component_id": invocation.component_id}
        )
    return True


def _sync_local_interrupt_to_cancellation_event(
    payload: dict[str, Any],
    cancellation_event: threading.Event | None,
) -> bool:
    """Mirror ComfyUI's interrupt flag into the current Modal cancellation event."""
    if cancellation_event is not None and cancellation_event.is_set():
        return True
    if not _local_processing_interrupted():
        return False
    if cancellation_event is not None and not cancellation_event.is_set():
        logger.info(
            "Observed local interrupt while Modal component=%s was running; requesting remote cancellation.",
            payload.get("component_id"),
        )
        cancellation_event.set()
    _abandon_local_modal_workflow_gate(payload, "observed local interrupt")
    return True


def _abandon_local_modal_workflow_gate(payload: dict[str, Any], reason: str) -> None:
    """Release the local prompt gate for a Modal prompt that ComfyUI has cancelled."""
    prompt_id = payload.get("prompt_id")
    if prompt_id is None:
        return

    try:
        from ..modal_executor_node import abandon_modal_workflow_execution_prompt
    except ImportError:
        logger.debug(
            "Unable to import Modal workflow gate helper while abandoning prompt."
        )
        return

    abandon_modal_workflow_execution_prompt(str(prompt_id), reason)


def _propagate_remote_interrupt_request(
    payload: dict[str, Any],
    interrupt_remote_call: Callable[[], Any] | None,
) -> None:
    """Send one best-effort remote cancellation request for an active Modal payload."""
    prompt_id, component_id = _remote_interrupt_key(payload)
    if interrupt_remote_call is None:
        logger.warning(
            "Local interrupt requested for component=%s, but no remote interrupt method is available.",
            component_id,
        )
        return
    try:
        interrupt_remote_call()
        logger.info(
            "Propagated local interrupt to Modal prompt=%s component=%s.",
            prompt_id,
            component_id,
        )
    except Exception:
        logger.exception(
            "Failed to propagate local interrupt to Modal prompt=%s component=%s.",
            prompt_id,
            component_id,
        )


def _handle_modal_wait_cancellation(
    payload: dict[str, Any],
    cancellation_event: threading.Event,
    *,
    interrupt_sent: bool,
    cancellation_started_at: float | None,
) -> tuple[bool, float | None]:
    """Propagate and bound local waiting after cancellation during any Modal wait phase."""
    if not _sync_local_interrupt_to_cancellation_event(payload, cancellation_event):
        return interrupt_sent, cancellation_started_at

    if not interrupt_sent:
        _request_remote_interrupt(payload)
        return True, time.monotonic()

    if cancellation_started_at is None:
        return interrupt_sent, time.monotonic()

    grace_seconds = max(0.0, get_settings().remote_cancel_grace_seconds)
    if time.monotonic() - cancellation_started_at >= grace_seconds:
        logger.info(
            "Modal component=%s did not reach a cancellable remote call within %.3fs of local interrupt; releasing the local prompt while remote cancellation continues.",
            payload.get("component_id"),
            grace_seconds,
        )
        raise ModalRemoteInvocationError(
            "Remote Modal call did not reach a cancellable remote phase after local interrupt propagation."
        )

    return interrupt_sent, cancellation_started_at


async def _handle_modal_wait_cancellation_async(
    payload: dict[str, Any],
    cancellation_event: threading.Event,
    *,
    interrupt_sent: bool,
    cancellation_started_at: float | None,
) -> tuple[bool, float | None]:
    """Propagate and bound local waiting after cancellation during an async Modal wait phase."""
    if not _sync_local_interrupt_to_cancellation_event(payload, cancellation_event):
        return interrupt_sent, cancellation_started_at

    if not interrupt_sent:
        await _request_remote_interrupt_async(payload)
        return True, time.monotonic()

    if cancellation_started_at is None:
        return interrupt_sent, time.monotonic()

    grace_seconds = max(0.0, get_settings().remote_cancel_grace_seconds)
    if time.monotonic() - cancellation_started_at >= grace_seconds:
        logger.info(
            "Modal component=%s did not reach a cancellable remote call within %.3fs of local interrupt; releasing the local prompt while remote cancellation continues.",
            payload.get("component_id"),
            grace_seconds,
        )
        raise ModalRemoteInvocationError(
            "Remote Modal call did not reach a cancellable remote phase after local interrupt propagation."
        )

    return interrupt_sent, cancellation_started_at


def _invoke_remote_call_with_interrupts(
    *,
    payload: dict[str, Any],
    invoke_remote_call: Callable[[], bytes],
    interrupt_remote_call: Callable[[], Any] | None,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Run one blocking remote call while optionally propagating cancellation to Modal."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    cancellation_started_at: float | None = None

    def execute_remote_call() -> None:
        """Run the blocking Modal request in a worker thread."""
        try:
            result_queue.put(("result", invoke_remote_call()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    request_thread = threading.Thread(
        target=execute_remote_call,
        name=f"modal-request-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    request_thread.start()
    interrupt_sent = False
    try:
        with _registered_active_remote_invocation(
            payload, cancellation_event, interrupt_remote_call
        ):
            while True:
                try:
                    result_kind, result_payload = result_queue.get(timeout=0.1)
                except queue.Empty:
                    if _sync_local_interrupt_to_cancellation_event(
                        payload, cancellation_event
                    ):
                        if not interrupt_sent:
                            _propagate_remote_interrupt_request(
                                payload, interrupt_remote_call
                            )
                            interrupt_sent = True
                            cancellation_started_at = time.monotonic()
                        elif cancellation_started_at is not None:
                            grace_seconds = max(
                                0.0, get_settings().remote_cancel_grace_seconds
                            )
                            if (
                                time.monotonic() - cancellation_started_at
                                >= grace_seconds
                            ):
                                logger.info(
                                    "Modal component=%s did not return within %.3fs of local interrupt propagation; releasing the local prompt while remote cancellation continues.",
                                    payload.get("component_id"),
                                    grace_seconds,
                                )
                                raise ModalRemoteInvocationError(
                                    "Remote Modal call did not finish after local interrupt propagation."
                                )
                    continue

                if result_kind == "result":
                    return bytes(result_payload)
                raise result_payload
    finally:
        request_thread.join(
            timeout=0.1
            if cancellation_event is not None and cancellation_event.is_set()
            else 1.0
        )


def _consume_remote_payload_stream(
    payload: dict[str, Any],
    stream_events: Iterator[dict[str, Any]],
) -> bytes:
    """Forward remote progress events into the local UI and return the final payload bytes."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    node_ids = [str(node_id) for node_id in payload.get("component_node_ids", [])]
    modal_gpu = (
        str(payload["modal_gpu"]) if payload.get("modal_gpu") is not None else None
    )
    suppress_status_stream = bool(payload.get("suppress_status_stream"))
    result_payload: bytes | bytearray | None = None
    suppressed_progress_node_metadata: dict[str, dict[str, str | None]] = {}
    active_remote_task_id: str | None = None
    active_remote_log_task_id: str | None = None
    should_close_stream = False
    component_id = str(payload.get("component_id") or "payload")
    invocation_id = str(payload.get("invocation_id") or "none")
    stream_started_at = time.monotonic()
    previous_event_at = stream_started_at
    event_count = 0
    progress_event_count = 0
    speculative_prewarm_checked = False
    logger.info(
        "Starting local Modal stream consumption component=%s prompt_id=%s invocation_id=%s.",
        component_id,
        prompt_id or "none",
        invocation_id,
    )

    try:
        for stream_event in stream_events:
            event_received_at = time.monotonic()
            seconds_since_previous_event = event_received_at - previous_event_at
            previous_event_at = event_received_at
            event_count += 1
            if not speculative_prewarm_checked:
                speculative_prewarm_checked = True
                _schedule_speculative_affinity_prewarm(
                    payload,
                    reason="current_remote_stream_started",
                )
            event_kind = str(stream_event.get("kind", ""))
            if event_kind == "remote_logs":
                task_id = _coerce_modal_task_id(stream_event.get("task_id"))
                if task_id is not None and active_remote_task_id is None:
                    active_remote_task_id = task_id
                if (
                    task_id is not None
                    and active_remote_log_task_id is None
                    and _is_remote_container_log_stream_enabled()
                ):
                    active_remote_log_task_id = _retain_remote_container_log_stream(
                        task_id
                    )
                continue
            if event_kind == "progress":
                progress_event_count += 1
                event_type = str(stream_event.get("event_type", ""))
                if event_type == "node_progress":
                    progress_metadata = _progress_stream_event_metadata(stream_event)
                    filter_node_id = (
                        progress_metadata["filter_node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    lane_id = (
                        str(stream_event["lane_id"])
                        if stream_event.get("lane_id") is not None
                        else (
                            str(payload["mapped_progress_lane_id"])
                            if payload.get("mapped_progress_lane_id") is not None
                            else None
                        )
                    )
                    aggregate_only = bool(stream_event.get("aggregate_only", False))
                    if (
                        suppress_status_stream
                        and lane_id is None
                        and not aggregate_only
                        and not _should_forward_suppressed_stream_event(
                            payload, filter_node_id
                        )
                    ):
                        logger.debug(
                            "Suppressing streamed Modal node progress for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            stream_event.get("real_node_id"),
                        )
                        continue
                    reported_node_id = (
                        progress_metadata["node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    if reported_node_id is not None:
                        display_node_id = (
                            progress_metadata["display_node_id"]
                            if progress_metadata is not None
                            else str(reported_node_id)
                        )
                        real_node_id = (
                            progress_metadata["real_node_id"]
                            if progress_metadata is not None
                            else None
                        )
                        progress_node_id = real_node_id or display_node_id
                        completed_ancestor_node_ids = _remote_prompt_ancestor_node_ids(
                            payload,
                            progress_node_id,
                        )
                        if lane_id is not None:
                            _remember_mapped_lane_node_id(
                                payload, lane_id, progress_node_id
                            )
                        elif (
                            suppress_status_stream
                            and not aggregate_only
                            and progress_metadata is not None
                        ):
                            suppressed_progress_node_metadata[
                                str(progress_metadata["filter_node_id"])
                            ] = {
                                "node_id": str(reported_node_id),
                                "display_node_id": display_node_id,
                                "real_node_id": real_node_id,
                            }
                        logger.debug(
                            "Forwarding streamed Modal node progress for component=%s node_id=%s real_node_id=%s value=%s max=%s lane_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                            real_node_id,
                            stream_event.get("value"),
                            stream_event.get("max"),
                            lane_id,
                        )
                        progress_kwargs = {
                            "prompt_id": prompt_id,
                            "client_id": client_id,
                            "node_id": str(reported_node_id),
                            "value": float(stream_event.get("value", 0.0)),
                            "max_value": float(stream_event.get("max", 1.0)),
                            "display_node_id": display_node_id,
                            "real_node_id": real_node_id,
                            "lane_id": lane_id,
                            "clear": bool(stream_event.get("clear", False)),
                            "item_index": (
                                int(stream_event["item_index"])
                                if stream_event.get("item_index") is not None
                                else (
                                    int(payload["map_item_index"])
                                    if payload.get("map_item_index") is not None
                                    else None
                                )
                            ),
                            "aggregate_only": aggregate_only,
                        }
                        progress_kwargs.update(
                            _remote_execution_identity(payload, active_remote_task_id)
                        )
                        for string_field in ("stage", "message", "unit"):
                            if stream_event.get(string_field) is not None:
                                progress_kwargs[string_field] = str(
                                    stream_event[string_field]
                                )
                        if stream_event.get("indeterminate") is not None:
                            progress_kwargs["indeterminate"] = bool(
                                stream_event["indeterminate"]
                            )
                        for numeric_field in (
                            "elapsed_seconds",
                            "time_to_first_token_seconds",
                            "tokens_per_second",
                        ):
                            if stream_event.get(numeric_field) is not None:
                                progress_kwargs[numeric_field] = float(
                                    stream_event[numeric_field]
                                )
                        if completed_ancestor_node_ids:
                            progress_kwargs[
                                "completed_ancestor_node_ids"
                            ] = completed_ancestor_node_ids
                        _emit_local_modal_progress(**progress_kwargs)
                    continue
                if event_type == "executed":
                    reported_node_id = stream_event.get("node_id")
                    if reported_node_id is not None:
                        if not _should_forward_suppressed_stream_event(
                            payload, reported_node_id
                        ):
                            logger.debug(
                                "Suppressing streamed Modal executed output for component=%s node_id=%s because it does not belong to this mapped/static payload.",
                                payload.get("component_id"),
                                reported_node_id,
                            )
                            continue
                        logger.debug(
                            "Forwarding streamed Modal executed output for component=%s node_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                        )
                        _emit_local_executed_output(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            display_node_id=(
                                str(stream_event["display_node_id"])
                                if stream_event.get("display_node_id") is not None
                                else None
                            ),
                            output_payload=deserialize_value(
                                stream_event.get("output")
                            ),
                        )
                    continue
                if event_type == "preview":
                    reported_node_id = stream_event.get("node_id")
                    image_bytes = deserialize_value(stream_event.get("image_bytes"))
                    if reported_node_id is not None and isinstance(image_bytes, bytes):
                        if not _should_forward_suppressed_stream_event(
                            payload, reported_node_id
                        ):
                            logger.debug(
                                "Suppressing streamed Modal preview image for component=%s node_id=%s because it does not belong to this mapped/static payload.",
                                payload.get("component_id"),
                                reported_node_id,
                            )
                            continue
                        logger.debug(
                            "Forwarding streamed Modal preview image for component=%s node_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                        )
                        _emit_local_preview_image(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            display_node_id=(
                                str(stream_event["display_node_id"])
                                if stream_event.get("display_node_id") is not None
                                else None
                            ),
                            parent_node_id=(
                                str(stream_event["parent_node_id"])
                                if stream_event.get("parent_node_id") is not None
                                else None
                            ),
                            real_node_id=(
                                str(stream_event["real_node_id"])
                                if stream_event.get("real_node_id") is not None
                                else None
                            ),
                            image_type=str(stream_event.get("image_type", "PNG")),
                            image_bytes=image_bytes,
                            max_size=(
                                int(stream_event["max_size"])
                                if stream_event.get("max_size") is not None
                                else None
                            ),
                        )
                    continue
                if event_type == "boundary_output":
                    preview_target_node_ids = [
                        str(node_id)
                        for node_id in stream_event.get("preview_target_node_ids", [])
                        if str(node_id)
                    ]
                    if preview_target_node_ids:
                        logger.debug(
                            "Forwarding streamed Modal boundary output previews for component=%s source_node=%s targets=%s.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            preview_target_node_ids,
                        )
                        _emit_local_preview_boundary_output(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            preview_target_node_ids=preview_target_node_ids,
                            image_value=deserialize_value(stream_event.get("value")),
                        )
                    continue
                if event_type == "node_cached":
                    progress_metadata = _progress_stream_event_metadata(stream_event)
                    filter_node_id = (
                        progress_metadata["filter_node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    if (
                        suppress_status_stream
                        and not _should_forward_suppressed_stream_event(
                            payload, filter_node_id
                        )
                    ):
                        logger.debug(
                            "Suppressing streamed Modal cached-node marker for component=%s node_id=%s real_node_id=%s because it does not belong to this mapped/static payload.",
                            payload.get("component_id"),
                            stream_event.get("node_id"),
                            stream_event.get("real_node_id"),
                        )
                        continue
                    reported_node_id = (
                        progress_metadata["node_id"]
                        if progress_metadata is not None
                        else None
                    )
                    if reported_node_id is not None:
                        display_node_id = (
                            progress_metadata["display_node_id"]
                            if progress_metadata is not None
                            else str(reported_node_id)
                        )
                        real_node_id = (
                            progress_metadata["real_node_id"]
                            if progress_metadata is not None
                            else None
                        )
                        logger.debug(
                            "Forwarding streamed Modal cached-node marker for component=%s node_id=%s real_node_id=%s.",
                            payload.get("component_id"),
                            reported_node_id,
                            real_node_id,
                        )
                        _emit_local_modal_progress(
                            prompt_id=prompt_id,
                            client_id=client_id,
                            node_id=str(reported_node_id),
                            value=0.0,
                            max_value=1.0,
                            display_node_id=display_node_id,
                            real_node_id=real_node_id,
                            cached_hit=True,
                            **_remote_execution_identity(
                                payload, active_remote_task_id
                            ),
                        )
                    continue
                logger.info(
                    "Forwarding streamed Modal progress for component=%s phase=%s active_node_id=%s.",
                    payload.get("component_id"),
                    stream_event.get("phase"),
                    stream_event.get("active_node_id"),
                )
                if suppress_status_stream:
                    remote_phase = str(stream_event.get("phase", "executing"))
                    if remote_phase in {
                        "execution_success",
                        "execution_error",
                        "execution_interrupted",
                    }:
                        for (
                            progress_metadata
                        ) in suppressed_progress_node_metadata.values():
                            _emit_local_modal_progress(
                                prompt_id=prompt_id,
                                client_id=client_id,
                                node_id=str(progress_metadata["node_id"]),
                                value=0.0,
                                max_value=1.0,
                                display_node_id=progress_metadata["display_node_id"],
                                real_node_id=progress_metadata["real_node_id"],
                                clear=True,
                            )
                        suppressed_progress_node_metadata.clear()
                    continue
                remote_phase = str(stream_event.get("phase", "executing"))
                if remote_phase == "execution_success":
                    _emit_local_modal_status(
                        prompt_id=prompt_id,
                        client_id=client_id,
                        phase="finalizing",
                        node_ids=node_ids,
                        modal_gpu=modal_gpu,
                        status_message=(
                            "Receiving remote outputs from "
                            f"{_remote_execution_destination(payload)}"
                        ),
                        **_remote_execution_identity(payload, active_remote_task_id),
                    )
                    continue
                _emit_local_modal_status(
                    prompt_id=prompt_id,
                    client_id=client_id,
                    phase=remote_phase,
                    node_ids=node_ids,
                    modal_gpu=modal_gpu,
                    active_node_id=(
                        str(stream_event["active_node_id"])
                        if stream_event.get("active_node_id") is not None
                        else None
                    ),
                    completed_ancestor_node_ids=_remote_prompt_ancestor_node_ids(
                        payload,
                        str(stream_event["active_node_id"])
                        if stream_event.get("active_node_id") is not None
                        else None,
                    )
                    or None,
                    active_node_class_type=(
                        str(stream_event["active_node_class_type"])
                        if stream_event.get("active_node_class_type") is not None
                        else None
                    ),
                    active_node_role=(
                        str(stream_event["active_node_role"])
                        if stream_event.get("active_node_role") is not None
                        else None
                    ),
                    **_remote_execution_identity(payload, active_remote_task_id),
                )
                continue
            if event_kind == "result":
                candidate_outputs = stream_event.get("outputs")
                candidate_bytes = (
                    len(candidate_outputs)
                    if isinstance(candidate_outputs, bytes | bytearray)
                    else -1
                )
                logger.info(
                    "Received streamed Modal result component=%s prompt_id=%s invocation_id=%s "
                    "result_bytes=%d stream_elapsed_seconds=%.3f seconds_since_previous_event=%.3f "
                    "event_count=%d progress_event_count=%d.",
                    component_id,
                    prompt_id or "none",
                    invocation_id,
                    candidate_bytes,
                    event_received_at - stream_started_at,
                    seconds_since_previous_event,
                    event_count,
                    progress_event_count,
                )
                try:
                    result_payload = coerce_serialized_node_outputs(candidate_outputs)
                except TypeError as exc:
                    raise ModalRemoteInvocationError(
                        "Modal streamed payload result did not include transport-safe outputs."
                    ) from exc
                should_close_stream = True
                break
            logger.debug(
                "Ignoring unexpected streamed Modal event kind=%s for component=%s.",
                event_kind,
                payload.get("component_id"),
            )
    finally:
        if should_close_stream:
            close_started_at = time.monotonic()
            logger.info(
                "Starting local Modal result stream close component=%s prompt_id=%s invocation_id=%s.",
                component_id,
                prompt_id or "none",
                invocation_id,
            )
            _close_remote_payload_stream(stream_events)
            logger.info(
                "Finished local Modal result stream close in %.3fs component=%s prompt_id=%s "
                "invocation_id=%s.",
                time.monotonic() - close_started_at,
                component_id,
                prompt_id or "none",
                invocation_id,
            )
        if active_remote_log_task_id is not None:
            _release_remote_container_log_stream(active_remote_log_task_id)

    if result_payload is None:
        raise ModalRemoteInvocationError(
            f"Modal streamed payload for component={payload.get('component_id')!r} did not yield a final result."
        )
    logger.info(
        "Finished local Modal stream consumption in %.3fs component=%s prompt_id=%s invocation_id=%s "
        "result_bytes=%d event_count=%d progress_event_count=%d.",
        time.monotonic() - stream_started_at,
        component_id,
        prompt_id or "none",
        invocation_id,
        len(result_payload),
        event_count,
        progress_event_count,
    )
    return bytes(result_payload)


def _mapped_execution_parallelism(item_count: int) -> int:
    """Return the local worker width used to schedule mapped Modal item executions."""
    settings = get_settings()
    configured_limit = settings.max_inflight_calls
    if settings.max_containers is not None:
        configured_limit = min(configured_limit, settings.max_containers)
    return max(1, min(item_count, configured_limit))


def _mapped_phase_definition(
    payload: dict[str, Any], phase_key: str
) -> dict[str, Any] | None:
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
        "terminate_container_on_error": bool(
            payload.get("terminate_container_on_error", True)
        ),
        "custom_nodes_bundle": payload.get("custom_nodes_bundle"),
    }
    snapshot_profile_key = payload.get("snapshot_profile_key")
    if isinstance(snapshot_profile_key, str) and snapshot_profile_key.strip():
        shared_fields["snapshot_profile_key"] = snapshot_profile_key.strip()
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
    *,
    suppress_status_stream: bool = False,
    lane_id: str | None = None,
    item_index: int | None = None,
) -> dict[str, Any]:
    """Return one explicit static or mapped subgraph payload."""
    phase_definition = _mapped_phase_definition(payload, phase_key)
    if phase_definition is None:
        raise KeyError(f"Mapped payload is missing phase definition {phase_key!r}.")

    phase_payload = {
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
    if suppress_status_stream:
        phase_payload["suppress_status_stream"] = True
    if lane_id is not None:
        phase_payload["mapped_progress_lane_id"] = str(lane_id)
        phase_payload["mapped_progress_display_node_id"] = str(
            payload.get("component_id", "modal-subgraph")
        )
    if item_index is not None:
        phase_payload["map_item_index"] = int(item_index)
    return phase_payload


def _split_phase_outputs(
    phase_outputs: tuple[Any, ...],
    boundary_outputs: list[dict[str, Any]],
    internal_output_names: set[str],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Split one phase result tuple into bridge values and external outputs."""
    internal_outputs: dict[str, Any] = {}
    external_outputs: list[Any] = []
    for boundary_output, output_value in zip(
        boundary_outputs, phase_outputs, strict=True
    ):
        output_name = str(boundary_output.get("proxy_output_name") or "")
        if output_name in internal_output_names:
            internal_outputs[output_name] = output_value
            continue
        external_outputs.append(output_value)
    return internal_outputs, tuple(external_outputs)


def _execute_mapped_subgraph_payload(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> tuple[Any, ...]:
    """Execute one mapped payload locally using explicit static and mapped phases."""
    mapped_input = payload.get("mapped_input") or {}
    mapped_input_name = str(mapped_input.get("proxy_input_name") or "")
    if not mapped_input_name:
        raise ModalRemoteInvocationError(
            "Mapped remote payloads must define mapped_input.proxy_input_name."
        )
    if mapped_input_name not in hydrated_inputs:
        raise KeyError(
            f"Mapped remote payload input {mapped_input_name!r} was not provided."
        )

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
    static_phase_payload: dict[str, Any] | None = None
    if payload.get("static_phase") is not None:
        static_phase_payload = _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
            suppress_status_stream=True,
        )
        static_phase_payload.pop("clear_remote_session", None)
    elif payload.get("static_execute_node_ids"):
        static_phase_payload = _build_static_mapped_payload(payload)

    if static_phase_payload is not None:
        if static_phase_payload.get("execute_node_ids"):
            logger.info(
                "Executing static mapped phase for component=%s with execute nodes=%s.",
                payload.get("component_id"),
                static_phase_payload.get("execute_node_ids", []),
            )
            static_phase_outputs = _execute_subgraph_prompt(
                static_phase_payload,
                dict(broadcast_inputs),
                node_mapping,
            )
            bridge_inputs, static_outputs = _split_phase_outputs(
                static_phase_outputs,
                list(static_phase_payload.get("boundary_outputs", [])),
                bridge_output_names,
            )
            broadcast_inputs.update(bridge_inputs)

    total_items = len(mapped_items)
    _emit_local_mapped_progress(payload, 0, total_items)
    per_item_outputs: list[tuple[Any, ...]] = []
    for item_index, item_value in enumerate(mapped_items):
        if _local_processing_interrupted():
            _raise_local_interrupt()
        if payload.get("mapped_phase") is not None:
            item_payload = _build_phase_subgraph_payload(
                payload,
                "mapped_phase",
                f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
                suppress_status_stream=True,
                lane_id="0",
                item_index=item_index,
            )
        else:
            item_payload = _build_mapped_item_payload(payload, item_index, 0)
        item_inputs = dict(broadcast_inputs)
        item_inputs[mapped_input_name] = item_value
        logger.info(
            "Executing mapped item %d/%d for component=%s with execute nodes=%s.",
            item_index + 1,
            total_items,
            payload.get("component_id"),
            item_payload.get("execute_node_ids", []),
        )
        per_item_outputs.append(
            _execute_subgraph_prompt(
                item_payload,
                item_inputs,
                node_mapping,
            )
        )
        _emit_local_mapped_progress(payload, item_index + 1, total_items)

    if payload.get("mapped_phase") is not None:
        mapped_phase_payload = _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::mapped",
        )
    else:
        mapped_phase_payload = {
            **payload,
            "boundary_outputs": [
                boundary_output
                for boundary_output in payload.get("boundary_outputs", [])
                if _is_mapped_boundary_output(boundary_output, payload)
            ],
        }
    mapped_outputs = _aggregate_mapped_outputs(
        per_item_outputs,
        {
            **payload,
            "boundary_outputs": list(mapped_phase_payload.get("boundary_outputs", [])),
        },
    )
    return _merge_static_and_mapped_outputs(
        static_outputs=static_outputs,
        mapped_outputs=mapped_outputs,
        payload=payload,
    )


def _build_mapped_item_payload(
    payload: dict[str, Any],
    item_index: int,
    lane_index: int,
) -> dict[str, Any]:
    """Return one per-item subgraph payload derived from a mapped remote component payload."""
    if _mapped_phase_definition(payload, "mapped_phase") is not None:
        return _build_phase_subgraph_payload(
            payload,
            "mapped_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}",
            suppress_status_stream=True,
            lane_id=str(lane_index),
            item_index=item_index,
        )
    item_payload = copy.deepcopy(payload)
    item_payload["payload_kind"] = "subgraph"
    item_payload[
        "component_id"
    ] = f"{payload.get('component_id', 'modal-subgraph')}::item:{item_index}"
    item_payload["mapped_input"] = None
    item_payload["suppress_status_stream"] = True
    item_payload["map_item_index"] = item_index
    item_payload["mapped_progress_lane_id"] = str(lane_index)
    item_payload["mapped_progress_display_node_id"] = str(
        payload.get("component_id", "modal-subgraph")
    )
    item_payload.pop("clear_remote_session", None)
    item_payload["execute_node_ids"] = list(
        payload.get("mapped_execute_node_ids") or payload.get("execute_node_ids", [])
    )
    item_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in payload.get("boundary_outputs", [])
        if _is_mapped_boundary_output(boundary_output, payload)
    ]
    return item_payload


def _aggregate_mapped_outputs(
    per_item_outputs: list[tuple[Any, ...]],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble ordered per-item outputs from mapped execution into one proxy result tuple."""
    if not per_item_outputs:
        raise ValueError("Mapped execution produced no per-item outputs to aggregate.")

    output_count = len(per_item_outputs[0])
    if any(len(item_outputs) != output_count for item_outputs in per_item_outputs):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced inconsistent output arity."
        )

    boundary_outputs = list(payload.get("boundary_outputs", []))
    aggregated_outputs: list[Any] = []
    for output_index in range(output_count):
        boundary_output = (
            boundary_outputs[output_index]
            if output_index < len(boundary_outputs)
            else {}
        )
        aggregated_outputs.append(
            join_mapped_values_for_scheduler(
                [item_outputs[output_index] for item_outputs in per_item_outputs],
                io_type=str(boundary_output.get("io_type", "*")),
                is_list=bool(boundary_output.get("is_list", False)),
                scheduler_is_list=bool(boundary_output.get("scheduler_is_list", False)),
            )
        )
    return tuple(aggregated_outputs)


def _build_remote_session_cleanup_payload(
    payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a dedicated cleanup payload that clears one shared remote session once."""
    remote_session = payload.get("remote_session")
    if not bool(
        payload.get("clear_remote_session")
    ) or not is_remote_session_handle_payload(remote_session):
        return None
    return {
        "payload_kind": "subgraph",
        "component_id": f"{payload.get('component_id', 'modal-subgraph')}::cleanup",
        **_shared_subgraph_payload_fields(payload),
        "component_node_ids": [],
        "subgraph_prompt": {},
        "boundary_inputs": [],
        "boundary_outputs": [],
        "execute_node_ids": [],
        "remote_session": copy.deepcopy(remote_session),
        "clear_remote_session": True,
        "suppress_status_stream": True,
        "terminate_container_on_error": False,
    }


def _is_mapped_boundary_output(
    boundary_output: dict[str, Any], payload: dict[str, Any]
) -> bool:
    """Return whether one boundary output belongs to the mapped per-item branch."""
    mapped_output = boundary_output.get("mapped_output")
    if mapped_output is not None:
        return bool(mapped_output)
    return bool(payload.get("mapped_input")) and not bool(
        payload.get("static_execute_node_ids")
    )


def _build_static_mapped_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the one-time static subgraph payload for a hybrid mapped component."""
    if _mapped_phase_definition(payload, "static_phase") is not None:
        return _build_phase_subgraph_payload(
            payload,
            "static_phase",
            f"{payload.get('component_id', 'modal-subgraph')}::static",
            suppress_status_stream=True,
        )
    static_payload = copy.deepcopy(payload)
    static_payload["payload_kind"] = "subgraph"
    static_payload[
        "component_id"
    ] = f"{payload.get('component_id', 'modal-subgraph')}::static"
    static_payload["mapped_input"] = None
    static_payload["suppress_status_stream"] = True
    static_payload.pop("clear_remote_session", None)
    static_payload["execute_node_ids"] = list(
        payload.get("static_execute_node_ids") or []
    )
    static_payload["boundary_outputs"] = [
        copy.deepcopy(boundary_output)
        for boundary_output in payload.get("boundary_outputs", [])
        if not _is_mapped_boundary_output(boundary_output, payload)
    ]
    return static_payload


def _merge_static_and_mapped_outputs(
    *,
    static_outputs: tuple[Any, ...],
    mapped_outputs: tuple[Any, ...],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    """Reassemble one hybrid mapped component's static and mapped outputs in original order."""
    combined_outputs: list[Any] = []
    static_output_index = 0
    mapped_output_index = 0

    for boundary_output in payload.get("boundary_outputs", []):
        if _is_mapped_boundary_output(boundary_output, payload):
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

    if static_output_index != len(static_outputs) or mapped_output_index != len(
        mapped_outputs
    ):
        raise RemoteSubgraphExecutionError(
            "Mapped remote execution produced extra outputs that did not match the declared boundary outputs."
        )
    return tuple(combined_outputs)


def _emit_local_mapped_progress(
    payload: dict[str, Any],
    completed_items: int,
    total_items: int,
) -> None:
    """Emit one aggregate mapped-execution progress update for the component representative node."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    display_node_id = str(
        payload.get("mapped_progress_display_node_id")
        or payload.get("component_id")
        or ""
    )
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=float(completed_items),
        max_value=float(total_items),
        display_node_id=display_node_id,
        aggregate_only=True,
    )


def _mapped_progress_owner_component_id(payload: dict[str, Any]) -> str | None:
    """Return the stable component id used to track one mapped worker lane locally."""
    owner_component_id = payload.get(
        "mapped_progress_display_node_id", payload.get("component_id")
    )
    if owner_component_id is None:
        return None
    owner_component = str(owner_component_id)
    return owner_component or None


def _remember_mapped_lane_node_id(
    payload: dict[str, Any], lane_id: str, node_id: str
) -> None:
    """Remember the last real node id that emitted progress for one mapped worker lane."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    owner_component_id = _mapped_progress_owner_component_id(payload)
    if not prompt_id or not owner_component_id or not node_id:
        return
    with _MAPPED_PROGRESS_NODE_IDS_LOCK:
        _MAPPED_PROGRESS_NODE_IDS[(prompt_id, owner_component_id, lane_id)] = node_id


def _pop_mapped_lane_node_id(payload: dict[str, Any], lane_id: str) -> str | None:
    """Forget and return the last real node id that emitted progress for one mapped worker lane."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    owner_component_id = _mapped_progress_owner_component_id(payload)
    if not prompt_id or not owner_component_id:
        return None
    with _MAPPED_PROGRESS_NODE_IDS_LOCK:
        return _MAPPED_PROGRESS_NODE_IDS.pop(
            (prompt_id, owner_component_id, lane_id), None
        )


def _clear_local_mapped_lane_progress(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int,
) -> None:
    """Remove one mapped worker lane from the local node overlay."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    lane_id = str(lane_index)
    display_node_id = _pop_mapped_lane_node_id(payload, lane_id) or str(
        payload.get("component_id") or ""
    )
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=lane_id,
        clear=True,
        item_index=item_index,
    )


def _emit_local_mapped_lane_progress_start(
    payload: dict[str, Any],
    lane_index: int,
    item_index: int | None = None,
) -> None:
    """Create or reset one mapped worker lane before remote progress begins arriving."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data.get("client_id"))
        if extra_data.get("client_id") is not None
        else None
    )
    display_node_id = str(payload.get("component_id") or "")
    if not prompt_id or not client_id or not display_node_id:
        return
    _emit_local_modal_progress(
        prompt_id=prompt_id,
        client_id=client_id,
        node_id=display_node_id,
        value=0.0,
        max_value=1.0,
        display_node_id=display_node_id,
        lane_id=str(lane_index),
        item_index=item_index,
        setup_only=True,
    )


def _implicit_batch_preserving_targets(
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
) -> list[str]:
    """Return target sockets that must receive one complete tensor batch."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return []

    preserving_targets: list[str] = []
    for target in boundary_input.get("targets", []):
        node_id = str(target.get("node_id") or "")
        input_name = str(target.get("input_name") or "")
        prompt_node = prompt.get(node_id)
        if not node_id or not input_name or not isinstance(prompt_node, dict):
            continue
        class_type = str(prompt_node.get("class_type") or "")
        if (class_type, input_name) in _IMPLICIT_BATCH_PRESERVING_TARGETS:
            preserving_targets.append(f"{node_id}.{input_name}")
    return sorted(preserving_targets)


def _split_batch_boundary_inputs(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
) -> tuple[dict[str, list[Any]], int] | None:
    """Return zipped per-item boundary inputs when an ordinary subgraph receives batched values."""
    implicitly_batchable_scalar_io_types = frozenset(
        {"BOOLEAN", "FLOAT", "INT", "STRING"}
    )
    implicitly_batchable_transport_io_types = frozenset(
        {"IMAGE", "LATENT", "MASK", "NOISE", "SIGMAS"}
    )
    split_inputs: dict[str, list[Any]] = {}

    def is_session_ref_list(value: Any) -> bool:
        """Return whether `value` is a non-empty list of remote session refs."""
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(
                is_remote_session_value_ref_payload(item)
                or is_remote_session_bridge_ref_payload(item)
                for item in value
            )
        )

    def unwrap_singleton_session_ref_list(value: Any) -> Any:
        """Unwrap Comfy's output-list wrapper around mapped remote session refs."""
        if (
            isinstance(value, list)
            and len(value) == 1
            and is_session_ref_list(value[0])
        ):
            return value[0]
        return value

    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if not proxy_input_name or proxy_input_name not in hydrated_inputs:
            continue
        input_value = unwrap_singleton_session_ref_list(
            hydrated_inputs[proxy_input_name]
        )
        io_type = _implicit_batch_boundary_effective_io_type(
            payload=payload,
            boundary_input=boundary_input,
            input_value=input_value,
        )
        preserving_targets = _implicit_batch_preserving_targets(
            payload,
            boundary_input,
        )
        if preserving_targets:
            logger.info(
                "Skipping implicit batch split for boundary input %s io_type=%s "
                "because target sockets consume the complete tensor batch: %s.",
                proxy_input_name,
                io_type,
                preserving_targets,
            )
            continue
        input_is_session_ref_list = is_session_ref_list(input_value)
        if (
            isinstance(input_value, list)
            and not is_mapped_output_value(input_value)
            and not input_is_session_ref_list
            and io_type
            not in (
                implicitly_batchable_scalar_io_types
                | implicitly_batchable_transport_io_types
            )
        ):
            logger.info(
                "Skipping implicit batch split for boundary input %s io_type=%s because list-backed non-scalar values stay broadcast.",
                proxy_input_name,
                io_type,
            )
            continue
        try:
            items = split_mapped_value(
                input_value,
                io_type,
            )
        except (TypeError, ValueError):
            continue
        if len(items) <= 1:
            continue
        split_inputs[proxy_input_name] = items

    if not split_inputs:
        return None

    item_counts = {input_name: len(items) for input_name, items in split_inputs.items()}
    unique_counts = set(item_counts.values())
    if len(unique_counts) != 1:
        raise ModalRemoteInvocationError(
            "Implicit Modal batch boundary inputs must all have the same item count. "
            f"Received counts: {item_counts!r}"
        )
    return split_inputs, next(iter(unique_counts))


def _is_latent_like_mapping(value: Any) -> bool:
    """Return whether one runtime value looks like a ComfyUI LATENT mapping."""
    return isinstance(value, Mapping) and "samples" in value


def _list_is_latent_like_batch(value: Any) -> bool:
    """Return whether one runtime value is a list of LATENT-like mappings."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_latent_like_mapping(item) for item in value)
    )


def _implicit_batch_boundary_target_input_types(
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
) -> set[str]:
    """Return the declared input types of one boundary input's target sockets."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return set()

    try:
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    except ModuleNotFoundError:
        logger.debug(
            "Skipping target input type discovery for implicit batching because ComfyUI nodes are unavailable."
        )
        return set()

    target_input_types: set[str] = set()
    for target in boundary_input.get("targets", []):
        node_id = str(target.get("node_id") or "")
        input_name = str(target.get("input_name") or "")
        if not node_id or not input_name:
            continue
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, dict):
            continue
        node_class = resolved_node_mapping.get(str(prompt_node.get("class_type")))
        if node_class is None:
            continue
        prompt_inputs = prompt_node.get("inputs") or {}
        declared_type = _node_input_type_map(node_class, prompt_inputs).get(input_name)
        if isinstance(declared_type, str) and declared_type:
            target_input_types.add(declared_type)
    return target_input_types


def _implicit_batch_boundary_effective_io_type(
    *,
    payload: dict[str, Any],
    boundary_input: dict[str, Any],
    input_value: Any,
) -> str:
    """Return the best effective io_type for implicit batching of one boundary input."""
    declared_io_type = str(boundary_input.get("io_type", "*"))
    if declared_io_type != "*":
        return declared_io_type

    if _list_is_latent_like_batch(input_value):
        return "LATENT"

    target_input_types = _implicit_batch_boundary_target_input_types(
        payload, boundary_input
    )
    if len(target_input_types) == 1:
        return next(iter(target_input_types))
    return declared_io_type


def _implicit_batch_input_is_list_target_node_ids(
    payload: dict[str, Any],
    split_inputs: dict[str, list[Any]],
) -> list[str]:
    """Return split-boundary target nodes that must consume the full list in one execution."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return []

    try:
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
    except ModuleNotFoundError:
        logger.debug(
            "Skipping INPUT_IS_LIST detection for implicit batching because ComfyUI nodes are unavailable."
        )
        return []
    target_node_ids: set[str] = set()
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if proxy_input_name not in split_inputs:
            continue
        for target in boundary_input.get("targets", []):
            target_node_id = str(target.get("node_id") or "")
            if not target_node_id:
                continue
            prompt_node = prompt.get(target_node_id)
            if prompt_node is None:
                continue
            class_type = str(prompt_node.get("class_type"))
            node_class = resolved_node_mapping.get(class_type)
            if node_class is None:
                continue
            if bool(getattr(node_class, "INPUT_IS_LIST", False)):
                target_node_ids.add(target_node_id)
    return sorted(target_node_ids)


def _partition_implicit_batched_execute_nodes(
    payload: dict[str, Any],
    split_inputs: dict[str, list[Any]],
) -> tuple[list[str], list[str]]:
    """Split one implicitly batched subgraph into static and per-item execute targets."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        execute_node_ids = [
            str(node_id) for node_id in payload.get("execute_node_ids", [])
        ]
        return [], execute_node_ids

    batched_target_node_ids: set[str] = set()
    for boundary_input in payload.get("boundary_inputs", []):
        proxy_input_name = str(boundary_input.get("proxy_input_name") or "")
        if proxy_input_name not in split_inputs:
            continue
        for target in boundary_input.get("targets", []):
            target_node_id = target.get("node_id")
            if target_node_id is not None:
                batched_target_node_ids.add(str(target_node_id))

    execute_node_ids = [str(node_id) for node_id in payload.get("execute_node_ids", [])]
    static_execute_node_ids: list[str] = []
    mapped_execute_node_ids: list[str] = []
    for execute_node_id in execute_node_ids:
        required_node_ids = set(
            _resolve_required_subgraph_nodes(
                prompt=prompt,
                execute_node_ids=[execute_node_id],
            )
        )
        if required_node_ids & batched_target_node_ids:
            mapped_execute_node_ids.append(execute_node_id)
            continue
        static_execute_node_ids.append(execute_node_id)

    if not mapped_execute_node_ids and execute_node_ids:
        logger.warning(
            "Implicitly batched Modal component=%s had batched inputs %s but no execute target depended on them; "
            "falling back to per-item execution for all execute nodes.",
            payload.get("component_id"),
            sorted(split_inputs),
        )
        return [], execute_node_ids

    logger.info(
        "Partitioned implicitly batched Modal component=%s into static execute nodes=%s and mapped execute nodes=%s.",
        payload.get("component_id"),
        static_execute_node_ids,
        mapped_execute_node_ids,
    )
    return static_execute_node_ids, mapped_execute_node_ids


def _annotate_implicit_batched_boundary_outputs(
    payload: dict[str, Any],
    mapped_execute_node_ids: list[str],
) -> list[dict[str, Any]]:
    """Mark which boundary outputs belong to the per-item branch of an implicitly batched subgraph."""
    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict) or not mapped_execute_node_ids:
        return [
            copy.deepcopy(boundary_output)
            for boundary_output in payload.get("boundary_outputs", [])
        ]

    mapped_required_node_ids = set(
        _resolve_required_subgraph_nodes(
            prompt=prompt,
            execute_node_ids=[str(node_id) for node_id in mapped_execute_node_ids],
        )
    )
    annotated_outputs: list[dict[str, Any]] = []
    for boundary_output in payload.get("boundary_outputs", []):
        annotated_output = copy.deepcopy(boundary_output)
        annotated_output["mapped_output"] = (
            str(boundary_output.get("node_id")) in mapped_required_node_ids
        )
        annotated_outputs.append(annotated_output)
    return annotated_outputs


def _log_detached_mapped_lane_result(
    task: asyncio.Task[None],
    *,
    component_id: str,
    lane_index: int,
) -> None:
    """Consume and log any late failure from one detached mapped worker lane."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        logger.warning(
            "Detached mapped worker lane=%d for component=%s finished with an unawaited failure: %s",
            lane_index,
            component_id,
            exc,
        )


async def _invoke_implicitly_mapped_subgraph_async(
    payload: dict[str, Any], kwargs_payload: bytes
) -> bytes:
    """Fan out one ordinary subgraph payload when batchable boundary inputs arrive zipped."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    split_batch_inputs = _split_batch_boundary_inputs(payload, hydrated_inputs)
    if split_batch_inputs is None:
        raise ModalRemoteInvocationError(
            "Implicit mapped subgraph execution requires at least one batched boundary input."
        )

    split_inputs, total_items = split_batch_inputs
    input_is_list_target_node_ids = _implicit_batch_input_is_list_target_node_ids(
        payload,
        split_inputs,
    )
    if input_is_list_target_node_ids:
        logger.info(
            "Executing implicitly batched Modal component=%s as one ordinary subgraph because split boundary inputs target INPUT_IS_LIST nodes=%s.",
            payload.get("component_id"),
            input_is_list_target_node_ids,
        )
        return await invoke_remote_engine_async(
            payload,
            kwargs_payload,
            allow_implicit_mapping=False,
        )

    parallelism, refined_prompt_warmup_target = boost_mapped_component_warmup(
        payload,
        total_items=total_items,
        reason="implicit_mapped_component_exact_parallelism",
    )
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    if prompt_id is not None:
        await _await_prompt_warmup_slots(
            prompt_id,
            list(range(refined_prompt_warmup_target)),
            _prompt_warmup_head_start_seconds(),
        )
    logger.info(
        "Scheduling implicitly mapped Modal component=%s for %d item(s) with local parallelism=%d prompt_warmup_target=%d across inputs=%s.",
        payload.get("component_id"),
        total_items,
        parallelism,
        refined_prompt_warmup_target,
        sorted(split_inputs),
    )
    _emit_local_mapped_progress(payload, 0, total_items)

    broadcast_inputs = {
        input_name: value
        for input_name, value in hydrated_inputs.items()
        if input_name not in split_inputs
    }
    (
        static_execute_node_ids,
        mapped_execute_node_ids,
    ) = _partition_implicit_batched_execute_nodes(
        payload,
        split_inputs,
    )
    hybrid_payload = copy.deepcopy(payload)
    hybrid_payload["static_execute_node_ids"] = static_execute_node_ids
    hybrid_payload["mapped_execute_node_ids"] = mapped_execute_node_ids
    hybrid_payload["boundary_outputs"] = _annotate_implicit_batched_boundary_outputs(
        payload,
        mapped_execute_node_ids,
    )
    cleanup_payload = _build_remote_session_cleanup_payload(hybrid_payload)
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    use_seeded_remote_lanes = (
        execution_mode != "local"
        and modal is not None
        and _payload_remote_session_handle(hybrid_payload) is not None
        and _mapped_phase_definition(hybrid_payload, "static_phase") is not None
    )

    static_outputs: tuple[Any, ...] = ()
    lane_remote_engines: list[Any] = []
    seeded_lane_indices: set[int] = set()
    skip_cleanup_after_interrupt = False
    try:
        if use_seeded_remote_lanes:
            lane_remote_engines = [
                _lookup_deployed_remote_engine(
                    hybrid_payload,
                    affinity_key_override=_mapped_lane_affinity_key(
                        hybrid_payload, lane_index
                    ),
                )
                for lane_index in range(parallelism)
            ]
            logger.info(
                "Seeding %d mapped worker lane(s) for component=%s and allowing ready lanes to start item dispatch independently.",
                parallelism,
                payload.get("component_id"),
            )

            async def seed_lane(lane_index: int) -> None:
                """Run the static bridge-producing phase once on one mapped worker lane."""
                lane_seed_payload = _build_phase_subgraph_payload(
                    hybrid_payload,
                    "static_phase",
                    f"{payload.get('component_id', 'modal-subgraph')}::seed:{lane_index}",
                    suppress_status_stream=True,
                )
                lane_seed_payload.pop("clear_remote_session", None)
                await _invoke_bound_remote_engine_async(
                    lane_remote_engines[lane_index],
                    lane_seed_payload,
                    serialize_node_inputs(broadcast_inputs),
                )
                seeded_lane_indices.add(lane_index)

        elif static_execute_node_ids:
            static_response = await invoke_remote_engine_async(
                _build_static_mapped_payload(hybrid_payload),
                serialize_node_inputs(broadcast_inputs),
            )
            static_outputs = deserialize_node_outputs(static_response)
        else:
            lane_remote_engines = []

        per_item_outputs: list[tuple[Any, ...] | None] = [None] * total_items
        completed_items = 0
        all_items_completed = asyncio.Event()
        stop_dispatch_requested = asyncio.Event()
        worker_failure: asyncio.Future[
            BaseException
        ] = asyncio.get_running_loop().create_future()
        item_queue: asyncio.Queue[int | None] = asyncio.Queue()
        for item_index in range(total_items):
            item_queue.put_nowait(item_index)
        for _ in range(parallelism):
            item_queue.put_nowait(None)

        def request_interrupt_stop() -> None:
            """Stop queued mapped item dispatch and suppress cleanup after interruption."""
            nonlocal skip_cleanup_after_interrupt
            skip_cleanup_after_interrupt = True
            stop_dispatch_requested.set()

        def raise_if_local_interrupted() -> None:
            """Raise the native ComfyUI interrupt after marking mapped dispatch as stopped."""
            if _local_processing_interrupted():
                request_interrupt_stop()
                _raise_local_interrupt()

        async def run_worker(lane_index: int) -> None:
            """Execute queued implicit mapped items through one stable local worker lane."""
            nonlocal completed_items
            try:
                raise_if_local_interrupted()
                if use_seeded_remote_lanes:
                    _emit_local_mapped_lane_progress_start(payload, lane_index)
                    await seed_lane(lane_index)
                while True:
                    if stop_dispatch_requested.is_set():
                        return
                    item_index = await item_queue.get()
                    if item_index is None:
                        return
                    if stop_dispatch_requested.is_set():
                        return
                    raise_if_local_interrupted()
                    try:
                        item_payload = _build_mapped_item_payload(
                            hybrid_payload, item_index, lane_index
                        )
                        item_inputs = dict(broadcast_inputs)
                        for input_name, items in split_inputs.items():
                            item_inputs[input_name] = items[item_index]
                        if use_seeded_remote_lanes:
                            item_response = await _invoke_bound_remote_engine_async(
                                lane_remote_engines[lane_index],
                                item_payload,
                                serialize_node_inputs(item_inputs),
                            )
                        else:
                            item_response = await invoke_remote_engine_async(
                                item_payload,
                                serialize_node_inputs(item_inputs),
                            )
                        raise_if_local_interrupted()
                        per_item_outputs[item_index] = deserialize_node_outputs(
                            item_response
                        )
                        completed_items += 1
                        _emit_local_mapped_progress(
                            payload, completed_items, total_items
                        )
                        if completed_items >= total_items:
                            all_items_completed.set()
                    finally:
                        _clear_local_mapped_lane_progress(
                            payload, lane_index, item_index
                        )
            except BaseException as exc:
                if all_items_completed.is_set():
                    logger.info(
                        "Ignoring late mapped worker failure after component=%s already completed all items lane=%d: %s",
                        payload.get("component_id"),
                        lane_index,
                        exc,
                    )
                    return
                stop_dispatch_requested.set()
                if _exception_indicates_interruption(exc):
                    request_interrupt_stop()
                if not worker_failure.done():
                    worker_failure.set_result(exc)
                raise

        tasks = [
            asyncio.create_task(run_worker(lane_index))
            for lane_index in range(parallelism)
        ]
        for lane_index, task in enumerate(tasks):
            task.add_done_callback(
                lambda completed_task, lane_index=lane_index: _log_detached_mapped_lane_result(
                    completed_task,
                    component_id=str(payload.get("component_id", "modal-subgraph")),
                    lane_index=lane_index,
                )
            )
        completion_task = asyncio.create_task(all_items_completed.wait())
        try:
            while not all_items_completed.is_set():
                wait_set: set[asyncio.Task[Any] | asyncio.Future[BaseException]] = {
                    completion_task,
                    worker_failure,
                    *tasks,
                }
                done, _ = await asyncio.wait(
                    wait_set, return_when=asyncio.FIRST_COMPLETED
                )
                if completion_task in done or all_items_completed.is_set():
                    break
                if worker_failure in done:
                    stop_dispatch_requested.set()
                    raise worker_failure.result()
                for task in done:
                    if task in tasks:
                        task_exc = task.exception()
                        if task_exc is not None:
                            stop_dispatch_requested.set()
                            raise task_exc
            if not all(item_outputs is not None for item_outputs in per_item_outputs):
                await asyncio.gather(*tasks)
        except BaseException as exc:
            stop_dispatch_requested.set()
            if _exception_indicates_interruption(exc):
                skip_cleanup_after_interrupt = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            completion_task.cancel()
            await asyncio.gather(completion_task, return_exceptions=True)

        return serialize_node_outputs(
            _merge_static_and_mapped_outputs(
                static_outputs=static_outputs,
                mapped_outputs=_aggregate_mapped_outputs(
                    [
                        item_outputs
                        for item_outputs in per_item_outputs
                        if item_outputs is not None
                    ],
                    {
                        **hybrid_payload,
                        "boundary_outputs": [
                            boundary_output
                            for boundary_output in hybrid_payload.get(
                                "boundary_outputs", []
                            )
                            if _is_mapped_boundary_output(
                                boundary_output, hybrid_payload
                            )
                        ],
                    },
                ),
                payload=hybrid_payload,
            )
        )
    finally:
        if cleanup_payload is not None and not skip_cleanup_after_interrupt:
            if use_seeded_remote_lanes and lane_remote_engines:
                await asyncio.gather(
                    *(
                        _invoke_bound_remote_engine_async(
                            lane_remote_engines[lane_index],
                            {
                                **cleanup_payload,
                                "component_id": (
                                    f"{payload.get('component_id', 'modal-subgraph')}::cleanup:{lane_index}"
                                ),
                            },
                            serialize_node_inputs({}),
                        )
                        for lane_index in sorted(seeded_lane_indices)
                    )
                )
            else:
                await invoke_remote_engine_async(
                    cleanup_payload,
                    serialize_node_inputs({}),
                )
        elif cleanup_payload is not None:
            logger.info(
                "Skipping remote-session cleanup for implicitly mapped Modal component=%s because execution was interrupted.",
                payload.get("component_id"),
            )


async def _invoke_mapped_remote_engine_async(
    payload: dict[str, Any], kwargs_payload: bytes
) -> bytes:
    """Execute one mapped payload locally using explicit static and mapped phases."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    return await asyncio.to_thread(
        lambda: serialize_node_outputs(
            _execute_mapped_subgraph_payload(
                payload,
                hydrated_inputs,
            )
        )
    )




def _emit_local_llm_staging_progress(
    payload: Mapping[str, Any],
    stage_event: Mapping[str, Any],
) -> None:
    """Render one CPU ModelStager update on its actual LLM node bars."""
    prompt_id = (
        str(payload["prompt_id"]) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data["client_id"])
        if isinstance(extra_data, Mapping) and extra_data.get("client_id") is not None
        else None
    )
    node_ids_by_reference = llm_model_reference_node_ids_from_payload(payload)
    model_reference = str(stage_event.get("model_reference") or "").strip()
    node_ids = node_ids_by_reference.get(model_reference, ())
    if not node_ids:
        node_ids = tuple(
            sorted(
                {
                    node_id
                    for reference_node_ids in node_ids_by_reference.values()
                    for node_id in reference_node_ids
                }
            )
        )
    if not node_ids:
        component_id = str(payload.get("component_id") or "")
        node_ids = (component_id,) if component_id else ()
    if not node_ids:
        return
    maximum = stage_event.get("max")
    for node_id in node_ids:
        _emit_local_modal_progress(
            prompt_id=prompt_id,
            client_id=client_id,
            node_id=node_id,
            value=float(stage_event.get("value") or 0.0),
            max_value=float(maximum) if maximum is not None else 1.0,
            stage=str(stage_event.get("stage") or "staging"),
            message=str(stage_event.get("message") or "Staging LLM snapshot"),
            unit=(
                str(stage_event["unit"])
                if stage_event.get("unit") is not None
                else None
            ),
            indeterminate=bool(stage_event.get("indeterminate", False)),
            pre_gpu=True,
        )


def _read_modal_stage_events(
    stage_events: Iterable[Any],
    output: queue.Queue[Any],
) -> None:
    """Read a blocking Modal generator while exposing controller timeouts."""
    try:
        for event in stage_events:
            output.put(event)
    except Exception as exc:
        output.put(_ModalStageStreamFailure(exc))
    finally:
        output.put(_MODAL_STAGE_STREAM_END)


def _close_modal_stage_events(stage_events: Iterable[Any]) -> None:
    """Ask Modal to cancel a stage generator that stopped reporting progress."""
    close = getattr(stage_events, "close", None)
    if not callable(close):
        return
    try:
        close()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Unable to close stalled Modal model staging stream: %s", exc)


def _bounded_modal_stage_events(stage_events: Iterable[Any]) -> Iterator[Any]:
    """Yield Modal staging events with a bounded interval between updates."""
    try:
        timeout_seconds = staging_no_progress_timeout_seconds()
    except ValueError:
        _close_modal_stage_events(stage_events)
        raise
    output: queue.Queue[Any] = queue.Queue()
    reader = threading.Thread(
        target=_read_modal_stage_events,
        args=(stage_events, output),
        name="modal-llm-stage-stream",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            try:
                item = output.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                _close_modal_stage_events(stage_events)
                raise ModalRemoteInvocationError(
                    "Modal model staging produced no progress for "
                    f"{timeout_seconds:.0f} seconds; the staging call was cancelled."
                ) from exc
            if item is _MODAL_STAGE_STREAM_END:
                return
            if isinstance(item, _ModalStageStreamFailure):
                raise item.error
            yield item
    finally:
        reader.join(timeout=1.0)


def _ensure_llm_profiles_staged(
    payload: dict[str, Any],
    deployment_app_name: str,
) -> None:
    """Stage every LLM profile in a payload on a CPU worker before GPU dispatch."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")
    model_references = llm_model_references_from_payload(payload)
    if not model_references:
        return
    with _STAGED_LLM_PROFILES_LOCK:
        if not _STAGED_LLM_PROFILES:
            _STAGED_LLM_PROFILE_RESULTS.clear()
        missing_model_references = [
            reference
            for reference in model_references
            if (deployment_app_name, reference) not in _STAGED_LLM_PROFILE_RESULTS
        ]
        if missing_model_references:
            resolved_profiles = resolved_llm_profile_payloads(
                payload,
                missing_model_references,
            )
            _emit_local_remote_startup_status(
                payload,
                phase="llm_staging",
                status_message=(
                    "Preparing LLM model snapshots on CPU; no GPU is allocated yet"
                ),
            )
            logger.info(
                "Dispatching CPU model resolution/staging app=%s models=%s "
                "before GPU component=%s.",
                deployment_app_name,
                missing_model_references,
                payload.get("component_id"),
            )
            stager_cls = modal.Cls.from_name(deployment_app_name, "ModelStager")
            stager = stager_cls()
            stage_results: list[dict[str, Any]] = []
            stage_stream = getattr(stager, "stage_profiles_stream", None)
            remote_generator = getattr(stage_stream, "remote_gen", None)
            if callable(remote_generator):
                stage_events = (
                    remote_generator(missing_model_references, resolved_profiles)
                    if resolved_profiles
                    else remote_generator(missing_model_references)
                )
                for stage_event in _bounded_modal_stage_events(stage_events):
                    if not isinstance(stage_event, Mapping):
                        continue
                    if stage_event.get("kind") == "result":
                        candidate_results = stage_event.get("results")
                        if isinstance(candidate_results, list):
                            stage_results = candidate_results
                        continue
                    if stage_event.get("kind") != "progress":
                        continue
                    _emit_local_llm_staging_progress(payload, stage_event)
            else:
                stage_results = (
                    stager.stage_profiles.remote(
                        missing_model_references,
                        resolved_profiles,
                    )
                    if resolved_profiles
                    else stager.stage_profiles.remote(missing_model_references)
                )
            confirmed_references: set[str] = set()
            for stage_result in stage_results:
                if not isinstance(stage_result, Mapping):
                    continue
                requested_reference = str(
                    stage_result.get("requested_reference")
                    or stage_result.get("profile_id")
                    or ""
                )
                profile_id = str(stage_result.get("profile_id") or "")
                revision = str(stage_result.get("revision") or "")
                if not revision and requested_reference == profile_id:
                    try:
                        revision = get_llm_profile(profile_id).revision
                    except ValueError:
                        pass
                if not requested_reference or not profile_id or not revision:
                    continue
                normalized_result = dict(stage_result)
                normalized_result["requested_reference"] = requested_reference
                normalized_result["profile_id"] = profile_id
                normalized_result["revision"] = revision
                _STAGED_LLM_PROFILE_RESULTS[
                    (deployment_app_name, requested_reference)
                ] = normalized_result
                _STAGED_LLM_PROFILE_RESULTS[
                    (deployment_app_name, profile_id)
                ] = normalized_result
                _STAGED_LLM_PROFILES.add((deployment_app_name, profile_id, revision))
                confirmed_references.add(requested_reference)
            missing_results = set(missing_model_references) - confirmed_references
            if missing_results:
                raise ModalRemoteInvocationError(
                    f"Modal ModelStager did not confirm models {sorted(missing_results)}."
                )
            downloaded_gib = sum(
                float(result.get("artifact_bytes") or 0) / 1024**3
                for result in stage_results
                if isinstance(result, Mapping) and result.get("downloaded")
            )
            _emit_local_remote_startup_status(
                payload,
                phase="llm_staged",
                status_message=(
                    f"LLM staging complete ({downloaded_gib:.1f} GiB downloaded); "
                    "starting GPU worker"
                ),
            )
        resolved_results = {
            reference: _STAGED_LLM_PROFILE_RESULTS[(deployment_app_name, reference)]
            for reference in model_references
        }
    profile_ids_by_reference = {
        reference: str(result["profile_id"])
        for reference, result in resolved_results.items()
    }
    rewrite_llm_model_references(payload, profile_ids_by_reference)
    revisions = ",".join(
        f"{result['profile_id']}:{result['revision']}"
        for result in sorted(
            resolved_results.values(),
            key=lambda value: str(value["profile_id"]),
        )
    )
    payload["requires_volume_reload"] = True
    payload["volume_reload_marker"] = hashlib.sha256(
        f"llm-profiles:{revisions}".encode("utf-8")
    ).hexdigest()
    logger.info(
        "Modal LLM models are resolved and staged for component=%s profiles=%s "
        "reload_marker=%s.",
        payload.get("component_id"),
        sorted(profile_ids_by_reference.values()),
        payload["volume_reload_marker"],
    )


def _rewrite_staged_llm_kwargs_payload(
    kwargs_payload: bytes,
    deployment_app_name: str,
) -> bytes:
    """Replace a direct node input model reference with its staged profile ID."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    if not isinstance(hydrated_inputs, Mapping):
        return kwargs_payload
    model_reference = hydrated_inputs.get("model_profile")
    if not isinstance(model_reference, str) or not model_reference.strip():
        return kwargs_payload
    normalized_reference = model_reference.strip()
    with _STAGED_LLM_PROFILES_LOCK:
        stage_result = _STAGED_LLM_PROFILE_RESULTS.get(
            (deployment_app_name, normalized_reference)
        )
    if stage_result is None:
        return kwargs_payload
    profile_id = str(stage_result.get("profile_id") or "").strip()
    if not profile_id or profile_id == normalized_reference:
        return kwargs_payload
    rewritten_inputs = dict(hydrated_inputs)
    rewritten_inputs["model_profile"] = profile_id
    logger.info(
        "Rewrote direct Modal LLM input model=%s to generated profile=%s.",
        normalized_reference,
        profile_id,
    )
    return serialize_node_inputs(rewritten_inputs)


async def _invoke_bound_remote_engine_async(
    remote_engine: Any,
    payload: dict[str, Any],
    kwargs_payload: bytes,
) -> bytes:
    """Invoke one pre-bound remote engine handle asynchronously with local interrupt mirroring."""
    logger.info(
        "Dispatching async Modal remote invocation via bound engine for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    cancellation_event = threading.Event()
    future = _remote_modal_call_executor().submit(
        _invoke_remote_engine_payload_with_recovery,
        remote_engine,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    wrapped_future = asyncio.wrap_future(future)
    interrupt_sent = False
    cancellation_started_at: float | None = None
    try:
        while True:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(wrapped_future), timeout=0.1
                )
                break
            except asyncio.TimeoutError:
                (
                    interrupt_sent,
                    cancellation_started_at,
                ) = await _handle_modal_wait_cancellation_async(
                    payload,
                    cancellation_event,
                    interrupt_sent=interrupt_sent,
                    cancellation_started_at=cancellation_started_at,
                )
                continue
    except asyncio.CancelledError:
        cancellation_event.set()
        prompt_id = (
            str(payload.get("prompt_id"))
            if payload.get("prompt_id") is not None
            else None
        )
        if prompt_id is not None:
            await request_remote_modal_prompt_interrupt_async(prompt_id)
        raise
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising async Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Async Modal remote invocation via bound engine failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Async bound-engine invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Async Modal remote invocation via bound engine completed for component=%s.",
        payload.get("component_id"),
    )
    return response














































def _build_remote_interrupt_callback(
    _remote_engine: Any, payload: dict[str, Any]
) -> Callable[[], Any] | None:
    """Return a callable that requests interruption for one active Modal payload."""
    interrupt_store = _lookup_modal_interrupt_store()
    if interrupt_store is None:
        return None
    return lambda: _request_remote_interrupt(payload)


def _invoke_remote_engine_payload(
    remote_engine: Any,
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Invoke one prepared remote engine instance with optional progress streaming."""
    _install_modal_cloud_exception_compatibility_module()
    if cancellation_event is not None and cancellation_event.is_set():
        _request_remote_interrupt(payload)
        raise ModalRemoteInvocationError(
            "Remote Modal payload dispatch was cancelled before the remote call started."
        )

    stream_method = getattr(remote_engine, "execute_payload_stream", None)
    interrupt_remote_call = _build_remote_interrupt_callback(remote_engine, payload)
    if _should_stream_remote_progress(payload) and hasattr(stream_method, "remote_gen"):
        logger.info(
            "Using streamed Modal progress path for component=%s via execute_payload_stream.remote_gen(...).",
            payload.get("component_id"),
        )
        return _invoke_remote_call_with_interrupts(
            payload=payload,
            invoke_remote_call=lambda: _consume_remote_payload_stream(
                payload,
                stream_method.remote_gen(payload, kwargs_payload),
            ),
            interrupt_remote_call=interrupt_remote_call,
            cancellation_event=cancellation_event,
        )

    if _should_stream_remote_progress(payload):
        logger.warning(
            "Streamed Modal progress is unavailable for component=%s; falling back to execute_payload.remote(...).",
            payload.get("component_id"),
        )
    return _invoke_remote_call_with_interrupts(
        payload=payload,
        invoke_remote_call=lambda: remote_engine.execute_payload.remote(
            payload, kwargs_payload
        ),
        interrupt_remote_call=interrupt_remote_call,
        cancellation_event=cancellation_event,
    )


def _fresh_llm_memory_recovery_affinity_key(payload: Mapping[str, Any]) -> str:
    """Return a one-use affinity identity that cannot select the dirty worker."""
    return (
        f"{_remote_worker_affinity_key(dict(payload))}:llm-memory-recovery:"
        f"{uuid.uuid4().hex[:12]}"
    )


def _retry_exhausted_llm_memory_on_fresh_worker(
    *,
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None,
    error: BaseException,
) -> bytes:
    """Retry one exhausted post-eviction admission on a distinct Modal worker."""
    recovery_payload = dict(payload)
    if exhausted_recovery_used_vllm_throughput(error):
        recovery_payload[LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY] = True
    recovery_affinity_key = _fresh_llm_memory_recovery_affinity_key(payload)
    logger.warning(
        "Modal LLM memory recovery timed out for component=%s; retrying "
        "invocation_id=%s once on fresh worker_affinity=%s.",
        payload.get("component_id"),
        payload["invocation_id"],
        recovery_affinity_key,
    )
    _emit_local_remote_startup_status(
        recovery_payload,
        phase="starting",
        status_message="Retrying LLM on a fresh Modal worker",
    )
    recovered_remote_engine = _lookup_deployed_remote_engine(
        recovery_payload,
        affinity_key_override=recovery_affinity_key,
    )
    return _invoke_remote_engine_payload(
        recovered_remote_engine,
        recovery_payload,
        kwargs_payload,
        cancellation_event,
    )


def _invoke_remote_engine_payload_with_recovery(
    remote_engine: Any,
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None,
) -> bytes:
    """Recover missing deployments and exhausted LLM workers with one safe retry."""
    payload = dict(payload)
    settings = _settings_for_payload(payload)
    if llm_model_references_from_payload(payload):
        kwargs_payload = _rewrite_staged_llm_kwargs_payload(
            kwargs_payload,
            modal_deployment_app_name(settings),
        )
    payload.setdefault("capture_remote_outputs", True)
    payload.setdefault(
        "invocation_id",
        stable_remote_invocation_id(payload, kwargs_payload),
    )
    lookup_error_types = _modal_lookup_error_types()
    invocation_error_types = tuple(
        dict.fromkeys((*lookup_error_types, RemoteSubgraphExecutionError, RuntimeError))
    )
    try:
        response = _invoke_remote_engine_payload(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )
    except invocation_error_types as exc:
        if is_llm_memory_recovery_exhausted(exc):
            response = _retry_exhausted_llm_memory_on_fresh_worker(
                payload=payload,
                kwargs_payload=kwargs_payload,
                cancellation_event=cancellation_event,
                error=exc,
            )
        elif not isinstance(exc, lookup_error_types):
            raise
        elif not settings.auto_deploy or not _is_missing_modal_deployment_error(exc):
            raise
        else:
            logger.warning(
                "Modal payload invocation failed for component=%s because the deployed app was missing at call time: %s. Recreating the app and retrying.",
                payload.get("component_id"),
                exc,
            )
            recovered_remote_engine = _auto_deploy_modal_app(payload, exc)
            _ensure_llm_profiles_staged(
                payload,
                modal_deployment_app_name(settings),
            )
            response = _invoke_remote_engine_payload(
                recovered_remote_engine,
                payload,
                kwargs_payload,
                cancellation_event,
            )
    return _materialize_remote_execution_result(response, settings=settings)


def _local_comfy_output_directory(settings: ModalSyncSettings) -> Path:
    """Return the effective output directory of the local ComfyUI process."""
    try:
        import folder_paths
    except ModuleNotFoundError as exc:
        if exc.name != "folder_paths":
            raise
    else:
        get_output_directory = getattr(folder_paths, "get_output_directory", None)
        if callable(get_output_directory):
            return Path(get_output_directory()).expanduser().resolve()
    if settings.comfyui_root is None:
        raise ModalRemoteInvocationError(
            "Modal returned ComfyUI output files, but the local ComfyUI output directory "
            "could not be resolved. Set COMFY_MODAL_COMFYUI_ROOT."
        )
    return (settings.comfyui_root / "output").resolve()


def _materialize_remote_execution_result(
    response: bytes | bytearray,
    *,
    settings: ModalSyncSettings | None = None,
) -> bytes:
    """Download bundled remote files and return the ordinary serialized node outputs."""
    result: RemoteExecutionResult = unpack_remote_execution_result(response)
    if not result.artifacts:
        return result.outputs
    resolved_settings = settings or get_settings()
    output_directory = _local_comfy_output_directory(resolved_settings)
    logger.info(
        "Materializing %d remote ComfyUI output file(s) into %s.",
        len(result.artifacts),
        output_directory,
    )
    materialized_paths = materialize_remote_output_artifacts(
        result,
        output_directory=output_directory,
        app_name=modal_deployment_app_name(resolved_settings),
    )
    logger.info(
        "Finished downloading %d remote ComfyUI output file(s): %s.",
        len(materialized_paths),
        [str(path) for path in materialized_paths],
    )
    return result.outputs


def _invoke_modal_payload_blocking(
    payload: dict[str, Any],
    kwargs_payload: bytes,
    cancellation_event: threading.Event | None = None,
) -> bytes:
    """Invoke the Modal runtime from a worker thread using deployed or ephemeral app state."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    _await_matching_speculative_prewarm(payload, cancellation_event)
    lookup_error_types = _modal_lookup_error_types()
    settings = _settings_for_payload(payload)
    deployment_app_name = modal_deployment_app_name(settings)
    if lookup_error_types:
        try:
            remote_engine = _lookup_protocol_current_remote_engine(payload)
            _ensure_llm_profiles_staged(payload, deployment_app_name)
            logger.info(
                "Using deployed Modal app %s for component %s.",
                deployment_app_name,
                payload.get("component_id"),
            )
            return _invoke_remote_engine_payload_with_recovery(
                remote_engine,
                payload,
                kwargs_payload,
                cancellation_event,
            )
        except lookup_error_types as exc:
            missing_deployment = _is_missing_modal_deployment_error(exc)
            if settings.auto_deploy and missing_deployment:
                remote_engine = _auto_deploy_modal_app(payload, exc)
                try:
                    _ensure_llm_profiles_staged(payload, deployment_app_name)
                    logger.info(
                        "Using auto-deployed Modal app %s for component %s.",
                        deployment_app_name,
                        payload.get("component_id"),
                    )
                    return _invoke_remote_engine_payload_with_recovery(
                        remote_engine,
                        payload,
                        kwargs_payload,
                        cancellation_event,
                    )
                except lookup_error_types as retry_exc:
                    exc = retry_exc
                    missing_deployment = _is_missing_modal_deployment_error(exc)
            if not missing_deployment:
                raise
            if not settings.allow_ephemeral_fallback:
                raise ModalRemoteInvocationError(
                    "Remote execution requires a deployed Modal app or a successful first-run auto-deploy. "
                    f"Lookup failed for app={deployment_app_name!r} component={payload.get('component_id')!r}: {exc}. "
                    "Ensure Modal credentials are configured so the custom node can auto-deploy, "
                    "or set COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true to allow slow ephemeral app.run() fallback behavior."
                ) from exc
            logger.warning(
                "Deployed Modal app lookup failed for app=%s component=%s: %s. Falling back to ephemeral app.run(); this creates a temporary Modal app session, not a persistent deployment or endpoint.",
                deployment_app_name,
                payload.get("component_id"),
                exc,
            )
    else:
        remote_engine = _lookup_protocol_current_remote_engine(payload)
        _ensure_llm_profiles_staged(payload, deployment_app_name)
        logger.info(
            "Using deployed Modal app %s for component %s.",
            deployment_app_name,
            payload.get("component_id"),
        )
        return _invoke_remote_engine_payload_with_recovery(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )

    if "app" not in globals() or "RemoteEngine" not in globals():
        logger.debug(
            "Local module Modal runtime objects are unavailable; loading stable cloud entry module."
        )

    with _modal_cloud_settings_override(settings):
        cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    cloud_remote_engine = getattr(cloud_module, "RemoteEngine", None)
    if cloud_app is None or cloud_remote_engine is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose app and RemoteEngine."
        )
    logger.info(
        "Starting ephemeral Modal app.run() for component %s. This does not create a persistent deployed app or web endpoint.",
        payload.get("component_id"),
    )
    run_context = cloud_app.run() if hasattr(cloud_app, "run") else nullcontext()
    with run_context:
        remote_engine = cloud_remote_engine()
        result = _invoke_remote_engine_payload_with_recovery(
            remote_engine,
            payload,
            kwargs_payload,
            cancellation_event,
        )
    logger.info(
        "Ephemeral Modal app.run() invocation completed for component %s.",
        payload.get("component_id"),
    )
    return result


def invoke_remote_engine(
    payload: dict[str, Any],
    kwargs_payload: bytes,
    *,
    allow_implicit_mapping: bool = True,
) -> bytes:
    """Invoke Modal when configured, or fall back to local in-process execution."""
    execution_mode = get_settings().execution_mode
    if payload.get("payload_kind") == "mapped_subgraph":
        if execution_mode == "local" or modal is None:
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
            return serialize_node_outputs(
                _execute_mapped_subgraph_payload(payload, hydrated_inputs)
            )
    if allow_implicit_mapping and payload.get("payload_kind") == "subgraph":
        hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        if _split_batch_boundary_inputs(payload, hydrated_inputs) is not None:
            return asyncio.run(
                _invoke_implicitly_mapped_subgraph_async(payload, kwargs_payload)
            )
    if execution_mode == "local" or modal is None:
        if payload.get("payload_kind") == "subgraph":
            return execute_subgraph_locally(payload, kwargs_payload)
        return execute_node_locally(payload, kwargs_payload)

    logger.info(
        "Dispatching Modal remote invocation for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    _emit_local_remote_dispatch_status(payload)
    cancellation_event = threading.Event()
    future = _remote_modal_call_executor().submit(
        _invoke_modal_payload_blocking,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    interrupt_sent = False
    cancellation_started_at: float | None = None
    try:
        while True:
            try:
                response = future.result(timeout=0.1)
                break
            except FutureTimeoutError:
                (
                    interrupt_sent,
                    cancellation_started_at,
                ) = _handle_modal_wait_cancellation(
                    payload,
                    cancellation_event,
                    interrupt_sent=interrupt_sent,
                    cancellation_started_at=cancellation_started_at,
                )
                continue
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Modal remote invocation failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Remote invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Modal remote invocation completed for component=%s.",
        payload.get("component_id"),
    )
    return response


async def invoke_remote_engine_async(
    payload: dict[str, Any],
    kwargs_payload: bytes,
    *,
    allow_implicit_mapping: bool = True,
) -> bytes:
    """Invoke Modal asynchronously so multiple proxy nodes can wait on remote work in parallel."""
    execution_mode = get_settings().execution_mode
    if (
        execution_mode != "local"
        and modal is not None
        and bool(payload.get("stop_local_gap_keepalive_before_remote_component"))
    ):
        _stop_local_gap_keepalive(payload, reason="next_remote_component_started")
    if payload.get("payload_kind") == "mapped_subgraph":
        if execution_mode == "local" or modal is None:
            return await _invoke_mapped_remote_engine_async(payload, kwargs_payload)
    if allow_implicit_mapping and payload.get("payload_kind") == "subgraph":
        hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        if _split_batch_boundary_inputs(payload, hydrated_inputs) is not None:
            if execution_mode != "local" and modal is not None:
                await asyncio.to_thread(
                    _lookup_protocol_current_remote_engine,
                    payload,
                )
            return await _invoke_implicitly_mapped_subgraph_async(
                payload, kwargs_payload
            )
    if execution_mode == "local" or modal is None:
        return await asyncio.to_thread(
            invoke_remote_engine,
            payload,
            kwargs_payload,
            allow_implicit_mapping=allow_implicit_mapping,
        )

    logger.info(
        "Dispatching async Modal remote invocation for component=%s payload_kind=%s.",
        payload.get("component_id"),
        payload.get("payload_kind"),
    )
    _emit_local_remote_dispatch_status(payload)
    cancellation_event = threading.Event()
    future = _remote_modal_call_executor().submit(
        _invoke_modal_payload_blocking,
        dict(payload),
        kwargs_payload,
        cancellation_event,
    )
    wrapped_future = asyncio.wrap_future(future)
    interrupt_sent = False
    cancellation_started_at: float | None = None
    try:
        while True:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(wrapped_future), timeout=0.1
                )
                break
            except asyncio.TimeoutError:
                (
                    interrupt_sent,
                    cancellation_started_at,
                ) = await _handle_modal_wait_cancellation_async(
                    payload,
                    cancellation_event,
                    interrupt_sent=interrupt_sent,
                    cancellation_started_at=cancellation_started_at,
                )
                continue
    except asyncio.CancelledError:
        cancellation_event.set()
        prompt_id = (
            str(payload.get("prompt_id"))
            if payload.get("prompt_id") is not None
            else None
        )
        if prompt_id is not None:
            await request_remote_modal_prompt_interrupt_async(prompt_id)
        raise
    except Exception:
        if cancellation_event.is_set() or _local_processing_interrupted():
            logger.info(
                "Reraising async Modal failure as a local interrupt for component=%s after cancellation.",
                payload.get("component_id"),
            )
            _raise_local_interrupt()
        logger.exception(
            "Async Modal remote invocation failed for component=%s.",
            payload.get("component_id"),
        )
        raise
    if cancellation_event.is_set() or _local_processing_interrupted():
        logger.info(
            "Async remote invocation for component=%s finished after interruption; raising local interrupt.",
            payload.get("component_id"),
        )
        _raise_local_interrupt()
    logger.info(
        "Async Modal remote invocation completed for component=%s.",
        payload.get("component_id"),
    )
    _start_local_gap_keepalive(payload)
    return response


configure_modal_warmup_hooks(
    ModalWarmupHooks(
        ensure_llm_profiles_staged=lambda payload, app_name: _ensure_llm_profiles_staged(
            payload, app_name
        ),
        mapped_execution_parallelism=lambda item_count: _mapped_execution_parallelism(
            item_count
        ),
    )
)


configure_modal_deployment_hooks(
    ModalDeploymentHooks(
        ensure_llm_profiles_staged=lambda payload, app_name: _ensure_llm_profiles_staged(
            payload, app_name
        ),
        schedule_post_deploy_runtime_seed=lambda payload: _schedule_post_deploy_runtime_seed(
            payload
        ),
        await_matching_speculative_prewarm=lambda payload, cancellation_event: _await_matching_speculative_prewarm(
            payload, cancellation_event
        ),
        prepare_snapshot_profile_fields=lambda payload: _prepare_snapshot_profile_fields(
            payload
        ),
        select_gpu_snapshot_for_profile=lambda payload, profile_key: _select_gpu_snapshot_for_profile(
            payload, profile_key
        ),
        prompt_parallelism_target=lambda payload: _prompt_parallelism_target(payload),
    )
)


if modal is not None:  # pragma: no branch - simple import-time configuration.
    settings = get_settings()
    app = modal.App(modal_deployment_app_name(settings))
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    image = modal.Image.debian_slim().pip_install(
        "torch", "safetensors", "pillow", "numpy"
    )

    @app.cls(
        gpu=settings.modal_gpu,
        volumes={settings.remote_storage_root: vol},
        scaledown_window=60,
        image=image,
    )
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""

        snapshot_profile_key: str = modal.parameter(default="")
        gpu_snapshot_enabled: bool = modal.parameter(default=False)
        worker_affinity_key: str = modal.parameter(default="worker-pool:slot:0")

        @modal.enter()
        def setup(self) -> None:
            """Prepare the container process for headless node execution."""
            logger.info(
                "RemoteEngine setup complete for snapshot_profile_key=%s gpu_snapshot_enabled=%s.",
                self.snapshot_profile_key or None,
                bool(self.gpu_snapshot_enabled),
            )

        @modal.method()
        def execute_payload(
            self, payload: dict[str, Any], kwargs_payload: bytes
        ) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            if payload.get("payload_kind") == "mapped_subgraph":
                hydrated_inputs = deserialize_node_inputs(kwargs_payload)
                return serialize_node_outputs(
                    _execute_mapped_subgraph_payload(payload, hydrated_inputs)
                )
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)

        @modal.method()
        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """No-op local warmup entrypoint for the simplified Modal runtime."""
            return {"component_id": str(payload.get("component_id") or "modal-warmup")}

        @modal.method()
        def keepalive_for_local_gap(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Return a lightweight keepalive acknowledgement for one affinity slot."""
            return {
                "component_id": str(payload.get("component_id") or "modal-keepalive"),
                "worker_affinity_key": self.worker_affinity_key,
            }

else:

    class RemoteEngine:
        """Local fallback runtime used when the Modal SDK is unavailable."""

        def __init__(
            self,
            snapshot_profile_key: str | None = None,
            gpu_snapshot_enabled: bool = False,
            worker_affinity_key: str = "worker-pool:slot:0",
        ) -> None:
            """Record the optional snapshot-profile key and GPU snapshot mode."""
            self.snapshot_profile_key = snapshot_profile_key
            self.gpu_snapshot_enabled = gpu_snapshot_enabled
            self.worker_affinity_key = worker_affinity_key

        def setup(self) -> None:
            """No-op setup for local fallback execution."""

        def execute_payload(
            self, payload: dict[str, Any], kwargs_payload: bytes
        ) -> bytes:
            """Execute the proxied node or subgraph locally."""
            if payload.get("payload_kind") == "mapped_subgraph":
                hydrated_inputs = deserialize_node_inputs(kwargs_payload)
                return serialize_node_outputs(
                    _execute_mapped_subgraph_payload(payload, hydrated_inputs)
                )
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)

        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Return a local no-op warmup result when Modal is unavailable."""
            return {"component_id": str(payload.get("component_id") or "modal-warmup")}

        def keepalive_for_local_gap(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Return a local keepalive acknowledgement when Modal is unavailable."""
            return {
                "component_id": str(payload.get("component_id") or "modal-keepalive"),
                "worker_affinity_key": self.worker_affinity_key,
            }
