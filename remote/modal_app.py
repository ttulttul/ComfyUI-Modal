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
from .mapped_execution import (
    MappedExecutionHooks,
    _aggregate_mapped_outputs,
    _annotate_implicit_batched_boundary_outputs,
    _build_mapped_item_payload,
    _build_phase_subgraph_payload,
    _build_remote_session_cleanup_payload,
    _build_static_mapped_payload,
    _clear_local_mapped_lane_progress,
    _emit_local_mapped_lane_progress_start,
    _emit_local_mapped_progress,
    _execute_mapped_subgraph_payload,
    _implicit_batch_boundary_effective_io_type,
    _implicit_batch_boundary_target_input_types,
    _implicit_batch_input_is_list_target_node_ids,
    _implicit_batch_preserving_targets,
    _invoke_implicitly_mapped_subgraph_async,
    _invoke_mapped_remote_engine_async,
    _is_latent_like_mapping,
    _is_mapped_boundary_output,
    _list_is_latent_like_batch,
    _log_detached_mapped_lane_result,
    _mapped_execution_parallelism,
    _mapped_phase_definition,
    _mapped_progress_owner_component_id,
    _merge_static_and_mapped_outputs,
    _partition_implicit_batched_execute_nodes,
    _pop_mapped_lane_node_id,
    _remember_mapped_lane_node_id,
    _shared_subgraph_payload_fields,
    _split_batch_boundary_inputs,
    _split_phase_outputs,
    configure_mapped_execution_hooks,
)
from .modal_llm_profile_staging import (
    _bounded_modal_stage_events,
    _close_modal_stage_events,
    _emit_local_llm_staging_progress,
    _ensure_llm_profiles_staged,
    _read_modal_stage_events,
    _rewrite_staged_llm_kwargs_payload,
)

logger = logging.getLogger(__name__)

# Remote durable-object handles.
_MODAL_INTERRUPT_DICTS_LOCK = threading.Lock()
_MODAL_INTERRUPT_DICTS: dict[tuple[str, str | None], Any] = {}

# Active invocation and local durable-session state.
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


configure_mapped_execution_hooks(
    MappedExecutionHooks(
        invoke_remote_engine_async=lambda *args, **kwargs: invoke_remote_engine_async(
            *args, **kwargs
        ),
        invoke_bound_remote_engine_async=lambda remote_engine, payload, kwargs_payload: _invoke_bound_remote_engine_async(
            remote_engine, payload, kwargs_payload
        ),
        local_processing_interrupted=lambda: _local_processing_interrupted(),
        raise_local_interrupt=lambda: _raise_local_interrupt(),
        exception_indicates_interruption=lambda exc: _exception_indicates_interruption(
            exc
        ),
    )
)


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
