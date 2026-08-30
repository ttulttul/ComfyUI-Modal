"""Remote Modal runtime and local execution fallback."""

import asyncio
import copy
from dataclasses import replace
import importlib
import importlib.util
import logging
import os
import threading
import time
from types import ModuleType
import uuid
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

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
from .modal_interrupts import (
    _ActiveRemoteInvocation,
    _abandon_local_modal_workflow_gate,
    _exception_indicates_interruption,
    _handle_modal_wait_cancellation,
    _handle_modal_wait_cancellation_async,
    _invoke_remote_call_with_interrupts,
    _local_processing_interrupted,
    _lookup_modal_interrupt_store,
    _propagate_remote_interrupt_request,
    _raise_local_interrupt,
    _registered_active_remote_invocation,
    _remote_interrupt_flag_key,
    _remote_interrupt_flag_value,
    _remote_interrupt_key,
    _request_remote_interrupt,
    _request_remote_interrupt_async,
    _sync_local_interrupt_to_cancellation_event,
    _write_remote_interrupt_flag,
    _write_remote_interrupt_flag_async,
    active_remote_modal_prompt_ids,
    request_remote_modal_prompt_interrupt,
    request_remote_modal_prompt_interrupt_async,
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
from .payload_stream import (
    _close_remote_payload_stream,
    _consume_remote_payload_stream,
)

logger = logging.getLogger(__name__)

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


try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local fallback tests.
    modal = None












































































































































































































































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
                input_transfer_bytes=len(kwargs_payload),
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
