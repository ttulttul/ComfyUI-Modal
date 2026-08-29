"""Stable Modal cloud entrypoint for ComfyUI Modal-Sync."""

import asyncio
import copy
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
from io import BytesIO
import inspect
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent
_REMOTE_REPO_ROOT = Path("/root/comfyui_modal_sync_repo")
_LOCAL_COMFYUI_ROOT = (Path.home() / "git" / "ComfyUI").resolve()
_REMOTE_COMFYUI_ROOT = Path("/root/comfyui_src")
_REMOTE_LLM_COMPILE_CACHE_ROOT = Path("/root/.cache/comfy-modal-llm")
for candidate in (
    _REPO_ROOT,
    _REMOTE_REPO_ROOT,
    _LOCAL_COMFYUI_ROOT,
    _REMOTE_COMFYUI_ROOT,
):
    candidate_str = str(candidate)
    try:
        candidate_exists = candidate.exists()
    except PermissionError:
        candidate_exists = False
    if candidate_exists and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from runtime_environment import (  # noqa: E402 - paths are bootstrapped above.
    COMFYUI_RUNTIME_SOURCE_DIRECTORIES as _COMFYUI_IMAGE_RUNTIME_DIRECTORIES,
    COMFYUI_RUNTIME_SOURCE_FILES as _COMFYUI_IMAGE_RUNTIME_FILES,
    REMOTE_APP_PROTOCOL_VERSION as _REMOTE_APP_PROTOCOL_VERSION,
    REMOTE_PYTHON_VERSION,
    RemoteTorchBuild as _RemoteTorchBuild,
    build_remote_runtime_identity,
    custom_node_runtime_packages as _custom_node_runtime_packages,
    remote_accelerator_packages as _remote_accelerator_packages,
    remote_accelerator_validation_command as _remote_accelerator_validation_command,
    remote_apt_packages as _comfyui_apt_packages,
    remote_huggingface_packages as _remote_huggingface_packages,
    remote_huggingface_validation_command as _remote_huggingface_validation_command,
    remote_runtime_packages as _comfyui_runtime_packages,
    select_remote_torch_build as _select_remote_torch_build,
)
from llm_recovery import (  # noqa: E402
    LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY,
    is_llm_memory_recovery_exhausted,
)
from llm_staging import resolve_and_stage_model_references  # noqa: E402
from durable_state import (  # noqa: E402 - paths are bootstrapped above.
    DurableObjectCommitBatch,
    DurableObjectRef,
    DurableStateError,
    FileDurableObjectStore,
    InMemoryRemoteInvocationStore,
    RemoteInvocationRecord,
    new_running_invocation_record,
    read_modal_volume_file,
)
from output_artifacts import (  # noqa: E402 - paths are bootstrapped above.
    RemoteOutputSnapshot,
    capture_execution_result,
    snapshot_output_directory,
)
from remote_protocol import (  # noqa: E402 - paths are bootstrapped above.
    BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
    PRIMITIVE_WIDGET_INPUT_TYPES as _PRIMITIVE_WIDGET_INPUT_TYPES,
)

from serialization import (  # noqa: E402 - paths are bootstrapped above.
    coerce_serialized_node_outputs,
    deserialize_node_inputs,
    deserialize_node_outputs,
    deserialize_value,
    serialize_mapping,
    serialize_node_inputs,
    serialize_node_outputs,
    serialize_value,
)
from session_state import (  # noqa: E402 - paths are bootstrapped above.
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
from settings import (  # noqa: E402 - paths are bootstrapped above.
    DEFAULT_MODAL_SECRET_NAME,
    get_settings,
    modal_deployment_app_name,
)
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_comfy_bootstrap import (
        CloudComfyBootstrapHooks,
        _active_comfyui_root,
        _alias_flux_rms_norm_weight_keys,
        _build_checkpoint_loader_cache_key,
        _build_clip_loader_cache_key,
        _build_dual_clip_loader_cache_key,
        _build_unet_loader_cache_key,
        _build_vae_loader_cache_key,
        _clone_loader_cache_outputs,
        _clone_loader_cache_value,
        _custom_node_package_for_candidate_file,
        _ensure_comfy_runtime_initialized,
        _ensure_comfyui_support_packages,
        _ensure_default_custom_nodes_dir,
        _ensure_headless_prompt_server_instance,
        _ensure_prompt_node_classes_registered,
        _extract_custom_nodes_bundle,
        _force_import_package_from_root,
        _install_loader_cache_wrappers,
        _install_model_state_dict_compatibility_wrappers,
        _iter_custom_nodes_manifest_assets,
        _iter_missing_class_candidate_files,
        _load_custom_nodes_manifest,
        _load_execution_module,
        _load_nodes_module,
        _loader_cache_metric_snapshot,
        _materialize_custom_nodes_manifest_assets,
        _materialize_remote_asset_path,
        _missing_node_class_diagnostics,
        _patched_folder_paths_absolute_lookup,
        _prompt_missing_node_class_types,
        _readthrough_cache_path,
        _record_loader_cache_metric,
        _register_custom_nodes_root,
        _register_modal_sync_runtime_nodes,
        _reload_external_custom_nodes_for_missing_classes,
        _resolve_custom_nodes_archives,
        _resolve_custom_nodes_bundle_path,
        _resolve_runtime_asset_path,
        _rewrite_modal_asset_references,
        _serialize_loader_cache_key,
        _validated_custom_node_asset_relative_path,
        _wrap_loader_method_with_cache,
        clear_warm_caches as clear_comfy_bootstrap_warm_caches,
        configure_cloud_comfy_bootstrap_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_comfy_bootstrap import (
        CloudComfyBootstrapHooks,
        _active_comfyui_root,
        _alias_flux_rms_norm_weight_keys,
        _build_checkpoint_loader_cache_key,
        _build_clip_loader_cache_key,
        _build_dual_clip_loader_cache_key,
        _build_unet_loader_cache_key,
        _build_vae_loader_cache_key,
        _clone_loader_cache_outputs,
        _clone_loader_cache_value,
        _custom_node_package_for_candidate_file,
        _ensure_comfy_runtime_initialized,
        _ensure_comfyui_support_packages,
        _ensure_default_custom_nodes_dir,
        _ensure_headless_prompt_server_instance,
        _ensure_prompt_node_classes_registered,
        _extract_custom_nodes_bundle,
        _force_import_package_from_root,
        _install_loader_cache_wrappers,
        _install_model_state_dict_compatibility_wrappers,
        _iter_custom_nodes_manifest_assets,
        _iter_missing_class_candidate_files,
        _load_custom_nodes_manifest,
        _load_execution_module,
        _load_nodes_module,
        _loader_cache_metric_snapshot,
        _materialize_custom_nodes_manifest_assets,
        _materialize_remote_asset_path,
        _missing_node_class_diagnostics,
        _patched_folder_paths_absolute_lookup,
        _prompt_missing_node_class_types,
        _readthrough_cache_path,
        _record_loader_cache_metric,
        _register_custom_nodes_root,
        _register_modal_sync_runtime_nodes,
        _reload_external_custom_nodes_for_missing_classes,
        _resolve_custom_nodes_archives,
        _resolve_custom_nodes_bundle_path,
        _resolve_runtime_asset_path,
        _rewrite_modal_asset_references,
        _serialize_loader_cache_key,
        _validated_custom_node_asset_relative_path,
        _wrap_loader_method_with_cache,
        clear_warm_caches as clear_comfy_bootstrap_warm_caches,
        configure_cloud_comfy_bootstrap_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_node_output_cache import (
        CloudNodeOutputCacheHooks,
        _NodeOutputCacheLookupResult,
        _PersistedNodeCacheRestoreState,
        _await_maybe,
        _boundary_output_node_ids,
        _build_node_output_cache_immediate_signature,
        _build_node_output_cache_signature_from_key_set_async,
        _build_node_output_cache_signature_from_key_set_sync,
        _cache_signature_link_output_index,
        _canonicalize_node_output_cache_key_part,
        _deserialize_node_output_cache_entry,
        _emit_restored_node_cache_events,
        _estimate_node_output_cache_value_size_bytes,
        _include_unique_id_in_input_signature,
        _install_prompt_executor_persisted_cache_restore,
        _is_input_signature_cache_key_set,
        _node_output_cache_ancestor_ids,
        _node_output_cache_key,
        _node_output_cache_key_from_key_set_async,
        _node_output_cache_key_from_key_set_sync,
        _node_output_cache_key_preview,
        _node_output_cache_store,
        _node_output_cache_store_get,
        _node_output_cache_store_put,
        _node_output_cache_value_preview,
        _persist_node_output_cache_entries,
        _prompt_executor_cache_get_sync,
        _restore_persisted_node_output_cache_entries,
        _restore_persisted_node_output_cache_entries_into_prepared_cache,
        _serialize_node_output_cache_entry,
        _tensor_cache_key_digest,
        configure_cloud_node_output_cache_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_node_output_cache import (
        CloudNodeOutputCacheHooks,
        _NodeOutputCacheLookupResult,
        _PersistedNodeCacheRestoreState,
        _await_maybe,
        _boundary_output_node_ids,
        _build_node_output_cache_immediate_signature,
        _build_node_output_cache_signature_from_key_set_async,
        _build_node_output_cache_signature_from_key_set_sync,
        _cache_signature_link_output_index,
        _canonicalize_node_output_cache_key_part,
        _deserialize_node_output_cache_entry,
        _emit_restored_node_cache_events,
        _estimate_node_output_cache_value_size_bytes,
        _include_unique_id_in_input_signature,
        _install_prompt_executor_persisted_cache_restore,
        _is_input_signature_cache_key_set,
        _node_output_cache_ancestor_ids,
        _node_output_cache_key,
        _node_output_cache_key_from_key_set_async,
        _node_output_cache_key_from_key_set_sync,
        _node_output_cache_key_preview,
        _node_output_cache_store,
        _node_output_cache_store_get,
        _node_output_cache_store_put,
        _node_output_cache_value_preview,
        _persist_node_output_cache_entries,
        _prompt_executor_cache_get_sync,
        _restore_persisted_node_output_cache_entries,
        _restore_persisted_node_output_cache_entries_into_prepared_cache,
        _serialize_node_output_cache_entry,
        _tensor_cache_key_digest,
        configure_cloud_node_output_cache_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_prompt_execution import (
        CloudPromptExecutionHooks,
        _ReusablePromptExecutorState,
        _aggregate_mapped_phase_outputs,
        _apply_boundary_inputs,
        _boundary_input_cache_signature,
        _build_phase_subgraph_payload,
        _coerce_primitive_prompt_input_value,
        _coerce_prompt_primitive_input_values,
        _collapse_cache_slot,
        _copy_json_safe_prompt_metadata,
        _execute_mapped_subgraph_payload,
        _execute_node_locally_raw,
        _execute_prompt_executor_compat,
        _execute_subgraph_prompt,
        _extract_prompt_executor_error,
        _extract_prompt_executor_error_payload,
        _format_prompt_executor_error_payload,
        _get_or_create_prompt_executor_state,
        _install_metadata_safe_dynamic_prompt_wrapper,
        _invoke_original_node,
        _is_link,
        _log_prompt_executor_failure_details,
        _mapped_phase_definition,
        _merge_static_and_mapped_outputs,
        _merge_static_or_mapped_values,
        _node_input_type_map,
        _node_input_types,
        _node_required_input_names,
        _normalize_link_output_index,
        _normalize_prompt_input_value,
        _normalize_subgraph_payload,
        _prompt_executor_cache_config,
        _prompt_executor_ram_thresholds,
        _remote_session_ref_cache_signature,
        _reset_prompt_executor_request_state,
        _resolve_required_subgraph_nodes,
        _serialize_prompt_executor_cache_scope,
        _shared_subgraph_payload_fields,
        _short_circuit_restored_session_output_subgraph,
        _split_phase_outputs,
        _summarize_suspicious_prompt_inputs,
        _temporary_node_mapping,
        _temporary_progress_hook,
        _temporary_prompt_metadata,
        _temporary_remote_interrupt_monitor,
        _trim_subgraph_payload_to_required_nodes,
        _unwrap_wrapped_prompt_link,
        _validate_prompt_input_shapes,
        _validate_required_prompt_inputs,
        clear_warm_caches as clear_cloud_prompt_execution_warm_caches,
        configure_cloud_prompt_execution_hooks,
        execute_node_locally,
        execute_subgraph_locally,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_prompt_execution import (
        CloudPromptExecutionHooks,
        _ReusablePromptExecutorState,
        _aggregate_mapped_phase_outputs,
        _apply_boundary_inputs,
        _boundary_input_cache_signature,
        _build_phase_subgraph_payload,
        _coerce_primitive_prompt_input_value,
        _coerce_prompt_primitive_input_values,
        _collapse_cache_slot,
        _copy_json_safe_prompt_metadata,
        _execute_mapped_subgraph_payload,
        _execute_node_locally_raw,
        _execute_prompt_executor_compat,
        _execute_subgraph_prompt,
        _extract_prompt_executor_error,
        _extract_prompt_executor_error_payload,
        _format_prompt_executor_error_payload,
        _get_or_create_prompt_executor_state,
        _install_metadata_safe_dynamic_prompt_wrapper,
        _invoke_original_node,
        _is_link,
        _log_prompt_executor_failure_details,
        _mapped_phase_definition,
        _merge_static_and_mapped_outputs,
        _merge_static_or_mapped_values,
        _node_input_type_map,
        _node_input_types,
        _node_required_input_names,
        _normalize_link_output_index,
        _normalize_prompt_input_value,
        _normalize_subgraph_payload,
        _prompt_executor_cache_config,
        _prompt_executor_ram_thresholds,
        _remote_session_ref_cache_signature,
        _reset_prompt_executor_request_state,
        _resolve_required_subgraph_nodes,
        _serialize_prompt_executor_cache_scope,
        _shared_subgraph_payload_fields,
        _short_circuit_restored_session_output_subgraph,
        _split_phase_outputs,
        _summarize_suspicious_prompt_inputs,
        _temporary_node_mapping,
        _temporary_progress_hook,
        _temporary_prompt_metadata,
        _temporary_remote_interrupt_monitor,
        _trim_subgraph_payload_to_required_nodes,
        _unwrap_wrapped_prompt_link,
        _validate_prompt_input_shapes,
        _validate_required_prompt_inputs,
        clear_warm_caches as clear_cloud_prompt_execution_warm_caches,
        configure_cloud_prompt_execution_hooks,
        execute_node_locally,
        execute_subgraph_locally,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_prompt_server_shims import (
        CloudPromptServerHooks,
        _HeadlessPromptQueue,
        _HeadlessPromptServerInstance,
        _NullPromptServer,
        _TracingPromptServer,
        configure_cloud_prompt_server_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_prompt_server_shims import (
        CloudPromptServerHooks,
        _HeadlessPromptQueue,
        _HeadlessPromptServerInstance,
        _NullPromptServer,
        _TracingPromptServer,
        configure_cloud_prompt_server_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_session_bridge import (
        CloudSessionBridgeHooks,
        _RemoteSessionBridgeResolutionStats,
        _bridge_record_replays_sampling_node,
        _build_durable_bridge_rehydration_plan,
        _build_remote_session_bridge_record,
        _deserialize_remote_session_bridge_producer_inputs,
        _get_remote_session_bridge_value,
        _json_payload_size_bytes,
        _load_loader_snapshot_profile,
        _load_remote_session_bridge_record,
        _log_remote_session_resolution_summary,
        _offload_large_bridge_payloads,
        _payload_remote_session_handle,
        _record_remote_session_resolution_event,
        _rehydrate_remote_session_bridge_value,
        _remote_session_bridge_recovery_input_names,
        _remote_session_bridge_replay_stack,
        _resolve_remote_session_inputs,
        _restore_planned_remote_session_bridge_value,
        _restore_serialized_remote_session_bridge_value,
        _sanitize_payload_for_session_bridge_record,
        _select_remote_session_bridge_recovery_kind,
        _serialize_durable_bridge_output,
        _session_bridge_store,
        _snapshot_profile_store,
        _store_remote_session_bridge_record,
        _store_remote_session_bridge_value,
        _subgraph_contains_sampling_node,
        configure_cloud_session_bridge_hooks,
        remote_session_store,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_session_bridge import (
        CloudSessionBridgeHooks,
        _RemoteSessionBridgeResolutionStats,
        _bridge_record_replays_sampling_node,
        _build_durable_bridge_rehydration_plan,
        _build_remote_session_bridge_record,
        _deserialize_remote_session_bridge_producer_inputs,
        _get_remote_session_bridge_value,
        _json_payload_size_bytes,
        _load_loader_snapshot_profile,
        _load_remote_session_bridge_record,
        _log_remote_session_resolution_summary,
        _offload_large_bridge_payloads,
        _payload_remote_session_handle,
        _record_remote_session_resolution_event,
        _rehydrate_remote_session_bridge_value,
        _remote_session_bridge_recovery_input_names,
        _remote_session_bridge_replay_stack,
        _resolve_remote_session_inputs,
        _restore_planned_remote_session_bridge_value,
        _restore_serialized_remote_session_bridge_value,
        _sanitize_payload_for_session_bridge_record,
        _select_remote_session_bridge_recovery_kind,
        _serialize_durable_bridge_output,
        _session_bridge_store,
        _snapshot_profile_store,
        _store_remote_session_bridge_record,
        _store_remote_session_bridge_value,
        _subgraph_contains_sampling_node,
        configure_cloud_session_bridge_hooks,
        remote_session_store,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_durable_invocation import (
        DurableInvocationErrors,
        _begin_remote_invocation,
        _canary_barrier_marker_exists,
        _canary_barrier_marker_key,
        _canary_interrupt_requested,
        _complete_remote_invocation,
        _durable_object_store,
        _execute_canary_payload,
        _execute_payload_with_output_capture,
        _execute_with_durable_invocation,
        _fail_remote_invocation,
        _invocation_record_store,
        _load_completed_remote_invocation_result,
        _load_remote_invocation_record,
        _put_canary_barrier_marker,
        _raise_if_canary_interrupted,
        _remote_comfy_output_directory,
        _remote_output_snapshot,
        _store_remote_invocation_record,
        _wait_for_canary_barrier,
        _wait_for_running_remote_invocation,
        configure_durable_invocation_errors,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_durable_invocation import (
        DurableInvocationErrors,
        _begin_remote_invocation,
        _canary_barrier_marker_exists,
        _canary_barrier_marker_key,
        _canary_interrupt_requested,
        _complete_remote_invocation,
        _durable_object_store,
        _execute_canary_payload,
        _execute_payload_with_output_capture,
        _execute_with_durable_invocation,
        _fail_remote_invocation,
        _invocation_record_store,
        _load_completed_remote_invocation_result,
        _load_remote_invocation_record,
        _put_canary_barrier_marker,
        _raise_if_canary_interrupted,
        _remote_comfy_output_directory,
        _remote_output_snapshot,
        _store_remote_invocation_record,
        _wait_for_canary_barrier,
        _wait_for_running_remote_invocation,
        configure_durable_invocation_errors,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_image_env import (
        _REMOTE_LLM_COMPILE_CACHE_ROOT,
        _install_custom_node_packages,
        _install_remote_accelerator_packages,
        _install_remote_torch_build,
        _modal_image_environment,
        _modal_secret_from_settings,
        _model_stager_image_environment,
        _remote_engine_cls_options,
        _should_ignore_comfyui_path,
        _should_ignore_repo_path,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_image_env import (
        _REMOTE_LLM_COMPILE_CACHE_ROOT,
        _install_custom_node_packages,
        _install_remote_accelerator_packages,
        _install_remote_torch_build,
        _modal_image_environment,
        _modal_secret_from_settings,
        _model_stager_image_environment,
        _remote_engine_cls_options,
        _should_ignore_comfyui_path,
        _should_ignore_repo_path,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_app_guard import guard_against_existing_modal_app
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_app_guard import guard_against_existing_modal_app
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_runtime_context import (
        interrupt_flag_store,
        invocation_record_store,
        node_output_cache_store,
        register_cloud_runtime_stores,
        session_bridge_store,
        snapshot_profile_store,
        volume_store,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_runtime_context import (
        interrupt_flag_store,
        invocation_record_store,
        node_output_cache_store,
        register_cloud_runtime_stores,
        session_bridge_store,
        snapshot_profile_store,
        volume_store,
    )

logger = logging.getLogger(__name__)

# Cloud app and ComfyUI bootstrap state.
_CLOUD_HANDLER_NAME = "comfyui-modal-sync-cloud-timestamped"

# Loader and prewarm state.
_LOADER_PREWARM_PLAN_KEYS_LOCK = threading.Lock()
_LOADER_PREWARM_PLAN_KEYS: set[str] = set()
_LLM_PREWARM_PLAN_KEYS_LOCK = threading.Lock()
_LLM_PREWARM_PLAN_KEYS: set[str] = set()

# Modal volume reload and poisoned-container retirement state.
_MODAL_VOLUME_RELOAD_MARKERS_LOCK = threading.Lock()
_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)
_MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT = 256
_MODAL_VOLUME_RELOAD_MARKERS: queue.SimpleQueue[str] | None = None
_MODAL_VOLUME_RELOAD_MARKER_SET: set[str] = set()
_CONTAINER_TERMINATION_LOCK = threading.Lock()
_REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS = 1.0
_CONTAINER_TERMINATION_SCHEDULED = False
# Durable invocation and session-bridge state.
_REMOTE_INVOCATION_ABANDON_JOIN_SECONDS = 0.5

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - remote entrypoint only.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


class RemoteInvocationInProgressError(RuntimeError):
    """Raised when an idempotent invocation is already running remotely."""


class RemoteInvocationAbandonedError(RuntimeError):
    """Raised when a streamed invocation loses its consumer before completion."""


class RemoteCanaryInterruptedError(RuntimeError):
    """Raised when a live remote canary observes its shared interrupt flag."""


class RemoteCanaryBarrierTimeoutError(TimeoutError):
    """Raised when live canary calls fail to overlap before their deadline."""


class ExistingModalAppError(RuntimeError):
    """Raised when deploying would overwrite an existing Modal app."""


configure_durable_invocation_errors(
    DurableInvocationErrors(
        invocation_in_progress=RemoteInvocationInProgressError,
        canary_interrupted=RemoteCanaryInterruptedError,
        canary_barrier_timeout=RemoteCanaryBarrierTimeoutError,
    )
)


class RemoteFailureDisposition(str, Enum):
    """Describe whether one remote failure implies poisoned worker state."""

    EXPECTED = "expected"
    DETERMINISTIC = "deterministic"
    POISONED_WORKER = "poisoned-worker"


def _guard_against_existing_modal_app(settings: Any, modal_module: Any) -> None:
    """Fail local Modal app construction when the configured app already exists."""
    guard_against_existing_modal_app(
        settings,
        modal_module,
        error_type=ExistingModalAppError,
    )


@dataclass
class _RemoteExecutionControl:
    """Track interruption state for one active remote payload execution."""

    cancellation_event: threading.Event
    interrupt_flag_key: str


def _meaningful_progress_values(
    node_state: dict[str, Any]
) -> tuple[float, float] | None:
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
        logger.error(
            "Exiting Modal container process with code=%s after remote failure.",
            exit_code,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    threading.Thread(
        target=exit_later,
        name="modal-container-exit",
        daemon=True,
    ).start()


def _schedule_process_exit_unless_cancelled(
    *,
    delay_seconds: float,
    exit_code: int,
    cancel_event: threading.Event,
    reason: str,
) -> None:
    """Exit the current process after a delay unless the caller cancels first."""

    def exit_later() -> None:
        """Wait for cancellation or exit the worker if the delay expires."""
        if delay_seconds > 0 and cancel_event.wait(timeout=delay_seconds):
            logger.debug("Cancelled delayed Modal container restart for %s.", reason)
            return
        logger.error(
            "Exiting Modal container process with code=%s after %s.", exit_code, reason
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    threading.Thread(
        target=exit_later,
        name="modal-container-cancel-restart",
        daemon=True,
    ).start()


def _schedule_remote_cancel_restart(
    *,
    component_id: str,
    completion_event: threading.Event,
) -> bool:
    """Restart the Modal worker if a cancelled remote prompt keeps executing."""
    if not _is_modal_container_runtime():
        return False

    delay_seconds = max(0.0, get_settings().remote_cancel_restart_seconds)
    logger.warning(
        "Remote cancellation requested for component=%s; scheduling container restart in %.3fs unless execution stops first.",
        component_id,
        delay_seconds,
    )
    _schedule_process_exit_unless_cancelled(
        delay_seconds=delay_seconds,
        exit_code=0,
        cancel_event=completion_event,
        reason=f"remote cancellation timeout for component={component_id}",
    )
    return True


def _is_interrupt_like_failure(exc: Exception) -> bool:
    """Return whether one remote failure represents an expected interruption rather than a crash."""
    return "interrupt" in str(exc).lower()


def _is_session_state_like_failure(exc: Exception) -> bool:
    """Return whether one remote failure came from prompt-scoped session routing/state issues."""
    if isinstance(exc, RemoteSessionStateError):
        return True
    return "remote session" in str(exc).lower()


def _remote_failure_disposition(exc: Exception) -> RemoteFailureDisposition:
    """Classify one execution failure for worker-retirement decisions."""
    if _is_interrupt_like_failure(exc) or _is_session_state_like_failure(exc):
        return RemoteFailureDisposition.EXPECTED
    if isinstance(exc, MemoryError) or is_llm_memory_recovery_exhausted(exc):
        return RemoteFailureDisposition.POISONED_WORKER

    message = str(exc).lower()
    poisoned_runtime_markers = (
        "cuda out of memory",
        "cuda error",
        "device-side assert",
        "illegal memory access",
        "cublas_status",
        "cudnn_status",
        "hip error",
    )
    if any(marker in message for marker in poisoned_runtime_markers):
        return RemoteFailureDisposition.POISONED_WORKER
    return RemoteFailureDisposition.DETERMINISTIC


def _maybe_schedule_container_termination_on_error(
    payload: dict[str, Any],
    exc: Exception,
) -> bool:
    """Retire the current Modal container after a remote execution crash when configured."""
    if not _is_modal_container_runtime():
        return False
    if not bool(payload.get("terminate_container_on_error", True)):
        return False
    disposition = _remote_failure_disposition(exc)
    if disposition is not RemoteFailureDisposition.POISONED_WORKER:
        logger.warning(
            "Skipping Modal container termination for component=%s failure_disposition=%s because the worker is safe to reuse.",
            payload.get("component_id"),
            disposition.value,
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
    prompt_id = str(
        payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph"
    )
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _observe_remote_workflow_for_llm_mode(payload: dict[str, Any]) -> None:
    """Record real workflow arrivals for container-local vLLM auto promotion."""
    if payload.get("payload_kind") == "canary":
        return
    from modal_llm_runtime import (
        force_modal_vllm_throughput_after_memory_recovery,
        observe_modal_workflow_execution,
    )

    prompt_id = payload.get("prompt_id")
    normalized_prompt_id = str(prompt_id).strip() if prompt_id is not None else None
    if bool(payload.get(LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY)):
        force_modal_vllm_throughput_after_memory_recovery(normalized_prompt_id)
        return
    observe_modal_workflow_execution(normalized_prompt_id)


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
    try:
        yield control
    finally:
        interrupt_store = interrupt_flag_store()
        if modal is not None and interrupt_store is not None:
            interrupt_store.pop(control.interrupt_flag_key, None)


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


class _BoundedStreamEventBuffer:
    """Bound progress memory while preserving terminal stream events."""

    def __init__(self, maxsize: int) -> None:
        """Initialize a bounded event queue and close signal."""
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max(4, maxsize))
        self._closed = threading.Event()
        self._dropped_progress_events = 0
        self._dropped_lock = threading.Lock()

    @property
    def dropped_progress_events(self) -> int:
        """Return how many stale progress events were coalesced away."""
        with self._dropped_lock:
            return self._dropped_progress_events

    @property
    def queue_size(self) -> int:
        """Return the approximate number of currently buffered events."""
        return self._queue.qsize()

    def publish_progress(self, payload: Any) -> None:
        """Publish the newest progress event without exceeding the queue bound."""
        event = ("progress", payload)
        while not self._closed.is_set():
            try:
                self._queue.put_nowait(event)
                return
            except queue.Full:
                try:
                    discarded_event = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if discarded_event[0] != "progress":
                    self._queue.put_nowait(discarded_event)
                    return
                with self._dropped_lock:
                    self._dropped_progress_events += 1

    def publish_terminal(self, event_kind: str, payload: Any) -> bool:
        """Publish a result, error, or completion unless the consumer closed."""
        while not self._closed.is_set():
            try:
                self._queue.put((event_kind, payload), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def get(self) -> tuple[str, Any]:
        """Wait for and return the next buffered stream event."""
        return self._queue.get()

    def close(self) -> None:
        """Release any producer waiting after the consumer stops."""
        self._closed.set()


def _abandon_streamed_remote_invocation(
    *,
    payload: Mapping[str, Any],
    running_record: RemoteInvocationRecord | None,
    cancellation_event: threading.Event | None,
    event_buffer: _BoundedStreamEventBuffer,
    worker_thread: threading.Thread,
) -> None:
    """Cancel unfinished compute and make its invocation record retryable."""
    component_id = str(payload.get("component_id") or "payload")
    if cancellation_event is not None:
        cancellation_event.set()
    event_buffer.close()
    worker_thread.join(timeout=max(0.0, _REMOTE_INVOCATION_ABANDON_JOIN_SECONDS))
    abandoned_error = RemoteInvocationAbandonedError(
        f"Remote invocation stream for component {component_id!r} closed before completion."
    )
    if running_record is not None:
        _fail_remote_invocation(running_record, abandoned_error)
    logger.warning(
        "Abandoned remote invocation stream component=%s invocation_id=%s worker_stopped=%s; the invocation is retryable.",
        component_id,
        running_record.invocation_id if running_record is not None else "none",
        not worker_thread.is_alive(),
    )


def _stream_remote_payload_events(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield progress and result events for one remote payload execution."""
    event_buffer = _BoundedStreamEventBuffer(get_settings().stream_event_queue_maxsize)
    task_id = os.getenv("MODAL_TASK_ID")
    component_id = str(payload.get("component_id") or "payload")
    invocation_id = str(payload.get("invocation_id") or "none")

    def publish_status(progress_state: dict[str, Any]) -> None:
        """Queue a progress envelope for the remote caller."""
        event_buffer.publish_progress(serialize_mapping(progress_state))

    def execute_once() -> bytes:
        """Run the underlying payload once and return serialized outputs."""
        if payload.get("payload_kind") == "canary":
            return _execute_canary_payload(
                payload,
                kwargs_payload,
                cancellation_event=cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            )
        if payload.get("payload_kind") == "mapped_subgraph":
            custom_nodes_root = _extract_custom_nodes_bundle(
                payload.get("custom_nodes_bundle")
            )
            _ensure_comfy_runtime_initialized(custom_nodes_root)
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
            return serialize_node_outputs(
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
        if payload.get("payload_kind") == "subgraph":
            execute_subgraph_kwargs: dict[str, Any] = {
                "status_callback": publish_status
            }
            if (
                "cancellation_event"
                in inspect.signature(execute_subgraph_locally).parameters
            ):
                execute_subgraph_kwargs["cancellation_event"] = cancellation_event
            if (
                "interrupt_store"
                in inspect.signature(execute_subgraph_locally).parameters
            ):
                execute_subgraph_kwargs["interrupt_store"] = interrupt_store
            if (
                "interrupt_flag_key"
                in inspect.signature(execute_subgraph_locally).parameters
            ):
                execute_subgraph_kwargs["interrupt_flag_key"] = interrupt_flag_key
            return execute_subgraph_locally(
                payload,
                kwargs_payload,
                **execute_subgraph_kwargs,
            )
        execute_node_kwargs: dict[str, Any] = {}
        if "cancellation_event" in inspect.signature(execute_node_locally).parameters:
            execute_node_kwargs["cancellation_event"] = cancellation_event
        if "interrupt_store" in inspect.signature(execute_node_locally).parameters:
            execute_node_kwargs["interrupt_store"] = interrupt_store
        if "interrupt_flag_key" in inspect.signature(execute_node_locally).parameters:
            execute_node_kwargs["interrupt_flag_key"] = interrupt_flag_key
        return execute_node_locally(payload, kwargs_payload, **execute_node_kwargs)

    object_store = _durable_object_store()

    def execute_payload() -> None:
        """Run compute in a worker thread while deferring Modal volume commits."""
        pending_batch: DurableObjectCommitBatch | None = None
        try:
            with object_store.batch_commits(commit_on_exit=False) as pending_batch:
                outputs = _execute_payload_with_output_capture(payload, execute_once)
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised through generator consumer tests.
            event_buffer.publish_terminal("error", (exc, pending_batch))
        else:
            result_bytes = len(outputs)
            pending_object_write = bool(pending_batch and pending_batch.wrote_object)
            logger.info(
                "Remote stream worker produced result component=%s invocation_id=%s task_id=%s "
                "result_bytes=%d pending_object_write=%s buffer_queue_size=%d.",
                component_id,
                invocation_id,
                task_id or "none",
                result_bytes,
                pending_object_write,
                event_buffer.queue_size,
            )
            publish_started_at = time.monotonic()
            published = event_buffer.publish_terminal("result", (outputs, pending_batch))
            logger.info(
                "Finished publishing remote stream result to event buffer in %.3fs component=%s "
                "invocation_id=%s task_id=%s published=%s result_bytes=%d buffer_queue_size=%d.",
                time.monotonic() - publish_started_at,
                component_id,
                invocation_id,
                task_id or "none",
                published,
                result_bytes,
                event_buffer.queue_size,
            )
        finally:
            event_buffer.publish_terminal("done", None)

    worker_thread = threading.Thread(
        target=execute_payload,
        name=f"modal-stream-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    if task_id:
        yield {"kind": "remote_logs", "task_id": task_id}
    running_record, completed_result = _begin_remote_invocation(payload)
    if completed_result is not None:
        yield {"kind": "result", "outputs": completed_result}
        return
    worker_thread.start()
    invocation_finalized = False
    try:
        while True:
            event_kind, event_payload = event_buffer.get()
            if event_kind == "progress":
                yield {"kind": "progress", **event_payload}
                continue
            if event_kind == "result":
                outputs, pending_batch = event_payload
                serialized_outputs = coerce_serialized_node_outputs(outputs)
                logger.info(
                    "Remote stream consumer received buffered result component=%s invocation_id=%s "
                    "task_id=%s result_bytes=%d pending_object_write=%s buffer_queue_size=%d.",
                    component_id,
                    invocation_id,
                    task_id or "none",
                    len(serialized_outputs),
                    bool(pending_batch and pending_batch.wrote_object),
                    event_buffer.queue_size,
                )
                if running_record is not None:
                    _complete_remote_invocation(
                        running_record,
                        serialized_outputs,
                        pending_batch=pending_batch,
                    )
                elif pending_batch is not None:
                    object_store.commit_batch(pending_batch)
                invocation_finalized = True
                yield_started_at = time.monotonic()
                logger.info(
                    "Yielding remote stream result to Modal transport component=%s invocation_id=%s "
                    "task_id=%s result_bytes=%d buffer_queue_size=%d.",
                    component_id,
                    invocation_id,
                    task_id or "none",
                    len(serialized_outputs),
                    event_buffer.queue_size,
                )
                try:
                    yield {"kind": "result", "outputs": serialized_outputs}
                finally:
                    logger.info(
                        "Remote stream result yield released after %.3fs component=%s invocation_id=%s "
                        "task_id=%s result_bytes=%d buffer_queue_size=%d.",
                        time.monotonic() - yield_started_at,
                        component_id,
                        invocation_id,
                        task_id or "none",
                        len(serialized_outputs),
                        event_buffer.queue_size,
                    )
                continue
            if event_kind == "error":
                error, pending_batch = event_payload
                if pending_batch is not None:
                    object_store.commit_batch(pending_batch)
                if running_record is not None:
                    _fail_remote_invocation(running_record, error)
                invocation_finalized = True
                raise error
            if event_kind == "done":
                return
    finally:
        if invocation_finalized:
            event_buffer.close()
            worker_thread.join(timeout=1.0)
        else:
            _abandon_streamed_remote_invocation(
                payload=payload,
                running_record=running_record,
                cancellation_event=cancellation_event,
                event_buffer=event_buffer,
                worker_thread=worker_thread,
            )
        if event_buffer.dropped_progress_events:
            logger.info(
                "Coalesced %d stale remote progress event(s) for component=%s to keep the stream buffer bounded.",
                event_buffer.dropped_progress_events,
                payload.get("component_id"),
            )




def _prewarm_snapshot_state(
    *,
    gpu_snapshot_enabled: bool,
    snapshot_profile_key: str = "",
) -> None:
    """Run snapshot-safe initialization before Modal captures a memory snapshot."""
    with _timed_phase(
        "prewarm_snapshot_state",
        gpu_snapshot=gpu_snapshot_enabled,
        snapshot_profile=snapshot_profile_key or None,
    ):
        _ensure_comfyui_support_packages()
        normalized_snapshot_profile_key = snapshot_profile_key.strip()
        if gpu_snapshot_enabled and normalized_snapshot_profile_key:
            _ensure_comfy_runtime_initialized(None)
            _load_execution_module()
            loader_prewarm_plans = _load_loader_snapshot_profile(
                normalized_snapshot_profile_key
            )
            if loader_prewarm_plans:
                _execute_loader_prewarm_plans(
                    component_id=f"snapshot-profile:{normalized_snapshot_profile_key}",
                    loader_prewarm_plans=loader_prewarm_plans,
                    custom_nodes_root=None,
                )
            _emit_cloud_info(
                "Completed GPU-snapshot ComfyUI prewarm before snapshot capture."
            )
            return

        if gpu_snapshot_enabled:
            _emit_cloud_info(
                "Skipping GPU-snapshot ComfyUI prewarm before snapshot capture because no snapshot profile was provided."
            )
        else:
            _emit_cloud_info(
                "Skipping full ComfyUI runtime prewarm during CPU-only snapshot to avoid accidental CUDA initialization."
            )


def _reload_compile_cache_volume(volume: Any | None) -> bool:
    """Refresh persistent compiler artifacts before a runtime opens its caches."""
    if volume is None:
        return False
    reload_method = getattr(volume, "reload", None)
    if not callable(reload_method):
        logger.warning("Modal compile-cache Volume does not expose reload().")
        return False
    with _timed_phase("llm_compile_cache_reload"):
        try:
            reload_method()
        except RuntimeError as exc:
            if _is_modal_volume_open_files_error(exc):
                _log_compile_cache_memory_maps()
            raise
    return True


def _mapped_process_files_under(
    volume_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[tuple[int, str], ...]:
    """Return process ids and files memory-mapped beneath one filesystem root."""
    try:
        resolved_root = volume_root.resolve(strict=True)
        process_directories = tuple(proc_root.iterdir())
    except OSError:
        return ()

    mapped_files: set[tuple[int, str]] = set()
    for process_directory in process_directories:
        if not process_directory.name.isdecimal():
            continue
        try:
            maps_text = (process_directory / "maps").read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue
        for line in maps_text.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            mapped_path = fields[5].removesuffix(" (deleted)")
            try:
                Path(mapped_path).relative_to(resolved_root)
            except ValueError:
                continue
            mapped_files.add((int(process_directory.name), mapped_path))
    return tuple(sorted(mapped_files))


def _log_compile_cache_memory_maps() -> None:
    """Log native mappings that explain a busy compile-cache Volume reload."""
    mapped_files = _mapped_process_files_under(_REMOTE_LLM_COMPILE_CACHE_ROOT)
    if not mapped_files:
        logger.warning(
            "Modal compile-cache Volume reload reported open files, but no "
            "memory-mapped cache files were visible in /proc."
        )
        return
    logger.error(
        "Modal compile-cache Volume reload is blocked by %d memory-mapped "
        "native cache file(s): %s",
        len(mapped_files),
        [
            {"pid": process_id, "path": mapped_path}
            for process_id, mapped_path in mapped_files[:8]
        ],
    )


def _prewarm_restored_runtime(compile_cache_volume: Any | None = None) -> None:
    """Run post-restore initialization that should be ready before serving requests."""
    with _timed_phase("prewarm_restored_runtime"):
        _reload_compile_cache_volume(compile_cache_volume)
        _ensure_comfy_runtime_initialized(None)
        _load_execution_module()




def _should_reload_modal_volume(payload: dict[str, Any]) -> bool:
    """Return whether this request needs the mounted Modal volume reloaded."""
    if _payload_volume_paths(payload) and not _payload_volume_paths_visible(payload):
        return True
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
        while (
            len(_MODAL_VOLUME_RELOAD_MARKER_SET)
            > _MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT
        ):
            expired_marker = _MODAL_VOLUME_RELOAD_MARKERS.get()
            _MODAL_VOLUME_RELOAD_MARKER_SET.discard(expired_marker)


def _clear_warm_remote_caches() -> None:
    """Drop warm-container caches that may retain references to mounted volume files."""
    clear_cloud_prompt_execution_warm_caches()
    clear_comfy_bootstrap_warm_caches()


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
        if bundle_path.is_absolute() and bundle_path.resolve().is_relative_to(
            remote_storage_root
        ):
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
                    and materialized_path_obj.resolve().is_relative_to(
                        remote_storage_root
                    )
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
    return all(_runtime_volume_path_visible(path) for path in uploaded_paths)


def _runtime_volume_path_visible(volume_path: Path) -> bool:
    """Return whether a mounted path is available directly or through read-through storage."""
    if volume_path.exists():
        return True
    cache_path = _readthrough_cache_path(volume_path)
    return cache_path is not None and cache_path.exists()


def _payload_volume_paths_visible(payload: dict[str, Any]) -> bool:
    """Return whether every mounted-volume path referenced by this payload is already visible."""
    referenced_paths = _payload_volume_paths(payload)
    if not referenced_paths:
        return False
    return all(_runtime_volume_path_visible(path) for path in referenced_paths)


def _download_committed_volume_path(
    volume: Any, volume_path: Path, cache_path: Path
) -> None:
    """Stream one committed Modal Volume file into the worker's ephemeral cache."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    relative_path = volume_path.resolve().relative_to(remote_storage_root).as_posix()
    read_file_into_fileobj = getattr(volume, "read_file_into_fileobj", None)
    read_file = getattr(volume, "read_file", None)
    if not callable(read_file_into_fileobj) and not callable(read_file):
        raise AttributeError(
            "The configured Modal Volume does not support committed file reads."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as cache_file:
            if callable(read_file_into_fileobj):
                read_file_into_fileobj(relative_path, cache_file)
            else:
                assert callable(read_file)
                for chunk in read_file(relative_path):
                    cache_file.write(chunk)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _custom_nodes_manifest_dependency_paths(
    volume_path: Path,
    runtime_path: Path,
) -> set[Path]:
    """Return mounted-volume dependencies declared by one custom-node manifest."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    custom_nodes_root = remote_storage_root / "custom_nodes"
    if (
        volume_path.suffix.lower() != ".json"
        or not volume_path.resolve().is_relative_to(custom_nodes_root)
    ):
        return set()
    try:
        manifest_payload = _load_custom_nodes_manifest(runtime_path)
    except RuntimeError:
        return set()

    dependency_paths: set[Path] = set()
    entry_payloads = manifest_payload.get("entries", [])
    if not isinstance(entry_payloads, list):
        return dependency_paths
    for entry_payload in entry_payloads:
        if not isinstance(entry_payload, dict):
            continue
        candidate_payloads = [entry_payload]
        asset_payloads = entry_payload.get("assets", [])
        if isinstance(asset_payloads, list):
            candidate_payloads.extend(
                asset_payload
                for asset_payload in asset_payloads
                if isinstance(asset_payload, dict)
            )
        for candidate_payload in candidate_payloads:
            remote_path = candidate_payload.get("remote_path")
            if not isinstance(remote_path, str) or not remote_path.strip():
                continue
            materialized_path = Path(_materialize_remote_asset_path(remote_path))
            if _readthrough_cache_path(materialized_path) is not None:
                dependency_paths.add(materialized_path)
    return dependency_paths


def _hydrate_missing_payload_volume_paths(
    volume: Any, payload: dict[str, Any]
) -> list[Path]:
    """Cache committed payload files that are absent from this worker's mounted snapshot."""
    candidate_paths = _payload_volume_paths(payload) | _payload_uploaded_volume_paths(
        payload
    )
    if not candidate_paths:
        return []
    if not callable(getattr(volume, "read_file_into_fileobj", None)) and not callable(
        getattr(volume, "read_file", None)
    ):
        return []

    hydrated_paths: list[Path] = []
    pending_paths = sorted(candidate_paths)
    visited_paths: set[Path] = set()
    component_id = str(payload.get("component_id") or "modal-subgraph")
    while pending_paths:
        volume_path = pending_paths.pop(0)
        if volume_path in visited_paths:
            continue
        visited_paths.add(volume_path)
        runtime_path = Path(_resolve_runtime_asset_path(str(volume_path)))
        if not _runtime_volume_path_visible(volume_path):
            cache_path = _readthrough_cache_path(volume_path)
            if cache_path is None:
                continue
            try:
                with _timed_phase(
                    "committed_volume_readthrough",
                    component=component_id,
                    path=volume_path.name,
                ):
                    _download_committed_volume_path(volume, volume_path, cache_path)
            except FileNotFoundError:
                logger.warning(
                    "Committed Modal Volume path %s was unavailable for component=%s; falling back to mounted-volume reload.",
                    volume_path,
                    component_id,
                )
                continue
            hydrated_paths.append(cache_path)
            runtime_path = cache_path

        pending_paths.extend(
            sorted(
                _custom_nodes_manifest_dependency_paths(volume_path, runtime_path)
                - visited_paths
            )
        )

    if hydrated_paths:
        _emit_cloud_info(
            "Hydrated %d missing committed volume file(s) through read-through storage for component=%s.",
            len(hydrated_paths),
            component_id,
        )
    return hydrated_paths


def _log_payload_volume_reload_diagnostics(
    component_id: str,
    payload: dict[str, Any] | None,
    *,
    context: str,
) -> None:
    """Log the mounted-volume paths relevant to one reload decision or failure."""
    if payload is None:
        return

    uploaded_paths = sorted(
        str(path) for path in _payload_uploaded_volume_paths(payload)
    )
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
                if attempt_index == len(
                    _MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS
                ):
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
    if reload_marker is not None and _has_seen_modal_volume_reload_marker(
        reload_marker
    ):
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


def _prepare_warm_container_for_request(
    volume: Any,
    payload: dict[str, Any],
    compile_cache_volume: Any | None = None,
) -> dict[str, Any]:
    """Prime one RemoteEngine container for a request before the first real execution payload arrives."""
    component_id = str(payload.get("component_id") or "modal-warmup")
    reload_marker = _modal_volume_reload_marker(payload)
    _hydrate_missing_payload_volume_paths(volume, payload)
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
        llm_prewarm_plans = payload.get("llm_prewarm_plans")
        llm_prewarm_results: list[dict[str, Any]] = []
        if isinstance(llm_prewarm_plans, list) and llm_prewarm_plans:
            llm_prewarm_results = _execute_llm_prewarm_plans(
                component_id=component_id,
                prompt_id=(
                    str(payload["prompt_id"])
                    if payload.get("prompt_id") is not None
                    else None
                ),
                llm_prewarm_plans=llm_prewarm_plans,
                compile_cache_volume=compile_cache_volume,
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
            "llm_prewarm_results": llm_prewarm_results,
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
    executable_plans: list[tuple[int, Mapping[str, Any], str | None]] = []
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
        executable_plans.append((plan_index, plan, plan_key))

    def execute_plan(
        plan_entry: tuple[int, Mapping[str, Any], str | None]
    ) -> None:
        """Execute one reserved loader plan and make failures retryable."""
        plan_index, plan, plan_key = plan_entry
        started_at = time.perf_counter()
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
        except Exception:
            if plan_key is not None:
                with _LOADER_PREWARM_PLAN_KEYS_LOCK:
                    _LOADER_PREWARM_PLAN_KEYS.discard(plan_key)
            raise
        logger.info(
            "Completed loader prewarm component=%s class_type=%s plan_index=%d elapsed_seconds=%.3f.",
            component_id,
            plan.get("class_type"),
            plan_index,
            time.perf_counter() - started_at,
        )

    worker_count = min(
        len(executable_plans),
        max(1, int(get_settings().loader_prewarm_workers)),
    )
    if worker_count > 1:
        logger.info(
            "Running %d independent loader prewarms with bounded concurrency=%d component=%s.",
            len(executable_plans),
            worker_count,
            component_id,
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="modal-loader-prewarm",
        ) as executor:
            futures = [
                executor.submit(execute_plan, plan_entry)
                for plan_entry in executable_plans
            ]
            for future in futures:
                future.result()
    else:
        for plan_entry in executable_plans:
            execute_plan(plan_entry)

    executed_plan_count = len(executable_plans)
    if executed_plan_count or skipped_plan_count:
        logger.info(
            "Warm container loader prewarm finished for component=%s executed=%d skipped=%d.",
            component_id,
            executed_plan_count,
            skipped_plan_count,
        )


def _llm_prewarm_model_profile(plan: Mapping[str, Any]) -> str:
    """Return the staged model profile from one rewritten LLM warmup plan."""
    prompt_node = plan.get("prompt_node")
    if isinstance(prompt_node, Mapping):
        inputs = prompt_node.get("inputs")
        if isinstance(inputs, Mapping):
            model_profile = inputs.get("model_profile")
            if isinstance(model_profile, str) and model_profile.strip():
                return model_profile.strip()
    model_profile = plan.get("model_profile")
    if not isinstance(model_profile, str) or not model_profile.strip():
        raise ValueError("LLM prewarm plan requires a fixed model_profile.")
    return model_profile.strip()


def _llm_compile_manifest_path(signature: str) -> Path:
    """Return the content-addressed completion marker for one JIT warmup plan."""
    cache_root = Path(
        os.getenv("TRITON_CACHE_DIR", str(_REMOTE_LLM_COMPILE_CACHE_ROOT))
    ).parent
    return cache_root / "manifests" / f"{signature}.json"


def _write_llm_compile_manifest(
    *,
    signature: str,
    model_profile: str,
    result: Mapping[str, Any],
) -> Path:
    """Atomically publish successful representative-warmup metadata."""
    manifest_path = _llm_compile_manifest_path(signature)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "model_profile": model_profile,
                "runtime_fingerprint": os.getenv(
                    "COMFY_MODAL_RUNTIME_FINGERPRINT", ""
                ),
                "completed_at": time.time(),
                "result": dict(result),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def _execute_llm_prewarm_plans(
    *,
    component_id: str,
    prompt_id: str | None,
    llm_prewarm_plans: list[dict[str, Any]],
    compile_cache_volume: Any | None,
) -> list[dict[str, Any]]:
    """Load resident LLMs, exercise representative shapes, and commit JIT caches."""
    from modal_llm_runtime import prewarm_modal_llm_profile

    results: list[dict[str, Any]] = []
    for plan in llm_prewarm_plans:
        if not isinstance(plan, Mapping):
            continue
        plan_signature = str(plan.get("signature") or "").strip()
        if not plan_signature:
            raise ValueError("LLM prewarm plan requires a stable signature.")
        model_profile = _llm_prewarm_model_profile(plan)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "model_profile": model_profile,
                    "plan_signature": plan_signature,
                    "runtime_fingerprint": os.getenv(
                        "COMFY_MODAL_RUNTIME_FINGERPRINT", ""
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path = _llm_compile_manifest_path(signature)
        representative_request_count = (
            1
            if manifest_path.exists()
            else max(1, int(plan.get("representative_request_count") or 3))
        )
        with _LLM_PREWARM_PLAN_KEYS_LOCK:
            already_resident = signature in _LLM_PREWARM_PLAN_KEYS
            _LLM_PREWARM_PLAN_KEYS.add(signature)
        if already_resident:
            logger.info(
                "Skipping duplicate resident LLM prewarm profile=%s component=%s.",
                model_profile,
                component_id,
            )
            continue
        try:
            compile_checkpoint = _LLMCompileMissCheckpoint(
                profiles=(model_profile,),
                signal_size=_triton_compile_miss_signal_size(),
                listener_engine_pids=_triton_compile_listener_engine_pids(),
            )
            with _timed_phase(
                "llm_representative_prewarm",
                component=component_id,
                profile=model_profile,
                requests=representative_request_count,
            ):
                result = prewarm_modal_llm_profile(
                    model_profile=model_profile,
                    representative_request_count=representative_request_count,
                    workflow_execution_id=prompt_id,
                )
            manifest_path = _write_llm_compile_manifest(
                signature=signature,
                model_profile=model_profile,
                result=result,
            )
            compile_cache_committed = _commit_actual_llm_compile_cache(
                compile_checkpoint,
                compile_cache_volume,
            )
            results.append(
                {
                    **result,
                    "manifest_path": str(manifest_path),
                    "manifest_cache_hit": representative_request_count == 1,
                    "compile_cache_committed": compile_cache_committed,
                }
            )
        except Exception:
            with _LLM_PREWARM_PLAN_KEYS_LOCK:
                _LLM_PREWARM_PLAN_KEYS.discard(signature)
            raise
    return results


def _llm_profiles_in_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect ModalLLM profiles in the executable subgraph dependency closure."""
    if payload.get("payload_kind") not in {"subgraph", "mapped_subgraph"}:
        return ()
    subgraph_prompt = payload.get("subgraph_prompt")
    execute_node_ids = payload.get("execute_node_ids")
    if not isinstance(subgraph_prompt, Mapping) or not isinstance(
        execute_node_ids, (list, tuple)
    ):
        return ()
    prompt = {str(node_id): node for node_id, node in subgraph_prompt.items()}
    profiles: set[str] = set()
    visited: set[str] = set()
    pending = [str(node_id) for node_id in execute_node_ids]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        if prompt_node.get("class_type") == "ModalLLM" and isinstance(
            inputs, Mapping
        ):
            profile = inputs.get("model_profile")
            if isinstance(profile, str) and profile.strip():
                profiles.add(profile.strip())
        if not isinstance(inputs, Mapping):
            continue
        for input_value in inputs.values():
            if _is_link(input_value):
                pending.append(str(input_value[0]))
    return tuple(sorted(profiles))


@dataclass(frozen=True)
class _LLMCompileMissCheckpoint:
    """Capture the genuine Triton miss signal before one LLM subgraph executes."""

    profiles: tuple[str, ...]
    signal_size: int
    listener_engine_pids: tuple[int, ...]


def _triton_compile_miss_signal_size() -> int:
    """Read the EngineCore compile-miss signal shared through container storage."""
    runtime_module = importlib.import_module("modal_llm_runtime")
    signal_reader = getattr(runtime_module, "triton_compile_miss_signal_size", None)
    if not callable(signal_reader):
        raise RuntimeError(
            "Modal LLM runtime does not expose triton_compile_miss_signal_size()."
        )
    return int(signal_reader())


def _triton_compile_listener_engine_pids() -> tuple[int, ...]:
    """Return live EngineCore processes with cache-aware Triton telemetry."""
    runtime_module = importlib.import_module("modal_llm_runtime")
    listener_reader = getattr(
        runtime_module,
        "triton_compile_listener_engine_pids",
        None,
    )
    if not callable(listener_reader):
        raise RuntimeError(
            "Modal LLM runtime does not expose "
            "triton_compile_listener_engine_pids()."
        )
    return tuple(int(pid) for pid in listener_reader())


def _llm_compile_miss_checkpoint(
    payload: Mapping[str, Any],
) -> _LLMCompileMissCheckpoint | None:
    """Capture the current miss signal for an executable ModalLLM subgraph."""
    profiles = _llm_profiles_in_payload(payload)
    if not profiles:
        return None
    return _LLMCompileMissCheckpoint(
        profiles=profiles,
        signal_size=_triton_compile_miss_signal_size(),
        listener_engine_pids=_triton_compile_listener_engine_pids(),
    )


def _commit_actual_llm_compile_cache(
    checkpoint: _LLMCompileMissCheckpoint | None,
    compile_cache_volume: Any | None,
) -> bool:
    """Commit the compile-cache Volume after a genuine Triton disk-cache miss."""
    if checkpoint is None or compile_cache_volume is None:
        return False
    listener_engine_pids = _triton_compile_listener_engine_pids()
    if not listener_engine_pids:
        logger.warning(
            "Skipping LLM compile-cache commit because no live vLLM EngineCore "
            "reported the cache-aware Triton listener profiles=%s "
            "listener_pids_before=%s.",
            checkpoint.profiles,
            checkpoint.listener_engine_pids,
        )
        return False
    signal_size = _triton_compile_miss_signal_size()
    if signal_size < checkpoint.signal_size:
        raise RuntimeError(
            "Triton compile-miss signal shrank during LLM execution: "
            f"before={checkpoint.signal_size} after={signal_size}."
        )
    if signal_size == checkpoint.signal_size:
        logger.info(
            "Skipping LLM compile-cache commit because every Triton lookup hit "
            "the persistent cache profiles=%s signal_size=%d listener_pids=%s.",
            checkpoint.profiles,
            signal_size,
            listener_engine_pids,
        )
        return False
    commit_method = getattr(compile_cache_volume, "commit", None)
    if not callable(commit_method):
        raise RuntimeError("Modal compile-cache Volume does not expose commit().")
    with _timed_phase(
        "llm_actual_compile_cache_commit",
        profiles=checkpoint.profiles,
        miss_signal_bytes=signal_size - checkpoint.signal_size,
    ):
        commit_method()
    return True




configure_cloud_session_bridge_hooks(
    CloudSessionBridgeHooks(
        clone_loader_cache_value=_clone_loader_cache_value,
        emit_cloud_info=_emit_cloud_info,
        execute_node_locally_raw=_execute_node_locally_raw,
        execute_subgraph_prompt=_execute_subgraph_prompt,
        is_link=_is_link,
        normalize_prompt_input_value=_normalize_prompt_input_value,
        resolve_required_subgraph_nodes=_resolve_required_subgraph_nodes,
    )
)

configure_cloud_comfy_bootstrap_hooks(
    CloudComfyBootstrapHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
        remote_subgraph_error=RemoteSubgraphExecutionError,
    )
)

configure_cloud_node_output_cache_hooks(
    CloudNodeOutputCacheHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
    )
)

configure_cloud_prompt_execution_hooks(
    CloudPromptExecutionHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
        schedule_remote_cancel_restart=_schedule_remote_cancel_restart,
        remote_subgraph_error=RemoteSubgraphExecutionError,
    )
)

configure_cloud_prompt_server_hooks(
    CloudPromptServerHooks(
        collapse_cache_slot=_collapse_cache_slot,
        emit_cloud_info=_emit_cloud_info,
        meaningful_progress_values=_meaningful_progress_values,
    )
)


if modal is not None:  # pragma: no branch - remote entrypoint configuration.
    settings = globals().get("__comfy_modal_settings_override__") or get_settings()
    __comfy_modal_gpu__ = settings.modal_gpu
    __comfy_modal_app_name__ = modal_deployment_app_name(settings)
    __comfy_modal_secret_name__ = str(
        getattr(settings, "modal_secret_name", DEFAULT_MODAL_SECRET_NAME)
    ).strip()
    _guard_against_existing_modal_app(settings, modal)
    app = modal.App(__comfy_modal_app_name__)
    modal_secret = _modal_secret_from_settings(settings, modal)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    llm_compile_cache_vol = modal.Volume.from_name(
        getattr(
            settings,
            "llm_compile_cache_volume_name",
            f"{settings.volume_name}-llm-compile-cache",
        ),
        create_if_missing=True,
    )
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
    invocation_records = modal.Dict.from_name(
        settings.invocation_dict_name,
        create_if_missing=True,
    )
    snapshot_profiles = modal.Dict.from_name(
        settings.snapshot_profile_dict_name,
        create_if_missing=True,
    )
    register_cloud_runtime_stores(
        session_bridge_cache=session_bridge_cache,
        invocation_records=invocation_records,
        volume=vol,
        snapshot_profiles=snapshot_profiles,
        node_output_cache=node_output_cache,
        interrupt_flags=interrupt_flags,
    )
    custom_node_packages = _custom_node_runtime_packages(settings.custom_nodes_dir)
    torch_build = _select_remote_torch_build(settings.modal_gpu)
    runtime_identity = build_remote_runtime_identity(
        repo_root=_REPO_ROOT,
        comfyui_root=settings.comfyui_root,
        custom_nodes_dir=settings.custom_nodes_dir,
        settings=settings,
    )
    logger.info(
        "Building Modal runtime fingerprint=%s protocol=%d python=%s.",
        runtime_identity.fingerprint,
        _REMOTE_APP_PROTOCOL_VERSION,
        REMOTE_PYTHON_VERSION,
    )
    logger.info(
        "Selected Modal PyTorch build gpu=%s cuda=%s install_layers=%s.",
        settings.modal_gpu,
        torch_build.cuda_version,
        torch_build.install_layers,
    )
    image = (
        modal.Image.debian_slim(python_version=REMOTE_PYTHON_VERSION)
        .apt_install(*_comfyui_apt_packages())
        .pip_install(*_comfyui_runtime_packages())
    )
    image = _install_custom_node_packages(image, custom_node_packages)
    image = _install_remote_torch_build(image, torch_build)
    image = _install_remote_accelerator_packages(image, settings.modal_gpu)
    image = image.env(
        _modal_image_environment(settings, runtime_identity.fingerprint)
    )
    image = image.add_local_dir(
        _REPO_ROOT,
        remote_path="/root/comfyui_modal_sync_repo",
        ignore=_should_ignore_repo_path,
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

    stager_image = (
        modal.Image.debian_slim(python_version=REMOTE_PYTHON_VERSION)
        .env(
            _model_stager_image_environment(
                settings,
                runtime_identity.fingerprint,
            )
        )
        .pip_install(*_remote_huggingface_packages())
        .run_commands(_remote_huggingface_validation_command())
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/comfyui_modal_sync_repo",
            ignore=_should_ignore_repo_path,
        )
    )

    @app.cls(
        image=stager_image,
        volumes={settings.remote_storage_root: vol},
        secrets=[modal_secret],
        cpu=4.0,
        memory=16384,
        max_containers=1,
        scaledown_window=300,
        timeout=7200,
    )
    @modal.concurrent(max_inputs=1)
    class ModelStager:
        """Resolve and stage pinned Hugging Face snapshots without consuming GPU time."""

        def _stage_profiles(
            self,
            model_references: list[str],
            resolved_profiles: Mapping[str, Any] | None = None,
            progress_callback: Callable[[dict[str, Any]], None] | None = None,
        ) -> list[dict[str, Any]]:
            """Resolve and stage profiles while optionally publishing progress."""
            staged_profiles = resolve_and_stage_model_references(
                model_references,
                settings.remote_storage_root,
                resolved_profiles=resolved_profiles,
                owner_id=f"modal:{os.getpid()}:{time.time_ns()}",
                progress_callback=(
                    lambda progress: progress_callback(
                        {
                            "stage": progress.stage,
                            "message": progress.message,
                            "value": progress.value,
                            "max": progress.maximum,
                            "unit": progress.unit,
                            "indeterminate": progress.indeterminate,
                            "model_reference": progress.model_reference,
                        }
                    )
                    if progress_callback is not None
                    else None
                ),
            )
            results = [profile.to_dict() for profile in staged_profiles]
            if any(
                result["downloaded"]
                or result["manifest_created"]
                for result in results
            ):
                vol.commit()
            else:
                logger.info(
                    "Skipping Modal Volume commit because all requested LLM "
                    "profiles and weights were already durable."
                )
            logger.info(
                "Modal LLM CPU resolution and staging completed for models=%s.",
                model_references,
            )
            return results

        @modal.method()
        def stage_profiles(
            self,
            model_references: list[str],
            resolved_profiles: Mapping[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            """Resolve model references, stage snapshots, and return metadata."""
            return self._stage_profiles(model_references, resolved_profiles)

        @modal.method()
        def stage_profiles_stream(
            self,
            model_references: list[str],
            resolved_profiles: Mapping[str, Any] | None = None,
        ) -> Iterator[dict[str, Any]]:
            """Stream CPU staging progress and finish with resolved profile data."""
            progress_events: queue.Queue[dict[str, Any]] = queue.Queue()
            results: list[list[dict[str, Any]]] = []
            errors: list[Exception] = []

            def run_staging() -> None:
                """Run blocking Hugging Face work while the generator yields events."""
                try:
                    results.append(
                        self._stage_profiles(
                            model_references,
                            resolved_profiles,
                            progress_events.put,
                        )
                    )
                except Exception as error:
                    errors.append(error)

            staging_thread = threading.Thread(
                target=run_staging,
                name="modal-llm-model-stager",
                daemon=True,
            )
            staging_thread.start()
            while staging_thread.is_alive() or not progress_events.empty():
                try:
                    progress = progress_events.get(timeout=0.25)
                except queue.Empty:
                    continue
                yield {"kind": "progress", **progress}
            staging_thread.join()
            if errors:
                raise errors[0]
            yield {"kind": "result", "results": results[0] if results else []}

        @modal.method()
        def runtime_version(self) -> dict[str, Any]:
            """Return deployment identity without allocating a GPU container."""
            return {
                "protocol_version": _REMOTE_APP_PROTOCOL_VERSION,
                "app_name": __comfy_modal_app_name__,
                "runtime_fingerprint": os.environ.get(
                    "COMFY_MODAL_RUNTIME_FINGERPRINT",
                    "",
                ),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            }

    @app.cls(
        **_remote_engine_cls_options(
            settings,
            vol,
            image,
            modal_secret,
            llm_compile_cache_vol,
        )
    )
    @modal.concurrent(max_inputs=1)
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""

        snapshot_profile_key: str = modal.parameter(default="")
        gpu_snapshot_enabled: bool = modal.parameter(default=False)
        worker_affinity_key: str = modal.parameter(default="worker-pool:slot:0")

        @modal.enter(snap=True)
        def setup_snapshot_state(self) -> None:
            """Prepare snapshot-friendly runtime state before Modal captures memory."""
            with _timed_phase("remote_engine_setup_snapshot"):
                _prewarm_snapshot_state(
                    gpu_snapshot_enabled=bool(self.gpu_snapshot_enabled),
                    snapshot_profile_key=self.snapshot_profile_key,
                )
                logger.info(
                    "RemoteEngine snapshot setup complete for snapshot_profile_key=%s gpu_snapshot_enabled=%s worker_affinity=%s.",
                    self.snapshot_profile_key or None,
                    bool(self.gpu_snapshot_enabled),
                    self.worker_affinity_key,
                )

        @modal.enter(snap=False)
        def setup_restored_runtime(self) -> None:
            """Prepare request-serving runtime state after a fresh boot or snapshot restore."""
            with _timed_phase("remote_engine_setup_restored"):
                _prewarm_restored_runtime(llm_compile_cache_vol)
                logger.info(
                    "RemoteEngine restored-runtime setup complete for snapshot_profile_key=%s.",
                    self.snapshot_profile_key or None,
                )

        @modal.method()
        def execute_payload(
            self, payload: dict[str, Any], kwargs_payload: bytes
        ) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            _observe_remote_workflow_for_llm_mode(payload)
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            try:
                with _registered_remote_execution(payload) as execution_control:
                    with _timed_phase(
                        "remote_engine_execute_payload",
                        component=component_id,
                        payload_kind=payload.get("payload_kind"),
                    ):
                        _hydrate_missing_payload_volume_paths(vol, payload)
                        if _should_reload_modal_volume(payload):
                            _reload_modal_volume_for_request(
                                vol,
                                str(component_id),
                                reload_marker=reload_marker,
                                payload=payload,
                            )
                        else:
                            _emit_modal_volume_reload_skip(component_id, payload)

                        def execute_once() -> bytes:
                            """Execute the underlying payload once inside this request context."""
                            if payload.get("payload_kind") == "canary":
                                return _execute_canary_payload(
                                    payload,
                                    kwargs_payload,
                                    cancellation_event=execution_control.cancellation_event,
                                    interrupt_store=interrupt_flags,
                                    interrupt_flag_key=execution_control.interrupt_flag_key,
                                )
                            if payload.get("payload_kind") == "mapped_subgraph":
                                custom_nodes_root = _extract_custom_nodes_bundle(
                                    payload.get("custom_nodes_bundle")
                                )
                                _ensure_comfy_runtime_initialized(custom_nodes_root)
                                hydrated_inputs = deserialize_node_inputs(
                                    kwargs_payload
                                )
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

                        compile_miss_checkpoint = _llm_compile_miss_checkpoint(payload)
                        result = _execute_with_durable_invocation(
                            payload, execute_once
                        )
                        _commit_actual_llm_compile_cache(
                            compile_miss_checkpoint,
                            llm_compile_cache_vol,
                        )
                        return result
            except Exception as exc:
                _maybe_schedule_container_termination_on_error(payload, exc)
                raise

        @modal.method()
        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Prime the current or a newly started Modal container for one prompt."""
            return _prepare_warm_container_for_request(
                vol,
                payload,
                llm_compile_cache_vol,
            )

        @modal.method()
        def keepalive_for_local_gap(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Keep this affinity slot active while the workflow executes locally."""
            logger.info(
                "Remote local-gap keepalive prompt=%s component=%s worker_affinity=%s.",
                payload.get("prompt_id"),
                payload.get("component_id"),
                self.worker_affinity_key,
            )
            return {
                "component_id": str(payload.get("component_id") or "modal-keepalive"),
                "task_id": os.getenv("MODAL_TASK_ID"),
                "worker_affinity_key": self.worker_affinity_key,
            }

        @modal.method()
        def runtime_version(self) -> dict[str, Any]:
            """Return the deployed runtime identity expected by the local client."""
            return {
                "protocol_version": _REMOTE_APP_PROTOCOL_VERSION,
                "app_name": __comfy_modal_app_name__,
                "runtime_fingerprint": os.environ.get(
                    "COMFY_MODAL_RUNTIME_FINGERPRINT",
                    "",
                ),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "vllm_version": importlib.metadata.version("vllm"),
            }

        @modal.method()
        def execute_payload_stream(
            self,
            payload: dict[str, Any],
            kwargs_payload: bytes,
        ) -> Iterator[dict[str, Any]]:
            """Stream progress envelopes and a final serialized result for one payload."""
            _observe_remote_workflow_for_llm_mode(payload)
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            try:
                with _registered_remote_execution(payload) as execution_control:
                    with _timed_phase(
                        "remote_engine_execute_payload",
                        component=component_id,
                        payload_kind=payload.get("payload_kind"),
                    ):
                        _hydrate_missing_payload_volume_paths(vol, payload)
                        if _should_reload_modal_volume(payload):
                            _reload_modal_volume_for_request(
                                vol,
                                str(component_id),
                                reload_marker=reload_marker,
                                payload=payload,
                            )
                        else:
                            _emit_modal_volume_reload_skip(component_id, payload)
                        compile_miss_checkpoint = _llm_compile_miss_checkpoint(payload)
                        yield from _stream_remote_payload_events(
                            payload,
                            kwargs_payload,
                            cancellation_event=execution_control.cancellation_event,
                            interrupt_store=interrupt_flags,
                            interrupt_flag_key=execution_control.interrupt_flag_key,
                        )
                        _commit_actual_llm_compile_cache(
                            compile_miss_checkpoint,
                            llm_compile_cache_vol,
                        )
            except Exception as exc:
                _maybe_schedule_container_termination_on_error(payload, exc)
                raise

else:
    app = None
    RemoteEngine = None
