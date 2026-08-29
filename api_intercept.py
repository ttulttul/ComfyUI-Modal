"""Prompt interception and graph rewriting for Modal-backed execution."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from aiohttp import web

if __package__:
    from .modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_MAP_INPUT_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        ensure_modal_parallel_local_passthrough_registered,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from .session_state import RemoteSessionHandle
    from .settings import (
        ModalSyncSettings,
        discover_comfyui_user_directory,
        get_settings,
        modal_gpu_from_workflow,
        settings_for_modal_gpu,
    )
    from .execution_environments import (
        ComponentResourceRequirements,
        CostAwareEnvironmentScheduler,
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionAssignment,
        ExecutionPolicy,
        ExecutionProvider,
        GpuCapability,
        NoCompatibleExecutionEnvironmentError,
        WorkflowExecutionPreferences,
    )
    from .execution_history import ExecutionHistory
    from .sync_engine import (
        AssetSyncRequestCache,
        MODEL_FILE_EXTENSIONS,
        ModalAssetSyncEngine,
        ModalVolumeBackend,
        SyncCancelledError,
        SyncedAsset,
        begin_r2_writeback_prompt,
        finish_r2_writeback_prompt,
        resolve_model_path,
    )
    from .remote_hosts import RemoteExecutionConfig, RemoteHostRegistry, SshHostConfig
    from .r2_cache import R2CacheClient, R2CacheError, R2StorageUsage
    from .r2_credentials import (
        R2CredentialError,
        R2CredentialStore,
        R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
        request_macos_keychain_unlock,
    )
    from .cloudflare_oauth import setup_r2_oauth_routes
    from .remote_configuration_nodes import (
        REMOTE_CONFIGURATION_NODE_IDS,
        REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
        compile_remote_configuration_set,
    )
    from .remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        R2StorageBackingConfiguration,
        SshRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from .ssh_docker import SshDockerController, SshDockerVolumeBackend
    from .ssh_runtime import SshRuntimeManager
    from .vast_config_node import extract_vast_profiles
    from .vast_service import VastProfileQuote, VastSearchRequirements, VastService
    from .vast_api import VastApiClient
    from .vast_leases import VastLeaseManager, VastLeaseRegistry
    from .remote_plan_types import (
        BoundaryInputSpec,
        BoundaryOutputSpec,
        ComponentExecutionPlan,
        ComponentMemoryEstimate,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        ProducedPhaseOutputSpec,
        PromptGraphLink,
        RemoteComponentPlan,
        RemoteExpansionReason,
        RemoteNodeAnalysis,
        RewriteSummary,
        StaticToMappedBoundarySpec,
        _EnvironmentAssetPreparationResult,
    )
    from .modal_hardware import (
        _HBM_GPU_NAME_MARKERS,
        _MODAL_GPU_COST_USD_PER_SECOND,
        _MODAL_GPU_VRAM_GB,
        _capabilities_hardware_payload,
        _gpu_memory_kind,
        _hardware_payload,
        _modal_hardware_payload,
        _vast_hardware_payload,
    )
    from .intercept_route_paths import (
        _analysis_route_path,
        _cancel_preparation_route_path,
        _container_status_route_path,
        _delete_modal_caches_route_path,
        _delete_modal_volume_route_path,
        _progress_state_route_path,
        _remote_environment_bootstrap_route_path,
        _remote_environment_probe_route_path,
        _remote_environment_status_route_path,
        _remote_environment_stop_route_path,
        _remote_environments_route_path,
    )
    from .modal_admin_ops import (
        _call_modal_sdk,
        _delete_modal_named_object,
        _modal_cache_dict_names,
        _modal_not_found_error_types,
        delete_modal_cache_dicts,
        delete_modal_volume,
    )
    from .modal_ui_events import (
        _emit_modal_status,
        modal_ui_events_for_client,
        record_modal_ui_event,
    )
    from .remote_graph_analysis import (
        _prompt_node_required_provider,
        _iter_payload_input_strings,
        _looks_like_workflow_node,
        _iter_workflow_nodes,
        _workflow_subgraph_definitions,
        _iter_workflow_nodes_with_ancestors,
        _resolve_prompt_node_ids_for_workflow_node,
        _workflow_node_path,
        _extract_marked_workflow_node_paths,
        _build_workflow_prompt_resolution_maps,
        _resolve_requested_prompt_node_ids,
        _best_workflow_path_for_prompt_node,
        _resolve_workflow_node_paths_for_prompt_nodes,
        extract_remote_node_ids,
        requested_remote_node_ids,
        _workflow_node_remote_enabled,
        _normalize_output_metadata,
        _is_transportable_output_type,
        _is_inexpensive_remote_boundary_type,
        _build_consumer_map,
        _sandwiched_local_node_ids,
        _node_output_refs,
        _downstream_node_ids_from_targets,
        _is_non_returning_tap_terminal_node,
        _non_returning_local_tap_node_ids,
        _non_returning_local_output_consumers,
        _output_supports_parallel_local_materialization,
        _expand_component_for_non_transportable_local_outputs,
        _remote_output_io_type,
        _remote_output_is_list,
        _remote_group_dependency_edges,
        _has_alternate_group_path_through_protected_group,
        _remote_component_partition_groups,
        _component_topological_order,
        _component_dependency_graph,
        _component_execution_stages,
        _estimated_stage_parallelism,
        _merge_cyclic_component_groups,
        _build_remote_components,
        _expand_remote_node_ids_for_non_transportable_inputs,
        _terminal_remote_video_source,
        _expand_remote_node_ids_for_terminal_video_sinks,
        analyze_remote_node_selection,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from modal_executor_node import (
        MODAL_ARTIFACT_FINALIZER_MAX_COMPONENTS,
        MODAL_ARTIFACT_FINALIZER_NODE_ID,
        MODAL_COMPONENT_COMPLETION_OUTPUT_NAME,
        MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID,
        MODAL_MAP_INPUT_NODE_ID,
        MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID,
        MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY,
        ensure_modal_artifact_finalizer_registered,
        ensure_modal_component_proxy_node_registered,
        ensure_modal_local_bridge_materializer_registered,
        ensure_modal_parallel_local_passthrough_registered,
        register_cache_friendly_proxy_payload,
        register_modal_map_input_warmup_context,
        registered_proxy_execution_payload,
        update_registered_proxy_payload_fields,
    )
    from session_state import RemoteSessionHandle
    from settings import (
        ModalSyncSettings,
        discover_comfyui_user_directory,
        get_settings,
        modal_gpu_from_workflow,
        settings_for_modal_gpu,
    )
    from execution_environments import (
        ComponentResourceRequirements,
        CostAwareEnvironmentScheduler,
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionAssignment,
        ExecutionPolicy,
        ExecutionProvider,
        GpuCapability,
        NoCompatibleExecutionEnvironmentError,
        WorkflowExecutionPreferences,
    )
    from execution_history import ExecutionHistory
    from sync_engine import (
        AssetSyncRequestCache,
        MODEL_FILE_EXTENSIONS,
        ModalAssetSyncEngine,
        ModalVolumeBackend,
        SyncCancelledError,
        SyncedAsset,
        begin_r2_writeback_prompt,
        finish_r2_writeback_prompt,
        resolve_model_path,
    )
    from remote_hosts import RemoteExecutionConfig, RemoteHostRegistry, SshHostConfig
    from r2_cache import R2CacheClient, R2CacheError, R2StorageUsage
    from r2_credentials import (
        R2CredentialError,
        R2CredentialStore,
        R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
        request_macos_keychain_unlock,
    )
    from cloudflare_oauth import setup_r2_oauth_routes
    from remote_configuration_nodes import (
        REMOTE_CONFIGURATION_NODE_IDS,
        REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
        compile_remote_configuration_set,
    )
    from remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        R2StorageBackingConfiguration,
        SshRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from ssh_docker import SshDockerController, SshDockerVolumeBackend
    from ssh_runtime import SshRuntimeManager
    from vast_config_node import extract_vast_profiles
    from vast_service import VastProfileQuote, VastSearchRequirements, VastService
    from vast_api import VastApiClient
    from vast_leases import VastLeaseManager, VastLeaseRegistry
    from remote_plan_types import (
        BoundaryInputSpec,
        BoundaryOutputSpec,
        ComponentExecutionPlan,
        ComponentMemoryEstimate,
        InputTarget,
        LinkedOutputRef,
        ModalPromptValidationError,
        ProducedPhaseOutputSpec,
        PromptGraphLink,
        RemoteComponentPlan,
        RemoteExpansionReason,
        RemoteNodeAnalysis,
        RewriteSummary,
        StaticToMappedBoundarySpec,
        _EnvironmentAssetPreparationResult,
    )
    from modal_hardware import (
        _HBM_GPU_NAME_MARKERS,
        _MODAL_GPU_COST_USD_PER_SECOND,
        _MODAL_GPU_VRAM_GB,
        _capabilities_hardware_payload,
        _gpu_memory_kind,
        _hardware_payload,
        _modal_hardware_payload,
        _vast_hardware_payload,
    )
    from intercept_route_paths import (
        _analysis_route_path,
        _cancel_preparation_route_path,
        _container_status_route_path,
        _delete_modal_caches_route_path,
        _delete_modal_volume_route_path,
        _progress_state_route_path,
        _remote_environment_bootstrap_route_path,
        _remote_environment_probe_route_path,
        _remote_environment_status_route_path,
        _remote_environment_stop_route_path,
        _remote_environments_route_path,
    )
    from modal_admin_ops import (
        _call_modal_sdk,
        _delete_modal_named_object,
        _modal_cache_dict_names,
        _modal_not_found_error_types,
        delete_modal_cache_dicts,
        delete_modal_volume,
    )
    from modal_ui_events import (
        _emit_modal_status,
        modal_ui_events_for_client,
        record_modal_ui_event,
    )
    from remote_graph_analysis import (
        _prompt_node_required_provider,
        _iter_payload_input_strings,
        _looks_like_workflow_node,
        _iter_workflow_nodes,
        _workflow_subgraph_definitions,
        _iter_workflow_nodes_with_ancestors,
        _resolve_prompt_node_ids_for_workflow_node,
        _workflow_node_path,
        _extract_marked_workflow_node_paths,
        _build_workflow_prompt_resolution_maps,
        _resolve_requested_prompt_node_ids,
        _best_workflow_path_for_prompt_node,
        _resolve_workflow_node_paths_for_prompt_nodes,
        extract_remote_node_ids,
        requested_remote_node_ids,
        _workflow_node_remote_enabled,
        _normalize_output_metadata,
        _is_transportable_output_type,
        _is_inexpensive_remote_boundary_type,
        _build_consumer_map,
        _sandwiched_local_node_ids,
        _node_output_refs,
        _downstream_node_ids_from_targets,
        _is_non_returning_tap_terminal_node,
        _non_returning_local_tap_node_ids,
        _non_returning_local_output_consumers,
        _output_supports_parallel_local_materialization,
        _expand_component_for_non_transportable_local_outputs,
        _remote_output_io_type,
        _remote_output_is_list,
        _remote_group_dependency_edges,
        _has_alternate_group_path_through_protected_group,
        _remote_component_partition_groups,
        _component_topological_order,
        _component_dependency_graph,
        _component_execution_stages,
        _estimated_stage_parallelism,
        _merge_cyclic_component_groups,
        _build_remote_components,
        _expand_remote_node_ids_for_non_transportable_inputs,
        _terminal_remote_video_source,
        _expand_remote_node_ids_for_terminal_video_sinks,
        analyze_remote_node_selection,
    )

if __package__:
    from .component_planning import (
        _REMOTE_REPLICA_NODE_PREFIX,
        _REPLICABLE_REMOTE_OBJECT_TYPES,
        _build_component_plan,
        _filter_boundary_inputs_for_node_ids,
        _filter_boundary_outputs_for_node_ids,
        _subset_component_prompt,
        _preview_target_node_ids,
        _component_downstream_closure,
        _component_upstream_closure,
        _component_ancestors_of_local_source,
        _component_has_local_reentry_dependency,
        _component_has_parallel_local_remote_fanout,
        _order_execute_node_ids_for_transportable_splits,
        _subgraph_topological_node_order,
        _remote_dependency_closure,
        _can_replicate_remote_dependency_closure,
        _replica_node_id_mapping,
        _install_remote_dependency_replica,
        _cross_group_replicable_boundaries,
        _replicate_safe_nontransportable_provider_boundaries,
        _workflow_visible_remote_node_ids,
        _build_component_plans,
        _mark_remote_to_remote_session_boundaries,
        _remote_session_component_ids,
        _boundary_output_payload,
        _proxy_boundary_output_is_list,
        _implicitly_mapped_boundary_output_sources,
        _mark_payload_scheduler_list_outputs,
        _mapped_boundary_origin_io_type,
        _describe_output_boundary_error,
        _describe_input_boundary_error,
        validate_remote_component_transport_compatibility,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from component_planning import (
        _REMOTE_REPLICA_NODE_PREFIX,
        _REPLICABLE_REMOTE_OBJECT_TYPES,
        _build_component_plan,
        _filter_boundary_inputs_for_node_ids,
        _filter_boundary_outputs_for_node_ids,
        _subset_component_prompt,
        _preview_target_node_ids,
        _component_downstream_closure,
        _component_upstream_closure,
        _component_ancestors_of_local_source,
        _component_has_local_reentry_dependency,
        _component_has_parallel_local_remote_fanout,
        _order_execute_node_ids_for_transportable_splits,
        _subgraph_topological_node_order,
        _remote_dependency_closure,
        _can_replicate_remote_dependency_closure,
        _replica_node_id_mapping,
        _install_remote_dependency_replica,
        _cross_group_replicable_boundaries,
        _replicate_safe_nontransportable_provider_boundaries,
        _workflow_visible_remote_node_ids,
        _build_component_plans,
        _mark_remote_to_remote_session_boundaries,
        _remote_session_component_ids,
        _boundary_output_payload,
        _proxy_boundary_output_is_list,
        _implicitly_mapped_boundary_output_sources,
        _mark_payload_scheduler_list_outputs,
        _mapped_boundary_origin_io_type,
        _describe_output_boundary_error,
        _describe_input_boundary_error,
        validate_remote_component_transport_compatibility,
    )

if __package__:
    from .prompt_payload_metadata import (
        _ROOT_LOADER_PREWARM_CLASS_TYPES,
        _prompt_value_signature_fragment,
        _prompt_node_signature_digest,
        _iter_loader_snapshot_prompt_payloads,
        _is_root_literal_loader_node,
        _loader_prewarm_plan_signature,
        _uses_llm_worker_affinity,
        _payload_loader_snapshot_profile_key,
        _stamp_snapshot_profile_key,
        _attach_snapshot_profile_key,
        _resolved_llm_profile_entry,
        _attach_resolved_llm_profiles,
        _boundary_source_signature,
        _serialize_boundary_input_specs,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from prompt_payload_metadata import (
        _ROOT_LOADER_PREWARM_CLASS_TYPES,
        _prompt_value_signature_fragment,
        _prompt_node_signature_digest,
        _iter_loader_snapshot_prompt_payloads,
        _is_root_literal_loader_node,
        _loader_prewarm_plan_signature,
        _uses_llm_worker_affinity,
        _payload_loader_snapshot_profile_key,
        _stamp_snapshot_profile_key,
        _attach_snapshot_profile_key,
        _resolved_llm_profile_entry,
        _attach_resolved_llm_profiles,
        _boundary_source_signature,
        _serialize_boundary_input_specs,
    )

if __package__:
    from .prompt_rewrite import (
        _SPECULATIVE_PREWARM_TARGET_KEY,
        _SPECULATIVE_PREWARM_PAYLOAD_FIELDS,
        _sync_component_prompt_inputs,
        _deduplicate_synced_assets,
        _build_component_payload,
        _component_uploaded_volume_paths,
        _rewrite_component_into_proxy,
        _modal_component_completion_output_index,
        _attach_modal_artifact_finalizer,
        _nearest_downstream_remote_component_ids,
        _parallelize_non_returning_local_branches,
        _configure_local_gap_keepalive_payloads,
        _remote_proxy_payload,
        _remote_proxy_dependency_edges,
        _component_descendant_distances,
        _speculative_prewarm_target_payload,
        _configure_speculative_affinity_prewarm_payloads,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from prompt_rewrite import (
        _SPECULATIVE_PREWARM_TARGET_KEY,
        _SPECULATIVE_PREWARM_PAYLOAD_FIELDS,
        _sync_component_prompt_inputs,
        _deduplicate_synced_assets,
        _build_component_payload,
        _component_uploaded_volume_paths,
        _rewrite_component_into_proxy,
        _modal_component_completion_output_index,
        _attach_modal_artifact_finalizer,
        _nearest_downstream_remote_component_ids,
        _parallelize_non_returning_local_branches,
        _configure_local_gap_keepalive_payloads,
        _remote_proxy_payload,
        _remote_proxy_dependency_edges,
        _component_descendant_distances,
        _speculative_prewarm_target_payload,
        _configure_speculative_affinity_prewarm_payloads,
    )

if __package__:
    from .execution_scheduling import (
        _modal_environment_state,
        _modal_configuration_environment_state,
        _ssh_host_registry,
        _configured_ssh_hosts,
        _schedulable_ssh_hosts,
        _refresh_ssh_host,
        _prompt_llm_model_references,
        _resolve_prompt_llm_profiles,
        _component_profile_memory_estimate,
        _component_required_provider,
        _iter_prompt_string_values,
        _is_additive_model_node,
        _component_model_asset_sizes,
        _component_memory_estimate,
        _component_execution_signature,
        _execution_history,
        _maximum_capacity_state,
        _reprobe_reclaimed_ssh_host,
        _probe_workflow_ssh_configuration,
        _optional_scheduler_choice,
        _require_scheduler_choice,
        _remove_idle_ssh_workers_for_reclaim,
        _reclaim_improves_assignment,
        _choose_with_idle_ssh_worker_reclaim,
        _plan_component_execution_assignments,
        _plan_component_execution,
        _plan_configured_component_execution,
        _safe_remote_configuration_payload,
        _cached_r2_storage_usage,
        _refresh_r2_storage_usage,
        _r2_storage_from_usage_payload,
        _planned_execution_assignments_payload,
        _configuration_field,
        _configuration_host,
        _assignment_hardware_payload,
        _probe_configured_ssh_hosts,
        _configured_vast_service,
        _configured_component_requirements,
        _configured_candidate_environments,
        _reclaim_idle_configured_ssh_capacity,
        _prefetch_configured_vast_offers,
        _configured_candidate_environment,
        _apply_historical_execution_estimates,
        _prepare_selected_vast_capacity,
        _emit_vast_capacity_status,
        _quote_selected_vast_slot,
        _ssh_sync_engine,
        _workflow_r2_cache,
        _stamp_execution_assignment,
        _ssh_hostname,
        _execution_location_for_assignment,
        _vast_provider_metadata,
        _configured_provider_metadata,
        _ensure_remote_sync_backend,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_scheduling import (
        _modal_environment_state,
        _modal_configuration_environment_state,
        _ssh_host_registry,
        _configured_ssh_hosts,
        _schedulable_ssh_hosts,
        _refresh_ssh_host,
        _prompt_llm_model_references,
        _resolve_prompt_llm_profiles,
        _component_profile_memory_estimate,
        _component_required_provider,
        _iter_prompt_string_values,
        _is_additive_model_node,
        _component_model_asset_sizes,
        _component_memory_estimate,
        _component_execution_signature,
        _execution_history,
        _maximum_capacity_state,
        _reprobe_reclaimed_ssh_host,
        _probe_workflow_ssh_configuration,
        _optional_scheduler_choice,
        _require_scheduler_choice,
        _remove_idle_ssh_workers_for_reclaim,
        _reclaim_improves_assignment,
        _choose_with_idle_ssh_worker_reclaim,
        _plan_component_execution_assignments,
        _plan_component_execution,
        _plan_configured_component_execution,
        _safe_remote_configuration_payload,
        _cached_r2_storage_usage,
        _refresh_r2_storage_usage,
        _r2_storage_from_usage_payload,
        _planned_execution_assignments_payload,
        _configuration_field,
        _configuration_host,
        _assignment_hardware_payload,
        _probe_configured_ssh_hosts,
        _configured_vast_service,
        _configured_component_requirements,
        _configured_candidate_environments,
        _reclaim_idle_configured_ssh_capacity,
        _prefetch_configured_vast_offers,
        _configured_candidate_environment,
        _apply_historical_execution_estimates,
        _prepare_selected_vast_capacity,
        _emit_vast_capacity_status,
        _quote_selected_vast_slot,
        _ssh_sync_engine,
        _workflow_r2_cache,
        _stamp_execution_assignment,
        _ssh_hostname,
        _execution_location_for_assignment,
        _vast_provider_metadata,
        _configured_provider_metadata,
        _ensure_remote_sync_backend,
    )

if __package__:
    from .prompt_diagnostics import (
        _prompt_node_class_type,
        _prompt_graph_links,
        _prompt_link_dict,
        _find_prompt_dependency_cycles,
        _modal_proxy_payload_summaries,
        _modal_rewritten_prompt_diagnostics,
        _log_modal_rewritten_prompt_diagnostics,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from prompt_diagnostics import (
        _prompt_node_class_type,
        _prompt_graph_links,
        _prompt_link_dict,
        _find_prompt_dependency_cycles,
        _modal_proxy_payload_summaries,
        _modal_rewritten_prompt_diagnostics,
        _log_modal_rewritten_prompt_diagnostics,
    )

if __package__:
    from .queue_bridge import (
        _queue_prompt_json,
        _install_modal_interrupt_queue_bridge,
        _queue_item_prompt_ids,
        _cancel_remote_preparation,
        _queued_ssh_environment_ids,
        _set_remote_preparation,
        _clear_remote_preparation,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from queue_bridge import (
        _queue_prompt_json,
        _install_modal_interrupt_queue_bridge,
        _queue_item_prompt_ids,
        _cancel_remote_preparation,
        _queued_ssh_environment_ids,
        _set_remote_preparation,
        _clear_remote_preparation,
    )

if __package__:
    from .route_context import RouteContext
    from .routes_modal_containers import register_modal_container_routes
    from .routes_queue import register_queue_routes
    from .routes_r2 import register_r2_routes
    from .routes_remote_environments import register_remote_environment_routes
    from .routes_vast import register_vast_routes
else:  # pragma: no cover - flat import inside the Modal container.
    from route_context import RouteContext
    from routes_modal_containers import register_modal_container_routes
    from routes_queue import register_queue_routes
    from routes_r2 import register_r2_routes
    from routes_remote_environments import register_remote_environment_routes
    from routes_vast import register_vast_routes

logger = logging.getLogger(__name__)
SetupStatusCallback = Callable[[str, int | None, int | None], None]
EnvironmentSetupStatusCallback = Callable[
    [str, str, int | None, int | None], None
]
ExecutionPlanStatusCallback = Callable[
    [dict[str, dict[str, Any]], list[dict[str, Any]]], None
]

_ROUTE_REGISTERED = False
_REMOTE_ASSET_PREPARATION_MAX_WORKERS = 8


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












def _environment_setup_status_callback(
    environment_id: str,
    status_callback: SetupStatusCallback | None,
    environment_status_callback: EnvironmentSetupStatusCallback | None,
) -> SetupStatusCallback | None:
    """Return a setup callback that updates prompt-wide and environment UI."""
    if status_callback is None and environment_status_callback is None:
        return None

    def emit_status(
        message: str,
        current: int | None,
        total: int | None,
    ) -> None:
        """Forward one setup update with its concrete execution environment."""
        if status_callback is not None:
            status_callback(message, current, total)
        if environment_status_callback is not None:
            environment_status_callback(environment_id, message, current, total)

    return emit_status


def _prepare_environment_assets(
    *,
    environment_id: str,
    components: Sequence[RemoteComponentPlan],
    sync_engine: ModalAssetSyncEngine,
    engine_lock: threading.Lock,
    rewritten_prompt: dict[str, Any],
    sync_custom_nodes: bool,
    completion_message: str,
    status_callback: SetupStatusCallback | None,
    environment_status_callback: EnvironmentSetupStatusCallback | None,
) -> _EnvironmentAssetPreparationResult:
    """Prepare one environment's custom nodes and prompt assets in order."""
    started_at = time.perf_counter()
    logger.info(
        "Starting remote asset preparation environment=%s components=%d.",
        environment_id,
        len(components),
    )
    environment_callback = _environment_setup_status_callback(
        environment_id,
        status_callback,
        environment_status_callback,
    )
    if environment_callback is not None:
        environment_callback("Preparing remote assets", None, None)
    with engine_lock:
        sync_engine.preflight_r2_access(status_callback=environment_callback)
        custom_nodes_bundle = (
            sync_engine.sync_custom_nodes_directory(
                status_callback=environment_callback,
            )
            if sync_custom_nodes
            else None
        )
        component_prompts, assets_by_component_id = (
            _sync_environment_prompt_assets(
                components=components,
                sync_engine=sync_engine,
                rewritten_prompt=rewritten_prompt,
                status_callback=environment_callback,
            )
        )
    if environment_callback is not None:
        environment_callback(completion_message, None, None)
    logger.info(
        "Finished remote asset preparation environment=%s components=%d assets=%d elapsed_seconds=%.3f.",
        environment_id,
        len(components),
        sum(len(assets) for assets in assets_by_component_id.values()),
        time.perf_counter() - started_at,
    )
    return _EnvironmentAssetPreparationResult(
        environment_id=environment_id,
        custom_nodes_bundle=custom_nodes_bundle,
        component_prompts=component_prompts,
        assets_by_component_id=assets_by_component_id,
    )


def _sync_environment_prompt_assets(
    *,
    components: Sequence[RemoteComponentPlan],
    sync_engine: ModalAssetSyncEngine,
    rewritten_prompt: dict[str, Any],
    status_callback: SetupStatusCallback | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[SyncedAsset]]]:
    """Sync every component assigned to one environment with one request cache."""
    request_cache = sync_engine.create_request_asset_cache(
        rewritten_prompt[node_id].get("inputs", {})
        for component in components
        for node_id in component.node_ids
    )
    component_prompts: dict[str, dict[str, Any]] = {}
    assets_by_component_id: dict[str, list[SyncedAsset]] = {}
    for component in components:
        component_prompt, synced_assets = _sync_component_prompt_inputs(
            component=component,
            rewritten_prompt=rewritten_prompt,
            sync_engine=sync_engine,
            request_cache=request_cache,
            status_callback=status_callback,
        )
        component_id = component.representative_node_id
        component_prompts[component_id] = component_prompt
        assets_by_component_id[component_id] = list(synced_assets)
    return component_prompts, assets_by_component_id


def _prepare_remote_environment_assets(
    *,
    components: Sequence[RemoteComponentPlan],
    assignments_by_component_id: Mapping[str, ExecutionAssignment],
    sync_engines_by_environment: Mapping[str, ModalAssetSyncEngine],
    rewritten_prompt: dict[str, Any],
    sync_custom_nodes: bool,
    status_callback: SetupStatusCallback | None,
    environment_status_callback: EnvironmentSetupStatusCallback | None,
) -> dict[str, _EnvironmentAssetPreparationResult]:
    """Prepare distinct remote environments concurrently in a bounded pool."""
    components_by_environment: dict[str, list[RemoteComponentPlan]] = {
        environment_id: [] for environment_id in sync_engines_by_environment
    }
    providers_by_environment: dict[str, ExecutionProvider] = {}
    for component in components:
        assignment = assignments_by_component_id[component.representative_node_id]
        components_by_environment[assignment.environment_id].append(component)
        providers_by_environment[assignment.environment_id] = assignment.provider
    locks_by_engine: dict[int, threading.Lock] = {}
    environment_ids = list(components_by_environment)
    if not environment_ids:
        return {}
    max_workers = min(_REMOTE_ASSET_PREPARATION_MAX_WORKERS, len(environment_ids))
    logger.info(
        "Preparing remote assets across %d environments with %d workers.",
        len(environment_ids),
        max_workers,
    )
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="comfy-remote-assets",
    ) as executor:
        futures_by_environment = {
            environment_id: executor.submit(
                _prepare_environment_assets,
                environment_id=environment_id,
                components=components_by_environment[environment_id],
                sync_engine=sync_engines_by_environment[environment_id],
                engine_lock=locks_by_engine.setdefault(
                    id(sync_engines_by_environment[environment_id]),
                    threading.Lock(),
                ),
                rewritten_prompt=rewritten_prompt,
                sync_custom_nodes=sync_custom_nodes,
                completion_message=_asset_preparation_completion_message(
                    providers_by_environment[environment_id]
                ),
                status_callback=status_callback,
                environment_status_callback=environment_status_callback,
            )
            for environment_id in environment_ids
        }
        return {
            environment_id: futures_by_environment[environment_id].result()
            for environment_id in environment_ids
        }


def _asset_preparation_completion_message(provider: ExecutionProvider) -> str:
    """Return the truthful state after assets, but not runtime startup, finish."""
    if provider is ExecutionProvider.SSH_DOCKER:
        return "Remote assets prepared; SSH runtime starts on dispatch"
    if provider is ExecutionProvider.MODAL:
        return "Remote assets prepared; Modal runtime starts on dispatch"
    return "Ready for remote execution"


def rewrite_prompt_for_modal(
    prompt: dict[str, Any],
    workflow: dict[str, Any] | None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
    nodes_module: Any | None = None,
    extra_data: dict[str, Any] | None = None,
    status_callback: SetupStatusCallback | None = None,
    environment_status_callback: EnvironmentSetupStatusCallback | None = None,
    plan_callback: ExecutionPlanStatusCallback | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    occupied_environment_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], RewriteSummary]:
    """Rewrite connected remote components into Modal proxy nodes."""
    if cancellation_check is not None and cancellation_check():
        raise SyncCancelledError("Remote workflow preparation was cancelled.")
    resolved_settings = settings or get_settings()
    remote_node_ids = requested_remote_node_ids(
        prompt=prompt,
        workflow=workflow,
        settings=resolved_settings,
    )
    summary = RewriteSummary(remote_node_ids=sorted(remote_node_ids))
    logger.info(
        "Found %d workflow nodes marked for remote execution.", len(remote_node_ids)
    )

    if not remote_node_ids:
        return copy.deepcopy(prompt), summary

    resolved_nodes_module = nodes_module or _get_nodes_module()
    resolved_sync_engine = sync_engine or ModalAssetSyncEngine.from_environment(
        resolved_settings
    )
    rewritten_prompt = copy.deepcopy(prompt)
    expanded_remote_node_ids, _ = _expand_remote_node_ids_for_non_transportable_inputs(
        prompt=rewritten_prompt,
        remote_node_ids=remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    expanded_remote_node_ids = _expand_remote_node_ids_for_terminal_video_sinks(
        prompt=rewritten_prompt,
        remote_node_ids=expanded_remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    summary.remote_node_ids = sorted(expanded_remote_node_ids)
    summary.sandwiched_local_node_ids = sorted(
        _sandwiched_local_node_ids(rewritten_prompt, expanded_remote_node_ids)
    )
    if summary.sandwiched_local_node_ids:
        logger.warning(
            "Detected local nodes sandwiched between remote graph regions; "
            "these nodes may force additional remote phases: %s",
            summary.sandwiched_local_node_ids,
        )
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

    execution_plan = _plan_component_execution(
        components=components,
        prompt=rewritten_prompt,
        workflow=workflow,
        settings=resolved_settings,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
        plan_callback=plan_callback,
        occupied_environment_ids=occupied_environment_ids,
    )
    assignments_by_component_id = execution_plan.assignments
    if execution_plan.configuration_set is not None:
        summary.remote_configurations = list(execution_plan.safe_configurations)
    for assignment in assignments_by_component_id.values():
        execution_location = _execution_location_for_assignment(
            assignment,
            execution_plan.ssh_hosts_by_id,
            execution_plan.vast_leases_by_environment,
        )
        if execution_location:
            summary.execution_locations_by_environment[
                assignment.environment_id
            ] = execution_location
    session_component_ids = _mark_remote_to_remote_session_boundaries(
        rewritten_prompt,
        components,
        resolved_nodes_module,
        assignments_by_component_id,
    )
    prompt_id = (extra_data or {}).get("prompt_id")
    remote_sessions_by_component_id = {
        component.representative_node_id: RemoteSessionHandle(
            session_id=uuid.uuid4().hex,
            prompt_id=(str(prompt_id) if prompt_id is not None else None),
            owner_component_id=component.representative_node_id,
        ).to_payload()
        for component in components
        if component.representative_node_id in session_component_ids
    }
    if remote_sessions_by_component_id:
        logger.info(
            "Enabled environment-local remote references for components=%s.",
            sorted(remote_sessions_by_component_id),
        )
    summary.execution_assignments_by_representative = dict(assignments_by_component_id)

    vast_service = execution_plan.vast_service
    vast_leases_by_environment = dict(
        execution_plan.vast_leases_by_environment
    )
    if vast_service is None and any(
        assignment.provider is ExecutionProvider.VAST
        for assignment in assignments_by_component_id.values()
    ):
        try:
            vast_service = VastService.from_environment(
                resolved_settings,
                repo_root=Path(__file__).resolve().parent,
            )
            vast_leases_by_environment = {
                assignment.environment_id: vast_service.lease_for_environment_id(
                    assignment.environment_id
                )
                for assignment in assignments_by_component_id.values()
                if assignment.provider is ExecutionProvider.VAST
            }
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise ModalPromptValidationError(
                f"Unable to resolve the acquired Vast.ai lease: {exc}"
            ) from exc

    workflow_r2_cache = (
        _workflow_r2_cache(execution_plan.configuration_set)
        if any(
            assignment.provider is not ExecutionProvider.MODAL
            for assignment in assignments_by_component_id.values()
        )
        else None
    )
    if workflow_r2_cache is not None and vast_service is not None:
        vast_service.r2_cache = workflow_r2_cache

    sync_engines_by_environment: dict[str, ModalAssetSyncEngine] = {}
    ssh_hosts_by_id = dict(execution_plan.ssh_hosts_by_id)
    if not ssh_hosts_by_id:
        ssh_hosts_by_id = {
            host.environment_id: host
            for host in _configured_ssh_hosts(resolved_settings)
        }
    session_environment_ids = {
        assignments_by_component_id[component_id].environment_id
        for component_id in session_component_ids
    }
    worker_counts_by_environment: dict[str, int] = defaultdict(int)
    for component in components:
        component_id = component.representative_node_id
        assignment = assignments_by_component_id[component_id]
        if assignment.provider is ExecutionProvider.MODAL:
            summary.execution_worker_indices_by_representative[component_id] = 0
            continue
        if assignment.provider is ExecutionProvider.VAST:
            summary.execution_worker_indices_by_representative[component_id] = 0
            continue
        if execution_plan.configuration_set is not None:
            summary.execution_worker_indices_by_representative[component_id] = (
                assignment.capacity_slot_index
            )
            logger.info(
                "Assigned workflow-configured remote component=%s environment=%s "
                "worker_index=%d.",
                component_id,
                assignment.environment_id,
                assignment.capacity_slot_index,
            )
            continue
        host = ssh_hosts_by_id.get(assignment.environment_id)
        maximum_workers = max(1, host.maximum_workers if host is not None else 1)
        if component_id in session_component_ids:
            worker_index = 0
        else:
            first_worker = (
                1
                if assignment.environment_id in session_environment_ids
                and maximum_workers > 1
                else 0
            )
            available_workers = max(1, maximum_workers - first_worker)
            worker_index = first_worker + (
                worker_counts_by_environment[assignment.environment_id]
                % available_workers
            )
            worker_counts_by_environment[assignment.environment_id] += 1
        summary.execution_worker_indices_by_representative[component_id] = worker_index
        logger.info(
            "Assigned remote component=%s environment=%s worker_index=%d.",
            component_id,
            assignment.environment_id,
            worker_index,
        )
    for assignment in assignments_by_component_id.values():
        if assignment.environment_id in sync_engines_by_environment:
            continue
        if assignment.provider is ExecutionProvider.MODAL:
            _ensure_remote_sync_backend(resolved_settings, resolved_sync_engine)
            sync_engines_by_environment[
                assignment.environment_id
            ] = resolved_sync_engine
            continue
        if assignment.provider is ExecutionProvider.VAST:
            if vast_service is None:
                raise ModalPromptValidationError(
                    "Vast.ai assignment has no active controller service."
                )
            lease = vast_leases_by_environment[assignment.environment_id]
            vast_sync_engine = vast_service.sync_engine(lease)
            vast_sync_engine.cancellation_check = cancellation_check
            sync_engines_by_environment[assignment.environment_id] = vast_sync_engine
            continue
        host = ssh_hosts_by_id.get(assignment.environment_id)
        if host is None:
            raise ModalPromptValidationError(
                f"Assigned SSH execution environment {assignment.environment_id!r} is no longer configured."
            )
        sync_engines_by_environment[assignment.environment_id] = _ssh_sync_engine(
            host=host,
            settings=resolved_settings,
            r2_cache=workflow_r2_cache,
        )

    if status_callback is not None:
        status_callback("Preparing assets for remote execution", None, None)

    try:
        environment_preparations = _prepare_remote_environment_assets(
            components=components,
            assignments_by_component_id=assignments_by_component_id,
            sync_engines_by_environment=sync_engines_by_environment,
            rewritten_prompt=rewritten_prompt,
            sync_custom_nodes=resolved_settings.sync_custom_nodes,
            status_callback=status_callback,
            environment_status_callback=environment_status_callback,
        )
    except R2CacheError as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    if resolved_settings.sync_custom_nodes:
        summary.custom_nodes_bundles_by_environment = {
            environment_id: preparation.custom_nodes_bundle
            for environment_id, preparation in environment_preparations.items()
        }
        modal_bundle = next(
            (
                summary.custom_nodes_bundles_by_environment[assignment.environment_id]
                for assignment in assignments_by_component_id.values()
                if assignment.provider is ExecutionProvider.MODAL
            ),
            None,
        )
        summary.custom_nodes_bundle = modal_bundle or next(
            iter(summary.custom_nodes_bundles_by_environment.values()),
            None,
        )
    else:
        logger.info(
            "Skipping custom_nodes bundle sync because sync is disabled for execution_mode=%s.",
            resolved_settings.execution_mode,
        )

    synced_component_prompts: dict[str, dict[str, Any]] = {}
    synced_assets_by_component_id: dict[str, list[SyncedAsset]] = {}
    for component in components:
        assignment = assignments_by_component_id[component.representative_node_id]
        preparation = environment_preparations[assignment.environment_id]
        component_id = component.representative_node_id
        synced_component_prompts[component_id] = preparation.component_prompts[
            component_id
        ]
        synced_assets_by_component_id[component_id] = (
            preparation.assets_by_component_id[component_id]
        )
        summary.synced_assets.extend(
            preparation.assets_by_component_id[component_id]
        )

    summary.synced_assets = _deduplicate_synced_assets(summary.synced_assets)

    modal_environment_ids = {
        assignment.environment_id
        for assignment in assignments_by_component_id.values()
        if assignment.provider is ExecutionProvider.MODAL
    }
    modal_component_ids = {
        component.representative_node_id
        for component in components
        if assignments_by_component_id[component.representative_node_id].provider
        is ExecutionProvider.MODAL
    }
    requires_volume_reload = any(
        asset.uploaded
        for component_id, assets in synced_assets_by_component_id.items()
        if component_id in modal_component_ids
        for asset in assets
    ) or any(
        bundle is not None and bundle.uploaded
        for environment_id, bundle in summary.custom_nodes_bundles_by_environment.items()
        if environment_id in modal_environment_ids
    )
    volume_reload_marker = uuid.uuid4().hex if requires_volume_reload else None
    logger.info(
        "Resolved request-wide Modal volume reload requirement: requires_volume_reload=%s volume_reload_marker=%s synced_assets=%d custom_nodes_uploaded=%s",
        requires_volume_reload,
        volume_reload_marker,
        len(summary.synced_assets),
        bool(
            summary.custom_nodes_bundle is not None
            and summary.custom_nodes_bundle.uploaded
        ),
    )
    summary.requires_volume_reload = requires_volume_reload
    summary.volume_reload_marker = volume_reload_marker
    summary.uploaded_volume_paths = [
        asset.remote_path for asset in summary.synced_assets if asset.uploaded
    ]
    mapped_proxy_component_ids: set[str] = set()
    sandwiched_local_node_id_set = set(summary.sandwiched_local_node_ids)
    for component in components:
        logger.info(
            "Rewriting remote component %s covering nodes %s.",
            component.representative_node_id,
            component.node_ids,
        )
        assignment = assignments_by_component_id[component.representative_node_id]
        component_custom_nodes_bundle = summary.custom_nodes_bundles_by_environment.get(
            assignment.environment_id
        )
        uploaded_volume_paths = _component_uploaded_volume_paths(
            component_prompt=synced_component_prompts[component.representative_node_id],
            synced_assets=synced_assets_by_component_id[
                component.representative_node_id
            ],
            custom_nodes_bundle=component_custom_nodes_bundle,
        )
        payload = _build_component_payload(
            component=component,
            component_prompt=synced_component_prompts[component.representative_node_id],
            signature_prompt=prompt,
            extra_data=extra_data,
            settings=resolved_settings,
            requires_volume_reload=(
                assignment.provider is ExecutionProvider.MODAL
                and bool(uploaded_volume_paths)
            ),
            volume_reload_marker=volume_reload_marker,
            custom_nodes_bundle=component_custom_nodes_bundle,
            uploaded_volume_paths=uploaded_volume_paths,
            terminate_container_on_error=resolved_settings.terminate_container_on_error,
            nodes_module=resolved_nodes_module,
            remote_session=remote_sessions_by_component_id.get(
                component.representative_node_id
            ),
        )
        _attach_resolved_llm_profiles(
            payload,
            execution_plan.resolved_llm_profiles,
            resolved_settings,
        )
        _stamp_execution_assignment(
            payload,
            assignment,
            summary.execution_worker_indices_by_representative.get(
                component.representative_node_id,
                0,
            ),
            _component_execution_signature(component, prompt),
            _execution_location_for_assignment(
                assignment,
                ssh_hosts_by_id,
                vast_leases_by_environment,
            ),
            _configured_provider_metadata(
                execution_plan=execution_plan,
                assignment=assignment,
                vast_leases_by_environment=vast_leases_by_environment,
            ),
        )
        implicitly_mapped_output_sources = _implicitly_mapped_boundary_output_sources(
            component=component,
            original_prompt=prompt,
            rewritten_prompt=rewritten_prompt,
            nodes_module=resolved_nodes_module,
        )
        _mark_payload_scheduler_list_outputs(
            payload,
            implicitly_mapped_output_sources,
        )
        proxy_node_ids = _rewrite_component_into_proxy(
            component=component,
            rewritten_prompt=rewritten_prompt,
            payload=payload,
            nodes_module=resolved_nodes_module,
        )
        for proxy_node_id in proxy_node_ids:
            summary.execution_assignments_by_representative[proxy_node_id] = assignment
            summary.execution_worker_indices_by_representative[proxy_node_id] = (
                summary.execution_worker_indices_by_representative.get(
                    component.representative_node_id,
                    0,
                )
            )
        split_proxy_payloads = payload.get("split_proxy_payloads")
        if isinstance(split_proxy_payloads, dict):
            static_proxy_node_id, mapped_proxy_node_id = proxy_node_ids
            summary.remote_component_ids.extend(proxy_node_ids)
            summary.component_node_ids_by_representative[static_proxy_node_id] = (
                _workflow_visible_remote_node_ids(component.static_node_ids)
            )
            summary.component_node_ids_by_representative[mapped_proxy_node_id] = (
                _workflow_visible_remote_node_ids(component.mapped_node_ids)
            )
            for node_id in _workflow_visible_remote_node_ids(
                component.static_node_ids
            ):
                summary.rewritten_node_id_map[node_id] = static_proxy_node_id
            for node_id in _workflow_visible_remote_node_ids(
                component.mapped_node_ids
            ):
                summary.rewritten_node_id_map[node_id] = mapped_proxy_node_id
            mapped_proxy_component_ids.add(mapped_proxy_node_id)
            continue
        if isinstance(split_proxy_payloads, list):
            summary.remote_component_ids.extend(proxy_node_ids)
            mapped_node_id_set = set(component.mapped_node_ids)
            for phase_payload in split_proxy_payloads:
                phase_proxy_node_id = str(phase_payload["component_id"])
                phase_component_node_ids = [
                    str(node_id) for node_id in phase_payload["component_node_ids"]
                ]
                summary.component_node_ids_by_representative[
                    phase_proxy_node_id
                ] = _workflow_visible_remote_node_ids(phase_component_node_ids)
                for node_id in _workflow_visible_remote_node_ids(
                    phase_component_node_ids
                ):
                    summary.rewritten_node_id_map[node_id] = phase_proxy_node_id
                if mapped_node_id_set and mapped_node_id_set.intersection(
                    phase_component_node_ids
                ):
                    mapped_proxy_component_ids.add(phase_proxy_node_id)
            continue

        summary.remote_component_ids.extend(proxy_node_ids)
        summary.component_node_ids_by_representative[proxy_node_ids[0]] = (
            _workflow_visible_remote_node_ids(component.node_ids)
        )
        for node_id in _workflow_visible_remote_node_ids(component.node_ids):
            summary.rewritten_node_id_map[node_id] = proxy_node_ids[0]
        if component.mapped_boundary_input_name or implicitly_mapped_output_sources:
            mapped_proxy_component_ids.add(proxy_node_ids[0])

    _configure_local_gap_keepalive_payloads(
        rewritten_prompt=rewritten_prompt,
        remote_component_ids=summary.remote_component_ids,
        sandwiched_local_node_ids=sandwiched_local_node_id_set,
    )
    summary.parallel_local_branch_node_ids = _parallelize_non_returning_local_branches(
        rewritten_prompt=rewritten_prompt,
        remote_component_ids=summary.remote_component_ids,
        nodes_module=resolved_nodes_module,
    )
    summary.artifact_finalizer_node_id = _attach_modal_artifact_finalizer(
        rewritten_prompt=rewritten_prompt,
        remote_component_ids=summary.remote_component_ids,
        nodes_module=resolved_nodes_module,
    )

    proxy_component_groups = {
        component_id: {component_id} for component_id in summary.remote_component_ids
    }
    _, dependency_edges, _ = _component_dependency_graph(
        rewritten_prompt, proxy_component_groups
    )
    execution_stages = _component_execution_stages(
        rewritten_prompt, proxy_component_groups
    )
    _configure_speculative_affinity_prewarm_payloads(
        rewritten_prompt=rewritten_prompt,
        execution_stages=execution_stages,
    )
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
    if execution_plan.configuration_set is not None:
        configured_capacity = sum(
            configuration.capacity_limit
            for configuration in execution_plan.configuration_set.capacity_configurations
        )
        summary.max_parallel_requests_upper_bound = min(
            summary.estimated_max_parallel_requests,
            configured_capacity,
        )
    elif resolved_settings.max_containers is not None:
        summary.max_parallel_requests_upper_bound = min(
            summary.estimated_max_parallel_requests,
            resolved_settings.max_containers,
        )
    else:
        summary.max_parallel_requests_upper_bound = (
            summary.estimated_max_parallel_requests
        )

    logger.info(
        "Estimated remote parallelism after proxy rewrite: known_max_parallel_requests=%d max_parallel_requests_upper_bound=%s mapped_components=%s execution_stages=%s",
        summary.estimated_max_parallel_requests,
        summary.max_parallel_requests_upper_bound,
        summary.mapped_component_ids,
        summary.component_execution_stages,
    )
    _log_modal_rewritten_prompt_diagnostics(
        prompt_id=(
            str(extra_data.get("prompt_id"))
            if isinstance(extra_data, Mapping) and extra_data.get("prompt_id")
            else None
        ),
        prompt=rewritten_prompt,
        summary=summary,
        reason="post_rewrite",
    )

    return rewritten_prompt, summary


async def rewrite_prompt_for_modal_async(
    prompt: dict[str, Any],
    workflow: dict[str, Any] | None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
    nodes_module: Any | None = None,
    extra_data: dict[str, Any] | None = None,
    status_callback: SetupStatusCallback | None = None,
    environment_status_callback: EnvironmentSetupStatusCallback | None = None,
    plan_callback: ExecutionPlanStatusCallback | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    occupied_environment_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], RewriteSummary]:
    """Prepare one Modal prompt without blocking ComfyUI's event loop."""
    return await asyncio.to_thread(
        rewrite_prompt_for_modal,
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=nodes_module,
        extra_data=extra_data,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
        plan_callback=plan_callback,
        cancellation_check=cancellation_check,
        occupied_environment_ids=occupied_environment_ids,
    )


def _execution_assignments_payload(
    summary: RewriteSummary,
    settings: ModalSyncSettings,
) -> dict[str, dict[str, Any]]:
    """Return planner assignments with member nodes and safe runtime labels."""
    ssh_hosts_by_id = {
        host.environment_id: host for host in _configured_ssh_hosts(settings)
    }
    vast_leases_by_environment: dict[str, Any] = {}
    vast_registry = _vast_registry(settings)
    if vast_registry is not None:
        try:
            vast_leases_by_environment = {
                lease.environment_id: lease for lease in vast_registry.load().leases
            }
        except ValueError as exc:
            logger.warning("Unable to load Vast lease labels for UI status: %s", exc)
    configurations_by_id = {
        str(configuration.get("configuration_id") or ""): configuration
        for configuration in summary.remote_configurations
    }
    return {
        component_id: {
            "provider": assignment.provider.value,
            "environment_id": assignment.environment_id,
            "configuration_id": assignment.configuration_id,
            "execution_location": _execution_location_for_assignment(
                assignment, ssh_hosts_by_id, vast_leases_by_environment
            )
            or summary.execution_locations_by_environment.get(
                assignment.environment_id
            ),
            "node_ids": list(
                summary.component_node_ids_by_representative.get(component_id, [])
            ),
            "predicted_cost_usd": assignment.predicted_cost_usd,
            "predicted_completion_seconds": assignment.predicted_completion_seconds,
            "worker_index": summary.execution_worker_indices_by_representative.get(
                component_id,
                0,
            ),
            "reasons": list(assignment.reasons),
            "hardware": _assignment_hardware_payload(
                component_id=component_id,
                assignment=assignment,
                configurations_by_id=configurations_by_id,
                ssh_hosts_by_id=ssh_hosts_by_id,
                vast_quotes={},
                vast_leases_by_environment=vast_leases_by_environment,
            ),
        }
        for component_id, assignment in sorted(
            summary.execution_assignments_by_representative.items()
        )
    }


def _prompt_uses_remote_execution_configurator(prompt: Mapping[str, Any]) -> bool:
    """Return whether the serialized prompt opts into workflow-scoped capacity."""
    return _remote_execution_configurator_node_id(prompt) is not None


def _remote_execution_configurator_node_id(
    prompt: Mapping[str, Any],
) -> str | None:
    """Return the serialized identity of the workflow's sole configurator node."""
    configurator_node_ids = [
        str(node_id)
        for node_id, prompt_node in prompt.items()
        if isinstance(prompt_node, Mapping)
        and str(prompt_node.get("class_type") or "")
        == REMOTE_EXECUTION_CONFIGURATOR_NODE_ID
    ]
    return configurator_node_ids[0] if len(configurator_node_ids) == 1 else None


def _selected_modal_gpus(
    summary: RewriteSummary,
    legacy_modal_gpu: str,
) -> list[str]:
    """Return distinct Modal GPUs selected by the completed execution plan."""
    modal_assignments = [
        assignment
        for assignment in summary.execution_assignments_by_representative.values()
        if assignment.provider is ExecutionProvider.MODAL
    ]
    if not modal_assignments:
        return []
    configured_gpus = {
        str(configuration.get("configuration_id") or ""): str(
            configuration.get("gpu_type") or ""
        )
        for configuration in summary.remote_configurations
        if configuration.get("provider") == ExecutionProvider.MODAL.value
        and configuration.get("configuration_id")
        and configuration.get("gpu_type")
    }
    selected_gpus = {
        configured_gpus.get(str(assignment.configuration_id or ""), "")
        for assignment in modal_assignments
    }
    selected_gpus.discard("")
    if not selected_gpus:
        selected_gpus.add(legacy_modal_gpu)
    return sorted(selected_gpus)




def _vast_registry(settings: ModalSyncSettings) -> VastLeaseRegistry | None:
    """Return persistent credential-free Vast lease state when available."""
    user_directory = discover_comfyui_user_directory(settings)
    if user_directory is None:
        return None
    return VastLeaseRegistry.for_user_directory(user_directory)




def setup_modal_queue_route(
    prompt_server: Any | None = None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
) -> None:
    """Register prompt interception routes once for the active PromptServer."""
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return

    try:
        resolved_server_module = _get_server_module()
    except ModuleNotFoundError:
        logger.debug(
            "ComfyUI server module is not available; skipping route registration."
        )
        return

    resolved_settings = settings or get_settings()
    prompt_server = prompt_server or getattr(
        resolved_server_module.PromptServer, "instance", None
    )
    if prompt_server is None:
        logger.debug(
            "PromptServer.instance is not available; skipping route registration."
        )
        return

    setup_r2_oauth_routes(prompt_server)
    resolved_sync_engine = sync_engine or ModalAssetSyncEngine.from_environment(
        resolved_settings
    )
    context = RouteContext(
        settings=resolved_settings,
        sync_engine=resolved_sync_engine,
        remote_host_registry=_ssh_host_registry(resolved_settings),
        vast_registry=_vast_registry(resolved_settings),
        analysis_route_path=_analysis_route_path(resolved_settings.route_path),
        progress_state_route_path=_progress_state_route_path(resolved_settings.route_path),
        container_status_route_path=_container_status_route_path(resolved_settings.route_path),
        container_stop_route_path=_container_status_route_path(
            resolved_settings.route_path
        ).replace("/container_status", "/container_stop"),
        delete_caches_route_path=_delete_modal_caches_route_path(resolved_settings.route_path),
        delete_volume_route_path=_delete_modal_volume_route_path(resolved_settings.route_path),
        cancel_preparation_route_path=_cancel_preparation_route_path(resolved_settings.route_path),
        remote_environments_route_path=_remote_environments_route_path(resolved_settings.route_path),
        remote_environment_probe_route_path=_remote_environment_probe_route_path(
            resolved_settings.route_path
        ),
        remote_environment_bootstrap_route_path=_remote_environment_bootstrap_route_path(
            resolved_settings.route_path
        ),
        remote_environment_status_route_path=_remote_environment_status_route_path(
            resolved_settings.route_path
        ),
        remote_environment_stop_route_path=_remote_environment_stop_route_path(
            resolved_settings.route_path
        ),
        rewrite_prompt=rewrite_prompt_for_modal_async,
        emit_status=_emit_modal_status,
        execution_assignments_payload=_execution_assignments_payload,
        prompt_uses_configurator=_prompt_uses_remote_execution_configurator,
        configurator_node_id=_remote_execution_configurator_node_id,
        selected_modal_gpus=_selected_modal_gpus,
    )
    _install_modal_interrupt_queue_bridge(prompt_server)
    register_r2_routes(prompt_server, context)
    register_remote_environment_routes(prompt_server, context)
    register_vast_routes(prompt_server, context)
    register_modal_container_routes(prompt_server, context)
    register_queue_routes(prompt_server, context)

    _ROUTE_REGISTERED = True
    logger.info(
        "Registered Modal queue and administration routes at %s.",
        resolved_settings.route_path,
    )
