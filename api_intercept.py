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
    from .prompt_interception import (
        _asset_preparation_completion_message,
        _environment_setup_status_callback,
        _prepare_environment_assets,
        _prepare_remote_environment_assets,
        _sync_environment_prompt_assets,
        rewrite_prompt_for_modal,
        rewrite_prompt_for_modal_async,
    )
    from .route_context import RouteContext
    from .routes_modal_containers import register_modal_container_routes
    from .routes_queue import register_queue_routes
    from .routes_r2 import register_r2_routes
    from .routes_remote_environments import register_remote_environment_routes
    from .routes_vast import register_vast_routes
else:  # pragma: no cover - flat import inside the Modal container.
    from prompt_interception import (
        _asset_preparation_completion_message,
        _environment_setup_status_callback,
        _prepare_environment_assets,
        _prepare_remote_environment_assets,
        _sync_environment_prompt_assets,
        rewrite_prompt_for_modal,
        rewrite_prompt_for_modal_async,
    )
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
