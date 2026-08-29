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

logger = logging.getLogger(__name__)
SetupStatusCallback = Callable[[str, int | None, int | None], None]
EnvironmentSetupStatusCallback = Callable[
    [str, str, int | None, int | None], None
]
ExecutionPlanStatusCallback = Callable[
    [dict[str, dict[str, Any]], list[dict[str, Any]]], None
]

_ROUTE_REGISTERED = False
_MODAL_INTERRUPT_QUEUE_BRIDGE_ATTR = "__comfy_modal_interrupt_queue_bridge_installed"
_REMOTE_PREPARATION_PROMPTS_ATTR = "__comfy_modal_remote_preparation_prompts"
_REMOTE_PREPARATION_CANCELLATIONS_ATTR = (
    "__comfy_modal_remote_preparation_cancellations"
)
_REMOTE_PREPARATION_LOCK_ATTR = "__comfy_modal_remote_preparation_lock"
_REMOTE_ASSET_PREPARATION_MAX_WORKERS = 8
_R2_STORAGE_USAGE_ROUTE = "/remote/storage/r2/usage"
_R2_KEYCHAIN_UNLOCK_ROUTE = "/remote/storage/r2/keychain/unlock"
_R2_STORAGE_USAGE_CACHE_SECONDS = 5 * 60
_R2_STORAGE_USAGE_CACHE_LOCK = threading.Lock()
_R2_STORAGE_USAGE_CACHE: dict[
    tuple[str, str, str], tuple[float, R2StorageUsage]
] = {}
_MODEL_VRAM_WEIGHT_MULTIPLIER = 1.20
_MODEL_VRAM_HEADROOM_BYTES = 4 * 1024**3
_MODEL_RAM_HEADROOM_BYTES = 4 * 1024**3
_ADDITIVE_MODEL_CLASS_TOKENS = (
    "adapter",
    "controlnet",
    "ipadapter",
    "lora",
)


def _modal_environment_state(settings: ModalSyncSettings) -> EnvironmentSchedulingState:
    """Return the scheduler-facing state of the selected Modal GPU target."""
    modal_gpu = settings.modal_gpu
    vram_bytes = int(_MODAL_GPU_VRAM_GB.get(modal_gpu, 0.0) * 1024**3)
    capabilities = EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=1,
        total_ram_bytes=max(vram_bytes, 1),
        available_ram_bytes=None,
        available_disk_bytes=None,
        docker_version="modal-managed",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(
            GpuCapability(
                uuid=f"modal-{modal_gpu.lower()}",
                name=modal_gpu,
                total_vram_bytes=vram_bytes,
            ),
        ),
    )
    return EnvironmentSchedulingState(
        environment_id=f"modal:{modal_gpu}",
        provider=ExecutionProvider.MODAL,
        enabled=True,
        health=EnvironmentHealth.READY,
        cost_usd_per_second=_MODAL_GPU_COST_USD_PER_SECOND.get(modal_gpu),
        capabilities=capabilities,
        maximum_workers=settings.max_containers or 1,
    )


def _modal_configuration_environment_state(
    configuration: ModalRemoteConfiguration,
    settings: ModalSyncSettings,
) -> EnvironmentSchedulingState:
    """Return one scheduler state for a workflow-declared Modal capacity pool."""
    selected_settings = settings_for_modal_gpu(settings, configuration.gpu_type)
    base_state = _modal_environment_state(selected_settings)
    return replace(
        base_state,
        environment_id=(
            f"modal:{configuration.configuration_id}:{configuration.gpu_type}"
        ),
        configuration_id=configuration.configuration_id,
        display_name=configuration.display_name,
        maximum_workers=configuration.instance_count,
    )


def _ssh_host_registry(settings: ModalSyncSettings) -> RemoteHostRegistry | None:
    """Return the persistent SSH host registry when a user directory exists."""
    user_directory = discover_comfyui_user_directory(settings)
    if user_directory is None:
        return None
    return RemoteHostRegistry.for_user_directory(user_directory)


def _configured_ssh_hosts(settings: ModalSyncSettings) -> tuple[SshHostConfig, ...]:
    """Return configured SSH hosts without probing during queue submission."""
    registry = _ssh_host_registry(settings)
    if registry is None:
        return ()
    return registry.load().hosts


def _schedulable_ssh_hosts(settings: ModalSyncSettings) -> tuple[SshHostConfig, ...]:
    """Probe every enabled host immediately before cost-aware scheduling."""
    hosts = _configured_ssh_hosts(settings)
    registry = _ssh_host_registry(settings)
    if registry is None:
        return hosts
    probe_hosts = [host for host in hosts if host.enabled and not host.draining]
    if not probe_hosts:
        return hosts

    with ThreadPoolExecutor(max_workers=min(8, len(probe_hosts))) as executor:
        refreshed = {
            host.environment_id: future.result()
            for host, future in (
                (host, executor.submit(_refresh_ssh_host, host, registry))
                for host in probe_hosts
            )
        }
    return tuple(refreshed.get(host.environment_id, host) for host in hosts)


def _refresh_ssh_host(
    host: SshHostConfig,
    registry: RemoteHostRegistry,
) -> SshHostConfig:
    """Probe and persist one host without aborting other candidates."""
    try:
        capabilities = SshDockerController(host).probe_capabilities()
        return registry.update_probe_result(
            host.environment_id,
            capabilities=capabilities,
            health=EnvironmentHealth.READY,
            last_error=None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Remote environment probe failed environment=%s: %s",
            host.environment_id,
            exc,
        )
        try:
            return registry.update_probe_result(
                host.environment_id,
                capabilities=host.capabilities,
                health=EnvironmentHealth.UNAVAILABLE,
                last_error=str(exc),
            )
        except (KeyError, OSError, ValueError):
            return replace(
                host,
                health=EnvironmentHealth.UNAVAILABLE,
                last_error=str(exc),
            )


def _prompt_llm_model_references(prompt: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every fixed Modal LLM model reference in a prompt."""
    references: set[str] = set()
    for prompt_node in prompt.values():
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_reference = inputs.get("model_profile")
        if isinstance(model_reference, str) and model_reference.strip():
            references.add(model_reference.strip())
    return tuple(sorted(references))


def _resolve_prompt_llm_profiles(
    prompt: Mapping[str, Any],
    settings: ModalSyncSettings,
) -> dict[str, Any]:
    """Resolve LLM metadata before environment admission and cost ranking."""
    from .llm_profiles import get_llm_profile
    from .llm_resolver import resolve_model_profile

    storage_root = Path(
        getattr(settings, "local_storage_root", "/tmp/comfyui-modal-sync-storage")
    )
    profiles: dict[str, Any] = {}
    for model_reference in _prompt_llm_model_references(prompt):
        try:
            profile = get_llm_profile(model_reference, storage_root=storage_root)
        except ValueError as profile_error:
            if model_reference.startswith("hf-"):
                raise ModalPromptValidationError(str(profile_error)) from profile_error
            try:
                profile = resolve_model_profile(model_reference, storage_root).profile
            except ValueError as resolution_error:
                raise ModalPromptValidationError(
                    str(resolution_error)
                ) from resolution_error
        profiles[model_reference] = profile
        logger.info(
            "Resolved planner LLM profile model=%s profile=%s weights_gib=%.2f "
            "estimated_vram_gib=%.2f.",
            model_reference,
            profile.profile_id,
            profile.artifact_bytes / 1024**3,
            profile.estimated_vram_gb,
        )
    return profiles


def _component_profile_memory_estimate(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    preferences: WorkflowExecutionPreferences,
    resolved_profiles: Mapping[str, Any],
) -> ComponentMemoryEstimate:
    """Estimate a component's RAM and VRAM floors from resolved LLM profiles."""
    minimum_vram_bytes = preferences.minimum_vram_bytes
    profiles: dict[str, Any] = {}
    for node_id in component.node_ids:
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_profile = inputs.get("model_profile")
        if not isinstance(model_profile, str) or not model_profile.strip():
            continue
        profile = resolved_profiles.get(model_profile.strip())
        if profile is None:
            continue
        profiles[profile.profile_id] = profile
        minimum_vram_bytes = max(
            minimum_vram_bytes,
            int(max(0.0, profile.estimated_vram_gb) * 1024**3),
        )
    artifact_sizes = [int(profile.artifact_bytes) for profile in profiles.values()]
    return ComponentMemoryEstimate(
        minimum_vram_bytes=minimum_vram_bytes,
        minimum_ram_bytes=(sum(artifact_sizes) + _MODEL_RAM_HEADROOM_BYTES)
        if artifact_sizes
        else 0,
        model_asset_count=len(artifact_sizes),
        largest_model_bytes=max(artifact_sizes, default=0),
    )


def _component_required_provider(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    resolved_profiles: Mapping[str, Any],
) -> ExecutionProvider | None:
    """Require SSH when a component uses an SSH-only resident backend."""
    for node_id in component.node_ids:
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_reference = inputs.get("model_profile")
        if not isinstance(model_reference, str):
            continue
        profile = resolved_profiles.get(model_reference.strip())
        if (
            profile is not None
            and getattr(profile, "backend", "") == "llama_cpp_server"
        ):
            return ExecutionProvider.SSH_DOCKER
    return None


def _iter_prompt_string_values(value: object) -> Iterator[str]:
    """Yield nested prompt string values while ignoring graph links and scalars."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_prompt_string_values(child)
        return
    if isinstance(value, (list, tuple)) and not _is_link(value):
        for child in value:
            yield from _iter_prompt_string_values(child)


def _is_additive_model_node(class_type: str) -> bool:
    """Return whether a loader contributes weights alongside a primary model."""
    normalized = class_type.casefold()
    return any(token in normalized for token in _ADDITIVE_MODEL_CLASS_TOKENS)


def _component_model_asset_sizes(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    settings: ModalSyncSettings,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return unique primary and additive model sizes referenced by a component."""
    asset_roles: dict[Path, bool] = {}
    for node_id in component.node_ids:
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        additive = _is_additive_model_node(str(prompt_node.get("class_type") or ""))
        for value in _iter_prompt_string_values(inputs):
            resolved_path = resolve_model_path(
                value,
                comfyui_root=getattr(settings, "comfyui_root", None),
                extensions=MODEL_FILE_EXTENSIONS,
            )
            if resolved_path is None:
                continue
            asset_roles[resolved_path] = (
                asset_roles.get(resolved_path, True) and additive
            )

    primary_sizes: list[int] = []
    additive_sizes: list[int] = []
    for asset_path, additive in asset_roles.items():
        try:
            size_bytes = asset_path.stat().st_size
        except OSError as exc:
            logger.warning(
                "Unable to inspect model size for placement path=%s: %s",
                asset_path,
                exc,
            )
            continue
        if size_bytes <= 0:
            continue
        target = additive_sizes if additive else primary_sizes
        target.append(size_bytes)
    return tuple(primary_sizes), tuple(additive_sizes)


def _component_memory_estimate(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    preferences: WorkflowExecutionPreferences,
    settings: ModalSyncSettings,
    resolved_llm_profiles: Mapping[str, Any] | None = None,
) -> ComponentMemoryEstimate:
    """Infer conservative RAM and VRAM floors from resident model weight sizes."""
    profile_estimate = _component_profile_memory_estimate(
        component,
        prompt,
        preferences,
        resolved_llm_profiles
        if resolved_llm_profiles is not None
        else _resolve_prompt_llm_profiles(prompt, settings),
    )
    primary_sizes, additive_sizes = _component_model_asset_sizes(
        component,
        prompt,
        settings,
    )
    all_sizes = primary_sizes + additive_sizes
    if not all_sizes:
        return profile_estimate

    primary_peak_bytes = max(primary_sizes or all_sizes)
    additive_bytes = sum(additive_sizes) if primary_sizes else 0
    resident_weight_bytes = primary_peak_bytes + additive_bytes
    model_vram_bytes = (
        math.ceil(resident_weight_bytes * _MODEL_VRAM_WEIGHT_MULTIPLIER)
        + _MODEL_VRAM_HEADROOM_BYTES
    )
    return ComponentMemoryEstimate(
        minimum_vram_bytes=max(
            profile_estimate.minimum_vram_bytes,
            model_vram_bytes,
        ),
        minimum_ram_bytes=max(
            profile_estimate.minimum_ram_bytes,
            resident_weight_bytes + _MODEL_RAM_HEADROOM_BYTES,
        ),
        model_asset_count=len(all_sizes) + profile_estimate.model_asset_count,
        largest_model_bytes=max(
            max(all_sizes),
            profile_estimate.largest_model_bytes,
        ),
    )


def _component_execution_signature(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
) -> str:
    """Return a stable workload signature for runtime-history estimates."""
    cost_shaping_input_names = frozenset(
        {
            "batch_size",
            "duration",
            "frames",
            "height",
            "max_new_tokens",
            "model_profile",
            "num_frames",
            "steps",
            "width",
        }
    )
    nodes: list[dict[str, Any]] = []
    for node_id in sorted(component.node_ids):
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        input_mapping = inputs if isinstance(inputs, Mapping) else {}
        nodes.append(
            {
                "class_type": str(prompt_node.get("class_type") or ""),
                "inputs": {
                    name: input_mapping[name]
                    for name in sorted(cost_shaping_input_names & set(input_mapping))
                    if not _is_link(input_mapping[name])
                },
            }
        )
    serialized = json.dumps(nodes, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def _execution_history(settings: ModalSyncSettings) -> ExecutionHistory | None:
    """Return local runtime history when the ComfyUI user directory is known."""
    user_directory = discover_comfyui_user_directory(settings)
    if user_directory is None:
        return None
    return ExecutionHistory.for_user_directory(user_directory)


def _maximum_capacity_state(
    environment: EnvironmentSchedulingState,
) -> EnvironmentSchedulingState:
    """Return an optimistic state used only to identify reclaimable SSH capacity."""
    capabilities = environment.capabilities
    if environment.provider is not ExecutionProvider.SSH_DOCKER or capabilities is None:
        return environment
    return replace(
        environment,
        capabilities=replace(
            capabilities,
            available_ram_bytes=capabilities.total_ram_bytes,
            gpus=tuple(
                replace(gpu, free_vram_bytes=gpu.total_vram_bytes)
                for gpu in capabilities.gpus
            ),
        ),
    )


def _reprobe_reclaimed_ssh_host(
    host: SshHostConfig,
    settings: ModalSyncSettings,
) -> SshHostConfig:
    """Probe a reclaimed host and persist its new free-memory measurements."""
    registry = _ssh_host_registry(settings)
    if registry is not None:
        return _refresh_ssh_host(host, registry)
    try:
        capabilities = SshDockerController(host).probe_capabilities()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Remote environment re-probe failed after worker reclaim "
            "environment=%s: %s",
            host.environment_id,
            exc,
        )
        return replace(
            host,
            health=EnvironmentHealth.UNAVAILABLE,
            last_error=str(exc),
        )
    return replace(
        host,
        capabilities=capabilities,
        health=EnvironmentHealth.READY,
        last_error=None,
    )


def _probe_workflow_ssh_configuration(
    configuration: SshRemoteConfiguration,
) -> SshHostConfig:
    """Probe one workflow-declared SSH host without mutating the global registry."""
    host = configuration.host
    try:
        capabilities = SshDockerController(host).probe_capabilities()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Workflow SSH environment probe failed configuration=%s environment=%s: %s",
            configuration.display_name,
            host.environment_id,
            exc,
        )
        return replace(
            host,
            capabilities=None,
            health=EnvironmentHealth.UNAVAILABLE,
            last_error=str(exc),
        )
    return replace(
        host,
        capabilities=capabilities,
        health=EnvironmentHealth.READY,
        last_error=None,
    )


def _optional_scheduler_choice(
    scheduler: CostAwareEnvironmentScheduler,
    environments: list[EnvironmentSchedulingState],
    requirements: ComponentResourceRequirements,
) -> tuple[
    ExecutionAssignment | None,
    NoCompatibleExecutionEnvironmentError | None,
]:
    """Return a scheduler choice and preserve a compatibility failure for fallback."""
    try:
        return scheduler.choose(environments, requirements), None
    except NoCompatibleExecutionEnvironmentError as exc:
        return None, exc


def _require_scheduler_choice(
    assignment: ExecutionAssignment | None,
    error: NoCompatibleExecutionEnvironmentError | None,
) -> ExecutionAssignment:
    """Return an existing scheduler choice or raise its compatibility detail."""
    if assignment is not None:
        return assignment
    if error is not None:
        raise error
    raise RuntimeError("Scheduler failed without an incompatibility reason.")


def _remove_idle_ssh_workers_for_reclaim(host: SshHostConfig) -> tuple[str, ...]:
    """Recycle idle managed workers without disrupting fallback placement."""
    try:
        return SshDockerController(host).remove_idle_managed_workers()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Could not reclaim managed SSH worker capacity environment=%s: %s",
            host.environment_id,
            exc,
        )
        return ()


def _reclaim_improves_assignment(
    optimistic: ExecutionAssignment,
    actual: ExecutionAssignment | None,
    requirements: ComponentResourceRequirements,
) -> bool:
    """Return whether reclaim unlocks required or materially better placement."""
    if actual is None:
        return True
    if optimistic.environment_id == actual.environment_id:
        return False
    preferred = requirements.preferred_environment_ids
    if optimistic.environment_id in preferred:
        if actual.environment_id not in preferred:
            return True
        return preferred.index(optimistic.environment_id) < preferred.index(
            actual.environment_id
        )
    if optimistic.predicted_cost_usd is None:
        return False
    if actual.predicted_cost_usd is None:
        return True
    return optimistic.predicted_cost_usd < actual.predicted_cost_usd


def _choose_with_idle_ssh_worker_reclaim(
    *,
    scheduler: CostAwareEnvironmentScheduler,
    environments: list[EnvironmentSchedulingState],
    requirements: ComponentResourceRequirements,
    ssh_hosts_by_id: dict[str, SshHostConfig],
    settings: ModalSyncSettings,
) -> ExecutionAssignment:
    """Choose an environment, recycling a cheaper idle SSH worker when necessary."""
    actual_assignment, actual_error = _optional_scheduler_choice(
        scheduler,
        environments,
        requirements,
    )
    optimistic_assignment, _ = _optional_scheduler_choice(
        scheduler,
        [_maximum_capacity_state(environment) for environment in environments],
        requirements,
    )
    if optimistic_assignment is None:
        return _require_scheduler_choice(actual_assignment, actual_error)

    if (
        optimistic_assignment.provider is not ExecutionProvider.SSH_DOCKER
        or not _reclaim_improves_assignment(
            optimistic_assignment,
            actual_assignment,
            requirements,
        )
    ):
        return _require_scheduler_choice(actual_assignment, actual_error)

    host = ssh_hosts_by_id.get(optimistic_assignment.environment_id)
    if host is None:
        return _require_scheduler_choice(actual_assignment, actual_error)

    removed_workers = _remove_idle_ssh_workers_for_reclaim(host)
    if not removed_workers:
        return _require_scheduler_choice(actual_assignment, actual_error)

    logger.info(
        "Recycled idle managed SSH worker(s) environment=%s containers=%s so the "
        "planner can re-evaluate full host capacity.",
        host.environment_id,
        removed_workers,
    )
    refreshed_host = _reprobe_reclaimed_ssh_host(host, settings)
    ssh_hosts_by_id[host.environment_id] = refreshed_host
    refreshed_environments = [
        (
            refreshed_host.scheduling_state()
            if environment.environment_id == host.environment_id
            else environment
        )
        for environment in environments
    ]
    return scheduler.choose(refreshed_environments, requirements)


def _plan_component_execution_assignments(
    *,
    components: list[RemoteComponentPlan],
    prompt: Mapping[str, Any],
    workflow: Mapping[str, Any] | None,
    settings: ModalSyncSettings,
    status_callback: SetupStatusCallback | None = None,
    resolved_llm_profiles: Mapping[str, Any] | None = None,
) -> dict[str, ExecutionAssignment]:
    """Assign components across Modal, SSH Docker, and workflow-declared Vast pools."""
    preferences = WorkflowExecutionPreferences.from_workflow(workflow)
    active_llm_profiles = (
        dict(resolved_llm_profiles)
        if resolved_llm_profiles is not None
        else _resolve_prompt_llm_profiles(prompt, settings)
    )
    if preferences.policy is ExecutionPolicy.MODAL:
        incompatible_component_ids = [
            component.representative_node_id
            for component in components
            if _component_required_provider(
                component,
                prompt,
                active_llm_profiles,
            )
            is ExecutionProvider.SSH_DOCKER
        ]
        if incompatible_component_ids:
            raise ModalPromptValidationError(
                "Modal-only execution cannot run SSH-only component(s) "
                f"{incompatible_component_ids}. Select Automatic or Self-hosted "
                "execution for the workflow."
            )
        modal_state = _modal_environment_state(settings)
        return {
            component.representative_node_id: ExecutionAssignment(
                environment_id=modal_state.environment_id,
                provider=ExecutionProvider.MODAL,
                predicted_cost_usd=None,
                predicted_completion_seconds=0.0,
                reasons=("workflow policy requires Modal",),
            )
            for component in components
        }

    try:
        vast_profiles = extract_vast_profiles(prompt)
    except ValueError as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    vast_service: VastService | None = None
    if preferences.policy in {ExecutionPolicy.VAST, ExecutionPolicy.AUTOMATIC}:
        if preferences.policy is ExecutionPolicy.VAST and not vast_profiles:
            raise ModalPromptValidationError(
                "Vast.ai-only execution requires at least one disconnected "
                "Vast.ai Lease Configuration node in the workflow."
            )
        if vast_profiles:
            try:
                vast_service = VastService.from_environment(
                    settings,
                    repo_root=Path(__file__).resolve().parent,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if preferences.policy is ExecutionPolicy.VAST:
                    raise ModalPromptValidationError(str(exc)) from exc
                logger.warning("Skipping Vast.ai automatic placement: %s", exc)

    ssh_hosts = (
        _schedulable_ssh_hosts(settings)
        if preferences.policy in {ExecutionPolicy.SELF_HOSTED, ExecutionPolicy.AUTOMATIC}
        else ()
    )
    ssh_hosts_by_id = {host.environment_id: host for host in ssh_hosts}
    modal_state = (
        _modal_environment_state(settings)
        if preferences.policy is ExecutionPolicy.AUTOMATIC
        else None
    )
    if not ssh_hosts and modal_state is None and vast_service is None:
        raise ModalPromptValidationError(
            "This workflow requests remote execution, but no compatible provider is configured."
        )

    scheduler = CostAwareEnvironmentScheduler()
    history = _execution_history(settings)
    component_signatures: dict[str, str] = {}
    component_memory_estimates: dict[str, ComponentMemoryEstimate] = {}
    component_required_providers: dict[str, ExecutionProvider | None] = {}
    for component in components:
        component_id = component.representative_node_id
        component_signatures[component_id] = _component_execution_signature(
            component,
            prompt,
        )
        memory_estimate = _component_memory_estimate(
            component,
            prompt,
            preferences,
            settings,
            active_llm_profiles,
        )
        component_memory_estimates[component_id] = memory_estimate
        component_required_providers[component_id] = _component_required_provider(
            component,
            prompt,
            active_llm_profiles,
        )
        if memory_estimate.model_asset_count:
            logger.info(
                "Estimated remote component=%s memory floor vram_gib=%.2f ram_gib=%.2f "
                "from model_assets=%d largest_model_gib=%.2f.",
                component_id,
                memory_estimate.minimum_vram_bytes / 1024**3,
                memory_estimate.minimum_ram_bytes / 1024**3,
                memory_estimate.model_asset_count,
                memory_estimate.largest_model_bytes / 1024**3,
            )
    if vast_service is not None:
        try:
            vast_service.prefetch_offers_sync(
                vast_profiles,
                tuple(
                    VastSearchRequirements(
                        minimum_vram_bytes=memory_estimate.minimum_vram_bytes,
                        minimum_ram_bytes=memory_estimate.minimum_ram_bytes,
                    )
                    for memory_estimate in component_memory_estimates.values()
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if preferences.policy is ExecutionPolicy.VAST:
                raise ModalPromptValidationError(str(exc)) from exc
            logger.warning(
                "Unable to prefetch Vast.ai marketplace offers: %s",
                exc,
            )
    assignments: dict[str, ExecutionAssignment] = {}
    for component in components:
        component_id = component.representative_node_id
        environments = [
            host.scheduling_state() for host in ssh_hosts_by_id.values()
        ]
        if modal_state is not None:
            environments.append(modal_state)
        component_signature = component_signatures[component_id]
        memory_estimate = component_memory_estimates[component_id]
        vast_quote: VastProfileQuote | None = None
        if vast_service is not None:
            try:
                vast_quote = vast_service.quote_best_profile_sync(
                    vast_profiles,
                    minimum_vram_bytes=memory_estimate.minimum_vram_bytes,
                    minimum_ram_bytes=memory_estimate.minimum_ram_bytes,
                    predicted_execution_seconds=60.0,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if preferences.policy is ExecutionPolicy.VAST:
                    raise ModalPromptValidationError(str(exc)) from exc
                logger.warning(
                    "No Vast.ai automatic candidate for component=%s: %s",
                    component.representative_node_id,
                    exc,
                )
            else:
                environments.append(vast_service.scheduling_state(vast_quote))
        environment_ids = [
            environment.environment_id for environment in environments
        ]
        historical_estimates = (
            history.estimates(component_signature, environment_ids)
            if history is not None
            else {}
        )
        component_required_provider = component_required_providers[component_id]
        if preferences.policy is ExecutionPolicy.VAST:
            if component_required_provider is ExecutionProvider.SSH_DOCKER:
                raise ModalPromptValidationError(
                    "Vast.ai-only execution cannot run an SSH-only component. "
                    "Select Automatic or Self-hosted execution for this workflow."
                )
            component_required_provider = ExecutionProvider.VAST
        requirements = ComponentResourceRequirements(
            minimum_vram_bytes=memory_estimate.minimum_vram_bytes,
            minimum_ram_bytes=memory_estimate.minimum_ram_bytes,
            gpu_required=True,
            architecture="x86_64",
            estimated_execution_seconds=60.0,
            estimated_execution_seconds_by_environment={
                environment_id: estimate.execution_seconds
                for environment_id, estimate in historical_estimates.items()
            },
            preferred_environment_ids=preferences.preferred_environment_ids,
            required_provider=component_required_provider,
        )
        try:
            assignment = _choose_with_idle_ssh_worker_reclaim(
                scheduler=scheduler,
                environments=environments,
                requirements=requirements,
                ssh_hosts_by_id=ssh_hosts_by_id,
                settings=settings,
            )
        except NoCompatibleExecutionEnvironmentError as exc:
            raise ModalPromptValidationError(str(exc)) from exc
        if assignment.provider is ExecutionProvider.VAST:
            if vast_service is None or vast_quote is None:
                raise ModalPromptValidationError(
                    "Vast.ai was selected without a current marketplace quote."
                )
            try:
                if status_callback is None:
                    lease = vast_service.acquire_sync(vast_quote)
                else:
                    lease = vast_service.acquire_sync(
                        vast_quote,
                        status_callback=lambda message: status_callback(
                            message,
                            None,
                            None,
                        ),
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModalPromptValidationError(
                    f"Unable to acquire Vast.ai capacity: {exc}"
                ) from exc
            assignment = replace(
                assignment,
                environment_id=lease.environment_id,
                predicted_cost_usd=vast_quote.predicted_incremental_cost_usd,
                reasons=assignment.reasons
                + (
                    f"Vast profile {vast_quote.profile.profile_name}",
                    f"idle retention {lease.idle_retention_seconds / 3600:.1f}h",
                ),
            )
        assignments[component.representative_node_id] = assignment
        logger.info(
            "Assigned remote component=%s provider=%s environment=%s predicted_cost_usd=%s reasons=%s.",
            component.representative_node_id,
            assignment.provider.value,
            assignment.environment_id,
            assignment.predicted_cost_usd,
            assignment.reasons,
        )
    return assignments


def _plan_component_execution(
    *,
    components: list[RemoteComponentPlan],
    prompt: Mapping[str, Any],
    workflow: Mapping[str, Any] | None,
    settings: ModalSyncSettings,
    status_callback: SetupStatusCallback | None = None,
    environment_status_callback: EnvironmentSetupStatusCallback | None = None,
    plan_callback: ExecutionPlanStatusCallback | None = None,
    occupied_environment_ids: frozenset[str] = frozenset(),
) -> ComponentExecutionPlan:
    """Plan through connected configurations or synthesize the legacy behavior."""
    try:
        configuration_set = compile_remote_configuration_set(prompt)
    except (TypeError, ValueError) as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    resolved_llm_profiles = _resolve_prompt_llm_profiles(prompt, settings)
    if configuration_set is None:
        return ComponentExecutionPlan(
            assignments=_plan_component_execution_assignments(
                components=components,
                prompt=prompt,
                workflow=workflow,
                settings=settings,
                status_callback=status_callback,
                resolved_llm_profiles=resolved_llm_profiles,
            ),
            resolved_llm_profiles=resolved_llm_profiles,
        )
    return _plan_configured_component_execution(
        components=components,
        prompt=prompt,
        workflow=workflow,
        settings=settings,
        configuration_set=configuration_set,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
        plan_callback=plan_callback,
        occupied_environment_ids=occupied_environment_ids,
        resolved_llm_profiles=resolved_llm_profiles,
    )


def _plan_configured_component_execution(
    *,
    components: list[RemoteComponentPlan],
    prompt: Mapping[str, Any],
    workflow: Mapping[str, Any] | None,
    settings: ModalSyncSettings,
    configuration_set: RemoteConfigurationSet,
    status_callback: SetupStatusCallback | None = None,
    environment_status_callback: EnvironmentSetupStatusCallback | None = None,
    plan_callback: ExecutionPlanStatusCallback | None = None,
    occupied_environment_ids: frozenset[str] = frozenset(),
    resolved_llm_profiles: Mapping[str, Any] | None = None,
) -> ComponentExecutionPlan:
    """Resolve and prepare a capacity-aware plan from connected configurations."""
    configurations_by_id = {
        configuration.configuration_id: configuration
        for configuration in configuration_set.capacity_configurations
    }
    ssh_hosts_by_id = _probe_configured_ssh_hosts(configuration_set)
    vast_service, vast_unavailable_reason = _configured_vast_service(
        configuration_set,
        settings,
    )
    requirements_by_component = _configured_component_requirements(
        components=components,
        prompt=prompt,
        workflow=workflow,
        settings=settings,
        resolved_llm_profiles=resolved_llm_profiles,
    )
    _prefetch_configured_vast_offers(
        configuration_set=configuration_set,
        requirements_by_component=requirements_by_component,
        vast_service=vast_service,
    )
    environments_by_component, vast_quotes = _configured_candidate_environments(
        configuration_set=configuration_set,
        requirements_by_component=requirements_by_component,
        settings=settings,
        ssh_hosts_by_id=ssh_hosts_by_id,
        vast_service=vast_service,
        vast_unavailable_reason=vast_unavailable_reason,
        occupied_environment_ids=occupied_environment_ids,
    )
    if _reclaim_idle_configured_ssh_capacity(
        configuration_set=configuration_set,
        requirements_by_component=requirements_by_component,
        ssh_hosts_by_id=ssh_hosts_by_id,
        occupied_environment_ids=occupied_environment_ids,
    ):
        environments_by_component, vast_quotes = _configured_candidate_environments(
            configuration_set=configuration_set,
            requirements_by_component=requirements_by_component,
            settings=settings,
            ssh_hosts_by_id=ssh_hosts_by_id,
            vast_service=vast_service,
            vast_unavailable_reason=vast_unavailable_reason,
            occupied_environment_ids=occupied_environment_ids,
        )
    _apply_historical_execution_estimates(
        components=components,
        environments_by_component=environments_by_component,
        requirements_by_component=requirements_by_component,
        prompt=prompt,
        settings=settings,
    )
    execution_stages = _component_execution_stages(
        dict(prompt),
        {
            component.representative_node_id: set(component.node_ids)
            for component in components
        },
    )
    try:
        assignments = CostAwareEnvironmentScheduler().plan(
            execution_stages=execution_stages,
            environments_by_component=environments_by_component,
            requirements_by_component=requirements_by_component,
        )
    except NoCompatibleExecutionEnvironmentError as exc:
        raise ModalPromptValidationError(str(exc)) from exc

    for component_id, assignment in tuple(assignments.items()):
        if assignment.environment_id not in occupied_environment_ids:
            continue
        assignments[component_id] = replace(
            assignment,
            reasons=assignment.reasons
            + ("capacity available after the earlier workflow completes",),
        )

    safe_configurations = _safe_remote_configuration_payload(configuration_set)
    if plan_callback is not None:
        plan_callback(
            _planned_execution_assignments_payload(
                assignments,
                components,
                configurations_by_id=configurations_by_id,
                ssh_hosts_by_id=ssh_hosts_by_id,
                vast_quotes=vast_quotes,
            ),
            safe_configurations,
        )

    vast_leases = _prepare_selected_vast_capacity(
        assignments=assignments,
        configuration_set=configuration_set,
        requirements_by_component=requirements_by_component,
        vast_quotes=vast_quotes,
        vast_service=vast_service,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
    )
    if plan_callback is not None and vast_leases:
        plan_callback(
            _planned_execution_assignments_payload(
                assignments,
                components,
                configurations_by_id=configurations_by_id,
                ssh_hosts_by_id=ssh_hosts_by_id,
                vast_quotes=vast_quotes,
                vast_leases_by_environment=vast_leases,
            ),
            safe_configurations,
        )
    return ComponentExecutionPlan(
        assignments=assignments,
        configuration_set=configuration_set,
        configurations_by_id=configurations_by_id,
        safe_configurations=safe_configurations,
        ssh_hosts_by_id=ssh_hosts_by_id,
        vast_service=vast_service,
        vast_leases_by_environment=vast_leases,
        resolved_llm_profiles=dict(resolved_llm_profiles or {}),
    )


def _safe_remote_configuration_payload(
    configuration_set: RemoteConfigurationSet,
) -> list[dict[str, Any]]:
    """Return safe configuration metadata enriched with best-effort storage usage."""
    payload = configuration_set.to_safe_list()
    payload_by_id = {
        str(configuration.get("configuration_id") or ""): configuration
        for configuration in payload
    }
    for storage in configuration_set.storage_configurations:
        if not isinstance(storage, R2StorageBackingConfiguration):
            continue
        safe_storage = payload_by_id.get(storage.configuration_id)
        if safe_storage is None:
            continue
        usage, error_code = _cached_r2_storage_usage(storage)
        if error_code is not None:
            safe_storage["credential_error_code"] = error_code
        if usage is None:
            continue
        safe_storage["storage_usage_bytes"] = usage.size_bytes
        safe_storage["storage_object_count"] = usage.object_count
    return payload


def _cached_r2_storage_usage(
    storage: R2StorageBackingConfiguration,
) -> tuple[R2StorageUsage | None, str | None]:
    """Return cached bucket usage and an optional credential recovery code."""
    cache_key = (storage.account_id, storage.bucket, storage.jurisdiction)
    now = time.monotonic()
    with _R2_STORAGE_USAGE_CACHE_LOCK:
        cached = _R2_STORAGE_USAGE_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _R2_STORAGE_USAGE_CACHE_SECONDS:
            return cached[1], None
    try:
        usage = _refresh_r2_storage_usage(storage)
    except R2CredentialError as exc:
        logger.warning(
            "Unable to read R2 storage usage for configuration=%s bucket=%s: %s",
            storage.configuration_id,
            storage.bucket,
            exc,
        )
        return None, exc.code
    except (R2CacheError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Unable to read R2 storage usage for configuration=%s bucket=%s: %s",
            storage.configuration_id,
            storage.bucket,
            exc,
        )
        return None, None
    return usage, None


def _refresh_r2_storage_usage(
    storage: R2StorageBackingConfiguration,
) -> R2StorageUsage:
    """Query current R2 bucket usage and replace the short-lived cache entry."""
    configuration = R2CredentialStore().cache_configuration(storage)
    usage = R2CacheClient(configuration).storage_usage()
    cache_key = (storage.account_id, storage.bucket, storage.jurisdiction)
    with _R2_STORAGE_USAGE_CACHE_LOCK:
        _R2_STORAGE_USAGE_CACHE[cache_key] = (time.monotonic(), usage)
    return usage


def _r2_storage_from_usage_payload(
    payload: Mapping[str, Any],
) -> R2StorageBackingConfiguration:
    """Build a validated R2 reference from a same-origin usage refresh request."""
    configuration_id = str(
        payload.get("configuration_id") or "r2-storage-refresh"
    ).strip()
    return R2StorageBackingConfiguration(
        configuration_id=configuration_id,
        display_name=str(payload.get("display_name") or "R2 storage").strip(),
        account_id=str(payload.get("account_id") or "").strip(),
        bucket=str(payload.get("bucket") or "").strip(),
        credential_id=str(payload.get("credential_id") or "").strip(),
        jurisdiction=str(payload.get("jurisdiction") or "default")
        .strip()
        .casefold(),
        key_prefix=str(
            payload.get("key_prefix") or "comfy-modal-cache/v1/blobs/sha256"
        ).strip(),
        write_back_mode=str(payload.get("write_back_mode") or "async")
        .strip()
        .casefold(),
    )


def _planned_execution_assignments_payload(
    assignments: Mapping[str, ExecutionAssignment],
    components: Sequence[RemoteComponentPlan],
    *,
    configurations_by_id: Mapping[str, Any] | None = None,
    ssh_hosts_by_id: Mapping[str, SshHostConfig] | None = None,
    vast_quotes: Mapping[tuple[str, str], VastProfileQuote] | None = None,
    vast_leases_by_environment: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Serialize scheduler choices before provider capacity is acquired."""
    component_nodes = {
        component.representative_node_id: list(component.node_ids)
        for component in components
    }
    return {
        component_id: {
            "provider": assignment.provider.value,
            "environment_id": assignment.environment_id,
            "configuration_id": assignment.configuration_id,
            "node_ids": component_nodes.get(component_id, [component_id]),
            "predicted_cost_usd": assignment.predicted_cost_usd,
            "predicted_completion_seconds": assignment.predicted_completion_seconds,
            "worker_index": assignment.capacity_slot_index,
            "reasons": list(assignment.reasons),
            "hardware": _assignment_hardware_payload(
                component_id=component_id,
                assignment=assignment,
                configurations_by_id=configurations_by_id or {},
                ssh_hosts_by_id=ssh_hosts_by_id or {},
                vast_quotes=vast_quotes or {},
                vast_leases_by_environment=vast_leases_by_environment or {},
            ),
        }
        for component_id, assignment in sorted(assignments.items())
    }


def _configuration_field(configuration: Any, field_name: str) -> Any:
    """Read one field from a configuration model or safe mapping."""
    if isinstance(configuration, Mapping):
        return configuration.get(field_name)
    return getattr(configuration, field_name, None)


def _configuration_host(configuration: Any) -> SshHostConfig | None:
    """Return a workflow SSH host from a model or safe configuration mapping."""
    host = _configuration_field(configuration, "host")
    if isinstance(host, SshHostConfig):
        return host
    if isinstance(host, Mapping):
        try:
            return SshHostConfig.from_dict(host)
        except (TypeError, ValueError):
            return None
    return None


def _assignment_hardware_payload(
    *,
    component_id: str,
    assignment: ExecutionAssignment,
    configurations_by_id: Mapping[str, Any],
    ssh_hosts_by_id: Mapping[str, SshHostConfig],
    vast_quotes: Mapping[tuple[str, str], VastProfileQuote],
    vast_leases_by_environment: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the best hardware identity known for one planned assignment."""
    configuration_id = str(assignment.configuration_id or "")
    configuration = configurations_by_id.get(configuration_id)
    if assignment.provider is ExecutionProvider.MODAL:
        gpu_type = str(
            _configuration_field(configuration, "gpu_type")
            or assignment.environment_id.rsplit(":", 1)[-1]
        )
        return _modal_hardware_payload(gpu_type)
    if assignment.provider is ExecutionProvider.SSH_DOCKER:
        host = ssh_hosts_by_id.get(configuration_id) or _configuration_host(
            configuration
        )
        return _capabilities_hardware_payload(
            host.capabilities if host is not None else None
        )
    lease = vast_leases_by_environment.get(assignment.environment_id)
    if lease is not None:
        return _vast_hardware_payload(lease)
    quote = vast_quotes.get((component_id, configuration_id))
    if quote is None:
        return None
    resource = getattr(quote, "existing_lease", None) or getattr(
        quote,
        "offer",
        None,
    )
    return _vast_hardware_payload(resource) if resource is not None else None


def _probe_configured_ssh_hosts(
    configuration_set: RemoteConfigurationSet,
) -> dict[str, SshHostConfig]:
    """Probe all workflow-declared SSH hosts concurrently."""
    ssh_configurations = [
        configuration
        for configuration in configuration_set.configurations
        if isinstance(configuration, SshRemoteConfiguration)
    ]
    if not ssh_configurations:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(ssh_configurations))) as executor:
        probed_hosts = {
            configuration.configuration_id: future.result()
            for configuration, future in (
                (
                    configuration,
                    executor.submit(
                        _probe_workflow_ssh_configuration,
                        configuration,
                    ),
                )
                for configuration in ssh_configurations
            )
        }
    return probed_hosts


def _configured_vast_service(
    configuration_set: RemoteConfigurationSet,
    settings: ModalSyncSettings,
) -> tuple[VastService | None, str | None]:
    """Construct one Vast service only when the connected graph requests it."""
    vast_configured = any(
        isinstance(configuration, VastRemoteConfiguration)
        for configuration in configuration_set.capacity_configurations
    )
    if not vast_configured:
        return None, None
    try:
        return (
            VastService.from_environment(
                settings,
                repo_root=Path(__file__).resolve().parent,
            ),
            None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        has_fallback_provider = any(
            not isinstance(configuration, VastRemoteConfiguration)
            for configuration in configuration_set.capacity_configurations
        )
        if not has_fallback_provider:
            raise ModalPromptValidationError(str(exc)) from exc
        logger.warning(
            "Skipping unavailable workflow Vast.ai capacity because another "
            "configured provider can be planned: %s",
            exc,
        )
        return None, str(exc)


def _configured_component_requirements(
    *,
    components: list[RemoteComponentPlan],
    prompt: Mapping[str, Any],
    workflow: Mapping[str, Any] | None,
    settings: ModalSyncSettings,
    resolved_llm_profiles: Mapping[str, Any] | None = None,
) -> dict[str, ComponentResourceRequirements]:
    """Build provider-neutral requirements for every remote component."""
    preferences = WorkflowExecutionPreferences.from_workflow(workflow)
    active_llm_profiles = (
        dict(resolved_llm_profiles)
        if resolved_llm_profiles is not None
        else _resolve_prompt_llm_profiles(prompt, settings)
    )
    requirements: dict[str, ComponentResourceRequirements] = {}
    for component in components:
        memory_estimate = _component_memory_estimate(
            component,
            prompt,
            preferences,
            settings,
            active_llm_profiles,
        )
        component_id = component.representative_node_id
        requirements[component_id] = ComponentResourceRequirements(
            minimum_vram_bytes=memory_estimate.minimum_vram_bytes,
            minimum_ram_bytes=memory_estimate.minimum_ram_bytes,
            gpu_required=True,
            architecture="x86_64",
            estimated_execution_seconds=60.0,
            preferred_environment_ids=preferences.preferred_environment_ids,
            required_provider=_component_required_provider(
                component,
                prompt,
                active_llm_profiles,
            ),
        )
    return requirements


def _configured_candidate_environments(
    *,
    configuration_set: RemoteConfigurationSet,
    requirements_by_component: Mapping[str, ComponentResourceRequirements],
    settings: ModalSyncSettings,
    ssh_hosts_by_id: Mapping[str, SshHostConfig],
    vast_service: VastService | None,
    vast_unavailable_reason: str | None,
    occupied_environment_ids: frozenset[str] = frozenset(),
) -> tuple[
    dict[str, list[EnvironmentSchedulingState]],
    dict[tuple[str, str], VastProfileQuote],
]:
    """Resolve candidate pool states for every component without acquiring capacity."""
    candidates: dict[str, list[EnvironmentSchedulingState]] = {
        component_id: [] for component_id in requirements_by_component
    }
    vast_quotes: dict[tuple[str, str], VastProfileQuote] = {}
    for component_id, requirements in requirements_by_component.items():
        for configuration in configuration_set.capacity_configurations:
            state, quote = _configured_candidate_environment(
                configuration=configuration,
                requirements=requirements,
                settings=settings,
                ssh_hosts_by_id=ssh_hosts_by_id,
                vast_service=vast_service,
                vast_unavailable_reason=vast_unavailable_reason,
                occupied_environment_ids=occupied_environment_ids,
            )
            if state is not None:
                candidates[component_id].append(state)
            if quote is not None:
                vast_quotes[(component_id, configuration.configuration_id)] = quote
    return candidates, vast_quotes


def _reclaim_idle_configured_ssh_capacity(
    *,
    configuration_set: RemoteConfigurationSet,
    requirements_by_component: Mapping[str, ComponentResourceRequirements],
    ssh_hosts_by_id: dict[str, SshHostConfig],
    occupied_environment_ids: frozenset[str],
) -> bool:
    """Recycle idle configured workers when only their cached memory blocks work."""
    scheduler = CostAwareEnvironmentScheduler()
    reclaimed = False
    for configuration in configuration_set.capacity_configurations:
        if not isinstance(configuration, SshRemoteConfiguration):
            continue
        host = ssh_hosts_by_id[configuration.configuration_id]
        if host.environment_id in occupied_environment_ids:
            continue
        state = replace(
            host.scheduling_state(),
            configuration_id=configuration.configuration_id,
            display_name=configuration.display_name,
        )
        optimistic_state = _maximum_capacity_state(state)
        reclaimable = any(
            _optional_scheduler_choice(scheduler, [state], requirements)[0] is None
            and _optional_scheduler_choice(
                scheduler,
                [optimistic_state],
                requirements,
            )[0]
            is not None
            for requirements in requirements_by_component.values()
        )
        if not reclaimable:
            continue
        removed_workers = _remove_idle_ssh_workers_for_reclaim(host)
        if not removed_workers:
            continue
        refreshed_host = _probe_workflow_ssh_configuration(configuration)
        ssh_hosts_by_id[configuration.configuration_id] = refreshed_host
        reclaimed = True
        logger.info(
            "Recycled idle workflow-configured SSH worker(s) environment=%s "
            "containers=%s and refreshed available memory.",
            host.environment_id,
            removed_workers,
        )
    return reclaimed


def _prefetch_configured_vast_offers(
    *,
    configuration_set: RemoteConfigurationSet,
    requirements_by_component: Mapping[str, ComponentResourceRequirements],
    vast_service: VastService | None,
) -> None:
    """Warm distinct Vast searches in parallel before deterministic planning."""
    if vast_service is None:
        return
    profiles = [
        configuration.profile
        for configuration in configuration_set.configurations
        if isinstance(configuration, VastRemoteConfiguration)
    ]
    prefetch = getattr(vast_service, "prefetch_offers_sync", None)
    if not callable(prefetch):
        return
    try:
        prefetch(
            profiles,
            tuple(
                VastSearchRequirements(
                    minimum_vram_bytes=requirements.minimum_vram_bytes,
                    minimum_ram_bytes=requirements.minimum_ram_bytes,
                )
                for requirements in requirements_by_component.values()
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Unable to prefetch configured Vast.ai marketplace offers: %s",
            exc,
        )


def _configured_candidate_environment(
    *,
    configuration: RemoteConfiguration,
    requirements: ComponentResourceRequirements,
    settings: ModalSyncSettings,
    ssh_hosts_by_id: Mapping[str, SshHostConfig],
    vast_service: VastService | None,
    vast_unavailable_reason: str | None,
    occupied_environment_ids: frozenset[str] = frozenset(),
) -> tuple[EnvironmentSchedulingState | None, VastProfileQuote | None]:
    """Resolve one configuration as a candidate for one component."""
    if isinstance(configuration, ModalRemoteConfiguration):
        return _modal_configuration_environment_state(configuration, settings), None
    if isinstance(configuration, SshRemoteConfiguration):
        host = ssh_hosts_by_id[configuration.configuration_id]
        state = replace(
            host.scheduling_state(),
            configuration_id=configuration.configuration_id,
            display_name=configuration.display_name,
        )
        if state.environment_id in occupied_environment_ids:
            state = _maximum_capacity_state(state)
            logger.info(
                "Planning queued work against reclaimable SSH capacity "
                "environment=%s because an earlier workflow owns that host.",
                state.environment_id,
            )
        return state, None
    if not isinstance(configuration, VastRemoteConfiguration):
        raise TypeError(
            f"Unsupported remote configuration type {type(configuration).__name__}."
        )
    if vast_service is None:
        return EnvironmentSchedulingState(
            environment_id=f"vast:{configuration.configuration_id}",
            provider=ExecutionProvider.VAST,
            enabled=True,
            health=EnvironmentHealth.UNAVAILABLE,
            cost_usd_per_second=None,
            capabilities=None,
            configuration_id=configuration.configuration_id,
            display_name=configuration.display_name,
            unavailable_reason=(
                vast_unavailable_reason
                or "Vast.ai controller is unavailable."
            ),
            maximum_workers=configuration.capacity_limit,
        ), None
    try:
        quote = vast_service.quote_best_profile_sync(
            [configuration.profile],
            minimum_vram_bytes=requirements.minimum_vram_bytes,
            minimum_ram_bytes=requirements.minimum_ram_bytes,
            predicted_execution_seconds=requirements.estimated_execution_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Vast configuration has no candidate configuration=%s: %s",
            configuration.display_name,
            exc,
        )
        return EnvironmentSchedulingState(
            environment_id=f"vast:{configuration.configuration_id}",
            provider=ExecutionProvider.VAST,
            enabled=True,
            health=EnvironmentHealth.UNAVAILABLE,
            cost_usd_per_second=None,
            capabilities=None,
            configuration_id=configuration.configuration_id,
            display_name=configuration.display_name,
            unavailable_reason=str(exc),
            maximum_workers=configuration.capacity_limit,
        ), None
    state = vast_service.scheduling_state(quote)
    return replace(
        state,
        environment_id=f"vast:{configuration.configuration_id}",
        configuration_id=configuration.configuration_id,
        display_name=configuration.display_name,
        maximum_workers=configuration.capacity_limit,
    ), quote


def _apply_historical_execution_estimates(
    *,
    components: list[RemoteComponentPlan],
    environments_by_component: Mapping[str, list[EnvironmentSchedulingState]],
    requirements_by_component: dict[str, ComponentResourceRequirements],
    prompt: Mapping[str, Any],
    settings: ModalSyncSettings,
) -> None:
    """Add environment-specific historical timing estimates in place."""
    history = _execution_history(settings)
    if history is None:
        return
    for component in components:
        component_id = component.representative_node_id
        environment_ids = [
            environment.environment_id
            for environment in environments_by_component.get(component_id, [])
        ]
        estimates = history.estimates(
            _component_execution_signature(component, prompt),
            environment_ids,
        )
        requirements_by_component[component_id] = replace(
            requirements_by_component[component_id],
            estimated_execution_seconds_by_environment={
                environment_id: estimate.execution_seconds
                for environment_id, estimate in estimates.items()
            },
        )


def _prepare_selected_vast_capacity(
    *,
    assignments: dict[str, ExecutionAssignment],
    configuration_set: RemoteConfigurationSet,
    requirements_by_component: Mapping[str, ComponentResourceRequirements],
    vast_quotes: Mapping[tuple[str, str], VastProfileQuote],
    vast_service: VastService | None,
    status_callback: SetupStatusCallback | None = None,
    environment_status_callback: EnvironmentSetupStatusCallback | None = None,
) -> dict[str, Any]:
    """Acquire only Vast slots selected by the completed provider-neutral plan."""
    selected_slots: dict[tuple[str, int], list[str]] = defaultdict(list)
    for component_id, assignment in assignments.items():
        if assignment.provider is not ExecutionProvider.VAST:
            continue
        if assignment.configuration_id is None:
            raise ModalPromptValidationError(
                "Vast assignment is missing its configuration identity."
            )
        selected_slots[
            (assignment.configuration_id, assignment.capacity_slot_index)
        ].append(component_id)
    if not selected_slots:
        return {}
    if vast_service is None:
        raise ModalPromptValidationError(
            "Vast capacity was selected without an initialized Vast service."
        )

    configurations_by_id = {
        configuration.configuration_id: configuration
        for configuration in configuration_set.configurations
        if isinstance(configuration, VastRemoteConfiguration)
    }
    leases_by_environment: dict[str, Any] = {}
    ordered_slots = sorted(selected_slots.items())
    total_slots = len(ordered_slots)
    for slot_number, ((configuration_id, slot_index), component_ids) in enumerate(
        ordered_slots,
        start=1,
    ):
        configuration = configurations_by_id[configuration_id]
        planned_environment_id = assignments[component_ids[0]].environment_id
        quote = _quote_selected_vast_slot(
            configuration=configuration,
            component_ids=component_ids,
            requirements_by_component=requirements_by_component,
            vast_quotes=vast_quotes,
            vast_service=vast_service,
        )
        try:
            _emit_vast_capacity_status(
                environment_id=planned_environment_id,
                message=f"Acquiring Vast.ai capacity {slot_number} of {total_slots}",
                current=slot_number - 1,
                total=total_slots,
                status_callback=status_callback,
                environment_status_callback=environment_status_callback,
            )
            if status_callback is None and environment_status_callback is None:
                lease = vast_service.acquire_sync(quote, slot=slot_index)
            else:
                lease = vast_service.acquire_sync(
                    quote,
                    slot=slot_index,
                    status_callback=(
                        lambda message, environment_id=planned_environment_id,
                        current=slot_number - 1: _emit_vast_capacity_status(
                            environment_id=environment_id,
                            message=message,
                            current=current,
                            total=total_slots,
                            status_callback=status_callback,
                            environment_status_callback=environment_status_callback,
                        )
                    ),
                )
        except SyncCancelledError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModalPromptValidationError(
                f"Unable to acquire Vast.ai capacity for configuration "
                f"{configuration.display_name!r} slot {slot_index}: {exc}"
            ) from exc
        leases_by_environment[lease.environment_id] = lease
        cost_share = quote.predicted_incremental_cost_usd / len(component_ids)
        for component_id in component_ids:
            assignment = assignments[component_id]
            assignments[component_id] = replace(
                assignment,
                environment_id=lease.environment_id,
                predicted_cost_usd=cost_share,
                reasons=assignment.reasons
                + (
                    f"Vast profile {configuration.display_name}",
                    f"Vast capacity slot {slot_index}",
                ),
            )
        _emit_vast_capacity_status(
            environment_id=lease.environment_id,
            message=f"Vast.ai capacity {slot_number} of {total_slots} is ready",
            environment_message="Vast.ai worker ready; preparing remote assets next",
            current=slot_number,
            total=total_slots,
            status_callback=status_callback,
            environment_status_callback=environment_status_callback,
        )
    return leases_by_environment


def _emit_vast_capacity_status(
    *,
    environment_id: str,
    message: str,
    environment_message: str | None = None,
    current: int,
    total: int,
    status_callback: SetupStatusCallback | None,
    environment_status_callback: EnvironmentSetupStatusCallback | None,
) -> None:
    """Publish one Vast setup update at prompt and environment scopes."""
    if status_callback is not None:
        status_callback(message, current, total)
    if environment_status_callback is not None:
        environment_status_callback(
            environment_id,
            environment_message or message,
            None,
            None,
        )


def _quote_selected_vast_slot(
    *,
    configuration: VastRemoteConfiguration,
    component_ids: list[str],
    requirements_by_component: Mapping[str, ComponentResourceRequirements],
    vast_quotes: Mapping[tuple[str, str], VastProfileQuote],
    vast_service: VastService,
) -> VastProfileQuote:
    """Quote aggregate slot requirements immediately before acquisition."""
    minimum_vram_bytes = max(
        requirements_by_component[component_id].minimum_vram_bytes
        for component_id in component_ids
    )
    minimum_ram_bytes = max(
        requirements_by_component[component_id].minimum_ram_bytes
        for component_id in component_ids
    )
    predicted_execution_seconds = sum(
        requirements_by_component[component_id].estimated_execution_seconds
        for component_id in component_ids
    )
    if len(component_ids) == 1:
        existing_quote = vast_quotes.get(
            (component_ids[0], configuration.configuration_id)
        )
        if existing_quote is not None:
            return existing_quote
    return vast_service.quote_best_profile_sync(
        [configuration.profile],
        minimum_vram_bytes=minimum_vram_bytes,
        minimum_ram_bytes=minimum_ram_bytes,
        predicted_execution_seconds=predicted_execution_seconds,
    )


def _ssh_sync_engine(
    *,
    host: SshHostConfig,
    settings: ModalSyncSettings,
    r2_cache: R2CacheClient | None = None,
) -> ModalAssetSyncEngine:
    """Build a content-addressed sync engine for one SSH Docker host."""
    ssh_settings = replace(
        settings,
        execution_mode="ssh_docker",
        volume_name=host.resolved_storage_volume_name,
        local_storage_root=(
            settings.local_storage_root / "ssh" / host.environment_id
        ).resolve(),
        remote_storage_root="/storage",
    )
    resolved_r2_cache = r2_cache or R2CacheClient.from_environment()
    controller = SshDockerController(host)
    runtime_manager = SshRuntimeManager(
        controller=controller,
        repo_root=Path(__file__).resolve().parent,
        settings=settings,
    )
    materializer_spec = runtime_manager.runtime_spec()

    def prepare_materializer_image() -> None:
        """Make the SSH runtime image available before an R2 helper starts."""
        runtime_manager.ensure_image(materializer_spec)

    volume = SshDockerVolumeBackend(
        controller,
        host.resolved_storage_volume_name,
        materializer_image=materializer_spec.image_tag,
        materializer_image_preparer=(
            prepare_materializer_image if resolved_r2_cache is not None else None
        ),
    )
    return ModalAssetSyncEngine(
        volume=volume,
        settings=ssh_settings,
        r2_cache=resolved_r2_cache,
    )


def _workflow_r2_cache(
    configuration_set: RemoteConfigurationSet | None,
) -> R2CacheClient | None:
    """Resolve the connected R2 backing through its opaque OS-keyring reference."""
    if configuration_set is None:
        return None
    r2_configurations = [
        configuration
        for configuration in configuration_set.storage_configurations
        if isinstance(configuration, R2StorageBackingConfiguration)
    ]
    if not r2_configurations:
        return None
    storage = r2_configurations[0]
    try:
        configuration = R2CredentialStore().cache_configuration(storage)
    except (RuntimeError, ValueError) as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    return R2CacheClient(configuration)


def _stamp_execution_assignment(
    payload: dict[str, Any],
    assignment: ExecutionAssignment,
    worker_index: int = 0,
    execution_history_signature: str | None = None,
    execution_location: str | None = None,
    provider_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Attach provider placement to a payload and every nested proxy phase."""
    payload["execution_provider"] = assignment.provider.value
    payload["execution_environment_id"] = assignment.environment_id
    payload["execution_worker_index"] = worker_index
    if execution_location:
        payload["execution_location"] = execution_location
    if execution_history_signature:
        payload["execution_history_signature"] = execution_history_signature
    if provider_metadata:
        payload.update(provider_metadata)
    split_payloads = payload.get("split_proxy_payloads")
    nested_payloads: list[dict[str, Any]] = []
    if isinstance(split_payloads, dict):
        nested_payloads.extend(
            nested_payload
            for nested_payload in split_payloads.values()
            if isinstance(nested_payload, dict)
        )
    elif isinstance(split_payloads, list):
        nested_payloads.extend(
            nested_payload
            for nested_payload in split_payloads
            if isinstance(nested_payload, dict)
        )
    for nested_payload in nested_payloads:
        _stamp_execution_assignment(
            nested_payload,
            assignment,
            worker_index,
            execution_history_signature,
            execution_location,
            provider_metadata,
        )


def _ssh_hostname(ssh_target: str) -> str:
    """Return the host portion of one validated OpenSSH destination."""
    host = ssh_target.rsplit("@", 1)[-1].strip()
    if host.startswith("[") and "]" in host:
        return host[1 : host.index("]")]
    return host


def _execution_location_for_assignment(
    assignment: ExecutionAssignment,
    ssh_hosts_by_id: Mapping[str, SshHostConfig],
    vast_leases_by_environment: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the runtime location label known before remote dispatch."""
    if assignment.provider is ExecutionProvider.MODAL:
        return None
    if assignment.provider is ExecutionProvider.VAST:
        lease = (vast_leases_by_environment or {}).get(assignment.environment_id)
        if lease is not None:
            return str(lease.ssh_host or lease.gpu_name)
        return assignment.environment_id
    host = ssh_hosts_by_id.get(assignment.environment_id)
    return _ssh_hostname(host.ssh_target) if host is not None else assignment.environment_id


def _vast_provider_metadata(lease: Any) -> dict[str, Any]:
    """Return credential-free lease details for execution and local status output."""
    return {
        "vast_instance_id": lease.instance_id,
        "vast_profile_id": lease.profile_id,
        "vast_profile_name": lease.profile_name,
        "vast_gpu_name": lease.gpu_name,
        "vast_gpu_count": lease.gpu_count,
        "vast_gpu_ram_mb": lease.gpu_ram_mb,
        "vast_hourly_cost_usd": lease.hourly_cost_usd,
        "vast_idle_retention_seconds": lease.idle_retention_seconds,
    }


def _configured_provider_metadata(
    *,
    execution_plan: ComponentExecutionPlan,
    assignment: ExecutionAssignment,
    vast_leases_by_environment: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return safe provider metadata needed to execute one prepared assignment."""
    if assignment.provider is ExecutionProvider.VAST:
        return _vast_provider_metadata(
            vast_leases_by_environment[assignment.environment_id]
        )
    configuration_id = assignment.configuration_id
    if configuration_id is None:
        return None
    configuration = execution_plan.configurations_by_id.get(configuration_id)
    if isinstance(configuration, ModalRemoteConfiguration):
        return {
            "remote_configuration_id": configuration.configuration_id,
            "remote_configuration_name": configuration.display_name,
            "modal_gpu": configuration.gpu_type,
            "modal_max_containers": configuration.instance_count,
        }
    if isinstance(configuration, SshRemoteConfiguration):
        host = execution_plan.ssh_hosts_by_id[configuration.configuration_id]
        portable_host = replace(
            host,
            health=EnvironmentHealth.UNKNOWN,
            last_error=None,
        )
        return {
            "remote_configuration_id": configuration.configuration_id,
            "remote_configuration_name": configuration.display_name,
            "ssh_host_config": portable_host.to_dict(),
        }
    return None


def _ensure_remote_sync_backend(
    settings: ModalSyncSettings,
    sync_engine: ModalAssetSyncEngine,
) -> None:
    """Fail before queueing when remote execution cannot write to Modal-visible storage."""
    if settings.execution_mode == "local":
        return
    if isinstance(sync_engine.volume, ModalVolumeBackend):
        return
    raise ModalPromptValidationError(
        "Remote Modal execution requires asset sync to use the Modal volume backend, "
        f"but the active sync backend is {type(sync_engine.volume).__name__}. "
        "Restart ComfyUI with COMFY_MODAL_EXECUTION_MODE=remote and the Modal SDK available "
        "so synced assets and custom_nodes bundles are visible inside Modal workers."
    )


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


async def _queue_prompt_json(
    prompt_server: Any,
    json_data: dict[str, Any],
    modal_response_payload: dict[str, Any] | None = None,
) -> web.Response:
    """Queue a possibly rewritten prompt using ComfyUI's native semantics."""
    execution = _get_execution_module()
    json_data = prompt_server.trigger_on_prompt(json_data)

    if "number" in json_data:
        number = float(json_data["number"])
    else:
        number = prompt_server.number
        if json_data.get("front"):
            number = -number
        prompt_server.number += 1

    if "prompt" not in json_data:
        return web.json_response(
            {
                "error": {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {},
                }
            },
            status=400,
        )

    prompt = json_data["prompt"]
    prompt_id = str(json_data.get("prompt_id", uuid.uuid4()))
    partial_execution_targets = json_data.get("partial_execution_targets")
    extra_data = dict(json_data.get("extra_data", {}))
    if "client_id" in json_data:
        extra_data["client_id"] = json_data["client_id"]
    valid = await execution.validate_prompt(
        prompt_id, prompt, partial_execution_targets
    )

    if not valid[0]:
        modal_extra = extra_data.get("modal")
        if isinstance(modal_extra, Mapping) and modal_extra.get("remote_component_ids"):
            logger.warning(
                "ComfyUI rejected rewritten Modal prompt prompt_id=%s error=%s node_errors=%s",
                prompt_id,
                valid[1],
                valid[3],
            )
            _log_modal_rewritten_prompt_diagnostics(
                prompt_id=prompt_id,
                prompt=prompt,
                reason="comfy_validation_failure",
                level=logging.WARNING,
            )
        else:
            logger.warning("invalid prompt: %s", valid[1])
        return web.json_response(
            {"error": valid[1], "node_errors": valid[3]}, status=400
        )

    outputs_to_execute = valid[2]
    sensitive: dict[str, Any] = {}
    for sensitive_key in execution.SENSITIVE_EXTRA_DATA_KEYS:
        if sensitive_key in extra_data:
            sensitive[sensitive_key] = extra_data.pop(sensitive_key)

    extra_data["create_time"] = int(time.time() * 1000)
    prompt_server.prompt_queue.put(
        (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
    )
    response_payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "number": number,
        "node_errors": valid[3],
    }
    if modal_response_payload:
        response_payload.update(modal_response_payload)
    return web.json_response(response_payload)


def _vast_registry(settings: ModalSyncSettings) -> VastLeaseRegistry | None:
    """Return persistent credential-free Vast lease state when available."""
    user_directory = discover_comfyui_user_directory(settings)
    if user_directory is None:
        return None
    return VastLeaseRegistry.for_user_directory(user_directory)


def _install_modal_interrupt_queue_bridge(prompt_server: Any) -> None:
    """Expose active remote work through every ComfyUI queue-state view."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    if prompt_queue is None or getattr(
        prompt_queue, _MODAL_INTERRUPT_QUEUE_BRIDGE_ATTR, False
    ):
        return

    original_get_current_queue = getattr(prompt_queue, "get_current_queue", None)
    original_get_current_queue_volatile = getattr(
        prompt_queue, "get_current_queue_volatile", None
    )
    original_get_tasks_remaining = getattr(prompt_queue, "get_tasks_remaining", None)
    original_interrupt_if_running = getattr(
        prompt_queue, "interrupt_if_running", None
    )
    original_task_done = getattr(prompt_queue, "task_done", None)
    original_wipe_queue = getattr(prompt_queue, "wipe_queue", None)
    original_delete_queue_item = getattr(prompt_queue, "delete_queue_item", None)
    if not any(
        callable(method)
        for method in (
            original_get_current_queue,
            original_get_current_queue_volatile,
            original_get_tasks_remaining,
        )
    ):
        logger.debug(
            "Prompt queue does not expose queue-state methods; skipping remote queue bridge."
        )
        return

    preparation_prompts: dict[str, tuple[Any, ...]] = {}
    preparation_cancellations: dict[str, threading.Event] = {}
    preparation_lock = threading.RLock()
    setattr(prompt_queue, _REMOTE_PREPARATION_PROMPTS_ATTR, preparation_prompts)
    setattr(
        prompt_queue,
        _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
        preparation_cancellations,
    )
    setattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, preparation_lock)

    def preparation_items() -> list[tuple[Any, ...]]:
        """Return a stable snapshot of prompts still preparing remote capacity."""
        with preparation_lock:
            return list(preparation_prompts.values())

    def append_missing_preparations(
        running: Iterable[Any],
        queued: Iterable[Any],
    ) -> tuple[list[Any], Any]:
        """Add preparation entries that are not already in ComfyUI's native queue."""
        running_items = list(running)
        queued_items = list(queued)
        native_prompt_ids = _queue_item_prompt_ids((*running_items, *queued_items))
        running_items.extend(
            item
            for item in preparation_items()
            if str(item[1]) not in native_prompt_ids
        )
        return running_items, queued

    if callable(original_get_current_queue):

        def remote_get_current_queue() -> tuple[list[Any], Any]:
            """Return native work plus preparation and active remote prompt entries."""
            running, queued = original_get_current_queue()
            running_items, queued = append_missing_preparations(running, queued)
            running_prompt_ids = _queue_item_prompt_ids(running_items)
            try:
                from .remote.modal_app import active_remote_modal_prompt_ids
            except ImportError:
                return running_items, queued

            for prompt_id in sorted(
                active_remote_modal_prompt_ids() - running_prompt_ids
            ):
                running_items.append((0, prompt_id, {}, {}, [], {}))
            return running_items, queued

        setattr(prompt_queue, "get_current_queue", remote_get_current_queue)

    if callable(original_get_current_queue_volatile):

        def remote_get_current_queue_volatile() -> tuple[list[Any], Any]:
            """Include remote preparation in ComfyUI's public `/queue` response."""
            running, queued = original_get_current_queue_volatile()
            return append_missing_preparations(running, queued)

        setattr(
            prompt_queue,
            "get_current_queue_volatile",
            remote_get_current_queue_volatile,
        )

    if callable(original_get_tasks_remaining):

        def remote_get_tasks_remaining() -> int:
            """Count remote preparation as work in websocket queue status."""
            remaining = int(original_get_tasks_remaining())
            native_prompt_ids: set[str] = set()
            native_queue_method = (
                original_get_current_queue_volatile
                if callable(original_get_current_queue_volatile)
                else original_get_current_queue
            )
            if callable(native_queue_method):
                running, queued = native_queue_method()
                native_prompt_ids = _queue_item_prompt_ids((*running, *queued))
            return remaining + sum(
                str(item[1]) not in native_prompt_ids
                for item in preparation_items()
            )

        setattr(prompt_queue, "get_tasks_remaining", remote_get_tasks_remaining)

    if callable(original_interrupt_if_running):

        def remote_interrupt_if_running(prompt_id: str) -> bool:
            """Cancel remote preparation or interrupt matching native execution."""
            if _cancel_remote_preparation(prompt_server, prompt_id):
                return True
            return bool(original_interrupt_if_running(prompt_id))

        setattr(prompt_queue, "interrupt_if_running", remote_interrupt_if_running)

    if callable(original_task_done):

        def remote_task_done(item_id: Any, *args: Any, **kwargs: Any) -> Any:
            """Release background cache work after the whole prompt terminates."""
            currently_running = getattr(prompt_queue, "currently_running", {})
            running_item = (
                currently_running.get(item_id)
                if isinstance(currently_running, Mapping)
                else None
            )
            prompt_id = (
                str(running_item[1])
                if isinstance(running_item, (list, tuple)) and len(running_item) > 1
                else None
            )
            try:
                return original_task_done(item_id, *args, **kwargs)
            finally:
                if prompt_id is not None:
                    finish_r2_writeback_prompt(prompt_id)

        setattr(prompt_queue, "task_done", remote_task_done)

    if callable(original_wipe_queue):

        def remote_wipe_queue() -> Any:
            """Release reservations belonging to every discarded queued prompt."""
            queued_items: list[Any] = []
            if callable(original_get_current_queue):
                _running, queued = original_get_current_queue()
                queued_items = list(queued)
            try:
                return original_wipe_queue()
            finally:
                for queued_prompt_id in _queue_item_prompt_ids(queued_items):
                    finish_r2_writeback_prompt(queued_prompt_id)

        setattr(prompt_queue, "wipe_queue", remote_wipe_queue)

    if callable(original_delete_queue_item):

        def remote_delete_queue_item(predicate: Callable[[Any], bool]) -> Any:
            """Release the exact reservation removed through ComfyUI's queue API."""
            before_prompt_ids: set[str] = set()
            if callable(original_get_current_queue):
                _running, queued = original_get_current_queue()
                before_prompt_ids = _queue_item_prompt_ids(queued)
            result = original_delete_queue_item(predicate)
            if result and callable(original_get_current_queue):
                _running, queued = original_get_current_queue()
                after_prompt_ids = _queue_item_prompt_ids(queued)
                for removed_prompt_id in before_prompt_ids - after_prompt_ids:
                    finish_r2_writeback_prompt(removed_prompt_id)
            return result

        setattr(prompt_queue, "delete_queue_item", remote_delete_queue_item)

    setattr(prompt_queue, _MODAL_INTERRUPT_QUEUE_BRIDGE_ATTR, True)
    logger.info("Installed remote preparation bridge on ComfyUI prompt queue.")


def _queue_item_prompt_ids(items: Iterable[Any]) -> set[str]:
    """Return prompt IDs from well-formed native or synthetic queue items."""
    return {
        str(item[1])
        for item in items
        if isinstance(item, (list, tuple)) and len(item) > 1
    }


def _cancel_remote_preparation(prompt_server: Any, prompt_id: str) -> bool:
    """Signal cancellation when the prompt is preparing remote capacity."""
    normalized_prompt_id = str(prompt_id).strip()
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    cancellations = getattr(
        prompt_queue,
        _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
        None,
    )
    preparation_lock = getattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, None)
    if (
        not normalized_prompt_id
        or not isinstance(cancellations, dict)
        or preparation_lock is None
    ):
        return False
    with preparation_lock:
        cancellation_event = cancellations.get(normalized_prompt_id)
        if cancellation_event is None:
            return False
        cancellation_event.set()
    logger.info(
        "Cancelled remote preparation for prompt %s.", normalized_prompt_id
    )
    return True


def _queued_ssh_environment_ids(
    prompt_server: Any,
    *,
    excluding_prompt_id: str | None = None,
) -> frozenset[str]:
    """Return SSH environments reserved by prompts already in ComfyUI's queue."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    get_current_queue = getattr(prompt_queue, "get_current_queue", None)
    if not callable(get_current_queue):
        return frozenset()
    queue_state = get_current_queue()
    if not isinstance(queue_state, (list, tuple)) or len(queue_state) != 2:
        return frozenset()
    running, queued = queue_state
    queue_items = [
        item
        for collection in (running, queued)
        if isinstance(collection, (list, tuple))
        for item in collection
    ]
    environment_ids: set[str] = set()
    for item in queue_items:
        if not isinstance(item, (list, tuple)) or len(item) <= 3:
            continue
        prompt_id = str(item[1]) if len(item) > 1 else ""
        if excluding_prompt_id is not None and prompt_id == excluding_prompt_id:
            continue
        extra_data = item[3]
        if not isinstance(extra_data, Mapping):
            continue
        remote_execution = extra_data.get("remote_execution")
        if not isinstance(remote_execution, Mapping):
            continue
        assignments = remote_execution.get("assignments")
        if not isinstance(assignments, Mapping):
            continue
        for assignment in assignments.values():
            if not isinstance(assignment, Mapping):
                continue
            provider = str(assignment.get("provider") or "").strip().lower()
            environment_id = str(assignment.get("environment_id") or "").strip()
            if provider == ExecutionProvider.SSH_DOCKER.value and environment_id:
                environment_ids.add(environment_id)
    return frozenset(environment_ids)


def _set_remote_preparation(
    prompt_server: Any,
    *,
    prompt_id: str,
    prompt: Mapping[str, Any],
    extra_data: Mapping[str, Any],
    cancellation_event: threading.Event | None = None,
) -> bool:
    """Register one pre-queue remote workflow and publish its active queue state."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    preparations = getattr(prompt_queue, _REMOTE_PREPARATION_PROMPTS_ATTR, None)
    preparation_lock = getattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, None)
    if not isinstance(preparations, dict) or preparation_lock is None:
        return False
    preparation_extra_data = copy.deepcopy(dict(extra_data))
    preparation_extra_data.setdefault("create_time", int(time.time() * 1000))
    with preparation_lock:
        preparations[prompt_id] = (
            0,
            prompt_id,
            copy.deepcopy(dict(prompt)),
            preparation_extra_data,
            [],
            {},
        )
        cancellations = getattr(
            prompt_queue,
            _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
            None,
        )
        if isinstance(cancellations, dict) and cancellation_event is not None:
            cancellations[prompt_id] = cancellation_event
    queue_updated = getattr(prompt_server, "queue_updated", None)
    if callable(queue_updated):
        queue_updated()
    return True


def _clear_remote_preparation(prompt_server: Any, prompt_id: str) -> None:
    """Remove one pre-queue remote workflow and publish the resulting queue state."""
    prompt_queue = getattr(prompt_server, "prompt_queue", None)
    preparations = getattr(prompt_queue, _REMOTE_PREPARATION_PROMPTS_ATTR, None)
    preparation_lock = getattr(prompt_queue, _REMOTE_PREPARATION_LOCK_ATTR, None)
    if not isinstance(preparations, dict) or preparation_lock is None:
        return
    with preparation_lock:
        removed = preparations.pop(prompt_id, None)
        cancellations = getattr(
            prompt_queue,
            _REMOTE_PREPARATION_CANCELLATIONS_ATTR,
            None,
        )
        if isinstance(cancellations, dict):
            cancellations.pop(prompt_id, None)
    if removed is not None:
        queue_updated = getattr(prompt_server, "queue_updated", None)
        if callable(queue_updated):
            queue_updated()


def setup_modal_queue_route(
    prompt_server: Any | None = None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
) -> None:
    """Register the `/modal/queue_prompt` route once for the active PromptServer."""
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
    analysis_route_path = _analysis_route_path(resolved_settings.route_path)
    progress_state_route_path = _progress_state_route_path(resolved_settings.route_path)
    container_status_route_path = _container_status_route_path(
        resolved_settings.route_path
    )
    container_stop_route_path = container_status_route_path.replace(
        "/container_status", "/container_stop"
    )
    delete_caches_route_path = _delete_modal_caches_route_path(
        resolved_settings.route_path
    )
    delete_volume_route_path = _delete_modal_volume_route_path(
        resolved_settings.route_path
    )
    cancel_preparation_route_path = _cancel_preparation_route_path(
        resolved_settings.route_path
    )
    remote_environments_route_path = _remote_environments_route_path(
        resolved_settings.route_path
    )
    remote_environment_probe_route_path = _remote_environment_probe_route_path(
        resolved_settings.route_path
    )
    remote_environment_bootstrap_route_path = _remote_environment_bootstrap_route_path(
        resolved_settings.route_path
    )
    remote_environment_status_route_path = _remote_environment_status_route_path(
        resolved_settings.route_path
    )
    remote_environment_stop_route_path = _remote_environment_stop_route_path(
        resolved_settings.route_path
    )
    remote_host_registry = _ssh_host_registry(resolved_settings)
    vast_registry = _vast_registry(resolved_settings)
    _install_modal_interrupt_queue_bridge(prompt_server)

    @prompt_server.routes.post(_R2_STORAGE_USAGE_ROUTE)
    async def refresh_r2_storage_usage(request: web.Request) -> web.Response:
        """Refresh one configured R2 bucket's safe aggregate storage state."""
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping):
                raise TypeError("R2 storage usage request must be a JSON object.")
            storage = _r2_storage_from_usage_payload(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        try:
            usage = await asyncio.to_thread(_refresh_r2_storage_usage, storage)
        except R2CredentialError as exc:
            logger.warning(
                "Unable to refresh R2 storage usage configuration=%s bucket=%s: %s",
                storage.configuration_id,
                storage.bucket,
                exc,
            )
            status = 423 if exc.code == R2_KEYCHAIN_UNLOCK_REQUIRED_CODE else 502
            return web.json_response(
                {"error": str(exc), "code": exc.code},
                status=status,
            )
        except (R2CacheError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Unable to refresh R2 storage usage configuration=%s bucket=%s: %s",
                storage.configuration_id,
                storage.bucket,
                exc,
            )
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {
                "configuration_id": storage.configuration_id,
                "storage_usage_bytes": usage.size_bytes,
                "storage_object_count": usage.object_count,
                "refreshed_at": time.time(),
            }
        )

    @prompt_server.routes.post(_R2_KEYCHAIN_UNLOCK_ROUTE)
    async def unlock_r2_keychain(request: web.Request) -> web.Response:
        """Display macOS's system-owned login-keychain unlock prompt."""
        del request
        try:
            await asyncio.to_thread(request_macos_keychain_unlock)
        except R2CredentialError as exc:
            logger.warning("Unable to unlock the macOS login keychain: %s", exc)
            return web.json_response({"error": str(exc)}, status=409)
        return web.json_response({"unlocked": True})

    @prompt_server.routes.post(cancel_preparation_route_path)
    async def cancel_remote_preparation(request: web.Request) -> web.Response:
        """Cancel one prompt while it is still preparing remote execution."""
        payload = await request.json()
        prompt_id = str(payload.get("prompt_id") or "").strip()
        cancelled = _cancel_remote_preparation(prompt_server, prompt_id)
        return web.json_response({"cancelled": cancelled, "prompt_id": prompt_id})

    if hasattr(prompt_server.routes, "get"):

        @prompt_server.routes.get(remote_environments_route_path)
        async def remote_environments(request: web.Request) -> web.Response:
            """Return credential-free SSH host configuration and discovered state."""
            del request
            if remote_host_registry is None:
                return web.json_response(
                    {
                        "error": "The ComfyUI user directory could not be resolved.",
                        "hosts": [],
                    },
                    status=503,
                )
            try:
                config = await asyncio.to_thread(remote_host_registry.load)
            except ValueError as exc:
                return web.json_response({"error": str(exc), "hosts": []}, status=500)
            return web.json_response(config.to_dict())

        @prompt_server.routes.get(progress_state_route_path)
        async def modal_progress_state(request: web.Request) -> web.Response:
            """Return recent Modal UI events for the requesting ComfyUI client."""
            client_id = request.query.get("client_id")
            return web.json_response({"events": modal_ui_events_for_client(client_id)})

        @prompt_server.routes.get(container_status_route_path)
        async def modal_container_status(request: web.Request) -> web.Response:
            """Return active containers and hourly billing for one selected GPU app."""
            from .remote.modal_app import (
                ModalBillingStatusError,
                ModalContainerStatusError,
                get_hourly_modal_app_billing,
                list_active_modal_containers,
            )

            requested_modal_gpu = request.query.get(
                "modal_gpu",
                resolved_settings.modal_gpu,
            )
            try:
                selected_settings = settings_for_modal_gpu(
                    resolved_settings,
                    requested_modal_gpu,
                )
            except ValueError as exc:
                return web.json_response(
                    {"containers": [], "error": str(exc), "polled_at": time.time()},
                    status=400,
                )

            containers_task = asyncio.create_task(
                list_active_modal_containers(resolved_settings)
            )
            include_billing = request.query.get("include_billing", "true").casefold() not in {
                "0",
                "false",
                "no",
            }
            billing_task = (
                asyncio.create_task(
                    get_hourly_modal_app_billing(
                        selected_settings.modal_gpu,
                        resolved_settings,
                    )
                )
                if include_billing
                else None
            )
            try:
                containers = await containers_task
            except ModalContainerStatusError as exc:
                if billing_task is not None:
                    billing_task.cancel()
                    await asyncio.gather(billing_task, return_exceptions=True)
                logger.warning("Unable to refresh Modal container status: %s", exc)
                return web.json_response(
                    {"containers": [], "error": str(exc), "polled_at": time.time()},
                    status=502,
                )
            billing = None
            billing_error = None
            if billing_task is not None:
                try:
                    billing = await billing_task
                except ModalBillingStatusError as exc:
                    billing_error = str(exc)
                    logger.warning("Unable to refresh Modal hourly billing: %s", exc)
            return web.json_response(
                {
                    "containers": [container.as_dict() for container in containers],
                    "billing": billing.as_dict() if billing is not None else None,
                    "billing_error": billing_error,
                    "polled_at": time.time(),
                }
            )

        @prompt_server.routes.post(container_stop_route_path)
        async def modal_container_stop(request: web.Request) -> web.Response:
            """Stop one exact active Modal container owned by this installation."""
            from .remote.modal_app import (
                ModalContainerStatusError,
                stop_managed_modal_container,
            )

            try:
                payload = await request.json()
                container_id = str(payload.get("container_id") or "").strip()
                stopped = await stop_managed_modal_container(
                    container_id,
                    resolved_settings,
                )
            except (ModalContainerStatusError, TypeError, ValueError) as exc:
                return web.json_response({"error": str(exc)}, status=502)
            return web.json_response(
                {"container_id": container_id, "stopped": stopped}
            )

        @prompt_server.routes.get("/remote/vast/status")
        async def vast_status(request: web.Request) -> web.Response:
            """Return refreshed credential-free managed Vast lease inventory."""
            del request
            if vast_registry is None:
                return web.json_response(
                    {"configured": False, "leases": [], "error": "ComfyUI user directory unavailable."},
                    status=503,
                )
            try:
                if not os.getenv("VAST_API_KEY"):
                    state = await asyncio.to_thread(vast_registry.load)
                    leases = state.leases
                else:
                    api_key = str(os.getenv("VAST_API_KEY") or "").strip()
                    base_url = str(
                        os.getenv("COMFY_MODAL_VAST_API_BASE_URL") or ""
                    ).strip()
                    api_client = VastApiClient(
                        api_key,
                        **({"base_url": base_url} if base_url else {}),
                    )
                    manager = VastLeaseManager.for_inventory(
                        api_client=api_client,
                        registry=vast_registry,
                        owner_id=resolved_settings.app_name,
                    )
                    leases = await manager.refresh_owned_leases()
            except (OSError, RuntimeError, ValueError) as exc:
                return web.json_response(
                    {"configured": bool(os.getenv("VAST_API_KEY")), "leases": [], "error": str(exc)},
                    status=502,
                )
            return web.json_response(
                {
                    "configured": bool(os.getenv("VAST_API_KEY")),
                    "image_configured": bool(os.getenv("COMFY_MODAL_VAST_IMAGE")),
                    "leases": [lease.to_dict() for lease in leases],
                }
            )

    if hasattr(prompt_server.routes, "put"):

        @prompt_server.routes.put(remote_environments_route_path)
        async def remote_environments_update(request: web.Request) -> web.Response:
            """Validate and atomically replace SSH host configuration."""
            if remote_host_registry is None:
                return web.json_response(
                    {"error": "The ComfyUI user directory could not be resolved."},
                    status=503,
                )
            try:
                payload = await request.json()
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        "Remote environment configuration must be a JSON object."
                    )
                config = RemoteExecutionConfig.from_dict(payload)
                await asyncio.to_thread(remote_host_registry.save, config)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)
            return web.json_response(config.to_dict())

    @prompt_server.routes.post(remote_environment_probe_route_path)
    async def remote_environment_probe(request: web.Request) -> web.Response:
        """Probe one configured SSH host and persist its discovered capabilities."""
        if remote_host_registry is None:
            return web.json_response(
                {"error": "The ComfyUI user directory could not be resolved."},
                status=503,
            )
        try:
            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            host = await asyncio.to_thread(
                remote_host_registry.get_host, environment_id
            )
            capabilities = await asyncio.to_thread(
                SshDockerController(host).probe_capabilities
            )
            updated_host = await asyncio.to_thread(
                remote_host_registry.update_probe_result,
                environment_id,
                capabilities=capabilities,
                health=EnvironmentHealth.READY,
                last_error=None,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            environment_id = str(locals().get("environment_id") or "").strip()
            if environment_id:
                try:
                    await asyncio.to_thread(
                        remote_host_registry.update_probe_result,
                        environment_id,
                        capabilities=None,
                        health=EnvironmentHealth.UNAVAILABLE,
                        last_error=str(exc),
                    )
                except (KeyError, ValueError):
                    pass
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(updated_host.to_dict())

    @prompt_server.routes.post("/remote/vast/verify")
    async def vast_verify(request: web.Request) -> web.Response:
        """Verify the configured Vast credential without returning it."""
        del request
        api_key = str(os.getenv("VAST_API_KEY") or "").strip()
        if not api_key:
            return web.json_response(
                {"verified": False, "error": "Set VAST_API_KEY first."},
                status=400,
            )
        base_url = str(os.getenv("COMFY_MODAL_VAST_API_BASE_URL") or "").strip()
        try:
            client = VastApiClient(api_key, **({"base_url": base_url} if base_url else {}))
            account = await client.verify_credentials()
        except (OSError, RuntimeError, ValueError) as exc:
            return web.json_response(
                {"verified": False, "error": str(exc)},
                status=502,
            )
        return web.json_response({"verified": True, "account": account})

    @prompt_server.routes.post("/remote/vast/reap")
    async def vast_reap(request: web.Request) -> web.Response:
        """Destroy only owned idle leases whose configured deadline has expired."""
        del request
        try:
            service = VastService.from_environment(
                resolved_settings,
                repo_root=Path(__file__).resolve().parent,
            )
            destroyed = await service.lease_manager.destroy_expired()
        except (OSError, RuntimeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response({"destroyed_instance_ids": list(destroyed)})

    @prompt_server.routes.post("/remote/vast/destroy")
    async def vast_destroy(request: web.Request) -> web.Response:
        """Destroy one exact idle registry-owned lease after server-side label checks."""
        try:
            payload = await request.json()
            instance_id = int(payload.get("instance_id"))
            api_key = str(os.getenv("VAST_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError("Set VAST_API_KEY before destroying Vast capacity.")
            if vast_registry is None:
                raise RuntimeError("ComfyUI user directory is unavailable.")
            base_url = str(
                os.getenv("COMFY_MODAL_VAST_API_BASE_URL") or ""
            ).strip()
            api_client = VastApiClient(
                api_key,
                **({"base_url": base_url} if base_url else {}),
            )
            manager = VastLeaseManager.for_inventory(
                api_client=api_client,
                registry=vast_registry,
                owner_id=resolved_settings.app_name,
            )
            destroyed = await manager.destroy_owned_lease(
                instance_id,
                allow_active_work=payload.get("force") is True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response({"instance_id": instance_id, "destroyed": destroyed})

    @prompt_server.routes.post(remote_environment_bootstrap_route_path)
    async def remote_environment_bootstrap(request: web.Request) -> web.Response:
        """Build the current runtime and start one compatible warm SSH worker."""
        if remote_host_registry is None:
            return web.json_response(
                {"error": "The ComfyUI user directory could not be resolved."},
                status=503,
            )
        environment_id = ""
        try:
            from .ssh_runtime import SshRuntimeManager

            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            worker_index = int(payload.get("worker_index", 0))
            host = await asyncio.to_thread(
                remote_host_registry.get_host, environment_id
            )
            manager = SshRuntimeManager(
                controller=SshDockerController(host),
                repo_root=Path(__file__).resolve().parent,
                settings=resolved_settings,
            )
            spec = await asyncio.to_thread(manager.ensure_worker, worker_index)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            if remote_host_registry is not None and environment_id:
                try:
                    await asyncio.to_thread(
                        remote_host_registry.update_probe_result,
                        environment_id,
                        capabilities=host.capabilities if "host" in locals() else None,
                        health=EnvironmentHealth.UNAVAILABLE,
                        last_error=str(exc),
                    )
                except (KeyError, ValueError):
                    pass
            return web.json_response({"error": str(exc)}, status=502)
        await asyncio.to_thread(
            remote_host_registry.update_probe_result,
            environment_id,
            capabilities=host.capabilities,
            health=EnvironmentHealth.READY,
            last_error=None,
        )
        return web.json_response(
            {
                "environment_id": environment_id,
                "worker_index": worker_index,
                "container_name": spec.container_name,
                "image_tag": spec.image_tag,
                "runtime_fingerprint": spec.identity.fingerprint,
            }
        )

    @prompt_server.routes.post(remote_environment_status_route_path)
    async def remote_environment_status(request: web.Request) -> web.Response:
        """Return managed worker state for one SSH execution environment."""
        if remote_host_registry is None:
            return web.json_response(
                {"error": "The ComfyUI user directory could not be resolved."},
                status=503,
            )
        try:
            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            host = await asyncio.to_thread(
                remote_host_registry.get_host, environment_id
            )
            workers = await asyncio.to_thread(
                SshDockerController(host).list_managed_workers
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {
                "environment_id": environment_id,
                "workers": [worker.to_dict() for worker in workers],
            }
        )

    @prompt_server.routes.post(remote_environment_stop_route_path)
    async def remote_environment_stop(request: web.Request) -> web.Response:
        """Stop all node-pack-managed workers on one configured SSH host."""
        if remote_host_registry is None:
            return web.json_response(
                {"error": "The ComfyUI user directory could not be resolved."},
                status=503,
            )
        try:
            from .ssh_runtime import SshRuntimeManager

            payload = await request.json()
            environment_id = str(payload.get("environment_id") or "").strip()
            host = await asyncio.to_thread(
                remote_host_registry.get_host, environment_id
            )
            manager = SshRuntimeManager(
                controller=SshDockerController(host),
                repo_root=Path(__file__).resolve().parent,
                settings=resolved_settings,
            )
            removed = await asyncio.to_thread(manager.stop_all_workers)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {"environment_id": environment_id, "removed_containers": list(removed)}
        )

    @prompt_server.routes.post(analysis_route_path)
    async def modal_analyze_remote_nodes(request: web.Request) -> web.Response:
        """Analyze which workflow nodes should be marked remote for the current graph."""
        logger.info("Received Modal remote-node analysis request.")
        try:
            request_started_at = time.perf_counter()
            json_data = await request.json()
            prompt = json_data.get("prompt")
            if not isinstance(prompt, dict):
                raise ValueError(
                    "Modal remote-node analysis requires a 'prompt' object."
                )

            workflow = json_data.get("workflow")
            seed_node_ids = json_data.get("seed_node_ids") or []
            if not isinstance(seed_node_ids, list):
                raise ValueError(
                    "Modal remote-node analysis requires 'seed_node_ids' to be a list."
                )

            analysis = analyze_remote_node_selection(
                prompt=prompt,
                workflow=workflow if isinstance(workflow, dict) else None,
                seed_workflow_node_paths=[
                    str(seed_node_id) for seed_node_id in seed_node_ids
                ],
                settings=resolved_settings,
            )
            logger.info(
                "Modal remote-node analysis finished in %.3fs with %d requested nodes and %d additions.",
                time.perf_counter() - request_started_at,
                len(analysis.requested_workflow_node_paths),
                len(analysis.added_workflow_node_paths),
            )
            return web.json_response(
                {
                    "requested_node_ids": analysis.requested_node_ids,
                    "requested_workflow_node_paths": analysis.requested_workflow_node_paths,
                    "current_remote_node_ids": analysis.current_remote_node_ids,
                    "current_remote_workflow_node_paths": (
                        analysis.current_remote_workflow_node_paths
                    ),
                    "resolved_remote_node_ids": analysis.resolved_remote_node_ids,
                    "resolved_workflow_node_paths": analysis.resolved_workflow_node_paths,
                    "added_node_ids": analysis.added_node_ids,
                    "added_workflow_node_paths": analysis.added_workflow_node_paths,
                    "sandwiched_local_node_ids": analysis.sandwiched_local_node_ids,
                    "reasons": [
                        {
                            "node_id": reason.node_id,
                            "class_type": reason.class_type,
                            "required_by_node_id": reason.required_by_node_id,
                            "required_by_class_type": reason.required_by_class_type,
                            "output_index": reason.output_index,
                            "io_type": reason.io_type,
                        }
                        for reason in analysis.reasons
                    ],
                }
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Modal remote-node analysis request was invalid: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)

    @prompt_server.routes.post(delete_caches_route_path)
    async def modal_delete_caches(request: web.Request) -> web.Response:
        """Delete persistent Modal cache Dicts for the active configuration."""
        del request
        logger.info("Received Modal cache deletion request.")
        try:
            return web.json_response(await delete_modal_cache_dicts(resolved_settings))
        except RuntimeError as exc:
            logger.warning("Modal cache deletion request failed: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)

    @prompt_server.routes.post(delete_volume_route_path)
    async def modal_delete_volume(request: web.Request) -> web.Response:
        """Delete the configured Modal Volume for the active configuration."""
        del request
        logger.info("Received Modal volume deletion request.")
        try:
            return web.json_response(await delete_modal_volume(resolved_settings))
        except RuntimeError as exc:
            logger.warning("Modal volume deletion request failed: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)

    @prompt_server.routes.post(resolved_settings.route_path)
    async def modal_queue_prompt(request: web.Request) -> web.Response:
        """Handle prompt queue requests that include Modal remote markers."""
        logger.info("Received Modal queue request.")
        json_data: dict[str, Any] | None = None
        workflow: dict[str, Any] | None = None
        remote_node_ids: list[str] = []
        request_modal_gpu: str | None = None
        summary = RewriteSummary()
        preparation_prompt_id: str | None = None
        preparation_cancellation = threading.Event()
        configurator_node_id: str | None = None
        r2_writeback_prompt_id: str | None = None
        prompt_queued = False
        try:
            request_started_at = time.perf_counter()
            json_data = await request.json()
            json_data.setdefault("prompt_id", str(uuid.uuid4()))
            json_data.setdefault("extra_data", {})
            json_data["extra_data"]["prompt_id"] = json_data["prompt_id"]
            if json_data.get("client_id") is not None:
                json_data["extra_data"]["client_id"] = json_data["client_id"]
            client_id = (
                str(json_data.get("client_id")) if json_data.get("client_id") else None
            )
            prompt_id = (
                str(json_data.get("prompt_id")) if json_data.get("prompt_id") else None
            )
            extra_pnginfo = (json_data.get("extra_data") or {}).get(
                "extra_pnginfo"
            ) or {}
            workflow = extra_pnginfo.get("workflow")
            remote_node_ids = sorted(
                requested_remote_node_ids(
                    prompt=json_data.get("prompt", {}),
                    workflow=workflow,
                    settings=resolved_settings,
                )
            )
            configurator_node_id = _remote_execution_configurator_node_id(
                json_data.get("prompt", {})
            )
            if "prompt" in json_data and not remote_node_ids:
                logger.info(
                    "No workflow nodes are marked for Modal execution; forwarding prompt without Modal status or rewrite."
                )
                response = await _queue_prompt_json(prompt_server, json_data)
                logger.info(
                    "Modal queue request completed in %.3fs.",
                    time.perf_counter() - request_started_at,
                )
                return response

            if prompt_id is not None:
                begin_r2_writeback_prompt(prompt_id)
                r2_writeback_prompt_id = prompt_id

            if prompt_id is not None:
                extra_pnginfo[MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY] = prompt_id
                json_data["extra_data"]["extra_pnginfo"] = extra_pnginfo
                logger.debug(
                    "Attached prompt-scoped Modal execution metadata prompt_id=%s.",
                    prompt_id,
                )
                if _set_remote_preparation(
                    prompt_server,
                    prompt_id=prompt_id,
                    prompt=json_data.get("prompt", {}),
                    extra_data=json_data.get("extra_data", {}),
                    cancellation_event=preparation_cancellation,
                ):
                    preparation_prompt_id = prompt_id

            try:
                request_settings = settings_for_modal_gpu(
                    resolved_settings,
                    modal_gpu_from_workflow(workflow, resolved_settings.modal_gpu),
                )
            except ValueError as exc:
                raise ModalPromptValidationError(str(exc)) from exc
            logger.info(
                "Resolved workflow Modal GPU selection gpu=%s prompt_id=%s.",
                request_settings.modal_gpu,
                prompt_id,
            )
            status_modal_gpu = (
                None
                if _prompt_uses_remote_execution_configurator(
                    json_data.get("prompt", {})
                )
                else request_settings.modal_gpu
            )
            request_modal_gpu = status_modal_gpu

            def emit_setup_status(
                message: str,
                current: int | None = None,
                total: int | None = None,
            ) -> None:
                """Forward one queue-time Modal setup update into the websocket stream."""
                if preparation_cancellation.is_set():
                    raise SyncCancelledError(
                        "Remote workflow preparation was cancelled."
                    )
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="setup",
                    client_id=client_id,
                    prompt_id=prompt_id,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=status_modal_gpu,
                    component_node_ids_by_representative=(
                        summary.component_node_ids_by_representative or None
                    ),
                    status_message=message,
                    status_current=current,
                    status_total=total,
                )

            def emit_execution_plan(
                assignments: dict[str, dict[str, Any]],
                configurations: list[dict[str, Any]],
            ) -> None:
                """Publish scheduler choices before capacity acquisition can block."""
                component_nodes = {
                    component_id: list(assignment.get("node_ids", []))
                    for component_id, assignment in assignments.items()
                }
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="setup",
                    client_id=client_id,
                    prompt_id=prompt_id,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=status_modal_gpu,
                    component_node_ids_by_representative=component_nodes,
                    status_message="Remote execution plan ready",
                    remote_execution_assignments=assignments,
                    remote_execution_configurations=configurations,
                )

            def emit_environment_setup_status(
                environment_id: str,
                message: str,
                current: int | None = None,
                total: int | None = None,
            ) -> None:
                """Publish setup progress for one concrete remote environment."""
                if preparation_cancellation.is_set():
                    raise SyncCancelledError(
                        "Remote workflow preparation was cancelled."
                    )
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="setup",
                    client_id=client_id,
                    prompt_id=prompt_id,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=status_modal_gpu,
                    component_node_ids_by_representative=(
                        summary.component_node_ids_by_representative or None
                    ),
                    status_message=message,
                    status_current=current,
                    status_total=total,
                    execution_environment_id=environment_id,
                )

            if "prompt" in json_data:
                emit_setup_status("Preparing remote workflow")
                rewrite_started_at = time.perf_counter()
                occupied_environment_ids = _queued_ssh_environment_ids(
                    prompt_server,
                    excluding_prompt_id=prompt_id,
                )
                if occupied_environment_ids:
                    logger.info(
                        "Queued workflow may reuse SSH capacity after earlier "
                        "prompts finish environments=%s.",
                        sorted(occupied_environment_ids),
                    )
                rewritten_prompt, summary = await rewrite_prompt_for_modal_async(
                    prompt=json_data["prompt"],
                    workflow=workflow,
                    sync_engine=resolved_sync_engine,
                    settings=request_settings,
                    extra_data=json_data.get("extra_data"),
                    status_callback=emit_setup_status,
                    environment_status_callback=emit_environment_setup_status,
                    plan_callback=emit_execution_plan,
                    cancellation_check=preparation_cancellation.is_set,
                    occupied_environment_ids=occupied_environment_ids,
                )
                selected_modal_gpus = _selected_modal_gpus(
                    summary,
                    request_settings.modal_gpu,
                )
                status_modal_gpu = (
                    selected_modal_gpus[0]
                    if len(selected_modal_gpus) == 1
                    else None
                )
                request_modal_gpu = status_modal_gpu
                logger.info(
                    "Modal prompt rewrite finished in %.3fs for %d remote nodes across %d components.",
                    time.perf_counter() - rewrite_started_at,
                    len(summary.remote_node_ids),
                    len(summary.remote_component_ids),
                )
                remote_node_ids = list(summary.remote_node_ids)
                json_data["prompt"] = rewritten_prompt
                if json_data.get("partial_execution_targets"):
                    rewritten_targets = {
                        summary.rewritten_node_id_map.get(str(target), str(target))
                        for target in json_data["partial_execution_targets"]
                    }
                    json_data["partial_execution_targets"] = sorted(rewritten_targets)
                json_data.setdefault("extra_data", {}).setdefault("modal", {})
                json_data["extra_data"]["modal"]["gpu"] = request_settings.modal_gpu
                json_data["extra_data"]["modal"][
                    "remote_node_ids"
                ] = summary.remote_node_ids
                json_data["extra_data"]["modal"][
                    "remote_component_ids"
                ] = summary.remote_component_ids
                json_data["extra_data"]["modal"][
                    "component_dependency_ids_by_representative"
                ] = summary.component_dependency_ids_by_representative
                json_data["extra_data"]["modal"][
                    "component_execution_stages"
                ] = summary.component_execution_stages
                json_data["extra_data"]["modal"][
                    "mapped_component_ids"
                ] = summary.mapped_component_ids
                json_data["extra_data"]["modal"][
                    "estimated_max_parallel_requests"
                ] = summary.estimated_max_parallel_requests
                json_data["extra_data"]["modal"][
                    "max_parallel_requests_upper_bound"
                ] = summary.max_parallel_requests_upper_bound
                json_data["extra_data"]["modal"]["synced_assets"] = [
                    asset.remote_path for asset in summary.synced_assets
                ]
                json_data["extra_data"]["remote_execution"] = {
                    "assignments": _execution_assignments_payload(
                        summary,
                        request_settings,
                    ),
                    "configurations": list(summary.remote_configurations),
                }
                if summary.custom_nodes_bundle is not None:
                    json_data["extra_data"]["modal"][
                        "custom_nodes_bundle"
                    ] = summary.custom_nodes_bundle.remote_path
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="setup",
                    client_id=client_id,
                    prompt_id=prompt_id,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=status_modal_gpu,
                    component_node_ids_by_representative=summary.component_node_ids_by_representative,
                    status_message="Submitting remote workflow",
                )
            response = await _queue_prompt_json(
                prompt_server,
                json_data,
                modal_response_payload=(
                    {
                        "modal_gpu": status_modal_gpu,
                        "remote_execution_configurator_node_id": configurator_node_id,
                        "remote_execution_modal_gpus": selected_modal_gpus,
                        "modal_remote_node_ids": list(summary.remote_node_ids),
                        "modal_sandwiched_local_node_ids": list(
                            summary.sandwiched_local_node_ids
                        ),
                        "modal_parallel_local_branch_node_ids": list(
                            summary.parallel_local_branch_node_ids
                        ),
                        "remote_execution_assignments": _execution_assignments_payload(
                            summary,
                            request_settings,
                        ),
                        "remote_execution_configurations": list(
                            summary.remote_configurations
                        ),
                        "modal_components": [
                            {
                                "representative_node_id": representative_node_id,
                                "node_ids": list(component_node_ids),
                            }
                            for representative_node_id, component_node_ids in sorted(
                                summary.component_node_ids_by_representative.items()
                            )
                        ],
                    }
                    if summary.remote_node_ids
                    else None
                ),
            )
            prompt_queued = response.status < 400
            logger.info(
                "Modal queue request completed in %.3fs.",
                time.perf_counter() - request_started_at,
            )
            return response
        except SyncCancelledError as exc:
            logger.info("Remote workflow preparation cancelled: %s", exc)
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="execution_interrupted",
                    client_id=str(json_data.get("client_id"))
                    if json_data.get("client_id")
                    else None,
                    prompt_id=str(json_data.get("prompt_id"))
                    if json_data.get("prompt_id")
                    else None,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=request_modal_gpu,
                    status_message=str(exc),
                )
            return web.json_response(
                {"error": str(exc), "node_errors": [], "cancelled": True},
                status=409,
            )
        except FileNotFoundError as exc:
            logger.exception("Modal asset sync failed.")
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="error",
                    client_id=str(json_data.get("client_id"))
                    if json_data.get("client_id")
                    else None,
                    prompt_id=str(json_data.get("prompt_id"))
                    if json_data.get("prompt_id")
                    else None,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=request_modal_gpu,
                    error_message=str(exc),
                )
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
        except ModalPromptValidationError as exc:
            logger.exception("Modal prompt validation failed.")
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="error",
                    client_id=str(json_data.get("client_id"))
                    if json_data.get("client_id")
                    else None,
                    prompt_id=str(json_data.get("prompt_id"))
                    if json_data.get("prompt_id")
                    else None,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=request_modal_gpu,
                    error_message=str(exc),
                )
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
        except Exception as exc:
            logger.exception("Modal queue handler failed.")
            if json_data is not None:
                _emit_modal_status(
                    prompt_server=prompt_server,
                    phase="error",
                    client_id=str(json_data.get("client_id"))
                    if json_data.get("client_id")
                    else None,
                    prompt_id=str(json_data.get("prompt_id"))
                    if json_data.get("prompt_id")
                    else None,
                    node_ids=remote_node_ids,
                    configurator_node_id=configurator_node_id,
                    modal_gpu=request_modal_gpu,
                    error_message=str(exc),
                )
            return web.json_response({"error": str(exc), "node_errors": []}, status=500)
        finally:
            if preparation_prompt_id is not None:
                _clear_remote_preparation(prompt_server, preparation_prompt_id)
            if r2_writeback_prompt_id is not None and not prompt_queued:
                finish_r2_writeback_prompt(r2_writeback_prompt_id)

    _ROUTE_REGISTERED = True
    logger.info(
        "Registered Modal queue route at %s, analysis route at %s, progress state route at %s, container status route at %s, cache deletion route at %s, and volume deletion route at %s",
        resolved_settings.route_path,
        analysis_route_path,
        progress_state_route_path,
        container_status_route_path,
        delete_caches_route_path,
        delete_volume_route_path,
    )
