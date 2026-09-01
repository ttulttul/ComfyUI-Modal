"""Queue-time remote prompt analysis, asset preparation, and proxy rewriting."""

from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

if __package__:
    from .component_planning import (
        _build_component_plans,
        _implicitly_mapped_boundary_output_sources,
        _mark_payload_scheduler_list_outputs,
        _mark_remote_to_remote_session_boundaries,
        _workflow_visible_remote_node_ids,
        validate_remote_component_transport_compatibility,
    )
    from .execution_assignment_runtime import (
        _configured_provider_metadata,
        _ensure_remote_sync_backend,
        _execution_location_for_assignment,
        _ssh_sync_engine,
        _stamp_execution_assignment,
        _workflow_r2_cache,
    )
    from .execution_environments import ExecutionAssignment, ExecutionProvider
    from .execution_resource_estimates import _component_execution_signature
    from .execution_scheduling import (
        _configured_ssh_hosts,
        _plan_component_execution,
    )
    from .prompt_affinity_planning import (
        _configure_local_gap_keepalive_payloads,
        _configure_speculative_affinity_prewarm_payloads,
        _parallelize_non_returning_local_branches,
    )
    from .prompt_diagnostics import _log_modal_rewritten_prompt_diagnostics
    from .prompt_payload_metadata import _attach_resolved_llm_profiles
    from .prompt_payload_building import _build_component_payload
    from .prompt_rewrite import (
        _attach_modal_artifact_finalizer,
        _component_uploaded_volume_paths,
        _deduplicate_synced_assets,
        _rewrite_component_into_proxy,
        _sync_component_prompt_inputs,
    )
    from .r2_cache import R2CacheError
    from .remote_graph_analysis import (
        _component_dependency_graph,
        _component_execution_stages,
        _estimated_stage_parallelism,
        _expand_remote_node_ids_for_non_transportable_inputs,
        _expand_remote_node_ids_for_terminal_video_sinks,
        _sandwiched_local_node_ids,
        requested_remote_node_ids,
    )
    from .remote_plan_types import (
        ComponentExecutionPlan,
        ModalPromptValidationError,
        RemoteComponentPlan,
        RewriteSummary,
        _EnvironmentAssetPreparationResult,
    )
    from .session_state import RemoteSessionHandle
    from .settings import ModalSyncSettings, get_settings
    from .sync_engine import (
        ModalAssetSyncEngine,
        SyncCancelledError,
        SyncedAsset,
    )
    from .vast_service import VastService
else:  # pragma: no cover - flat import inside the Modal container.
    from component_planning import (
        _build_component_plans,
        _implicitly_mapped_boundary_output_sources,
        _mark_payload_scheduler_list_outputs,
        _mark_remote_to_remote_session_boundaries,
        _workflow_visible_remote_node_ids,
        validate_remote_component_transport_compatibility,
    )
    from execution_assignment_runtime import (
        _configured_provider_metadata,
        _ensure_remote_sync_backend,
        _execution_location_for_assignment,
        _ssh_sync_engine,
        _stamp_execution_assignment,
        _workflow_r2_cache,
    )
    from execution_environments import ExecutionAssignment, ExecutionProvider
    from execution_resource_estimates import _component_execution_signature
    from execution_scheduling import (
        _configured_ssh_hosts,
        _plan_component_execution,
    )
    from prompt_affinity_planning import (
        _configure_local_gap_keepalive_payloads,
        _configure_speculative_affinity_prewarm_payloads,
        _parallelize_non_returning_local_branches,
    )
    from prompt_diagnostics import _log_modal_rewritten_prompt_diagnostics
    from prompt_payload_metadata import _attach_resolved_llm_profiles
    from prompt_payload_building import _build_component_payload
    from prompt_rewrite import (
        _attach_modal_artifact_finalizer,
        _component_uploaded_volume_paths,
        _deduplicate_synced_assets,
        _rewrite_component_into_proxy,
        _sync_component_prompt_inputs,
    )
    from r2_cache import R2CacheError
    from remote_graph_analysis import (
        _component_dependency_graph,
        _component_execution_stages,
        _estimated_stage_parallelism,
        _expand_remote_node_ids_for_non_transportable_inputs,
        _expand_remote_node_ids_for_terminal_video_sinks,
        _sandwiched_local_node_ids,
        requested_remote_node_ids,
    )
    from remote_plan_types import (
        ComponentExecutionPlan,
        ModalPromptValidationError,
        RemoteComponentPlan,
        RewriteSummary,
        _EnvironmentAssetPreparationResult,
    )
    from session_state import RemoteSessionHandle
    from settings import ModalSyncSettings, get_settings
    from sync_engine import (
        ModalAssetSyncEngine,
        SyncCancelledError,
        SyncedAsset,
    )
    from vast_service import VastService

logger = logging.getLogger(__name__)

SetupStatusCallback = Callable[[str, int | None, int | None], None]
EnvironmentSetupStatusCallback = Callable[
    [str, str, int | None, int | None], None
]
ExecutionPlanStatusCallback = Callable[
    [dict[str, dict[str, Any]], list[dict[str, Any]]], None
]

_REMOTE_ASSET_PREPARATION_MAX_WORKERS = 8


def _get_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


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
        finalize_manifest = getattr(sync_engine, "finalize_manifest", None)
        asset_manifest_id = (
            str(finalize_manifest()).strip()
            if callable(finalize_manifest)
            else None
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
        asset_manifest_id=asset_manifest_id,
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


@dataclass
class _PromptRewriteState:
    """Hold mutable queue-time state across prompt rewrite stages."""

    prompt: dict[str, Any]
    workflow: dict[str, Any] | None
    rewritten_prompt: dict[str, Any]
    settings: ModalSyncSettings
    nodes_module: Any
    sync_engine: ModalAssetSyncEngine
    extra_data: dict[str, Any] | None
    summary: RewriteSummary
    components: list[RemoteComponentPlan] = field(default_factory=list)
    execution_plan: ComponentExecutionPlan | None = None
    assignments: dict[str, ExecutionAssignment] = field(default_factory=dict)
    remote_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    vast_service: VastService | None = None
    vast_leases: dict[str, Any] = field(default_factory=dict)
    workflow_r2_cache: Any | None = None
    ssh_hosts: dict[str, Any] = field(default_factory=dict)
    sync_engines: dict[str, ModalAssetSyncEngine] = field(default_factory=dict)
    synced_component_prompts: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    synced_assets_by_component: dict[str, list[SyncedAsset]] = field(
        default_factory=dict
    )
    mapped_proxy_component_ids: set[str] = field(default_factory=set)
    asset_manifest_ids_by_environment: dict[str, str] = field(default_factory=dict)


def _analyze_remote_components(
    state: _PromptRewriteState,
    remote_node_ids: set[str],
) -> None:
    """Expand remote selection, build components, and validate transport."""
    expanded_node_ids, _ = _expand_remote_node_ids_for_non_transportable_inputs(
        prompt=state.rewritten_prompt,
        remote_node_ids=remote_node_ids,
        nodes_module=state.nodes_module,
    )
    expanded_node_ids = _expand_remote_node_ids_for_terminal_video_sinks(
        prompt=state.rewritten_prompt,
        remote_node_ids=expanded_node_ids,
        nodes_module=state.nodes_module,
    )
    state.summary.remote_node_ids = sorted(expanded_node_ids)
    state.summary.sandwiched_local_node_ids = sorted(
        _sandwiched_local_node_ids(state.rewritten_prompt, expanded_node_ids)
    )
    if state.summary.sandwiched_local_node_ids:
        logger.warning(
            "Detected local nodes sandwiched between remote graph regions; "
            "these nodes may force additional remote phases: %s",
            state.summary.sandwiched_local_node_ids,
        )
    state.components = _build_component_plans(
        state.rewritten_prompt,
        expanded_node_ids,
        state.nodes_module,
    )
    validate_remote_component_transport_compatibility(
        prompt=state.rewritten_prompt,
        components=state.components,
        nodes_module=state.nodes_module,
    )


def _remote_sessions_for_components(
    state: _PromptRewriteState,
    session_component_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Create prompt-scoped sessions for components with remote-to-remote links."""
    prompt_id = (state.extra_data or {}).get("prompt_id")
    sessions = {
        component.representative_node_id: RemoteSessionHandle(
            session_id=uuid.uuid4().hex,
            prompt_id=(str(prompt_id) if prompt_id is not None else None),
            owner_component_id=component.representative_node_id,
        ).to_payload()
        for component in state.components
        if component.representative_node_id in session_component_ids
    }
    if sessions:
        logger.info(
            "Enabled environment-local remote references for components=%s.",
            sorted(sessions),
        )
    return sessions


def _plan_remote_component_execution(
    state: _PromptRewriteState,
    *,
    status_callback: SetupStatusCallback | None,
    environment_status_callback: EnvironmentSetupStatusCallback | None,
    plan_callback: ExecutionPlanStatusCallback | None,
    occupied_environment_ids: frozenset[str],
) -> set[str]:
    """Plan providers, record locations, and mark session boundaries."""
    execution_plan = _plan_component_execution(
        components=state.components,
        prompt=state.rewritten_prompt,
        workflow=state.workflow,
        settings=state.settings,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
        plan_callback=plan_callback,
        occupied_environment_ids=occupied_environment_ids,
    )
    state.execution_plan = execution_plan
    state.assignments = execution_plan.assignments
    if execution_plan.configuration_set is not None:
        state.summary.remote_configurations = list(execution_plan.safe_configurations)
    for assignment in state.assignments.values():
        execution_location = _execution_location_for_assignment(
            assignment,
            execution_plan.ssh_hosts_by_id,
            execution_plan.vast_leases_by_environment,
        )
        if execution_location:
            state.summary.execution_locations_by_environment[
                assignment.environment_id
            ] = execution_location
    session_component_ids = _mark_remote_to_remote_session_boundaries(
        state.rewritten_prompt,
        state.components,
        state.nodes_module,
        state.assignments,
    )
    state.remote_sessions = _remote_sessions_for_components(
        state, session_component_ids
    )
    state.summary.execution_assignments_by_representative = dict(state.assignments)
    return session_component_ids


def _resolve_vast_service(state: _PromptRewriteState) -> None:
    """Resolve the planner-provided or registry-backed Vast controller state."""
    assert state.execution_plan is not None
    state.vast_service = state.execution_plan.vast_service
    state.vast_leases = dict(state.execution_plan.vast_leases_by_environment)
    if state.vast_service is not None or not any(
        assignment.provider is ExecutionProvider.VAST
        for assignment in state.assignments.values()
    ):
        return
    try:
        runtime_fingerprints = {
            lease.runtime_fingerprint for lease in state.vast_leases.values()
        }
        worker_images = {
            lease.worker_image
            for lease in state.vast_leases.values()
            if lease.worker_image is not None
        }
        runtime_fingerprint = (
            next(iter(runtime_fingerprints))
            if len(runtime_fingerprints) == 1
            else None
        )
        worker_image = next(iter(worker_images)) if len(worker_images) == 1 else None
        state.vast_service = VastService.from_environment(
            state.settings,
            repo_root=Path(__file__).resolve().parent,
            runtime_fingerprint=runtime_fingerprint,
            worker_image=worker_image,
        )
        state.vast_leases = {
            assignment.environment_id: state.vast_service.lease_for_environment_id(
                assignment.environment_id
            )
            for assignment in state.assignments.values()
            if assignment.provider is ExecutionProvider.VAST
        }
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise ModalPromptValidationError(
            f"Unable to resolve the acquired Vast.ai lease: {exc}"
        ) from exc


def _resolve_execution_resources(state: _PromptRewriteState) -> None:
    """Resolve Vast, R2, and SSH resources needed after placement."""
    assert state.execution_plan is not None
    _resolve_vast_service(state)
    state.workflow_r2_cache = (
        _workflow_r2_cache(state.execution_plan.configuration_set)
        if any(
            assignment.provider is not ExecutionProvider.MODAL
            for assignment in state.assignments.values()
        )
        else None
    )
    if state.workflow_r2_cache is not None and state.vast_service is not None:
        state.vast_service.r2_cache = state.workflow_r2_cache
    state.ssh_hosts = dict(state.execution_plan.ssh_hosts_by_id)
    if not state.ssh_hosts:
        state.ssh_hosts = {
            host.environment_id: host for host in _configured_ssh_hosts(state.settings)
        }


def _assign_component_worker_indices(
    state: _PromptRewriteState,
    session_component_ids: set[str],
) -> None:
    """Assign deterministic worker slots while reserving session-affine slot zero."""
    assert state.execution_plan is not None
    session_environment_ids = {
        state.assignments[component_id].environment_id
        for component_id in session_component_ids
    }
    worker_counts: dict[str, int] = defaultdict(int)
    for component in state.components:
        component_id = component.representative_node_id
        assignment = state.assignments[component_id]
        if assignment.provider in {ExecutionProvider.MODAL, ExecutionProvider.VAST}:
            worker_index = 0
        elif state.execution_plan.configuration_set is not None:
            worker_index = assignment.capacity_slot_index
            logger.info(
                "Assigned workflow-configured remote component=%s environment=%s worker_index=%d.",
                component_id,
                assignment.environment_id,
                worker_index,
            )
        else:
            host = state.ssh_hosts.get(assignment.environment_id)
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
                    worker_counts[assignment.environment_id] % available_workers
                )
                worker_counts[assignment.environment_id] += 1
            logger.info(
                "Assigned remote component=%s environment=%s worker_index=%d.",
                component_id,
                assignment.environment_id,
                worker_index,
            )
        state.summary.execution_worker_indices_by_representative[component_id] = (
            worker_index
        )


def _sync_engine_for_assignment(
    state: _PromptRewriteState,
    assignment: ExecutionAssignment,
    cancellation_check: Callable[[], bool] | None,
) -> ModalAssetSyncEngine:
    """Return the provider-specific sync engine for one assignment."""
    if assignment.provider is ExecutionProvider.MODAL:
        _ensure_remote_sync_backend(state.settings, state.sync_engine)
        return state.sync_engine
    if assignment.provider is ExecutionProvider.VAST:
        if state.vast_service is None:
            raise ModalPromptValidationError(
                "Vast.ai assignment has no active controller service."
            )
        lease = state.vast_leases[assignment.environment_id]
        sync_engine = state.vast_service.sync_engine(lease)
        sync_engine.cancellation_check = cancellation_check
        return sync_engine
    if assignment.provider is ExecutionProvider.SUBROSA:
        if __package__:
            from .remote_configurations import SubrosaRemoteConfiguration
            from .subrosa_sync import subrosa_asset_sync_engine
        else:  # pragma: no cover - flat Modal-container import.
            from remote_configurations import SubrosaRemoteConfiguration
            from subrosa_sync import subrosa_asset_sync_engine

        if state.execution_plan is None or assignment.configuration_id is None:
            raise ModalPromptValidationError(
                "Subrosa assignment has no compiled configuration."
            )
        configuration = state.execution_plan.configurations_by_id.get(
            assignment.configuration_id
        )
        if not isinstance(configuration, SubrosaRemoteConfiguration):
            raise ModalPromptValidationError(
                "Subrosa assignment references an invalid configuration."
            )
        return subrosa_asset_sync_engine(
            state.settings,
            cancellation_check,
            relay_url=configuration.relay_url,
            credential_id=configuration.credential_id,
        )
    host = state.ssh_hosts.get(assignment.environment_id)
    if host is None:
        raise ModalPromptValidationError(
            f"Assigned SSH execution environment {assignment.environment_id!r} is no longer configured."
        )
    return _ssh_sync_engine(
        host=host,
        settings=state.settings,
        r2_cache=state.workflow_r2_cache,
    )


def _build_environment_sync_engines(
    state: _PromptRewriteState,
    cancellation_check: Callable[[], bool] | None,
) -> None:
    """Build one asset synchronization engine per selected environment."""
    for assignment in state.assignments.values():
        if assignment.environment_id in state.sync_engines:
            continue
        state.sync_engines[assignment.environment_id] = _sync_engine_for_assignment(
            state, assignment, cancellation_check
        )


def _record_environment_preparations(
    state: _PromptRewriteState,
    preparations: dict[str, _EnvironmentAssetPreparationResult],
) -> None:
    """Record custom-node bundles, synced component prompts, and unique assets."""
    state.asset_manifest_ids_by_environment = {
        environment_id: preparation.asset_manifest_id
        for environment_id, preparation in preparations.items()
        if preparation.asset_manifest_id
    }
    if state.settings.sync_custom_nodes:
        state.summary.custom_nodes_bundles_by_environment = {
            environment_id: preparation.custom_nodes_bundle
            for environment_id, preparation in preparations.items()
        }
        modal_bundle = next(
            (
                state.summary.custom_nodes_bundles_by_environment[
                    assignment.environment_id
                ]
                for assignment in state.assignments.values()
                if assignment.provider is ExecutionProvider.MODAL
            ),
            None,
        )
        state.summary.custom_nodes_bundle = modal_bundle or next(
            iter(state.summary.custom_nodes_bundles_by_environment.values()), None
        )
    else:
        logger.info(
            "Skipping custom_nodes bundle sync because sync is disabled for execution_mode=%s.",
            state.settings.execution_mode,
        )
    for component in state.components:
        component_id = component.representative_node_id
        assignment = state.assignments[component_id]
        preparation = preparations[assignment.environment_id]
        state.synced_component_prompts[component_id] = (
            preparation.component_prompts[component_id]
        )
        assets = preparation.assets_by_component_id[component_id]
        state.synced_assets_by_component[component_id] = assets
        state.summary.synced_assets.extend(assets)
    state.summary.synced_assets = _deduplicate_synced_assets(
        state.summary.synced_assets
    )


def _prepare_rewrite_assets(
    state: _PromptRewriteState,
    *,
    status_callback: SetupStatusCallback | None,
    environment_status_callback: EnvironmentSetupStatusCallback | None,
) -> None:
    """Prepare all environment assets and record their component-scoped results."""
    if __package__:
        from .subrosa_sync import SubrosaAssetSyncError
    else:  # pragma: no cover - flat Modal-container import.
        from subrosa_sync import SubrosaAssetSyncError

    if status_callback is not None:
        status_callback("Preparing assets for remote execution", None, None)
    try:
        preparations = _prepare_remote_environment_assets(
            components=state.components,
            assignments_by_component_id=state.assignments,
            sync_engines_by_environment=state.sync_engines,
            rewritten_prompt=state.rewritten_prompt,
            sync_custom_nodes=state.settings.sync_custom_nodes,
            status_callback=status_callback,
            environment_status_callback=environment_status_callback,
        )
    except (R2CacheError, SubrosaAssetSyncError) as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    _record_environment_preparations(state, preparations)


def _resolve_modal_volume_reload(state: _PromptRewriteState) -> None:
    """Compute request-wide Modal volume reload state after asset preparation."""
    modal_environment_ids = {
        assignment.environment_id
        for assignment in state.assignments.values()
        if assignment.provider is ExecutionProvider.MODAL
    }
    modal_component_ids = {
        component.representative_node_id
        for component in state.components
        if state.assignments[component.representative_node_id].provider
        is ExecutionProvider.MODAL
    }
    requires_reload = any(
        asset.uploaded
        for component_id, assets in state.synced_assets_by_component.items()
        if component_id in modal_component_ids
        for asset in assets
    ) or any(
        bundle is not None and bundle.uploaded
        for environment_id, bundle in (
            state.summary.custom_nodes_bundles_by_environment.items()
        )
        if environment_id in modal_environment_ids
    )
    state.summary.requires_volume_reload = requires_reload
    state.summary.volume_reload_marker = uuid.uuid4().hex if requires_reload else None
    state.summary.uploaded_volume_paths = [
        asset.remote_path for asset in state.summary.synced_assets if asset.uploaded
    ]
    logger.info(
        "Resolved request-wide Modal volume reload requirement: requires_volume_reload=%s volume_reload_marker=%s synced_assets=%d custom_nodes_uploaded=%s",
        requires_reload,
        state.summary.volume_reload_marker,
        len(state.summary.synced_assets),
        bool(
            state.summary.custom_nodes_bundle is not None
            and state.summary.custom_nodes_bundle.uploaded
        ),
    )


def _build_stamped_component_payload(
    state: _PromptRewriteState,
    component: RemoteComponentPlan,
) -> tuple[dict[str, Any], set[Any]]:
    """Build one component payload and stamp its provider assignment metadata."""
    assert state.execution_plan is not None
    component_id = component.representative_node_id
    assignment = state.assignments[component_id]
    custom_nodes_bundle = state.summary.custom_nodes_bundles_by_environment.get(
        assignment.environment_id
    )
    uploaded_paths = _component_uploaded_volume_paths(
        component_prompt=state.synced_component_prompts[component_id],
        synced_assets=state.synced_assets_by_component[component_id],
        custom_nodes_bundle=custom_nodes_bundle,
    )
    payload = _build_component_payload(
        component=component,
        component_prompt=state.synced_component_prompts[component_id],
        signature_prompt=state.prompt,
        extra_data=state.extra_data,
        settings=state.settings,
        requires_volume_reload=(
            assignment.provider is ExecutionProvider.MODAL and bool(uploaded_paths)
        ),
        volume_reload_marker=state.summary.volume_reload_marker,
        custom_nodes_bundle=custom_nodes_bundle,
        uploaded_volume_paths=uploaded_paths,
        terminate_container_on_error=state.settings.terminate_container_on_error,
        nodes_module=state.nodes_module,
        remote_session=state.remote_sessions.get(component_id),
    )
    _attach_resolved_llm_profiles(
        payload, state.execution_plan.resolved_llm_profiles, state.settings
    )
    _stamp_execution_assignment(
        payload,
        assignment,
        state.summary.execution_worker_indices_by_representative.get(component_id, 0),
        _component_execution_signature(component, state.prompt),
        _execution_location_for_assignment(
            assignment, state.ssh_hosts, state.vast_leases
        ),
        _configured_provider_metadata(
            execution_plan=state.execution_plan,
            assignment=assignment,
            vast_leases_by_environment=state.vast_leases,
        ),
    )
    asset_manifest_id = state.asset_manifest_ids_by_environment.get(
        assignment.environment_id
    )
    if asset_manifest_id is not None:
        payload["asset_manifest_id"] = asset_manifest_id
    implicitly_mapped_sources = _implicitly_mapped_boundary_output_sources(
        component=component,
        original_prompt=state.prompt,
        rewritten_prompt=state.rewritten_prompt,
        nodes_module=state.nodes_module,
    )
    _mark_payload_scheduler_list_outputs(payload, implicitly_mapped_sources)
    return payload, implicitly_mapped_sources


def _record_hybrid_component_summary(
    state: _PromptRewriteState,
    component: RemoteComponentPlan,
    proxy_node_ids: list[str],
) -> None:
    """Record node mappings for paired static and mapped proxies."""
    static_proxy_node_id, mapped_proxy_node_id = proxy_node_ids
    state.summary.remote_component_ids.extend(proxy_node_ids)
    for proxy_node_id, component_node_ids in (
        (static_proxy_node_id, component.static_node_ids),
        (mapped_proxy_node_id, component.mapped_node_ids),
    ):
        visible_node_ids = _workflow_visible_remote_node_ids(component_node_ids)
        state.summary.component_node_ids_by_representative[proxy_node_id] = (
            visible_node_ids
        )
        for node_id in visible_node_ids:
            state.summary.rewritten_node_id_map[node_id] = proxy_node_id
    state.mapped_proxy_component_ids.add(mapped_proxy_node_id)


def _record_ordered_component_summary(
    state: _PromptRewriteState,
    component: RemoteComponentPlan,
    proxy_node_ids: list[str],
    split_payloads: list[Any],
) -> None:
    """Record node mappings for an ordered proxy sequence."""
    state.summary.remote_component_ids.extend(proxy_node_ids)
    mapped_node_ids = set(component.mapped_node_ids)
    for phase_payload in split_payloads:
        proxy_node_id = str(phase_payload["component_id"])
        component_node_ids = [
            str(node_id) for node_id in phase_payload["component_node_ids"]
        ]
        visible_node_ids = _workflow_visible_remote_node_ids(component_node_ids)
        state.summary.component_node_ids_by_representative[proxy_node_id] = (
            visible_node_ids
        )
        for node_id in visible_node_ids:
            state.summary.rewritten_node_id_map[node_id] = proxy_node_id
        if mapped_node_ids.intersection(component_node_ids):
            state.mapped_proxy_component_ids.add(proxy_node_id)


def _record_single_component_summary(
    state: _PromptRewriteState,
    component: RemoteComponentPlan,
    proxy_node_id: str,
    implicitly_mapped_sources: set[Any],
) -> None:
    """Record node mappings for one ordinary component proxy."""
    state.summary.remote_component_ids.append(proxy_node_id)
    visible_node_ids = _workflow_visible_remote_node_ids(component.node_ids)
    state.summary.component_node_ids_by_representative[proxy_node_id] = (
        visible_node_ids
    )
    for node_id in visible_node_ids:
        state.summary.rewritten_node_id_map[node_id] = proxy_node_id
    if component.mapped_boundary_input_name or implicitly_mapped_sources:
        state.mapped_proxy_component_ids.add(proxy_node_id)


def _record_component_proxy_summary(
    state: _PromptRewriteState,
    component: RemoteComponentPlan,
    payload: dict[str, Any],
    proxy_node_ids: list[str],
    implicitly_mapped_sources: set[Any],
) -> None:
    """Record assignment and workflow-node metadata for rewritten proxies."""
    assignment = state.assignments[component.representative_node_id]
    worker_index = state.summary.execution_worker_indices_by_representative.get(
        component.representative_node_id, 0
    )
    for proxy_node_id in proxy_node_ids:
        state.summary.execution_assignments_by_representative[proxy_node_id] = (
            assignment
        )
        state.summary.execution_worker_indices_by_representative[proxy_node_id] = (
            worker_index
        )
    split_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_payloads, dict):
        _record_hybrid_component_summary(state, component, proxy_node_ids)
    elif isinstance(split_payloads, list):
        _record_ordered_component_summary(
            state, component, proxy_node_ids, split_payloads
        )
    else:
        _record_single_component_summary(
            state, component, proxy_node_ids[0], implicitly_mapped_sources
        )


def _rewrite_remote_components(state: _PromptRewriteState) -> None:
    """Build, stamp, and rewrite every planned remote component."""
    for component in state.components:
        logger.info(
            "Rewriting remote component %s covering nodes %s.",
            component.representative_node_id,
            component.node_ids,
        )
        payload, implicitly_mapped_sources = _build_stamped_component_payload(
            state, component
        )
        proxy_node_ids = _rewrite_component_into_proxy(
            component=component,
            rewritten_prompt=state.rewritten_prompt,
            payload=payload,
            nodes_module=state.nodes_module,
        )
        _record_component_proxy_summary(
            state,
            component,
            payload,
            proxy_node_ids,
            implicitly_mapped_sources,
        )


def _parallelism_upper_bound(
    state: _PromptRewriteState,
    estimated_parallelism: int,
) -> int:
    """Return the configured or runtime upper bound for remote parallelism."""
    assert state.execution_plan is not None
    if state.execution_plan.configuration_set is not None:
        configured_capacity = sum(
            configuration.capacity_limit
            for configuration in (
                state.execution_plan.configuration_set.capacity_configurations
            )
        )
        return min(estimated_parallelism, configured_capacity)
    if state.settings.max_containers is not None:
        return min(estimated_parallelism, state.settings.max_containers)
    return estimated_parallelism


def _finalize_rewritten_prompt(state: _PromptRewriteState) -> None:
    """Attach affinity/finalizer nodes and calculate dependency summary metadata."""
    _configure_local_gap_keepalive_payloads(
        rewritten_prompt=state.rewritten_prompt,
        remote_component_ids=state.summary.remote_component_ids,
        sandwiched_local_node_ids=set(state.summary.sandwiched_local_node_ids),
    )
    state.summary.parallel_local_branch_node_ids = (
        _parallelize_non_returning_local_branches(
            rewritten_prompt=state.rewritten_prompt,
            remote_component_ids=state.summary.remote_component_ids,
            nodes_module=state.nodes_module,
        )
    )
    state.summary.artifact_finalizer_node_id = _attach_modal_artifact_finalizer(
        rewritten_prompt=state.rewritten_prompt,
        remote_component_ids=state.summary.remote_component_ids,
        nodes_module=state.nodes_module,
    )
    proxy_groups = {
        component_id: {component_id}
        for component_id in state.summary.remote_component_ids
    }
    _, dependency_edges, _ = _component_dependency_graph(
        state.rewritten_prompt, proxy_groups
    )
    execution_stages = _component_execution_stages(
        state.rewritten_prompt, proxy_groups
    )
    _configure_speculative_affinity_prewarm_payloads(
        rewritten_prompt=state.rewritten_prompt,
        execution_stages=execution_stages,
    )
    state.summary.component_dependency_ids_by_representative = {
        component_id: sorted(
            upstream_id
            for upstream_id, downstream_ids in dependency_edges.items()
            if component_id in downstream_ids
        )
        for component_id in sorted(proxy_groups)
    }
    state.summary.component_execution_stages = [
        list(stage) for stage in execution_stages
    ]
    state.summary.mapped_component_ids = sorted(state.mapped_proxy_component_ids)
    state.summary.estimated_max_parallel_requests = _estimated_stage_parallelism(
        execution_stages,
        state.mapped_proxy_component_ids,
        mapped_component_weight=1,
    )
    state.summary.max_parallel_requests_upper_bound = _parallelism_upper_bound(
        state, state.summary.estimated_max_parallel_requests
    )


def _log_rewrite_completion(state: _PromptRewriteState) -> None:
    """Log final parallelism and diagnostics for one rewritten prompt."""
    logger.info(
        "Estimated remote parallelism after proxy rewrite: known_max_parallel_requests=%d max_parallel_requests_upper_bound=%s mapped_components=%s execution_stages=%s",
        state.summary.estimated_max_parallel_requests,
        state.summary.max_parallel_requests_upper_bound,
        state.summary.mapped_component_ids,
        state.summary.component_execution_stages,
    )
    _log_modal_rewritten_prompt_diagnostics(
        prompt_id=(
            str(state.extra_data.get("prompt_id"))
            if isinstance(state.extra_data, Mapping)
            and state.extra_data.get("prompt_id")
            else None
        ),
        prompt=state.rewritten_prompt,
        summary=state.summary,
        reason="post_rewrite",
    )


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
    """Rewrite connected remote components into provider-backed proxy nodes."""
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

    state = _PromptRewriteState(
        prompt=prompt,
        workflow=workflow,
        rewritten_prompt=copy.deepcopy(prompt),
        settings=resolved_settings,
        nodes_module=nodes_module or _get_nodes_module(),
        sync_engine=sync_engine
        or ModalAssetSyncEngine.from_environment(resolved_settings),
        extra_data=extra_data,
        summary=summary,
    )
    _analyze_remote_components(state, remote_node_ids)
    session_component_ids = _plan_remote_component_execution(
        state,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
        plan_callback=plan_callback,
        occupied_environment_ids=occupied_environment_ids,
    )
    _resolve_execution_resources(state)
    _assign_component_worker_indices(state, session_component_ids)
    _build_environment_sync_engines(state, cancellation_check)
    _prepare_rewrite_assets(
        state,
        status_callback=status_callback,
        environment_status_callback=environment_status_callback,
    )
    _resolve_modal_volume_reload(state)
    _rewrite_remote_components(state)
    _finalize_rewritten_prompt(state)
    _log_rewrite_completion(state)
    return state.rewritten_prompt, state.summary

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
