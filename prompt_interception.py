"""Queue-time remote prompt analysis, asset preparation, and proxy rewriting."""

from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    from .prompt_rewrite import (
        _attach_modal_artifact_finalizer,
        _build_component_payload,
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
    from prompt_rewrite import (
        _attach_modal_artifact_finalizer,
        _build_component_payload,
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


