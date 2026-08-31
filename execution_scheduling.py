"""Provider-neutral execution planning, capacity acquisition, and assignment metadata."""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__:
    from .execution_assignment_runtime import (
        _configured_provider_metadata,
        _ensure_remote_sync_backend,
        _execution_location_for_assignment,
        _ssh_hostname,
        _ssh_sync_engine,
        _stamp_execution_assignment,
        _vast_provider_metadata,
        _workflow_r2_cache,
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
    from .execution_plan_reporting import (
        _assignment_hardware_payload,
        _cached_r2_storage_usage,
        _configuration_field,
        _configuration_host,
        _planned_execution_assignments_payload,
        _r2_storage_from_usage_payload,
        _refresh_r2_storage_usage,
        _safe_remote_configuration_payload,
    )
    from .execution_resource_estimates import (
        _component_execution_signature,
        _component_memory_estimate,
        _component_model_asset_sizes,
        _component_profile_memory_estimate,
        _component_required_provider,
        _is_additive_model_node,
        _iter_prompt_string_values,
        _prompt_llm_model_references,
        _resolve_prompt_llm_profiles,
    )
    from .modal_hardware import (
        _MODAL_GPU_COST_USD_PER_SECOND,
        _MODAL_GPU_VRAM_GB,
    )
    from .remote_configuration_nodes import compile_remote_configuration_set
    from .remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        SshRemoteConfiguration,
        SubrosaRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from .remote_graph_analysis import _component_execution_stages
    from .remote_hosts import RemoteHostRegistry, SshHostConfig
    from .remote_plan_types import (
        ComponentExecutionPlan,
        ComponentMemoryEstimate,
        ModalPromptValidationError,
        RemoteComponentPlan,
    )
    from .settings import (
        ModalSyncSettings,
        discover_comfyui_user_directory,
        settings_for_modal_gpu,
    )
    from .ssh_docker import SshDockerController
    from .sync_engine import SyncCancelledError
    from .vast_config_node import extract_vast_profiles
    from .vast_service import VastProfileQuote, VastSearchRequirements, VastService
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_assignment_runtime import (
        _configured_provider_metadata,
        _ensure_remote_sync_backend,
        _execution_location_for_assignment,
        _ssh_hostname,
        _ssh_sync_engine,
        _stamp_execution_assignment,
        _vast_provider_metadata,
        _workflow_r2_cache,
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
    from execution_plan_reporting import (
        _assignment_hardware_payload,
        _cached_r2_storage_usage,
        _configuration_field,
        _configuration_host,
        _planned_execution_assignments_payload,
        _r2_storage_from_usage_payload,
        _refresh_r2_storage_usage,
        _safe_remote_configuration_payload,
    )
    from execution_resource_estimates import (
        _component_execution_signature,
        _component_memory_estimate,
        _component_model_asset_sizes,
        _component_profile_memory_estimate,
        _component_required_provider,
        _is_additive_model_node,
        _iter_prompt_string_values,
        _prompt_llm_model_references,
        _resolve_prompt_llm_profiles,
    )
    from modal_hardware import (
        _MODAL_GPU_COST_USD_PER_SECOND,
        _MODAL_GPU_VRAM_GB,
    )
    from remote_configuration_nodes import compile_remote_configuration_set
    from remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        SshRemoteConfiguration,
        SubrosaRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from remote_graph_analysis import _component_execution_stages
    from remote_hosts import RemoteHostRegistry, SshHostConfig
    from remote_plan_types import (
        ComponentExecutionPlan,
        ComponentMemoryEstimate,
        ModalPromptValidationError,
        RemoteComponentPlan,
    )
    from settings import (
        ModalSyncSettings,
        discover_comfyui_user_directory,
        settings_for_modal_gpu,
    )
    from ssh_docker import SshDockerController
    from sync_engine import SyncCancelledError
    from vast_config_node import extract_vast_profiles
    from vast_service import VastProfileQuote, VastSearchRequirements, VastService

logger = logging.getLogger(__name__)

SetupStatusCallback = Callable[[str, int | None, int | None], None]
EnvironmentSetupStatusCallback = Callable[
    [str, str, int | None, int | None], None
]
ExecutionPlanStatusCallback = Callable[
    [dict[str, dict[str, Any]], list[dict[str, Any]]], None
]


@dataclass(frozen=True)
class SubrosaPoolCapability:
    """Describe the synthetic hardware advertised for one Subrosa relay pool."""

    gpu_name: str
    total_vram_bytes: int
    cpu_count: int
    total_ram_bytes: int


_SUBROSA_DEFAULT_POOL_CAPABILITY = SubrosaPoolCapability(
    gpu_name="",
    total_vram_bytes=24 * 1024**3,
    cpu_count=16,
    total_ram_bytes=64 * 1024**3,
)

_SUBROSA_POOL_CAPABILITIES: Mapping[str, SubrosaPoolCapability] = {
    "mock-4090": SubrosaPoolCapability(
        gpu_name="NVIDIA GeForce RTX 4090",
        total_vram_bytes=24 * 1024**3,
        cpu_count=16,
        total_ram_bytes=64 * 1024**3,
    ),
    "mock-B300": SubrosaPoolCapability(
        gpu_name="NVIDIA B300",
        total_vram_bytes=288 * 1024**3,
        cpu_count=128,
        total_ram_bytes=1024 * 1024**3,
    ),
}


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
    if isinstance(configuration, SubrosaRemoteConfiguration):
        capability = _SUBROSA_POOL_CAPABILITIES.get(
            configuration.pool, _SUBROSA_DEFAULT_POOL_CAPABILITY
        )
        gpu_name = capability.gpu_name or f"Subrosa pool {configuration.pool}"
        capabilities = EnvironmentCapabilities(
            architecture="x86_64",
            operating_system="linux",
            cpu_count=capability.cpu_count,
            total_ram_bytes=capability.total_ram_bytes,
            available_ram_bytes=capability.total_ram_bytes,
            available_disk_bytes=None,
            docker_version="subrosa-managed",
            docker_rootless=False,
            nvidia_container_runtime=True,
            gpus=(
                GpuCapability(
                    uuid=f"subrosa-pool:{configuration.pool}",
                    name=gpu_name,
                    total_vram_bytes=capability.total_vram_bytes,
                    free_vram_bytes=capability.total_vram_bytes,
                ),
            ),
        )
        return EnvironmentSchedulingState(
            environment_id=f"subrosa:{configuration.configuration_id}",
            provider=ExecutionProvider.SUBROSA,
            enabled=True,
            health=EnvironmentHealth.READY,
            cost_usd_per_second=None,
            capabilities=capabilities,
            configuration_id=configuration.configuration_id,
            display_name=configuration.display_name,
            maximum_workers=configuration.capacity_limit,
        ), None
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
