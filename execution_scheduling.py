"""Provider-neutral execution planning, capacity acquisition, and assignment metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

if __package__:
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
    from .llm_profiles import get_llm_profile
    from .llm_resolver import resolve_model_profile
    from .modal_hardware import (
        _MODAL_GPU_COST_USD_PER_SECOND,
        _MODAL_GPU_VRAM_GB,
        _capabilities_hardware_payload,
        _modal_hardware_payload,
        _vast_hardware_payload,
    )
    from .r2_cache import R2CacheClient, R2CacheError, R2StorageUsage
    from .r2_credentials import R2CredentialError, R2CredentialStore
    from .remote_configuration_nodes import compile_remote_configuration_set
    from .remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        R2StorageBackingConfiguration,
        SshRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from .remote_graph_analysis import _component_execution_stages, _is_link
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
    from .ssh_docker import SshDockerController, SshDockerVolumeBackend
    from .ssh_runtime import SshRuntimeManager
    from .sync_engine import (
        MODEL_FILE_EXTENSIONS,
        ModalAssetSyncEngine,
        ModalVolumeBackend,
        SyncCancelledError,
        resolve_model_path,
    )
    from .vast_config_node import extract_vast_profiles
    from .vast_service import VastProfileQuote, VastSearchRequirements, VastService
else:  # pragma: no cover - flat import inside the Modal container.
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
    from llm_profiles import get_llm_profile
    from llm_resolver import resolve_model_profile
    from modal_hardware import (
        _MODAL_GPU_COST_USD_PER_SECOND,
        _MODAL_GPU_VRAM_GB,
        _capabilities_hardware_payload,
        _modal_hardware_payload,
        _vast_hardware_payload,
    )
    from r2_cache import R2CacheClient, R2CacheError, R2StorageUsage
    from r2_credentials import R2CredentialError, R2CredentialStore
    from remote_configuration_nodes import compile_remote_configuration_set
    from remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        R2StorageBackingConfiguration,
        SshRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from remote_graph_analysis import _component_execution_stages, _is_link
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
    from ssh_docker import SshDockerController, SshDockerVolumeBackend
    from ssh_runtime import SshRuntimeManager
    from sync_engine import (
        MODEL_FILE_EXTENSIONS,
        ModalAssetSyncEngine,
        ModalVolumeBackend,
        SyncCancelledError,
        resolve_model_path,
    )
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
