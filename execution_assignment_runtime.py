"""Runtime backend construction and placement metadata for execution assignments."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from .execution_environments import (
        EnvironmentHealth,
        ExecutionAssignment,
        ExecutionProvider,
    )
    from .r2_cache import R2CacheClient
    from .r2_credentials import R2CredentialStore
    from .remote_configurations import (
        ModalRemoteConfiguration,
        R2StorageBackingConfiguration,
        RemoteConfigurationSet,
        SshRemoteConfiguration,
        SubrosaRemoteConfiguration,
    )
    from .remote_hosts import SshHostConfig
    from .remote_plan_types import ComponentExecutionPlan, ModalPromptValidationError
    from .settings import ModalSyncSettings
    from .ssh_docker import SshDockerController, SshDockerVolumeBackend
    from .ssh_runtime import SshRuntimeManager
    from .sync_engine import ModalAssetSyncEngine, ModalVolumeBackend
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import (
        EnvironmentHealth,
        ExecutionAssignment,
        ExecutionProvider,
    )
    from r2_cache import R2CacheClient
    from r2_credentials import R2CredentialStore
    from remote_configurations import (
        ModalRemoteConfiguration,
        R2StorageBackingConfiguration,
        RemoteConfigurationSet,
        SshRemoteConfiguration,
        SubrosaRemoteConfiguration,
    )
    from remote_hosts import SshHostConfig
    from remote_plan_types import ComponentExecutionPlan, ModalPromptValidationError
    from settings import ModalSyncSettings
    from ssh_docker import SshDockerController, SshDockerVolumeBackend
    from ssh_runtime import SshRuntimeManager
    from sync_engine import ModalAssetSyncEngine, ModalVolumeBackend

logger = logging.getLogger(__name__)


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
    if assignment.provider is ExecutionProvider.SUBROSA:
        return assignment.environment_id
    host = ssh_hosts_by_id.get(assignment.environment_id)
    return _ssh_hostname(host.ssh_target) if host is not None else assignment.environment_id


def _vast_provider_metadata(lease: Any) -> dict[str, Any]:
    """Return credential-free lease details for execution and local status output."""
    metadata = {
        "vast_instance_id": lease.instance_id,
        "vast_profile_id": lease.profile_id,
        "vast_profile_name": lease.profile_name,
        "vast_gpu_name": lease.gpu_name,
        "vast_gpu_count": lease.gpu_count,
        "vast_gpu_ram_mb": lease.gpu_ram_mb,
        "vast_hourly_cost_usd": lease.hourly_cost_usd,
        "vast_idle_retention_seconds": lease.idle_retention_seconds,
    }
    runtime_fingerprint = getattr(lease, "runtime_fingerprint", None)
    worker_image = getattr(lease, "worker_image", None)
    if runtime_fingerprint is not None and worker_image is not None:
        metadata["vast_runtime_fingerprint"] = runtime_fingerprint
        metadata["vast_worker_image"] = worker_image
    return metadata


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
    if isinstance(configuration, SubrosaRemoteConfiguration):
        return {
            "relay_url": configuration.relay_url,
            "pool": configuration.pool,
            "configuration_id": configuration.configuration_id,
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
