"""Credential-safe configuration and assignment reporting for execution plans."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping, Sequence

if __package__:
    from .execution_environments import ExecutionAssignment, ExecutionProvider
    from .modal_hardware import (
        _capabilities_hardware_payload,
        _modal_hardware_payload,
        _vast_hardware_payload,
    )
    from .r2_cache import R2CacheClient, R2CacheError, R2StorageUsage
    from .r2_credentials import R2CredentialError, R2CredentialStore
    from .remote_configurations import (
        R2StorageBackingConfiguration,
        RemoteConfigurationSet,
    )
    from .remote_hosts import SshHostConfig
    from .remote_plan_types import RemoteComponentPlan
    from .vast_service import VastProfileQuote
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import ExecutionAssignment, ExecutionProvider
    from modal_hardware import (
        _capabilities_hardware_payload,
        _modal_hardware_payload,
        _vast_hardware_payload,
    )
    from r2_cache import R2CacheClient, R2CacheError, R2StorageUsage
    from r2_credentials import R2CredentialError, R2CredentialStore
    from remote_configurations import (
        R2StorageBackingConfiguration,
        RemoteConfigurationSet,
    )
    from remote_hosts import SshHostConfig
    from remote_plan_types import RemoteComponentPlan
    from vast_service import VastProfileQuote

logger = logging.getLogger(__name__)

_R2_STORAGE_USAGE_CACHE_SECONDS = 5 * 60
_R2_STORAGE_USAGE_CACHE_LOCK = threading.Lock()
_R2_STORAGE_USAGE_CACHE: dict[
    tuple[str, str, str], tuple[float, R2StorageUsage]
] = {}


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


