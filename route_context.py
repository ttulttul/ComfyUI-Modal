"""Immutable dependencies shared by prompt interception route registrars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

if __package__:
    from .remote_hosts import RemoteHostRegistry
    from .settings import ModalSyncSettings
    from .sync_engine import ModalAssetSyncEngine
    from .vast_leases import VastLeaseRegistry
else:  # pragma: no cover - flat import inside the Modal container.
    from remote_hosts import RemoteHostRegistry
    from settings import ModalSyncSettings
    from sync_engine import ModalAssetSyncEngine
    from vast_leases import VastLeaseRegistry


@dataclass(frozen=True)
class RouteContext:
    """Hold immutable route paths, services, registries, and orchestration hooks."""

    settings: ModalSyncSettings
    sync_engine: ModalAssetSyncEngine
    remote_host_registry: RemoteHostRegistry | None
    vast_registry: VastLeaseRegistry | None
    analysis_route_path: str
    progress_state_route_path: str
    container_status_route_path: str
    container_stop_route_path: str
    delete_caches_route_path: str
    delete_volume_route_path: str
    cancel_preparation_route_path: str
    remote_environments_route_path: str
    remote_environment_probe_route_path: str
    remote_environment_bootstrap_route_path: str
    remote_environment_status_route_path: str
    remote_environment_stop_route_path: str
    rewrite_prompt: Callable[..., Any]
    emit_status: Callable[..., None]
    execution_assignments_payload: Callable[..., dict[str, Any]]
    prompt_uses_configurator: Callable[..., bool]
    configurator_node_id: Callable[..., str | None]
    selected_modal_gpus: Callable[..., list[str]]
