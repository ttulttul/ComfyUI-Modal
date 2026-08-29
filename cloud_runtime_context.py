"""Registered Modal runtime stores shared by extracted cloud subsystems."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True)
class CloudRuntimeStores:
    """Live Modal stores created by the stable cloud entrypoint."""

    session_bridge_cache: Any | None = None
    invocation_records: Any | None = None
    volume: Any | None = None
    snapshot_profiles: Any | None = None
    node_output_cache: Any | None = None
    interrupt_flags: Any | None = None


_RUNTIME_STORES_LOCK = threading.Lock()
_RUNTIME_STORES = CloudRuntimeStores()


def register_cloud_runtime_stores(
    *,
    session_bridge_cache: Any | None,
    invocation_records: Any | None,
    volume: Any | None,
    snapshot_profiles: Any | None,
    node_output_cache: Any | None,
    interrupt_flags: Any | None,
) -> None:
    """Atomically publish the live stores created by the cloud entrypoint."""
    global _RUNTIME_STORES

    stores = CloudRuntimeStores(
        session_bridge_cache=session_bridge_cache,
        invocation_records=invocation_records,
        volume=volume,
        snapshot_profiles=snapshot_profiles,
        node_output_cache=node_output_cache,
        interrupt_flags=interrupt_flags,
    )
    with _RUNTIME_STORES_LOCK:
        _RUNTIME_STORES = stores


def clear_cloud_runtime_stores() -> None:
    """Clear registered stores, primarily for isolated imports and tests."""
    global _RUNTIME_STORES

    with _RUNTIME_STORES_LOCK:
        _RUNTIME_STORES = CloudRuntimeStores()


def session_bridge_store() -> Any | None:
    """Return the registered durable session-bridge store, if available."""
    return _RUNTIME_STORES.session_bridge_cache


def invocation_record_store() -> Any | None:
    """Return the registered durable invocation-record store, if available."""
    return _RUNTIME_STORES.invocation_records


def volume_store() -> Any | None:
    """Return the registered Modal volume, if available."""
    return _RUNTIME_STORES.volume


def snapshot_profile_store() -> Any | None:
    """Return the registered snapshot-profile store, if available."""
    return _RUNTIME_STORES.snapshot_profiles


def node_output_cache_store() -> Any | None:
    """Return the registered persisted node-output cache, if available."""
    return _RUNTIME_STORES.node_output_cache


def interrupt_flag_store() -> Any | None:
    """Return the registered remote-interrupt flag store, if available."""
    return _RUNTIME_STORES.interrupt_flags
