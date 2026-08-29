"""Tests for routing workflow-scoped R2 storage into remote sync transports."""

from __future__ import annotations

from typing import Any


def test_workflow_r2_configuration_resolves_controller_signing_client(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    remote_configurations_module: Any,
    r2_cache_module: Any,
    monkeypatch: Any,
) -> None:
    """The connected storage node should become the cache used by Vast and SSH sync."""
    storage = remote_configurations_module.R2StorageBackingConfiguration(
        configuration_id="r2-node",
        display_name="shared-r2",
        account_id="a" * 32,
        bucket="models",
        credential_id="opaque-reference",
    )
    capacity = remote_configurations_module.ModalRemoteConfiguration(
        configuration_id="modal-node",
        display_name="modal",
        gpu_type="H200",
    )
    configuration_set = remote_configurations_module.RemoteConfigurationSet(
        (capacity, storage)
    )
    expected = r2_cache_module.R2CacheConfiguration(
        account_id="a" * 32,
        bucket="models",
        access_key_id="access-id",
        secret_access_key="secret-key",
        endpoint_url=f"https://{'a' * 32}.r2.cloudflarestorage.com",
    )

    class Store:
        """Resolve the expected credential-bearing configuration."""

        def cache_configuration(self, requested_storage: Any) -> Any:
            """Require the exact workflow storage object."""
            assert requested_storage is storage
            return expected

    monkeypatch.setattr(execution_scheduling_module, "R2CredentialStore", Store)

    client = execution_scheduling_module._workflow_r2_cache(configuration_set)

    assert client is not None
    assert client.configuration is expected


def test_safe_configuration_payload_adds_cached_r2_bucket_usage(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    remote_configurations_module: Any,
    r2_cache_module: Any,
    monkeypatch: Any,
) -> None:
    """Planning metadata should expose bucket usage without exposing credentials."""
    storage = remote_configurations_module.R2StorageBackingConfiguration(
        configuration_id="r2-node",
        display_name="shared-r2",
        account_id="a" * 32,
        bucket="models",
        credential_id="opaque-reference",
    )
    capacity = remote_configurations_module.ModalRemoteConfiguration(
        configuration_id="modal-node",
        display_name="modal",
        gpu_type="H200",
    )
    configuration_set = remote_configurations_module.RemoteConfigurationSet(
        (capacity, storage)
    )
    expected = r2_cache_module.R2CacheConfiguration(
        account_id="a" * 32,
        bucket="models",
        access_key_id="access-id",
        secret_access_key="secret-key",
        endpoint_url=f"https://{'a' * 32}.r2.cloudflarestorage.com",
    )
    usage_calls = 0
    usage_size_bytes = 5 * 1024**3

    class Store:
        """Resolve the expected credential-bearing configuration."""

        def cache_configuration(self, requested_storage: Any) -> Any:
            """Require the exact workflow storage object."""
            assert requested_storage is storage
            return expected

    class Client:
        """Return deterministic bucket usage without provider I/O."""

        def __init__(self, configuration: Any) -> None:
            """Require the resolved credential-bearing configuration."""
            assert configuration is expected

        def storage_usage(self) -> Any:
            """Record the provider query and return safe aggregate metrics."""
            nonlocal usage_calls, usage_size_bytes
            usage_calls += 1
            return r2_cache_module.R2StorageUsage(
                size_bytes=usage_size_bytes,
                object_count=42,
            )

    execution_scheduling_module._R2_STORAGE_USAGE_CACHE.clear()
    monkeypatch.setattr(execution_scheduling_module, "R2CredentialStore", Store)
    monkeypatch.setattr(execution_scheduling_module, "R2CacheClient", Client)

    first = execution_scheduling_module._safe_remote_configuration_payload(
        configuration_set
    )
    second = execution_scheduling_module._safe_remote_configuration_payload(
        configuration_set
    )

    safe_storage = next(
        item for item in first if item["configuration_id"] == "r2-node"
    )
    assert safe_storage["storage_usage_bytes"] == 5 * 1024**3
    assert safe_storage["storage_object_count"] == 42
    assert "credential" not in str(safe_storage).casefold()
    assert second == first
    assert usage_calls == 1

    usage_size_bytes = 6 * 1024**3
    refreshed = execution_scheduling_module._refresh_r2_storage_usage(storage)
    third = execution_scheduling_module._safe_remote_configuration_payload(
        configuration_set
    )

    assert refreshed.size_bytes == 6 * 1024**3
    assert next(
        item for item in third if item["configuration_id"] == "r2-node"
    )["storage_usage_bytes"] == 6 * 1024**3
    assert usage_calls == 2
