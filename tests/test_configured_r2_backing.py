"""Tests for routing workflow-scoped R2 storage into remote sync transports."""

from __future__ import annotations

from typing import Any


def test_workflow_r2_configuration_resolves_controller_signing_client(
    api_intercept_module: Any,
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

    monkeypatch.setattr(api_intercept_module, "R2CredentialStore", Store)

    client = api_intercept_module._workflow_r2_cache(configuration_set)

    assert client is not None
    assert client.configuration is expected
