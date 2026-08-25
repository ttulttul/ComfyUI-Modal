"""Tests for secure controller-side R2 credential persistence."""

from __future__ import annotations

from typing import Any


class MemoryPasswordStore:
    """Implement the keyring surface entirely in test memory."""

    def __init__(self) -> None:
        """Initialize the empty secret mapping."""
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return one stored value."""
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Store one value."""
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        """Delete one value."""
        self.values.pop((service_name, username), None)


def test_keyring_record_resolves_workflow_r2_configuration(
    r2_credentials_module: Any,
    remote_configurations_module: Any,
) -> None:
    """An opaque workflow reference should resolve to validated signing credentials."""
    password_store = MemoryPasswordStore()
    store = r2_credentials_module.R2CredentialStore(password_store=password_store)
    record = r2_credentials_module.R2CredentialRecord(
        account_id="a" * 32,
        bucket="models",
        access_key_id="access-id",
        secret_access_key="secret-key",
        jurisdiction="eu",
    )
    storage = remote_configurations_module.R2StorageBackingConfiguration(
        configuration_id="node-20",
        display_name="shared-r2",
        account_id="a" * 32,
        bucket="models",
        credential_id="credential-reference",
        jurisdiction="eu",
        key_prefix="custom/cache",
        write_back_mode="sync",
    )

    store.save(storage.credential_id, record)
    configuration = store.cache_configuration(storage)

    assert configuration.access_key_id == "access-id"
    assert configuration.secret_access_key == "secret-key"
    assert configuration.endpoint_url == f"https://{'a' * 32}.eu.r2.cloudflarestorage.com"
    assert configuration.key_prefix == "custom/cache"
    assert configuration.write_back_mode == "sync"
    status = store.status(storage.credential_id)
    assert status == {
        "connected": True,
        "account_id": "a" * 32,
        "bucket": "models",
        "jurisdiction": "eu",
    }
    assert "secret-key" not in str(status)


def test_keyring_record_must_match_workflow_bucket(
    r2_credentials_module: Any,
    remote_configurations_module: Any,
) -> None:
    """Editing a node after Login should require a new bucket-scoped credential."""
    password_store = MemoryPasswordStore()
    store = r2_credentials_module.R2CredentialStore(password_store=password_store)
    store.save(
        "credential-reference",
        r2_credentials_module.R2CredentialRecord(
            account_id="a" * 32,
            bucket="old-bucket",
            access_key_id="access-id",
            secret_access_key="secret-key",
        ),
    )
    storage = remote_configurations_module.R2StorageBackingConfiguration(
        configuration_id="node-20",
        display_name="shared-r2",
        account_id="a" * 32,
        bucket="new-bucket",
        credential_id="credential-reference",
    )

    try:
        store.cache_configuration(storage)
    except r2_credentials_module.R2CredentialError as error:
        assert "changed after Login" in str(error)
    else:
        raise AssertionError("Expected a mismatched R2 credential to be rejected.")
