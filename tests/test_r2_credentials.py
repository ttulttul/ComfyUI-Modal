"""Tests for secure controller-side R2 credential persistence."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest


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
    assert configuration.write_back_mode == "async"
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
        assert "changed after credentials were saved" in str(error)
    else:
        raise AssertionError("Expected a mismatched R2 credential to be rejected.")


def test_macos_interaction_error_requests_keychain_unlock(
    r2_credentials_module: Any,
) -> None:
    """macOS interaction denial should become a credential-safe recovery code."""

    class LockedPasswordStore(MemoryPasswordStore):
        """Raise the nested status emitted by keyring's macOS backend."""

        def get_password(self, service_name: str, username: str) -> str | None:
            """Fail as though the login keychain cannot display authentication."""
            del service_name, username
            try:
                raise RuntimeError(-25308, "Unknown Error")
            except RuntimeError as exc:
                raise r2_credentials_module.KeyringError(
                    "Can't get password from keychain"
                ) from exc

    store = r2_credentials_module.R2CredentialStore(
        password_store=LockedPasswordStore()
    )

    with pytest.raises(r2_credentials_module.R2CredentialError) as captured:
        store.load("credential-reference")

    assert captured.value.code == "keychain_unlock_required"
    assert "macOS login keychain must be unlocked" in str(captured.value)


def test_macos_unlock_uses_system_owned_prompt(
    r2_credentials_module: Any,
) -> None:
    """Unlock recovery must never place a password in the child command."""
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run_command(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Capture the security invocation and report successful authentication."""
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    r2_credentials_module.request_macos_keychain_unlock(
        command_runner=run_command,
        platform_name="darwin",
        timeout_seconds=15,
    )

    assert calls[0][0] == ["/usr/bin/security", "unlock-keychain", "-u"]
    assert calls[0][1] == {
        "capture_output": True,
        "check": False,
        "text": True,
        "timeout": 15,
    }
