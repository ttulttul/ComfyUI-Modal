"""Secure controller-side storage for Cloudflare R2 S3 credentials."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import keyring
from keyring.errors import KeyringError

if __package__:
    from .r2_cache import R2CacheClient, R2CacheConfiguration, R2CacheError
    from .remote_configurations import R2StorageBackingConfiguration
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from r2_cache import R2CacheClient, R2CacheConfiguration, R2CacheError
    from remote_configurations import R2StorageBackingConfiguration

logger = logging.getLogger(__name__)

R2_KEYRING_SERVICE = "comfyui-modal-sync-cloudflare-r2"
R2_KEYCHAIN_UNLOCK_REQUIRED_CODE = "keychain_unlock_required"
_CREDENTIAL_SCHEMA_VERSION = 1
_MACOS_INTERACTION_NOT_ALLOWED_STATUS = -25308
_MACOS_KEYCHAIN_UNLOCK_TIMEOUT_SECONDS = 120.0


class R2CredentialError(RuntimeError):
    """Raised when controller-held R2 credentials cannot be stored or loaded."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Retain an optional credential-safe recovery code for the browser."""
        super().__init__(message)
        self.code = code


class PasswordStore(Protocol):
    """Describe the keyring methods used by the R2 credential store."""

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return one stored secret value."""

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """Persist one secret value."""

    def delete_password(self, service_name: str, username: str) -> None:
        """Delete one secret value."""


@dataclass(frozen=True)
class R2CredentialRecord:
    """Hold one user-created bucket-scoped R2 S3 credential."""

    account_id: str
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    jurisdiction: str = "default"

    @property
    def endpoint_url(self) -> str:
        """Return the jurisdiction-specific R2 S3 endpoint."""
        jurisdiction_segment = (
            "" if self.jurisdiction == "default" else f".{self.jurisdiction}"
        )
        return (
            f"https://{self.account_id}{jurisdiction_segment}."
            "r2.cloudflarestorage.com"
        )

    def to_secret_json(self) -> str:
        """Serialize the record for an operating-system credential vault."""
        return json.dumps(
            {
                "schema_version": _CREDENTIAL_SCHEMA_VERSION,
                "account_id": self.account_id,
                "bucket": self.bucket,
                "access_key_id": self.access_key_id,
                "secret_access_key": self.secret_access_key,
                "jurisdiction": self.jurisdiction,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_secret_json(cls, value: str) -> "R2CredentialRecord":
        """Parse and validate a record retrieved from the credential vault."""
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise R2CredentialError("Stored R2 credentials are not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise R2CredentialError("Stored R2 credentials must be a JSON object.")
        if payload.get("schema_version") != _CREDENTIAL_SCHEMA_VERSION:
            raise R2CredentialError("Stored R2 credentials use an unsupported schema.")
        record = cls(
            account_id=str(payload.get("account_id") or "").strip(),
            bucket=str(payload.get("bucket") or "").strip(),
            access_key_id=str(payload.get("access_key_id") or "").strip(),
            secret_access_key=str(payload.get("secret_access_key") or "").strip(),
            jurisdiction=str(payload.get("jurisdiction") or "default")
            .strip()
            .casefold(),
        )
        record.as_cache_configuration()
        return record

    def as_cache_configuration(
        self,
        storage: R2StorageBackingConfiguration | None = None,
    ) -> R2CacheConfiguration:
        """Build the validated cache configuration used for S3 signing."""
        return R2CacheConfiguration(
            account_id=self.account_id,
            bucket=self.bucket,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            key_prefix=(
                storage.key_prefix
                if storage is not None
                else "comfy-modal-cache/v1/blobs/sha256"
            ),
            write_back_mode=(storage.write_back_mode if storage is not None else "async"),
        )


@dataclass
class R2CredentialStore:
    """Persist R2 credentials in the operating system's secure keyring."""

    password_store: PasswordStore = field(default=keyring, repr=False)
    service_name: str = R2_KEYRING_SERVICE

    def save(self, credential_id: str, record: R2CredentialRecord) -> None:
        """Store one record without writing its secret into project files."""
        normalized_id = _validated_credential_id(credential_id)
        record.as_cache_configuration()
        try:
            self.password_store.set_password(
                self.service_name,
                normalized_id,
                record.to_secret_json(),
            )
        except KeyringError as exc:
            raise R2CredentialError(
                "The operating-system credential vault could not store Cloudflare "
                "R2 credentials. Configure a supported keyring backend and try again."
            ) from exc
        logger.info(
            "Stored Cloudflare R2 credentials in the OS keyring "
            "credential_id=%s account=%s bucket=%s.",
            normalized_id,
            record.account_id,
            record.bucket,
        )

    def load(self, credential_id: str) -> R2CredentialRecord | None:
        """Load one record, returning None when Login has not completed."""
        normalized_id = _validated_credential_id(credential_id)
        try:
            value = self.password_store.get_password(self.service_name, normalized_id)
        except KeyringError as exc:
            if _exception_contains_status(
                exc,
                _MACOS_INTERACTION_NOT_ALLOWED_STATUS,
            ):
                raise R2CredentialError(
                    "The macOS login keychain must be unlocked before Cloudflare "
                    "R2 credentials can be read.",
                    code=R2_KEYCHAIN_UNLOCK_REQUIRED_CODE,
                ) from exc
            raise R2CredentialError(
                "The operating-system credential vault could not read Cloudflare "
                "R2 credentials."
            ) from exc
        return None if value is None else R2CredentialRecord.from_secret_json(value)

    def cache_configuration(
        self,
        storage: R2StorageBackingConfiguration,
    ) -> R2CacheConfiguration:
        """Resolve one workflow reference into credential-bearing cache settings."""
        record = self.load(storage.credential_id)
        if record is None:
            raise R2CredentialError(
                f"R2 API credentials have not been saved for configuration "
                f"{storage.display_name!r}."
            )
        if record.account_id != storage.account_id or record.bucket != storage.bucket:
            raise R2CredentialError(
                "The R2 node account or bucket changed after credentials were saved; "
                "import credentials for the configured bucket again."
            )
        if record.jurisdiction != storage.jurisdiction:
            raise R2CredentialError(
                "The R2 node jurisdiction changed after credentials were saved; "
                "import the credentials again."
            )
        return record.as_cache_configuration(storage)

    def status(self, credential_id: str) -> dict[str, Any]:
        """Return credential-free connection status for the browser UI."""
        record = self.load(credential_id)
        if record is None:
            return {"connected": False}
        return {
            "connected": True,
            "account_id": record.account_id,
            "bucket": record.bucket,
            "jurisdiction": record.jurisdiction,
        }


def validate_r2_credentials(record: R2CredentialRecord) -> None:
    """Verify imported S3 credentials against their exact configured R2 bucket."""
    try:
        R2CacheClient(record.as_cache_configuration()).validate_bucket_access()
    except (R2CacheError, RuntimeError, ValueError) as exc:
        raise R2CredentialError(str(exc)) from exc
    logger.info(
        "Validated imported Cloudflare R2 credentials account=%s bucket=%s.",
        record.account_id,
        record.bucket,
    )


def _validated_credential_id(credential_id: str) -> str:
    """Reject empty or control-character-bearing credential references."""
    normalized_id = str(credential_id).strip()
    if not normalized_id or any(
        character in normalized_id for character in ("\x00", "\n", "\r")
    ):
        raise ValueError("R2 credential ID must be a non-empty single-line value.")
    return normalized_id


def _exception_contains_status(error: BaseException, status: int) -> bool:
    """Return whether an exception chain contains one integer OS status."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if getattr(current, "status", None) == status or status in current.args:
            return True
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return False


def request_macos_keychain_unlock(
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    platform_name: str | None = None,
    timeout_seconds: float = _MACOS_KEYCHAIN_UNLOCK_TIMEOUT_SECONDS,
) -> None:
    """Ask macOS SecurityAgent to display its default-keychain unlock dialog."""
    if (platform_name or sys.platform) != "darwin":
        raise R2CredentialError(
            "Interactive keychain unlock is available only on macOS."
        )
    try:
        result = command_runner(
            ["/usr/bin/security", "unlock-keychain", "-u"],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise R2CredentialError(
            "The macOS keychain unlock prompt timed out."
        ) from exc
    except OSError as exc:
        raise R2CredentialError(
            "macOS could not start the system keychain unlock prompt."
        ) from exc
    if result.returncode != 0:
        raise R2CredentialError(
            "The macOS login keychain was not unlocked."
        )
    logger.info("The macOS login keychain was unlocked through SecurityAgent.")


__all__ = [
    "R2CredentialError",
    "R2CredentialRecord",
    "R2CredentialStore",
    "R2_KEYCHAIN_UNLOCK_REQUIRED_CODE",
    "R2_KEYRING_SERVICE",
    "request_macos_keychain_unlock",
    "validate_r2_credentials",
]
