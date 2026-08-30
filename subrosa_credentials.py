"""Secure controller-side storage for Subrosa extension tokens."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import keyring
from keyring.errors import KeyringError

if __package__:
    from .r2_credentials import PasswordStore
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from r2_credentials import PasswordStore

logger = logging.getLogger(__name__)

SUBROSA_KEYRING_SERVICE = "comfyui-modal-sync-subrosa"
SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE = "keychain_unlock_required"
_MACOS_INTERACTION_NOT_ALLOWED_STATUS = -25308


class SubrosaCredentialError(RuntimeError):
    """Raised when a Subrosa extension token cannot be stored or loaded."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Retain an optional credential-safe recovery code for the UI."""
        super().__init__(message)
        self.code = code


@dataclass
class SubrosaCredentialStore:
    """Persist Subrosa extension tokens in the operating-system keyring."""

    password_store: PasswordStore = field(default=keyring, repr=False)
    service_name: str = SUBROSA_KEYRING_SERVICE

    def save(self, credential_id: str, token: str) -> None:
        """Store one validated extension token under an opaque workflow reference."""
        normalized_id = _validated_credential_id(credential_id)
        normalized_token = _validated_extension_token(token)
        try:
            self.password_store.set_password(
                self.service_name,
                normalized_id,
                normalized_token,
            )
        except KeyringError as exc:
            if _exception_contains_status(exc, _MACOS_INTERACTION_NOT_ALLOWED_STATUS):
                raise SubrosaCredentialError(
                    "The macOS login keychain must be unlocked before the Subrosa "
                    "token can be saved.",
                    code=SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE,
                ) from exc
            raise SubrosaCredentialError(
                "The operating-system credential vault could not store the Subrosa token."
            ) from exc
        logger.info("Stored a Subrosa extension token in the OS keyring credential_id=%s.", normalized_id)

    def load(self, credential_id: str) -> str | None:
        """Load one extension token, returning None when it has not been configured."""
        normalized_id = _validated_credential_id(credential_id)
        try:
            token = self.password_store.get_password(self.service_name, normalized_id)
        except KeyringError as exc:
            if _exception_contains_status(exc, _MACOS_INTERACTION_NOT_ALLOWED_STATUS):
                raise SubrosaCredentialError(
                    "The macOS login keychain must be unlocked before the Subrosa "
                    "token can be read.",
                    code=SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE,
                ) from exc
            raise SubrosaCredentialError(
                "The operating-system credential vault could not read the Subrosa token."
            ) from exc
        return None if token is None else _validated_extension_token(token)

    def require(self, credential_id: str) -> str:
        """Return one token or raise a credential-safe configuration error."""
        token = self.load(credential_id)
        if token is None:
            raise SubrosaCredentialError(
                f"A Subrosa extension token has not been saved for credential reference {credential_id!r}."
            )
        return token

    def delete(self, credential_id: str) -> None:
        """Remove one extension token from the operating-system keyring."""
        normalized_id = _validated_credential_id(credential_id)
        try:
            self.password_store.delete_password(self.service_name, normalized_id)
        except KeyringError as exc:
            raise SubrosaCredentialError(
                "The operating-system credential vault could not delete the Subrosa token."
            ) from exc


def _validated_credential_id(value: str) -> str:
    """Return one non-empty, bounded keyring lookup identifier."""
    normalized = str(value).strip()
    if not normalized:
        raise SubrosaCredentialError("Subrosa credential_id must not be empty.")
    if len(normalized) > 256:
        raise SubrosaCredentialError("Subrosa credential_id is too long.")
    return normalized


def _validated_extension_token(value: str) -> str:
    """Return one syntactically valid Subrosa extension token."""
    normalized = str(value).strip()
    if not normalized.startswith("srk_") or len(normalized) <= 4:
        raise SubrosaCredentialError("Subrosa extension tokens must begin with 'srk_'.")
    return normalized


def _exception_contains_status(error: BaseException, status: int) -> bool:
    """Return whether an exception chain contains one macOS Security status."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if status in getattr(current, "args", ()):
            return True
        if str(status) in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = [
    "SUBROSA_KEYCHAIN_UNLOCK_REQUIRED_CODE",
    "SUBROSA_KEYRING_SERVICE",
    "SubrosaCredentialError",
    "SubrosaCredentialStore",
]
