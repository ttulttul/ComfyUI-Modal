"""Persistent per-ComfyUI identity for Modal resource namespacing."""

from __future__ import annotations

import base64
import logging
import os
import secrets
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

INSTANCE_ID_BYTES = 8
INSTANCE_ID_ENTROPY_BITS = INSTANCE_ID_BYTES * 8
INSTANCE_ID_FILENAME = ".comfy-modal-sync-instance-id"
MODAL_APP_PREFIX = "comfy-modal-sync"


class ComfyInstanceIdentityError(RuntimeError):
    """Raised when a persistent ComfyUI instance identity cannot be used safely."""


def _read_instance_id(identity_path: Path) -> bytes:
    """Read and validate one persisted hexadecimal instance identifier."""
    try:
        encoded_identity = identity_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ComfyInstanceIdentityError(
            f"Unable to read ComfyUI instance identity from {identity_path}: {exc}"
        ) from exc

    try:
        instance_id = bytes.fromhex(encoded_identity)
    except ValueError as exc:
        raise ComfyInstanceIdentityError(
            f"ComfyUI instance identity at {identity_path} is not valid hexadecimal data."
        ) from exc
    if len(instance_id) != INSTANCE_ID_BYTES or encoded_identity != instance_id.hex():
        raise ComfyInstanceIdentityError(
            f"ComfyUI instance identity at {identity_path} must contain exactly "
            f"{INSTANCE_ID_BYTES} bytes encoded as lowercase hexadecimal."
        )
    return instance_id


def _remove_temporary_identity_file(temporary_path: Path) -> None:
    """Remove an unpublished identity candidate without masking the primary result."""
    try:
        temporary_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Unable to remove temporary instance identity %s: %s", temporary_path, exc)


def _publish_instance_id(identity_path: Path, instance_id: bytes) -> bool:
    """Atomically publish an identifier and return whether this process won creation."""
    try:
        identity_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ComfyInstanceIdentityError(
            f"Unable to create instance identity directory {identity_path.parent}: {exc}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=identity_path.parent,
            prefix=f".{identity_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(f"{instance_id.hex()}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.link(temporary_path, identity_path)
        except FileExistsError:
            return False
        except OSError as exc:
            raise ComfyInstanceIdentityError(
                f"Unable to publish ComfyUI instance identity at {identity_path}: {exc}"
            ) from exc
        return True
    finally:
        if temporary_path is not None:
            _remove_temporary_identity_file(temporary_path)


def load_or_create_instance_id(identity_path: Path) -> bytes:
    """Return the stable 64-bit identifier stored at the requested path."""
    try:
        return _read_instance_id(identity_path)
    except FileNotFoundError:
        pass

    candidate = secrets.token_bytes(INSTANCE_ID_BYTES)
    if _publish_instance_id(identity_path, candidate):
        logger.info(
            "Created %d-bit ComfyUI instance identity at %s.",
            INSTANCE_ID_ENTROPY_BITS,
            identity_path,
        )
        return candidate
    return _read_instance_id(identity_path)


def instance_id_base64(instance_id: bytes) -> str:
    """Encode a 64-bit instance identifier as unpadded URL-safe Base64."""
    if len(instance_id) != INSTANCE_ID_BYTES:
        raise ValueError(f"Instance identifier must contain exactly {INSTANCE_ID_BYTES} bytes.")
    return base64.urlsafe_b64encode(instance_id).decode("ascii").rstrip("=")


def modal_app_name_for_instance(instance_id: bytes) -> str:
    """Return the Modal app name namespaced to one persistent ComfyUI instance."""
    return f"{MODAL_APP_PREFIX}-{instance_id_base64(instance_id)}"
