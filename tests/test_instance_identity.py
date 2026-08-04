"""Tests for persistent per-ComfyUI Modal app identities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest


def test_instance_identity_is_persisted_and_reused(
    instance_identity_module: Any,
    tmp_path: Path,
) -> None:
    """A generated 64-bit identifier should remain stable across later loads."""
    identity_path = tmp_path / "user" / instance_identity_module.INSTANCE_ID_FILENAME

    first_identity = instance_identity_module.load_or_create_instance_id(identity_path)
    second_identity = instance_identity_module.load_or_create_instance_id(identity_path)

    assert first_identity == second_identity
    assert len(first_identity) == instance_identity_module.INSTANCE_ID_BYTES
    assert identity_path.read_text(encoding="ascii").strip() == first_identity.hex()


def test_instance_identity_uses_unpadded_url_safe_base64(
    instance_identity_module: Any,
) -> None:
    """The app suffix should encode all 64 bits into eleven URL-safe characters."""
    instance_id = bytes.fromhex("0001020304050607")

    app_name = instance_identity_module.modal_app_name_for_instance(instance_id)

    assert app_name == "comfy-modal-sync-AAECAwQFBgc"


def test_concurrent_identity_creation_publishes_one_winner(
    instance_identity_module: Any,
    tmp_path: Path,
) -> None:
    """Concurrent first startups should all observe the same atomically published identity."""
    identity_path = tmp_path / "user" / instance_identity_module.INSTANCE_ID_FILENAME

    with ThreadPoolExecutor(max_workers=16) as executor:
        identities = list(
            executor.map(
                lambda _: instance_identity_module.load_or_create_instance_id(identity_path),
                range(64),
            )
        )

    assert len(set(identities)) == 1
    assert list(identity_path.parent.glob(f".{identity_path.name}.*.tmp")) == []


def test_corrupt_instance_identity_is_not_silently_replaced(
    instance_identity_module: Any,
    tmp_path: Path,
) -> None:
    """Malformed persistent state should fail rather than silently select a new Modal app."""
    identity_path = tmp_path / instance_identity_module.INSTANCE_ID_FILENAME
    identity_path.write_text("not-an-identity\n", encoding="ascii")

    with pytest.raises(instance_identity_module.ComfyInstanceIdentityError):
        instance_identity_module.load_or_create_instance_id(identity_path)

    assert identity_path.read_text(encoding="ascii") == "not-an-identity\n"
