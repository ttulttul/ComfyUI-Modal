"""Validation and compatibility helpers for immutable Vast worker images."""

from __future__ import annotations

import re

_IMMUTABLE_IMAGE_PATTERN = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")


def require_immutable_vast_image(image: str) -> str:
    """Return a normalized OCI digest reference or reject a mutable image tag."""
    normalized = image.strip()
    if not _IMMUTABLE_IMAGE_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Vast worker image must be an immutable OCI reference ending in "
            "@sha256:<64 lowercase hex characters>."
        )
    return normalized


def vast_worker_images_compatible(expected: str | None, actual: str | None) -> bool:
    """Return whether two lease images name the same immutable OCI digest."""
    if expected is None or actual is None:
        return False
    try:
        return require_immutable_vast_image(expected) == require_immutable_vast_image(
            actual
        )
    except ValueError:
        return False


__all__ = ["require_immutable_vast_image", "vast_worker_images_compatible"]
