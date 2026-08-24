"""Persistent provenance records for remotely materialized Hugging Face assets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

logger = logging.getLogger(__name__)

HUGGINGFACE_ASSET_REGISTRY_FILENAME = "huggingface-assets.json"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Hugging Face asset not found: {resolved_path}")
    digest = hashlib.sha256()
    with resolved_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_repo_id(value: object) -> str:
    """Return one conservative Hugging Face model repository identifier."""
    normalized = str(value).strip()
    parts = normalized.split("/")
    if len(parts) != 2 or not all(_REPO_COMPONENT.fullmatch(part) for part in parts):
        raise ValueError(
            "Hugging Face repo_id must have the form 'owner/model' using safe characters."
        )
    return normalized


def _validated_revision(value: object) -> str:
    """Require an immutable 40-character Hugging Face commit revision."""
    normalized = str(value).strip().lower()
    if not _HEX_COMMIT.fullmatch(normalized):
        raise ValueError(
            "Hugging Face asset revision must be an exact 40-character commit SHA."
        )
    return normalized


def _validated_filename(value: object) -> str:
    """Return one traversal-safe repository-relative Hugging Face filename."""
    normalized = str(value).strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in normalized
    ):
        raise ValueError("Hugging Face asset filename must be repository-relative and safe.")
    return path.as_posix()


def _validated_sha256(value: object) -> str:
    """Return one lowercase SHA-256 digest or raise a validation error."""
    normalized = str(value).strip().lower()
    if not _HEX_SHA256.fullmatch(normalized):
        raise ValueError("Hugging Face asset sha256 must contain 64 lowercase hex digits.")
    return normalized


def _validated_size_bytes(value: object) -> int:
    """Return one positive file size without accepting booleans as integers."""
    if isinstance(value, bool):
        raise ValueError("Hugging Face asset size_bytes must be a positive integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Hugging Face asset size_bytes must be a positive integer."
        ) from exc
    if normalized <= 0:
        raise ValueError("Hugging Face asset size_bytes must be a positive integer.")
    return normalized


@dataclass(frozen=True)
class HuggingFaceAssetSource:
    """Identify one immutable Hugging Face file and its expected local contents."""

    repo_id: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Normalize and validate every persisted provenance field."""
        object.__setattr__(self, "repo_id", _validated_repo_id(self.repo_id))
        object.__setattr__(self, "revision", _validated_revision(self.revision))
        object.__setattr__(self, "filename", _validated_filename(self.filename))
        object.__setattr__(self, "sha256", _validated_sha256(self.sha256))
        object.__setattr__(self, "size_bytes", _validated_size_bytes(self.size_bytes))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HuggingFaceAssetSource":
        """Build a validated source from one JSON-compatible mapping."""
        return cls(
            repo_id=str(payload.get("repo_id") or ""),
            revision=str(payload.get("revision") or ""),
            filename=str(payload.get("filename") or ""),
            sha256=str(payload.get("sha256") or ""),
            size_bytes=payload.get("size_bytes", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible provenance record."""
        return {
            "filename": self.filename,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @property
    def display_reference(self) -> str:
        """Return a compact immutable reference suitable for status messages."""
        return f"{self.repo_id}@{self.revision[:12]}/{self.filename}"


@dataclass
class HuggingFaceAssetRegistry:
    """Atomically persist Hugging Face provenance indexed by content SHA-256."""

    config_path: Path
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def for_user_directory(cls, user_directory: Path) -> "HuggingFaceAssetRegistry":
        """Create a registry below one persistent ComfyUI user directory."""
        return cls(
            config_path=(
                user_directory.expanduser().resolve()
                / "comfyui-modal"
                / HUGGINGFACE_ASSET_REGISTRY_FILENAME
            )
        )

    def get(self, sha256: str) -> HuggingFaceAssetSource | None:
        """Return one source by content digest when it is registered."""
        normalized_sha256 = _validated_sha256(sha256)
        with self._lock:
            payload = self._load_payload().get(normalized_sha256)
        if not isinstance(payload, Mapping):
            return None
        return HuggingFaceAssetSource.from_dict(payload)

    def upsert(self, source: HuggingFaceAssetSource) -> HuggingFaceAssetSource:
        """Insert or replace one validated source and persist it atomically."""
        validated = HuggingFaceAssetSource.from_dict(source.to_dict())
        with self._lock:
            payload = self._load_payload()
            payload[validated.sha256] = validated.to_dict()
            self._save_payload(payload)
        logger.info(
            "Registered Hugging Face asset sha256=%s source=%s.",
            validated.sha256,
            validated.display_reference,
        )
        return validated

    def _load_payload(self) -> dict[str, dict[str, Any]]:
        """Load and validate the registry document or return an empty registry."""
        if not self.config_path.exists():
            return {}
        try:
            raw_payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Hugging Face asset registry {self.config_path} is unreadable."
            ) from exc
        if not isinstance(raw_payload, Mapping):
            raise ValueError("Hugging Face asset registry must be a JSON object.")
        records = raw_payload.get("assets", raw_payload)
        if not isinstance(records, Mapping):
            raise ValueError("Hugging Face asset registry assets must be an object.")
        normalized: dict[str, dict[str, Any]] = {}
        for key, value in records.items():
            if not isinstance(value, Mapping):
                raise ValueError("Hugging Face asset registry records must be objects.")
            source = HuggingFaceAssetSource.from_dict(value)
            normalized_key = _validated_sha256(key)
            if normalized_key != source.sha256:
                raise ValueError(
                    "Hugging Face asset registry key does not match its record SHA-256."
                )
            normalized[normalized_key] = source.to_dict()
        return normalized

    def _save_payload(self, assets: Mapping[str, Mapping[str, Any]]) -> None:
        """Write one complete registry document with fsync and atomic replacement."""
        serialized = json.dumps(
            {"version": 1, "assets": assets},
            indent=2,
            sort_keys=True,
        ) + "\n"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.config_path.name}.",
            suffix=".tmp",
            dir=self.config_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
                output_file.write(serialized)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, self.config_path)
        finally:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "HUGGINGFACE_ASSET_REGISTRY_FILENAME",
    "HuggingFaceAssetRegistry",
    "HuggingFaceAssetSource",
    "sha256_file",
]
