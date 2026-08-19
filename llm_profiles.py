"""Curated, revision-pinned model profiles for Modal LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping

logger = logging.getLogger(__name__)

MODAL_LLM_NODE_ID = "ModalLLM"
LLM_PROFILE_FILENAME = "llm_profiles.json"
LLM_MODEL_DIRECTORY_NAME = "llm_models"
_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTED_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_SUPPORTED_MODALITIES = frozenset({"text", "image", "video", "file"})


@dataclass(frozen=True)
class LLMModelProfile:
    """Describe one reviewed Hugging Face model and its resource limits."""

    profile_id: str
    display_name: str
    repository: str
    revision: str
    dtype: str
    modalities: frozenset[str]
    estimated_vram_gb: float
    max_context_tokens: int
    max_images: int
    max_video_frames: int
    max_file_bytes: int
    max_file_characters: int
    allow_mixed_image_video: bool
    trust_remote_code: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LLMModelProfile":
        """Validate and construct one profile from decoded JSON data."""
        profile_id = str(value.get("id", "")).strip()
        revision = str(value.get("revision", "")).strip().lower()
        dtype = str(value.get("dtype", "")).strip().lower()
        modalities = frozenset(str(item).strip().lower() for item in value.get("modalities", []))
        if not _PROFILE_ID_PATTERN.fullmatch(profile_id):
            raise ValueError(f"Invalid Modal LLM profile id {profile_id!r}.")
        if not _REVISION_PATTERN.fullmatch(revision):
            raise ValueError(
                f"Modal LLM profile {profile_id!r} must pin an exact 40-character Git revision."
            )
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r} in Modal LLM profile {profile_id!r}.")
        unsupported_modalities = modalities - _SUPPORTED_MODALITIES
        if not modalities or unsupported_modalities:
            raise ValueError(
                f"Invalid modalities for Modal LLM profile {profile_id!r}: "
                f"{sorted(unsupported_modalities or modalities)!r}."
            )
        profile = cls(
            profile_id=profile_id,
            display_name=str(value.get("display_name", profile_id)).strip() or profile_id,
            repository=str(value.get("repository", "")).strip(),
            revision=revision,
            dtype=dtype,
            modalities=modalities,
            estimated_vram_gb=float(value.get("estimated_vram_gb", 0.0)),
            max_context_tokens=int(value.get("max_context_tokens", 0)),
            max_images=int(value.get("max_images", 0)),
            max_video_frames=int(value.get("max_video_frames", 0)),
            max_file_bytes=int(value.get("max_file_bytes", 0)),
            max_file_characters=int(value.get("max_file_characters", 0)),
            allow_mixed_image_video=bool(value.get("allow_mixed_image_video", False)),
            trust_remote_code=bool(value.get("trust_remote_code", False)),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        """Raise when a profile is unsafe or internally inconsistent."""
        if not self.repository or "/" not in self.repository:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has invalid repository {self.repository!r}."
            )
        if self.trust_remote_code:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} enables trust_remote_code; "
                "review and package custom model code instead."
            )
        positive_limits = {
            "estimated_vram_gb": self.estimated_vram_gb,
            "max_context_tokens": self.max_context_tokens,
            "max_images": self.max_images,
            "max_video_frames": self.max_video_frames,
            "max_file_bytes": self.max_file_bytes,
            "max_file_characters": self.max_file_characters,
        }
        invalid_limits = [name for name, limit in positive_limits.items() if limit <= 0]
        if invalid_limits:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has non-positive limits: {invalid_limits}."
            )

    def storage_relative_path(self) -> Path:
        """Return the stable volume-relative directory for this immutable snapshot."""
        return Path(LLM_MODEL_DIRECTORY_NAME) / self.profile_id / self.revision


def _default_profile_path() -> Path:
    """Return the checked-in profile registry path."""
    return Path(__file__).resolve().with_name(LLM_PROFILE_FILENAME)


@lru_cache(maxsize=4)
def load_llm_profiles(profile_path: Path | None = None) -> dict[str, LLMModelProfile]:
    """Load and validate the curated LLM profile registry."""
    resolved_path = (profile_path or _default_profile_path()).resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    raw_profiles = payload.get("profiles") if isinstance(payload, Mapping) else None
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"Modal LLM profile registry {resolved_path} has no profiles.")
    profiles: dict[str, LLMModelProfile] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"Modal LLM profile registry {resolved_path} contains a non-object entry.")
        profile = LLMModelProfile.from_mapping(raw_profile)
        if profile.profile_id in profiles:
            raise ValueError(f"Duplicate Modal LLM profile id {profile.profile_id!r}.")
        profiles[profile.profile_id] = profile
    logger.info("Loaded %d curated Modal LLM profile(s) from %s.", len(profiles), resolved_path)
    return profiles


def get_llm_profile(profile_id: str) -> LLMModelProfile:
    """Return one curated profile or raise a user-facing error."""
    normalized_id = profile_id.strip()
    try:
        return load_llm_profiles()[normalized_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Modal LLM profile {normalized_id!r}; choose one of "
            f"{sorted(load_llm_profiles())}."
        ) from exc


def llm_profile_options() -> list[str]:
    """Return stable profile ids for the ComfyUI combo widget."""
    return list(load_llm_profiles())


def llm_profile_ids_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Find curated LLM profiles referenced anywhere in a remote payload."""
    profile_ids: set[str] = set()

    def visit(value: Any) -> None:
        """Visit nested payload values and collect Modal LLM prompt nodes."""
        if isinstance(value, Mapping):
            if value.get("class_type") == MODAL_LLM_NODE_ID:
                inputs = value.get("inputs")
                if isinstance(inputs, Mapping):
                    profile_id = inputs.get("model_profile")
                    if isinstance(profile_id, str) and profile_id.strip():
                        get_llm_profile(profile_id)
                        profile_ids.add(profile_id.strip())
                    else:
                        raise ValueError(
                            "Modal LLM model_profile must be a fixed combo value so its pinned "
                            "snapshot can be staged on CPU before GPU allocation."
                        )
            for nested_value in value.values():
                visit(nested_value)
            return
        if isinstance(value, list | tuple):
            for nested_value in value:
                visit(nested_value)

    visit(payload)
    return tuple(sorted(profile_ids))


__all__ = [
    "LLM_MODEL_DIRECTORY_NAME",
    "LLMModelProfile",
    "MODAL_LLM_NODE_ID",
    "get_llm_profile",
    "llm_profile_ids_from_payload",
    "llm_profile_options",
    "load_llm_profiles",
]
