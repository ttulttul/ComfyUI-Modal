"""Immutable curated and generated profiles for Modal LLM inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping

if __package__:
    from .llm_compatibility import (
        LLM_COMPATIBILITY_POLICY_VERSION,
        LLM_PROFILE_SCHEMA_VERSION,
        LOCAL_MLX_VLM_VERSION,
        LLMExecutionTarget,
    )
else:  # pragma: no cover - stable cloud entrypoint imports top-level modules.
    from llm_compatibility import (
        LLM_COMPATIBILITY_POLICY_VERSION,
        LLM_PROFILE_SCHEMA_VERSION,
        LOCAL_MLX_VLM_VERSION,
        LLMExecutionTarget,
    )

logger = logging.getLogger(__name__)

MODAL_LLM_NODE_ID = "ModalLLM"
LLM_PROFILE_FILENAME = "llm_profiles.json"
LLM_MODEL_DIRECTORY_NAME = "llm_models"
LLM_GENERATED_PROFILE_DIRECTORY_NAME = "llm_generated_profiles"
_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTED_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_SUPPORTED_MODALITIES = frozenset({"text", "image", "video", "file"})
_SUPPORTED_BACKENDS = frozenset({"transformers", "vllm", "mlx_vlm", "llama_cpp_server"})
_SUPPORTED_EXECUTION_TARGETS = frozenset({"modal", "local_apple"})
_SUPPORTED_REASONING_PARSERS = frozenset({"", "none", "qwen3"})
_PROFILE_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LLMModelProfile:
    """Describe one immutable Hugging Face model and its reviewed runtime policy."""

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
    schema_version: int = 1
    source: str = "curated"
    profile_digest: str = ""
    compatibility_policy_version: int = LLM_COMPATIBILITY_POLICY_VERSION
    backend: str = "transformers"
    architecture: str = ""
    quantization_method: str = "none"
    artifact_bytes: int = 0
    advertised_context_tokens: int = 0
    reasoning_parser: str = ""
    backend_options: tuple[tuple[str, str | int | float | bool], ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    execution_target: str = "modal"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LLMModelProfile":
        """Validate and construct one profile from decoded JSON data."""
        profile_id = str(value.get("id", "")).strip()
        revision = str(value.get("revision", "")).strip().lower()
        dtype = str(value.get("dtype", "")).strip().lower()
        modalities = frozenset(
            str(item).strip().lower() for item in value.get("modalities", [])
        )
        if not _PROFILE_ID_PATTERN.fullmatch(profile_id):
            raise ValueError(f"Invalid Modal LLM profile id {profile_id!r}.")
        if not _REVISION_PATTERN.fullmatch(revision):
            raise ValueError(
                f"Modal LLM profile {profile_id!r} must pin an exact 40-character "
                "Git revision."
            )
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype!r} in Modal LLM profile {profile_id!r}."
            )
        unsupported_modalities = modalities - _SUPPORTED_MODALITIES
        if not modalities or unsupported_modalities:
            raise ValueError(
                f"Invalid modalities for Modal LLM profile {profile_id!r}: "
                f"{sorted(unsupported_modalities or modalities)!r}."
            )
        profile = cls(
            profile_id=profile_id,
            display_name=str(value.get("display_name", profile_id)).strip()
            or profile_id,
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
            schema_version=int(value.get("schema_version", 1)),
            source=str(value.get("source", "curated")).strip().lower(),
            profile_digest=str(value.get("profile_digest", "")).strip().lower(),
            compatibility_policy_version=int(
                value.get(
                    "compatibility_policy_version", LLM_COMPATIBILITY_POLICY_VERSION
                )
            ),
            backend=str(value.get("backend", "transformers")).strip().lower(),
            architecture=str(value.get("architecture", "")).strip(),
            quantization_method=str(value.get("quantization_method", "none"))
            .strip()
            .lower(),
            artifact_bytes=int(value.get("artifact_bytes", 0)),
            advertised_context_tokens=int(
                value.get(
                    "advertised_context_tokens", value.get("max_context_tokens", 0)
                )
            ),
            reasoning_parser=str(value.get("reasoning_parser", "")).strip().lower(),
            backend_options=tuple(
                sorted(
                    (str(key), option)
                    for key, option in _mapping_value(value, "backend_options").items()
                )
            ),
            runtime_requirements=tuple(
                str(requirement)
                for requirement in value.get("runtime_requirements", [])
            ),
            execution_target=str(value.get("execution_target", "modal"))
            .strip()
            .lower(),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        """Raise when a profile is unsafe or internally inconsistent."""
        if not self.repository or "/" not in self.repository:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has invalid repository "
                f"{self.repository!r}."
            )
        if self.trust_remote_code:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} enables trust_remote_code; "
                "review and package custom model code instead."
            )
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has unsupported backend "
                f"{self.backend!r}."
            )
        if self.execution_target not in _SUPPORTED_EXECUTION_TARGETS:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has unsupported execution "
                f"target {self.execution_target!r}."
            )
        if self.reasoning_parser not in _SUPPORTED_REASONING_PARSERS:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has unsupported reasoning "
                f"parser {self.reasoning_parser!r}."
            )
        model_filename = self.backend_option("model_filename")
        if model_filename is not None:
            normalized_filename = str(model_filename).strip()
            if (
                not normalized_filename
                or Path(normalized_filename).name != normalized_filename
            ):
                raise ValueError(
                    f"Modal LLM profile {self.profile_id!r} has unsafe model "
                    f"filename {model_filename!r}."
                )
        if self.source not in {"curated", "generated"}:
            raise ValueError(
                f"Modal LLM profile {self.profile_id!r} has invalid source "
                f"{self.source!r}."
            )
        if self.source == "generated":
            if self.schema_version != LLM_PROFILE_SCHEMA_VERSION:
                raise ValueError(
                    f"Generated profile {self.profile_id!r} uses schema "
                    f"{self.schema_version}; "
                    f"runtime requires schema {LLM_PROFILE_SCHEMA_VERSION}."
                )
            if not _PROFILE_DIGEST_PATTERN.fullmatch(self.profile_digest):
                raise ValueError(
                    f"Generated profile {self.profile_id!r} has an invalid "
                    "content digest."
                )
            if self.profile_id != generated_profile_id(self.profile_digest):
                raise ValueError(
                    f"Generated profile id {self.profile_id!r} does not match its "
                    "content digest."
                )
            if self.compatibility_policy_version != LLM_COMPATIBILITY_POLICY_VERSION:
                raise ValueError(
                    f"Generated profile {self.profile_id!r} uses compatibility policy "
                    f"{self.compatibility_policy_version}; runtime requires "
                    f"{LLM_COMPATIBILITY_POLICY_VERSION}."
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
                f"Modal LLM profile {self.profile_id!r} has non-positive limits: "
                f"{invalid_limits}."
            )

    def storage_relative_path(self) -> Path:
        """Return this repository revision's runtime-independent weight path."""
        repository_digest = hashlib.sha256(self.repository.encode("utf-8")).hexdigest()
        return (
            Path(LLM_MODEL_DIRECTORY_NAME) / f"repo-{repository_digest}" / self.revision
        )

    def backend_option(self, name: str, default: Any = None) -> Any:
        """Return one immutable backend option by name."""
        return dict(self.backend_options).get(name, default)

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable JSON representation used by generated manifests."""
        mapping = {
            "id": self.profile_id,
            "display_name": self.display_name,
            "repository": self.repository,
            "revision": self.revision,
            "dtype": self.dtype,
            "modalities": sorted(self.modalities),
            "estimated_vram_gb": self.estimated_vram_gb,
            "max_context_tokens": self.max_context_tokens,
            "max_images": self.max_images,
            "max_video_frames": self.max_video_frames,
            "max_file_bytes": self.max_file_bytes,
            "max_file_characters": self.max_file_characters,
            "allow_mixed_image_video": self.allow_mixed_image_video,
            "trust_remote_code": self.trust_remote_code,
            "schema_version": self.schema_version,
            "source": self.source,
            "profile_digest": self.profile_digest,
            "compatibility_policy_version": self.compatibility_policy_version,
            "backend": self.backend,
            "architecture": self.architecture,
            "quantization_method": self.quantization_method,
            "artifact_bytes": self.artifact_bytes,
            "advertised_context_tokens": self.advertised_context_tokens,
            "reasoning_parser": self.reasoning_parser,
            "backend_options": dict(self.backend_options),
            "runtime_requirements": list(self.runtime_requirements),
        }
        if self.execution_target != "modal":
            mapping["execution_target"] = self.execution_target
        return mapping


def profile_for_execution_target(
    profile: LLMModelProfile,
    execution_target: LLMExecutionTarget,
) -> LLMModelProfile:
    """Return a curated profile adapted to one execution target."""
    if profile.source == "generated":
        if profile.execution_target != execution_target:
            raise ValueError(
                f"Generated LLM profile {profile.profile_id!r} targets "
                f"{profile.execution_target!r}, not {execution_target!r}. Resolve the "
                "original Hugging Face model reference for this execution target."
            )
        return profile
    if execution_target == "modal":
        return profile
    adapted = replace(
        profile,
        backend="mlx_vlm",
        execution_target="local_apple",
        runtime_requirements=(f"mlx-vlm=={LOCAL_MLX_VLM_VERSION}",),
    )
    adapted.validate()
    return adapted


def _mapping_value(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one nested mapping or reject a malformed profile field."""
    nested = value.get(name, {})
    if not isinstance(nested, Mapping):
        raise ValueError(f"Modal LLM profile field {name!r} must be an object.")
    return nested


def generated_profile_id(profile_digest: str) -> str:
    """Return the unambiguous identifier for one content-addressed profile."""
    if not _PROFILE_DIGEST_PATTERN.fullmatch(profile_digest):
        raise ValueError(
            "Generated Modal LLM profile digest must be 64 lowercase hex characters."
        )
    return f"hf-{profile_digest}"


def generated_profile_manifest_path(storage_root: str | Path, profile_id: str) -> Path:
    """Return the authoritative Volume path for one generated profile manifest."""
    return (
        Path(storage_root).resolve()
        / LLM_GENERATED_PROFILE_DIRECTORY_NAME
        / profile_id
        / "profile.json"
    )


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
            raise ValueError(
                f"Modal LLM profile registry {resolved_path} contains a "
                "non-object entry."
            )
        profile = LLMModelProfile.from_mapping(raw_profile)
        if profile.profile_id in profiles:
            raise ValueError(f"Duplicate Modal LLM profile id {profile.profile_id!r}.")
        profiles[profile.profile_id] = profile
    logger.info(
        "Loaded %d curated Modal LLM profile(s) from %s.", len(profiles), resolved_path
    )
    return profiles


def get_llm_profile(
    profile_id: str,
    *,
    storage_root: str | Path | None = None,
) -> LLMModelProfile:
    """Return one curated or Volume-backed generated immutable profile."""
    normalized_id = profile_id.strip()
    try:
        return load_llm_profiles()[normalized_id]
    except KeyError as exc:
        if normalized_id.startswith("hf-") and storage_root is not None:
            manifest_path = generated_profile_manifest_path(storage_root, normalized_id)
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            except (json.JSONDecodeError, OSError) as manifest_exc:
                raise ValueError(
                    f"Generated Modal LLM profile {normalized_id!r} is unreadable at "
                    f"{manifest_path}: {manifest_exc}"
                ) from manifest_exc
            else:
                profile_payload = (
                    payload.get("profile") if isinstance(payload, Mapping) else None
                )
                if not isinstance(profile_payload, Mapping):
                    raise ValueError(
                        f"Generated Modal LLM manifest {manifest_path} has no "
                        "profile object."
                    )
                return LLMModelProfile.from_mapping(profile_payload)
        raise ValueError(
            f"Unknown Modal LLM profile {normalized_id!r}. Use a curated profile "
            "or let the "
            "CPU resolver turn a Hugging Face model ID into a generated profile first."
        ) from exc


def llm_profile_options() -> list[str]:
    """Return stable profile ids for the ComfyUI combo widget."""
    return list(load_llm_profiles())


def llm_model_references_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Find fixed curated IDs or Hugging Face references in a remote payload."""
    model_references: set[str] = set()

    def visit(value: Any) -> None:
        """Visit nested payload values and collect Modal LLM model inputs."""
        if isinstance(value, Mapping):
            if value.get("class_type") == MODAL_LLM_NODE_ID:
                inputs = value.get("inputs")
                if isinstance(inputs, Mapping):
                    model_reference = inputs.get("model_profile")
                    if isinstance(model_reference, str) and model_reference.strip():
                        model_references.add(model_reference.strip())
                    else:
                        raise ValueError(
                            "Modal LLM model must be a fixed string so it can be "
                            "resolved and staged on CPU before GPU allocation."
                        )
            for nested_value in value.values():
                visit(nested_value)
            return
        if isinstance(value, list | tuple):
            for nested_value in value:
                visit(nested_value)

    visit(payload)
    return tuple(sorted(model_references))


def rewrite_llm_model_references(
    payload: Mapping[str, Any],
    profile_ids_by_reference: Mapping[str, str],
) -> None:
    """Replace user-facing model references with generated immutable profile IDs."""

    def visit(value: Any) -> None:
        """Rewrite matching Modal LLM inputs anywhere in the payload."""
        if isinstance(value, Mapping):
            if value.get("class_type") == MODAL_LLM_NODE_ID:
                inputs = value.get("inputs")
                if isinstance(inputs, dict):
                    model_reference = inputs.get("model_profile")
                    if isinstance(model_reference, str):
                        replacement = profile_ids_by_reference.get(
                            model_reference.strip()
                        )
                        if replacement:
                            inputs["model_profile"] = replacement
            for nested_value in value.values():
                visit(nested_value)
            return
        if isinstance(value, list | tuple):
            for nested_value in value:
                visit(nested_value)

    visit(payload)


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
                            "Modal LLM model_profile must be a fixed combo value so "
                            "its pinned "
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
    "LLM_GENERATED_PROFILE_DIRECTORY_NAME",
    "LLMModelProfile",
    "MODAL_LLM_NODE_ID",
    "get_llm_profile",
    "generated_profile_id",
    "generated_profile_manifest_path",
    "llm_model_references_from_payload",
    "llm_profile_ids_from_payload",
    "llm_profile_options",
    "load_llm_profiles",
    "profile_for_execution_target",
    "rewrite_llm_model_references",
]
