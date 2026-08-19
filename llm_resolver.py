"""CPU-side Hugging Face inspection and immutable profile generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Protocol

from huggingface_hub.errors import HfHubHTTPError

if __package__:
    from .llm_compatibility import (
        LLM_COMPATIBILITY_POLICY_VERSION,
        LLM_PROFILE_SCHEMA_VERSION,
        resolve_compatibility,
    )
    from .llm_profiles import (
        LLMModelProfile,
        generated_profile_id,
        generated_profile_manifest_path,
    )
else:  # pragma: no cover - stable cloud entrypoint imports top-level modules.
    from llm_compatibility import (
        LLM_COMPATIBILITY_POLICY_VERSION,
        LLM_PROFILE_SCHEMA_VERSION,
        resolve_compatibility,
    )
    from llm_profiles import (
        LLMModelProfile,
        generated_profile_id,
        generated_profile_manifest_path,
    )

logger = logging.getLogger(__name__)

_MODEL_REFERENCE_PATTERN = re.compile(
    r"^(?P<repository>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:@(?P<revision>[A-Za-z0-9][A-Za-z0-9._/-]*))?$"
)
_EXACT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_MAX_DOWNLOAD_BYTES = 96 * 1024**3


class HuggingFaceApi(Protocol):
    """Describe the Hugging Face metadata method used by the resolver."""

    def model_info(self, repo_id: str, **kwargs: Any) -> Any:
        """Return repository metadata for one model revision."""


@dataclass(frozen=True)
class HuggingFaceModelReference:
    """Hold a validated repository and optional user-requested revision."""

    repository: str
    requested_revision: str | None

    @classmethod
    def parse(cls, value: str) -> "HuggingFaceModelReference":
        """Parse ``owner/model`` or ``owner/model@revision`` syntax."""
        normalized = value.strip()
        match = _MODEL_REFERENCE_PATTERN.fullmatch(normalized)
        if match is None:
            raise ValueError(
                "Modal LLM model must be a Hugging Face ID like 'owner/model' or "
                "'owner/model@revision'."
            )
        return cls(
            repository=match.group("repository"),
            requested_revision=match.group("revision"),
        )

    def display(self) -> str:
        """Return the normalized user-facing reference."""
        if self.requested_revision:
            return f"{self.repository}@{self.requested_revision}"
        return self.repository


@dataclass(frozen=True)
class ResolvedModelProfile:
    """Describe a generated immutable profile persisted to shared storage."""

    requested_reference: str
    profile: LLMModelProfile
    manifest_path: str
    manifest_created: bool
    security_scan_complete: bool


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from a Hugging Face object or test mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _token() -> str | None:
    """Return the first supported Hugging Face access-token environment value."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _repository_files(model_info: Any) -> tuple[tuple[str, int], ...]:
    """Return stable repository filenames and known sizes from model metadata."""
    siblings = _field(model_info, "siblings", ()) or ()
    files: list[tuple[str, int]] = []
    for sibling in siblings:
        filename = str(_field(sibling, "rfilename", "")).strip()
        if not filename:
            continue
        raw_size = _field(sibling, "size", 0) or 0
        files.append((filename, int(raw_size)))
    return tuple(sorted(files))


def _validate_repository_metadata(
    model_info: Any,
    *,
    max_download_bytes: int,
) -> tuple[int, bool]:
    """Reject unsafe, incomplete, or over-budget repositories before download."""
    files = _repository_files(model_info)
    filenames = {filename for filename, _ in files}
    if "config.json" not in filenames:
        raise ValueError("The Hugging Face model repository has no config.json.")
    safetensor_bytes = sum(
        size for filename, size in files if filename.endswith(".safetensors")
    )
    if safetensor_bytes <= 0:
        raise ValueError(
            "Modal LLM requires safetensors weights; pickle-only checkpoints "
            "are rejected."
        )
    if safetensor_bytes > max_download_bytes:
        raise ValueError(
            f"Model weights require {safetensor_bytes / 1024**3:.1f} GiB, exceeding "
            f"COMFY_MODAL_LLM_MAX_DOWNLOAD_GB={max_download_bytes / 1024**3:.1f}."
        )
    security = _field(model_info, "security_repo_status", None)
    if security is None:
        security = _field(model_info, "securityStatus", {})
    issues = (
        _field(security, "filesWithIssues", ()) or _field(security, "issues", ()) or ()
    )
    if issues:
        raise ValueError(
            f"Hugging Face security metadata reports {len(issues)} file issue(s); "
            "the model will not be staged."
        )
    scan_complete = bool(
        _field(security, "scansDone", _field(security, "scans_done", False))
    )
    if not scan_complete:
        logger.warning(
            "Hugging Face security scan is not complete for repository=%s revision=%s.",
            _field(model_info, "id", "unknown"),
            _field(model_info, "sha", "unknown"),
        )
    return safetensor_bytes, scan_complete


def _read_config(
    repository: str,
    revision: str,
    *,
    hf_hub_download: Callable[..., str],
) -> Mapping[str, Any]:
    """Download and decode only config.json during the compatibility phase."""
    config_path = Path(
        hf_hub_download(
            repo_id=repository,
            filename="config.json",
            revision=revision,
            token=_token(),
        )
    )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Unable to read Hugging Face config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Hugging Face config {config_path} must contain a JSON object."
        )
    return payload


def _canonical_profile_identity(profile_fields: Mapping[str, Any]) -> bytes:
    """Serialize profile-defining fields for content addressing."""
    return json.dumps(profile_fields, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _generated_profile(
    *,
    reference: HuggingFaceModelReference,
    revision: str,
    config: Mapping[str, Any],
    artifact_bytes: int,
) -> LLMModelProfile:
    """Build a content-addressed profile from reviewed compatibility metadata."""
    decision = resolve_compatibility(config, artifact_bytes=artifact_bytes)
    identity: dict[str, Any] = {
        "schema_version": LLM_PROFILE_SCHEMA_VERSION,
        "compatibility_policy_version": LLM_COMPATIBILITY_POLICY_VERSION,
        "repository": reference.repository,
        "revision": revision,
        "dtype": decision.dtype,
        "modalities": sorted(decision.modalities),
        "backend": decision.backend,
        "architecture": decision.architecture,
        "quantization_method": decision.quantization_method,
        "artifact_bytes": artifact_bytes,
        "advertised_context_tokens": decision.advertised_context_tokens,
        "max_context_tokens": decision.default_context_tokens,
        "estimated_vram_gb": decision.estimated_vram_gb,
        "backend_options": dict(decision.backend_options),
        "runtime_requirements": list(decision.runtime_requirements),
    }
    digest = hashlib.sha256(_canonical_profile_identity(identity)).hexdigest()
    profile_id = generated_profile_id(digest)
    return LLMModelProfile.from_mapping(
        {
            **identity,
            "id": profile_id,
            "display_name": reference.repository.rsplit("/", maxsplit=1)[-1],
            "source": "generated",
            "profile_digest": digest,
            "max_images": 8,
            "max_video_frames": 32,
            "max_file_bytes": 32 * 1024**2,
            "max_file_characters": 200000,
            "allow_mixed_image_video": False,
            "trust_remote_code": False,
        }
    )


def _write_generated_manifest(
    storage_root: str | Path,
    requested_reference: str,
    profile: LLMModelProfile,
    *,
    security_scan_complete: bool,
) -> tuple[Path, bool]:
    """Persist one immutable manifest atomically, reusing an identical existing file."""
    manifest_path = generated_profile_manifest_path(storage_root, profile.profile_id)
    manifest_payload = {
        "schema_version": LLM_PROFILE_SCHEMA_VERSION,
        "compatibility_policy_version": LLM_COMPATIBILITY_POLICY_VERSION,
        "requested_reference": requested_reference,
        "resolved_at_unix": time.time(),
        "security_scan_complete": security_scan_complete,
        "profile": profile.to_mapping(),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_profile = (
            existing.get("profile") if isinstance(existing, Mapping) else None
        )
        if existing_profile != manifest_payload["profile"]:
            raise RuntimeError(
                f"Generated profile collision at {manifest_path}; refusing to "
                "overwrite it."
            )
        return manifest_path, False
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path, True


def resolve_model_profile(
    model_reference: str,
    storage_root: str | Path,
    *,
    api: HuggingFaceApi | None = None,
    hf_hub_download: Callable[..., str] | None = None,
    max_download_bytes: int | None = None,
) -> ResolvedModelProfile:
    """Inspect one Hugging Face ID and persist its immutable generated profile."""
    reference = HuggingFaceModelReference.parse(model_reference)
    if api is None or hf_hub_download is None:
        from huggingface_hub import HfApi, hf_hub_download as download_file

        api = api or HfApi()
        hf_hub_download = hf_hub_download or download_file
    token = _token()
    try:
        model_info = api.model_info(
            reference.repository,
            revision=reference.requested_revision,
            files_metadata=True,
            securityStatus=True,
            token=token,
        )
    except (HfHubHTTPError, OSError, ValueError) as exc:
        raise ValueError(
            f"Unable to access Hugging Face model {reference.display()!r}. For "
            "gated or "
            "private models, add HF_TOKEN to the Modal Secret: "
            f"{exc}"
        ) from exc
    revision = str(_field(model_info, "sha", "")).lower()
    if not _EXACT_REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            f"Hugging Face did not resolve {reference.display()!r} to an exact commit."
        )
    resolved_max_bytes = max_download_bytes
    if resolved_max_bytes is None:
        configured_gb = float(os.getenv("COMFY_MODAL_LLM_MAX_DOWNLOAD_GB", "96"))
        if configured_gb <= 0:
            raise ValueError("COMFY_MODAL_LLM_MAX_DOWNLOAD_GB must be positive.")
        resolved_max_bytes = int(configured_gb * 1024**3)
    artifact_bytes, scan_complete = _validate_repository_metadata(
        model_info,
        max_download_bytes=resolved_max_bytes,
    )
    config = _read_config(
        reference.repository,
        revision,
        hf_hub_download=hf_hub_download,
    )
    profile = _generated_profile(
        reference=reference,
        revision=revision,
        config=config,
        artifact_bytes=artifact_bytes,
    )
    manifest_path, manifest_created = _write_generated_manifest(
        storage_root,
        reference.display(),
        profile,
        security_scan_complete=scan_complete,
    )
    logger.info(
        "Resolved Modal LLM model=%s revision=%s profile=%s backend=%s "
        "weights_gib=%.2f manifest_created=%s.",
        reference.display(),
        revision,
        profile.profile_id,
        profile.backend,
        artifact_bytes / 1024**3,
        manifest_created,
    )
    return ResolvedModelProfile(
        requested_reference=reference.display(),
        profile=profile,
        manifest_path=str(manifest_path),
        manifest_created=manifest_created,
        security_scan_complete=scan_complete,
    )


__all__ = [
    "HuggingFaceModelReference",
    "ResolvedModelProfile",
    "resolve_model_profile",
]
