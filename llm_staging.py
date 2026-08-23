"""CPU-side Hugging Face snapshot staging for Modal LLM profiles."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

if __package__:
    from .llm_profiles import LLMModelProfile, get_llm_profile, load_llm_profiles
else:  # pragma: no cover - the stable cloud entrypoint imports top-level modules.
    from llm_profiles import LLMModelProfile, get_llm_profile, load_llm_profiles

logger = logging.getLogger(__name__)

_COMPLETE_MARKER_FILENAME = ".comfy-modal-llm-complete.json"
_DEFAULT_LEASE_TIMEOUT_SECONDS = 7200.0
_LEASE_POLL_SECONDS = 2.0
_SNAPSHOT_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.model",
    "*.tiktoken",
    "added_tokens.json",
    "chat_template*",
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor*",
    "special_tokens_map.json",
    "tokenizer*",
    "video_preprocessor_config.json",
    "vocab*",
)
_TOKENIZER_ALLOW_PATTERNS = tuple(
    pattern for pattern in _SNAPSHOT_ALLOW_PATTERNS if "safetensors" not in pattern
)


@dataclass(frozen=True)
class StagedModelSnapshot:
    """Describe a model snapshot made visible on the shared Modal Volume."""

    profile_id: str
    repository: str
    revision: str
    path: str
    downloaded: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class LLMStagingProgress:
    """Describe one numeric Hugging Face snapshot-staging update."""

    stage: str
    message: str
    value: float | None = None
    maximum: float | None = None
    unit: str | None = None
    indeterminate: bool = False


StagingProgressCallback = Callable[[LLMStagingProgress], None]


@dataclass(frozen=True)
class ResolvedStagedModelProfile:
    """Describe one resolved immutable profile and its staged snapshot."""

    requested_reference: str
    profile_id: str
    repository: str
    revision: str
    backend: str
    quantization_method: str
    artifact_bytes: int
    manifest_path: str | None
    manifest_created: bool
    security_scan_complete: bool
    path: str
    downloaded: bool
    resolve_elapsed_seconds: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible staging metadata."""
        return asdict(self)


def _snapshot_tqdm_class(
    progress_callback: StagingProgressCallback,
) -> type[Any]:
    """Build a tqdm subclass that forwards Hugging Face file progress."""
    from tqdm.auto import tqdm

    class SnapshotProgressTqdm(tqdm):
        """Mirror snapshot_download's aggregate file bar to the caller."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the Hugging Face bar and report its initial state."""
            super().__init__(*args, **kwargs)
            self._report_progress()

        def update(self, n: float | None = 1) -> bool | None:
            """Forward every aggregate file-count update."""
            updated = super().update(n)
            self._report_progress()
            return updated

        def _report_progress(self) -> None:
            """Publish the current tqdm count with compact user-facing copy."""
            maximum = float(self.total) if self.total is not None else None
            value = float(self.n)
            description = str(
                getattr(self, "desc", None) or "Downloading model snapshot"
            ).strip()
            progress_callback(
                LLMStagingProgress(
                    stage="download",
                    message=description,
                    value=value,
                    maximum=maximum,
                    unit=str(getattr(self, "unit", None) or "files"),
                    indeterminate=maximum is None,
                )
            )

    return SnapshotProgressTqdm


def model_snapshot_path(storage_root: str | Path, profile: LLMModelProfile) -> Path:
    """Return the shared path, reusing a valid legacy profile path in place."""
    resolved_root = Path(storage_root).resolve()
    canonical_path = resolved_root / profile.storage_relative_path()
    if canonical_path.exists():
        return canonical_path
    model_root = resolved_root / "llm_models"
    for legacy_path in sorted(model_root.glob(f"*/{profile.revision}")):
        if legacy_path != canonical_path and _legacy_snapshot_matches(
            legacy_path,
            profile,
        ):
            logger.info(
                "Reusing legacy Modal LLM weight path repository=%s revision=%s "
                "path=%s.",
                profile.repository,
                profile.revision,
                legacy_path,
            )
            return legacy_path
    return canonical_path


def _marker_path(snapshot_path: Path) -> Path:
    """Return the completion marker path for one staged snapshot."""
    return snapshot_path / _COMPLETE_MARKER_FILENAME


def _required_model_path(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> Path:
    """Return the profile's required model artifact inside one snapshot."""
    model_filename = profile.backend_option("model_filename")
    if model_filename is None:
        return snapshot_path / "config.json"
    return snapshot_path / str(model_filename)


def _tokenizer_source(profile: LLMModelProfile) -> tuple[str, str] | None:
    """Return an optional separately pinned tokenizer snapshot source."""
    repository = profile.backend_option("tokenizer_repository")
    revision = profile.backend_option("tokenizer_revision")
    if repository is None and revision is None:
        return None
    normalized_repository = str(repository or "").strip()
    normalized_revision = str(revision or "").strip().lower()
    if "/" not in normalized_repository or len(normalized_revision) != 40:
        raise ValueError(
            f"Modal LLM profile {profile.profile_id!r} has an invalid separately "
            "pinned tokenizer source."
        )
    return normalized_repository, normalized_revision


def _snapshot_has_required_artifacts(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> bool:
    """Return whether model and optional tokenizer artifacts are complete."""
    if not _required_model_path(snapshot_path, profile).is_file():
        return False
    return (
        _tokenizer_source(profile) is None
        or (snapshot_path / "tokenizer_config.json").is_file()
    )


def _model_allow_patterns(profile: LLMModelProfile) -> tuple[str, ...]:
    """Return a bounded allowlist for this profile's primary repository."""
    model_filename = profile.backend_option("model_filename")
    if model_filename is None:
        return _SNAPSHOT_ALLOW_PATTERNS
    return (*_TOKENIZER_ALLOW_PATTERNS, str(model_filename))


def _read_marker(snapshot_path: Path) -> dict[str, Any] | None:
    """Return a valid completion marker, if present."""
    marker_path = _marker_path(snapshot_path)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def is_model_snapshot_staged(
    storage_root: str | Path, profile: LLMModelProfile
) -> bool:
    """Return whether the requested immutable snapshot is completely staged."""
    snapshot_path = model_snapshot_path(storage_root, profile)
    marker = _read_marker(snapshot_path)
    return bool(
        marker
        and marker.get("repository") == profile.repository
        and marker.get("revision") == profile.revision
        and _snapshot_has_required_artifacts(snapshot_path, profile)
    )


def _write_marker(snapshot_path: Path, profile: LLMModelProfile) -> None:
    """Atomically record that an immutable model snapshot is complete."""
    marker_path = _marker_path(snapshot_path)
    temporary_path = marker_path.with_suffix(".tmp")
    marker_payload = {
        "marker_version": 2,
        "profile_id": profile.profile_id,
        "repository": profile.repository,
        "revision": profile.revision,
        "completed_at_unix": time.time(),
    }
    temporary_path.write_text(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_path, marker_path)


def _legacy_snapshot_matches(
    snapshot_path: Path,
    profile: LLMModelProfile,
) -> bool:
    """Return whether an old profile-keyed directory has these exact weights."""
    marker = _read_marker(snapshot_path)
    return bool(
        marker
        and marker.get("repository") == profile.repository
        and marker.get("revision") == profile.revision
        and _snapshot_has_required_artifacts(snapshot_path, profile)
    )


@contextmanager
def _snapshot_lease(snapshot_path: Path) -> Iterator[None]:
    """Serialize a snapshot download and recover an abandoned lease."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path = snapshot_path.parent / f".{snapshot_path.name}.download.lock"
    timeout_seconds = float(
        os.getenv("COMFY_MODAL_LLM_STAGE_LEASE_TIMEOUT_SECONDS", "7200")
    )
    if timeout_seconds <= 0:
        raise ValueError(
            "COMFY_MODAL_LLM_STAGE_LEASE_TIMEOUT_SECONDS must be positive."
        )
    started_at = time.monotonic()
    acquired = False
    while not acquired:
        try:
            descriptor = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                lease_age = time.time() - lease_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lease_age >= _DEFAULT_LEASE_TIMEOUT_SECONDS:
                logger.warning(
                    "Removing abandoned Modal LLM staging lease %s.", lease_path
                )
                try:
                    lease_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - started_at >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting {timeout_seconds:.0f}s for model staging lease "
                    f"{lease_path}."
                )
            time.sleep(_LEASE_POLL_SECONDS)
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as lease_file:
                lease_file.write(f"pid={os.getpid()} acquired_at={time.time()}\n")
            acquired = True
    try:
        yield
    finally:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass


def stage_model_profile(
    profile_id: str,
    storage_root: str | Path,
    *,
    profile: LLMModelProfile | None = None,
    snapshot_download: Callable[..., str] | None = None,
    progress_callback: StagingProgressCallback | None = None,
) -> StagedModelSnapshot:
    """Download one pinned curated or generated snapshot on a CPU worker."""
    profile = profile or get_llm_profile(profile_id, storage_root=storage_root)
    if profile.profile_id != profile_id:
        raise ValueError(
            f"Stage request id {profile_id!r} does not match profile "
            f"{profile.profile_id!r}."
        )
    snapshot_path = model_snapshot_path(storage_root, profile)
    started_at = time.perf_counter()
    if is_model_snapshot_staged(storage_root, profile):
        logger.info(
            "Reusing staged Modal LLM profile=%s revision=%s path=%s.",
            profile.profile_id,
            profile.revision,
            snapshot_path,
        )
        if progress_callback is not None:
            progress_callback(
                LLMStagingProgress(
                    stage="cached",
                    message="Model snapshot already staged",
                    value=1,
                    maximum=1,
                    unit="model",
                )
            )
        return StagedModelSnapshot(
            profile_id=profile.profile_id,
            repository=profile.repository,
            revision=profile.revision,
            path=str(snapshot_path),
            downloaded=False,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    with _snapshot_lease(snapshot_path):
        if is_model_snapshot_staged(storage_root, profile):
            return StagedModelSnapshot(
                profile_id=profile.profile_id,
                repository=profile.repository,
                revision=profile.revision,
                path=str(snapshot_path),
                downloaded=False,
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if snapshot_download is None:
            from huggingface_hub import (
                snapshot_download as huggingface_snapshot_download,
            )

            snapshot_download = huggingface_snapshot_download
        snapshot_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Staging Modal LLM profile=%s repository=%s revision=%s to %s.",
            profile.profile_id,
            profile.repository,
            profile.revision,
            snapshot_path,
        )
        download_options: dict[str, Any] = {
            "repo_id": profile.repository,
            "revision": profile.revision,
            "local_dir": str(snapshot_path),
            "token": os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            "allow_patterns": _model_allow_patterns(profile),
        }
        if progress_callback is not None:
            download_options["tqdm_class"] = _snapshot_tqdm_class(progress_callback)
        resolved_path = Path(snapshot_download(**download_options)).resolve()
        if resolved_path != snapshot_path.resolve():
            raise RuntimeError(
                f"Hugging Face staged {profile.profile_id!r} at unexpected path "
                f"{resolved_path}; expected {snapshot_path.resolve()}."
            )
        tokenizer_source = _tokenizer_source(profile)
        if tokenizer_source is not None:
            tokenizer_repository, tokenizer_revision = tokenizer_source
            tokenizer_options: dict[str, Any] = {
                "repo_id": tokenizer_repository,
                "revision": tokenizer_revision,
                "local_dir": str(snapshot_path),
                "token": download_options["token"],
                "allow_patterns": _TOKENIZER_ALLOW_PATTERNS,
            }
            if progress_callback is not None:
                tokenizer_options["tqdm_class"] = _snapshot_tqdm_class(
                    progress_callback
                )
            tokenizer_path = Path(snapshot_download(**tokenizer_options)).resolve()
            if tokenizer_path != snapshot_path.resolve():
                raise RuntimeError(
                    f"Hugging Face staged tokenizer for {profile.profile_id!r} at "
                    f"unexpected path {tokenizer_path}; expected "
                    f"{snapshot_path.resolve()}."
                )
        if not _snapshot_has_required_artifacts(snapshot_path, profile):
            raise RuntimeError(
                f"Staged Modal LLM profile {profile.profile_id!r} is missing its "
                "required model or tokenizer artifacts."
            )
        _write_marker(snapshot_path, profile)
    result = StagedModelSnapshot(
        profile_id=profile.profile_id,
        repository=profile.repository,
        revision=profile.revision,
        path=str(snapshot_path),
        downloaded=True,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    logger.info("Completed Modal LLM staging: %s", asdict(result))
    return result


def _resolve_profile_for_staging(
    model_reference: str,
    storage_root: str | Path,
) -> tuple[LLMModelProfile, str | None, bool, bool]:
    """Resolve one curated, generated, or Hugging Face model reference."""
    curated_profiles = load_llm_profiles()
    if model_reference in curated_profiles:
        return curated_profiles[model_reference], None, False, True
    if model_reference.startswith("hf-"):
        return (
            get_llm_profile(model_reference, storage_root=storage_root),
            None,
            False,
            True,
        )
    if __package__:
        from .llm_resolver import resolve_model_profile
    else:  # pragma: no cover - stable cloud entrypoint imports top-level modules.
        from llm_resolver import resolve_model_profile

    resolved = resolve_model_profile(model_reference, storage_root)
    return (
        resolved.profile,
        resolved.manifest_path,
        resolved.manifest_created,
        resolved.security_scan_complete,
    )


def resolve_and_stage_model_references(
    model_references: list[str],
    storage_root: str | Path,
    *,
    progress_callback: StagingProgressCallback | None = None,
) -> list[ResolvedStagedModelProfile]:
    """Resolve and stage model references on any CPU-backed remote worker."""
    results: list[ResolvedStagedModelProfile] = []
    for model_reference in model_references:
        if progress_callback is not None:
            progress_callback(
                LLMStagingProgress(
                    stage="resolve",
                    message=f"Inspecting {model_reference}",
                    indeterminate=True,
                )
            )
        resolve_started_at = time.perf_counter()
        (
            profile,
            manifest_path,
            manifest_created,
            scan_complete,
        ) = _resolve_profile_for_staging(model_reference, storage_root)
        resolve_elapsed_seconds = time.perf_counter() - resolve_started_at
        staged = stage_model_profile(
            profile.profile_id,
            storage_root,
            profile=profile,
            progress_callback=progress_callback,
        )
        results.append(
            ResolvedStagedModelProfile(
                requested_reference=model_reference,
                profile_id=staged.profile_id,
                repository=staged.repository,
                revision=staged.revision,
                backend=profile.backend,
                quantization_method=profile.quantization_method,
                artifact_bytes=profile.artifact_bytes,
                manifest_path=manifest_path,
                manifest_created=manifest_created,
                security_scan_complete=scan_complete,
                path=staged.path,
                downloaded=staged.downloaded,
                resolve_elapsed_seconds=resolve_elapsed_seconds,
                elapsed_seconds=staged.elapsed_seconds,
            )
        )
    return results


__all__ = [
    "LLMStagingProgress",
    "ResolvedStagedModelProfile",
    "StagedModelSnapshot",
    "is_model_snapshot_staged",
    "model_snapshot_path",
    "resolve_and_stage_model_references",
    "stage_model_profile",
]
