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
    from .llm_profiles import LLMModelProfile, get_llm_profile
else:  # pragma: no cover - the stable cloud entrypoint imports top-level modules.
    from llm_profiles import LLMModelProfile, get_llm_profile

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
        and (snapshot_path / "config.json").is_file()
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
        and (snapshot_path / "config.json").is_file()
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
            "token": os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            "allow_patterns": _SNAPSHOT_ALLOW_PATTERNS,
        }
        if progress_callback is not None:
            download_options["tqdm_class"] = _snapshot_tqdm_class(
                progress_callback
            )
        resolved_path = Path(snapshot_download(**download_options)).resolve()
        if resolved_path != snapshot_path.resolve():
            raise RuntimeError(
                f"Hugging Face staged {profile.profile_id!r} at unexpected path "
                f"{resolved_path}; expected {snapshot_path.resolve()}."
            )
        if not (snapshot_path / "config.json").is_file():
            raise RuntimeError(
                f"Staged Modal LLM profile {profile.profile_id!r} is missing "
                "config.json."
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


__all__ = [
    "StagedModelSnapshot",
    "LLMStagingProgress",
    "is_model_snapshot_staged",
    "model_snapshot_path",
    "stage_model_profile",
]
