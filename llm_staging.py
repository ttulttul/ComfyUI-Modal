"""CPU-side Hugging Face snapshot staging for Modal LLM profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

if __package__:
    from .llm_profiles import LLMModelProfile, get_llm_profile
else:  # pragma: no cover - the stable cloud entrypoint imports top-level modules.
    from llm_profiles import LLMModelProfile, get_llm_profile

logger = logging.getLogger(__name__)

_COMPLETE_MARKER_FILENAME = ".comfy-modal-llm-complete.json"


@dataclass(frozen=True)
class StagedModelSnapshot:
    """Describe a model snapshot made visible on the shared Modal Volume."""

    profile_id: str
    repository: str
    revision: str
    path: str
    downloaded: bool
    elapsed_seconds: float


def model_snapshot_path(storage_root: str | Path, profile: LLMModelProfile) -> Path:
    """Return the absolute storage path for one profile's immutable snapshot."""
    return Path(storage_root).resolve() / profile.storage_relative_path()


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


def is_model_snapshot_staged(storage_root: str | Path, profile: LLMModelProfile) -> bool:
    """Return whether the requested immutable snapshot is completely staged."""
    snapshot_path = model_snapshot_path(storage_root, profile)
    marker = _read_marker(snapshot_path)
    return bool(
        marker
        and marker.get("profile_id") == profile.profile_id
        and marker.get("repository") == profile.repository
        and marker.get("revision") == profile.revision
        and (snapshot_path / "config.json").is_file()
    )


def _write_marker(snapshot_path: Path, profile: LLMModelProfile) -> None:
    """Atomically record that an immutable model snapshot is complete."""
    marker_path = _marker_path(snapshot_path)
    temporary_path = marker_path.with_suffix(".tmp")
    marker_payload = {
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


def stage_model_profile(
    profile_id: str,
    storage_root: str | Path,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> StagedModelSnapshot:
    """Download one pinned profile snapshot to shared storage on a CPU worker."""
    profile = get_llm_profile(profile_id)
    snapshot_path = model_snapshot_path(storage_root, profile)
    started_at = time.perf_counter()
    if is_model_snapshot_staged(storage_root, profile):
        logger.info(
            "Reusing staged Modal LLM profile=%s revision=%s path=%s.",
            profile.profile_id,
            profile.revision,
            snapshot_path,
        )
        return StagedModelSnapshot(
            profile_id=profile.profile_id,
            repository=profile.repository,
            revision=profile.revision,
            path=str(snapshot_path),
            downloaded=False,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    if snapshot_download is None:
        from huggingface_hub import snapshot_download as huggingface_snapshot_download

        snapshot_download = huggingface_snapshot_download
    snapshot_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Staging Modal LLM profile=%s repository=%s revision=%s to %s.",
        profile.profile_id,
        profile.repository,
        profile.revision,
        snapshot_path,
    )
    resolved_path = Path(
        snapshot_download(
            repo_id=profile.repository,
            revision=profile.revision,
            local_dir=str(snapshot_path),
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        )
    ).resolve()
    if resolved_path != snapshot_path.resolve():
        raise RuntimeError(
            f"Hugging Face staged {profile.profile_id!r} at unexpected path {resolved_path}; "
            f"expected {snapshot_path.resolve()}."
        )
    if not (snapshot_path / "config.json").is_file():
        raise RuntimeError(
            f"Staged Modal LLM profile {profile.profile_id!r} is missing config.json."
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
    "is_model_snapshot_staged",
    "model_snapshot_path",
    "stage_model_profile",
]
