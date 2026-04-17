"""Runtime configuration helpers for ComfyUI Modal-Sync."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalSyncSettings:
    """Resolved configuration for local and remote Modal-Sync operations."""

    app_name: str
    auto_deploy: bool
    allow_ephemeral_fallback: bool
    enable_memory_snapshot: bool
    enable_gpu_memory_snapshot: bool
    execution_mode: str
    sync_custom_nodes: bool
    volume_name: str
    route_path: str
    marker_property: str
    local_storage_root: Path
    remote_storage_root: str
    custom_nodes_archive_name: str
    comfyui_root: Path | None
    custom_nodes_dir: Path | None


def _read_path_env(name: str) -> Path | None:
    """Resolve an environment variable into a path when present."""
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _read_bool_env(name: str) -> bool | None:
    """Resolve an environment variable into a boolean when present."""
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got {value!r}.")


def _discover_repo_root() -> Path:
    """Return the repository root containing this module."""
    return Path(__file__).resolve().parent


def _discover_comfyui_root() -> Path | None:
    """Locate the local ComfyUI checkout used for tests and path resolution."""
    env_root = _read_path_env("COMFYUI_ROOT")
    if env_root is not None:
        return env_root

    default_root = Path.home() / "git" / "ComfyUI"
    if default_root.exists():
        return default_root.resolve()

    return None


def _discover_custom_nodes_dir(repo_root: Path, comfyui_root: Path | None) -> Path | None:
    """Locate the custom_nodes directory that should be mirrored to Modal."""
    env_dir = _read_path_env("COMFY_MODAL_CUSTOM_NODES_DIR")
    if env_dir is not None:
        return env_dir

    if repo_root.parent.name == "custom_nodes":
        return repo_root.parent.resolve()

    if comfyui_root is not None:
        candidate = comfyui_root / "custom_nodes"
        if candidate.exists():
            return candidate.resolve()

    return None


@lru_cache(maxsize=1)
def get_settings() -> ModalSyncSettings:
    """Return cached extension settings derived from the environment."""
    repo_root = _discover_repo_root()
    comfyui_root = _discover_comfyui_root()
    custom_nodes_dir = _discover_custom_nodes_dir(repo_root, comfyui_root)
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local").strip().lower()
    sync_custom_nodes = _read_bool_env("COMFY_MODAL_SYNC_CUSTOM_NODES")
    if sync_custom_nodes is None:
        sync_custom_nodes = execution_mode != "local"
    local_storage_root = (
        _read_path_env("COMFY_MODAL_LOCAL_STORAGE_ROOT")
        or Path("/tmp/comfyui-modal-sync-storage")
    )

    settings = ModalSyncSettings(
        app_name=os.getenv("COMFY_MODAL_APP_NAME", "comfy-modal-sync"),
        auto_deploy=_read_bool_env("COMFY_MODAL_AUTO_DEPLOY") is not False,
        allow_ephemeral_fallback=_read_bool_env("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK") or False,
        enable_memory_snapshot=_read_bool_env("COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT") is not False,
        enable_gpu_memory_snapshot=_read_bool_env("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT") or False,
        execution_mode=execution_mode,
        sync_custom_nodes=sync_custom_nodes,
        volume_name=os.getenv("COMFY_MODAL_VOLUME_NAME", "comfy-universal-storage"),
        route_path=os.getenv("COMFY_MODAL_ROUTE_PATH", "/modal/queue_prompt"),
        marker_property=os.getenv("COMFY_MODAL_MARKER_PROPERTY", "is_modal_remote"),
        local_storage_root=local_storage_root.resolve(),
        remote_storage_root=os.getenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", "/storage"),
        custom_nodes_archive_name=os.getenv(
            "COMFY_MODAL_CUSTOM_NODES_ARCHIVE",
            "custom_nodes_bundle.zip",
        ),
        comfyui_root=comfyui_root,
        custom_nodes_dir=custom_nodes_dir,
    )
    logger.debug("Resolved Modal-Sync settings: %s", settings)
    return settings
