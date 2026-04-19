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
    interrupt_dict_name: str = "comfy-modal-sync-interrupts"
    terminate_container_on_error: bool = True
    modal_gpu: str = "A100"
    scaledown_window_seconds: int = 600
    min_containers: int = 0
    max_containers: int | None = None
    buffer_containers: int | None = None
    enable_proactive_warmup: bool = True


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


def _read_int_env(name: str, default: int) -> int:
    """Resolve an environment variable into an integer when present."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.") from exc


def _read_optional_int_env(name: str) -> int | None:
    """Resolve an environment variable into an optional integer when present."""
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.") from exc


def _looks_like_comfyui_root(candidate: Path) -> bool:
    """Return whether a path appears to be a ComfyUI checkout root."""
    return (candidate / "main.py").exists() and (candidate / "nodes.py").exists()


def _discover_comfyui_root(repo_root: Path) -> Path | None:
    """Locate the local ComfyUI checkout used for tests and path resolution."""
    env_root = _read_path_env("COMFYUI_ROOT")
    if env_root is not None:
        return env_root

    modal_env_root = _read_path_env("COMFY_MODAL_COMFYUI_ROOT")
    if modal_env_root is not None:
        return modal_env_root

    if repo_root.parent.name == "custom_nodes":
        install_root = repo_root.parent.parent.resolve()
        if _looks_like_comfyui_root(install_root):
            return install_root

    default_root = Path.home() / "git" / "ComfyUI"
    if _looks_like_comfyui_root(default_root):
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
    comfyui_root = _discover_comfyui_root(repo_root)
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
        app_name=(app_name := os.getenv("COMFY_MODAL_APP_NAME", "comfy-modal-sync")),
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
        interrupt_dict_name=os.getenv(
            "COMFY_MODAL_INTERRUPT_DICT_NAME",
            f"{app_name}-interrupts",
        ),
        terminate_container_on_error=_read_bool_env("COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR")
        is not False,
        modal_gpu=os.getenv("COMFY_MODAL_GPU", "A100").strip() or "A100",
        scaledown_window_seconds=_read_int_env("COMFY_MODAL_SCALEDOWN_WINDOW", 600),
        min_containers=_read_int_env("COMFY_MODAL_MIN_CONTAINERS", 0),
        max_containers=_read_optional_int_env("COMFY_MODAL_MAX_CONTAINERS"),
        buffer_containers=_read_optional_int_env("COMFY_MODAL_BUFFER_CONTAINERS"),
        enable_proactive_warmup=_read_bool_env("COMFY_MODAL_ENABLE_PROACTIVE_WARMUP") is not False,
    )
    logger.debug("Resolved Modal-Sync settings: %s", settings)
    return settings
