"""Runtime configuration helpers for ComfyUI Modal-Sync."""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

if __package__:
    from .instance_identity import (
        INSTANCE_ID_FILENAME,
        load_or_create_instance_id,
        modal_app_name_for_instance,
    )
else:  # pragma: no cover - the stable cloud entrypoint imports this module top-level.
    from instance_identity import (
        INSTANCE_ID_FILENAME,
        load_or_create_instance_id,
        modal_app_name_for_instance,
    )

logger = logging.getLogger(__name__)

MODAL_GPU_TYPES = (
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "RTX-PRO-6000",
    "H100",
    "H100!",
    "H200",
    "B200",
    "B200+",
    "B300",
)
WORKFLOW_MODAL_CONFIG_KEY = "comfy_modal"
WORKFLOW_MODAL_GPU_KEY = "gpu"
DEFAULT_MODAL_GPU = "RTX-PRO-6000"
DEFAULT_MODAL_SECRET_NAME = "comfy"
_LEGACY_BASE_APP_GPU = "A100"
_MODAL_APP_NAME_MAX_LENGTH = 64
_MODAL_GPU_SLUG_MAX_LENGTH = 28

_SETTINGS_ENV_KEYS = (
    "COMFYUI_ROOT",
    "COMFY_MODAL_COMFYUI_ROOT",
    "COMFY_MODAL_CUSTOM_NODES_DIR",
    "COMFY_MODAL_EXECUTION_MODE",
    "COMFY_MODAL_SECRET_NAME",
    "COMFY_MODAL_SYNC_CUSTOM_NODES",
    "COMFY_MODAL_LOCAL_STORAGE_ROOT",
    "COMFY_MODAL_APP_NAME",
    "COMFY_MODAL_INSTANCE_ID_PATH",
    "COMFY_MODAL_AUTO_DEPLOY",
    "COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK",
    "COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT",
    "COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT",
    "COMFY_MODAL_VOLUME_NAME",
    "COMFY_MODAL_ROUTE_PATH",
    "COMFY_MODAL_MARKER_PROPERTY",
    "COMFY_MODAL_REMOTE_STORAGE_ROOT",
    "COMFY_MODAL_CUSTOM_NODES_ARCHIVE",
    "COMFY_MODAL_INTERRUPT_DICT_NAME",
    "COMFY_MODAL_NODE_CACHE_DICT_NAME",
    "COMFY_MODAL_SESSION_BRIDGE_DICT_NAME",
    "COMFY_MODAL_INVOCATION_DICT_NAME",
    "COMFY_MODAL_SYNC_INDEX_DICT_NAME",
    "COMFY_MODAL_SNAPSHOT_PROFILE_DICT_NAME",
    "COMFY_MODAL_NODE_CACHE_MAX_BYTES",
    "COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES",
    "COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES",
    "COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR",
    "COMFY_MODAL_GPU",
    "COMFY_MODAL_MIN_CONTAINERS",
    "COMFY_MODAL_MAX_CONTAINERS",
    "COMFY_MODAL_BUFFER_CONTAINERS",
    "COMFY_MODAL_MAX_INFLIGHT_CALLS",
    "COMFY_MODAL_SCALEDOWN_WINDOW",
    "COMFY_MODAL_LOCAL_GAP_KEEPALIVE_SECONDS",
    "COMFY_MODAL_LOCAL_GAP_KEEPALIVE_INTERVAL_SECONDS",
    "COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS",
    "COMFY_MODAL_STARTUP_TIMEOUT_SECONDS",
    "COMFY_MODAL_ENABLE_LOADER_PREWARM",
    "COMFY_MODAL_LOADER_PREWARM_WORKERS",
    "COMFY_MODAL_MAX_LOADER_PREWARMS_PER_COMPONENT",
    "COMFY_MODAL_ENABLE_PROACTIVE_WARMUP",
    "COMFY_MODAL_PROACTIVE_WARMUP_HEAD_START_SECONDS",
    "COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS",
    "COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS",
    "COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE",
    "COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS",
    "COMFY_MODAL_LLM_MAX_RESIDENT_MODELS",
    "COMFY_MODAL_LLM_RESERVE_FREE_GB",
    "COMFY_MODAL_LLM_COMPILE_CACHE_VOLUME_NAME",
    "COMFY_MODAL_LLM_VLLM_EXECUTION_MODE",
    "COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS",
)

VLLM_EXECUTION_MODES = ("auto", "eager", "throughput")


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
    modal_secret_name: str = DEFAULT_MODAL_SECRET_NAME
    interrupt_dict_name: str = "comfy-modal-sync-interrupts"
    node_output_cache_dict_name: str = "comfy-modal-sync-node-cache"
    session_bridge_dict_name: str = "comfy-modal-sync-session-bridges"
    invocation_dict_name: str = "comfy-modal-sync-invocations"
    sync_index_dict_name: str = "comfy-modal-sync-sync-index"
    snapshot_profile_dict_name: str = "comfy-modal-sync-snapshot-profiles"
    node_output_cache_max_bytes: int = 5 * 1024 * 1024
    bridge_inline_max_bytes: int = 4 * 1024 * 1024
    invocation_result_inline_max_bytes: int = 4 * 1024 * 1024
    terminate_container_on_error: bool = True
    modal_gpu: str = DEFAULT_MODAL_GPU
    scaledown_window_seconds: int = 600
    local_gap_keepalive_seconds: float = 900.0
    local_gap_keepalive_interval_seconds: float = 15.0
    min_containers: int = 0
    max_containers: int | None = None
    buffer_containers: int | None = None
    max_inflight_calls: int = 4
    execution_timeout_seconds: int = 3600
    startup_timeout_seconds: int = 900
    enable_proactive_warmup: bool = True
    enable_loader_prewarm: bool = True
    loader_prewarm_workers: int = 2
    proactive_warmup_head_start_seconds: float = 2.0
    remote_cancel_grace_seconds: float = 2.0
    remote_cancel_restart_seconds: float = 1.0
    stream_event_queue_maxsize: int = 256
    stream_remote_container_logs: bool = False
    llm_max_resident_models: int = 2
    llm_reserve_free_vram_gb: float = 24.0
    llm_compile_cache_volume_name: str = "comfy-universal-storage-llm-compile-cache"
    llm_vllm_execution_mode: str = "auto"
    llm_memory_recovery_timeout_seconds: float = 15.0


def normalize_vllm_execution_mode(value: object) -> str:
    """Return one supported vLLM execution profile name."""
    normalized = str(value).strip().lower()
    if normalized not in VLLM_EXECUTION_MODES:
        supported_values = ", ".join(VLLM_EXECUTION_MODES)
        raise ValueError(
            f"Unsupported vLLM execution mode {value!r}. "
            f"Choose one of: {supported_values}."
        )
    return normalized


def normalize_modal_gpu_selection(value: object) -> str:
    """Return one supported workflow-level Modal GPU selection."""
    if not isinstance(value, str):
        raise ValueError("The workflow Modal GPU selection must be a string.")
    normalized = value.strip().upper()
    if normalized not in MODAL_GPU_TYPES:
        supported_values = ", ".join(MODAL_GPU_TYPES)
        raise ValueError(
            f"Unsupported workflow Modal GPU selection {value!r}. "
            f"Choose one of: {supported_values}."
        )
    return normalized


def modal_gpu_from_workflow(
    workflow: object,
    default_gpu: str,
) -> str:
    """Resolve the saved workflow GPU selection or return the configured fallback."""
    if not isinstance(workflow, dict):
        return default_gpu
    extra = workflow.get("extra")
    if not isinstance(extra, dict):
        return default_gpu
    modal_config = extra.get(WORKFLOW_MODAL_CONFIG_KEY)
    if not isinstance(modal_config, dict) or WORKFLOW_MODAL_GPU_KEY not in modal_config:
        return default_gpu
    return normalize_modal_gpu_selection(modal_config[WORKFLOW_MODAL_GPU_KEY])


def settings_for_modal_gpu(
    settings: ModalSyncSettings,
    modal_gpu: object,
) -> ModalSyncSettings:
    """Return settings with one validated workflow-level GPU override."""
    return replace(settings, modal_gpu=normalize_modal_gpu_selection(modal_gpu))


def _modal_gpu_app_slug(modal_gpu: str) -> str:
    """Return a readable, collision-resistant app-name suffix for one GPU target."""
    expanded = (
        modal_gpu.strip()
        .lower()
        .replace("!", "-priority")
        .replace("+", "-plus")
        .replace(":", "-x")
        .replace(",", "-or-")
    )
    slug = "-".join(re.findall(r"[a-z0-9]+", expanded))
    if not slug:
        slug = f"target-{hashlib.sha256(modal_gpu.encode('utf-8')).hexdigest()[:8]}"
    if len(slug) <= _MODAL_GPU_SLUG_MAX_LENGTH:
        return slug
    digest = hashlib.sha256(modal_gpu.encode("utf-8")).hexdigest()[:8]
    prefix_length = _MODAL_GPU_SLUG_MAX_LENGTH - len(digest) - 1
    return f"{slug[:prefix_length].rstrip('-')}-{digest}"


def modal_deployment_app_name(settings: ModalSyncSettings) -> str:
    """Return the persistent Modal app name dedicated to the configured GPU target."""
    modal_gpu = str(getattr(settings, "modal_gpu", DEFAULT_MODAL_GPU)).strip()
    if modal_gpu.upper() == _LEGACY_BASE_APP_GPU:
        return settings.app_name

    suffix = f"-gpu-{_modal_gpu_app_slug(modal_gpu)}"
    candidate = f"{settings.app_name}{suffix}"
    if len(candidate) <= _MODAL_APP_NAME_MAX_LENGTH:
        return candidate

    digest = hashlib.sha256(settings.app_name.encode("utf-8")).hexdigest()[:8]
    prefix_length = _MODAL_APP_NAME_MAX_LENGTH - len(suffix) - len(digest) - 1
    prefix = settings.app_name[:prefix_length].rstrip("-_.") or "app"
    return f"{prefix}-{digest}{suffix}"


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


def _read_float_env(name: str, default: float) -> float:
    """Resolve an environment variable into a float when present."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}.") from exc


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

    for default_name in ("ComfyUI", "Latest_ComfyUI"):
        default_root = Path.home() / "git" / default_name
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


def _discover_comfyui_user_directory(comfyui_root: Path | None) -> Path | None:
    """Return ComfyUI's effective user directory when it can be resolved."""
    try:
        folder_paths = importlib.import_module("folder_paths")
    except ModuleNotFoundError as exc:
        if exc.name != "folder_paths":
            logger.debug("Unable to import folder_paths while resolving the user directory: %s", exc)
    else:
        get_user_directory = getattr(folder_paths, "get_user_directory", None)
        if callable(get_user_directory):
            return Path(get_user_directory()).expanduser().resolve()
    if comfyui_root is None:
        return None
    return (comfyui_root / "user").resolve()


def discover_comfyui_user_directory(
    settings: ModalSyncSettings | None = None,
) -> Path | None:
    """Return the effective ComfyUI user directory for extension-owned state."""
    resolved_settings = settings or get_settings()
    return _discover_comfyui_user_directory(resolved_settings.comfyui_root)


def _resolve_modal_app_name(comfyui_root: Path | None, execution_mode: str) -> str:
    """Return the explicit or persistent per-ComfyUI Modal app name."""
    configured_name = os.getenv("COMFY_MODAL_APP_NAME")
    if configured_name is not None:
        resolved_name = configured_name.strip()
        if not resolved_name:
            raise ValueError("COMFY_MODAL_APP_NAME must not be empty.")
        return resolved_name

    identity_path = _read_path_env("COMFY_MODAL_INSTANCE_ID_PATH")
    if identity_path is None:
        user_directory = _discover_comfyui_user_directory(comfyui_root)
        if user_directory is not None:
            identity_path = user_directory / INSTANCE_ID_FILENAME
    if identity_path is None:
        if execution_mode == "remote":
            raise RuntimeError(
                "Remote execution requires a persistent ComfyUI instance identity, but the "
                "ComfyUI user directory could not be resolved. Set COMFY_MODAL_COMFYUI_ROOT, "
                "COMFY_MODAL_INSTANCE_ID_PATH, or COMFY_MODAL_APP_NAME explicitly."
            )
        logger.debug("Using the legacy local app name because no ComfyUI user directory was found.")
        return "comfy-modal-sync"

    app_name = modal_app_name_for_instance(load_or_create_instance_id(identity_path))
    logger.info("Using per-ComfyUI Modal app %s from identity file %s.", app_name, identity_path)
    return app_name


def _settings_env_signature() -> tuple[tuple[str, str | None], ...]:
    """Return the environment values that affect resolved Modal-Sync settings."""
    return tuple((key, os.getenv(key)) for key in _SETTINGS_ENV_KEYS)


@lru_cache(maxsize=8)
def _get_settings_cached(
    env_signature: tuple[tuple[str, str | None], ...],
) -> ModalSyncSettings:
    """Return cached extension settings for one environment signature."""
    del env_signature
    repo_root = _discover_repo_root()
    comfyui_root = _discover_comfyui_root(repo_root)
    custom_nodes_dir = _discover_custom_nodes_dir(repo_root, comfyui_root)
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local").strip().lower()
    app_name = _resolve_modal_app_name(comfyui_root, execution_mode)
    sync_custom_nodes = _read_bool_env("COMFY_MODAL_SYNC_CUSTOM_NODES")
    if sync_custom_nodes is None:
        sync_custom_nodes = execution_mode != "local"
    local_storage_root = (
        _read_path_env("COMFY_MODAL_LOCAL_STORAGE_ROOT")
        or Path("/tmp/comfyui-modal-sync-storage")
    )
    volume_name = os.getenv("COMFY_MODAL_VOLUME_NAME", "comfy-universal-storage")

    settings = ModalSyncSettings(
        app_name=app_name,
        auto_deploy=_read_bool_env("COMFY_MODAL_AUTO_DEPLOY") is not False,
        allow_ephemeral_fallback=_read_bool_env("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK") or False,
        enable_memory_snapshot=_read_bool_env("COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT") is not False,
        enable_gpu_memory_snapshot=_read_bool_env("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT")
        is not False,
        execution_mode=execution_mode,
        sync_custom_nodes=sync_custom_nodes,
        volume_name=volume_name,
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
        modal_secret_name=(
            os.getenv("COMFY_MODAL_SECRET_NAME", DEFAULT_MODAL_SECRET_NAME).strip()
            or DEFAULT_MODAL_SECRET_NAME
        ),
        interrupt_dict_name=os.getenv(
            "COMFY_MODAL_INTERRUPT_DICT_NAME",
            f"{app_name}-interrupts",
        ),
        node_output_cache_dict_name=os.getenv(
            "COMFY_MODAL_NODE_CACHE_DICT_NAME",
            f"{app_name}-node-cache",
        ),
        session_bridge_dict_name=os.getenv(
            "COMFY_MODAL_SESSION_BRIDGE_DICT_NAME",
            f"{app_name}-session-bridges",
        ),
        invocation_dict_name=os.getenv(
            "COMFY_MODAL_INVOCATION_DICT_NAME",
            f"{app_name}-invocations",
        ),
        snapshot_profile_dict_name=os.getenv(
            "COMFY_MODAL_SNAPSHOT_PROFILE_DICT_NAME",
            f"{app_name}-snapshot-profiles",
        ),
        sync_index_dict_name=os.getenv(
            "COMFY_MODAL_SYNC_INDEX_DICT_NAME",
            f"{app_name}-sync-index",
        ),
        node_output_cache_max_bytes=_read_int_env(
            "COMFY_MODAL_NODE_CACHE_MAX_BYTES",
            5 * 1024 * 1024,
        ),
        bridge_inline_max_bytes=max(
            0,
            _read_int_env("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", 4 * 1024 * 1024),
        ),
        invocation_result_inline_max_bytes=max(
            0,
            _read_int_env(
                "COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES",
                4 * 1024 * 1024,
            ),
        ),
        terminate_container_on_error=_read_bool_env("COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR")
        is not False,
        modal_gpu=os.getenv("COMFY_MODAL_GPU", DEFAULT_MODAL_GPU).strip() or DEFAULT_MODAL_GPU,
        scaledown_window_seconds=_read_int_env("COMFY_MODAL_SCALEDOWN_WINDOW", 600),
        local_gap_keepalive_seconds=max(
            0.0,
            _read_float_env("COMFY_MODAL_LOCAL_GAP_KEEPALIVE_SECONDS", 900.0),
        ),
        local_gap_keepalive_interval_seconds=max(
            1.0,
            _read_float_env(
                "COMFY_MODAL_LOCAL_GAP_KEEPALIVE_INTERVAL_SECONDS",
                15.0,
            ),
        ),
        min_containers=_read_int_env("COMFY_MODAL_MIN_CONTAINERS", 0),
        max_containers=_read_optional_int_env("COMFY_MODAL_MAX_CONTAINERS"),
        buffer_containers=_read_optional_int_env("COMFY_MODAL_BUFFER_CONTAINERS"),
        max_inflight_calls=max(
            1,
            _read_int_env("COMFY_MODAL_MAX_INFLIGHT_CALLS", 4),
        ),
        execution_timeout_seconds=max(
            1,
            _read_int_env("COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS", 3600),
        ),
        startup_timeout_seconds=max(
            1,
            _read_int_env("COMFY_MODAL_STARTUP_TIMEOUT_SECONDS", 900),
        ),
        enable_proactive_warmup=_read_bool_env("COMFY_MODAL_ENABLE_PROACTIVE_WARMUP") is not False,
        enable_loader_prewarm=_read_bool_env("COMFY_MODAL_ENABLE_LOADER_PREWARM") is not False,
        loader_prewarm_workers=max(
            1,
            _read_int_env("COMFY_MODAL_LOADER_PREWARM_WORKERS", 2),
        ),
        proactive_warmup_head_start_seconds=_read_float_env(
            "COMFY_MODAL_PROACTIVE_WARMUP_HEAD_START_SECONDS",
            2.0,
        ),
        remote_cancel_grace_seconds=_read_float_env(
            "COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS",
            2.0,
        ),
        remote_cancel_restart_seconds=_read_float_env(
            "COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS",
            1.0,
        ),
        stream_event_queue_maxsize=max(
            4,
            _read_int_env("COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE", 256),
        ),
        stream_remote_container_logs=_read_bool_env("COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS")
        or False,
        llm_max_resident_models=max(
            1,
            _read_int_env("COMFY_MODAL_LLM_MAX_RESIDENT_MODELS", 2),
        ),
        llm_reserve_free_vram_gb=max(
            0.0,
            _read_float_env("COMFY_MODAL_LLM_RESERVE_FREE_GB", 24.0),
        ),
        llm_compile_cache_volume_name=(
            os.getenv(
                "COMFY_MODAL_LLM_COMPILE_CACHE_VOLUME_NAME",
                f"{volume_name}-llm-compile-cache",
            ).strip()
            or f"{volume_name}-llm-compile-cache"
        ),
        llm_vllm_execution_mode=normalize_vllm_execution_mode(
            os.getenv("COMFY_MODAL_LLM_VLLM_EXECUTION_MODE", "auto")
        ),
        llm_memory_recovery_timeout_seconds=max(
            0.0,
            _read_float_env(
                "COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS",
                15.0,
            ),
        ),
    )
    logger.debug("Resolved Modal-Sync settings: %s", settings)
    return settings


def get_settings() -> ModalSyncSettings:
    """Return extension settings derived from the current environment."""
    return _get_settings_cached(_settings_env_signature())


get_settings.cache_clear = _get_settings_cached.cache_clear  # type: ignore[attr-defined]
