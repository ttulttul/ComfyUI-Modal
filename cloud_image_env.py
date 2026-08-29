"""Modal image filtering, package layering, and container configuration."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

try:
    from .runtime_environment import (
        COMFYUI_RUNTIME_SOURCE_DIRECTORIES as _COMFYUI_IMAGE_RUNTIME_DIRECTORIES,
        COMFYUI_RUNTIME_SOURCE_FILES as _COMFYUI_IMAGE_RUNTIME_FILES,
        REMOTE_PYTHON_VERSION,
        RemoteTorchBuild as _RemoteTorchBuild,
        remote_accelerator_packages as _remote_accelerator_packages,
        remote_accelerator_validation_command as _remote_accelerator_validation_command,
        remote_runtime_packages as _comfyui_runtime_packages,
        select_remote_torch_build as _select_remote_torch_build,
    )
    from .settings import DEFAULT_MODAL_SECRET_NAME, get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from runtime_environment import (
        COMFYUI_RUNTIME_SOURCE_DIRECTORIES as _COMFYUI_IMAGE_RUNTIME_DIRECTORIES,
        COMFYUI_RUNTIME_SOURCE_FILES as _COMFYUI_IMAGE_RUNTIME_FILES,
        REMOTE_PYTHON_VERSION,
        RemoteTorchBuild as _RemoteTorchBuild,
        remote_accelerator_packages as _remote_accelerator_packages,
        remote_accelerator_validation_command as _remote_accelerator_validation_command,
        remote_runtime_packages as _comfyui_runtime_packages,
        select_remote_torch_build as _select_remote_torch_build,
    )
    from settings import DEFAULT_MODAL_SECRET_NAME, get_settings

logger = logging.getLogger(__name__)

_REMOTE_LLM_COMPILE_CACHE_ROOT = Path("/root/.cache/comfy-modal-llm")
_COMFYUI_IMAGE_EXCLUDED_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".engine",
        ".gguf",
        ".log",
        ".onnx",
        ".pt",
        ".pth",
        ".pyc",
        ".pyo",
        ".safetensors",
        ".swp",
        ".tmp",
        ".vae",
    }
)


def _should_ignore_repo_path(path: Path) -> bool:
    """Return whether a local repo path should be omitted from the Modal image mount."""
    parts = set(path.parts)
    if {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    } & parts:
        return True
    return path.suffix.lower() in {".log", ".pyc", ".pyo", ".swp", ".tmp"}


def _comfyui_image_relative_parts(path: Path) -> tuple[str, ...]:
    """Return normalized ComfyUI-relative path parts for one image candidate."""
    candidate = path
    if path.is_absolute():
        comfyui_root = get_settings().comfyui_root
        if comfyui_root is None:
            return ()
        try:
            candidate = path.relative_to(comfyui_root)
        except ValueError:
            return ()
    return tuple(part for part in candidate.parts if part not in {"", "."})


def _should_ignore_comfyui_path(path: Path) -> bool:
    """Allow only source and configuration required by the headless Modal runtime."""
    parts = _comfyui_image_relative_parts(path)
    if not parts:
        return False

    if {
        ".cache",
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    } & set(parts):
        return True
    if path.suffix.lower() in _COMFYUI_IMAGE_EXCLUDED_SUFFIXES:
        return True
    if parts[0] in _COMFYUI_IMAGE_RUNTIME_DIRECTORIES:
        return False
    return not (
        len(parts) == 1
        and (path.suffix.lower() == ".py" or parts[0] in _COMFYUI_IMAGE_RUNTIME_FILES)
    )


def _remote_engine_cls_options(
    settings: Any,
    vol: Any,
    image: Any,
    modal_secret: Any | None = None,
    llm_compile_cache_vol: Any | None = None,
) -> dict[str, Any]:
    """Build the Modal class options for the deployed remote execution runtime."""
    volumes = {settings.remote_storage_root: vol}
    if llm_compile_cache_vol is not None:
        volumes[str(_REMOTE_LLM_COMPILE_CACHE_ROOT)] = llm_compile_cache_vol
    options: dict[str, Any] = {
        "gpu": settings.modal_gpu,
        "volumes": volumes,
        "scaledown_window": settings.scaledown_window_seconds,
        "min_containers": settings.min_containers,
        "image": image,
        "enable_memory_snapshot": settings.enable_memory_snapshot,
        "timeout": int(getattr(settings, "execution_timeout_seconds", 3600)),
        "startup_timeout": int(getattr(settings, "startup_timeout_seconds", 900)),
    }
    if modal_secret is not None:
        options["secrets"] = [modal_secret]
    max_containers = getattr(settings, "max_containers", None)
    buffer_containers = getattr(settings, "buffer_containers", None)
    if max_containers is not None:
        options["max_containers"] = max_containers
    if buffer_containers is not None:
        options["buffer_containers"] = buffer_containers
    if settings.enable_gpu_memory_snapshot:
        options["experimental_options"] = {"enable_gpu_snapshot": True}
    return options


def _modal_secret_from_settings(settings: Any, modal_module: Any) -> Any:
    """Return the existing named Modal secret configured for remote workers."""
    secret_name = str(
        getattr(settings, "modal_secret_name", DEFAULT_MODAL_SECRET_NAME)
    ).strip()
    if not secret_name:
        raise ValueError("The Modal secret collection name must not be empty.")
    logger.info(
        "Attaching Modal secret collection %s to the remote worker.", secret_name
    )
    return modal_module.Secret.from_name(secret_name)


def _llm_compile_cache_namespace(settings: Any) -> str:
    """Return a stable cache namespace for accelerator-compatible JIT artifacts."""
    cache_identity = {
        "accelerator_packages": _remote_accelerator_packages(settings.modal_gpu),
        "gpu": settings.modal_gpu,
        "python": REMOTE_PYTHON_VERSION,
        "torch_build": _select_remote_torch_build(settings.modal_gpu),
    }
    payload = json.dumps(cache_identity, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _modal_image_environment(settings: Any, runtime_fingerprint: str) -> dict[str, str]:
    """Return environment values that keep the worker aligned with local settings."""
    compile_cache_root = (
        _REMOTE_LLM_COMPILE_CACHE_ROOT / _llm_compile_cache_namespace(settings)
    )
    return {
        "COMFY_MODAL_APP_NAME": settings.app_name,
        "COMFY_MODAL_GPU": settings.modal_gpu,
        "COMFY_MODAL_REMOTE_STORAGE_ROOT": getattr(
            settings,
            "remote_storage_root",
            "/storage",
        ),
        "COMFY_MODAL_REMOTE_WORKER": "1",
        "COMFY_MODAL_LLM_VLLM_EXECUTION_MODE": str(
            getattr(settings, "llm_vllm_execution_mode", "auto")
        ),
        "VLLM_CACHE_ROOT": str(compile_cache_root / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(compile_cache_root / "torchinductor"),
        "TRITON_CACHE_DIR": str(compile_cache_root / "triton"),
        "CUDA_CACHE_PATH": str(compile_cache_root / "cuda"),
        "TORCH_EXTENSIONS_DIR": str(compile_cache_root / "torch-extensions"),
        "NUMBA_CACHE_DIR": str(compile_cache_root / "numba"),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "COMFY_MODAL_SECRET_NAME": getattr(
            settings,
            "modal_secret_name",
            DEFAULT_MODAL_SECRET_NAME,
        ),
        "COMFY_MODAL_RUNTIME_FINGERPRINT": runtime_fingerprint,
        "COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE": str(
            settings.stream_event_queue_maxsize
        ),
        "COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES": str(settings.bridge_inline_max_bytes),
        "COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES": str(
            settings.invocation_result_inline_max_bytes
        ),
        "COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS": str(
            settings.execution_timeout_seconds
        ),
        "COMFY_MODAL_STARTUP_TIMEOUT_SECONDS": str(settings.startup_timeout_seconds),
        "COMFY_MODAL_LLM_MAX_RESIDENT_MODELS": str(
            getattr(settings, "llm_max_resident_models", 2)
        ),
        "COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS": str(
            getattr(settings, "llm_memory_recovery_timeout_seconds", 15.0)
        ),
        "COMFY_MODAL_LLM_RESERVE_FREE_GB": str(
            getattr(settings, "llm_reserve_free_vram_gb", 24.0)
        ),
    }


def _model_stager_image_environment(
    settings: Any,
    runtime_fingerprint: str,
) -> dict[str, str]:
    """Return deployment identity and staging values for the CPU-only helper."""
    return {
        "COMFY_MODAL_APP_NAME": settings.app_name,
        "COMFY_MODAL_GPU": settings.modal_gpu,
        "COMFY_MODAL_REMOTE_STORAGE_ROOT": settings.remote_storage_root,
        "COMFY_MODAL_RUNTIME_FINGERPRINT": runtime_fingerprint,
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def _install_remote_torch_build(image: Any, torch_build: _RemoteTorchBuild) -> Any:
    """Install and validate the ordered package layers for one remote Torch build."""
    for layer_number, install_layer in enumerate(torch_build.install_layers, start=1):
        logger.info(
            "Installing Modal PyTorch layer=%d index=%s packages=%s extra_options=%s.",
            layer_number,
            install_layer.index_url,
            install_layer.packages,
            install_layer.extra_options or "<none>",
        )
        image = image.pip_install(
            *install_layer.packages,
            index_url=install_layer.index_url,
            extra_options=install_layer.extra_options,
        )
    return image.run_commands(torch_build.validation_command())


def _install_remote_accelerator_packages(image: Any, modal_gpu: str) -> Any:
    """Install and validate the vLLM wheel shared by all supported GPU images."""
    accelerator_packages = _remote_accelerator_packages(modal_gpu)
    logger.info(
        "Installing Modal accelerator packages gpu=%s packages=%s.",
        modal_gpu,
        accelerator_packages,
    )
    image = image.pip_install(*accelerator_packages)
    return image.run_commands(_remote_accelerator_validation_command(modal_gpu))


def _install_custom_node_packages(image: Any, packages: tuple[str, ...]) -> Any:
    """Install custom requirements and then restore deployment-owned package pins."""
    if not packages:
        return image
    image = image.pip_install(*packages)
    return image.pip_install(*_comfyui_runtime_packages())

