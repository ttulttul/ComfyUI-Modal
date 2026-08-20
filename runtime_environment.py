"""Pinned remote environment and deployment identity helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__:
    from .settings import DEFAULT_MODAL_GPU, DEFAULT_MODAL_SECRET_NAME
else:  # pragma: no cover - the stable cloud entrypoint imports this module top-level.
    from settings import DEFAULT_MODAL_GPU, DEFAULT_MODAL_SECRET_NAME

logger = logging.getLogger(__name__)

REMOTE_APP_PROTOCOL_VERSION = 9
REMOTE_PYTHON_VERSION = "3.13"
REMOTE_MODAL_SDK_SPEC = "modal==1.4.2"
REMOTE_HUGGINGFACE_HUB_SPEC = "huggingface-hub==1.28.0"
REMOTE_HF_XET_SPEC = "hf-xet==1.6.0"
REMOTE_VLLM_SPEC = "vllm==0.27.1"
REMOTE_VLLM_WHEEL_URL = (
    "https://wheels.vllm.ai/"
    "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/"
    "vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl"
)
REMOTE_NUMPY_SPEC = "numpy==2.3.5"
REMOTE_OPENCV_SPEC = "opencv-python-headless==4.13.0.92"

_REMOTE_APT_PACKAGES = (
    "libgl1",
    "libglib2.0-0",
)
_REMOTE_RUNTIME_PACKAGES = (
    "aiohttp==3.13.3",
    "accelerate==1.14.0",
    "alembic==1.18.4",
    "av==16.1.0",
    "blake3==1.0.9",
    "comfy-angle==0.1.0",
    "comfy-aimdo==0.4.13",
    "comfy-kitchen==0.2.31",
    "einops==0.8.2",
    "kornia==0.8.2",
    REMOTE_NUMPY_SPEC,
    "num2words==0.5.14",
    REMOTE_OPENCV_SPEC,
    "packaging==26.0",
    "pillow==12.1.0",
    "psutil==7.2.2",
    "pypdf==6.16.1",
    "pydantic==2.12.5",
    "pydantic-settings==2.13.0",
    "pyopengl==3.1.10",
    "pyyaml==6.0.3",
    "requests==2.32.5",
    "safetensors==0.8.0",
    "scipy==1.17.0",
    "sentencepiece==0.2.1",
    "simpleeval==1.0.7",
    "spandrel==0.4.1",
    "sqlalchemy==2.0.46",
    "torchsde==0.2.6",
    "tqdm==4.67.3",
    "transformers==5.15.0",
    REMOTE_HUGGINGFACE_HUB_SPEC,
    REMOTE_HF_XET_SPEC,
)
_CUDA_130_TORCH_PACKAGES = ("torch==2.13.0", "torchvision==0.28.0")
_CUDA_130_CPU_AUDIO_PACKAGES = ("torchaudio==2.11.0+cpu",)
_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)
COMFYUI_RUNTIME_SOURCE_DIRECTORIES = frozenset(
    {
        "alembic_db",
        "api_server",
        "app",
        "blueprints",
        "comfy",
        "comfy_api",
        "comfy_api_nodes",
        "comfy_config",
        "comfy_execution",
        "comfy_extras",
        "middleware",
        "utils",
        "web",
    }
)
COMFYUI_RUNTIME_SOURCE_FILES = frozenset(
    {
        "alembic.ini",
        "manager_requirements.txt",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
    }
)


class RemoteRuntimeSettings(Protocol):
    """Describe settings that shape one deployed Modal runtime."""

    app_name: str
    buffer_containers: int | None
    enable_gpu_memory_snapshot: bool
    enable_memory_snapshot: bool
    execution_timeout_seconds: int
    invocation_dict_name: str
    invocation_result_inline_max_bytes: int
    interrupt_dict_name: str
    loader_prewarm_workers: int
    max_containers: int | None
    min_containers: int
    modal_gpu: str
    modal_secret_name: str
    node_output_cache_dict_name: str
    remote_storage_root: str
    scaledown_window_seconds: int
    session_bridge_dict_name: str
    bridge_inline_max_bytes: int
    snapshot_profile_dict_name: str
    startup_timeout_seconds: int
    stream_event_queue_maxsize: int
    sync_index_dict_name: str
    volume_name: str
    llm_max_resident_models: int
    llm_reserve_free_vram_gb: float


@dataclass(frozen=True)
class RemoteRuntimeIdentity:
    """Hold a deterministic fingerprint and its diagnostic manifest."""

    fingerprint: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RemoteTorchInstallLayer:
    """Describe one ordered package layer in a remote PyTorch installation."""

    index_url: str
    packages: tuple[str, ...]
    extra_options: str = ""


@dataclass(frozen=True)
class RemoteTorchBuild:
    """Describe an ordered, pinned PyTorch installation for one Modal GPU type."""

    cuda_version: str
    install_layers: tuple[RemoteTorchInstallLayer, ...]

    @property
    def packages(self) -> tuple[str, ...]:
        """Return every package pin installed for this PyTorch build."""
        return tuple(
            package
            for install_layer in self.install_layers
            for package in install_layer.packages
        )

    def validation_command(self) -> str:
        """Return a build-time import and CUDA compatibility check."""
        validation_script = (
            "import torch, torchaudio, torchvision; "
            f"expected_cuda={self.cuda_version!r}; "
            "actual_cuda=torch.version.cuda; "
            "assert actual_cuda == expected_cuda, "
            "f'Expected PyTorch CUDA {expected_cuda}, found {actual_cuda}'; "
            "print('Validated Modal PyTorch stack:', "
            "torch.__version__, torchvision.__version__, torchaudio.__version__, "
            "'CUDA', actual_cuda)"
        )
        return f"python -c {shlex.quote(validation_script)}"


def remote_runtime_packages() -> tuple[str, ...]:
    """Return the exact ComfyUI support package set used by Modal images."""
    return _REMOTE_RUNTIME_PACKAGES


def remote_huggingface_packages() -> tuple[str, ...]:
    """Return the pinned Hub client and its Xet transfer accelerator."""
    return (REMOTE_HUGGINGFACE_HUB_SPEC, REMOTE_HF_XET_SPEC)


def remote_huggingface_validation_command() -> str:
    """Return a build-time check for the remote Hugging Face transfer stack."""
    validation_script = (
        "from importlib import metadata; import hf_xet, huggingface_hub; "
        f"expected_hub={REMOTE_HUGGINGFACE_HUB_SPEC.split('==', maxsplit=1)[1]!r}; "
        f"expected_xet={REMOTE_HF_XET_SPEC.split('==', maxsplit=1)[1]!r}; "
        "actual_hub=metadata.version('huggingface-hub'); "
        "actual_xet=metadata.version('hf-xet'); "
        "assert actual_hub == expected_hub, "
        "f'Expected huggingface-hub {expected_hub}, found {actual_hub}'; "
        "assert actual_xet == expected_xet, "
        "f'Expected hf-xet {expected_xet}, found {actual_xet}'; "
        "print('Validated Hugging Face transfer stack:', "
        "'huggingface-hub', actual_hub, 'hf-xet', actual_xet)"
    )
    return f"python -c {shlex.quote(validation_script)}"


def remote_accelerator_packages(modal_gpu: str) -> tuple[str, ...]:
    """Return the immutable vLLM wheel installed for every supported GPU."""
    normalized_modal_gpu_type(modal_gpu)
    return (REMOTE_VLLM_WHEEL_URL,)


def remote_accelerator_validation_command(modal_gpu: str) -> str:
    """Return a build-time import and version check for the GPU inference stack."""
    normalized_modal_gpu_type(modal_gpu)
    validation_script = (
        "from importlib import metadata; import cv2, numpy, torch, vllm; "
        f"expected_vllm={REMOTE_VLLM_SPEC.split('==', maxsplit=1)[1]!r}; "
        f"expected_numpy={REMOTE_NUMPY_SPEC.split('==', maxsplit=1)[1]!r}; "
        f"expected_opencv={REMOTE_OPENCV_SPEC.split('==', maxsplit=1)[1]!r}; "
        "actual_vllm=metadata.version('vllm'); "
        "actual_numpy=metadata.version('numpy'); "
        "actual_opencv=metadata.version('opencv-python-headless'); "
        "assert actual_vllm == expected_vllm, "
        "f'Expected vLLM {expected_vllm}, found {actual_vllm}'; "
        "assert actual_numpy == expected_numpy, "
        "f'Expected NumPy {expected_numpy}, found {actual_numpy}'; "
        "assert actual_opencv == expected_opencv, "
        "f'Expected OpenCV {expected_opencv}, found {actual_opencv}'; "
        "print('Validated Modal accelerator stack:', "
        "'vLLM', actual_vllm, 'Torch', torch.__version__, "
        "'CUDA', torch.version.cuda, 'NumPy', numpy.__version__, "
        "'OpenCV', cv2.__version__)"
    )
    return f"python -c {shlex.quote(validation_script)}"


def remote_apt_packages() -> tuple[str, ...]:
    """Return system libraries required by Python packages in the Modal image."""
    return _REMOTE_APT_PACKAGES


def normalized_modal_gpu_type(modal_gpu: str) -> str:
    """Return a canonical Modal GPU type without an optional count suffix."""
    gpu_type = modal_gpu.strip().split(":", maxsplit=1)[0].upper()
    if not gpu_type:
        raise ValueError("Modal GPU type cannot be empty when selecting a PyTorch build.")
    return gpu_type


def select_remote_torch_build(modal_gpu: str) -> RemoteTorchBuild:
    """Select the CUDA 13 stack shared by every supported Modal GPU."""
    normalized_modal_gpu_type(modal_gpu)
    return RemoteTorchBuild(
        cuda_version="13.0",
        install_layers=(
            RemoteTorchInstallLayer(
                index_url="https://download.pytorch.org/whl/cu130",
                packages=_CUDA_130_TORCH_PACKAGES,
            ),
            RemoteTorchInstallLayer(
                index_url="https://download.pytorch.org/whl/cpu",
                packages=_CUDA_130_CPU_AUDIO_PACKAGES,
                extra_options="--no-deps",
            ),
        ),
    )


def remote_torch_packages(modal_gpu: str = DEFAULT_MODAL_GPU) -> tuple[str, ...]:
    """Return the exact PyTorch package set for one Modal GPU specification."""
    return select_remote_torch_build(modal_gpu).packages


def _strip_requirement_comment(line: str) -> str:
    """Remove a requirements.txt comment while preserving URL fragments."""
    for index, character in enumerate(line):
        if character != "#":
            continue
        previous = line[index - 1] if index > 0 else ""
        if not previous or previous.isspace():
            return line[:index].strip()
    return line.strip()


def _custom_node_requirement_files(custom_nodes_dir: Path | None) -> tuple[Path, ...]:
    """Return top-level custom-node requirement files in stable order."""
    if custom_nodes_dir is None or not custom_nodes_dir.exists():
        return ()
    requirement_files: list[Path] = []
    for entry_path in sorted(custom_nodes_dir.iterdir(), key=lambda path: path.name):
        if not entry_path.is_dir():
            continue
        requirements_path = entry_path / "requirements.txt"
        if requirements_path.is_file():
            requirement_files.append(requirements_path)
    return tuple(requirement_files)


def _read_requirement_file(requirements_path: Path, seen: set[Path]) -> tuple[str, ...]:
    """Read package specs from one requirement file, following relative includes."""
    resolved_path = requirements_path.resolve()
    if resolved_path in seen:
        return ()
    seen.add(resolved_path)

    requirements: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = _strip_requirement_comment(raw_line)
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            _, include_path = line.split(maxsplit=1)
            requirements.extend(
                _read_requirement_file((requirements_path.parent / include_path).resolve(), seen)
            )
            continue
        if line.startswith(("-c ", "--constraint ")):
            logger.info(
                "Skipping custom-node pip constraint line from %s: %s",
                requirements_path,
                line,
            )
            continue
        if line.startswith("-"):
            logger.info(
                "Skipping custom-node pip option line from %s: %s",
                requirements_path,
                line,
            )
            continue
        requirements.append(line)
    return tuple(requirements)


def custom_node_runtime_packages(custom_nodes_dir: Path | None) -> tuple[str, ...]:
    """Return deduplicated package specs declared by bundled custom nodes."""
    requirements: list[str] = []
    seen_specs: set[str] = set()
    seen_files: set[Path] = set()
    for requirements_path in _custom_node_requirement_files(custom_nodes_dir):
        for requirement in _read_requirement_file(requirements_path, seen_files):
            if requirement in seen_specs:
                continue
            seen_specs.add(requirement)
            requirements.append(requirement)
    if requirements:
        logger.info(
            "Including %d custom-node Python requirement(s) in the Modal image from %s.",
            len(requirements),
            custom_nodes_dir,
        )
    return tuple(requirements)


def _tree_digest(
    root: Path | None,
    *,
    included_suffixes: frozenset[str],
    included_names: frozenset[str] = frozenset(),
    ignored_top_level_directories: frozenset[str] = frozenset(),
    included_top_level_directories: frozenset[str] | None = None,
) -> str:
    """Hash the selected files in one directory tree using stable relative paths."""
    if root is None or not root.is_dir():
        return "missing"

    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for directory, directory_names, file_names in os.walk(resolved_root):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(resolved_root)
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if directory_name not in _IGNORED_DIRECTORY_NAMES
            and not (
                relative_directory == Path(".")
                and directory_name in ignored_top_level_directories
            )
            and not (
                relative_directory == Path(".")
                and included_top_level_directories is not None
                and directory_name not in included_top_level_directories
            )
        )
        for file_name in sorted(file_names):
            file_path = directory_path / file_name
            if file_name not in included_names and file_path.suffix.lower() not in included_suffixes:
                continue
            relative_path = file_path.relative_to(resolved_root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _runtime_options(settings: RemoteRuntimeSettings) -> dict[str, Any]:
    """Return deployment settings that materially shape the remote runtime."""
    return {
        "app_name": settings.app_name,
        "buffer_containers": settings.buffer_containers,
        "enable_gpu_memory_snapshot": settings.enable_gpu_memory_snapshot,
        "enable_memory_snapshot": settings.enable_memory_snapshot,
        "execution_timeout_seconds": settings.execution_timeout_seconds,
        "invocation_dict_name": settings.invocation_dict_name,
        "invocation_result_inline_max_bytes": settings.invocation_result_inline_max_bytes,
        "interrupt_dict_name": settings.interrupt_dict_name,
        "loader_prewarm_workers": getattr(
            settings,
            "loader_prewarm_workers",
            2,
        ),
        "max_containers": settings.max_containers,
        "min_containers": settings.min_containers,
        "modal_gpu": settings.modal_gpu,
        "modal_secret_name": getattr(
            settings,
            "modal_secret_name",
            DEFAULT_MODAL_SECRET_NAME,
        ),
        "node_output_cache_dict_name": settings.node_output_cache_dict_name,
        "remote_storage_root": settings.remote_storage_root,
        "scaledown_window_seconds": settings.scaledown_window_seconds,
        "session_bridge_dict_name": settings.session_bridge_dict_name,
        "bridge_inline_max_bytes": settings.bridge_inline_max_bytes,
        "snapshot_profile_dict_name": settings.snapshot_profile_dict_name,
        "startup_timeout_seconds": settings.startup_timeout_seconds,
        "stream_event_queue_maxsize": settings.stream_event_queue_maxsize,
        "sync_index_dict_name": settings.sync_index_dict_name,
        "volume_name": settings.volume_name,
        "llm_max_resident_models": getattr(settings, "llm_max_resident_models", 2),
        "llm_reserve_free_vram_gb": getattr(settings, "llm_reserve_free_vram_gb", 24.0),
        "llm_compile_cache_volume_name": getattr(
            settings,
            "llm_compile_cache_volume_name",
            f"{settings.volume_name}-llm-compile-cache",
        ),
        "llm_vllm_execution_mode": getattr(
            settings,
            "llm_vllm_execution_mode",
            "auto",
        ),
    }


def build_remote_runtime_identity(
    *,
    repo_root: Path,
    comfyui_root: Path | None,
    custom_nodes_dir: Path | None,
    settings: RemoteRuntimeSettings,
) -> RemoteRuntimeIdentity:
    """Build the deterministic identity expected from one deployed Modal runtime."""
    torch_build = select_remote_torch_build(settings.modal_gpu)
    manifest: dict[str, Any] = {
        "protocol_version": REMOTE_APP_PROTOCOL_VERSION,
        "python_version": REMOTE_PYTHON_VERSION,
        "modal_sdk_spec": REMOTE_MODAL_SDK_SPEC,
        "apt_packages": list(remote_apt_packages()),
        "runtime_packages": list(remote_runtime_packages()),
        "accelerator_packages": list(remote_accelerator_packages(settings.modal_gpu)),
        "torch_packages": list(torch_build.packages),
        "torch_build": {
            "cuda_version": torch_build.cuda_version,
            "install_layers": [
                {
                    "index_url": install_layer.index_url,
                    "packages": list(install_layer.packages),
                    "extra_options": install_layer.extra_options,
                }
                for install_layer in torch_build.install_layers
            ],
        },
        "custom_node_packages": list(custom_node_runtime_packages(custom_nodes_dir)),
        "repo_source_digest": _tree_digest(
            repo_root,
            included_suffixes=frozenset({".py"}),
            included_names=frozenset({"llm_profiles.json"}),
            ignored_top_level_directories=frozenset({"tests"}),
        ),
        "comfyui_source_digest": _tree_digest(
            comfyui_root,
            included_suffixes=frozenset({".py"}),
            included_names=COMFYUI_RUNTIME_SOURCE_FILES,
            included_top_level_directories=COMFYUI_RUNTIME_SOURCE_DIRECTORIES,
        ),
        "runtime_options": _runtime_options(settings),
    }
    canonical_manifest = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return RemoteRuntimeIdentity(
        fingerprint=hashlib.sha256(canonical_manifest).hexdigest(),
        manifest=manifest,
    )
