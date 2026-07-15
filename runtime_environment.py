"""Pinned remote environment and deployment identity helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

REMOTE_APP_PROTOCOL_VERSION = 5
REMOTE_PYTHON_VERSION = "3.11"
REMOTE_MODAL_SDK_SPEC = "modal==1.4.2"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"

_REMOTE_RUNTIME_PACKAGES = (
    "aiohttp==3.13.3",
    "alembic==1.18.4",
    "av==16.1.0",
    "comfy-kitchen==0.2.7",
    "einops==0.8.2",
    "kornia==0.8.2",
    "numpy==2.4.2",
    "opencv-python-headless==4.13.0.92",
    "packaging==26.0",
    "pillow==12.1.0",
    "psutil==7.2.2",
    "pydantic==2.12.5",
    "pydantic-settings==2.13.0",
    "pyyaml==6.0.3",
    "requests==2.32.5",
    "safetensors==0.7.0",
    "scipy==1.17.0",
    "sentencepiece==0.2.1",
    "spandrel==0.4.1",
    "sqlalchemy==2.0.46",
    "torchsde==0.2.6",
    "tqdm==4.67.3",
    "transformers==5.1.0",
)
_REMOTE_TORCH_PACKAGES = (
    "torch==2.10.0",
    "torchvision==0.25.0",
    "torchaudio==2.10.0",
)
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
_IGNORED_COMFYUI_TOP_LEVEL_DIRECTORIES = frozenset(
    {"custom_nodes", "input", "models", "output", "temp", "user"}
)


class RemoteRuntimeSettings(Protocol):
    """Describe settings that shape one deployed Modal runtime."""

    app_name: str
    buffer_containers: int | None
    enable_gpu_memory_snapshot: bool
    enable_memory_snapshot: bool
    execution_timeout_seconds: int
    interrupt_dict_name: str
    max_containers: int | None
    min_containers: int
    modal_gpu: str
    node_output_cache_dict_name: str
    remote_storage_root: str
    scaledown_window_seconds: int
    session_bridge_dict_name: str
    snapshot_profile_dict_name: str
    startup_timeout_seconds: int
    stream_event_queue_maxsize: int
    sync_index_dict_name: str
    volume_name: str


@dataclass(frozen=True)
class RemoteRuntimeIdentity:
    """Hold a deterministic fingerprint and its diagnostic manifest."""

    fingerprint: str
    manifest: dict[str, Any]


def remote_runtime_packages() -> tuple[str, ...]:
    """Return the exact ComfyUI support package set used by Modal images."""
    return _REMOTE_RUNTIME_PACKAGES


def remote_torch_packages() -> tuple[str, ...]:
    """Return the exact CUDA PyTorch package set used by Modal images."""
    return _REMOTE_TORCH_PACKAGES


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
        "interrupt_dict_name": settings.interrupt_dict_name,
        "max_containers": settings.max_containers,
        "min_containers": settings.min_containers,
        "modal_gpu": settings.modal_gpu,
        "node_output_cache_dict_name": settings.node_output_cache_dict_name,
        "remote_storage_root": settings.remote_storage_root,
        "scaledown_window_seconds": settings.scaledown_window_seconds,
        "session_bridge_dict_name": settings.session_bridge_dict_name,
        "snapshot_profile_dict_name": settings.snapshot_profile_dict_name,
        "startup_timeout_seconds": settings.startup_timeout_seconds,
        "stream_event_queue_maxsize": settings.stream_event_queue_maxsize,
        "sync_index_dict_name": settings.sync_index_dict_name,
        "volume_name": settings.volume_name,
    }


def build_remote_runtime_identity(
    *,
    repo_root: Path,
    comfyui_root: Path | None,
    custom_nodes_dir: Path | None,
    settings: RemoteRuntimeSettings,
) -> RemoteRuntimeIdentity:
    """Build the deterministic identity expected from one deployed Modal runtime."""
    manifest: dict[str, Any] = {
        "protocol_version": REMOTE_APP_PROTOCOL_VERSION,
        "python_version": REMOTE_PYTHON_VERSION,
        "modal_sdk_spec": REMOTE_MODAL_SDK_SPEC,
        "runtime_packages": list(remote_runtime_packages()),
        "torch_packages": list(remote_torch_packages()),
        "custom_node_packages": list(custom_node_runtime_packages(custom_nodes_dir)),
        "repo_source_digest": _tree_digest(
            repo_root,
            included_suffixes=frozenset({".py"}),
            ignored_top_level_directories=frozenset({"tests"}),
        ),
        "comfyui_source_digest": _tree_digest(
            comfyui_root,
            included_suffixes=frozenset({".py"}),
            included_names=frozenset({"pyproject.toml", "requirements.txt", "uv.lock"}),
            ignored_top_level_directories=_IGNORED_COMFYUI_TOP_LEVEL_DIRECTORIES,
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
