"""Pytest fixtures for loading the ComfyUI Modal-Sync extension package."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path.home() / "git" / "ComfyUI"
PACKAGE_NAME = "comfyui_modal_sync_under_test"


def _ensure_import_paths() -> None:
    """Add the repository and local ComfyUI checkout to sys.path when present."""
    if "av" not in sys.modules:
        av_module = types.ModuleType("av")
        av_module.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
        av_module.open = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("The av stub should not be used in tests.")
        )
        av_module.time_base = 1
        av_module.VideoStream = type("VideoStream", (), {})
        av_module.AVError = RuntimeError
        av_module.FFmpegError = RuntimeError
        av_module.logging = types.SimpleNamespace(
            ERROR="ERROR",
            set_level=lambda *args, **kwargs: None,
        )
        av_module.video = types.SimpleNamespace(
            frame=types.SimpleNamespace(VideoFrame=type("VideoFrame", (), {"pict_type": None}))
        )

        av_container_module = types.ModuleType("av.container")
        av_container_module.InputContainer = type("InputContainer", (), {})

        av_subtitles_module = types.ModuleType("av.subtitles")
        av_subtitles_stream_module = types.ModuleType("av.subtitles.stream")
        av_subtitles_stream_module.SubtitleStream = type("SubtitleStream", (), {})

        sys.modules["av"] = av_module
        sys.modules["av.container"] = av_container_module
        sys.modules["av.subtitles"] = av_subtitles_module
        sys.modules["av.subtitles.stream"] = av_subtitles_stream_module
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if COMFYUI_ROOT.exists() and str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))


def _load_extension_package() -> object:
    """Load the extension root as an importable package for test modules."""
    _ensure_import_paths()
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create an import spec for the extension package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def extension_package() -> object:
    """Return the loaded extension package."""
    return _load_extension_package()


@pytest.fixture(scope="session")
def api_intercept_module(extension_package: object) -> object:
    """Return the prompt interception module."""
    return importlib.import_module(f"{PACKAGE_NAME}.api_intercept")


@pytest.fixture(scope="session")
def modal_executor_module(extension_package: object) -> object:
    """Return the dynamic proxy node module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_executor_node")


@pytest.fixture(scope="session")
def remote_modal_app_module(extension_package: object) -> object:
    """Return the remote execution module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_app")


@pytest.fixture(scope="session")
def serialization_module(extension_package: object) -> object:
    """Return the serialization helpers module."""
    return importlib.import_module(f"{PACKAGE_NAME}.serialization")


@pytest.fixture(scope="session")
def settings_module(extension_package: object) -> object:
    """Return the settings module."""
    return importlib.import_module(f"{PACKAGE_NAME}.settings")


@pytest.fixture(scope="session")
def sync_engine_module(extension_package: object) -> object:
    """Return the sync engine module."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_engine")


@pytest.fixture(scope="session")
def modal_cloud_module() -> object:
    """Return the stable Modal cloud entry module."""
    _ensure_import_paths()
    return importlib.import_module("comfyui_modal_sync_cloud")


@pytest.fixture(autouse=True)
def reset_modal_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate Modal-Sync environment variables between tests."""
    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "local")
    monkeypatch.setenv("COMFY_MODAL_LOCAL_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.delenv("COMFY_MODAL_AUTO_DEPLOY", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK", raising=False)
    monkeypatch.delenv("COMFY_MODAL_CUSTOM_NODES_DIR", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", raising=False)
    monkeypatch.delenv("COMFY_MODAL_INTERRUPT_DICT_NAME", raising=False)
    monkeypatch.delenv("COMFY_MODAL_MAX_CONTAINERS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_BUFFER_CONTAINERS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_SCALEDOWN_WINDOW", raising=False)
    monkeypatch.delenv("COMFY_MODAL_MIN_CONTAINERS", raising=False)
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)
